"""Locked HumanCurated443 data handling for the train399-only Stage-B model.

This module deliberately does not know about the historical RHAxiscc work
directory.  It consumes only the canonical, hash-audited dataset release and
an optional *read-only* image cache.  The locked 44-image development split is
never returned by :func:`training_records`.

The original RHAxiscc crop dataset kept one ``numpy.Generator`` on the Dataset
object.  With multiple persistent DataLoader workers each worker inherited an
identical generator state and therefore produced duplicate augmentation
streams.  Here every sample RNG is a pure function of ``(seed, epoch, index)``;
it is consequently independent of worker count, scheduling and resume.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset, Sampler

from ..io import atomic_write_json, sha256_file, sha256_json
from .model import HEADS
from .preprocess import make_input_channels, resample_to_physical_scale, to_gray


EXPECTED_TASKS = 443
EXPECTED_TRAIN = 399
EXPECTED_VAL = 44
LENGTH_SCALE_UM = 100.0


@dataclass(frozen=True)
class HairRecord:
    instance_id: str
    points: tuple[tuple[float, float], ...]
    length_um: float
    vertex_order_flipped: bool

    @property
    def base(self) -> tuple[float, float]:
        return self.points[0]

    @property
    def tip(self) -> tuple[float, float]:
        return self.points[-1]


@dataclass(frozen=True)
class StageBImageRecord:
    task_id: str
    split: str
    family_key: str
    image_path: str
    image_sha256: str
    raw_annotation_sha256: str
    canonical_annotation_sha256: str
    source_um_per_px: float
    width: int
    height: int
    root_polygon: tuple[tuple[float, float], ...]
    hairs: tuple[HairRecord, ...]

    @property
    def n_hairs(self) -> int:
        return len(self.hairs)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _point_in_polygon(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float64)
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    x, y = points[:, 0], points[:, 1]
    x1, y1 = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    inside = np.zeros(len(points), dtype=bool)
    for xa, ya, xb, yb in zip(x1, y1, x2, y2, strict=True):
        crosses = (ya > y) != (yb > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            crossing_x = (xb - xa) * (y - ya) / (
                np.nan if yb == ya else yb - ya
            ) + xa
        inside ^= crosses & (x < crossing_x)
    return inside


def _signed_distance_to_polygon(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float64)
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    start = polygon
    end = np.roll(polygon, -1, axis=0)
    segment = end - start
    squared = np.einsum("ij,ij->i", segment, segment)
    relative = points[:, None, :] - start[None, :, :]
    fraction = np.einsum("nmi,mi->nm", relative, segment) / np.maximum(
        squared[None, :], 1e-12
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    projection = start[None, :, :] + fraction[..., None] * segment[None, :, :]
    distance = np.linalg.norm(points[:, None, :] - projection, axis=-1).min(axis=1)
    return np.where(_point_in_polygon(polygon, points), -distance, distance)


def _integrity_by_task(dataset_root: Path) -> dict[str, dict[str, dict[str, str]]]:
    rows = _read_csv(dataset_root / "manifests" / "integrity_sha256.csv")
    if len(rows) != EXPECTED_TASKS * 9:
        raise RuntimeError(f"expected 3987 integrity rows, found {len(rows)}")
    returned: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        task = returned.setdefault(row["task_id"], {})
        if row["role"] in task:
            raise RuntimeError(
                f"duplicate integrity role {row['task_id']} / {row['role']}"
            )
        task[row["role"]] = row
    if len(returned) != EXPECTED_TASKS or any(len(roles) != 9 for roles in returned.values()):
        raise RuntimeError("integrity manifest does not contain 9 roles for every task")
    return returned


def verify_integrity_readonly(dataset_root: str | Path) -> dict[str, Any]:
    """Rehash the release integrity manifest without writing into the dataset."""

    root = Path(dataset_root).resolve()
    manifest_path = root / "manifests" / "integrity_sha256.csv"
    rows = _read_csv(manifest_path)
    if len(rows) != EXPECTED_TASKS * 9:
        raise RuntimeError(f"expected 3987 integrity rows, found {len(rows)}")
    role_counts: dict[str, int] = {}
    verified_entries: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        path = root / row["relative_path"]
        if not path.is_file():
            raise RuntimeError(f"missing integrity file: {path}")
        expected_size = int(row["size_bytes"])
        if path.stat().st_size != expected_size:
            raise RuntimeError(f"integrity size mismatch: {path}")
        observed_sha256 = sha256_file(path)
        if observed_sha256 != row["sha256"]:
            raise RuntimeError(f"integrity SHA-256 mismatch: {path}")
        role_counts[row["role"]] = role_counts.get(row["role"], 0) + 1
        verified_entries.append(
            {
                "task_id": row["task_id"],
                "role": row["role"],
                "sha256": observed_sha256,
                "size_bytes": expected_size,
            }
        )
        if index % 250 == 0 or index == len(rows):
            print(f"[integrity-readonly] {index}/{len(rows)}", flush=True)
    if len(role_counts) != 9 or any(value != EXPECTED_TASKS for value in role_counts.values()):
        raise RuntimeError(f"unexpected integrity role counts: {role_counts}")
    return {
        "schema_version": "PHAxis-readonly-integrity-recheck-1.0",
        "status": "passed",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "integrity_manifest_sha256": sha256_file(manifest_path),
        "files_hashed": len(rows),
        "role_counts": role_counts,
        "verified_entries_identity_sha256": sha256_json(verified_entries),
        "dataset_files_written": 0,
        "blind_images_used": 0,
    }


def load_locked_records(
    dataset_root: str | Path,
    *,
    split_manifest: str | Path | None = None,
    split_lock: str | Path | None = None,
) -> tuple[list[StageBImageRecord], dict[str, Any]]:
    """Load canonical vectors and independently enforce the locked split."""

    root = Path(dataset_root).resolve()
    dataset_manifest_path = root / "manifests" / "dataset_manifest.csv"
    source_split_manifest_path = root / "manifests" / "split_manifest.csv"
    split_manifest_path = (
        Path(split_manifest).resolve()
        if split_manifest is not None
        else source_split_manifest_path
    )
    dataset_rows = _read_csv(dataset_manifest_path)
    split_rows = _read_csv(split_manifest_path)
    if len(dataset_rows) != EXPECTED_TASKS or len(split_rows) != EXPECTED_TASKS:
        raise RuntimeError("HumanCurated443 must contain exactly 443 manifest rows")
    dataset_by_id = {row["task_id"]: row for row in dataset_rows}
    split_by_id = {row["task_id"]: row for row in split_rows}
    if len(dataset_by_id) != EXPECTED_TASKS or len(split_by_id) != EXPECTED_TASKS:
        raise RuntimeError("duplicate task_id in dataset or split manifest")
    if set(dataset_by_id) != set(split_by_id):
        raise RuntimeError("dataset_manifest and split_manifest task sets differ")
    integrity = _integrity_by_task(root)

    records: list[StageBImageRecord] = []
    flipped = 0
    for task_id in sorted(dataset_by_id):
        row = dataset_by_id[task_id]
        split_row = split_by_id[task_id]
        if row["family_key"] != split_row["family_key"]:
            raise RuntimeError(f"family_key disagreement for {task_id}")
        annotation_path = root / row["canonical_annotation_relpath"]
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        if payload.get("task_id") != task_id:
            raise RuntimeError(f"canonical task_id mismatch for {task_id}")
        roots: list[np.ndarray] = []
        raw_hairs: list[tuple[str, np.ndarray]] = []
        for shape in payload.get("shapes", []):
            points = np.asarray(shape.get("points", []), dtype=np.float64)
            if shape.get("label") == "root" and len(points) >= 3:
                roots.append(points)
            elif shape.get("label") == "root_hair" and len(points) >= 2:
                raw_hairs.append(
                    (str(shape.get("instance_id") or f"H{len(raw_hairs)}"), points)
                )
        if not roots:
            raise RuntimeError(f"no root polygon in {task_id}")
        root_polygon = max(roots, key=len)
        hairs: list[HairRecord] = []
        if raw_hairs:
            endpoints = np.asarray([[points[0], points[-1]] for _, points in raw_hairs])
            distances = _signed_distance_to_polygon(
                root_polygon, endpoints.reshape(-1, 2)
            ).reshape(-1, 2)
            for index, (instance_id, points) in enumerate(raw_hairs):
                is_flipped = bool(distances[index, 1] < distances[index, 0])
                if is_flipped:
                    points = points[::-1].copy()
                    flipped += 1
                hairs.append(
                    HairRecord(
                        instance_id=instance_id,
                        points=tuple((float(x), float(y)) for x, y in points),
                        length_um=_polyline_length(points)
                        * float(row["source_um_per_px"]),
                        vertex_order_flipped=is_flipped,
                    )
                )
        expected_hairs = int(row["root_hair_count"])
        if len(hairs) != expected_hairs:
            raise RuntimeError(
                f"root-hair count mismatch for {task_id}: {len(hairs)} != {expected_hairs}"
            )
        roles = integrity[task_id]
        if roles["image"]["sha256"] != row["image_sha256"]:
            raise RuntimeError(f"image hash manifest disagreement for {task_id}")
        if roles["raw_annotation"]["sha256"] != row["raw_annotation_sha256"]:
            raise RuntimeError(f"raw annotation hash disagreement for {task_id}")
        records.append(
            StageBImageRecord(
                task_id=task_id,
                split=split_row["split"],
                family_key=row["family_key"],
                image_path=str((root / row["image_relpath"]).resolve()),
                image_sha256=row["image_sha256"],
                raw_annotation_sha256=row["raw_annotation_sha256"],
                canonical_annotation_sha256=roles["canonical_annotation"]["sha256"],
                source_um_per_px=float(row["source_um_per_px"]),
                width=int(row["image_width"]),
                height=int(row["image_height"]),
                root_polygon=tuple(
                    (float(x), float(y)) for x, y in root_polygon
                ),
                hairs=tuple(hairs),
            )
        )

    train = [record for record in records if record.split == "train"]
    val = [record for record in records if record.split == "val"]
    if len(train) != EXPECTED_TRAIN or len(val) != EXPECTED_VAL:
        raise RuntimeError(f"locked split must be 399/44, observed {len(train)}/{len(val)}")
    train_families = {record.family_key for record in train}
    val_families = {record.family_key for record in val}
    overlap = sorted(train_families & val_families)
    if overlap:
        raise RuntimeError(f"family leakage in locked split: {overlap}")

    train_ids = [record.task_id for record in train]
    val_ids = [record.task_id for record in val]
    train_family_rows = sorted((record.task_id, record.family_key) for record in train)
    val_family_rows = sorted((record.task_id, record.family_key) for record in val)
    verification_path = root / "verification_report.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if verification.get("status") != "passed" or verification.get("blind_images_used") != 0:
        raise RuntimeError("dataset verification report is not a blind-free pass")
    split_lock_payload: dict[str, Any] | None = None
    split_lock_sha256: str | None = None
    locked_split_identity_sha256: str | None = None
    if split_lock is not None:
        split_lock_path = Path(split_lock).resolve()
        split_lock_payload = json.loads(split_lock_path.read_text(encoding="utf-8"))
        split_lock_sha256 = sha256_file(split_lock_path)
        locked_split_identity_sha256 = split_lock_payload.get("split_identity_sha256")
        expected_split_sha256 = split_lock_payload.get("files", {}).get(
            "split_manifest.csv", {}
        ).get("sha256")
        if expected_split_sha256 != sha256_file(split_manifest_path):
            raise RuntimeError("split lock does not match the selected split manifest")
        if split_lock_payload.get("counts") != {
            "all": 443,
            "train": 399,
            "val": 44,
            "family_key_overlap": 0,
        }:
            raise RuntimeError("unexpected counts in split lock")
        if split_lock_payload.get("blind_images_used") != 0:
            raise RuntimeError("split lock is not blind-free")

    stable_identity = {
        "dataset_version": dataset_rows[0]["dataset_version"],
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "source_dataset_split_manifest_sha256": sha256_file(
            source_split_manifest_path
        ),
        "split_lock_sha256": split_lock_sha256,
        "locked_split_identity_sha256": locked_split_identity_sha256,
        "integrity_manifest_sha256": sha256_file(
            root / "manifests" / "integrity_sha256.csv"
        ),
        "train_ids": train_ids,
        "train_ids_sha256": sha256_json(train_ids),
        "train_task_family_rows": train_family_rows,
        "train_task_family_sha256": sha256_json(train_family_rows),
        "train_families": sorted(train_families),
        "train_families_sha256": sha256_json(sorted(train_families)),
        "excluded_val_ids": val_ids,
        "excluded_val_ids_sha256": sha256_json(val_ids),
        "excluded_val_task_family_rows": val_family_rows,
        "excluded_val_task_family_sha256": sha256_json(val_family_rows),
        "excluded_val_families": sorted(val_families),
        "excluded_val_families_sha256": sha256_json(sorted(val_families)),
        "family_key_overlap": overlap,
        "records": len(records),
        "train_records": len(train),
        "excluded_val_records": len(val),
        "train_root_hairs": sum(record.n_hairs for record in train),
        "excluded_val_root_hairs": sum(record.n_hairs for record in val),
        "vertex_orders_geometrically_flipped": flipped,
        "validation_labels_used_for_gradient": False,
        "validation_labels_used_for_early_stopping": False,
        "blind_images_used": 0,
        "pyRootHair_called_or_copied": False,
    }
    identity = {
        **stable_identity,
        "verification_report_sha256": sha256_file(verification_path),
        "verification_report_verified_utc": verification.get("verified_utc"),
        "dataset_split_identity_sha256": sha256_json(stable_identity),
    }
    return records, identity


def write_dataset_audit(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    split_manifest: str | Path | None = None,
    split_lock: str | Path | None = None,
    rehash_integrity: bool = True,
    write_side_effect_incident: dict[str, Any] | None = None,
) -> tuple[list[StageBImageRecord], dict[str, Any]]:
    records, identity = load_locked_records(
        dataset_root, split_manifest=split_manifest, split_lock=split_lock
    )
    readonly_integrity = (
        verify_integrity_readonly(dataset_root) if rehash_integrity else None
    )
    payload = {
        "schema_version": "PHAxis-StageB-train399-dataset-audit-1.0",
        "status": "passed",
        "dataset_root": str(Path(dataset_root).resolve()),
        "selected_split_manifest": (
            str(Path(split_manifest).resolve()) if split_manifest is not None else None
        ),
        "selected_split_lock": (
            str(Path(split_lock).resolve()) if split_lock is not None else None
        ),
        **identity,
        "integrity_recheck": readonly_integrity,
        "dataset_files_written_by_audit": 0,
        "write_side_effect_incident": write_side_effect_incident,
        "gradient_authorization": "only locked split=train task IDs",
        "model_initialization_prohibition": (
            "no RHAxiscc fold checkpoint or any state trained on the locked 44 may be loaded"
        ),
    }
    atomic_write_json(output_path, payload)
    return records, payload


def training_records(records: Sequence[StageBImageRecord]) -> list[StageBImageRecord]:
    returned = [record for record in records if record.split == "train"]
    if len(returned) != EXPECTED_TRAIN or any(record.split != "train" for record in returned):
        raise RuntimeError("training_records failed the strict train399 gate")
    return returned


def excluded_validation_records(
    records: Sequence[StageBImageRecord],
) -> list[StageBImageRecord]:
    returned = [record for record in records if record.split == "val"]
    if len(returned) != EXPECTED_VAL:
        raise RuntimeError("expected exactly 44 excluded validation records")
    return returned


def _atomic_save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_legacy_cache(
    record: StageBImageRecord, image_path: Path, metadata_path: Path, target_um_per_px: float
) -> tuple[np.ndarray, dict[str, Any]]:
    if not image_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(image_path)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    array = np.load(image_path, mmap_mode="r", allow_pickle=False)
    expected_scale = record.source_um_per_px / target_um_per_px
    expected_shape = (
        max(1, int(round(record.height * expected_scale))),
        max(1, int(round(record.width * expected_scale))),
    )
    checks = (
        metadata.get("task_id") == record.task_id,
        tuple(metadata.get("shape", ())) == expected_shape,
        tuple(array.shape) == expected_shape,
        array.dtype == np.uint8,
        math.isclose(float(metadata.get("scale")), expected_scale, rel_tol=0, abs_tol=1e-8),
        math.isclose(
            float(metadata.get("src_um_px")),
            record.source_um_per_px,
            rel_tol=0,
            abs_tol=1e-10,
        ),
        math.isclose(
            float(metadata.get("um_per_px")),
            target_um_per_px,
            rel_tol=0,
            abs_tol=1e-10,
        ),
    )
    if not all(checks):
        raise RuntimeError(f"legacy cache metadata mismatch for {record.task_id}")
    return array, metadata


def _realized_cache_geometry(
    record: StageBImageRecord,
    target_um_per_px: float,
    cached_shape: Sequence[int],
) -> dict[str, Any]:
    """Return the requested and actually realized resize geometry.

    A scalar physical-scale request generally cannot be represented exactly by
    integer output dimensions. Vector annotations must therefore be mapped by
    the realized x/y scales, not by the requested scalar. The latter is kept
    only as provenance and as the nominal scale used for approximately
    isotropic line-width targets.
    """

    if len(cached_shape) != 2:
        raise RuntimeError(f"invalid cached shape for {record.task_id}: {cached_shape}")
    cached_height, cached_width = (int(cached_shape[0]), int(cached_shape[1]))
    if min(cached_height, cached_width, record.height, record.width) <= 0:
        raise RuntimeError(f"non-positive cache geometry for {record.task_id}")
    requested_scale = float(record.source_um_per_px) / float(target_um_per_px)
    expected_shape = (
        max(1, int(round(record.height * requested_scale))),
        max(1, int(round(record.width * requested_scale))),
    )
    if (cached_height, cached_width) != expected_shape:
        raise RuntimeError(
            f"cached shape does not realize the requested resize for {record.task_id}: "
            f"{(cached_height, cached_width)} != {expected_shape}"
        )
    scale_x = cached_width / float(record.width)
    scale_y = cached_height / float(record.height)
    realized_um_per_px_x = float(record.source_um_per_px) / scale_x
    realized_um_per_px_y = float(record.source_um_per_px) / scale_y
    axis_errors = (
        abs(realized_um_per_px_x - float(target_um_per_px)),
        abs(realized_um_per_px_y - float(target_um_per_px)),
    )
    return {
        "requested_source_to_cached_scale": requested_scale,
        "source_to_cached_scale_xy": [scale_x, scale_y],
        "scale_xy": [scale_x, scale_y],
        "realized_um_per_px_xy": [realized_um_per_px_x, realized_um_per_px_y],
        "realized_um_per_px_max_axis_abs_error": max(axis_errors),
        "realized_um_per_px_axis_difference": abs(
            realized_um_per_px_x - realized_um_per_px_y
        ),
    }


def _validate_realized_geometry_fields(
    metadata: dict[str, Any], expected: dict[str, Any], task_id: str
) -> None:
    """Fail closed if a v1.1 entry's redundant geometry fields disagree."""

    pair_fields = (
        "source_to_cached_scale_xy",
        "scale_xy",
        "realized_um_per_px_xy",
    )
    scalar_fields = (
        "requested_source_to_cached_scale",
        "realized_um_per_px_max_axis_abs_error",
        "realized_um_per_px_axis_difference",
    )
    for field in pair_fields:
        observed = metadata.get(field)
        if not isinstance(observed, list) or len(observed) != 2 or not np.allclose(
            np.asarray(observed, dtype=np.float64),
            np.asarray(expected[field], dtype=np.float64),
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(f"stale cache realized geometry for {task_id}: {field}")
    for field in scalar_fields:
        try:
            observed_scalar = float(metadata[field])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"stale cache realized geometry for {task_id}: {field}"
            ) from error
        if not math.isclose(
            observed_scalar, float(expected[field]), rel_tol=0, abs_tol=1e-12
        ):
            raise RuntimeError(f"stale cache realized geometry for {task_id}: {field}")


def materialize_image_cache(
    records: Sequence[StageBImageRecord],
    cache_root: str | Path,
    *,
    target_um_per_px: float = 2.0,
    readonly_reuse_root: str | Path | None = None,
    hash_arrays: bool = True,
) -> dict[str, Any]:
    """Build or validate a provenance-bearing cache in the project workspace.

    When ``readonly_reuse_root`` points to the audited RHAxiscc cache, its
    arrays are byte-copied after strict geometry/scale validation.  A copy is
    intentional: changing the read-only attribute of an NTFS hard link would
    also mutate the external cache's file record.  Every project-local array is
    marked read-only and is only ever opened with a read-only mmap.  If reuse is
    unavailable, the canonical image is decoded and resampled afresh.
    """

    cache = Path(cache_root).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    reuse = Path(readonly_reuse_root).resolve() if readonly_reuse_root else None
    entries: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        destination = cache / f"{record.task_id}.npy"
        metadata_destination = cache / f"{record.task_id}.meta.json"
        method = "existing_validated"
        requested_scale = record.source_um_per_px / target_um_per_px
        expected_shape = (
            max(1, int(round(record.height * requested_scale))),
            max(1, int(round(record.width * requested_scale))),
        )
        realized_geometry = _realized_cache_geometry(
            record, target_um_per_px, expected_shape
        )
        if destination.exists() and metadata_destination.exists():
            metadata = json.loads(metadata_destination.read_text(encoding="utf-8"))
            if (
                metadata.get("task_id") != record.task_id
                or metadata.get("source_image_sha256") != record.image_sha256
                or metadata.get("raw_annotation_sha256")
                != record.raw_annotation_sha256
                or metadata.get("canonical_annotation_sha256")
                != record.canonical_annotation_sha256
                or not math.isclose(
                    float(metadata.get("target_um_per_px")),
                    target_um_per_px,
                    rel_tol=0,
                    abs_tol=1e-10,
                )
                or not math.isclose(
                    float(metadata.get("source_to_cached_scale")),
                    requested_scale,
                    rel_tol=0,
                    abs_tol=1e-10,
                )
                or tuple(metadata.get("source_shape", ()))
                != (record.height, record.width)
                or tuple(metadata.get("cached_shape", ())) != expected_shape
                or metadata.get("dtype") != "uint8"
                or not metadata.get("array_sha256")
            ):
                raise RuntimeError(f"stale cache provenance for {record.task_id}")
            array = np.load(destination, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.uint8 or tuple(array.shape) != expected_shape:
                raise RuntimeError(f"existing cache array mismatch for {record.task_id}")
            observed_sha256 = sha256_file(destination) if hash_arrays else metadata.get(
                "array_sha256"
            )
            if not observed_sha256 or observed_sha256 != metadata["array_sha256"]:
                raise RuntimeError(f"cache SHA-256 mismatch for {record.task_id}")
            if os.access(destination, os.W_OK):
                # ``os.access`` follows effective Windows ACLs and can report
                # writable for administrators despite the DOS attribute, so
                # also require the explicit read-only mode bit below.
                pass
            if destination.stat().st_mode & stat.S_IWRITE:
                raise RuntimeError(f"existing cache is not read-only: {record.task_id}")
            geometry_fields = {
                "requested_source_to_cached_scale",
                "source_to_cached_scale_xy",
                "scale_xy",
                "realized_um_per_px_xy",
                "realized_um_per_px_max_axis_abs_error",
                "realized_um_per_px_axis_difference",
            }
            present_geometry_fields = geometry_fields.intersection(metadata)
            if present_geometry_fields and present_geometry_fields != geometry_fields:
                raise RuntimeError(
                    f"partial realized cache geometry metadata for {record.task_id}"
                )
            if present_geometry_fields:
                _validate_realized_geometry_fields(
                    metadata, realized_geometry, record.task_id
                )
                if (
                    metadata.get("schema_version")
                    != "PHAxis-StageB-physical-cache-entry-1.1"
                    or metadata.get("line_width_physical_scale_policy")
                    != "nominal_target_um_per_px_scalar_approximation"
                ):
                    raise RuntimeError(
                        f"stale cache geometry policy for {record.task_id}"
                    )
            else:
                # Atomic metadata-only upgrade. The already hash-validated,
                # read-only pixel array is deliberately not rewritten.
                metadata = {
                    **metadata,
                    "schema_version": "PHAxis-StageB-physical-cache-entry-1.1",
                    **realized_geometry,
                    "line_width_physical_scale_policy": (
                        "nominal_target_um_per_px_scalar_approximation"
                    ),
                }
                atomic_write_json(metadata_destination, metadata)
        else:
            if destination.exists() != metadata_destination.exists():
                raise RuntimeError(f"partial cache entry for {record.task_id}")
            reused = False
            if reuse is not None:
                source = reuse / f"{record.task_id}.npy"
                source_meta = reuse / f"{record.task_id}.meta.npy"
                if source.is_file() and source_meta.is_file():
                    array, _legacy_metadata = _validate_legacy_cache(
                        record, source, source_meta, target_um_per_px
                    )
                    temporary = destination.with_name(
                        f".{destination.name}.{os.getpid()}.copy.tmp"
                    )
                    try:
                        shutil.copyfile(source, temporary)
                        os.replace(temporary, destination)
                    finally:
                        temporary.unlink(missing_ok=True)
                    reused = True
                    method = "byte_copy_from_readonly_validated_cache"
            if not reused:
                source_array = tifffile.imread(record.image_path)
                gray = to_gray(source_array)
                array, scale = resample_to_physical_scale(
                    gray, record.source_um_per_px, target_um_per_px
                )
                _atomic_save_numpy(destination, np.asarray(array, dtype=np.uint8))
                method = "canonical_image_decode_and_resample"
            os.chmod(destination, stat.S_IREAD)
            array = np.load(destination, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.uint8 or tuple(array.shape) != expected_shape:
                raise RuntimeError(f"materialized cache geometry mismatch for {record.task_id}")
            realized_geometry = _realized_cache_geometry(
                record, target_um_per_px, array.shape
            )
            array_sha256 = sha256_file(destination) if hash_arrays else None
            metadata = {
                "schema_version": "PHAxis-StageB-physical-cache-entry-1.1",
                "task_id": record.task_id,
                "source_image_sha256": record.image_sha256,
                "raw_annotation_sha256": record.raw_annotation_sha256,
                "canonical_annotation_sha256": record.canonical_annotation_sha256,
                "source_um_per_px": record.source_um_per_px,
                "target_um_per_px": target_um_per_px,
                "source_shape": [record.height, record.width],
                "cached_shape": list(array.shape),
                # Backwards-compatible scalar alias: requested, never used to
                # map canonical vector coordinates in v1.1.
                "source_to_cached_scale": requested_scale,
                **realized_geometry,
                "dtype": str(array.dtype),
                "array_sha256": array_sha256,
                "materialization": method,
                "application_write_policy": "read_only_no_in_place_mutation",
                "line_width_physical_scale_policy": (
                    "nominal_target_um_per_px_scalar_approximation"
                ),
                "blind_images_used": 0,
            }
            atomic_write_json(metadata_destination, metadata)
        observed_sha256 = sha256_file(destination) if hash_arrays else metadata.get(
            "array_sha256"
        )
        if metadata.get("array_sha256") and observed_sha256 != metadata["array_sha256"]:
            raise RuntimeError(f"cache SHA-256 mismatch for {record.task_id}")
        entries.append(
            {
                "task_id": record.task_id,
                "split": record.split,
                "array_sha256": observed_sha256,
                "size_bytes": destination.stat().st_size,
                "shape": list(array.shape),
                "materialization": metadata.get("materialization", method),
                "requested_source_to_cached_scale": metadata[
                    "requested_source_to_cached_scale"
                ],
                "source_to_cached_scale_xy": metadata["source_to_cached_scale_xy"],
                "realized_um_per_px_xy": metadata["realized_um_per_px_xy"],
                "realized_um_per_px_max_axis_abs_error": metadata[
                    "realized_um_per_px_max_axis_abs_error"
                ],
                "realized_um_per_px_axis_difference": metadata[
                    "realized_um_per_px_axis_difference"
                ],
                "metadata_sha256": sha256_file(metadata_destination),
            }
        )
        if index % 25 == 0 or index == len(records):
            print(f"[cache] {index}/{len(records)}", flush=True)
    stable_entries = [
        {
            "task_id": entry["task_id"],
            "split": entry["split"],
            "array_sha256": entry["array_sha256"],
            "size_bytes": entry["size_bytes"],
            "shape": entry["shape"],
            "metadata_sha256": entry["metadata_sha256"],
            "source_to_cached_scale_xy": entry["source_to_cached_scale_xy"],
            "realized_um_per_px_xy": entry["realized_um_per_px_xy"],
        }
        for entry in entries
    ]
    cache_identity_payload = {
        "schema_version": "PHAxis-StageB-train399-cache-identity-1.1",
        "target_um_per_px": target_um_per_px,
        "vector_coordinate_mapping": (
            "per_axis_realized_source_to_cached_scale_xy"
        ),
        "requested_scale_role": "provenance_only_not_vector_coordinate_mapping",
        "line_width_physical_scale_policy": (
            "nominal_target_um_per_px_scalar_approximation"
        ),
        "entries": stable_entries,
    }
    payload = {
        "schema_version": "PHAxis-StageB-train399-cache-audit-1.1",
        "status": "passed",
        "cache_root": str(cache),
        "target_um_per_px": target_um_per_px,
        "records": len(entries),
        "task_ids_sha256": sha256_json([entry["task_id"] for entry in entries]),
        "cache_identity_sha256": sha256_json(cache_identity_payload),
        "cache_identity_payload_sha256": sha256_json(cache_identity_payload),
        "entries": entries,
        "vector_coordinate_mapping": "per_axis_realized_source_to_cached_scale_xy",
        "requested_scale_role": "provenance_only_not_vector_coordinate_mapping",
        "line_width_physical_scale_policy": (
            "nominal_target_um_per_px_scalar_approximation"
        ),
        "max_realized_um_per_px_axis_abs_error": max(
            entry["realized_um_per_px_max_axis_abs_error"] for entry in entries
        ),
        "max_realized_um_per_px_axis_difference": max(
            entry["realized_um_per_px_axis_difference"] for entry in entries
        ),
        "cache_is_rebuildable": True,
        "application_write_policy": "read_only_no_in_place_mutation",
        "blind_images_used": 0,
    }
    atomic_write_json(cache.parent / "cache_audit.json", payload)
    return payload


def _resample_polyline(points: np.ndarray, step: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return points
    segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    total = float(cumulative[-1])
    if total <= 0:
        return points[:1]
    count = max(2, int(np.ceil(total / max(step, 1e-6))) + 1)
    samples = np.linspace(0.0, total, count)
    return np.column_stack(
        (
            np.interp(samples, cumulative, points[:, 0]),
            np.interp(samples, cumulative, points[:, 1]),
        )
    ).astype(np.float32)


def _draw_gaussian(
    heatmap: np.ndarray, center_x: float, center_y: float, sigma: float
) -> None:
    height, width = heatmap.shape
    center_x, center_y = float(np.floor(center_x)), float(np.floor(center_y))
    radius = int(max(1, np.ceil(3.0 * sigma)))
    x0, x1 = int(center_x) - radius, int(center_x) + radius + 1
    y0, y1 = int(center_y) - radius, int(center_y) + radius + 1
    clipped_x0, clipped_x1 = max(0, x0), min(width, x1)
    clipped_y0, clipped_y1 = max(0, y0), min(height, y1)
    if clipped_x0 >= clipped_x1 or clipped_y0 >= clipped_y1:
        return
    xs = np.arange(clipped_x0, clipped_x1, dtype=np.float32)
    ys = np.arange(clipped_y0, clipped_y1, dtype=np.float32)
    gaussian = np.exp(
        -(
            (xs[None, :] - center_x) ** 2
            + (ys[:, None] - center_y) ** 2
        )
        / (2.0 * sigma**2)
    )
    np.maximum(
        heatmap[clipped_y0:clipped_y1, clipped_x0:clipped_x1],
        gaussian,
        out=heatmap[clipped_y0:clipped_y1, clipped_x0:clipped_x1],
    )


def build_training_targets(
    hairs: Sequence[dict[str, Any]],
    root_mask: np.ndarray,
    *,
    crop: int,
    out_stride: int,
    um_per_px: float,
    base_sigma_um: float,
    tip_sigma_um: float,
    line_halfwidth_um: float,
) -> dict[str, np.ndarray]:
    output_height = crop // out_stride
    output_width = crop // out_stride
    target = {
        name: np.zeros((channels, output_height, output_width), dtype=np.float32)
        for name, channels in HEADS.items()
    }
    base_mask = np.zeros((output_height, output_width), dtype=np.float32)
    tip_mask = np.zeros((output_height, output_width), dtype=np.float32)
    scale = 1.0 / out_stride
    base_sigma = max(0.8, base_sigma_um / um_per_px * scale)
    tip_sigma = max(0.8, tip_sigma_um / um_per_px * scale)

    halfwidth_px = max(1.0, line_halfwidth_um / um_per_px)
    thickness = int(max(1, round(2 * halfwidth_px)))
    line_full = np.zeros((crop, crop), dtype=np.float32)
    flow_full = np.zeros((2, crop, crop), dtype=np.float32)
    for hair in hairs:
        points = np.asarray(hair["points"], dtype=np.float32)
        if len(points) < 2:
            continue
        resampled = _resample_polyline(points, max(1.0, halfwidth_px))
        integer = np.round(resampled).astype(np.int32)
        for index in range(len(integer) - 1):
            first, second = integer[index], integer[index + 1]
            cv2.line(line_full, tuple(first), tuple(second), 1.0, thickness, cv2.LINE_AA)
            vector = resampled[index + 1] - resampled[index]
            norm = float(np.linalg.norm(vector))
            if norm < 1e-6:
                continue
            vector /= norm
            cv2.line(flow_full[0], tuple(first), tuple(second), float(vector[0]), thickness)
            cv2.line(flow_full[1], tuple(first), tuple(second), float(vector[1]), thickness)
    if out_stride > 1:
        line_output = cv2.resize(
            line_full, (output_width, output_height), interpolation=cv2.INTER_AREA
        )
        flow_x = cv2.resize(
            flow_full[0], (output_width, output_height), interpolation=cv2.INTER_AREA
        )
        flow_y = cv2.resize(
            flow_full[1], (output_width, output_height), interpolation=cv2.INTER_AREA
        )
    else:
        line_output, flow_x, flow_y = line_full, flow_full[0], flow_full[1]
    target["line"][0] = np.clip(
        line_output * (2.0 if out_stride > 1 else 1.0), 0, 1
    )
    flow_norm = np.sqrt(flow_x**2 + flow_y**2)
    valid_flow = flow_norm > 1e-3
    target["flow"][0, valid_flow] = flow_x[valid_flow] / flow_norm[valid_flow]
    target["flow"][1, valid_flow] = flow_y[valid_flow] / flow_norm[valid_flow]

    for hair in hairs:
        base_x, base_y = np.asarray(hair["points"])[0] * scale
        tip_x, tip_y = np.asarray(hair["points"])[-1] * scale
        if 0 <= base_x < output_width and 0 <= base_y < output_height:
            _draw_gaussian(target["base_hm"][0], base_x, base_y, base_sigma)
            ix, iy = int(base_x), int(base_y)
            base_mask[iy, ix] = 1.0
            target["base_off"][:, iy, ix] = (base_x - ix, base_y - iy)
            vector = np.asarray(hair["points"])[-1] - np.asarray(hair["points"])[0]
            norm = float(np.linalg.norm(vector))
            if norm > 1e-6:
                target["base_dir"][:, iy, ix] = vector / norm
            target["base_len"][0, iy, ix] = np.log(
                max(float(hair["length_um"]), 1.0) / LENGTH_SCALE_UM
            )
        if 0 <= tip_x < output_width and 0 <= tip_y < output_height:
            _draw_gaussian(target["tip_hm"][0], tip_x, tip_y, tip_sigma)
            ix, iy = int(tip_x), int(tip_y)
            tip_mask[iy, ix] = 1.0
            target["tip_off"][:, iy, ix] = (tip_x - ix, tip_y - iy)
    target["root"][0] = (
        cv2.resize(
            root_mask.astype(np.float32),
            (output_width, output_height),
            interpolation=cv2.INTER_AREA,
        )
        if out_stride > 1
        else root_mask.astype(np.float32)
    )
    target["_base_mask"] = base_mask[None]
    target["_tip_mask"] = tip_mask[None]
    return target


class DeterministicEpochSampler(Sampler[tuple[int, int]]):
    """Epoch-aware deterministic sampler whose indices carry the epoch."""

    def __init__(self, size: int, seed: int):
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        generator = torch.Generator()
        generator.manual_seed((self.seed + 1_000_003 * self.epoch) % (2**63 - 1))
        for index in torch.randperm(self.size, generator=generator).tolist():
            yield self.epoch, int(index)

    def __len__(self) -> int:
        return self.size


class Train399HairCropDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic crop/augmentation dataset over exactly 399 train records."""

    def __init__(
        self,
        records: Sequence[StageBImageRecord],
        cache_root: str | Path,
        *,
        crop: int = 768,
        out_stride: int = 2,
        target_um_per_px: float = 2.0,
        crops_per_image: int = 8,
        background_fraction: float = 0.12,
        input_channels: int = 3,
        base_sigma_um: float = 6.0,
        tip_sigma_um: float = 8.0,
        line_halfwidth_um: float = 3.0,
        seed: int,
    ):
        if len(records) != EXPECTED_TRAIN or any(record.split != "train" for record in records):
            raise RuntimeError("Train399HairCropDataset accepts only the locked 399 train records")
        self.records = tuple(records)
        self.cache_root = Path(cache_root).resolve()
        self.crop = int(crop)
        self.out_stride = int(out_stride)
        self.target_um_per_px = float(target_um_per_px)
        self.crops_per_image = int(crops_per_image)
        self.background_fraction = float(background_fraction)
        self.input_channels = int(input_channels)
        self.base_sigma_um = float(base_sigma_um)
        self.tip_sigma_um = float(tip_sigma_um)
        self.line_halfwidth_um = float(line_halfwidth_um)
        self.seed = int(seed)
        self.geometry: dict[str, dict[str, Any]] = {}
        self._prepare_geometry()

    def _prepare_geometry(self) -> None:
        for record in self.records:
            meta = json.loads(
                (self.cache_root / f"{record.task_id}.meta.json").read_text(
                    encoding="utf-8"
                )
            )
            if meta["source_image_sha256"] != record.image_sha256:
                raise RuntimeError(f"cache/source mismatch for {record.task_id}")
            scale_xy = np.asarray(
                meta["source_to_cached_scale_xy"], dtype=np.float32
            )
            if scale_xy.shape != (2,) or not np.isfinite(scale_xy).all():
                raise RuntimeError(f"invalid realized cache scale for {record.task_id}")
            root = np.asarray(record.root_polygon, dtype=np.float32) * scale_xy
            hairs = []
            for hair in record.hairs:
                points = np.asarray(hair.points, dtype=np.float32) * scale_xy
                hairs.append(
                    {
                        "points": points,
                        "length_um": hair.length_um,
                    }
                )
            bases = (
                np.asarray([hair["points"][0] for hair in hairs], dtype=np.float32)
                if hairs
                else np.zeros((0, 2), dtype=np.float32)
            )
            self.geometry[record.task_id] = {
                "shape": tuple(meta["cached_shape"]),
                "root": root,
                "hairs": hairs,
                "bases": bases,
                "source_to_cached_scale_xy": scale_xy,
                "realized_um_per_px_xy": np.asarray(
                    meta["realized_um_per_px_xy"], dtype=np.float32
                ),
            }

    def __len__(self) -> int:
        return len(self.records) * self.crops_per_image

    def _rng(self, epoch: int, index: int) -> np.random.Generator:
        sequence = np.random.SeedSequence(
            [self.seed & 0xFFFFFFFF, int(epoch) & 0xFFFFFFFF, int(index) & 0xFFFFFFFF]
        )
        return np.random.default_rng(sequence)

    def _sample_origin(
        self, geometry: dict[str, Any], rng: np.random.Generator, source_crop: int
    ) -> tuple[int, int]:
        height, width = geometry["shape"]
        bases = geometry["bases"]
        root = geometry["root"]
        draw = rng.random()
        if len(bases) and draw > self.background_fraction:
            base = bases[rng.integers(len(bases))]
            center_x = base[0] + rng.normal(0, source_crop * 0.18)
            center_y = base[1] + rng.normal(0, source_crop * 0.28)
        elif draw > self.background_fraction * 0.5:
            point = root[rng.integers(len(root))]
            center_x = point[0] + rng.normal(0, source_crop * 0.20)
            center_y = point[1] + rng.normal(0, source_crop * 0.30)
        else:
            center_x = rng.uniform(0, width)
            center_y = rng.uniform(0, height)
        x0 = int(np.clip(center_x - source_crop / 2, 0, max(0, width - source_crop)))
        y0 = int(np.clip(center_y - source_crop / 2, 0, max(0, height - source_crop)))
        return x0, y0

    def __getitem__(self, key: tuple[int, int] | int) -> dict[str, torch.Tensor]:
        if isinstance(key, tuple):
            epoch, index = int(key[0]), int(key[1])
        else:
            epoch, index = 0, int(key)
        record = self.records[index // self.crops_per_image]
        geometry = self.geometry[record.task_id]
        rng = self._rng(epoch, index)
        image = np.load(
            self.cache_root / f"{record.task_id}.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        height, width = geometry["shape"]

        zoom = float(np.exp(rng.normal(0, 0.10)))
        source_crop = int(round(self.crop * zoom))
        if min(height, width) >= 64:
            source_crop = max(64, min(source_crop, min(height, width)))
        else:
            source_crop = self.crop
        x0, y0 = self._sample_origin(geometry, rng, source_crop)
        x1, y1 = min(width, x0 + source_crop), min(height, y0 + source_crop)
        patch = np.asarray(image[y0:y1, x0:x1], dtype=np.uint8)
        if patch.shape[0] < source_crop or patch.shape[1] < source_crop:
            patch = np.pad(
                patch,
                (
                    (0, source_crop - patch.shape[0]),
                    (0, source_crop - patch.shape[1]),
                ),
                mode="edge",
            )
        if source_crop != self.crop:
            patch = cv2.resize(
                patch,
                (self.crop, self.crop),
                interpolation=(
                    cv2.INTER_AREA if source_crop > self.crop else cv2.INTER_CUBIC
                ),
            )
        coordinate_scale = self.crop / source_crop
        effective_um_per_px = self.target_um_per_px / coordinate_scale

        def to_crop(points: np.ndarray) -> np.ndarray:
            returned = points.copy()
            returned[:, 0] = (returned[:, 0] - x0) * coordinate_scale
            returned[:, 1] = (returned[:, 1] - y0) * coordinate_scale
            return returned

        hairs: list[dict[str, Any]] = []
        margin = 8.0
        for hair in geometry["hairs"]:
            points = to_crop(hair["points"])
            base_x, base_y = points[0]
            base_inside = (
                -margin <= base_x < self.crop + margin
                and -margin <= base_y < self.crop + margin
            )
            x_intersects = (points[:, 0] > -margin).any() and (
                points[:, 0] < self.crop + margin
            ).any()
            y_intersects = (points[:, 1] > -margin).any() and (
                points[:, 1] < self.crop + margin
            ).any()
            if base_inside or (x_intersects and y_intersects):
                hairs.append({"points": points, "length_um": hair["length_um"]})
        root_polygon = to_crop(geometry["root"])
        root_mask = np.zeros((self.crop, self.crop), dtype=np.uint8)
        cv2.fillPoly(
            root_mask,
            [np.round(root_polygon).astype(np.int32).reshape(-1, 1, 2)],
            1,
        )

        augmented = patch.astype(np.float32)
        augmented = augmented * float(np.exp(rng.normal(0, 0.06))) + rng.normal(0, 6.0)
        if rng.random() < 0.30:
            augmented = cv2.GaussianBlur(
                augmented, (0, 0), float(rng.uniform(0.4, 1.2))
            )
        if rng.random() < 0.30:
            augmented += rng.normal(
                0, float(rng.uniform(1.0, 4.0)), augmented.shape
            )
        if rng.random() < 0.25:
            grid_y, grid_x = np.mgrid[0 : self.crop, 0 : self.crop].astype(np.float32)
            grid_y /= self.crop
            grid_x /= self.crop
            augmented += rng.normal(0, 12) * grid_x + rng.normal(0, 12) * grid_y
        patch = np.clip(augmented, 0, 255).astype(np.uint8)

        if rng.random() < 0.5:
            patch = patch[:, ::-1].copy()
            root_mask = root_mask[:, ::-1].copy()
            for hair in hairs:
                hair["points"][:, 0] = self.crop - hair["points"][:, 0]
        if rng.random() < 0.5:
            patch = patch[::-1].copy()
            root_mask = root_mask[::-1].copy()
            for hair in hairs:
                hair["points"][:, 1] = self.crop - hair["points"][:, 1]
        rotations = int(rng.integers(4))
        if rotations:
            patch = np.rot90(patch, rotations).copy()
            root_mask = np.rot90(root_mask, rotations).copy()
            for hair in hairs:
                points = hair["points"]
                for _ in range(rotations):
                    points = np.column_stack(
                        (points[:, 1], self.crop - points[:, 0])
                    )
                hair["points"] = points.astype(np.float32)

        inputs = make_input_channels(
            patch, effective_um_per_px, self.input_channels
        )
        targets = build_training_targets(
            hairs,
            root_mask,
            crop=self.crop,
            out_stride=self.out_stride,
            um_per_px=effective_um_per_px,
            base_sigma_um=self.base_sigma_um,
            tip_sigma_um=self.tip_sigma_um,
            line_halfwidth_um=self.line_halfwidth_um,
        )
        returned = {"image": torch.from_numpy(inputs)}
        returned.update(
            {name: torch.from_numpy(value) for name, value in targets.items()}
        )
        return returned


def deterministic_worker_init(worker_id: int) -> None:
    """Seed incidental library RNGs; crop RNG itself is index-derived."""

    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    import random

    random.seed(worker_seed)


def serializable_records(records: Iterable[StageBImageRecord]) -> list[dict[str, Any]]:
    """Small helper for audit/debug output; not used in model checkpoints."""

    return [asdict(record) for record in records]
