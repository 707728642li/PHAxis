"""Hash-verified canonical root-hair ground truth for QC development only."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from ..io import read_json, sha256_file, sha256_json

if TYPE_CHECKING:
    from .training_data import StageBImageRecord


CANONICAL_GT_AUTHORITY = (
    "HumanCurated443 canonical vectors with per-file annotation/source-image "
    "hashes, source shape, explicit um/px and physical-geometry identity; "
    "root-hair vertex order oriented by the same root-polygon endpoint-distance "
    "rule as train399 targets"
)


class CanonicalGroundTruthError(RuntimeError):
    """Canonical label bytes, geometry or split identity failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanonicalGroundTruthError(message)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(bool(rows), f"empty CSV: {path}")
    _require(
        len({row.get("task_id") for row in rows}) == len(rows),
        f"duplicate task_id in {path}",
    )
    return rows


def _geometry(record: "StageBImageRecord") -> tuple[dict[str, Any], dict[str, Any]]:
    polylines_source = [np.asarray(hair.points, dtype=np.float64) for hair in record.hairs]
    for index, polyline in enumerate(polylines_source):
        _require(
            polyline.ndim == 2
            and polyline.shape[1:] == (2,)
            and len(polyline) >= 2
            and np.all(np.isfinite(polyline)),
            f"{record.task_id}: invalid canonical root-hair polyline {index}",
        )
        _require(
            np.all(polyline[:, 0] >= 0)
            and np.all(polyline[:, 0] <= record.width)
            and np.all(polyline[:, 1] >= 0)
            and np.all(polyline[:, 1] <= record.height),
            f"{record.task_id}: canonical root-hair coordinate is outside the image",
        )
    if polylines_source:
        base_source = np.asarray([polyline[0] for polyline in polylines_source])
        tip_source = np.asarray([polyline[-1] for polyline in polylines_source])
    else:
        base_source = np.empty((0, 2), dtype=np.float64)
        tip_source = np.empty((0, 2), dtype=np.float64)
    source = {
        "base": base_source.tolist(),
        "tip": tip_source.tolist(),
        "polys": [polyline.tolist() for polyline in polylines_source],
    }
    scale = float(record.source_um_per_px)
    physical = {
        "base": base_source * scale,
        "tip": tip_source * scale,
        "polys": [polyline * scale for polyline in polylines_source],
        "length_um": np.asarray([hair.length_um for hair in record.hairs]),
    }
    return source, physical


def load_canonical_qcdev_ground_truth(
    *,
    dataset_root: str | Path,
    dataset_manifest: str | Path,
    split_manifest: str | Path,
    expected_task_ids: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load only canonical vectors and seal their current byte/geometry identity."""

    # Selection-receipt *validation* is a deployment-time, CPU-only operation
    # and must remain importable before the mandatory GPU preflight.  The
    # training-data module imports torch/cv2/tifffile, so load it only for the
    # development-only operation that actually opens canonical ground truth.
    from .training_data import load_locked_records

    root = Path(dataset_root).resolve()
    dataset_manifest_path = Path(dataset_manifest).resolve()
    split_manifest_path = Path(split_manifest).resolve()
    authoritative_manifest = (root / "manifests" / "dataset_manifest.csv").resolve()
    _require(
        dataset_manifest_path == authoritative_manifest,
        "dataset manifest must be the canonical file inside dataset_root",
    )
    records, dataset_identity = load_locked_records(
        root,
        split_manifest=split_manifest_path,
    )
    _require(
        dataset_identity["dataset_manifest_sha256"] == sha256_file(dataset_manifest_path),
        "canonical dataset manifest identity drifted while loading ground truth",
    )
    _require(
        dataset_identity["split_manifest_sha256"] == sha256_file(split_manifest_path),
        "canonical split manifest identity drifted while loading ground truth",
    )
    metadata_rows = _csv_rows(dataset_manifest_path)
    metadata = {row["task_id"]: row for row in metadata_rows}
    selected = [record for record in records if record.split == "val"]
    task_ids = [record.task_id for record in selected]
    if expected_task_ids is not None:
        _require(
            task_ids == list(expected_task_ids),
            "canonical QC-development task order differs from the locked split",
        )
    _require(len(task_ids) == 44 and len(set(task_ids)) == 44, "canonical GT is not val44")

    returned: dict[str, dict[str, Any]] = {}
    locks: list[dict[str, Any]] = []
    for record in selected:
        row = metadata[record.task_id]
        annotation_relpath = row.get("canonical_annotation_relpath")
        _require(bool(annotation_relpath), f"{record.task_id}: canonical path is missing")
        annotation_path = (root / str(annotation_relpath)).resolve()
        try:
            annotation_path.relative_to(root)
        except ValueError as error:
            raise CanonicalGroundTruthError(
                f"{record.task_id}: canonical path escapes dataset_root"
            ) from error
        _require(annotation_path.is_file(), f"{record.task_id}: canonical JSON is missing")
        observed_sha256 = sha256_file(annotation_path)
        _require(
            observed_sha256 == record.canonical_annotation_sha256,
            f"{record.task_id}: canonical annotation SHA-256 mismatch",
        )
        annotation = read_json(annotation_path)
        _require(
            annotation.get("schema_version") == "RHAxis-human-curated-vector-1.0"
            and annotation.get("task_id") == record.task_id,
            f"{record.task_id}: canonical annotation schema/task mismatch",
        )
        _require(
            annotation.get("dataset_version") == row.get("dataset_version"),
            f"{record.task_id}: canonical annotation dataset version mismatch",
        )
        annotation_image = annotation.get("image")
        _require(
            isinstance(annotation_image, Mapping),
            f"{record.task_id}: canonical source-image identity is missing",
        )
        # In the frozen HumanCurated443 schema this embedded field carries the
        # immutable *raw annotation* SHA (443/443 files), despite its historical
        # name ``image.sha256``.  Source-image bytes are independently bound by
        # dataset_manifest/integrity_sha256 and by ``record.image_sha256`` below.
        _require(
            annotation_image.get("sha256") == record.raw_annotation_sha256
            and int(annotation_image.get("width", -1)) == record.width
            and int(annotation_image.get("height", -1)) == record.height,
            f"{record.task_id}: canonical embedded raw-annotation hash/shape mismatch",
        )
        annotation_scale = float(
            annotation.get("calibration", {}).get("source_um_per_px", np.nan)
        )
        _require(
            np.isfinite(annotation_scale)
            and np.isclose(
                annotation_scale,
                record.source_um_per_px,
                rtol=0.0,
                atol=1e-12,
            ),
            f"{record.task_id}: canonical calibration differs from dataset manifest",
        )
        source_geometry, physical = _geometry(record)
        source_identity = sha256_json(source_geometry)
        root_polygon_identity = sha256_json(
            {"points": [list(point) for point in record.root_polygon]}
        )
        instance_id_order_identity = sha256_json(
            [hair.instance_id for hair in record.hairs]
        )
        physical_json = {
            "base": physical["base"].tolist(),
            "tip": physical["tip"].tolist(),
            "polys": [polyline.tolist() for polyline in physical["polys"]],
            "length_um": physical["length_um"].tolist(),
        }
        physical_identity = sha256_json(physical_json)
        fully_manual = int(row.get("model_prediction_shapes_retained") or 0) == 0
        returned[record.task_id] = {
            **physical,
            "source_um_per_px": record.source_um_per_px,
            "source_image_shape_hw": [record.height, record.width],
            "source_image_sha256": record.image_sha256,
            "canonical_embedded_raw_annotation_sha256": record.raw_annotation_sha256,
            "fully_manual": fully_manual,
            "canonical_annotation_sha256": observed_sha256,
            "root_polygon_source_geometry_identity_sha256": root_polygon_identity,
            "root_hair_instance_id_order_sha256": instance_id_order_identity,
            "oriented_source_geometry_identity_sha256": source_identity,
            "physical_geometry_identity_sha256": physical_identity,
        }
        locks.append(
            {
                "task_id": record.task_id,
                "canonical_annotation_relpath": str(annotation_relpath).replace("\\", "/"),
                "canonical_annotation_sha256": observed_sha256,
                "canonical_annotation_size_bytes": annotation_path.stat().st_size,
                "source_image_sha256": record.image_sha256,
                "canonical_embedded_raw_annotation_sha256": (
                    record.raw_annotation_sha256
                ),
                "source_image_shape_hw": [record.height, record.width],
                "source_um_per_px": record.source_um_per_px,
                "root_hair_count": record.n_hairs,
                "vertex_orders_geometrically_flipped": sum(
                    int(hair.vertex_order_flipped) for hair in record.hairs
                ),
                "root_polygon_source_geometry_identity_sha256": root_polygon_identity,
                "root_hair_instance_id_order_sha256": instance_id_order_identity,
                "oriented_source_geometry_identity_sha256": source_identity,
                "physical_geometry_identity_sha256": physical_identity,
            }
        )
    lock_identity = sha256_json(locks)
    provenance = {
        "authority": CANONICAL_GT_AUTHORITY,
        "coordinate_space": "physical_um_xy",
        "dataset_root": str(root),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "integrity_manifest_sha256": dataset_identity["integrity_manifest_sha256"],
        "canonical_annotation_locks": locks,
        "canonical_ground_truth_lock_identity_sha256": lock_identity,
        "images": 44,
        "root_hairs": sum(lock["root_hair_count"] for lock in locks),
        "blind_images_used": 0,
    }
    return returned, provenance
