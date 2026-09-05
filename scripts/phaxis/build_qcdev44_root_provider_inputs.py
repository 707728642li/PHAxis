#!/usr/bin/env python3
"""Build a label-free QC-development44 root-provider input suite.

The locked QC-development IDs must not be post-hoc filtered after their labels
are visible.  This producer therefore performs byte/geometry/scale validation
only and writes a permissive acquisition *compatibility* gate: corrupt or
unreadable images still fail in the preparer, but acquisition metrics cannot
silently remove a member of the predeclared exact44 cohort.  No annotation
JSON, mask, centreline, phenotype, or condition is read or used for routing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.io import atomic_write_json, sha256_file, sha256_json  # noqa: E402


SCHEMA = "PHAxis-QCdevelopment44-root-provider-input-suite-1.0"
STATUS = "completed_locked_exact44_label_free_source_contract"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TASK = re.compile(r"^RHAUD-[0-9]{3}$")


class QCdevRootInputError(RuntimeError):
    """The exact44 source authorities cannot form a root-provider suite."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QCdevRootInputError(message)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), [dict(row) for row in reader]


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _locked_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    values = [value for value in values if value]
    _require(
        len(values) == 44
        and len(set(values)) == 44
        and all(_TASK.fullmatch(value) is not None for value in values),
        "locked QC-development IDs are not 44 unique RHAUD tasks",
    )
    return values


def build_suite(
    *,
    manifest: Path,
    dataset_root: Path,
    dataset_manifest: Path,
    locked_val_ids: Path,
    output: Path,
) -> dict[str, Any]:
    destination = output.resolve()
    _require(not destination.exists(), f"refusing to overwrite QCdev root suite: {destination}")
    root = dataset_root.resolve()
    _require(root.is_dir(), "canonical443 dataset root is absent")
    for source in (manifest, dataset_manifest, locked_val_ids):
        _require(source.resolve().is_file(), f"QCdev source authority is absent: {source}")

    ids = _locked_ids(locked_val_ids.resolve())
    manifest_fields, manifest_rows = _read_csv(manifest.resolve())
    required = {"task_id", "image_path", "image_sha256", "um_per_px", "source_megapixels"}
    _require(required <= set(manifest_fields), "QCdev inference manifest columns are incomplete")
    _require(
        [row.get("task_id", "").strip() for row in manifest_rows] == ids,
        "QCdev inference manifest order differs from the locked ID authority",
    )
    dataset_fields, dataset_rows = _read_csv(dataset_manifest.resolve())
    required_dataset = {
        "task_id",
        "split",
        "image_relpath",
        "image_sha256",
        "image_width",
        "image_height",
        "source_um_per_px",
        "source_megapixels",
    }
    _require(required_dataset <= set(dataset_fields), "dataset source columns are incomplete")
    dataset_by_task = {row["task_id"].strip(): row for row in dataset_rows}
    _require(len(dataset_by_task) == len(dataset_rows), "dataset manifest task IDs are duplicated")

    source_records: list[dict[str, Any]] = []
    for row in manifest_rows:
        task = row["task_id"].strip()
        _require(task in dataset_by_task, f"{task}: absent from dataset manifest")
        dataset = dataset_by_task[task]
        _require(dataset.get("split") == "val", f"{task}: is not in the locked val split")
        image = Path(row["image_path"]).resolve()
        try:
            relative = image.relative_to(root)
        except ValueError as error:
            raise QCdevRootInputError(f"{task}: image leaves canonical443 root") from error
        digest = str(row["image_sha256"]).casefold()
        _require(_SHA256.fullmatch(digest) is not None, f"{task}: image SHA is invalid")
        _require(
            image.is_file()
            and not image.is_symlink()
            and sha256_file(image) == digest
            and dataset["image_sha256"].casefold() == digest,
            f"{task}: source image byte identity drifted",
        )
        _require(
            Path(dataset["image_relpath"]).as_posix() == relative.as_posix(),
            f"{task}: dataset/image-root relative path differs",
        )
        try:
            width = int(dataset["image_width"])
            height = int(dataset["image_height"])
            scale = float(row["um_per_px"])
            dataset_scale = float(dataset["source_um_per_px"])
            megapixels = float(row["source_megapixels"])
            dataset_megapixels = float(dataset["source_megapixels"])
        except (TypeError, ValueError) as error:
            raise QCdevRootInputError(f"{task}: source geometry is invalid") from error
        _require(
            width > 0
            and height > 0
            and math.isfinite(scale)
            and scale > 0
            and math.isclose(scale, dataset_scale, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(megapixels, width * height / 1e6, rel_tol=0.0, abs_tol=5e-7)
            and math.isclose(megapixels, dataset_megapixels, rel_tol=0.0, abs_tol=5e-7),
            f"{task}: source geometry/scale authority differs",
        )
        source_records.append(
            {
                "task_id": task,
                "image_path": image,
                "image_relpath": relative.as_posix(),
                "image_sha256": digest,
                "width": width,
                "height": height,
                "source_megapixels": megapixels,
                "um_per_px": scale,
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = destination.parent / f".{destination.name}.attempt-{os.getpid()}"
    _require(not attempt.exists(), f"QCdev root-suite attempt already exists: {attempt}")
    attempt.mkdir()
    try:
        raw_rows = [
            {
                "image_id": row["task_id"],
                "input_path": row["image_path"],
                "analysis_mode": "sparse_instance",
                "source_um_per_px": repr(row["um_per_px"]),
                "source_scale_provenance": "raw_image_classical_train399_locked",
            }
            for row in source_records
        ]
        _write_csv(
            attempt / "root_input_manifest.csv",
            (
                "image_id",
                "input_path",
                "analysis_mode",
                "source_um_per_px",
                "source_scale_provenance",
            ),
            raw_rows,
        )
        deployment_rows = [
            {
                "task_id": row["task_id"],
                "image_relpath": row["image_relpath"],
                "image_sha256": row["image_sha256"],
                "width": row["width"],
                "height": row["height"],
                "source_megapixels": repr(row["source_megapixels"]),
                "um_per_px": repr(row["um_per_px"]),
                "scale_provenance": "raw_image_classical_train399_locked",
                "analysis_scale_eligible": "true",
                "review_id": row["task_id"],
                "experiment_key": "",
                "condition_code": "",
                "study_role": "QCdevelopment44_nonindependent_development",
                "developmental_day": "",
                "genotype_or_construct": "",
                "temperature_c": "",
                "qc_disposition": "locked_development_member",
                "eligible_complete_root_geometry": "true",
                "eligible_distal_point_and_first_hair": "true",
                "eligible_local_root_hair_morphology": "true",
                "whole_hair_zone_is_right_censored_by_fov": "true",
            }
            for row in source_records
        ]
        deployment_fields = tuple(deployment_rows[0])
        deployment_manifest_path = attempt / "deployment_manifest.csv"
        _write_csv(deployment_manifest_path, deployment_fields, deployment_rows)
        _write_csv(attempt / "deployment_metadata.csv", deployment_fields, deployment_rows)

        canonical_rows = [
            {
                "biological_unit_id": row["task_id"],
                "canonical_view_selected": "true",
                "acquisition_desirability_score": "",
                "focus_score": "",
                "robust_contrast": "",
            }
            for row in source_records
        ]
        _write_csv(
            attempt / "canonical_unit_manifest.csv",
            tuple(canonical_rows[0]),
            canonical_rows,
        )

        source_lock = [
            {
                "task_id": row["task_id"],
                "image_sha256": row["image_sha256"],
                "width": row["width"],
                "height": row["height"],
                "um_per_px": row["um_per_px"],
            }
            for row in source_records
        ]
        source_lock_identity = sha256_json(source_lock)
        projection = [
            {
                "task_id": row["task_id"],
                "image_relpath": row["image_relpath"],
                "image_sha256": row["image_sha256"],
                "width": row["width"],
                "height": row["height"],
                "um_per_px": row["um_per_px"],
            }
            for row in source_records
        ]
        manifest_sha = sha256_file(deployment_manifest_path)
        deployment_identity = sha256_json(
            {
                "schema_version": "RHAxis-NextGen-deployment-identity-1.0",
                "manifest_sha256": manifest_sha,
                "source_qc_lock_identity_sha256": source_lock_identity,
                "samples": projection,
            }
        )
        deployment_lock = {
            "schema_version": "RHAxis-NextGen-deployment-manifest-lock-1.0",
            "status": "locked_before_phenotype_inference",
            "study": "PHAxis QC-development44 root-provider compatibility",
            "samples": 44,
            "manifest": "deployment_manifest.csv",
            "manifest_sha256": manifest_sha,
            "deployment_identity_sha256": deployment_identity,
            "source_qc_lock_identity_sha256": source_lock_identity,
            "source_inputs": {
                "qcdev_manifest_sha256": sha256_file(manifest.resolve()),
                "dataset_manifest_sha256": sha256_file(dataset_manifest.resolve()),
                "locked_val_ids_sha256": sha256_file(locked_val_ids.resolve()),
            },
            "ordering": "locked QC-development44 ID order",
            "scale_policy": "raw_image_classical_train399_locked_fail_closed",
            "unscaled_units_excluded_from_this_physical_deployment": 0,
            "canonical_annotations_read": False,
            "phenotype_model_predictions_used": False,
            "condition_used_for_model_routing": False,
            "blind_images_used": 0,
        }
        atomic_write_json(attempt / "deployment_manifest_lock.json", deployment_lock)

        acquisition_gate = {
            "schema_version": "RHPheno-acquisition-gate-1.0",
            "phenotype_model_independent": True,
            "manual_phenotype_truth_used": False,
            "reference_image_count": 44,
            "max_side": 3072,
            "calibration": {
                "method": "locked_exact44_no_posthoc_acquisition_exclusion",
                "role": "format compatibility; exact44 membership remains immutable",
                "prediction_or_phenotype_used": False,
            },
            "thresholds": {
                name: {"minimum": 0.0}
                for name in (
                    "robust_contrast",
                    "focus_score",
                    "fine_detail_score",
                    "edge_occupancy",
                    "axis_dominance",
                )
            }
            | {
                name: {"maximum": 1.0e12}
                for name in (
                    "saturation_fraction",
                    "corner_illumination_range",
                    "cross_axis_border_occupancy",
                )
            },
            "blind_images_used": 0,
        }
        atomic_write_json(attempt / "acquisition_gate.json", acquisition_gate)

        artifact_names = (
            "root_input_manifest.csv",
            "deployment_metadata.csv",
            "canonical_unit_manifest.csv",
            "deployment_manifest.csv",
            "deployment_manifest_lock.json",
            "acquisition_gate.json",
        )
        summary: dict[str, Any] = {
            "schema_version": SCHEMA,
            "status": STATUS,
            "tasks": 44,
            "task_ids": ids,
            "source_lock_identity_sha256": source_lock_identity,
            "deployment_identity_sha256": deployment_identity,
            "artifacts": {
                name: {"path": name, "sha256": sha256_file(attempt / name)}
                for name in artifact_names
            },
            "dataset_root": str(root),
            "image_bytes_rehashed": True,
            "locked_members_posthoc_filtered": False,
            "acquisition_gate_can_remove_locked_member": False,
            "labels_or_annotation_files_read": False,
            "condition_metadata_used_for_routing": False,
            "canonical_annotations_read": False,
            "blind_images_used": 0,
        }
        summary["summary_identity_sha256"] = sha256_json(summary)
        atomic_write_json(attempt / "summary.json", summary)
        os.replace(attempt, destination)
        return summary
    except BaseException:
        # Retain a non-official attempt for forensic diagnosis.  No summary can
        # appear at the requested destination before the single directory move.
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--locked-val-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_suite(
            manifest=args.manifest,
            dataset_root=args.dataset_root,
            dataset_manifest=args.dataset_manifest,
            locked_val_ids=args.locked_val_ids,
            output=args.output,
        )
    except (QCdevRootInputError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
