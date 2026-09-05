#!/usr/bin/env python3
"""Build a portable one-image real/nonblind clean-install workflow input.

The source unit is inherited from the result-independent Figure 1 case lock.
The producer projects the full release workflow's root, Stage-B, and trait
source manifests to that one task, copies the hash-identical raw image, and
reseals a relative-path workflow manifest.  It validates a PHAxis plan but
never executes inference.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402
from phaxis.workflow import (  # noqa: E402
    WORKFLOW_MANIFEST_SCHEMA,
    build_analysis_plan,
    load_analysis_manifest,
)


CASE_SCHEMA = "PHAxis-figure1-case-selection-1.0"
CASE_STATUS = "locked_before_model_result_consumption"
SCHEMA_VERSION = "PHAxis-clean-install-sample-input-suite-1.0"
STATUS = "completed_real_nonblind_release_example_manifest"
DEPLOYMENT_LOCK_SCHEMA = "RHAxis-NextGen-deployment-manifest-lock-1.0"


class SampleManifestError(RuntimeError):
    """The full workflow cannot be projected to the locked release example."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SampleManifestError(message)


def _sealed(payload: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field, None)
    _require(isinstance(observed, str) and observed == sha256_json(unsigned), f"{role}: identity mismatch")
    return observed


def _resolve_ref(value: Any, *, base: Path, role: str) -> Path:
    _require(isinstance(value, Mapping), f"{role}: locked file reference is absent")
    supplied = Path(str(value.get("path") or ""))
    path = supplied if supplied.is_absolute() else base / supplied
    path = path.resolve()
    digest = str(value.get("sha256") or "").casefold()
    _require(path.is_file() and not path.is_symlink(), f"{role}: locked file is absent or symlinked")
    _require(sha256_file(path) == digest, f"{role}: locked file SHA-256 mismatch")
    return path


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or ())
    _require(bool(fields) and bool(rows), f"empty CSV authority: {path}")
    return fields, rows


def _task_id(row: Mapping[str, str]) -> str:
    return str(row.get("task_id") or row.get("image_id") or "")


def _source_hash(row: Mapping[str, str]) -> str:
    return str(row.get("image_sha256") or row.get("source_image_sha256") or "").casefold()


def _scale(row: Mapping[str, str]) -> float:
    value = row.get("um_per_px") or row.get("source_um_per_px")
    try:
        observed = float(str(value))
    except (TypeError, ValueError) as error:
        raise SampleManifestError("sample source scale is absent") from error
    _require(observed > 0.0, "sample source scale is invalid")
    return observed


def _project_row(path: Path, task_id: str, role: str) -> tuple[list[str], dict[str, str]]:
    fields, rows = _read_csv(path)
    by_task = {_task_id(row): row for row in rows}
    _require(len(by_task) == len(rows) and task_id in by_task, f"{role}: task set is duplicate or omits sample")
    return fields, dict(by_task[task_id])


def _project_canonical_row(
    path: Path, task_id: str
) -> tuple[list[str], dict[str, str]]:
    _fields, rows = _read_csv(path)
    matches = [
        row
        for row in rows
        if str(
            row.get("biological_unit_id")
            or row.get("task_id")
            or row.get("image_id")
            or ""
        )
        == task_id
    ]
    _require(len(matches) == 1, "root canonical manifest does not uniquely contain sample")
    row = matches[0]
    # The frozen root materializer reaches only these label-free acquisition
    # fields.  Omitting historical source-path columns is essential: those
    # columns point to the authoring workstation and are not runtime inputs.
    fields = [
        "biological_unit_id",
        "canonical_view_selected",
        "acquisition_desirability_score",
        "focus_score",
        "robust_contrast",
    ]
    projected = {
        "biological_unit_id": task_id,
        "canonical_view_selected": str(
            row.get("canonical_view_selected") or "true"
        ),
        "acquisition_desirability_score": str(
            row.get("acquisition_desirability_score") or ""
        ),
        "focus_score": str(row.get("focus_score") or ""),
        "robust_contrast": str(row.get("robust_contrast") or ""),
    }
    _require(
        projected["canonical_view_selected"].casefold() == "true",
        "sample is not the selected canonical root view",
    )
    return fields, projected


def _deployment_identity(
    *, manifest_sha256: str, source_qc_identity: str, row: Mapping[str, str]
) -> str:
    projection = {
        "task_id": str(row["task_id"]),
        "image_relpath": Path(str(row["image_relpath"])).as_posix(),
        "image_sha256": str(row["image_sha256"]).casefold(),
        "width": int(row["width"]),
        "height": int(row["height"]),
        "um_per_px": float(row["um_per_px"]),
    }
    return sha256_json(
        {
            "schema_version": "RHAxis-NextGen-deployment-identity-1.0",
            "manifest_sha256": manifest_sha256,
            "source_qc_lock_identity_sha256": source_qc_identity,
            "samples": [projection],
        }
    )


def _write_csv(path: Path, fields: Sequence[str], row: Mapping[str, str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerow(dict(row))
        handle.flush()
        os.fsync(handle.fileno())


def _relative_ref(path: Path, *, base: Path) -> dict[str, str]:
    return {"path": path.relative_to(base).as_posix(), "sha256": sha256_file(path)}


def build_sample_manifest(
    *,
    analysis_workflow_manifest: str | Path,
    case_selection: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    full_path = Path(analysis_workflow_manifest).resolve()
    case_path = Path(case_selection).resolve()
    destination = Path(output).resolve()
    for role, path in (("analysis workflow manifest", full_path), ("case selection", case_path)):
        _require(path.is_file() and not path.is_symlink(), f"{role} is absent or symlinked")
        _require("blind" not in str(path).casefold(), f"{role} has a blind-labelled path")
    _require(not destination.exists(), f"refusing to overwrite {destination}")

    full = load_analysis_manifest(full_path)
    _require(full.get("schema_version") == WORKFLOW_MANIFEST_SCHEMA, "full workflow schema changed")
    full_identity = str(full["manifest_identity_sha256"])
    base = full_path.parent
    case = read_json(case_path)
    case_identity = _sealed(case, "figure1_case_selection_identity_sha256", "case selection")
    _require(
        case.get("schema_version") == CASE_SCHEMA
        and case.get("status") == CASE_STATUS
        and case.get("selected_before_model_result_consumption") is True
        and case.get("selected_by_prediction_or_trait_outcome") is False
        and case.get("classic_challenge_panel_task") is False
        and case.get("condition_metadata_read") is False
        and case.get("canonical_annotations_read") is False
        and case.get("blind_images_used") == 0,
        "case selection is not the result-independent real release example",
    )
    task_id = str(case.get("task_id") or "")
    source = Path(str(case.get("source_image_path") or "")).resolve()
    source_sha = str(case.get("source_image_sha256") or "").casefold()
    _require(
        bool(task_id)
        and source.is_file()
        and not source.is_symlink()
        and "blind" not in str(source).casefold()
        and sha256_file(source) == source_sha
        and source.stat().st_size == case.get("source_image_bytes"),
        "case source image differs from its prelock",
    )

    root_section = full.get("root_provider")
    stageb_section = full.get("stageb")
    traits_section = full.get("traits")
    _require(
        isinstance(root_section, Mapping)
        and isinstance(stageb_section, Mapping)
        and isinstance(traits_section, Mapping),
        "full workflow source sections are incomplete",
    )
    root_authority = _resolve_ref(root_section.get("input_manifest"), base=base, role="root input manifest")
    root_acquisition_gate = _resolve_ref(
        root_section.get("acquisition_gate"), base=base, role="root acquisition gate"
    )
    root_deployment_metadata = _resolve_ref(
        root_section.get("deployment_metadata"),
        base=base,
        role="root deployment metadata",
    )
    root_canonical_manifest = _resolve_ref(
        root_section.get("canonical_manifest"),
        base=base,
        role="root canonical manifest",
    )
    root_deployment_manifest = _resolve_ref(
        root_section.get("deployment_manifest"),
        base=base,
        role="root deployment manifest",
    )
    root_deployment_lock = _resolve_ref(
        root_section.get("deployment_lock"), base=base, role="root deployment lock"
    )
    root_reference_registry = (
        _resolve_ref(
            root_section.get("reference_registry"),
            base=base,
            role="root reference registry",
        )
        if root_section.get("reference_registry") is not None
        else None
    )
    stageb_authority = _resolve_ref(stageb_section.get("input_manifest"), base=base, role="Stage-B input manifest")
    traits_authority = _resolve_ref(traits_section.get("metadata_csv"), base=base, role="traits metadata")
    root_fields, root_row = _project_row(root_authority, task_id, "root input manifest")
    stageb_fields, stageb_row = _project_row(stageb_authority, task_id, "Stage-B input manifest")
    traits_fields, traits_row = _project_row(traits_authority, task_id, "traits metadata")
    deployment_metadata_fields, deployment_metadata_row = _project_row(
        root_deployment_metadata, task_id, "root deployment metadata"
    )
    deployment_manifest_fields, deployment_manifest_row = _project_row(
        root_deployment_manifest, task_id, "root deployment manifest"
    )
    canonical_fields, canonical_row = _project_canonical_row(
        root_canonical_manifest, task_id
    )
    _require(
        _source_hash(root_row) in {"", source_sha}
        and _source_hash(stageb_row) == source_sha
        and _source_hash(traits_row) == source_sha,
        "sample source hash differs across full workflow authorities",
    )
    scales = (_scale(root_row), _scale(stageb_row), _scale(traits_row))
    _require(max(scales) - min(scales) <= 1e-12, "sample scale differs across full workflow authorities")

    suffixes = "".join(source.suffixes[-2:]) or source.suffix or ".tif"
    sample_filename = "sample_source_image" + suffixes.casefold()
    for field in ("input_path", "image_path"):
        if field in root_row:
            root_row[field] = sample_filename
        if field in stageb_row:
            stageb_row[field] = sample_filename
    for row in (deployment_metadata_row, deployment_manifest_row):
        if "image_relpath" in row:
            row["image_relpath"] = sample_filename

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        inputs = staging / "inputs"
        inputs.mkdir()
        sample_image = inputs / sample_filename
        shutil.copyfile(source, sample_image)
        _require(sha256_file(sample_image) == source_sha, "copied sample source-image hash changed")
        root_manifest = inputs / "root_input_manifest.csv"
        stageb_manifest = inputs / "stageb_input_manifest.csv"
        traits_metadata = inputs / "traits_metadata.csv"
        acquisition_gate = inputs / "acquisition_gate.json"
        deployment_metadata = inputs / "deployment_metadata.csv"
        canonical_manifest = inputs / "canonical_manifest.csv"
        deployment_manifest = inputs / "deployment_manifest.csv"
        deployment_lock = inputs / "deployment_manifest_lock.json"
        reference_registry = inputs / "reference_registry.json"
        _write_csv(root_manifest, root_fields, root_row)
        _write_csv(stageb_manifest, stageb_fields, stageb_row)
        _write_csv(traits_metadata, traits_fields, traits_row)
        _write_csv(
            deployment_metadata,
            deployment_metadata_fields,
            deployment_metadata_row,
        )
        _write_csv(canonical_manifest, canonical_fields, canonical_row)
        _write_csv(
            deployment_manifest,
            deployment_manifest_fields,
            deployment_manifest_row,
        )
        shutil.copyfile(root_acquisition_gate, acquisition_gate)
        if root_reference_registry is not None:
            shutil.copyfile(root_reference_registry, reference_registry)

        source_lock = read_json(root_deployment_lock)
        _require(
            source_lock.get("schema_version") == DEPLOYMENT_LOCK_SCHEMA
            and source_lock.get("status") == "locked_before_phenotype_inference"
            and source_lock.get("blind_images_used") == 0
            and source_lock.get("canonical_annotations_read") is False
            and source_lock.get("phenotype_model_predictions_used") is False
            and source_lock.get("manifest_sha256")
            == sha256_file(root_deployment_manifest),
            "root deployment lock is invalid or differs from its manifest",
        )
        source_qc_identity = str(
            source_lock.get("source_qc_lock_identity_sha256") or ""
        ).casefold()
        _require(
            len(source_qc_identity) == 64,
            "root deployment lock source-QC identity is absent",
        )
        projected_lock = deepcopy(source_lock)
        projected_lock["samples"] = 1
        projected_lock["manifest"] = "deployment_manifest.csv"
        projected_lock["manifest_sha256"] = sha256_file(deployment_manifest)
        projected_lock["deployment_identity_sha256"] = _deployment_identity(
            manifest_sha256=projected_lock["manifest_sha256"],
            source_qc_identity=source_qc_identity,
            row=deployment_manifest_row,
        )
        projected_lock["portable_one_task_projection"] = {
            "task_id": task_id,
            "source_manifest_sha256": sha256_file(root_deployment_manifest),
            "source_lock_sha256": sha256_file(root_deployment_lock),
            "condition_metadata_used_for_model_routing": False,
        }
        atomic_write_json(deployment_lock, projected_lock)

        sample = deepcopy(full)
        sample.pop("manifest_identity_sha256", None)
        sample["root_provider"]["input_manifest"] = _relative_ref(root_manifest, base=staging)
        sample["root_provider"]["image_root"] = "inputs"
        sample["root_provider"]["project"] = "."
        sample["root_provider"].pop("python_executable", None)
        sample["root_provider"]["acquisition_gate"] = _relative_ref(
            acquisition_gate, base=staging
        )
        sample["root_provider"]["deployment_metadata"] = _relative_ref(
            deployment_metadata, base=staging
        )
        sample["root_provider"]["canonical_manifest"] = _relative_ref(
            canonical_manifest, base=staging
        )
        sample["root_provider"]["deployment_manifest"] = _relative_ref(
            deployment_manifest, base=staging
        )
        sample["root_provider"]["deployment_lock"] = _relative_ref(
            deployment_lock, base=staging
        )
        if root_reference_registry is not None:
            sample["root_provider"]["reference_registry"] = _relative_ref(
                reference_registry, base=staging
            )
        sample["stageb"]["input_manifest"] = _relative_ref(stageb_manifest, base=staging)
        sample["stageb"]["image_root"] = "inputs"
        sample["traits"]["metadata_csv"] = _relative_ref(traits_metadata, base=staging)
        if isinstance(sample.get("benchmark_contract"), dict):
            sample["benchmark_contract"]["ordered_raw_source_manifest"] = _relative_ref(
                stageb_manifest, base=staging
            )
        sample["release_example"] = {
            "input_kind": "real_nonblind_release_example",
            "release_authorized": True,
            "development_or_synthetic_smoke": False,
            "tasks": 1,
            "task_id": task_id,
            "source_image_relpath": f"inputs/{sample_filename}",
            "source_image_sha256": source_sha,
            "case_selection_sha256": sha256_file(case_path),
            "case_selection_identity_sha256": case_identity,
            "full_workflow_manifest_sha256": sha256_file(full_path),
            "full_workflow_manifest_identity_sha256": full_identity,
            "blind_images_used": 0,
        }
        sample["manifest_identity_sha256"] = sha256_json(sample)
        sample_path = staging / "release_example_manifest.json"
        atomic_write_json(sample_path, sample)

        plan = build_analysis_plan(
            sample_path,
            output=staging / ".plan-only-output-must-remain-absent",
            review_overlays=False,
        )
        _require(
            plan.get("tasks") == 1
            and plan.get("manifest_identity_sha256") == sample["manifest_identity_sha256"]
            and not (staging / ".plan-only-output-must-remain-absent").exists(),
            "release-example workflow plan validation failed",
        )
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "input_kind": "real_nonblind_release_example",
            "release_authorized": True,
            "development_or_synthetic_smoke": False,
            "tasks": 1,
            "task_id": task_id,
            "source_image_relpath": f"inputs/{sample_filename}",
            "source_image_sha256": source_sha,
            "full_workflow_manifest_sha256": sha256_file(full_path),
            "full_workflow_manifest_identity_sha256": full_identity,
            "case_selection_sha256": sha256_file(case_path),
            "case_selection_identity_sha256": case_identity,
            "release_example_manifest_sha256": sha256_file(sample_path),
            "release_example_manifest_identity_sha256": sample[
                "manifest_identity_sha256"
            ],
            "projected_inputs": {
                "root_input_manifest_sha256": sha256_file(root_manifest),
                "stageb_input_manifest_sha256": sha256_file(stageb_manifest),
                "traits_metadata_sha256": sha256_file(traits_metadata),
                "root_acquisition_gate_sha256": sha256_file(acquisition_gate),
                "root_deployment_metadata_sha256": sha256_file(
                    deployment_metadata
                ),
                "root_canonical_manifest_sha256": sha256_file(canonical_manifest),
                "root_deployment_manifest_sha256": sha256_file(
                    deployment_manifest
                ),
                "root_deployment_lock_sha256": sha256_file(deployment_lock),
                "root_reference_registry_sha256": (
                    sha256_file(reference_registry)
                    if root_reference_registry is not None
                    else None
                ),
            },
            "portable_relative_input_paths": True,
            "portable_data_and_root_authority_paths": True,
            "runtime_python_inherited_not_authoring_environment_pinned": True,
            "model_asset_capsule_finalization_required": True,
            "plan_validated_without_execution": True,
            "model_outputs_read": False,
            "canonical_annotations_read": False,
            "condition_metadata_read": False,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        }
        receipt["sample_input_suite_identity_sha256"] = sha256_json(receipt)
        atomic_write_json(staging / "receipt.json", receipt)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return deepcopy(receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-workflow-manifest", type=Path, required=True)
    parser.add_argument("--case-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_sample_manifest(
        analysis_workflow_manifest=args.analysis_workflow_manifest,
        case_selection=args.case_selection,
        output=args.output,
    )
    print(receipt["sample_input_suite_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
