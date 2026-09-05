"""Fail-closed post-training release orchestration for PHAxis 1.0.0.

The module is intentionally an orchestrator, not another implementation of
training, inference, fusion, traits, benchmarking, or publication.  A sealed
manifest supplies explicit commands and hash-locked inputs for those existing
entry points.  The default operation is a read-only deterministic plan.  An
execution requires an explicit caller action and commits one hash-closed stage
sentinel at a time.

No torch/CUDA import occurs in this module.  Every GPU command is preceded by
an injected or real ``nvidia-smi`` capacity check, and commands are launched
without a shell.  Production stages consume a run-scoped proposal pin.  The
official model contract is applied by compare-and-swap only after the final
model/evidence receipts exist; post-apply packaging is then verified and an
internal terminal ``release_finalize`` stage seals complete release closure.
"""

from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_policy
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote
import uuid
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement

from .contracts import ContractError
from .hair_stageb.candidate_bundle import (
    CANDIDATE_MANIFEST_SCHEMA,
    CANDIDATE_STATUS,
    FORMAL_TRAIN399_SEEDS,
    build_candidate_manifest,
    validate_candidate_manifest,
)
from .hair_stageb.selection import SELECTION_RECEIPT_SCHEMA
from .io import atomic_write_json, read_json, sha256_file, sha256_json
from .model_contract_binding import (
    APPLIED_OFFICIAL_LIFECYCLE,
    RUN_SCOPED_AUTHORITY_PIN_SCHEMA,
    build_run_scoped_authority_pin,
    read_model_contract_authority,
    read_model_contract_proposal,
    require_output_identity,
)
from .narrative_decision import (
    NarrativeDecisionError,
    validate_narrative_decision,
)
from .publication_titles import title_contract
from .release_topology import (
    MANDATORY_STAGE_ORDER,
    ReleaseTopologyError,
    require_manifest_stage_dependencies,
)
from .root_provider import BundleError, verify_bundle
from .supplementary_tables import (
    BUNDLE_RECEIPT as SUPPLEMENTARY_TABLE_BUNDLE_RECEIPT,
    SupplementaryTableError,
    TABLE_STEMS as SUPPLEMENTARY_TABLE_STEMS,
    validate_supplementary_table_data_bundle,
)


MANIFEST_SCHEMA = "PHAxis-post-training-release-manifest-1.3"
PLAN_SCHEMA = "PHAxis-post-training-release-plan-1.3"
STATE_SCHEMA = "PHAxis-post-training-release-state-1.3"
SENTINEL_SCHEMA = "PHAxis-post-training-release-stage-sentinel-1.3"
TRAINING_RECEIPT_SCHEMA = "PHAxis-StageB-train399-training-receipt-1.0"
ROOT_EXACT283_SCHEMA = "PHAxis-root-provider-fresh-reference283-audit-1.0"
APPLICATION_RECEIPT_SCHEMA = "PHAxis-model-contract-promotion-application-1.0"
DEFERRED_HUMAN_AUTHORITY_CONTRACT_SCHEMA = (
    "PHAxis-deferred-human-authority-contract-1.0"
)
DEFERRED_HUMAN_WORK_ITEM_SCHEMA = "PHAxis-deferred-human-authority-work-item-1.0"
EXPECTED_HUMAN_GATE_EXIT_CODE = 4
EXPECTED_GPU_HOLD_EXIT_CODE = 5
PENDING_RELEASE_AUTHORITY_REGISTRY_SCHEMA = "PHAxis-release-authority-registry-1.0"
PROMOTED_RELEASE_AUTHORITY_REGISTRY_SCHEMA = "PHAxis-release-authority-registry-1.1"
PEP517_SDIST_GENERATED_MEMBERS = (
    "PKG-INFO",
    "setup.cfg",
    "src/phaxis.egg-info/PKG-INFO",
    "src/phaxis.egg-info/SOURCES.txt",
    "src/phaxis.egg-info/dependency_links.txt",
    "src/phaxis.egg-info/entry_points.txt",
    "src/phaxis.egg-info/requires.txt",
    "src/phaxis.egg-info/top_level.txt",
)

PUBLICATION_FIGURE_INPUT_RESOURCE_ROLE_ORDER = (
    "trait_contract",
    "figure1_image",
    "figure1_geometry",
    "development_per_image",
    "development_tolerance",
    "development_threshold",
    "development_strata",
    "assurance_metrics",
    "assurance_pairs",
    "assurance_support",
    "qcdev_assignment",
    "overlay_selection",
    "overlay_audit",
    "phenotype_points",
    "phenotype_effects",
    "narrative_decision",
    "multitrait_atlas",
    "axial_profiles",
    "cohort_flow",
    "workflow_stages",
    "runtime_summary",
    "runtime_per_image",
    "wt_within_experiment_contrasts",
    "wt_within_day_meta_analysis",
    "wt_temperature_qc_flow",
)
PUBLICATION_FIGURE_INPUT_RESOURCE_ROLES = frozenset(
    PUBLICATION_FIGURE_INPUT_RESOURCE_ROLE_ORDER
)
PUBLICATION_FIGURE_INPUT_RESOURCE_CANONICAL_KEY_ORDER = tuple(
    sorted(PUBLICATION_FIGURE_INPUT_RESOURCE_ROLES)
)

GPU_STAGE_NAMES = frozenset(
    {
        "qcdev_candidate_pool",
        "qcdev_evaluation_inference",
        "root_provider_exact283",
        "qcdev_root_provider",
        "production_stageb_exact283",
        "benchmark_phaxis_production",
        "benchmark_frozen_v1_production",
        "benchmark_phaxis_sequential",
        "benchmark_frozen_v1_sequential",
        "clean_install_expected_identity",
        "clean_install",
    }
)

STRICT_PHYSICAL_GPU_ENV = "PHAXIS_REQUIRE_EXACT_PHYSICAL_GPU"
STRICT_PHYSICAL_GPU_STAGE_NAMES = frozenset(
    {
        "root_provider_exact283",
        "qcdev_root_provider",
        "benchmark_phaxis_production",
        "benchmark_frozen_v1_production",
        "benchmark_phaxis_sequential",
        "benchmark_frozen_v1_sequential",
        "clean_install_expected_identity",
        "clean_install",
    }
)
DIRECT_BENCHMARK_STAGE_NAMES = frozenset(
    {
        "benchmark_phaxis_production",
        "benchmark_frozen_v1_production",
        "benchmark_phaxis_sequential",
        "benchmark_frozen_v1_sequential",
    }
)

KNOWN_STAGE_SCHEMAS = {
    "candidate_manifest": CANDIDATE_MANIFEST_SCHEMA,
    "production_manifest": "PHAxis-production-manifest-1.0",
    "release_case_prelocks": "PHAxis-release-case-prelocks-1.0",
    "direct_benchmark_provider_descriptor": "PHAxis-formal-direct-benchmark-provider-descriptor-1.0",
    "qcdev_candidate_pool": (
        "PHAxis-StageB-train399-QCdev44-candidate-pool-run-1.0"
    ),
    "selection": SELECTION_RECEIPT_SCHEMA,
    "qcdev_evaluation_inference": (
        "PHAxis-StageB-train399-QCdev44-evaluation-inference-run-1.0"
    ),
    "qcdev_evaluation": (
        "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2"
    ),
    "root_provider_exact283": ROOT_EXACT283_SCHEMA,
    "root_bundle_materialization": "PHAxis-root-provider-model-bundle-verification-1.0",
    "proposal": "PHAxis-model-contract-1.0.0",
    "authority_pin": RUN_SCOPED_AUTHORITY_PIN_SCHEMA,
    "analysis_workflow_manifest": "PHAxis-analysis-workflow-manifest-1.0",
    "clean_install_sample_manifest": "PHAxis-clean-install-sample-input-suite-1.0",
    "qcdev_root_inputs": "PHAxis-QCdevelopment44-root-provider-input-suite-1.0",
    "qcdev_root_provider": "PHAxis-root-provider-portable-pipeline-1.0",
    "qcdev_fusion": "PHAxis-fusion-run-1.1",
    "production_stageb_exact283": "PHAxis-StageB-inference-run-1.1",
    "fusion_exact283": "PHAxis-fusion-run-1.1",
    "traits_exact283": "PHAxis-trait-export-1.0",
    "cohorts_exact283": "PHAxis-biological-cohorts-1.0",
    "biological_analysis": "PHAxis-exploratory-biological-analysis-1.0",
    "profiles_exact283": "PHAxis-distal-axis-cohort-profile-bundle-1.0.0",
    "profile_analysis": "PHAxis-distal-axis-profile-analysis-1.0.0",
    "historical_oof_evidence": "PHAxis-historical-OOF443-development-receipt-1.0",
    "measurement_assurance": "PHAxis-measurement-assurance-receipt-1.0",
    "overlay_evidence": "PHAxis-manuscript-overlay-selection-receipt-1.2",
    "figure1_geometry_materialization": "PHAxis-figure1-geometry-materialization-1.0",
    "benchmark_phaxis_production": "PHAxis-full-workflow-production-batch-benchmark-1.0",
    "benchmark_frozen_v1_production": "PHAxis-full-workflow-production-batch-benchmark-1.0",
    "benchmark_phaxis_sequential": "PHAxis-full-workflow-sequential-latency-benchmark-1.0",
    "benchmark_frozen_v1_sequential": "PHAxis-full-workflow-sequential-latency-benchmark-1.0",
    "benchmark_production_comparison": "PHAxis-full-workflow-benchmark-comparison-1.0",
    "benchmark_sequential_comparison": "PHAxis-full-workflow-benchmark-comparison-1.0",
    "benchmark_same_hardware": "PHAxis-same-hardware-benchmark-receipt-1.0",
    "benchmark_artifact_inventory": "PHAxis-benchmark-artifact-inventory-1.0",
    "figure_inputs": "PHAxis-publication-figure-input-assembly-1.0",
    "evidence": "PHAxis-manuscript-release-evidence-graph-1.1",
    "figures": "PHAxis-publication-figure-suite-1.0",
    "values": "PHAxis-manuscript-values-1.2",
    "handover": "PHAxis-reuse-handover-build-receipt-1.0",
    "handover_dataset_manifest": "PHAxis-handover-materialisation-plan-1.0",
    "handover_image_manifest": "PHAxis-handover-materialisation-plan-1.0",
    "handover_model_source_manifest": "PHAxis-handover-materialisation-plan-1.0",
    "handover_model_asset_manifest": "PHAxis-handover-materialisation-plan-1.0",
    "handover_benchmark_manifest": "PHAxis-handover-materialisation-plan-1.0",
    "handover_contract": "PHAxis-handover-build-contract-assembly-report-1.0",
    "source_release": "PHAxis-source-release-manifest-2.0",
    "distributions": "PHAxis-release-distributions-1.0",
    "offline_dependencies": "PHAxis-offline-dependency-materialization-1.0",
    "clean_install_expected_identity": "PHAxis-clean-install-reference-output-1.0",
    "clean_install": "PHAxis-clean-install-verification-1.0",
    "manuscript": "PHAxis-manuscript-compile-receipt-1.2",
    "supplementary_manuscript": "PHAxis-supplementary-manuscript-compile-receipt-1.0",
    "submission_docx": "PHAxis-submission-docx-build-2.0",
    "supplementary_docx": "PHAxis-supplementary-docx-build-2.0",
    "manuscript_artifact_qa": "PHAxis-manuscript-artifact-structural-qa-2.0",
    "manuscript_render": "PHAxis-manuscript-pdf-page-render-2.0",
    "manuscript_visual_qa": "PHAxis-manuscript-human-visual-qa-receipt-2.0",
    "official_apply": APPLICATION_RECEIPT_SCHEMA,
    "release_finalize": "PHAxis-post-training-release-finalization-1.0",
}

FINAL_PROMOTION_RECEIPT_IDENTITIES = {
    "production_stageb_exact283": "summary_identity_sha256",
    "fusion_exact283": "summary_identity_sha256",
    "traits_exact283": "export_identity_sha256",
    "evidence": "manifest_identity_sha256",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
EXTERNAL_AUTHORITY_CLASSES = frozenset(
    {
        "immutable_raw_data",
        "frozen_read_only_asset",
        "static_contract",
        "author_metadata",
        "completed_training_authority",
    }
)
_UNSAFE_COMMAND_BASENAMES = frozenset(
    {
        "bash",
        "bash.exe",
        "cmd",
        "cmd.exe",
        "del",
        "erase",
        "kill",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "remove-item",
        "rm",
        "sh",
        "sh.exe",
        "stop-process",
        "suspend-process",
        "taskkill",
        "taskkill.exe",
    }
)


class ReleaseOrchestratorError(RuntimeError):
    """A release prerequisite, command, receipt, or resume Gate failed."""


class _DeferredHumanAuthorityPending(RuntimeError):
    """An expected, recoverable human-authority Gate is not final yet."""


@dataclass(frozen=True)
class _Context:
    manifest_path: Path
    manifest_file_sha256: str
    manifest: dict[str, Any]
    workspace: Path
    run_dir: Path
    external_locks: dict[str, dict[str, Any]]
    artifact_paths: dict[tuple[str, str], Path]
    candidate_preview: dict[str, Any]


CommandRunner = Callable[..., Any]
GpuProbe = Callable[..., Any]
CandidateBuilder = Callable[..., dict[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseOrchestratorError(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def validate_deferred_human_authority_contract(
    spec: Mapping[str, Any],
    *,
    authority_name: str,
) -> None:
    """Validate the byte-independent contract for one future human authority.

    The contract deliberately identifies the future authority without naming
    its not-yet-final hash.  Exact bytes become authoritative only in the
    sentinel of the first stage that actually consumes the file.
    """

    expected_keys = {
        "authority_class",
        "deferred",
        "deferred_contract_schema_version",
        "document_schema_version",
        "draft_template_path",
        "final_status",
        "first_consumer_stage",
        "human_authority_id",
        "identity_field",
        "kind",
        "path",
        "status_field",
    }
    _require(
        set(spec) == expected_keys,
        f"deferred human authority contract fields are invalid: {authority_name}",
    )
    _require(
        spec.get("authority_class") == "author_metadata",
        f"only author_metadata may be deferred: {authority_name}",
    )
    _require(
        spec.get("deferred") is True
        and spec.get("deferred_contract_schema_version")
        == DEFERRED_HUMAN_AUTHORITY_CONTRACT_SCHEMA,
        f"deferred human authority contract schema is invalid: {authority_name}",
    )
    _require(
        spec.get("kind") == "file",
        f"deferred human authority must target one file: {authority_name}",
    )
    for field in (
        "path",
        "draft_template_path",
        "document_schema_version",
        "final_status",
        "first_consumer_stage",
        "human_authority_id",
        "identity_field",
        "status_field",
    ):
        _require(
            isinstance(spec.get(field), str) and bool(str(spec[field]).strip()),
            f"deferred human authority {field} is invalid: {authority_name}",
        )
    _require(
        _SHA256.fullmatch(str(spec["human_authority_id"])) is None,
        f"human_authority_id must be a stable logical ID, not a byte hash: {authority_name}",
    )


def _sealed(payload: Mapping[str, Any], field: str, *, role: str) -> str:
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field, None)
    _require(_is_sha256(observed), f"{role}: {field} is absent or invalid")
    _require(
        sha256_json(unsigned) == observed,
        f"{role}: {field} does not seal the complete JSON object",
    )
    return str(observed)


def _expand(value: str, *, workspace: Path, run_dir: Path) -> str:
    expanded = value.replace("{workspace}", str(workspace)).replace(
        "{run_dir}", str(run_dir)
    ).replace("{python}", sys.executable)
    _require("{" not in expanded and "}" not in expanded, f"unknown placeholder: {value}")
    _require("*" not in expanded and "?" not in expanded, f"glob discovery is forbidden: {value}")
    return expanded


def _resolve_input_path(value: str, *, workspace: Path, run_dir: Path) -> Path:
    expanded = _expand(value, workspace=workspace, run_dir=run_dir)
    path = Path(expanded)
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def _resolve_artifact_path(value: str, *, workspace: Path, run_dir: Path) -> Path:
    expanded = _expand(value, workspace=workspace, run_dir=run_dir)
    path = Path(expanded)
    if not path.is_absolute():
        path = run_dir / path
    resolved = path.resolve()
    _require(
        resolved == run_dir or run_dir in resolved.parents,
        f"stage artifact escapes the new run directory: {value}",
    )
    return resolved


def _directory_lock(path: Path) -> dict[str, Any]:
    _require(path.is_dir(), f"directory input/artifact is missing: {path}")
    records: list[dict[str, Any]] = []
    for member in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        _require(not member.is_symlink(), f"symlink is forbidden in sealed directory: {member}")
        if member.is_file():
            records.append(
                {
                    "path": member.relative_to(path).as_posix(),
                    "size_bytes": member.stat().st_size,
                    "sha256": sha256_file(member),
                }
            )
    return {
        "kind": "directory",
        "sha256": sha256_json(records),
        "files": len(records),
        "size_bytes": sum(record["size_bytes"] for record in records),
        "members": records,
    }


def _path_lock(path: Path, kind: str) -> dict[str, Any]:
    if kind == "file":
        _require(path.is_file(), f"file input/artifact is missing: {path}")
        _require(not path.is_symlink(), f"symlink is forbidden: {path}")
        return {
            "kind": "file",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    _require(kind == "directory", f"unsupported path kind: {kind}")
    return _directory_lock(path)


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Commit a JSON object atomically without ever replacing an existing file."""

    _require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ReleaseOrchestratorError(f"refusing to overwrite: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_training_receipt(
    receipt_path: Path,
    checkpoint_path: Path,
    *,
    seed: int,
) -> None:
    receipt = read_json(receipt_path)
    expected = {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "status": "completed",
        "formal_training": True,
        "seed": seed,
        "epochs": 60,
        "steps_per_epoch": 399,
        "global_steps": 23940,
        "nvidia_smi_preflight_status": "passed",
        "nvidia_smi_training_monitor_status": "passed",
        "validation_evaluated_during_training": False,
        "blind_images_used": 0,
    }
    for field, value in expected.items():
        _require(
            receipt.get(field) == value,
            f"seed {seed} training receipt field mismatch: {field}",
        )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    _require(
        receipt.get("checkpoint_sha256") == checkpoint_sha256,
        f"seed {seed} training receipt/checkpoint SHA mismatch",
    )
    receipt_checkpoint = receipt.get("checkpoint")
    _require(
        isinstance(receipt_checkpoint, str) and bool(receipt_checkpoint),
        f"seed {seed} training receipt checkpoint path is absent",
    )
    receipt_checkpoint_path = Path(receipt_checkpoint)
    if not receipt_checkpoint_path.is_absolute():
        receipt_checkpoint_path = receipt_path.parent / receipt_checkpoint_path
    _require(
        receipt_checkpoint_path.resolve() == checkpoint_path,
        f"seed {seed} training receipt names a different checkpoint",
    )


def _validate_root_exact283(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    _require(payload.get("schema_version") == ROOT_EXACT283_SCHEMA, "root exact283 schema changed")
    _require(payload.get("status") == "pass_exact_283", "root exact283 audit did not pass")
    layers = payload.get("layers")
    expected_layers = {
        "v12_strip_root_mask",
        "v20_root_polygon",
        "final_hybrid_root_mask",
    }
    _require(isinstance(layers, Mapping) and set(layers) == expected_layers, "root exact283 layers changed")
    for name, layer in layers.items():
        _require(
            isinstance(layer, Mapping)
            and layer.get("exact") == 283
            and layer.get("expected") == 283
            and layer.get("mismatch_count") == 0
            and layer.get("mismatch_task_ids") == []
            and layer.get("gate_pass") is True,
            f"root exact283 layer failed: {name}",
        )
    guards = {
        "fresh_portable_raw_image_rerun_completed": True,
        "fresh_283_exact_reproduction_claim_allowed": True,
        "pipeline_raw_image_provenance_gate": True,
        "pipeline_stage_evidence_gate": True,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }
    for field, value in guards.items():
        _require(payload.get(field) == value, f"root exact283 guard changed: {field}")
    _require(payload.get("source_image_mismatch_task_ids") == [], "root exact283 source image mismatch")
    identity_payload = {
        key: payload.get(key)
        for key in (
            "schema_version",
            "reference_identity_sha256",
            "fresh_reference_identity_sha256",
            "bundle_identity_sha256",
            "pipeline_identity_sha256",
            "layers",
            "source_image_mismatch_task_ids",
            "prepared_radius_fallback_task_ids",
            "attachment_supported_extension_rescue_task_ids",
            "pipeline_raw_image_provenance_gate",
            "pipeline_stage_evidence_gate",
        )
    }
    _require(
        _is_sha256(payload.get("audit_identity_sha256"))
        and sha256_json(identity_payload) == payload["audit_identity_sha256"],
        "root exact283 audit identity is invalid",
    )
    for field in (
        "reference_identity_sha256",
        "fresh_reference_identity_sha256",
        "bundle_identity_sha256",
        "pipeline_identity_sha256",
    ):
        _require(_is_sha256(payload.get(field)), f"root exact283 {field} is invalid")
    return payload


def _validate_production_manifest(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    required = {"task_id", "image_path", "image_sha256", "um_per_px", "source_megapixels"}
    _require(len(rows) == 283, "production manifest must contain exactly 283 images")
    _require(required.issubset(rows[0] if rows else {}), "production manifest columns are incomplete")
    task_ids = [row["task_id"] for row in rows]
    _require(all(task_ids) and len(set(task_ids)) == 283, "production manifest task IDs are not 283 unique values")
    locks: list[dict[str, Any]] = []
    total_pixels = 0.0
    for row in rows:
        task_id = row["task_id"]
        image_path = Path(row["image_path"])
        _require(image_path.is_absolute(), f"{task_id}: production image path must be explicit and absolute")
        image_path = image_path.resolve()
        _require(image_path.is_file(), f"{task_id}: production source image is missing")
        observed = sha256_file(image_path)
        _require(observed == row["image_sha256"].casefold(), f"{task_id}: production source image SHA mismatch")
        try:
            um_per_px = float(row["um_per_px"])
            megapixels = float(row["source_megapixels"])
        except (TypeError, ValueError) as error:
            raise ReleaseOrchestratorError(f"{task_id}: invalid physical/image scale") from error
        _require(math.isfinite(um_per_px) and um_per_px > 0.0, f"{task_id}: invalid um_per_px")
        _require(math.isfinite(megapixels) and megapixels > 0.0, f"{task_id}: invalid source_megapixels")
        total_pixels += megapixels
        locks.append(
            {
                "task_id": task_id,
                "image_sha256": observed,
                "um_per_px": um_per_px,
                "source_megapixels": megapixels,
            }
        )
    return {
        "images": 283,
        "manifest_sha256": sha256_file(path),
        "source_set_identity_sha256": sha256_json(locks),
        "source_megapixels": total_pixels,
    }


def _validate_qcdev_inputs(
    *,
    audit: Mapping[str, Any],
    qcdev_manifest: Path,
    locked_val_ids: Path,
) -> None:
    expected = audit.get("excluded_val_ids")
    _require(
        isinstance(expected, list)
        and len(expected) == 44
        and len(set(expected)) == 44,
        "dataset audit does not define exact QCdev44",
    )
    rows = _read_csv(qcdev_manifest)
    observed = [row.get("task_id") for row in rows]
    _require(observed == expected, "QCdev manifest order differs from train399 exclusion lock")
    locked = [value.strip() for value in locked_val_ids.read_text(encoding="utf-8").splitlines() if value.strip()]
    _require(locked == expected, "locked validation IDs differ from train399 exclusion lock")


def _recursive_release_guards(value: Any, *, role: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).casefold()
            if lowered == "blind_images_used":
                _require(item == 0, f"{role}: blind image guard failed")
            if lowered in {
                "canonical_annotations_read",
                "canonical_annotations_read_during_inference",
                "condition_metadata_used_for_routing",
                "condition_metadata_used_for_model_routing",
                "root_cap_region_output",
                "root_cap_region_statistics_included",
            }:
                _require(item is False, f"{role}: release red-line guard failed: {key}")
            _recursive_release_guards(item, role=role)
    elif isinstance(value, list):
        for item in value:
            _recursive_release_guards(item, role=role)


def _is_absolute_host_path(value: str) -> bool:
    return bool(
        re.match(r"^[A-Za-z]:[\\/]", value)
        or value.startswith("\\\\")
        or value.startswith("/")
    )


def _value_at(payload: Mapping[str, Any], dotted: str) -> Any:
    value: Any = payload
    for part in dotted.split("."):
        _require(isinstance(value, Mapping) and part in value, f"receipt field is absent: {dotted}")
        value = value[part]
    return value


def _recover_pre_cas_external_lock_from_stage(
    *,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    run_dir: Path,
    external_name: str,
    cas_path: Path,
    expected_sha256: str,
    sentinel_stage: str,
) -> dict[str, Any]:
    """Recover an immutable pre-CAS lock from this run's producer sentinel.

    A resumed formal run may legitimately observe the official contract after
    ``official_apply`` atomically replaced the predecessor.  The release plan
    must nevertheless remain byte-for-byte anchored to the manifest's pre-CAS
    authority.  The already sealed proposal sentinel is the only run-scoped
    record of that file's original size; accepting the newly observed size
    here would change the plan identity and make post-CAS resume impossible.

    This helper does *not* authorize the applied file.  The existing
    cross-stage CAS validator subsequently reconstructs the application
    receipt from the applied contract and binds it to this run's proposal.
    """

    stages = manifest.get("stages")
    _require(isinstance(stages, list), "cannot recover CAS lock without release stages")
    producer_indices = [
        index
        for index, stage in enumerate(stages)
        if isinstance(stage, Mapping) and stage.get("name") == sentinel_stage
    ]
    _require(
        len(producer_indices) == 1,
        f"cannot recover CAS lock without one {sentinel_stage} stage",
    )
    producer_index = producer_indices[0]
    sentinel_path = _sentinel_path(run_dir, producer_index, sentinel_stage)
    _require(
        sentinel_path.is_file(),
        f"CAS target drifted without this run's sealed {sentinel_stage} sentinel",
    )
    sentinel = read_json(sentinel_path)
    _sealed(
        sentinel,
        "sentinel_identity_sha256",
        role=f"{sentinel_stage} sentinel CAS recovery",
    )
    _require(
        sentinel.get("schema_version") == SENTINEL_SCHEMA
        and sentinel.get("status") == "completed_and_hash_verified"
        and sentinel.get("run_id") == manifest.get("run_id")
        and sentinel.get("manifest_file_sha256") == manifest_file_sha256
        and sentinel.get("stage_index") == producer_index
        and sentinel.get("stage_name") == sentinel_stage,
        f"{sentinel_stage} sentinel cannot recover this release CAS predecessor",
    )
    matches = []
    for lock in sentinel.get("input_locks", []):
        if not isinstance(lock, Mapping) or lock.get("external") != external_name:
            continue
        try:
            lock_path = Path(str(lock.get("path"))).resolve()
        except (OSError, ValueError):
            continue
        if lock_path == cas_path:
            matches.append(lock)
    _require(
        len(matches) == 1,
        f"{sentinel_stage} sentinel CAS predecessor lock is absent or ambiguous",
    )
    lock = matches[0]
    _require(
        lock.get("kind") == "file"
        and lock.get("sha256") == expected_sha256
        and isinstance(lock.get("size_bytes"), int)
        and int(lock["size_bytes"]) >= 0,
        f"{sentinel_stage} sentinel CAS predecessor lock is invalid",
    )
    return {
        "kind": "file",
        "sha256": expected_sha256,
        "size_bytes": int(lock["size_bytes"]),
    }


def _recover_pre_cas_external_lock(
    *,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    run_dir: Path,
    external_name: str,
    cas_path: Path,
    expected_sha256: str,
) -> dict[str, Any]:
    """Backward-compatible official-contract predecessor recovery."""

    return _recover_pre_cas_external_lock_from_stage(
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
        run_dir=run_dir,
        external_name=external_name,
        cas_path=cas_path,
        expected_sha256=expected_sha256,
        sentinel_stage="proposal",
    )


def _validate_promoted_registry_recovery_target(
    path: Path,
    *,
    expected_predecessor_sha256: str,
    manifest: Mapping[str, Any],
    manifest_file_sha256: str,
    workspace: Path,
    run_dir: Path,
) -> None:
    """Reject arbitrary registry drift before reconstructing its predecessor lock."""

    payload = read_json(path)
    _require(
        payload.get("schema_version") == PROMOTED_RELEASE_AUTHORITY_REGISTRY_SCHEMA
        and payload.get("status") == "formal_release_materialized_and_verified",
        "release authority registry drift is not a promoted formal registry",
    )
    _sealed(
        payload,
        "registry_identity_sha256",
        role="promoted release authority registry",
    )
    promotion = payload.get("promotion")
    _require(
        isinstance(promotion, Mapping)
        and promotion.get("predecessor_sha256") == expected_predecessor_sha256
        and promotion.get("run_id") == manifest.get("run_id")
        and promotion.get("manifest_file_sha256") == manifest_file_sha256,
        "promoted release authority registry does not belong to this manifest",
    )
    finalization = payload.get("current_release_finalization")
    _require(
        isinstance(finalization, Mapping)
        and _is_sha256(finalization.get("sha256"))
        and _is_sha256(finalization.get("release_finalization_identity_sha256")),
        "promoted release authority registry lacks a sealed finalization reference",
    )
    finalization_path = Path(str(finalization.get("path", "")))
    if not finalization_path.is_absolute():
        finalization_path = (workspace / finalization_path).resolve()
    _require(
        finalization_path.is_file()
        and run_dir in finalization_path.resolve().parents
        and sha256_file(finalization_path) == finalization["sha256"],
        "promoted release authority registry finalization reference drifted",
    )


def _manifest_context(
    manifest_path: str | Path,
    run_dir: str | Path,
    *,
    candidate_builder: CandidateBuilder,
) -> _Context:
    source = Path(manifest_path).resolve()
    _require(source.is_file(), f"release manifest does not exist: {source}")
    payload = read_json(source)
    manifest_file_sha256 = sha256_file(source)
    _require(payload.get("schema_version") == MANIFEST_SCHEMA, "unsupported release manifest schema")
    _sealed(payload, "manifest_identity_sha256", role="release manifest")
    expected_guards = {
        "product": "PHAxis",
        "product_version": "1.0.0",
        "blind_images_used": 0,
        "canonical_annotations_read": False,
        "condition_metadata_used_for_routing": False,
        "root_cap_region_statistics_included": False,
        "frozen_v1_read_only": True,
    }
    for field, expected in expected_guards.items():
        _require(payload.get(field) == expected, f"release manifest guard changed: {field}")
    run_id = payload.get("run_id")
    _require(isinstance(run_id, str) and bool(run_id.strip()), "release manifest run_id is invalid")
    workspace_raw = payload.get("workspace")
    _require(isinstance(workspace_raw, str) and bool(workspace_raw), "release workspace is absent")
    workspace = Path(_expand(workspace_raw, workspace=source.parent, run_dir=Path(run_dir).resolve()))
    if not workspace.is_absolute():
        workspace = source.parent / workspace
    workspace = workspace.resolve()
    _require(workspace.is_dir(), f"release workspace is missing: {workspace}")
    destination = Path(run_dir).resolve()

    # Identify the sole compare-and-swap target before external authorities
    # are locked.  A post-apply resume is allowed to observe this exact path in
    # its strictly validated applied state; no other external hash may drift.
    raw_stages = payload.get("stages")
    cas_path_for_resume: Path | None = None
    cas_expected_sha256: str | None = None
    registry_cas_path_for_resume: Path | None = None
    registry_cas_expected_sha256: str | None = None
    registry_cas_external_name: str | None = None
    if isinstance(raw_stages, list):
        official_candidates = [
            stage
            for stage in raw_stages
            if isinstance(stage, Mapping) and stage.get("name") == "official_apply"
        ]
        if len(official_candidates) == 1:
            raw_cas = official_candidates[0].get("cas")
            if (
                isinstance(raw_cas, Mapping)
                and isinstance(raw_cas.get("path"), str)
                and _is_sha256(raw_cas.get("expected_sha256"))
            ):
                cas_path_for_resume = _resolve_input_path(
                    str(raw_cas["path"]), workspace=workspace, run_dir=destination
                )
                cas_expected_sha256 = str(raw_cas["expected_sha256"])
        registry_candidates = [
            stage
            for stage in raw_stages
            if isinstance(stage, Mapping) and stage.get("name") == "release_finalize"
        ]
        if len(registry_candidates) == 1:
            raw_registry_cas = registry_candidates[0].get("release_registry_cas")
            if (
                isinstance(raw_registry_cas, Mapping)
                and set(raw_registry_cas) == {"external", "path", "expected_sha256"}
                and isinstance(raw_registry_cas.get("external"), str)
                and isinstance(raw_registry_cas.get("path"), str)
                and _is_sha256(raw_registry_cas.get("expected_sha256"))
            ):
                registry_cas_external_name = str(raw_registry_cas["external"])
                registry_cas_path_for_resume = _resolve_input_path(
                    str(raw_registry_cas["path"]),
                    workspace=workspace,
                    run_dir=destination,
                )
                registry_cas_expected_sha256 = str(
                    raw_registry_cas["expected_sha256"]
                )

    external = payload.get("external_inputs")
    _require(isinstance(external, Mapping) and external, "release external_inputs are absent")
    external_locks: dict[str, dict[str, Any]] = {}
    external_paths: dict[str, Path] = {}
    for name in sorted(external):
        _require(
            isinstance(name, str)
            and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)),
            f"external input name is unsafe: {name}",
        )
        spec = external[name]
        _require(isinstance(spec, Mapping), f"external input is invalid: {name}")
        if spec.get("deferred") is True:
            validate_deferred_human_authority_contract(
                spec,
                authority_name=str(name),
            )
            resolved = _resolve_input_path(
                str(spec["path"]), workspace=workspace, run_dir=destination
            )
            try:
                deferred_relative = resolved.relative_to(destination)
            except ValueError as error:
                raise ReleaseOrchestratorError(
                    f"deferred human authority target leaves this release run: {name}"
                ) from error
            _require(
                len(deferred_relative.parts) == 2
                and deferred_relative.parts[0] == "human_authorities",
                f"deferred human authority target must be run-scoped under "
                f"human_authorities/: {name}",
            )
            draft_template = _resolve_input_path(
                str(spec["draft_template_path"]),
                workspace=workspace,
                run_dir=destination,
            )
            try:
                draft_template.relative_to(workspace)
            except ValueError as error:
                raise ReleaseOrchestratorError(
                    f"deferred human draft template leaves the workspace: {name}"
                ) from error
            external_paths[str(name)] = resolved
            external_locks[str(name)] = {
                **dict(spec),
                "path": str(resolved),
                "draft_template_path": str(draft_template),
            }
            continue
        raw_path = spec.get("path")
        kind = spec.get("kind")
        expected_sha = spec.get("sha256")
        authority_class = spec.get("authority_class")
        _require(isinstance(raw_path, str) and bool(raw_path), f"external path is absent: {name}")
        _require(kind in {"file", "directory"}, f"external kind is invalid: {name}")
        _require(_is_sha256(expected_sha), f"external SHA is invalid: {name}")
        _require(
            authority_class in EXTERNAL_AUTHORITY_CLASSES,
            f"external authority_class is invalid: {name}",
        )
        resolved = _resolve_input_path(raw_path, workspace=workspace, run_dir=destination)
        observed = _path_lock(resolved, str(kind))
        locked = observed
        if observed["sha256"] != expected_sha:
            official_recovery = bool(
                kind == "file"
                and cas_path_for_resume is not None
                and cas_expected_sha256 == expected_sha
                and resolved == cas_path_for_resume
            )
            registry_recovery = bool(
                kind == "file"
                and registry_cas_path_for_resume is not None
                and registry_cas_expected_sha256 == expected_sha
                and registry_cas_external_name == name
                and resolved == registry_cas_path_for_resume
            )
            _require(
                official_recovery or registry_recovery,
                f"external input hash drifted: {name}",
            )
            if official_recovery:
                locked = _recover_pre_cas_external_lock(
                    manifest=payload,
                    manifest_file_sha256=manifest_file_sha256,
                    run_dir=destination,
                    external_name=str(name),
                    cas_path=resolved,
                    expected_sha256=str(expected_sha),
                )
            else:
                _validate_promoted_registry_recovery_target(
                    resolved,
                    expected_predecessor_sha256=str(expected_sha),
                    manifest=payload,
                    manifest_file_sha256=manifest_file_sha256,
                    workspace=workspace,
                    run_dir=destination,
                )
                locked = _recover_pre_cas_external_lock_from_stage(
                    manifest=payload,
                    manifest_file_sha256=manifest_file_sha256,
                    run_dir=destination,
                    external_name=str(name),
                    cas_path=resolved,
                    expected_sha256=str(expected_sha),
                    sentinel_stage="release_finalize",
                )
        external_paths[str(name)] = resolved
        external_locks[str(name)] = {
            "path": str(resolved),
            "authority_class": str(authority_class),
            **locked,
        }

    frozen = payload.get("frozen_v1_inputs")
    _require(isinstance(frozen, list) and frozen, "at least one frozen-v1 input lock is required")
    _require(len(set(frozen)) == len(frozen), "duplicate frozen-v1 input lock")
    for name in frozen:
        _require(name in external_locks, f"unknown frozen-v1 input: {name}")
        _require(
            external_locks[str(name)]["authority_class"]
            == "frozen_read_only_asset",
            f"frozen-v1 input has the wrong authority class: {name}",
        )

    required_names = {}
    for field in (
        "dataset_audit_input",
        "qcdev_manifest_input",
        "locked_val_ids_input",
    ):
        value = payload.get(field)
        _require(isinstance(value, str) and value in external_paths, f"{field} is not a locked external input")
        required_names[field] = value

    members = payload.get("training_members")
    _require(isinstance(members, list) and len(members) == 5, "training_members must contain exactly five seeds")
    member_by_seed: dict[int, tuple[Path, Path]] = {}
    for member in members:
        _require(isinstance(member, Mapping), "invalid training member")
        seed = member.get("seed")
        receipt_name = member.get("completion_receipt_input")
        checkpoint_name = member.get("checkpoint_input")
        _require(seed in FORMAL_TRAIN399_SEEDS, f"unexpected formal seed: {seed}")
        _require(receipt_name in external_paths, f"seed {seed}: unknown completion receipt")
        _require(checkpoint_name in external_paths, f"seed {seed}: unknown checkpoint")
        _require(
            external_locks[str(receipt_name)]["authority_class"]
            == "completed_training_authority"
            and external_locks[str(checkpoint_name)]["authority_class"]
            == "completed_training_authority",
            f"seed {seed}: training inputs have the wrong authority class",
        )
        _require(seed not in member_by_seed, f"duplicate training seed: {seed}")
        receipt_path = external_paths[str(receipt_name)]
        checkpoint_path = external_paths[str(checkpoint_name)]
        _validate_training_receipt(receipt_path, checkpoint_path, seed=int(seed))
        member_by_seed[int(seed)] = (receipt_path, checkpoint_path)
    _require(tuple(sorted(member_by_seed)) == FORMAL_TRAIN399_SEEDS, "formal five-seed set is incomplete")

    audit_path = external_paths[required_names["dataset_audit_input"]]
    checkpoint_paths = [member_by_seed[seed][1] for seed in FORMAL_TRAIN399_SEEDS]
    try:
        candidate_preview = candidate_builder(
            checkpoint_paths,
            dataset_audit_path=audit_path,
        )
        validate_candidate_manifest(candidate_preview)
    except Exception as error:
        raise ReleaseOrchestratorError(
            f"five formal checkpoints failed the existing candidate contract: {error}"
        ) from error
    audit = read_json(audit_path)
    _validate_qcdev_inputs(
        audit=audit,
        qcdev_manifest=external_paths[required_names["qcdev_manifest_input"]],
        locked_val_ids=external_paths[required_names["locked_val_ids_input"]],
    )
    stages = payload.get("stages")
    _require(isinstance(stages, list) and stages, "release stages are absent")
    names = [stage.get("name") if isinstance(stage, Mapping) else None for stage in stages]
    _require(all(isinstance(name, str) and name for name in names), "release stage name is invalid")
    _require(len(set(names)) == len(names), "release stage names are not unique")
    _require(
        tuple(names) == MANDATORY_STAGE_ORDER,
        "release stages must exactly follow the authoritative producer topology",
    )
    _require(
        names[-1] == "release_finalize",
        "release_finalize must be the terminal release-closure stage",
    )
    official_apply_index = names.index("official_apply")
    for external_name, lock in external_locks.items():
        if lock.get("deferred") is not True:
            continue
        declared_consumer = str(lock["first_consumer_stage"])
        _require(
            declared_consumer in names,
            f"deferred human first consumer is not a release stage: {external_name}",
        )
        consumer_indices = [
            index
            for index, stage in enumerate(stages)
            if any(
                isinstance(reference, Mapping)
                and reference.get("external") == external_name
                for reference in stage.get("inputs", [])
            )
        ]
        _require(
            consumer_indices,
            f"deferred human authority is never consumed: {external_name}",
        )
        first_consumer_index = min(consumer_indices)
        _require(
            names[first_consumer_index] == declared_consumer,
            f"deferred human first-consumer contract drifted: {external_name}",
        )
        _require(
            first_consumer_index > official_apply_index,
            f"deferred human authority is referenced before official_apply: {external_name}",
        )
        relative_target = Path(str(lock["path"])).relative_to(destination).as_posix()
        for prior_stage in stages[:first_consumer_index]:
            serialized = json.dumps(prior_stage, ensure_ascii=False)
            _require(
                external_name not in serialized
                and str(lock["path"]) not in serialized
                and relative_target not in serialized.replace("\\", "/"),
                f"pre-consumer stage references deferred human authority: "
                f"{prior_stage.get('name')}/{external_name}",
            )

    artifact_paths: dict[tuple[str, str], Path] = {}
    seen_paths: dict[Path, tuple[str, str]] = {}
    stage_index = {str(name): index for index, name in enumerate(names)}
    for index, stage in enumerate(stages):
        _require(isinstance(stage, Mapping), f"invalid stage at index {index}")
        name = str(stage["name"])
        artifacts = stage.get("artifacts")
        _require(isinstance(artifacts, list) and artifacts, f"{name}: artifacts are absent")
        artifact_names: set[str] = set()
        for artifact in artifacts:
            _require(isinstance(artifact, Mapping), f"{name}: invalid artifact")
            artifact_name = artifact.get("name")
            raw_path = artifact.get("path")
            kind = artifact.get("kind")
            _require(isinstance(artifact_name, str) and artifact_name, f"{name}: artifact name is invalid")
            _require(artifact_name not in artifact_names, f"{name}: duplicate artifact name")
            _require(isinstance(raw_path, str) and raw_path, f"{name}: artifact path is invalid")
            _require(kind in {"file", "directory"}, f"{name}: artifact kind is invalid")
            path = _resolve_artifact_path(raw_path, workspace=workspace, run_dir=destination)
            try:
                relative_artifact = path.relative_to(destination)
            except ValueError as error:
                raise ReleaseOrchestratorError(
                    f"{name}: artifact path leaves the release run directory"
                ) from error
            _require(
                relative_artifact.parts
                and relative_artifact.parts[0] == name,
                f"{name}: artifact path must be owned by its producing stage",
            )
            if path in seen_paths:
                owner = seen_paths[path]
                raise ReleaseOrchestratorError(
                    f"stage artifact path reused by {name}/{artifact_name}: {owner}"
                )
            seen_paths[path] = (name, str(artifact_name))
            artifact_names.add(str(artifact_name))
            artifact_paths[(name, str(artifact_name))] = path
        receipt = stage.get("receipt")
        _require(isinstance(receipt, Mapping), f"{name}: receipt contract is absent")
        receipt_artifact = receipt.get("artifact")
        _require(receipt_artifact in artifact_names, f"{name}: receipt artifact is undeclared")
        _require(
            next(item["kind"] for item in artifacts if item["name"] == receipt_artifact) == "file",
            f"{name}: receipt artifact must be a JSON file",
        )
        schema = receipt.get("schema_version")
        _require(isinstance(schema, str) and schema, f"{name}: receipt schema is absent")
        expected_schema = KNOWN_STAGE_SCHEMAS.get(name)
        if expected_schema is not None:
            _require(schema == expected_schema, f"{name}: receipt schema differs from production contract")
        status_field = receipt.get("status_field", "status")
        _require(isinstance(status_field, str) and status_field, f"{name}: receipt status_field is invalid")
        _require("status" in receipt, f"{name}: receipt expected status is absent")
        required_fields = receipt.get("required_fields", {})
        _require(isinstance(required_fields, Mapping), f"{name}: receipt required_fields is invalid")
        expected_final_identity = FINAL_PROMOTION_RECEIPT_IDENTITIES.get(name)
        if expected_final_identity is not None:
            _require(
                receipt.get("identity_field") == expected_final_identity
                and receipt.get("identity_seals_complete_object") is True,
                f"{name}: final promotion receipt must be completely self-sealed by {expected_final_identity}",
            )

        inputs = stage.get("inputs")
        _require(isinstance(inputs, list), f"{name}: inputs must be an explicit list")
        for reference in inputs:
            _require(isinstance(reference, Mapping), f"{name}: invalid input reference")
            if set(reference) == {"external"}:
                _require(reference["external"] in external_locks, f"{name}: unknown external input")
            elif set(reference) == {"stage", "artifact"}:
                upstream = reference["stage"]
                artifact_name = reference["artifact"]
                _require(upstream in stage_index and stage_index[upstream] < index, f"{name}: input is not from an earlier stage")
                _require((upstream, artifact_name) in artifact_paths, f"{name}: unknown stage artifact input")
            else:
                raise ReleaseOrchestratorError(f"{name}: input reference must be external or stage/artifact")

        command = stage.get("command")
        if name in {"authority_pin", "release_finalize"}:
            _require(command is None, f"{name} is an internal atomic stage")
            _require(
                len(artifacts) == 1,
                f"{name} must have exactly one sealed JSON artifact",
            )
        else:
            _require(isinstance(command, list) and command, f"{name}: command argv is absent")
            _require(
                all(isinstance(token, str) and token for token in command),
                f"{name}: every command argv token must be a non-empty string",
            )
            expanded_command = [
                _expand(token, workspace=workspace, run_dir=destination)
                for token in command
            ]
            _validate_command(expanded_command, stage=name)
        _validate_command_declared_authorities(
            stage,
            index=index,
            stage_index=stage_index,
            external_inputs=external_locks,
        )
        if name == "official_apply":
            _require(
                len(artifacts) == 1,
                "official_apply must have exactly one application-receipt artifact",
            )
        cwd = stage.get("cwd", ".")
        environment = stage.get("environment", {})
        _require(isinstance(cwd, str) and cwd, f"{name}: cwd is invalid")
        _require(
            isinstance(environment, Mapping)
            and all(
                isinstance(key, str)
                and key
                and isinstance(value, str)
                for key, value in environment.items()
            ),
            f"{name}: environment must be a string mapping",
        )
        _require(
            "CUDA_VISIBLE_DEVICES" not in environment,
            f"{name}: CVD belongs only in the explicit GPU mapping",
        )
        if name in STRICT_PHYSICAL_GPU_STAGE_NAMES:
            _require(
                environment.get(STRICT_PHYSICAL_GPU_ENV) == "1",
                f"{name}: formal exact physical-GPU mode is absent",
            )
        gpu = stage.get("gpu")
        if name in GPU_STAGE_NAMES:
            _validate_gpu_spec(gpu, stage=name)
            if name in {
                "qcdev_candidate_pool",
                "qcdev_evaluation_inference",
                "production_stageb_exact283",
            }:
                _require(
                    str(gpu["internal_device"]) in expanded_command,
                    f"{name}: command does not carry the mapped internal CUDA device",
                )
            elif name in {"root_provider_exact283", "qcdev_root_provider"}:
                for flag in ("--v1-physical-gpu", "--q8-physical-gpu"):
                    values = {
                        expanded_command[index + 1]
                        for index, token in enumerate(expanded_command[:-1])
                        if token == flag
                    }
                    _require(
                        values
                        and values
                        <= {str(index) for index in gpu["physical_gpus"]},
                        f"{name}: {flag} leaves the explicit physical GPU mapping",
                    )
                _require(
                    "--strict-physical-gpu" in expanded_command,
                    f"{name}: root provider lacks strict physical-GPU argv",
                )
            else:
                flag_index = (
                    expanded_command.index("--cuda-visible-devices")
                    if "--cuda-visible-devices" in expanded_command
                    else -1
                )
                _require(
                    flag_index >= 0
                    and flag_index + 1 < len(expanded_command)
                    and expanded_command[flag_index + 1]
                    == str(gpu["cuda_visible_devices"]),
                    f"{name}: provider command does not carry the exact CVD mapping",
                )
        else:
            _require(gpu is None, f"{name}: only declared GPU stages may request a GPU")

    _validate_cross_stage_contracts(payload, artifact_paths, workspace, destination)
    return _Context(
        manifest_path=source,
        manifest_file_sha256=manifest_file_sha256,
        manifest=payload,
        workspace=workspace,
        run_dir=destination,
        external_locks=external_locks,
        artifact_paths=artifact_paths,
        candidate_preview=candidate_preview,
    )


def _validate_command(command: Sequence[str], *, stage: str) -> None:
    _require(all(isinstance(token, str) and token for token in command), f"{stage}: command contains an empty token")
    first = Path(command[0]).name.casefold()
    _require(first not in _UNSAFE_COMMAND_BASENAMES, f"{stage}: shell/destructive command is forbidden")
    for index, token in enumerate(command):
        base = Path(token).name.casefold()
        pinned_manuscript_renderer = (
            stage == "manuscript_render"
            and len(command) > 1
            and Path(command[1]).name.casefold() == "render_manuscript_bundle.py"
            and index > 1
            and command[index - 1] == "--powershell"
            and command.count("--powershell") == 1
            and token.casefold() == "powershell.exe"
        )
        _require(
            base not in _UNSAFE_COMMAND_BASENAMES or pinned_manuscript_renderer,
            f"{stage}: destructive command token is forbidden: {token}",
        )
        _require(token.casefold() != "--gpu-reset", f"{stage}: GPU reset is forbidden")
        _require(
            token.casefold() not in {"-c", "/c", "-command"},
            f"{stage}: inline command execution is forbidden",
        )
    joined = " ".join(command).casefold()
    _require("--latest" not in joined and " latest " not in f" {joined} ", f"{stage}: latest discovery is forbidden")
    _require(
        re.search(
            r"(?<![a-z])(kill|taskkill|suspend|stop-process|gpu-reset)(?![a-z])",
            joined,
        )
        is None,
        f"{stage}: kill/suspend/reset semantics are forbidden",
    )


def _validate_command_declared_authorities(
    stage: Mapping[str, Any],
    *,
    index: int,
    stage_index: Mapping[str, int],
    external_inputs: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject command authority that is absent, future, or undeclared.

    Stage inputs are the only admissible authority edges.  Merely embedding an
    external token or another stage's run path in argv must not create a
    second, unsealed dependency graph beside the manifest inputs.
    """

    name = str(stage["name"])
    command = stage.get("command")
    if command is None:
        return
    external_refs = {
        str(reference["external"])
        for reference in stage["inputs"]
        if "external" in reference
    }
    stage_refs = {
        str(reference["stage"])
        for reference in stage["inputs"]
        if "stage" in reference
    }
    joined = json.dumps(command, ensure_ascii=False)
    token_external_refs = set(
        re.findall(r"\{(?:external|external_sha256):([^{}]+)\}", joined)
    )
    _require(
        token_external_refs <= external_refs,
        f"{name}: command uses undeclared external authorities: "
        f"{sorted(token_external_refs - external_refs)}",
    )
    run_owners = set(
        re.findall(r"\{run_dir\}[\\/]+([^\\/={}]+)", joined)
    )
    for owner in sorted(run_owners):
        if owner == "human_authorities":
            _require(
                any(
                    external_inputs.get(external_name, {}).get("deferred") is True
                    for external_name in external_refs
                ),
                f"{name}: run-scoped human authority path is undeclared",
            )
            continue
        _require(owner in stage_index, f"{name}: command uses unknown run-stage path: {owner}")
        if owner == name:
            continue
        _require(
            stage_index[owner] < index,
            f"{name}: command uses future-stage authority: {owner}",
        )
        _require(
            owner in stage_refs,
            f"{name}: command uses undeclared stage authority: {owner}",
        )



def _validate_gpu_spec(gpu: Any, *, stage: str) -> None:
    _require(isinstance(gpu, Mapping), f"{stage}: explicit GPU mapping is required")
    physical = gpu.get("physical_gpus")
    _require(
        isinstance(physical, list)
        and physical
        and len(set(physical)) == len(physical)
        and all(isinstance(index, int) and not isinstance(index, bool) and index >= 0 for index in physical),
        f"{stage}: physical_gpus is invalid",
    )
    expected_cvd = ",".join(str(index) for index in physical)
    _require(gpu.get("cuda_visible_devices") == expected_cvd, f"{stage}: CVD/physical GPU mapping mismatch")
    internal = gpu.get("internal_device")
    _require(isinstance(internal, str) and internal.startswith("cuda:"), f"{stage}: internal_device is invalid")
    try:
        ordinal = int(internal.split(":", 1)[1])
    except (ValueError, IndexError) as error:
        raise ReleaseOrchestratorError(f"{stage}: internal_device is invalid") from error
    _require(0 <= ordinal < len(physical), f"{stage}: internal CUDA ordinal is outside CVD")
    peak = gpu.get("estimated_peak_memory_mib")
    reserve = gpu.get("reserve_memory_mib")
    utilization = gpu.get("maximum_utilization_pct")
    _require(isinstance(peak, int) and not isinstance(peak, bool) and peak > 0, f"{stage}: estimated peak VRAM is invalid")
    _require(isinstance(reserve, int) and not isinstance(reserve, bool) and reserve >= 2048, f"{stage}: GPU reserve must be at least 2048 MiB")
    _require(
        isinstance(utilization, (int, float))
        and not isinstance(utilization, bool)
        and 0.0 <= float(utilization) <= 80.0,
        f"{stage}: utilization Gate must be at most 80%",
    )


def _stage_ref_set(stage: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (str(reference["stage"]), str(reference["artifact"]))
        for reference in stage["inputs"]
        if "stage" in reference
    }


def _command_path_equals(token: str, expected: Path) -> bool:
    try:
        return Path(token).resolve() == expected
    except (OSError, ValueError):
        return False


def _validate_cross_stage_contracts(
    manifest: Mapping[str, Any],
    artifact_paths: Mapping[tuple[str, str], Path],
    workspace: Path,
    run_dir: Path,
) -> None:
    stages = {str(stage["name"]): stage for stage in manifest["stages"]}
    receipt_artifact = {
        name: str(stage["receipt"]["artifact"]) for name, stage in stages.items()
    }

    def external_refs(stage_name: str) -> set[str]:
        return {
            str(reference["external"])
            for reference in stages[stage_name]["inputs"]
            if "external" in reference
        }

    stage_input_names = {
        name: {upstream for upstream, _artifact in _stage_ref_set(stage)}
        for name, stage in stages.items()
    }
    try:
        require_manifest_stage_dependencies(stage_input_names)
    except ReleaseTopologyError as error:
        raise ReleaseOrchestratorError(str(error)) from error

    ordered_names = list(stages)
    ordered_index = {name: index for index, name in enumerate(ordered_names)}
    for stage_name, stage in stages.items():
        command = stage.get("command")
        if command is None:
            continue
        expanded = [
            _expand(str(token), workspace=workspace, run_dir=run_dir)
            for token in command
        ]
        declared_owners = stage_input_names[stage_name]
        for (owner, _artifact), path in artifact_paths.items():
            if owner == stage_name:
                continue
            path_text = str(path)
            if not any(path_text in token for token in expanded):
                continue
            _require(
                ordered_index[owner] < ordered_index[stage_name],
                f"{stage_name}: command uses future-stage authority: {owner}",
            )
            _require(
                owner in declared_owners,
                f"{stage_name}: command uses undeclared stage authority: {owner}",
            )

    forbidden_evaluation_sources = {"qcdev_candidate_pool", "qcdev_evaluation_inference"}
    for stage_name in ("production_stageb_exact283", "fusion_exact283", "traits_exact283"):
        refs = {name for name, _artifact in _stage_ref_set(stages[stage_name])}
        _require(not (refs & forbidden_evaluation_sources), f"{stage_name}: eval-only artifact entered production")

    training_checkpoint_inputs = {
        str(member["checkpoint_input"])
        for member in manifest["training_members"]
    }
    training_receipt_inputs = {
        str(member["completion_receipt_input"])
        for member in manifest["training_members"]
    }
    _require(
        external_refs("candidate_manifest")
        >= training_checkpoint_inputs
        | training_receipt_inputs
        | {str(manifest["dataset_audit_input"])},
        "candidate_manifest stage does not explicitly consume all five completion receipts/checkpoints and the dataset audit",
    )
    for stage_name in (
        "qcdev_candidate_pool",
        "qcdev_evaluation_inference",
        "proposal",
        "production_stageb_exact283",
    ):
        _require(
            external_refs(stage_name) >= training_checkpoint_inputs,
            f"{stage_name}: all five checkpoint file locks must be explicit inputs",
        )
    _require(
        str(manifest["qcdev_manifest_input"])
        in external_refs("qcdev_candidate_pool"),
        "candidate-pool stage does not consume the locked QCdev manifest",
    )
    _require(
        {
            str(manifest["qcdev_manifest_input"]),
            str(manifest["locked_val_ids_input"]),
        }
        <= external_refs("qcdev_evaluation_inference"),
        "evaluation-only inference does not consume both QCdev locks",
    )
    _require(
        "root_provider_exact283"
        in {name for name, _artifact in _stage_ref_set(stages["proposal"])},
        "proposal stage does not consume the stage-produced fresh exact283 root authority",
    )
    _require(
        "production_manifest"
        in {
            name
            for name, _artifact in _stage_ref_set(
                stages["production_stageb_exact283"]
            )
        },
        "production StageB does not consume the stage-produced exact283 manifest",
    )

    pin_path = artifact_paths[("authority_pin", receipt_artifact["authority_pin"])]
    production_command = [
        _expand(str(token), workspace=workspace, run_dir=run_dir)
        for token in stages["production_stageb_exact283"]["command"]
    ]
    _require(
        any(
            _command_path_equals(token, pin_path)
            for token in production_command
        ),
        "production StageB command does not consume the sealed authority pin",
    )
    benchmark_dependencies = {
        name for name, _artifact in _stage_ref_set(stages["benchmark_same_hardware"])
    }
    _require(
        benchmark_dependencies
        >= {
            "benchmark_phaxis_production",
            "benchmark_frozen_v1_production",
            "benchmark_phaxis_sequential",
            "benchmark_frozen_v1_sequential",
            "benchmark_production_comparison",
            "benchmark_sequential_comparison",
        },
        "same-hardware aggregate does not consume all four direct runs and two comparisons",
    )
    apply_stage = stages["official_apply"]
    apply_command = [
        _expand(str(token), workspace=workspace, run_dir=run_dir)
        for token in apply_stage["command"]
    ]
    _require("--apply" in apply_command, "official_apply command lacks --apply")
    cas = apply_stage.get("cas")
    _require(isinstance(cas, Mapping), "official_apply compare-and-swap contract is absent")
    _require(isinstance(cas.get("path"), str) and _is_sha256(cas.get("expected_sha256")), "official_apply CAS path/hash is invalid")
    cas_path = _resolve_input_path(cas["path"], workspace=workspace, run_dir=run_dir)
    _require(cas_path.is_file(), "official model-contract CAS target is missing")
    official_index = list(stages).index("official_apply")
    for stage_name, stage in stages.items():
        if stage_name in {"authority_pin", "proposal", "official_apply", "release_finalize"}:
            continue
        expanded = [
            _expand(str(token), workspace=workspace, run_dir=run_dir)
            for token in stage["command"]
        ]
        if list(stages).index(stage_name) < official_index:
            _require(
                not any(_command_path_equals(token, cas_path) for token in expanded),
                f"{stage_name}: official CAS target cannot enter a pre-apply producer",
            )
    observed_cas_sha256 = sha256_file(cas_path)
    proposal_command = [
        _expand(str(token), workspace=workspace, run_dir=run_dir)
        for token in stages["proposal"]["command"]
    ]
    _require(
        "--apply" not in proposal_command,
        "proposal stage must remain unapplied and cannot carry --apply",
    )
    if observed_cas_sha256 != cas["expected_sha256"]:
        receipt_path = artifact_paths[
            ("official_apply", receipt_artifact["official_apply"])
        ]
        proposal_path = artifact_paths[("proposal", receipt_artifact["proposal"])]
        _require(proposal_path.is_file(), "official CAS target drifted before this run produced a proposal")
        applied = read_model_contract_authority(cas_path)
        proposal = read_model_contract_proposal(proposal_path)
        _require(
            applied.authority_lifecycle == APPLIED_OFFICIAL_LIFECYCLE
            and applied.file_sha256 == proposal.file_sha256
            and applied.identity_sha256 == proposal.identity_sha256,
            "official CAS target does not preserve this run proposal",
        )
        expected_application = _application_receipt_from_official(cas_path)
        _require(
            expected_application["expected_previous_model_contract_sha256"]
            == cas["expected_sha256"]
            and expected_application["final_model_contract_sha256"]
            == observed_cas_sha256,
            "official CAS target/application authority mismatch",
        )
        if receipt_path.exists():
            _require(
                receipt_path.is_file()
                and read_json(receipt_path) == expected_application,
                "existing model-contract application receipt is not recoverable",
            )
    expected_flag = apply_command.index("--expected-current-sha256") if "--expected-current-sha256" in apply_command else -1
    _require(expected_flag >= 0 and expected_flag + 1 < len(apply_command) and apply_command[expected_flag + 1] == cas["expected_sha256"], "official_apply command does not carry the exact CAS SHA")

    finalize_stage = stages["release_finalize"]
    registry_cas = finalize_stage.get("release_registry_cas")
    _require(
        isinstance(registry_cas, Mapping)
        and set(registry_cas) == {"external", "path", "expected_sha256"},
        "release_finalize registry compare-and-swap contract is absent",
    )
    registry_external = registry_cas.get("external")
    _require(
        isinstance(registry_external, str)
        and registry_external == "release_authority_registry"
        and registry_external in external_refs("release_finalize"),
        "release_finalize does not consume the release authority registry",
    )
    registry_lock = manifest["external_inputs"][registry_external]
    registry_path = _resolve_input_path(
        str(registry_cas.get("path")), workspace=workspace, run_dir=run_dir
    )
    _require(
        registry_lock.get("kind") == "file"
        and registry_lock.get("authority_class") == "static_contract"
        and _resolve_input_path(
            str(registry_lock.get("path")), workspace=workspace, run_dir=run_dir
        )
        == registry_path
        and registry_cas.get("expected_sha256") == registry_lock.get("sha256")
        and _is_sha256(registry_cas.get("expected_sha256")),
        "release authority registry CAS path/hash/authority differs from its external lock",
    )
    if sha256_file(registry_path) == registry_cas["expected_sha256"]:
        registry_payload = read_json(registry_path)
        _require(
            registry_payload.get("schema_version")
            == PENDING_RELEASE_AUTHORITY_REGISTRY_SCHEMA
            and registry_payload.get("current_formal_source_release") is None
            and registry_payload.get("current_formal_release_gate_receipt") is None
            and registry_payload.get("blind_images_used") == 0,
            "release authority registry predecessor is not a pending unpromoted authority",
        )


def _application_receipt_from_official(path: Path) -> dict[str, Any]:
    contract = read_json(path)
    binding = read_model_contract_authority(path)
    _require(
        binding.authority_lifecycle == APPLIED_OFFICIAL_LIFECYCLE,
        "application-receipt recovery requires an applied official contract",
    )
    promotion = contract.get("promotion")
    _require(isinstance(promotion, Mapping), "applied official contract lacks promotion evidence")
    sources = promotion.get("final_receipt_source_sha256")
    identities = promotion.get("final_receipt_identity_sha256")
    _require(
        isinstance(sources, Mapping) and isinstance(identities, Mapping),
        "applied official contract lacks final receipt recovery hashes",
    )
    payload: dict[str, Any] = {
        "schema_version": APPLICATION_RECEIPT_SCHEMA,
        "status": "applied",
        "official_model_contract_replaced": True,
        "expected_previous_model_contract_sha256": promotion.get(
            "expected_source_model_contract_sha256"
        ),
        "proposal_file_sha256": promotion.get("proposal_file_sha256"),
        "proposal_identity_sha256": promotion.get("proposal_identity_sha256"),
        "final_model_contract_sha256": sha256_file(path),
        "final_model_contract_identity_sha256": contract.get(
            "model_contract_identity_sha256"
        ),
        "final_evidence_manifest_sha256": sources.get("evidence"),
        "final_evidence_manifest_identity_sha256": identities.get("evidence"),
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }
    for field in (
        "expected_previous_model_contract_sha256",
        "proposal_file_sha256",
        "proposal_identity_sha256",
        "final_model_contract_sha256",
        "final_model_contract_identity_sha256",
        "final_evidence_manifest_sha256",
        "final_evidence_manifest_identity_sha256",
    ):
        _require(_is_sha256(payload.get(field)), f"application recovery field is invalid: {field}")
    payload["application_identity_sha256"] = sha256_json(payload)
    return payload


def _stage_plan(
    context: _Context,
    stage: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    name = str(stage["name"])
    inputs: list[dict[str, Any]] = []
    for reference in stage["inputs"]:
        if "external" in reference:
            external_name = str(reference["external"])
            inputs.append({"external": external_name, **context.external_locks[external_name]})
        else:
            upstream = str(reference["stage"])
            artifact = str(reference["artifact"])
            inputs.append(
                {
                    "stage": upstream,
                    "artifact": artifact,
                    "path": str(context.artifact_paths[(upstream, artifact)]),
                    "hash_source": "validated_upstream_stage_sentinel",
                }
            )
    artifacts = [
        {
            "name": str(artifact["name"]),
            "kind": str(artifact["kind"]),
            "path": str(context.artifact_paths[(name, str(artifact["name"]))]),
        }
        for artifact in stage["artifacts"]
    ]
    command = None
    if stage.get("command") is not None:
        command = [
            _expand(str(token), workspace=context.workspace, run_dir=context.run_dir)
            for token in stage["command"]
        ]
    gpu = deepcopy(stage.get("gpu"))
    raw_registry_cas = stage.get("release_registry_cas")
    release_registry_cas = None
    if raw_registry_cas is not None:
        release_registry_cas = {
            "external": str(raw_registry_cas["external"]),
            "path": str(
                _resolve_input_path(
                    str(raw_registry_cas["path"]),
                    workspace=context.workspace,
                    run_dir=context.run_dir,
                )
            ),
            "expected_sha256": str(raw_registry_cas["expected_sha256"]),
        }
    payload = {
        "index": index,
        "name": name,
        "command": command,
        "cwd": str(
            _resolve_input_path(
                str(stage.get("cwd", ".")),
                workspace=context.workspace,
                run_dir=context.run_dir,
            )
        ),
        "environment": dict(sorted((stage.get("environment") or {}).items())),
        "inputs": inputs,
        "artifacts": artifacts,
        "receipt_contract": deepcopy(dict(stage["receipt"])),
        "gpu": gpu,
        "same_hardware_as": stage.get("same_hardware_as"),
        "cas": deepcopy(stage.get("cas")),
        "release_registry_cas": release_registry_cas,
    }
    payload["stage_plan_identity_sha256"] = sha256_json(payload)
    return payload


def _release_plan_from_context(context: _Context) -> dict[str, Any]:
    stages = [
        _stage_plan(context, stage, index)
        for index, stage in enumerate(context.manifest["stages"])
    ]
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "status": "validated_plan_only_no_commands_started",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "run_id": context.manifest["run_id"],
        "workspace": str(context.workspace),
        "run_dir": str(context.run_dir),
        "manifest_file_sha256": context.manifest_file_sha256,
        "manifest_identity_sha256": context.manifest["manifest_identity_sha256"],
        "external_inputs": context.external_locks,
        "five_seed_candidate_preview": {
            "candidate_bundle_identity_sha256": context.candidate_preview[
                "candidate_bundle_identity_sha256"
            ],
            "candidate_manifest_identity_sha256": context.candidate_preview[
                "candidate_manifest_identity_sha256"
            ],
            "checkpoint_sha256_in_member_order": [
                member["checkpoint_sha256"]
                for member in context.candidate_preview["identity_payload"]["members"]
            ],
        },
        "fresh_root_exact283_gate": "explicit_stage_producer_not_preexisting_external",
        "production_manifest_gate": "explicit_stage_producer_not_preexisting_external",
        "official_apply_policy": (
            "compare_and_swap_after_final_model_evidence_then_post_apply_release_closure"
        ),
        "release_finalize_policy": (
            "terminal_internal_seal_after_source_clean_install_values_manuscript_handover"
        ),
        "deferred_human_authority_policy": (
            "manifest_contract_then_first_consumer_sentinel_exact_byte_activation"
        ),
        "stages": stages,
        "canonical_annotations_read": False,
        "condition_metadata_used_for_routing": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    payload["release_plan_identity_sha256"] = sha256_json(payload)
    return payload


def build_release_plan(
    manifest_path: str | Path,
    run_dir: str | Path,
    *,
    candidate_builder: CandidateBuilder = build_candidate_manifest,
) -> dict[str, Any]:
    """Validate every immutable prerequisite and return a read-only plan."""

    context = _manifest_context(
        manifest_path,
        run_dir,
        candidate_builder=candidate_builder,
    )
    return _release_plan_from_context(context)


def _state_payload(
    plan: Mapping[str, Any],
    *,
    status: str,
    completed: Sequence[str],
    current_stage: str | None,
    failure: Mapping[str, Any] | None = None,
    official_apply_performed: bool | None = None,
    human_authority_gate: Mapping[str, Any] | None = None,
    gpu_hold_gate: Mapping[str, Any] | None = None,
    release_authority_registry_promotion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    applied = (
        "official_apply" in completed
        if official_apply_performed is None
        else bool(official_apply_performed)
    )
    payload: dict[str, Any] = {
        "schema_version": STATE_SCHEMA,
        "status": status,
        "run_id": plan["run_id"],
        "manifest_file_sha256": plan["manifest_file_sha256"],
        "manifest_identity_sha256": plan["manifest_identity_sha256"],
        "release_plan_identity_sha256": plan["release_plan_identity_sha256"],
        "completed_stage_names": list(completed),
        "current_stage": current_stage,
        "official_apply_is_authority_checkpoint": True,
        "official_apply_performed": applied,
        "official_apply_sentinel_committed": "official_apply" in completed,
        "release_finalize_is_final_stage": True,
        "release_finalize_sentinel_committed": "release_finalize" in completed,
        "failure": deepcopy(dict(failure)) if failure is not None else None,
        "human_authority_gate": (
            deepcopy(dict(human_authority_gate))
            if human_authority_gate is not None
            else None
        ),
        "gpu_hold_gate": (
            deepcopy(dict(gpu_hold_gate))
            if gpu_hold_gate is not None
            else None
        ),
        "release_authority_registry_promotion": (
            deepcopy(dict(release_authority_registry_promotion))
            if release_authority_registry_promotion is not None
            else None
        ),
        "algorithm_or_training_failure": (
            False
            if human_authority_gate is not None or gpu_hold_gate is not None
            else None
        ),
        "all_pending_deferred_authorities": (
            deepcopy(
                list(
                    human_authority_gate.get(
                        "all_pending_deferred_authorities", []
                    )
                )
            )
            if human_authority_gate is not None
            else []
        ),
        "blind_images_used": 0,
    }
    payload["state_identity_sha256"] = sha256_json(payload)
    return payload


def _sentinel_path(run_dir: Path, index: int, name: str) -> Path:
    return run_dir / "sentinels" / f"{index:02d}_{name}.json"


def _deferred_human_authority_lock(
    descriptor: Mapping[str, Any],
    *,
    external_name: str,
    activation_origin_stage: str,
) -> dict[str, Any]:
    """Validate and exact-lock one final run-scoped human authority."""

    path = Path(str(descriptor["path"]))
    if not path.exists():
        raise _DeferredHumanAuthorityPending("target file is absent")
    _require(
        path.is_file() and not path.is_symlink(),
        f"deferred human authority target is not a regular file: {external_name}",
    )
    try:
        payload = read_json(path)
    except (OSError, ValueError, TypeError) as error:
        raise _DeferredHumanAuthorityPending(
            f"target is not a readable JSON object: {type(error).__name__}"
        ) from error
    status_field = str(descriptor["status_field"])
    if payload.get(status_field) != descriptor["final_status"]:
        raise _DeferredHumanAuthorityPending(
            f"{status_field} is not {descriptor['final_status']}"
        )
    _require(
        payload.get("schema_version") == descriptor["document_schema_version"],
        f"final deferred human authority schema drifted: {external_name}",
    )
    identity_field = str(descriptor["identity_field"])
    logical_identity = _sealed(
        payload,
        identity_field,
        role=f"deferred human authority {external_name}",
    )
    exact = _path_lock(path, "file")
    return {
        "external": external_name,
        "path": str(path.resolve()),
        "kind": "file",
        "authority_class": "author_metadata",
        "deferred_authority_activated": True,
        "deferred_contract_schema_version": descriptor[
            "deferred_contract_schema_version"
        ],
        "human_authority_id": descriptor["human_authority_id"],
        "document_schema_version": descriptor["document_schema_version"],
        "status_field": status_field,
        "final_status": descriptor["final_status"],
        "identity_field": identity_field,
        "logical_identity_sha256": logical_identity,
        "activation_origin_stage": activation_origin_stage,
        "exact_file_hash_locked": True,
        **exact,
    }


def _validate_deferred_activation_lock(lock: Mapping[str, Any]) -> None:
    """Revalidate the exact authority bytes named by a completed sentinel."""

    _require(
        lock.get("deferred_authority_activated") is True
        and lock.get("authority_class") == "author_metadata"
        and lock.get("kind") == "file"
        and lock.get("exact_file_hash_locked") is True
        and _is_sha256(lock.get("sha256"))
        and _is_sha256(lock.get("logical_identity_sha256")),
        "deferred human activation lock is incomplete",
    )
    path = Path(str(lock.get("path")))
    _require(
        path.is_file() and not path.is_symlink(),
        f"activated deferred human authority is missing: {path}",
    )
    observed = _path_lock(path, "file")
    _require(
        observed.get("sha256") == lock.get("sha256")
        and observed.get("size_bytes") == lock.get("size_bytes"),
        f"activated deferred human authority exact bytes drifted: {path}",
    )
    payload = read_json(path)
    _require(
        payload.get("schema_version") == lock.get("document_schema_version")
        and payload.get(str(lock.get("status_field"))) == lock.get("final_status"),
        f"activated deferred human authority schema/status drifted: {path}",
    )
    _require(
        _sealed(
            payload,
            str(lock.get("identity_field")),
            role=f"activated deferred human authority {lock.get('human_authority_id')}",
        )
        == lock.get("logical_identity_sha256"),
        f"activated deferred human authority self-identity drifted: {path}",
    )


def _prior_deferred_activation(
    context: _Context,
    plan: Mapping[str, Any],
    *,
    external_name: str,
    before_index: int,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for earlier in plan["stages"][:before_index]:
        sentinel_path = _sentinel_path(
            context.run_dir, int(earlier["index"]), str(earlier["name"])
        )
        if not sentinel_path.is_file():
            continue
        sentinel = read_json(sentinel_path)
        _sealed(
            sentinel,
            "sentinel_identity_sha256",
            role=f"{earlier['name']} sentinel deferred-authority lookup",
        )
        _require(
            sentinel.get("release_plan_identity_sha256")
            == plan["release_plan_identity_sha256"],
            "deferred-authority activation belongs to another release plan",
        )
        for raw_lock in sentinel.get("input_locks", []):
            if (
                isinstance(raw_lock, Mapping)
                and raw_lock.get("external") == external_name
                and raw_lock.get("deferred_authority_activated") is True
            ):
                matches.append(dict(raw_lock))
    if not matches:
        return None
    first = matches[0]
    for match in matches:
        _require(
            match == first,
            f"deferred human authority has conflicting prior activations: {external_name}",
        )
    _validate_deferred_activation_lock(first)
    return first


def _deferred_work_item_path(
    run_dir: Path, *, stage_index: int, external_name: str
) -> Path:
    return (
        run_dir
        / "human_authority_work_items"
        / f"{stage_index:02d}_{external_name}.json"
    )


def _copy_deferred_draft_if_possible(descriptor: Mapping[str, Any]) -> bool:
    """Create a run-scoped editable draft; it is never release authority yet."""

    target = Path(str(descriptor["path"]))
    if target.exists():
        return False
    draft = Path(str(descriptor["draft_template_path"]))
    if not draft.is_file() or draft.is_symlink():
        return False
    try:
        payload = read_json(draft)
    except (OSError, ValueError, TypeError):
        return False
    if payload.get("schema_version") != descriptor["document_schema_version"]:
        return False
    _atomic_write_new_json(target, payload)
    return True


def _ensure_deferred_work_item(
    context: _Context,
    plan: Mapping[str, Any],
    plan_stage: Mapping[str, Any],
    *,
    external_name: str,
    pending_reason: str,
) -> tuple[Path, dict[str, Any]]:
    descriptor = context.external_locks[external_name]
    copied = _copy_deferred_draft_if_possible(descriptor)
    path = _deferred_work_item_path(
        context.run_dir,
        stage_index=int(plan_stage["index"]),
        external_name=external_name,
    )
    payload: dict[str, Any] = {
        "schema_version": DEFERRED_HUMAN_WORK_ITEM_SCHEMA,
        "status": "action_required_expected_human_gate_not_success_artifact",
        "counts_as_completed_stage": False,
        "counts_as_formal_release_success": False,
        "expected_pause_not_algorithm_or_training_failure": True,
        "resume_command_required_after_completion": True,
        "expected_process_exit_code": EXPECTED_HUMAN_GATE_EXIT_CODE,
        "run_id": plan["run_id"],
        "manifest_file_sha256": plan["manifest_file_sha256"],
        "release_plan_identity_sha256": plan["release_plan_identity_sha256"],
        "blocked_stage_index": plan_stage["index"],
        "blocked_stage_name": plan_stage["name"],
        "external_name": external_name,
        "human_authority_id": descriptor["human_authority_id"],
        "target_path": descriptor["path"],
        "document_schema_version": descriptor["document_schema_version"],
        "status_field": descriptor["status_field"],
        "required_final_status": descriptor["final_status"],
        "identity_field": descriptor["identity_field"],
        "draft_template_path": descriptor["draft_template_path"],
        "run_scoped_draft_created": copied,
        "pending_reason_at_creation": pending_reason,
        "instructions": (
            "Edit only target_path, replace all placeholders, set the required "
            "final status, recompute identity_field over the complete JSON object "
            "with identity_field omitted, then rerun the same manifest/output with "
            "--execute --resume. Do not edit this work item."
        ),
        "blind_images_used": 0,
        "work_item_is_not_success_artifact": True,
    }
    payload["work_item_identity_sha256"] = sha256_json(payload)
    if path.is_file():
        existing = read_json(path)
        _sealed(
            existing,
            "work_item_identity_sha256",
            role=f"deferred work item {external_name}",
        )
        for field in (
            "schema_version",
            "status",
            "run_id",
            "manifest_file_sha256",
            "release_plan_identity_sha256",
            "blocked_stage_index",
            "blocked_stage_name",
            "external_name",
            "human_authority_id",
            "target_path",
            "document_schema_version",
            "required_final_status",
            "identity_field",
            "work_item_is_not_success_artifact",
        ):
            _require(
                existing.get(field) == payload.get(field),
                f"deferred work item drifted: {external_name}/{field}",
            )
        return path, existing
    _atomic_write_new_json(path, payload)
    return path, payload


def _pending_deferred_authority(
    context: _Context,
    plan: Mapping[str, Any],
    plan_stage: Mapping[str, Any],
) -> tuple[str, str] | None:
    for reference in plan_stage["inputs"]:
        if "external" not in reference:
            continue
        external_name = str(reference["external"])
        descriptor = context.external_locks[external_name]
        if descriptor.get("deferred") is not True:
            continue
        prior = _prior_deferred_activation(
            context,
            plan,
            external_name=external_name,
            before_index=int(plan_stage["index"]),
        )
        if prior is not None:
            continue
        try:
            _deferred_human_authority_lock(
                descriptor,
                external_name=external_name,
                activation_origin_stage=str(plan_stage["name"]),
            )
        except _DeferredHumanAuthorityPending as pending:
            return external_name, str(pending)
    return None


def _all_pending_deferred_work_items(
    context: _Context,
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Materialize every currently incomplete human task at the first Gate."""

    plan_stage_by_name = {
        str(stage["name"]): stage for stage in plan["stages"]
    }
    ordered = sorted(
        (
            (name, descriptor)
            for name, descriptor in context.external_locks.items()
            if descriptor.get("deferred") is True
        ),
        key=lambda item: int(
            plan_stage_by_name[str(item[1]["first_consumer_stage"])]["index"]
        ),
    )
    pending_items: list[dict[str, Any]] = []
    for external_name, descriptor in ordered:
        prior = _prior_deferred_activation(
            context,
            plan,
            external_name=external_name,
            before_index=len(plan["stages"]),
        )
        if prior is not None:
            continue
        consumer_stage = plan_stage_by_name[str(descriptor["first_consumer_stage"])]
        try:
            _deferred_human_authority_lock(
                descriptor,
                external_name=external_name,
                activation_origin_stage=str(consumer_stage["name"]),
            )
        except _DeferredHumanAuthorityPending as pending:
            reason = str(pending)
        except ReleaseOrchestratorError as invalid_final:
            # A future consumer has not used these bytes.  Surface the exact
            # correction in the same batch work list instead of creating a
            # second surprise stop later.  The current consumer is separately
            # checked before this helper and still fails closed if it claims a
            # final but invalid authority.
            reason = f"final-looking target is invalid: {invalid_final}"
        else:
            continue
        work_item_path, work_item = _ensure_deferred_work_item(
            context,
            plan,
            consumer_stage,
            external_name=external_name,
            pending_reason=reason,
        )
        pending_items.append(
            {
                "status": "action_required_not_success_artifact",
                "external_name": external_name,
                "human_authority_id": descriptor["human_authority_id"],
                "first_consumer_stage": descriptor["first_consumer_stage"],
                "target_path": descriptor["path"],
                "work_item_path": str(work_item_path),
                "work_item_sha256": sha256_file(work_item_path),
                "work_item_identity_sha256": work_item[
                    "work_item_identity_sha256"
                ],
                "work_item_is_not_success_artifact": True,
            }
        )
    return pending_items


def _refresh_external_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": lock["path"],
        "authority_class": lock["authority_class"],
        **_path_lock(Path(lock["path"]), str(lock["kind"])),
    }


def _validate_frozen_inputs(context: _Context) -> dict[str, dict[str, Any]]:
    locks: dict[str, dict[str, Any]] = {}
    for name in context.manifest["frozen_v1_inputs"]:
        observed = _refresh_external_lock(context.external_locks[name])
        _require(observed == context.external_locks[name], f"frozen-v1 input was modified: {name}")
        locks[str(name)] = observed
    return locks


def _validate_official_contract_guard(
    context: _Context,
    plan_stage: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Require predecessor authority before CAS and this run's authority after."""

    official_stage = next(
        stage
        for stage in context.manifest["stages"]
        if stage["name"] == "official_apply"
    )
    cas = official_stage["cas"]
    path = _resolve_input_path(
        str(cas["path"]),
        workspace=context.workspace,
        run_dir=context.run_dir,
    )
    observed = sha256_file(path)
    current_index = int(plan_stage["index"])
    official_index = next(
        index
        for index, stage in enumerate(context.manifest["stages"])
        if stage["name"] == "official_apply"
    )
    if current_index == official_index:
        return None
    if current_index < official_index:
        _require(
            observed == cas["expected_sha256"],
            "official model contract changed before the authority CAS checkpoint",
        )
        return {
            "path": str(path),
            "sha256": observed,
            "authority_phase": "pending_predecessor_read_only",
        }

    _require(
        observed != cas["expected_sha256"],
        "post-apply producer started before the authority CAS checkpoint",
    )
    stages = {
        str(stage["name"]): stage for stage in context.manifest["stages"]
    }
    proposal_path = context.artifact_paths[
        ("proposal", str(stages["proposal"]["receipt"]["artifact"]))
    ]
    applied = read_model_contract_authority(path)
    proposal = read_model_contract_proposal(proposal_path)
    _require(
        applied.authority_lifecycle == APPLIED_OFFICIAL_LIFECYCLE
        and applied.file_sha256 == proposal.file_sha256
        and applied.identity_sha256 == proposal.identity_sha256,
        "post-apply official authority does not preserve this run proposal",
    )
    recovered = _application_receipt_from_official(path)
    _require(
        recovered["expected_previous_model_contract_sha256"]
        == cas["expected_sha256"]
        and recovered["final_model_contract_sha256"] == observed,
        "post-apply official authority does not recover this release CAS",
    )
    receipt_path = context.artifact_paths[
        ("official_apply", str(stages["official_apply"]["receipt"]["artifact"]))
    ]
    _require(
        receipt_path.is_file() and read_json(receipt_path) == recovered,
        "post-apply application receipt is absent or differs from official authority",
    )
    return {
        "path": str(path),
        "sha256": observed,
        "authority_phase": "applied_for_post_apply_release_closure",
        "application_identity_sha256": recovered["application_identity_sha256"],
    }


def _input_locks(
    context: _Context,
    plan: Mapping[str, Any],
    plan_stage: Mapping[str, Any],
    *,
    official_guard: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    locks: list[dict[str, Any]] = []
    for reference in plan_stage["inputs"]:
        if "external" in reference:
            name = str(reference["external"])
            if context.external_locks[name].get("deferred") is True:
                prior = _prior_deferred_activation(
                    context,
                    plan,
                    external_name=name,
                    before_index=int(plan_stage["index"]),
                )
                if prior is None:
                    observed_deferred = _deferred_human_authority_lock(
                        context.external_locks[name],
                        external_name=name,
                        activation_origin_stage=str(plan_stage["name"]),
                    )
                else:
                    observed_deferred = prior
                locks.append(observed_deferred)
                continue
            observed = _refresh_external_lock(context.external_locks[name])
            expected = context.external_locks[name]
            if observed != expected:
                # The one deliberately mutable external authority is the
                # official model-contract CAS target.  Its manifest lock is
                # necessarily the pre-apply predecessor, while post-apply
                # producers must consume the newly applied authority.  The
                # caller has already reconstructed and validated that
                # authority against this run's proposal and application
                # receipt; no other external drift is accepted here.
                _require(
                    isinstance(official_guard, Mapping)
                    and official_guard.get("authority_phase")
                    == "applied_for_post_apply_release_closure"
                    and Path(str(expected["path"])).resolve()
                    == Path(str(official_guard.get("path"))).resolve()
                    and observed.get("sha256") == official_guard.get("sha256"),
                    f"stage input drifted: {name}",
                )
                locks.append(
                    {
                        "external": name,
                        **observed,
                        "authority_phase": (
                            "applied_for_post_apply_release_closure"
                        ),
                    }
                )
            else:
                locks.append({"external": name, **observed})
        else:
            upstream = str(reference["stage"])
            artifact = str(reference["artifact"])
            upstream_index = next(
                int(item["index"])
                for item in plan["stages"]
                if item["name"] == upstream
            )
            sentinel = _validate_sentinel(
                context,
                plan,
                upstream_index,
            )
            artifact_lock = next(
                item for item in sentinel["artifact_locks"] if item["name"] == artifact
            )
            locks.append(
                {
                    "stage": upstream,
                    "artifact": artifact,
                    "path": artifact_lock["path"],
                    "kind": artifact_lock["kind"],
                    "sha256": artifact_lock["sha256"],
                    "size_bytes": artifact_lock["size_bytes"],
                }
            )
    return locks


def _default_command_runner(*, command: Sequence[str], cwd: Path, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _default_gpu_probe(*, stage: str) -> subprocess.CompletedProcess[str]:
    del stage
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _result_fields(result: Any) -> tuple[int, str, str]:
    if isinstance(result, int):
        return result, "", ""
    if isinstance(result, Mapping):
        return int(result.get("returncode", -1)), str(result.get("stdout", "")), str(result.get("stderr", ""))
    return int(getattr(result, "returncode")), str(getattr(result, "stdout", "") or ""), str(getattr(result, "stderr", "") or "")


def _parse_gpu_probe(result: Any, gpu: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    returncode, stdout, stderr = _result_fields(result)
    _require(returncode == 0, f"{stage}: nvidia-smi preflight failed: {stderr[-500:]}")
    devices: dict[int, dict[str, Any]] = {}
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        _require(len(parts) == 7, f"{stage}: malformed nvidia-smi output")
        try:
            index = int(parts[0])
            total = int(float(parts[3]))
            used = int(float(parts[4]))
            utilization = float(parts[5])
        except ValueError as error:
            raise ReleaseOrchestratorError(f"{stage}: malformed nvidia-smi numeric field") from error
        devices[index] = {
            "physical_index": index,
            "uuid": parts[1],
            "name": parts[2],
            "memory_total_mib": total,
            "memory_used_mib": used,
            "memory_free_mib": total - used,
            "utilization_pct": utilization,
            "driver_version": parts[6],
        }
    selected: list[dict[str, Any]] = []
    required = int(gpu["estimated_peak_memory_mib"]) + int(gpu["reserve_memory_mib"])
    maximum = float(gpu["maximum_utilization_pct"])
    for index in gpu["physical_gpus"]:
        _require(index in devices, f"{stage}: requested physical GPU is absent: {index}")
        device = devices[index]
        _require(device["memory_free_mib"] >= required, f"{stage}: GPU {index} has insufficient free VRAM")
        _require(device["utilization_pct"] <= maximum, f"{stage}: GPU {index} exceeds utilization Gate")
        selected.append(device)
    return {
        "schema_version": "PHAxis-release-orchestrator-nvidia-smi-preflight-1.0",
        "status": "passed",
        "stage": stage,
        "cuda_visible_devices": gpu["cuda_visible_devices"],
        "internal_device": gpu["internal_device"],
        "estimated_peak_memory_mib": gpu["estimated_peak_memory_mib"],
        "reserve_memory_mib": gpu["reserve_memory_mib"],
        "maximum_utilization_pct": maximum,
        "selected_devices": selected,
        "nvidia_smi_stdout_sha256": sha256_json(stdout.splitlines()),
    }


def _stable_hardware(preflight: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "physical_index": row["physical_index"],
            "uuid": row["uuid"],
            "name": row["name"],
            "memory_total_mib": row["memory_total_mib"],
            "driver_version": row["driver_version"],
        }
        for row in preflight["selected_devices"]
    ]


def _validate_direct_benchmark_gpu_authority(
    payload: Mapping[str, Any],
    plan_stage: Mapping[str, Any],
    gpu_preflight: Mapping[str, Any] | None,
    *,
    stage: str,
) -> None:
    _require(
        isinstance(gpu_preflight, Mapping)
        and gpu_preflight.get("status") == "passed",
        f"{stage}: benchmark lacks its orchestrator GPU preflight authority",
    )
    gpu = plan_stage.get("gpu")
    _require(isinstance(gpu, Mapping), f"{stage}: benchmark GPU plan is absent")
    physical = list(gpu["physical_gpus"])
    _require(
        payload.get("physical_gpu_mapping") == physical,
        f"{stage}: receipt physical GPU mapping differs from the release plan",
    )
    cvd = payload.get("cuda_visible_devices_by_stage")
    _require(
        isinstance(cvd, Mapping)
        and cvd.get("direct_provider") == gpu["cuda_visible_devices"],
        f"{stage}: receipt CVD differs from the release plan",
    )
    hardware = payload.get("hardware")
    _require(isinstance(hardware, Mapping), f"{stage}: hardware receipt is absent")
    _require(
        payload.get("hardware_identity_sha256") == sha256_json(hardware),
        f"{stage}: hardware identity does not seal the receipt object",
    )
    gpu_rows = hardware.get("gpus")
    _require(isinstance(gpu_rows, list), f"{stage}: hardware GPU inventory is absent")
    observed = [
        {
            "physical_index": row.get("physical_index"),
            "uuid": row.get("uuid"),
            "name": row.get("name"),
            "memory_total_mib": row.get("memory_total_mib"),
            "driver_version": row.get("driver_version"),
        }
        for row in gpu_rows
        if isinstance(row, Mapping)
    ]
    _require(
        observed == _stable_hardware(gpu_preflight),
        f"{stage}: receipt GPU index/UUID/driver differs from orchestrator preflight",
    )
    provider_preflight = payload.get("nvidia_smi_preflight")
    _require(
        isinstance(provider_preflight, Mapping)
        and provider_preflight.get("physical_gpus") == physical
        and payload.get("nvidia_smi_preflight_identity_sha256")
        == sha256_json(provider_preflight),
        f"{stage}: provider nvidia-smi preflight authority is invalid",
    )
    if stage.startswith("benchmark_phaxis_"):
        binding = payload.get("q8_exact_device_binding")
        _require(
            isinstance(binding, Mapping)
            and binding.get("status") == "passed_exact_physical_gpu_and_uuid"
            and isinstance(binding.get("bindings"), list)
            and bool(binding["bindings"]),
            f"{stage}: PHAxis Q8 exact-device binding is absent",
        )
        unsigned = deepcopy(dict(binding))
        identity = unsigned.pop("binding_identity_sha256", None)
        _require(
            _is_sha256(identity) and sha256_json(unsigned) == identity,
            f"{stage}: PHAxis Q8 exact-device binding is not self-sealed",
        )
        expected_uuid = {
            row["physical_index"]: row["uuid"] for row in observed
        }
        _require(
            all(
                item.get("requested_physical_gpu")
                == item.get("selected_physical_gpu")
                and item.get("requested_physical_gpu") in physical
                and item.get("physical_gpu_uuid")
                == expected_uuid[item.get("requested_physical_gpu")]
                and _is_sha256(item.get("selection_receipt_sha256"))
                for item in binding["bindings"]
                if isinstance(item, Mapping)
            )
            and len(binding["bindings"])
            == sum(isinstance(item, Mapping) for item in binding["bindings"]),
            f"{stage}: PHAxis Q8 shard mapping/UUID binding is invalid",
        )


def _validate_root_provider_q8_gpu_authority(
    context: _Context,
    plan_stage: Mapping[str, Any],
    gpu_preflight: Mapping[str, Any] | None,
    *,
    stage: str,
) -> None:
    _require(
        isinstance(gpu_preflight, Mapping)
        and gpu_preflight.get("status") == "passed",
        f"{stage}: root provider lacks its orchestrator GPU preflight authority",
    )
    output = context.artifact_paths.get((stage, "output"))
    _require(output is not None, f"{stage}: root-provider output artifact is absent")
    binding_path = output / "q8_shards" / "exact_device_binding.json"
    _require(binding_path.is_file(), f"{stage}: Q8 pre-merge binding is absent")
    binding = read_json(binding_path)
    unsigned = deepcopy(binding)
    identity = unsigned.pop("binding_identity_sha256", None)
    _require(
        binding.get("schema_version") == "PHAxis-Q8-shard-device-binding-1.0"
        and binding.get("status") == "passed_before_q8_merge"
        and binding.get("exact_physical_gpu_required") is True
        and _is_sha256(identity)
        and sha256_json(unsigned) == identity,
        f"{stage}: Q8 pre-merge binding is invalid",
    )
    gpu = plan_stage.get("gpu")
    _require(isinstance(gpu, Mapping), f"{stage}: GPU plan is absent")
    physical = list(gpu["physical_gpus"])
    _require(
        binding.get("planned_physical_gpus") == physical,
        f"{stage}: Q8 planned physical GPUs differ from the release plan",
    )
    outer_uuid = {
        row["physical_index"]: row["uuid"]
        for row in _stable_hardware(gpu_preflight)
    }
    records = binding.get("records")
    _require(
        isinstance(records, list)
        and len(records) == binding.get("shards")
        and bool(records),
        f"{stage}: Q8 shard bindings are incomplete",
    )
    for record in records:
        _require(isinstance(record, Mapping), f"{stage}: Q8 shard binding is invalid")
        requested = record.get("requested_physical_gpu")
        _require(
            record.get("planned_physical_gpu") == requested
            and record.get("selected_physical_gpu") == requested
            and requested in physical
            and record.get("physical_gpu_uuid") == outer_uuid[requested]
            and _is_sha256(record.get("selection_receipt_sha256")),
            f"{stage}: Q8 shard index/UUID differs from orchestrator preflight",
        )


def _validate_figure_table_bundle(
    summary_path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    relative = payload.get("supplementary_table_bundle_receipt")
    _require(
        isinstance(relative, str)
        and Path(relative).name == SUPPLEMENTARY_TABLE_BUNDLE_RECEIPT,
        "figures: supplementary Table/Data receipt path is invalid",
    )
    table_receipt = (summary_path.parent / relative).resolve()
    _require(
        table_receipt.is_relative_to(summary_path.parent.resolve()),
        "figures: supplementary Table/Data receipt escapes the figure suite",
    )
    try:
        bundle = validate_supplementary_table_data_bundle(
            table_receipt, require_final=True
        )
    except SupplementaryTableError as error:
        raise ReleaseOrchestratorError(
            f"figures: supplementary Table/Data S1--S10 validation failed: {error}"
        ) from error
    identities = {
        stem: record["item_identity_sha256"]
        for stem, record in bundle["items"].items()
    }
    _require(
        list(bundle["items"]) == list(SUPPLEMENTARY_TABLE_STEMS)
        and payload.get("supplementary_tables") == bundle["items"]
        and payload.get("supplementary_table_bundle_receipt_sha256")
        == bundle["receipt_sha256"]
        and payload.get("supplementary_table_bundle_identity_sha256")
        == bundle["bundle_identity_sha256"]
        and payload.get("supplementary_table_bundle_sha256")
        == bundle["bundle_file_sha256"]
        and payload.get("supplementary_table_source_authority_sha256")
        == bundle["source_authority_sha256"]
        and payload.get("supplementary_table_source_authority_identity")
        == bundle["source_authority_identity"]
        and payload.get("claim_contract", {}).get(
            "supplementary_table_data_resource_count"
        )
        == 10,
        "figures: supplementary Table/Data summary/physical closure differs",
    )
    return {
        "ordered_item_count": 10,
        "bundle_receipt_sha256": bundle["receipt_sha256"],
        "bundle_identity_sha256": bundle["bundle_identity_sha256"],
        "source_authority_sha256": bundle["source_authority_sha256"],
        "ordered_item_identity_sha256": identities,
    }


def _figure_table_binding(
    context: _Context,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    figure_stage = next(
        stage for stage in plan["stages"] if stage["name"] == "figures"
    )
    artifact = str(figure_stage["receipt_contract"]["artifact"])
    summary_path = context.artifact_paths[("figures", artifact)]
    return _validate_figure_table_bundle(summary_path, read_json(summary_path))


def _validate_publication_figure_input_resource_roles(
    resources: Mapping[str, Any],
) -> tuple[str, ...]:
    """Require the exact deterministic 25-role figure-input closure."""

    observed = tuple(resources)
    _require(
        observed == PUBLICATION_FIGURE_INPUT_RESOURCE_CANONICAL_KEY_ORDER,
        (
            "figure_inputs: final ordered 25-resource manifest is incomplete "
            "or noncanonical"
        ),
    )
    return observed


def _validate_publication_decision_bundle(
    context: _Context,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate the physical stage36 narrative/assignment/audit authority."""

    stage = next(
        item for item in plan["stages"] if item["name"] == "figure_inputs"
    )
    summary_path = context.artifact_paths[
        ("figure_inputs", str(stage["receipt_contract"]["artifact"]))
    ]
    manifest_path = context.artifact_paths[("figure_inputs", "manifest")]
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    manifest_identity = _sealed(
        manifest,
        "figure_input_assembly_identity_sha256",
        role="publication figure-input manifest",
    )
    resources = manifest.get("resources")
    _require(
        manifest.get("schema_version") == "PHAxis-manuscript-figure-inputs-2.0"
        and manifest.get("assembler_schema_version")
        == KNOWN_STAGE_SCHEMAS["figure_inputs"]
        and manifest.get("status") == "final"
        and isinstance(resources, Mapping),
        "figure_inputs: final ordered 25-resource manifest is incomplete or noncanonical",
    )
    _validate_publication_figure_input_resource_roles(resources)
    resource_paths: dict[str, Path] = {}
    resource_sha256: dict[str, str] = {}
    for role in PUBLICATION_FIGURE_INPUT_RESOURCE_ROLE_ORDER:
        record = resources[role]
        _require(
            isinstance(record, Mapping)
            and isinstance(record.get("path"), str)
            and _is_sha256(record.get("sha256")),
            f"figure_inputs: invalid resource record: {role}",
        )
        path = (manifest_path.parent / str(record["path"])).resolve()
        _require(
            path.is_relative_to(manifest_path.parent.resolve())
            and path.is_file()
            and not path.is_symlink()
            and sha256_file(path) == record["sha256"],
            f"figure_inputs: physical resource hash mismatch: {role}",
        )
        resource_paths[role] = path
        resource_sha256[role] = str(record["sha256"])

    try:
        decision = validate_narrative_decision(
            read_json(resource_paths["narrative_decision"])
        )
    except (NarrativeDecisionError, OSError, ValueError, TypeError) as error:
        raise ReleaseOrchestratorError(
            f"figure_inputs: narrative decision validation failed: {error}"
        ) from error
    assignment = read_json(resource_paths["qcdev_assignment"])
    assignment_identity = _sealed(
        assignment,
        "assignment_identity_sha256",
        role="QC-development instance assignment",
    )
    _require(
        assignment.get("schema_version")
        == "PHAxis-qcdev-instance-assignment-1.0"
        and assignment.get("status")
        == "completed_recomputed_from_sealed_geometry"
        and assignment.get("evidence_role")
        == "selected_qc_development_non_independent"
        and isinstance(assignment.get("matcher_contract"), Mapping)
        and assignment.get("matcher_contract_sha256")
        == sha256_json(assignment["matcher_contract"])
        and _is_sha256(assignment.get("source_input_sha256"))
        and _is_sha256(assignment.get("source_input_identity_sha256"))
        and _is_sha256(assignment.get("stage7_lock_set_identity_sha256"))
        and isinstance(assignment.get("display_source_unit"), str)
        and isinstance(assignment.get("pooled"), Mapping)
        and isinstance(assignment.get("assignments"), list)
        and bool(assignment["assignments"])
        and assignment.get("independent_accuracy_claim_allowed") is False,
        "figure_inputs: QC-development assignment is not a sealed non-independent one-to-one reconstruction",
    )
    overlay_rows = _read_csv(resource_paths["overlay_audit"])
    overlay_fields = {
        "schema_version",
        "case_id",
        "case_role",
        "task_id",
        "source_image_sha256",
        "prediction_sha256",
        "formal_state",
        "axis_in_root_coverage_fraction",
        "axis_single_component_coverage_fraction",
        "longest_unsupported_axis_gap_um",
        "formal_identity_count",
        "endpoint_complete_support_count",
        "endpoint_complete_support_fraction",
        "distal_window_1_4mm_eligible",
        "distal_window_1_4mm_reason",
        "profile_0_5mm_eligible",
        "profile_0_5mm_reason",
        "downstream_eligible",
        "downstream_reason",
        "condition_metadata_used",
    }
    _require(
        len(overlay_rows) == 5
        and len({row.get("case_id") for row in overlay_rows}) == 5
        and {row.get("case_role") for row in overlay_rows}
        == {
            "representative", "low_contrast", "curved_dense",
            "continuity", "fail_closed",
        }
        and {
            row.get("case_role"): row.get("task_id") for row in overlay_rows
        }.get("low_contrast")
        == "RHSCU-aa5b6e37df15821f"
        and {
            row.get("case_role"): row.get("task_id") for row in overlay_rows
        }.get("curved_dense")
        == "RHSCU-bbf649822174e0a2"
        and all(set(row) == overlay_fields for row in overlay_rows)
        and all(
            row.get("schema_version") == "PHAxis-Fig4-case-audit-2.0"
            for row in overlay_rows
        )
        and all(row.get("formal_state") in {"formal", "review_only"} for row in overlay_rows)
        and all(row.get("condition_metadata_used", "").casefold() == "false" for row in overlay_rows),
        "figure_inputs: Fig.4 five-case audit is incomplete or condition-aware",
    )
    formal_audit_fields = {
        "axis_in_root_coverage_fraction",
        "axis_single_component_coverage_fraction",
        "longest_unsupported_axis_gap_um",
        "formal_identity_count",
        "endpoint_complete_support_count",
        "endpoint_complete_support_fraction",
    }
    _require(
        all(
            (
                all(str(row.get(field, "")).strip() for field in formal_audit_fields)
                and row.get("downstream_eligible", "").casefold() == "true"
            )
            if row.get("formal_state") == "formal"
            else (
                all(not str(row.get(field, "")).strip() for field in formal_audit_fields)
                and row.get("downstream_eligible", "").casefold() == "false"
            )
            for row in overlay_rows
        ),
        "figure_inputs: Fig.4 formal/review null and eligibility semantics changed",
    )
    decision_identity = decision["narrative_decision_identity_sha256"]
    branch = decision["branch_id"]
    _require(
        summary.get("status") == "completed_final"
        and summary.get("figure_inputs_sha256") == sha256_file(manifest_path)
        and summary.get("figure_input_assembly_identity_sha256")
        == manifest_identity
        and summary.get("resource_sha256") == resource_sha256
        and summary.get("narrative_decision_identity_sha256")
        == manifest.get("narrative_decision_identity_sha256")
        == decision_identity
        and summary.get("narrative_branch_id")
        == manifest.get("narrative_branch_id")
        == branch
        and summary.get("qcdev_assignment_identity_sha256")
        == manifest.get("qcdev_assignment_identity_sha256")
        == assignment_identity,
        "figure_inputs: summary/manifest/decision/assignment bindings differ",
    )
    titles = title_contract(decision)
    return {
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_identity_sha256": manifest_identity,
        "narrative_decision_identity_sha256": decision_identity,
        "narrative_branch_id": branch,
        "qcdev_assignment_identity_sha256": assignment_identity,
        "title_contract": titles,
    }


def _normal_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _resolved_target_environment() -> dict[str, str]:
    environment = default_environment()
    environment.update(
        {
            "python_version": "3.12",
            "python_full_version": "3.12.0",
            "platform_system": "Windows",
            "sys_platform": "win32",
            "platform_machine": "AMD64",
            "implementation_name": "cpython",
            "implementation_version": "3.12.0",
            "extra": "",
        }
    )
    return environment


def _is_wheel_license_member(path: PurePosixPath) -> bool:
    name = path.name.upper()
    return any(
        name == stem
        or name.startswith(stem + ".")
        or name.startswith(stem + "-")
        or name.startswith(stem + "_")
        for stem in ("LICENSE", "LICENCE", "COPYING", "COPYRIGHT", "NOTICE")
    )


def _inspect_locked_wheel(path: Path) -> dict[str, Any]:
    """Read the exact package, dependency and license evidence sealed in a wheel."""

    _require(
        path.is_file() and not path.is_symlink() and path.suffix.casefold() == ".whl",
        f"offline_dependencies: locked wheel is absent/invalid: {path}",
    )
    try:
        with zipfile.ZipFile(path) as archive:
            _require(
                archive.testzip() is None,
                f"offline_dependencies: corrupt locked wheel: {path.name}",
            )
            metadata_members = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            _require(
                len(metadata_members) == 1,
                f"offline_dependencies: wheel METADATA closure is invalid: {path.name}",
            )
            metadata = BytesParser(policy=email_policy).parsebytes(
                archive.read(metadata_members[0])
            )
            license_files: list[dict[str, Any]] = []
            for member in sorted(archive.infolist(), key=lambda item: item.filename):
                if member.is_dir():
                    continue
                relative = PurePosixPath(member.filename)
                _require(
                    not relative.is_absolute()
                    and ".." not in relative.parts
                    and relative.as_posix() == member.filename,
                    f"offline_dependencies: unsafe wheel member: {path.name}:{member.filename}",
                )
                if not _is_wheel_license_member(relative):
                    continue
                _require(
                    ((member.external_attr >> 16) & 0o170000) != 0o120000,
                    f"offline_dependencies: symlinked wheel license member: {path.name}:{member.filename}",
                )
                content = archive.read(member)
                license_files.append(
                    {
                        "path": relative.as_posix(),
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
    except (OSError, zipfile.BadZipFile) as error:
        raise ReleaseOrchestratorError(
            f"offline_dependencies: cannot inspect {path.name}: {error}"
        ) from error
    name = metadata.get("Name")
    version = metadata.get("Version")
    _require(
        isinstance(name, str) and bool(name) and isinstance(version, str) and bool(version),
        f"offline_dependencies: wheel Name/Version is absent: {path.name}",
    )
    legacy_license = str(metadata.get("License") or "").strip()
    if legacy_license.casefold() in {"unknown", "n/a", "none"}:
        legacy_license = ""
    return {
        "filename": path.name,
        "distribution": _normal_distribution_name(name),
        "version": version,
        "requires_dist": list(metadata.get_all("Requires-Dist") or ()),
        "metadata_license_expression": str(
            metadata.get("License-Expression") or ""
        ).strip(),
        "metadata_legacy_license": legacy_license,
        "metadata_license_classifiers": sorted(
            str(value).strip()
            for value in metadata.get_all("Classifier") or ()
            if str(value).strip().startswith("License ::")
        ),
        "metadata_license_files": sorted(
            str(value).strip()
            for value in metadata.get_all("License-File") or ()
            if str(value).strip()
        ),
        "license_files": license_files,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _active_wheel_dependencies(
    metadata: Mapping[str, Any], *, available: frozenset[str], extra: str = ""
) -> list[str]:
    environment = _resolved_target_environment()
    environment["extra"] = extra
    dependencies: set[str] = set()
    for raw in metadata.get("requires_dist", ()):
        try:
            requirement = Requirement(str(raw))
        except Exception as error:
            raise ReleaseOrchestratorError(
                f"offline_dependencies: invalid Requires-Dist in {metadata.get('filename')}: {raw}"
            ) from error
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        name = _normal_distribution_name(requirement.name)
        _require(
            name in available,
            f"offline_dependencies: active dependency {name} required by "
            f"{metadata.get('distribution')} is absent from wheelhouse",
        )
        dependencies.add(name)
    return sorted(dependencies)


def _active_wheel_requirement_records(
    metadata: Mapping[str, Any], *, extra: str
) -> list[dict[str, str]]:
    environment = _resolved_target_environment()
    environment["extra"] = extra
    records: list[dict[str, str]] = []
    for raw in metadata.get("requires_dist", ()):
        try:
            requirement = Requirement(str(raw))
        except Exception as error:
            raise ReleaseOrchestratorError(
                f"offline_dependencies: invalid Requires-Dist in {metadata.get('filename')}: {raw}"
            ) from error
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        records.append(
            {
                "name": _normal_distribution_name(requirement.name),
                "specifier": str(requirement.specifier),
                "marker": str(requirement.marker or ""),
            }
        )
    records.sort(key=lambda row: (row["name"], row["specifier"], row["marker"]))
    return records


def _pypi_purl(distribution: str, version: str) -> str:
    return (
        "pkg:pypi/"
        + quote(distribution, safe=".-_~")
        + "@"
        + quote(version, safe=".-_~")
    )


def _validate_offline_dependencies(
    context: _Context,
    plan_stage: Mapping[str, Any],
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> None:
    """Fail closed on the physical and semantic resolved supply-chain closure."""

    artifact_paths = {
        str(artifact["name"]): Path(str(artifact["path"])).resolve()
        for artifact in plan_stage.get("artifacts", ())
        if isinstance(artifact, Mapping)
    }
    required_artifacts = {
        "receipt",
        "output",
        "dependency_lock",
        "wheelhouse",
        "resolved_sbom",
        "resolved_license_inventory",
    }
    _require(
        required_artifacts.issubset(artifact_paths),
        "offline_dependencies: stage contract omits a resolved supply-chain artifact",
    )
    output = receipt_path.parent.resolve()
    canonical = {
        "receipt": output / "receipt.json",
        "output": output,
        "dependency_lock": output / "requirements.lock.txt",
        "wheelhouse": output / "wheelhouse",
        "resolved_sbom": output / "SBOM.resolved.cdx.json",
        "resolved_license_inventory": output
        / "THIRD_PARTY_LICENSES.resolved.json",
    }
    _require(
        all(artifact_paths[name] == path.resolve() for name, path in canonical.items()),
        "offline_dependencies: resolved artifact canonical path mismatch",
    )
    sbom_path = canonical["resolved_sbom"]
    licenses_path = canonical["resolved_license_inventory"]
    lock_path = canonical["dependency_lock"]
    wheelhouse = canonical["wheelhouse"]
    for path in (sbom_path, licenses_path, lock_path):
        _require(
            path.is_file() and not path.is_symlink(),
            f"offline_dependencies: sealed file is absent/symlinked: {path}",
        )
    _require(
        wheelhouse.is_dir() and not wheelhouse.is_symlink(),
        "offline_dependencies: wheelhouse is absent/symlinked",
    )

    wheels = sorted(wheelhouse.iterdir(), key=lambda path: path.name.casefold())
    _require(
        bool(wheels)
        and all(path.is_file() and not path.is_symlink() and path.suffix.casefold() == ".whl" for path in wheels),
        "offline_dependencies: wheelhouse contains a non-wheel or is empty",
    )
    inspected = [_inspect_locked_wheel(path) for path in wheels]
    by_name = {str(record["distribution"]): record for record in inspected}
    _require(
        len(by_name) == len(inspected),
        "offline_dependencies: duplicate resolved distributions",
    )
    records = [
        {
            key: metadata[key]
            for key in ("distribution", "version", "filename", "bytes", "sha256")
        }
        for metadata in sorted(inspected, key=lambda row: str(row["distribution"]))
    ]
    _require(
        receipt.get("wheelhouse_files") == records
        and receipt.get("wheelhouse_file_count") == len(records)
        and receipt.get("wheelhouse_identity_sha256") == sha256_json(records),
        "offline_dependencies: receipt/wheelhouse byte closure mismatch",
    )
    reduced_records = [
        {
            "distribution": row["distribution"],
            "version": row["version"],
            "sha256": row["sha256"],
        }
        for row in records
    ]
    _require(
        receipt.get("resolved_requirement_set_identity_sha256")
        == sha256_json(reduced_records),
        "offline_dependencies: resolved requirement-set identity mismatch",
    )
    expected_lock = "".join(
        f"{row['distribution']}=={row['version']} --hash=sha256:{row['sha256']}\n"
        for row in records
    )
    _require(
        lock_path.read_text(encoding="utf-8") == expected_lock
        and receipt.get("requirements_lock_sha256") == sha256_file(lock_path)
        and receipt.get("pip_require_hashes") is True,
        "offline_dependencies: hash-required lock does not exactly close the wheelhouse",
    )

    direct_rows = receipt.get("active_direct_requirements")
    _require(
        isinstance(direct_rows, list)
        and bool(direct_rows)
        and all(isinstance(row, Mapping) for row in direct_rows),
        "offline_dependencies: active direct requirement contract is absent",
    )
    direct_names = [str(row.get("name")) for row in direct_rows]
    _require(
        len(direct_names) == len(set(direct_names))
        and set(direct_names).issubset(by_name),
        "offline_dependencies: direct requirement closure is invalid",
    )

    formal_wheel = context.artifact_paths.get(("distributions", "wheel"))
    _require(
        formal_wheel is not None and formal_wheel.is_file(),
        "offline_dependencies: upstream formal wheel artifact is absent",
    )
    formal_metadata = _inspect_locked_wheel(formal_wheel)
    _require(
        formal_metadata["distribution"] == "phaxis"
        and formal_metadata["version"] == "1.0.0"
        and receipt.get("formal_wheel_sha256") == formal_metadata["sha256"],
        "offline_dependencies: formal PHAxis wheel authority mismatch",
    )
    formal_direct_names = _active_wheel_dependencies(
        formal_metadata,
        available=frozenset(by_name),
        extra="deployment",
    )
    formal_direct_records = _active_wheel_requirement_records(
        formal_metadata,
        extra="deployment",
    )
    _require(
        direct_rows == formal_direct_records
        and sorted(direct_names) == formal_direct_names,
        "offline_dependencies: receipt direct requirements differ from formal wheel METADATA",
    )
    _require(
        receipt.get("formal_wheel_requires_dist_identity_sha256")
        == sha256_json(formal_metadata["requires_dist"]),
        "offline_dependencies: formal wheel Requires-Dist identity mismatch",
    )

    sbom_record = receipt.get("resolved_cyclonedx_sbom")
    license_record = receipt.get("resolved_license_inventory")
    _require(
        isinstance(sbom_record, Mapping)
        and sbom_record.get("filename") == sbom_path.name
        and sbom_record.get("sha256") == sha256_file(sbom_path)
        and isinstance(license_record, Mapping)
        and license_record.get("filename") == licenses_path.name
        and license_record.get("sha256") == sha256_file(licenses_path),
        "offline_dependencies: receipt/resolved artifact hash interlock failed",
    )

    sbom = read_json(sbom_path)
    try:
        uuid.UUID(str(sbom.get("serialNumber", "")).removeprefix("urn:uuid:"))
        serial_valid = str(sbom.get("serialNumber", "")).startswith("urn:uuid:")
    except (ValueError, AttributeError):
        serial_valid = False
    root_ref = _pypi_purl("phaxis", "1.0.0")
    root_component = sbom.get("metadata", {}).get("component")
    components = sbom.get("components")
    dependencies = sbom.get("dependencies")
    _require(
        sbom.get("bomFormat") == "CycloneDX"
        and sbom.get("specVersion") == "1.6"
        and sbom.get("version") == 1
        and serial_valid
        and isinstance(root_component, Mapping)
        and root_component.get("type") == "application"
        and root_component.get("bom-ref") == root_ref
        and root_component.get("name") == "phaxis"
        and root_component.get("version") == "1.0.0"
        and root_component.get("purl") == root_ref
        and root_component.get("hashes")
        == [{"alg": "SHA-256", "content": formal_metadata["sha256"]}]
        and isinstance(components, list)
        and isinstance(dependencies, list),
        "offline_dependencies: resolved document is not a basic CycloneDX 1.6 SBOM",
    )
    refs = {
        name: _pypi_purl(name, str(metadata["version"]))
        for name, metadata in by_name.items()
    }
    component_by_name = {
        str(component.get("name")): component
        for component in components
        if isinstance(component, Mapping)
    }
    _require(
        len(component_by_name) == len(components)
        and set(component_by_name) == set(by_name),
        "offline_dependencies: CycloneDX component set differs from wheelhouse",
    )
    for name, metadata in by_name.items():
        component = component_by_name[name]
        properties = {
            str(item.get("name")): item.get("value")
            for item in component.get("properties", ())
            if isinstance(item, Mapping)
        }
        _require(
            component.get("type") == "library"
            and component.get("bom-ref") == refs[name]
            and component.get("purl") == refs[name]
            and component.get("version") == metadata["version"]
            and component.get("hashes")
            == [{"alg": "SHA-256", "content": metadata["sha256"]}]
            and isinstance(component.get("licenses"), list)
            and bool(component["licenses"])
            and properties.get("phaxis:locked-wheel-filename")
            == metadata["filename"],
            f"offline_dependencies: CycloneDX component is not wheel-bound: {name}",
        )
    graph = {
        str(row.get("ref")): row.get("dependsOn")
        for row in dependencies
        if isinstance(row, Mapping)
    }
    expected_refs = {root_ref, *refs.values()}
    _require(
        len(graph) == len(dependencies)
        and set(graph) == expected_refs
        and graph[root_ref] == [refs[name] for name in sorted(direct_names)],
        "offline_dependencies: CycloneDX root dependency graph is incomplete",
    )
    for name, metadata in by_name.items():
        expected = [
            refs[dependency]
            for dependency in _active_wheel_dependencies(
                metadata, available=frozenset(by_name)
            )
        ]
        _require(
            graph[refs[name]] == expected,
            f"offline_dependencies: CycloneDX dependency relation mismatch: {name}",
        )
    properties = {
        str(item.get("name")): item.get("value")
        for item in sbom.get("metadata", {}).get("properties", ())
        if isinstance(item, Mapping)
    }
    _require(
        properties.get("phaxis:resolution-target") == "cp312-win_amd64"
        and properties.get("phaxis:resolved-transitive-closure-claimed") == "true"
        and properties.get("phaxis:wheelhouse-identity-sha256")
        == sha256_json(reduced_records)
        and sbom_record.get("spec_version") == "1.6"
        and sbom_record.get("serial_number") == sbom["serialNumber"]
        and sbom_record.get("component_count_including_phaxis")
        == len(records) + 1
        and sbom_record.get("exact_versions_and_wheel_sha256_included") is True
        and sbom_record.get("dependency_graph_included") is True,
        "offline_dependencies: CycloneDX receipt properties/summary mismatch",
    )

    license_inventory = read_json(licenses_path)
    _sealed(
        license_inventory,
        "resolved_license_inventory_identity_sha256",
        role="offline_dependencies resolved license inventory",
    )
    expected_license_artifacts: list[dict[str, Any]] = []
    for metadata in sorted(inspected, key=lambda row: str(row["distribution"])):
        evidence = {
            "metadata_license_expression": metadata["metadata_license_expression"],
            "metadata_legacy_license": metadata["metadata_legacy_license"],
            "metadata_license_classifiers": metadata[
                "metadata_license_classifiers"
            ],
            "metadata_license_files": metadata["metadata_license_files"],
            "license_files": metadata["license_files"],
        }
        expected_license_artifacts.append(
            {
                "distribution": metadata["distribution"],
                "version": metadata["version"],
                "filename": metadata["filename"],
                "bytes": metadata["bytes"],
                "sha256": metadata["sha256"],
                **evidence,
                "license_evidence_identity_sha256": sha256_json(evidence),
                "machine_readable_spdx_expression_present": bool(
                    metadata["metadata_license_expression"]
                ),
                "license_evidence_present": bool(
                    metadata["metadata_license_expression"]
                    or metadata["metadata_legacy_license"]
                    or metadata["metadata_license_classifiers"]
                    or metadata["license_files"]
                ),
            }
        )
    _require(
        license_inventory.get("schema_version")
        == "PHAxis-resolved-third-party-license-inventory-1.0"
        and license_inventory.get("status")
        == "resolved_artifact_evidence_inventory_requires_review"
        and license_inventory.get("product") == "PHAxis"
        and license_inventory.get("product_version") == "1.0.0"
        and license_inventory.get("target")
        == {
            "platform": "win_amd64",
            "python_version": "3.12",
            "implementation": "cp",
            "abi": "cp312",
            "extras": ["deployment"],
        }
        and license_inventory.get("artifact_count") == len(records)
        and license_inventory.get("artifacts") == expected_license_artifacts
        and license_inventory.get("all_artifacts_have_license_evidence") is True
        and license_inventory.get("artifact_specific_license_review_required") is True
        and license_inventory.get("license_clearance_claimed") is False
        and license_inventory.get("resolved_transitive_dependency_claimed") is True
        and license_record.get("identity_sha256")
        == license_inventory["resolved_license_inventory_identity_sha256"]
        and license_record.get("artifact_count") == len(records)
        and license_record.get("all_artifacts_have_license_evidence") is True
        and license_record.get("artifact_specific_license_review_required") is True,
        "offline_dependencies: resolved license inventory does not close wheel METADATA/license members",
    )
    _require(
        receipt.get("target") == license_inventory["target"]
        and receipt.get("resolver", {}).get("binary_only") is True
        and receipt.get("resolver", {}).get("pip_require_hashes_for_install")
        is True,
        "offline_dependencies: target/resolver receipt contract mismatch",
    )
    _require(
        receipt.get("resolved_software_supply_chain_generated") is True
        and receipt.get("sdists_used") is False
        and receipt.get("credentials_recorded") is False,
        "offline_dependencies: resolved supply-chain receipt guards failed",
    )


def _validate_cohort_profile_bundle(
    context: _Context,
    plan: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    """Prove that Stage 23 materialised distinct clean261/full283 profiles."""

    expected_counts = {
        "human_curated443": 443,
        "biological_full": 283,
        "human_curated_overlap": 22,
        "biological_clean": 261,
    }
    _require(
        payload.get("cohort_directories")
        == {"primary": "primary_clean261", "sensitivity": "sensitivity_full283"}
        and payload.get("counts") == expected_counts
        and payload.get("primary_is_strict_task_subset_of_sensitivity") is True
        and payload.get("primary_sensitivity_task_overlap") == 261
        and payload.get("sensitivity_only_human443_overlap_tasks") == 22
        and payload.get("root_cap_region_output") is False
        and payload.get("stageb_two_point_vector_used_as_length") is False
        and payload.get("canonical_annotations_read") is False
        and payload.get("blind_images_used") == 0,
        "profiles_exact283: clean261/full283 bundle contract is incomplete",
    )
    for field in (
        "cohort_build_summary_sha256",
        "cohort_build_identity_sha256",
        "cohort_lock_sha256",
        "cohort_membership_csv_sha256",
        "traits_summary_sha256",
        "profile_contract_sha256",
    ):
        _require(
            _is_sha256(payload.get(field)),
            f"profiles_exact283: bundle hash is invalid: {field}",
        )

    pin_stage = next(stage for stage in plan["stages"] if stage["name"] == "authority_pin")
    pin_path = context.artifact_paths[
        ("authority_pin", str(pin_stage["receipt_contract"]["artifact"]))
    ]
    binding = read_model_contract_authority(pin_path)
    try:
        require_output_identity(payload, binding, role="profiles_exact283")
    except ContractError as error:
        raise ReleaseOrchestratorError(str(error)) from error

    cohort_stage = next(
        stage for stage in plan["stages"] if stage["name"] == "cohorts_exact283"
    )
    cohort_summary_path = context.artifact_paths[
        ("cohorts_exact283", str(cohort_stage["receipt_contract"]["artifact"]))
    ]
    cohort_summary = read_json(cohort_summary_path)
    _require(
        payload.get("cohort_build_summary_sha256")
        == sha256_file(cohort_summary_path)
        and payload.get("cohort_build_identity_sha256")
        == cohort_summary.get("cohort_build_identity_sha256"),
        "profiles_exact283: biological cohort authority is misbound",
    )
    traits_stage = next(
        stage for stage in plan["stages"] if stage["name"] == "traits_exact283"
    )
    traits_summary_path = context.artifact_paths[
        ("traits_exact283", str(traits_stage["receipt_contract"]["artifact"]))
    ]
    _require(
        payload.get("traits_summary_sha256") == sha256_file(traits_summary_path),
        "profiles_exact283: full283 trait authority is misbound",
    )

    exports = payload.get("cohort_exports")
    _require(
        isinstance(exports, Mapping)
        and set(exports) == {"primary_clean261", "sensitivity_full283"},
        "profiles_exact283: cohort export inventory is incomplete",
    )
    specifications = (
        (
            "primary_clean261",
            "primary_SHA_disjoint",
            261,
            "primary_summary",
            "primary_profiles",
        ),
        (
            "sensitivity_full283",
            "overlap_contaminated_sensitivity",
            283,
            "sensitivity_summary",
            "sensitivity_profiles",
        ),
    )
    task_sets: dict[str, set[str]] = {}
    profile_rows_by_cohort: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    child_summaries: dict[str, Mapping[str, Any]] = {}
    cohort_hashes = cohort_summary.get("output_sha256")
    _require(
        isinstance(cohort_hashes, Mapping),
        "profiles_exact283: cohort table hash authority is absent",
    )
    for cohort_name, cohort_role, expected_tasks, summary_artifact, csv_artifact in specifications:
        summary_path = context.artifact_paths[("profiles_exact283", summary_artifact)]
        profile_path = context.artifact_paths[("profiles_exact283", csv_artifact)]
        child = read_json(summary_path)
        child_summaries[cohort_name] = child
        _require(
            child.get("schema_version") == "PHAxis-distal-axis-profile-export-1.0.0"
            and child.get("status") == "completed"
            and child.get("tasks") == expected_tasks
            and child.get("bins_per_task") == 5
            and child.get("rows") == expected_tasks * 5
            and child.get("profiles_csv_sha256") == sha256_file(profile_path)
            and child.get("profile_contract_sha256")
            == payload.get("profile_contract_sha256")
            and child.get("blind_images_used") == 0,
            f"profiles_exact283: {cohort_name} child export is incomplete",
        )
        _sealed(child, "export_identity_sha256", role=f"profiles_exact283/{cohort_name}")
        try:
            require_output_identity(
                child,
                binding,
                role=f"profiles_exact283/{cohort_name}",
            )
        except ContractError as error:
            raise ReleaseOrchestratorError(str(error)) from error
        cohort_binding = child.get("cohort_binding")
        _require(
            isinstance(cohort_binding, Mapping)
            and cohort_binding.get("schema_version")
            == "PHAxis-distal-axis-profile-cohort-binding-1.0.0"
            and cohort_binding.get("cohort_name") == cohort_name
            and cohort_binding.get("cohort_role") == cohort_role
            and cohort_binding.get("cohort_tasks") == expected_tasks
            and cohort_binding.get("cohort_build_summary_sha256")
            == sha256_file(cohort_summary_path)
            and cohort_binding.get("cohort_build_identity_sha256")
            == cohort_summary.get("cohort_build_identity_sha256")
            and cohort_binding.get("cohort_membership_csv_sha256")
            == payload.get("cohort_membership_csv_sha256")
            and cohort_binding.get("blind_images_used") == 0,
            f"profiles_exact283: {cohort_name} cohort binding is invalid",
        )
        table_authority = cohort_hashes.get(cohort_name)
        _require(
            isinstance(table_authority, Mapping)
            and child.get("traits_csv_sha256") == table_authority.get("traits")
            and child.get("hair_instances_csv_sha256")
            == table_authority.get("hair_instances"),
            f"profiles_exact283: {cohort_name} source tables are misbound",
        )
        record = exports[cohort_name]
        _require(
            isinstance(record, Mapping)
            and record.get("summary_sha256") == sha256_file(summary_path)
            and record.get("profiles_csv_sha256") == sha256_file(profile_path)
            and record.get("export_identity_sha256")
            == child.get("export_identity_sha256")
            and record.get("cohort_task_membership_sha256")
            == cohort_binding.get("cohort_task_membership_sha256"),
            f"profiles_exact283: {cohort_name} bundle inventory is misbound",
        )
        rows = _read_csv(profile_path)
        task_ids = {str(row.get("task_id", "")).strip() for row in rows}
        keyed_rows = {
            (
                str(row.get("task_id", "")).strip(),
                str(row.get("bin_index", "")).strip(),
            ): dict(row)
            for row in rows
        }
        source_by_task: dict[str, str] = {}
        for row in rows:
            task_id = str(row.get("task_id", "")).strip()
            source_sha = str(row.get("source_image_sha256", "")).casefold()
            _require(
                _is_sha256(source_sha)
                and source_by_task.setdefault(task_id, source_sha) == source_sha,
                f"profiles_exact283: {cohort_name} source-image membership drifted",
            )
        _require(
            "" not in task_ids
            and len(rows) == expected_tasks * 5
            and len(keyed_rows) == len(rows)
            and len(task_ids) == expected_tasks
            and all(
                {
                    str(row.get("bin_index", "")).strip()
                    for row in rows
                    if str(row.get("task_id", "")).strip() == task_id
                }
                == {"0", "1", "2", "3", "4"}
                for task_id in task_ids
            )
            and sha256_json(sorted(task_ids))
            == cohort_binding.get("cohort_task_membership_sha256"),
            f"profiles_exact283: {cohort_name} profile membership does not close",
        )
        _require(
            len(set(source_by_task.values())) == expected_tasks
            and sha256_json(sorted(source_by_task.values()))
            == cohort_binding.get("cohort_source_image_membership_sha256"),
            f"profiles_exact283: {cohort_name} source-image digest is misbound",
        )
        task_sets[cohort_name] = task_ids
        profile_rows_by_cohort[cohort_name] = keyed_rows

    primary = task_sets["primary_clean261"]
    sensitivity = task_sets["sensitivity_full283"]
    _require(
        primary < sensitivity
        and len(sensitivity - primary) == 22
        and child_summaries["primary_clean261"].get("profiles_csv_sha256")
        != child_summaries["sensitivity_full283"].get("profiles_csv_sha256")
        and child_summaries["primary_clean261"].get("export_identity_sha256")
        != child_summaries["sensitivity_full283"].get("export_identity_sha256"),
        "profiles_exact283: primary and sensitivity exports are aliased or not nested",
    )
    _require(
        all(
            profile_rows_by_cohort["primary_clean261"][key]
            == profile_rows_by_cohort["sensitivity_full283"].get(key)
            for key in profile_rows_by_cohort["primary_clean261"]
        ),
        "profiles_exact283: shared clean261 profile rows differ from full283",
    )


def _validate_receipt(
    context: _Context,
    plan: Mapping[str, Any],
    plan_stage: Mapping[str, Any],
    gpu_preflight: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    name = str(plan_stage["name"])
    contract = plan_stage["receipt_contract"]
    receipt_artifact = str(contract["artifact"])
    receipt_path = context.artifact_paths[(name, receipt_artifact)]
    payload = read_json(receipt_path)
    _require(payload.get("schema_version") == contract["schema_version"], f"{name}: receipt schema mismatch")
    status_field = str(contract.get("status_field", "status"))
    _require(_value_at(payload, status_field) == contract["status"], f"{name}: receipt status mismatch")
    for field, expected in contract.get("required_fields", {}).items():
        _require(_value_at(payload, str(field)) == expected, f"{name}: receipt field mismatch: {field}")
    identity_field = contract.get("identity_field")
    if identity_field is not None:
        identity = _value_at(payload, str(identity_field))
        _require(_is_sha256(identity), f"{name}: receipt logical identity is invalid")
        if contract.get("identity_seals_complete_object") is True:
            _sealed(payload, str(identity_field), role=f"{name} receipt")
    _recursive_release_guards(payload, role=f"{name} receipt")

    if name == "candidate_manifest":
        validate_candidate_manifest(payload)
        _require(payload == context.candidate_preview, "candidate CLI output differs from the prevalidated five-checkpoint manifest")
    elif name == "production_manifest":
        _require(
            payload.get("images") == 283
            and payload.get("status") == "completed",
            "stage-produced production manifest is not completed exact283",
        )
        manifest_artifact = next(
            (
                artifact
                for artifact in plan_stage["artifacts"]
                if artifact["name"] == "manifest_all"
            ),
            None,
        )
        _require(
            isinstance(manifest_artifact, Mapping),
            "production_manifest must publish the normalized manifest_all artifact",
        )
        manifest_gate = _validate_production_manifest(
            Path(str(manifest_artifact["path"]))
        )
        _require(
            payload.get("manifest_all_sha256") == manifest_gate["manifest_sha256"],
            "production manifest summary/CSV authority mismatch",
        )
    elif name == "qcdev_candidate_pool":
        _require(payload.get("images") == 44 and payload.get("resumed_images") == 0, "candidate-pool run was not fresh exact44")
        _require(
            payload.get("checkpoint_sha256")
            == [
                member["checkpoint_sha256"]
                for member in context.candidate_preview["identity_payload"]["members"]
            ],
            "candidate-pool checkpoint order differs from the five-seed Gate",
        )
    elif name == "selection":
        _require(payload.get("images") == 44 and payload.get("independent_accuracy_claim_allowed") is False, "selection receipt is not QC-development44-only")
    elif name == "qcdev_evaluation_inference":
        expected = {
            "images": 44,
            "resumed_images": 0,
            "production_consumption_allowed": False,
            "fusion_consumption_allowed": False,
            "traits_consumption_allowed": False,
            "model_contract_proposal_present": False,
        }
        for field, value in expected.items():
            _require(payload.get(field) == value, f"evaluation-only inference guard changed: {field}")
    elif name == "qcdev_evaluation":
        _require(payload.get("independent_accuracy_claim_allowed") is False, "QCdev evaluation overclaims independence")
        _require(payload.get("overall", {}).get("stageb_train399", {}).get("images") == 44, "QCdev evaluation is not exact44")
        _require(
            payload.get("training_contract", {}).get("checkpoint_sha256")
            == [
                member["checkpoint_sha256"]
                for member in context.candidate_preview["identity_payload"]["members"]
            ],
            "QCdev evaluation checkpoint order differs from the five-seed Gate",
        )
    elif name == "root_provider_exact283":
        _validate_root_exact283(receipt_path)
        _validate_root_provider_q8_gpu_authority(
            context,
            plan_stage,
            gpu_preflight,
            stage=name,
        )
    elif name == "root_bundle_materialization":
        _require(
            payload.get("status") == "pass"
            and payload.get("materialized_exact_closure") is True
            and payload.get("exact_file_closure_required") is True
            and payload.get("exact_file_closure_passed") is True
            and payload.get("unlisted_file_count") == 0
            and payload.get("missing_closure_file_count") == 0
            and payload.get("files_verified", 0) > 0
            and payload.get("bytes_verified", 0) > 0
            and payload.get("source_bundle_mutated") is False,
            "materialized root-provider bundle is not an exact verified closure",
        )
    elif name == "proposal":
        binding = read_model_contract_proposal(receipt_path)
        _require(binding.file_sha256 == sha256_file(receipt_path), "proposal file authority mismatch")
        candidate = context.candidate_preview
        _require(
            list(binding.stageb_binding["checkpoint_sha256"])
            == [member["checkpoint_sha256"] for member in candidate["identity_payload"]["members"]],
            "proposal checkpoint authority differs from the five completed seeds",
        )
        stage_specs = {
            str(stage["name"]): stage for stage in context.manifest["stages"]
        }
        proposal_payload = read_json(receipt_path)
        promotion = proposal_payload["promotion"]
        formal_sources = promotion["formal_gate_source_sha256"]
        source_paths = {
            "train399_candidate": context.artifact_paths[
                (
                    "candidate_manifest",
                    stage_specs["candidate_manifest"]["receipt"]["artifact"],
                )
            ],
            "train399_selection": context.artifact_paths[
                ("selection", stage_specs["selection"]["receipt"]["artifact"])
            ],
            "train399_evaluation": context.artifact_paths[
                (
                    "qcdev_evaluation",
                    stage_specs["qcdev_evaluation"]["receipt"]["artifact"],
                )
            ],
            "root_exact283": context.artifact_paths[
                (
                    "root_provider_exact283",
                    stage_specs["root_provider_exact283"]["receipt"]["artifact"],
                )
            ],
        }
        for role, path in source_paths.items():
            _require(
                formal_sources.get(role) == sha256_file(path),
                f"proposal formal source hash mismatch: {role}",
            )
        official_stage = stage_specs["official_apply"]
        _require(
            promotion.get("source_model_contract_sha256")
            == official_stage["cas"]["expected_sha256"],
            "proposal source model-contract SHA differs from the authority CAS predecessor",
        )
    elif name == "authority_pin":
        binding = read_model_contract_authority(receipt_path)
        _require(binding.authority_file_sha256 == sha256_file(receipt_path), "authority pin file authority mismatch")
    elif name == "qcdev_root_provider":
        _require(
            payload.get("status") == "completed_uncompared"
            and payload.get("canonical_annotations_read") is False
            and payload.get("blind_images_used") == 0,
            "QCdev root-provider label-free run is not completed-uncompared or violates release guards",
        )
        _validate_root_provider_q8_gpu_authority(
            context,
            plan_stage,
            gpu_preflight,
            stage=name,
        )
    elif name == "qcdev_fusion":
        _require(
            payload.get("status") == "completed"
            and payload.get("images") == 44,
            "QCdev fusion is not completed exact44",
        )
    elif name in {"production_stageb_exact283", "fusion_exact283", "traits_exact283"}:
        pin_stage = next(stage for stage in plan["stages"] if stage["name"] == "authority_pin")
        pin_path = context.artifact_paths[("authority_pin", pin_stage["receipt_contract"]["artifact"])]
        binding = read_model_contract_authority(pin_path)
        root_field = "root_expert" if name == "fusion_exact283" else "root_expert_id"
        try:
            require_output_identity(payload, binding, role=name, root_field=root_field)
        except ContractError as error:
            raise ReleaseOrchestratorError(str(error)) from error
        count_field = "tasks" if name == "traits_exact283" else "images"
        _require(payload.get(count_field) == 283, f"{name}: output is not exact283")
        if name == "production_stageb_exact283":
            _require(
                payload.get("checkpoint_sha256")
                == list(binding.stageb_binding["checkpoint_sha256"]),
                "production StageB checkpoint order differs from the proposal",
            )
            _require(payload.get("resumed_images") == 0, "production StageB benchmark/release run was not fresh")
            _require(all(record.get("resumed") is False for record in payload.get("records", [])), "production StageB contains cached image results")
        role_text = json.dumps(payload, ensure_ascii=False).casefold()
        _require("evaluation_only_not_deployable" not in role_text, f"{name}: eval-only artifact entered production")
    elif name == "profiles_exact283":
        _validate_cohort_profile_bundle(context, plan, payload)
    elif name in DIRECT_BENCHMARK_STAGE_NAMES:
        _require(
            payload.get("status") == "completed_direct_full283"
            and payload.get("images") == 283
            and payload.get("fresh_direct_run") is True
            and payload.get("resume_or_cache_used") is False
            and payload.get("measurement_scope")
            == "raw_image_to_final_traits_and_profiles_direct",
            f"{name}: benchmark is not a fresh direct full-workflow exact283 run",
        )
        _validate_direct_benchmark_gpu_authority(
            payload,
            plan_stage,
            gpu_preflight,
            stage=name,
        )
        artifact_by_name = {
            str(artifact["name"]): Path(str(artifact["path"]))
            for artifact in plan_stage["artifacts"]
        }
        for field, artifact_name in (
            ("gpu_telemetry_artifact", "gpu_telemetry"),
            ("hardware_preflight_artifact", "hardware_preflight"),
        ):
            record = payload.get(field)
            path = artifact_by_name.get(artifact_name)
            _require(
                isinstance(record, Mapping)
                and path is not None
                and path.is_file()
                and record.get("path") == path.name
                and record.get("sha256") == sha256_file(path),
                f"{name}: {field} is absent or differs from its declared artifact",
            )
    elif name == "analysis_workflow_manifest":
        _require(
            payload.get("status") == "ready_hash_locked_full_workflow"
            and payload.get("guards", {}).get("blind_images_used") == 0
            and payload.get("guards", {}).get("canonical_annotations_read") is False
            and payload.get("benchmark_contract", {}).get("warmup_runs") == 0
            and payload.get("benchmark_contract", {}).get("measured_repeats") == 1,
            "analysis workflow manifest is not a sealed fresh-direct authority",
        )
    elif name == "qcdev_root_inputs":
        _require(
            payload.get("status")
            == "completed_locked_exact44_label_free_source_contract"
            and payload.get("tasks") == 44
            and payload.get("labels_or_annotation_files_read") is False
            and payload.get("locked_members_posthoc_filtered") is False
            and payload.get("acquisition_gate_can_remove_locked_member") is False
            and payload.get("condition_metadata_used_for_routing") is False,
            "QCdev root-provider source suite is not locked exact44/label-free",
        )
    elif name == "benchmark_artifact_inventory":
        counts = payload.get("role_counts")
        _require(
            payload.get("status") == "completed_explicit_benchmark_inventory"
            and isinstance(counts, Mapping)
            and all(
                counts.get(role) == 1
                for role in (
                    "same_hardware_receipt",
                    "phaxis_production_summary",
                    "v1_production_summary",
                    "phaxis_sequential_summary",
                    "v1_sequential_summary",
                    "production_comparison_receipt",
                    "sequential_comparison_receipt",
                )
            )
            and counts.get("per_image_latency_csv") == 2
            and counts.get("gpu_telemetry") == 4
            and counts.get("hardware_preflight") == 4,
            "benchmark inventory does not close all formal result roles",
        )
    elif name == "benchmark_same_hardware":
        expected = {
            "status": "passed",
            "images": 283,
            "measurement_scope": (
                "raw_image_to_final_traits_and_profiles_direct"
            ),
            "same_ordered_exact283_sources": True,
            "same_hardware_uuid_and_driver": True,
            "same_io_and_full_workflow_scope": True,
            "fresh_no_cache": True,
            "historical_98_47_min_component_receipt_used": False,
            "forward_only_runtime_used": False,
        }
        for field, value in expected.items():
            _require(
                payload.get(field) == value,
                f"same-hardware benchmark Gate failed: {field}",
            )
        runs = payload.get("runs")
        _require(
            isinstance(runs, list)
            and len(runs) == 4
            and all(
                run.get("fresh_direct_run") is True
                and run.get("resume_or_cache_used") is False
                and run.get("full_workflow_io_included") is True
                for run in runs
            ),
            "same-hardware benchmark contains cached or component-only runs",
        )
    elif name == "figure_inputs":
        _validate_publication_decision_bundle(context, plan)
    elif name == "figures":
        _validate_figure_table_bundle(receipt_path, payload)
        decision_bundle = _validate_publication_decision_bundle(context, plan)
        claim_contract = payload.get("claim_contract")
        _require(
            payload.get("narrative_decision_identity_sha256")
            == decision_bundle["narrative_decision_identity_sha256"]
            and payload.get("narrative_branch_id")
            == decision_bundle["narrative_branch_id"]
            and payload.get("title_contract")
            == decision_bundle["title_contract"]
            and isinstance(claim_contract, Mapping)
            and claim_contract.get("narrative_decision_identity_sha256")
            == decision_bundle["narrative_decision_identity_sha256"]
            and claim_contract.get("profile_hypothesis_tests_added") is False
            and claim_contract.get("profiles_select_or_veto_narrative_branch")
            is False,
            "figures: narrative decision, title, or profile non-inference contract differs from stage36",
        )
    elif name == "evidence":
        table = payload.get("supplementary_table_data")
        expected = _figure_table_binding(context, plan)
        _require(
            isinstance(table, Mapping)
            and dict(table) == expected,
            "evidence: supplementary Table/Data S1--S10 binding differs from figures",
        )
    elif name == "distributions":
        artifacts = payload.get("artifacts")
        release_assets = payload.get("release_assets")
        asset_inventory_record = payload.get("release_asset_inventory")
        checksum_record = payload.get("release_checksums")
        source_supply_chain = payload.get("source_supply_chain")
        sdist_audit = payload.get("sdist_archive_audit")
        wheel_audit = payload.get("wheel_archive_audit")
        build_toolchain = payload.get("build_toolchain")
        private_build_input = payload.get("private_build_input")
        source_before_lock = payload.get("source_release_before_lock")
        source_after_lock = payload.get("source_release_after_lock")
        commands = payload.get("commands")
        expected_release_asset_names = {
            "wheel": "phaxis-1.0.0-py3-none-any.whl",
            "sdist": "phaxis-1.0.0.tar.gz",
            "cyclonedx_sbom": "phaxis-1.0.0.cdx.json",
            "third_party_notices": "phaxis-1.0.0-THIRD_PARTY_NOTICES.md",
            "third_party_license_inventory": (
                "phaxis-1.0.0-THIRD_PARTY_LICENSES.json"
            ),
        }
        sdist_artifact = next(
            (
                artifact
                for artifact in artifacts
                if isinstance(artifact, Mapping) and artifact.get("kind") == "sdist"
            ),
            None,
        ) if isinstance(artifacts, list) else None
        _require(
            payload.get("status") == "completed_wheel_sdist_verified"
            and isinstance(artifacts, list)
            and {item.get("kind") for item in artifacts} == {"wheel", "sdist"}
            and len(artifacts) == 2
            and payload.get("twine_check_passed") is True,
            "distribution stage did not verify exactly one wheel and one sdist",
        )
        _require(
            isinstance(release_assets, list)
            and len(release_assets) == 5
            and [row.get("filename") for row in release_assets]
            == sorted(row.get("filename") for row in release_assets)
            and {row.get("kind") for row in release_assets}
            == set(expected_release_asset_names)
            and {
                str(row.get("kind")): row.get("filename")
                for row in release_assets
                if isinstance(row, Mapping)
            }
            == expected_release_asset_names
            and all(
                isinstance(row, Mapping)
                and isinstance(row.get("filename"), str)
                and isinstance(row.get("bytes"), int)
                and row["bytes"] > 0
                and _is_sha256(row.get("sha256"))
                for row in release_assets
            )
            and isinstance(source_supply_chain, list)
            and {row.get("path") for row in source_supply_chain}
            == {
                "NOTICE",
                "THIRD_PARTY_NOTICES.md",
                "THIRD_PARTY_LICENSES.json",
                "SBOM.cdx.json",
            }
            and all(
                isinstance(row, Mapping)
                and isinstance(row.get("bytes"), int)
                and row["bytes"] > 0
                and _is_sha256(row.get("sha256"))
                for row in source_supply_chain
            ),
            "distribution release asset/SBOM/notice inventory is incomplete",
        )
        release_asset_by_kind = {
            str(row["kind"]): row for row in release_assets
        }
        _require(
            isinstance(wheel_audit, Mapping)
            and wheel_audit.get("archive_filename")
            == expected_release_asset_names["wheel"]
            and wheel_audit.get("archive_sha256")
            == release_asset_by_kind["wheel"]["sha256"]
            and wheel_audit.get("distribution") == "phaxis"
            and wheel_audit.get("version") == "1.0.0"
            and wheel_audit.get("wheel_tag") == "py3-none-any"
            and isinstance(wheel_audit.get("metadata_member"), str)
            and str(wheel_audit["metadata_member"]).endswith(
                ".dist-info/METADATA"
            )
            and _is_sha256(wheel_audit.get("metadata_sha256"))
            and isinstance(wheel_audit.get("entry_points_member"), str)
            and str(wheel_audit["entry_points_member"]).endswith(
                ".dist-info/entry_points.txt"
            )
            and _is_sha256(wheel_audit.get("entry_points_sha256"))
            and wheel_audit.get("entry_point") == "phaxis = phaxis.cli:main"
            and isinstance(wheel_audit.get("record_member"), str)
            and str(wheel_audit["record_member"]).endswith(".dist-info/RECORD")
            and _is_sha256(wheel_audit.get("record_sha256"))
            and isinstance(wheel_audit.get("record_member_count"), int)
            and wheel_audit["record_member_count"] > 0
            and wheel_audit.get("record_verified") is True
            and wheel_audit.get("metadata_license_files")
            == ["LICENSE", "src/phaxis/_vendor/tomli/LICENSE.txt"]
            and wheel_audit.get("pep639_license_member_count") == 2
            and wheel_audit.get("license_file_hashes_verified") is True
            and isinstance(wheel_audit.get("source_package_file_count"), int)
            and wheel_audit["source_package_file_count"] > 0
            and _is_sha256(wheel_audit.get("source_package_identity_sha256"))
            and wheel_audit.get("source_package_hashes_verified") is True
            and wheel_audit.get("unexpected_payload_members") == 0
            and wheel_audit.get("prohibited_payload_members") == 0,
            "distribution wheel archive audit does not prove exact PHAxis code/metadata/license/RECORD closure",
        )
        unsigned_toolchain = (
            dict(build_toolchain) if isinstance(build_toolchain, Mapping) else {}
        )
        toolchain_identity = unsigned_toolchain.pop(
            "build_toolchain_identity_sha256", None
        )
        _require(
            isinstance(build_toolchain, Mapping)
            and toolchain_identity == sha256_json(unsigned_toolchain)
            and _is_sha256(toolchain_identity)
            and build_toolchain.get("implementation") == "CPython"
            and isinstance(build_toolchain.get("python_version"), str)
            and isinstance(build_toolchain.get("python_cache_tag"), str)
            and isinstance(build_toolchain.get("python_executable_filename"), str)
            and _is_sha256(build_toolchain.get("python_executable_sha256"))
            and isinstance(build_toolchain.get("packages"), Mapping)
            and set(build_toolchain["packages"])
            == {"build", "setuptools", "wheel", "twine"}
            and all(
                isinstance(value, str) and bool(value)
                for value in build_toolchain["packages"].values()
            )
            and build_toolchain.get("probe_isolated") is True
            and build_toolchain.get("cuda_visible_devices") == "-1"
            and build_toolchain.get("exact_versions_recorded") is True
            and build_toolchain.get("build_isolation_used") is False
            and payload.get("build_isolation_used") is False,
            "distribution build toolchain is absent, unsealed, or not exact/no-isolation",
        )
        _require(
            payload.get("source_release_input_immutable") is True
            and isinstance(private_build_input, Mapping)
            and private_build_input.get("role")
            == "private_manifest_exact_source_copy"
            and private_build_input.get("manifest_exact_copy_verified") is True
            and _is_sha256(private_build_input.get("source_manifest_sha256"))
            and private_build_input.get("source_manifest_sha256")
            == payload.get("source_release_manifest_sha256")
            and _is_sha256(private_build_input.get("tree_identity_sha256"))
            and isinstance(private_build_input.get("file_count_including_manifest"), int)
            and private_build_input["file_count_including_manifest"] > 1
            and isinstance(source_before_lock, Mapping)
            and source_before_lock == source_after_lock
            and _is_sha256(source_before_lock.get("identity_sha256"))
            and isinstance(source_before_lock.get("file_count"), int)
            and source_before_lock["file_count"] > 1,
            "distribution build did not prove private-copy construction and source immutability",
        )
        _require(
            isinstance(commands, list)
            and len(commands) == 2
            and all(
                isinstance(command, Mapping)
                and isinstance(command.get("argv"), list)
                and command["argv"]
                and all(
                    isinstance(argument, str)
                    and not _is_absolute_host_path(argument)
                    for argument in command["argv"]
                )
                for command in commands
            )
            and "<PRIVATE_DISTRIBUTION_OUTPUT>" in commands[0]["argv"]
            and "<PRIVATE_MANIFEST_EXACT_SOURCE_COPY>" in commands[0]["argv"]
            and "<PRIVATE_WHEEL>" in commands[1]["argv"]
            and "<PRIVATE_SDIST>" in commands[1]["argv"],
            "distribution command audit exposes a host path or lacks role argv",
        )
        source_supply_chain_by_path = {
            str(row["path"]): row for row in source_supply_chain
        }
        _require(
            all(
                (receipt_path.parent / str(row["filename"])).is_file()
                and not (receipt_path.parent / str(row["filename"])).is_symlink()
                and (receipt_path.parent / str(row["filename"])).stat().st_size
                == row["bytes"]
                and sha256_file(receipt_path.parent / str(row["filename"]))
                == row["sha256"]
                for row in release_assets
            )
            and release_asset_by_kind["cyclonedx_sbom"]["sha256"]
            == source_supply_chain_by_path["SBOM.cdx.json"]["sha256"]
            and release_asset_by_kind["third_party_notices"]["sha256"]
            == source_supply_chain_by_path["THIRD_PARTY_NOTICES.md"]["sha256"]
            and release_asset_by_kind["third_party_license_inventory"]["sha256"]
            == source_supply_chain_by_path["THIRD_PARTY_LICENSES.json"]["sha256"],
            "distribution release asset bytes do not match the sealed receipt/source closure",
        )
        _require(
            isinstance(asset_inventory_record, Mapping)
            and asset_inventory_record.get("filename")
            == "release_asset_inventory.json"
            and _is_sha256(asset_inventory_record.get("sha256"))
            and _is_sha256(asset_inventory_record.get("identity_sha256"))
            and isinstance(checksum_record, Mapping)
            and checksum_record.get("filename") == "SHA256SUMS"
            and checksum_record.get("algorithm") == "SHA-256"
            and checksum_record.get("entries") == len(release_assets)
            and _is_sha256(checksum_record.get("sha256")),
            "distribution release asset control-file identities are invalid",
        )
        inventory_path = receipt_path.parent / str(asset_inventory_record["filename"])
        checksum_path = receipt_path.parent / str(checksum_record["filename"])
        _require(
            inventory_path.is_file()
            and checksum_path.is_file()
            and sha256_file(inventory_path) == asset_inventory_record["sha256"]
            and sha256_file(checksum_path) == checksum_record["sha256"],
            "distribution release asset control files are absent or hash-mismatched",
        )
        asset_inventory = read_json(inventory_path)
        unsigned_asset_inventory = dict(asset_inventory)
        observed_inventory_identity = unsigned_asset_inventory.pop(
            "release_asset_inventory_identity_sha256", None
        )
        _require(
            asset_inventory.get("schema_version")
            == "PHAxis-release-asset-inventory-1.0"
            and asset_inventory.get("status") == "sealed_release_assets"
            and asset_inventory.get("distribution") == "phaxis"
            and asset_inventory.get("version") == "1.0.0"
            and asset_inventory.get("assets") == release_assets
            and asset_inventory.get("asset_count") == len(release_assets)
            and asset_inventory.get("source_supply_chain") == source_supply_chain
            and asset_inventory.get("blind_images_used") == 0
            and observed_inventory_identity
            == asset_inventory_record["identity_sha256"]
            == sha256_json(unsigned_asset_inventory),
            "distribution release asset inventory is not completely sealed",
        )
        expected_checksum_text = "".join(
            f"{row['sha256']}  {row['filename']}\n" for row in release_assets
        )
        _require(
            checksum_path.read_text(encoding="utf-8") == expected_checksum_text,
            "distribution SHA256SUMS does not exactly cover the release assets",
        )
        observed_generated = (
            sdist_audit.get("observed_pep517_generated_members")
            if isinstance(sdist_audit, Mapping)
            else None
        )
        _require(
            isinstance(sdist_audit, Mapping)
            and sdist_audit.get("source_manifest_self_covered") is True
            and sdist_audit.get("source_manifest_member") == "SOURCE_MANIFEST.json"
            and _is_sha256(sdist_audit.get("source_manifest_member_sha256"))
            and sdist_audit.get("authored_member_hashes_verified") is True
            and sdist_audit.get("allowed_pep517_generated_members")
            == list(PEP517_SDIST_GENERATED_MEMBERS)
            and isinstance(observed_generated, list)
            and [record.get("path") for record in observed_generated]
            == list(PEP517_SDIST_GENERATED_MEMBERS)
            and all(
                isinstance(record, Mapping)
                and isinstance(record.get("bytes"), int)
                and record["bytes"] >= 0
                and _is_sha256(record.get("sha256"))
                for record in observed_generated
            )
            and sdist_audit.get("unexpected_generated_members") == 0
            and sdist_audit.get("unexpected_generated_member_paths") == []
            and sdist_audit.get("missing_allowed_generated_members") == 0
            and sdist_audit.get("missing_allowed_generated_member_paths") == []
            and isinstance(sdist_artifact, Mapping)
            and _is_sha256(sdist_artifact.get("sha256"))
            and sdist_audit.get("archive_sha256") == sdist_artifact["sha256"],
            "distribution sdist audit does not preserve authored closure plus the exact eight PEP 517 metadata members",
        )
    elif name == "offline_dependencies":
        _validate_offline_dependencies(
            context,
            plan_stage,
            receipt_path,
            payload,
        )
    elif name == "clean_install":
        formal_wheel = payload.get("formal_wheel")
        distribution_wheel = context.artifact_paths.get(("distributions", "wheel"))
        source_stage = next(
            stage for stage in plan["stages"] if stage["name"] == "source_release"
        )
        source_manifest_path = context.artifact_paths[
            (
                "source_release",
                str(source_stage["receipt_contract"]["artifact"]),
            )
        ]
        _require(
            payload.get("status") == "completed_final_clean_install"
            and isinstance(formal_wheel, Mapping)
            and distribution_wheel is not None
            and distribution_wheel.is_file()
            and formal_wheel.get("sha256") == sha256_file(distribution_wheel)
            and formal_wheel.get("record_verified") is True
            and formal_wheel.get("source_package_hashes_verified") is True
            and formal_wheel.get("metadata_license_files")
            == ["LICENSE", "src/phaxis/_vendor/tomli/LICENSE.txt"]
            and formal_wheel.get("pep639_license_member_count") == 2
            and formal_wheel.get("license_file_hashes_verified") is True
            and payload.get("source_release_manifest_sha256")
            == sha256_file(source_manifest_path),
            "clean install does not reverify the exact distribution code/license/source closure",
        )
    elif name == "values":
        source_stage = next(
            stage for stage in plan["stages"] if stage["name"] == "source_release"
        )
        source_manifest_path = context.artifact_paths[
            (
                "source_release",
                str(source_stage["receipt_contract"]["artifact"]),
            )
        ]
        source_manifest = read_json(source_manifest_path)
        source_files = source_manifest.get("files")
        _require(
            source_manifest.get("schema_version")
            == KNOWN_STAGE_SCHEMAS["source_release"]
            and source_manifest.get("release_mode") == "formal"
            and source_manifest.get("distribution") == "phaxis"
            and source_manifest.get("version") == "1.0.0"
            and isinstance(source_files, list)
            and bool(source_files)
            and source_manifest.get("tree_identity_sha256")
            == sha256_json(source_files),
            "values: formal source-release authority is invalid",
        )
        source_records = {
            str(record["path"]): record
            for record in source_files
            if isinstance(record, Mapping) and isinstance(record.get("path"), str)
        }
        _require(
            len(source_records) == len(source_files)
            and {
                "RELEASE_HUMAN_METADATA.json",
                "LICENSE",
                "pyproject.toml",
                "CITATION.cff",
            }
            <= set(source_records),
            "values: source-release public-coordinate files are incomplete",
        )
        metadata_path = source_manifest_path.parent / "RELEASE_HUMAN_METADATA.json"
        metadata = read_json(metadata_path)
        metadata_identity = _sealed(
            metadata,
            "metadata_identity_sha256",
            role="values source-release human metadata",
        )
        _require(
            metadata.get("schema_version") == "PHAxis-release-human-metadata-1.3"
            and metadata.get("status") == "author_verified_release_authority"
            and sha256_file(metadata_path)
            == source_records["RELEASE_HUMAN_METADATA.json"].get("sha256"),
            "values: source-release human metadata is not final/hash-bound",
        )
        coordinates = metadata.get("release_coordinates")
        project_urls = metadata.get("project_urls")
        rights = metadata.get("rights")
        cross_binding = {
            "repository_url": (
                project_urls.get("Repository")
                if isinstance(project_urls, Mapping)
                else None
            ),
            "release_tag": (
                coordinates.get("github_release_tag")
                if isinstance(coordinates, Mapping)
                else None
            ),
            "version": source_manifest.get("version"),
            "release_doi": (
                coordinates.get("release_doi")
                if isinstance(coordinates, Mapping)
                else None
            ),
            "software_license": (
                rights.get("source_license_spdx")
                if isinstance(rights, Mapping)
                else None
            ),
            "source_release_tree_identity_sha256": source_manifest.get(
                "tree_identity_sha256"
            ),
            "source_release_manifest_sha256": sha256_file(source_manifest_path),
            "release_metadata_identity_sha256": metadata_identity,
            "release_metadata_sha256": sha256_file(metadata_path),
            "license_file_sha256": (
                rights.get("license_file_sha256")
                if isinstance(rights, Mapping)
                else None
            ),
            "pyproject_sha256": source_records["pyproject.toml"].get("sha256"),
            "citation_cff_sha256": source_records["CITATION.cff"].get("sha256"),
        }
        values_source_files = payload.get("source_files")
        manifest_source_record = (
            values_source_files.get("source_release_manifest")
            if isinstance(values_source_files, Mapping)
            else None
        )
        metadata_source_record = (
            values_source_files.get("source_release_metadata")
            if isinstance(values_source_files, Mapping)
            else None
        )
        decision_bundle = _validate_publication_decision_bundle(context, plan)
        figure_stage = next(
            stage for stage in plan["stages"] if stage["name"] == "figures"
        )
        figure_receipt = read_json(
            context.artifact_paths[
                ("figures", str(figure_stage["receipt_contract"]["artifact"]))
            ]
        )
        _require(
            payload.get("status") == "final_values_machine_derived_locked"
            and payload.get("builder_schema_version")
            == "PHAxis-manuscript-values-builder-1.1"
            and payload.get("source_release_manifest_file_sha256")
            == sha256_file(source_manifest_path)
            and payload.get("source_release_tree_identity_sha256")
            == source_manifest["tree_identity_sha256"]
            and payload.get("source_release_metadata_file_sha256")
            == sha256_file(metadata_path)
            and payload.get("source_release_metadata_identity_sha256")
            == metadata_identity
            and payload.get("software_release_cross_binding_identity_sha256")
            == sha256_json(cross_binding)
            and isinstance(manifest_source_record, Mapping)
            and manifest_source_record.get("sha256")
            == sha256_file(source_manifest_path)
            and manifest_source_record.get("logical_identity_sha256")
            == source_manifest["tree_identity_sha256"]
            and isinstance(metadata_source_record, Mapping)
            and metadata_source_record.get("sha256") == sha256_file(metadata_path)
            and metadata_source_record.get("logical_identity_sha256")
            == metadata_identity
            and payload.get("narrative_decision_identity_sha256")
            == decision_bundle["narrative_decision_identity_sha256"]
            == figure_receipt.get("narrative_decision_identity_sha256")
            and payload.get("narrative_branch_id")
            == decision_bundle["narrative_branch_id"]
            == figure_receipt.get("narrative_branch_id")
            and payload.get("narrative_branch_id") in {"A", "B", "C"},
            "values: source-release or narrative-decision cross-binding is incomplete",
        )
        _require(
            payload.get("publication_title_contract")
            == decision_bundle["title_contract"]
            == figure_receipt.get("title_contract"),
            "values: publication title contract differs from stage36/figures",
        )
    elif name == "manuscript":
        _require(
            payload.get("status") == "completed_strict_final_manuscript_compilation"
            and payload.get("unresolved_token_count") == 0
            and payload.get("author_metadata_complete") is True
            and payload.get("output_sha256") is not None,
            "compiled main Markdown is not final",
        )
    elif name == "supplementary_manuscript":
        expected_table = _figure_table_binding(context, plan)
        _require(
            payload.get("status") == "completed_strict_final_supplementary_compilation"
            and payload.get("unresolved_token_count") == 0
            and payload.get("numeric_or_author_values_inserted") == 0
            and payload.get("status_frontmatter_replacements") == 1
            and payload.get("supplementary_table_data_materialized") is True
            and payload.get("supplementary_table_data_resource_count") == 10
            and payload.get("supplementary_table_bundle_receipt_sha256")
            == expected_table["bundle_receipt_sha256"]
            and payload.get("supplementary_table_bundle_identity_sha256")
            == expected_table["bundle_identity_sha256"]
            and payload.get("supplementary_table_item_identity_sha256")
            == expected_table["ordered_item_identity_sha256"],
            "compiled supplementary Markdown is not authority-bound final",
        )
    elif name in {"submission_docx", "supplementary_docx"}:
        expected = (
            "completed_final_double_anonymous_submission_bundle"
            if name == "submission_docx"
            else "completed_final_anonymized_supplementary_docx"
        )
        artifact_names = {
            str(artifact["name"])
            for artifact in plan_stage["artifacts"]
            if isinstance(artifact, Mapping)
        }
        _require(
            payload.get("status") == expected
            and payload.get("mode") == "final"
            and payload.get("submission_use_allowed") is True
            and _is_sha256(payload.get("docx_sha256")),
            f"{name} is not a final hash-bound OOXML build",
        )
        if name == "submission_docx":
            title_path = context.artifact_paths[(name, "title_page")]
            anonymous_path = context.artifact_paths[(name, "anonymized_main")]
            _require(
                artifact_names == {"receipt", "title_page", "anonymized_main"}
                and payload.get("title_page_separate") is True
                and payload.get("anonymized_main_separate") is True
                and payload.get("reviewer_visible_identity_declarations_removed")
                is True
                and payload.get("anonymous_core_creator_empty") is True
                and _is_sha256(payload.get("submission_metadata_sha256"))
                and _is_sha256(
                    payload.get("submission_metadata_identity_sha256")
                )
                and isinstance(
                    payload.get("editor_only_declaration_sha256"), Mapping
                )
                and payload.get("title_page_docx_sha256")
                == sha256_file(title_path)
                and payload.get("anonymized_main_docx_sha256")
                == sha256_file(anonymous_path)
                and payload.get("docx_sha256") == sha256_file(anonymous_path),
                "submission_docx does not seal separate editor-only and reviewer-visible documents",
            )
        else:
            supplement_path = context.artifact_paths[(name, "anonymized_supplement")]
            expected_table = _figure_table_binding(context, plan)
            _require(
                artifact_names == {"receipt", "anonymized_supplement"}
                and payload.get("submission_metadata_consumed") is False
                and payload.get("reviewer_visible") is True
                and payload.get("anonymized_supplement_separate") is True
                and payload.get("anonymous_core_creator_empty") is True
                and payload.get("docx_sha256") == sha256_file(supplement_path)
                and payload.get("supplementary_table_data_materialized") is True
                and payload.get("supplementary_table_data_resource_count") == 10
                and payload.get("supplementary_table_bundle_receipt_sha256")
                == expected_table["bundle_receipt_sha256"]
                and payload.get("supplementary_table_bundle_identity_sha256")
                == expected_table["bundle_identity_sha256"]
                and payload.get("supplementary_table_item_identity_sha256")
                == expected_table["ordered_item_identity_sha256"],
                "supplementary_docx: Table/Data S1--S10 binding differs from figures",
            )
    elif name == "manuscript_artifact_qa":
        expected_table = _figure_table_binding(context, plan)
        upload_path = context.artifact_paths[(name, "upload_manifest")]
        upload = read_json(upload_path)
        upload_identity = _sealed(
            upload,
            "upload_manifest_identity_sha256",
            role="submission upload-role manifest",
        )
        roles = upload.get("roles")
        _require(
            payload.get("status")
            == "passed_double_anonymous_three_role_ooxml_closure"
            and payload.get("ooxml_zip_magic_and_required_structure_passed") is True
            and payload.get("master_authority_closure_passed") is True
            and payload.get("figure_input_closure_passed") is True
            and payload.get("supplementary_table_data_closure_passed") is True
            and payload.get("supplementary_table_data_closure")
            == expected_table
            and payload.get("data_and_code_availability_present") is True
            and isinstance(payload.get("availability_statement_closure"), Mapping)
            and isinstance(payload.get("title_page_ooxml"), Mapping)
            and isinstance(payload.get("main_ooxml"), Mapping)
            and isinstance(payload.get("supplement_ooxml"), Mapping)
            and payload.get("document_roles")
            == {
                "editor_only": ["title_page"],
                "reviewer_visible": [
                    "anonymized_main",
                    "anonymized_supplement",
                ],
            }
            and payload.get("reviewer_visible_identity_occurrence_count") == 0
            and payload.get("reviewer_visible_core_identity_occurrence_count")
            == 0
            and payload.get("reviewer_visible_tracked_change_count") == 0
            and payload.get("reviewer_visible_hidden_text_count") == 0
            and payload.get(
                "reviewer_visible_embedded_image_identity_occurrence_count"
            )
            == 0
            and payload.get("deep_ooxml_anonymity_scan_passed") is True
            and payload.get("editor_only_title_page_completeness_passed") is True
            and payload.get("submission_upload_role_manifest_sha256")
            == sha256_file(upload_path)
            and payload.get("submission_upload_role_manifest_identity_sha256")
            == upload_identity
            and payload.get("submission_use_allowed_before_visual_qa") is False,
            "manuscript Markdown/OOXML structural authority QA did not pass",
        )
        editor_only_roles = (
            roles.get("editor_only") if isinstance(roles, Mapping) else None
        )
        reviewer_visible_roles = (
            roles.get("reviewer_visible") if isinstance(roles, Mapping) else None
        )
        _require(
            upload.get("schema_version")
            == "PHAxis-submission-upload-role-manifest-1.0"
            and upload.get("status") == "sealed_editor_and_reviewer_upload_roles"
            and upload.get("submission_model") == "double_anonymous"
            and isinstance(roles, Mapping)
            and set(roles) == {"editor_only", "reviewer_visible"}
            and isinstance(editor_only_roles, Mapping)
            and set(editor_only_roles) == {"title_page"}
            and isinstance(reviewer_visible_roles, Mapping)
            and set(reviewer_visible_roles)
            == {"anonymized_main", "anonymized_supplement"}
            and upload.get("editor_only_document_count") == 1
            and upload.get("reviewer_visible_document_count") == 2
            and upload.get("reviewer_visible_identity_occurrence_count") == 0
            and upload.get("reviewer_visible_ooxml_deep_scan_passed") is True,
            "submission upload-role manifest is not an exact double-anonymous three-role closure",
        )
    elif name == "manuscript_render":
        documents = payload.get("documents")
        qa_stage = next(
            stage
            for stage in plan["stages"]
            if stage["name"] == "manuscript_artifact_qa"
        )
        upload_path = context.artifact_paths[("manuscript_artifact_qa", "upload_manifest")]
        upload = read_json(upload_path)
        _require(
            payload.get("status")
            == "completed_three_role_word_pdf_and_page_png_render"
            and payload.get("pdf_magic_passed") is True
            and payload.get("page_rasterization_completed") is True
            and payload.get("visual_qa_completed") is False
            and payload.get("submission_use_allowed") is False
            and payload.get("structural_qa_sha256")
            == sha256_file(
                context.artifact_paths[
                    (
                        "manuscript_artifact_qa",
                        str(qa_stage["receipt_contract"]["artifact"]),
                    )
                ]
            )
            and payload.get("submission_upload_role_manifest_sha256")
            == sha256_file(upload_path)
            and payload.get("submission_upload_role_manifest_identity_sha256")
            == upload.get("upload_manifest_identity_sha256")
            and isinstance(documents, Mapping)
            and set(documents)
            == {"title_page", "anonymized_main", "anonymized_supplement"}
            and all(
                isinstance(record, Mapping)
                and isinstance(record.get("pages"), int)
                and record["pages"] > 0
                and len(record.get("page_png_records", [])) == record["pages"]
                for record in documents.values()
            ),
            "final DOCX PDF/page rendering is incomplete",
        )
    elif name == "manuscript_visual_qa":
        _require(
            payload.get("status")
            == "passed_author_verified_three_role_page_visual_qa"
            and payload.get("documents_reviewed") == 3
            and payload.get("editor_only_documents_reviewed") == 1
            and payload.get("reviewer_visible_documents_reviewed") == 2
            and payload.get("reviewer_visible_identity_occurrence_count") == 0
            and payload.get("pages_reviewed", 0) > 0
            and payload.get("all_pages_reviewed_at_original_resolution") is True
            and payload.get("submission_visual_gate_passed") is True
            and payload.get("submission_use_allowed") is True,
            "author-verified final page visual QA is absent or incomplete",
        )
    elif name.startswith("handover_") and name.endswith("_manifest"):
        expected_role = name.removeprefix("handover_")
        _require(
            payload.get("status") == "created"
            and payload.get("materialisation_role") == expected_role,
            f"{name}: materialisation receipt role/status mismatch",
        )
    elif name == "handover_contract":
        _require(
            payload.get("status") == "created"
            and payload.get("bindings") == 16,
            "handover contract assembly did not close all 16 bindings",
        )
    elif name == "official_apply":
        _require(payload.get("official_model_contract_replaced") is True, "official apply receipt did not replace the CAS target")
    elif name == "release_finalize":
        registry_cas = plan_stage.get("release_registry_cas")
        _require(
            payload.get("formal_release_closed") is True
            and payload.get("terminal_stage") is True
            and payload.get("official_apply_preceded_post_apply_release_closure")
            is True
            and isinstance(registry_cas, Mapping)
            and payload.get("release_authority_registry_promotion_required") is True
            and payload.get("release_authority_registry_path")
            == _public_registry_path(
                Path(str(registry_cas.get("path"))), workspace=context.workspace
            )
            and payload.get("release_authority_registry_predecessor_sha256")
            == registry_cas.get("expected_sha256")
            and payload.get(
                "release_authority_registry_promotion_occurs_after_terminal_sentinel"
            )
            is True,
            "release finalization receipt does not close the post-apply release",
        )
        expected_upstream = {
            str(item["name"])
            for item in plan["stages"]
            if item["name"] != "release_finalize"
        }
        sentinel_identities = payload.get("upstream_sentinel_identity_sha256")
        receipt_sha256 = payload.get("upstream_receipt_sha256")
        _require(
            isinstance(sentinel_identities, Mapping)
            and set(sentinel_identities) == expected_upstream
            and all(_is_sha256(value) for value in sentinel_identities.values())
            and isinstance(receipt_sha256, Mapping)
            and set(receipt_sha256) == expected_upstream
            and all(_is_sha256(value) for value in receipt_sha256.values()),
            "release finalization does not seal every upstream sentinel/receipt",
        )
        expected_supply_chain = {
            stage: receipt_sha256[stage]
            for stage in (
                "source_release",
                "distributions",
                "offline_dependencies",
                "handover_model_source_manifest",
                "handover",
            )
        }
        _require(
            payload.get("software_supply_chain_closure_included") is True
            and payload.get("software_supply_chain_receipt_sha256")
            == expected_supply_chain,
            "release finalization does not expose the exact software supply-chain closure",
        )
    return payload, {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "logical_identity_field": identity_field,
        "logical_identity_sha256": (
            _value_at(payload, str(identity_field)) if identity_field is not None else None
        ),
    }


def _artifact_locks(context: _Context, plan_stage: Mapping[str, Any]) -> list[dict[str, Any]]:
    locks: list[dict[str, Any]] = []
    for artifact in plan_stage["artifacts"]:
        path = Path(artifact["path"])
        observed = _path_lock(path, str(artifact["kind"]))
        locks.append({"name": artifact["name"], "path": str(path), **observed})
    return locks


def _cas_preflight(context: _Context, plan_stage: Mapping[str, Any]) -> dict[str, Any] | None:
    cas = plan_stage.get("cas")
    if not cas:
        return None
    path = _resolve_input_path(str(cas["path"]), workspace=context.workspace, run_dir=context.run_dir)
    observed = sha256_file(path)
    if observed == cas["expected_sha256"]:
        return {
            "path": str(path),
            "expected_sha256": observed,
            "recovery_required": False,
        }
    stages = {str(stage["name"]): stage for stage in context.manifest["stages"]}
    proposal_artifact = str(stages["proposal"]["receipt"]["artifact"])
    proposal_path = context.artifact_paths[("proposal", proposal_artifact)]
    _require(proposal_path.is_file(), "official CAS target drifted before the proposal stage completed")
    applied = read_model_contract_authority(path)
    proposal = read_model_contract_proposal(proposal_path)
    _require(
        applied.authority_lifecycle == APPLIED_OFFICIAL_LIFECYCLE
        and applied.file_sha256 == proposal.file_sha256
        and applied.identity_sha256 == proposal.identity_sha256,
        "official CAS target drifted to an authority outside this release run",
    )
    recovered = _application_receipt_from_official(path)
    _require(
        recovered["expected_previous_model_contract_sha256"]
        == cas["expected_sha256"]
        and recovered["final_model_contract_sha256"] == observed,
        "applied official contract cannot recover this release CAS",
    )
    return {
        "path": str(path),
        "expected_sha256": str(cas["expected_sha256"]),
        "final_sha256": observed,
        "recovery_required": True,
        "recovered_application_identity_sha256": recovered[
            "application_identity_sha256"
        ],
    }


def _cas_postflight(
    context: _Context,
    plan: Mapping[str, Any],
    plan_stage: Mapping[str, Any],
    receipt: Mapping[str, Any],
    preflight: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if preflight is None:
        return None
    path = Path(preflight["path"])
    observed = sha256_file(path)
    _require(receipt.get("expected_previous_model_contract_sha256") == preflight["expected_sha256"], "application receipt CAS predecessor mismatch")
    _require(receipt.get("final_model_contract_sha256") == observed, "application receipt final contract SHA mismatch")
    binding = read_model_contract_authority(path)
    _require(binding.authority_lifecycle == APPLIED_OFFICIAL_LIFECYCLE, "authority CAS target is not an applied official contract")
    proposal_stage = next(
        stage for stage in plan["stages"] if stage["name"] == "proposal"
    )
    proposal_path = context.artifact_paths[("proposal", proposal_stage["receipt_contract"]["artifact"])]
    proposal = read_model_contract_proposal(proposal_path)
    _require(binding.file_sha256 == proposal.file_sha256 and binding.identity_sha256 == proposal.identity_sha256, "applied official contract does not preserve the run proposal authority")
    official = read_json(path)
    promotion = official.get("promotion")
    _require(isinstance(promotion, Mapping), "applied official contract lacks promotion evidence")
    source_authority = promotion.get("final_receipt_source_sha256")
    logical_authority = promotion.get("final_receipt_identity_sha256")
    _require(
        isinstance(source_authority, Mapping)
        and isinstance(logical_authority, Mapping),
        "applied official contract lacks final receipt authority",
    )
    role_stages = {
        "stageb": "production_stageb_exact283",
        "fusion": "fusion_exact283",
        "traits": "traits_exact283",
        "evidence": "evidence",
    }
    for role, stage_name in role_stages.items():
        final_stage = next(
            stage for stage in plan["stages"] if stage["name"] == stage_name
        )
        final_path = context.artifact_paths[
            (stage_name, final_stage["receipt_contract"]["artifact"])
        ]
        final_payload = read_json(final_path)
        identity_field = FINAL_PROMOTION_RECEIPT_IDENTITIES[stage_name]
        _require(
            source_authority.get(role) == sha256_file(final_path)
            and logical_authority.get(role) == final_payload.get(identity_field),
            f"applied official contract final receipt authority mismatch: {role}",
        )
    _require(
        receipt.get("final_evidence_manifest_sha256")
        == source_authority.get("evidence")
        and receipt.get("final_evidence_manifest_identity_sha256")
        == logical_authority.get("evidence"),
        "application receipt final evidence authority mismatch",
    )
    return {"path": str(path), "previous_sha256": preflight["expected_sha256"], "final_sha256": observed, "application_identity_sha256": receipt.get("application_identity_sha256")}


def _make_sentinel(
    *,
    plan: Mapping[str, Any],
    plan_stage: Mapping[str, Any],
    input_locks: Sequence[Mapping[str, Any]],
    artifact_locks: Sequence[Mapping[str, Any]],
    receipt_lock: Mapping[str, Any],
    command_result: Mapping[str, Any],
    gpu_preflight: Mapping[str, Any] | None,
    frozen_v1_locks: Mapping[str, Any],
    official_contract_guard: Mapping[str, Any] | None,
    cas: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SENTINEL_SCHEMA,
        "status": "completed_and_hash_verified",
        "run_id": plan["run_id"],
        "manifest_file_sha256": plan["manifest_file_sha256"],
        "release_plan_identity_sha256": plan["release_plan_identity_sha256"],
        "stage_index": plan_stage["index"],
        "stage_name": plan_stage["name"],
        "stage_plan_identity_sha256": plan_stage["stage_plan_identity_sha256"],
        "input_locks": list(input_locks),
        "artifact_locks": list(artifact_locks),
        "receipt_lock": dict(receipt_lock),
        "command_result": dict(command_result),
        "gpu_preflight": deepcopy(gpu_preflight),
        "frozen_v1_locks_after_stage": deepcopy(dict(frozen_v1_locks)),
        "official_contract_guard_after_stage": deepcopy(official_contract_guard),
        "official_contract_compare_and_swap": deepcopy(cas),
        "canonical_annotations_read": False,
        "condition_metadata_used_for_routing": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    payload["sentinel_identity_sha256"] = sha256_json(payload)
    return payload


def _validate_sentinel(
    context: _Context,
    plan: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    plan_stage = plan["stages"][index]
    path = _sentinel_path(context.run_dir, index, plan_stage["name"])
    _require(path.is_file(), f"resume sentinel is missing: {plan_stage['name']}")
    payload = read_json(path)
    _sealed(payload, "sentinel_identity_sha256", role=f"{plan_stage['name']} sentinel")
    expected = {
        "schema_version": SENTINEL_SCHEMA,
        "status": "completed_and_hash_verified",
        "run_id": plan["run_id"],
        "manifest_file_sha256": plan["manifest_file_sha256"],
        "release_plan_identity_sha256": plan["release_plan_identity_sha256"],
        "stage_index": index,
        "stage_name": plan_stage["name"],
        "stage_plan_identity_sha256": plan_stage["stage_plan_identity_sha256"],
        "blind_images_used": 0,
    }
    for field, value in expected.items():
        _require(payload.get(field) == value, f"resume sentinel mismatch: {plan_stage['name']} {field}")
    for input_lock in payload.get("input_locks", []):
        if (
            isinstance(input_lock, Mapping)
            and input_lock.get("deferred_authority_activated") is True
        ):
            external_name = str(input_lock.get("external"))
            descriptor = context.external_locks.get(external_name)
            _require(
                isinstance(descriptor, Mapping)
                and descriptor.get("deferred") is True
                and descriptor.get("human_authority_id")
                == input_lock.get("human_authority_id")
                and descriptor.get("document_schema_version")
                == input_lock.get("document_schema_version")
                and Path(str(descriptor.get("path"))).resolve()
                == Path(str(input_lock.get("path"))).resolve(),
                f"resume deferred human authority contract drifted: {external_name}",
            )
            _validate_deferred_activation_lock(input_lock)
    for lock in payload.get("artifact_locks", []):
        observed = _path_lock(Path(lock["path"]), str(lock["kind"]))
        for field in ("kind", "sha256", "size_bytes"):
            _require(observed[field] == lock[field], f"resume artifact drifted: {plan_stage['name']}/{lock.get('name')}")
    receipt, receipt_lock = _validate_receipt(
        context,
        plan,
        plan_stage,
        payload.get("gpu_preflight"),
    )
    _require(receipt_lock == payload.get("receipt_lock"), f"resume receipt drifted: {plan_stage['name']}")
    _validate_frozen_inputs(context)
    if plan_stage["name"] == "official_apply":
        cas = payload.get("official_contract_compare_and_swap")
        _require(
            isinstance(cas, Mapping)
            and _is_sha256(cas.get("final_sha256"))
            and Path(str(cas.get("path"))).is_file()
            and sha256_file(Path(str(cas["path"]))) == cas["final_sha256"]
            and receipt.get("final_model_contract_sha256") == cas["final_sha256"],
            "applied official model contract drifted after authority CAS",
        )
    official_index = next(
        i for i, item in enumerate(plan["stages"])
        if item["name"] == "official_apply"
    )
    if index > official_index:
        guard = payload.get("official_contract_guard_after_stage")
        _require(
            isinstance(guard, Mapping)
            and guard.get("authority_phase")
            == "applied_for_post_apply_release_closure"
            and _is_sha256(guard.get("sha256"))
            and Path(str(guard.get("path"))).is_file()
            and sha256_file(Path(str(guard["path"]))) == guard["sha256"],
            f"post-apply official authority drifted: {plan_stage['name']}",
        )
    return payload


def _write_internal_authority_pin(
    context: _Context,
    plan: Mapping[str, Any],
    plan_stage: Mapping[str, Any],
) -> None:
    proposal_stage = next(stage for stage in plan["stages"] if stage["name"] == "proposal")
    proposal_path = context.artifact_paths[("proposal", proposal_stage["receipt_contract"]["artifact"])]
    pin_path = context.artifact_paths[("authority_pin", plan_stage["receipt_contract"]["artifact"])]
    pin = build_run_scoped_authority_pin(
        proposal_path,
        pin_path=pin_path,
        run_id=str(plan["run_id"]),
        release_manifest_sha256=str(plan["manifest_file_sha256"]),
        release_plan_identity_sha256=str(plan["release_plan_identity_sha256"]),
    )
    _atomic_write_new_json(pin_path, pin)


def _write_internal_release_finalize(
    context: _Context,
    plan: Mapping[str, Any],
    plan_stage: Mapping[str, Any],
) -> None:
    """Seal release closure after every real post-apply producer succeeded."""

    _require(
        plan_stage["name"] == "release_finalize"
        and int(plan_stage["index"]) == len(plan["stages"]) - 1,
        "release finalization is not the terminal plan stage",
    )
    sentinel_identities: dict[str, str] = {}
    receipt_sha256: dict[str, str] = {}
    for index, upstream in enumerate(plan["stages"][:-1]):
        sentinel = _validate_sentinel(context, plan, index)
        sentinel_identities[str(upstream["name"])] = str(
            sentinel["sentinel_identity_sha256"]
        )
        receipt_sha256[str(upstream["name"])] = str(
            sentinel["receipt_lock"]["sha256"]
        )
    official_guard = _validate_official_contract_guard(context, plan_stage)
    _require(
        isinstance(official_guard, Mapping)
        and official_guard.get("authority_phase")
        == "applied_for_post_apply_release_closure",
        "release finalization lacks the applied official authority",
    )
    registry_cas = plan_stage.get("release_registry_cas")
    _require(
        isinstance(registry_cas, Mapping)
        and registry_cas.get("external") == "release_authority_registry"
        and _is_sha256(registry_cas.get("expected_sha256")),
        "release finalization lacks the terminal registry CAS contract",
    )
    registry_path = Path(str(registry_cas["path"])).resolve()
    _require(
        registry_path.is_file()
        and sha256_file(registry_path) == registry_cas["expected_sha256"],
        "release authority registry predecessor drifted before finalization",
    )
    payload: dict[str, Any] = {
        "schema_version": "PHAxis-post-training-release-finalization-1.0",
        "status": "completed_formal_release_closure",
        "formal_release_closed": True,
        "run_id": plan["run_id"],
        "release_plan_identity_sha256": plan["release_plan_identity_sha256"],
        "official_model_contract_sha256": official_guard["sha256"],
        "application_identity_sha256": official_guard[
            "application_identity_sha256"
        ],
        "upstream_sentinel_identity_sha256": sentinel_identities,
        "upstream_receipt_sha256": receipt_sha256,
        "software_supply_chain_receipt_sha256": {
            stage: receipt_sha256[stage]
            for stage in (
                "source_release",
                "distributions",
                "offline_dependencies",
                "handover_model_source_manifest",
                "handover",
            )
        },
        "software_supply_chain_closure_included": True,
        "terminal_stage": True,
        "official_apply_preceded_post_apply_release_closure": True,
        "release_authority_registry_promotion_required": True,
        "release_authority_registry_path": _public_registry_path(
            registry_path, workspace=context.workspace
        ),
        "release_authority_registry_predecessor_sha256": registry_cas[
            "expected_sha256"
        ],
        "release_authority_registry_promotion_occurs_after_terminal_sentinel": True,
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }
    payload["release_finalization_identity_sha256"] = sha256_json(payload)
    destination = context.artifact_paths[
        (
            "release_finalize",
            str(plan_stage["receipt_contract"]["artifact"]),
        )
    ]
    _atomic_write_new_json(destination, payload)


def _workspace_path(path: Path, *, workspace: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _public_registry_path(path: Path, *, workspace: Path) -> str:
    """Render a registry reference without disclosing a build-host path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return "<RELEASE_AUTHORITY_REGISTRY>"


def _promoted_release_registry_payload(
    context: _Context,
    plan: Mapping[str, Any],
    registry_cas: Mapping[str, Any],
) -> dict[str, Any]:
    registry_path = Path(str(registry_cas["path"])).resolve()
    predecessor = read_json(registry_path)
    _require(
        predecessor.get("schema_version")
        == PENDING_RELEASE_AUTHORITY_REGISTRY_SCHEMA
        and predecessor.get("current_formal_source_release") is None
        and predecessor.get("current_formal_release_gate_receipt") is None
        and predecessor.get("blind_images_used") == 0,
        "release authority registry predecessor cannot be promoted",
    )

    source_stage = next(stage for stage in plan["stages"] if stage["name"] == "source_release")
    source_manifest = context.artifact_paths[
        ("source_release", str(source_stage["receipt_contract"]["artifact"]))
    ]
    source_root = source_manifest.parent
    gate_path = source_root / "FORMAL_RELEASE_GATE_RECEIPT.json"
    final_stage = next(stage for stage in plan["stages"] if stage["name"] == "release_finalize")
    finalization_path = context.artifact_paths[
        ("release_finalize", str(final_stage["receipt_contract"]["artifact"]))
    ]
    _require(
        source_manifest.is_file()
        and gate_path.is_file()
        and finalization_path.is_file(),
        "terminal registry promotion inputs are incomplete",
    )
    source = read_json(source_manifest)
    gate = read_json(gate_path)
    finalization = read_json(finalization_path)
    _sealed(
        finalization,
        "release_finalization_identity_sha256",
        role="release finalization for registry promotion",
    )
    _require(
        source.get("schema_version") == KNOWN_STAGE_SCHEMAS["source_release"]
        and source.get("release_mode") == "formal"
        and source.get("distribution") == "phaxis"
        and source.get("version") == "1.0.0"
        and _is_sha256(source.get("tree_identity_sha256")),
        "formal source manifest is invalid for registry promotion",
    )
    _require(
        gate.get("status") == "passed"
        and gate.get("formal_release_allowed") is True
        and gate.get("release_mode") == "formal",
        "formal source release gate is not passed",
    )
    _recursive_release_guards(gate, role="formal source release gate")
    _require(
        finalization.get("status") == "completed_formal_release_closure"
        and finalization.get("run_id") == plan["run_id"]
        and finalization.get("release_plan_identity_sha256")
        == plan["release_plan_identity_sha256"]
        and finalization.get("release_authority_registry_predecessor_sha256")
        == registry_cas["expected_sha256"],
        "release finalization does not authorize this registry promotion",
    )
    metadata = gate.get("release_human_metadata")
    coordinates = metadata.get("release_coordinates") if isinstance(metadata, Mapping) else None
    release_date = coordinates.get("release_date") if isinstance(coordinates, Mapping) else None
    _require(
        isinstance(release_date, str)
        and bool(re.fullmatch(r"20\d{2}-\d{2}-\d{2}", release_date)),
        "formal release date is absent from the sealed human metadata",
    )

    payload = deepcopy(predecessor)
    payload.pop("registry_identity_sha256", None)
    payload["schema_version"] = PROMOTED_RELEASE_AUTHORITY_REGISTRY_SCHEMA
    payload["status"] = "formal_release_materialized_and_verified"
    payload["current_formal_source_release"] = {
        "path": _workspace_path(source_root, workspace=context.workspace),
        "source_manifest": _workspace_path(source_manifest, workspace=context.workspace),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_tree_identity_sha256": source["tree_identity_sha256"],
        "release_mode": "formal",
        "distribution": "phaxis",
        "version": "1.0.0",
    }
    payload["current_formal_release_gate_receipt"] = {
        "path": _workspace_path(gate_path, workspace=context.workspace),
        "sha256": sha256_file(gate_path),
        "schema_version": gate.get("schema_version"),
        "status": "passed",
        "formal_release_allowed": True,
    }
    payload["current_release_finalization"] = {
        "path": _workspace_path(finalization_path, workspace=context.workspace),
        "sha256": sha256_file(finalization_path),
        "release_finalization_identity_sha256": finalization[
            "release_finalization_identity_sha256"
        ],
        "run_id": plan["run_id"],
        "release_plan_identity_sha256": plan["release_plan_identity_sha256"],
    }
    release_control = deepcopy(dict(payload.get("release_control") or {}))
    release_control.update(
        {
            "release_manifest": _workspace_path(
                context.manifest_path, workspace=context.workspace
            ),
            "release_manifest_sha256": context.manifest_file_sha256,
            "release_plan_identity_sha256": plan["release_plan_identity_sha256"],
        }
    )
    payload["release_control"] = release_control
    payload["promotion"] = {
        "operation": "terminal_compare_and_swap_after_release_finalize_sentinel",
        "predecessor_sha256": registry_cas["expected_sha256"],
        "run_id": plan["run_id"],
        "manifest_file_sha256": context.manifest_file_sha256,
        "release_plan_identity_sha256": plan["release_plan_identity_sha256"],
        "blind_images_used": 0,
    }
    payload["updated_utc"] = f"{release_date}T00:00:00Z"
    payload["blind_images_used"] = 0
    payload["registry_identity_sha256"] = sha256_json(payload)
    return payload


def _validate_promoted_registry_release_bindings(
    context: _Context,
    plan: Mapping[str, Any],
    registry_cas: Mapping[str, Any],
    promoted: Mapping[str, Any],
) -> None:
    source_stage = next(stage for stage in plan["stages"] if stage["name"] == "source_release")
    source_manifest = context.artifact_paths[
        ("source_release", str(source_stage["receipt_contract"]["artifact"]))
    ]
    source_root = source_manifest.parent
    gate_path = source_root / "FORMAL_RELEASE_GATE_RECEIPT.json"
    final_stage = next(stage for stage in plan["stages"] if stage["name"] == "release_finalize")
    finalization_path = context.artifact_paths[
        ("release_finalize", str(final_stage["receipt_contract"]["artifact"]))
    ]
    source = read_json(source_manifest)
    finalization = read_json(finalization_path)
    source_ref = promoted.get("current_formal_source_release")
    gate_ref = promoted.get("current_formal_release_gate_receipt")
    final_ref = promoted.get("current_release_finalization")
    promotion = promoted.get("promotion")
    control = promoted.get("release_control")
    _require(
        promoted.get("public_identity")
        == {
            "product": "PHAxis",
            "version": "1.0.0",
            "distribution": "phaxis",
            "import_namespace": "phaxis",
            "cli": "phaxis",
            "release_tag": "v1.0.0",
        }
        and promoted.get("blind_images_used") == 0,
        "promoted release registry public identity or blind guard drifted",
    )
    _require(
        isinstance(source_ref, Mapping)
        and source_ref.get("path")
        == _workspace_path(source_root, workspace=context.workspace)
        and source_ref.get("source_manifest")
        == _workspace_path(source_manifest, workspace=context.workspace)
        and source_ref.get("source_manifest_sha256") == sha256_file(source_manifest)
        and source_ref.get("source_tree_identity_sha256")
        == source.get("tree_identity_sha256")
        and source_ref.get("release_mode") == "formal"
        and source_ref.get("distribution") == "phaxis"
        and source_ref.get("version") == "1.0.0",
        "promoted release registry source-release binding drifted",
    )
    _require(
        isinstance(gate_ref, Mapping)
        and gate_ref.get("path")
        == _workspace_path(gate_path, workspace=context.workspace)
        and gate_ref.get("sha256") == sha256_file(gate_path)
        and gate_ref.get("status") == "passed"
        and gate_ref.get("formal_release_allowed") is True,
        "promoted release registry formal-gate binding drifted",
    )
    _require(
        isinstance(final_ref, Mapping)
        and final_ref.get("path")
        == _workspace_path(finalization_path, workspace=context.workspace)
        and final_ref.get("sha256") == sha256_file(finalization_path)
        and final_ref.get("release_finalization_identity_sha256")
        == finalization.get("release_finalization_identity_sha256")
        and final_ref.get("run_id") == plan["run_id"]
        and final_ref.get("release_plan_identity_sha256")
        == plan["release_plan_identity_sha256"],
        "promoted release registry finalization binding drifted",
    )
    _require(
        isinstance(promotion, Mapping)
        and promotion.get("predecessor_sha256")
        == registry_cas["expected_sha256"]
        and promotion.get("run_id") == plan["run_id"]
        and promotion.get("manifest_file_sha256")
        == context.manifest_file_sha256
        and promotion.get("release_plan_identity_sha256")
        == plan["release_plan_identity_sha256"]
        and promotion.get("blind_images_used") == 0,
        "promoted release registry CAS provenance drifted",
    )
    _require(
        isinstance(control, Mapping)
        and control.get("release_manifest")
        == _workspace_path(context.manifest_path, workspace=context.workspace)
        and control.get("release_manifest_sha256")
        == context.manifest_file_sha256
        and control.get("release_plan_identity_sha256")
        == plan["release_plan_identity_sha256"],
        "promoted release registry release-control binding drifted",
    )


def _atomic_promote_release_registry(
    context: _Context,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    final_stage = next(stage for stage in plan["stages"] if stage["name"] == "release_finalize")
    registry_cas = final_stage.get("release_registry_cas")
    _require(
        isinstance(registry_cas, Mapping),
        "terminal release registry CAS is absent from the sealed plan",
    )
    path = Path(str(registry_cas["path"])).resolve()
    expected_sha256 = str(registry_cas["expected_sha256"])
    current_sha256 = sha256_file(path)
    if current_sha256 != expected_sha256:
        _validate_promoted_registry_recovery_target(
            path,
            expected_predecessor_sha256=expected_sha256,
            manifest=context.manifest,
            manifest_file_sha256=context.manifest_file_sha256,
            workspace=context.workspace,
            run_dir=context.run_dir,
        )
        promoted = read_json(path)
        _validate_promoted_registry_release_bindings(
            context, plan, registry_cas, promoted
        )
        status = "already_promoted_exact_idempotent_recovery"
    else:
        promoted = _promoted_release_registry_payload(context, plan, registry_cas)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".promotion.tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    promoted,
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _require(
                sha256_file(path) == expected_sha256,
                "release authority registry changed during compare-and-swap",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        _require(
            read_json(path) == promoted,
            "release authority registry replacement did not publish exact promoted bytes",
        )
        _validate_promoted_registry_release_bindings(
            context, plan, registry_cas, promoted
        )
        status = "promoted_by_terminal_compare_and_swap"
    return {
        "status": status,
        "path": str(path),
        "predecessor_sha256": expected_sha256,
        "promoted_sha256": sha256_file(path),
        "registry_identity_sha256": promoted["registry_identity_sha256"],
        "current_formal_source_release": deepcopy(
            promoted["current_formal_source_release"]
        ),
        "current_formal_release_gate_receipt": deepcopy(
            promoted["current_formal_release_gate_receipt"]
        ),
        "current_release_finalization": deepcopy(
            promoted["current_release_finalization"]
        ),
    }


def _recoverable_root_bundle_materialization(
    plan_stage: Mapping[str, Any],
) -> bool:
    """Validate a fully published root-bundle container after hard interruption.

    The producer publishes ``bundle/`` and ``verification.json`` in one
    directory rename.  A process can still die after that rename but before the
    orchestrator writes its sentinel.  Only that complete, strictly reverified
    state is recoverable; partial or tampered output remains fail-closed.
    """

    if plan_stage.get("name") != "root_bundle_materialization":
        return False
    artifacts = {
        str(artifact["name"]): Path(str(artifact["path"]))
        for artifact in plan_stage.get("artifacts", [])
        if isinstance(artifact, Mapping)
    }
    required = {"receipt", "bundle", "bundle_manifest"}
    if set(artifacts) != required:
        return False
    existing = {name for name, path in artifacts.items() if path.exists()}
    if not existing:
        return False
    _require(
        existing == required,
        "root_bundle_materialization interrupted with a partial atomic container",
    )
    _require(
        artifacts["receipt"].is_file()
        and artifacts["bundle"].is_dir()
        and artifacts["bundle_manifest"].is_file()
        and artifacts["bundle_manifest"].resolve()
        == (artifacts["bundle"] / "root_provider_bundle.json").resolve(),
        "root_bundle_materialization recovery paths are invalid",
    )
    try:
        observed = verify_bundle(
            artifacts["bundle"], require_exact_closure=True
        )
    except BundleError as error:
        raise ReleaseOrchestratorError(
            f"root_bundle_materialization recovery verification failed: {error}"
        ) from error
    receipt = read_json(artifacts["receipt"])
    for field, value in observed.items():
        _require(
            receipt.get(field) == value,
            f"root_bundle_materialization recovery receipt mismatch: {field}",
        )
    _require(
        receipt.get("source_bundle_mutated") is False
        and receipt.get("materialized_exact_closure") is True,
        "root_bundle_materialization recovery receipt lacks materialisation guards",
    )
    return True


def execute_release(
    manifest_path: str | Path,
    run_dir: str | Path,
    *,
    resume: bool = False,
    held_physical_gpus: Sequence[int] = (),
    candidate_builder: CandidateBuilder = build_candidate_manifest,
    command_runner: CommandRunner = _default_command_runner,
    gpu_probe: GpuProbe = _default_gpu_probe,
) -> dict[str, Any]:
    """Execute a validated release plan one sealed stage at a time."""

    held_values = tuple(held_physical_gpus)
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in held_values
        ),
        "held physical GPUs must be non-negative integers",
    )
    _require(
        len(set(held_values)) == len(held_values),
        "held physical GPUs contain duplicates",
    )
    held_gpu_set = set(held_values)

    context = _manifest_context(manifest_path, run_dir, candidate_builder=candidate_builder)
    plan = _release_plan_from_context(context)
    terminal_registry_required = isinstance(
        plan["stages"][-1].get("release_registry_cas"), Mapping
    )
    destination = context.run_dir
    state_path = destination / "state.json"
    if resume:
        _require(state_path.is_file(), "--resume requires an existing sealed state")
        state = read_json(state_path)
        _sealed(state, "state_identity_sha256", role="release state")
        _require(state.get("release_plan_identity_sha256") == plan["release_plan_identity_sha256"], "resume plan identity changed")
    else:
        _require(not destination.exists(), f"new release output already exists: {destination}")
        destination.mkdir(parents=True)
        state = _state_payload(plan, status="running", completed=[], current_stage=None)
        atomic_write_json(state_path, state)

    completed: list[str] = []
    first_missing = len(plan["stages"])
    for index, plan_stage in enumerate(plan["stages"]):
        sentinel_path = _sentinel_path(destination, index, plan_stage["name"])
        if sentinel_path.is_file():
            _validate_sentinel(context, plan, index)
            completed.append(str(plan_stage["name"]))
            continue
        first_missing = index
        break
    for later in plan["stages"][first_missing + 1 :]:
        _require(not _sentinel_path(destination, int(later["index"]), str(later["name"])).exists(), "release sentinels are not a contiguous prefix")
    if resume:
        atomic_write_json(
            state_path,
            _state_payload(
                plan,
                status=(
                    (
                        "finalizing_release_authority_registry"
                        if terminal_registry_required
                        else "completed"
                    )
                    if len(completed) == len(plan["stages"])
                    else "running"
                ),
                completed=completed,
                current_stage=None,
            ),
        )
    for stage in plan["stages"][first_missing:]:
        recoverable_final_apply = False
        if stage["name"] == "official_apply" and stage.get("cas"):
            cas_target = _resolve_input_path(
                str(stage["cas"]["path"]),
                workspace=context.workspace,
                run_dir=context.run_dir,
            )
            recoverable_final_apply = (
                sha256_file(cas_target) != stage["cas"]["expected_sha256"]
            )
        recoverable_root_bundle = bool(
            resume and _recoverable_root_bundle_materialization(stage)
        )
        for artifact in stage["artifacts"]:
            _require(
                not Path(artifact["path"]).exists()
                or recoverable_final_apply
                or recoverable_root_bundle,
                f"unsealed stage artifact already exists: {stage['name']}/{artifact['name']}",
            )

        required_physical_gpus = (
            set(int(value) for value in stage["gpu"]["physical_gpus"])
            if stage["gpu"] is not None
            else set()
        )
        blocked_physical_gpus = sorted(
            required_physical_gpus & held_gpu_set
        )
        if blocked_physical_gpus:
            gate = {
                "schema_version": "PHAxis-user-GPU-hold-gate-1.0",
                "status": "expected_user_gpu_hold_gate",
                "expected_pause_not_algorithm_or_training_failure": True,
                "resume_required": True,
                "expected_process_exit_code": EXPECTED_GPU_HOLD_EXIT_CODE,
                "stage_index": stage["index"],
                "stage_name": stage["name"],
                "stage_physical_gpus": sorted(required_physical_gpus),
                "held_physical_gpus": sorted(held_gpu_set),
                "blocked_physical_gpus": blocked_physical_gpus,
                "gpu_probe_called": False,
                "stage_command_started": False,
                "resume_policy": (
                    "resume only after the user explicitly releases every "
                    "blocked physical GPU"
                ),
            }
            paused_state = _state_payload(
                plan,
                status="paused_for_user_gpu_hold",
                completed=completed,
                current_stage=str(stage["name"]),
                gpu_hold_gate=gate,
            )
            atomic_write_json(state_path, paused_state)
            return paused_state

        pending_human = _pending_deferred_authority(context, plan, stage)
        if pending_human is not None:
            external_name, pending_reason = pending_human
            all_pending = _all_pending_deferred_work_items(context, plan)
            _require(
                all_pending,
                "current deferred human Gate was not represented in the batched work list",
            )
            primary = next(
                (
                    item
                    for item in all_pending
                    if item["external_name"] == external_name
                ),
                None,
            )
            _require(
                isinstance(primary, Mapping),
                "current deferred human Gate is absent from the batched work list",
            )
            gate = {
                "status": "expected_deferred_human_authority_gate",
                "expected_pause_not_algorithm_or_training_failure": True,
                "resume_required": True,
                "expected_process_exit_code": EXPECTED_HUMAN_GATE_EXIT_CODE,
                "stage_index": stage["index"],
                "stage_name": stage["name"],
                "external_name": external_name,
                "human_authority_id": context.external_locks[external_name][
                    "human_authority_id"
                ],
                "target_path": primary["target_path"],
                "work_item_path": primary["work_item_path"],
                "work_item_sha256": primary["work_item_sha256"],
                "work_item_identity_sha256": primary[
                    "work_item_identity_sha256"
                ],
                "work_item_is_not_success_artifact": True,
                "all_pending_deferred_authorities": all_pending,
                "all_pending_deferred_authority_count": len(all_pending),
            }
            paused_state = _state_payload(
                plan,
                status="paused_for_deferred_human_authority",
                completed=completed,
                current_stage=str(stage["name"]),
                human_authority_gate=gate,
            )
            atomic_write_json(state_path, paused_state)
            return paused_state

        current_state = _state_payload(
            plan,
            status="running",
            completed=completed,
            current_stage=str(stage["name"]),
        )
        atomic_write_json(state_path, current_state)
        try:
            frozen_before = _validate_frozen_inputs(context)
            official_guard_before = _validate_official_contract_guard(
                context, stage
            )
            input_locks = _input_locks(
                context,
                plan,
                stage,
                official_guard=official_guard_before,
            )
            cas_preflight = _cas_preflight(context, stage)
            environment = os.environ.copy()
            for key, value in stage["environment"].items():
                _require(key != "CUDA_VISIBLE_DEVICES", f"{stage['name']}: CVD must come only from the GPU mapping")
                environment[str(key)] = str(value)
            gpu_preflight = None
            if stage["gpu"] is not None:
                environment["CUDA_VISIBLE_DEVICES"] = str(stage["gpu"]["cuda_visible_devices"])
                # No model command is invoked until this result passes capacity checks.
                probe_result = gpu_probe(stage=str(stage["name"]))
                gpu_preflight = _parse_gpu_probe(
                    probe_result,
                    stage["gpu"],
                    stage=str(stage["name"]),
                )
            if stage["name"] == "authority_pin":
                _write_internal_authority_pin(context, plan, stage)
                result_code, stdout, stderr = 0, "internal atomic authority pin", ""
            elif stage["name"] == "release_finalize":
                _write_internal_release_finalize(context, plan, stage)
                result_code, stdout, stderr = (
                    0,
                    "internal atomic release finalization",
                    "",
                )
            elif (
                stage["name"] == "official_apply"
                and cas_preflight is not None
                and cas_preflight.get("recovery_required") is True
            ):
                receipt_path = context.artifact_paths[
                    ("official_apply", stage["receipt_contract"]["artifact"])
                ]
                recovered = _application_receipt_from_official(
                    Path(cas_preflight["path"])
                )
                if receipt_path.exists():
                    _require(
                        read_json(receipt_path) == recovered,
                        "existing application receipt differs from deterministic recovery",
                    )
                else:
                    _atomic_write_new_json(receipt_path, recovered)
                result_code, stdout, stderr = (
                    0,
                    "recovered application receipt from applied official authority",
                    "",
                )
            elif recoverable_root_bundle:
                result_code, stdout, stderr = (
                    0,
                    "recovered exact root-bundle materialization after atomic publication",
                    "",
                )
            else:
                result = command_runner(
                    command=stage["command"],
                    cwd=Path(stage["cwd"]),
                    env=environment,
                )
                result_code, stdout, stderr = _result_fields(result)
            _require(result_code == 0, f"{stage['name']}: command failed with exit code {result_code}: {stderr[-1000:]}")
            receipt, receipt_lock = _validate_receipt(
                context,
                plan,
                stage,
                gpu_preflight,
            )
            cas_postflight = _cas_postflight(
                context,
                plan,
                stage,
                receipt,
                cas_preflight,
            )
            artifact_locks = _artifact_locks(context, stage)
            for input_lock in input_locks:
                if input_lock.get("deferred_authority_activated") is True:
                    _validate_deferred_activation_lock(input_lock)
            frozen_after = _validate_frozen_inputs(context)
            _require(frozen_after == frozen_before, f"{stage['name']}: frozen-v1 inputs changed during stage")
            official_guard_after = _validate_official_contract_guard(
                context, stage
            )
            _require(
                official_guard_after == official_guard_before,
                f"{stage['name']}: official authority changed during stage",
            )
            command_result = {
                "returncode": result_code,
                "stdout_sha256": sha256_json(stdout.splitlines()),
                "stderr_sha256": sha256_json(stderr.splitlines()),
            }
            sentinel = _make_sentinel(
                plan=plan,
                plan_stage=stage,
                input_locks=input_locks,
                artifact_locks=artifact_locks,
                receipt_lock=receipt_lock,
                command_result=command_result,
                gpu_preflight=gpu_preflight,
                frozen_v1_locks=frozen_after,
                official_contract_guard=official_guard_after,
                cas=cas_postflight or cas_preflight,
            )
            _atomic_write_new_json(
                _sentinel_path(destination, int(stage["index"]), str(stage["name"])),
                sentinel,
            )
            completed.append(str(stage["name"]))
            atomic_write_json(
                state_path,
                _state_payload(
                    plan,
                    status=(
                        "finalizing_release_authority_registry"
                        if len(completed) == len(plan["stages"])
                        and terminal_registry_required
                        else "completed"
                        if len(completed) == len(plan["stages"])
                        else "running"
                    ),
                    completed=completed,
                    current_stage=None,
                ),
            )
        except Exception as error:
            applied_during_failed_stage = False
            if stage["name"] == "official_apply" and stage.get("cas"):
                failed_cas_path = _resolve_input_path(
                    str(stage["cas"]["path"]),
                    workspace=context.workspace,
                    run_dir=context.run_dir,
                )
                applied_during_failed_stage = (
                    failed_cas_path.is_file()
                    and sha256_file(failed_cas_path)
                    != stage["cas"]["expected_sha256"]
                )
            failure = {
                "stage": stage["name"],
                "error_type": type(error).__name__,
                "message": str(error),
            }
            atomic_write_json(
                state_path,
                _state_payload(
                    plan,
                    status="failed_closed",
                    completed=completed,
                    current_stage=str(stage["name"]),
                    failure=failure,
                    official_apply_performed=(
                        "official_apply" in completed
                        or applied_during_failed_stage
                    ),
                ),
            )
            raise
    registry_promotion = None
    if terminal_registry_required:
        try:
            registry_promotion = _atomic_promote_release_registry(context, plan)
            atomic_write_json(
                state_path,
                _state_payload(
                    plan,
                    status="completed",
                    completed=completed,
                    current_stage=None,
                    release_authority_registry_promotion=registry_promotion,
                ),
            )
        except Exception as error:
            failure = {
                "stage": "release_authority_registry_promotion",
                "error_type": type(error).__name__,
                "message": str(error),
            }
            atomic_write_json(
                state_path,
                _state_payload(
                    plan,
                    status="failed_closed",
                    completed=completed,
                    current_stage="release_authority_registry_promotion",
                    failure=failure,
                    official_apply_performed="official_apply" in completed,
                ),
            )
            raise
    final_state = read_json(state_path)
    _sealed(final_state, "state_identity_sha256", role="release state")
    _require(final_state.get("status") == "completed", "release execution did not complete")
    if terminal_registry_required:
        _require(
            final_state.get("release_authority_registry_promotion")
            == registry_promotion,
            "completed release state lacks the terminal registry promotion receipt",
        )
    return final_state


__all__ = [
    "APPLICATION_RECEIPT_SCHEMA",
    "DEFERRED_HUMAN_AUTHORITY_CONTRACT_SCHEMA",
    "DEFERRED_HUMAN_WORK_ITEM_SCHEMA",
    "EXPECTED_GPU_HOLD_EXIT_CODE",
    "EXPECTED_HUMAN_GATE_EXIT_CODE",
    "GPU_STAGE_NAMES",
    "MANIFEST_SCHEMA",
    "MANDATORY_STAGE_ORDER",
    "PLAN_SCHEMA",
    "ReleaseOrchestratorError",
    "SENTINEL_SCHEMA",
    "STATE_SCHEMA",
    "build_release_plan",
    "execute_release",
    "validate_deferred_human_authority_contract",
]
