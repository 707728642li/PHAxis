"""Installable, fail-closed PHAxis batch-analysis orchestration.

The public entry point is deliberately plan-only unless :func:`run_analysis`
is called (the CLI exposes that call only through ``phaxis analyze --execute``).
Importing this module does not import torch, inspect CUDA, start a subprocess,
or create an output directory.

The workflow manifest is an immutable deployment receipt.  Every file input is
represented by ``{"path": ..., "sha256": ...}``, the root-provider bundle is
bound by both its registry file and logical identity, and the complete JSON
object is sealed by ``manifest_identity_sha256``.  Paths may be relative to the
manifest, which keeps a copied GitHub checkout or installed wheel portable.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .constants import PRODUCT_NAME, PRODUCT_VERSION
from .contracts import ContractError, validate_hybrid_prediction, validate_stageb_detection_payload
from .io import atomic_write_json, read_json, sha256_file, sha256_json
from .model_contract_binding import (
    ModelContractProposalBinding,
    read_model_contract_authority,
    require_output_identity,
    require_receipt_binding,
    validate_stageb_proposal_binding,
)
from .root_provider.runtime import (
    PipelineConfig,
    build_execution_plan as build_root_provider_plan,
    run_pipeline as run_root_provider_pipeline,
)


WORKFLOW_MANIFEST_SCHEMA = "PHAxis-analysis-workflow-manifest-1.0"
WORKFLOW_PLAN_SCHEMA = "PHAxis-analysis-workflow-plan-1.0"
WORKFLOW_STATE_SCHEMA = "PHAxis-analysis-workflow-state-1.1"
STAGEB_BATCH_SCHEMA = "PHAxis-StageB-inference-run-1.1"
GPU_PREFLIGHT_SCHEMA = "PHAxis-StageB-GPU-preflight-1.0"

_SHA256_LENGTH = 64
_ROOT_FILE_KEYS = (
    "input_manifest",
    "acquisition_gate",
    "deployment_metadata",
    "canonical_manifest",
    "deployment_manifest",
    "deployment_lock",
)
_GUARDS = {
    "condition_metadata_used_for_routing": False,
    "canonical_annotations_read": False,
    "blind_images_used": 0,
    "root_cap_region_output": False,
}


@dataclass(frozen=True)
class LockedFile:
    """One manifest-relative file whose exact bytes are authorized."""

    name: str
    path: Path
    sha256: str

    def verify(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        observed = sha256_file(self.path)
        if observed.casefold() != self.sha256.casefold():
            raise ContractError(
                f"locked input hash mismatch for {self.name}: "
                f"expected {self.sha256}, got {observed}"
            )

    def plan_record(self) -> dict[str, str]:
        return {"path": str(self.path), "sha256": self.sha256.lower()}


@dataclass(frozen=True)
class StageBBatchConfig:
    """Strict train399 batch configuration used by the package runtime."""

    input_manifest: Path
    checkpoints: tuple[Path, ...]
    candidate_manifest: Path
    selected_model_metadata: Path
    selection_receipt: Path
    physical_gpu: int
    internal_device: str = "cuda:0"
    shared_input_acceleration: bool = False
    shared_input_max_host_bytes: int = 2 * 1024**3
    shared_input_max_device_bytes: int = 1 * 1024**3
    shared_input_device_reserve_bytes: int = 2 * 1024**3
    estimated_peak_vram_mib: int = 8192
    required_free_vram_reserve_mib: int = 2048
    image_root: Path | None = None
    file_sha256: Mapping[str, str] | None = None


@dataclass(frozen=True)
class _WorkflowContext:
    manifest_path: Path
    manifest: Mapping[str, Any]
    output: Path
    plan: Mapping[str, Any]
    root_config: PipelineConfig
    stageb_config: StageBBatchConfig
    stageb_rows: tuple[Mapping[str, Any], ...]
    model_contract_proposal: LockedFile
    model_contract_binding: ModelContractProposalBinding
    traits_metadata: LockedFile
    profile_contract: LockedFile
    review_overlays: bool


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _resolve_path(value: Any, *, base: Path, field: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise ContractError(f"{field} must be a non-empty path")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _locked_file(value: Any, *, base: Path, name: str) -> LockedFile:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a locked {{path, sha256}} object")
    digest = value.get("sha256")
    if not _is_sha256(digest):
        raise ContractError(f"{name}.sha256 is absent or invalid")
    return LockedFile(
        name=name,
        path=_resolve_path(value.get("path"), base=base, field=f"{name}.path"),
        sha256=str(digest).lower(),
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ContractError(f"empty CSV is not a valid workflow input: {path}")
    return rows


def _sealed_identity(payload: Mapping[str, Any], field: str) -> str:
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field, None)
    if not _is_sha256(observed):
        raise ContractError(f"{field} is absent or invalid")
    expected = sha256_json(unsigned)
    if str(observed).casefold() != expected.casefold():
        raise ContractError(f"{field} does not seal the complete JSON receipt")
    return expected


def load_analysis_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate the immutable top-level policy receipt.

    File hashes and cross-file train399 bindings are checked while the plan is
    constructed.  This first layer intentionally rejects a partial Gate before
    any GPU preflight can be reached.
    """

    manifest_path = Path(path).resolve()
    payload = read_json(manifest_path)
    if payload.get("schema_version") != WORKFLOW_MANIFEST_SCHEMA:
        raise ContractError("unsupported PHAxis analysis workflow manifest schema")
    guards = payload.get("guards")
    if not isinstance(guards, Mapping):
        raise ContractError("workflow manifest has no inference guards")
    for field, expected in _GUARDS.items():
        if guards.get(field) != expected:
            raise ContractError(f"workflow guard must remain {field}={expected!r}")
    stageb = payload.get("stageb")
    if not isinstance(stageb, Mapping):
        raise ContractError("workflow manifest has no strict Stage-B section")
    gate_fields = (
        "candidate_manifest",
        "selected_model_metadata",
        "selection_receipt",
    )
    present = [stageb.get(field) is not None for field in gate_fields]
    if any(present) and not all(present):
        raise ContractError(
            "strict train399 Stage-B requires candidate_manifest, "
            "selected_model_metadata and selection_receipt together"
        )
    if not all(present):
        raise ContractError("strict train399 Stage-B requires all three Gate receipts")
    if not isinstance(payload.get("model_contract_proposal"), Mapping):
        raise ContractError("workflow manifest requires a locked model_contract_proposal")
    _sealed_identity(payload, "manifest_identity_sha256")
    return payload


def _validate_train399_gate(
    *,
    candidate_manifest: Path,
    selected_model_metadata: Path,
    selection_receipt: Path,
    checkpoints: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the three deployment receipts without importing torch."""

    from .hair_stageb.candidate_bundle import read_candidate_manifest
    from .hair_stageb.selection import (
        read_selection_receipt,
        validate_selected_operating_point_binding,
    )

    candidate = read_candidate_manifest(candidate_manifest)
    selected = read_json(selected_model_metadata)
    receipt = read_selection_receipt(selection_receipt)
    validate_selected_operating_point_binding(
        candidate_manifest=candidate,
        selected_model_metadata=selected,
        selection_receipt=receipt,
        selection_receipt_file_sha256=sha256_file(selection_receipt),
    )
    expected = list(candidate["detection_model_metadata"]["checkpoint_sha256"])
    observed = [sha256_file(path) for path in checkpoints]
    if len(checkpoints) != 5 or len(set(observed)) != 5 or set(observed) != set(expected):
        raise ContractError(
            "Stage-B requires exactly the five distinct checkpoints sealed by "
            "the train399 candidate manifest"
        )
    if selected.get("checkpoint_policy") != "five_seed_train399_last_epoch_60":
        raise ContractError("selected Stage-B metadata is not strict train399")
    if selected.get("ensemble_members") != 5 or selected.get("training_images") != 399:
        raise ContractError("selected Stage-B metadata is not the five-member train399 expert")
    if selected.get("blind_images_used") != 0:
        raise ContractError("selected Stage-B metadata is blind-tainted")
    return candidate, selected


def _stageb_rows(
    config: StageBBatchConfig,
    *,
    verify_images: bool,
) -> tuple[dict[str, Any], ...]:
    rows = _read_csv(config.input_manifest)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    base = config.image_root.resolve() if config.image_root is not None else config.input_manifest.parent
    for index, row in enumerate(rows, start=2):
        task_id = str(row.get("task_id") or row.get("image_id") or "")
        if not task_id or task_id in seen or Path(task_id).name != task_id:
            raise ContractError(f"invalid or duplicate Stage-B task_id at CSV row {index}")
        seen.add(task_id)
        raw_path = row.get("image_path") or row.get("input_path")
        image_path = _resolve_path(raw_path, base=base, field=f"row {index}.image_path")
        image_sha256 = str(
            row.get("image_sha256") or row.get("source_image_sha256") or ""
        ).lower()
        if not _is_sha256(image_sha256):
            raise ContractError(f"{task_id}: image_sha256 is absent or invalid")
        try:
            um_per_px = float(row.get("um_per_px") or row.get("source_um_per_px"))
        except (TypeError, ValueError) as error:
            raise ContractError(f"{task_id}: source scale is absent or invalid") from error
        if not math.isfinite(um_per_px) or um_per_px <= 0:
            raise ContractError(f"{task_id}: source scale must be finite and positive")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if verify_images:
            observed = sha256_file(image_path)
            if observed.casefold() != image_sha256.casefold():
                raise ContractError(f"{task_id}: locked source-image hash mismatch")
        normalized.append(
            {
                "task_id": task_id,
                "image_path": str(image_path),
                "image_sha256": image_sha256,
                "um_per_px": um_per_px,
            }
        )
    return tuple(sorted(normalized, key=lambda item: str(item["task_id"])))


def _task_scale_map(
    path: Path,
    *,
    task_fields: Sequence[str],
    scale_fields: Sequence[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in _read_csv(path):
        task_id = next((str(row.get(name) or "") for name in task_fields if row.get(name)), "")
        if not task_id or task_id in result:
            raise ContractError(f"empty or duplicate task identity in {path}")
        raw_scale = next((row.get(name) for name in scale_fields if row.get(name)), None)
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError) as error:
            raise ContractError(f"{task_id}: missing physical scale in {path}") from error
        if not math.isfinite(scale) or scale <= 0:
            raise ContractError(f"{task_id}: invalid physical scale in {path}")
        result[task_id] = scale
    return result


def _root_source_locks(path: Path) -> dict[str, dict[str, Any]]:
    """Bind the raw images actually consumed by the root-provider manifest."""

    result: dict[str, dict[str, Any]] = {}
    for row in _read_csv(path):
        task_id = str(row.get("image_id") or row.get("task_id") or "")
        if not task_id or task_id in result:
            raise ContractError(f"empty or duplicate root-provider task in {path}")
        source = _resolve_path(
            row.get("input_path") or row.get("image_path"),
            base=path.parent,
            field=f"{task_id}.root_provider.input_path",
        )
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            scale = float(row.get("source_um_per_px") or row.get("um_per_px"))
        except (TypeError, ValueError) as error:
            raise ContractError(f"{task_id}: missing root-provider physical scale") from error
        if not math.isfinite(scale) or scale <= 0:
            raise ContractError(f"{task_id}: invalid root-provider physical scale")
        result[task_id] = {
            "image_path": str(source),
            "image_sha256": sha256_file(source),
            "um_per_px": scale,
        }
    return result


def _traits_source_locks(path: Path) -> dict[str, dict[str, Any]]:
    """Read only identity/scale fields; condition fields never leave traits."""

    result: dict[str, dict[str, Any]] = {}
    for row in _read_csv(path):
        task_id = str(row.get("task_id") or row.get("image_id") or "")
        if not task_id or task_id in result:
            raise ContractError(f"empty or duplicate trait task in {path}")
        digest = str(
            row.get("image_sha256") or row.get("source_image_sha256") or ""
        ).lower()
        if not _is_sha256(digest):
            raise ContractError(f"{task_id}: trait metadata image hash is invalid")
        try:
            scale = float(row.get("um_per_px") or row.get("source_um_per_px"))
        except (TypeError, ValueError) as error:
            raise ContractError(f"{task_id}: missing trait physical scale") from error
        if not math.isfinite(scale) or scale <= 0:
            raise ContractError(f"{task_id}: invalid trait physical scale")
        result[task_id] = {"image_sha256": digest, "um_per_px": scale}
    return result


def _tree_files(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise ContractError(f"stage evidence cannot contain a symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    if not result:
        raise ContractError(f"completed stage has no evidence files: {root}")
    return result


def _plan_stage(
    *,
    name: str,
    input_hashes: Mapping[str, Any],
    output: Path,
    estimated_gpu: Mapping[str, Any],
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stage: dict[str, Any] = {
        "name": name,
        "input_hashes": dict(sorted(input_hashes.items())),
        "output": str(output),
        "estimated_gpu": dict(estimated_gpu),
    }
    if detail:
        stage["detail"] = dict(detail)
    stage["stage_plan_identity_sha256"] = sha256_json(stage)
    return stage


def _context(
    manifest_path: str | Path,
    *,
    output: str | Path,
    review_overlays: bool | None,
) -> _WorkflowContext:
    manifest_path = Path(manifest_path).resolve()
    payload = load_analysis_manifest(manifest_path)
    base = manifest_path.parent
    output_path = Path(output).resolve()

    proposal_ref = _locked_file(
        payload["model_contract_proposal"],
        base=base,
        name="model_contract_proposal",
    )
    proposal_ref.verify()
    proposal_binding = read_model_contract_authority(
        proposal_ref.path,
        expected_file_sha256=proposal_ref.sha256,
    )

    root = payload.get("root_provider")
    if not isinstance(root, Mapping):
        raise ContractError("workflow manifest has no root_provider section")
    root_refs = {
        key: _locked_file(root.get(key), base=base, name=f"root_provider.{key}")
        for key in _ROOT_FILE_KEYS
    }
    for ref in root_refs.values():
        ref.verify()

    bundle_value = root.get("bundle")
    if not isinstance(bundle_value, Mapping):
        raise ContractError("root_provider.bundle must bind a directory registry")
    bundle = _resolve_path(
        bundle_value.get("path"), base=base, field="root_provider.bundle.path"
    )
    registry_digest = bundle_value.get("registry_sha256")
    bundle_identity = bundle_value.get("bundle_identity_sha256")
    if not _is_sha256(registry_digest) or not _is_sha256(bundle_identity):
        raise ContractError(
            "root_provider.bundle requires registry_sha256 and bundle_identity_sha256"
        )
    bundle_registry = bundle / "root_provider_bundle.json"
    if not bundle_registry.is_file():
        raise FileNotFoundError(bundle_registry)
    if sha256_file(bundle_registry) != str(registry_digest).lower():
        raise ContractError("root-provider bundle registry hash mismatch")
    registry_payload = read_json(bundle_registry)
    if registry_payload.get("bundle_identity_sha256") != str(bundle_identity).lower():
        raise ContractError("root-provider logical bundle identity mismatch")
    if proposal_binding.root_bundle_identity_sha256 != str(bundle_identity).lower():
        raise ContractError(
            "workflow root-provider bundle differs from proposal root authority"
        )

    project = _resolve_path(root.get("project", "."), base=base, field="root_provider.project")
    python_executable = _resolve_path(
        root.get("python_executable", sys.executable),
        base=base,
        field="root_provider.python_executable",
    )
    image_root = _resolve_path(root.get("image_root"), base=base, field="root_provider.image_root")
    if not project.is_dir():
        raise FileNotFoundError(project)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    if not python_executable.is_file():
        raise FileNotFoundError(python_executable)

    def _gpu_tuple(name: str) -> tuple[int, ...]:
        value = root.get(name)
        if not isinstance(value, list) or not value:
            raise ContractError(f"root_provider.{name} must be a non-empty list")
        if any(isinstance(item, bool) or int(item) < 0 for item in value):
            raise ContractError(f"root_provider.{name} contains an invalid GPU")
        return tuple(int(item) for item in value)

    reference_ref: LockedFile | None = None
    if root.get("reference_registry") is not None:
        reference_ref = _locked_file(
            root["reference_registry"], base=base, name="root_provider.reference_registry"
        )
        reference_ref.verify()
    strict_physical_gpu = root.get("strict_physical_gpu", False)
    if not isinstance(strict_physical_gpu, bool):
        raise ContractError("root_provider.strict_physical_gpu must be boolean")
    root_config = PipelineConfig(
        project=project,
        bundle=bundle,
        input_manifest=root_refs["input_manifest"].path,
        acquisition_gate=root_refs["acquisition_gate"].path,
        deployment_metadata=root_refs["deployment_metadata"].path,
        canonical_manifest=root_refs["canonical_manifest"].path,
        deployment_manifest=root_refs["deployment_manifest"].path,
        deployment_lock=root_refs["deployment_lock"].path,
        image_root=image_root,
        output=output_path / "root_provider",
        v1_physical_gpus=_gpu_tuple("v1_physical_gpus"),
        q8_physical_gpus=_gpu_tuple("q8_physical_gpus"),
        python_executable=python_executable,
        v1_shards=int(root.get("v1_shards", 4)),
        v1_concurrency=int(root.get("v1_concurrency", 2)),
        v20_shards=int(root.get("v20_shards", 8)),
        v20_concurrency=int(root.get("v20_concurrency", 8)),
        q8_shards=int(root.get("q8_shards", 8)),
        q8_concurrency=int(root.get("q8_concurrency", 1)),
        field_batch_size=int(root.get("field_batch_size", 10)),
        query_batch_size=int(root.get("query_batch_size", 32)),
        reference_registry=reference_ref.path if reference_ref else None,
        strict_physical_gpu=strict_physical_gpu,
    )

    stageb = payload["stageb"]
    stageb_manifest_value = stageb.get("input_manifest", stageb.get("manifest"))
    stageb_manifest = _locked_file(
        stageb_manifest_value, base=base, name="stageb.input_manifest"
    )
    checkpoint_values = stageb.get("checkpoints")
    if not isinstance(checkpoint_values, list) or len(checkpoint_values) != 5:
        raise ContractError("stageb.checkpoints must contain exactly five locked files")
    checkpoint_refs = tuple(
        _locked_file(value, base=base, name=f"stageb.checkpoints[{index}]")
        for index, value in enumerate(checkpoint_values)
    )
    candidate_ref = _locked_file(
        stageb["candidate_manifest"], base=base, name="stageb.candidate_manifest"
    )
    selected_ref = _locked_file(
        stageb["selected_model_metadata"],
        base=base,
        name="stageb.selected_model_metadata",
    )
    selection_ref = _locked_file(
        stageb["selection_receipt"], base=base, name="stageb.selection_receipt"
    )
    stageb_refs = (
        stageb_manifest,
        *checkpoint_refs,
        candidate_ref,
        selected_ref,
        selection_ref,
    )
    for ref in stageb_refs:
        ref.verify()

    physical_gpu = stageb.get("physical_gpu")
    if isinstance(physical_gpu, bool) or not isinstance(physical_gpu, int) or physical_gpu < 0:
        raise ContractError("stageb.physical_gpu must be an explicit non-negative integer")
    internal_device = str(stageb.get("internal_device", "cuda:0"))
    if internal_device != "cuda:0":
        raise ContractError(
            "Stage-B uses a one-card CUDA_VISIBLE_DEVICES mapping; internal_device must be cuda:0"
        )
    shared = stageb.get("shared_input_acceleration", False)
    if not isinstance(shared, bool):
        raise ContractError("stageb.shared_input_acceleration must be boolean")
    stageb_config = StageBBatchConfig(
        input_manifest=stageb_manifest.path,
        checkpoints=tuple(ref.path for ref in checkpoint_refs),
        candidate_manifest=candidate_ref.path,
        selected_model_metadata=selected_ref.path,
        selection_receipt=selection_ref.path,
        physical_gpu=physical_gpu,
        internal_device=internal_device,
        shared_input_acceleration=shared,
        shared_input_max_host_bytes=int(stageb.get("shared_input_max_host_bytes", 2 * 1024**3)),
        shared_input_max_device_bytes=int(stageb.get("shared_input_max_device_bytes", 1 * 1024**3)),
        shared_input_device_reserve_bytes=int(stageb.get("shared_input_device_reserve_bytes", 2 * 1024**3)),
        estimated_peak_vram_mib=int(stageb.get("estimated_peak_vram_mib", 8192)),
        required_free_vram_reserve_mib=int(stageb.get("required_free_vram_reserve_mib", 2048)),
        image_root=_resolve_path(
            stageb.get("image_root", image_root), base=base, field="stageb.image_root"
        ),
        file_sha256={ref.name: ref.sha256 for ref in stageb_refs},
    )
    for field in (
        stageb_config.shared_input_max_host_bytes,
        stageb_config.shared_input_max_device_bytes,
        stageb_config.estimated_peak_vram_mib,
        stageb_config.required_free_vram_reserve_mib,
    ):
        if field <= 0:
            raise ContractError("Stage-B memory limits and estimates must be positive")
    if stageb_config.shared_input_device_reserve_bytes < 0:
        raise ContractError("Stage-B device-memory reserve cannot be negative")

    candidate, selected_metadata = _validate_train399_gate(
        candidate_manifest=candidate_ref.path,
        selected_model_metadata=selected_ref.path,
        selection_receipt=selection_ref.path,
        checkpoints=stageb_config.checkpoints,
    )
    validate_stageb_proposal_binding(
        proposal_binding,
        candidate_manifest_path=candidate_ref.path,
        candidate_manifest=candidate,
        selected_model_metadata_path=selected_ref.path,
        selected_model_metadata=selected_metadata,
        selection_receipt_path=selection_ref.path,
        selection_receipt=read_json(selection_ref.path),
        checkpoints=stageb_config.checkpoints,
    )
    rows = _stageb_rows(stageb_config, verify_images=True)

    traits_value = payload.get("traits")
    if not isinstance(traits_value, Mapping):
        raise ContractError("workflow manifest has no traits section")
    traits_ref = _locked_file(
        traits_value.get("metadata_csv"), base=base, name="traits.metadata_csv"
    )
    traits_ref.verify()
    profiles_value = payload.get("distal_axis_profiles", payload.get("profiles"))
    if not isinstance(profiles_value, Mapping):
        raise ContractError("workflow manifest has no distal_axis_profiles section")
    profile_ref = _locked_file(
        profiles_value.get("contract_json"),
        base=base,
        name="distal_axis_profiles.contract_json",
    )
    profile_ref.verify()
    profile_payload = read_json(profile_ref.path)
    if (
        profile_payload.get("root_cap_region_output") is not False
        or profile_payload.get("canonical_annotations_read") is not False
        or profile_payload.get("blind_images_used") != 0
        or profile_payload.get("stageb_two_point_vector_used_as_length") is not False
    ):
        raise ContractError("distal-axis profile contract violates PHAxis inference guards")

    stage_locks = {
        str(row["task_id"]): {
            "image_sha256": str(row["image_sha256"]),
            "um_per_px": float(row["um_per_px"]),
        }
        for row in rows
    }
    root_locks = _root_source_locks(root_refs["input_manifest"].path)
    trait_locks = _traits_source_locks(traits_ref.path)
    if set(stage_locks) != set(root_locks) or set(stage_locks) != set(trait_locks):
        raise ContractError("root-provider/Stage-B/traits task sets differ")
    for task_id, stage_lock in stage_locks.items():
        scale = float(stage_lock["um_per_px"])
        if root_locks[task_id]["image_sha256"] != stage_lock["image_sha256"]:
            raise ContractError(f"{task_id}: root-provider/Stage-B source-image mismatch")
        if trait_locks[task_id]["image_sha256"] != stage_lock["image_sha256"]:
            raise ContractError(f"{task_id}: Stage-B/traits source-image mismatch")
        if not math.isclose(
            scale, float(root_locks[task_id]["um_per_px"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ContractError(f"{task_id}: root-provider/Stage-B scale mismatch")
        if not math.isclose(
            scale, float(trait_locks[task_id]["um_per_px"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ContractError(f"{task_id}: Stage-B/traits scale mismatch")

    overlay_section = payload.get("review_overlays", {})
    if overlay_section is None:
        overlay_section = {}
    if not isinstance(overlay_section, Mapping):
        raise ContractError("review_overlays must be an object")
    manifest_overlay = overlay_section.get("enabled", False)
    if not isinstance(manifest_overlay, bool):
        raise ContractError("review_overlays.enabled must be boolean")
    overlay_enabled = manifest_overlay if review_overlays is None else bool(review_overlays)

    manifest_digest = sha256_file(manifest_path)
    module_digest = sha256_file(Path(__file__).resolve())
    image_locks = [
        {
            "task_id": row["task_id"],
            "image_sha256": row["image_sha256"],
            "um_per_px": row["um_per_px"],
        }
        for row in rows
    ]
    source_image_lock_identity = sha256_json(image_locks)
    root_inputs: dict[str, Any] = {
        ref.name: ref.sha256 for ref in root_refs.values()
    }
    root_inputs.update(
        {
            "root_provider.bundle_registry": str(registry_digest).lower(),
            "root_provider.bundle_identity": str(bundle_identity).lower(),
            "source_image_lock_identity": source_image_lock_identity,
            "workflow_manifest": manifest_digest,
            "workflow_module": module_digest,
            **proposal_binding.receipt_fields(),
        }
    )
    if reference_ref is not None:
        root_inputs[reference_ref.name] = reference_ref.sha256
    root_plan = build_root_provider_plan(root_config)
    stages: list[dict[str, Any]] = []
    root_stage = _plan_stage(
        name="root_provider",
        input_hashes=root_inputs,
        output=root_config.output,
        estimated_gpu={
            "required": True,
            "v1_physical_gpus": list(root_config.v1_physical_gpus),
            "q8_physical_gpus": list(root_config.q8_physical_gpus),
            "selection": "root-provider per-program nvidia-smi preflight",
        },
        detail={"package_api": "phaxis.root_provider.runtime.run_pipeline", "plan": root_plan},
    )
    stages.append(root_stage)
    stageb_inputs = {
        ref.name: ref.sha256 for ref in stageb_refs
    }
    stageb_inputs.update(proposal_binding.receipt_fields())
    stageb_inputs["source_image_lock_identity"] = source_image_lock_identity
    stageb_inputs["root_provider.stage_plan_identity"] = root_stage[
        "stage_plan_identity_sha256"
    ]
    stageb_stage = _plan_stage(
        name="stageb_train399",
        input_hashes=stageb_inputs,
        output=output_path / "stageb",
        estimated_gpu={
            "required": True,
            "physical_gpu": physical_gpu,
            "cuda_visible_devices": str(physical_gpu),
            "internal_device": internal_device,
            "estimated_peak_vram_mib": stageb_config.estimated_peak_vram_mib,
            "required_free_vram_reserve_mib": stageb_config.required_free_vram_reserve_mib,
        },
        detail={
            "tasks": len(rows),
            "checkpoint_policy": selected_metadata["checkpoint_policy"],
            "ensemble_members": 5,
            "shared_input_acceleration_requested": shared,
            "shared_input_acceleration_default_enabled": False,
        },
    )
    stages.append(stageb_stage)
    fusion_stage = _plan_stage(
        name="fusion",
        input_hashes={
            "root_provider.stage_plan_identity": root_stage["stage_plan_identity_sha256"],
            "stageb.stage_plan_identity": stageb_stage["stage_plan_identity_sha256"],
            **proposal_binding.receipt_fields(),
        },
        output=output_path / "fusion",
        estimated_gpu={"required": False},
        detail={
            "package_api": "phaxis.fusion.fuse_hybrid_root_with_stageb_hairs",
            "root_expert": proposal_binding.root_expert_id,
        },
    )
    stages.append(fusion_stage)
    traits_stage = _plan_stage(
        name="traits",
        input_hashes={
            "fusion.stage_plan_identity": fusion_stage["stage_plan_identity_sha256"],
            traits_ref.name: traits_ref.sha256,
            **proposal_binding.receipt_fields(),
        },
        output=output_path / "traits",
        estimated_gpu={"required": False},
        detail={"package_api": "phaxis.traits.export_traits"},
    )
    stages.append(traits_stage)
    profile_stage = _plan_stage(
        name="distal_axis_profiles",
        input_hashes={
            "traits.stage_plan_identity": traits_stage["stage_plan_identity_sha256"],
            profile_ref.name: profile_ref.sha256,
            **proposal_binding.receipt_fields(),
        },
        output=output_path / "distal_axis_profiles",
        estimated_gpu={"required": False},
        detail={"package_api": "phaxis.axial_profiles.export_distal_axis_profiles"},
    )
    stages.append(profile_stage)
    if overlay_enabled:
        stages.append(
            _plan_stage(
                name="review_overlays",
                input_hashes={
                    "fusion.stage_plan_identity": fusion_stage["stage_plan_identity_sha256"],
                    "source_image_lock_identity": source_image_lock_identity,
                    **proposal_binding.receipt_fields(),
                },
                output=output_path / "review_overlays",
                estimated_gpu={"required": False},
                detail={
                    "optional_review_artifact": True,
                    "used_for_model_routing": False,
                },
            )
        )

    plan: dict[str, Any] = {
        "schema_version": WORKFLOW_PLAN_SCHEMA,
        "status": "planned_not_executed",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "manifest_identity_sha256": payload["manifest_identity_sha256"],
        **proposal_binding.output_identity_fields(),
        "output": str(output_path),
        "tasks": len(rows),
        "task_ids": [str(row["task_id"]) for row in rows],
        "task_identity_sha256": sha256_json(image_locks),
        "review_overlays_enabled": overlay_enabled,
        "stages": stages,
        "guards": dict(_GUARDS),
        "default_plan_only": True,
        "execute_requires_explicit_flag": True,
    }
    plan["plan_identity_sha256"] = sha256_json(plan)
    return _WorkflowContext(
        manifest_path=manifest_path,
        manifest=payload,
        output=output_path,
        plan=plan,
        root_config=root_config,
        stageb_config=stageb_config,
        stageb_rows=rows,
        model_contract_proposal=proposal_ref,
        model_contract_binding=proposal_binding,
        traits_metadata=traits_ref,
        profile_contract=profile_ref,
        review_overlays=overlay_enabled,
    )


def build_analysis_plan(
    manifest: str | Path,
    *,
    output: str | Path,
    review_overlays: bool | None = None,
) -> dict[str, Any]:
    """Build a deterministic, read-only plan with no GPU/model side effects."""

    return deepcopy(
        dict(
            _context(
                manifest, output=output, review_overlays=review_overlays
            ).plan
        )
    )


def _stageb_batch_identity(
    config: StageBBatchConfig, rows: Sequence[Mapping[str, Any]]
) -> str:
    return sha256_json(
        {
            "input_manifest_sha256": sha256_file(config.input_manifest),
            "checkpoints_sha256": [sha256_file(path) for path in config.checkpoints],
            "candidate_manifest_sha256": sha256_file(config.candidate_manifest),
            "selected_model_metadata_sha256": sha256_file(config.selected_model_metadata),
            "selection_receipt_sha256": sha256_file(config.selection_receipt),
            "physical_gpu": config.physical_gpu,
            "internal_device": config.internal_device,
            "shared_input_acceleration": config.shared_input_acceleration,
            "shared_input_max_host_bytes": config.shared_input_max_host_bytes,
            "shared_input_max_device_bytes": config.shared_input_max_device_bytes,
            "shared_input_device_reserve_bytes": config.shared_input_device_reserve_bytes,
            "estimated_peak_vram_mib": config.estimated_peak_vram_mib,
            "required_free_vram_reserve_mib": config.required_free_vram_reserve_mib,
            "image_locks": [
                {
                    "task_id": row["task_id"],
                    "image_sha256": row["image_sha256"],
                    "um_per_px": row["um_per_px"],
                }
                for row in rows
            ],
        }
    )


def _parse_gpu_query(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            raise ContractError(f"unexpected nvidia-smi GPU row: {line!r}")
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mib": int(parts[3]),
                    "memory_used_mib": int(parts[4]),
                    "utilization_percent": int(parts[5]),
                    "temperature_c": int(parts[6]),
                }
            )
        except ValueError as error:
            raise ContractError(f"non-numeric nvidia-smi GPU row: {line!r}") from error
    if not rows:
        raise ContractError("nvidia-smi returned an empty GPU inventory")
    return rows


def _restore_cuda_visible_devices(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _gpu_preflight(config: StageBBatchConfig) -> dict[str, Any]:
    """Capture physical mapping/capacity before the first torch import."""

    early_modules = sorted(
        name
        for name in sys.modules
        if name == "torch"
        or name.startswith("torch.")
        or name == "torchvision"
        or name.startswith("torchvision.")
        or name in {"phaxis.hair_stageb.runtime", "phaxis.hair_stageb.model"}
    )
    if early_modules:
        raise ContractError(
            "torch/CUDA runtime was imported before mandatory nvidia-smi preflight: "
            + ", ".join(early_modules[:8])
        )
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.physical_gpu)
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    snapshots: list[dict[str, Any]] = []
    raw_samples: list[str] = []
    try:
        for sample_index in range(5):
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            raw_samples.append(completed.stdout)
            inventory = _parse_gpu_query(completed.stdout)
            matches = [row for row in inventory if row["index"] == config.physical_gpu]
            if len(matches) != 1:
                raise ContractError(
                    f"physical GPU {config.physical_gpu} is absent or duplicated in nvidia-smi"
                )
            snapshots.append(matches[0])
            if sample_index < 4:
                time.sleep(1.0)
        if len({row["uuid"] for row in snapshots}) != 1:
            raise ContractError("physical GPU UUID changed during preflight sampling")
        total = int(snapshots[0]["memory_total_mib"])
        peak_used = max(int(row["memory_used_mib"]) for row in snapshots)
        sustained = float(
            statistics.median(int(row["utilization_percent"]) for row in snapshots)
        )
        required = (
            peak_used
            + config.estimated_peak_vram_mib
            + config.required_free_vram_reserve_mib
        )
        if required > total:
            raise ContractError(
                "Stage-B GPU preflight failed closed: existing + estimated peak + "
                f"reserve is {required} MiB, total is {total} MiB"
            )
        if sustained >= 80.0:
            raise ContractError(
                f"Stage-B GPU preflight failed closed: sustained utilization is {sustained:.1f}%"
            )
        process_command = [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
        processes = subprocess.run(
            process_command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except BaseException:
        _restore_cuda_visible_devices(previous)
        raise
    return {
        "schema_version": GPU_PREFLIGHT_SCHEMA,
        "status": "pass_capacity_and_utilization_safe",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": str(config.physical_gpu),
        "internal_device": config.internal_device,
        "mapping": {
            "logical_cuda_index": 0,
            "physical_index": config.physical_gpu,
            "uuid": snapshots[0]["uuid"],
            "name": snapshots[0]["name"],
        },
        "utilization_samples_percent": [
            int(row["utilization_percent"]) for row in snapshots
        ],
        "sustained_utilization_percent": sustained,
        "memory_total_mib": total,
        "maximum_existing_memory_used_mib": peak_used,
        "estimated_stageb_peak_vram_mib": config.estimated_peak_vram_mib,
        "required_free_vram_reserve_mib": config.required_free_vram_reserve_mib,
        "capacity_after_estimate_and_reserve_mib": total - required,
        "gpu_query_command": command,
        "gpu_query_stdout_samples": raw_samples,
        "compute_process_query_command": process_command,
        "compute_process_rows": processes.stdout.splitlines(),
        "no_process_killed_or_suspended": True,
        "torch_imported_before_preflight": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }


def _validate_stageb_summary(
    *,
    output: Path,
    config: StageBBatchConfig,
    rows: Sequence[Mapping[str, Any]],
    expected_model_metadata: Mapping[str, Any],
    model_contract_binding: ModelContractProposalBinding,
) -> dict[str, Any]:
    summary_path = output / "summary.json"
    summary = read_json(summary_path)
    if summary.get("schema_version") != STAGEB_BATCH_SCHEMA:
        raise ContractError("unsupported Stage-B batch summary schema")
    if summary.get("status") != "completed":
        raise ContractError("Stage-B batch summary is not completed")
    _sealed_identity(summary, "summary_identity_sha256")
    require_output_identity(summary, model_contract_binding, role="Stage-B summary")
    expected_batch = _stageb_batch_identity(config, rows)
    if summary.get("batch_identity_sha256") != expected_batch:
        raise ContractError("Stage-B resume batch identity mismatch")
    if summary.get("condition_metadata_used_for_routing") is not False:
        raise ContractError("Stage-B summary claims condition-based routing")
    if summary.get("canonical_annotations_read") is not False:
        raise ContractError("Stage-B summary claims canonical annotation access")
    if summary.get("blind_images_used") != 0 or summary.get("root_cap_region_output") is not False:
        raise ContractError("Stage-B summary violates blind/root-cap guards")
    shared = summary.get("shared_input_acceleration", {})
    if shared.get("requested") is not config.shared_input_acceleration:
        raise ContractError("Stage-B shared-input receipt differs from the lock")
    if shared.get("default_enabled") is not False:
        raise ContractError("Stage-B shared-input default must remain disabled")
    records = summary.get("records")
    if not isinstance(records, list) or {record.get("task_id") for record in records} != {
        row["task_id"] for row in rows
    }:
        raise ContractError("Stage-B summary task set mismatch")
    by_task = {str(row["task_id"]): row for row in rows}
    for record in records:
        task_id = str(record["task_id"])
        path = output / "detections" / f"{task_id}.json"
        if sha256_file(path) != record.get("detection_file_sha256"):
            raise ContractError(f"{task_id}: Stage-B detection file hash mismatch")
        payload = read_json(path)
        validate_stageb_detection_payload(
            payload,
            expected_task_id=task_id,
            expected_image_sha256=str(by_task[task_id]["image_sha256"]),
            expected_model_metadata=expected_model_metadata,
        )
        require_output_identity(
            payload,
            model_contract_binding,
            role=f"Stage-B detection {task_id}",
        )
    return summary


def run_stageb_batch(
    config: StageBBatchConfig,
    *,
    output: str | Path,
    resume: bool = False,
    model_contract_binding: ModelContractProposalBinding,
) -> dict[str, Any]:
    """Run strict five-checkpoint train399 inference on a locked CSV manifest.

    All file/Gate/image checks and the physical ``nvidia-smi`` mapping happen
    before importing torch or any module that imports torch.
    """

    output = Path(output).resolve()
    if len(config.checkpoints) != 5:
        raise ContractError("exactly five Stage-B checkpoints are required")
    if config.internal_device != "cuda:0" or config.physical_gpu < 0:
        raise ContractError("invalid Stage-B physical/logical CUDA mapping")
    if config.file_sha256 is not None:
        paths = {
            "stageb.input_manifest": config.input_manifest,
            **{
                f"stageb.checkpoints[{index}]": path
                for index, path in enumerate(config.checkpoints)
            },
            "stageb.candidate_manifest": config.candidate_manifest,
            "stageb.selected_model_metadata": config.selected_model_metadata,
            "stageb.selection_receipt": config.selection_receipt,
        }
        for name, path in paths.items():
            expected = config.file_sha256.get(name)
            if not _is_sha256(expected) or sha256_file(path) != str(expected).lower():
                raise ContractError(f"Stage-B locked file drift: {name}")
    _candidate, selected_metadata = _validate_train399_gate(
        candidate_manifest=config.candidate_manifest,
        selected_model_metadata=config.selected_model_metadata,
        selection_receipt=config.selection_receipt,
        checkpoints=config.checkpoints,
    )
    rows = _stageb_rows(config, verify_images=True)
    batch_identity = _stageb_batch_identity(config, rows)
    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite Stage-B output: {output}")
        if (output / "summary.json").is_file():
            return _validate_stageb_summary(
                output=output,
                config=config,
                rows=rows,
                expected_model_metadata=selected_metadata,
                model_contract_binding=model_contract_binding,
            )
    else:
        output.mkdir(parents=True)

    preflight = _gpu_preflight(config)
    preflight_path = output / "nvidia_smi_preflight.json"
    if preflight_path.exists():
        if not resume or read_json(preflight_path).get("mapping") != preflight["mapping"]:
            raise FileExistsError(f"refusing to overwrite GPU preflight: {preflight_path}")
    else:
        atomic_write_json(preflight_path, preflight)

    # Mandatory boundary: no torch or StageBEnsemble import may move above the
    # preflight receipt write.
    import numpy as np
    import tifffile
    import torch

    from .hair_stageb.runtime import StageBEnsemble
    from .hair_stageb.serialization import make_detection_payload

    if not torch.cuda.is_available():
        raise ContractError("CUDA is unavailable after a passing nvidia-smi preflight")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    load_started = time.perf_counter()
    ensemble = StageBEnsemble(
        config.checkpoints,
        device=config.internal_device,
        use_amp=False,
        candidate_manifest=config.candidate_manifest,
        selected_model_metadata=config.selected_model_metadata,
        selection_receipt=config.selection_receipt,
        shared_input_acceleration=config.shared_input_acceleration,
        shared_input_max_host_bytes=config.shared_input_max_host_bytes,
        shared_input_max_device_bytes=config.shared_input_max_device_bytes,
        shared_input_device_reserve_bytes=config.shared_input_device_reserve_bytes,
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()
    records: list[dict[str, Any]] = []
    resumed_images = 0
    batch_started = time.perf_counter()
    for row in rows:
        task_id = str(row["task_id"])
        detection_path = output / "detections" / f"{task_id}.json"
        if detection_path.is_file():
            if not resume:
                raise FileExistsError(f"refusing to overwrite Stage-B detection: {detection_path}")
            existing = read_json(detection_path)
            validate_stageb_detection_payload(
                existing,
                expected_task_id=task_id,
                expected_image_sha256=str(row["image_sha256"]),
                expected_model_metadata=ensemble.detection_model_metadata,
            )
            require_output_identity(
                existing,
                model_contract_binding,
                role=f"Stage-B detection {task_id}",
            )
            records.append(
                {
                    "task_id": task_id,
                    "source_megapixels": 0.0,
                    "detections": int(existing["n"]),
                    "wall_seconds_including_io": 0.0,
                    "input_io_seconds": 0.0,
                    "preprocess_seconds": 0.0,
                    "inference_seconds": 0.0,
                    "postprocess_and_output_io_seconds": 0.0,
                    "timing_trace_kind": "cached_not_measured",
                    "detection_identity_sha256": existing["detection_identity_sha256"],
                    "detection_file_sha256": sha256_file(detection_path),
                    "resumed": True,
                    "shared_input_runtime_audit": None,
                }
            )
            resumed_images += 1
            continue
        image_started = time.perf_counter()
        io_started = time.perf_counter()
        image = tifffile.imread(row["image_path"])
        input_io_seconds = time.perf_counter() - io_started
        inference_started = time.perf_counter()
        prediction = ensemble.predict(image, source_um_per_px=float(row["um_per_px"]))
        observed_audit = getattr(ensemble, "last_shared_input_audit", None)
        shared_audit = dict(observed_audit) if observed_audit is not None else None
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        postprocess_started = time.perf_counter()
        payload = make_detection_payload(
            task_id=task_id,
            source_image_sha256=str(row["image_sha256"]),
            source_um_per_px=float(row["um_per_px"]),
            prediction=prediction,
            precision_mode="fp32_locked",
            model_metadata=ensemble.detection_model_metadata,
            score_threshold=ensemble.score_threshold,
        )
        payload.pop("detection_identity_sha256", None)
        payload.update(model_contract_binding.output_identity_fields())
        payload["detection_identity_sha256"] = sha256_json(payload)
        validate_stageb_detection_payload(
            payload,
            expected_task_id=task_id,
            expected_image_sha256=str(row["image_sha256"]),
            expected_model_metadata=ensemble.detection_model_metadata,
        )
        require_output_identity(
            payload,
            model_contract_binding,
            role=f"Stage-B detection {task_id}",
        )
        atomic_write_json(detection_path, payload)
        postprocess_seconds = time.perf_counter() - postprocess_started
        elapsed = time.perf_counter() - image_started
        records.append(
            {
                "task_id": task_id,
                "source_megapixels": float(np.prod(image.shape[:2]) / 1e6),
                "detections": int(payload["n"]),
                "wall_seconds_including_io": elapsed,
                "input_io_seconds": input_io_seconds,
                "preprocess_seconds": 0.0,
                "inference_seconds": inference_seconds,
                "postprocess_and_output_io_seconds": postprocess_seconds,
                "timing_trace_kind": "direct_per_source_nonoverlapping",
                "detection_identity_sha256": payload["detection_identity_sha256"],
                "detection_file_sha256": sha256_file(detection_path),
                "resumed": False,
                "shared_input_runtime_audit": shared_audit,
            }
        )
    torch.cuda.synchronize()
    batch_seconds = time.perf_counter() - batch_started
    executed = [
        record["shared_input_runtime_audit"]
        for record in records
        if record["shared_input_runtime_audit"] is not None
    ]
    path_counts = Counter(str(item.get("runtime_path")) for item in executed)
    fallback_counts = Counter(str(item.get("fallback_reason")) for item in executed)
    timings = [
        float(record["wall_seconds_including_io"])
        for record in records
        if not record["resumed"]
    ]
    summary: dict[str, Any] = {
        "schema_version": STAGEB_BATCH_SCHEMA,
        "status": "completed",
        "batch_identity_sha256": batch_identity,
        "images": len(records),
        "detections": sum(int(record["detections"]) for record in records),
        "model_load_seconds": model_load_seconds,
        "batch_wall_seconds_including_io": batch_seconds,
        "median_seconds_per_image": float(np.median(timings)) if timings else 0.0,
        "p95_seconds_per_image": float(np.quantile(timings, 0.95)) if timings else 0.0,
        "peak_allocated_vram_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_vram_mib": float(torch.cuda.max_memory_reserved() / 2**20),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_mapping": preflight["mapping"],
        "internal_device": config.internal_device,
        "precision_mode": "fp32_locked",
        "shared_input_acceleration": {
            "requested": config.shared_input_acceleration,
            "default_enabled": False,
            "effective_max_host_bytes": config.shared_input_max_host_bytes,
            "effective_max_device_bytes": config.shared_input_max_device_bytes,
            "effective_device_reserve_bytes": config.shared_input_device_reserve_bytes,
            "executed_images": len(executed),
            "resumed_images_not_executed": resumed_images,
            "runtime_path_counts": dict(sorted(path_counts.items())),
            "fallback_reason_counts": dict(sorted(fallback_counts.items())),
        },
        "resumed_images": resumed_images,
        "checkpoint_sha256": list(ensemble.checkpoint_sha256),
        "detection_model_metadata": ensemble.detection_model_metadata,
        "score_threshold": ensemble.score_threshold,
        "per_source_timing_semantics": (
            "direct_nonoverlapping_wall; model-owned preprocessing is included "
            "inside inference_seconds"
        ),
        "nvidia_smi_preflight_sha256": sha256_file(preflight_path),
        "records": records,
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read": False,
        "root_cap_region_output": False,
        "blind_images_used": 0,
        **model_contract_binding.output_identity_fields(),
    }
    summary["summary_identity_sha256"] = sha256_json(summary)
    atomic_write_json(output / "summary.json", summary)
    return _validate_stageb_summary(
        output=output,
        config=config,
        rows=rows,
        expected_model_metadata=selected_metadata,
        model_contract_binding=model_contract_binding,
    )


def _copy_locked_root_artifacts(
    prediction: Mapping[str, Any], *, source_root: Path, output_root: Path
) -> None:
    for relpath_field, digest_field in (
        ("root_mask_relpath", "root_mask_sha256"),
        ("root_axis_geometry_relpath", "root_axis_geometry_sha256"),
        ("root_continuity_added_mask_relpath", "root_continuity_added_mask_sha256"),
        ("root_width_reference_mask_relpath", "root_width_reference_mask_sha256"),
        (
            "root_width_reference_axis_geometry_relpath",
            "root_width_reference_axis_geometry_sha256",
        ),
    ):
        relpath = prediction.get(relpath_field)
        expected = prediction.get(digest_field)
        if not relpath or not expected:
            continue
        source = source_root / str(relpath)
        destination = output_root / str(relpath)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite fused root artifact: {destination}")
        if sha256_file(source) != str(expected):
            raise ContractError(f"source root artifact hash mismatch: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != str(expected):
            raise ContractError(f"copied root artifact hash mismatch: {destination}")


def _run_fusion_batch(
    *,
    hybrid_root: Path,
    stageb_root: Path,
    output: Path,
    model_contract_binding: ModelContractProposalBinding,
) -> dict[str, Any]:
    from .fusion import fuse_hybrid_root_with_stageb_hairs

    prediction_paths = sorted((hybrid_root / "predictions").glob("*.json"))
    if not prediction_paths:
        raise ContractError(f"no root-provider predictions in {hybrid_root / 'predictions'}")
    public_identity = model_contract_binding.public_identity_fields()
    records: list[dict[str, Any]] = []
    for prediction_path in prediction_paths:
        task_started = time.perf_counter()
        hybrid = read_json(prediction_path)
        task_id = str(hybrid["task_id"])
        detection_path = stageb_root / "detections" / f"{task_id}.json"
        if not detection_path.is_file():
            raise ContractError(f"missing Stage-B detection: {detection_path}")
        fused = fuse_hybrid_root_with_stageb_hairs(
            hybrid,
            read_json(detection_path),
            hybrid_artifact_root=hybrid_root,
            model_contract_proposal=model_contract_binding.receipt_fields(),
            model_contract_public_identity=public_identity,
        )
        phaxis = fused.get("phaxis")
        if not isinstance(phaxis, Mapping):
            raise ContractError(f"{task_id}: fused prediction has no PHAxis provenance")
        fused["phaxis"] = {
            **dict(phaxis),
            **model_contract_binding.receipt_fields(),
            "model_bundle_id": public_identity["model_bundle_id"],
            "root_expert": public_identity["root_expert_id"],
        }
        if (
            fused.get("condition_metadata_used_for_routing") is not False
            or fused.get("canonical_annotations_read_during_inference") is not False
            or fused.get("blind_images_used") != 0
            or fused.get("root_cap_region_output") is not False
        ):
            raise ContractError(f"{task_id}: fused prediction violates workflow guards")
        _copy_locked_root_artifacts(fused, source_root=hybrid_root, output_root=output)
        destination = output / "predictions" / f"{task_id}.json"
        atomic_write_json(destination, fused)
        elapsed = time.perf_counter() - task_started
        records.append(
            {
                "task_id": task_id,
                "hair_identity_count_expert": fused["phaxis"]["hair_identity_count_expert"],
                "root_lock_sha256": fused["phaxis"]["root_lock_sha256"],
                "prediction_sha256": sha256_file(destination),
                "wall_seconds_including_io": elapsed,
                "timing_trace_kind": "direct_per_source_nonoverlapping",
            }
        )
    experts = sorted({str(record["hair_identity_count_expert"]) for record in records})
    if len(experts) != 1:
        raise ContractError(f"mixed Stage-B experts in fusion batch: {experts}")
    summary: dict[str, Any] = {
        "schema_version": "PHAxis-fusion-run-1.1",
        "status": "completed",
        "software": {"name": PRODUCT_NAME, "version": PRODUCT_VERSION},
        "model_bundle_id": public_identity["model_bundle_id"],
        "root_expert": public_identity["root_expert_id"],
        "hair_identity_count_expert": experts[0],
        "images": len(records),
        "source_root_provider_summary_sha256": (
            sha256_file(hybrid_root / "summary.json")
            if (hybrid_root / "summary.json").is_file()
            else None
        ),
        "source_stageb_summary_sha256": sha256_file(stageb_root / "summary.json"),
        "records": records,
        "per_source_timing_semantics": "direct fusion input_io_compute_output_io wall",
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read": False,
        "root_cap_region_output": False,
        "blind_images_used": 0,
        **model_contract_binding.receipt_fields(),
    }
    summary["summary_identity_sha256"] = sha256_json(summary)
    atomic_write_json(output / "fusion_summary.json", summary)
    return summary


def _validate_fusion_output(
    output: Path,
    expected_tasks: set[str],
    model_contract_binding: ModelContractProposalBinding | None = None,
) -> dict[str, Any]:
    summary = read_json(output / "fusion_summary.json")
    if summary.get("status") != "completed":
        raise ContractError("fusion summary is not completed")
    _sealed_identity(summary, "summary_identity_sha256")
    if model_contract_binding is not None:
        require_output_identity(
            summary,
            model_contract_binding,
            role="fusion summary",
            root_field="root_expert",
        )
    if (
        summary.get("condition_metadata_used_for_routing") is not False
        or summary.get("canonical_annotations_read") is not False
        or summary.get("blind_images_used") != 0
        or summary.get("root_cap_region_output") is not False
    ):
        raise ContractError("fusion summary violates workflow guards")
    records = summary.get("records")
    if not isinstance(records, list) or {record.get("task_id") for record in records} != expected_tasks:
        raise ContractError("fusion output task set mismatch")
    for record in records:
        task_id = str(record["task_id"])
        path = output / "predictions" / f"{task_id}.json"
        if sha256_file(path) != record.get("prediction_sha256"):
            raise ContractError(f"{task_id}: fused prediction hash mismatch")
        prediction = read_json(path)
        validate_hybrid_prediction(prediction, artifact_root=output)
        if model_contract_binding is not None:
            phaxis = prediction.get("phaxis")
            if not isinstance(phaxis, Mapping):
                raise ContractError(f"{task_id}: missing PHAxis provenance")
            require_output_identity(
                phaxis,
                model_contract_binding,
                role=f"{task_id} fused prediction",
                root_field="root_expert",
            )
    return summary


def _publish_directory(
    destination: Path,
    stage_name: str,
    producer: Callable[[Path], Any],
) -> Any:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite stage output: {destination}")
    index = 1
    while True:
        attempt = destination.parent / f".{destination.name}.{stage_name}.attempt-{index:03d}"
        if not attempt.exists():
            break
        index += 1
    attempt.mkdir(parents=True)
    try:
        result = producer(attempt)
        os.replace(attempt, destination)
        return result
    except BaseException as error:
        atomic_write_json(
            attempt / "PHAXIS_STAGE_FAILURE.json",
            {
                "schema_version": "PHAxis-analysis-stage-failure-1.0",
                "stage": stage_name,
                "error_type": type(error).__name__,
                "error": str(error),
                "official_output_published": False,
                "blind_images_used": 0,
            },
        )
        raise


def _validate_traits_output(
    output: Path,
    expected_tasks: int,
    model_contract_binding: ModelContractProposalBinding | None = None,
) -> dict[str, Any]:
    summary = read_json(output / "summary.json")
    if summary.get("status") != "completed":
        raise ContractError("trait summary is not completed")
    _sealed_identity(summary, "export_identity_sha256")
    if model_contract_binding is not None:
        require_output_identity(
            summary, model_contract_binding, role="traits summary"
        )
    if summary.get("tasks") != expected_tasks:
        raise ContractError("trait output task count mismatch")
    for field, filename in (
        ("traits_sha256", "traits.csv"),
        ("image_traits_sha256", "image_traits.csv"),
        ("detailed_root_statistics_sha256", "detailed_root_statistics.csv"),
        ("hair_instances_sha256", "hair_instances.csv"),
    ):
        if sha256_file(output / filename) != summary.get(field):
            raise ContractError(f"trait output hash mismatch: {filename}")
    if (
        summary.get("condition_metadata_used_for_model_routing") is not False
        or summary.get("canonical_annotations_read") is not False
        or summary.get("blind_images_used") != 0
        or summary.get("root_cap_region_statistics_included") is not False
    ):
        raise ContractError("trait summary violates workflow guards")
    return summary


def _validate_profile_output(
    output: Path,
    expected_tasks: int,
    model_contract_binding: ModelContractProposalBinding | None = None,
) -> dict[str, Any]:
    summary = read_json(output / "summary.json")
    if summary.get("status") != "completed":
        raise ContractError("distal-axis profile summary is not completed")
    _sealed_identity(summary, "export_identity_sha256")
    if model_contract_binding is not None:
        require_output_identity(
            summary, model_contract_binding, role="distal-axis profiles summary"
        )
    if summary.get("tasks") != expected_tasks:
        raise ContractError("distal-axis profile task count mismatch")
    if sha256_file(output / "distal_axis_profiles.csv") != summary.get("profiles_csv_sha256"):
        raise ContractError("distal-axis profile table hash mismatch")
    if (
        summary.get("condition_metadata_used_for_model_routing") is not False
        or summary.get("canonical_annotations_read") is not False
        or summary.get("blind_images_used") != 0
        or summary.get("root_cap_region_output") is not False
        or summary.get("stageb_two_point_vector_used_as_length") is not False
    ):
        raise ContractError("distal-axis profile summary violates workflow guards")
    return summary


def _render_overlays(context: _WorkflowContext, output: Path) -> dict[str, Any]:
    import numpy as np
    import tifffile
    from PIL import Image

    from .rendering import render_prediction_overlay

    rows = {str(row["task_id"]): row for row in context.stageb_rows}
    records: list[dict[str, Any]] = []
    for task_id in sorted(rows):
        prediction = read_json(context.output / "fusion" / "predictions" / f"{task_id}.json")
        image = tifffile.imread(rows[task_id]["image_path"])
        overlay = render_prediction_overlay(
            image,
            prediction,
            artifact_root=context.output / "fusion",
        )
        destination = output / f"{task_id}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(overlay)[..., ::-1]).save(destination, format="PNG")
        records.append({"task_id": task_id, "overlay_sha256": sha256_file(destination)})
    summary: dict[str, Any] = {
        "schema_version": "PHAxis-review-overlay-run-1.0",
        "status": "completed_optional_review_artifacts",
        "images": len(records),
        "records": records,
        "used_for_model_routing": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }
    summary["summary_identity_sha256"] = sha256_json(summary)
    atomic_write_json(output / "summary.json", summary)
    return summary


def _validate_overlay_output(output: Path, expected_tasks: set[str]) -> dict[str, Any]:
    summary = read_json(output / "summary.json")
    _sealed_identity(summary, "summary_identity_sha256")
    records = summary.get("records")
    if not isinstance(records, list) or {record.get("task_id") for record in records} != expected_tasks:
        raise ContractError("review-overlay task set mismatch")
    for record in records:
        path = output / f"{record['task_id']}.png"
        if sha256_file(path) != record.get("overlay_sha256"):
            raise ContractError(f"review-overlay hash mismatch: {path}")
    if summary.get("used_for_model_routing") is not False or summary.get("blind_images_used") != 0:
        raise ContractError("review-overlay summary violates workflow guards")
    return summary


def _write_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(dict(state))
    payload.pop("state_identity_sha256", None)
    payload["state_identity_sha256"] = sha256_json(payload)
    atomic_write_json(path, payload)
    return payload


def _read_state(path: Path) -> dict[str, Any]:
    state = read_json(path)
    if state.get("schema_version") != WORKFLOW_STATE_SCHEMA:
        raise ContractError("unsupported PHAxis workflow state schema")
    _sealed_identity(state, "state_identity_sha256")
    return state


def _record_evidence(context: _WorkflowContext, stage: str, root: Path) -> dict[str, Any]:
    files = _tree_files(root)
    receipt = {
        "schema_version": "PHAxis-analysis-stage-evidence-1.0",
        "stage": stage,
        "output": str(root),
        "files": files,
        "tree_identity_sha256": sha256_json(files),
        "blind_images_used": 0,
    }
    path = context.output / "evidence" / f"{stage}.json"
    atomic_write_json(path, receipt)
    return {
        "receipt": str(path.relative_to(context.output).as_posix()),
        "receipt_sha256": sha256_file(path),
        "tree_identity_sha256": receipt["tree_identity_sha256"],
        "files": len(files),
    }


def _verify_evidence(context: _WorkflowContext, stage: str, record: Mapping[str, Any]) -> None:
    receipt_path = context.output / str(record.get("receipt", ""))
    if sha256_file(receipt_path) != record.get("receipt_sha256"):
        raise ContractError(f"{stage}: evidence receipt hash mismatch")
    receipt = read_json(receipt_path)
    root = Path(str(receipt.get("output"))).resolve()
    expected_root = context.output / (
        "root_provider" if stage == "root_provider" else
        "stageb" if stage == "stageb_train399" else stage
    )
    if root != expected_root.resolve():
        raise ContractError(f"{stage}: evidence output path mismatch")
    observed = _tree_files(root)
    if observed != receipt.get("files") or sha256_json(observed) != record.get(
        "tree_identity_sha256"
    ):
        raise ContractError(f"{stage}: completed output tree drift")


def _root_guards(state: Mapping[str, Any]) -> None:
    if state.get("canonical_annotations_read") is not False or state.get("blind_images_used") != 0:
        raise ContractError("root-provider state violates canonical/blind guards")


def _execute_stage(
    context: _WorkflowContext, stage: str, *, resume: bool
) -> dict[str, Any]:
    task_ids = {str(row["task_id"]) for row in context.stageb_rows}
    destination = context.output / (
        "root_provider" if stage == "root_provider" else
        "stageb" if stage == "stageb_train399" else stage
    )
    if stage == "root_provider":
        root_resume = destination.exists()
        state = run_root_provider_pipeline(context.root_config, resume=root_resume)
        _root_guards(state)
        if (
            state.get("bundle_identity_sha256")
            != context.model_contract_binding.root_bundle_identity_sha256
        ):
            raise ContractError(
                "executed root-provider bundle differs from proposal root authority"
            )
        hybrid_predictions = destination / "hybrid" / "predictions"
        if {path.stem for path in hybrid_predictions.glob("*.json")} != task_ids:
            raise ContractError("root-provider output task set differs from locked manifest")
        return {
            "execution_status": (
                "resumed_or_cached_stage" if root_resume else "executed_fresh"
            ),
            "destination_existed_before": root_resume,
        }
    if stage == "stageb_train399":
        existed = destination.exists()
        summary = run_stageb_batch(
            context.stageb_config,
            output=destination,
            resume=destination.exists(),
            model_contract_binding=context.model_contract_binding,
        )
        resumed_images = int(summary.get("resumed_images", 0))
        return {
            "execution_status": (
                "executed_fresh"
                if not existed and resumed_images == 0
                else "resumed_or_cached_stage"
            ),
            "destination_existed_before": existed,
            "resumed_images": resumed_images,
        }
    if stage == "fusion":
        if destination.exists():
            _validate_fusion_output(
                destination, task_ids, context.model_contract_binding
            )
            return {
                "execution_status": "resumed_or_cached_stage",
                "destination_existed_before": True,
            }
        _publish_directory(
            destination,
            stage,
            lambda attempt: _run_fusion_batch(
                hybrid_root=context.output / "root_provider" / "hybrid",
                stageb_root=context.output / "stageb",
                output=attempt,
                model_contract_binding=context.model_contract_binding,
            ),
        )
        _validate_fusion_output(destination, task_ids, context.model_contract_binding)
        return {
            "execution_status": "executed_fresh",
            "destination_existed_before": False,
        }
    if stage == "traits":
        if destination.exists():
            _validate_traits_output(
                destination, len(task_ids), context.model_contract_binding
            )
            return {
                "execution_status": "resumed_or_cached_stage",
                "destination_existed_before": True,
            }
        from .traits import export_traits

        _publish_directory(
            destination,
            stage,
            lambda attempt: export_traits(
                prediction_root=context.output / "fusion" / "predictions",
                metadata_csv=context.traits_metadata.path,
                output=attempt,
                model_contract_proposal=context.model_contract_binding.receipt_fields(),
                model_contract_public_identity=(
                    context.model_contract_binding.public_identity_fields()
                ),
            ),
        )
        _validate_traits_output(
            destination, len(task_ids), context.model_contract_binding
        )
        return {
            "execution_status": "executed_fresh",
            "destination_existed_before": False,
        }
    if stage == "distal_axis_profiles":
        if destination.exists():
            _validate_profile_output(
                destination, len(task_ids), context.model_contract_binding
            )
            return {
                "execution_status": "resumed_or_cached_stage",
                "destination_existed_before": True,
            }
        from .axial_profiles import export_distal_axis_profiles

        def produce_profiles(attempt: Path) -> dict[str, Any]:
            return export_distal_axis_profiles(
                traits_csv=context.output / "traits" / "traits.csv",
                hair_instances_csv=context.output / "traits" / "hair_instances.csv",
                contract_json=context.profile_contract.path,
                output=attempt,
                model_contract_proposal=context.model_contract_binding.receipt_fields(),
                model_contract_public_identity=(
                    context.model_contract_binding.public_identity_fields()
                ),
            )

        _publish_directory(
            destination,
            stage,
            produce_profiles,
        )
        _validate_profile_output(
            destination, len(task_ids), context.model_contract_binding
        )
        return {
            "execution_status": "executed_fresh",
            "destination_existed_before": False,
        }
    if stage == "review_overlays":
        if destination.exists():
            _validate_overlay_output(destination, task_ids)
            return {
                "execution_status": "resumed_or_cached_stage",
                "destination_existed_before": True,
            }
        _publish_directory(
            destination,
            stage,
            lambda attempt: _render_overlays(context, attempt),
        )
        _validate_overlay_output(destination, task_ids)
        return {
            "execution_status": "executed_fresh",
            "destination_existed_before": False,
        }
    raise AssertionError(f"unknown workflow stage: {stage}")


def run_analysis(
    manifest: str | Path,
    *,
    output: str | Path,
    resume: bool = False,
    review_overlays: bool | None = None,
) -> dict[str, Any]:
    """Execute the locked plan; callers must opt in explicitly to this API."""

    invocation_started = time.perf_counter()
    invocation_started_utc = datetime.now(timezone.utc).isoformat()
    context = _context(manifest, output=output, review_overlays=review_overlays)
    context_validation_seconds = time.perf_counter() - invocation_started
    state_path = context.output / "workflow_state.json"
    workflow_identity = sha256_json(
        {
            "manifest_sha256": context.plan["manifest_sha256"],
            "manifest_identity_sha256": context.plan["manifest_identity_sha256"],
            "plan_identity_sha256": context.plan["plan_identity_sha256"],
            "output": str(context.output),
        }
    )
    if context.output.exists():
        if not resume or not state_path.is_file():
            raise FileExistsError(
                f"workflow output exists; pass --resume with valid state: {context.output}"
            )
        state = _read_state(state_path)
        require_output_identity(
            state,
            context.model_contract_binding,
            role="workflow resume state",
        )
        if state.get("workflow_identity_sha256") != workflow_identity:
            raise ContractError("workflow resume identity mismatch")
        plan_path = context.output / "analysis_plan.json"
        if sha256_file(plan_path) != state.get("analysis_plan_sha256"):
            raise ContractError("workflow resume plan hash mismatch")
    else:
        if resume:
            raise FileNotFoundError(f"cannot resume absent workflow output: {context.output}")
        context.output.mkdir(parents=True)
        plan_path = context.output / "analysis_plan.json"
        atomic_write_json(plan_path, context.plan)
        state = _write_state(
            state_path,
            {
                "schema_version": WORKFLOW_STATE_SCHEMA,
                "status": "running",
                "workflow_identity_sha256": workflow_identity,
                "manifest_sha256": context.plan["manifest_sha256"],
                "manifest_identity_sha256": context.plan["manifest_identity_sha256"],
                "plan_identity_sha256": context.plan["plan_identity_sha256"],
                "analysis_plan_sha256": sha256_file(plan_path),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "completed_stages": [],
                "stage_evidence": {},
                "execution_attempts": [],
                "review_overlays_enabled": context.review_overlays,
                **context.model_contract_binding.output_identity_fields(),
                **_GUARDS,
            },
        )

    stage_names = [str(stage["name"]) for stage in context.plan["stages"]]
    attempts = state.setdefault("execution_attempts", [])
    if not isinstance(attempts, list):
        raise ContractError("workflow state execution_attempts is invalid")
    attempt_id = len(attempts) + 1
    attempts.append(
        {
            "attempt_id": attempt_id,
            "started_utc": invocation_started_utc,
            "resume_requested": resume,
            "context_and_locked_input_validation_seconds": context_validation_seconds,
            "measurement_scope": "workflow_invocation_including_input_hash_io",
            "stages": [],
            "status": "running",
        }
    )
    state = _write_state(state_path, state)
    for stage in stage_names:
        stage_started = time.perf_counter()
        stage_started_utc = datetime.now(timezone.utc).isoformat()
        if stage in state.get("completed_stages", []):
            evidence = state.get("stage_evidence", {}).get(stage)
            if not isinstance(evidence, Mapping):
                raise ContractError(f"{stage}: completed state has no evidence")
            _verify_evidence(context, stage, evidence)
            stage_result: Mapping[str, Any] = {
                "execution_status": "cached_completed_evidence_validated",
                "destination_existed_before": True,
            }
            stage_evidence = evidence
        else:
            observed_result = _execute_stage(context, stage, resume=resume)
            stage_result = (
                observed_result
                if isinstance(observed_result, Mapping)
                else {
                    "execution_status": (
                        "executed_fresh" if not resume else "resumed_or_cached_stage"
                    ),
                    "destination_existed_before": resume,
                }
            )
            stage_evidence = _record_evidence(
                context,
                stage,
                context.output / (
                    "root_provider" if stage == "root_provider" else
                    "stageb" if stage == "stageb_train399" else stage
                ),
            )
            state.setdefault("completed_stages", []).append(stage)
            state.setdefault("stage_evidence", {})[stage] = stage_evidence
            state["last_completed_stage"] = stage
            state["last_completed_utc"] = datetime.now(timezone.utc).isoformat()

        stage_record = {
            "stage": stage,
            "started_utc": stage_started_utc,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds_including_stage_io_and_evidence_hashing": (
                time.perf_counter() - stage_started
            ),
            **dict(stage_result),
            "evidence_receipt_sha256": stage_evidence.get("receipt_sha256"),
            "evidence_tree_identity_sha256": stage_evidence.get(
                "tree_identity_sha256"
            ),
        }
        state["execution_attempts"][attempt_id - 1]["stages"].append(stage_record)
        state = _write_state(state_path, state)

    attempt = state["execution_attempts"][attempt_id - 1]
    core_stages = {
        "root_provider",
        "stageb_train399",
        "fusion",
        "traits",
        "distal_axis_profiles",
    }
    observed_core = {
        str(record.get("stage")): str(record.get("execution_status"))
        for record in attempt["stages"]
        if record.get("stage") in core_stages
    }
    fresh_direct = (
        not resume
        and not context.review_overlays
        and set(observed_core) == core_stages
        and all(value == "executed_fresh" for value in observed_core.values())
    )
    attempt.update(
        {
            "status": "completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds_including_locked_input_and_interstage_state_io": (
                time.perf_counter() - invocation_started
            ),
            "final_state_atomic_publish_included": False,
            "resume_or_cache_used": any(
                value != "executed_fresh" for value in observed_core.values()
            ),
            "fresh_direct_benchmark_eligible": fresh_direct,
            "review_overlays_excluded_from_benchmark_scope": not context.review_overlays,
        }
    )

    state.update(
        {
            "status": "completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "official_fused_predictions": str(context.output / "fusion" / "predictions"),
            "official_traits": str(context.output / "traits" / "image_traits.csv"),
            "official_distal_axis_profiles": str(
                context.output / "distal_axis_profiles" / "distal_axis_profiles.csv"
            ),
            "latest_execution_attempt_id": attempt_id,
            "latest_execution_fresh_direct_benchmark_eligible": fresh_direct,
            **context.model_contract_binding.output_identity_fields(),
            **_GUARDS,
        }
    )
    return _write_state(state_path, state)


__all__ = [
    "LockedFile",
    "StageBBatchConfig",
    "WORKFLOW_MANIFEST_SCHEMA",
    "build_analysis_plan",
    "load_analysis_manifest",
    "run_analysis",
    "run_stageb_batch",
]
