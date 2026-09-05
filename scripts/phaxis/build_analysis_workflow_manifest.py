#!/usr/bin/env python3
"""Build one sealed, executable PHAxis analysis-workflow manifest.

This producer is deliberately CPU-only.  It converts explicit, hash-verified
authorities into the package-owned ``PHAxis-analysis-workflow-manifest-1.0``
contract used by both normal analysis and the direct benchmark producers.  It
never discovers a ``latest`` result and refuses to overwrite an existing
manifest.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402
from phaxis.workflow import (  # noqa: E402
    WORKFLOW_MANIFEST_SCHEMA,
    build_analysis_plan,
)


STATUS = "ready_hash_locked_full_workflow"


class WorkflowManifestError(RuntimeError):
    """An input authority cannot form an executable workflow manifest."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkflowManifestError(message)


def _file_ref(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    _require(resolved.is_file(), f"locked workflow input is absent: {resolved}")
    _require(not resolved.is_symlink(), f"locked workflow input is a symlink: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _bundle_ref(bundle: Path) -> dict[str, str]:
    resolved = bundle.resolve()
    registry = resolved / "root_provider_bundle.json"
    _require(resolved.is_dir() and registry.is_file(), "root-provider bundle registry is absent")
    payload = read_json(registry)
    identity = payload.get("bundle_identity_sha256")
    _require(
        isinstance(identity, str) and len(identity) == 64,
        "root-provider bundle identity is invalid",
    )
    return {
        "path": str(resolved),
        "registry_sha256": sha256_file(registry),
        "bundle_identity_sha256": identity,
    }


def build_manifest(
    *,
    project: Path,
    bundle: Path,
    root_input_manifest: Path,
    acquisition_gate: Path,
    deployment_metadata: Path,
    canonical_manifest: Path,
    deployment_manifest: Path,
    deployment_lock: Path,
    image_root: Path,
    model_contract_proposal: Path,
    stageb_input_manifest: Path,
    checkpoints: Sequence[Path],
    candidate_manifest: Path,
    selected_model_metadata: Path,
    selection_receipt: Path,
    traits_metadata: Path,
    profile_contract: Path,
    output: Path,
    python_executable: Path,
    v1_physical_gpus: Sequence[int],
    q8_physical_gpus: Sequence[int],
    stageb_physical_gpu: int,
    stageb_internal_device: str,
    reference_registry: Path | None = None,
    shared_input_acceleration: bool = True,
    strict_physical_gpu: bool = True,
) -> dict[str, Any]:
    destination = output.resolve()
    _require(not destination.exists(), f"refusing to overwrite workflow manifest: {destination}")
    _require(project.resolve().is_dir(), f"workflow project is absent: {project.resolve()}")
    _require(image_root.resolve().is_dir(), f"workflow image root is absent: {image_root.resolve()}")
    _require(python_executable.resolve().is_file(), f"workflow Python is absent: {python_executable.resolve()}")
    _require(len(checkpoints) == 5, "workflow requires exactly five Stage-B checkpoints")
    checkpoint_refs = [_file_ref(Path(path)) for path in checkpoints]
    _require(
        len({item["sha256"] for item in checkpoint_refs}) == 5,
        "workflow Stage-B checkpoints are not five distinct files",
    )
    _require(
        bool(v1_physical_gpus)
        and bool(q8_physical_gpus)
        and all(isinstance(value, int) and value >= 0 for value in (*v1_physical_gpus, *q8_physical_gpus)),
        "root-provider physical GPU mapping is invalid",
    )
    _require(
        isinstance(stageb_physical_gpu, int) and stageb_physical_gpu >= 0,
        "Stage-B physical GPU is invalid",
    )
    _require(
        stageb_internal_device.startswith("cuda:")
        and stageb_internal_device[5:].isdigit(),
        "Stage-B internal device must be an explicit CUDA ordinal",
    )

    root: dict[str, Any] = {
        "project": str(project.resolve()),
        "bundle": _bundle_ref(bundle),
        "input_manifest": _file_ref(root_input_manifest),
        "acquisition_gate": _file_ref(acquisition_gate),
        "deployment_metadata": _file_ref(deployment_metadata),
        "canonical_manifest": _file_ref(canonical_manifest),
        "deployment_manifest": _file_ref(deployment_manifest),
        "deployment_lock": _file_ref(deployment_lock),
        "image_root": str(image_root.resolve()),
        "python_executable": str(python_executable.resolve()),
        "v1_physical_gpus": list(v1_physical_gpus),
        "q8_physical_gpus": list(q8_physical_gpus),
        "strict_physical_gpu": bool(strict_physical_gpu),
        "v1_shards": max(1, len(v1_physical_gpus) * 2),
        "v20_shards": 8,
        "q8_shards": 8,
        "v1_concurrency": len(v1_physical_gpus),
        "v20_concurrency": 8,
        "q8_concurrency": len(q8_physical_gpus),
    }
    if reference_registry is not None:
        root["reference_registry"] = _file_ref(reference_registry)

    payload: dict[str, Any] = {
        "schema_version": WORKFLOW_MANIFEST_SCHEMA,
        "status": STATUS,
        "model_contract_proposal": _file_ref(model_contract_proposal),
        "root_provider": root,
        "stageb": {
            "input_manifest": _file_ref(stageb_input_manifest),
            "checkpoints": checkpoint_refs,
            "candidate_manifest": _file_ref(candidate_manifest),
            "selected_model_metadata": _file_ref(selected_model_metadata),
            "selection_receipt": _file_ref(selection_receipt),
            "physical_gpu": stageb_physical_gpu,
            "internal_device": stageb_internal_device,
            "shared_input_acceleration": bool(shared_input_acceleration),
        },
        "traits": {"metadata_csv": _file_ref(traits_metadata)},
        "distal_axis_profiles": {"contract_json": _file_ref(profile_contract)},
        "review_overlays": {"enabled": False},
        "guards": {
            "condition_metadata_used_for_routing": False,
            "canonical_annotations_read": False,
            "blind_images_used": 0,
            "root_cap_region_output": False,
        },
        "benchmark_contract": {
            "ordered_raw_source_manifest": _file_ref(stageb_input_manifest),
            "warmup_runs": 0,
            "measured_repeats": 1,
            "fresh_no_resume_required": True,
        },
    }
    payload["manifest_identity_sha256"] = sha256_json(payload)

    # Validate the exact object before publication.  This is deterministic and
    # CPU-only: build_analysis_plan hashes inputs but performs no nvidia-smi or
    # model execution and creates no output directory.
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = destination.with_name(f".{destination.name}.validation-attempt")
    _require(not attempt.exists(), f"workflow validation attempt already exists: {attempt}")
    atomic_write_json(attempt, payload)
    try:
        plan = build_analysis_plan(
            attempt,
            output=destination.parent / f".{destination.stem}.planned-workflow-output",
            review_overlays=False,
        )
        _require(
            int(plan.get("tasks", -1)) > 0
            and plan.get("manifest_identity_sha256")
            == payload["manifest_identity_sha256"],
            "workflow plan did not preserve the sealed manifest authority",
        )
    finally:
        if attempt.exists():
            attempt.unlink()
    atomic_write_json(destination, payload)
    return deepcopy(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--root-input-manifest", type=Path, required=True)
    parser.add_argument("--acquisition-gate", type=Path, required=True)
    parser.add_argument("--deployment-metadata", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--deployment-lock", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--reference-registry", type=Path)
    parser.add_argument("--model-contract-proposal", type=Path, required=True)
    parser.add_argument("--stageb-input-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--selected-model-metadata", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--traits-metadata", type=Path, required=True)
    parser.add_argument("--profile-contract", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--v1-physical-gpu", type=int, action="append", required=True)
    parser.add_argument("--q8-physical-gpu", type=int, action="append", required=True)
    parser.add_argument("--stageb-physical-gpu", type=int, required=True)
    parser.add_argument("--stageb-internal-device", required=True)
    parser.add_argument("--disable-shared-input-acceleration", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = build_manifest(
            project=args.project,
            bundle=args.bundle,
            root_input_manifest=args.root_input_manifest,
            acquisition_gate=args.acquisition_gate,
            deployment_metadata=args.deployment_metadata,
            canonical_manifest=args.canonical_manifest,
            deployment_manifest=args.deployment_manifest,
            deployment_lock=args.deployment_lock,
            image_root=args.image_root,
            reference_registry=args.reference_registry,
            model_contract_proposal=args.model_contract_proposal,
            stageb_input_manifest=args.stageb_input_manifest,
            checkpoints=args.checkpoint,
            candidate_manifest=args.candidate_manifest,
            selected_model_metadata=args.selected_model_metadata,
            selection_receipt=args.selection_receipt,
            traits_metadata=args.traits_metadata,
            profile_contract=args.profile_contract,
            output=args.output,
            python_executable=args.python,
            v1_physical_gpus=args.v1_physical_gpu,
            q8_physical_gpus=args.q8_physical_gpu,
            stageb_physical_gpu=args.stageb_physical_gpu,
            stageb_internal_device=args.stageb_internal_device,
            shared_input_acceleration=not args.disable_shared_input_acceleration,
        )
    except (WorkflowManifestError, OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
