#!/usr/bin/env python3
"""Execute the locked real release example and seal its canonical identity.

``--check`` is CPU-only and validates all available authorities without
starting inference.  ``--execute`` performs one fresh PHAxis workflow run after
an explicit nvidia-smi preflight, inventories path-independent canonical
outputs, and publishes both the reference-output receipt and the exact identity
consumed by the later isolated clean-install verification.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402
from phaxis.public_identity import validate_proposal_public_identity  # noqa: E402
from phaxis.workflow import load_analysis_manifest  # noqa: E402


EXPECTED_SCHEMA = "PHAxis-clean-install-example-expected-identity-1.0"
EXPECTED_STATUS = "locked_final_real_example_before_clean_install"
RECEIPT_SCHEMA = "PHAxis-clean-install-reference-output-1.0"
RECEIPT_STATUS = "completed_fresh_real_nonblind_reference"
PLAN_SCHEMA = "PHAxis-clean-install-reference-output-plan-1.0"
CAPSULE_SCHEMA = "PHAxis-portable-model-runtime-capsule-1.0"
CANONICAL_REQUIRED_OUTPUTS = {
    "traits/traits.csv",
    "traits/image_traits.csv",
    "traits/detailed_root_statistics.csv",
    "traits/hair_instances.csv",
    "traits/analysis_metadata.csv",
    "distal_axis_profiles/distal_axis_profiles.csv",
}
SUMMARY_PATHS = {
    "fusion/fusion_summary.json",
    "traits/summary.json",
    "distal_axis_profiles/summary.json",
}


class ExpectedIdentityError(RuntimeError):
    """The real release example cannot authorize a clean-install prelock."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpectedIdentityError(message)


def _sealed(payload: Mapping[str, Any], field: str, role: str) -> str:
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field, None)
    _require(isinstance(observed, str) and observed == sha256_json(unsigned), f"{role}: identity mismatch")
    return observed


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _portable_capsule_context(
    *,
    capsule_root: Path,
    example_manifest: Path,
    proposal: Path,
    applied: Path,
    model_bundle: Path,
) -> dict[str, Any]:
    capsule = capsule_root.resolve()
    _require(
        capsule.is_dir() and not capsule.is_symlink(),
        "portable capsule root is absent or symlinked",
    )
    canonical_example = (
        capsule / "model/examples/clean_install/release_example_manifest.json"
    ).resolve()
    canonical_proposal = (
        capsule / "model/assets/runtime/model_contract_proposal.json"
    ).resolve()
    canonical_applied = (
        capsule / "model/assets/runtime/applied_model_contract.json"
    ).resolve()
    canonical_bundle = (
        capsule / "model/assets/MODEL_BUNDLE_MANIFEST.json"
    ).resolve()
    for observed, canonical, role in (
        (example_manifest.resolve(), canonical_example, "example manifest"),
        (proposal.resolve(), canonical_proposal, "model-contract proposal"),
        (applied.resolve(), canonical_applied, "applied model contract"),
        (model_bundle.resolve(), canonical_bundle, "model-bundle manifest"),
    ):
        _require(
            canonical.is_file()
            and not canonical.is_symlink()
            and sha256_file(observed) == sha256_file(canonical),
            f"{role} differs from its canonical portable-capsule copy",
        )

    bundle = read_json(canonical_bundle)
    members = bundle.get("members")
    _require(isinstance(members, list) and bool(members), "portable model bundle members are absent")
    expected = {"model/assets/MODEL_BUNDLE_MANIFEST.json"}
    for index, row in enumerate(members):
        _require(isinstance(row, Mapping), f"portable model member {index} is invalid")
        relative = Path(str(row.get("path") or ""))
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"portable model member {index} path is unsafe",
        )
        member = (capsule / relative).resolve()
        _require(
            _inside(member, capsule)
            and member.is_file()
            and not member.is_symlink()
            and sha256_file(member) == row.get("sha256")
            and member.stat().st_size == row.get("bytes"),
            f"portable model member {index} differs from the sealed bundle",
        )
        expected.add(relative.as_posix())
    observed = {
        path.relative_to(capsule).as_posix()
        for path in capsule.rglob("*")
        if path.is_file()
    }
    _require(observed == expected, "portable capsule has unlisted or missing files")

    workflow = read_json(canonical_example)
    workflow_base = canonical_example.parent

    def local_path(value: Any, role: str, *, directory: bool = False) -> Path:
        _require(isinstance(value, str) and value, f"{role} path is absent")
        supplied = Path(value)
        _require(not supplied.is_absolute(), f"{role} must be capsule-relative")
        resolved = (workflow_base / supplied).resolve()
        _require(_inside(resolved, capsule), f"{role} escapes portable capsule")
        _require(
            resolved.is_dir() if directory else resolved.is_file(),
            f"{role} is absent from portable capsule",
        )
        return resolved

    def local_ref(value: Any, role: str) -> Path:
        _require(isinstance(value, Mapping), f"{role} locked reference is absent")
        resolved = local_path(value.get("path"), role)
        _require(
            sha256_file(resolved) == value.get("sha256"),
            f"{role} SHA-256 differs in portable capsule",
        )
        return resolved

    root_section = workflow.get("root_provider")
    stageb_section = workflow.get("stageb")
    _require(
        isinstance(root_section, Mapping) and isinstance(stageb_section, Mapping),
        "portable workflow root/Stage-B sections are absent",
    )
    _require(
        "python_executable" not in root_section,
        "portable workflow pins an authoring Python executable",
    )
    local_path(str(root_section.get("project") or "."), "root project", directory=True)
    local_path(str(root_section.get("image_root") or ""), "root image root", directory=True)
    local_path(str(stageb_section.get("image_root") or ""), "Stage-B image root", directory=True)
    bundle_ref = root_section.get("bundle")
    _require(isinstance(bundle_ref, Mapping), "portable root bundle reference is absent")
    bundle_path = local_path(bundle_ref.get("path"), "root bundle", directory=True)
    _require(
        sha256_file(bundle_path / "root_provider_bundle.json")
        == bundle_ref.get("registry_sha256"),
        "portable root bundle registry SHA-256 differs",
    )
    local_ref(workflow.get("model_contract_proposal"), "model-contract proposal")
    for field in (
        "input_manifest",
        "acquisition_gate",
        "deployment_metadata",
        "canonical_manifest",
        "deployment_manifest",
        "deployment_lock",
    ):
        local_ref(root_section.get(field), f"root {field}")
    _require(
        root_section.get("reference_registry") is None,
        "one-task portable workflow must not run the exact283 reference audit",
    )
    for field in (
        "input_manifest",
        "candidate_manifest",
        "selected_model_metadata",
        "selection_receipt",
    ):
        local_ref(stageb_section.get(field), f"Stage-B {field}")
    checkpoints = stageb_section.get("checkpoints")
    _require(
        isinstance(checkpoints, list) and len(checkpoints) == 5,
        "portable Stage-B checkpoint closure is not exactly five",
    )
    for index, value in enumerate(checkpoints):
        local_ref(value, f"Stage-B checkpoint {index}")
    local_ref(workflow.get("traits", {}).get("metadata_csv"), "traits metadata")
    profile_section = workflow.get(
        "distal_axis_profiles", workflow.get("profiles", {})
    )
    local_ref(profile_section.get("contract_json"), "distal-axis profile contract")
    release_example = workflow.get("release_example")
    _require(isinstance(release_example, Mapping), "portable release-example block is absent")
    source_image = local_path(
        release_example.get("source_image_relpath"), "release-example source image"
    )
    _require(
        sha256_file(source_image) == release_example.get("source_image_sha256"),
        "portable release-example source image SHA-256 differs",
    )

    receipt_path = capsule / "model/examples/clean_install/receipt.json"
    receipt = read_json(receipt_path)
    capsule_identity = _sealed(
        receipt, "portable_capsule_identity_sha256", "portable capsule receipt"
    )
    _require(
        receipt.get("schema_version") == CAPSULE_SCHEMA
        and receipt.get("status")
        == "completed_self_contained_raw_to_profiles_runtime"
        and receipt.get("authoring_workspace_paths_required") is False
        and receipt.get("root_subprocess_python_rebound_to_active_interpreter")
        is True
        and receipt.get("blind_images_used") == 0,
        "portable capsule receipt is not release eligible",
    )
    return {
        "root": capsule,
        "identity": capsule_identity,
        "members": len(members),
        "tree_identity_sha256": sha256_json(
            [
                {
                    "path": row["path"],
                    "bytes": row["bytes"],
                    "sha256": row["sha256"],
                }
                for row in members
            ]
        ),
    }


def _public_identity(proposal: Mapping[str, Any]) -> dict[str, str]:
    try:
        result = validate_proposal_public_identity(proposal)
    except Exception as error:  # package ContractError is deliberately normalized
        raise ExpectedIdentityError(f"proposal public identity is invalid: {error}") from error
    root_expert = proposal.get("root_expert")
    hair_expert = proposal.get("hair_identity_count_expert")
    _require(
        isinstance(root_expert, Mapping)
        and isinstance(hair_expert, Mapping)
        and isinstance(root_expert.get("bundle_identity_sha256"), str)
        and isinstance(hair_expert.get("expert_id"), str),
        "proposal public expert identities are incomplete",
    )
    return {
        **{str(key): str(value) for key, value in result.items()},
        "root_bundle_identity_sha256": str(root_expert["bundle_identity_sha256"]),
        "hair_identity_count_expert": str(hair_expert["expert_id"]),
    }


def _authority_context(
    *,
    example_manifest: Path,
    model_contract_proposal: Path,
    applied_model_contract: Path,
    model_bundle_manifest: Path,
    portable_capsule_root: Path,
    source_release_root: Path,
    formal_wheel: Path,
    python_executable: Path,
    physical_gpu: int,
) -> dict[str, Any]:
    paths = {
        "example_manifest": example_manifest.resolve(),
        "proposal": model_contract_proposal.resolve(),
        "applied": applied_model_contract.resolve(),
        "model_bundle": model_bundle_manifest.resolve(),
        "wheel": formal_wheel.resolve(),
        "python": python_executable.resolve(),
    }
    for role, path in paths.items():
        _require(path.is_file() and not path.is_symlink(), f"{role} is absent or symlinked")
        _require("blind" not in str(path).casefold(), f"{role} has a blind-labelled path")
    source_root = source_release_root.resolve()
    source_manifest_path = source_root / "SOURCE_MANIFEST.json"
    _require(source_root.is_dir() and not source_root.is_symlink(), "source release root is absent or symlinked")
    _require(source_manifest_path.is_file(), "source release manifest is absent")
    capsule = _portable_capsule_context(
        capsule_root=portable_capsule_root,
        example_manifest=paths["example_manifest"],
        proposal=paths["proposal"],
        applied=paths["applied"],
        model_bundle=paths["model_bundle"],
    )

    example = load_analysis_manifest(paths["example_manifest"])
    example_identity = str(example["manifest_identity_sha256"])
    release_example = example.get("release_example")
    _require(
        isinstance(release_example, Mapping)
        and release_example.get("input_kind") == "real_nonblind_release_example"
        and release_example.get("release_authorized") is True
        and release_example.get("development_or_synthetic_smoke") is False
        and release_example.get("tasks") == 1
        and release_example.get("blind_images_used") == 0,
        "workflow manifest is not the locked real/nonblind one-task example",
    )
    root_provider = example.get("root_provider")
    stageb = example.get("stageb")
    _require(isinstance(root_provider, Mapping) and isinstance(stageb, Mapping), "example GPU mapping is absent")
    _require(
        root_provider.get("v1_physical_gpus") == [physical_gpu]
        and root_provider.get("q8_physical_gpus") == [physical_gpu]
        and stageb.get("physical_gpu") == physical_gpu
        and stageb.get("internal_device") == "cuda:0",
        "example workflow physical/internal GPU mapping differs from the declared card",
    )

    proposal = read_json(paths["proposal"])
    proposal_identity = _sealed(proposal, "model_contract_identity_sha256", "model-contract proposal")
    _require(
        proposal.get("schema_version") == "PHAxis-model-contract-1.0.0"
        and proposal.get("product") == "PHAxis"
        and proposal.get("product_version") == "1.0.0"
        and proposal.get("formal_release_status") == "passed_proposal_not_official",
        "model-contract proposal is not the final unapplied authority",
    )
    public = _public_identity(proposal)
    applied = read_json(paths["applied"])
    applied_identity = _sealed(applied, "model_contract_identity_sha256", "applied model contract")
    promotion = applied.get("promotion")
    _require(
        applied.get("schema_version") == "PHAxis-model-contract-1.0.0"
        and applied.get("product") == "PHAxis"
        and applied.get("product_version") == "1.0.0"
        and applied.get("formal_release_status") == "passed"
        and isinstance(promotion, Mapping)
        and promotion.get("status") == "applied_formal_release"
        and promotion.get("official_apply_performed") is True
        and promotion.get("proposal_file_sha256") == sha256_file(paths["proposal"])
        and promotion.get("proposal_identity_sha256") == proposal_identity
        and _public_identity(applied) == public,
        "applied model contract does not bind the proposal/public identity",
    )
    bundle = read_json(paths["model_bundle"])
    bundle_identity = _sealed(bundle, "model_bundle_manifest_identity_sha256", "model bundle manifest")
    _require(
        bundle.get("schema_version") == "PHAxis-model-bundle-release-manifest-1.0"
        and bundle.get("status") == "completed_final_immutable_bundle"
        and bundle.get("applied_model_contract_sha256") == sha256_file(paths["applied"])
        and bundle.get("applied_model_contract_identity_sha256") == applied_identity
        and bundle.get("model_contract_proposal_sha256") == sha256_file(paths["proposal"])
        and bundle.get("model_contract_proposal_identity_sha256") == proposal_identity
        and bundle.get("model_bundle_id") == public["model_bundle_id"]
        and bundle.get("root_expert_id") == public["root_expert_id"]
        and bundle.get("root_bundle_identity_sha256")
        == public["root_bundle_identity_sha256"]
        and bundle.get("hair_identity_count_expert")
        == public["hair_identity_count_expert"],
        "model bundle does not bind the applied/proposal public identity",
    )
    source = read_json(source_manifest_path)
    _require(
        source.get("schema_version") == "PHAxis-source-release-manifest-2.0"
        and source.get("distribution") == "phaxis"
        and source.get("version") == "1.0.0"
        and source.get("release_mode") == "formal"
        and isinstance(source.get("tree_identity_sha256"), str),
        "source release manifest is not formal PHAxis 1.0.0",
    )
    _require(
        paths["wheel"].suffix.casefold() == ".whl"
        and paths["wheel"].name.casefold().startswith("phaxis-1.0.0-"),
        "formal wheel filename is not PHAxis 1.0.0",
    )
    _require(isinstance(physical_gpu, int) and physical_gpu >= 0, "physical GPU is invalid")
    return {
        "paths": paths,
        "source_root": source_root,
        "source_manifest_path": source_manifest_path,
        "source": source,
        "example": example,
        "example_identity": example_identity,
        "proposal_identity": proposal_identity,
        "applied_identity": applied_identity,
        "bundle_identity": bundle_identity,
        "public": public,
        "physical_gpu": physical_gpu,
        "capsule": capsule,
    }


def _canonical_output_records(output: Path) -> list[dict[str, Any]]:
    actual: set[str] = set()
    for directory in ("fusion", "traits", "distal_axis_profiles"):
        root = output / directory
        _require(root.is_dir() and not root.is_symlink(), f"reference output directory is absent: {directory}")
        for path in root.rglob("*"):
            _require(not path.is_symlink(), f"reference output symlink is forbidden: {path}")
            if path.is_file():
                relative = path.relative_to(output).as_posix()
                if relative not in SUMMARY_PATHS:
                    actual.add(relative)
    _require(CANONICAL_REQUIRED_OUTPUTS.issubset(actual), "reference output omits canonical traits/profiles files")
    _require(
        len([path for path in actual if path.startswith("fusion/predictions/") and path.endswith(".json")]) == 1,
        "reference output must contain exactly one final prediction",
    )
    return [
        {"path": relative, "bytes": (output / relative).stat().st_size, "sha256": sha256_file(output / relative)}
        for relative in sorted(actual)
    ]


def _validate_reference_summaries(context: Mapping[str, Any], output: Path) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    for role, relative, schema, identity_field, count_field in (
        ("fusion", "fusion/fusion_summary.json", "PHAxis-fusion-run-1.1", "summary_identity_sha256", "images"),
        ("traits", "traits/summary.json", "PHAxis-trait-export-1.0", "export_identity_sha256", "tasks"),
        ("profiles", "distal_axis_profiles/summary.json", "PHAxis-distal-axis-profile-export-1.0.0", "export_identity_sha256", "tasks"),
    ):
        path = output / relative
        _require(path.is_file(), f"reference {role} summary is absent")
        payload = read_json(path)
        identity = _sealed(payload, identity_field, f"reference {role} summary")
        _require(
            payload.get("schema_version") == schema
            and payload.get("status") == "completed"
            and payload.get(count_field) == 1
            and payload.get("model_contract_proposal_sha256")
            == sha256_file(context["paths"]["proposal"])
            and payload.get("model_contract_proposal_identity_sha256")
            == context["proposal_identity"]
            and payload.get("blind_images_used") == 0,
            f"reference {role} summary is not completed one-task/proposal-bound output",
        )
        bindings[role] = {
            "path": relative,
            "sha256": sha256_file(path),
            "identity_sha256": identity,
        }
    return bindings


def _expected_payload(context: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [dict(record) for record in records]
    expected: dict[str, Any] = {
        "schema_version": EXPECTED_SCHEMA,
        "status": EXPECTED_STATUS,
        "input_kind": "real_nonblind_release_example",
        "release_authorized": True,
        "development_or_synthetic_smoke": False,
        "tasks": 1,
        "canonical_output_files": normalized,
        "expected_example_output_identity_sha256": sha256_json(normalized),
        "example_manifest_sha256": sha256_file(context["paths"]["example_manifest"]),
        "example_manifest_identity_sha256": context["example_identity"],
        "model_contract_proposal_sha256": sha256_file(context["paths"]["proposal"]),
        "model_contract_proposal_identity_sha256": context["proposal_identity"],
        "applied_model_contract_sha256": sha256_file(context["paths"]["applied"]),
        "applied_model_contract_identity_sha256": context["applied_identity"],
        "model_bundle_manifest_sha256": sha256_file(context["paths"]["model_bundle"]),
        "model_bundle_manifest_identity_sha256": context["bundle_identity"],
        "source_release_manifest_sha256": sha256_file(context["source_manifest_path"]),
        "source_release_tree_identity_sha256": context["source"]["tree_identity_sha256"],
        "formal_wheel_sha256": sha256_file(context["paths"]["wheel"]),
        "portable_capsule_identity_sha256": context["capsule"]["identity"],
        "portable_capsule_tree_identity_sha256": context["capsule"][
            "tree_identity_sha256"
        ],
        **context["public"],
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    expected["expected_identity_receipt_identity_sha256"] = sha256_json(expected)
    return expected


def check_plan(**kwargs: Any) -> dict[str, Any]:
    context = _authority_context(**kwargs)
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "status": "validated_not_executed",
        "default_check_only": True,
        "execute_requires_explicit_flag": True,
        "tasks": 1,
        "physical_gpu": context["physical_gpu"],
        "cuda_visible_devices": str(context["physical_gpu"]),
        "example_manifest_identity_sha256": context["example_identity"],
        "model_contract_proposal_identity_sha256": context["proposal_identity"],
        "applied_model_contract_identity_sha256": context["applied_identity"],
        "model_bundle_manifest_identity_sha256": context["bundle_identity"],
        "portable_capsule_identity_sha256": context["capsule"]["identity"],
        "portable_capsule_tree_identity_sha256": context["capsule"][
            "tree_identity_sha256"
        ],
        "formal_wheel_sha256": sha256_file(context["paths"]["wheel"]),
        "fresh_no_resume_required": True,
        "canonical_annotations_read": False,
        "condition_metadata_read": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    plan["plan_identity_sha256"] = sha256_json(plan)
    return plan


def execute_expected_identity(*, output: Path, **kwargs: Any) -> dict[str, Any]:
    context = _authority_context(**kwargs)
    destination = output.resolve()
    _require(not destination.exists(), f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        preflight = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        _require(preflight.returncode == 0 and preflight.stdout.strip(), "nvidia-smi preflight failed")
        preflight_path = staging / "nvidia_smi_preflight.txt"
        preflight_path.write_text(preflight.stdout, encoding="utf-8")
        analysis = staging / "reference_analysis"
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(context["physical_gpu"])
        command = [
            str(context["paths"]["python"]),
            "-m",
            "phaxis.cli",
            "analyze",
            "--manifest",
            str(context["paths"]["example_manifest"]),
            "--output",
            str(analysis),
            "--execute",
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        (staging / "workflow_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (staging / "workflow_stderr.txt").write_text(completed.stderr, encoding="utf-8")
        _require(completed.returncode == 0, "fresh release-example workflow execution failed")
        state_path = analysis / "workflow_state.json"
        _require(state_path.is_file(), "fresh reference workflow state is absent")
        state = read_json(state_path)
        state_identity = _sealed(state, "state_identity_sha256", "fresh reference workflow state")
        attempts = state.get("execution_attempts")
        _require(
            state.get("schema_version") == "PHAxis-analysis-workflow-state-1.1"
            and state.get("status") == "completed"
            and isinstance(attempts, list)
            and len(attempts) == 1
            and attempts[0].get("status") == "completed"
            and attempts[0].get("resume_requested") is False
            and attempts[0].get("resume_or_cache_used") is False
            and attempts[0].get("fresh_direct_benchmark_eligible") is True
            and state.get("blind_images_used") == 0,
            "reference workflow was not one fresh completed execution",
        )
        summary_bindings = _validate_reference_summaries(context, analysis)
        records = _canonical_output_records(analysis)
        expected = _expected_payload(context, records)
        expected_path = staging / "expected_identity.json"
        atomic_write_json(expected_path, expected)
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "status": RECEIPT_STATUS,
            "tasks": 1,
            "physical_gpu": context["physical_gpu"],
            "cuda_visible_devices": str(context["physical_gpu"]),
            "nvidia_smi_preflight_sha256": sha256_file(preflight_path),
            "example_manifest_sha256": sha256_file(context["paths"]["example_manifest"]),
            "example_manifest_identity_sha256": context["example_identity"],
            "workflow_state_sha256": sha256_file(state_path),
            "workflow_state_identity_sha256": state_identity,
            "summary_bindings": summary_bindings,
            "canonical_output_files": len(records),
            "canonical_output_identity_sha256": sha256_json(records),
            "expected_identity_sha256": sha256_file(expected_path),
            "expected_identity_receipt_identity_sha256": expected[
                "expected_identity_receipt_identity_sha256"
            ],
            "portable_capsule_identity_sha256": context["capsule"]["identity"],
            "portable_capsule_tree_identity_sha256": context["capsule"][
                "tree_identity_sha256"
            ],
            "fresh_execution": True,
            "resume_or_cache_used": False,
            "canonical_annotations_read": False,
            "condition_metadata_read": False,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        }
        receipt["reference_output_identity_sha256"] = sha256_json(receipt)
        atomic_write_json(staging / "receipt.json", receipt)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return deepcopy(receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-manifest", type=Path, required=True)
    parser.add_argument("--model-contract-proposal", type=Path, required=True)
    parser.add_argument("--applied-model-contract", type=Path, required=True)
    parser.add_argument("--model-bundle-manifest", type=Path, required=True)
    parser.add_argument("--portable-capsule-root", type=Path, required=True)
    parser.add_argument("--source-release-root", type=Path, required=True)
    parser.add_argument("--formal-wheel", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument(
        "--cuda-visible-devices",
        default="0",
        help="explicit single-card CVD mapping; must equal --physical-gpu",
    )
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cuda_visible_devices != str(args.physical_gpu):
        print(
            "clean-install expected-identity blocked: CVD/physical GPU mapping differs",
            file=sys.stderr,
        )
        return 2
    kwargs = {
        "example_manifest": args.example_manifest,
        "model_contract_proposal": args.model_contract_proposal,
        "applied_model_contract": args.applied_model_contract,
        "model_bundle_manifest": args.model_bundle_manifest,
        "portable_capsule_root": args.portable_capsule_root,
        "source_release_root": args.source_release_root,
        "formal_wheel": args.formal_wheel,
        "python_executable": args.python,
        "physical_gpu": args.physical_gpu,
    }
    try:
        result = (
            execute_expected_identity(output=args.output, **kwargs)
            if args.execute
            else check_plan(**kwargs)
        )
    except ExpectedIdentityError as error:
        print(f"clean-install expected-identity blocked: {error}", file=sys.stderr)
        return 2
    identity = result.get("reference_output_identity_sha256") or result.get("plan_identity_sha256")
    print(json.dumps({"status": result["status"], "identity_sha256": identity}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
