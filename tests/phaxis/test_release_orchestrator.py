from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

import phaxis.release_orchestrator as release_orchestrator_module
from phaxis.hair_stageb.candidate_bundle import (
    AMP_AMENDMENT_SCHEMA,
    CANDIDATE_MANIFEST_SCHEMA,
    CANDIDATE_STATUS,
    FORMAL_TRAIN399_SEEDS,
    TRAIN399_CHECKPOINT_POLICY,
    TRAINING_FAILURE_SCHEMA,
    amp_backward_retry_policy_lock,
    operating_point_selection_contract,
)
from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json
from phaxis.model_contract_binding import (
    RUN_SCOPED_AUTHORITY_PIN_LIFECYCLE,
    read_model_contract_authority,
)
from phaxis.narrative_decision import (
    COHORT_ORDER,
    EFFECT_ORDER,
    ENDPOINT_ORDER,
    build_narrative_decision,
)
from phaxis.public_identity import (
    PUBLIC_SYSTEM_DERIVATION,
    PUBLIC_SYSTEM_IDENTITY_SCHEMA,
    derive_public_identity,
)
from phaxis.publication_titles import title_contract
from phaxis.release_orchestrator import (
    DEFERRED_HUMAN_AUTHORITY_CONTRACT_SCHEMA,
    DEFERRED_HUMAN_WORK_ITEM_SCHEMA,
    EXPECTED_GPU_HOLD_EXIT_CODE,
    EXPECTED_HUMAN_GATE_EXIT_CODE,
    KNOWN_STAGE_SCHEMAS,
    MANDATORY_STAGE_ORDER,
    PEP517_SDIST_GENERATED_MEMBERS,
    ReleaseOrchestratorError,
    _validate_figure_table_bundle,
    build_release_plan,
    execute_release,
)
from phaxis.release_topology import STAGE_DEPENDENCIES
from phaxis.supplementary_tables import (
    FINAL_STATUS as SUPPLEMENTARY_TABLE_FINAL_STATUS,
    materialize_supplementary_table_data_bundle,
)
from tests.phaxis.test_supplementary_table_data_bundle import source_fixture as _supplementary_source_fixture
from scripts.phaxis import materialize_offline_dependencies as dependency_materializer


def test_public_release_registry_reference_never_discloses_host_path(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    registry = workspace / "authority/release_registry.json"
    outside = tmp_path.parent / "external-release-registry.json"
    assert release_orchestrator_module._public_registry_path(
        registry, workspace=workspace
    ) == "authority/release_registry.json"
    assert release_orchestrator_module._public_registry_path(
        outside, workspace=workspace
    ) == "<RELEASE_AUTHORITY_REGISTRY>"
    for host_path in (
        "D:" + r"\private\release_registry.json",
        "\\" + r"\server\share\release_registry.json",
        "/" + "home/private/release_registry.json",
    ):
        assert release_orchestrator_module._is_absolute_host_path(host_path)
    for public_role in (
        "authority/release_registry.json",
        "<RELEASE_AUTHORITY_REGISTRY>",
        "<PRIVATE_MANIFEST_EXACT_SOURCE_COPY>",
    ):
        assert not release_orchestrator_module._is_absolute_host_path(public_role)


def test_command_guard_allows_only_pinned_manuscript_renderer_powershell_argument() -> None:
    valid = [
        "python.exe",
        "scripts/phaxis/render_manuscript_bundle.py",
        "--powershell",
        "powershell.exe",
        "--output",
        "outputs/render",
    ]
    release_orchestrator_module._validate_command(valid, stage="manuscript_render")

    invalid = (
        (valid, "another_stage"),
        ([*valid[:2], "--renderer", "powershell.exe", *valid[4:]], "manuscript_render"),
        ([valid[0], "scripts/phaxis/other.py", *valid[2:]], "manuscript_render"),
        ([*valid[:3], "pwsh.exe", *valid[4:]], "manuscript_render"),
        (["powershell.exe", "-File", "anything.ps1"], "manuscript_render"),
    )
    for command, stage in invalid:
        with pytest.raises(ReleaseOrchestratorError):
            release_orchestrator_module._validate_command(command, stage=stage)


def _write_json(path: Path, payload: dict) -> Path:
    atomic_write_json(path, payload)
    return path


def _synthetic_narrative_decision() -> dict:
    rows = [
        {
            "endpoint_key": endpoint,
            "effect_key": effect,
            "cohort": cohort,
            "estimate": 1.0,
            "ci_low": 0.9,
            "ci_high": 1.1,
            "endpoint_n": 261 if cohort == "primary_clean261" else 283,
            "effect_scale": "ratio",
        }
        for endpoint in ENDPOINT_ORDER
        for effect in EFFECT_ORDER
        for cohort in COHORT_ORDER
    ]
    return build_narrative_decision(
        rows,
        source_sha256={"phenotype_effects": sha256_json(rows)},
    )


def _synthetic_cohort_table_hashes() -> dict[str, dict[str, str]]:
    return {
        cohort: {
            table: sha256_json(["synthetic-cohort-table", cohort, table])
            for table in (
                "traits",
                "detailed_root_statistics",
                "hair_instances",
                "image_traits",
            )
        }
        for cohort in ("primary_clean261", "sensitivity_full283")
    }


def _write_synthetic_profile_csv(path: Path, task_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_id", "source_image_sha256", "bin_index"),
        )
        writer.writeheader()
        writer.writerows(
            {
                "task_id": task_id,
                "source_image_sha256": sha256_json(
                    ["synthetic-profile-source", task_id]
                ),
                "bin_index": bin_index,
            }
            for task_id in task_ids
            for bin_index in range(5)
        )


def _synthetic_wheel(
    path: Path,
    *,
    name: str,
    version: str,
    requires: list[str] | None = None,
    license_expression: str = "MIT",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        "Metadata-Version: 2.4",
        f"Name: {name}",
        f"Version: {version}",
        f"License-Expression: {license_expression}",
        "License-File: LICENSE.txt",
    ]
    metadata.extend(f"Requires-Dist: {item}" for item in (requires or ()))
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", "\n".join(metadata) + "\n")
        archive.writestr(
            f"{dist_info}/licenses/LICENSE.txt",
            f"Synthetic {license_expression} license for {name}.\n",
        )
    return path


def _synthetic_candidate_preview(
    checkpoints: list[Path], audit: Path
) -> dict:
    training_lock = {
        "training_task_ids_sha256": sha256_json(
            [f"T{index:03d}" for index in range(399)]
        ),
        "split_manifest_sha256": sha256_json(["synthetic-split"]),
    }
    training_lock_identity = sha256_json(training_lock)
    members = []
    for index, (seed, checkpoint) in enumerate(
        zip(FORMAL_TRAIN399_SEEDS, checkpoints, strict=True)
    ):
        retry_count = 1 if seed == FORMAL_TRAIN399_SEEDS[2] else 0
        member = {
            "member_index": index,
            "member_id": f"seed_{seed}",
            "seed": seed,
            "epoch": 60,
            "global_step": 23_940,
            "checkpoint_sha256": sha256_file(checkpoint),
            "model_state_sha256": sha256_json(["model-state", seed]),
            "training_receipt_sha256": sha256_json(
                ["synthetic-training-receipt", seed]
            ),
            "training_receipt_filename": "training_receipt.json",
            "optimizer_name": "AdamW",
            "optimizer_parameter_state_count": 1,
            "optimizer_parameter_step": 23_940,
            "amp_final_scale": 512.0 if retry_count else 1024.0,
            "amp_growth_tracker": 23_934 if retry_count else 23_940,
            "amp_backward_retry_count": retry_count,
            "amp_backward_retry_mode": (
                "same_forward_graph_backoff"
                if retry_count
                else "legacy_or_amended_zero_retry"
            ),
            "optimizer_steps_skipped_due_nonfinite_gradients": 0,
        }
        if retry_count:
            member["amp_backward_retry_audit_sha256"] = sha256_json(
                ["synthetic-amp-backward-retry-audit", seed]
            )
        members.append(member)

    amendment_lock = {
        "schema_version": AMP_AMENDMENT_SCHEMA,
        "status": "applied_before_authoritative_seed3_optimizer_trajectory",
        "amendment_sha256": sha256_json(["synthetic-amp-amendment"]),
        "failure_receipt_schema_version": TRAINING_FAILURE_SCHEMA,
        "failure_receipt_sha256": sha256_json(
            ["synthetic-seed3-superseded-failure-receipt"]
        ),
        "fixed_numeric_policy": amp_backward_retry_policy_lock(),
        "unchanged_scientific_contract_sha256": sha256_json(
            ["synthetic-unchanged-scientific-contract"]
        ),
        "training_source_sha256": sha256_json(["synthetic-training-source"]),
    }
    operating = operating_point_selection_contract()
    identity = {
        "checkpoint_policy": TRAIN399_CHECKPOINT_POLICY,
        "members": members,
        "training_lock": training_lock,
        "training_lock_identity_sha256": training_lock_identity,
        "amp_backward_retry_amendment_lock": amendment_lock,
        "operating_point_selection_contract": operating,
    }
    candidate_identity = sha256_json(identity)
    metadata = {
        "expert_id": "PHAxis-StageB-train399-five-seed",
        "ensemble_members": 5,
        "checkpoint_policy": TRAIN399_CHECKPOINT_POLICY,
        "deployment_role": CANDIDATE_STATUS,
        "operating_point_status": "pending_QCdevelopment44_selection",
        "selected_score_threshold": None,
        "selection_receipt_sha256": None,
        "selection_receipt_identity_sha256": None,
        "candidate_pool_identity_sha256": None,
        "selected_model_metadata_identity_sha256": None,
        "training_images": 399,
        "validation_images": 44,
        "validation_labels_used_for_gradient_or_early_stopping": False,
        "blind_images_used": 0,
        "seeds": list(FORMAL_TRAIN399_SEEDS),
        "member_ids": [member["member_id"] for member in members],
        "checkpoint_sha256": [member["checkpoint_sha256"] for member in members],
        "model_state_sha256": [member["model_state_sha256"] for member in members],
        "training_task_ids_sha256": training_lock["training_task_ids_sha256"],
        "split_manifest_sha256": training_lock["split_manifest_sha256"],
        "training_lock_identity_sha256": training_lock_identity,
        "candidate_bundle_identity_sha256": candidate_identity,
        "operating_point_selection_contract_sha256": sha256_json(operating),
    }
    payload = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA,
        "status": CANDIDATE_STATUS,
        "candidate_only": True,
        "official_constants_modified": False,
        "official_model_contract_modified": False,
        "automatic_promotion_performed": False,
        "dataset_audit_path": str(audit.resolve()),
        "dataset_audit_sha256": sha256_file(audit),
        "amp_backward_retry_amendment_path": str(
            (audit.parent / "synthetic_amp_backward_retry_amendment.json").resolve()
        ),
        "superseded_failure_receipt_path": str(
            (audit.parent / "synthetic_seed3_training_failure.json").resolve()
        ),
        "source_checkpoint_paths_in_member_order": [
            str(path.resolve()) for path in checkpoints
        ],
        "identity_payload": identity,
        "candidate_bundle_identity_sha256": candidate_identity,
        "detection_model_metadata": metadata,
        "blind_images_used": 0,
    }
    payload["candidate_manifest_identity_sha256"] = sha256_json(payload)
    return payload


def _root_exact283(path: Path) -> Path:
    layer = {
        "exact": 283,
        "expected": 283,
        "mismatch_count": 0,
        "mismatch_task_ids": [],
        "gate_pass": True,
    }
    identity = {
        "schema_version": "PHAxis-root-provider-fresh-reference283-audit-1.0",
        "reference_identity_sha256": "1" * 64,
        "fresh_reference_identity_sha256": "2" * 64,
        "bundle_identity_sha256": "3" * 64,
        "pipeline_identity_sha256": "4" * 64,
        "layers": {
            "v12_strip_root_mask": dict(layer),
            "v20_root_polygon": dict(layer),
            "final_hybrid_root_mask": dict(layer),
        },
        "source_image_mismatch_task_ids": [],
        "prepared_radius_fallback_task_ids": [],
        "attachment_supported_extension_rescue_task_ids": [],
        "pipeline_raw_image_provenance_gate": True,
        "pipeline_stage_evidence_gate": True,
    }
    payload = {
        **identity,
        "status": "pass_exact_283",
        "audit_identity_sha256": sha256_json(identity),
        "fresh_portable_raw_image_rerun_completed": True,
        "fresh_283_exact_reproduction_claim_allowed": True,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }
    return _write_json(path, payload)


def _receipt_spec(
    schema: str,
    status: object,
    *,
    status_field: str = "status",
    identity_field: str | None = None,
    seals: bool = False,
) -> dict:
    payload = {
        "artifact": "receipt",
        "schema_version": schema,
        "status_field": status_field,
        "status": status,
        "required_fields": {},
    }
    if identity_field:
        payload["identity_field"] = identity_field
        payload["identity_seals_complete_object"] = seals
    return payload


def _gpu() -> dict:
    return {
        "physical_gpus": [1],
        "cuda_visible_devices": "1",
        "internal_device": "cuda:0",
        "estimated_peak_memory_mib": 256,
        "reserve_memory_mib": 2048,
        "maximum_utilization_pct": 80,
    }


def _input_stage(name: str) -> dict[str, str]:
    return {"stage": name, "artifact": "receipt"}


def _stage(
    name: str,
    schema: str,
    status: object,
    inputs: list[dict[str, str]],
    *,
    run_dir: Path,
    status_field: str = "status",
    identity_field: str | None = None,
    seals: bool = False,
    gpu: dict | None = None,
    same_hardware_as: str | None = None,
    command_extra: list[str] | None = None,
) -> dict:
    receipt_path = run_dir / name / "receipt.json"
    command = None
    if name not in {"authority_pin", "release_finalize"}:
        command = ["synthetic-stage", name]
        if gpu is not None and name != "benchmark_same_hardware":
            command.append("cuda:0")
        command.extend(command_extra or [])
    payload = {
        "name": name,
        "command": command,
        "inputs": inputs,
        "artifacts": [
            {"name": "receipt", "path": str(receipt_path), "kind": "file"}
        ],
        "receipt": _receipt_spec(
            schema,
            status,
            status_field=status_field,
            identity_field=identity_field,
            seals=seals,
        ),
    }
    if gpu is not None:
        payload["gpu"] = gpu
    if same_hardware_as is not None:
        payload["same_hardware_as"] = same_hardware_as
    return payload


def _manifest_fixture(tmp_path: Path) -> dict:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    candidate_inputs = workspace / "candidate_inputs"
    candidate_inputs.mkdir()
    train_ids = [f"T{index:03d}" for index in range(399)]
    val_ids = [f"V{index:03d}" for index in range(44)]
    audit = _write_json(
        candidate_inputs / "dataset_audit.json",
        {
            "schema_version": "PHAxis-StageB-train399-dataset-audit-1.0",
            "status": "passed",
            "train_ids": train_ids,
            "excluded_val_ids": val_ids,
            "blind_images_used": 0,
        },
    )
    checkpoints = []
    for seed in FORMAL_TRAIN399_SEEDS:
        checkpoint = candidate_inputs / f"seed_{seed}.pt"
        checkpoint.write_bytes(f"synthetic-checkpoint-{seed}".encode())
        checkpoints.append(checkpoint)
    preview = _synthetic_candidate_preview(checkpoints, audit)
    audit_payload = read_json(audit)

    training_members = []
    external_paths: dict[str, Path] = {"dataset_audit": audit}
    for seed, checkpoint in zip(FORMAL_TRAIN399_SEEDS, checkpoints, strict=True):
        receipt = _write_json(
            workspace / "training" / str(seed) / "training_receipt.json",
            {
                "schema_version": "PHAxis-StageB-train399-training-receipt-1.0",
                "status": "completed",
                "formal_training": True,
                "seed": seed,
                "epochs": 60,
                "steps_per_epoch": 399,
                "global_steps": 23940,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint),
                "nvidia_smi_preflight_status": "passed",
                "nvidia_smi_training_monitor_status": "passed",
                "validation_evaluated_during_training": False,
                "blind_images_used": 0,
            },
        )
        receipt_name = f"seed_{seed}_receipt"
        checkpoint_name = f"seed_{seed}_checkpoint"
        external_paths[receipt_name] = receipt
        external_paths[checkpoint_name] = checkpoint
        training_members.append(
            {
                "seed": seed,
                "completion_receipt_input": receipt_name,
                "checkpoint_input": checkpoint_name,
            }
        )

    qcdev = workspace / "qcdev44.csv"
    with qcdev.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task_id",))
        writer.writeheader()
        writer.writerows(
            {"task_id": task_id} for task_id in audit_payload["excluded_val_ids"]
        )
    locked_val = workspace / "locked_val_ids.txt"
    locked_val.write_text(
        "\n".join(audit_payload["excluded_val_ids"]) + "\n",
        encoding="utf-8",
    )
    source_image = workspace / "production_source.tif"
    source_image.write_bytes(b"one source may be reused by synthetic task IDs")
    production = workspace / "production283.csv"
    with production.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task_id",
                "image_path",
                "image_sha256",
                "um_per_px",
                "source_megapixels",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {
                "task_id": f"P{index:03d}",
                "image_path": str(source_image.resolve()),
                "image_sha256": sha256_file(source_image),
                "um_per_px": "1.0",
                "source_megapixels": "0.001",
            }
            for index in range(283)
        )
    root = _root_exact283(workspace / "root_exact283.json")
    frozen = workspace / "frozen_v1_read_only.lock"
    frozen.write_bytes(b"frozen-v1-read-only")
    official = _write_json(workspace / "official_model_contract.json", {"status": "not_yet_applied"})
    expected_official_sha256 = sha256_file(official)
    release_registry = _write_json(
        workspace / "release" / "RELEASE_AUTHORITY_REGISTRY.json",
        {
            "schema_version": "PHAxis-release-authority-registry-1.0",
            "status": "formal_release_not_yet_materialized_pending_scientific_and_human_authority_gates",
            "public_identity": {
                "product": "PHAxis",
                "version": "1.0.0",
                "distribution": "phaxis",
                "import_namespace": "phaxis",
                "cli": "phaxis",
                "release_tag": "v1.0.0",
            },
            "current_formal_source_release": None,
            "current_formal_release_gate_receipt": None,
            "release_control": {},
            "blind_images_used": 0,
            "updated_utc": "2026-08-30T00:00:00Z",
        },
    )
    external_paths.update(
        {
            "qcdev_manifest": qcdev,
            "locked_val_ids": locked_val,
            "production_manifest": production,
            "root_exact283": root,
            "frozen_v1": frozen,
            "release_authority_registry": release_registry,
        }
    )
    external = {
        name: {
            "path": path.relative_to(workspace).as_posix(),
            "kind": "file",
            "sha256": sha256_file(path),
        }
        for name, path in external_paths.items()
    }

    run_dir = workspace / "release_run"
    checkpoint_refs = [
        {"external": member["checkpoint_input"]} for member in training_members
    ]
    training_receipt_refs = [
        {"external": member["completion_receipt_input"]}
        for member in training_members
    ]
    stages = [
        _stage(
            "candidate_manifest",
            "PHAxis-StageB-train399-candidate-bundle-1.0",
            "candidate_gate_passed_not_promoted",
            [
                {"external": "dataset_audit"},
                *checkpoint_refs,
                *training_receipt_refs,
            ],
            run_dir=run_dir,
            identity_field="candidate_manifest_identity_sha256",
            seals=True,
        ),
        _stage(
            "qcdev_candidate_pool",
            "PHAxis-StageB-train399-QCdev44-candidate-pool-run-1.0",
            "completed",
            [
                {"external": "qcdev_manifest"},
                *checkpoint_refs,
                _input_stage("candidate_manifest"),
            ],
            run_dir=run_dir,
            gpu=_gpu(),
        ),
        _stage(
            "selection",
            "PHAxis-StageB-train399-QCdev44-selection-receipt-1.3",
            "completed",
            [_input_stage("candidate_manifest"), _input_stage("qcdev_candidate_pool")],
            run_dir=run_dir,
            identity_field="selection_receipt_identity_sha256",
            seals=True,
        ),
        _stage(
            "qcdev_evaluation_inference",
            "PHAxis-StageB-train399-QCdev44-evaluation-inference-run-1.0",
            "completed",
            [
                {"external": "qcdev_manifest"},
                {"external": "locked_val_ids"},
                *checkpoint_refs,
                _input_stage("candidate_manifest"),
                _input_stage("selection"),
            ],
            run_dir=run_dir,
            gpu=_gpu(),
        ),
        _stage(
            "qcdev_evaluation",
            "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2",
            "completed",
            [
                _input_stage("candidate_manifest"),
                _input_stage("selection"),
                _input_stage("qcdev_evaluation_inference"),
            ],
            run_dir=run_dir,
        ),
        _stage(
            "proposal",
            "PHAxis-model-contract-1.0.0",
            "passed_proposal_not_official",
            [
                {"external": "root_exact283"},
                *checkpoint_refs,
                _input_stage("candidate_manifest"),
                _input_stage("selection"),
                _input_stage("qcdev_evaluation"),
            ],
            run_dir=run_dir,
            status_field="formal_release_status",
            identity_field="model_contract_identity_sha256",
            seals=True,
        ),
        _stage(
            "authority_pin",
            "PHAxis-run-scoped-model-contract-authority-pin-1.0",
            "sealed_unapplied_proposal_for_production",
            [_input_stage("proposal")],
            run_dir=run_dir,
            identity_field="authority_pin_identity_sha256",
            seals=True,
        ),
        _stage(
            "production_stageb_exact283",
            "PHAxis-StageB-inference-run-1.1",
            "completed",
            [
                {"external": "production_manifest"},
                *checkpoint_refs,
                _input_stage("candidate_manifest"),
                _input_stage("selection"),
                _input_stage("authority_pin"),
            ],
            run_dir=run_dir,
            gpu=_gpu(),
            command_extra=[str(run_dir / "authority_pin" / "receipt.json")],
            identity_field="summary_identity_sha256",
            seals=True,
        ),
        _stage(
            "fusion_exact283",
            "PHAxis-fusion-run-1.1",
            "completed",
            [
                _input_stage("production_stageb_exact283"),
                _input_stage("authority_pin"),
            ],
            run_dir=run_dir,
            identity_field="summary_identity_sha256",
            seals=True,
        ),
        _stage(
            "traits_exact283",
            "PHAxis-trait-export-1.0",
            "completed",
            [_input_stage("fusion_exact283"), _input_stage("authority_pin")],
            run_dir=run_dir,
            identity_field="export_identity_sha256",
            seals=True,
        ),
        _stage(
            "benchmark_same_hardware",
            "PHAxis-same-hardware-benchmark-receipt-1.0",
            "passed",
            [
                {"external": "frozen_v1"},
                _input_stage("production_stageb_exact283"),
                _input_stage("fusion_exact283"),
                _input_stage("traits_exact283"),
            ],
            run_dir=run_dir,
            gpu=_gpu(),
            same_hardware_as="production_stageb_exact283",
        ),
        _stage(
            "figures",
            "PHAxis-publication-figure-suite-1.0",
            "final",
            [
                _input_stage("proposal"),
                _input_stage("traits_exact283"),
                _input_stage("benchmark_same_hardware"),
            ],
            run_dir=run_dir,
        ),
        _stage(
            "evidence",
            "PHAxis-manuscript-release-evidence-graph-1.1",
            "passed_formal_evidence_graph",
            [
                _input_stage("candidate_manifest"),
                _input_stage("selection"),
                _input_stage("qcdev_evaluation"),
                _input_stage("proposal"),
                _input_stage("production_stageb_exact283"),
                _input_stage("fusion_exact283"),
                _input_stage("traits_exact283"),
                _input_stage("figures"),
            ],
            run_dir=run_dir,
            identity_field="manifest_identity_sha256",
            seals=True,
        ),
    ]
    official_inputs = [
        _input_stage(name)
        for name in (
            "proposal",
            "production_stageb_exact283",
            "fusion_exact283",
            "traits_exact283",
            "evidence",
        )
    ]
    official_apply = _stage(
        "official_apply",
        "PHAxis-model-contract-promotion-application-1.0",
        "applied",
        official_inputs,
        run_dir=run_dir,
        identity_field="application_identity_sha256",
        seals=True,
        command_extra=[
            "--apply",
            "--expected-current-sha256",
            expected_official_sha256,
        ],
    )
    official_apply["cas"] = {
        "path": official.relative_to(workspace).as_posix(),
        "expected_sha256": expected_official_sha256,
    }
    stages.extend(
        [
            official_apply,
            _stage(
                "source_release",
                "PHAxis-source-release-manifest-2.0",
                "formal",
                [
                    _input_stage("official_apply"),
                    _input_stage("candidate_manifest"),
                    _input_stage("selection"),
                    _input_stage("qcdev_evaluation"),
                    _input_stage("fusion_exact283"),
                    _input_stage("traits_exact283"),
                ],
                run_dir=run_dir,
                status_field="release_mode",
            ),
            _stage(
                "clean_install",
                "PHAxis-clean-install-verification-1.0",
                "completed_final_clean_install",
                [_input_stage("official_apply"), _input_stage("source_release")],
                run_dir=run_dir,
                gpu=_gpu(),
                command_extra=["--cuda-visible-devices", "1"],
            ),
            _stage(
                "values",
                "PHAxis-manuscript-values-1.2",
                "final_values_machine_derived_locked",
                [
                    _input_stage("evidence"),
                    _input_stage("figures"),
                    _input_stage("clean_install"),
                ],
                run_dir=run_dir,
            ),
            _stage(
                "manuscript",
                "PHAxis-manuscript-compile-receipt-1.2",
                "completed_strict_final_manuscript_compilation",
                [
                    _input_stage("evidence"),
                    _input_stage("figures"),
                    _input_stage("values"),
                ],
                run_dir=run_dir,
            ),
            _stage(
                "handover",
                "PHAxis-reuse-handover-build-receipt-1.0",
                "passed",
                [
                    _input_stage("official_apply"),
                    _input_stage("candidate_manifest"),
                    _input_stage("selection"),
                    _input_stage("qcdev_evaluation"),
                    _input_stage("fusion_exact283"),
                    _input_stage("traits_exact283"),
                    _input_stage("benchmark_same_hardware"),
                    _input_stage("source_release"),
                    _input_stage("clean_install"),
                ],
                run_dir=run_dir,
            ),
            _stage(
                "release_finalize",
                "PHAxis-post-training-release-finalization-1.0",
                "completed_formal_release_closure",
                [
                    _input_stage("official_apply"),
                    _input_stage("source_release"),
                    _input_stage("clean_install"),
                    _input_stage("values"),
                    _input_stage("manuscript"),
                    _input_stage("handover"),
                ],
                run_dir=run_dir,
                identity_field="release_finalization_identity_sha256",
                seals=True,
            ),
        ]
    )
    manifest = {
        "schema_version": "PHAxis-post-training-release-manifest-1.1",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "run_id": "synthetic-post-training-release",
        "workspace": ".",
        "external_inputs": external,
        "training_members": training_members,
        "dataset_audit_input": "dataset_audit",
        "fresh_root_exact283_input": "root_exact283",
        "qcdev_manifest_input": "qcdev_manifest",
        "locked_val_ids_input": "locked_val_ids",
        "production_manifest_input": "production_manifest",
        "frozen_v1_inputs": ["frozen_v1"],
        "stages": stages,
        "frozen_v1_read_only": True,
        "canonical_annotations_read": False,
        "condition_metadata_used_for_routing": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    manifest["manifest_identity_sha256"] = sha256_json(manifest)
    manifest_path = _write_json(workspace / "release_manifest.json", manifest)
    return {
        "workspace": workspace,
        "manifest": manifest_path,
        "manifest_payload": manifest,
        "run_dir": run_dir,
        "preview": preview,
        "root": root,
        "official": official,
        "expected_official_sha256": expected_official_sha256,
        "release_registry": release_registry,
    }


def _proposal_payload(fixture: dict, run_dir: Path) -> dict:
    candidate_path = run_dir / "candidate_manifest" / "receipt.json"
    selection_path = run_dir / "selection" / "receipt.json"
    evaluation_path = run_dir / "qcdev_evaluation" / "receipt.json"
    candidate = read_json(candidate_path)
    selection = read_json(selection_path)
    root = read_json(fixture["root"])
    checkpoints = [
        member["checkpoint_sha256"]
        for member in candidate["identity_payload"]["members"]
    ]
    selected_metadata_identity = "9" * 64
    stageb = {
        "expert_id": "PHAxis-StageB-train399-five-seed",
        "checkpoint_sha256": checkpoints,
        "selected_score_threshold": 0.225,
        "candidate_bundle_identity_sha256": candidate[
            "candidate_bundle_identity_sha256"
        ],
        "selection_receipt_identity_sha256": selection[
            "selection_receipt_identity_sha256"
        ],
        "selected_model_metadata_identity_sha256": selected_metadata_identity,
    }
    public = derive_public_identity(
        stageb,
        root_bundle_identity_sha256=root["bundle_identity_sha256"],
    )
    payload = {
        "schema_version": "PHAxis-model-contract-1.0.0",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "model_bundle_id": public["model_bundle_id"],
        "public_system_identity": {
            "schema_version": PUBLIC_SYSTEM_IDENTITY_SCHEMA,
            "identity_sha256": public["public_system_identity_sha256"],
            "derivation": PUBLIC_SYSTEM_DERIVATION,
        },
        "root_expert": {
            "provider_role": public["root_provider_role"],
            "expert_id": public["root_expert_id"],
            "fresh_exact283_audit_identity_sha256": root["audit_identity_sha256"],
            "bundle_identity_sha256": root["bundle_identity_sha256"],
            "pipeline_identity_sha256": root["pipeline_identity_sha256"],
            "root_bundle_authority": {
                "binding": "transitively_sealed_by_fresh_exact283_pipeline_identity",
                "bundle_identity_sha256": root["bundle_identity_sha256"],
                "pipeline_identity_sha256": root["pipeline_identity_sha256"],
            },
        },
        "expert_boundary": {
            "root_point_scale_continuity_statistics": public["root_expert_id"],
            "hair_identity_and_count": stageb["expert_id"],
        },
        "formal_release_status": "passed_proposal_not_official",
        "hair_identity_count_expert": {
            "current_checkpoint_role": "formal_train399_only_deployment_candidate",
            "deployment_ensemble_used_qcdev44_labels_in_some_members": False,
            "strict_train399_only_retraining_gate": "passed_proposal_not_official",
            "score_threshold": stageb["selected_score_threshold"],
            "checkpoint_sha256_in_member_order": checkpoints,
            "expert_id": stageb["expert_id"],
        },
        "red_lines": {
            "blind_images_used": 0,
            "canonical_annotations_read_during_inference": False,
            "condition_metadata_used_for_routing": False,
            "root_cap_region_statistics_included": False,
        },
        "promotion": {
            "schema_version": "PHAxis-model-contract-promotion-1.0",
            "status": "validated_proposal_not_applied",
            "official_apply_performed": False,
            "source_model_contract_sha256": fixture[
                "expected_official_sha256"
            ],
            "formal_gate_source_sha256": {
                "train399_candidate": sha256_file(candidate_path),
                "train399_selection": sha256_file(selection_path),
                "train399_evaluation": sha256_file(evaluation_path),
                "root_exact283": sha256_file(fixture["root"]),
            },
            "formal_gate_identity_sha256": {
                "candidate_bundle_identity_sha256": stageb[
                    "candidate_bundle_identity_sha256"
                ],
                "selection_receipt_identity_sha256": stageb[
                    "selection_receipt_identity_sha256"
                ],
                "selected_model_metadata_identity_sha256": (
                    selected_metadata_identity
                ),
                "root_exact283_audit_identity_sha256": root[
                    "audit_identity_sha256"
                ],
            },
            "checkpoint_file_sha256_in_member_order": checkpoints,
            "stageb_binding": stageb,
        },
    }
    payload["model_contract_identity_sha256"] = sha256_json(payload)
    return payload


def _apply_official(fixture: dict, run_dir: Path, receipt_path: Path) -> None:
    proposal_path = run_dir / "proposal" / "receipt.json"
    proposal = read_json(proposal_path)
    proposal_file_sha256 = sha256_file(proposal_path)
    official = deepcopy(proposal)
    official.pop("model_contract_identity_sha256")
    official["formal_release_status"] = "passed"
    official["hair_identity_count_expert"]["current_checkpoint_role"] = (
        "formal_train399_only_deployment"
    )
    official["hair_identity_count_expert"][
        "strict_train399_only_retraining_gate"
    ] = "passed"
    public = {
        "model_bundle_id": official["model_bundle_id"],
        "root_expert_id": official["root_expert"]["expert_id"],
    }
    final_paths = {
        "stageb": run_dir / "production_stageb_exact283" / "receipt.json",
        "fusion": run_dir / "fusion_exact283" / "receipt.json",
        "traits": run_dir / "traits_exact283" / "receipt.json",
        "evidence": run_dir / "evidence" / "receipt.json",
    }
    identity_fields = {
        "stageb": "summary_identity_sha256",
        "fusion": "summary_identity_sha256",
        "traits": "export_identity_sha256",
        "evidence": "manifest_identity_sha256",
    }
    promotion = official["promotion"]
    promotion.update(
        {
            "status": "applied_formal_release",
            "official_apply_performed": True,
            "proposal_file_sha256": proposal_file_sha256,
            "proposal_identity_sha256": proposal[
                "model_contract_identity_sha256"
            ],
            "expected_source_model_contract_sha256": fixture[
                "expected_official_sha256"
            ],
            "final_receipt_source_sha256": {
                role: sha256_file(path) for role, path in final_paths.items()
            },
            "final_receipt_identity_sha256": {
                role: read_json(path)[identity_fields[role]]
                for role, path in final_paths.items()
            },
            "final_receipt_public_identity": {
                role: dict(public) for role in ("fusion", "traits")
            },
        }
    )
    official["model_contract_identity_sha256"] = sha256_json(official)
    atomic_write_json(fixture["official"], official)
    receipt = {
        "schema_version": "PHAxis-model-contract-promotion-application-1.0",
        "status": "applied",
        "official_model_contract_replaced": True,
        "expected_previous_model_contract_sha256": fixture[
            "expected_official_sha256"
        ],
        "proposal_file_sha256": proposal_file_sha256,
        "proposal_identity_sha256": proposal[
            "model_contract_identity_sha256"
        ],
        "final_model_contract_sha256": sha256_file(fixture["official"]),
        "final_model_contract_identity_sha256": official[
            "model_contract_identity_sha256"
        ],
        "final_evidence_manifest_sha256": promotion[
            "final_receipt_source_sha256"
        ]["evidence"],
        "final_evidence_manifest_identity_sha256": promotion[
            "final_receipt_identity_sha256"
        ]["evidence"],
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }
    receipt["application_identity_sha256"] = sha256_json(receipt)
    _write_json(receipt_path, receipt)


def _runner(fixture: dict, events: list[str], fail_stage: str | None = None):
    def run(*, command: list[str], cwd: Path, env: dict[str, str]):
        del cwd
        stage = command[1]
        events.append(f"run:{stage}")
        if stage in {
            "qcdev_candidate_pool",
            "qcdev_evaluation_inference",
            "production_stageb_exact283",
            "benchmark_same_hardware",
        }:
            assert env["CUDA_VISIBLE_DEVICES"] == "1"
        if stage == fail_stage:
            return SimpleNamespace(returncode=17, stdout="", stderr="synthetic failure")
        receipt_path = fixture["run_dir"] / stage / "receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        if stage == "candidate_manifest":
            payload = fixture["preview"]
        elif stage == "qcdev_candidate_pool":
            payload = {
                "schema_version": "PHAxis-StageB-train399-QCdev44-candidate-pool-run-1.0",
                "status": "completed",
                "images": 44,
                "resumed_images": 0,
                "checkpoint_sha256": [
                    member["checkpoint_sha256"]
                    for member in fixture["preview"]["identity_payload"]["members"]
                ],
                "blind_images_used": 0,
            }
        elif stage == "selection":
            payload = {
                "schema_version": "PHAxis-StageB-train399-QCdev44-selection-receipt-1.3",
                "status": "completed",
                "images": 44,
                "independent_accuracy_claim_allowed": False,
                "blind_images_used": 0,
            }
            payload["selection_receipt_identity_sha256"] = sha256_json(payload)
        elif stage == "qcdev_evaluation_inference":
            payload = {
                "schema_version": "PHAxis-StageB-train399-QCdev44-evaluation-inference-run-1.0",
                "status": "completed",
                "images": 44,
                "resumed_images": 0,
                "production_consumption_allowed": False,
                "fusion_consumption_allowed": False,
                "traits_consumption_allowed": False,
                "model_contract_proposal_present": False,
                "canonical_annotations_read_during_inference": False,
                "condition_metadata_used_for_routing": False,
                "blind_images_used": 0,
            }
        elif stage == "qcdev_evaluation":
            payload = {
                "schema_version": "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2",
                "status": "completed",
                "independent_accuracy_claim_allowed": False,
                "overall": {"stageb_train399": {"images": 44}},
                "training_contract": {
                    "checkpoint_sha256": [
                        member["checkpoint_sha256"]
                        for member in fixture["preview"]["identity_payload"]["members"]
                    ]
                },
                "blind_images_used": 0,
            }
        elif stage == "proposal":
            payload = _proposal_payload(fixture, fixture["run_dir"])
        elif stage in {
            "production_stageb_exact283",
            "fusion_exact283",
            "traits_exact283",
        }:
            pin = read_model_contract_authority(
                fixture["run_dir"] / "authority_pin" / "receipt.json"
            )
            payload = {
                "schema_version": {
                    "production_stageb_exact283": "PHAxis-StageB-inference-run-1.1",
                    "fusion_exact283": "PHAxis-fusion-run-1.1",
                    "traits_exact283": "PHAxis-trait-export-1.0",
                }[stage],
                "status": "completed",
                "model_bundle_id": pin.model_bundle_id,
                "model_contract_proposal_sha256": pin.file_sha256,
                "model_contract_proposal_identity_sha256": pin.identity_sha256,
                "blind_images_used": 0,
            }
            if stage == "production_stageb_exact283":
                payload.update(
                    {
                        "images": 283,
                        "resumed_images": 0,
                        "records": [
                            {"task_id": f"P{index:03d}", "resumed": False}
                            for index in range(283)
                        ],
                        "checkpoint_sha256": list(
                            pin.stageb_binding["checkpoint_sha256"]
                        ),
                        "root_expert_id": pin.root_expert_id,
                    }
                )
            elif stage == "fusion_exact283":
                payload.update({"images": 283, "root_expert": pin.root_expert_id})
            else:
                payload.update({"tasks": 283, "root_expert_id": pin.root_expert_id})
            identity_field = (
                "export_identity_sha256"
                if stage == "traits_exact283"
                else "summary_identity_sha256"
            )
            payload[identity_field] = sha256_json(payload)
        elif stage == "benchmark_same_hardware":
            payload = {
                "schema_version": "PHAxis-same-hardware-benchmark-receipt-1.0",
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
                "runs": [
                    {
                        "role": role,
                        "fresh_direct_run": True,
                        "resume_or_cache_used": False,
                        "full_workflow_io_included": True,
                    }
                    for role in (
                        "phaxis_production",
                        "phaxis_sequential",
                        "frozen_v1_production",
                        "frozen_v1_sequential",
                    )
                ],
                "blind_images_used": 0,
            }
        elif stage == "evidence":
            payload = {
                "schema_version": "PHAxis-manuscript-release-evidence-graph-1.1",
                "status": "passed_formal_evidence_graph",
                "blind_images_used": 0,
            }
            payload["manifest_identity_sha256"] = sha256_json(payload)
        elif stage == "figures":
            payload = {
                "schema_version": "PHAxis-publication-figure-suite-1.0",
                "status": "final",
                "blind_images_used": 0,
            }
        elif stage == "values":
            source_manifest_path = (
                fixture["run_dir"] / "source_release" / "receipt.json"
            )
            source_manifest = read_json(source_manifest_path)
            metadata_path = source_manifest_path.parent / "RELEASE_HUMAN_METADATA.json"
            metadata = read_json(metadata_path)
            source_records = {
                record["path"]: record for record in source_manifest["files"]
            }
            cross_binding = {
                "repository_url": metadata["project_urls"]["Repository"],
                "release_tag": metadata["release_coordinates"][
                    "github_release_tag"
                ],
                "version": source_manifest["version"],
                "release_doi": metadata["release_coordinates"]["release_doi"],
                "software_license": metadata["rights"]["source_license_spdx"],
                "source_release_tree_identity_sha256": source_manifest[
                    "tree_identity_sha256"
                ],
                "source_release_manifest_sha256": sha256_file(
                    source_manifest_path
                ),
                "release_metadata_identity_sha256": metadata[
                    "metadata_identity_sha256"
                ],
                "release_metadata_sha256": sha256_file(metadata_path),
                "license_file_sha256": metadata["rights"][
                    "license_file_sha256"
                ],
                "pyproject_sha256": source_records["pyproject.toml"]["sha256"],
                "citation_cff_sha256": source_records["CITATION.cff"]["sha256"],
            }
            figure_inputs = read_json(
                fixture["run_dir"] / "figure_inputs" / "figure_inputs.json"
            )
            figures = read_json(
                fixture["run_dir"] / "figures" / "receipt.json"
            )
            payload = {
                "schema_version": "PHAxis-manuscript-values-1.2",
                "builder_schema_version": (
                    "PHAxis-manuscript-values-builder-1.1"
                ),
                "status": "final_values_machine_derived_locked",
                "source_release_manifest_file_sha256": sha256_file(
                    source_manifest_path
                ),
                "source_release_tree_identity_sha256": source_manifest[
                    "tree_identity_sha256"
                ],
                "source_release_metadata_file_sha256": sha256_file(metadata_path),
                "source_release_metadata_identity_sha256": metadata[
                    "metadata_identity_sha256"
                ],
                "software_release_cross_binding_identity_sha256": sha256_json(
                    cross_binding
                ),
                "source_files": {
                    "source_release_manifest": {
                        "sha256": sha256_file(source_manifest_path),
                        "logical_identity_sha256": source_manifest[
                            "tree_identity_sha256"
                        ],
                    },
                    "source_release_metadata": {
                        "sha256": sha256_file(metadata_path),
                        "logical_identity_sha256": metadata[
                            "metadata_identity_sha256"
                        ],
                    },
                },
                "narrative_decision_identity_sha256": figure_inputs[
                    "narrative_decision_identity_sha256"
                ],
                "narrative_branch_id": figure_inputs["narrative_branch_id"],
                "publication_title_contract": figures["title_contract"],
                "blind_images_used": 0,
            }
            payload["values_identity_sha256"] = sha256_json(payload)
        elif stage == "handover":
            payload = {
                "schema_version": "PHAxis-reuse-handover-build-receipt-1.0",
                "status": "passed",
                "blind_images_used": 0,
            }
        elif stage == "source_release":
            license_path = receipt_path.parent / "LICENSE"
            license_path.write_text(
                "Synthetic Apache-2.0 license\n", encoding="utf-8"
            )
            pyproject_path = receipt_path.parent / "pyproject.toml"
            pyproject_path.write_text(
                '[project]\nname = "phaxis"\nversion = "1.0.0"\n'
                'license = "Apache-2.0"\n[project.urls]\n'
                'Repository = "https://github.com/example/phaxis"\n',
                encoding="utf-8",
            )
            citation_path = receipt_path.parent / "CITATION.cff"
            citation_path.write_text(
                'cff-version: "1.2.0"\nversion: "1.0.0"\n'
                'license: "Apache-2.0"\n'
                'repository-code: "https://github.com/example/phaxis"\n'
                'doi: "10.5281/zenodo.1234567"\n',
                encoding="utf-8",
            )
            metadata = {
                "schema_version": "PHAxis-release-human-metadata-1.3",
                "status": "author_verified_release_authority",
                "product": "PHAxis",
                "product_version": "1.0.0",
                "distribution": "phaxis",
                "project_urls": {
                    "Repository": "https://github.com/example/phaxis"
                },
                "release_coordinates": {
                    "github_repository_url": "https://github.com/example/phaxis",
                    "github_release_tag": "v1.0.0",
                    "github_release_url": (
                        "https://github.com/example/phaxis/releases/tag/v1.0.0"
                    ),
                    "pypi_project": "phaxis",
                    "pypi_version": "1.0.0",
                    "pypi_project_url": "https://pypi.org/project/phaxis/1.0.0/",
                    "release_date": "2026-08-30",
                    "release_doi": "10.5281/zenodo.1234567",
                },
                "rights": {
                    "source_license_spdx": "Apache-2.0",
                    "source_release_authorized": True,
                    "license_file_sha256": sha256_file(license_path),
                },
                "blind_images_used": 0,
            }
            metadata["metadata_identity_sha256"] = sha256_json(metadata)
            metadata_path = receipt_path.parent / "RELEASE_HUMAN_METADATA.json"
            _write_json(metadata_path, metadata)
            files = sorted(
                [
                    {
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in (
                        metadata_path,
                        license_path,
                        pyproject_path,
                        citation_path,
                    )
                ],
                key=lambda record: record["path"],
            )
            payload = {
                "schema_version": "PHAxis-source-release-manifest-2.0",
                "release_mode": "formal",
                "distribution": "phaxis",
                "version": "1.0.0",
                "files": files,
                "tree_identity_sha256": sha256_json(files),
                "blind_images_used": 0,
            }
            gate = {
                "schema_version": "PHAxis-formal-release-gate-1.0",
                "status": "passed",
                "formal_release_allowed": True,
                "release_mode": "formal",
                "release_human_metadata": metadata,
                "blind_images_used": 0,
            }
            _write_json(
                receipt_path.parent / "FORMAL_RELEASE_GATE_RECEIPT.json", gate
            )
        elif stage == "clean_install":
            formal_wheel = (
                fixture["run_dir"]
                / "distributions/phaxis-1.0.0-py3-none-any.whl"
            )
            source_manifest = (
                fixture["run_dir"] / "source_release/receipt.json"
            )
            payload = {
                "schema_version": "PHAxis-clean-install-verification-1.0",
                "status": "completed_final_clean_install",
                "source_release_manifest_sha256": sha256_file(source_manifest),
                "formal_wheel": {
                    "sha256": sha256_file(formal_wheel),
                    "record_verified": True,
                    "source_package_hashes_verified": True,
                    "metadata_license_files": [
                        "LICENSE",
                        "src/phaxis/_vendor/tomli/LICENSE.txt",
                    ],
                    "pep639_license_member_count": 2,
                    "license_file_hashes_verified": True,
                },
                "blind_images_used": 0,
            }
        elif stage == "manuscript":
            payload = {
                "schema_version": "PHAxis-manuscript-compile-receipt-1.2",
                "status": "completed_strict_final_manuscript_compilation",
                "blind_images_used": 0,
            }
        elif stage == "official_apply":
            _apply_official(fixture, fixture["run_dir"], receipt_path)
            return SimpleNamespace(returncode=0, stdout="applied", stderr="")
        else:  # pragma: no cover - fixture and mandatory order must agree
            raise AssertionError(stage)
        _write_json(receipt_path, payload)
        return SimpleNamespace(returncode=0, stdout=stage, stderr="")

    return run


def _probe(events: list[str]):
    def probe(*, stage: str):
        events.append(f"probe:{stage}")
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "1, GPU-SYNTHETIC-1, NVIDIA RTX 3090, "
                "24576, 1024, 10, 999.0\n"
            ),
            stderr="",
        )

    return probe


def _cached_builder(preview: dict):
    def builder(_checkpoints, *, dataset_audit_path):
        assert Path(dataset_audit_path).is_file()
        return deepcopy(preview)

    return builder


# Schema 1.2 fixture: every release-derived authority is produced by an
# explicit stage.  The older fixture above is retained as useful construction
# scaffolding, then rewritten here before any test consumes it.
_manifest_fixture_v11 = _manifest_fixture
_runner_v11 = _runner


def _manifest_fixture(tmp_path: Path) -> dict:
    fixture = _manifest_fixture_v11(tmp_path)
    payload = fixture["manifest_payload"]
    workspace = fixture["workspace"]
    run_dir = fixture["run_dir"]
    release_registry = fixture["release_registry"]

    # Root exact283 and the normalized production manifest are producer
    # outputs in 1.2, not external inputs required before plan construction.
    fixture["root_source"] = fixture["root"]
    fixture["production_source"] = (
        workspace / payload["external_inputs"]["production_manifest"]["path"]
    )
    fixture["root"] = run_dir / "root_provider_exact283" / "receipt.json"
    payload["external_inputs"].pop("root_exact283")
    payload["external_inputs"].pop("production_manifest")
    payload.pop("fresh_root_exact283_input")
    payload.pop("production_manifest_input")

    training_names = {
        member[key]
        for member in payload["training_members"]
        for key in ("completion_receipt_input", "checkpoint_input")
    }
    for name, spec in payload["external_inputs"].items():
        if name in training_names:
            spec["authority_class"] = "completed_training_authority"
        elif name == "frozen_v1":
            spec["authority_class"] = "frozen_read_only_asset"
        elif name in {"dataset_audit", "qcdev_manifest", "locked_val_ids"}:
            spec["authority_class"] = "immutable_raw_data"
        else:
            spec["authority_class"] = "static_contract"

    checkpoint_refs = [
        {"external": member["checkpoint_input"]}
        for member in payload["training_members"]
    ]
    training_receipt_refs = [
        {"external": member["completion_receipt_input"]}
        for member in payload["training_members"]
    ]
    schema_status: dict[str, tuple[str, object]] = {
        "candidate_manifest": ("PHAxis-StageB-train399-candidate-bundle-1.0", "candidate_gate_passed_not_promoted"),
        "production_manifest": ("PHAxis-production-manifest-1.0", "completed"),
        "direct_benchmark_provider_descriptor": ("PHAxis-formal-direct-benchmark-provider-descriptor-1.0", "ready_hash_locked_direct_execution"),
        "release_case_prelocks": ("PHAxis-release-case-prelocks-1.0", "completed_result_independent_exact283_case_prelocks"),
        "qcdev_candidate_pool": ("PHAxis-StageB-train399-QCdev44-candidate-pool-run-1.0", "completed"),
        "selection": (KNOWN_STAGE_SCHEMAS["selection"], "completed"),
        "qcdev_evaluation_inference": ("PHAxis-StageB-train399-QCdev44-evaluation-inference-run-1.0", "completed"),
        "qcdev_evaluation": ("PHAxis-StageB-train399-QCdev44-development-evaluation-1.2", "completed"),
        "root_provider_exact283": ("PHAxis-root-provider-fresh-reference283-audit-1.0", "pass_exact_283"),
        "root_bundle_materialization": ("PHAxis-root-provider-model-bundle-verification-1.0", "pass"),
        "proposal": ("PHAxis-model-contract-1.0.0", "passed_proposal_not_official"),
        "authority_pin": ("PHAxis-run-scoped-model-contract-authority-pin-1.0", "sealed_unapplied_proposal_for_production"),
        "analysis_workflow_manifest": ("PHAxis-analysis-workflow-manifest-1.0", "ready_hash_locked_full_workflow"),
        "clean_install_sample_manifest": ("PHAxis-clean-install-sample-input-suite-1.0", "completed_real_nonblind_release_example_manifest"),
        "qcdev_root_inputs": ("PHAxis-QCdevelopment44-root-provider-input-suite-1.0", "completed_locked_exact44_label_free_source_contract"),
        "qcdev_root_provider": ("PHAxis-root-provider-portable-pipeline-1.0", "completed_uncompared"),
        "qcdev_fusion": ("PHAxis-fusion-run-1.1", "completed"),
        "production_stageb_exact283": ("PHAxis-StageB-inference-run-1.1", "completed"),
        "fusion_exact283": ("PHAxis-fusion-run-1.1", "completed"),
        "figure1_geometry_materialization": ("PHAxis-figure1-geometry-materialization-1.0", "completed_from_preselected_case_and_final_prediction"),
        "traits_exact283": ("PHAxis-trait-export-1.0", "completed"),
        "cohorts_exact283": ("PHAxis-biological-cohorts-1.0", "completed_without_fitting_biological_effect_models"),
        "biological_analysis": ("PHAxis-exploratory-biological-analysis-1.0", "completed_exploratory_clean_primary_full_sensitivity"),
        "profiles_exact283": ("PHAxis-distal-axis-cohort-profile-bundle-1.0.0", "completed"),
        "profile_analysis": ("PHAxis-distal-axis-profile-analysis-1.0.0", "completed_exploratory_source_unit_profile_summaries"),
        "historical_oof_evidence": ("PHAxis-historical-OOF443-development-receipt-1.0", "completed_locked_historical_oof443_development"),
        "measurement_assurance": ("PHAxis-measurement-assurance-receipt-1.0", "completed_locked_qc_development_assurance"),
        "overlay_evidence": (
            "PHAxis-manuscript-overlay-selection-receipt-1.2",
            "completed_locked_preselected_gallery_and_exact_cohort_review_export",
        ),
        "benchmark_phaxis_production": ("PHAxis-full-workflow-production-batch-benchmark-1.0", "completed_direct_full283"),
        "benchmark_frozen_v1_production": ("PHAxis-full-workflow-production-batch-benchmark-1.0", "completed_direct_full283"),
        "benchmark_phaxis_sequential": ("PHAxis-full-workflow-sequential-latency-benchmark-1.0", "completed_direct_full283"),
        "benchmark_frozen_v1_sequential": ("PHAxis-full-workflow-sequential-latency-benchmark-1.0", "completed_direct_full283"),
        "benchmark_production_comparison": ("PHAxis-full-workflow-benchmark-comparison-1.0", "comparable_direct_full283"),
        "benchmark_sequential_comparison": ("PHAxis-full-workflow-benchmark-comparison-1.0", "comparable_direct_full283"),
        "benchmark_same_hardware": ("PHAxis-same-hardware-benchmark-receipt-1.0", "passed"),
        "benchmark_artifact_inventory": ("PHAxis-benchmark-artifact-inventory-1.0", "completed_explicit_benchmark_inventory"),
        "figure_inputs": ("PHAxis-publication-figure-input-assembly-1.0", "completed_final"),
        "figures": ("PHAxis-publication-figure-suite-1.0", "final_sealed_strict_train399_only"),
        "evidence": ("PHAxis-manuscript-release-evidence-graph-1.1", "passed_formal_evidence_graph"),
        "official_apply": ("PHAxis-model-contract-promotion-application-1.0", "applied"),
        "source_release": ("PHAxis-source-release-manifest-2.0", "formal"),
        "distributions": ("PHAxis-release-distributions-1.0", "completed_wheel_sdist_verified"),
        "offline_dependencies": ("PHAxis-offline-dependency-materialization-1.0", "completed_locked_cp312_win_amd64"),
        "handover_dataset_manifest": ("PHAxis-handover-materialisation-plan-1.0", "created"),
        "handover_image_manifest": ("PHAxis-handover-materialisation-plan-1.0", "created"),
        "handover_model_source_manifest": ("PHAxis-handover-materialisation-plan-1.0", "created"),
        "handover_model_asset_manifest": ("PHAxis-handover-materialisation-plan-1.0", "created"),
        "clean_install_expected_identity": ("PHAxis-clean-install-reference-output-1.0", "completed_fresh_real_nonblind_reference"),
        "handover_benchmark_manifest": ("PHAxis-handover-materialisation-plan-1.0", "created"),
        "clean_install": ("PHAxis-clean-install-verification-1.0", "completed_final_clean_install"),
        "values": ("PHAxis-manuscript-values-1.2", "final_values_machine_derived_locked"),
        "manuscript": ("PHAxis-manuscript-compile-receipt-1.2", "completed_strict_final_manuscript_compilation"),
        "supplementary_manuscript": ("PHAxis-supplementary-manuscript-compile-receipt-1.0", "completed_strict_final_supplementary_compilation"),
        "submission_docx": ("PHAxis-submission-docx-build-2.0", "completed_final_double_anonymous_submission_bundle"),
        "supplementary_docx": ("PHAxis-supplementary-docx-build-2.0", "completed_final_anonymized_supplementary_docx"),
        "manuscript_artifact_qa": ("PHAxis-manuscript-artifact-structural-qa-2.0", "passed_double_anonymous_three_role_ooxml_closure"),
        "manuscript_render": ("PHAxis-manuscript-pdf-page-render-2.0", "completed_three_role_word_pdf_and_page_png_render"),
        "manuscript_visual_qa": ("PHAxis-manuscript-human-visual-qa-receipt-2.0", "passed_author_verified_three_role_page_visual_qa"),
        "handover_contract": ("PHAxis-handover-build-contract-assembly-report-1.0", "created"),
        "handover": ("PHAxis-reuse-handover-build-receipt-1.0", "passed"),
        "release_finalize": ("PHAxis-post-training-release-finalization-1.0", "completed_formal_release_closure"),
    }
    identity_fields = {
        "candidate_manifest": "candidate_manifest_identity_sha256",
        "direct_benchmark_provider_descriptor": "descriptor_identity_sha256",
        "release_case_prelocks": "case_prelock_identity_sha256",
        "selection": "selection_receipt_identity_sha256",
        "proposal": "model_contract_identity_sha256",
        "authority_pin": "authority_pin_identity_sha256",
        "analysis_workflow_manifest": "manifest_identity_sha256",
        "clean_install_sample_manifest": "sample_input_suite_identity_sha256",
        "qcdev_root_inputs": "summary_identity_sha256",
        "production_stageb_exact283": "summary_identity_sha256",
        "fusion_exact283": "summary_identity_sha256",
        "figure1_geometry_materialization": "figure1_geometry_materialization_identity_sha256",
        "traits_exact283": "export_identity_sha256",
        "profiles_exact283": "cohort_profile_bundle_identity_sha256",
        "evidence": "manifest_identity_sha256",
        "official_apply": "application_identity_sha256",
        "offline_dependencies": "dependency_materialization_identity_sha256",
        "clean_install_expected_identity": "reference_output_identity_sha256",
        "values": "values_identity_sha256",
        "manuscript": "receipt_identity_sha256",
        "supplementary_manuscript": "receipt_identity_sha256",
        "submission_docx": "receipt_identity_sha256",
        "supplementary_docx": "receipt_identity_sha256",
        "manuscript_artifact_qa": "qa_identity_sha256",
        "manuscript_render": "render_identity_sha256",
        "manuscript_visual_qa": "visual_qa_identity_sha256",
        "release_finalize": "release_finalization_identity_sha256",
    }
    gpu_names = {
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
    stages: list[dict] = []
    for name in MANDATORY_STAGE_ORDER:
        schema, status = schema_status[name]
        inputs = [_input_stage(dep) for dep in STAGE_DEPENDENCIES[name]]
        if name == "candidate_manifest":
            inputs = [
                {"external": "dataset_audit"},
                *checkpoint_refs,
                *training_receipt_refs,
            ]
        elif name == "qcdev_candidate_pool":
            inputs.extend([{"external": "qcdev_manifest"}, *checkpoint_refs])
        elif name == "qcdev_evaluation_inference":
            inputs.extend(
                [
                    {"external": "qcdev_manifest"},
                    {"external": "locked_val_ids"},
                    *checkpoint_refs,
                ]
            )
        elif name in {"proposal", "production_stageb_exact283"}:
            inputs.extend(checkpoint_refs)
        elif name in {
            "benchmark_frozen_v1_production",
            "benchmark_frozen_v1_sequential",
        }:
            inputs.append({"external": "frozen_v1"})
        elif name == "release_finalize":
            inputs.append({"external": "release_authority_registry"})
        status_field = (
            "formal_release_status"
            if name == "proposal"
            else "release_mode" if name == "source_release" else "status"
        )
        command_extra: list[str] = []
        if name == "production_stageb_exact283":
            command_extra.append(str(run_dir / "authority_pin" / "receipt.json"))
        if name in {"root_provider_exact283", "qcdev_root_provider"}:
            command_extra.extend(
                [
                    "--v1-physical-gpu",
                    "1",
                    "--q8-physical-gpu",
                    "1",
                    "--strict-physical-gpu",
                ]
            )
        if name in {
            "benchmark_phaxis_production",
            "benchmark_frozen_v1_production",
            "benchmark_phaxis_sequential",
            "benchmark_frozen_v1_sequential",
            "clean_install_expected_identity",
            "clean_install",
        }:
            command_extra.extend(["--cuda-visible-devices", "1"])
        stage = _stage(
            name,
            schema,
            status,
            inputs,
            run_dir=run_dir,
            status_field=status_field,
            identity_field=identity_fields.get(name),
            seals=name in identity_fields,
            gpu=_gpu() if name in gpu_names else None,
            command_extra=command_extra,
        )
        if name in {
            "root_provider_exact283",
            "qcdev_root_provider",
            "benchmark_phaxis_production",
            "benchmark_frozen_v1_production",
            "benchmark_phaxis_sequential",
            "benchmark_frozen_v1_sequential",
            "clean_install_expected_identity",
            "clean_install",
        }:
            stage["environment"] = {
                "PHAXIS_REQUIRE_EXACT_PHYSICAL_GPU": "1"
            }
        if name == "production_manifest":
            stage["artifacts"].append(
                {
                    "name": "manifest_all",
                    "path": str(run_dir / name / "manifest_all.csv"),
                    "kind": "file",
                }
            )
        if name == "profiles_exact283":
            stage["artifacts"].extend(
                [
                    {
                        "name": "primary_summary",
                        "path": str(run_dir / name / "primary_summary.json"),
                        "kind": "file",
                    },
                    {
                        "name": "primary_profiles",
                        "path": str(run_dir / name / "primary_profiles.csv"),
                        "kind": "file",
                    },
                    {
                        "name": "sensitivity_summary",
                        "path": str(run_dir / name / "sensitivity_summary.json"),
                        "kind": "file",
                    },
                    {
                        "name": "sensitivity_profiles",
                        "path": str(run_dir / name / "sensitivity_profiles.csv"),
                        "kind": "file",
                    },
                ]
            )
        if name == "figure_inputs":
            stage["artifacts"].append(
                {
                    "name": "manifest",
                    "path": str(run_dir / name / "figure_inputs.json"),
                    "kind": "file",
                }
            )
        if name == "submission_docx":
            stage["artifacts"].extend(
                [
                    {
                        "name": "title_page",
                        "path": str(run_dir / name / "title_page.docx"),
                        "kind": "file",
                    },
                    {
                        "name": "anonymized_main",
                        "path": str(run_dir / name / "anonymized_main.docx"),
                        "kind": "file",
                    },
                ]
            )
        if name == "supplementary_docx":
            stage["artifacts"].append(
                {
                    "name": "anonymized_supplement",
                    "path": str(run_dir / name / "anonymized_supplement.docx"),
                    "kind": "file",
                }
            )
        if name == "manuscript_artifact_qa":
            stage["artifacts"].append(
                {
                    "name": "upload_manifest",
                    "path": str(run_dir / name / "upload-role-manifest.json"),
                    "kind": "file",
                }
            )
        if name in {"root_provider_exact283", "qcdev_root_provider"}:
            stage["artifacts"].append(
                {
                    "name": "output",
                    "path": str(run_dir / name / "output"),
                    "kind": "directory",
                }
            )
        if name in {
            "benchmark_phaxis_production",
            "benchmark_frozen_v1_production",
            "benchmark_phaxis_sequential",
            "benchmark_frozen_v1_sequential",
        }:
            for artifact_name in ("gpu_telemetry", "hardware_preflight"):
                stage["artifacts"].append(
                    {
                        "name": artifact_name,
                        "path": str(run_dir / name / f"{artifact_name}.json"),
                        "kind": "file",
                    }
                )
        if name == "distributions":
            stage["artifacts"].append(
                {
                    "name": "wheel",
                    "path": str(
                        run_dir
                        / name
                        / "phaxis-1.0.0-py3-none-any.whl"
                    ),
                    "kind": "file",
                }
            )
        if name == "offline_dependencies":
            stage["artifacts"].extend(
                [
                    {
                        "name": "output",
                        "path": str(run_dir / name),
                        "kind": "directory",
                    },
                    {
                        "name": "dependency_lock",
                        "path": str(run_dir / name / "requirements.lock.txt"),
                        "kind": "file",
                    },
                    {
                        "name": "wheelhouse",
                        "path": str(run_dir / name / "wheelhouse"),
                        "kind": "directory",
                    },
                    {
                        "name": "resolved_sbom",
                        "path": str(
                            run_dir
                            / name
                            / dependency_materializer.RESOLVED_SBOM_NAME
                        ),
                        "kind": "file",
                    },
                    {
                        "name": "resolved_license_inventory",
                        "path": str(
                            run_dir
                            / name
                            / dependency_materializer.RESOLVED_LICENSE_INVENTORY_NAME
                        ),
                        "kind": "file",
                    },
                ]
            )
        if name == "release_finalize":
            stage["release_registry_cas"] = {
                "external": "release_authority_registry",
                "path": release_registry.relative_to(workspace).as_posix(),
                "expected_sha256": sha256_file(release_registry),
            }
        stages.append(stage)

    official = next(stage for stage in stages if stage["name"] == "official_apply")
    official["command"].extend(
        ["--apply", "--expected-current-sha256", fixture["expected_official_sha256"]]
    )
    official["cas"] = {
        "path": fixture["official"].relative_to(workspace).as_posix(),
        "expected_sha256": fixture["expected_official_sha256"],
    }
    payload.update(
        {
            "schema_version": release_orchestrator_module.MANIFEST_SCHEMA,
            "stages": stages,
        }
    )
    payload.pop("manifest_identity_sha256", None)
    payload["manifest_identity_sha256"] = sha256_json(payload)
    _write_json(fixture["manifest"], payload)
    fixture["manifest_payload"] = payload
    fixture["release_registry"] = release_registry
    fixture["expected_release_registry_sha256"] = sha256_file(release_registry)
    return fixture


def _runner(fixture: dict, events: list[str], fail_stage: str | None = None):
    legacy = _runner_v11(fixture, events, fail_stage=fail_stage)
    legacy_names = {
        "candidate_manifest",
        "qcdev_candidate_pool",
        "selection",
        "qcdev_evaluation_inference",
        "qcdev_evaluation",
        "proposal",
        "production_stageb_exact283",
        "fusion_exact283",
        "traits_exact283",
        "official_apply",
        "source_release",
        "clean_install",
        "values",
        "handover",
    }

    def run(*, command: list[str], cwd: Path, env: dict[str, str]):
        stage = command[1]
        if stage in legacy_names:
            return legacy(command=command, cwd=cwd, env=env)
        events.append(f"run:{stage}")
        if stage == fail_stage:
            return SimpleNamespace(returncode=17, stdout="", stderr="synthetic failure")
        if stage in {
            "root_provider_exact283",
            "qcdev_root_provider",
            "benchmark_phaxis_production",
            "benchmark_frozen_v1_production",
            "benchmark_phaxis_sequential",
            "benchmark_frozen_v1_sequential",
            "clean_install_expected_identity",
            "clean_install",
        }:
            assert env["CUDA_VISIBLE_DEVICES"] == "1"
            assert env["PHAXIS_REQUIRE_EXACT_PHYSICAL_GPU"] == "1"
        receipt_path = fixture["run_dir"] / stage / "receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        spec = next(
            item for item in fixture["manifest_payload"]["stages"] if item["name"] == stage
        )["receipt"]
        if stage in {"root_provider_exact283", "qcdev_root_provider"}:
            binding = {
                "schema_version": "PHAxis-Q8-shard-device-binding-1.0",
                "status": "passed_before_q8_merge",
                "exact_physical_gpu_required": True,
                "planned_physical_gpus": [1],
                "shards": 1,
                "records": [
                    {
                        "shard_index": 0,
                        "planned_physical_gpu": 1,
                        "requested_physical_gpu": 1,
                        "selected_physical_gpu": 1,
                        "physical_gpu_uuid": "GPU-SYNTHETIC-1",
                        "selection_receipt": "synthetic",
                        "selection_receipt_sha256": "b" * 64,
                    }
                ],
                "requested_equals_selected_equals_planned": True,
                "uuid_bound_to_each_selection_receipt": True,
                "merge_started": False,
                "blind_images_used": 0,
            }
            binding["binding_identity_sha256"] = sha256_json(binding)
            _write_json(
                fixture["run_dir"]
                / stage
                / "output"
                / "q8_shards"
                / "exact_device_binding.json",
                binding,
            )
        if stage == "production_manifest":
            manifest_path = fixture["run_dir"] / stage / "manifest_all.csv"
            manifest_path.write_bytes(fixture["production_source"].read_bytes())
            payload = {
                "schema_version": spec["schema_version"],
                "status": "completed",
                "images": 283,
                "manifest_all_sha256": sha256_file(manifest_path),
                "blind_images_used": 0,
            }
        elif stage == "root_provider_exact283":
            payload = read_json(fixture["root_source"])
        elif stage == "root_bundle_materialization":
            payload = {
                "schema_version": spec["schema_version"],
                "status": "pass",
                "materialized_exact_closure": True,
                "exact_file_closure_required": True,
                "exact_file_closure_passed": True,
                "unlisted_file_count": 0,
                "missing_closure_file_count": 0,
                "files_verified": 313,
                "bytes_verified": 1024,
                "source_bundle_mutated": False,
                "blind_images_used": 0,
            }
        elif stage == "qcdev_root_provider":
            payload = {
                "schema_version": spec["schema_version"],
                "status": "completed_uncompared",
                "canonical_annotations_read": False,
                "blind_images_used": 0,
            }
        elif stage == "analysis_workflow_manifest":
            payload = {
                "schema_version": spec["schema_version"],
                "status": "ready_hash_locked_full_workflow",
                "guards": {
                    "blind_images_used": 0,
                    "canonical_annotations_read": False,
                    "condition_metadata_used_for_routing": False,
                },
                "benchmark_contract": {
                    "warmup_runs": 0,
                    "measured_repeats": 1,
                },
            }
        elif stage == "qcdev_root_inputs":
            payload = {
                "schema_version": spec["schema_version"],
                "status": "completed_locked_exact44_label_free_source_contract",
                "tasks": 44,
                "labels_or_annotation_files_read": False,
                "locked_members_posthoc_filtered": False,
                "acquisition_gate_can_remove_locked_member": False,
                "condition_metadata_used_for_routing": False,
                "canonical_annotations_read": False,
                "blind_images_used": 0,
            }
        elif stage == "qcdev_fusion":
            payload = {
                "schema_version": spec["schema_version"],
                "status": "completed",
                "images": 44,
                "blind_images_used": 0,
            }
        elif stage == "cohorts_exact283":
            payload = {
                "schema_version": spec["schema_version"],
                "status": "completed_without_fitting_biological_effect_models",
                "cohort_directories": {
                    "primary": "primary_clean261",
                    "sensitivity": "sensitivity_full283",
                },
                "counts": {
                    "human_curated443": 443,
                    "biological_full": 283,
                    "human_curated_overlap": 22,
                    "biological_clean": 261,
                },
                "output_sha256": _synthetic_cohort_table_hashes(),
                "cohort_build_identity_sha256": sha256_json(
                    ["synthetic-cohort-build", 261, 283]
                ),
                "canonical_annotations_read": False,
                "root_cap_region_statistics_included": False,
                "blind_images_used": 0,
            }
        elif stage == "profiles_exact283":
            pin = read_model_contract_authority(
                fixture["run_dir"] / "authority_pin" / "receipt.json"
            )
            cohort_summary_path = (
                fixture["run_dir"] / "cohorts_exact283" / "receipt.json"
            )
            cohort_summary = read_json(cohort_summary_path)
            traits_summary_path = (
                fixture["run_dir"] / "traits_exact283" / "receipt.json"
            )
            all_tasks = [f"P{index:03d}" for index in range(283)]
            cohort_specs = (
                (
                    "primary_clean261",
                    "primary_SHA_disjoint",
                    all_tasks[:261],
                    "primary_summary.json",
                    "primary_profiles.csv",
                ),
                (
                    "sensitivity_full283",
                    "overlap_contaminated_sensitivity",
                    all_tasks,
                    "sensitivity_summary.json",
                    "sensitivity_profiles.csv",
                ),
            )
            exports: dict[str, dict[str, str]] = {}
            membership_sha = sha256_json(["synthetic-cohort-membership", 283])
            profile_contract_sha = sha256_json(["synthetic-profile-contract"])
            for (
                cohort_name,
                cohort_role,
                task_ids,
                summary_name,
                profiles_name,
            ) in cohort_specs:
                profiles_path = receipt_path.parent / profiles_name
                summary_path = receipt_path.parent / summary_name
                _write_synthetic_profile_csv(profiles_path, task_ids)
                task_membership_sha = sha256_json(sorted(task_ids))
                source_membership_sha = sha256_json(
                    sorted(
                        sha256_json(["synthetic-profile-source", task_id])
                        for task_id in task_ids
                    )
                )
                cohort_binding = {
                    "schema_version": (
                        "PHAxis-distal-axis-profile-cohort-binding-1.0.0"
                    ),
                    "cohort_name": cohort_name,
                    "cohort_role": cohort_role,
                    "cohort_tasks": len(task_ids),
                    "cohort_build_summary_sha256": sha256_file(
                        cohort_summary_path
                    ),
                    "cohort_build_identity_sha256": cohort_summary[
                        "cohort_build_identity_sha256"
                    ],
                    "cohort_membership_csv_sha256": membership_sha,
                    "cohort_task_membership_sha256": task_membership_sha,
                    "cohort_source_image_membership_sha256": (
                        source_membership_sha
                    ),
                    "blind_images_used": 0,
                }
                child = {
                    "schema_version": "PHAxis-distal-axis-profile-export-1.0.0",
                    "status": "completed",
                    "tasks": len(task_ids),
                    "bins_per_task": 5,
                    "rows": len(task_ids) * 5,
                    "locked_1_4mm_trait_crosscheck_mismatches": 0,
                    "traits_csv_sha256": cohort_summary["output_sha256"][
                        cohort_name
                    ]["traits"],
                    "hair_instances_csv_sha256": cohort_summary[
                        "output_sha256"
                    ][cohort_name]["hair_instances"],
                    "profile_contract_sha256": profile_contract_sha,
                    "profiles_csv_sha256": sha256_file(profiles_path),
                    "cohort_binding": cohort_binding,
                    "model_bundle_id": pin.model_bundle_id,
                    "root_expert_id": pin.root_expert_id,
                    "model_contract_proposal_sha256": pin.file_sha256,
                    "model_contract_proposal_identity_sha256": pin.identity_sha256,
                    "root_cap_region_output": False,
                    "stageb_two_point_vector_used_as_length": False,
                    "canonical_annotations_read": False,
                    "blind_images_used": 0,
                }
                child["export_identity_sha256"] = sha256_json(child)
                _write_json(summary_path, child)
                exports[cohort_name] = {
                    "summary_sha256": sha256_file(summary_path),
                    "profiles_csv_sha256": sha256_file(profiles_path),
                    "export_identity_sha256": child[
                        "export_identity_sha256"
                    ],
                    "cohort_task_membership_sha256": task_membership_sha,
                }
            payload = {
                "schema_version": spec["schema_version"],
                "status": "completed",
                "cohort_directories": {
                    "primary": "primary_clean261",
                    "sensitivity": "sensitivity_full283",
                },
                "counts": {
                    "human_curated443": 443,
                    "biological_full": 283,
                    "human_curated_overlap": 22,
                    "biological_clean": 261,
                },
                "primary_is_strict_task_subset_of_sensitivity": True,
                "primary_sensitivity_task_overlap": 261,
                "sensitivity_only_human443_overlap_tasks": 22,
                "cohort_build_summary_sha256": sha256_file(cohort_summary_path),
                "cohort_build_identity_sha256": cohort_summary[
                    "cohort_build_identity_sha256"
                ],
                "cohort_lock_sha256": sha256_json(["synthetic-cohort-lock"]),
                "cohort_membership_csv_sha256": membership_sha,
                "traits_summary_sha256": sha256_file(traits_summary_path),
                "profile_contract_sha256": profile_contract_sha,
                "cohort_exports": exports,
                "model_bundle_id": pin.model_bundle_id,
                "root_expert_id": pin.root_expert_id,
                "model_contract_proposal_sha256": pin.file_sha256,
                "model_contract_proposal_identity_sha256": pin.identity_sha256,
                "root_cap_region_output": False,
                "stageb_two_point_vector_used_as_length": False,
                "canonical_annotations_read": False,
                "blind_images_used": 0,
            }
        elif stage.startswith("benchmark_") and stage in {
            "benchmark_phaxis_production",
            "benchmark_frozen_v1_production",
            "benchmark_phaxis_sequential",
            "benchmark_frozen_v1_sequential",
        }:
            hardware = {
                "host": "synthetic-host",
                "platform": "synthetic-platform",
                "processor": "synthetic-processor",
                "gpus": [
                    {
                        "physical_index": 1,
                        "uuid": "GPU-SYNTHETIC-1",
                        "name": "NVIDIA RTX 3090",
                        "memory_total_mib": 24576,
                        "driver_version": "999.0",
                    }
                ],
            }
            provider_preflight = {
                "command": ["nvidia-smi", "synthetic"],
                "stdout_sha256": "a" * 64,
                "physical_gpus": [1],
            }
            payload = {
                "schema_version": spec["schema_version"],
                "status": "completed_direct_full283",
                "images": 283,
                "measurement_scope": "raw_image_to_final_traits_and_profiles_direct",
                "fresh_direct_run": True,
                "resume_or_cache_used": False,
                "physical_gpu_mapping": [1],
                "cuda_visible_devices_by_stage": {"direct_provider": "1"},
                "hardware": hardware,
                "hardware_identity_sha256": sha256_json(hardware),
                "nvidia_smi_preflight": provider_preflight,
                "nvidia_smi_preflight_identity_sha256": sha256_json(
                    provider_preflight
                ),
                "blind_images_used": 0,
            }
            if stage.startswith("benchmark_phaxis_"):
                q8_binding = {
                    "status": "passed_exact_physical_gpu_and_uuid",
                    "selection_receipts": 1,
                    "bindings": [
                        {
                            "selection_receipt_sha256": "b" * 64,
                            "requested_physical_gpu": 1,
                            "selected_physical_gpu": 1,
                            "physical_gpu_uuid": "GPU-SYNTHETIC-1",
                        }
                    ],
                }
                q8_binding["binding_identity_sha256"] = sha256_json(q8_binding)
                payload["q8_exact_device_binding"] = q8_binding
            for artifact_name in ("gpu_telemetry", "hardware_preflight"):
                artifact_path = fixture["run_dir"] / stage / f"{artifact_name}.json"
                _write_json(artifact_path, {"stage": stage, "role": artifact_name})
                payload[f"{artifact_name}_artifact"] = {
                    "path": artifact_path.name,
                    "sha256": sha256_file(artifact_path),
                }
        elif stage == "benchmark_same_hardware":
            payload = {
                "schema_version": spec["schema_version"],
                "status": "passed",
                "images": 283,
                "measurement_scope": "raw_image_to_final_traits_and_profiles_direct",
                "same_ordered_exact283_sources": True,
                "same_hardware_uuid_and_driver": True,
                "same_io_and_full_workflow_scope": True,
                "fresh_no_cache": True,
                "historical_98_47_min_component_receipt_used": False,
                "forward_only_runtime_used": False,
                "runs": [
                    {
                        "role": role,
                        "fresh_direct_run": True,
                        "resume_or_cache_used": False,
                        "full_workflow_io_included": True,
                    }
                    for role in (
                        "phaxis_production",
                        "phaxis_sequential",
                        "frozen_v1_production",
                        "frozen_v1_sequential",
                    )
                ],
                "blind_images_used": 0,
            }
        elif stage == "benchmark_artifact_inventory":
            payload = {
                "schema_version": spec["schema_version"],
                "status": "completed_explicit_benchmark_inventory",
                "role_counts": {
                    role: 1
                    for role in (
                        "same_hardware_receipt",
                        "phaxis_production_summary",
                        "v1_production_summary",
                        "phaxis_sequential_summary",
                        "v1_sequential_summary",
                        "production_comparison_receipt",
                        "sequential_comparison_receipt",
                    )
                },
                "blind_images_used": 0,
            }
            payload["role_counts"].update(
                {
                    "per_image_latency_csv": 2,
                    "gpu_telemetry": 4,
                    "hardware_preflight": 4,
                }
            )
        elif stage == "figure_inputs":
            resource_roles = (
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
            assert len(resource_roles) == 25
            assert set(resource_roles) == (
                release_orchestrator_module.PUBLICATION_FIGURE_INPUT_RESOURCE_ROLES
            )
            resource_root = receipt_path.parent / "resources"
            resource_root.mkdir()

            decision = _synthetic_narrative_decision()
            narrative_path = _write_json(
                resource_root / "narrative_decision.json",
                decision,
            )
            matcher_contract = {
                "assignment_mode": "synthetic_one_to_one",
                "distance_tolerance_px": 3.0,
            }
            assignment = {
                "schema_version": "PHAxis-qcdev-instance-assignment-1.0",
                "status": "completed_recomputed_from_sealed_geometry",
                "evidence_role": "selected_qc_development_non_independent",
                "matcher_contract": matcher_contract,
                "matcher_contract_sha256": sha256_json(matcher_contract),
                "source_input_sha256": sha256_json(
                    ["synthetic-qcdev-assignment", "source"]
                ),
                "source_input_identity_sha256": sha256_json(
                    ["synthetic-qcdev-assignment", "source-identity"]
                ),
                "stage7_lock_set_identity_sha256": sha256_json(
                    ["synthetic-qcdev-assignment", "stage7-lock"]
                ),
                "display_source_unit": "V000",
                "pooled": {"matched_instances": 1},
                "assignments": [
                    {
                        "task_id": "V000",
                        "prediction_index": 0,
                        "reference_index": 0,
                    }
                ],
                "independent_accuracy_claim_allowed": False,
                "blind_images_used": 0,
            }
            assignment["assignment_identity_sha256"] = sha256_json(
                assignment
            )
            assignment_path = _write_json(
                resource_root / "qcdev_assignment.json",
                assignment,
            )
            overlay_fields = (
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
            )
            overlay_path = resource_root / "overlay_audit.csv"
            with overlay_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=overlay_fields)
                writer.writeheader()
                case_roles = (
                    "representative", "low_contrast", "curved_dense",
                    "continuity", "fail_closed",
                )
                anchor_ids = {
                    "low_contrast": "RHSCU-aa5b6e37df15821f",
                    "curved_dense": "RHSCU-bbf649822174e0a2",
                }
                writer.writerows(
                    {
                        "schema_version": "PHAxis-Fig4-case-audit-2.0",
                        "case_id": f"case-{index}",
                        "case_role": case_roles[index],
                        "task_id": anchor_ids.get(case_roles[index], f"P{index:03d}"),
                        "source_image_sha256": sha256_json(
                            ["synthetic-overlay-source", index]
                        ),
                        "prediction_sha256": sha256_json(
                            ["synthetic-overlay-prediction", index]
                        ),
                        "formal_state": (
                            "formal" if index < 4 else "review_only"
                        ),
                        "axis_in_root_coverage_fraction": "1.0" if index < 4 else None,
                        "axis_single_component_coverage_fraction": "1.0" if index < 4 else None,
                        "longest_unsupported_axis_gap_um": "0.0" if index < 4 else None,
                        "formal_identity_count": "1" if index < 4 else None,
                        "endpoint_complete_support_count": "1" if index < 4 else None,
                        "endpoint_complete_support_fraction": "1.0" if index < 4 else None,
                        "distal_window_1_4mm_eligible": "true" if index < 4 else "false",
                        "distal_window_1_4mm_reason": "eligible_visible_axis_reaches_4mm" if index < 4 else "formal_statistics_ineligible:synthetic_fixture",
                        "profile_0_5mm_eligible": "true" if index < 4 else "false",
                        "profile_0_5mm_reason": "eligible_visible_axis_reaches_5mm" if index < 4 else "formal_statistics_ineligible:synthetic_fixture",
                        "downstream_eligible": "true" if index < 4 else "false",
                        "downstream_reason": "synthetic_fixture",
                        "condition_metadata_used": "false",
                    }
                    for index in range(5)
                )

            resource_paths = {
                "narrative_decision": narrative_path,
                "qcdev_assignment": assignment_path,
                "overlay_audit": overlay_path,
            }
            for role in resource_roles:
                if role in resource_paths:
                    continue
                resource_paths[role] = _write_json(
                    resource_root / f"{role}.json",
                    {
                        "role": role,
                        "synthetic_resource_contract": True,
                        "blind_images_used": 0,
                    },
                )
            resources = {
                role: {
                    "path": resource_paths[role]
                    .relative_to(receipt_path.parent)
                    .as_posix(),
                    "sha256": sha256_file(resource_paths[role]),
                }
                for role in resource_roles
            }
            figure_manifest = {
                "schema_version": "PHAxis-manuscript-figure-inputs-2.0",
                "assembler_schema_version": spec["schema_version"],
                "status": "final",
                "resources": resources,
                "narrative_decision_identity_sha256": decision[
                    "narrative_decision_identity_sha256"
                ],
                "narrative_branch_id": decision["branch_id"],
                "qcdev_assignment_identity_sha256": assignment[
                    "assignment_identity_sha256"
                ],
                "blind_images_used": 0,
            }
            figure_manifest["figure_input_assembly_identity_sha256"] = (
                sha256_json(figure_manifest)
            )
            figure_manifest_path = _write_json(
                fixture["run_dir"] / stage / "figure_inputs.json",
                figure_manifest,
            )
            payload = {
                "schema_version": spec["schema_version"],
                "status": spec["status"],
                "figure_inputs_sha256": sha256_file(figure_manifest_path),
                "figure_input_assembly_identity_sha256": figure_manifest[
                    "figure_input_assembly_identity_sha256"
                ],
                "resource_sha256": {
                    role: record["sha256"] for role, record in resources.items()
                },
                "narrative_decision_identity_sha256": decision[
                    "narrative_decision_identity_sha256"
                ],
                "narrative_branch_id": decision["branch_id"],
                "qcdev_assignment_identity_sha256": assignment[
                    "assignment_identity_sha256"
                ],
                "blind_images_used": 0,
            }
        elif stage == "figures":
            figure_inputs_path = (
                fixture["run_dir"] / "figure_inputs" / "figure_inputs.json"
            )
            figure_inputs = read_json(figure_inputs_path)
            narrative_path = (
                figure_inputs_path.parent
                / figure_inputs["resources"]["narrative_decision"]["path"]
            )
            decision = read_json(narrative_path)
            locked_titles = title_contract(decision)
            table_sources = _supplementary_source_fixture(
                fixture["run_dir"] / stage / "table_sources"
            )
            table_bundle = materialize_supplementary_table_data_bundle(
                output=(
                    fixture["run_dir"]
                    / stage
                    / "supplementary_tables_and_data"
                ),
                status=SUPPLEMENTARY_TABLE_FINAL_STATUS,
                source_paths=table_sources,
                source_identities={},
                figure_input_manifest_sha256=sha256_json("figure-inputs"),
                figure_input_assembly_identity_sha256=sha256_json(
                    "figure-input-assembly"
                ),
                model_contract_proposal_identity_sha256=sha256_json(
                    "model-contract-proposal"
                ),
            )
            payload = {
                "schema_version": spec["schema_version"],
                "status": spec["status"],
                "supplementary_tables": table_bundle["items"],
                "supplementary_table_bundle_receipt": (
                    "supplementary_tables_and_data/bundle_receipt.json"
                ),
                "supplementary_table_bundle_receipt_sha256": table_bundle[
                    "receipt_sha256"
                ],
                "supplementary_table_bundle_identity_sha256": table_bundle[
                    "bundle_identity_sha256"
                ],
                "supplementary_table_bundle_sha256": table_bundle[
                    "bundle_file_sha256"
                ],
                "supplementary_table_source_authority_sha256": table_bundle[
                    "source_authority_sha256"
                ],
                "supplementary_table_source_authority_identity": table_bundle[
                    "source_authority_identity"
                ],
                "claim_contract": {
                    "supplementary_table_data_resource_count": 10,
                    "narrative_decision_identity_sha256": decision[
                        "narrative_decision_identity_sha256"
                    ],
                    "profile_hypothesis_tests_added": False,
                    "profiles_select_or_veto_narrative_branch": False,
                },
                "narrative_decision_identity_sha256": figure_inputs[
                    "narrative_decision_identity_sha256"
                ],
                "narrative_branch_id": figure_inputs["narrative_branch_id"],
                "title_contract": locked_titles,
                "blind_images_used": 0,
            }
        elif stage == "evidence":
            figures = read_json(fixture["run_dir"] / "figures" / "receipt.json")
            payload = {
                "schema_version": spec["schema_version"],
                "status": spec["status"],
                "supplementary_table_data": {
                    "ordered_item_count": 10,
                    "bundle_receipt_sha256": figures[
                        "supplementary_table_bundle_receipt_sha256"
                    ],
                    "bundle_identity_sha256": figures[
                        "supplementary_table_bundle_identity_sha256"
                    ],
                    "source_authority_sha256": figures[
                        "supplementary_table_source_authority_sha256"
                    ],
                    "ordered_item_identity_sha256": {
                        stem: record["item_identity_sha256"]
                        for stem, record in figures["supplementary_tables"].items()
                    },
                },
                "blind_images_used": 0,
            }
        elif stage == "distributions":
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            asset_specs = (
                (
                    "phaxis-1.0.0-py3-none-any.whl",
                    "wheel",
                    b"synthetic wheel",
                ),
                ("phaxis-1.0.0.tar.gz", "sdist", b"synthetic sdist"),
                ("phaxis-1.0.0.cdx.json", "cyclonedx_sbom", b"synthetic sbom"),
                (
                    "phaxis-1.0.0-THIRD_PARTY_NOTICES.md",
                    "third_party_notices",
                    b"synthetic notices",
                ),
                (
                    "phaxis-1.0.0-THIRD_PARTY_LICENSES.json",
                    "third_party_license_inventory",
                    b"synthetic licenses",
                ),
            )
            for filename, _, content in asset_specs:
                path = receipt_path.parent / filename
                if filename.endswith(".whl"):
                    _synthetic_wheel(
                        path,
                        name="phaxis",
                        version="1.0.0",
                        requires=[
                            f'{name}>=1; extra == "deployment"'
                            for name in sorted(
                                dependency_materializer.REQUIRED_DEPLOYMENT_DISTRIBUTIONS
                            )
                        ],
                        license_expression="Apache-2.0",
                    )
                else:
                    path.write_bytes(content)
            release_assets = sorted(
                [
                    {
                        "filename": filename,
                        "kind": kind,
                        "bytes": (receipt_path.parent / filename).stat().st_size,
                        "sha256": sha256_file(receipt_path.parent / filename),
                    }
                    for filename, kind, _ in asset_specs
                ],
                key=lambda row: row["filename"],
            )
            release_asset_by_kind = {
                row["kind"]: row for row in release_assets
            }
            wheel_archive_audit = {
                "archive_filename": "phaxis-1.0.0-py3-none-any.whl",
                "archive_sha256": release_asset_by_kind["wheel"]["sha256"],
                "distribution": "phaxis",
                "version": "1.0.0",
                "wheel_tag": "py3-none-any",
                "metadata_member": "phaxis-1.0.0.dist-info/METADATA",
                "metadata_sha256": "5" * 64,
                "entry_points_member": (
                    "phaxis-1.0.0.dist-info/entry_points.txt"
                ),
                "entry_points_sha256": "6" * 64,
                "entry_point": "phaxis = phaxis.cli:main",
                "record_member": "phaxis-1.0.0.dist-info/RECORD",
                "record_sha256": "7" * 64,
                "record_member_count": 3,
                "record_verified": True,
                "metadata_license_files": [
                    "LICENSE",
                    "src/phaxis/_vendor/tomli/LICENSE.txt",
                ],
                "pep639_license_member_count": 2,
                "license_file_hashes_verified": True,
                "source_package_file_count": 1,
                "source_package_identity_sha256": "8" * 64,
                "source_package_hashes_verified": True,
                "unexpected_payload_members": 0,
                "prohibited_payload_members": 0,
            }
            build_toolchain = {
                "implementation": "CPython",
                "python_version": "3.11.9",
                "python_cache_tag": "cpython-311",
                "python_executable_filename": "python.exe",
                "python_executable_sha256": "9" * 64,
                "packages": {
                    "build": "1.2.2.post1",
                    "setuptools": "80.9.0",
                    "wheel": "0.45.1",
                    "twine": "6.2.0",
                },
                "probe_isolated": True,
                "cuda_visible_devices": "-1",
                "exact_versions_recorded": True,
                "build_isolation_used": False,
            }
            build_toolchain["build_toolchain_identity_sha256"] = sha256_json(
                build_toolchain
            )
            sdist_sha256 = release_asset_by_kind["sdist"]["sha256"]
            source_supply_chain = [
                {"path": path, "bytes": 1, "sha256": character * 64}
                for path, character in (
                    ("NOTICE", "1"),
                    ("THIRD_PARTY_NOTICES.md", "2"),
                    ("THIRD_PARTY_LICENSES.json", "3"),
                    ("SBOM.cdx.json", "4"),
                )
            ]
            source_supply_chain_by_path = {
                row["path"]: row for row in source_supply_chain
            }
            source_supply_chain_by_path["SBOM.cdx.json"]["sha256"] = (
                release_asset_by_kind["cyclonedx_sbom"]["sha256"]
            )
            source_supply_chain_by_path["THIRD_PARTY_NOTICES.md"]["sha256"] = (
                release_asset_by_kind["third_party_notices"]["sha256"]
            )
            source_supply_chain_by_path[
                "THIRD_PARTY_LICENSES.json"
            ]["sha256"] = release_asset_by_kind[
                "third_party_license_inventory"
            ]["sha256"]
            checksum_path = receipt_path.parent / "SHA256SUMS"
            checksum_path.write_text(
                "".join(
                    f"{row['sha256']}  {row['filename']}\n"
                    for row in release_assets
                ),
                encoding="utf-8",
                newline="\n",
            )
            asset_inventory = {
                "schema_version": "PHAxis-release-asset-inventory-1.0",
                "status": "sealed_release_assets",
                "distribution": "phaxis",
                "version": "1.0.0",
                "assets": release_assets,
                "asset_count": len(release_assets),
                "source_supply_chain": source_supply_chain,
                "blind_images_used": 0,
            }
            asset_inventory["release_asset_inventory_identity_sha256"] = (
                sha256_json(asset_inventory)
            )
            asset_inventory_path = receipt_path.parent / "release_asset_inventory.json"
            _write_json(asset_inventory_path, asset_inventory)
            payload = {
                "schema_version": spec["schema_version"],
                "status": "completed_wheel_sdist_verified",
                "source_release_manifest_sha256": sha256_file(
                    fixture["run_dir"] / "source_release" / "receipt.json"
                ),
                "artifacts": [
                    row
                    for row in release_assets
                    if row["kind"] in {"wheel", "sdist"}
                ],
                "release_assets": release_assets,
                "release_asset_inventory": {
                    "filename": asset_inventory_path.name,
                    "sha256": sha256_file(asset_inventory_path),
                    "identity_sha256": asset_inventory[
                        "release_asset_inventory_identity_sha256"
                    ],
                },
                "release_checksums": {
                    "filename": checksum_path.name,
                    "algorithm": "SHA-256",
                    "entries": len(release_assets),
                    "sha256": sha256_file(checksum_path),
                },
                "source_supply_chain": source_supply_chain,
                "sdist_archive_audit": {
                    "archive_sha256": sdist_sha256,
                    "source_manifest_self_covered": True,
                    "source_manifest_member": "SOURCE_MANIFEST.json",
                    "source_manifest_member_sha256": "a" * 64,
                    "authored_member_hashes_verified": True,
                    "allowed_pep517_generated_members": list(
                        PEP517_SDIST_GENERATED_MEMBERS
                    ),
                    "observed_pep517_generated_members": [
                        {
                            "path": path,
                            "bytes": 1,
                            "sha256": "e" * 64,
                        }
                        for path in PEP517_SDIST_GENERATED_MEMBERS
                    ],
                    "unexpected_generated_members": 0,
                    "unexpected_generated_member_paths": [],
                    "missing_allowed_generated_members": 0,
                    "missing_allowed_generated_member_paths": [],
                },
                "wheel_archive_audit": wheel_archive_audit,
                "build_toolchain": build_toolchain,
                "private_build_input": {
                    "role": "private_manifest_exact_source_copy",
                    "source_manifest_sha256": sha256_file(
                        fixture["run_dir"] / "source_release" / "receipt.json"
                    ),
                    "file_count_including_manifest": 2,
                    "tree_identity_sha256": "b" * 64,
                    "manifest_exact_copy_verified": True,
                },
                "source_release_input_immutable": True,
                "source_release_before_lock": {
                    "file_count": 2,
                    "identity_sha256": "c" * 64,
                },
                "source_release_after_lock": {
                    "file_count": 2,
                    "identity_sha256": "c" * 64,
                },
                "commands": [
                    {
                        "argv": [
                            "<BUILD_PYTHON>",
                            "-m",
                            "build",
                            "--outdir",
                            "<PRIVATE_DISTRIBUTION_OUTPUT>",
                            "<PRIVATE_MANIFEST_EXACT_SOURCE_COPY>",
                        ]
                    },
                    {
                        "argv": [
                            "<BUILD_PYTHON>",
                            "-m",
                            "twine",
                            "check",
                            "<PRIVATE_WHEEL>",
                            "<PRIVATE_SDIST>",
                        ]
                    },
                ],
                "build_isolation_used": False,
                "twine_check_passed": True,
                "blind_images_used": 0,
            }
        elif stage == "offline_dependencies":
            formal = (
                fixture["run_dir"]
                / "distributions"
                / "phaxis-1.0.0-py3-none-any.whl"
            )
            resolver_wheels = [
                _synthetic_wheel(
                    fixture["workspace"]
                    / "synthetic_resolver"
                    / f"{name.replace('-', '_')}-1.0-py3-none-any.whl",
                    name=name,
                    version="1.0",
                    requires=["torch>=1"] if name == "timm" else None,
                )
                for name in sorted(
                    dependency_materializer.REQUIRED_DEPLOYMENT_DISTRIBUTIONS
                )
            ]

            def resolver(
                argv: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess[str]:
                destination = Path(argv[argv.index("--dest") + 1])
                shutil.copyfile(formal, destination / formal.name)
                for wheel in resolver_wheels:
                    shutil.copyfile(wheel, destination / wheel.name)
                return subprocess.CompletedProcess(argv, 0, "resolved", "")

            materialized = fixture["workspace"] / "synthetic_offline_output"
            dependency_materializer.materialize_dependencies(
                formal_wheel=formal,
                python_executable=Path(sys.executable),
                output=materialized,
                runner=resolver,
            )
            shutil.copytree(materialized, receipt_path.parent, dirs_exist_ok=True)
            payload = read_json(materialized / "receipt.json")
        elif stage == "manuscript":
            payload = {
                "schema_version": spec["schema_version"],
                "status": "completed_strict_final_manuscript_compilation",
                "unresolved_token_count": 0,
                "author_metadata_complete": True,
                "output_sha256": "a" * 64,
                "blind_images_used": 0,
            }
        elif stage == "supplementary_manuscript":
            figures = read_json(fixture["run_dir"] / "figures" / "receipt.json")
            payload = {
                "schema_version": spec["schema_version"],
                "status": "completed_strict_final_supplementary_compilation",
                "unresolved_token_count": 0,
                "numeric_or_author_values_inserted": 0,
                "status_frontmatter_replacements": 1,
                "supplementary_table_data_materialized": True,
                "supplementary_table_data_resource_count": 10,
                "supplementary_table_bundle_receipt_sha256": figures[
                    "supplementary_table_bundle_receipt_sha256"
                ],
                "supplementary_table_bundle_identity_sha256": figures[
                    "supplementary_table_bundle_identity_sha256"
                ],
                "supplementary_table_item_identity_sha256": {
                    stem: record["item_identity_sha256"]
                    for stem, record in figures["supplementary_tables"].items()
                },
                "blind_images_used": 0,
            }
        elif stage in {"submission_docx", "supplementary_docx"}:
            document_root = fixture["run_dir"] / stage
            if stage == "submission_docx":
                title_path = document_root / "title_page.docx"
                anonymous_path = document_root / "anonymized_main.docx"
                title_path.write_bytes(b"synthetic editor-only title page")
                anonymous_path.write_bytes(b"synthetic anonymous main")
                document_sha = sha256_file(anonymous_path)
            else:
                supplement_path = document_root / "anonymized_supplement.docx"
                supplement_path.write_bytes(b"synthetic anonymous supplement")
                document_sha = sha256_file(supplement_path)
            payload = {
                "schema_version": spec["schema_version"],
                "status": spec["status"],
                "mode": "final",
                "submission_use_allowed": True,
                "docx_sha256": document_sha,
                "blind_images_used": 0,
            }
            if stage == "submission_docx":
                payload.update(
                    {
                        "submission_metadata_sha256": "a" * 64,
                        "submission_metadata_identity_sha256": "b" * 64,
                        "title_page_docx_sha256": sha256_file(title_path),
                        "anonymized_main_docx_sha256": sha256_file(
                            anonymous_path
                        ),
                        "title_page_separate": True,
                        "anonymized_main_separate": True,
                        "editor_only_declaration_sha256": {
                            "authors": "c" * 64,
                            "affiliations": "d" * 64,
                        },
                        "reviewer_visible_identity_declarations_removed": True,
                        "anonymous_core_creator_empty": True,
                    }
                )
            else:
                figures = read_json(
                    fixture["run_dir"] / "figures" / "receipt.json"
                )
                payload.update(
                    {
                        "submission_metadata_consumed": False,
                        "reviewer_visible": True,
                        "anonymized_supplement_separate": True,
                        "anonymous_core_creator_empty": True,
                        "supplementary_table_data_materialized": True,
                        "supplementary_table_data_resource_count": 10,
                        "supplementary_table_bundle_receipt_sha256": figures[
                            "supplementary_table_bundle_receipt_sha256"
                        ],
                        "supplementary_table_bundle_identity_sha256": figures[
                            "supplementary_table_bundle_identity_sha256"
                        ],
                        "supplementary_table_item_identity_sha256": {
                            stem: record["item_identity_sha256"]
                            for stem, record in figures[
                                "supplementary_tables"
                            ].items()
                        },
                    }
                )
        elif stage == "manuscript_artifact_qa":
            figures = read_json(fixture["run_dir"] / "figures" / "receipt.json")
            upload = {
                "schema_version": "PHAxis-submission-upload-role-manifest-1.0",
                "status": "sealed_editor_and_reviewer_upload_roles",
                "submission_model": "double_anonymous",
                "roles": {
                    "editor_only": {
                        "title_page": {
                            "filename": "title_page.docx",
                            "sha256": "1" * 64,
                        }
                    },
                    "reviewer_visible": {
                        "anonymized_main": {
                            "filename": "anonymized_main.docx",
                            "sha256": "2" * 64,
                        },
                        "anonymized_supplement": {
                            "filename": "anonymized_supplement.docx",
                            "sha256": "3" * 64,
                        },
                    },
                },
                "editor_only_document_count": 1,
                "reviewer_visible_document_count": 2,
                "reviewer_visible_identity_occurrence_count": 0,
                "reviewer_visible_ooxml_deep_scan_passed": True,
                "blind_images_used": 0,
            }
            upload["upload_manifest_identity_sha256"] = sha256_json(upload)
            upload_path = receipt_path.parent / "upload-role-manifest.json"
            _write_json(upload_path, upload)
            payload = {
                "schema_version": spec["schema_version"],
                "status": "passed_double_anonymous_three_role_ooxml_closure",
                "ooxml_zip_magic_and_required_structure_passed": True,
                "master_authority_closure_passed": True,
                "figure_input_closure_passed": True,
                "supplementary_table_data_closure_passed": True,
                "supplementary_table_data_closure": {
                    "ordered_item_count": 10,
                    "bundle_receipt_sha256": figures[
                        "supplementary_table_bundle_receipt_sha256"
                    ],
                    "bundle_identity_sha256": figures[
                        "supplementary_table_bundle_identity_sha256"
                    ],
                    "source_authority_sha256": figures[
                        "supplementary_table_source_authority_sha256"
                    ],
                    "ordered_item_identity_sha256": {
                        stem: record["item_identity_sha256"]
                        for stem, record in figures["supplementary_tables"].items()
                    },
                },
                "availability_statement_closure": {
                    "public_source_url_present": True,
                    "license_present": True,
                },
                "data_and_code_availability_present": True,
                "title_page_ooxml": {"reviewer_visible": False},
                "main_ooxml": {"reviewer_visible": True},
                "supplement_ooxml": {"reviewer_visible": True},
                "document_roles": {
                    "editor_only": ["title_page"],
                    "reviewer_visible": [
                        "anonymized_main",
                        "anonymized_supplement",
                    ],
                },
                "reviewer_visible_identity_occurrence_count": 0,
                "reviewer_visible_core_identity_occurrence_count": 0,
                "reviewer_visible_tracked_change_count": 0,
                "reviewer_visible_hidden_text_count": 0,
                "reviewer_visible_embedded_image_identity_occurrence_count": 0,
                "deep_ooxml_anonymity_scan_passed": True,
                "editor_only_title_page_completeness_passed": True,
                "submission_upload_role_manifest_sha256": sha256_file(
                    upload_path
                ),
                "submission_upload_role_manifest_identity_sha256": upload[
                    "upload_manifest_identity_sha256"
                ],
                "submission_use_allowed_before_visual_qa": False,
                "blind_images_used": 0,
            }
        elif stage == "manuscript_render":
            structural_path = (
                fixture["run_dir"] / "manuscript_artifact_qa" / "receipt.json"
            )
            structural = read_json(structural_path)
            upload_path = (
                fixture["run_dir"]
                / "manuscript_artifact_qa"
                / "upload-role-manifest.json"
            )
            upload = read_json(upload_path)
            payload = {
                "schema_version": spec["schema_version"],
                "status": "completed_three_role_word_pdf_and_page_png_render",
                "structural_qa_sha256": sha256_file(structural_path),
                "structural_qa_identity_sha256": structural[
                    "qa_identity_sha256"
                ],
                "submission_upload_role_manifest_sha256": sha256_file(
                    upload_path
                ),
                "submission_upload_role_manifest_identity_sha256": upload[
                    "upload_manifest_identity_sha256"
                ],
                "pdf_magic_passed": True,
                "page_rasterization_completed": True,
                "visual_qa_completed": False,
                "submission_use_allowed": False,
                "documents": {
                    role: {"pages": 1, "page_png_records": [{"page": 1}]}
                    for role in (
                        "title_page",
                        "anonymized_main",
                        "anonymized_supplement",
                    )
                },
                "blind_images_used": 0,
            }
        elif stage == "manuscript_visual_qa":
            payload = {
                "schema_version": spec["schema_version"],
                "status": "passed_author_verified_three_role_page_visual_qa",
                "documents_reviewed": 3,
                "editor_only_documents_reviewed": 1,
                "reviewer_visible_documents_reviewed": 2,
                "reviewer_visible_identity_occurrence_count": 0,
                "pages_reviewed": 3,
                "all_pages_reviewed_at_original_resolution": True,
                "submission_visual_gate_passed": True,
                "submission_use_allowed": True,
                "blind_images_used": 0,
            }
        elif stage.startswith("handover_") and stage.endswith("_manifest"):
            payload = {
                "schema_version": spec["schema_version"],
                "status": "created",
                "materialisation_role": stage.removeprefix("handover_"),
                "blind_images_used": 0,
            }
        elif stage == "handover_contract":
            payload = {
                "schema_version": spec["schema_version"],
                "status": "created",
                "bindings": 16,
                "blind_images_used": 0,
            }
        else:
            status_field = spec.get("status_field", "status")
            payload = {
                "schema_version": spec["schema_version"],
                status_field: spec["status"],
                "blind_images_used": 0,
            }
        identity_field = spec.get("identity_field")
        if identity_field:
            payload.pop(identity_field, None)
            payload[identity_field] = sha256_json(payload)
        _write_json(receipt_path, payload)
        return SimpleNamespace(returncode=0, stdout=stage, stderr="")

    return run


def test_plan_is_deterministic_read_only_and_uses_terminal_release_finalize(tmp_path: Path) -> None:
    fixture = _manifest_fixture(tmp_path)
    builder = _cached_builder(fixture["preview"])
    first = build_release_plan(
        fixture["manifest"], fixture["run_dir"], candidate_builder=builder
    )
    second = build_release_plan(
        fixture["manifest"], fixture["run_dir"], candidate_builder=builder
    )
    assert first == second
    assert not fixture["run_dir"].exists()
    assert [stage["name"] for stage in first["stages"]] == list(
        MANDATORY_STAGE_ORDER
    )
    assert first["stages"][-1]["name"] == "release_finalize"
    assert first["official_apply_policy"].startswith("compare_and_swap")
    assert [stage["name"] for stage in first["stages"]].index(
        "official_apply"
    ) < [stage["name"] for stage in first["stages"]].index("source_release")
    assert all(
        stage["gpu"]["cuda_visible_devices"] == "1"
        for stage in first["stages"]
        if stage["gpu"] is not None
    )
    assert sha256_file(fixture["official"]) == fixture["expected_official_sha256"]


def test_orchestrator_figures_gate_revalidates_physical_s1_s10_bundle(
    tmp_path: Path,
) -> None:
    figures_root = tmp_path / "figures"
    bundle = materialize_supplementary_table_data_bundle(
        output=figures_root / "supplementary_tables_and_data",
        status=SUPPLEMENTARY_TABLE_FINAL_STATUS,
        source_paths=_supplementary_source_fixture(tmp_path / "sources"),
        source_identities={},
        figure_input_manifest_sha256=sha256_json("figure-inputs"),
        figure_input_assembly_identity_sha256=sha256_json("assembly"),
        model_contract_proposal_identity_sha256=sha256_json("proposal"),
    )
    summary_path = figures_root / "figure_assembly_summary.json"
    payload = {
        "supplementary_table_bundle_receipt": (
            "supplementary_tables_and_data/bundle_receipt.json"
        ),
        "supplementary_tables": bundle["items"],
        "supplementary_table_bundle_receipt_sha256": bundle["receipt_sha256"],
        "supplementary_table_bundle_identity_sha256": bundle[
            "bundle_identity_sha256"
        ],
        "supplementary_table_bundle_sha256": bundle["bundle_file_sha256"],
        "supplementary_table_source_authority_sha256": bundle[
            "source_authority_sha256"
        ],
        "supplementary_table_source_authority_identity": bundle[
            "source_authority_identity"
        ],
        "claim_contract": {"supplementary_table_data_resource_count": 10},
    }
    _write_json(summary_path, payload)
    assert _validate_figure_table_bundle(summary_path, payload)[
        "ordered_item_count"
    ] == 10
    relative = next(iter(bundle["bundle_file_sha256"]))
    target = figures_root / "supplementary_tables_and_data" / relative
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ReleaseOrchestratorError, match="validation failed"):
        _validate_figure_table_bundle(summary_path, payload)


def test_figure_input_resource_role_contract_is_exact_ordered_25() -> None:
    logical_order = (
        release_orchestrator_module.PUBLICATION_FIGURE_INPUT_RESOURCE_ROLE_ORDER
    )
    order = (
        release_orchestrator_module.PUBLICATION_FIGURE_INPUT_RESOURCE_CANONICAL_KEY_ORDER
    )
    assert len(logical_order) == 25
    assert len(set(logical_order)) == 25
    assert set(logical_order) == (
        release_orchestrator_module.PUBLICATION_FIGURE_INPUT_RESOURCE_ROLES
    )
    assert order == tuple(sorted(logical_order))
    resources = {role: {} for role in order}
    assert (
        release_orchestrator_module._validate_publication_figure_input_resource_roles(
            resources
        )
        == order
    )


@pytest.mark.parametrize("mutation", ("missing", "extra", "order"))
def test_figure_input_resource_role_contract_fails_closed(
    mutation: str,
) -> None:
    order = (
        release_orchestrator_module.PUBLICATION_FIGURE_INPUT_RESOURCE_CANONICAL_KEY_ORDER
    )
    resources = {role: {} for role in order}
    if mutation == "missing":
        resources.pop(order[-1])
    elif mutation == "extra":
        resources["undeclared_resource"] = {}
    else:
        items = list(resources.items())
        items[0], items[1] = items[1], items[0]
        resources = dict(items)

    with pytest.raises(
        ReleaseOrchestratorError,
        match="final ordered 25-resource manifest",
    ):
        release_orchestrator_module._validate_publication_figure_input_resource_roles(
            resources
        )


def test_plan_fails_closed_on_missing_tampered_and_semantically_invalid_inputs(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    builder = _cached_builder(fixture["preview"])
    manifest = fixture["manifest"]
    payload = fixture["manifest_payload"]
    receipt_name = payload["training_members"][0]["completion_receipt_input"]
    receipt_path = fixture["workspace"] / payload["external_inputs"][receipt_name]["path"]
    original = receipt_path.read_bytes()
    receipt_path.unlink()
    with pytest.raises(ReleaseOrchestratorError, match="missing"):
        build_release_plan(manifest, fixture["run_dir"], candidate_builder=builder)
    receipt_path.write_bytes(original)

    receipt_path.write_bytes(original + b"tamper")
    with pytest.raises(ReleaseOrchestratorError, match="hash drifted"):
        build_release_plan(manifest, fixture["run_dir"], candidate_builder=builder)
    receipt_path.write_bytes(original)

    checkpoint_name = payload["training_members"][0]["checkpoint_input"]
    checkpoint_path = (
        fixture["workspace"]
        / payload["external_inputs"][checkpoint_name]["path"]
    )
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_path.unlink()
    with pytest.raises(ReleaseOrchestratorError, match="missing"):
        build_release_plan(manifest, fixture["run_dir"], candidate_builder=builder)
    checkpoint_path.write_bytes(checkpoint_bytes)

    # A future release-derived artifact cannot be relabelled as an external
    # authority.  Schema 1.2 accepts only irreducible raw/frozen/static/author
    # or completed-training classes.
    payload = read_json(manifest)
    payload["external_inputs"][receipt_name]["authority_class"] = (
        "derived_release_artifact"
    )
    payload.pop("manifest_identity_sha256")
    payload["manifest_identity_sha256"] = sha256_json(payload)
    _write_json(manifest, payload)
    with pytest.raises(ReleaseOrchestratorError, match="authority_class"):
        build_release_plan(manifest, fixture["run_dir"], candidate_builder=builder)


def test_plan_rejects_undeclared_future_artifact_embedded_only_in_command(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    payload = read_json(fixture["manifest"])
    candidate = payload["stages"][0]
    selection = next(stage for stage in payload["stages"] if stage["name"] == "selection")
    selection_receipt = next(
        artifact
        for artifact in selection["artifacts"]
        if artifact["name"] == selection["receipt"]["artifact"]
    )
    candidate["command"].append(selection_receipt["path"])
    payload.pop("manifest_identity_sha256")
    payload["manifest_identity_sha256"] = sha256_json(payload)
    _write_json(fixture["manifest"], payload)

    with pytest.raises(ReleaseOrchestratorError, match="future-stage authority"):
        build_release_plan(
            fixture["manifest"],
            fixture["run_dir"],
            candidate_builder=_cached_builder(fixture["preview"]),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("primary_sensitivity_task_overlap", 260, "bundle contract is incomplete"),
        (
            "sensitivity_only_human443_overlap_tasks",
            21,
            "bundle contract is incomplete",
        ),
    ),
)
def test_profile_bundle_runtime_gate_rejects_resealed_cohort_drift(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    events: list[str] = []
    base_runner = _runner(fixture, events)

    def corrupting_runner(*, command: list[str], cwd: Path, env: dict[str, str]):
        result = base_runner(command=command, cwd=cwd, env=env)
        if command[1] == "profiles_exact283":
            receipt = fixture["run_dir"] / "profiles_exact283" / "receipt.json"
            payload = read_json(receipt)
            payload[field] = value
            payload.pop("cohort_profile_bundle_identity_sha256", None)
            payload["cohort_profile_bundle_identity_sha256"] = sha256_json(payload)
            _write_json(receipt, payload)
        return result

    with pytest.raises(ReleaseOrchestratorError, match=message):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=corrupting_runner,
            gpu_probe=_probe(events),
        )


def test_execute_orders_stages_preflights_gpu_and_resumes_without_relaunch(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    events: list[str] = []
    state = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, events),
        gpu_probe=_probe(events),
    )
    assert state["status"] == "completed"
    assert state["completed_stage_names"] == list(MANDATORY_STAGE_ORDER)
    finalization = read_json(
        fixture["run_dir"] / "release_finalize" / "receipt.json"
    )
    assert finalization["release_authority_registry_path"] == (
        fixture["release_registry"]
        .resolve()
        .relative_to(fixture["workspace"].resolve())
        .as_posix()
    )
    assert str(tmp_path) not in json.dumps(finalization, sort_keys=True)
    distribution = read_json(fixture["run_dir"] / "distributions" / "receipt.json")
    assert str(tmp_path) not in json.dumps(distribution["commands"], sort_keys=True)
    registry = read_json(fixture["release_registry"])
    assert registry["schema_version"] == "PHAxis-release-authority-registry-1.1"
    assert registry["status"] == "formal_release_materialized_and_verified"
    assert registry["current_formal_source_release"]["version"] == "1.0.0"
    assert registry["current_formal_release_gate_receipt"]["status"] == "passed"
    assert state["release_authority_registry_promotion"]["promoted_sha256"] == sha256_file(
        fixture["release_registry"]
    )
    run_events = [event.removeprefix("run:") for event in events if event.startswith("run:")]
    assert run_events == [
        name
        for name in MANDATORY_STAGE_ORDER
        if name not in {"authority_pin", "release_finalize"}
    ]
    for name in (
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
    ):
        probe_index = events.index(f"probe:{name}")
        assert events[probe_index + 1] == f"run:{name}"
    assert events[-1] == "run:handover"
    assert events.index("run:official_apply") < events.index("run:source_release")
    assert sha256_file(fixture["official"]) != fixture["expected_official_sha256"]
    pin = read_model_contract_authority(
        fixture["run_dir"] / "authority_pin" / "receipt.json"
    )
    assert pin.authority_lifecycle == RUN_SCOPED_AUTHORITY_PIN_LIFECYCLE

    resumed_events: list[str] = []
    with pytest.raises(ReleaseOrchestratorError, match="already exists"):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=_runner(fixture, []),
            gpu_probe=_probe([]),
        )
    resumed = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        resume=True,
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, resumed_events),
        gpu_probe=_probe(resumed_events),
    )
    assert resumed["status"] == "completed"
    assert resumed_events == []

    traits = fixture["run_dir"] / "traits_exact283" / "receipt.json"
    traits.write_bytes(traits.read_bytes() + b"tamper")
    with pytest.raises((ReleaseOrchestratorError, json.JSONDecodeError)):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            resume=True,
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=_runner(fixture, []),
            gpu_probe=_probe([]),
        )


def test_terminal_registry_cas_recovers_after_replace_before_state_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _manifest_fixture(tmp_path)
    events: list[str] = []
    real_promote = release_orchestrator_module._atomic_promote_release_registry
    interrupted = {"raised": False}

    def promote_then_interrupt(context, plan):
        result = real_promote(context, plan)
        if not interrupted["raised"]:
            interrupted["raised"] = True
            raise RuntimeError("synthetic crash after registry replacement")
        return result

    monkeypatch.setattr(
        release_orchestrator_module,
        "_atomic_promote_release_registry",
        promote_then_interrupt,
    )
    with pytest.raises(RuntimeError, match="synthetic crash"):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=_runner(fixture, events),
            gpu_probe=_probe(events),
        )
    promoted_sha256 = sha256_file(fixture["release_registry"])
    assert promoted_sha256 != fixture["expected_release_registry_sha256"]
    failed_state = read_json(fixture["run_dir"] / "state.json")
    assert failed_state["status"] == "failed_closed"
    assert failed_state["current_stage"] == "release_authority_registry_promotion"
    assert failed_state["release_finalize_sentinel_committed"] is True

    monkeypatch.setattr(
        release_orchestrator_module,
        "_atomic_promote_release_registry",
        real_promote,
    )
    resumed_events: list[str] = []
    resumed = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        resume=True,
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, resumed_events),
        gpu_probe=_probe(resumed_events),
    )
    assert resumed["status"] == "completed"
    assert resumed_events == []
    assert sha256_file(fixture["release_registry"]) == promoted_sha256
    assert resumed["release_authority_registry_promotion"]["status"] == (
        "already_promoted_exact_idempotent_recovery"
    )


def test_terminal_finalize_rejects_resealed_supply_chain_stage_hash_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _manifest_fixture(tmp_path)
    real_finalize = release_orchestrator_module._write_internal_release_finalize

    def tampered_finalize(context, plan, plan_stage) -> None:
        real_finalize(context, plan, plan_stage)
        destination = context.artifact_paths[
            ("release_finalize", plan_stage["receipt_contract"]["artifact"])
        ]
        payload = read_json(destination)
        payload["software_supply_chain_receipt_sha256"]["source_release"] = (
            "f" * 64
        )
        payload.pop("release_finalization_identity_sha256")
        payload["release_finalization_identity_sha256"] = sha256_json(payload)
        atomic_write_json(destination, payload)

    monkeypatch.setattr(
        release_orchestrator_module,
        "_write_internal_release_finalize",
        tampered_finalize,
    )
    with pytest.raises(
        ReleaseOrchestratorError,
        match="exact software supply-chain closure",
    ):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=_runner(fixture, []),
            gpu_probe=_probe([]),
        )


def test_distribution_gate_rejects_release_asset_byte_tamper(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    events: list[str] = []
    base_runner = _runner(fixture, events)

    def tampering_runner(*, command: list[str], cwd: Path, env: dict[str, str]):
        result = base_runner(command=command, cwd=cwd, env=env)
        if command[1] == "distributions":
            target = (
                fixture["run_dir"]
                / "distributions"
                / "phaxis-1.0.0-THIRD_PARTY_LICENSES.json"
            )
            target.write_bytes(target.read_bytes() + b"tamper")
        return result

    with pytest.raises(
        ReleaseOrchestratorError,
        match="release asset bytes do not match",
    ):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=tampering_runner,
            gpu_probe=_probe(events),
        )


def test_user_gpu_hold_pauses_before_probe_or_command_and_resumes_exact_prefix(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    events: list[str] = []
    paused = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        held_physical_gpus=(1,),
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, events),
        gpu_probe=_probe(events),
    )

    assert paused["status"] == "paused_for_user_gpu_hold"
    assert paused["algorithm_or_training_failure"] is False
    assert paused["completed_stage_names"] == list(MANDATORY_STAGE_ORDER[:4])
    assert paused["current_stage"] == "qcdev_candidate_pool"
    assert paused["human_authority_gate"] is None
    assert paused["gpu_hold_gate"] == {
        "schema_version": "PHAxis-user-GPU-hold-gate-1.0",
        "status": "expected_user_gpu_hold_gate",
        "expected_pause_not_algorithm_or_training_failure": True,
        "resume_required": True,
        "expected_process_exit_code": EXPECTED_GPU_HOLD_EXIT_CODE,
        "stage_index": 4,
        "stage_name": "qcdev_candidate_pool",
        "stage_physical_gpus": [1],
        "held_physical_gpus": [1],
        "blocked_physical_gpus": [1],
        "gpu_probe_called": False,
        "stage_command_started": False,
        "resume_policy": (
            "resume only after the user explicitly releases every blocked "
            "physical GPU"
        ),
    }
    assert "probe:qcdev_candidate_pool" not in events
    assert "run:qcdev_candidate_pool" not in events
    assert not (fixture["run_dir"] / "qcdev_candidate_pool").exists()

    repeated_events: list[str] = []
    repeated = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        resume=True,
        held_physical_gpus=(1,),
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, repeated_events),
        gpu_probe=_probe(repeated_events),
    )
    assert repeated["status"] == "paused_for_user_gpu_hold"
    assert repeated["completed_stage_names"] == paused["completed_stage_names"]
    assert repeated_events == []

    resumed_events: list[str] = []
    completed = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        resume=True,
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, resumed_events),
        gpu_probe=_probe(resumed_events),
    )
    assert completed["status"] == "completed"
    assert resumed_events[0] == "probe:qcdev_candidate_pool"
    assert resumed_events[1] == "run:qcdev_candidate_pool"


@pytest.mark.parametrize("held", [(-1,), (True,), (1, 1)])
def test_user_gpu_hold_rejects_invalid_indices(
    tmp_path: Path, held: tuple[object, ...]
) -> None:
    fixture = _manifest_fixture(tmp_path)
    with pytest.raises(ReleaseOrchestratorError, match="held physical GPUs"):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            held_physical_gpus=held,  # type: ignore[arg-type]
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=_runner(fixture, []),
            gpu_probe=_probe([]),
        )


def test_execution_rejects_root_q8_uuid_and_benchmark_hardware_misreporting(
    tmp_path: Path,
) -> None:
    root_fixture = _manifest_fixture(tmp_path / "root")
    root_normal = _runner(root_fixture, [])

    def wrong_root_uuid(*, command, cwd, env):
        result = root_normal(command=command, cwd=cwd, env=env)
        if command[1] == "root_provider_exact283":
            path = (
                root_fixture["run_dir"]
                / "root_provider_exact283"
                / "output"
                / "q8_shards"
                / "exact_device_binding.json"
            )
            payload = read_json(path)
            payload["records"][0]["physical_gpu_uuid"] = "GPU-DIFFERENT"
            payload.pop("binding_identity_sha256")
            payload["binding_identity_sha256"] = sha256_json(payload)
            _write_json(path, payload)
        return result

    with pytest.raises(
        ReleaseOrchestratorError,
        match="Q8 shard index/UUID differs from orchestrator preflight",
    ):
        execute_release(
            root_fixture["manifest"],
            root_fixture["run_dir"],
            candidate_builder=_cached_builder(root_fixture["preview"]),
            command_runner=wrong_root_uuid,
            gpu_probe=_probe([]),
        )

    benchmark_fixture = _manifest_fixture(tmp_path / "benchmark")
    benchmark_normal = _runner(benchmark_fixture, [])

    def wrong_benchmark_uuid(*, command, cwd, env):
        result = benchmark_normal(command=command, cwd=cwd, env=env)
        if command[1] == "benchmark_phaxis_production":
            path = (
                benchmark_fixture["run_dir"]
                / "benchmark_phaxis_production"
                / "receipt.json"
            )
            payload = read_json(path)
            payload["hardware"]["gpus"][0]["uuid"] = "GPU-DIFFERENT"
            payload["hardware_identity_sha256"] = sha256_json(payload["hardware"])
            _write_json(path, payload)
        return result

    with pytest.raises(
        ReleaseOrchestratorError,
        match="receipt GPU index/UUID/driver differs from orchestrator preflight",
    ):
        execute_release(
            benchmark_fixture["manifest"],
            benchmark_fixture["run_dir"],
            candidate_builder=_cached_builder(benchmark_fixture["preview"]),
            command_runner=wrong_benchmark_uuid,
            gpu_probe=_probe([]),
        )


def test_post_apply_stage_can_consume_the_designated_cas_authority(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    payload = read_json(fixture["manifest"])
    payload["external_inputs"]["official_contract"] = {
        "path": fixture["official"].relative_to(fixture["workspace"]).as_posix(),
        "kind": "file",
        "sha256": fixture["expected_official_sha256"],
        "authority_class": "static_contract",
    }
    clean_install = next(
        stage for stage in payload["stages"] if stage["name"] == "clean_install"
    )
    clean_install["inputs"].append({"external": "official_contract"})
    payload.pop("manifest_identity_sha256")
    payload["manifest_identity_sha256"] = sha256_json(payload)
    _write_json(fixture["manifest"], payload)
    fixture["manifest_payload"] = payload

    state = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, []),
        gpu_probe=_probe([]),
    )
    assert state["status"] == "completed"
    sentinel = next(
        read_json(path)
        for path in (fixture["run_dir"] / "sentinels").glob("*.json")
        if read_json(path).get("stage_name") == "clean_install"
    )
    authority_lock = next(
        lock
        for lock in sentinel["input_locks"]
        if lock.get("external") == "official_contract"
    )
    assert authority_lock["authority_phase"] == (
        "applied_for_post_apply_release_closure"
    )
    assert authority_lock["sha256"] == sha256_file(fixture["official"])
    assert authority_lock["sha256"] != fixture["expected_official_sha256"]


def test_gpu_preflight_failure_or_command_failure_stops_before_later_stage(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    events: list[str] = []
    with pytest.raises(ReleaseOrchestratorError, match="exit code 17"):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=_runner(
                fixture,
                events,
                fail_stage="qcdev_evaluation_inference",
            ),
            gpu_probe=_probe(events),
        )
    assert events[-2:] == [
        "probe:qcdev_evaluation_inference",
        "run:qcdev_evaluation_inference",
    ]
    assert "run:qcdev_evaluation" not in events
    state = read_json(fixture["run_dir"] / "state.json")
    assert state["status"] == "failed_closed"
    assert state["failure"]["stage"] == "qcdev_evaluation_inference"

    preflight_fixture = _manifest_fixture(tmp_path / "preflight")
    preflight_events: list[str] = []

    def insufficient(*, stage: str):
        preflight_events.append(f"probe:{stage}")
        return SimpleNamespace(
            returncode=0,
            stdout=(
                "1, GPU-SYNTHETIC-1, NVIDIA RTX 3090, "
                "24576, 24000, 10, 999.0\n"
            ),
            stderr="",
        )

    with pytest.raises(ReleaseOrchestratorError, match="insufficient free VRAM"):
        execute_release(
            preflight_fixture["manifest"],
            preflight_fixture["run_dir"],
            candidate_builder=_cached_builder(preflight_fixture["preview"]),
            command_runner=_runner(preflight_fixture, preflight_events),
            gpu_probe=insufficient,
        )
    assert preflight_events[-1] == "probe:qcdev_candidate_pool"
    assert "run:qcdev_candidate_pool" not in preflight_events
    preflight_state = read_json(preflight_fixture["run_dir"] / "state.json")
    assert preflight_state["status"] == "failed_closed"
    assert preflight_state["failure"]["stage"] == "qcdev_candidate_pool"


def test_resume_recovers_application_receipt_after_cas_replace_crash(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    events: list[str] = []
    normal = _runner(fixture, events)

    def crash_after_replace(*, command, cwd, env):
        if command[1] != "official_apply":
            return normal(command=command, cwd=cwd, env=env)
        events.append("run:official_apply")
        receipt = fixture["run_dir"] / "official_apply" / "receipt.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        _apply_official(fixture, fixture["run_dir"], receipt)
        receipt.unlink()
        return SimpleNamespace(
            returncode=29,
            stdout="official replaced",
            stderr="synthetic receipt publication crash",
        )

    with pytest.raises(ReleaseOrchestratorError, match="exit code 29"):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=crash_after_replace,
            gpu_probe=_probe(events),
        )
    assert sha256_file(fixture["official"]) != fixture["expected_official_sha256"]
    assert not (fixture["run_dir"] / "official_apply" / "receipt.json").exists()
    failed_state = read_json(fixture["run_dir"] / "state.json")
    assert failed_state["official_apply_performed"] is True
    assert failed_state["official_apply_sentinel_committed"] is False

    resumed_events: list[str] = []
    state = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        resume=True,
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, resumed_events),
        gpu_probe=_probe(resumed_events),
    )
    assert state["status"] == "completed"
    official_index = MANDATORY_STAGE_ORDER.index("official_apply")
    expected_resumed_events: list[str] = []
    for name in MANDATORY_STAGE_ORDER[official_index + 1 :]:
        if name == "release_finalize":
            continue
        if name in {"clean_install_expected_identity", "clean_install"}:
            expected_resumed_events.append(f"probe:{name}")
        expected_resumed_events.append(f"run:{name}")
    assert resumed_events == expected_resumed_events
    recovered = read_json(
        fixture["run_dir"] / "official_apply" / "receipt.json"
    )
    assert recovered["status"] == "applied"
    assert recovered["final_model_contract_sha256"] == sha256_file(
        fixture["official"]
    )


def test_post_apply_failure_keeps_release_open_and_resumes_against_same_authority(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    events: list[str] = []
    with pytest.raises(ReleaseOrchestratorError, match="exit code 17"):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=_runner(
                fixture,
                events,
                fail_stage="source_release",
            ),
            gpu_probe=_probe(events),
        )
    failed = read_json(fixture["run_dir"] / "state.json")
    assert failed["status"] == "failed_closed"
    assert failed["failure"]["stage"] == "source_release"
    assert failed["official_apply_performed"] is True
    assert failed["official_apply_sentinel_committed"] is True
    assert failed["release_finalize_sentinel_committed"] is False
    applied_sha = sha256_file(fixture["official"])
    assert applied_sha != fixture["expected_official_sha256"]

    resumed_events: list[str] = []
    completed = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        resume=True,
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, resumed_events),
        gpu_probe=_probe(resumed_events),
    )
    assert completed["status"] == "completed"
    assert completed["release_finalize_sentinel_committed"] is True
    assert sha256_file(fixture["official"]) == applied_sha
    official_index = MANDATORY_STAGE_ORDER.index("official_apply")
    expected_resumed_events: list[str] = []
    for name in MANDATORY_STAGE_ORDER[official_index + 1 :]:
        if name == "release_finalize":
            continue
        if name in {"clean_install_expected_identity", "clean_install"}:
            expected_resumed_events.append(f"probe:{name}")
        expected_resumed_events.append(f"run:{name}")
    assert resumed_events == expected_resumed_events


def test_values_blocked_diagnostic_is_not_a_success_artifact_and_can_resume(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    events: list[str] = []
    normal = _runner(fixture, events)
    work_item = fixture["run_dir"] / "values" / "missing_human.json"

    def block_values(*, command, cwd, env):
        if command[1] != "values":
            return normal(command=command, cwd=cwd, env=env)
        events.append("run:values")
        _write_json(work_item, {"status": "missing_human_authority"})
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="human metadata authority is incomplete",
        )

    with pytest.raises(ReleaseOrchestratorError, match="values.*exit code 2"):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=block_values,
            gpu_probe=_probe(events),
        )
    assert work_item.is_file()
    values_stage = next(
        stage
        for stage in fixture["manifest_payload"]["stages"]
        if stage["name"] == "values"
    )
    assert {artifact["name"] for artifact in values_stage["artifacts"]} == {
        "receipt"
    }

    resumed_events: list[str] = []
    completed = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        resume=True,
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, resumed_events),
        gpu_probe=_probe(resumed_events),
    )
    assert completed["status"] == "completed"
    assert resumed_events[0] == "run:values"
    assert work_item.is_file()


def test_visual_review_work_item_survives_post_cas_pause_and_resume(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    events: list[str] = []
    normal = _runner(fixture, events)
    attestation = (
        fixture["run_dir"]
        / "manuscript_visual_qa"
        / "VISUAL_QA_ATTESTATION.json"
    )

    def pause_for_human_review(*, command, cwd, env):
        if command[1] != "manuscript_visual_qa":
            return normal(command=command, cwd=cwd, env=env)
        events.append("run:manuscript_visual_qa")
        _write_json(
            attestation,
            {"status": "incomplete_human_page_review_not_for_submission"},
        )
        return SimpleNamespace(
            returncode=3,
            stdout="",
            stderr="human visual QA is required",
        )

    with pytest.raises(
        ReleaseOrchestratorError,
        match="manuscript_visual_qa.*exit code 3",
    ):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=pause_for_human_review,
            gpu_probe=_probe(events),
        )
    assert attestation.is_file()
    assert sha256_file(fixture["official"]) != fixture["expected_official_sha256"]
    visual_stage = next(
        stage
        for stage in fixture["manifest_payload"]["stages"]
        if stage["name"] == "manuscript_visual_qa"
    )
    assert {artifact["name"] for artifact in visual_stage["artifacts"]} == {
        "receipt"
    }

    # The production validator would read the reviewer-edited, sealed work
    # item.  The synthetic runner emits the corresponding successful receipt;
    # the critical integration assertion is that neither the preexisting work
    # item nor the already-applied CAS authority deadlocks resume.
    resumed_events: list[str] = []
    completed = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        resume=True,
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, resumed_events),
        gpu_probe=_probe(resumed_events),
    )
    assert completed["status"] == "completed"
    assert resumed_events[0] == "run:manuscript_visual_qa"
    assert "run:handover_contract" in resumed_events
    assert "run:handover" in resumed_events


def _add_deferred_release_metadata(fixture: dict) -> tuple[Path, Path]:
    """Attach one run-scoped human authority to the real first consumer."""

    workspace = fixture["workspace"]
    run_dir = fixture["run_dir"]
    draft = _write_json(
        workspace / "release_author_metadata_template.json",
        {
            "schema_version": "Synthetic-release-human-metadata-1.0",
            "status": "BLOCKED_TEMPLATE_NOT_AUTHORITY",
            "authors": [{"name": "REQUIRED_AUTHOR_NAME"}],
            "metadata_identity_sha256": "COMPUTE_AFTER_AUTHOR_VERIFICATION",
        },
    )
    target_token = "{run_dir}/human_authorities/release_author_metadata.json"
    fixture["manifest_payload"]["external_inputs"]["release_author_metadata"] = {
        "path": target_token,
        "kind": "file",
        "authority_class": "author_metadata",
        "deferred": True,
        "deferred_contract_schema_version": DEFERRED_HUMAN_AUTHORITY_CONTRACT_SCHEMA,
        "human_authority_id": "synthetic-phaxis-release-author-metadata",
        "document_schema_version": "Synthetic-release-human-metadata-1.0",
        "status_field": "status",
        "final_status": "author_verified_release_authority",
        "identity_field": "metadata_identity_sha256",
        "first_consumer_stage": "source_release",
        "draft_template_path": str(draft.resolve()),
    }
    source_release = next(
        stage
        for stage in fixture["manifest_payload"]["stages"]
        if stage["name"] == "source_release"
    )
    source_release["inputs"].append({"external": "release_author_metadata"})
    source_release["command"].append(target_token)
    fixture["manifest_payload"].pop("manifest_identity_sha256", None)
    fixture["manifest_payload"]["manifest_identity_sha256"] = sha256_json(
        fixture["manifest_payload"]
    )
    _write_json(fixture["manifest"], fixture["manifest_payload"])
    return draft, run_dir / "human_authorities" / "release_author_metadata.json"


def _final_synthetic_release_metadata(target: Path) -> dict:
    payload = {
        "schema_version": "Synthetic-release-human-metadata-1.0",
        "status": "author_verified_release_authority",
        "authors": [{"name": "A. Researcher"}],
    }
    payload["metadata_identity_sha256"] = sha256_json(payload)
    atomic_write_json(target, payload)
    return payload


def _add_future_deferred_authority(
    fixture: dict,
    *,
    external_name: str,
    consumer_stage: str,
) -> None:
    workspace = fixture["workspace"]
    schema = f"Synthetic-{external_name}-1.0"
    identity_field = f"{external_name}_identity_sha256"
    draft = _write_json(
        workspace / f"{external_name}_template.json",
        {
            "schema_version": schema,
            "status": "INCOMPLETE_DO_NOT_USE",
            identity_field: "COMPUTE_AFTER_AUTHOR_VERIFICATION",
        },
    )
    target_token = f"{{run_dir}}/human_authorities/{external_name}.json"
    fixture["manifest_payload"]["external_inputs"][external_name] = {
        "path": target_token,
        "kind": "file",
        "authority_class": "author_metadata",
        "deferred": True,
        "deferred_contract_schema_version": DEFERRED_HUMAN_AUTHORITY_CONTRACT_SCHEMA,
        "human_authority_id": f"synthetic-{external_name}",
        "document_schema_version": schema,
        "status_field": "status",
        "final_status": "author_verified_final",
        "identity_field": identity_field,
        "first_consumer_stage": consumer_stage,
        "draft_template_path": str(draft.resolve()),
    }
    consumer = next(
        stage
        for stage in fixture["manifest_payload"]["stages"]
        if stage["name"] == consumer_stage
    )
    consumer["inputs"].append({"external": external_name})
    consumer["command"].append(target_token)


def test_first_human_pause_batches_all_four_pending_authorities(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    _add_deferred_release_metadata(fixture)
    for external_name, consumer in (
        ("author_release_attestation", "handover_dataset_manifest"),
        ("author_verified_manuscript_metadata", "values"),
        ("submission_title_metadata", "submission_docx"),
    ):
        _add_future_deferred_authority(
            fixture,
            external_name=external_name,
            consumer_stage=consumer,
        )
    fixture["manifest_payload"].pop("manifest_identity_sha256", None)
    fixture["manifest_payload"]["manifest_identity_sha256"] = sha256_json(
        fixture["manifest_payload"]
    )
    _write_json(fixture["manifest"], fixture["manifest_payload"])

    paused = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, []),
        gpu_probe=_probe([]),
    )
    expected = {
        "release_author_metadata",
        "author_release_attestation",
        "author_verified_manuscript_metadata",
        "submission_title_metadata",
    }
    assert paused["status"] == "paused_for_deferred_human_authority"
    assert paused["human_authority_gate"][
        "all_pending_deferred_authority_count"
    ] == 4
    batched = paused["all_pending_deferred_authorities"]
    assert {item["external_name"] for item in batched} == expected
    assert batched == paused["human_authority_gate"][
        "all_pending_deferred_authorities"
    ]
    assert all(item["work_item_is_not_success_artifact"] is True for item in batched)
    assert all(Path(item["work_item_path"]).is_file() for item in batched)
    assert all(Path(item["target_path"]).is_file() for item in batched)
    assert all(
        not _sentinel_path_for_test(
            fixture["run_dir"],
            MANDATORY_STAGE_ORDER.index(item["first_consumer_stage"]),
            item["first_consumer_stage"],
        ).exists()
        for item in batched
    )

    for item in batched:
        descriptor = fixture["manifest_payload"]["external_inputs"][
            item["external_name"]
        ]
        final_payload = {
            "schema_version": descriptor["document_schema_version"],
            descriptor["status_field"]: descriptor["final_status"],
            "verified_by": "Synthetic Author",
        }
        final_payload[descriptor["identity_field"]] = sha256_json(final_payload)
        atomic_write_json(Path(item["target_path"]), final_payload)
    completed = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        resume=True,
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, []),
        gpu_probe=_probe([]),
    )
    assert completed["status"] == "completed"
    assert completed["all_pending_deferred_authorities"] == []
    for item in batched:
        consumer_index = MANDATORY_STAGE_ORDER.index(item["first_consumer_stage"])
        sentinel = read_json(
            _sentinel_path_for_test(
                fixture["run_dir"],
                consumer_index,
                item["first_consumer_stage"],
            )
        )
        activation = next(
            lock
            for lock in sentinel["input_locks"]
            if lock.get("external") == item["external_name"]
        )
        assert activation["deferred_authority_activated"] is True
        assert activation["sha256"] == sha256_file(Path(item["target_path"]))


def test_deferred_human_authority_executes_science_prefix_pauses_then_resumes(
    tmp_path: Path,
) -> None:
    fixture = _manifest_fixture(tmp_path)
    draft, target = _add_deferred_release_metadata(fixture)
    draft_sha = sha256_file(draft)
    manifest_external = fixture["manifest_payload"]["external_inputs"][
        "release_author_metadata"
    ]
    assert "sha256" not in manifest_external
    assert draft_sha not in fixture["manifest"].read_text(encoding="utf-8")

    events: list[str] = []
    paused = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, events),
        gpu_probe=_probe(events),
    )

    assert paused["status"] == "paused_for_deferred_human_authority"
    assert paused["algorithm_or_training_failure"] is False
    assert paused["official_apply_sentinel_committed"] is True
    assert paused["release_finalize_sentinel_committed"] is False
    gate = paused["human_authority_gate"]
    assert gate["expected_process_exit_code"] == EXPECTED_HUMAN_GATE_EXIT_CODE
    assert gate["work_item_is_not_success_artifact"] is True
    assert events[-1] == "run:official_apply"
    assert "run:source_release" not in events
    source_index = MANDATORY_STAGE_ORDER.index("source_release")
    assert not _sentinel_path_for_test(
        fixture["run_dir"], source_index, "source_release"
    ).exists()
    work_item_path = Path(gate["work_item_path"])
    work_item = read_json(work_item_path)
    assert work_item["schema_version"] == DEFERRED_HUMAN_WORK_ITEM_SCHEMA
    assert work_item["counts_as_completed_stage"] is False
    assert work_item["counts_as_formal_release_success"] is False
    assert work_item["work_item_is_not_success_artifact"] is True
    assert target.is_file()
    assert read_json(target)["status"] == "BLOCKED_TEMPLATE_NOT_AUTHORITY"
    source_stage = next(
        stage
        for stage in fixture["manifest_payload"]["stages"]
        if stage["name"] == "source_release"
    )
    assert all(
        artifact["path"] != str(work_item_path)
        for artifact in source_stage["artifacts"]
    )

    _final_synthetic_release_metadata(target)
    resumed_events: list[str] = []
    completed = execute_release(
        fixture["manifest"],
        fixture["run_dir"],
        resume=True,
        candidate_builder=_cached_builder(fixture["preview"]),
        command_runner=_runner(fixture, resumed_events),
        gpu_probe=_probe(resumed_events),
    )
    assert completed["status"] == "completed"
    assert resumed_events[0] == "run:source_release"
    source_sentinel = read_json(
        _sentinel_path_for_test(
            fixture["run_dir"], source_index, "source_release"
        )
    )
    activation = next(
        lock
        for lock in source_sentinel["input_locks"]
        if lock.get("external") == "release_author_metadata"
    )
    assert activation["deferred_authority_activated"] is True
    assert activation["exact_file_hash_locked"] is True
    assert activation["sha256"] == sha256_file(target)
    assert activation["logical_identity_sha256"] == read_json(target)[
        "metadata_identity_sha256"
    ]

    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(
        ReleaseOrchestratorError,
        match="exact bytes drifted",
    ):
        execute_release(
            fixture["manifest"],
            fixture["run_dir"],
            resume=True,
            candidate_builder=_cached_builder(fixture["preview"]),
            command_runner=_runner(fixture, []),
            gpu_probe=_probe([]),
        )


def _sentinel_path_for_test(run_dir: Path, index: int, name: str) -> Path:
    return run_dir / "sentinels" / f"{index:02d}_{name}.json"


def test_deferred_human_authority_rejects_nonhuman_class_and_final_tamper(
    tmp_path: Path,
) -> None:
    nonhuman = _manifest_fixture(tmp_path / "nonhuman")
    _draft, _target = _add_deferred_release_metadata(nonhuman)
    nonhuman["manifest_payload"]["external_inputs"]["release_author_metadata"][
        "authority_class"
    ] = "static_contract"
    nonhuman["manifest_payload"].pop("manifest_identity_sha256", None)
    nonhuman["manifest_payload"]["manifest_identity_sha256"] = sha256_json(
        nonhuman["manifest_payload"]
    )
    _write_json(nonhuman["manifest"], nonhuman["manifest_payload"])
    with pytest.raises(ReleaseOrchestratorError, match="only author_metadata"):
        build_release_plan(
            nonhuman["manifest"],
            nonhuman["run_dir"],
            candidate_builder=_cached_builder(nonhuman["preview"]),
        )

    tampered = _manifest_fixture(tmp_path / "tampered")
    _draft, target = _add_deferred_release_metadata(tampered)
    paused = execute_release(
        tampered["manifest"],
        tampered["run_dir"],
        candidate_builder=_cached_builder(tampered["preview"]),
        command_runner=_runner(tampered, []),
        gpu_probe=_probe([]),
    )
    assert paused["status"] == "paused_for_deferred_human_authority"
    payload = _final_synthetic_release_metadata(target)
    payload["authors"] = [{"name": "Changed after sealing"}]
    atomic_write_json(target, payload)
    with pytest.raises(
        ReleaseOrchestratorError,
        match="does not seal the complete JSON object",
    ):
        execute_release(
            tampered["manifest"],
            tampered["run_dir"],
            resume=True,
            candidate_builder=_cached_builder(tampered["preview"]),
            command_runner=_runner(tampered, []),
            gpu_probe=_probe([]),
        )


@pytest.mark.parametrize("leak_kind", ("input", "argv"))
def test_preconsumer_stage_cannot_reference_deferred_human_authority(
    tmp_path: Path, leak_kind: str
) -> None:
    fixture = _manifest_fixture(tmp_path)
    _add_deferred_release_metadata(fixture)
    prior = next(
        stage
        for stage in fixture["manifest_payload"]["stages"]
        if stage["name"] == "evidence"
    )
    if leak_kind == "input":
        prior["inputs"].append({"external": "release_author_metadata"})
    else:
        prior["command"].append(
            "{run_dir}/human_authorities/release_author_metadata.json"
        )
    fixture["manifest_payload"].pop("manifest_identity_sha256", None)
    fixture["manifest_payload"]["manifest_identity_sha256"] = sha256_json(
        fixture["manifest_payload"]
    )
    _write_json(fixture["manifest"], fixture["manifest_payload"])
    with pytest.raises(
        ReleaseOrchestratorError,
        match=(
            "first-consumer contract drifted"
            if leak_kind == "input"
            else "pre-consumer stage references"
        ),
    ):
        build_release_plan(
            fixture["manifest"],
            fixture["run_dir"],
            candidate_builder=_cached_builder(fixture["preview"]),
        )


def test_cli_requires_explicit_execute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script_root = Path(__file__).resolve().parents[2] / "scripts" / "phaxis"
    sys.path.insert(0, str(script_root))
    try:
        import run_post_training_release as cli
    finally:
        sys.path.remove(str(script_root))
    calls: list[str] = []
    monkeypatch.setattr(
        cli,
        "build_release_plan",
        lambda *_args, **_kwargs: calls.append("plan") or {"status": "plan"},
    )
    monkeypatch.setattr(
        cli,
        "execute_release",
        lambda *_args, **_kwargs: calls.append("execute") or {"status": "completed"},
    )
    assert cli.main(["--manifest", str(tmp_path / "m.json"), "--output", str(tmp_path / "o")]) == 0
    assert calls == ["plan"]
    assert cli.main(["--manifest", str(tmp_path / "m.json"), "--output", str(tmp_path / "o"), "--execute"]) == 0
    assert calls == ["plan", "execute"]
    captured_holds: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        cli,
        "execute_release",
        lambda *_args, **kwargs: (
            captured_holds.append(tuple(kwargs["held_physical_gpus"]))
            or {"status": "paused_for_user_gpu_hold"}
        ),
    )
    assert (
        cli.main(
            [
                "--manifest",
                str(tmp_path / "m.json"),
                "--output",
                str(tmp_path / "o"),
                "--execute",
                "--hold-physical-gpu",
                "0",
            ]
        )
        == EXPECTED_GPU_HOLD_EXIT_CODE
    )
    assert captured_holds == [(0,)]
    monkeypatch.setattr(
        cli,
        "execute_release",
        lambda *_args, **_kwargs: {
            "status": "paused_for_deferred_human_authority"
        },
    )
    assert (
        cli.main(
            [
                "--manifest",
                str(tmp_path / "m.json"),
                "--output",
                str(tmp_path / "o"),
                "--execute",
                "--resume",
            ]
        )
        == EXPECTED_HUMAN_GATE_EXIT_CODE
    )
    with pytest.raises(SystemExit):
        cli.main(["--manifest", "m", "--output", "o", "--resume"])
    with pytest.raises(SystemExit):
        cli.main(
            [
                "--manifest",
                "m",
                "--output",
                "o",
                "--hold-physical-gpu",
                "0",
            ]
        )
