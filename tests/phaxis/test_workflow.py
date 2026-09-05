from __future__ import annotations

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from phaxis.contracts import ContractError
from phaxis.io import sha256_file, sha256_json
from phaxis.public_identity import (
    PUBLIC_SYSTEM_DERIVATION,
    PUBLIC_SYSTEM_IDENTITY_SCHEMA,
    derive_public_identity,
)
import phaxis.workflow as workflow
import phaxis.cli as cli_module


def _write(path: Path, value: bytes = b"locked") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _ref(path: Path, base: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": sha256_file(path),
    }


def _manifest(tmp_path: Path) -> Path:
    image_root = tmp_path / "images"
    image = _write(image_root / "T1.tif", b"source-image")
    image_sha = sha256_file(image)
    root_input = _csv(
        tmp_path / "root_input.csv",
        ("image_id", "input_path", "source_um_per_px"),
        [{"image_id": "T1", "input_path": str(image), "source_um_per_px": 1.0}],
    )
    stageb_input = _csv(
        tmp_path / "stageb_input.csv",
        ("task_id", "image_path", "image_sha256", "um_per_px"),
        [{
            "task_id": "T1",
            "image_path": str(image),
            "image_sha256": image_sha,
            "um_per_px": 1.0,
        }],
    )
    traits = _csv(
        tmp_path / "traits_metadata.csv",
        ("task_id", "image_sha256", "um_per_px", "condition_code"),
        [{
            "task_id": "T1",
            "image_sha256": image_sha,
            "um_per_px": 1.0,
            "condition_code": "never_a_route",
        }],
    )
    placeholders = {
        name: _write(tmp_path / f"{name}.lock", name.encode())
        for name in (
            "acquisition_gate",
            "deployment_metadata",
            "canonical_manifest",
            "deployment_manifest",
            "deployment_lock",
        )
    }
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": "PHAxis-distal-axis-profile-contract-1.0.0",
                "bins_um": [[0, 1000]],
                "root_cap_region_output": False,
                "stageb_two_point_vector_used_as_length": False,
                "canonical_annotations_read": False,
                "blind_images_used": 0,
            }
        ),
        encoding="utf-8",
    )
    checkpoints = [_write(tmp_path / f"member{index}.pt", f"cp{index}".encode()) for index in range(5)]
    checkpoint_sha256 = [sha256_file(path) for path in checkpoints]
    candidate = tmp_path / "candidate.json"
    selected = tmp_path / "selected.json"
    receipt = tmp_path / "selection.json"
    expert = "PHAxis-StageB-train399-five-seed"
    candidate_identity = sha256_json({"fixture": "candidate"})
    candidate_payload = {
        "candidate_bundle_identity_sha256": candidate_identity,
        "detection_model_metadata": {
            "expert_id": expert,
            "deployment_role": "candidate_gate_passed_not_promoted",
            "checkpoint_sha256": checkpoint_sha256,
        },
    }
    candidate.write_text(json.dumps(candidate_payload), encoding="utf-8")
    receipt_payload = {
        "selected": {"threshold": 0.225},
    }
    receipt_payload["selection_receipt_identity_sha256"] = sha256_json(receipt_payload)
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    selected_payload = {
        "checkpoint_policy": "five_seed_train399_last_epoch_60",
        "ensemble_members": 5,
        "training_images": 399,
        "blind_images_used": 0,
        "expert_id": expert,
        "deployment_role": "candidate_gate_passed_not_promoted",
        "operating_point_status": "selected_on_locked_QCdevelopment44",
        "selected_score_threshold": 0.225,
        "checkpoint_sha256": checkpoint_sha256,
        "candidate_bundle_identity_sha256": candidate_identity,
        "selection_receipt_identity_sha256": receipt_payload[
            "selection_receipt_identity_sha256"
        ],
    }
    selected_payload["selected_model_metadata_identity_sha256"] = sha256_json(
        selected_payload
    )
    selected.write_text(json.dumps(selected_payload), encoding="utf-8")
    stageb_binding = {
        "expert_id": expert,
        "checkpoint_sha256": checkpoint_sha256,
        "selected_score_threshold": 0.225,
        "candidate_bundle_identity_sha256": candidate_identity,
        "selection_receipt_identity_sha256": receipt_payload[
            "selection_receipt_identity_sha256"
        ],
        "selected_model_metadata_identity_sha256": selected_payload[
            "selected_model_metadata_identity_sha256"
        ],
    }
    root_audit = "9" * 64
    root_pipeline = "8" * 64
    root_bundle = "b" * 64
    public = derive_public_identity(
        stageb_binding,
        root_bundle_identity_sha256=root_bundle,
    )
    proposal_payload = {
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
            "fresh_exact283_audit_identity_sha256": root_audit,
            "bundle_identity_sha256": root_bundle,
            "pipeline_identity_sha256": root_pipeline,
            "root_bundle_authority": {
                "binding": "transitively_sealed_by_fresh_exact283_pipeline_identity",
                "bundle_identity_sha256": root_bundle,
                "pipeline_identity_sha256": root_pipeline,
            },
        },
        "expert_boundary": {
            "root_point_scale_continuity_statistics": public["root_expert_id"],
            "hair_identity_and_count": expert,
        },
        "formal_release_status": "passed_proposal_not_official",
        "hair_identity_count_expert": {
            "current_checkpoint_role": "formal_train399_only_deployment_candidate",
            "deployment_ensemble_used_qcdev44_labels_in_some_members": False,
            "strict_train399_only_retraining_gate": "passed_proposal_not_official",
            "score_threshold": 0.225,
            "checkpoint_sha256_in_member_order": checkpoint_sha256,
            "expert_id": expert,
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
            "source_model_contract_sha256": "a" * 64,
            "formal_gate_source_sha256": {
                "train399_candidate": sha256_file(candidate),
                "train399_selection": sha256_file(receipt),
                "train399_evaluation": "e" * 64,
                "root_exact283": "f" * 64,
            },
            "formal_gate_identity_sha256": {
                "candidate_bundle_identity_sha256": candidate_identity,
                "selection_receipt_identity_sha256": receipt_payload[
                    "selection_receipt_identity_sha256"
                ],
                "selected_model_metadata_identity_sha256": selected_payload[
                    "selected_model_metadata_identity_sha256"
                ],
                "root_exact283_audit_identity_sha256": root_audit,
            },
            "checkpoint_file_sha256_in_member_order": checkpoint_sha256,
            "stageb_binding": stageb_binding,
        },
    }
    proposal_payload["model_contract_identity_sha256"] = sha256_json(proposal_payload)
    proposal = tmp_path / "model_contract_proposal.json"
    proposal.write_text(json.dumps(proposal_payload), encoding="utf-8")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    bundle_identity = "b" * 64
    registry = bundle / "root_provider_bundle.json"
    registry.write_text(
        json.dumps({"bundle_identity_sha256": bundle_identity}), encoding="utf-8"
    )
    project = tmp_path / "installed_project_cwd"
    project.mkdir()
    payload: dict[str, object] = {
        "schema_version": workflow.WORKFLOW_MANIFEST_SCHEMA,
        "model_contract_proposal": _ref(proposal, tmp_path),
        "root_provider": {
            "project": project.relative_to(tmp_path).as_posix(),
            "bundle": {
                "path": bundle.relative_to(tmp_path).as_posix(),
                "registry_sha256": sha256_file(registry),
                "bundle_identity_sha256": bundle_identity,
            },
            "input_manifest": _ref(root_input, tmp_path),
            **{name: _ref(path, tmp_path) for name, path in placeholders.items()},
            "image_root": image_root.relative_to(tmp_path).as_posix(),
            "python_executable": str(Path(sys.executable).resolve()),
            "v1_physical_gpus": [0],
            "q8_physical_gpus": [1],
            "v1_shards": 1,
            "v20_shards": 1,
            "q8_shards": 1,
            "v1_concurrency": 1,
            "v20_concurrency": 1,
            "q8_concurrency": 1,
        },
        "stageb": {
            "input_manifest": _ref(stageb_input, tmp_path),
            "checkpoints": [_ref(path, tmp_path) for path in checkpoints],
            "candidate_manifest": _ref(candidate, tmp_path),
            "selected_model_metadata": _ref(selected, tmp_path),
            "selection_receipt": _ref(receipt, tmp_path),
            "physical_gpu": 1,
            "internal_device": "cuda:0",
            "shared_input_acceleration": False,
        },
        "traits": {"metadata_csv": _ref(traits, tmp_path)},
        "distal_axis_profiles": {"contract_json": _ref(profile, tmp_path)},
        "review_overlays": {"enabled": False},
        "guards": {
            "condition_metadata_used_for_routing": False,
            "canonical_annotations_read": False,
            "blind_images_used": 0,
            "root_cap_region_output": False,
        },
    }
    payload["manifest_identity_sha256"] = sha256_json(payload)
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _apply_model_contract_in_manifest(manifest: Path) -> dict[str, str]:
    """Apply the exact production lifecycle transform to the locked fixture."""

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    contract_ref = manifest_payload["model_contract_proposal"]
    contract = manifest.parent / contract_ref["path"]
    proposal = json.loads(contract.read_text(encoding="utf-8"))
    proposal_bytes = (
        json.dumps(
            proposal,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    contract.write_bytes(proposal_bytes)
    proposal_file_sha256 = sha256_file(contract)
    proposal_identity_sha256 = proposal["model_contract_identity_sha256"]

    official = deepcopy(proposal)
    official.pop("model_contract_identity_sha256")
    official["formal_release_status"] = "passed"
    official["hair_identity_count_expert"]["current_checkpoint_role"] = (
        "formal_train399_only_deployment"
    )
    official["hair_identity_count_expert"][
        "strict_train399_only_retraining_gate"
    ] = "passed"
    public_identity = {
        "model_bundle_id": official["model_bundle_id"],
        "root_expert_id": official["root_expert"]["expert_id"],
    }
    official["promotion"].update(
        {
            "status": "applied_formal_release",
            "official_apply_performed": True,
            "proposal_file_sha256": proposal_file_sha256,
            "proposal_identity_sha256": proposal_identity_sha256,
            "expected_source_model_contract_sha256": official["promotion"][
                "source_model_contract_sha256"
            ],
            "final_receipt_source_sha256": {
                role: sha256_json({"source": role})
                for role in ("stageb", "fusion", "traits", "evidence")
            },
            "final_receipt_identity_sha256": {
                role: sha256_json({"identity": role})
                for role in ("stageb", "fusion", "traits", "evidence")
            },
            "final_receipt_public_identity": {
                role: dict(public_identity) for role in ("fusion", "traits")
            },
        }
    )
    official["model_contract_identity_sha256"] = sha256_json(official)
    contract.write_bytes(
        (
            json.dumps(
                official,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )
    contract_ref["sha256"] = sha256_file(contract)
    manifest_payload.pop("manifest_identity_sha256")
    manifest_payload["manifest_identity_sha256"] = sha256_json(manifest_payload)
    manifest.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return {
        "model_contract_proposal_sha256": proposal_file_sha256,
        "model_contract_proposal_identity_sha256": proposal_identity_sha256,
        **public_identity,
    }


@pytest.fixture
def fake_gate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        workflow,
        "_validate_train399_gate",
        lambda **kwargs: (
            json.loads(Path(kwargs["candidate_manifest"]).read_text(encoding="utf-8")),
            json.loads(Path(kwargs["selected_model_metadata"]).read_text(encoding="utf-8")),
        ),
    )


def test_partial_train399_gate_fails_before_any_gpu_preflight(tmp_path: Path, monkeypatch):
    payload = {
        "schema_version": workflow.WORKFLOW_MANIFEST_SCHEMA,
        "stageb": {"candidate_manifest": {"path": "candidate.json", "sha256": "a" * 64}},
        "guards": {
            "condition_metadata_used_for_routing": False,
            "canonical_annotations_read": False,
            "blind_images_used": 0,
            "root_cap_region_output": False,
        },
    }
    payload["manifest_identity_sha256"] = sha256_json(payload)
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        workflow,
        "_gpu_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("GPU preflight must not run")
        ),
    )
    with pytest.raises(ContractError, match="requires candidate_manifest"):
        workflow.build_analysis_plan(path, output=tmp_path / "output")


def test_plan_is_deterministic_complete_and_side_effect_free(
    tmp_path: Path, fake_gate
) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "analysis"
    first = workflow.build_analysis_plan(manifest, output=output)
    second = workflow.build_analysis_plan(manifest, output=output)
    assert first == second
    assert first["status"] == "planned_not_executed"
    assert first["default_plan_only"] is True
    assert first["guards"]["condition_metadata_used_for_routing"] is False
    assert not output.exists()
    assert [stage["name"] for stage in first["stages"]] == [
        "root_provider",
        "stageb_train399",
        "fusion",
        "traits",
        "distal_axis_profiles",
    ]
    assert all(stage["input_hashes"] for stage in first["stages"])
    assert all(stage["output"] for stage in first["stages"])
    assert all("estimated_gpu" in stage for stage in first["stages"])
    stageb = first["stages"][1]
    assert stageb["detail"]["shared_input_acceleration_requested"] is False
    assert stageb["detail"]["shared_input_acceleration_default_enabled"] is False
    assert stageb["estimated_gpu"]["physical_gpu"] == 1
    assert stageb["estimated_gpu"]["internal_device"] == "cuda:0"
    assert all(
        stage["input_hashes"]["model_contract_proposal_sha256"]
        == first["model_contract_proposal_sha256"]
        for stage in first["stages"]
    )
    selected_path = tmp_path / "selected.json"
    assert json.loads(selected_path.read_text(encoding="utf-8"))["deployment_role"] == (
        "candidate_gate_passed_not_promoted"
    )


def test_analyze_plan_only_accepts_applied_official_contract(
    tmp_path: Path, fake_gate, capsys
) -> None:
    manifest = _manifest(tmp_path)
    expected_identity = _apply_model_contract_in_manifest(manifest)
    output = tmp_path / "official-plan-only-output"

    assert cli_module.main(
        [
            "analyze",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ]
    ) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "planned_not_executed"
    assert not output.exists()
    for field, expected in expected_identity.items():
        assert plan[field] == expected


def test_atomic_state_resume_and_overwrite_guards(
    tmp_path: Path, fake_gate, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "analysis"
    calls: list[str] = []

    def fake_execute(context, stage: str, *, resume: bool) -> None:
        calls.append(stage)
        destination = context.output / (
            "root_provider" if stage == "root_provider" else
            "stageb" if stage == "stageb_train399" else stage
        )
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "oracle.json").write_text(
            json.dumps({"stage": stage, "blind_images_used": 0}), encoding="utf-8"
        )

    monkeypatch.setattr(workflow, "_execute_stage", fake_execute)
    completed = workflow.run_analysis(manifest, output=output)
    assert completed["status"] == "completed"
    assert completed["completed_stages"] == [
        "root_provider",
        "stageb_train399",
        "fusion",
        "traits",
        "distal_axis_profiles",
    ]
    assert (output / "workflow_state.json").is_file()
    assert completed["latest_execution_fresh_direct_benchmark_eligible"] is True
    first_attempt = completed["execution_attempts"][0]
    assert first_attempt["fresh_direct_benchmark_eligible"] is True
    assert first_attempt["resume_or_cache_used"] is False
    assert [row["execution_status"] for row in first_attempt["stages"]] == [
        "executed_fresh"
    ] * 5
    with pytest.raises(FileExistsError, match="pass --resume"):
        workflow.run_analysis(manifest, output=output)
    calls.clear()
    resumed = workflow.run_analysis(manifest, output=output, resume=True)
    assert resumed["status"] == "completed"
    assert calls == []
    assert resumed["latest_execution_fresh_direct_benchmark_eligible"] is False
    assert resumed["execution_attempts"][-1]["resume_or_cache_used"] is True
    assert all(
        row["execution_status"] == "cached_completed_evidence_validated"
        for row in resumed["execution_attempts"][-1]["stages"]
    )
    (output / "traits" / "oracle.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ContractError, match="output tree drift"):
        workflow.run_analysis(manifest, output=output, resume=True)


def test_gpu_preflight_rejects_early_torch_without_running_nvidia_smi(monkeypatch):
    sentinel = object()
    previous = sys.modules.get("torch")
    sys.modules["torch"] = sentinel  # type: ignore[assignment]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("nvidia-smi must not run after an early torch import")
        ),
    )
    config = workflow.StageBBatchConfig(
        input_manifest=Path("unused.csv"),
        checkpoints=tuple(Path(f"cp{index}") for index in range(5)),
        candidate_manifest=Path("candidate.json"),
        selected_model_metadata=Path("selected.json"),
        selection_receipt=Path("receipt.json"),
        physical_gpu=1,
    )
    try:
        with pytest.raises(ContractError, match="imported before mandatory"):
            workflow._gpu_preflight(config)
    finally:
        if previous is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous


def test_analyze_cli_is_plan_only_until_execute_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls: list[str] = []

    def fake_plan(*_args, **_kwargs):
        calls.append("plan")
        return {"status": "planned_not_executed", "blind_images_used": 0}

    def fake_run(*_args, **_kwargs):
        calls.append("execute")
        return {"status": "completed", "blind_images_used": 0}

    monkeypatch.setattr(workflow, "build_analysis_plan", fake_plan)
    monkeypatch.setattr(workflow, "run_analysis", fake_run)
    base = [
        "analyze",
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--output",
        str(tmp_path / "output"),
    ]
    assert cli_module.main(base) == 0
    assert calls == ["plan"]
    assert "planned_not_executed" in capsys.readouterr().out
    calls.clear()
    assert cli_module.main([*base, "--execute"]) == 0
    assert calls == ["execute"]
    calls.clear()
    assert cli_module.main([*base, "--resume"]) == 2
    assert calls == []
    assert "only together with explicit --execute" in capsys.readouterr().err


def test_completed_stageb_summary_resume_is_validated_without_gpu(
    tmp_path: Path,
    fake_gate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = workflow._context(
        _manifest(tmp_path), output=tmp_path / "analysis", review_overlays=None
    )
    config = context.stageb_config
    rows = context.stageb_rows
    output = tmp_path / "stageb-complete"
    detection = output / "detections" / "T1.json"
    detection.parent.mkdir(parents=True)
    detection.write_text(
        json.dumps(context.model_contract_binding.output_identity_fields()),
        encoding="utf-8",
    )
    summary = {
        "schema_version": workflow.STAGEB_BATCH_SCHEMA,
        "status": "completed",
        "batch_identity_sha256": workflow._stageb_batch_identity(config, rows),
        "records": [
            {
                "task_id": "T1",
                "detection_file_sha256": sha256_file(detection),
            }
        ],
        "shared_input_acceleration": {
            "requested": False,
            "default_enabled": False,
        },
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read": False,
        "root_cap_region_output": False,
        "blind_images_used": 0,
            **context.model_contract_binding.output_identity_fields(),
    }
    summary["summary_identity_sha256"] = sha256_json(summary)
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    validations: list[str] = []
    monkeypatch.setattr(
        workflow,
        "validate_stageb_detection_payload",
        lambda _payload, **kwargs: validations.append(kwargs["expected_task_id"]),
    )
    monkeypatch.setattr(
        workflow,
        "_gpu_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed-summary resume must not touch GPU")
        ),
    )
    resumed = workflow.run_stageb_batch(
        config,
        output=output,
        resume=True,
        model_contract_binding=context.model_contract_binding,
    )
    assert resumed["status"] == "completed"
    assert validations == ["T1"]


def test_optional_overlay_stage_is_explicit_and_never_routes(
    tmp_path: Path, fake_gate
) -> None:
    manifest = _manifest(tmp_path)
    default = workflow.build_analysis_plan(manifest, output=tmp_path / "default")
    enabled = workflow.build_analysis_plan(
        manifest, output=tmp_path / "enabled", review_overlays=True
    )
    assert "review_overlays" not in [stage["name"] for stage in default["stages"]]
    overlay = enabled["stages"][-1]
    assert overlay["name"] == "review_overlays"
    assert overlay["estimated_gpu"] == {"required": False}
    assert overlay["detail"]["optional_review_artifact"] is True
    assert overlay["detail"]["used_for_model_routing"] is False


def test_plan_rejects_manifest_and_source_image_tamper(
    tmp_path: Path, fake_gate
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["review_overlays"]["enabled"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="does not seal"):
        workflow.build_analysis_plan(manifest, output=tmp_path / "manifest-tamper")

    manifest = _manifest(tmp_path / "image_case")
    image = tmp_path / "image_case" / "images" / "T1.tif"
    image.write_bytes(b"tampered-image")
    with pytest.raises(ContractError, match="source-image hash mismatch"):
        workflow.build_analysis_plan(manifest, output=tmp_path / "image-tamper")


def test_plan_rejects_model_contract_proposal_tamper(
    tmp_path: Path, fake_gate
) -> None:
    manifest = _manifest(tmp_path)
    proposal = tmp_path / "model_contract_proposal.json"
    payload = json.loads(proposal.read_text(encoding="utf-8"))
    payload["formal_release_status"] = "formally_promoted"
    proposal.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="locked input hash mismatch"):
        workflow.build_analysis_plan(manifest, output=tmp_path / "output")


def test_fusion_predictions_and_summary_inherit_proposal_without_promoting_metadata(
    tmp_path: Path, fake_gate, phaxis_case
) -> None:
    hybrid, stageb, artifact_root = phaxis_case
    context = workflow._context(
        _manifest(tmp_path / "locks"),
        output=tmp_path / "analysis",
        review_overlays=None,
    )
    hybrid_root = artifact_root
    atomic_predictions = hybrid_root / "predictions"
    atomic_predictions.mkdir()
    (atomic_predictions / "T1.json").write_text(json.dumps(hybrid), encoding="utf-8")
    stageb_root = tmp_path / "stageb"
    (stageb_root / "detections").mkdir(parents=True)
    stageb.pop("detection_identity_sha256", None)
    stageb.update(context.model_contract_binding.output_identity_fields())
    stageb["detection_identity_sha256"] = sha256_json(stageb)
    (stageb_root / "detections/T1.json").write_text(
        json.dumps(stageb), encoding="utf-8"
    )
    (stageb_root / "summary.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "fusion"
    output.mkdir()
    summary = workflow._run_fusion_batch(
        hybrid_root=hybrid_root,
        stageb_root=stageb_root,
        output=output,
        model_contract_binding=context.model_contract_binding,
    )
    fused = json.loads((output / "predictions/T1.json").read_text(encoding="utf-8"))
    for field, expected in context.model_contract_binding.receipt_fields().items():
        assert summary[field] == expected
        assert fused["phaxis"][field] == expected
    selected = json.loads(
        context.stageb_config.selected_model_metadata.read_text(encoding="utf-8")
    )
    assert selected["deployment_role"] == "candidate_gate_passed_not_promoted"


def test_plan_gate_validation_imports_no_gpu_or_image_runtime() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source = r'''
import builtins
import sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"torch", "torchvision", "cv2", "tifffile"}:
        raise AssertionError("optional GPU/image runtime imported during plan Gate validation: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import phaxis.workflow
import phaxis.hair_stageb.selection
assert "phaxis.hair_stageb.training_data" not in sys.modules
assert "torch" not in sys.modules
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_plan_rejects_root_and_stageb_images_with_same_task_id(
    tmp_path: Path, fake_gate
) -> None:
    manifest = _manifest(tmp_path)
    other = _write(tmp_path / "images" / "other.tif", b"different-root-image")
    root_input = tmp_path / "root_input.csv"
    _csv(
        root_input,
        ("image_id", "input_path", "source_um_per_px"),
        [{"image_id": "T1", "input_path": str(other), "source_um_per_px": 1.0}],
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["root_provider"]["input_manifest"]["sha256"] = sha256_file(root_input)
    payload.pop("manifest_identity_sha256")
    payload["manifest_identity_sha256"] = sha256_json(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="root-provider/Stage-B source-image mismatch"):
        workflow.build_analysis_plan(manifest, output=tmp_path / "output")
