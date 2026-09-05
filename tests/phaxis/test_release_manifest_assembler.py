from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

from phaxis.io import sha256_file, sha256_json
from phaxis.release_manifest_assembler import (
    ASSEMBLY_CONFIG_SCHEMA,
    STAGE_TEMPLATE_SCHEMA,
    ReleaseManifestAssemblyError,
    assemble_release_manifest,
    inspect_release_readiness,
    _materialize_template_value,
)
from phaxis.release_orchestrator import (
    EXPECTED_GPU_HOLD_EXIT_CODE,
    KNOWN_STAGE_SCHEMAS,
    build_release_plan,
)
from phaxis.release_topology import FORMAL_RELEASE_PRODUCERS, MANDATORY_STAGE_ORDER
from scripts.phaxis import assemble_post_training_release_manifest as assembly_cli
from scripts.phaxis import build_post_training_release_stage_contract as contract_builder


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_current_workspace_assembly_config_targets_release_control_1_9() -> None:
    config_path = (
        PROJECT_ROOT
        / "configs/phaxis/v1_0/post_training_release_assembly_config_1_9.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["stage_contract_template"] == (
        "configs/phaxis/v1_0/post_training_release_stage_contract_1_9.json"
    )
    assert config["output_root"] == (
        "outputs/phaxis_post_training_formal_release_run1_9"
    )
    assert config["manifest_fields"]["product"] == "PHAxis"
    assert config["manifest_fields"]["product_version"] == "1.0.0"
    assert config["manifest_fields"]["run_id"] == (
        "PHAxis-1.0.0-post-training-formal-release"
    )


def test_current_qcdev_hybrid_authority_is_exact_locked_legacy_comparator() -> None:
    config = json.loads(
        (
            PROJECT_ROOT
            / "configs/phaxis/v1_0/post_training_release_assembly_config_1_9.json"
        ).read_text(encoding="utf-8")
    )
    authority = config["authorities"]["qcdev_hybrid_predictions"]
    expected_relative_path = (
        "outputs/rhaxis_nextgen_hybrid_max_qcdev44_run4_final/predictions"
    )
    prohibited_fused_path = (
        "outputs/phaxis_v1_0_qcdev44_fusion_run3_final_contract/predictions"
    )
    assert authority == {
        "authority_class": "frozen_read_only_asset",
        "kind": "directory",
        "path": expected_relative_path,
    }
    assert authority["path"] != prohibited_fused_path

    prediction_root = PROJECT_ROOT / authority["path"]
    locked_ids_path = PROJECT_ROOT / config["authorities"]["locked_val_ids"]["path"]
    locked_ids = [
        line.strip()
        for line in locked_ids_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(locked_ids) == len(set(locked_ids)) == 44

    prediction_paths = sorted(
        prediction_root.glob("*.json"), key=lambda path: path.name.casefold()
    )
    assert len(prediction_paths) == 44
    assert {path.name for path in prediction_paths} == {
        f"{task_id}.json" for task_id in locked_ids
    }
    assert {path.resolve() for path in prediction_root.iterdir()} == {
        path.resolve() for path in prediction_paths
    }

    manifest_path = PROJECT_ROOT / config["authorities"]["qcdev_manifest"]["path"]
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_by_task = {row["task_id"]: row for row in manifest_rows}
    assert set(locked_ids) <= set(manifest_by_task)

    ordered_prediction_file_locks = []
    for task_id in locked_ids:
        prediction_path = prediction_root / f"{task_id}.json"
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        expected_source_sha256 = (
            manifest_by_task[task_id].get("image_sha256")
            or manifest_by_task[task_id].get("source_image_sha256")
        )
        assert payload["schema_version"] == (
            "RHAxis-NextGen-Hybrid-Max-qc-development-evaluation-1.0"
        )
        assert payload["task_id"] == task_id
        assert payload["source_image_sha256"] == expected_source_sha256
        assert len(payload["source_image_sha256"]) == 64
        int(payload["source_image_sha256"], 16)
        assert payload["identity_hair_variant"] == "hybrid_verified_increment"
        assert payload["count_hair_variant"] == "hybrid_verified_increment"
        assert isinstance(payload["identity_hairs"], list)
        assert isinstance(payload["count_hairs"], list)
        assert payload["count_hairs"] == payload["identity_hairs"]
        assert payload["blind_images_used"] == 0
        assert payload["canonical_annotations_read_during_inference"] is False
        assert "phaxis" not in payload
        assert all("phaxis" not in str(key).casefold() for key in payload)
        assert all(
            hair.get("source")
            not in {"phaxis_stage_b_train399", "rhaxiscc_stage_b"}
            for hair in payload["identity_hairs"]
        )
        ordered_prediction_file_locks.append(
            {"task_id": task_id, "sha256": sha256_file(prediction_path)}
        )

    assert sha256_json(ordered_prediction_file_locks) == (
        "ede309b8a828aec35be64d9f8afbc2ac9bf92b5a9e1b1b262d5acf603a746f36"
    )


def test_release_control_1_8_is_exact_successor_of_immutable_1_7() -> None:
    config_root = PROJECT_ROOT / "configs/phaxis/v1_0"
    historical_contract_path = (
        config_root / "post_training_release_stage_contract_1_7.json"
    )
    historical_config_path = (
        config_root / "post_training_release_assembly_config_1_7.json"
    )
    current_contract_path = (
        config_root / "post_training_release_stage_contract_1_8.json"
    )
    current_config_path = (
        config_root / "post_training_release_assembly_config_1_8.json"
    )

    # Release-control 1.7 remains a byte-exact historical authority.  The
    # successor is create-only and must never be implemented by mutating it.
    assert sha256_file(historical_contract_path) == (
        "311a27501d6a77d103e7dd16bde37ad4a7811b2927fe9511eadb928fb87dd9d0"
    )
    assert sha256_file(historical_config_path) == (
        "fe42bf6a7ad92325376dfc589269f7550e445c759224147795721c43623a7c95"
    )

    historical_contract = json.loads(
        historical_contract_path.read_text(encoding="utf-8")
    )
    current_contract = json.loads(current_contract_path.read_text(encoding="utf-8"))
    assert current_contract == contract_builder.build(primary_physical_gpu=1)
    assert historical_contract["stage_count"] == current_contract["stage_count"] == 61
    assert [stage["name"] for stage in historical_contract["stages"]] == [
        stage["name"] for stage in current_contract["stages"]
    ]
    assert {
        key: value for key, value in historical_contract.items() if key != "stages"
    } == {key: value for key, value in current_contract.items() if key != "stages"}
    assert [
        index
        for index, (historical, current) in enumerate(
            zip(historical_contract["stages"], current_contract["stages"], strict=True)
        )
        if historical != current
    ] == [27]

    historical_overlay = historical_contract["stages"][27]
    current_overlay = current_contract["stages"][27]
    assert historical_overlay["name"] == current_overlay["name"] == "overlay_evidence"
    assert historical_overlay["receipt"]["status"] == (
        "completed_locked_preselected_acquisition_challenge_gallery"
    )
    assert "identity_field" not in historical_overlay["receipt"]
    assert [item["name"] for item in historical_overlay["artifacts"]] == [
        "receipt",
        "selection",
    ]
    assert "--expected-task-count" not in historical_overlay["command"]

    assert KNOWN_STAGE_SCHEMAS["overlay_evidence"] == (
        "PHAxis-manuscript-overlay-selection-receipt-1.2"
    )
    assert current_overlay["receipt"] == {
        "artifact": "receipt",
        "schema_version": "{known_stage_schema:overlay_evidence}",
        "status_field": "status",
        "status": (
            "completed_locked_preselected_gallery_and_exact_cohort_review_export"
        ),
        "required_fields": {},
        "identity_field": "overlay_selection_identity_sha256",
        "identity_seals_complete_object": True,
    }
    expected_artifacts = {
        "receipt": ("file", "overlay_selection_receipt.json"),
        "selection": ("file", "overlay_selection.csv"),
        "full283_review_overlays": ("directory", "full283_review_overlays"),
        "full283_review_index": ("file", "full283_review_index.csv"),
        "full283_review_checklist": ("file", "full283_review_checklist.csv"),
        "full283_review_summary": ("file", "full283_review_summary.json"),
        "full283_review_readme": ("file", "README_CN.md"),
    }
    assert {
        item["name"]: (item["kind"], item["path"].rsplit("/", 1)[-1])
        for item in current_overlay["artifacts"]
    } == expected_artifacts
    assert current_overlay["command"].count("--expected-task-count") == 1
    expected_count_index = current_overlay["command"].index("--expected-task-count")
    assert current_overlay["command"][expected_count_index + 1] == "283"

    first_gpu0_stage = next(
        index
        for index, stage in enumerate(current_contract["stages"])
        if 0 in stage.get("gpu", {}).get("physical_gpus", [])
    )
    assert first_gpu0_stage == 28
    assert current_contract["stages"][28]["name"] == (
        "benchmark_phaxis_production"
    )

    historical_config = json.loads(
        historical_config_path.read_text(encoding="utf-8")
    )
    current_config = json.loads(current_config_path.read_text(encoding="utf-8"))
    expected_config = deepcopy(historical_config)
    expected_config["stage_contract_template"] = (
        "configs/phaxis/v1_0/post_training_release_stage_contract_1_8.json"
    )
    expected_config["output_root"] = (
        "outputs/phaxis_post_training_formal_release_run1_8"
    )
    assert current_config == expected_config


def test_release_control_1_9_is_exact_successor_of_immutable_1_8() -> None:
    config_root = PROJECT_ROOT / "configs/phaxis/v1_0"
    historical_contract_path = (
        config_root / "post_training_release_stage_contract_1_8.json"
    )
    historical_config_path = (
        config_root / "post_training_release_assembly_config_1_8.json"
    )
    current_contract_path = (
        config_root / "post_training_release_stage_contract_1_9.json"
    )
    current_config_path = (
        config_root / "post_training_release_assembly_config_1_9.json"
    )

    # Release-control 1.8 is now immutable historical authority.  The 1.9
    # successor corrects only its comparator binding and run-local paths.
    assert sha256_file(historical_contract_path) == (
        "4303648ddb0db1fd41e5dd36d3181527176a35baf70145d7b22712f617533a8e"
    )
    assert sha256_file(historical_config_path) == (
        "393172f73308e48f21510485e6cb742756158f642134b0ccb62b02869821e82b"
    )

    historical_contract = json.loads(
        historical_contract_path.read_text(encoding="utf-8")
    )
    current_contract = json.loads(current_contract_path.read_text(encoding="utf-8"))
    assert current_contract_path.read_bytes() == historical_contract_path.read_bytes()
    assert current_contract == historical_contract == contract_builder.build(
        primary_physical_gpu=1
    )
    assert current_contract["stage_count"] == len(current_contract["stages"]) == 61

    historical_config = json.loads(
        historical_config_path.read_text(encoding="utf-8")
    )
    current_config = json.loads(current_config_path.read_text(encoding="utf-8"))
    expected_config = deepcopy(historical_config)
    expected_config["stage_contract_template"] = (
        "configs/phaxis/v1_0/post_training_release_stage_contract_1_9.json"
    )
    expected_config["output_root"] = (
        "outputs/phaxis_post_training_formal_release_run1_9"
    )
    expected_config["authorities"]["qcdev_hybrid_predictions"]["path"] = (
        "outputs/rhaxis_nextgen_hybrid_max_qcdev44_run4_final/predictions"
    )
    assert current_config == expected_config


def test_release_authority_registry_pins_current_release_control_1_9() -> None:
    registry = json.loads(
        (PROJECT_ROOT / "release/RELEASE_AUTHORITY_REGISTRY.json").read_text(
            encoding="utf-8"
        )
    )
    release_control = registry["release_control"]
    expected_contract = (
        PROJECT_ROOT
        / "configs/phaxis/v1_0/post_training_release_stage_contract_1_9.json"
    )
    expected_config = (
        PROJECT_ROOT
        / "configs/phaxis/v1_0/post_training_release_assembly_config_1_9.json"
    )
    assert release_control["stage_contract"] == (
        "configs/phaxis/v1_0/post_training_release_stage_contract_1_9.json"
    )
    assert release_control["stage_contract_sha256"] == sha256_file(
        expected_contract
    )
    assert release_control["assembly_config"] == (
        "configs/phaxis/v1_0/post_training_release_assembly_config_1_9.json"
    )
    assert release_control["assembly_config_sha256"] == sha256_file(expected_config)


def test_stage_contract_builder_output_cli_writes_the_generated_contract(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stage_contract.json"
    assert contract_builder.main(["--output", str(output)]) == 0
    generated = contract_builder.build()
    assert json.loads(output.read_text(encoding="utf-8")) == generated
    overlay_stage = {
        stage["name"]: stage for stage in generated["stages"]
    }["overlay_evidence"]
    overlay_command = overlay_stage["command"]
    assert overlay_command[
        overlay_command.index("--expected-task-count") + 1
    ] == "283"
    overlay_artifacts = {
        item["name"]: item for item in overlay_stage["artifacts"]
    }
    assert overlay_artifacts["full283_review_overlays"]["kind"] == "directory"
    assert overlay_artifacts["full283_review_overlays"]["path"].endswith(
        "/full283_review_overlays"
    )
    assert overlay_artifacts["full283_review_index"]["path"].endswith(
        "/full283_review_index.csv"
    )
    assert overlay_artifacts["full283_review_checklist"]["path"].endswith(
        "/full283_review_checklist.csv"
    )
    assert overlay_artifacts["full283_review_summary"]["path"].endswith(
        "/full283_review_summary.json"
    )
    assert overlay_artifacts["full283_review_readme"]["path"].endswith(
        "/README_CN.md"
    )
    assert overlay_stage["receipt"]["identity_field"] == (
        "overlay_selection_identity_sha256"
    )


def test_assembly_launch_forwards_user_gpu_hold_and_returns_expected_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        assembly_cli,
        "assemble_release_manifest",
        lambda config, manifest, *, run_dir: {
            "status": "assembled",
            "config": str(config),
            "manifest": str(manifest),
            "run_dir": str(run_dir),
        },
    )

    def fake_execute(manifest, run_dir, *, held_physical_gpus=()):
        captured.update(
            manifest=str(manifest),
            run_dir=str(run_dir),
            held_physical_gpus=tuple(held_physical_gpus),
        )
        return {"status": "paused_for_user_gpu_hold"}

    monkeypatch.setattr(assembly_cli, "execute_release", fake_execute)
    manifest = tmp_path / "manifest.json"
    run_dir = tmp_path / "run"
    assert (
        assembly_cli.main(
            [
                "--config",
                str(tmp_path / "config.json"),
                "--manifest-output",
                str(manifest),
                "--run-output",
                str(run_dir),
                "--launch",
                "--hold-physical-gpu",
                "0",
            ]
        )
        == EXPECTED_GPU_HOLD_EXIT_CODE
    )
    assert captured == {
        "manifest": str(manifest),
        "run_dir": str(run_dir),
        "held_physical_gpus": (0,),
    }

    with pytest.raises(SystemExit):
        assembly_cli.main(
            [
                "--config",
                str(tmp_path / "config.json"),
                "--check",
                "--hold-physical-gpu",
                "0",
            ]
        )


def _orchestrator_helpers() -> dict:
    return runpy.run_path(str(PROJECT_ROOT / "tests" / "phaxis" / "test_release_orchestrator.py"))


def _assembly_fixture(tmp_path: Path) -> tuple[Path, Path, object]:
    helpers = _orchestrator_helpers()
    fixture_builder = helpers["_manifest_fixture"]
    # The orchestrator fixture can briefly lag a newly inserted producer while
    # the topology is edited concurrently.  Build its previous complete
    # contract, then add the current explicit analysis-workflow authority.
    old_order = tuple(name for name in MANDATORY_STAGE_ORDER if name != "analysis_workflow_manifest")
    fixture_builder.__globals__["MANDATORY_STAGE_ORDER"] = old_order
    fixture = fixture_builder(tmp_path)
    original = fixture["manifest_payload"]
    if "analysis_workflow_manifest" in MANDATORY_STAGE_ORDER:
        run_dir = fixture["run_dir"]
        stage = {
            "name": "analysis_workflow_manifest",
            "command": ["synthetic-stage", "analysis_workflow_manifest"],
            "inputs": [
                {"stage": name, "artifact": "receipt"}
                for name in (
                    "production_manifest",
                    "candidate_manifest",
                    "selection",
                    "proposal",
                    "authority_pin",
                )
            ],
            "artifacts": [
                {
                    "name": "receipt",
                    "path": str(run_dir / "analysis_workflow_manifest" / "receipt.json"),
                    "kind": "file",
                }
            ],
            "receipt": {
                "artifact": "receipt",
                "schema_version": "PHAxis-analysis-workflow-manifest-1.0",
                "status_field": "status",
                "status": "ready_hash_locked_full_workflow",
                "required_fields": {},
            },
        }
        insert_at = MANDATORY_STAGE_ORDER.index("analysis_workflow_manifest")
        original["stages"].insert(insert_at, stage)
        for benchmark_name in ("benchmark_phaxis_production", "benchmark_phaxis_sequential"):
            benchmark = next(item for item in original["stages"] if item["name"] == benchmark_name)
            benchmark["inputs"].append({"stage": "analysis_workflow_manifest", "artifact": "receipt"})
        for benchmark_name in (
            "benchmark_phaxis_production",
            "benchmark_frozen_v1_production",
            "benchmark_phaxis_sequential",
            "benchmark_frozen_v1_sequential",
        ):
            benchmark = next(item for item in original["stages"] if item["name"] == benchmark_name)
            if "--cuda-visible-devices" not in benchmark["command"]:
                benchmark["command"].extend(["--cuda-visible-devices", "1"])
    original_workspace = fixture["workspace"]
    author = tmp_path / "author_metadata.json"
    author.write_text(
        json.dumps(
            {
                "schema_version": "PHAxis-release-human-metadata-1.1",
                "status": "author_verified",
                "authors": [{"name": "A. Researcher", "email": "a@example.org"}],
            }
        ),
        encoding="utf-8",
    )
    stage_template = tmp_path / "stage_contract.json"
    stage_template.write_text(
        json.dumps(
            {
                "schema_version": STAGE_TEMPLATE_SCHEMA,
                "stages": original["stages"],
            }
        ),
        encoding="utf-8",
    )

    authorities = deepcopy(original["external_inputs"])
    for spec in authorities.values():
        path = Path(spec["path"])
        if not path.is_absolute():
            path = original_workspace / path
        spec["path"] = str(path.resolve())

    members = []
    for member in original["training_members"]:
        receipt_name = member["completion_receipt_input"]
        checkpoint_name = member["checkpoint_input"]
        members.append(
            {
                **member,
                "completion_receipt": authorities[receipt_name]["path"],
                "checkpoint": authorities[checkpoint_name]["path"],
            }
        )
    excluded = {
        "schema_version",
        "workspace",
        "external_inputs",
        "training_members",
        "stages",
        "manifest_identity_sha256",
    }
    config = {
        "schema_version": ASSEMBLY_CONFIG_SCHEMA,
        "workspace": str(original_workspace),
        "project_root": str(PROJECT_ROOT),
        "output_root": str(fixture["run_dir"]),
        "author_metadata_template": str(author),
        "stage_contract_template": str(stage_template),
        "authorities": authorities,
        "training_members": members,
        "manifest_fields": {key: value for key, value in original.items() if key not in excluded},
    }
    config_path = tmp_path / "assembly_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    preview = helpers["_synthetic_candidate_preview"]

    def candidate_builder(checkpoints, *, dataset_audit_path):
        return preview(checkpoints, dataset_audit_path)

    return config_path, fixture["run_dir"], candidate_builder


def test_check_reports_blockers_without_preexisting_manifest(tmp_path: Path) -> None:
    config_path, _run_dir, _candidate_builder = _assembly_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed3 = config["training_members"][2]
    Path(seed3["completion_receipt"]).unlink()
    config["author_metadata_template"] = str(tmp_path / "missing_author_metadata.json")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report = inspect_release_readiness(config_path)

    assert report["status"] == "blocked_current_prerequisites"
    assert report["formal_manifest_present_or_required_for_check"] is False
    assert any(item["code"] == "TRAINING_MEMBER_NOT_COMPLETE" and item["seed"] == 2026082803 for item in report["blockers"])
    assert any(item["code"] == "AUTHOR_METADATA_NOT_FINAL" for item in report["blockers"])
    assert tuple(report["pending_derived_stages"]) == MANDATORY_STAGE_ORDER
    assert report["pending_derived_assets_are_not_external_authorities"] is True


def test_formal_assembly_is_self_sealed_plan_valid_and_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, run_dir, candidate_builder = _assembly_fixture(tmp_path)
    output = tmp_path / "formal_manifest.json"
    monkeypatch.setattr(
        "phaxis.release_manifest_assembler.validate_release_topology",
        lambda **_kwargs: {"real_producer_source_checks": []},
    )

    receipt = assemble_release_manifest(
        config_path,
        output,
        run_dir=run_dir,
        candidate_builder=candidate_builder,
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    unsigned = deepcopy(manifest)
    identity = unsigned.pop("manifest_identity_sha256")
    assert sha256_json(unsigned) == identity
    assert receipt["status"] == "formal_manifest_created_no_overwrite"
    assert tuple(receipt["stage_order"]) == MANDATORY_STAGE_ORDER
    plan = build_release_plan(output, run_dir, candidate_builder=candidate_builder)
    assert tuple(item["name"] for item in plan["stages"]) == MANDATORY_STAGE_ORDER
    with pytest.raises(ReleaseManifestAssemblyError, match="refusing to overwrite"):
        assemble_release_manifest(
            config_path,
            output,
            run_dir=run_dir,
            candidate_builder=candidate_builder,
        )


def test_deferred_human_authority_allows_manifest_assembly_without_template_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, run_dir, candidate_builder = _assembly_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workspace = Path(config["workspace"])
    draft = workspace / "release_author_metadata_template.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": "Synthetic-release-human-metadata-1.0",
                "status": "BLOCKED_TEMPLATE_NOT_AUTHORITY",
                "metadata_identity_sha256": "COMPUTE_AFTER_AUTHOR_VERIFICATION",
            }
        ),
        encoding="utf-8",
    )
    config["author_metadata_template"] = str(draft)
    config["authorities"]["release_author_metadata"] = {
        "path": str(draft),
        "kind": "file",
        "authority_class": "author_metadata",
        "deferred_authority": {
            "schema_version": "PHAxis-deferred-human-authority-contract-1.0",
            "human_authority_id": "synthetic-release-author-metadata",
            "document_schema_version": "Synthetic-release-human-metadata-1.0",
            "status_field": "status",
            "final_status": "author_verified_release_authority",
            "identity_field": "metadata_identity_sha256",
            "first_consumer_stage": "source_release",
            "target_path": "{run_dir}/human_authorities/release_author_metadata.json",
        },
    }
    template_path = Path(config["stage_contract_template"])
    template = json.loads(template_path.read_text(encoding="utf-8"))
    source_release = next(
        stage for stage in template["stages"] if stage["name"] == "source_release"
    )
    source_release["inputs"].append({"external": "release_author_metadata"})
    source_release["command"].append("{external:release_author_metadata}")
    template_path.write_text(json.dumps(template), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        "phaxis.release_manifest_assembler.validate_release_topology",
        lambda **_kwargs: {"real_producer_source_checks": []},
    )

    readiness = inspect_release_readiness(config_path)
    assert (
        readiness["status"]
        == "ready_to_assemble_science_prefix_human_gate_deferred"
    )
    assert readiness["formal_release_allowed"] is True
    assert readiness["expected_pause_is_not_training_or_algorithm_failure"] is True
    assert readiness["expected_first_deferred_consumer_stage"] == "source_release"
    deferred = readiness["external_authorities"]["release_author_metadata"]
    assert deferred["unfinished_template_bytes_locked_by_manifest"] is False
    assert deferred["exact_final_bytes_locked_by_first_consumer_sentinel"] is True

    output = tmp_path / "deferred_manifest.json"
    assemble_release_manifest(
        config_path,
        output,
        run_dir=run_dir,
        candidate_builder=candidate_builder,
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    authority = manifest["external_inputs"]["release_author_metadata"]
    assert authority["path"].startswith("{run_dir}/human_authorities/")
    assert "sha256" not in authority
    plan = build_release_plan(output, run_dir, candidate_builder=candidate_builder)
    source_plan = next(
        stage for stage in plan["stages"] if stage["name"] == "source_release"
    )
    assert any(
        item.get("external") == "release_author_metadata"
        and item.get("deferred") is True
        and "sha256" not in item
        for item in source_plan["inputs"]
    )


def test_non_author_metadata_cannot_opt_into_deferred_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _run_dir, _candidate_builder = _assembly_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    any_name, spec = next(iter(config["authorities"].items()))
    spec["deferred_authority"] = {
        "schema_version": "PHAxis-deferred-human-authority-contract-1.0",
        "human_authority_id": "invalid-nonhuman-deferred",
        "document_schema_version": "invalid-1.0",
        "status_field": "status",
        "final_status": "final",
        "identity_field": "identity_sha256",
        "first_consumer_stage": "source_release",
        "target_path": "{run_dir}/human_authorities/invalid.json",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        "phaxis.release_manifest_assembler.validate_release_topology",
        lambda **_kwargs: {"real_producer_source_checks": []},
    )
    report = inspect_release_readiness(config_path)
    blocker = next(
        item
        for item in report["blockers"]
        if item.get("authority") == any_name
    )
    assert "only author_metadata may be deferred" in blocker["detail"]


def test_stage_derived_output_cannot_be_external_authority(tmp_path: Path) -> None:
    config_path, _run_dir, _candidate_builder = _assembly_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    any_file = next(iter(config["authorities"].values()))["path"]
    config["authorities"]["candidate_manifest"] = {
        "path": any_file,
        "kind": "file",
        "authority_class": "static_contract",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report = inspect_release_readiness(config_path)

    assert any(
        blocker["code"] == "EXTERNAL_AUTHORITY_NOT_READY"
        and blocker["authority"] == "candidate_manifest"
        for blocker in report["blockers"]
    )


@pytest.mark.parametrize(
    ("token", "message"),
    (
        ("{external:qcdev_manifest}", "undeclared external authorities"),
        ("{run_dir}/selection/selection_receipt.json", "future-stage authority"),
    ),
)
def test_stage_contract_rejects_command_authority_outside_declared_inputs(
    tmp_path: Path, token: str, message: str
) -> None:
    config_path, _run_dir, _candidate_builder = _assembly_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    template_path = Path(config["stage_contract_template"])
    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["stages"][0]["command"].append(token)
    template_path.write_text(json.dumps(template), encoding="utf-8")

    report = inspect_release_readiness(config_path)

    blocker = next(
        item for item in report["blockers"] if item["code"] == "STAGE_CONTRACT_NOT_READY"
    )
    assert message in blocker["detail"]


def test_readiness_rejects_duplicate_artifact_path_before_formal_manifest(
    tmp_path: Path,
) -> None:
    config_path, _run_dir, _candidate_builder = _assembly_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    template_path = Path(config["stage_contract_template"])
    template = json.loads(template_path.read_text(encoding="utf-8"))
    stage = template["stages"][0]
    duplicate = deepcopy(stage["artifacts"][0])
    duplicate["name"] = "same_bytes_second_name"
    stage["artifacts"].append(duplicate)
    template_path.write_text(json.dumps(template), encoding="utf-8")

    report = inspect_release_readiness(config_path)

    blocker = next(
        item for item in report["blockers"] if item["code"] == "STAGE_CONTRACT_NOT_READY"
    )
    assert "stage artifact path reused" in blocker["detail"]
    assert report["formal_release_allowed"] is False


@pytest.mark.parametrize("alias_segment", (".", "temporary/.."))
def test_readiness_rejects_normalized_artifact_path_alias_before_formal_manifest(
    tmp_path: Path,
    alias_segment: str,
) -> None:
    config_path, _run_dir, _candidate_builder = _assembly_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    template_path = Path(config["stage_contract_template"])
    template = json.loads(template_path.read_text(encoding="utf-8"))
    stage = template["stages"][0]
    duplicate = deepcopy(stage["artifacts"][0])
    duplicate["name"] = "normalized_alias_of_same_bytes"
    normalized = str(duplicate["path"]).replace("\\", "/")
    parent, filename = normalized.rsplit("/", 1)
    duplicate["path"] = f"{parent}/{alias_segment}/{filename}"
    stage["artifacts"].append(duplicate)
    template_path.write_text(json.dumps(template), encoding="utf-8")

    report = inspect_release_readiness(config_path)

    blocker = next(
        item for item in report["blockers"] if item["code"] == "STAGE_CONTRACT_NOT_READY"
    )
    assert "stage artifact path reused" in blocker["detail"]
    assert report["formal_release_allowed"] is False


def test_readiness_rejects_artifact_path_outside_configured_run_root(
    tmp_path: Path,
) -> None:
    config_path, _run_dir, _candidate_builder = _assembly_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    template_path = Path(config["stage_contract_template"])
    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["stages"][0]["artifacts"][0]["path"] = str(
        (tmp_path.parent / "outside-release-receipt.json").resolve()
    )
    template_path.write_text(json.dumps(template), encoding="utf-8")

    report = inspect_release_readiness(config_path)

    blocker = next(
        item for item in report["blockers"] if item["code"] == "STAGE_CONTRACT_NOT_READY"
    )
    assert "leaves the release run directory" in blocker["detail"]
    assert report["formal_release_allowed"] is False


def test_readiness_rejects_non_string_artifact_path(tmp_path: Path) -> None:
    config_path, _run_dir, _candidate_builder = _assembly_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    template_path = Path(config["stage_contract_template"])
    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["stages"][0]["artifacts"][0]["path"] = None
    template_path.write_text(json.dumps(template), encoding="utf-8")

    report = inspect_release_readiness(config_path)

    blocker = next(
        item for item in report["blockers"] if item["code"] == "STAGE_CONTRACT_NOT_READY"
    )
    assert "artifact path is invalid" in blocker["detail"]
    assert report["formal_release_allowed"] is False


def test_readiness_rejects_directory_receipt_artifact(tmp_path: Path) -> None:
    config_path, _run_dir, _candidate_builder = _assembly_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    template_path = Path(config["stage_contract_template"])
    template = json.loads(template_path.read_text(encoding="utf-8"))
    template["stages"][0]["artifacts"][0]["kind"] = "directory"
    template_path.write_text(json.dumps(template), encoding="utf-8")

    report = inspect_release_readiness(config_path)

    blocker = next(
        item for item in report["blockers"] if item["code"] == "STAGE_CONTRACT_NOT_READY"
    )
    assert "receipt artifact must be a JSON file" in blocker["detail"]
    assert report["formal_release_allowed"] is False


def test_author_metadata_authority_template_is_not_reported_ready(tmp_path: Path) -> None:
    config_path, _run_dir, _candidate_builder = _assembly_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    template = tmp_path / "blocked_attestation.json"
    template.write_text(
        json.dumps(
            {
                "schema_version": "example-authority-1.0",
                "status": "BLOCKED_TEMPLATE_NOT_AUTHORITY",
                "authority_name": "REQUIRED_AUTHORITY_NAME",
            }
        ),
        encoding="utf-8",
    )
    config["authorities"]["release_attestation"] = {
        "path": str(template),
        "kind": "file",
        "authority_class": "author_metadata",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    report = inspect_release_readiness(config_path)

    assert report["external_authorities"]["release_attestation"]["ready"] is False
    assert any(
        blocker["code"] == "EXTERNAL_AUTHORITY_NOT_READY"
        and blocker["authority"] == "release_attestation"
        for blocker in report["blockers"]
    )


def test_real_stage_contract_tracks_current_topology_and_provider_descriptor() -> None:
    contract_path = (
        PROJECT_ROOT
        / "configs"
        / "phaxis"
        / "v1_0"
        / "post_training_release_stage_contract_1_9.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["synthetic_commands_present"] is False
    assert tuple(stage["name"] for stage in contract["stages"]) == MANDATORY_STAGE_ORDER
    assert all(
        stage["command"] is None or "synthetic" not in " ".join(stage["command"]).casefold()
        for stage in contract["stages"]
    )
    by_name = {stage["name"]: stage for stage in contract["stages"]}
    for producer in FORMAL_RELEASE_PRODUCERS:
        if producer.producer.startswith("internal:"):
            assert by_name[producer.name]["command"] is None
            continue
        command = by_name[producer.name]["command"]
        assert command is not None
        assert set(producer.required_cli_options) <= set(command)
    for name in (
        "benchmark_phaxis_production",
        "benchmark_frozen_v1_production",
        "benchmark_phaxis_sequential",
        "benchmark_frozen_v1_sequential",
    ):
        command = by_name[name]["command"]
        assert "{workspace}/scripts/phaxis/run_external_direct_benchmark.py" in command
        assert "--image-root" in command
        assert command[command.index("--producer-interface") + 1].endswith(
            "/direct_benchmark_provider_descriptor/provider.json"
        )
        assert command[command.index("--output") + 1].endswith("/benchmark")
        assert command[command.index("--cuda-visible-devices") + 1] == "0"
        assert by_name[name]["receipt"]["schema_version"] == f"{{known_stage_schema:{name}}}"
        assert by_name[name]["artifacts"][0]["path"].endswith("/benchmark/runtime_summary.json")
        assert by_name[name]["gpu"]["physical_gpus"] == [0]
        assert by_name[name]["environment"] == {
            "PHAXIS_REQUIRE_EXACT_PHYSICAL_GPU": "1"
        }
    assert by_name["qcdev_root_inputs"]["command"][1].endswith(
        "/scripts/phaxis/build_qcdev44_root_provider_inputs.py"
    )
    assert by_name["release_case_prelocks"]["command"][1].endswith(
        "/scripts/phaxis/build_release_case_prelocks.py"
    )
    assert by_name["direct_benchmark_provider_descriptor"]["command"][1].endswith(
        "/scripts/phaxis/build_direct_benchmark_provider_descriptor.py"
    )
    assert by_name["direct_benchmark_provider_descriptor"]["artifacts"] == [
        {
            "kind": "file",
            "name": "receipt",
            "path": "{run_dir}/direct_benchmark_provider_descriptor/provider.json",
        }
    ]
    assert by_name["overlay_evidence"]["command"][
        by_name["overlay_evidence"]["command"].index("--case-plan") + 1
    ].endswith("/release_case_prelocks/output/overlay_case_plan.csv")
    geometry_artifacts = {
        item["name"]: item
        for item in by_name["figure1_geometry_materialization"]["artifacts"]
    }
    assert geometry_artifacts["figure1_image"]["path"].endswith(
        "/figure1_source_image.tif"
    )
    sample_artifacts = {
        item["name"]: item
        for item in by_name["clean_install_sample_manifest"]["artifacts"]
    }
    assert sample_artifacts["example_manifest"]["path"].endswith(
        "/release_example_manifest.json"
    )
    assert by_name["clean_install_expected_identity"]["gpu"][
        "physical_gpus"
    ] == [0]
    assert by_name["clean_install"]["gpu"]["physical_gpus"] == [0]
    for name in (
        "root_provider_exact283",
        "qcdev_root_provider",
        "clean_install_expected_identity",
        "clean_install",
    ):
        assert by_name[name]["environment"] == {
            "PHAXIS_REQUIRE_EXACT_PHYSICAL_GPU": "1"
        }
    for name in ("root_provider_exact283", "qcdev_root_provider"):
        assert "--strict-physical-gpu" in by_name[name]["command"]
    clean_command = by_name["clean_install"]["command"]
    assert clean_command[clean_command.index("--cuda-visible-devices") + 1] == "0"
    offline_artifacts = {
        item["name"]: item for item in by_name["offline_dependencies"]["artifacts"]
    }
    assert offline_artifacts["dependency_lock"]["path"].endswith(
        "/requirements.lock.txt"
    )
    assert offline_artifacts["resolved_sbom"]["path"].endswith(
        "/SBOM.resolved.cdx.json"
    )
    assert offline_artifacts["resolved_license_inventory"]["path"].endswith(
        "/THIRD_PARTY_LICENSES.resolved.json"
    )
    assert "{external:root_raw_input_manifest}" in by_name["root_provider_exact283"]["command"]
    assert "{external:root_raw_input_manifest}" in by_name["analysis_workflow_manifest"]["command"]
    assert by_name["source_release"]["artifacts"][0]["path"].endswith("/SOURCE_MANIFEST.json")
    assert by_name["distributions"]["artifacts"][0]["path"].endswith("/distribution_receipt.json")
    assert "--execute" not in by_name["handover"]["command"]
    assert by_name["handover"]["artifacts"][0]["path"].endswith("/BUILD_RECEIPT.json")


def test_stage_contract_can_retarget_scientific_gpu_without_moving_frozen_comparator() -> None:
    contract = contract_builder.build(primary_physical_gpu=2)
    by_name = {stage["name"]: stage for stage in contract["stages"]}
    gpu_stages = [stage for stage in contract["stages"] if "gpu" in stage]

    assert len(gpu_stages) == 11
    assert all(stage["gpu"]["internal_device"] == "cuda:0" for stage in gpu_stages)

    for name in (
        "qcdev_candidate_pool",
        "qcdev_evaluation_inference",
        "root_provider_exact283",
        "qcdev_root_provider",
        "production_stageb_exact283",
    ):
        assert by_name[name]["gpu"]["physical_gpus"] == [2]
        assert by_name[name]["gpu"]["cuda_visible_devices"] == "2"

    # Formal Stage-B inference is intentionally FP32.  Keep the generated
    # release command aligned with run_stageb_inference.py's pre-GPU guard so
    # the scientific prefix cannot deterministically fail at stage 17.
    assert "--amp" not in by_name["production_stageb_exact283"]["command"]

    for name in (
        "benchmark_phaxis_production",
        "benchmark_frozen_v1_production",
        "benchmark_phaxis_sequential",
        "benchmark_frozen_v1_sequential",
        "clean_install_expected_identity",
        "clean_install",
    ):
        assert by_name[name]["gpu"]["physical_gpus"] == [0]
        assert by_name[name]["gpu"]["cuda_visible_devices"] == "0"

    descriptor = by_name["direct_benchmark_provider_descriptor"]["command"]
    assert descriptor[descriptor.index("--physical-gpu") + 1] == "0"

    analysis = by_name["analysis_workflow_manifest"]["command"]
    for flag in ("--v1-physical-gpu", "--q8-physical-gpu", "--stageb-physical-gpu"):
        assert analysis[analysis.index(flag) + 1] == "0"

    for name in (
        "benchmark_phaxis_production",
        "benchmark_frozen_v1_production",
        "benchmark_phaxis_sequential",
        "benchmark_frozen_v1_sequential",
        "clean_install",
    ):
        command = by_name[name]["command"]
        assert command[command.index("--cuda-visible-devices") + 1] == "0"

    expected = by_name["clean_install_expected_identity"]["command"]
    assert expected[expected.index("--physical-gpu") + 1] == "0"
    assert expected[expected.index("--cuda-visible-devices") + 1] == "0"


@pytest.mark.parametrize("value", [-1, True, "1"])
def test_stage_contract_rejects_invalid_physical_gpu(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        contract_builder.build(primary_physical_gpu=value)  # type: ignore[arg-type]


def test_real_assembly_config_names_every_external_authority() -> None:
    contract = json.loads(
        (
            PROJECT_ROOT
            / "configs"
            / "phaxis"
            / "v1_0"
            / "post_training_release_stage_contract_1_9.json"
        ).read_text(encoding="utf-8")
    )
    config = json.loads(
        (
            PROJECT_ROOT
            / "configs"
            / "phaxis"
            / "v1_0"
            / "post_training_release_assembly_config_1_9.json"
        ).read_text(encoding="utf-8")
    )
    required = {
        item["external"]
        for stage in contract["stages"]
        for item in stage["inputs"]
        if "external" in item
    }
    assert required <= set(config["authorities"])
    for replaced_static in (
        "static_condition_blinded_case_plan",
        "static_figure1_image",
        "static_figure1_geometry",
        "immutable_example_manifest",
        "expected_example_identity",
        "dependency_lock",
        "offline_wheelhouse",
        "formal_direct_benchmark_provider_descriptor",
    ):
        assert replaced_static not in config["authorities"]
    assert config["authorities"]["biological_image_root"]["path"].endswith(
        "rhaxis_six_condition_images_by_batch_v2"
    )
    assert config["authorities"]["release_authority_registry"] == {
        "authority_class": "static_contract",
        "kind": "file",
        "path": "release/RELEASE_AUTHORITY_REGISTRY.json",
    }
    assert config["authorities"]["master_manuscript"]["path"].endswith(
        "PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
    )
    assert config["authorities"]["supplement_master"]["path"].endswith(
        "PHAXIS_SUPPLEMENTARY_MASTER_DRAFT_20260830.md"
    )
    assert config["authorities"]["submission_title_metadata"]["path"].endswith(
        "SUBMISSION_TITLE_METADATA_TEMPLATE_2_0.json"
    )
    expected_receipts = {
        2026082801: "training_receipt_resume_001.json",
        2026082802: "training_receipt.json",
        2026082803: "training_receipt_resume_001.json",
        2026082804: "training_receipt_resume_001.json",
        2026082805: "training_receipt_resume_001.json",
    }
    assert {
        int(member["seed"]): Path(member["completion_receipt"]).name
        for member in config["training_members"]
    } == expected_receipts
    assert {
        seed: Path(config["authorities"][f"seed_{seed}_receipt"]["path"]).name
        for seed in expected_receipts
    } == expected_receipts


def test_known_stage_schema_token_is_materialized_dynamically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        __import__(
            "phaxis.release_manifest_assembler", fromlist=["KNOWN_STAGE_SCHEMAS"]
        ).KNOWN_STAGE_SCHEMAS,
        "selection",
        "PHAxis-selection-future-schema-9.9",
    )
    assert (
        _materialize_template_value(
            "{known_stage_schema:selection}", {}
        )
        == "PHAxis-selection-future-schema-9.9"
    )
