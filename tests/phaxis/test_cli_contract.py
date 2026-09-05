from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path

import pytest

from phaxis.cli import main
from phaxis.hair_stageb.candidate_bundle import FORMAL_TRAIN399_SEEDS
from phaxis.io import read_json, sha256_file, sha256_json
from phaxis.public_identity import (
    PUBLIC_SYSTEM_DERIVATION,
    PUBLIC_SYSTEM_IDENTITY_SCHEMA,
    derive_public_identity,
)
from phaxis.traits import IMAGE_TRAIT_FIELDS, ROOT_TRAIT_FIELDS
import phaxis.cli as cli_module


def _formal_contract(root: Path) -> tuple[Path, dict, dict[str, str]]:
    checkpoint_sha256 = [f"{index + 1:064x}" for index in range(5)]
    model = {
        "expert_id": "PHAxis-StageB-train399-five-seed",
        "ensemble_members": 5,
        "checkpoint_policy": "five_seed_train399_last_epoch_60",
        "deployment_role": "candidate_gate_passed_not_promoted",
        "operating_point_status": "selected_on_locked_QCdevelopment44",
        "selected_score_threshold": 0.225,
        "selection_receipt_sha256": "a" * 64,
        "selection_receipt_identity_sha256": "b" * 64,
        "candidate_pool_identity_sha256": "c" * 64,
        "selected_model_metadata_identity_sha256": None,
        "training_images": 399,
        "validation_images": 44,
        "validation_labels_used_for_gradient_or_early_stopping": False,
        "blind_images_used": 0,
        "seeds": list(FORMAL_TRAIN399_SEEDS),
        "member_ids": [f"seed_{seed}" for seed in FORMAL_TRAIN399_SEEDS],
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": [f"{index + 11:064x}" for index in range(5)],
        "training_task_ids_sha256": "d" * 64,
        "split_manifest_sha256": "e" * 64,
        "training_lock_identity_sha256": "f" * 64,
        "candidate_bundle_identity_sha256": "1" * 64,
        "operating_point_selection_contract_sha256": "2" * 64,
    }
    unsigned_model = dict(model)
    unsigned_model.pop("selected_model_metadata_identity_sha256")
    model["selected_model_metadata_identity_sha256"] = sha256_json(unsigned_model)
    stageb_binding = {
        "expert_id": model["expert_id"],
        "checkpoint_sha256": checkpoint_sha256,
        "selected_score_threshold": 0.225,
        "candidate_bundle_identity_sha256": "1" * 64,
        "selection_receipt_identity_sha256": "b" * 64,
        "selected_model_metadata_identity_sha256": model[
            "selected_model_metadata_identity_sha256"
        ],
    }
    root_audit = "7" * 64
    root_pipeline = "8" * 64
    root_bundle = "9" * 64
    public = derive_public_identity(
        stageb_binding,
        root_bundle_identity_sha256=root_bundle,
    )
    proposal = {
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
                "bundle_identity_sha256": root_bundle,
                "pipeline_identity_sha256": root_pipeline,
            },
        },
        "expert_boundary": {
            "root_point_scale_continuity_statistics": public["root_expert_id"],
            "hair_identity_and_count": model["expert_id"],
        },
        "formal_release_status": "passed_proposal_not_official",
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
            "formal_gate_source_sha256": {
                "train399_candidate": "3" * 64,
                "train399_selection": "4" * 64,
                "train399_evaluation": "5" * 64,
                "root_exact283": "6" * 64,
            },
            "formal_gate_identity_sha256": {
                "candidate_bundle_identity_sha256": "1" * 64,
                "selection_receipt_identity_sha256": "b" * 64,
                "selected_model_metadata_identity_sha256": model[
                    "selected_model_metadata_identity_sha256"
                ],
                "root_exact283_audit_identity_sha256": root_audit,
            },
            "checkpoint_file_sha256_in_member_order": checkpoint_sha256,
            "stageb_binding": stageb_binding,
        },
    }
    proposal["model_contract_identity_sha256"] = sha256_json(proposal)
    proposal_path = root / "model-contract-proposal.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    receipt = {
        "model_contract_proposal_sha256": sha256_file(proposal_path),
        "model_contract_proposal_identity_sha256": proposal[
            "model_contract_identity_sha256"
        ],
        "model_bundle_id": proposal["model_bundle_id"],
        "root_expert_id": proposal["root_expert"]["expert_id"],
    }
    return proposal_path, model, receipt


def _formalize_stageb(stageb: dict, model: dict, receipt: dict[str, str]) -> dict:
    formal = deepcopy(stageb)
    formal["model"] = {**model, "precision_mode": "fp32_locked"}
    formal["operating_point"]["score_threshold"] = 0.225
    formal["coordinate_space"].update(
        {
            "source_shape": [128, 128],
            "working_shape": [64, 64],
            "source_to_working_scale_xy": [0.5, 0.5],
            "realized_um_per_px_xy": [2.0, 2.0],
        }
    )
    for detection in formal["detections"]:
        detection["tip_snapped"] = False
        detection["predicted_length_semantics"] = (
            "regressed_polyline_arc_length_um_diagnostic_only"
        )
    formal.update(receipt)
    formal.pop("detection_identity_sha256", None)
    formal["detection_identity_sha256"] = sha256_json(formal)
    return formal


def _write_inputs(case, root: Path):
    hybrid, stageb, artifact_root = case
    proposal, model, receipt = _formal_contract(root)
    stageb = _formalize_stageb(stageb, model, receipt)
    hybrid_dir = root / "hybrid"
    hair_dir = root / "hair"
    hybrid_dir.mkdir(parents=True)
    hair_dir.mkdir(parents=True)
    (hybrid_dir / "T1.json").write_text(
        json.dumps(hybrid), encoding="utf-8"
    )
    (hair_dir / "T1.json").write_text(
        json.dumps(stageb), encoding="utf-8"
    )
    return hybrid_dir, hair_dir, artifact_root, proposal, stageb


def test_public_cli_fuse_then_export_traits_accepts_model_contract_and_zero_hairs(
    phaxis_case, tmp_path: Path
) -> None:
    hybrid, stageb, artifact_root = phaxis_case
    hybrid["detailed_root_statistics"] = {
        **{field: 1.0 for field in ROOT_TRAIT_FIELDS},
        "visible_root_axis_length_um": 5000.0,
        "median_root_width_um": 12.0,
    }
    stageb["detections"] = []
    stageb["n"] = 0
    hybrid_dir, hair_dir, _artifact_root, contract, _stageb = _write_inputs(
        (hybrid, stageb, artifact_root), tmp_path / "inputs"
    )
    fused = tmp_path / "fused"
    assert main(
        [
            "fuse",
            "--root-predictions",
            str(hybrid_dir),
            "--root-artifacts",
            str(artifact_root),
            "--hair-detections",
            str(hair_dir),
            "--model-contract",
            str(contract),
            "--output",
            str(fused),
        ]
    ) == 0
    fused_prediction = read_json(fused / "predictions" / "T1.json")
    assert fused_prediction["phaxis"]["hair_length_expert"] == (
        "PHAxis-1.0.0-endpoint-complete-root-hair-length"
    )
    for tier in fused_prediction["phenotypes"].values():
        assert tier["hair_length_semantics"] == (
            "PHAxis-1.0.0-conditional-on-endpoint-complete-centrelines"
        )

    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task_id",
                "image_sha256",
                "um_per_px",
                "experiment_key",
                "condition_code",
                "study_role",
                "developmental_day",
                "genotype_or_construct",
                "temperature_c",
                "qc_disposition",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "T1",
                "image_sha256": "a" * 64,
                "um_per_px": 1.0,
                "experiment_key": "synthetic",
                "condition_code": "zero-hair",
                "study_role": "unit_test",
                "developmental_day": "",
                "genotype_or_construct": "",
                "temperature_c": "",
                "qc_disposition": "eligible",
            }
        )
    output = tmp_path / "exported"
    assert main(
        [
            "export-traits",
            "--predictions",
            str(fused / "predictions"),
            "--metadata",
            str(metadata),
            "--model-contract",
            str(contract),
            "--output",
            str(output),
        ]
    ) == 0
    with (output / "image_traits.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        canonical = next(csv.DictReader(handle))
    with (output / "hair_instances.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        assert list(csv.DictReader(handle)) == []
    assert tuple(canonical) == IMAGE_TRAIT_FIELDS
    assert canonical["hair_count"] == "0"
    assert canonical["formal_statistics_eligible"] == "True"


def test_cli_fuse_materializes_locked_artifacts_and_refuses_overwrite(
    phaxis_case, tmp_path: Path, capsys
):
    hybrid_dir, hair_dir, artifact_root, proposal, _stageb = _write_inputs(
        phaxis_case, tmp_path / "inputs"
    )
    output = tmp_path / "output"
    arguments = [
        "fuse",
        "--root-predictions",
        str(hybrid_dir),
        "--root-artifacts",
        str(artifact_root),
        "--hair-detections",
        str(hair_dir),
        "--model-contract",
        str(proposal),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    fused = read_json(output / "predictions" / "T1.json")
    summary = read_json(output / "fusion_summary.json")
    assert summary["status"] == "completed"
    assert summary["images"] == 1
    assert summary["blind_images_used"] == 0
    assert fused["phaxis"]["formal_stageb_identity_count"] == 2
    for relpath, digest in (
        (fused["root_mask_relpath"], fused["root_mask_sha256"]),
        (fused["root_axis_geometry_relpath"], fused["root_axis_geometry_sha256"]),
    ):
        assert sha256_file(output / relpath) == digest

    assert main(arguments) == 2
    assert "refusing to overwrite" in capsys.readouterr().err


def test_cli_missing_requested_task_fails_with_nonzero_status(
    phaxis_case, tmp_path: Path, capsys
):
    hybrid_dir, hair_dir, artifact_root, proposal, _stageb = _write_inputs(
        phaxis_case, tmp_path / "inputs"
    )
    status = main(
        [
            "fuse",
            "--hybrid-predictions",
            str(hybrid_dir),
            "--hybrid-root",
            str(artifact_root),
            "--hair-detections",
            str(hair_dir),
            "--model-contract-proposal",
            str(proposal),
            "--output",
            str(tmp_path / "output"),
            "--task-id",
            "absent",
        ]
    )
    assert status == 2
    assert "requested root-provider task IDs are absent" in capsys.readouterr().err


def test_public_cli_help_uses_phaxis_terms_and_documents_public_fuse_inputs() -> None:
    parser = cli_module.build_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    help_text = "\n".join(
        [
            parser.format_help(),
            subparsers.choices["analyze"].format_help(),
            subparsers.choices["fuse"].format_help(),
            subparsers.choices["infer-hairs"].format_help(),
            subparsers.choices["export-traits"].format_help(),
        ]
    )
    assert "--root-predictions" in help_text
    assert "--root-artifacts" in help_text
    assert "--model-contract" in help_text
    assert "--hybrid-predictions" not in help_text
    assert "--hybrid-root" not in help_text
    assert "--model-contract-proposal" not in help_text
    assert "RHAxiscc" not in help_text
    assert "Hybrid-Max" not in help_text
    assert "nextgen" not in help_text.casefold()
    assert "v2.0" not in help_text.casefold()


def test_cli_verify_prediction_reports_valid_contract(phaxis_case, tmp_path: Path, capsys):
    hybrid, _, artifact_root = phaxis_case
    prediction_path = tmp_path / "prediction.json"
    prediction_path.write_text(json.dumps(hybrid), encoding="utf-8")
    assert main(
        [
            "verify-prediction",
            "--prediction",
            str(prediction_path),
            "--artifact-root",
            str(artifact_root),
        ]
    ) == 0
    output = capsys.readouterr().out
    assert '"status": "valid"' in output
    assert '"blind_images_used": 0' in output


def test_cli_version_matches_public_product_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "PHAxis 1.0.0" in capsys.readouterr().out


def test_infer_hairs_train399_gate_arguments_are_all_or_none(tmp_path: Path, capsys):
    arguments = [
        "infer-hairs",
        "--image",
        str(tmp_path / "image.tif"),
        "--task-id",
        "T1",
        "--source-um-per-px",
        "1.0",
        "--device",
        "cuda:0",
        "--output",
        str(tmp_path / "prediction.json"),
        "--candidate-manifest",
        str(tmp_path / "candidate.json"),
    ]
    for index in range(5):
        arguments.extend(("--checkpoint", str(tmp_path / f"member{index}.pt")))
    assert main(arguments) == 2
    error = capsys.readouterr().err
    assert "requires --model-contract-proposal" in error
    assert "legacy 443CV fallback is forbidden" in error


def test_fusion_summary_uses_the_unique_expert_identity_from_fused_predictions(
    phaxis_case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hybrid_dir, hair_dir, artifact_root, proposal, _stageb = _write_inputs(
        phaxis_case, tmp_path / "inputs"
    )
    train399_expert = "PHAxis-StageB-train399-five-seed-identity-count"

    def fake_fusion(hybrid, _stageb, **_kwargs):
        fused = deepcopy(hybrid)
        fused["phaxis"] = {
            "hair_identity_count_expert": train399_expert,
            "previous_hybrid_identity_count": 1,
            "formal_stageb_identity_count": 2,
            "attachment_valid_fraction": 1.0,
            "root_lock_sha256": "f" * 64,
            "length_identity_association": {"matched_length_identities": 1},
        }
        return fused

    monkeypatch.setattr(
        cli_module, "fuse_hybrid_root_with_stageb_hairs", fake_fusion
    )
    output = tmp_path / "output"
    assert cli_module.main(
        [
            "fuse",
            "--hybrid-predictions",
            str(hybrid_dir),
            "--hybrid-root",
            str(artifact_root),
            "--hair-detections",
            str(hair_dir),
            "--model-contract-proposal",
            str(proposal),
            "--output",
            str(output),
        ]
    ) == 0
    assert read_json(output / "fusion_summary.json")[
        "hair_identity_count_expert"
    ] == train399_expert


def test_fusion_summary_rejects_mixed_expert_identities(
    phaxis_case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    hybrid, _legacy_stageb, artifact_root = phaxis_case
    hybrid_dir, hair_dir, _artifact_root, proposal, stageb = _write_inputs(
        phaxis_case, tmp_path / "inputs"
    )
    for task_id in ("T1", "T2"):
        one_hybrid = deepcopy(hybrid)
        one_hybrid["task_id"] = task_id
        one_stageb = deepcopy(stageb)
        one_stageb["task_id"] = task_id
        one_stageb.pop("detection_identity_sha256", None)
        one_stageb["detection_identity_sha256"] = sha256_json(one_stageb)
        (hybrid_dir / f"{task_id}.json").write_text(
            json.dumps(one_hybrid), encoding="utf-8"
        )
        (hair_dir / f"{task_id}.json").write_text(
            json.dumps(one_stageb), encoding="utf-8"
        )

    def fake_fusion(one_hybrid, _stageb, **_kwargs):
        fused = deepcopy(one_hybrid)
        fused["phaxis"] = {
            "hair_identity_count_expert": (
                "train399-expert" if one_hybrid["task_id"] == "T1" else "legacy-expert"
            ),
            "previous_hybrid_identity_count": 1,
            "formal_stageb_identity_count": 2,
            "attachment_valid_fraction": 1.0,
            "root_lock_sha256": "f" * 64,
            "length_identity_association": {"matched_length_identities": 1},
        }
        return fused

    monkeypatch.setattr(cli_module, "fuse_hybrid_root_with_stageb_hairs", fake_fusion)
    status = cli_module.main(
        [
            "fuse",
            "--hybrid-predictions",
            str(hybrid_dir),
            "--hybrid-root",
            str(artifact_root),
            "--hair-detections",
            str(hair_dir),
            "--model-contract-proposal",
            str(proposal),
            "--output",
            str(tmp_path / "output"),
        ]
    )
    assert status == 2
    assert "mixed Stage-B expert identities" in capsys.readouterr().err
