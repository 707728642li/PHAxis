from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from phaxis import _toml_compat as tomllib
import zipfile

import pytest
from packaging.requirements import Requirement


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "phaxis"
FULL_PROJECT_PARITY_EVIDENCE = (
    PROJECT_ROOT
    / "outputs"
    / "phaxis_rhaxiscc_metric_parity_20260828"
    / "audit.json"
)

sys.path.insert(0, str(SCRIPT_ROOT))

import source_release_common as source_release_module  # noqa: E402
from phaxis.public_identity import (  # noqa: E402
    PUBLIC_SYSTEM_DERIVATION,
    PUBLIC_SYSTEM_IDENTITY_SCHEMA,
    derive_public_identity,
)
from phaxis.evaluation_metrics import (  # noqa: E402
    biological_hair_presence_matcher_contract,
)
from phaxis.hair_stageb.candidate_bundle import (  # noqa: E402
    PREREGISTERED_SCORE_THRESHOLDS,
    operating_point_selection_contract,
)

from source_release_common import (  # noqa: E402
    BLOCKED_RECEIPT_NAME,
    SourceReleaseError,
    build_source_release,
    inspect_formal_release_gate,
    verify_source_release,
)


def _canonical_hash(payload) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _selection_prf(true_positive: int, predicted: int, ground_truth: int) -> dict:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / ground_truth if ground_truth else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": true_positive,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _biological_selection_rows() -> tuple[list[dict], dict]:
    """Synthetic sufficient statistics with a unique biological-presence optimum."""

    rows: list[dict] = []
    for threshold in PREREGISTERED_SCORE_THRESHOLDS:
        if threshold < 0.225:
            predicted, biological_tp = 2, 1
        elif threshold == 0.225:
            predicted, biological_tp = 1, 1
        else:
            predicted, biological_tp = 0, 0
        per_image = []
        for index in range(44):
            matches = (
                [
                    {
                        "predicted_index_after_threshold": 0,
                        "annotated_index": 0,
                        "prediction_coverage": 0.50,
                        "truth_coverage": 0.75,
                        "proximal_direction_cosine": 0.50,
                    }
                ]
                if biological_tp
                else []
            )
            per_image.append(
                {
                    "task_id": f"QCDEV-{index:02d}",
                    "predicted": predicted,
                    "ground_truth": 1,
                    "biological_presence_true_positive_20um": biological_tp,
                    # Deliberately independent: a truncated but biologically
                    # matched hair need not pass the attachment localization tier.
                    "attachment_proxy_true_positive_20um": 0,
                    "biological_presence_matched_pairs": matches,
                    "count_error": predicted - 1,
                }
            )
        predicted_total = 44 * predicted
        biological_total = 44 * biological_tp
        rows.append(
            {
                "threshold": float(threshold),
                "tolerant_biological_presence_20um": _selection_prf(
                    biological_total, predicted_total, 44
                ),
                "identity_attachment_proxy_20um": _selection_prf(
                    0, predicted_total, 44
                ),
                "count_mae": float(abs(predicted - 1)),
                "count_bias": float(predicted - 1),
                "per_image": per_image,
            }
        )
    selected = next(row for row in rows if row["threshold"] == 0.225)
    return rows, selected


def _synthetic_hybrid_lock_identity() -> str:
    records = [
        {
            "task_id": f"QCDEV-{index:02d}",
            "sha256": _hash(f"hybrid-QCDEV-{index:02d}"),
        }
        for index in range(44)
    ]
    return _canonical_hash(records)


def test_source_release_binds_exact_runtime_biological_matcher_contract() -> None:
    assert (
        source_release_module.BIOLOGICAL_PRESENCE_MATCHER_CONTRACT
        == biological_hair_presence_matcher_contract()
    )


@pytest.fixture(autouse=True)
def _use_synthetic_hybrid_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep fixtures synthetic while production retains the fixed comparator hash."""

    monkeypatch.setattr(
        source_release_module,
        "LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256",
        _synthetic_hybrid_lock_identity(),
    )


def _evaluation_v12_contract_fixture() -> dict:
    per_image = [
        {
            "task_id": f"QCDEV-{index:02d}",
            "stageb_train399": {
                "biological_presence_tp": {"5.0": 1, "10.0": 1, "20.0": 1}
            },
            "hybrid_max": {
                "biological_presence_tp": {"5.0": 1, "10.0": 1, "20.0": 1}
            },
        }
        for index in range(44)
    ]
    stageb_locks = [
        {"task_id": row["task_id"], "sha256": _hash(f"stageb-{row['task_id']}")}
        for row in per_image
    ]
    hybrid_locks = [
        {"task_id": row["task_id"], "sha256": _hash(f"hybrid-{row['task_id']}")}
        for row in per_image
    ]
    metric_set = {
        name: {tolerance: {} for tolerance in ("5", "10", "20")}
        for name in (
            "tolerant_biological_presence",
            "identity_attachment_proxy",
            "strict_whole_line_correspondence",
        )
    }
    hybrid_identity = _canonical_hash(hybrid_locks)
    matcher = biological_hair_presence_matcher_contract()
    return {
        "metric_hierarchy": {
            "primary": (
                "one-to-one tolerant biological-hair presence; bidirectional "
                "partial centreline coverage without endpoint gates"
            ),
            "primary_minimum_truth_coverage": 0.25,
            "primary_minimum_prediction_coverage": 0.25,
            "primary_minimum_direction_cosine": 0.0,
            "primary_tolerance_um": 20.0,
            "primary_matcher_contract": matcher,
            "primary_matcher_contract_sha256": _canonical_hash(matcher),
        },
        "overall": {
            "stageb_train399": {"images": 44, **metric_set},
            "hybrid_max": {"images": 44, **metric_set},
        },
        "paired_bootstrap_95ci": {
            "method": "paired image-level nonparametric bootstrap",
            "repetitions": 10_000,
            "seed": 20260828,
            "delta_stageb_train399_minus_hybrid": {
                "biological_presence_f1_20um": {
                    "lower_2_5": -0.1,
                    "upper_97_5": 0.1,
                }
            },
        },
        "per_image": per_image,
        "prediction_input_locks": {
            "stageb_detection_files": stageb_locks,
            "stageb_detection_set_identity_sha256": _canonical_hash(stageb_locks),
            "hybrid_prediction_files": hybrid_locks,
            "hybrid_prediction_set_identity_sha256": hybrid_identity,
        },
        "comparator_contract": {
            "hybrid_max": {
                "evidence_role": "locked_legacy_development_comparator",
                "schema_version": (
                    "RHAxis-NextGen-Hybrid-Max-qc-development-evaluation-1.0"
                ),
                "identity_hair_variant": "hybrid_verified_increment",
                "count_hair_variant": "hybrid_verified_increment",
                "endpoint_complete_identity_layer": True,
                "phaxis_payload_allowed": False,
                "stageb_identity_source_allowed": False,
                "prediction_set_identity_sha256": hybrid_identity,
                "expected_prediction_set_identity_sha256": hybrid_identity,
            }
        },
    }


def _release_human_metadata(project: Path, output: Path) -> Path:
    license_path = project / "LICENSE"
    license_path.parent.mkdir(parents=True, exist_ok=True)
    if not license_path.exists():
        license_path.write_text("Apache License 2.0 test fixture\n", encoding="utf-8")
    repository = "https://github.com/example/phaxis"
    payload = {
        "schema_version": "PHAxis-release-human-metadata-1.3",
        "status": "author_verified_release_authority",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "distribution": "phaxis",
        "authors": [
            {
                "display_name": "PHAxis Test Author",
                "given_names": "PHAxis Test",
                "family_names": "Author",
                "email": "author@example.org",
                "affiliation": "Example Plant Science Institute",
                "orcid": "https://orcid.org/0000-0002-1825-0097",
            }
        ],
        "maintainers": [
            {
                "display_name": "PHAxis Test Maintainer",
                "given_names": "PHAxis Test",
                "family_names": "Maintainer",
                "email": "maintainer@example.org",
                "affiliation": "Example Plant Science Institute",
                "orcid": "https://orcid.org/0000-0001-5109-3700",
            }
        ],
        "project_urls": {
            "Homepage": "https://example.org/phaxis",
            "Repository": repository,
            "Issues": f"{repository}/issues",
            "Documentation": "https://example.org/phaxis/docs",
        },
        "release_coordinates": {
            "github_repository_url": repository,
            "github_release_tag": "v1.0.0",
            "github_release_url": f"{repository}/releases/tag/v1.0.0",
            "pypi_project": "phaxis",
            "pypi_version": "1.0.0",
            "pypi_project_url": "https://pypi.org/project/phaxis/1.0.0/",
            "release_date": "2026-08-29",
            "release_doi": "10.5281/zenodo.1234567",
        },
        "rights": {
            "source_license_spdx": "Apache-2.0",
            "license_file_sha256": _file_hash(license_path),
            "source_release_authorized": True,
            "model_weights_included": False,
            "images_included": False,
            "annotations_included": False,
            "separate_asset_rights_not_conferred": True,
        },
    }
    payload["metadata_identity_sha256"] = _canonical_hash(payload)
    _write_json(output, payload)
    return output


def _formal_gate_fixture(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "project"
    evaluator = project / "scripts/phaxis/evaluate_stageb_train399_qcdev44.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text(
        "from phaxis.evaluation_metrics import evaluate_image, presence_match_strict, prf\n",
        encoding="utf-8",
    )
    model_contract = project / "configs/phaxis/v1_0/model_contract.json"
    _write_json(model_contract, {"formal_release_status": "passed"})

    training_lock = {
        "dataset_manifest_sha256": _hash("dataset"),
        "split_manifest_sha256": _hash("split"),
        "dataset_split_identity_sha256": _hash("split-identity"),
        "integrity_manifest_sha256": _hash("integrity"),
    }
    checkpoint_hashes = [_hash(f"checkpoint-{index}") for index in range(5)]
    identity_payload = {
        "ensemble_members": 5,
        "training_images": 399,
        "validation_images": 44,
        "validation_labels_used_for_gradient_or_early_stopping": False,
        "blind_images_used": 0,
        "training_lock": training_lock,
        "training_lock_identity_sha256": _canonical_hash(training_lock),
        "operating_point_selection_contract": operating_point_selection_contract(),
        "members": [
            {"member_index": index, "checkpoint_sha256": digest}
            for index, digest in enumerate(checkpoint_hashes)
        ],
    }
    candidate = {
        "schema_version": "PHAxis-StageB-train399-candidate-bundle-1.0",
        "status": "candidate_gate_passed_not_promoted",
        "candidate_only": True,
        "automatic_promotion_performed": False,
        "official_constants_modified": False,
        "official_model_contract_modified": False,
        "blind_images_used": 0,
        "identity_payload": identity_payload,
        "candidate_bundle_identity_sha256": _canonical_hash(identity_payload),
        "detection_model_metadata": {
            "expert_id": "PHAxis-StageB-train399-five-seed",
            "deployment_role": "candidate_gate_passed_not_promoted",
            "ensemble_members": 5,
            "training_images": 399,
            "validation_images": 44,
            "validation_labels_used_for_gradient_or_early_stopping": False,
            "blind_images_used": 0,
            "checkpoint_sha256": checkpoint_hashes,
            "candidate_bundle_identity_sha256": _canonical_hash(identity_payload),
        },
    }
    candidate["candidate_manifest_identity_sha256"] = _canonical_hash(candidate)
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, candidate)

    matcher = biological_hair_presence_matcher_contract()
    threshold_metrics, selected = _biological_selection_rows()
    selection = {
        "schema_version": "PHAxis-StageB-train399-QCdev44-selection-receipt-1.3",
        "status": "completed",
        "images": 44,
        "blind_images_used": 0,
        "independent_accuracy_claim_allowed": False,
        "validation_labels_used_for_gradient_or_early_stopping": False,
        "straight_base_to_tip_presence_proxy_evaluated_during_selection": True,
        "distal_endpoint_error_used_as_selection_gate": False,
        "complete_line_overlap_used_as_selection_gate": False,
        "length_error_used_as_selection_gate": False,
        "manual_hair_width_assumed": False,
        "primary_matcher_contract": matcher,
        "primary_matcher_contract_sha256": _canonical_hash(matcher),
        "selection_contract": operating_point_selection_contract(),
        "candidate_bundle_identity_sha256": candidate[
            "candidate_bundle_identity_sha256"
        ],
        "candidate_pool_identity_sha256": _hash("candidate-pool"),
        "canonical_ground_truth_lock_identity_sha256": _hash("canonical"),
        "threshold_metrics": threshold_metrics,
        "selected": selected,
        **training_lock,
    }
    selection["selection_receipt_identity_sha256"] = _canonical_hash(selection)
    selection_path = tmp_path / "selection.json"
    _write_json(selection_path, selection)

    evaluation = {
        "schema_version": (
            "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2"
        ),
        "status": "completed",
        "blind_images_used": 0,
        "independent_accuracy_claim_allowed": False,
        "training_contract": {
            "training_images": 399,
            "validation_images": 44,
            "validation_labels_used_for_gradient_or_early_stopping": False,
            "candidate_bundle_identity_sha256": candidate[
                "candidate_bundle_identity_sha256"
            ],
            "selection_receipt_identity_sha256": selection[
                "selection_receipt_identity_sha256"
            ],
            "selected_model_metadata_identity_sha256": _hash(
                "selected-model-metadata-identity"
            ),
            "checkpoint_sha256": checkpoint_hashes,
        },
        "inputs_sha256": {
            "candidate_manifest": _file_hash(candidate_path),
            "selection_receipt": _file_hash(selection_path),
            "dataset_manifest": training_lock["dataset_manifest_sha256"],
            "split_manifest": training_lock["split_manifest_sha256"],
            "integrity_manifest": training_lock["integrity_manifest_sha256"],
            "canonical_ground_truth_lock_identity": selection[
                "canonical_ground_truth_lock_identity_sha256"
            ],
            "selected_model_metadata": _hash("selected-metadata"),
        },
        **_evaluation_v12_contract_fixture(),
    }
    evaluation_path = tmp_path / "evaluation.json"
    _write_json(evaluation_path, evaluation)

    layers = {
        name: {
            "exact": 283,
            "expected": 283,
            "mismatch_count": 0,
            "mismatch_task_ids": [],
            "gate_pass": True,
        }
        for name in (
            "v12_strip_root_mask",
            "v20_root_polygon",
            "final_hybrid_root_mask",
        )
    }
    root_identity = {
        "schema_version": "PHAxis-root-provider-fresh-reference283-audit-1.0",
        "reference_identity_sha256": _hash("root-reference"),
        "fresh_reference_identity_sha256": _hash("fresh-reference"),
        "bundle_identity_sha256": _hash("root-bundle"),
        "pipeline_identity_sha256": _hash("pipeline"),
        "layers": layers,
        "source_image_mismatch_task_ids": [],
        "prepared_radius_fallback_task_ids": ["fallback-a", "fallback-b"],
        "attachment_supported_extension_rescue_task_ids": [
            f"rescue-{index}" for index in range(7)
        ],
        "pipeline_raw_image_provenance_gate": True,
        "pipeline_stage_evidence_gate": True,
    }
    root_receipt = {
        **root_identity,
        "status": "pass_exact_283",
        "created_utc": "ignored-by-logical-identity",
        "audit_identity_sha256": _canonical_hash(root_identity),
        "fresh_portable_raw_image_rerun_completed": True,
        "fresh_283_exact_reproduction_claim_allowed": True,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }
    root_path = tmp_path / "root-exact283.json"
    _write_json(root_path, root_receipt)
    stageb_binding = {
        "expert_id": candidate["detection_model_metadata"]["expert_id"],
        "checkpoint_sha256": checkpoint_hashes,
        "selected_score_threshold": 0.225,
        "candidate_bundle_identity_sha256": candidate[
            "candidate_bundle_identity_sha256"
        ],
        "selection_receipt_identity_sha256": selection[
            "selection_receipt_identity_sha256"
        ],
        "selected_model_metadata_identity_sha256": evaluation[
            "training_contract"
        ]["selected_model_metadata_identity_sha256"],
    }
    public = derive_public_identity(
        stageb_binding,
        root_bundle_identity_sha256=root_receipt["bundle_identity_sha256"],
    )
    model_bundle_id = public["model_bundle_id"]
    root_provider_id = public["root_expert_id"]
    proposal_file_sha256 = _hash("proposal-file")
    proposal_identity_sha256 = _hash("proposal-identity")
    proposal_receipt_binding = {
        "model_contract_proposal_sha256": proposal_file_sha256,
        "model_contract_proposal_identity_sha256": proposal_identity_sha256,
    }
    fusion_summary = {
        "schema_version": "PHAxis-fusion-run-1.1",
        "status": "completed",
        "images": 283,
        "model_bundle_id": model_bundle_id,
        "root_expert": root_provider_id,
        **proposal_receipt_binding,
    }
    fusion_summary["summary_identity_sha256"] = _canonical_hash(fusion_summary)
    fusion_path = tmp_path / "final-fusion-summary.json"
    _write_json(fusion_path, fusion_summary)
    traits_summary = {
        "schema_version": "PHAxis-trait-export-1.0",
        "status": "completed",
        "tasks": 283,
        "model_bundle_id": model_bundle_id,
        "root_expert_id": root_provider_id,
        **proposal_receipt_binding,
    }
    traits_summary["export_identity_sha256"] = _canonical_hash(traits_summary)
    traits_path = tmp_path / "final-traits-summary.json"
    _write_json(traits_path, traits_summary)
    qcdev_evidence = {
        "role": "locked_qcdevelopment44_non_independent_development_evidence",
        "schema_version": evaluation["schema_version"],
        "metric_hierarchy": evaluation["metric_hierarchy"],
        "stageb_train399": evaluation["overall"]["stageb_train399"],
        "same_run_historical_endpoint_complete_comparator": {
            "role": "historical_comparator_recomputed_by_evaluator_1_2",
            "source_prediction_contract": evaluation["comparator_contract"][
                "hybrid_max"
            ],
        },
        "paired_bootstrap_95ci": evaluation["paired_bootstrap_95ci"],
        "prediction_input_locks": evaluation["prediction_input_locks"],
        "source": {
            "evaluation_sha256": _file_hash(evaluation_path),
            "evaluation_content_identity_sha256": _canonical_hash(evaluation),
        },
    }
    model_contract_payload = {
        "schema_version": "PHAxis-model-contract-1.0.0",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "model_bundle_id": model_bundle_id,
        "public_system_identity": {
            "schema_version": PUBLIC_SYSTEM_IDENTITY_SCHEMA,
            "identity_sha256": public["public_system_identity_sha256"],
            "derivation": PUBLIC_SYSTEM_DERIVATION,
        },
        "formal_release_status": "passed",
        "expert_boundary": {
            "root_point_scale_continuity_statistics": root_provider_id,
            "hair_identity_and_count": stageb_binding["expert_id"],
        },
        "root_expert": {
            "provider_role": public["root_provider_role"],
            "expert_id": root_provider_id,
            "fresh_exact283_receipt_sha256": _file_hash(root_path),
            "fresh_exact283_audit_identity_sha256": root_receipt[
                "audit_identity_sha256"
            ],
            "reference_identity_sha256": root_receipt[
                "reference_identity_sha256"
            ],
            "fresh_reference_identity_sha256": root_receipt[
                "fresh_reference_identity_sha256"
            ],
            "bundle_identity_sha256": root_receipt["bundle_identity_sha256"],
            "pipeline_identity_sha256": root_receipt["pipeline_identity_sha256"],
            "root_bundle_authority": {
                "bundle_identity_sha256": root_receipt[
                    "bundle_identity_sha256"
                ],
                "pipeline_identity_sha256": root_receipt[
                    "pipeline_identity_sha256"
                ],
            },
            "root_cap_region_output": False,
        },
        "hair_identity_count_expert": {
            "expert_id": stageb_binding["expert_id"],
            "checkpoint_sha256_in_member_order": checkpoint_hashes,
            "current_checkpoint_role": "formal_train399_only_deployment",
        },
        "development_evidence": {"qcdev44": qcdev_evidence},
        "red_lines": {
            "blind_images_used": 0,
            "canonical_annotations_read_during_inference": False,
            "condition_metadata_used_for_routing": False,
            "validation_labels_used_for_training_by_current_five_member_deployment_ensemble": False,
            "formal_train399_only_stageb_weights_available": True,
            "independent_accuracy_claimed": False,
            "root_cap_region_statistics_included": False,
        },
        "promotion": {
            "schema_version": "PHAxis-model-contract-promotion-1.0",
            "status": "applied_formal_release",
            "official_apply_performed": True,
            "formal_gate_source_sha256": {
                "train399_candidate": _file_hash(candidate_path),
                "train399_selection": _file_hash(selection_path),
                "train399_evaluation": _file_hash(evaluation_path),
                "root_exact283": _file_hash(root_path),
            },
            "formal_gate_identity_sha256": {
                "candidate_bundle_identity_sha256": candidate[
                    "candidate_bundle_identity_sha256"
                ],
                "selection_receipt_identity_sha256": selection[
                    "selection_receipt_identity_sha256"
                ],
                "selected_model_metadata_identity_sha256": evaluation[
                    "training_contract"
                ]["selected_model_metadata_identity_sha256"],
                "root_exact283_audit_identity_sha256": root_receipt[
                    "audit_identity_sha256"
                ],
            },
            "stageb_binding": stageb_binding,
            "proposal_file_sha256": proposal_file_sha256,
            "proposal_identity_sha256": proposal_identity_sha256,
            "expected_source_model_contract_sha256": _hash("blocked-contract"),
            "final_receipt_source_sha256": {
                "stageb": _hash("final-source-stageb"),
                "fusion": _file_hash(fusion_path),
                "traits": _file_hash(traits_path),
                "evidence": _hash("final-source-evidence"),
            },
            "final_receipt_identity_sha256": {
                "stageb": _hash("final-identity-stageb"),
                "fusion": fusion_summary["summary_identity_sha256"],
                "traits": traits_summary["export_identity_sha256"],
                "evidence": _hash("final-identity-evidence"),
            },
            "final_receipt_public_identity": {
                role: {
                    "model_bundle_id": model_bundle_id,
                    "root_expert_id": root_provider_id,
                }
                for role in ("fusion", "traits")
            },
        },
    }
    model_contract_payload["model_contract_identity_sha256"] = _canonical_hash(
        model_contract_payload
    )
    _write_json(model_contract, model_contract_payload)
    metadata_path = _release_human_metadata(
        project,
        tmp_path / "release-human-metadata.json",
    )
    return {
        "project": project,
        "root": root_path,
        "candidate": candidate_path,
        "selection": selection_path,
        "evaluation": evaluation_path,
        "fusion": fusion_path,
        "traits": traits_path,
        "metadata": metadata_path,
    }


def test_release_allowlist_and_extras_cover_phaxis_owned_paper_pipeline() -> None:
    for path in (
        "scripts/phaxis/analyze_biological_cohorts.py",
        "scripts/phaxis/assemble_post_training_release_manifest.py",
        "scripts/phaxis/assemble_handover_build_contract.py",
        "scripts/phaxis/audit_biological_analysis_equivalence.py",
        "scripts/phaxis/benchmark_full_workflow.py",
        "scripts/phaxis/build_analysis_workflow_manifest.py",
        "scripts/phaxis/build_benchmark_artifact_inventory.py",
        "scripts/phaxis/build_clean_install_expected_identity.py",
        "scripts/phaxis/build_clean_install_sample_manifest.py",
        "scripts/phaxis/build_clean_install_verification.py",
        "scripts/phaxis/build_direct_benchmark_provider_descriptor.py",
        "scripts/phaxis/build_handover_benchmark_manifest.py",
        "scripts/phaxis/build_handover_dataset_manifest.py",
        "scripts/phaxis/build_handover_image_manifest.py",
        "scripts/phaxis/build_handover_model_asset_manifest.py",
        "scripts/phaxis/build_handover_model_source_manifest.py",
        "scripts/phaxis/build_post_training_release_stage_contract.py",
        "scripts/phaxis/build_manuscript_evidence_manifest.py",
        "scripts/phaxis/build_manuscript_values.py",
        "scripts/phaxis/build_submission_docx.py",
        "scripts/phaxis/build_supplementary_docx.py",
        "scripts/phaxis/build_supplementary_table_data_bundle.py",
        "scripts/phaxis/build_publication_figure_inputs.py",
        "scripts/phaxis/build_publication_figures.py",
        "scripts/phaxis/build_qcdev44_root_provider_inputs.py",
        "scripts/phaxis/build_release_case_prelocks.py",
        "scripts/phaxis/build_release_distributions.py",
        "scripts/phaxis/check_post_training_release_topology.py",
        "scripts/phaxis/compile_manuscript.py",
        "scripts/phaxis/compare_hair_experts_biological_presence.py",
        "scripts/phaxis/export_traits.py",
        "scripts/phaxis/handover_manifest_producers.py",
        "scripts/phaxis/materialize_figure1_geometry.py",
        "scripts/phaxis/materialize_offline_dependencies.py",
        "scripts/phaxis/promote_model_contract.py",
        "scripts/phaxis/run_external_direct_benchmark.py",
        "scripts/phaxis/run_post_training_release.py",
        "scripts/phaxis/run_stageb_evaluation_inference.py",
    ):
        assert path in source_release_module.SCRIPT_FILES
    assert any(
        rule.source == "src/phaxis" and ".py" in rule.suffixes
        for rule in source_release_module.TREE_RULES
    ), "release_topology.py must travel with the complete phaxis package tree"
    for path in (
        "docs/phaxis/PHAXIS_BIOLOGICAL_ACQUISITION_METADATA_COMPLETION_CN_20260829.md",
        "docs/phaxis/PHAXIS_GITHUB_PYPI_RELEASE_GUIDE_CN_20260828.md",
        "docs/phaxis/PHAXIS_MANUSCRIPT_FIGURE_INPUT_CONTRACT_20260828.md",
        "docs/phaxis/PHAXIS_MANUSCRIPT_REFERENCE_AUDIT_20260828.md",
        "docs/phaxis/PHAXIS_MEASUREMENT_ASSURANCE_CONTRACT_CN_20260829.md",
        "docs/phaxis/PHAXIS_SUBMISSION_DOCX_LAYOUT_QA_20260828.md",
        "docs/phaxis/PHAXIS_SUPPLEMENTARY_DOCX_LAYOUT_QA_20260829.md",
        "docs/phaxis/PHAXIS_MANUSCRIPT_VALUES_COMPILER_CONTRACT_20260828.md",
        "docs/phaxis/TRAIT_CONTRACT_CN.md",
        "docs/phaxis/USER_GUIDE.md",
    ):
        assert path in source_release_module.DOCUMENT_FILES
    assert set(source_release_module.UNCOMPILED_MANUSCRIPT_FILES).isdisjoint(
        source_release_module.DOCUMENT_FILES
    )
    assert (
        "docs/phaxis/PHAXIS_POSTTRAIN_RELEASE_REACHABILITY_AUDIT_20260829.md"
        not in source_release_module.DOCUMENT_FILES
    )
    assert (
        "manuscript/phaxis_v1_0/PUBLICATION_MISSING_RESULTS_MATRIX_20260829_R2"
        not in source_release_module.DOCUMENT_FILES
    )
    release_guide = (
        PROJECT_ROOT / "docs/phaxis/PHAXIS_GITHUB_PYPI_RELEASE_GUIDE_CN_20260828.md"
    ).read_text(encoding="utf-8")
    for token in (
        "--hold-physical-gpu 0",
        "--launch",
        "paused_for_user_gpu_hold",
        "退出码 `5`",
        "--execute --resume",
        "不会为该阶段调用 `nvidia-smi`、GPU probe 或",
        "frozen-v1 GPU0",
    ):
        assert token in " ".join(release_guide.split())
    assert "configs/phaxis/v1_0/locked_qcdevelopment44_ids.txt" in (
        source_release_module.SINGLE_FILES
    )
    release_manifest_path = PROJECT_ROOT / "SOURCE_MANIFEST.json"
    if release_manifest_path.is_file():
        release_manifest = json.loads(
            release_manifest_path.read_text(encoding="utf-8")
        )
        assert release_manifest["schema_version"] == (
            "PHAxis-source-release-manifest-2.0"
        )
        collected_destinations = {
            row["path"] for row in release_manifest["files"]
        }
    else:
        collected_destinations = {
            destination
            for _source, destination in source_release_module.collect_allowlisted_sources(
                PROJECT_ROOT
            )
        }
    for path in (
        "src/phaxis/hair_stageb/training.py",
        "src/phaxis/hair_stageb/candidate_bundle.py",
        "src/phaxis/root_trait_assurance.py",
        "src/phaxis/multitrait_atlas.py",
        "scripts/phaxis/run_stageb_train399_gpu_queue.ps1",
        "tests/phaxis/test_root_trait_assurance.py",
        "tests/phaxis/test_multitrait_atlas.py",
        "tests/phaxis/test_phenotype_catalog.py",
        "tests/phaxis/test_stageb_amp_amendment.py",
        "tests/phaxis/test_submission_docx_builder.py",
        "tests/phaxis/test_supplementary_docx_builder.py",
        "tests/phaxis/test_supplementary_table_data_bundle.py",
        "scripts/phaxis/build_supplementary_table_data_bundle.py",
        "tests/phaxis/fixtures/hair_attachment_assurance_input.json",
        "tests/phaxis/fixtures/root_continuity_assurance_input.json",
        "configs/phaxis/v1_0/stageb_train399_training_config.json",
        "evidence/stageb_amp_backward_retry_amendment.json",
        "MODEL_CARD.md",
        "DATA_CARD.md",
    ):
        assert path in collected_destinations
    assert (
        source_release_module.AMP_AMENDMENT_SOURCE,
        source_release_module.AMP_AMENDMENT_DESTINATION,
    ) in source_release_module.MAPPED_FILES
    plugin = source_release_module._source_release_pytest_plugin()
    assert "tests/phaxis/test_stageb_amp_amendment.py::" in plugin
    assert (
        "test_amp_amendment_binds_failure_source_restart_and_legacy_zero_retry"
        in plugin
    )
    for node in (
        "test_descriptor_builder_seals_real_four_mode_closures_and_is_create_only",
        "test_descriptor_builder_check_subprocess_is_explicitly_cpu_only",
        "test_provider_object_seal_rejects_any_descriptor_tamper",
    ):
        assert node in plugin
    pyproject = tomllib.loads(source_release_module._pyproject(formal=False))
    assert pyproject["build-system"]["requires"] == [
        "setuptools>=77",
        "wheel>=0.45",
    ]
    assert pyproject["project"]["scripts"] == {"phaxis": "phaxis.cli:main"}
    assert "packaging>=24,<26" in pyproject["project"]["dependencies"]
    assert all(
        Requirement(requirement).name != "tomli"
        for requirement in pyproject["project"]["dependencies"]
    )
    assert pyproject["project"]["license-files"] == [
        "LICENSE",
        "src/phaxis/_vendor/tomli/LICENSE.txt",
    ]
    assert pyproject["tool"]["setuptools"]["package-data"] == {
        "phaxis._vendor.tomli": ["LICENSE.txt", "py.typed"]
    }
    extras = pyproject["project"]["optional-dependencies"]
    for name in ("analysis", "test"):
        assert "pandas>=2.2,<4" in extras[name]
        assert "scikit-image>=0.24,<0.27" in extras[name]
        assert "statsmodels>=0.14,<1" in extras[name]
    assert "opencv-python-headless>=4.9,<6" in extras["analysis"]
    assert "tifffile>=2024.8,<2027" in extras["analysis"]
    assert "Pillow>=10,<13" in extras["publication"]
    assert "python-docx>=1.1,<2" in extras["publication"]
    assert "python-docx>=1.1,<2" in extras["test"]
    assert set(extras["deployment"]) == {
        "imageio>=2.35,<3",
        "joblib>=1.4,<2",
        "matplotlib>=3.8,<4",
        "opencv-python-headless>=4.9,<6",
        "pandas>=2.2,<4",
        "Pillow>=10,<13",
        "scikit-image>=0.24,<0.27",
        "scikit-learn>=1.5,<2",
        "statsmodels>=0.14,<1",
        "tifffile>=2024.8,<2027",
        "timm>=1.0.28,<2",
        "torch>=2.6,<3",
        "torchvision>=0.21,<1",
    }
    readme = source_release_module._readme(formal=False)
    assert 'python -m pip install ".[deployment]"' in readme
    compact_readme = " ".join(readme.split())
    assert "five-member root-hair identity/count expert" in compact_readme
    assert "32 canonical image-derived descriptors" in compact_readme
    assert "does not report 82 phenotypes" in readme
    assert "docs/phaxis/TRAIT_CONTRACT_CN.md" in readme
    assert "docs/phaxis/USER_GUIDE.md" in readme
    assert "--model-contract official-contract.json" in readme
    assert "--execute --resume" in readme
    assert "build_manuscript_evidence_manifest.py" not in readme
    assert "Stage-B" not in readme
    assert "Stage B" not in readme
    assert "r16" not in readme.casefold()
    assert "r17" not in readme.casefold()


def test_public_phenotype_catalog_verifier_is_positive_and_fail_closed(
    tmp_path: Path,
) -> None:
    for relative in (
        "README.md",
        "MODEL_CARD.md",
        "DATA_CARD.md",
        "docs/phaxis/TRAIT_CONTRACT_CN.md",
        "docs/phaxis/USER_GUIDE.md",
        "configs/phaxis/v1_0/trait_contract.json",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((PROJECT_ROOT / relative).read_bytes())

    failures: list[str] = []
    source_release_module._verify_public_phenotype_catalog(tmp_path, failures)
    assert failures == []

    catalog = tmp_path / "docs/phaxis/TRAIT_CONTRACT_CN.md"
    catalog.write_text(
        catalog.read_text(encoding="utf-8")
        + "\nHybrid-Max\nscripts/phaxis/obsolete_internal_command.py\n",
        encoding="utf-8",
    )
    failures = []
    source_release_module._verify_public_phenotype_catalog(tmp_path, failures)
    combined = "\n".join(failures).casefold()
    assert "hybrid-max" in combined
    assert "scripts/phaxis/" in combined


def test_source_release_staging_does_not_repeat_long_destination_name(
    tmp_path: Path,
) -> None:
    destination = tmp_path / ("PHAxis-source-release-journal-facing-name-" * 5)
    staging = source_release_module._private_staging_path(destination)
    assert staging.parent == tmp_path
    assert staging.name.startswith(".source-release-staging-")
    assert destination.name not in staging.name


@pytest.mark.parametrize(
    "script",
    tuple(
        path
        for path in source_release_module.SCRIPT_FILES
        if path.endswith(".py")
        and path != "scripts/phaxis/source_release_common.py"
    ),
)
def test_allowlisted_cli_runs_isolated_from_checkout_without_installed_package(
    script: str,
) -> None:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", script, "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr


def test_biological_equivalence_evidence_requires_zero_difference_pass() -> None:
    receipt = {
        "schema_version": "PHAxis-biological-analysis-native-equivalence-audit-1.0",
        "status": "passed",
        "blind_images_used": 0,
        "canonical_annotations_read": False,
        "production_wrapper_imports_frozen_predecessor": False,
        "tables_equivalent": True,
        "tables_byte_identical": True,
        "total_differing_cells": 0,
        "tables": {
            f"table-{index}.csv": {"equivalent": True, "byte_identical": True}
            for index in range(6)
        },
    }
    assert source_release_module._biological_equivalence_receipt_ok(receipt)
    for field, invalid in (
        ("status", "failed"),
        ("blind_images_used", 1),
        ("tables_equivalent", False),
        ("tables_byte_identical", False),
        ("total_differing_cells", 1),
    ):
        tampered = json.loads(json.dumps(receipt))
        tampered[field] = invalid
        assert not source_release_module._biological_equivalence_receipt_ok(tampered)


def test_projected_evidence_specs_are_exact_distinct_roles() -> None:
    specs = source_release_module.PROJECTED_EVIDENCE_SPECS
    assert len(specs) == 2
    assert [spec.evidence_role for spec in specs] == [
        "pre_amendment_biological_equivalence_historical_baseline",
        "h11_raw_median_contract_amendment_current",
    ]
    assert len({spec.source for spec in specs}) == 2
    assert len({spec.destination for spec in specs}) == 2
    assert len({spec.evidence_role for spec in specs}) == 2
    assert specs[0].expected_source_receipt_sha256 == (
        source_release_module.PRE_AMENDMENT_EQUIVALENCE_AUTHORITY_SHA256
    )
    assert specs[0].expected_payload_identity_field is None
    assert specs[0].expected_payload_identity_sha256 is None
    assert specs[0].expected_projected_payload_identity_sha256 == (
        source_release_module.PRE_AMENDMENT_EQUIVALENCE_PROJECTED_IDENTITY_SHA256
    )
    assert specs[1].expected_source_receipt_sha256 == (
        source_release_module.H11_RAW_MEDIAN_AMENDMENT_AUTHORITY_SHA256
    )
    assert source_release_module.H11_RAW_MEDIAN_AMENDMENT_AUTHORITY_SHA256 == (
        "82570646cc28357e0a48b5c333ac9c978da76521695dfd643d6d103196393896"
    )
    assert specs[1].expected_payload_identity_field == (
        "amendment_audit_identity_sha256"
    )
    assert specs[1].expected_payload_identity_sha256 == (
        source_release_module.H11_RAW_MEDIAN_AMENDMENT_IDENTITY_SHA256
    )
    assert specs[1].expected_projected_payload_identity_sha256 is None
    assert source_release_module.H11_RAW_MEDIAN_AMENDMENT_IDENTITY_SHA256 == (
        "378b19e9b44d2bc563eee5bf4a8b864094bac803a0e5e9ee30d0fe99ece337bd"
    )
    assert source_release_module.H11_HISTORICAL_COHORT_ANALYSIS_CONTRACT_SHA256 == (
        "aaf5fb794986e705c6245217f88f67d5459b476f965a6185f12123cefd3625bf"
    )
    assert source_release_module.H11_HISTORICAL_COHORT_ANALYSIS_CONTRACT_SHA256 != (
        _file_hash(
            PROJECT_ROOT / "configs/phaxis/v1_0/biological_analysis_contract.json"
        )
    )
    assert (
        "scripts/phaxis/audit_stage22_h11_raw_median_amendment.py"
        in source_release_module.SCRIPT_FILES
    )
    assert not any(
        source_release_module.PROJECTED_EVIDENCE_SPECS[1].source == source
        for source, _destination in source_release_module.MAPPED_FILES
    )


def _real_h11_amendment_receipt() -> dict[str, Any]:
    source = (
        PROJECT_ROOT
        / source_release_module.PROJECTED_EVIDENCE_SPECS[1].source
    )
    return json.loads(source.read_text(encoding="utf-8"))


def _reseal_h11_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    receipt.pop("amendment_audit_identity_sha256", None)
    receipt.pop("release_projection", None)
    receipt["amendment_audit_identity_sha256"] = source_release_module._sha256_json(
        receipt
    )
    return receipt


def test_h11_amendment_receipt_semantic_validator_rejects_tampering() -> None:
    receipt = _real_h11_amendment_receipt()
    assert source_release_module._h11_raw_median_amendment_receipt_ok(receipt)

    for field, invalid in (
        ("status", "failed"),
        ("blind_images_used", 1),
        ("gpu_programs_started", True),
        ("canonical_annotations_read", True),
        ("root_cap_region_statistics_included", True),
        ("new_hypothesis_tests_added", 1),
        ("unauthorized_differing_cells", 1),
    ):
        tampered = deepcopy(receipt)
        tampered[field] = invalid
        _reseal_h11_receipt(tampered)
        assert not source_release_module._h11_raw_median_amendment_receipt_ok(
            tampered
        )

    mutations = []
    baseline = deepcopy(receipt)
    baseline["pre_amendment_baseline"]["authority_sha256"] = "0" * 64
    mutations.append(baseline)
    formula = deepcopy(receipt)
    formula["change_contract"]["construct_effect"] += "+0"
    mutations.append(formula)
    replicates = deepcopy(receipt)
    replicates["change_contract"]["bootstrap_replicates"] = 4999
    mutations.append(replicates)
    whitelist = deepcopy(receipt)
    whitelist["change_contract"]["changed_existing_columns_whitelist"].append(
        "estimate"
    )
    mutations.append(whitelist)
    cells = deepcopy(receipt)
    cells["tables"]["primary_clean_exploratory_factorial_tests.csv"]["H11"][
        "cell_counts"
    ]["EV22"] = 4
    mutations.append(cells)
    protected = deepcopy(receipt)
    protected["tables"]["full283_sensitivity_factorial_tests.csv"][
        "protected_differing_cells"
    ] = 1
    mutations.append(protected)
    standardized = deepcopy(receipt)
    standardized["tables"]["primary_clean_exploratory_factorial_tests.csv"][
        "H11"
    ]["effects"]["construct_OE_minus_EV"]["standardized_ci95_high"] += 0.01
    mutations.append(standardized)
    resealed_raw_effect = deepcopy(receipt)
    resealed_raw_effect_record = resealed_raw_effect["tables"][
        "primary_clean_exploratory_factorial_tests.csv"
    ]["H11"]
    resealed_raw_effect_values = resealed_raw_effect_record["effects"][
        "construct_OE_minus_EV"
    ]
    resealed_raw_effect_values["raw_effect_estimate"] += 1.0
    resealed_raw_effect_values["standardized_effect"] = (
        resealed_raw_effect_values["raw_effect_estimate"]
        / resealed_raw_effect_record["sample_standard_deviation"]
    )
    mutations.append(resealed_raw_effect)
    resealed_interval = deepcopy(receipt)
    resealed_interval_record = resealed_interval["tables"][
        "primary_clean_exploratory_factorial_tests.csv"
    ]["H11"]
    resealed_interval_values = resealed_interval_record["effects"][
        "construct_OE_minus_EV"
    ]
    resealed_interval_values["raw_effect_ci95_high"] += 1.0
    resealed_interval_values["standardized_ci95_high"] = (
        resealed_interval_values["raw_effect_ci95_high"]
        / resealed_interval_record["sample_standard_deviation"]
    )
    mutations.append(resealed_interval)
    locked_contract = deepcopy(receipt)
    locked_contract["locked_inputs"]["analysis_contract_sha256"] = "0" * 64
    mutations.append(locked_contract)
    extra_effect = deepcopy(receipt)
    extra_effect["tables"]["full283_sensitivity_factorial_tests.csv"]["H11"][
        "effects"
    ]["unexpected"] = deepcopy(
        extra_effect["tables"]["full283_sensitivity_factorial_tests.csv"]["H11"][
            "effects"
        ]["construct_OE_minus_EV"]
    )
    mutations.append(extra_effect)
    for tampered in mutations:
        _reseal_h11_receipt(tampered)
        assert not source_release_module._h11_raw_median_amendment_receipt_ok(
            tampered
        )

    identity_only = deepcopy(receipt)
    identity_only["change_contract"]["effective_seed"] = 1
    assert not source_release_module._h11_raw_median_amendment_receipt_ok(
        identity_only
    )


@pytest.mark.parametrize("summary_kind", ("median", "mean"))
def test_h11_amendment_validator_independently_checks_cell_summary_algebra(
    monkeypatch: pytest.MonkeyPatch,
    summary_kind: str,
) -> None:
    receipt = _real_h11_amendment_receipt()
    table = receipt["tables"]["primary_clean_exploratory_factorial_tests.csv"][
        "H11"
    ]
    values = table["effects"]["construct_OE_minus_EV"]
    if summary_kind == "median":
        table["cell_medians"]["OE22"] += 1.0
        values["raw_effect_estimate"] += 0.5
        values["standardized_effect"] = (
            values["raw_effect_estimate"] / table["sample_standard_deviation"]
        )
        # Break only the independent four-cell algebra after making the row's
        # raw/standardized relation internally self-consistent.
        values["raw_effect_estimate"] += 0.25
        values["standardized_effect"] = (
            values["raw_effect_estimate"] / table["sample_standard_deviation"]
        )
    else:
        table["cell_means"]["OE22"] += 1.0
        values["historical_raw_mean_contrast"] += 0.75
    _reseal_h11_receipt(receipt)
    monkeypatch.setattr(
        source_release_module,
        "H11_RAW_MEDIAN_AMENDMENT_IDENTITY_SHA256",
        receipt["amendment_audit_identity_sha256"],
    )
    assert not source_release_module._h11_raw_median_amendment_receipt_ok(receipt)


@pytest.mark.parametrize(
    "field",
    (
        "unauthorized_differing_cells",
        "new_hypothesis_tests_added",
        "gpu_programs_started",
        "blind_images_used",
    ),
)
def test_h11_amendment_validator_rejects_boolean_integer_lookalikes(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    receipt = _real_h11_amendment_receipt()
    receipt[field] = False
    _reseal_h11_receipt(receipt)
    monkeypatch.setattr(
        source_release_module,
        "H11_RAW_MEDIAN_AMENDMENT_IDENTITY_SHA256",
        receipt["amendment_audit_identity_sha256"],
    )
    assert not source_release_module._h11_raw_median_amendment_receipt_ok(receipt)


def test_historical_validator_rejects_boolean_integer_lookalike() -> None:
    spec = source_release_module.PROJECTED_EVIDENCE_SPECS[0]
    receipt = json.loads(
        (PROJECT_ROOT / spec.source).read_text(encoding="utf-8")
    )
    receipt["blind_images_used"] = False
    assert not source_release_module._pre_amendment_biological_equivalence_receipt_ok(
        receipt
    )


def test_h11_historical_contract_binding_is_field_level_not_identity_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _real_h11_amendment_receipt()
    receipt["locked_inputs"]["analysis_contract_sha256"] = "0" * 64
    _reseal_h11_receipt(receipt)
    monkeypatch.setattr(
        source_release_module,
        "H11_RAW_MEDIAN_AMENDMENT_IDENTITY_SHA256",
        receipt["amendment_audit_identity_sha256"],
    )
    assert not source_release_module._h11_raw_median_amendment_receipt_ok(receipt)


def test_release_safe_biological_projection_rebases_and_binds_authority(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = (
        project
        / "outputs"
        / "phaxis_biological_analysis_native_modelspec_audit_final_20260828"
        / "equivalence_audit.json"
    )
    receipt = {
        "schema_version": "PHAxis-biological-analysis-native-equivalence-audit-1.0",
        "status": "passed",
        "blind_images_used": 0,
        "canonical_annotations_read": False,
        "production_wrapper_imports_frozen_predecessor": False,
        "tables_equivalent": True,
        "tables_byte_identical": True,
        "total_differing_cells": 0,
        "baseline_analysis": str(project / "outputs" / "baseline"),
        "candidate_analysis": str(project / "outputs" / "candidate"),
        "tables": {
            f"table-{index}.csv": {
                "equivalent": True,
                "byte_identical": True,
                "baseline_path": str(
                    project / "outputs" / "baseline" / f"table-{index}.csv"
                ),
            }
            for index in range(6)
        },
    }
    _write_json(source, receipt)
    synthetic_spec = source_release_module.ProjectedEvidenceSpec(
        source=source.relative_to(project).as_posix(),
        destination="evidence/biological_analysis_equivalence_audit.json",
        evidence_role="synthetic_rebasing_test_only",
        validator_kind="pre_amendment_biological_equivalence",
    )
    projected = source_release_module._project_evidence_receipt(
        project_root=project,
        source=source,
        spec=synthetic_spec,
    )
    projection = projected["release_projection"]
    assert projection["source_receipt_sha256"] == _file_hash(source)
    assert projection["rebased_project_absolute_paths"] == 8
    assert projected["baseline_analysis"] == "project:outputs/baseline"
    assert projected["candidate_analysis"] == "project:outputs/candidate"
    assert not source_release_module._absolute_host_path_markers(
        json.dumps(projected)
    )

    receipt["baseline_analysis"] = "Z:" + "\\outside\\baseline"
    _write_json(source, receipt)
    with pytest.raises(SourceReleaseError, match="outside the project root"):
        source_release_module._project_evidence_receipt(
            project_root=project,
            source=source,
            spec=synthetic_spec,
        )


def test_real_historical_projection_binds_raw_and_projected_identities() -> None:
    spec = source_release_module.PROJECTED_EVIDENCE_SPECS[0]
    source = PROJECT_ROOT / spec.source
    projected = source_release_module._project_evidence_receipt(
        project_root=PROJECT_ROOT,
        source=source,
        spec=spec,
    )
    projection = projected.pop("release_projection")
    assert projection["source_receipt_sha256"] == (
        spec.expected_source_receipt_sha256
    )
    assert spec.expected_source_receipt_sha256 == (
        source_release_module.PRE_AMENDMENT_EQUIVALENCE_AUTHORITY_SHA256
    )
    assert source_release_module._sha256_json(projected) == (
        spec.expected_projected_payload_identity_sha256
    )
    assert spec.expected_projected_payload_identity_sha256 == (
        source_release_module.PRE_AMENDMENT_EQUIVALENCE_PROJECTED_IDENTITY_SHA256
    )


def test_real_historical_projection_rejects_resealed_raw_authority_source(
    tmp_path: Path,
) -> None:
    spec = source_release_module.PROJECTED_EVIDENCE_SPECS[0]
    project = tmp_path / "project"
    source = project / spec.source
    receipt = json.loads(
        (PROJECT_ROOT / spec.source).read_text(encoding="utf-8")
    )
    table = next(iter(receipt["tables"].values()))
    table["baseline_sha256"] = "0" * 64
    table["candidate_sha256"] = "0" * 64
    _write_json(source, receipt)
    with pytest.raises(
        SourceReleaseError,
        match="immutable standalone release anchor",
    ):
        source_release_module._project_evidence_receipt(
            project_root=project,
            source=source,
            spec=spec,
        )


def test_release_safe_h11_projection_binds_current_role_and_identity() -> None:
    spec = source_release_module.PROJECTED_EVIDENCE_SPECS[1]
    source = PROJECT_ROOT / spec.source
    projected = source_release_module._project_evidence_receipt(
        project_root=PROJECT_ROOT,
        source=source,
        spec=spec,
    )
    projection = projected["release_projection"]
    assert projection == {
        "schema_version": "PHAxis-release-safe-evidence-projection-1.0",
        "evidence_role": "h11_raw_median_contract_amendment_current",
        "source_path": spec.source,
        "source_receipt_sha256": _file_hash(source),
        "rebased_project_absolute_paths": 0,
        "policy": "project_absolute_paths_to_project_relative_posix",
    }
    assert projection["source_receipt_sha256"] == (
        spec.expected_source_receipt_sha256
    )
    assert spec.expected_source_receipt_sha256 == (
        source_release_module.H11_RAW_MEDIAN_AMENDMENT_AUTHORITY_SHA256
    )
    assert projected["amendment_audit_identity_sha256"] == (
        spec.expected_payload_identity_sha256
    )
    assert spec.expected_payload_identity_sha256 == (
        source_release_module.H11_RAW_MEDIAN_AMENDMENT_IDENTITY_SHA256
    )
    assert source_release_module._h11_raw_median_amendment_receipt_ok(projected)
    assert not source_release_module._absolute_host_path_markers(
        json.dumps(projected)
    )


def test_release_safe_h11_projection_rejects_resealed_authority_source(
    tmp_path: Path,
) -> None:
    spec = source_release_module.PROJECTED_EVIDENCE_SPECS[1]
    project = tmp_path / "project"
    source = project / spec.source
    receipt = _real_h11_amendment_receipt()
    record = receipt["tables"]["primary_clean_exploratory_factorial_tests.csv"][
        "H11"
    ]
    values = record["effects"]["construct_OE_minus_EV"]
    values["raw_effect_estimate"] += 1.0
    values["standardized_effect"] = (
        values["raw_effect_estimate"] / record["sample_standard_deviation"]
    )
    _reseal_h11_receipt(receipt)
    _write_json(source, receipt)
    with pytest.raises(
        SourceReleaseError,
        match="immutable standalone release anchor",
    ):
        source_release_module._project_evidence_receipt(
            project_root=project,
            source=source,
            spec=spec,
        )


def test_h11_projection_rejects_semantically_identical_raw_json_reformat(
    tmp_path: Path,
) -> None:
    spec = source_release_module.PROJECTED_EVIDENCE_SPECS[1]
    project = tmp_path / "project"
    source = project / spec.source
    receipt = _real_h11_amendment_receipt()
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        json.dumps(
            receipt,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
        newline="\n",
    )
    assert json.loads(source.read_text(encoding="utf-8")) == receipt
    assert _file_hash(source) != spec.expected_source_receipt_sha256
    with pytest.raises(
        SourceReleaseError,
        match="immutable standalone release anchor",
    ):
        source_release_module._project_evidence_receipt(
            project_root=project,
            source=source,
            spec=spec,
        )


@pytest.mark.parametrize("mutation", ("source_sha", "extra_field"))
def test_h11_projected_payload_helpers_reject_projection_metadata_tampering(
    mutation: str,
) -> None:
    spec = source_release_module.PROJECTED_EVIDENCE_SPECS[1]
    projected = source_release_module._project_evidence_receipt(
        project_root=PROJECT_ROOT,
        source=PROJECT_ROOT / spec.source,
        spec=spec,
    )
    if mutation == "source_sha":
        projected["release_projection"]["source_receipt_sha256"] = "0" * 64
    else:
        projected["release_projection"]["analysis_contract_is_current"] = True
    assert not source_release_module._release_projection_metadata_ok(
        spec, projected
    )
    assert not source_release_module._h11_raw_median_amendment_receipt_ok(
        projected
    )
    assert not source_release_module._projected_evidence_payload_ok(
        spec, projected
    )


def test_release_amp_amendment_evidence_is_semantically_verified(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    training_target = release / "src/phaxis/hair_stageb/training.py"
    amendment_target = release / source_release_module.AMP_AMENDMENT_DESTINATION
    training_target.parent.mkdir(parents=True)
    amendment_target.parent.mkdir(parents=True)
    training_target.write_bytes(
        (PROJECT_ROOT / "src/phaxis/hair_stageb/training.py").read_bytes()
    )
    amendment_authority = PROJECT_ROOT / source_release_module.AMP_AMENDMENT_SOURCE
    if not amendment_authority.is_file():
        amendment_authority = (
            PROJECT_ROOT / source_release_module.AMP_AMENDMENT_DESTINATION
        )
    amendment_target.write_bytes(amendment_authority.read_bytes())
    manifest = {
        "files": [
            {
                "path": source_release_module.AMP_AMENDMENT_DESTINATION,
                "origin": f"project:{source_release_module.AMP_AMENDMENT_SOURCE}",
            }
        ]
    }
    failures: list[str] = []
    source_release_module._verify_amp_amendment_evidence(
        release,
        failures,
        manifest=manifest,
    )
    assert failures == []

    tampered = json.loads(amendment_target.read_text(encoding="utf-8"))
    tampered["amended_numeric_policy"]["forward_recomputed"] = True
    _write_json(amendment_target, tampered)
    failures = []
    source_release_module._verify_amp_amendment_evidence(
        release,
        failures,
        manifest=manifest,
    )
    assert "Stage-B AMP backward amendment semantics are invalid" in failures


def test_release_boundary_rejects_content_level_host_absolute_paths(
    tmp_path: Path,
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    (release / "leak.txt").write_text(
        "workspace=" + "C:" + "\\Users\\person\\project",
        encoding="utf-8",
    )
    failures: list[str] = []
    source_release_module._verify_boundary(release, failures, formal=False)
    assert any("host-absolute path content (Windows drive)" in row for row in failures)


def test_formal_gate_requires_cross_bound_exact283_and_train399_receipts(
    tmp_path: Path,
) -> None:
    fixture = _formal_gate_fixture(tmp_path)
    report = inspect_formal_release_gate(
        project_root=fixture["project"],
        root_provider_receipt=fixture["root"],
        train399_candidate_manifest=fixture["candidate"],
        train399_selection_receipt=fixture["selection"],
        train399_evaluation_receipt=fixture["evaluation"],
        final_fusion_summary=fixture["fusion"],
        final_traits_summary=fixture["traits"],
        release_human_metadata=fixture["metadata"],
    )
    assert report["status"] == "passed"
    assert report["formal_release_allowed"] is True
    assert all(record["passed"] for record in report["checks"])

    evaluation = json.loads(fixture["evaluation"].read_text(encoding="utf-8"))
    evaluation["inputs_sha256"]["candidate_manifest"] = "0" * 64
    _write_json(fixture["evaluation"], evaluation)
    tampered = inspect_formal_release_gate(
        project_root=fixture["project"],
        root_provider_receipt=fixture["root"],
        train399_candidate_manifest=fixture["candidate"],
        train399_selection_receipt=fixture["selection"],
        train399_evaluation_receipt=fixture["evaluation"],
        final_fusion_summary=fixture["fusion"],
        final_traits_summary=fixture["traits"],
        release_human_metadata=fixture["metadata"],
    )
    assert tampered["formal_release_allowed"] is False
    assert next(
        row for row in tampered["checks"] if row["code"] == "train399_receipt_file_binding"
    )["passed"] is False


def test_formal_gate_rejects_legacy_base_only_selection_receipt(
    tmp_path: Path,
) -> None:
    """A sealed 1.2/base-only receipt is historical, never formal authority."""

    fixture = _formal_gate_fixture(tmp_path)
    selection = json.loads(fixture["selection"].read_text(encoding="utf-8"))
    selection.pop("selection_receipt_identity_sha256")
    selection["schema_version"] = (
        "PHAxis-StageB-train399-QCdev44-selection-receipt-1.2"
    )
    for field in (
        "straight_base_to_tip_presence_proxy_evaluated_during_selection",
        "distal_endpoint_error_used_as_selection_gate",
        "complete_line_overlap_used_as_selection_gate",
        "length_error_used_as_selection_gate",
        "manual_hair_width_assumed",
        "primary_matcher_contract",
        "primary_matcher_contract_sha256",
        "selection_contract",
        "threshold_metrics",
    ):
        selection.pop(field, None)
    selection["whole_line_and_tip_geometry_evaluated_during_selection"] = False
    selection["selection_receipt_identity_sha256"] = _canonical_hash(selection)
    _write_json(fixture["selection"], selection)

    report = inspect_formal_release_gate(
        project_root=fixture["project"],
        root_provider_receipt=fixture["root"],
        train399_candidate_manifest=fixture["candidate"],
        train399_selection_receipt=fixture["selection"],
        train399_evaluation_receipt=fixture["evaluation"],
        final_fusion_summary=fixture["fusion"],
        final_traits_summary=fixture["traits"],
        release_human_metadata=fixture["metadata"],
    )
    by_code = {row["code"]: row["passed"] for row in report["checks"]}
    assert by_code["train399_selection_schema_status"] is False
    assert by_code["train399_selection_primary_matcher"] is False
    assert by_code["train399_selection_protocol"] is False
    assert by_code["train399_selection_metrics"] is False
    assert report["formal_release_allowed"] is False


def test_formal_gate_requires_sealed_release_human_metadata(tmp_path: Path) -> None:
    fixture = _formal_gate_fixture(tmp_path)
    missing = inspect_formal_release_gate(
        project_root=fixture["project"],
        root_provider_receipt=fixture["root"],
        train399_candidate_manifest=fixture["candidate"],
        train399_selection_receipt=fixture["selection"],
        train399_evaluation_receipt=fixture["evaluation"],
        final_fusion_summary=fixture["fusion"],
        final_traits_summary=fixture["traits"],
    )
    assert missing["formal_release_allowed"] is False
    assert next(
        row
        for row in missing["checks"]
        if row["code"] == "release_human_metadata_present"
    )["passed"] is False

    metadata = json.loads(fixture["metadata"].read_text(encoding="utf-8"))
    metadata.pop("metadata_identity_sha256")
    metadata["rights"]["source_release_authorized"] = False
    metadata["metadata_identity_sha256"] = _canonical_hash(metadata)
    _write_json(fixture["metadata"], metadata)
    unauthorized = inspect_formal_release_gate(
        project_root=fixture["project"],
        root_provider_receipt=fixture["root"],
        train399_candidate_manifest=fixture["candidate"],
        train399_selection_receipt=fixture["selection"],
        train399_evaluation_receipt=fixture["evaluation"],
        final_fusion_summary=fixture["fusion"],
        final_traits_summary=fixture["traits"],
        release_human_metadata=fixture["metadata"],
    )
    assert unauthorized["formal_release_allowed"] is False
    assert next(
        row
        for row in unauthorized["checks"]
        if row["code"] == "release_human_metadata_rights"
    )["passed"] is False


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("author_orcid", "https://orcid.org/0000-0002-1825-0098"),
        ("author_orcid", "REQUIRED_AUTHOR_ORCID_URL"),
        ("author_given_names", "REQUIRED_AUTHOR_GIVEN_NAMES"),
        ("release_doi", "REQUIRED_RELEASE_DOI"),
        ("release_date", "2026-02-30"),
        ("unexpected_commit_field", "not-a-sealed-git-commit"),
    ),
)
def test_formal_gate_fails_closed_on_unverified_citation_authority(
    tmp_path: Path,
    field: str,
    invalid_value: str,
) -> None:
    fixture = _formal_gate_fixture(tmp_path)
    metadata = json.loads(fixture["metadata"].read_text(encoding="utf-8"))
    metadata.pop("metadata_identity_sha256")
    if field == "author_orcid":
        metadata["authors"][0]["orcid"] = invalid_value
    elif field == "author_given_names":
        metadata["authors"][0]["given_names"] = invalid_value
    else:
        metadata["release_coordinates"][field] = invalid_value
    metadata["metadata_identity_sha256"] = _canonical_hash(metadata)
    _write_json(fixture["metadata"], metadata)

    report = inspect_formal_release_gate(
        project_root=fixture["project"],
        root_provider_receipt=fixture["root"],
        train399_candidate_manifest=fixture["candidate"],
        train399_selection_receipt=fixture["selection"],
        train399_evaluation_receipt=fixture["evaluation"],
        final_fusion_summary=fixture["fusion"],
        final_traits_summary=fixture["traits"],
        release_human_metadata=fixture["metadata"],
    )
    assert report["formal_release_allowed"] is False
    assert next(
        row
        for row in report["checks"]
        if row["code"] == "release_human_metadata_public_coordinates"
    )["passed"] is False


def test_formal_gate_accepts_explicit_null_orcid(tmp_path: Path) -> None:
    fixture = _formal_gate_fixture(tmp_path)
    metadata = json.loads(fixture["metadata"].read_text(encoding="utf-8"))
    metadata.pop("metadata_identity_sha256")
    metadata["authors"][0]["orcid"] = None
    metadata["metadata_identity_sha256"] = _canonical_hash(metadata)
    _write_json(fixture["metadata"], metadata)
    report = inspect_formal_release_gate(
        project_root=fixture["project"],
        root_provider_receipt=fixture["root"],
        train399_candidate_manifest=fixture["candidate"],
        train399_selection_receipt=fixture["selection"],
        train399_evaluation_receipt=fixture["evaluation"],
        final_fusion_summary=fixture["fusion"],
        final_traits_summary=fixture["traits"],
        release_human_metadata=fixture["metadata"],
    )
    assert next(
        row
        for row in report["checks"]
        if row["code"] == "release_human_metadata_public_coordinates"
    )["passed"] is True
    assert report["formal_release_allowed"] is True
    zenodo = json.loads(source_release_module._zenodo_metadata(metadata))
    assert "orcid" not in zenodo["creators"][0]


def test_formal_wheel_metadata_projects_author_and_project_urls(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    metadata_path = _release_human_metadata(
        package,
        tmp_path / "release-human-metadata.json",
    )
    metadata_authority = json.loads(metadata_path.read_text(encoding="utf-8"))
    (package / "README.md").write_text("# PHAxis test package\n", encoding="utf-8")
    module = package / "src/phaxis/__init__.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text('__version__ = "1.0.0"\n', encoding="utf-8")
    shutil.copytree(
        PROJECT_ROOT / "src/phaxis/_vendor",
        package / "src/phaxis/_vendor",
    )
    (package / "pyproject.toml").write_text(
        source_release_module._pyproject(
            formal=True,
            release_metadata=metadata_authority,
        ),
        encoding="utf-8",
    )
    dist = tmp_path / "dist"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist),
        ],
        cwd=package,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    wheel = next(dist.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        wheel_members = set(archive.namelist())
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        wheel_metadata = archive.read(metadata_name).decode("utf-8")
    assert "Author-email:" in wheel_metadata
    assert "PHAxis Test Author <author@example.org>" in wheel_metadata
    assert "Maintainer-email:" in wheel_metadata
    assert "Project-URL: Repository, https://github.com/example/phaxis" in wheel_metadata
    assert "Project-URL: Homepage, https://example.org/phaxis" in wheel_metadata
    assert "Requires-Dist: packaging<26,>=24" in wheel_metadata
    assert "Requires-Dist: tomli" not in wheel_metadata
    assert "phaxis/_vendor/tomli/_parser.py" in wheel_members
    assert "phaxis/_vendor/tomli/LICENSE.txt" in wheel_members


def test_formal_gate_rejects_implicit_external_evaluator_source(tmp_path: Path) -> None:
    fixture = _formal_gate_fixture(tmp_path)
    evaluator = (
        fixture["project"]
        / "scripts/phaxis/evaluate_stageb_train399_qcdev44.py"
    )
    evaluator.write_text(
        "# production anti-fixture\nfrom rhaxiscc.evaluate import evaluate_image\n",
        encoding="utf-8",
    )
    report = inspect_formal_release_gate(
        project_root=fixture["project"],
        root_provider_receipt=fixture["root"],
        train399_candidate_manifest=fixture["candidate"],
        train399_selection_receipt=fixture["selection"],
        train399_evaluation_receipt=fixture["evaluation"],
        final_fusion_summary=fixture["fusion"],
        final_traits_summary=fixture["traits"],
        release_human_metadata=fixture["metadata"],
    )
    check = next(
        row
        for row in report["checks"]
        if row["code"] == "train399_evaluator_self_contained"
    )
    assert check["passed"] is False
    assert report["formal_release_allowed"] is False


def test_formal_gate_rejects_status_only_or_unapplied_model_contract(
    tmp_path: Path,
) -> None:
    fixture = _formal_gate_fixture(tmp_path)
    model_path = fixture["project"] / "configs/phaxis/v1_0/model_contract.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model.pop("model_contract_identity_sha256")
    model["promotion"]["status"] = "validated_proposal_not_applied"
    model["promotion"]["official_apply_performed"] = False
    model["model_contract_identity_sha256"] = _canonical_hash(model)
    _write_json(model_path, model)
    report = inspect_formal_release_gate(
        project_root=fixture["project"],
        root_provider_receipt=fixture["root"],
        train399_candidate_manifest=fixture["candidate"],
        train399_selection_receipt=fixture["selection"],
        train399_evaluation_receipt=fixture["evaluation"],
        final_fusion_summary=fixture["fusion"],
        final_traits_summary=fixture["traits"],
        release_human_metadata=fixture["metadata"],
    )
    promotion = next(
        row
        for row in report["checks"]
        if row["code"] == "model_contract_applied_promotion_authority"
    )
    assert promotion["passed"] is False
    assert report["formal_release_allowed"] is False


def test_formal_gate_rejects_final_public_identity_mismatch(tmp_path: Path) -> None:
    fixture = _formal_gate_fixture(tmp_path)
    fusion = json.loads(fixture["fusion"].read_text(encoding="utf-8"))
    fusion.pop("summary_identity_sha256")
    fusion["model_bundle_id"] = "PHAXIS-V1.0.0-STRICT-TRAIN399-DIFFERENT"
    fusion["summary_identity_sha256"] = _canonical_hash(fusion)
    _write_json(fixture["fusion"], fusion)

    model_path = fixture["project"] / "configs/phaxis/v1_0/model_contract.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model.pop("model_contract_identity_sha256")
    model["promotion"]["final_receipt_source_sha256"]["fusion"] = _file_hash(
        fixture["fusion"]
    )
    model["promotion"]["final_receipt_identity_sha256"]["fusion"] = fusion[
        "summary_identity_sha256"
    ]
    model["model_contract_identity_sha256"] = _canonical_hash(model)
    _write_json(model_path, model)

    report = inspect_formal_release_gate(
        project_root=fixture["project"],
        root_provider_receipt=fixture["root"],
        train399_candidate_manifest=fixture["candidate"],
        train399_selection_receipt=fixture["selection"],
        train399_evaluation_receipt=fixture["evaluation"],
        final_fusion_summary=fixture["fusion"],
        final_traits_summary=fixture["traits"],
        release_human_metadata=fixture["metadata"],
    )
    check = next(
        row
        for row in report["checks"]
        if row["code"] == "final_fusion_traits_public_identity"
    )
    assert check["passed"] is False
    assert report["formal_release_allowed"] is False


def test_generated_community_files_are_phaxis_only_and_supply_chain_pinned() -> None:
    files = source_release_module._community_files(
        formal=False,
        release_metadata=None,
    )
    assert set(files) == {
        ".gitattributes",
        ".gitignore",
        "CITATION.cff",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "SUPPORT.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/pull_request_template.md",
        ".github/workflows/ci.yml",
        ".github/workflows/release.yml",
    }
    combined = "\n".join(files.values())
    assert "PHAxis 1.0.0" in combined
    assert "RHAxis version" not in combined
    assert "RHAxiscc" not in combined
    assert "rhaxis_nextgen" not in combined
    assert "v2.0" not in combined.casefold()
    assert "sparse_instance" not in combined
    assert "dense_aggregate" not in combined
    assert "blind/final-validation" in combined
    workflow = files[".github/workflows/ci.yml"]
    assert (
        "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683"
        in workflow
    )
    assert (
        "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065"
        in workflow
    )
    assert "actions/checkout@v" not in workflow
    assert "actions/setup-python@v" not in workflow
    assert 'python-version: ["3.10", "3.11", "3.12"]' in workflow
    assert "os: [ubuntu-latest, windows-latest]" in workflow
    assert "runs-on: ${{ matrix.os }}" in workflow
    assert 'python -B -m pip install ".[test]"' in workflow
    assert "python -B -m pytest tests/phaxis -q" in workflow
    assert "python -B -m build" in workflow
    assert "python -B -m twine check dist/*" in workflow
    assert "-m pip install dist/*.whl" in workflow
    assert "-m pip check" in workflow
    assert "-m phaxis --help" in workflow
    assert "python -B scripts/phaxis/verify_source_release.py ." in workflow
    release_workflow = files[".github/workflows/release.yml"]
    for action, reference in source_release_module.GITHUB_ACTION_PINS.items():
        assert f"{action}@{reference}" in release_workflow
    assert 'PHAXIS_RELEASE_ENABLED: "false"' in release_workflow
    assert "refs/tags/v1.0.0" in release_workflow
    assert "distribution_receipt.json" in release_workflow
    assert "clean_install_receipt.json" in release_workflow
    assert "release_finalization.json" in release_workflow
    assert "environment:\n      name: pypi" in release_workflow
    assert "id-token: write" in release_workflow
    assert "attestations: true" in release_workflow
    assert "skip-existing: false" in release_workflow
    assert "gh release edit v1.0.0" in release_workflow
    assert not re.search(r"(?m)^\s*-?\s*uses:\s*[^@\s]+@v", release_workflow)
    readme = source_release_module._readme(
        formal=False,
        release_metadata=None,
    )
    compact_readme = " ".join(readme.split())
    assert "exact source verifier is for the authored source tree" in compact_readme
    assert "not an unpacked sdist" in compact_readme
    assert "Distribution receipts separately bind wheel/sdist archive hashes" in (
        compact_readme
    )
    assert "author-verified metadata" in compact_readme
    assert "Stage-B" not in readme
    assert "Stage B" not in readme
    assert "* text=auto eol=lf" in files[".gitattributes"]
    assert "__pycache__/" in files[".gitignore"]
    manifest = source_release_module._manifest_in(formal=False)
    for relative in (
        ".gitattributes",
        ".gitignore",
        "CITATION.cff",
        "CHANGELOG.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "MODEL_CARD.md",
        "DATA_CARD.md",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "THIRD_PARTY_LICENSES.json",
        "SBOM.cdx.json",
        "SECURITY.md",
        "SUPPORT.md",
    ):
        assert f"include {relative}" in manifest
    assert "recursive-include .github *.md *.yml *.yaml" in manifest
    assert "recursive-include configs/phaxis *.json *.txt" in manifest
    assert "recursive-include tests/phaxis/fixtures *.json" in manifest
    assert "include docs/phaxis/USER_GUIDE.md" in manifest
    assert "include docs/phaxis/TRAIT_CONTRACT_CN.md" in manifest
    assert f"include {source_release_module.ZENODO_METADATA_NAME}" not in manifest
    assert f"include {source_release_module.ZENODO_METADATA_NAME}" in (
        source_release_module._manifest_in(formal=True, release_metadata=True)
    )


def test_formal_public_metadata_renders_real_support_and_model_asset_coordinates() -> None:
    release_url = "https://github.com/example/phaxis/releases/tag/v1.0.0"
    issues_url = "https://github.com/example/phaxis/issues"
    docs_url = "https://example.org/phaxis/docs"
    metadata = {
        "authors": [
            {
                "display_name": "PHAxis Test Author",
                "given_names": "PHAxis Test",
                "family_names": "Author",
                "email": "author@example.org",
                "affiliation": "Example Plant Science Institute",
                "orcid": "https://orcid.org/0000-0002-1825-0097",
            }
        ],
        "project_urls": {
            "Homepage": "https://example.org/phaxis",
            "Repository": "https://github.com/example/phaxis",
            "Issues": issues_url,
            "Documentation": docs_url,
        },
        "release_coordinates": {
            "github_repository_url": "https://github.com/example/phaxis",
            "github_release_tag": "v1.0.0",
            "github_release_url": release_url,
            "pypi_project": "phaxis",
            "pypi_version": "1.0.0",
            "pypi_project_url": "https://pypi.org/project/phaxis/1.0.0/",
            "release_date": "2026-08-29",
            "release_doi": "10.5281/zenodo.1234567",
        },
    }
    files = source_release_module._community_files(
        formal=True,
        release_metadata=metadata,
    )
    assert "This release establishes" in files["CHANGELOG.md"]
    assert "This released establishes" not in files["CHANGELOG.md"]
    assert issues_url in files["SUPPORT.md"]
    assert docs_url in files["SUPPORT.md"]
    readme = source_release_module._readme(
        formal=True,
        release_metadata=metadata,
    )
    assert 'python -m pip install "phaxis[deployment]==1.0.0"' in readme
    assert release_url in readme
    assert docs_url in readme
    assert "authorized asset manifest" in readme
    citation = files["CITATION.cff"]
    assert (
        f'title: "{source_release_module.SOFTWARE_CITATION_TITLE}"' in citation
    )
    assert 'affiliation: "Example Plant Science Institute"' in citation
    assert 'given-names: "PHAxis Test"' in citation
    assert 'family-names: "Author"' in citation
    assert 'orcid: "https://orcid.org/0000-0002-1825-0097"' in citation
    assert 'date-released: "2026-08-29"' in citation
    assert 'doi: "10.5281/zenodo.1234567"' in citation
    zenodo = json.loads(files[source_release_module.ZENODO_METADATA_NAME])
    assert zenodo["version"] == "1.0.0"
    assert zenodo["doi"] == "10.5281/zenodo.1234567"
    assert zenodo["publication_date"] == "2026-08-29"
    assert zenodo["creators"] == [
        {
            "name": "Author, PHAxis Test",
            "affiliation": "Example Plant Science Institute",
            "orcid": "0000-0002-1825-0097",
        }
    ]
    assert "conceptdoi" not in zenodo
    assert "concept DOI" in zenodo["notes"]
    assert {row["identifier"] for row in zenodo["related_identifiers"]} == {
        release_url,
        "https://pypi.org/project/phaxis/1.0.0/",
    }
    assert all(
        "/commit/" not in row["identifier"]
        for row in zenodo["related_identifiers"]
    )
    formal_release_workflow = files[".github/workflows/release.yml"]
    assert 'PHAXIS_RELEASE_ENABLED: "true"' in formal_release_workflow
    assert 'PHAXIS_EXPECTED_REPOSITORY: "example/phaxis"' in formal_release_workflow
    assert release_url in formal_release_workflow
    assert "10.5281/zenodo.1234567" in formal_release_workflow
    assert "PHAXIS_EXPECTED_GIT_COMMIT" not in formal_release_workflow
    assert '"git_commit": os.environ["GITHUB_SHA"]' in formal_release_workflow
    assert 'distribution["wheel_archive_audit"]["metadata_license_files"]' in (
        formal_release_workflow
    )
    assert 'distribution["wheel_archive_audit"]["license_file_hashes_verified"]' in (
        formal_release_workflow
    )
    assert 'clean["formal_wheel"]["license_file_hashes_verified"]' in (
        formal_release_workflow
    )
    assert 'for command in distribution["commands"]:' in formal_release_workflow
    assert (
        'registry_reference = finalization["release_authority_registry_path"]'
        in formal_release_workflow
    )


def test_python310_uses_one_manifested_vendored_tomli_backend_without_bootstrap() -> None:
    source_common = (SCRIPT_ROOT / "source_release_common.py").read_text(
        encoding="utf-8"
    )
    manuscript_values = (
        PROJECT_ROOT / "src/phaxis/manuscript_values.py"
    ).read_text(encoding="utf-8")
    compat = (PROJECT_ROOT / "src/phaxis/_toml_compat.py").read_text(
        encoding="utf-8"
    )
    assert "from phaxis import _toml_compat as tomllib" in source_common
    assert "from . import _toml_compat as tomllib" in manuscript_values
    assert "from ._vendor import tomli as _backend" in compat
    assert "\n        import tomli as _backend" not in compat

    project = tomllib.loads(source_release_module._pyproject(formal=False))["project"]
    assert all(
        Requirement(requirement).name != "tomli"
        for requirement in project["dependencies"]
    )

    vendored = source_release_module._vendored_tomli_inventory()
    assert vendored["version"] == "2.4.0"
    assert vendored["purl"] == "pkg:pypi/tomli@2.4.0"
    assert vendored["license_expression"] == "MIT"
    assert vendored["source_bytes_unmodified"] is True
    for record in vendored["source_files"]:
        path = PROJECT_ROOT / record["path"]
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]

    workflow = source_release_module._community_files(
        formal=False,
        release_metadata=None,
    )[".github/workflows/ci.yml"]
    verifier = "python -B scripts/phaxis/verify_source_release.py ."
    no_site = "python -B -S scripts/phaxis/verify_source_release.py ."
    isolated_no_site = "python -B -I -S scripts/phaxis/verify_source_release.py ."
    assert 'pip install "tomli' not in workflow
    assert workflow.index(verifier) < workflow.index(no_site)
    assert workflow.index(no_site) < workflow.index(isolated_no_site)


@pytest.mark.parametrize(
    "document",
    [
        "[project]\nname = \"phaxis\"\ndeps = [\"a\", \"b\"]\n",
        "owner = {name = \"PHAxis\", active = true}\n",
        source_release_module._pyproject(formal=False),
    ],
)
def test_vendored_tomli_matches_stdlib_tomllib_for_release_documents(
    document: str,
) -> None:
    if sys.version_info < (3, 11):
        pytest.skip("stdlib tomllib parity requires Python 3.11+")
    import tomllib as stdlib_tomllib
    from phaxis._vendor import tomli as vendored_tomli

    assert vendored_tomli.loads(document) == stdlib_tomllib.loads(document)


@pytest.mark.parametrize(
    "document",
    [
        "duplicate = 1\nduplicate = 2\n",
        "[project]\nname = \"unterminated\n",
        "[project]\nvalue = {a = 1, a = 2}\n",
    ],
)
def test_vendored_tomli_and_stdlib_both_reject_malformed_release_toml(
    document: str,
) -> None:
    if sys.version_info < (3, 11):
        pytest.skip("stdlib tomllib parity requires Python 3.11+")
    import tomllib as stdlib_tomllib
    from phaxis._vendor import tomli as vendored_tomli

    with pytest.raises(stdlib_tomllib.TOMLDecodeError):
        stdlib_tomllib.loads(document)
    with pytest.raises(vendored_tomli.TOMLDecodeError):
        vendored_tomli.loads(document)


def test_python310_source_verifier_passes_no_site_and_rejects_resealed_vendor_tamper(
    tmp_path: Path,
) -> None:
    if sys.version_info[:2] != (3, 10):
        pytest.skip("real no-site E2E is executed by the Python 3.10 CI matrix")

    if (PROJECT_ROOT / "SOURCE_MANIFEST.json").is_file():
        release_root = PROJECT_ROOT
    elif FULL_PROJECT_PARITY_EVIDENCE.is_file():
        release_root = tmp_path / "blocked-source-release"
        build_source_release(
            project_root=PROJECT_ROOT,
            output=release_root,
            allow_blocked_development_staging=True,
        )
    else:
        pytest.skip("neither a source release nor the private build authority is present")

    verifier_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "-1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for mode in (("-B", "-S"), ("-B", "-I", "-S")):
        completed = subprocess.run(
            [
                sys.executable,
                *mode,
                "scripts/phaxis/verify_source_release.py",
                ".",
            ],
            cwd=release_root,
            env=verifier_env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout

    backend_probe = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            "-S",
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(release_root / 'src')!r}); "
                "from phaxis import _toml_compat as t; "
                "assert t.BACKEND_NAME == 'phaxis._vendor.tomli'; "
                "assert t.loads('[x]\\na=1')['x']['a'] == 1"
            ),
        ],
        cwd=release_root,
        env=verifier_env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert backend_probe.returncode == 0, backend_probe.stderr + backend_probe.stdout

    tampered = tmp_path / "resealed-vendor-tamper"
    shutil.copytree(release_root, tampered, ignore=shutil.ignore_patterns(".git"))
    vendor_type_source = tampered / "src/phaxis/_vendor/tomli/_types.py"
    vendor_type_source.write_text(
        vendor_type_source.read_text(encoding="utf-8") + "# unauthorized change\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = tampered / "SOURCE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(
        row
        for row in manifest["files"]
        if row["path"] == "src/phaxis/_vendor/tomli/_types.py"
    )
    record["bytes"] = vendor_type_source.stat().st_size
    record["sha256"] = hashlib.sha256(vendor_type_source.read_bytes()).hexdigest()
    manifest["tree_identity_sha256"] = _canonical_hash(manifest["files"])
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    rejected = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            "-S",
            "scripts/phaxis/verify_source_release.py",
            ".",
        ],
        cwd=tampered,
        env=verifier_env,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert rejected.returncode == 2
    assert "vendored Tomli 2.4.0 file identity mismatch" in (
        rejected.stderr + rejected.stdout
    )


def test_source_verifier_accepts_only_exact_root_git_control_plane(
    tmp_path: Path,
) -> None:
    if not FULL_PROJECT_PARITY_EVIDENCE.is_file():
        pytest.skip("full project authority/evidence tree is not present")
    release = tmp_path / "release"
    build_source_release(
        project_root=PROJECT_ROOT,
        output=release,
        allow_blocked_development_staging=True,
    )
    manifest_before_git = (release / "SOURCE_MANIFEST.json").read_bytes()

    initialized = subprocess.run(
        ["git", "init", "--quiet"],
        cwd=release,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert initialized.returncode == 0, initialized.stderr
    assert (release / ".git/HEAD").is_file()
    assert verify_source_release(release)["status"] == "passed"
    assert (release / "SOURCE_MANIFEST.json").read_bytes() == manifest_before_git

    worktree_release = tmp_path / "worktree-release"
    shutil.copytree(release, worktree_release, ignore=shutil.ignore_patterns(".git"))
    (worktree_release / ".git").write_text(
        "gitdir: ../git-control/worktrees/phaxis\n", encoding="utf-8"
    )
    assert verify_source_release(worktree_release)["status"] == "passed"

    nested_release = tmp_path / "nested-release"
    shutil.copytree(release, nested_release, ignore=shutil.ignore_patterns(".git"))
    nested_git = nested_release / "docs/.git/config"
    nested_git.parent.mkdir()
    nested_git.write_text("[core]\n", encoding="utf-8")
    with pytest.raises(
        SourceReleaseError,
        match=r"unmanifested file: docs/\.git/config",
    ):
        verify_source_release(nested_release)


def test_release_workflow_binds_real_tag_checkout_without_predeclared_commit(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "tagged-repository"
    repository.mkdir()
    (repository / "README.md").write_text("PHAxis 1.0.0\n", encoding="utf-8")
    for command in (
        ["git", "init", "--quiet", "--initial-branch=main"],
        ["git", "config", "user.email", "phaxis-test@example.org"],
        ["git", "config", "user.name", "PHAxis release test"],
        ["git", "add", "README.md"],
        ["git", "commit", "--quiet", "-m", "Release PHAxis 1.0.0"],
        ["git", "tag", "-a", "v1.0.0", "-m", "PHAxis 1.0.0"],
    ):
        completed = subprocess.run(
            command,
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert completed.returncode == 0, completed.stderr
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    tagged = subprocess.run(
        ["git", "rev-list", "-n", "1", "v1.0.0"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    assert tagged == head

    workflow = source_release_module._release_workflow(
        formal=True,
        release_metadata={
            "release_coordinates": {
                "github_repository_url": "https://github.com/example/phaxis",
                "github_release_url": (
                    "https://github.com/example/phaxis/releases/tag/v1.0.0"
                ),
                "release_doi": "10.5281/zenodo.1234567",
            }
        },
    )
    assert 'test "$(git rev-list -n 1 v1.0.0)" = "$GITHUB_SHA"' in workflow
    assert '"git_commit": os.environ["GITHUB_SHA"]' in workflow
    assert "PHAXIS_EXPECTED_GIT_COMMIT" not in workflow


def test_real_project_blocked_build_is_atomic_deterministic_and_current(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not FULL_PROJECT_PARITY_EVIDENCE.is_file():
        pytest.skip("full project authority/evidence tree is not present")
    first = tmp_path / "release-a"
    second = tmp_path / "release-b"
    with pytest.raises(SourceReleaseError, match="formal release gate is blocked"):
        build_source_release(project_root=PROJECT_ROOT, output=first)
    assert not first.exists()

    first_manifest = build_source_release(
        project_root=PROJECT_ROOT,
        output=first,
        allow_blocked_development_staging=True,
    )
    second_manifest = build_source_release(
        project_root=PROJECT_ROOT,
        output=second,
        allow_blocked_development_staging=True,
    )
    assert first_manifest == second_manifest
    assert (first / "SOURCE_MANIFEST.json").read_bytes() == (
        second / "SOURCE_MANIFEST.json"
    ).read_bytes()
    assert "* text=auto eol=lf" in (first / ".gitattributes").read_text(
        encoding="utf-8"
    )
    assert "__pycache__/" in (first / ".gitignore").read_text(encoding="utf-8")
    assert (first / BLOCKED_RECEIPT_NAME).is_file()
    receipt = json.loads((first / BLOCKED_RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["formal_release_allowed"] is False
    assert "DO NOT PUBLISH" in receipt["warning"]
    assert (first / source_release_module.RELEASE_HUMAN_METADATA_TEMPLATE_NAME).is_file()
    assert not (first / source_release_module.RELEASE_HUMAN_METADATA_NAME).exists()
    template = json.loads(
        (
            first / source_release_module.RELEASE_HUMAN_METADATA_TEMPLATE_NAME
        ).read_text(encoding="utf-8")
    )
    assert template["status"] == "BLOCKED_TEMPLATE_NOT_AUTHORITY"
    assert template["rights"]["source_release_authorized"] is False
    assert template["schema_version"] == "PHAxis-release-human-metadata-1.3"
    assert template["authors"][0]["affiliation"].startswith("REQUIRED_")
    assert template["authors"][0]["orcid"] is None
    assert template["release_coordinates"]["release_doi"].startswith(
        "REQUIRED_"
    )
    assert template["release_coordinates"]["release_date"].startswith(
        "REQUIRED_"
    )
    assert "github_git_commit" not in template["release_coordinates"]
    assert next(
        row
        for row in receipt["checks"]
        if row["code"] == "train399_evaluator_self_contained"
    )["passed"] is True
    verified = verify_source_release(first, project_root=PROJECT_ROOT)
    assert verified["status"] == "passed"
    subprocess_verification = subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/phaxis/verify_source_release.py",
            ".",
        ],
        cwd=first,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert subprocess_verification.returncode == 0, subprocess_verification.stderr
    assert verified["phaxis_package_included"] is True
    assert verified["phaxis_cli_entry_point"] == "phaxis.cli:main"
    assert (first / "src/phaxis/evaluation_metrics.py").is_file()
    assert (first / "src/phaxis/workflow.py").is_file()
    assert (first / "tests/phaxis/test_workflow.py").is_file()
    supply_chain_paths = {
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "THIRD_PARTY_LICENSES.json",
        "SBOM.cdx.json",
    }
    assert all((first / relative).is_file() for relative in supply_chain_paths)
    source_manifest_paths = {
        row["path"] for row in first_manifest["files"]
    }
    assert supply_chain_paths.issubset(source_manifest_paths)
    sbom = json.loads((first / "SBOM.cdx.json").read_text(encoding="utf-8"))
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["specVersion"] == "1.6"
    assert sbom["metadata"]["component"]["name"] == "phaxis"
    assert sbom["metadata"]["component"]["version"] == "1.0.0"
    inventory = json.loads(
        (first / "THIRD_PARTY_LICENSES.json").read_text(encoding="utf-8")
    )
    unsigned_inventory = dict(inventory)
    inventory_identity = unsigned_inventory.pop("inventory_identity_sha256")
    assert inventory_identity == _canonical_hash(unsigned_inventory)
    assert inventory["dependency_count"] == len(
        source_release_module.THIRD_PARTY_DEPENDENCIES
    )
    dependency_by_name = {
        row["name"]: row for row in inventory["dependencies"]
    }
    assert dependency_by_name["packaging"]["requirement"] == "packaging>=24,<26"
    assert "tomli" not in dependency_by_name
    assert inventory["vendored_component_count"] == 1
    vendored_tomli = inventory["vendored_components"][0]
    assert vendored_tomli["name"] == "tomli"
    assert vendored_tomli["version"] == "2.4.0"
    assert vendored_tomli["relationship"] == "vendored_source_no_site_fallback"
    assert vendored_tomli["license_expression"] == "MIT"
    assert vendored_tomli["license_text_sha256"] == hashlib.sha256(
        (first / vendored_tomli["license_text_path"]).read_bytes()
    ).hexdigest()
    assert vendored_tomli["source_tree_identity_sha256"] == _canonical_hash(
        vendored_tomli["source_files"]
    )
    assert {
        row["path"] for row in vendored_tomli["source_files"]
    }.issubset(source_manifest_paths)
    assert dependency_by_name["Pillow"]["license_expression"] == "MIT-CMU"
    opencv_dependency = dependency_by_name["opencv-python-headless"]
    assert opencv_dependency["license_expression"] == (
        "LicenseRef-opencv-python-headless-wheel-multiple"
    )
    assert "LICENSE-3RD-PARTY.txt" in opencv_dependency["license_note"]
    assert "FFmpeg" in opencv_dependency["license_note"]
    sbom_opencv = next(
        component
        for component in sbom["components"]
        if component["name"] == "opencv-python-headless"
    )
    sbom_opencv_properties = {
        row["name"]: row["value"] for row in sbom_opencv["properties"]
    }
    assert sbom_opencv["licenses"] == [
        {"expression": "LicenseRef-opencv-python-headless-wheel-multiple"}
    ]
    assert "LICENSE-3RD-PARTY.txt" in sbom_opencv_properties[
        "phaxis:license-note"
    ]
    sbom_tomli = next(
        component
        for component in sbom["components"]
        if component.get("purl") == "pkg:pypi/tomli@2.4.0"
    )
    assert sbom_tomli["version"] == "2.4.0"
    assert sbom_tomli["licenses"] == [{"expression": "MIT"}]
    sbom_tomli_properties = {
        row["name"]: row["value"] for row in sbom_tomli["properties"]
    }
    assert sbom_tomli_properties[
        "phaxis:vendored-source-tree-identity-sha256"
    ] == vendored_tomli["source_tree_identity_sha256"]
    assert "RHPheno" not in (first / "NOTICE").read_text(encoding="utf-8")
    biological_metadata_guide = (
        first
        / "docs/phaxis/PHAXIS_BIOLOGICAL_ACQUISITION_METADATA_COMPLETION_CN_20260829.md"
    )
    assert biological_metadata_guide.read_bytes() == (
        PROJECT_ROOT
        / "docs/phaxis/PHAXIS_BIOLOGICAL_ACQUISITION_METADATA_COMPLETION_CN_20260829.md"
    ).read_bytes()
    assert biological_metadata_guide.read_text(encoding="utf-8").count(
        "| `FINAL_BIOLOGICAL_"
    ) == 15
    assert (first / "evidence/evaluation_metric_parity_audit.json").is_file()
    historical_evidence_path = (
        first / "evidence/biological_analysis_equivalence_audit.json"
    )
    h11_evidence_path = first / "evidence/h11_raw_median_amendment_audit.json"
    assert historical_evidence_path.is_file()
    assert h11_evidence_path.is_file()
    historical_evidence = json.loads(
        historical_evidence_path.read_text(encoding="utf-8")
    )
    h11_evidence = json.loads(h11_evidence_path.read_text(encoding="utf-8"))
    assert historical_evidence["release_projection"]["evidence_role"] == (
        "pre_amendment_biological_equivalence_historical_baseline"
    )
    assert h11_evidence["release_projection"]["evidence_role"] == (
        "h11_raw_median_contract_amendment_current"
    )
    assert h11_evidence["pre_amendment_baseline"]["authority_sha256"] == (
        historical_evidence["release_projection"]["source_receipt_sha256"]
    )
    projected_origins = {
        row["path"]: row["origin"]
        for row in first_manifest["files"]
        if row["path"]
        in {spec.destination for spec in source_release_module.PROJECTED_EVIDENCE_SPECS}
    }
    assert projected_origins == {
        spec.destination: f"derived-project:{spec.source}"
        for spec in source_release_module.PROJECTED_EVIDENCE_SPECS
    }
    amp_amendment_path = first / "evidence/stageb_amp_backward_retry_amendment.json"
    assert amp_amendment_path.is_file()
    amp_amendment = json.loads(amp_amendment_path.read_text(encoding="utf-8"))
    assert amp_amendment["schema_version"] == (
        "PHAxis-StageB-train399-AMP-backward-amendment-1.0"
    )
    assert amp_amendment["superseded_failed_attempt"]["blind_images_used"] == 0
    assert amp_amendment["amended_numeric_policy"] == {
        "contract_policy_string": "fail_closed_no_optimizer_step_skip",
        "interpretation": (
            "A scaled-backward overflow may be retried on the identical retained "
            "forward graph after GradScaler backoff; the batch and optimizer update "
            "are never skipped, and exhaustion remains a hard failure."
        ),
        "initial_scale": 1024.0,
        "backoff_factor": 0.5,
        "maximum_backward_retries_per_batch": 16,
        "same_forward_graph_replayed": True,
        "forward_recomputed": False,
        "batchnorm_buffers_updated_again": False,
        "rng_or_data_order_advanced": False,
        "optimizer_step_before_finite_unscaled_gradient": False,
        "optimizer_steps_skipped_due_nonfinite_gradients": 0,
        "failure_after_retry_exhaustion": True,
    }
    for relative in source_release_module._community_files(
        formal=False,
        release_metadata=None,
    ):
        assert (first / relative).is_file()
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
    assert "family-names: \"contributors\"" in (first / "CITATION.cff").read_text(
        encoding="utf-8"
    )
    readme = (first / "README.md").read_text(encoding="utf-8")
    compact_readme = " ".join(readme.split())
    assert "phaxis analyze --manifest workflow.json --output analysis-output" in readme
    assert "--execute --resume" in readme
    assert "PHAxis 1.0.0 is the sole public software and model-system version" in (
        compact_readme
    )
    assert "five-member root-hair identity/count expert" in compact_readme
    assert "not a segmented root-cap region" in readme
    assert "32 canonical image-derived descriptors" in readme
    assert "does not report 82 phenotypes" in readme
    assert "docs/phaxis/TRAIT_CONTRACT_CN.md" in readme
    assert "docs/phaxis/USER_GUIDE.md" in readme
    assert "not independent external accuracy" in " ".join(readme.split())
    assert "--model-contract official-contract.json" in readme
    assert "no authorized public model-asset download coordinate" in " ".join(
        readme.split()
    )
    assert "RHAxiscc" not in readme
    assert "rhaxis_nextgen" not in readme
    assert "Hybrid-Max" not in readme
    assert "Stage-B" not in readme
    assert "Stage B" not in readme
    assert "r16" not in readme.casefold()
    assert "r17" not in readme.casefold()
    assert "v2.0" not in readme.casefold()
    assert not (first / "scripts/phaxis/audit_rhaxiscc_metric_parity.py").exists()
    for internal_development_document in (
        "docs/phaxis/PHAXIS_PAPER_EVIDENCE_AND_ARCHITECTURE_20260828.md",
        "docs/phaxis/PHAXIS_STAGEB_TRAIN399_CANDIDATE_GATE_20260828.md",
    ):
        assert not (first / internal_development_document).exists()
    catalog_relative = "docs/phaxis/TRAIT_CONTRACT_CN.md"
    catalog_path = first / catalog_relative
    assert catalog_path.is_file()
    assert catalog_path.read_bytes() == (PROJECT_ROOT / catalog_relative).read_bytes()
    assert any(row["path"] == catalog_relative for row in first_manifest["files"])
    guide_relative = "docs/phaxis/USER_GUIDE.md"
    assert (first / guide_relative).is_file()
    assert (first / guide_relative).read_bytes() == (
        PROJECT_ROOT / guide_relative
    ).read_bytes()
    assert any(row["path"] == guide_relative for row in first_manifest["files"])
    catalog = catalog_path.read_text(encoding="utf-8")
    compact_catalog = " ".join(catalog.split())
    assert "five-member root-hair identity/count expert" in compact_catalog
    assert "32 canonical image-derived descriptors" in compact_catalog
    assert "does not report 82 phenotypes" in compact_catalog
    for forbidden in (
        "rhaxiscc",
        "rhaxis_nextgen",
        "hybrid-max",
        "stage-b",
        "stage b",
        "v2.0",
        "outputs/",
        "models/",
        "scripts/phaxis/",
    ):
        assert forbidden not in catalog.casefold()
    assert not (first / "src/rhaxis_nextgen").exists()
    assert not (first / "src/rhizoweave").exists()
    assert not (first / "scripts/analyze_six_condition_hybrid_max.py").exists()
    for uncompiled in source_release_module.UNCOMPILED_MANUSCRIPT_FILES:
        assert not (first / uncompiled).exists()
        assert all(row["path"] != uncompiled for row in first_manifest["files"])

    # Verification is observational under ordinary, isolated, and no-site
    # interpreters. Generated docs/CI still use explicit -B as defense in depth.
    verifier_env = dict(os.environ)
    verifier_env.pop("PYTHONDONTWRITEBYTECODE", None)
    manifest_before_probes = (first / "SOURCE_MANIFEST.json").read_bytes()
    verifier_modes = ((), ("-I",), ("-S",), ("-B", "-I", "-S"))
    for mode in verifier_modes:
        completed = subprocess.run(
            [
                sys.executable,
                *mode,
                "scripts/phaxis/verify_source_release.py",
                ".",
            ],
            cwd=first,
            env=verifier_env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        assert (first / "SOURCE_MANIFEST.json").read_bytes() == manifest_before_probes
        assert not list(first.rglob("__pycache__"))
        assert not [path for path in first.rglob("*") if path.suffix in {".pyc", ".pyo"}]

    # Even an attacker who keeps an interval/standardized-interval pair
    # internally coherent,
    # recomputes the amendment identity, and reseals the source-tree manifest
    # cannot replace the immutable H11 authority in standalone verification.
    tampered_h11 = tmp_path / "release-h11-resealed-tamper"
    shutil.copytree(first, tampered_h11)
    tampered_path = tampered_h11 / "evidence/h11_raw_median_amendment_audit.json"
    tampered_payload = json.loads(tampered_path.read_text(encoding="utf-8"))
    projection = tampered_payload.pop("release_projection")
    tampered_payload.pop("amendment_audit_identity_sha256")
    tampered_h11_record = tampered_payload["tables"][
        "primary_clean_exploratory_factorial_tests.csv"
    ]["H11"]
    tampered_effect = tampered_h11_record["effects"]["construct_OE_minus_EV"]
    tampered_effect["raw_effect_ci95_high"] += 1.0
    tampered_effect["standardized_ci95_high"] = (
        tampered_effect["raw_effect_ci95_high"]
        / tampered_h11_record["sample_standard_deviation"]
    )
    tampered_payload["amendment_audit_identity_sha256"] = _canonical_hash(
        tampered_payload
    )
    tampered_payload["release_projection"] = projection
    _write_json(tampered_path, tampered_payload)
    tampered_manifest_path = tampered_h11 / "SOURCE_MANIFEST.json"
    tampered_manifest = json.loads(
        tampered_manifest_path.read_text(encoding="utf-8")
    )
    tampered_record = next(
        row
        for row in tampered_manifest["files"]
        if row["path"] == "evidence/h11_raw_median_amendment_audit.json"
    )
    tampered_record["bytes"] = tampered_path.stat().st_size
    tampered_record["sha256"] = _file_hash(tampered_path)
    tampered_manifest["tree_identity_sha256"] = _canonical_hash(
        tampered_manifest["files"]
    )
    _write_json(tampered_manifest_path, tampered_manifest)
    with pytest.raises(
        SourceReleaseError,
        match="h11_raw_median_contract_amendment_current",
    ):
        verify_source_release(tampered_h11)

    # The immutable raw-authority SHA is separately enforced. Recomputing only
    # the source-tree manifest cannot legitimize forged projection metadata.
    tampered_source_sha = tmp_path / "release-h11-source-sha-tamper"
    shutil.copytree(first, tampered_source_sha)
    source_sha_path = (
        tampered_source_sha / "evidence/h11_raw_median_amendment_audit.json"
    )
    source_sha_payload = json.loads(source_sha_path.read_text(encoding="utf-8"))
    source_sha_payload["release_projection"]["source_receipt_sha256"] = "0" * 64
    _write_json(source_sha_path, source_sha_payload)
    source_sha_manifest_path = tampered_source_sha / "SOURCE_MANIFEST.json"
    source_sha_manifest = json.loads(
        source_sha_manifest_path.read_text(encoding="utf-8")
    )
    source_sha_record = next(
        row
        for row in source_sha_manifest["files"]
        if row["path"] == "evidence/h11_raw_median_amendment_audit.json"
    )
    source_sha_record["bytes"] = source_sha_path.stat().st_size
    source_sha_record["sha256"] = _file_hash(source_sha_path)
    source_sha_manifest["tree_identity_sha256"] = _canonical_hash(
        source_sha_manifest["files"]
    )
    _write_json(source_sha_manifest_path, source_sha_manifest)
    with pytest.raises(
        SourceReleaseError,
        match="h11_raw_median_contract_amendment_current",
    ):
        verify_source_release(tampered_source_sha)

    # The historical half of the two-evidence chain has no embedded identity,
    # so its release-safe payload identity is an explicit standalone anchor.
    tampered_historical = tmp_path / "release-historical-resealed-tamper"
    shutil.copytree(first, tampered_historical)
    historical_path = (
        tampered_historical
        / "evidence/biological_analysis_equivalence_audit.json"
    )
    historical_payload = json.loads(historical_path.read_text(encoding="utf-8"))
    historical_table = next(iter(historical_payload["tables"].values()))
    historical_table["baseline_sha256"] = "0" * 64
    historical_table["candidate_sha256"] = "0" * 64
    _write_json(historical_path, historical_payload)
    historical_manifest_path = tampered_historical / "SOURCE_MANIFEST.json"
    historical_manifest = json.loads(
        historical_manifest_path.read_text(encoding="utf-8")
    )
    historical_record = next(
        row
        for row in historical_manifest["files"]
        if row["path"] == "evidence/biological_analysis_equivalence_audit.json"
    )
    historical_record["bytes"] = historical_path.stat().st_size
    historical_record["sha256"] = _file_hash(historical_path)
    historical_manifest["tree_identity_sha256"] = _canonical_hash(
        historical_manifest["files"]
    )
    _write_json(historical_manifest_path, historical_manifest)
    with pytest.raises(
        SourceReleaseError,
        match="pre_amendment_biological_equivalence_historical_baseline",
    ):
        verify_source_release(tampered_historical)

    # Projection metadata is a closed six-field contract; an attacker cannot
    # append a plausible-looking current-workspace hash to relabel history.
    tampered_metadata = tmp_path / "release-h11-metadata-extension-tamper"
    shutil.copytree(first, tampered_metadata)
    metadata_path = (
        tampered_metadata / "evidence/h11_raw_median_amendment_audit.json"
    )
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_payload["release_projection"][
        "current_analysis_contract_sha256"
    ] = "f" * 64
    _write_json(metadata_path, metadata_payload)
    metadata_manifest_path = tampered_metadata / "SOURCE_MANIFEST.json"
    metadata_manifest = json.loads(
        metadata_manifest_path.read_text(encoding="utf-8")
    )
    metadata_record = next(
        row
        for row in metadata_manifest["files"]
        if row["path"] == "evidence/h11_raw_median_amendment_audit.json"
    )
    metadata_record["bytes"] = metadata_path.stat().st_size
    metadata_record["sha256"] = _file_hash(metadata_path)
    metadata_manifest["tree_identity_sha256"] = _canonical_hash(
        metadata_manifest["files"]
    )
    _write_json(metadata_manifest_path, metadata_manifest)
    with pytest.raises(
        SourceReleaseError,
        match="h11_raw_median_contract_amendment_current",
    ):
        verify_source_release(tampered_metadata)

    # The source-contained producers most likely to be used as release probes
    # must also be bytecode-free under isolated execution.
    source_probe_commands = (
        (
            sys.executable,
            "-B",
            "-I",
            "scripts/phaxis/check_post_training_release_topology.py",
            "--project-root",
            ".",
        ),
        (sys.executable, "-B", "-I", "scripts/phaxis/build_source_release.py", "--help"),
        (
            sys.executable,
            "-B",
            "-I",
            "scripts/phaxis/build_release_distributions.py",
            "--help",
        ),
        (
            sys.executable,
            "-B",
            "-I",
            "scripts/phaxis/build_clean_install_verification.py",
            "--help",
        ),
    )
    for command in source_probe_commands:
        completed = subprocess.run(
            command,
            cwd=first,
            env=verifier_env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        assert (first / "SOURCE_MANIFEST.json").read_bytes() == manifest_before_probes
    assert not list(first.rglob("__pycache__"))
    assert not [path for path in first.rglob("*") if path.suffix in {".pyc", ".pyo"}]
    assert verify_source_release(first)["status"] == "passed"

    # Exact closure rejects both generic extras and common generated-state
    # classes. Each negative probe runs in its own independent copied tree.
    extra_release = tmp_path / "release-extra"
    shutil.copytree(first, extra_release)
    (extra_release / "extra.txt").write_text("not manifested\n", encoding="utf-8")
    with pytest.raises(SourceReleaseError, match=r"unmanifested file: extra\.txt"):
        verify_source_release(extra_release)

    cache_release = tmp_path / "release-bytecode-cache"
    shutil.copytree(first, cache_release)
    bytecode = cache_release / "src/phaxis/__pycache__/injected.cpython-312.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"injected bytecode")
    with pytest.raises(SourceReleaseError, match=r"unmanifested file: src/phaxis/__pycache__"):
        verify_source_release(cache_release)

    pytest_cache_release = tmp_path / "release-pytest-cache"
    shutil.copytree(first, pytest_cache_release)
    pytest_cache = pytest_cache_release / ".pytest_cache/README.md"
    pytest_cache.parent.mkdir()
    pytest_cache.write_text("generated cache\n", encoding="utf-8")
    with pytest.raises(SourceReleaseError, match=r"unmanifested file: \.pytest_cache"):
        verify_source_release(pytest_cache_release)

    symlink_release = tmp_path / "release-symlink"
    shutil.copytree(first, symlink_release)
    symlink_path = symlink_release / "linked-readme.md"
    symlink_path.write_text("simulated link target\n", encoding="utf-8")
    original_is_symlink = Path.is_symlink
    with monkeypatch.context() as symlink_context:
        symlink_context.setattr(
            Path,
            "is_symlink",
            lambda path: path == symlink_path or original_is_symlink(path),
        )
        with pytest.raises(
            SourceReleaseError,
            match=r"source release may not contain symlinks: linked-readme\.md",
        ):
            verify_source_release(symlink_release)

    authoritative = source_release_module.collect_allowlisted_sources(PROJECT_ROOT)
    monkeypatch.setattr(
        source_release_module,
        "collect_allowlisted_sources",
        lambda _root: authoritative
        + [
            (
                PROJECT_ROOT / "src/phaxis/io.py",
                "src/phaxis/new_authority_module.py",
            )
        ],
    )
    with pytest.raises(
        SourceReleaseError,
        match="current allowlisted project source absent from release",
    ):
        verify_source_release(first, project_root=PROJECT_ROOT)


def test_nonempty_output_and_postbuild_tamper_fail_without_overwrite(
    tmp_path: Path,
) -> None:
    if not FULL_PROJECT_PARITY_EVIDENCE.is_file():
        pytest.skip("full project authority/evidence tree is not present")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    marker = occupied / "owned-by-user.txt"
    marker.write_text("preserve me", encoding="utf-8")
    with pytest.raises(SourceReleaseError, match="new or empty"):
        build_source_release(
            project_root=PROJECT_ROOT,
            output=occupied,
            allow_blocked_development_staging=True,
        )
    assert marker.read_text(encoding="utf-8") == "preserve me"

    release = tmp_path / "release"
    build_source_release(
        project_root=PROJECT_ROOT,
        output=release,
        allow_blocked_development_staging=True,
    )
    with (release / "src/phaxis/cli.py").open("a", encoding="utf-8") as handle:
        handle.write("\n# tamper\n")
    with pytest.raises(SourceReleaseError, match="manifest SHA-256 mismatch"):
        verify_source_release(release, project_root=PROJECT_ROOT)
