from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "phaxis"
sys.path.insert(0, str(SCRIPT_ROOT))

import build_manuscript_evidence_manifest as evidence_module  # noqa: E402
import source_release_common  # noqa: E402
from build_manuscript_evidence_manifest import (  # noqa: E402
    EvidenceManifestError,
    build_manuscript_evidence_manifest,
)
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
from phaxis.multitrait_atlas import (  # noqa: E402
    COHORTS as MULTITRAIT_COHORTS,
    CONDITION_ROW_UNIT,
    CONDITION_SUMMARY_STATUS,
    EFFECT_KEYS as MULTITRAIT_EFFECT_KEYS,
    GROUP_ORDER as MULTITRAIT_GROUP_ORDER,
    MEASUREMENT_FAMILY_ORDER,
    MEASUREMENT_FAMILY_TRAIT_IDS,
    NOT_ESTIMATED_REASON,
    PRIMARY_ENDPOINTS as MULTITRAIT_PRIMARY_ENDPOINTS,
    SCHEMA_VERSION as MULTITRAIT_SCHEMA_VERSION,
)
from phaxis.narrative_decision import (  # noqa: E402
    COHORT_ORDER as NARRATIVE_COHORT_ORDER,
    EFFECT_ORDER as NARRATIVE_EFFECT_ORDER,
    ENDPOINT_ORDER as NARRATIVE_ENDPOINT_ORDER,
    build_narrative_decision,
)
from phaxis.publication_titles import title_contract  # noqa: E402
from phaxis.supplementary_tables import (  # noqa: E402
    BUNDLE_RECEIPT as SUPPLEMENTARY_TABLE_BUNDLE_RECEIPT,
    BUNDLE_DIRECTORY as SUPPLEMENTARY_TABLE_BUNDLE_DIRECTORY,
    FINAL_STATUS as FINAL_SUPPLEMENTARY_TABLE_STATUS,
    TABLE_SPECS as SUPPLEMENTARY_TABLE_SPECS,
    materialize_supplementary_table_data_bundle,
)
from tests.phaxis.test_supplementary_table_data_bundle import (  # noqa: E402
    source_fixture as _supplementary_source_fixture,
)
from tests.phaxis.test_publication_figure_input_evidence import (  # noqa: E402
    _wt_secondary_fixture,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


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
    rows: list[dict] = []
    for threshold in PREREGISTERED_SCORE_THRESHOLDS:
        if threshold < 0.225:
            predicted, biological_tp = 2, 1
        elif threshold == 0.225:
            predicted, biological_tp = 1, 1
        else:
            predicted, biological_tp = 0, 0
        per_image = [
            {
                "task_id": f"qc-{index:02d}",
                "predicted": predicted,
                "ground_truth": 1,
                "biological_presence_true_positive_20um": biological_tp,
                "attachment_proxy_true_positive_20um": 0,
                "biological_presence_matched_pairs": (
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
                ),
                "count_error": predicted - 1,
            }
            for index in range(44)
        ]
        predicted_total = 44 * predicted
        rows.append(
            {
                "threshold": float(threshold),
                "tolerant_biological_presence_20um": _selection_prf(
                    44 * biological_tp, predicted_total, 44
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


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _multitrait_atlas_fixture(
    contract: dict, source_sha256: dict[str, str]
) -> dict:
    descriptors = []
    estimated = 0
    measurement_family_by_id = {
        trait_id: measurement_family
        for measurement_family, trait_ids in MEASUREMENT_FAMILY_TRAIT_IDS.items()
        for trait_id in trait_ids
    }
    for ordinal, (family_key, family_name, record) in enumerate(
        [
            (family_key, family_name, record)
            for family_key, family_name in (
                ("primary_root_traits", "primary_root"),
                ("root_hair_traits", "root_hair"),
            )
            for record in contract[family_key]
        ],
        start=1,
    ):
        cohorts = {}
        for cohort, total in zip(MULTITRAIT_COHORTS, (261, 283), strict=True):
            by_condition = {
                condition: total // 4 + (index < total % 4)
                for index, condition in enumerate(MULTITRAIT_GROUP_ORDER)
            }
            effects = {}
            for effect_key in MULTITRAIT_EFFECT_KEYS:
                if record["field"] in MULTITRAIT_PRIMARY_ENDPOINTS:
                    effects[effect_key] = {
                        "status": "estimated_fixed_15_effect_family",
                        "estimate": 1.0,
                        "ci95_low": 0.9,
                        "ci95_high": 1.1,
                        "endpoint_n": total,
                        "effect_scale": "ratio",
                        "not_estimable_reason": None,
                    }
                    estimated += 1
                else:
                    effects[effect_key] = {
                        "status": "not_estimated",
                        "estimate": None,
                        "ci95_low": None,
                        "ci95_high": None,
                        "endpoint_n": None,
                        "effect_scale": None,
                        "not_estimable_reason": NOT_ESTIMATED_REASON,
                    }
            condition_summaries = {
                condition: {
                    "source_unit_total": by_condition[condition],
                    "non_null_source_unit_n": by_condition[condition],
                    "observability_fraction": 1.0,
                    "median": 1.0,
                    "q25": 1.0,
                    "q75": 1.0,
                    "iqr": 0.0,
                    "minimum": 1.0,
                    "maximum": 1.0,
                    "summary_status": CONDITION_SUMMARY_STATUS,
                    "not_estimable_reason": None,
                    "raw_unadjusted": True,
                    "unit_of_analysis": CONDITION_ROW_UNIT,
                }
                for condition in MULTITRAIT_GROUP_ORDER
            }
            cohorts[cohort] = {
                "source_unit_total": total,
                "non_null_source_unit_n": total,
                "support_fraction": 1.0,
                "summary_status": "estimated_descriptive_source_unit_summary",
                "mean": 1.0,
                "median": 1.0,
                "q25": 1.0,
                "q75": 1.0,
                "minimum": 1.0,
                "maximum": 1.0,
                "effect_source_unit_n_by_condition": by_condition,
                "effect_source_unit_n": total,
                "condition_summaries": condition_summaries,
                "effects": effects,
            }
        descriptors.append(
            {
                "ordinal": ordinal,
                "trait_id": record["id"],
                "trait_family": family_name,
                "measurement_family": measurement_family_by_id[record["id"]],
                "field": record["field"],
                "display_name_cn": record["display_name_cn"],
                "unit": record["unit"],
                "value_type": record["type"],
                "source_definition": record["source"],
                "cohorts": cohorts,
            }
        )
    payload = {
        "schema_version": MULTITRAIT_SCHEMA_VERSION,
        "status": "completed_source_derived_32_trait_atlas",
        "row_unit": "one visible primary root per canonical source image",
        "descriptor_count": 32,
        "root_descriptor_count": 19,
        "hair_descriptor_count": 13,
        "cohort_order": list(MULTITRAIT_COHORTS),
        "effect_order": list(MULTITRAIT_EFFECT_KEYS),
        "condition_order": list(MULTITRAIT_GROUP_ORDER),
        "measurement_family_order": list(MEASUREMENT_FAMILY_ORDER),
        "prespecified_inferential_endpoint_fields": list(
            MULTITRAIT_PRIMARY_ENDPOINTS
        ),
        "effect_slot_count": 192,
        "estimated_effect_slot_count": estimated,
        "not_estimated_effect_slot_count": 192 - estimated,
        "condition_summary_slot_count": 256,
        "estimated_condition_summary_slot_count": 256,
        "not_estimated_condition_summary_slot_count": 0,
        "source_sha256": source_sha256,
        "descriptors": descriptors,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    payload["atlas_identity_sha256"] = _canonical_hash(payload)
    return payload


def _write(path: Path, payload) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _seal(payload: dict, field: str) -> dict:
    payload[field] = _canonical_hash(payload)
    return payload


def _fixture_legacy_prediction_locks() -> list[dict[str, str]]:
    return [
        {"task_id": f"qc-{index:02d}", "sha256": _hash(f"legacy-qc-{index:02d}")}
        for index in range(44)
    ]


def _fixture_legacy_prediction_identity() -> str:
    return _canonical_hash(_fixture_legacy_prediction_locks())


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> dict[str, Path]:
    hashes = [_hash(f"checkpoint-{index}") for index in range(5)]
    expert = "PHAxis-StageB-train399-five-seed"
    training_lock = {
        "dataset_manifest_sha256": _hash("dataset"),
        "split_manifest_sha256": _hash("split"),
        "dataset_split_identity_sha256": _hash("split-identity"),
        "integrity_manifest_sha256": _hash("integrity"),
    }
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
            for index, digest in enumerate(hashes)
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
            "expert_id": expert,
            "deployment_role": "candidate_gate_passed_not_promoted",
            "ensemble_members": 5,
            "training_images": 399,
            "validation_images": 44,
            "validation_labels_used_for_gradient_or_early_stopping": False,
            "blind_images_used": 0,
            "checkpoint_sha256": hashes,
            "candidate_bundle_identity_sha256": _canonical_hash(identity_payload),
        },
    }
    _seal(candidate, "candidate_manifest_identity_sha256")
    paths = {"train399_candidate": _write(tmp_path / "candidate.json", candidate)}

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
        "candidate_bundle_identity_sha256": candidate["candidate_bundle_identity_sha256"],
        "candidate_pool_identity_sha256": _hash("pool"),
        "canonical_ground_truth_lock_identity_sha256": _hash("canonical"),
        "threshold_metrics": threshold_metrics,
        "selected": selected,
        **training_lock,
    }
    _seal(selection, "selection_receipt_identity_sha256")
    paths["train399_selection"] = _write(tmp_path / "selection.json", selection)

    selected_metadata = {
        "expert_id": expert,
        "deployment_role": "candidate_gate_passed_not_promoted",
        "operating_point_status": "selected_on_locked_QCdevelopment44",
        "selected_score_threshold": 0.225,
        "checkpoint_sha256": hashes,
        "candidate_bundle_identity_sha256": candidate["candidate_bundle_identity_sha256"],
        "selection_receipt_identity_sha256": selection["selection_receipt_identity_sha256"],
    }
    selected_metadata["selected_model_metadata_identity_sha256"] = _canonical_hash(
        selected_metadata
    )
    task_ids = [f"qc-{index:02d}" for index in range(44)]
    stageb_prediction_locks = [
        {"task_id": task_id, "sha256": _hash(f"stageb-{task_id}")}
        for task_id in task_ids
    ]
    legacy_prediction_locks = _fixture_legacy_prediction_locks()
    stageb_prediction_identity = _canonical_hash(stageb_prediction_locks)
    legacy_prediction_identity = _canonical_hash(legacy_prediction_locks)
    evaluation_authority = {
        "schema_version": (
            "PHAxis-StageB-train399-QCdev44-evaluation-inference-run-1.0"
        ),
        "artifact_role": (
            "locked_qcdevelopment44_full_geometry_evaluation_only_not_deployable"
        ),
        "evaluation_detection_schema_version": (
            "PHAxis-StageB-train399-QCdev44-evaluation-only-full-geometry-1.0"
        ),
        "evaluation_inference_summary_sha256": _hash(
            "evaluation-inference-summary-file"
        ),
        "evaluation_inference_summary_identity_sha256": _hash(
            "evaluation-inference-summary-identity"
        ),
        "evaluation_gate_identity_sha256": _hash("evaluation-inference-gate"),
        "evaluation_detection_set_identity_sha256": stageb_prediction_identity,
        "model_contract_proposal_required_for_artifact": False,
        "model_contract_proposal_present": False,
        "production_consumption_allowed": False,
        "fusion_consumption_allowed": False,
        "traits_consumption_allowed": False,
        "canonical_annotations_read_during_inference": False,
        "condition_metadata_used_for_routing": False,
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
    }
    if monkeypatch is not None:
        monkeypatch.setattr(
            source_release_common,
            "LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256",
            legacy_prediction_identity,
        )
        monkeypatch.setattr(
            evidence_module,
            "LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256",
            legacy_prediction_identity,
        )
    per_image = [
        {
            "task_id": task_id,
            "stageb_train399": {
                "n_pred": 20,
                "n_gt": 20,
                "biological_presence_tp": {"5.0": 15, "10.0": 17, "20.0": 19},
            },
            "hybrid_max": {
                "n_pred": 20,
                "n_gt": 20,
                "biological_presence_tp": {"5.0": 13, "10.0": 15, "20.0": 17},
            },
        }
        for task_id in task_ids
    ]
    pooled_metric = {
        tolerance: {"tp": 44, "n_pred": 44, "n_gt": 44, "precision": 1.0, "recall": 1.0, "f1": 1.0}
        for tolerance in ("5", "10", "20")
    }
    evaluation = {
        "schema_version": "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2",
        "status": "completed",
        "scope": "locked overlay-visible QC-development44; not independent accuracy",
        "blind_images_used": 0,
        "independent_accuracy_claim_allowed": False,
        "metric_hierarchy": {
            "primary": "one-to-one tolerant biological-hair presence; bidirectional partial centreline coverage without endpoint gates",
            "primary_minimum_truth_coverage": 0.25,
            "primary_minimum_prediction_coverage": 0.25,
            "primary_minimum_direction_cosine": 0.0,
            "primary_tolerance_um": 20.0,
            "primary_matcher_contract": matcher,
            "primary_matcher_contract_sha256": _canonical_hash(matcher),
        },
        "training_contract": {
            "training_images": 399,
            "validation_images": 44,
            "validation_labels_used_for_gradient_or_early_stopping": False,
            "candidate_bundle_identity_sha256": candidate["candidate_bundle_identity_sha256"],
            "selection_receipt_identity_sha256": selection["selection_receipt_identity_sha256"],
            "selected_model_metadata_identity_sha256": selected_metadata[
                "selected_model_metadata_identity_sha256"
            ],
            "checkpoint_sha256": hashes,
            "evaluation_gate_identity_sha256": evaluation_authority[
                "evaluation_gate_identity_sha256"
            ],
            "evaluation_inference_summary_identity_sha256": evaluation_authority[
                "evaluation_inference_summary_identity_sha256"
            ],
        },
        "inputs_sha256": {
            "candidate_manifest": _file_hash(paths["train399_candidate"]),
            "selection_receipt": _file_hash(paths["train399_selection"]),
            "dataset_manifest": training_lock["dataset_manifest_sha256"],
            "split_manifest": training_lock["split_manifest_sha256"],
            "integrity_manifest": training_lock["integrity_manifest_sha256"],
            "canonical_ground_truth_lock_identity": selection[
                "canonical_ground_truth_lock_identity_sha256"
            ],
            "selected_model_metadata": _hash("selected-metadata-file"),
            "evaluation_inference_summary": evaluation_authority[
                "evaluation_inference_summary_sha256"
            ],
        },
        "evaluation_inference_authority": evaluation_authority,
        "overall": {
            expert_id: {
                "images": 44,
                "tolerant_biological_presence": pooled_metric,
                "identity_attachment_proxy": pooled_metric,
                "strict_whole_line_correspondence": pooled_metric,
            }
            for expert_id in ("stageb_train399", "hybrid_max")
        },
        "paired_bootstrap_95ci": {
            "method": "paired image-level nonparametric bootstrap",
            "repetitions": 10000,
            "seed": 20260828,
            "delta_stageb_train399_minus_hybrid": {
                "biological_presence_f1_20um": {
                    "estimate": 0.1,
                    "ci95_low": 0.05,
                    "ci95_high": 0.15,
                }
            },
        },
        "per_image": per_image,
        "prediction_input_locks": {
            "stageb_detection_files": stageb_prediction_locks,
            "stageb_detection_set_identity_sha256": stageb_prediction_identity,
            "hybrid_prediction_files": legacy_prediction_locks,
            "hybrid_prediction_set_identity_sha256": legacy_prediction_identity,
        },
        "comparator_contract": {
            "hybrid_max": {
                "evidence_role": "locked_legacy_development_comparator",
                "schema_version": "RHAxis-NextGen-Hybrid-Max-qc-development-evaluation-1.0",
                "identity_hair_variant": "hybrid_verified_increment",
                "count_hair_variant": "hybrid_verified_increment",
                "endpoint_complete_identity_layer": True,
                "phaxis_payload_allowed": False,
                "stageb_identity_source_allowed": False,
                "prediction_set_identity_sha256": legacy_prediction_identity,
                "expected_prediction_set_identity_sha256": legacy_prediction_identity,
            }
        },
    }
    paths["train399_evaluation"] = _write(tmp_path / "evaluation.json", evaluation)

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
        "bundle_identity_sha256": _hash("root-provider-bundle"),
        "pipeline_identity_sha256": _hash("pipeline"),
        "layers": layers,
        "source_image_mismatch_task_ids": [],
        "prepared_radius_fallback_task_ids": [],
        "attachment_supported_extension_rescue_task_ids": [],
        "pipeline_raw_image_provenance_gate": True,
        "pipeline_stage_evidence_gate": True,
    }
    root = {
        **root_identity,
        "status": "pass_exact_283",
        "audit_identity_sha256": _canonical_hash(root_identity),
        "fresh_portable_raw_image_rerun_completed": True,
        "fresh_283_exact_reproduction_claim_allowed": True,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }
    paths["root_exact283"] = _write(tmp_path / "root.json", root)

    stageb_binding = {
        "expert_id": expert,
        "checkpoint_sha256": hashes,
        "selected_score_threshold": 0.225,
        "candidate_bundle_identity_sha256": candidate[
            "candidate_bundle_identity_sha256"
        ],
        "selection_receipt_identity_sha256": selection[
            "selection_receipt_identity_sha256"
        ],
        "selected_model_metadata_identity_sha256": selected_metadata[
            "selected_model_metadata_identity_sha256"
        ],
    }
    public_identity = derive_public_identity(
        stageb_binding,
        root_bundle_identity_sha256=root["bundle_identity_sha256"],
    )
    model_bundle_id = public_identity["model_bundle_id"]
    root_expert_id = public_identity["root_expert_id"]
    proposal = {
        "schema_version": "PHAxis-model-contract-1.0.0",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "formal_release_status": "passed_proposal_not_official",
        "model_bundle_id": model_bundle_id,
        "public_system_identity": {
            "schema_version": PUBLIC_SYSTEM_IDENTITY_SCHEMA,
            "identity_sha256": public_identity["public_system_identity_sha256"],
            "derivation": PUBLIC_SYSTEM_DERIVATION,
        },
        "expert_boundary": {
            "root_point_scale_continuity_statistics": root_expert_id,
            "hair_identity_and_count": expert,
            "hair_length": "endpoint-complete comparator associated one-to-one",
        },
        "root_expert": {
            "provider_role": public_identity["root_provider_role"],
            "expert_id": root_expert_id,
            "fresh_exact283_audit_identity_sha256": root[
                "audit_identity_sha256"
            ],
            "bundle_identity_sha256": root["bundle_identity_sha256"],
            "pipeline_identity_sha256": root["pipeline_identity_sha256"],
            "root_bundle_authority": {
                "bundle_identity_sha256": root["bundle_identity_sha256"],
                "pipeline_identity_sha256": root["pipeline_identity_sha256"],
            },
            "root_cap_region_output": False,
        },
        "promotion": {
            "schema_version": "PHAxis-model-contract-promotion-1.0",
            "status": "validated_proposal_not_applied",
            "official_apply_performed": False,
            "formal_gate_source_sha256": {
                role: _file_hash(paths[role])
                for role in (
                    "train399_candidate",
                    "train399_selection",
                    "train399_evaluation",
                    "root_exact283",
                )
            },
            "formal_gate_identity_sha256": {
                "candidate_bundle_identity_sha256": candidate[
                    "candidate_bundle_identity_sha256"
                ],
                "selection_receipt_identity_sha256": selection[
                    "selection_receipt_identity_sha256"
                ],
                "selected_model_metadata_identity_sha256": selected_metadata[
                    "selected_model_metadata_identity_sha256"
                ],
                "root_exact283_audit_identity_sha256": root[
                    "audit_identity_sha256"
                ],
            },
            "stageb_binding": stageb_binding,
        },
    }
    _seal(proposal, "model_contract_identity_sha256")
    paths["model_contract_proposal"] = _write(tmp_path / "proposal.json", proposal)
    proposal_binding = {
        "model_contract_proposal_sha256": _file_hash(
            paths["model_contract_proposal"]
        ),
        "model_contract_proposal_identity_sha256": proposal[
            "model_contract_identity_sha256"
        ],
    }

    stageb = _seal(
        {
            "schema_version": "PHAxis-StageB-inference-run-1.1",
            "status": "completed",
            "images": 283,
            "checkpoint_sha256": hashes,
            "detection_model_metadata": selected_metadata,
            "score_threshold": 0.225,
            "model_bundle_id": model_bundle_id,
            "root_expert_id": root_expert_id,
            "root_cap_region_output": False,
            "blind_images_used": 0,
            **proposal_binding,
        },
        "summary_identity_sha256",
    )
    paths["stageb"] = _write(tmp_path / "stageb.json", stageb)
    prediction_sha = {f"task-{index:03d}": _hash(f"prediction-{index}") for index in range(283)}
    fusion = _seal(
        {
            "schema_version": "PHAxis-fusion-run-1.1",
            "status": "completed",
            "images": 283,
            "hair_identity_count_expert": expert,
            "model_bundle_id": model_bundle_id,
            "root_expert": root_expert_id,
            "source_stageb_summary_sha256": _file_hash(paths["stageb"]),
            "records": [
                {"task_id": task, "prediction_sha256": digest}
                for task, digest in prediction_sha.items()
            ],
            "root_cap_region_output": False,
            "blind_images_used": 0,
            **proposal_binding,
        },
        "summary_identity_sha256",
    )
    paths["fusion"] = _write(tmp_path / "fusion.json", fusion)
    full_image_traits_path = tmp_path / "full-image-traits.csv"
    full_image_traits_path.write_text(
        "task_id,model_bundle_id,root_expert_id\n"
        f"task-000,{model_bundle_id},{root_expert_id}\n",
        encoding="utf-8",
    )
    traits = _seal(
        {
            "schema_version": "PHAxis-trait-export-1.0",
            "status": "completed",
            "tasks": 283,
            "hair_identity_count_expert": expert,
            "model_bundle_id": model_bundle_id,
            "root_expert_id": root_expert_id,
            "prediction_sha256": prediction_sha,
            "image_traits_sha256": _file_hash(full_image_traits_path),
            "traits_sha256": _hash("traits-full"),
            "hair_instances_sha256": _hash("hairs-full"),
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
            **proposal_binding,
        },
        "export_identity_sha256",
    )
    paths["traits"] = _write(tmp_path / "traits.json", traits)
    primary_traits = _hash("traits-clean261")
    primary_hairs = _hash("hairs-clean261")
    cohorts = _seal(
        {
            "schema_version": "PHAxis-biological-cohorts-1.0",
            "status": "completed_without_fitting_biological_effect_models",
            "counts": {"biological_full": 283, "biological_clean": 261},
            "cohort_directories": {
                "primary": "primary_clean261",
                "sensitivity": "sensitivity_full283",
            },
            "input_sha256": {"trait_export_summary": _file_hash(paths["traits"])},
            "output_sha256": {
                "primary_clean261": {
                    "traits": primary_traits,
                    "hair_instances": primary_hairs,
                },
                "sensitivity_full283": {"traits": _hash("traits-283")},
            },
            "model_bundle_id": model_bundle_id,
            "root_expert_id": root_expert_id,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
            **proposal_binding,
        },
        "cohort_build_identity_sha256",
    )
    paths["cohorts"] = _write(tmp_path / "cohorts.json", cohorts)
    wt_table_root = tmp_path / "wt-secondary"
    wt_table_root.mkdir(parents=True)
    wt_contrasts, wt_meta, wt_flow = _wt_secondary_fixture()
    wt_table_paths: dict[str, Path] = {}
    for role, frame in (
        ("wt_within_experiment_contrasts", wt_contrasts),
        ("wt_within_day_meta_analysis", wt_meta),
        ("wt_temperature_qc_flow", wt_flow),
    ):
        wt_table_paths[role] = wt_table_root / f"{role}.csv"
        frame.to_csv(wt_table_paths[role], index=False, lineterminator="\n")
    wt_evidence = evidence_module.validate_wt_secondary_evidence(
        contrasts=wt_contrasts.to_dict("records"),
        meta=wt_meta.to_dict("records"),
        flow=wt_flow.to_dict("records"),
    )
    analysis = _seal(
        {
            "schema_version": "PHAxis-exploratory-biological-analysis-1.0",
            "status": "completed_exploratory_clean_primary_full_sensitivity",
            "primary_cohort": "primary_clean261",
            "sensitivity_cohort": "sensitivity_full283",
            "cohort_build_summary_sha256": _file_hash(paths["cohorts"]),
            "output_table_sha256": {
                "primary": _hash("analysis-table"),
                **{
                    role: _file_hash(path)
                    for role, path in wt_table_paths.items()
                },
            },
            "D15_fixed_effect_rows": 15,
            "D15_fixed_effect_family_changed_by_WT_secondary": False,
            "wt_secondary_within_experiment_rows": wt_evidence[
                "within_experiment_rows"
            ],
            "wt_secondary_estimable_within_experiment_rows": wt_evidence[
                "estimated_within_experiment_rows"
            ],
            "wt_secondary_unknown_day_contrast_rows": wt_evidence[
                "unknown_day_contrast_rows"
            ],
            "wt_secondary_within_day_meta_rows": wt_evidence[
                "within_day_meta_rows"
            ],
            "wt_secondary_estimable_within_day_meta_rows": wt_evidence[
                "estimated_within_day_meta_rows"
            ],
            "wt_secondary_typed_not_estimable_meta_rows": wt_evidence[
                "typed_not_estimable_meta_rows"
            ],
            "wt_secondary_cross_day_pooling_performed": False,
            "wt_secondary_unknown_day_meta_analysis_performed": False,
            "wt_secondary_clean_full_pooling_performed": False,
            "wt_secondary_claim_status": (
                "secondary exploratory blocked replication; pooled estimates "
                "require at least three estimable experiments within one "
                "developmental day"
            ),
            "wt_secondary_analysis": {
                "schema_version": "PHAxis-WT-temperature-secondary-1.0",
                "status": "materialized_as_separate_secondary_family",
                "endpoint_count": 5,
                "within_experiment_estimand": (
                    "30C_over_22C_ratio_on_log_or_log_link_scale"
                ),
                "minimum_per_temperature_base_and_endpoint": 3,
                "minimum_experiments_per_day_meta_analysis": 3,
                "meta_analysis": (
                    "random_effects_REML_with_Hartung_Knapp_interval"
                ),
                "within_experiment_multiplicity": (
                    "Benjamini-Hochberg within each cohort across every "
                    "estimated experiment-by-endpoint contrast, including "
                    "unknown-day contrasts"
                ),
                "within_day_meta_multiplicity": (
                    "Benjamini-Hochberg within each cohort across every "
                    "estimated developmental-day-by-endpoint meta-analysis"
                ),
                "cross_day_pooling_performed": False,
                "unknown_day_meta_analysis_performed": False,
                "clean_full_pooling_performed": False,
                "D15_fixed_effect_family_changed": False,
            },
            "model_bundle_id": model_bundle_id,
            "root_expert_id": root_expert_id,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
            **proposal_binding,
        },
        "analysis_identity_sha256",
    )
    paths["analysis"] = _write(tmp_path / "analysis.json", analysis)
    profiles = _seal(
        {
            "schema_version": "PHAxis-distal-axis-profile-export-1.0.0",
            "status": "completed",
            "tasks": 261,
            "locked_1_4mm_trait_crosscheck_tasks": 261,
            "locked_1_4mm_trait_crosscheck_mismatches": 0,
            "traits_csv_sha256": primary_traits,
            "hair_instances_csv_sha256": primary_hairs,
            "profiles_csv_sha256": _hash("profiles-table"),
            "model_bundle_id": model_bundle_id,
            "root_expert_id": root_expert_id,
            "root_cap_region_output": False,
            "blind_images_used": 0,
            **proposal_binding,
        },
        "export_identity_sha256",
    )
    paths["profiles"] = _write(tmp_path / "profiles.json", profiles)

    source_summary_sha = {
        "train399_evaluation": _file_hash(paths["train399_evaluation"]),
        "root_exact283": _file_hash(paths["root_exact283"]),
        **{role: _file_hash(paths[role]) for role in ("stageb", "fusion", "traits", "cohorts", "analysis", "profiles")},
    }
    assembly_root = tmp_path / "assembled"
    resource_root = assembly_root / "resources"
    source_root = assembly_root / "source_inputs"
    provenance_root = assembly_root / "provenance"
    resource_root.mkdir(parents=True)
    source_root.mkdir()
    provenance_root.mkdir()
    supplementary_authorities = _supplementary_source_fixture(
        tmp_path / "supplementary-table-authorities"
    )
    resources = {}
    resource_files: dict[str, Path] = {}
    narrative_rows = [
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
        for endpoint in NARRATIVE_ENDPOINT_ORDER
        for effect in NARRATIVE_EFFECT_ORDER
        for cohort in NARRATIVE_COHORT_ORDER
    ]
    narrative_decision = build_narrative_decision(
        narrative_rows,
        source_sha256={"synthetic_effect_family": _hash("narrative-effect-family")},
    )
    qcdev_assignment = _seal(
        {
            "schema_version": "PHAxis-qcdev-instance-assignment-1.0",
            "status": "completed_locked_qc_development_assignment",
            "scope": "QC-development only; non-independent",
            "assignments": [],
            "blind_images_used": 0,
            "independent_accuracy_claim_allowed": False,
        },
        "assignment_identity_sha256",
    )
    for role in evidence_module.FIGURE_RESOURCE_ROLES:
        if role in {"trait_contract", "multitrait_atlas"}:
            continue
        authority_path = wt_table_paths.get(role)
        if authority_path is None:
            authority_path = supplementary_authorities.get(f"resource/{role}")
        suffix = authority_path.suffix if authority_path is not None else ".dat"
        resource_path = resource_root / f"{role}{suffix}"
        if role == "narrative_decision":
            _write(resource_path, narrative_decision)
        elif role == "qcdev_assignment":
            _write(resource_path, qcdev_assignment)
        elif authority_path is None:
            resource_path.write_bytes(f"resource:{role}".encode())
        else:
            resource_path.write_bytes(authority_path.read_bytes())
        resource_files[role] = resource_path
        resources[role] = {
            "path": str(resource_path.relative_to(assembly_root)).replace("\\", "/"),
            "sha256": _file_hash(resource_path),
        }
    source_inputs = {}
    source_files: dict[str, Path] = {}
    for role in evidence_module.FIGURE_SOURCE_INPUT_ROLES:
        authority_path = wt_table_paths.get(role)
        if authority_path is None:
            authority_path = supplementary_authorities.get(f"source/{role}")
        if role == "model_contract_proposal":
            authority_path = paths["model_contract_proposal"]
        suffix = authority_path.suffix if authority_path is not None else ".csv"
        source_path = source_root / f"{role}{suffix}"
        if role == "full_image_traits":
            source_path.write_bytes(full_image_traits_path.read_bytes())
        elif authority_path is not None:
            source_path.write_bytes(authority_path.read_bytes())
        else:
            source_path.write_text(f"role,value\n{role},1\n", encoding="utf-8")
        source_files[role] = source_path
        source_inputs[role] = {
            "path": str(source_path.relative_to(assembly_root)).replace("\\", "/"),
            "sha256": _file_hash(source_path),
        }
    # The H11 reviewer-facing raw-median companion is derived from these exact
    # primary/sensitivity table bytes.  Keep the synthetic analysis receipt as
    # strict as the production receipt by binding those files before resealing
    # the receipt and refreshing its downstream source-summary hash.
    analysis["output_table_sha256"].update(
        {
            "primary_tests": source_inputs["analysis_primary_table"]["sha256"],
            "sensitivity_tests": source_inputs["analysis_sensitivity_table"][
                "sha256"
            ],
        }
    )
    analysis.pop("analysis_identity_sha256")
    _seal(analysis, "analysis_identity_sha256")
    _write(paths["analysis"], analysis)
    source_summary_sha["analysis"] = _file_hash(paths["analysis"])
    trait_contract = json.loads(
        (PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json").read_text(
            encoding="utf-8"
        )
    )
    trait_contract_path = resource_root / "trait_contract.json"
    _write(trait_contract_path, trait_contract)
    resources["trait_contract"] = {
        "path": str(trait_contract_path.relative_to(assembly_root)).replace(
            "\\", "/"
        ),
        "sha256": _file_hash(trait_contract_path),
    }
    resource_files["trait_contract"] = trait_contract_path
    atlas_source_sha256 = {
        "trait_contract": resources["trait_contract"]["sha256"],
        "canonical_image_traits": source_inputs["full_image_traits"][
            "sha256"
        ],
        **{
            role: source_inputs[role]["sha256"]
            for role in (
                "clean_traits",
                "full_traits",
                "analysis_primary_table",
                "analysis_sensitivity_table",
            )
        },
    }
    # Reuse the same semantically complete atlas fixture that produced the
    # copied clean/full analysis tables.  Its H11 raw-median companion and
    # four-cell condition medians are therefore exact by construction; only
    # replace the source-byte bindings with this assembled fixture's hashes.
    multitrait_atlas = json.loads(
        supplementary_authorities["resource/multitrait_atlas"].read_text(
            encoding="utf-8"
        )
    )
    multitrait_atlas["source_sha256"] = atlas_source_sha256
    multitrait_atlas.pop("atlas_identity_sha256")
    multitrait_atlas["atlas_identity_sha256"] = _canonical_hash(multitrait_atlas)
    multitrait_atlas_path = resource_root / "multitrait_atlas.json"
    _write(multitrait_atlas_path, multitrait_atlas)
    resources["multitrait_atlas"] = {
        "path": str(multitrait_atlas_path.relative_to(assembly_root)).replace(
            "\\", "/"
        ),
        "sha256": _file_hash(multitrait_atlas_path),
    }
    resource_files["multitrait_atlas"] = multitrait_atlas_path
    assert set(resources) == set(evidence_module.FIGURE_RESOURCE_ROLES)
    provenance_specs = {
        "historical_development": "historical_development_identity_sha256",
        "measurement_assurance": "measurement_assurance_identity_sha256",
        "overlay_index": "overlay_selection_identity_sha256",
        "profile_analysis": "analysis_identity_sha256",
        "runtime_latency": "summary_identity_sha256",
        "runtime_production": "summary_identity_sha256",
        "runtime_latency_comparison": "comparison_identity_sha256",
        "runtime_production_comparison": "comparison_identity_sha256",
        "baseline_runtime_latency": "summary_identity_sha256",
        "baseline_runtime_production": "summary_identity_sha256",
    }
    provenance = {}
    for role, identity_field in provenance_specs.items():
        receipt = {
            "schema_version": f"synthetic-{role}-1.0",
            "status": "completed",
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        }
        if role == "measurement_assurance":
            receipt.update(
                {
                    "schema_version": "PHAxis-measurement-assurance-receipt-1.0",
                    "status": "completed_locked_qc_development_assurance",
                    "scope": "QC-development measurement assurance; non-independent",
                    "independent_accuracy_claim_allowed": False,
                    "source_table_sha256": {
                        short: source_inputs[source_role]["sha256"]
                        for short, source_role in (
                            ("metrics", "assurance_metrics"),
                            ("pairs", "assurance_pairs"),
                            ("support", "assurance_support"),
                            ("topology", "assurance_topology"),
                        )
                    },
                }
            )
        _seal(receipt, identity_field)
        receipt_path = provenance_root / f"{role}.json"
        _write(receipt_path, receipt)
        provenance[role] = {
            "path": str(receipt_path.relative_to(assembly_root)).replace("\\", "/"),
            "sha256": _file_hash(receipt_path),
            "identity_field": identity_field,
            "identity_sha256": receipt[identity_field],
        }
    train399_prediction_provenance = {
        "task_order_identity_sha256": _canonical_hash(task_ids),
        "stageb_train399": {
            "schema_version": evaluation_authority[
                "evaluation_detection_schema_version"
            ],
            "artifact_role": evaluation_authority["artifact_role"],
            "evaluation_inference_summary_sha256": evaluation_authority[
                "evaluation_inference_summary_sha256"
            ],
            "evaluation_inference_summary_identity_sha256": evaluation_authority[
                "evaluation_inference_summary_identity_sha256"
            ],
            "evaluation_gate_identity_sha256": evaluation_authority[
                "evaluation_gate_identity_sha256"
            ],
            "production_consumption_allowed": False,
            "fusion_consumption_allowed": False,
            "traits_consumption_allowed": False,
            "ordered_file_set_identity_sha256": stageb_prediction_identity,
        },
        "legacy_hybrid_endpoint_complete_identity_layer": {
            **evaluation["comparator_contract"]["hybrid_max"],
            "ordered_file_set_identity_sha256": legacy_prediction_identity,
        },
    }
    public_identity = {
        "model_bundle_id": model_bundle_id,
        "root_expert_id": root_expert_id,
        "root_provider_role": "PHAxis-portable-root-provider",
    }
    wt_binding = evidence_module.validate_wt_secondary_analysis_binding(
        analysis_summary=analysis,
        evidence_summary=wt_evidence,
        table_sha256={
            role: source_inputs[role]["sha256"]
            for role in evidence_module.WT_SECONDARY_RESOURCE_ROLES
        },
    )
    figure_inputs = {
        "schema_version": "PHAxis-manuscript-figure-inputs-2.0",
        "assembler_schema_version": "PHAxis-publication-figure-input-assembly-1.0",
        "status": "final",
        "source_summary_sha256": source_summary_sha,
        **proposal_binding,
        "model_contract_public_identity": public_identity,
        "model_bundle_id": model_bundle_id,
        "root_expert_id": root_expert_id,
        "hair_identity_expert_id": expert,
        "train399_prediction_input_provenance": train399_prediction_provenance,
        "train399_selection_sha256": _file_hash(paths["train399_selection"]),
        "train399_selection_identity_sha256": selection[
            "selection_receipt_identity_sha256"
        ],
        "narrative_decision_identity_sha256": narrative_decision[
            "narrative_decision_identity_sha256"
        ],
        "narrative_branch_id": narrative_decision["branch_id"],
        "qcdev_assignment_identity_sha256": qcdev_assignment[
            "assignment_identity_sha256"
        ],
        "resources": resources,
        "resource_lineage": {
            role: [role] for role in resources
        },
        "source_inputs": source_inputs,
        "wt_secondary_evidence": wt_binding,
        "provenance_receipts": provenance,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    supplementary_contract = evidence_module.supplementary_figure_contract()
    supplementary_contract["contract_identity_sha256"] = _canonical_hash(
        supplementary_contract
    )
    figure_inputs["supplementary_figure_contract"] = supplementary_contract
    _seal(figure_inputs, "figure_input_assembly_identity_sha256")
    paths["figure_inputs"] = _write(
        assembly_root / "figure_inputs.json", figure_inputs
    )
    supplementary_source_paths: dict[str, Path] = {}
    for spec in SUPPLEMENTARY_TABLE_SPECS:
        for authority_role in spec["source_roles"]:
            authority_kind, role = authority_role.split("/", 1)
            if authority_kind == "source":
                supplementary_source_paths[authority_role] = source_files[role]
            elif authority_kind == "resource":
                supplementary_source_paths[authority_role] = resource_files[role]
            elif authority_kind == "receipt":
                supplementary_source_paths[authority_role] = paths[role]
            elif authority_kind == "proposal":
                supplementary_source_paths[authority_role] = paths[
                    "model_contract_proposal"
                ]
            else:  # pragma: no cover - fixed exact-ten test contract
                raise AssertionError(authority_role)
    supplementary_table_bundle = materialize_supplementary_table_data_bundle(
        output=tmp_path / SUPPLEMENTARY_TABLE_BUNDLE_DIRECTORY,
        status=FINAL_SUPPLEMENTARY_TABLE_STATUS,
        source_paths=supplementary_source_paths,
        source_identities={},
        figure_input_manifest_sha256=_file_hash(paths["figure_inputs"]),
        figure_input_assembly_identity_sha256=figure_inputs[
            "figure_input_assembly_identity_sha256"
        ],
        model_contract_proposal_identity_sha256=proposal[
            "model_contract_identity_sha256"
        ],
    )
    bundle_sha = {
        "main_figure": {"pdf": _hash("figure-pdf"), "tiff": _hash("figure-tiff")},
        "profile_figure": {"pdf": _hash("profile-pdf")},
    }
    supplementary_bundle_sha = {
        stem: {
            "pdf": _hash(f"{stem}-pdf"),
            "png": _hash(f"{stem}-png"),
            "tiff": _hash(f"{stem}-tiff"),
            "source_data": {
                f"source_data/{stem}.csv": _hash(f"{stem}-source-data")
            },
        }
        for stem in evidence_module.SUPPLEMENTARY_FIGURE_STEMS
    }
    figures = {
        "schema_version": "PHAxis-publication-figure-suite-1.0",
        "status": "final_sealed_strict_train399_only",
        "formal_train399_only_gate_passed": True,
        "deployment_figures_generated": True,
        "deployment_figures_provisional": False,
        "submission_use_allowed": True,
        "source_summary_sha256": source_summary_sha,
        "figure_input_manifest_sha256": _file_hash(paths["figure_inputs"]),
        "figure_input_assembly_identity_sha256": figure_inputs[
            "figure_input_assembly_identity_sha256"
        ],
        "figure_resource_sha256": {
            role: record["sha256"] for role, record in resources.items()
        },
        "wt_secondary_evidence": wt_binding,
        "train399_prediction_input_provenance": train399_prediction_provenance,
        "model_contract_public_identity": public_identity,
        "model_bundle_id": model_bundle_id,
        "root_expert_id": root_expert_id,
        "hair_identity_expert_id": expert,
        "multitrait_atlas_identity_sha256": multitrait_atlas[
            "atlas_identity_sha256"
        ],
        "narrative_decision_identity_sha256": narrative_decision[
            "narrative_decision_identity_sha256"
        ],
        "narrative_branch_id": narrative_decision["branch_id"],
        "title_contract": title_contract(narrative_decision),
        "figures": {name: {"sha256": value} for name, value in bundle_sha.items()},
        "figure_bundle_sha256": bundle_sha,
        "supplementary_figures": {
            stem: {
                "number": f"S{index}",
                "title": supplementary_contract["figures"][index - 1][
                    "title"
                ],
                "status": "final",
                "resource_roles": supplementary_contract["figures"][
                    index - 1
                ]["resource_roles"],
                "receipt_roles": supplementary_contract["figures"][
                    index - 1
                ]["receipt_roles"],
                "receipt_file_sha256": {
                    role: source_summary_sha[role]
                    for role in supplementary_contract["figures"][index - 1][
                        "receipt_roles"
                    ]
                },
                "source_data_sha256": supplementary_bundle_sha[stem][
                    "source_data"
                ],
            }
            for index, stem in enumerate(
                evidence_module.SUPPLEMENTARY_FIGURE_STEMS, start=1
            )
        },
        "supplementary_figure_bundle_sha256": supplementary_bundle_sha,
        "supplementary_figure_bundle_identity_sha256": _canonical_hash(
            supplementary_bundle_sha
        ),
        "legends_alt_text_sha256": _hash("legends"),
        "supplementary_figure_contract": supplementary_contract,
        "supplementary_figure_contract_identity_sha256": supplementary_contract[
            "contract_identity_sha256"
        ],
        "claim_contract": {
            "main_figure_count": 6,
            "supplementary_figure_count": 9,
            "supplementary_table_data_resource_count": 10,
            "root_cap_region_statistics_included": False,
            "narrative_decision_identity_sha256": narrative_decision[
                "narrative_decision_identity_sha256"
            ],
            "profiles_select_or_veto_narrative_branch": False,
            "wt_secondary_alters_D15_fixed_effect_family": False,
            "wt_cross_day_pooling_performed": False,
            "wt_unknown_day_meta_analysis_performed": False,
            "wt_clean_full_pooling_performed": False,
        },
        "supplementary_tables": supplementary_table_bundle["items"],
        "supplementary_table_bundle_receipt": (
            f"{SUPPLEMENTARY_TABLE_BUNDLE_DIRECTORY}/"
            f"{SUPPLEMENTARY_TABLE_BUNDLE_RECEIPT}"
        ),
        "supplementary_table_bundle_receipt_sha256": (
            supplementary_table_bundle["receipt_sha256"]
        ),
        "supplementary_table_bundle_identity_sha256": (
            supplementary_table_bundle["bundle_identity_sha256"]
        ),
        "supplementary_table_bundle_sha256": supplementary_table_bundle[
            "bundle_file_sha256"
        ],
        "supplementary_table_source_input_sha256": {
            role: record["sha256"] for role, record in source_inputs.items()
        },
        "supplementary_table_source_authority_sha256": (
            supplementary_table_bundle["source_authority_sha256"]
        ),
        "supplementary_table_source_authority_identity": (
            supplementary_table_bundle["source_authority_identity"]
        ),
        "blind_images_used": 0,
        **proposal_binding,
    }
    figures["figure_suite_identity_sha256"] = _canonical_hash(
        evidence_module.figure_suite_identity_preimage(
            status="final",
            figure_hashes=bundle_sha,
            source_hashes=source_summary_sha,
            figure_input_assembly_identity_sha256=figure_inputs[
                "figure_input_assembly_identity_sha256"
            ],
            model_contract_proposal_identity_sha256=proposal[
                "model_contract_identity_sha256"
            ],
            model_contract_public_identity=public_identity,
            train399_prediction_input_provenance=train399_prediction_provenance,
            supplementary_table_bundle_identity_sha256=(
                supplementary_table_bundle["bundle_identity_sha256"]
            ),
            supplementary_table_bundle_receipt_sha256=(
                supplementary_table_bundle["receipt_sha256"]
            ),
        )
    )
    paths["figures"] = _write(tmp_path / "figures.json", figures)
    return paths


def _kwargs(paths: dict[str, Path], output: Path) -> dict:
    return {
        "model_contract_proposal": paths["model_contract_proposal"],
        "train399_candidate": paths["train399_candidate"],
        "train399_selection": paths["train399_selection"],
        "train399_evaluation": paths["train399_evaluation"],
        "root_exact283": paths["root_exact283"],
        "stageb_summary": paths["stageb"],
        "fusion_summary": paths["fusion"],
        "traits_summary": paths["traits"],
        "cohorts_summary": paths["cohorts"],
        "analysis_summary": paths["analysis"],
        "profiles_summary": paths["profiles"],
        "figure_inputs": paths["figure_inputs"],
        "figures_summary": paths["figures"],
        "output": output,
    }


def test_builds_atomic_deterministic_hash_closed_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    first_path = tmp_path / "evidence-a.json"
    second_path = tmp_path / "evidence-b.json"
    first = build_manuscript_evidence_manifest(**_kwargs(paths, first_path))
    second = build_manuscript_evidence_manifest(**_kwargs(paths, second_path))
    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["status"] == "passed_formal_evidence_graph"
    assert first["cohort_identities"] == {"full": "full283", "primary": "clean261"}
    proposal = json.loads(
        paths["model_contract_proposal"].read_text(encoding="utf-8")
    )
    assert first["model_bundle_id"] == proposal["model_bundle_id"]
    assert first["root_expert_id"] == proposal["root_expert"]["expert_id"]
    assert first["model_contract_public_identity"]["root_provider_role"] == (
        "PHAxis-portable-root-provider"
    )
    prediction_provenance = first["train399_prediction_input_provenance"]
    assert prediction_provenance["stageb_detection_schema_version"] == (
        "PHAxis-StageB-train399-QCdev44-evaluation-only-full-geometry-1.0"
    )
    assert prediction_provenance["stageb_detection_artifact_role"] == (
        "locked_qcdevelopment44_full_geometry_evaluation_only_not_deployable"
    )
    assert prediction_provenance["stageb_production_consumption_allowed"] is False
    assert prediction_provenance["stageb_fusion_consumption_allowed"] is False
    assert prediction_provenance["stageb_traits_consumption_allowed"] is False
    assert len(first["figure_input_assembly"]["source_input_sha256"]) == len(
        evidence_module.FIGURE_SOURCE_INPUT_ROLES
    )
    assert len(first["figure_input_assembly"]["resource_sha256"]) == 25
    assert set(evidence_module.WT_SECONDARY_RESOURCE_ROLES) <= set(
        first["figure_input_assembly"]["resource_sha256"]
    )
    assert first["figure_input_assembly"]["wt_secondary_evidence"][
        "estimated_within_day_meta_rows"
    ] == 10
    assert first["figure_input_assembly"]["wt_secondary_evidence"][
        "typed_not_estimable_meta_rows"
    ] == 10
    assert first["figure_input_assembly"]["wt_secondary_evidence"][
        "unknown_day_meta_analysis_performed"
    ] is False
    assert first["figure_input_assembly"]["narrative_branch_id"] == "C"
    figure_input_payload = json.loads(paths["figure_inputs"].read_text(encoding="utf-8"))
    narrative_path = paths["figure_inputs"].parent / figure_input_payload["resources"][
        "narrative_decision"
    ]["path"]
    assert first["figure_input_assembly"]["publication_title_contract"] == title_contract(
        json.loads(narrative_path.read_text(encoding="utf-8"))
    )
    assert len(
        first["figure_input_assembly"]["multitrait_atlas_identity_sha256"]
    ) == 64
    assert list(first["artifacts"]) == [
        "model_contract_proposal",
        "train399_candidate",
        "train399_selection",
        "train399_evaluation",
        "root_exact283",
        "stageb",
        "fusion",
        "traits",
        "cohorts",
        "analysis",
        "profiles",
        "figure_inputs",
        "figures",
    ]
    unsigned = dict(first)
    identity = unsigned.pop("manifest_identity_sha256")
    assert identity == _canonical_hash(unsigned)
    assert not list(tmp_path.glob(".evidence-a.json.*.tmp"))


def test_evidence_validator_rejects_production_schema_for_eval_only_qcdev44(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    evaluation = json.loads(
        paths["train399_evaluation"].read_text(encoding="utf-8")
    )
    provenance = evidence_module._validate_evaluator12(evaluation)
    assert provenance["stageb_detection_schema_version"].startswith(
        "PHAxis-StageB-train399-QCdev44-evaluation-only"
    )
    evaluation["evaluation_inference_authority"][
        "evaluation_detection_schema_version"
    ] = "PHAxis-RHAxiscc-StageB-detections-1.0"
    with pytest.raises(
        EvidenceManifestError,
        match="evaluation-only inference authority/schema missing",
    ):
        evidence_module._validate_evaluator12(evaluation)


def test_rejects_broken_cross_stage_sha_and_leaves_no_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    fusion = json.loads(paths["fusion"].read_text(encoding="utf-8"))
    fusion["source_stageb_summary_sha256"] = _hash("wrong-stageb")
    fusion.pop("summary_identity_sha256")
    _seal(fusion, "summary_identity_sha256")
    _write(paths["fusion"], fusion)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(EvidenceManifestError, match="exact eight-source receipt closure mismatch"):
        build_manuscript_evidence_manifest(**_kwargs(paths, output))
    assert not output.exists()


@pytest.mark.parametrize(
    ("role", "mutation", "message"),
    [
        ("cohorts", lambda value: value["counts"].update(biological_clean=260), "exact eight-source receipt closure mismatch"),
        ("profiles", lambda value: value.update(blind_images_used=1), "blind_images_used must be 0"),
        ("figures", lambda value: value.update(status="provisional_debug_only_not_for_submission"), "development/provisional marker"),
    ],
)
def test_fail_closed_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role, mutation, message) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payload = json.loads(paths[role].read_text(encoding="utf-8"))
    identity_field = {
        "cohorts": "cohort_build_identity_sha256",
        "profiles": "export_identity_sha256",
    }.get(role)
    if identity_field:
        payload.pop(identity_field)
    mutation(payload)
    if identity_field:
        _seal(payload, identity_field)
    _write(paths[role], payload)
    with pytest.raises(EvidenceManifestError, match=message):
        build_manuscript_evidence_manifest(**_kwargs(paths, tmp_path / "blocked.json"))


def test_refuses_to_overwrite_existing_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "owned.json"
    output.write_text("owned\n", encoding="utf-8")
    with pytest.raises(EvidenceManifestError, match="output already exists"):
        build_manuscript_evidence_manifest(**_kwargs(paths, output))
    assert output.read_text(encoding="utf-8") == "owned\n"


def test_evidence_rejects_missing_supplementary_figure_even_if_bundle_resealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    figures = json.loads(paths["figures"].read_text(encoding="utf-8"))
    missing = evidence_module.SUPPLEMENTARY_FIGURE_STEMS[7]
    figures["supplementary_figures"].pop(missing)
    figures["supplementary_figure_bundle_sha256"].pop(missing)
    figures["supplementary_figure_bundle_identity_sha256"] = _canonical_hash(
        figures["supplementary_figure_bundle_sha256"]
    )
    _write(paths["figures"], figures)
    with pytest.raises(
        EvidenceManifestError,
        match="ordered supplementary S1--S9 receipts are not hash-closed",
    ):
        build_manuscript_evidence_manifest(
            **_kwargs(paths, tmp_path / "missing-s8-must-not-exist.json")
        )


def test_evidence_rejects_supplementary_source_authority_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    figures = json.loads(paths["figures"].read_text(encoding="utf-8"))
    role = next(iter(figures["supplementary_table_source_authority_sha256"]))
    figures["supplementary_table_source_authority_sha256"][role] = "0" * 64
    _write(paths["figures"], figures)
    with pytest.raises(
        EvidenceManifestError,
        match="Table/Data S1--S10 hash/denominator closure differs",
    ):
        build_manuscript_evidence_manifest(
            **_kwargs(paths, tmp_path / "authority-drift-must-not-exist.json")
        )


def test_evidence_rejects_wt_secondary_figure_summary_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    figures = json.loads(paths["figures"].read_text(encoding="utf-8"))
    figures["wt_secondary_evidence"]["estimated_within_day_meta_rows"] += 1
    _write(paths["figures"], figures)
    with pytest.raises(
        EvidenceManifestError,
        match="WT secondary evidence/independence binding differs",
    ):
        build_manuscript_evidence_manifest(
            **_kwargs(paths, tmp_path / "wt-drift-must-not-exist.json")
        )


def test_rejects_self_consistent_proposal_with_wrong_root_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    payloads = {
        role: json.loads(path.read_text(encoding="utf-8"))
        for role, path in paths.items()
        if role
        in {
            "train399_candidate",
            "train399_selection",
            "train399_evaluation",
            "root_exact283",
            "stageb",
        }
    }
    proposal = json.loads(
        paths["model_contract_proposal"].read_text(encoding="utf-8")
    )
    wrong_bundle = _hash("different-root-provider-bundle")
    derived = derive_public_identity(
        proposal["promotion"]["stageb_binding"],
        root_bundle_identity_sha256=wrong_bundle,
    )
    proposal["root_expert"]["bundle_identity_sha256"] = wrong_bundle
    proposal["root_expert"]["root_bundle_authority"][
        "bundle_identity_sha256"
    ] = wrong_bundle
    proposal["root_expert"]["expert_id"] = derived["root_expert_id"]
    proposal["model_bundle_id"] = derived["model_bundle_id"]
    proposal["public_system_identity"]["identity_sha256"] = derived[
        "public_system_identity_sha256"
    ]
    proposal["expert_boundary"][
        "root_point_scale_continuity_statistics"
    ] = derived["root_expert_id"]
    with pytest.raises(
        EvidenceManifestError,
        match="root-provider bundle/pipeline/audit differs",
    ):
        evidence_module._validate_proposal_gate_binding(
            proposal,
            paths=paths,
            payloads=payloads,
            stageb_binding=evidence_module._selected_stageb_binding(payloads),
        )
