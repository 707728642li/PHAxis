from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import pytest

from phaxis.biological_analysis import (
    RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
    RAW_EFFECT_HC3_INTERVAL,
    RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
    RAW_EFFECT_OLS_MEAN_CONTRAST,
    raw_median_bootstrap_seed,
)
from phaxis.hair_stageb import preprocess as stageb_preprocess
from phaxis.io import sha256_file
from phaxis.multitrait_atlas import EFFECT_NAME_TO_KEY, build_multitrait_atlas
from phaxis.narrative_decision import build_narrative_decision
from phaxis.public_identity import (
    PUBLIC_SYSTEM_DERIVATION,
    PUBLIC_SYSTEM_IDENTITY_SCHEMA,
    derive_public_identity,
)
from phaxis.root_trait_assurance import ROOT_TRAIT_FAMILY_BY_FIELD
from tests.phaxis.test_supplementary_table_data_bundle import (
    source_fixture as _supplementary_source_fixture,
)
from tests.phaxis.test_publication_figure_input_evidence import (
    _wt_secondary_fixture,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/phaxis/build_publication_figures.py"
SPEC = importlib.util.spec_from_file_location("phaxis_manuscript_figures", SCRIPT)
assert SPEC and SPEC.loader
figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figures)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _guarded(payload: dict) -> dict:
    return {
        **payload,
        "root_cap_region_output": False,
        "blind_images_used": 0,
    }


def _test_stageb_binding() -> dict[str, object]:
    return {
        "candidate_bundle_identity_sha256": figures.sha256_json("candidate"),
        "selection_receipt_identity_sha256": figures.sha256_json("selection"),
        "selected_model_metadata_identity_sha256": figures.sha256_json(
            "selected-metadata"
        ),
    }


def _test_root_pipeline_identity() -> str:
    return figures.sha256_json("root-pipeline")


def _test_root_bundle_identity() -> str:
    return figures.sha256_json("root-provider-bundle")


def _test_root_audit_identity() -> str:
    return figures.sha256_json("root-audit")


def _test_public_identity() -> dict[str, str]:
    return derive_public_identity(
        _test_stageb_binding(),
        root_bundle_identity_sha256=_test_root_bundle_identity(),
    )


def _test_narrative_decision() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for endpoint_index, endpoint in enumerate(figures.PRIMARY_ENDPOINTS):
        for effect_index, effect in enumerate(("OE_vs_EV", "30C_vs_22C", "interaction")):
            for cohort, offset in (("primary_clean261", 0.0), ("sensitivity_full283", 0.02)):
                value = 0.85 + endpoint_index * 0.05 + effect_index * 0.03 + offset
                rows.append(
                    {
                        "cohort": cohort,
                        "endpoint_key": endpoint,
                        "effect_key": effect,
                        "estimate": value,
                        "ci_low": value - 0.08,
                        "ci_high": value + 0.08,
                        "endpoint_n": 25 + endpoint_index,
                        "effect_scale": "ratio",
                    }
                )
    return build_narrative_decision(rows, source_sha256={"fixture": "a" * 64})


def _evaluation_inference_authority(stageb_identity: str) -> dict[str, object]:
    return {
        "schema_version": (
            "PHAxis-StageB-train399-QCdev44-evaluation-inference-run-1.0"
        ),
        "artifact_role": figures.EVALUATION_ARTIFACT_ROLE,
        "evaluation_detection_schema_version": figures.STAGEB_DETECTION_SCHEMA,
        "evaluation_inference_summary_sha256": figures.sha256_json(
            "evaluation-inference-summary-file"
        ),
        "evaluation_inference_summary_identity_sha256": figures.sha256_json(
            "evaluation-inference-summary-identity"
        ),
        "evaluation_gate_identity_sha256": figures.sha256_json(
            "evaluation-inference-gate"
        ),
        "evaluation_detection_set_identity_sha256": stageb_identity,
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


def _receipts(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    task_ids = [f"qc-{index:02d}" for index in range(44)]
    stageb_locks = [
        {"task_id": task_id, "sha256": figures.sha256_json({"stageb": task_id})}
        for task_id in task_ids
    ]
    legacy_locks = [
        {"task_id": task_id, "sha256": figures.sha256_json({"legacy": task_id})}
        for task_id in task_ids
    ]
    stageb_identity = figures.sha256_json(stageb_locks)
    legacy_identity = figures.sha256_json(legacy_locks)
    evaluation_authority = _evaluation_inference_authority(stageb_identity)
    figures.LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256 = legacy_identity
    per_image = []
    for index, task_id in enumerate(task_ids):
        n_gt = 20 + index
        per_image.append(
            {
                "task_id": task_id,
                "stageb_train399": {
                    "n_pred": n_gt + 1,
                    "n_gt": n_gt,
                    "biological_presence_tp": {
                        "5.0": n_gt - 5,
                        "10.0": n_gt - 3,
                        "20.0": n_gt - 1,
                    },
                },
                "hybrid_max": {
                    "n_pred": n_gt - 2,
                    "n_gt": n_gt,
                    "biological_presence_tp": {
                        "5.0": n_gt - 7,
                        "10.0": n_gt - 5,
                        "20.0": n_gt - 3,
                    },
                },
            }
        )
    paths["train399_evaluation"] = _write_json(
        root / "train399-evaluation.json",
        _guarded(
            {
                "schema_version": "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2",
                "status": "completed",
                "independent_accuracy_claim_allowed": False,
                "metric_hierarchy": {
                    "primary": "one-to-one tolerant biological-hair presence; bidirectional partial centreline coverage without endpoint gates",
                    "primary_minimum_truth_coverage": 0.25,
                    "primary_minimum_prediction_coverage": 0.25,
                    "primary_minimum_direction_cosine": 0.0,
                },
                "training_contract": {
                    "training_images": 399,
                    "validation_images": 44,
                    "validation_labels_used_for_gradient_or_early_stopping": False,
                    "evaluation_gate_identity_sha256": evaluation_authority[
                        "evaluation_gate_identity_sha256"
                    ],
                    "evaluation_inference_summary_identity_sha256": (
                        evaluation_authority[
                            "evaluation_inference_summary_identity_sha256"
                        ]
                    ),
                },
                "inputs_sha256": {
                    "evaluation_inference_summary": evaluation_authority[
                        "evaluation_inference_summary_sha256"
                    ]
                },
                "evaluation_inference_authority": evaluation_authority,
                "per_image": per_image,
                "prediction_input_locks": {
                    "stageb_detection_files": stageb_locks,
                    "stageb_detection_set_identity_sha256": stageb_identity,
                    "hybrid_prediction_files": legacy_locks,
                    "hybrid_prediction_set_identity_sha256": legacy_identity,
                },
                "comparator_contract": {
                    "hybrid_max": {
                        "evidence_role": "locked_legacy_development_comparator",
                        "schema_version": figures.LEGACY_HYBRID_COMPARATOR_SCHEMA,
                        "identity_hair_variant": figures.LEGACY_HYBRID_IDENTITY_VARIANT,
                        "count_hair_variant": figures.LEGACY_HYBRID_IDENTITY_VARIANT,
                        "endpoint_complete_identity_layer": True,
                        "phaxis_payload_allowed": False,
                        "stageb_identity_source_allowed": False,
                        "prediction_set_identity_sha256": legacy_identity,
                        "expected_prediction_set_identity_sha256": legacy_identity,
                    }
                },
                "paired_bootstrap_95ci": {
                    "method": "paired image-level nonparametric bootstrap",
                    "repetitions": 10000,
                    "seed": 20260828,
                    "delta_stageb_train399_minus_hybrid": {
                        "biological_presence_f1_20um": {
                            "estimate": 0.05,
                            "ci95_low": 0.02,
                            "ci95_high": 0.08,
                        }
                    },
                },
            }
        ),
    )
    layer = {
        "exact": 283,
        "expected": 283,
        "mismatch_count": 0,
        "mismatch_task_ids": [],
        "gate_pass": True,
    }
    paths["root_exact283"] = _write_json(
        root / "root-exact.json",
        _guarded(
            {
                "schema_version": "PHAxis-root-provider-fresh-reference283-audit-1.0",
                "status": "pass_exact_283",
                "fresh_portable_raw_image_rerun_completed": True,
                "fresh_283_exact_reproduction_claim_allowed": True,
                "pipeline_raw_image_provenance_gate": True,
                "pipeline_stage_evidence_gate": True,
                "canonical_annotations_read": False,
                "audit_identity_sha256": _test_root_audit_identity(),
                "bundle_identity_sha256": _test_root_bundle_identity(),
                "pipeline_identity_sha256": _test_root_pipeline_identity(),
                "layers": {
                    "v12_strip_root_mask": layer,
                    "v20_root_polygon": layer,
                    "final_hybrid_root_mask": layer,
                },
            }
        ),
    )
    expert = "PHAxis-StageB-train399-five-seed"
    public_identity = _test_public_identity()
    model_bundle_id = public_identity["model_bundle_id"]
    root_expert_id = public_identity["root_expert_id"]
    paths["stageb"] = _write_json(
        root / "stageb.json",
        _guarded(
            {
                "schema_version": "PHAxis-StageB-inference-run-1.1",
                "status": "completed",
                "images": 283,
                "detection_model_metadata": {"expert_id": expert},
                "shared_input_acceleration": {
                    "requested": True,
                    "executed_images": 283,
                    "resumed_images_not_executed": 0,
                    "runtime_path_counts": {
                        "shared_input_acceleration": 283
                    },
                    "fallback_reason_counts": {},
                },
                "checkpoint_sha256": [
                    figures.sha256_json({"checkpoint": index})
                    for index in range(5)
                ],
                "model_bundle_id": model_bundle_id,
                "root_expert_id": root_expert_id,
            }
        ),
    )
    paths["fusion"] = _write_json(
        root / "fusion.json",
        _guarded(
            {
                "schema_version": "PHAxis-fusion-run-1.1",
                "status": "completed",
                "images": 283,
                "source_stageb_summary_sha256": sha256_file(paths["stageb"]),
                "hair_identity_count_expert": expert,
                "model_bundle_id": model_bundle_id,
                "root_expert": root_expert_id,
            }
        ),
    )
    paths["traits"] = _write_json(
        root / "traits.json",
        _guarded(
            {
                "schema_version": "PHAxis-trait-export-1.0",
                "status": "completed",
                "tasks": 283,
                "hair_identity_count_expert": expert,
                "model_bundle_id": model_bundle_id,
                "root_expert_id": root_expert_id,
            }
        ),
    )
    paths["cohorts"] = _write_json(
        root / "cohorts.json",
        _guarded(
            {
                "schema_version": "PHAxis-biological-cohorts-1.0",
                "status": "completed",
                "counts": {"biological_full": 283, "biological_clean": 261},
                "input_sha256": {"trait_export_summary": sha256_file(paths["traits"])},
                "model_bundle_id": model_bundle_id,
                "root_expert_id": root_expert_id,
            }
        ),
    )
    paths["analysis"] = _write_json(
        root / "analysis.json",
        _guarded(
            {
                "schema_version": "PHAxis-exploratory-biological-analysis-1.0",
                "status": "completed_exploratory_clean_primary_full_sensitivity",
                "primary_cohort": "primary_clean261",
                "sensitivity_cohort": "sensitivity_full283",
                "cohort_build_summary_sha256": sha256_file(paths["cohorts"]),
                "model_bundle_id": model_bundle_id,
                "root_expert_id": root_expert_id,
            }
        ),
    )
    paths["profiles"] = _write_json(
        root / "profiles.json",
        _guarded(
            {
                "schema_version": "PHAxis-distal-axis-profile-export-1.0.0",
                "status": "completed",
                "tasks": 261,
                "locked_1_4mm_trait_crosscheck_tasks": 261,
                "locked_1_4mm_trait_crosscheck_mismatches": 0,
                "model_bundle_id": model_bundle_id,
                "root_expert_id": root_expert_id,
            }
        ),
    )
    return paths


def _save_csv(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _resources(root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    trait_contract_payload = json.loads(
        (PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json").read_text(
            encoding="utf-8"
        )
    )
    paths["trait_contract"] = _write_json(
        root / "trait-contract.json",
        trait_contract_payload,
    )
    image = np.full((80, 120, 3), 210, dtype=np.uint8)
    image[:, 55:65, :] = 45
    paths["figure1_image"] = root / "measurement-image.png"
    Image.fromarray(image).save(paths["figure1_image"])
    paths["figure1_geometry"] = _write_json(
        root / "measurement-geometry.json",
        {
            "source_image_sha256": sha256_file(paths["figure1_image"]),
            "display": {"kind": "linear_global", "lower": 0, "upper": 255},
            "scale_bar": {"pixels": 25, "micrometres": 100},
            "root_polygon_xy": [[54, 5], [66, 5], [66, 75], [54, 75]],
            "axis_xy": [[60, 72], [60, 8]],
            "distal_point_xy": [60, 72],
            "hair_identities": [
                {
                    "attachment_xy": [55, 45],
                    "identity_xy": [[55, 45], [35, 40]],
                    "length_curve_xy": [[55, 45], [45, 43], [35, 40]],
                },
                {
                    "attachment_xy": [65, 30],
                    "identity_xy": [[65, 30], [82, 27]],
                    "length_curve_xy": None,
                },
            ],
            "root_cap_region_output": False,
            "blind_images_used": 0,
        },
    )

    per_image: list[dict] = []
    task_ids = [f"qc-{index:02d}" for index in range(44)]
    stageb_locks = [
        {"task_id": task_id, "sha256": figures.sha256_json({"stageb": task_id})}
        for task_id in task_ids
    ]
    legacy_locks = [
        {"task_id": task_id, "sha256": figures.sha256_json({"legacy": task_id})}
        for task_id in task_ids
    ]
    lock_by_comparator = {
        figures.COMPARATORS[0]: (
            stageb_locks,
            figures.sha256_json(stageb_locks),
            figures.STAGEB_DETECTION_SCHEMA,
            "stageb_train399_identity",
            figures.EVALUATION_ARTIFACT_ROLE,
        ),
        figures.COMPARATORS[1]: (
            legacy_locks,
            figures.sha256_json(legacy_locks),
            figures.LEGACY_HYBRID_COMPARATOR_SCHEMA,
            figures.LEGACY_HYBRID_IDENTITY_VARIANT,
            "locked_legacy_development_comparator",
        ),
    }
    for index in range(44):
        for comparator, delta in ((figures.COMPARATORS[0], 1), (figures.COMPARATORS[1], -2)):
            n_gt = 20 + index
            offsets = (-5, -3, -1) if comparator == figures.COMPARATORS[0] else (-7, -5, -3)
            locks, set_identity, schema, variant, evidence_role = lock_by_comparator[comparator]
            per_image.append(
                {
                    "source_unit": f"qc-{index:02d}",
                    "source_unit_order": index,
                    "family_key": f"family-{index:02d}",
                    "comparator": comparator,
                    "gt_count": n_gt,
                    "predicted_count": n_gt + delta,
                    "biological_presence_tp_5um": n_gt + offsets[0],
                    "biological_presence_tp_10um": n_gt + offsets[1],
                    "biological_presence_tp_20um": n_gt + offsets[2],
                    "prediction_input_sha256": locks[index]["sha256"],
                    "prediction_input_set_identity_sha256": set_identity,
                    "prediction_input_schema_version": schema,
                    "identity_hair_variant": variant,
                    "evidence_role": evidence_role,
                }
            )
    paths["development_per_image"] = _save_csv(root / "development-per-image.csv", per_image)
    tolerance_rows = []
    per_image_frame = pd.DataFrame(per_image)
    for comparator in figures.COMPARATORS:
        selected = per_image_frame[per_image_frame["comparator"] == comparator]
        for tolerance in (5, 10, 20):
            tp = int(selected[f"biological_presence_tp_{tolerance}um"].sum())
            n_pred = int(selected["predicted_count"].sum())
            n_gt = int(selected["gt_count"].sum())
            precision = tp / n_pred
            recall = tp / n_gt
            f1 = 2 * tp / (n_pred + n_gt)
            tolerance_rows.append(
                {
                    "comparator": comparator,
                    "tolerance_um": tolerance,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "ci_low": max(0.0, f1 - 0.02),
                    "ci_high": min(1.0, f1 + 0.02),
                    "paired_delta_stageb_minus_legacy_f1": 0.05,
                    "paired_delta_ci_low": 0.02,
                    "paired_delta_ci_high": 0.08,
                    "primary_metric": "one_to_one_tolerant_biological_hair_presence",
                    "minimum_truth_coverage": 0.25,
                    "minimum_prediction_coverage": 0.25,
                    "minimum_direction_cosine": 0.0,
                    "endpoint_gate_used": False,
                }
            )
    paths["development_tolerance"] = _save_csv(root / "development-tolerance.csv", tolerance_rows)
    paths["development_threshold"] = _save_csv(
        root / "development-threshold.csv",
        [
            {"threshold": 0.15, "f1_20um": 0.77, "attachment_proxy_f1_20um": 0.70, "count_mae": 7.2, "selected": False, "selection_metric": "tolerant_biological_presence_f1_20um", "straight_base_to_tip_presence_proxy_used": True, "distal_endpoint_or_length_used_as_selection_gate": False},
            {"threshold": 0.20, "f1_20um": 0.81, "attachment_proxy_f1_20um": 0.72, "count_mae": 6.1, "selected": True, "selection_metric": "tolerant_biological_presence_f1_20um", "straight_base_to_tip_presence_proxy_used": True, "distal_endpoint_or_length_used_as_selection_gate": False},
            {"threshold": 0.25, "f1_20um": 0.79, "attachment_proxy_f1_20um": 0.73, "count_mae": 6.8, "selected": False, "selection_metric": "tolerant_biological_presence_f1_20um", "straight_base_to_tip_presence_proxy_used": True, "distal_endpoint_or_length_used_as_selection_gate": False},
        ],
    )
    paths["development_strata"] = _save_csv(
        root / "development-strata.csv",
        [
            {"dimension": dimension, "stratum": stratum, "comparator": figures.HISTORICAL_COMPARATOR, "f1_20um": value, "ci_low": value - 0.05, "ci_high": value + 0.05, "n_images": n}
            for dimension, stratum, value, n in (
                ("density", "sparse", 0.84, 80),
                ("density", "very dense", 0.64, 55),
                ("quality", "Q2", 0.77, 120),
                ("quality", "Q4", 0.82, 95),
                ("annotation", "fully manual", 0.80, 123),
                ("annotation", "assisted-refined", 0.79, 320),
            )
        ],
    )

    assignment = {
        "schema_version": "PHAxis-qcdev-instance-assignment-1.0",
        "status": "completed_recomputed_from_sealed_geometry",
        "evidence_role": "selected_qc_development_non_independent",
        "display_source_unit": "qc-00",
        "assignments": [
            {
                "source_unit": "qc-00",
                "annotated_polylines_xy_um": [
                    [[0.0, 0.0], [8.0, 4.0], [16.0, 9.0]],
                    [[0.0, 16.0], [7.0, 20.0], [14.0, 24.0]],
                ],
                "predicted_polylines_xy_um": [
                    [[0.5, 0.5], [8.5, 4.0], [15.5, 8.5]],
                    [[1.0, 30.0], [8.0, 34.0], [15.0, 38.0]],
                ],
                "matches": [{"predicted_index": 0, "annotated_index": 0}],
                "unmatched_prediction_indices": [1],
                "unmatched_truth_indices": [1],
            }
        ],
        "blind_images_used": 0,
        "independent_accuracy_claim_allowed": False,
    }
    assignment["assignment_identity_sha256"] = figures.sha256_json(assignment)
    paths["qcdev_assignment"] = _write_json(root / "qcdev-assignment.json", assignment)

    metric_rows = []
    metric_spec = (
        ("root", "root_dice", "Root Dice", 0.96, "ratio"),
        ("root", "root_boundary_f1", "Boundary F1", 0.91, "ratio"),
        ("root", "root_hd95_um", "HD95", 22.0, "µm"),
        ("distal", "distal_median_error_um", "Median error", 14.0, "µm"),
        ("distal", "distal_pck", "PCK", 0.95, "ratio"),
        ("scale", "scale_detection_coverage", "Scale coverage", 6 / 38, "ratio"),
        ("scale", "scale_geometry_endpoint_error_um", "Scale-line endpoint error", 3.5, "µm"),
        ("scale", "scale_relative_error_percent", "Relative error", 1.2, "%"),
        ("conditional_length", "conditional_length_mae_um", "Length MAE", 18.0, "µm"),
        ("conditional_length", "conditional_length_ccc", "Length CCC", 0.93, "ratio"),
        ("conditional_length", "matched_endpoint_error_um", "Endpoint error", 14.0, "µm"),
        ("conditional_length", "matched_trajectory_continuity", "Trajectory continuity", 0.91, "fraction"),
        ("conditional_length", "endpoint_complete_support_fraction", "Support", 0.72, "ratio"),
        ("root_trait", "root_trait_agreement", "Median trait CCC", 0.97, "ratio"),
        ("application_topology", "axis_containment_median", "Median axis containment", 1.0, "fraction"),
        ("application_topology", "axis_containment_min", "Minimum axis containment", 0.998, "fraction"),
        ("application_topology", "unsupported_attachment_n", "Unsupported attachments", 0.0, "count"),
        ("provider_equivalence", "provider_exact_fraction", "Provider exact", 1.0, "fraction"),
        ("root_continuity", "root_continuity_reference_axis_coverage_mean", "Mean union reference-axis coverage", 0.97, "fraction"),
        ("root_continuity", "root_continuity_maximum_single_component_coverage_mean", "Mean maximum single-component root coverage", 0.93, "fraction"),
        ("root_continuity", "root_continuity_maximum_single_component_coverage_median", "Median maximum single-component root coverage", 0.95, "fraction"),
        ("root_continuity", "root_continuity_best_component_gap_median_um", "Median longest gap on the best root component", 8.0, "um"),
        ("root_continuity", "root_continuity_break_free_rate", "Break-free root image rate", 0.82, "fraction"),
        ("root_continuity", "root_continuity_visible_axis_extent_mae_um", "Visible root-axis extent MAE", 12.0, "um"),
        ("hair_attachment", "hair_attachment_qualified_precision_20um", "Attachment-qualified precision @20 µm", 0.88, "fraction"),
        ("hair_attachment", "hair_attachment_qualified_recall_20um", "Attachment-qualified recall @20 µm", 0.86, "fraction"),
        ("hair_attachment", "hair_attachment_qualified_f1_20um", "Attachment-qualified F1 @20 µm", 0.87, "fraction"),
        ("hair_attachment", "hair_attachment_error_median_um", "Median base error on formal hair identities", 6.0, "um"),
        ("hair_attachment", "hair_attachment_error_p95_um", "P95 base error on formal hair identities", 18.0, "um"),
    )
    for domain, key, label, value, unit in metric_spec:
        scale_conditional = key in {
            "scale_geometry_endpoint_error_um",
            "scale_relative_error_percent",
        }
        metric_rows.append(
            {
                "domain": domain,
                "metric_key": key,
                "label": label,
                "value": value,
                "ci_low": value * 0.95,
                "ci_high": value * 1.05,
                "unit": unit,
                "n": 6 if scale_conditional else 38 if key == "scale_detection_coverage" else 44,
                "instances": 6 if key.startswith("scale_") else 44,
                "evidence_role": "exact_equivalence" if domain == "provider_equivalence" else "annotated_accuracy",
                "scale_visible_truth_n": 38,
                "scale_trusted_metadata_n": 6,
                "scale_absence_test_n": 0,
                "scale_absence_specificity_status": "not_estimable_no_absent_or_untrusted_scale_cases",
                "scale_fail_closed_evidence_basis": "software_contract_and_unit_tests",
            }
        )
    paths["assurance_metrics"] = _save_csv(root / "assurance-metrics.csv", metric_rows)
    root_pair_rows: list[dict] = []
    for trait_index, trait in enumerate(
        trait_contract_payload["primary_root_traits"]
    ):
        trait_id = str(trait["id"])
        trait_family = ROOT_TRAIT_FAMILY_BY_FIELD[str(trait["field"])]
        for source_index in range(3):
            observed = 10.0 + trait_index * 2.0 + source_index
            root_pair_rows.append(
                {
                    "pair_type": "root_trait",
                    "source_unit": f"r-{trait_id}-{source_index}",
                    "observed": observed,
                    "predicted": observed * 1.01 + 0.05 * source_index,
                    "unit": trait["unit"],
                    "endpoint_error_um": np.nan,
                    "trajectory_continuity": np.nan,
                    "trait_id": trait_id,
                    "trait_key": trait["field"],
                    "trait_family": trait_family,
                }
            )
    paths["assurance_pairs"] = _save_csv(
        root / "assurance-pairs.csv",
        [
            *[{"pair_type": "scale", "source_unit": f"s-{i}", "observed": 2.0 + i / 10, "predicted": 2.02 + i / 10, "unit": "um_per_px", "relative_error_percent": abs((2.02 + i / 10) - (2.0 + i / 10)) / (2.0 + i / 10) * 100, "scale_line_endpoint_error_um": 1.0 + i, "source_image_sha256": f"{i + 1:064x}", "endpoint_error_um": np.nan, "trajectory_continuity": np.nan} for i in range(6)],
            *[{"pair_type": "conditional_length", "source_unit": f"h-{i}", "observed": 80 + i * 15, "predicted": 82 + i * 14, "unit": "um", "endpoint_error_um": 10 + i / 2, "trajectory_continuity": 0.85 + i / 100} for i in range(12)],
            *root_pair_rows,
        ],
    )
    paths["assurance_support"] = _save_csv(
        root / "assurance-support.csv",
        [
            {"condition_code": code, "support_fraction": (60 + index * 4) / 90, "supported_hairs": 60 + index * 4, "identity_hairs": 90, "source_units": 12 + index}
            for index, code in enumerate(figures.GROUP_ORDER)
        ],
    )

    overlay_rows = []
    overlay_audit_rows = []
    for index, role in enumerate(figures.CASE_ROLES):
        task_id = figures.FIGURE4_LOCKED_ANCHOR_TASK_IDS.get(
            role, f"app-{index:03d}"
        )
        source = root / f"case-{index}-source.png"
        overlay = root / f"case-{index}-overlay.png"
        Image.fromarray(image).save(source)
        overlay_array = image.copy()
        overlay_array[:, 59:61] = np.asarray([40, 215, 229], dtype=np.uint8)
        Image.fromarray(overlay_array).save(overlay)
        source_sha = sha256_file(source)
        prediction_sha = figures.sha256_json({"prediction": index})
        formal = role != "fail_closed"
        inset_required = role in {"low_contrast", "curved_dense"}
        overlay_rows.append(
            {
                "case_id": f"case-{index}",
                "case_role": role,
                "source_path": source.name,
                "source_sha256": source_sha,
                "overlay_path": overlay.name,
                "overlay_sha256": sha256_file(overlay),
                "full_cohort_review_overlay_path": (
                    f"full283_review_overlays/synthetic/synthetic/"
                    f"{'formal' if formal else 'review_only'}/"
                    f"{task_id}.phaxis_overlay.png"
                ),
                "full_cohort_review_overlay_sha256": sha256_file(overlay),
                "overlay_bytes_reused_from_full_cohort_review_export": True,
                "scale_bar_um": 100,
                "scale_bar_px": 25,
                "display_lower": 0,
                "display_upper": 255,
                "selection_rule": "preselected morphology/acquisition-challenge role",
                "case_selection_basis": figures.OVERLAY_CASE_SELECTION_BASIS,
                "random_or_representative_performance_sample": False,
                "experimental_condition_metadata_used_for_rendering": False,
                "experimental_condition_metadata_used_for_evidence_assembly": False,
                "experimental_condition_metadata_used_for_evidence_assembly_scope": (
                    figures.OVERLAY_CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
                ),
                "formal_statistics_eligible": formal,
                "task_id": task_id,
                "prediction_sha256": prediction_sha,
                "root_boundary_colour": "#19AADC",
                "axis_colour": "#E6E6E6",
                "distal_colour": "#DC3CFF",
                "length_curve_colour": "#73F55A",
                "identity_vector_colour": "#FFCD14",
                "hair_base_colour": "#FFFF00",
                "visible_endpoint_colour": "#FF6919",
                "inset_required": inset_required,
                "inset_rule": (
                    "deterministic_test_axis_window"
                    if inset_required else "not_applicable"
                ),
                "inset_x0": 20 if inset_required else np.nan,
                "inset_y0": 10 if inset_required else np.nan,
                "inset_x1": 100 if inset_required else np.nan,
                "inset_y1": 70 if inset_required else np.nan,
                "inset_geometry_sha256": (
                    figures.sha256_json({"inset": role})
                    if inset_required else np.nan
                ),
            }
        )
        overlay_audit_rows.append(
            {
                "schema_version": "PHAxis-Fig4-case-audit-2.0",
                "case_id": f"case-{index}",
                "case_role": role,
                "task_id": task_id,
                "source_image_sha256": source_sha,
                "prediction_sha256": prediction_sha,
                "formal_state": "formal" if formal else "review_only",
                "axis_in_root_coverage_fraction": 0.995 if formal else np.nan,
                "axis_single_component_coverage_fraction": 0.990 if formal else np.nan,
                "longest_unsupported_axis_gap_um": 8.0 if formal else np.nan,
                "formal_identity_count": 10 if formal else np.nan,
                "endpoint_complete_support_count": 7 if formal else np.nan,
                "endpoint_complete_support_fraction": 0.70 if formal else np.nan,
                "distal_window_1_4mm_eligible": formal,
                "distal_window_1_4mm_reason": (
                    "eligible_visible_axis_reaches_4mm"
                    if formal else "formal_statistics_ineligible:review_only_fail_closed"
                ),
                "profile_0_5mm_eligible": formal,
                "profile_0_5mm_reason": (
                    "eligible_visible_axis_reaches_5mm"
                    if formal else "formal_statistics_ineligible:review_only_fail_closed"
                ),
                "downstream_eligible": formal,
                "downstream_reason": (
                    "formal_statistics_eligible" if formal else "review_only_fail_closed"
                ),
                "condition_metadata_used": False,
            }
        )
    paths["overlay_selection"] = _save_csv(root / "overlay-selection.csv", overlay_rows)
    paths["overlay_audit"] = _save_csv(root / "overlay-audit.csv", overlay_audit_rows)

    point_rows = []
    units = {
        figures.PRIMARY_ENDPOINTS[0]: "count",
        figures.PRIMARY_ENDPOINTS[1]: "µm",
        figures.PRIMARY_ENDPOINTS[2]: "µm",
        figures.PRIMARY_ENDPOINTS[3]: "µm",
        figures.PRIMARY_ENDPOINTS[4]: "µm",
    }
    for endpoint_index, endpoint in enumerate(figures.PRIMARY_ENDPOINTS):
        for group_index, group in enumerate(figures.GROUP_ORDER):
            for replicate in range(3):
                point_rows.append({"source_unit": f"p-{endpoint_index}-{group_index}-{replicate}", "cohort": "primary_clean261", "condition_code": group, "formal_eligible": True, "endpoint_key": endpoint, "value": 10 + endpoint_index * 20 + group_index * 4 + replicate, "unit": units[endpoint]})
    paths["phenotype_points"] = _save_csv(root / "phenotype-points.csv", point_rows)
    effect_rows = []
    effects = ("OE_vs_EV", "30C_vs_22C", "interaction")
    for cohort, offset in (("primary_clean261", 0.0), ("sensitivity_full283", 0.02)):
        for endpoint_index, endpoint in enumerate(figures.PRIMARY_ENDPOINTS):
            for effect_index, effect in enumerate(effects):
                value = 0.85 + endpoint_index * 0.05 + effect_index * 0.03 + offset
                effect_rows.append({"cohort": cohort, "endpoint_key": endpoint, "effect_key": effect, "estimate": value, "ci_low": value - 0.08, "ci_high": value + 0.08, "endpoint_n": 25 + endpoint_index})
    paths["phenotype_effects"] = _save_csv(root / "phenotype-effects.csv", effect_rows)
    narrative_decision = build_narrative_decision(
        effect_rows,
        source_sha256={"phenotype_effects": sha256_file(paths["phenotype_effects"])},
    )
    paths["narrative_decision"] = _write_json(
        root / "narrative-decision.json",
        narrative_decision,
    )
    profile_rows = []
    for metric in ("identity_abundance", "conditional_median_length_um", "length_support_fraction"):
        for group_index, group in enumerate(figures.GROUP_ORDER):
            for start in range(5):
                base = {"identity_abundance": 20, "conditional_median_length_um": 120, "length_support_fraction": 0.65}[metric]
                value = base + group_index * (3 if metric != "length_support_fraction" else 0.04) + start * (2 if metric != "length_support_fraction" else 0.02)
                profile_rows.append({"cohort": "primary_clean261", "condition_code": group, "bin_start_mm": start, "bin_end_mm": start + 1, "metric_key": metric, "estimate": value, "ci_low": value - (3 if metric != "length_support_fraction" else 0.05), "ci_high": value + (3 if metric != "length_support_fraction" else 0.05), "eligible_n": 15 + group_index, "length_supported_n": 10 + group_index})
    paths["axial_profiles"] = _save_csv(root / "axial-profiles.csv", profile_rows)

    flow_rows = [
        {"node_id": "human443", "label": "HumanCurated", "count": 443, "parent_id": "", "role": "development"},
        {"node_id": "train399", "label": "Train", "count": 399, "parent_id": "human443", "role": "training"},
        {"node_id": "qcdevelopment44", "label": "QC-development", "count": 44, "parent_id": "human443", "role": "selection"},
        {"node_id": "bio_full", "label": "Application", "count": 283, "parent_id": "", "role": "application"},
        {"node_id": "overlap", "label": "SHA overlap", "count": 22, "parent_id": "bio_full", "role": "sensitivity"},
        {"node_id": "bio_clean", "label": "Clean primary", "count": 261, "parent_id": "bio_full", "role": "primary"},
        {"node_id": "formal", "label": "Formal", "count": 280, "parent_id": "bio_full", "role": "formal"},
        {"node_id": "review_only", "label": "Review-only", "count": 3, "parent_id": "bio_full", "role": "review"},
    ]
    paths["cohort_flow"] = _save_csv(root / "cohort-flow.csv", flow_rows)
    paths["workflow_stages"] = _save_csv(
        root / "workflow-stages.csv",
        [
            {"stage_order": index, "stage_name": role.replace("_", " "), "receipt_role": role, "output_identity_sha256": figures.sha256_json({"role": role})}
            for index, role in enumerate(figures.RECEIPT_ROLES)
        ],
    )
    runtime_rows = []
    for index in range(283):
        wall = 7.2 + (index % 9) * 0.1
        runtime_rows.append(
            {
                "source_unit": f"runtime-{index:03d}",
                "wall_seconds": wall,
                "megapixels": 32.0,
                "io_seconds": 0.4,
                "preprocess_seconds": 0.8,
                "inference_seconds": 4.9,
                "postprocess_seconds": wall - 6.1,
            }
        )
    paths["runtime_per_image"] = _save_csv(root / "runtime-per-image.csv", runtime_rows)
    latency = {
        "schema_version": "PHAxis-full-workflow-sequential-latency-benchmark-1.0",
        "status": "completed_direct_full283",
        "benchmark_mode": "sequential_persistent_full283",
        "measurement_scope": "raw_image_to_final_traits_and_profiles_direct",
        "images": 283,
        "stage_timing_semantics": "nonoverlapping_wall_components",
        "startup_included_in_per_image_wall": False,
        "median_seconds_per_image": float(pd.DataFrame(runtime_rows)["wall_seconds"].median()),
        "p95_seconds_per_image": float(pd.DataFrame(runtime_rows)["wall_seconds"].quantile(0.95)),
        "per_image_csv_sha256": sha256_file(paths["runtime_per_image"]),
        "hardware": {"gpu_names": ["RTX 3090"]},
    }
    production = {
        "schema_version": "PHAxis-full-workflow-production-batch-benchmark-1.0",
        "status": "completed_direct_full283",
        "benchmark_mode": "production_batch_full283",
        "measurement_scope": "raw_image_to_final_traits_and_profiles_direct",
        "images": 283,
        "stage_timing_semantics": "nonoverlapping_wall_components",
        "batch_wall_seconds": 720.0,
        "images_per_min": 23.5833,
        "megapixels_per_second": 12.5778,
        "peak_vram_mib": 9216.0,
        "mean_gpu_utilization_pct": 76.0,
        "stage_timings": [
            {"stage": "I/O", "wall_seconds": 58.0},
            {"stage": "preprocess", "wall_seconds": 74.0},
            {"stage": "inference", "wall_seconds": 430.0},
            {"stage": "fusion + traits + profiles", "wall_seconds": 138.0},
        ],
        "hardware": {"gpu_names": ["RTX 3090"]},
    }
    paths["runtime_summary"] = _write_json(
        root / "runtime-summary.json",
        _guarded(
            {
                "schema_version": "PHAxis-manuscript-two-mode-runtime-input-1.0",
                "status": "completed_two_mode_direct_full283",
                "measurement_scope": "raw_image_to_final_traits_and_profiles_direct",
                "latency_mode": "sequential_persistent_full283",
                "sequential_latency_full283": latency,
                "production_batch_full283": production,
                "baseline_sequential_latency_full283": {
                    **latency,
                    "median_seconds_per_image": latency["median_seconds_per_image"] * 2.0,
                    "p95_seconds_per_image": latency["p95_seconds_per_image"] * 2.0,
                },
                "baseline_production_batch_full283": {
                    **production,
                    "batch_wall_seconds": 1440.0,
                    "images_per_min": production["images_per_min"] / 2.0,
                    "megapixels_per_second": production["megapixels_per_second"] / 2.0,
                    "peak_vram_mib": 12800.0,
                    "mean_gpu_utilization_pct": 54.0,
                },
                "latency_comparison": {
                    "comparable": True,
                    "benchmark_mode": "sequential_persistent_full283",
                    "median_latency_speedup_frozen_v1_over_phaxis": 2.0,
                },
                "production_comparison": {
                    "comparable": True,
                    "batch_wall_speedup_frozen_v1_over_phaxis": 2.0,
                },
                "per_image_csv_sha256": sha256_file(paths["runtime_per_image"]),
                "batch_latency_is_never_derived_per_image": True,
            }
        ),
    )
    wt_contrasts, wt_meta, wt_flow = _wt_secondary_fixture()
    for role, frame in (
        ("wt_within_experiment_contrasts", wt_contrasts),
        ("wt_within_day_meta_analysis", wt_meta),
        ("wt_temperature_qc_flow", wt_flow),
    ):
        paths[role] = root / f"{role}.csv"
        frame.to_csv(paths[role], index=False, lineterminator="\n")
    assert set(paths) == set(figures.RESOURCE_ROLES) - {"multitrait_atlas"}
    return paths


def _fixture(root: Path, *, status: str = "final") -> tuple[dict[str, Path], Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    receipts = _receipts(root)
    resources = _resources(root)
    wt_contract = figures.validate_wt_secondary_evidence(
        contrasts=pd.read_csv(
            resources["wt_within_experiment_contrasts"]
        ).to_dict("records"),
        meta=pd.read_csv(resources["wt_within_day_meta_analysis"]).to_dict(
            "records"
        ),
        flow=pd.read_csv(resources["wt_temperature_qc_flow"]).to_dict(
            "records"
        ),
    )
    analysis_payload = json.loads(receipts["analysis"].read_text(encoding="utf-8"))
    analysis_payload.update(
        {
            "output_table_sha256": {
                role: sha256_file(resources[role])
                for role in figures.WT_SECONDARY_RESOURCE_ROLES
            },
            "D15_fixed_effect_rows": 15,
            "D15_fixed_effect_family_changed_by_WT_secondary": False,
            "wt_secondary_within_experiment_rows": wt_contract[
                "within_experiment_rows"
            ],
            "wt_secondary_estimable_within_experiment_rows": wt_contract[
                "estimated_within_experiment_rows"
            ],
            "wt_secondary_unknown_day_contrast_rows": wt_contract[
                "unknown_day_contrast_rows"
            ],
            "wt_secondary_within_day_meta_rows": wt_contract[
                "within_day_meta_rows"
            ],
            "wt_secondary_estimable_within_day_meta_rows": wt_contract[
                "estimated_within_day_meta_rows"
            ],
            "wt_secondary_typed_not_estimable_meta_rows": wt_contract[
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
            "root_cap_region_statistics_included": False,
            "wt_secondary_analysis": {
                "schema_version": "PHAxis-WT-temperature-secondary-1.0",
                "status": "materialized_as_separate_secondary_family",
                "endpoint_count": 5,
                "within_experiment_estimand": "30C_over_22C_ratio_on_log_or_log_link_scale",
                "minimum_per_temperature_base_and_endpoint": 3,
                "minimum_experiments_per_day_meta_analysis": 3,
                "meta_analysis": "random_effects_REML_with_Hartung_Knapp_interval",
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
        }
    )
    _write_json(receipts["analysis"], analysis_payload)
    assurance_topology = _save_csv(
        root / "assurance-topology.csv",
        [
            {
                "source_unit": f"app-{index:03d}",
                "axis_containment_fraction": 1.0 if index else 0.998,
                "axis_in_root_coverage_fraction": 1.0 if index else 0.998,
                "axis_single_component_coverage_fraction": 0.995 if index else 0.990,
                "longest_unsupported_axis_gap_um": 5.0,
                "root_mask_component_count": 1,
                "axis_support_component_label": 1,
                "unsupported_attachment_n": 0,
                "identity_hair_n": 10,
            }
            for index in range(261)
        ],
    )
    trait_contract_payload = json.loads(
        resources["trait_contract"].read_text(encoding="utf-8")
    )
    canonical_fields = [
        record["field"]
        for family in ("primary_root_traits", "root_hair_traits")
        for record in trait_contract_payload[family]
    ]

    def _trait_rows(count: int) -> list[dict]:
        rows: list[dict] = []
        for index in range(count):
            row = {
                "task_id": f"app-{index:03d}",
                "source_image_sha256": figures.sha256_json(
                    {"source-image": index}
                ),
                "experiment_key": "D15_8d",
                "study_role": "rhd6_factorial_8d_primary",
                "condition_code": figures.GROUP_ORDER[index % 4],
                "formal_statistics_eligible": True,
            }
            for field_index, field in enumerate(canonical_fields):
                row[field] = float(1 + field_index + index / 100.0)
            rows.append(row)
        return rows

    clean_trait_rows = _trait_rows(261)
    full_trait_rows = _trait_rows(283)
    clean_traits_path = _save_csv(root / "clean-traits.csv", clean_trait_rows)
    full_traits_path = _save_csv(root / "full-traits.csv", full_trait_rows)
    trait_rows_by_cohort = {
        "primary_clean261": clean_trait_rows,
        "sensitivity_full283": full_trait_rows,
    }
    analysis_paths: dict[str, Path] = {}
    phenotype_effect_rows: list[dict] = []
    for cohort, count, offset, filename in (
        ("primary_clean261", 261, 0.0, "analysis-primary.csv"),
        ("sensitivity_full283", 283, 0.02, "analysis-sensitivity.csv"),
    ):
        analysis_rows: list[dict] = []
        cohort_traits = pd.DataFrame(trait_rows_by_cohort[cohort])
        h11_medians = {
            condition: float(
                cohort_traits.loc[
                    cohort_traits["condition_code"].astype(str).eq(condition),
                    "local_median_hair_length_um_1_4mm",
                ].median()
            )
            for condition in figures.GROUP_ORDER
        }
        ev22, ev30, oe22, oe30 = (
            h11_medians[condition] for condition in figures.GROUP_ORDER
        )
        h11_raw_effects = {
            "construct_OE_minus_EV": 0.5
            * ((oe22 - ev22) + (oe30 - ev30)),
            "temperature_30C_minus_22C": 0.5
            * ((ev30 - ev22) + (oe30 - oe22)),
            "construct_by_temperature_interaction": (oe30 - oe22)
            - (ev30 - ev22),
        }
        for endpoint_index, endpoint in enumerate(figures.PRIMARY_ENDPOINTS):
            for effect_index, (effect_name, effect_key) in enumerate(
                EFFECT_NAME_TO_KEY.items()
            ):
                value = 0.85 + endpoint_index * 0.05 + effect_index * 0.03 + offset
                is_h11 = endpoint == "local_median_hair_length_um_1_4mm"
                raw_estimate = (
                    h11_raw_effects[effect_name]
                    if is_h11
                    else (value - 1.0) * 100.0
                )
                raw_ci_low = raw_estimate - 5.0
                raw_ci_high = raw_estimate + 5.0
                raw_estimand = (
                    RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
                    if is_h11
                    else RAW_EFFECT_OLS_MEAN_CONTRAST
                )
                raw_interval_method = (
                    RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
                    if is_h11
                    else RAW_EFFECT_HC3_INTERVAL
                )
                raw_bootstrap_replicates = 5000 if is_h11 else 0
                raw_bootstrap_seed = (
                    raw_median_bootstrap_seed(
                        seed=20260823,
                        field="local_median_hair_length_um_1_4mm",
                        component="continuous",
                    )
                    if is_h11
                    else None
                )
                analysis_rows.append(
                    {
                        "cohort": cohort,
                        "endpoint": endpoint,
                        "model_component": (
                            "count_rate"
                            if endpoint == figures.PRIMARY_ENDPOINTS[0]
                            else "continuous"
                        ),
                        "effect": effect_name,
                        "n": count,
                        "estimate": value,
                        "ci95_low": value - 0.08,
                        "ci95_high": value + 0.08,
                        "effect_scale": "ratio",
                        "raw_effect_estimate": raw_estimate,
                        "raw_effect_ci95_low": raw_ci_low,
                        "raw_effect_ci95_high": raw_ci_high,
                        "raw_effect_estimand": raw_estimand,
                        "raw_effect_interval_method": raw_interval_method,
                        "raw_effect_bootstrap_replicates": raw_bootstrap_replicates,
                        "raw_effect_bootstrap_seed": raw_bootstrap_seed,
                        "standardized_effect": raw_estimate / 50.0,
                        "standardized_ci95_low": raw_ci_low / 50.0,
                        "standardized_ci95_high": raw_ci_high / 50.0,
                        "causal_treatment_claim_allowed": False,
                    }
                )
                phenotype_effect_rows.append(
                    {
                        "cohort": cohort,
                        "endpoint_key": endpoint,
                        "effect_key": effect_key,
                        "estimate": value,
                        "ci_low": value - 0.08,
                        "ci_high": value + 0.08,
                        "endpoint_n": count,
                        "effect_scale": "ratio",
                        "raw_effect_estimate": raw_estimate,
                        "raw_effect_ci_low": raw_ci_low,
                        "raw_effect_ci_high": raw_ci_high,
                        "raw_effect_unit": (
                            "count"
                            if endpoint == figures.PRIMARY_ENDPOINTS[0]
                            else "um"
                        ),
                        "raw_effect_estimand": raw_estimand,
                        "raw_effect_interval_method": raw_interval_method,
                        "raw_effect_bootstrap_replicates": raw_bootstrap_replicates,
                        "raw_effect_bootstrap_seed": raw_bootstrap_seed,
                        "standardized_effect": raw_estimate / 50.0,
                        "standardized_ci_low": raw_ci_low / 50.0,
                        "standardized_ci_high": raw_ci_high / 50.0,
                    }
                )
        analysis_paths[cohort] = _save_csv(root / filename, analysis_rows)
    analysis_payload["output_table_sha256"].update(
        {
            "primary_tests": sha256_file(analysis_paths["primary_clean261"]),
            "sensitivity_tests": sha256_file(
                analysis_paths["sensitivity_full283"]
            ),
        }
    )
    _write_json(receipts["analysis"], analysis_payload)
    resources["phenotype_effects"] = _save_csv(
        root / "phenotype-effects.csv", phenotype_effect_rows
    )
    narrative_decision = build_narrative_decision(
        phenotype_effect_rows,
        source_sha256={
            "phenotype_effects": sha256_file(resources["phenotype_effects"])
        },
    )
    resources["narrative_decision"] = _write_json(
        root / "narrative-decision.json",
        narrative_decision,
    )
    atlas_source_hashes = {
        "trait_contract": sha256_file(resources["trait_contract"]),
        "clean_traits": sha256_file(clean_traits_path),
        "full_traits": sha256_file(full_traits_path),
        "canonical_image_traits": sha256_file(full_traits_path),
        "analysis_primary_table": sha256_file(
            analysis_paths["primary_clean261"]
        ),
        "analysis_sensitivity_table": sha256_file(
            analysis_paths["sensitivity_full283"]
        ),
    }
    atlas_payload = build_multitrait_atlas(
        trait_contract=trait_contract_payload,
        clean_traits=pd.read_csv(clean_traits_path),
        full_traits=pd.read_csv(full_traits_path),
        canonical_image_traits=pd.read_csv(full_traits_path),
        primary_analysis=pd.read_csv(analysis_paths["primary_clean261"]),
        sensitivity_analysis=pd.read_csv(
            analysis_paths["sensitivity_full283"]
        ),
        source_sha256=atlas_source_hashes,
    )
    resources["multitrait_atlas"] = _write_json(
        root / "multitrait-atlas.json", atlas_payload
    )
    assert set(resources) == set(figures.RESOURCE_ROLES)
    public_identity = _test_public_identity()
    stageb_binding = _test_stageb_binding()
    proposal_payload = {
        "schema_version": "PHAxis-model-contract-1.0.0",
        "formal_release_status": "passed_proposal_not_official",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "promotion": {
            "schema_version": "PHAxis-model-contract-promotion-1.0",
            "status": "validated_proposal_not_applied",
            "official_apply_performed": False,
            "stageb_binding": stageb_binding,
            "formal_gate_identity_sha256": {
                "root_exact283_audit_identity_sha256": _test_root_audit_identity(),
            },
        },
        "model_bundle_id": public_identity["model_bundle_id"],
        "public_system_identity": {
            "schema_version": PUBLIC_SYSTEM_IDENTITY_SCHEMA,
            "identity_sha256": public_identity["public_system_identity_sha256"],
            "derivation": PUBLIC_SYSTEM_DERIVATION,
        },
        "expert_boundary": {
            "root_point_scale_continuity_statistics": public_identity[
                "root_expert_id"
            ],
        },
        "root_expert": {
            "provider_role": public_identity["root_provider_role"],
            "expert_id": public_identity["root_expert_id"],
            "fresh_exact283_audit_identity_sha256": _test_root_audit_identity(),
            "bundle_identity_sha256": _test_root_bundle_identity(),
            "pipeline_identity_sha256": _test_root_pipeline_identity(),
            "root_bundle_authority": {
                "bundle_identity_sha256": _test_root_bundle_identity(),
                "pipeline_identity_sha256": _test_root_pipeline_identity(),
            },
            "root_cap_region_output": False,
        },
        "red_lines": {"blind_images_used": 0},
    }
    proposal_payload["model_contract_identity_sha256"] = figures.sha256_json(
        proposal_payload
    )
    proposal = _write_json(root / "model-contract-proposal.json", proposal_payload)
    source_hashes = {role: sha256_file(path) for role, path in receipts.items()}
    provenance_specs = {
        "historical_development": (
            "historical_development_identity_sha256",
            {
                "schema_version": "PHAxis-historical-OOF443-development-receipt-1.0",
                "status": "completed_locked_historical_oof443_development",
            },
        ),
        "measurement_assurance": (
            "measurement_assurance_identity_sha256",
            {
                "schema_version": "PHAxis-measurement-assurance-receipt-1.0",
                "status": "completed_locked_qc_development_assurance",
                "scope": "QC-development measurement assurance; non-independent",
                "independent_accuracy_claim_allowed": False,
                "source_table_sha256": {
                    "metrics": sha256_file(resources["assurance_metrics"]),
                    "pairs": sha256_file(resources["assurance_pairs"]),
                    "support": sha256_file(resources["assurance_support"]),
                    "topology": sha256_file(assurance_topology),
                },
            },
        ),
        "overlay_index": (
            "overlay_selection_identity_sha256",
            {
                "schema_version": "PHAxis-manuscript-overlay-selection-receipt-1.2",
                "status": "completed_locked_preselected_gallery_and_exact_cohort_review_export",
                "case_plan_columns": ["case_role", "task_id"],
                "case_selection_basis": figures.OVERLAY_CASE_SELECTION_BASIS,
                "random_or_representative_performance_sample": False,
                "experimental_condition_metadata_used_for_rendering": False,
                "experimental_condition_metadata_used_for_evidence_assembly": False,
                "experimental_condition_metadata_used_for_evidence_assembly_scope": (
                    figures.OVERLAY_CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
                ),
                "experimental_condition_metadata_used_for_output_organization": True,
                "exact_cohort_review_images": 283,
                "paper_overlay_bytes_reused_from_full_cohort_review_export": True,
                "paper_overlay_sha256_matches_full_cohort_review_export": True,
                "source_authority_sha256": {
                    "case_plan": "1" * 64,
                    "application_manifest": "2" * 64,
                    "full_traits": "3" * 64,
                    "fusion_summary": "4" * 64,
                    "overlay_builder_source": "5" * 64,
                    "renderer_source": "6" * 64,
                },
                "full_cohort_review_export": {
                    "schema_version": "PHAxis-exact-cohort-review-overlay-export-1.0",
                    "status": "completed_exact_cohort_final_fusion_review_export",
                    "expected_task_count": 283,
                    "images": 283,
                    "index_rows": 283,
                    "checklist_rows": 283,
                    "review_root": "full283_review_overlays",
                    "index_csv": "full283_review_index.csv",
                    "index_csv_sha256": "7" * 64,
                    "checklist_csv": "full283_review_checklist.csv",
                    "checklist_csv_sha256": "8" * 64,
                    "readme_cn": "README_CN.md",
                    "readme_cn_sha256": "9" * 64,
                    "summary_json": "full283_review_summary.json",
                    "summary_json_sha256": "a" * 64,
                    "review_export_identity_sha256": "b" * 64,
                    "ordered_task_set_identity_sha256": "c" * 64,
                    "overlay_png_set_identity_sha256": "d" * 64,
                    "review_status_on_export": "pending_manual_visual_review",
                    "organization_fields": [
                        "experiment_key",
                        "condition_code",
                        "formal_statistics_eligible",
                    ],
                    "experimental_condition_metadata_used_for_prediction": False,
                    "experimental_condition_metadata_used_for_rendering": False,
                    "experimental_condition_metadata_used_for_evidence_assembly": False,
                    "experimental_condition_metadata_used_for_evidence_assembly_scope": (
                        figures.OVERLAY_CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
                    ),
                    "experimental_condition_metadata_used_for_output_organization": True,
                    "create_only": True,
                    "canonical_annotations_read": False,
                    "root_cap_region_statistics_included": False,
                    "blind_images_used": 0,
                },
                "inset_contract": {
                    "roles": ["low_contrast", "curved_dense"],
                    "locked_anchor_task_ids": figures.FIGURE4_LOCKED_ANCHOR_TASK_IDS,
                    "source_and_overlay_use_identical_crop_coordinates": True,
                    "whole_image_context_retained": True,
                    "performance_based_crop_selection": False,
                },
            },
        ),
        "profile_analysis": (
            "analysis_identity_sha256",
            {"schema_version": "PHAxis-distal-axis-profile-analysis-1.0.0", "status": "completed_exploratory_source_unit_profile_summaries"},
        ),
        "runtime_latency": (
            "summary_identity_sha256",
            {"schema_version": "PHAxis-full-workflow-sequential-latency-benchmark-1.0", "status": "completed_direct_full283"},
        ),
        "runtime_production": (
            "summary_identity_sha256",
            {"schema_version": "PHAxis-full-workflow-production-batch-benchmark-1.0", "status": "completed_direct_full283"},
        ),
        "runtime_latency_comparison": (
            "comparison_identity_sha256",
            {"schema_version": "PHAxis-full-workflow-benchmark-comparison-1.0", "status": "comparable_direct_full283"},
        ),
        "runtime_production_comparison": (
            "comparison_identity_sha256",
            {"schema_version": "PHAxis-full-workflow-benchmark-comparison-1.0", "status": "comparable_direct_full283"},
        ),
        "baseline_runtime_latency": (
            "summary_identity_sha256",
            {"schema_version": "PHAxis-full-workflow-sequential-latency-benchmark-1.0", "status": "completed_direct_full283"},
        ),
        "baseline_runtime_production": (
            "summary_identity_sha256",
            {"schema_version": "PHAxis-full-workflow-production-batch-benchmark-1.0", "status": "completed_direct_full283"},
        ),
    }
    provenance = {}
    for role, (field, payload) in provenance_specs.items():
        payload = {
            **payload,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        }
        payload[field] = figures.sha256_json(payload)
        path = _write_json(root / f"{role}.json", payload)
        provenance[role] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "identity_field": field,
            "identity_sha256": payload[field],
        }
    source_inputs = {
        role: {"path": path.name, "sha256": sha256_file(path)}
        for role, path in (
            ("assurance_metrics", resources["assurance_metrics"]),
            ("assurance_pairs", resources["assurance_pairs"]),
            ("assurance_support", resources["assurance_support"]),
            ("assurance_topology", assurance_topology),
            ("clean_traits", clean_traits_path),
            ("full_traits", full_traits_path),
            ("full_image_traits", full_traits_path),
            (
                "analysis_primary_table",
                analysis_paths["primary_clean261"],
            ),
            (
                "analysis_sensitivity_table",
                analysis_paths["sensitivity_full283"],
            ),
            *(
                (role, resources[role])
                for role in figures.WT_SECONDARY_RESOURCE_ROLES
            ),
        )
    }
    supplementary_sources = _supplementary_source_fixture(
        root / "supplementary-table-authorities"
    )
    runtime_frame = pd.read_csv(resources["runtime_per_image"])
    ordered_runtime_identity = figures.sha256_json(
        runtime_frame["source_unit"].astype(str).tolist()
    )
    pd.DataFrame(
        {
            "source_unit": runtime_frame["source_unit"],
            "wall_seconds": runtime_frame["wall_seconds"] * 2.0,
        }
    ).to_csv(
        supplementary_sources["source/baseline_runtime_per_image"],
        index=False,
        lineterminator="\n",
    )
    runtime_summary_payload = json.loads(
        resources["runtime_summary"].read_text(encoding="utf-8")
    )
    runtime_summary_payload["source_unit_ordered_set_identity_sha256"] = (
        ordered_runtime_identity
    )
    _write_json(resources["runtime_summary"], runtime_summary_payload)
    same_hardware_path = supplementary_sources["source/benchmark_same_hardware"]
    same_hardware_payload = json.loads(same_hardware_path.read_text(encoding="utf-8"))
    same_hardware_payload["source_unit_ordered_set_identity_sha256"] = (
        ordered_runtime_identity
    )
    for run in same_hardware_payload["runs"]:
        run["source_unit_ordered_set_identity_sha256"] = ordered_runtime_identity
    same_hardware_payload.pop("receipt_identity_sha256", None)
    same_hardware_payload["receipt_identity_sha256"] = figures.sha256_json(
        same_hardware_payload
    )
    _write_json(same_hardware_path, same_hardware_payload)
    fallback_source_paths = {
        "profile_analysis_table": analysis_paths["primary_clean261"],
        "sensitivity_profiles_summary": root / "profile_analysis.json",
        "runtime_latency": root / "runtime_latency.json",
        "runtime_production": root / "runtime_production.json",
        "runtime_per_image": resources["runtime_per_image"],
        "baseline_runtime_latency": root / "baseline_runtime_latency.json",
        "baseline_runtime_production": root / "baseline_runtime_production.json",
        "baseline_runtime_per_image": supplementary_sources[
            "source/baseline_runtime_per_image"
        ],
        "model_contract_proposal": proposal,
    }
    for role in figures.FIGURE_SOURCE_INPUT_ROLES:
        if role in source_inputs:
            continue
        source_path = supplementary_sources.get(f"source/{role}")
        if source_path is None:
            source_path = fallback_source_paths[role]
        source_inputs[role] = {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
        }
    assert set(source_inputs) == set(figures.FIGURE_SOURCE_INPUT_ROLES)
    evaluation = json.loads(
        receipts["train399_evaluation"].read_text(encoding="utf-8")
    )
    prediction_locks = evaluation["prediction_input_locks"]
    comparator = evaluation["comparator_contract"]["hybrid_max"]
    evaluation_authority = evaluation["evaluation_inference_authority"]
    wt_binding = figures.validate_wt_secondary_analysis_binding(
        analysis_summary=analysis_payload,
        evidence_summary=wt_contract,
        table_sha256={
            role: sha256_file(resources[role])
            for role in figures.WT_SECONDARY_RESOURCE_ROLES
        },
    )
    manifest_payload = {
        "schema_version": figures.INPUT_SCHEMA_VERSION,
        "assembler_schema_version": figures.ASSEMBLER_SCHEMA_VERSION,
        "status": status,
        "source_summary_sha256": source_hashes,
        "model_contract_proposal_sha256": sha256_file(proposal),
        "model_contract_proposal_identity_sha256": proposal_payload[
            "model_contract_identity_sha256"
        ],
        "model_contract_public_identity": {
            "model_bundle_id": proposal_payload["model_bundle_id"],
            "root_expert_id": proposal_payload["root_expert"]["expert_id"],
            "root_provider_role": proposal_payload["root_expert"][
                "provider_role"
            ],
        },
        "model_bundle_id": proposal_payload["model_bundle_id"],
        "root_expert_id": proposal_payload["root_expert"]["expert_id"],
        "hair_identity_expert_id": json.loads(
            receipts["stageb"].read_text(encoding="utf-8")
        )[
            "detection_model_metadata"
        ]["expert_id"],
        "narrative_decision_identity_sha256": narrative_decision[
            "narrative_decision_identity_sha256"
        ],
        "narrative_branch_id": narrative_decision["branch_id"],
        "wt_secondary_evidence": wt_binding,
        "resources": {
            role: {"path": path.name, "sha256": sha256_file(path)}
            for role, path in resources.items()
        },
        "resource_lineage": {role: ["synthetic_test_authority"] for role in resources},
        "source_inputs": source_inputs,
        "provenance_receipts": provenance,
        "train399_prediction_input_provenance": {
            "task_order_identity_sha256": figures.sha256_json(
                [row["task_id"] for row in evaluation["per_image"]]
            ),
            "stageb_train399": {
                "schema_version": figures.STAGEB_DETECTION_SCHEMA,
                "artifact_role": evaluation_authority["artifact_role"],
                "evaluation_inference_summary_sha256": evaluation_authority[
                    "evaluation_inference_summary_sha256"
                ],
                "evaluation_inference_summary_identity_sha256": (
                    evaluation_authority[
                        "evaluation_inference_summary_identity_sha256"
                    ]
                ),
                "evaluation_gate_identity_sha256": evaluation_authority[
                    "evaluation_gate_identity_sha256"
                ],
                "production_consumption_allowed": False,
                "fusion_consumption_allowed": False,
                "traits_consumption_allowed": False,
                "ordered_file_set_identity_sha256": prediction_locks[
                    "stageb_detection_set_identity_sha256"
                ],
            },
            "legacy_hybrid_endpoint_complete_identity_layer": {
                **comparator,
                "ordered_file_set_identity_sha256": prediction_locks[
                    "hybrid_prediction_set_identity_sha256"
                ],
            },
        },
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    supplementary_contract = figures.supplementary_figure_contract()
    supplementary_contract["contract_identity_sha256"] = figures.sha256_json(
        supplementary_contract
    )
    manifest_payload["supplementary_figure_contract"] = supplementary_contract
    manifest_payload["figure_input_assembly_identity_sha256"] = figures.sha256_json(
        manifest_payload
    )
    figure_inputs = _write_json(
        root / "figure-inputs.json",
        manifest_payload,
    )
    return receipts, figure_inputs, proposal


def _fast_bundle(figure, base_path: Path, *, width_mm: float, height_mm: float, **_) -> dict:
    files = {}
    hashes = {}
    for kind, suffix in (("pdf", ".pdf"), ("png", ".png"), ("tiff", ".tiff")):
        path = base_path.with_suffix(suffix)
        path.write_bytes(f"{kind}:{base_path.name}".encode())
        files[kind] = str(path.resolve())
        hashes[kind] = sha256_file(path)
    plt.close(figure)
    return {
        "width_mm": width_mm,
        "height_mm": height_mm,
        "files": files,
        "sha256": hashes,
        "png_pixels": [1, 1],
        "tiff_mode": "RGB",
        "edge_ink_pixels_outer_2px": 0,
    }


def test_final_six_figure_route_is_hash_closed(monkeypatch, tmp_path: Path) -> None:
    receipts, figure_inputs, proposal = _fixture(tmp_path / "final-inputs")
    monkeypatch.setattr(figures, "save_figure_bundle", _fast_bundle)
    output = tmp_path / "final-suite"
    summary = figures.build_figure_suite(
        mode="final",
        figure_inputs=figure_inputs,
        model_contract_proposal=proposal,
        output=output,
        receipt_paths=receipts,
    )

    assert summary["status"] == "final_sealed_strict_train399_only"
    assert summary["submission_use_allowed"] is True
    assert list(summary["figures"]) == list(figures.FIGURE_STEMS)
    assert [record["number"] for record in summary["figures"].values()] == [1, 2, 3, 4, 5, 6]
    assert summary["source_summary_sha256"] == {
        role: sha256_file(receipts[role]) for role in figures.RECEIPT_ROLES
    }
    assert summary["model_contract_proposal_sha256"] == sha256_file(proposal)
    assert len(summary["model_contract_proposal_identity_sha256"]) == 64
    expected_identity = figures.sha256_json(
        {
            "status": "final",
            "figure_hashes": summary["figure_bundle_sha256"],
            "source_hashes": summary["source_summary_sha256"],
            "figure_input_assembly_identity_sha256": summary[
                "figure_input_assembly_identity_sha256"
            ],
            "model_contract_proposal_identity_sha256": summary[
                "model_contract_proposal_identity_sha256"
            ],
            "model_contract_public_identity": summary[
                "model_contract_public_identity"
            ],
                "train399_prediction_input_provenance": summary[
                    "train399_prediction_input_provenance"
                ],
                "supplementary_table_bundle_identity_sha256": summary[
                    "supplementary_table_bundle_identity_sha256"
                ],
                "supplementary_table_bundle_receipt_sha256": summary[
                    "supplementary_table_bundle_receipt_sha256"
                ],
            }
        )
    assert summary["figure_suite_identity_sha256"] == expected_identity
    assert not any(path.name.startswith("PROVISIONAL_") for path in output.iterdir())
    assert (output / "figure_legends_and_alt_text.md").is_file()
    assert sha256_file(output / "source_hashes.json") == summary["source_hashes_manifest_sha256"]
    source_data_files = sorted(
        path for path in (output / "source_data").rglob("*") if path.is_file()
    )
    assert len(source_data_files) == 61
    assert any(
        path.name
        == "Figure_02_train399_development_evidence_assurance_metrics.csv"
        for path in source_data_files
    )
    assert tuple(
        figures._source_groups()[stem] for stem in figures.FIGURE_STEMS
    ) == figures.MAIN_FIGURE_RESOURCE_ROLES
    figure_source_maps = {
        **{
            stem: record["source_data_sha256"]
            for stem, record in summary["figures"].items()
        },
        **{
            stem: record["source_data_sha256"]
            for stem, record in summary["supplementary_figures"].items()
        },
    }
    assert sum(len(value) for value in figure_source_maps.values()) == 71
    declared_source_data: dict[str, str] = {}
    for source_map in figure_source_maps.values():
        for relative_path, digest in source_map.items():
            assert relative_path not in declared_source_data or (
                declared_source_data[relative_path] == digest
            )
            declared_source_data[relative_path] = digest
    observed_source_data = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in source_data_files
    }
    assert observed_source_data == declared_source_data
    assert observed_source_data == summary["physical_source_data_sha256"]
    assert figure_source_maps == summary["figure_source_data_sha256"]
    assert summary["source_data_identity_sha256"] == figures.sha256_json(
        {
            "figure_source_data_sha256": figure_source_maps,
            "physical_source_data_sha256": observed_source_data,
        }
    )
    figure1_source_map = figure_source_maps[figures.FIGURE_STEMS[0]]
    assert any("_figure1_image." in path for path in figure1_source_map)
    for stem in (figures.FIGURE_STEMS[3], figures.SUPPLEMENTARY_STEMS[5]):
        source_map = figure_source_maps[stem]
        copied_selection_path = next(
            output / path
            for path in source_map
            if "_overlay_selection." in path
        )
        copied_selection = pd.read_csv(copied_selection_path)
        for row in copied_selection.to_dict("records"):
            for prefix in ("source", "overlay"):
                relative = Path(str(row[f"{prefix}_path"]))
                assert not relative.is_absolute()
                copied_image = copied_selection_path.parent / relative
                assert copied_image.is_file()
                assert sha256_file(copied_image) == row[f"{prefix}_sha256"]
                assert copied_image.relative_to(output).as_posix() in source_map
    source_hash_manifest = json.loads(
        (output / "source_hashes.json").read_text(encoding="utf-8")
    )
    assert source_hash_manifest["figure_source_data_sha256"] == figure_source_maps
    assert source_hash_manifest["physical_source_data_sha256"] == observed_source_data
    assert source_hash_manifest["source_data_identity_sha256"] == summary[
        "source_data_identity_sha256"
    ]
    assert list(summary["supplementary_figures"]) == list(
        figures.SUPPLEMENTARY_STEMS
    )
    assert [record["number"] for record in summary["supplementary_figures"].values()] == [
        f"S{index}" for index in range(1, 10)
    ]
    assert all(
        record["status"] == "final"
        for record in summary["supplementary_figures"].values()
    )
    assert summary["claim_contract"]["main_figure_count"] == 6
    assert summary["claim_contract"]["supplementary_figure_count"] == 9
    assert summary["supplementary_figure_bundle_identity_sha256"] == figures.sha256_json(
        summary["supplementary_figure_bundle_sha256"]
    )
    assert summary["supplementary_figure_contract"]["ordered_figure_count"] == 9
    assert summary["supplementary_figure_contract_identity_sha256"] == figures.sha256_json(
        {
            key: value
            for key, value in summary["supplementary_figure_contract"].items()
            if key != "contract_identity_sha256"
        }
    )
    s9_contract = summary["supplementary_figure_contract"]["figures"][-1]
    assert set(s9_contract["resource_roles"]) == {
        "multitrait_atlas",
        *figures.WT_SECONDARY_RESOURCE_ROLES,
    }
    assert "block/day-stratified WT temperature secondary evidence" in s9_contract[
        "title"
    ]
    assert summary["wt_secondary_evidence"][
        "estimated_within_day_meta_rows"
    ] == 10
    assert summary["wt_secondary_evidence"][
        "typed_not_estimable_meta_rows"
    ] == 10
    assert summary["claim_contract"][
        "wt_secondary_alters_D15_fixed_effect_family"
    ] is False
    assert summary["claim_contract"]["wt_cross_day_pooling_performed"] is False
    assert summary["claim_contract"][
        "wt_unknown_day_meta_analysis_performed"
    ] is False

    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    original_resources = {
        role: figure_inputs.parent / record["path"]
        for role, record in manifest["resources"].items()
    }
    copied_resources: dict[str, Path] = {}
    for role in ("figure1_image", "figure1_geometry", "trait_contract"):
        copied_resources[role] = (
            output
            / "source_data"
            / f"{figures.FIGURE_STEMS[0]}_{role}{original_resources[role].suffix.lower()}"
        )
    for role in ("overlay_selection", "overlay_audit"):
        copied_resources[role] = (
            output
            / "source_data"
            / f"{figures.FIGURE_STEMS[3]}_{role}{original_resources[role].suffix.lower()}"
        )
    copied_s6_selection = (
        output
        / "source_data"
        / (
            f"{figures.SUPPLEMENTARY_STEMS[5]}_overlay_selection"
            f"{original_resources['overlay_selection'].suffix.lower()}"
        )
    )
    shutil.rmtree(figure_inputs.parent)
    replay_figure1 = figures._figure1(copied_resources, provisional=False)
    replay_figure4, replay_records = figures._figure4(
        copied_resources,
        provisional=False,
        final=True,
    )
    replay_s6 = figures._supplementary_s6(
        {"overlay_selection": copied_s6_selection},
        provisional=False,
    )
    try:
        replay_figure1.canvas.draw()
        replay_figure4.canvas.draw()
        replay_s6.canvas.draw()
        assert len(replay_records) == 5
    finally:
        plt.close(replay_figure1)
        plt.close(replay_figure4)
        plt.close(replay_s6)


def test_s9_wt_forest_draws_only_eligible_same_day_diamonds(
    tmp_path: Path,
) -> None:
    _receipts_by_role, figure_inputs, _proposal = _fixture(
        tmp_path / "s9-wt-forest-inputs"
    )
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    resources = {
        role: figure_inputs.parent / record["path"]
        for role, record in manifest["resources"].items()
    }
    figure = figures._supplementary_multitrait_atlas(
        resources, provisional=False
    )
    try:
        figure.canvas.draw()
        contract = getattr(figure, "_phaxis_wt_secondary_figure_contract")
        assert contract["known_day_panel"] == {
            "experiment_point_count": 50,
            "pooled_diamond_count": 10,
            "not_estimable_meta_rows_rendered": 10,
            "unknown_day": 0,
        }
        assert contract["unknown_day_panel"] == {
            "experiment_point_count": 10,
            "pooled_diamond_count": 0,
            "not_estimable_meta_rows_rendered": 0,
            "unknown_day": 1,
        }
        assert contract["D15_fixed_effect_family_changed"] is False
        assert contract["unknown_day_is_descriptive_only"] is True
        known_text = " ".join(
            item.get_text() for item in figure.axes[4].texts
        )
        assert "Not estimable" in known_text
        assert "k=2" in known_text
        assert "never pooled" in figure.axes[5].get_title(loc="left")
    finally:
        plt.close(figure)


def test_final_overlay_rows_enforce_preselected_nonperformance_contract(
    tmp_path: Path,
) -> None:
    _, figure_inputs, _ = _fixture(tmp_path / "overlay-contract-inputs")
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    resources = {
        role: figure_inputs.parent / record["path"]
        for role, record in manifest["resources"].items()
    }
    overlay_path = figure_inputs.parent / manifest["resources"]["overlay_selection"][
        "path"
    ]
    selection = pd.read_csv(overlay_path)
    records = figures._verify_overlay_rows(selection, overlay_path.parent, final=True)
    assert len(records) == len(figures.CASE_ROLES) == 5

    wrong_basis = selection.copy()
    wrong_basis.loc[0, "case_selection_basis"] = "representative_performance_sample"
    with pytest.raises(figures.FigureSuiteError, match="case-selection basis changed"):
        figures._verify_overlay_rows(wrong_basis, overlay_path.parent, final=True)

    for field in (
        "random_or_representative_performance_sample",
        "experimental_condition_metadata_used_for_rendering",
        "experimental_condition_metadata_used_for_evidence_assembly",
    ):
        unsafe = selection.copy()
        unsafe.loc[0, field] = True
        with pytest.raises(figures.FigureSuiteError, match=field):
            figures._verify_overlay_rows(unsafe, overlay_path.parent, final=True)

    wrong_scope = selection.copy()
    wrong_scope.loc[0, "experimental_condition_metadata_used_for_evidence_assembly_scope"] = (
        "all_evidence_and_output_organization"
    )
    with pytest.raises(figures.FigureSuiteError, match="evidence-assembly scope changed"):
        figures._verify_overlay_rows(wrong_scope, overlay_path.parent, final=True)

    attacks: list[tuple[str, pd.DataFrame, str]] = []
    absolute = selection.copy()
    absolute.loc[0, "source_path"] = str(
        (overlay_path.parent / selection.loc[0, "source_path"]).resolve()
    )
    attacks.append(("absolute", absolute, "absolute source source-data path"))

    traversal = selection.copy()
    traversal.loc[0, "source_path"] = "../outside.png"
    attacks.append(("traversal", traversal, "non-portable source source-data path"))

    tampered = selection.copy()
    tampered.loc[0, "source_sha256"] = "0" * 64
    attacks.append(("tampered", tampered, "source source-data image hash mismatch"))

    blind_source = overlay_path.parent / "blind-case.png"
    shutil.copyfile(
        overlay_path.parent / selection.loc[0, "source_path"],
        blind_source,
    )
    blind = selection.copy()
    blind.loc[0, "source_path"] = blind_source.name
    blind.loc[0, "source_sha256"] = sha256_file(blind_source)
    attacks.append(("blind", blind, "blind-labelled source source-data path refused"))

    source_target = overlay_path.parent / selection.loc[0, "source_path"]
    symlink_path = overlay_path.parent / "source-data-symlink.png"
    try:
        symlink_path.symlink_to(source_target)
    except OSError:
        pass
    else:
        symlinked = selection.copy()
        symlinked.loc[0, "source_path"] = symlink_path.name
        symlinked.loc[0, "source_sha256"] = sha256_file(source_target)
        attacks.append(("symlink", symlinked, "symlink source source-data path is forbidden"))

    for name, attacked_selection, message in attacks:
        attacked_path = overlay_path.parent / f"overlay-selection-{name}.csv"
        attacked_selection.to_csv(attacked_path, index=False, lineterminator="\n")
        attacked_resources = dict(resources)
        attacked_resources["overlay_selection"] = attacked_path
        with pytest.raises(figures.FigureSuiteError, match=message):
            figures._copy_source_data(
                tmp_path / f"source-data-{name}",
                attacked_resources,
            )


def test_provisional_route_marks_every_figure_and_forbids_submission(monkeypatch, tmp_path: Path) -> None:
    receipts, figure_inputs, proposal = _fixture(tmp_path / "layout-inputs", status="provisional")
    monkeypatch.setattr(figures, "save_figure_bundle", _fast_bundle)
    output = tmp_path / "layout-suite"
    summary = figures.build_figure_suite(
        mode="provisional",
        figure_inputs=figure_inputs,
        model_contract_proposal=proposal,
        output=output,
        receipt_paths=receipts,
    )

    assert summary["status"] == "provisional_not_for_submission"
    assert summary["submission_use_allowed"] is False
    assert summary["deployment_figures_provisional"] is True
    assert all(record["status"] == "provisional_not_for_submission" for record in summary["figures"].values())
    assert len(list(output.glob("PROVISIONAL_Figure_*.pdf"))) == 6
    assert len(list(output.glob("PROVISIONAL_Supplementary_Figure_*.pdf"))) == 9
    assert all(
        record["status"] == "provisional_not_for_submission"
        for record in summary["supplementary_figures"].values()
    )
    assert "PROVISIONAL — NOT FOR SUBMISSION" in (output / "figure_legends_and_alt_text.md").read_text(encoding="utf-8")


def test_final_mode_fails_closed_on_provisional_or_unbound_inputs(monkeypatch, tmp_path: Path) -> None:
    receipts, figure_inputs, proposal = _fixture(tmp_path / "blocked-inputs", status="provisional")
    monkeypatch.setattr(figures, "save_figure_bundle", _fast_bundle)
    with pytest.raises(figures.FigureSuiteError):
        figures.build_figure_suite(
            mode="final",
            figure_inputs=figure_inputs,
            model_contract_proposal=proposal,
            output=tmp_path / "must-not-exist",
            receipt_paths=receipts,
        )
    assert not (tmp_path / "must-not-exist").exists()

    payload = json.loads(figure_inputs.read_text(encoding="utf-8"))
    payload["status"] = "final"
    payload["source_summary_sha256"]["stageb"] = "0" * 64
    payload.pop("figure_input_assembly_identity_sha256")
    payload["figure_input_assembly_identity_sha256"] = figures.sha256_json(payload)
    figure_inputs.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(figures.FigureSuiteError, match="exact eight named receipts"):
        figures.build_figure_suite(
            mode="final",
            figure_inputs=figure_inputs,
            model_contract_proposal=proposal,
            output=tmp_path / "still-must-not-exist",
            receipt_paths=receipts,
        )


def test_builder_has_only_the_final_six_figure_architecture() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "rhaxis_nextgen" not in source.casefold()
    assert "47-unit" not in source
    assert "260 formal" not in source
    assert "Figure_01_development" not in source
    assert len(figures.FIGURE_STEMS) == 6
    assert figures.FIGURE_STEMS[0].endswith("biological_measurement_design")
    assert figures.FIGURE_STEMS[5].endswith("reproducibility_and_efficiency")
    assert len(figures.SUPPLEMENTARY_STEMS) == 9
    assert figures.SUPPLEMENTARY_STEMS[0].endswith(
        "stageb_input_architecture_targets"
    )
    assert figures.SUPPLEMENTARY_STEM.endswith(
        "multitrait_atlas_coverage_effect_heatmap"
    )
    assert 'mkdtemp(prefix=".figures-"' in source
    s2_source = inspect.getsource(figures._supplementary_s2)
    assert 'label="Biological-presence F1@20 µm"' in s2_source
    assert 'label="Attachment F1@20 µm"' not in s2_source


def test_main_figure_legends_use_the_locked_biological_terminology() -> None:
    decision = _test_narrative_decision()
    legends = figures._legends_and_alt_text(
        provisional=False,
        runtime={
            "latency_mode": "sequential_persistent_full283",
            "measurement_scope": "raw_image_to_final_traits_and_profiles_direct",
        },
        decision=decision,
    )
    assert (
        "## Figure 1. PHAxis anchors visible-hair population, supported "
        "morphology and primary-root form to one physical axis"
    ) in legends
    assert "five measurement families" in legends
    for family in (
        "visible-hair abundance",
        "conditional projected length",
        "axial deployment",
        "visible-root extent",
        "root form/trajectory",
    ):
        assert family in legends
    assert (
        "## Figure 2. PHAxis recovers visible root-hair populations at "
        "individual-hair resolution"
    ) in legends
    assert (
        "## Figure 3. Continuity, calibration and conditional geometry make "
        "organ-anchored traits physically interpretable"
    ) in legends
    assert "absence specificity is not estimable" in legends
    assert "fail-closed behaviour is a software contract" in legends
    assert (
        "## Figure 4. PHAxis exposes interpretable measurement support across "
        "challenging image contexts"
    ) in legends
    assert "preselected acquisition-challenge roles" in legends
    assert "RHSCU-aa5b6e37df15821f" in legends
    assert "RHSCU-bbf649822174e0a2" in legends
    assert "Condition-blinded illustrative cases" not in legends
    assert "These morphology-driven illustrations are not a performance sample" in legends
    assert "only after pixels were fixed" in legends
    assert "orange point on an amber vector marks only the Stage-B vector terminus" in legends
    assert "only the terminus of a green one-to-one matched curve" in legends
    assert f"## Figure 5. {figures.title_contract(decision)['figures']['5']}" in legends
    assert "(a) Clean-cohort D15 source-unit observations" in legends
    assert "(b) The fixed 15-effect family" in legends
    assert "(c) Visible-hair abundance" in legends
    assert "(d) Length-support fraction" in legends
    assert "(e) Per-image conditional projected length" in legends
    assert "Exact condition-specific denominators remain in the source-data table" in legends
    assert "distal profiles never select or veto it" in legends
    assert "complete four-condition map and all 32 descriptors" in legends
    assert (
        "## Figure 6. PHAxis carries raw images to a reusable, benchmarked "
        "root-hair phenotype atlas"
    ) in legends
    assert "`hybrid_verified_increment`" not in legends
    assert "crop locator" not in legends.casefold()
    assert "yellow an attachment" not in legends.casefold()
    figure6_alt = legends.split("## Figure 6.", 1)[1].split(
        "## Supplementary Figure S1.", 1
    )[0]
    assert "utilization" not in figure6_alt.casefold()
    assert "memory" not in figure6_alt.casefold()
    assert "does not imply that later release or clean-install gates have passed" in figure6_alt
    assert "An external laboratory supplies a raw image and calibration manifest" in figure6_alt
    assert "32 image-level descriptors" in figure6_alt
    assert "declared reuse contract" in figure6_alt
    assert "both panels in every pair carry the same calibrated physical scale bar" in legends
    assert (
        "## Supplementary Figure S7. Clean-cohort D15 analysis, full-cohort "
        "D15 sensitivity, and observability"
    ) in legends
    assert (
        "Only panels (a) and (b) compare the clean-cohort D15 primary analysis "
        "with full-cohort D15 sensitivity"
    ) in legends
    assert (
        "## Supplementary Figure S9. Clean-cohort D15 32-descriptor phenotype "
        "map and block/day-stratified WT temperature secondary evidence"
    ) in legends
    s7_only = legends.split("## Supplementary Figure S7.", 1)[1].split(
        "## Supplementary Figure S8.", 1
    )[0]
    assert "Clean261" not in s7_only
    assert "Full283" not in s7_only
    s9_only = legends.split("## Supplementary Figure S9.", 1)[1]
    assert "developmental-day-specific random-effects REML/Hartung–Knapp diamonds" in s9_only
    assert "typed `Not estimable` label" in s9_only
    assert "unknown developmental day remain descriptive" in s9_only
    assert "never pooled" in s9_only
    assert (
        "## Supplementary Figure S3. Identity, formal attachment, endpoint, "
        "and conditional-length assurance"
    ) in legends
    assert (
        "## Supplementary Figure S5. Root-provider equivalence, same-component "
        "root continuity, formal attachment, and tiled-inference assurance"
    ) in legends
    assert (
        "Hair-curve trajectory continuity, distinct from primary-root "
        "connected-component continuity"
    ) in legends
    assert "Base-only proxies are excluded from panel f" in legends
    assert "complete formal continuity family" in legends
    assert "complete formal attachment family" in legends
    for deprecated in (
        "conditional elongation",
        "root-growth extent",
        "five plant-response families",
        "PHAxis preserves biological interpretability",
        "Exploratory RHD6 × temperature phenotype atlas",
        "Cohort integrity, fail-closed flow",
    ):
        assert deprecated not in legends


def test_figure1_renders_exactly_five_measurement_families(tmp_path: Path) -> None:
    _receipts_by_role, figure_inputs, _proposal = _fixture(
        tmp_path / "figure1-terminology"
    )
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    resources = {
        role: figure_inputs.parent / record["path"]
        for role, record in manifest["resources"].items()
    }
    figure = figures._figure1(resources, provisional=False)
    try:
        text = {
            item.get_text()
            for axis in figure.axes
            for item in axis.texts
        }
        assert {
            "H08 / N — visible population",
            "H11 / L — supported morphology",
            "H07 / F — deployment boundary",
            "R07 / W — carrying-root calibre",
            "R01 / A — visible organ extent",
        }.issubset(text)
        assert (
            sum(
                axis.get_title(loc="left") == "Measurement families"
                for axis in figure.axes
            )
            == 1
        )
        assert "Conditional elongation" not in text
        assert "Root-growth extent" not in text
    finally:
        plt.close(figure)


def test_supplementary_s1_uses_production_three_channel_input_and_all_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.tile(np.arange(48, dtype=np.uint8), (32, 1))
    image = np.stack((image, image, image), axis=-1)
    image_path = tmp_path / "s1-input.png"
    Image.fromarray(image).save(image_path)
    geometry_path = _write_json(
        tmp_path / "s1-geometry.json",
        {"scale_bar": {"pixels": 20, "micrometres": 80}},
    )
    resources = {
        "figure1_image": image_path,
        "figure1_geometry": geometry_path,
    }
    receipts = {
        "stageb": {
            "status": "completed",
            "detection_model_metadata": {
                "expert_id": "sealed-test-expert",
                "ensemble_members": 5,
            },
            "summary_identity_sha256": figures.sha256_json("s1-test"),
        }
    }

    production_make_input_channels = stageb_preprocess.make_input_channels
    assert figures.make_input_channels is production_make_input_channels
    calls: list[dict[str, object]] = []

    def recording_make_input_channels(
        gray: np.ndarray, um_per_px: float, n_channels: int = 3
    ) -> np.ndarray:
        channels = production_make_input_channels(
            gray, um_per_px, n_channels=n_channels
        )
        calls.append(
            {
                "um_per_px": um_per_px,
                "n_channels": n_channels,
                "output_shape": channels.shape,
            }
        )
        return channels

    monkeypatch.setattr(figures, "make_input_channels", recording_make_input_channels)
    figure = figures._supplementary_s1(resources, receipts, provisional=False)
    try:
        assert len(calls) == 1
        assert calls[0]["um_per_px"] == pytest.approx(2.0)
        assert calls[0]["n_channels"] == 3
        assert calls[0]["output_shape"][0] == 3

        rendered_text = {
            item.get_text()
            for axis in figure.axes
            for item in axis.texts
        }
        expected_head_labels = {
            "base heatmap",
            "base offset",
            "base direction",
            "base length",
            "tip heatmap",
            "tip offset",
            "line support",
            "local flow",
            "root support",
        }
        assert expected_head_labels <= rendered_text
    finally:
        plt.close(figure)


def test_figure5_panel_order_and_profile_denominator_annotations(
    tmp_path: Path,
) -> None:
    _receipts_by_role, figure_inputs, _proposal = _fixture(
        tmp_path / "figure5-panel-contract"
    )
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    resources = {
        role: figure_inputs.parent / record["path"]
        for role, record in manifest["resources"].items()
    }
    figure = figures._figure5(resources, provisional=False, final=True)
    try:
        by_ylabel = {axis.get_ylabel(): axis for axis in figure.axes}
        abundance = by_ylabel["Hair identities per bin"]
        support = by_ylabel["Length-support fraction"]
        length = by_ylabel["Conditional projected length (µm)"]
        effect_axes = [
            axis
            for axis in figure.axes
            if [item.get_text() for item in axis.get_yticklabels()]
            == ["Construct", "Temperature", "Interaction"]
        ]
        assert len(effect_axes) == 5
        assert sum(len(axis.get_yticklabels()) for axis in effect_axes) == 15
        assert all(
            axis.get_title(loc="left").startswith("n clean/full=")
            for axis in effect_axes
        )
        assert (
            sum(
                axis.get_xlabel()
                == "Ratio (log scale; 95% model-based interval)"
                for axis in effect_axes
            )
            == 1
        )
        assert all(axis.get_xscale() == "log" for axis in effect_axes)
        assert all(
            np.allclose(effect_axes[0].get_xlim(), axis.get_xlim())
            for axis in effect_axes[1:]
        )
        raw_titles = [axis.get_title(loc="left") for axis in figure.axes[:5]]
        assert [title.splitlines()[0] for title in raw_titles] == [
            "N · H08",
            "L · H11",
            "F · H07",
            "W · R07",
            "A · R01",
        ]
        h11_ticks = [item.get_text() for item in figure.axes[1].get_xticklabels()]
        h07_ticks = [item.get_text() for item in figure.axes[2].get_xticklabels()]
        assert all("\nn " in text and "\nL " in text for text in h11_ticks)
        assert all("\nF " in text for text in h07_ticks)
        assert "(c)" in {item.get_text() for item in abundance.texts}
        assert "(d)" in {item.get_text() for item in support.texts}
        assert "(e)" in {item.get_text() for item in length.texts}

        abundance_n = [
            item.get_text()
            for item in abundance.texts
            if item.get_gid() == "profile-denominator"
        ]
        support_n = [
            item.get_text()
            for item in support.texts
            if item.get_gid() == "profile-denominator"
        ]
        length_n = [
            item.get_text()
            for item in length.texts
            if item.get_gid() == "profile-denominator"
        ]
        assert len(abundance_n) == len(support_n) == len(length_n) == 5
        assert set(abundance_n) == {"15–18"}
        assert set(support_n) == {"15–18\n10–13"}
        assert set(length_n) == set(support_n)
        assert [
            item.get_text()
            for item in abundance.texts
            if item.get_gid() == "profile-denominator-header"
        ] == ["n"]
        for axis in (support, length):
            assert [
                item.get_text()
                for item in axis.texts
                if item.get_gid() == "profile-denominator-header"
            ] == ["n\nL"]
        for axis in (abundance, support, length):
            header = next(
                item
                for item in axis.texts
                if item.get_gid() == "profile-denominator-header"
            )
            assert header.get_position()[0] < 0.0
            assert header.get_horizontalalignment() == "right"
        assert not figure.texts, "Figure 5 explanatory prose belongs in the caption"
    finally:
        plt.close(figure)


def test_figure5_submission_size_readability_and_s9_delegation(
    tmp_path: Path,
) -> None:
    _, figure_inputs, _ = _fixture(tmp_path / "figure5-submission-qa")
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    resources = {
        role: figure_inputs.parent / record["path"]
        for role, record in manifest["resources"].items()
    }
    figure = figures._figure5(resources, provisional=False, final=True)
    try:
        report = figure._phaxis_submission_readability_qa
        assert report["status"] == "pass_submission_size_readability"
        assert report["width_mm"] == 178.0
        assert report["height_mm"] == 148.0
        assert report["minimum_text_pt"] >= 6.0
        assert report["minimum_required_text_pt"] == 6.0
        assert report["minimum_data_symbol_diameter_pt"] >= 6.0
        assert report["minimum_required_data_symbol_diameter_pt"] == 6.0
        assert report["minimum_data_symbol_diameter_300dpi_px"] >= 25.0
        assert report["minimum_required_data_symbol_diameter_300dpi_px"] == 25.0
        assert report["panel_labels"] == ["(a)", "(b)", "(c)", "(d)", "(e)"]
        assert report["category_coloured_text_count"] == 0
        assert report["outside_canvas_text_count"] == 0
        assert report["minimum_text_canvas_inset_px_at_100dpi"] >= 3.0
        assert report["overlapping_tick_pair_count"] == 0
        assert report["overlapping_profile_annotation_pair_count"] == 0
        assert (
            report["profile_annotation_minimum_marker_clearance_px_at_100dpi"]
            >= report["profile_annotation_required_data_clearance_px_at_100dpi"]
        )
        assert report["profile_annotation_marker_collision_count"] == 0
        assert report["profile_annotation_line_collision_count"] == 0
        assert report["text_budget_pass"] is True
        assert report["endpoint_panels"] == 5
        assert report["effect_slots"] == 15
        assert report["effect_axis_scale"] == "log2_ratio"
        assert report["sentinel_badges"] == [
            "N · H08",
            "L · H11",
            "F · H07",
            "W · R07",
            "A · R01",
        ]
        assert report["h11_support_annotation"].startswith(
            "n non-null/formal"
        )
        assert report["h11_tick_text_budget_exception"] == (
            "three_line_semantic_denominator_labels_pass_canvas_overlap_and_font_gates"
        )
        assert report["h07_observability_annotation"] == (
            "F observable/formal source roots"
        )
        assert report["profile_panels"] == 3
        assert report["main_figure_descriptor_heatmap_rows"] == 0
        assert report["complete_32_descriptor_heatmap_figure"] == "Figure S9"
        assert len(figure.axes) == 13

        rendered = tmp_path / "figure5-submission-300dpi.png"
        figure.savefig(
            rendered,
            dpi=300,
            bbox_inches=None,
            pad_inches=0,
            facecolor="white",
        )
        with Image.open(rendered) as opened:
            assert opened.size == (
                round(178.0 / 25.4 * 300),
                round(148.0 / 25.4 * 300),
            )
            pixels = np.asarray(opened.convert("RGB"))
        outer_two = np.concatenate(
            (
                pixels[:2].reshape(-1, 3),
                pixels[-2:].reshape(-1, 3),
                pixels[2:-2, :2].reshape(-1, 3),
                pixels[2:-2, -2:].reshape(-1, 3),
            ),
            axis=0,
        )
        assert int(np.sum(np.min(outer_two, axis=1) < 245)) == 0
    finally:
        plt.close(figure)


def test_figure5_rejects_inconsistent_h11_support_denominators(
    tmp_path: Path,
) -> None:
    _, figure_inputs, _ = _fixture(tmp_path / "figure5-bad-support")
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    resources = {
        role: figure_inputs.parent / record["path"]
        for role, record in manifest["resources"].items()
    }
    support_path = resources["assurance_support"]
    support = pd.read_csv(support_path)
    support.loc[0, "support_fraction"] = float(support.loc[0, "support_fraction"]) + 0.01
    support.to_csv(support_path, index=False)

    with pytest.raises(
        figures.FigureSuiteError,
        match="assurance support counts/fractions are inconsistent",
    ):
        figures._figure5(resources, provisional=False, final=True)


def test_figure5_rejects_nonpositive_effects_before_log_rendering(
    tmp_path: Path,
) -> None:
    _, figure_inputs, _ = _fixture(tmp_path / "figure5-nonpositive-effect")
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    resources = {
        role: figure_inputs.parent / record["path"]
        for role, record in manifest["resources"].items()
    }
    effects_path = resources["phenotype_effects"]
    effects = pd.read_csv(effects_path)
    effects.loc[0, "ci_low"] = 0.0
    effects.to_csv(effects_path, index=False)

    with pytest.raises(
        figures.FigureSuiteError,
        match="strictly positive for log display",
    ):
        figures._figure5(resources, provisional=False, final=True)


def test_matcher_schematic_encodes_partial_curve_support_and_direction() -> None:
    figure, axis = plt.subplots()
    try:
        figures._draw_matcher_contract(axis)
        text = {item.get_text() for item in axis.texts}
        assert "32-point equal-arc resampling" in text
        assert "truth→prediction support ≥25%" in text
        assert "prediction→truth support ≥25%" in text
        assert "proximal directions non-opposing (cosine ≥0)" in text
        assert {line.get_label() for line in axis.lines} >= {
            "Annotated centreline",
            "Predicted centreline",
        }
    finally:
        plt.close(figure)


def test_fixed_effect_grid_rejects_duplicates_and_preserves_locked_order() -> None:
    rows = [
        {
            "endpoint_key": endpoint,
            "effect_key": effect,
            "cohort": cohort,
            "estimate": 1.0,
        }
        for cohort in reversed(figures.PHENOTYPE_EFFECT_COHORT_ORDER)
        for effect in reversed(figures.EFFECT_ORDER)
        for endpoint in reversed(figures.PRIMARY_ENDPOINTS)
    ]
    shuffled = pd.DataFrame(rows).sample(frac=1.0, random_state=29)
    ordered = figures._ordered_fixed_effects(shuffled, "test effects")
    expected = [
        (endpoint, effect, cohort)
        for endpoint in figures.PRIMARY_ENDPOINTS
        for effect in figures.EFFECT_ORDER
        for cohort in figures.PHENOTYPE_EFFECT_COHORT_ORDER
    ]
    assert list(
        ordered[["endpoint_key", "effect_key", "cohort"]].itertuples(
            index=False, name=None
        )
    ) == expected
    duplicate = pd.concat([shuffled, shuffled.iloc[[0]]], ignore_index=True)
    with pytest.raises(figures.FigureSuiteError, match="duplicate fixed"):
        figures._ordered_fixed_effects(duplicate, "test effects")


def test_s4_uses_six_subgroup_colours_and_retains_nonestimable_ccc_as_na(
    tmp_path: Path,
) -> None:
    _, figure_inputs, _ = _fixture(tmp_path / "s4-six-family-contract")
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    resources = {
        role: figure_inputs.parent / record["path"]
        for role, record in manifest["resources"].items()
    }
    pairs_path = resources["assurance_pairs"]
    pairs = pd.read_csv(pairs_path)
    constant = (
        (pairs["pair_type"].astype(str) == "root_trait")
        & (pairs["trait_id"].astype(str) == "R19")
    )
    pairs.loc[constant, ["observed", "predicted"]] = 5.0
    pairs.to_csv(pairs_path, index=False)

    figure = figures._supplementary_s4(resources, provisional=False)
    try:
        assert len(figure.axes) == 3
        legend_labels = {
            item.get_text()
            for legend in figure.legends
            for item in legend.get_texts()
        }
        assert legend_labels == set(figures.ROOT_TRAIT_FAMILY_LABELS.values())
        rendered = {
            item.get_text()
            for axis in figure.axes
            for item in [*axis.texts, *axis.get_yticklabels()]
        }
        assert "NA" in rendered
        assert any("R19" in value and "CCC=NA" in value for value in rendered)
        assert all("MAE" not in label for label in legend_labels)
    finally:
        plt.close(figure)


def test_s6_draws_a_physical_scale_bar_on_every_source_and_overlay_panel(
    tmp_path: Path,
) -> None:
    _, figure_inputs, _ = _fixture(tmp_path / "s6-scale-bars")
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    resources = {
        role: figure_inputs.parent / record["path"]
        for role, record in manifest["resources"].items()
    }
    figure = figures._supplementary_s6(resources, provisional=False)
    try:
        assert len(figure.axes) == 10
        for axis in figure.axes:
            assert any(item.get_text() == "100 µm" for item in axis.texts)
            assert any(line.get_color() == "white" for line in axis.lines)
    finally:
        plt.close(figure)


def test_figure3_scale_panel_reports_three_metrics_and_applicability(
    tmp_path: Path,
) -> None:
    _, figure_inputs, _ = _fixture(tmp_path / "figure3-scale-assurance")
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    resources = {
        role: figure_inputs.parent / record["path"]
        for role, record in manifest["resources"].items()
    }
    figure = figures._figure3(resources, provisional=False)
    try:
        rendered = "\n".join(text.get_text() for text in figure.axes[2].texts)
        assert "Scale coverage 6/38" in rendered
        assert "line error 3.5 µm" in rendered
        assert "Calibration error 1.2%" in rendered
        assert "38 visible + 6 metadata; absent/untrusted n=0" in rendered
        assert "Absence specificity: not estimable" in rendered
        assert "Fail-closed: software contract + unit tests" in rendered
    finally:
        plt.close(figure)


def test_supplementary_missing_trait_is_placeholder_only_in_provisional(
    tmp_path: Path,
) -> None:
    _, figure_inputs, _ = _fixture(tmp_path / "missing-s4-trait")
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    resources = {
        role: figure_inputs.parent / record["path"]
        for role, record in manifest["resources"].items()
    }
    pairs_path = resources["assurance_pairs"]
    pairs = pd.read_csv(pairs_path)
    pairs = pairs[~(
        (pairs["pair_type"].astype(str) == "root_trait")
        & (pairs["trait_id"].astype(str) == "R19")
    )]
    pairs.to_csv(pairs_path, index=False)

    builder = lambda: figures._supplementary_s4(resources, True)
    placeholder = figures._guarded_supplementary(
        stem=figures.SUPPLEMENTARY_STEMS[3], final=False, builder=builder
    )
    text = "\n".join(item.get_text() for axis in placeholder.axes for item in axis.texts)
    assert "FINAL EVIDENCE PENDING" in text
    assert "No quantitative value has been substituted" in text
    plt.close(placeholder)
    with pytest.raises(figures.FigureSuiteError, match="all 19 root descriptors"):
        figures._guarded_supplementary(
            stem=figures.SUPPLEMENTARY_STEMS[3], final=True, builder=builder
        )


def test_final_builder_rejects_resealed_supplementary_contract_drift(
    monkeypatch, tmp_path: Path
) -> None:
    receipts, figure_inputs, proposal = _fixture(tmp_path / "supp-contract-drift")
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    contract = manifest["supplementary_figure_contract"]
    contract["figures"][0]["number"] = "S0"
    contract.pop("contract_identity_sha256")
    contract["contract_identity_sha256"] = figures.sha256_json(contract)
    manifest.pop("figure_input_assembly_identity_sha256")
    manifest["figure_input_assembly_identity_sha256"] = figures.sha256_json(
        manifest
    )
    _write_json(figure_inputs, manifest)
    monkeypatch.setattr(figures, "save_figure_bundle", _fast_bundle)
    output = tmp_path / "supp-contract-drift-output"
    with pytest.raises(
        figures.FigureSuiteError,
        match="supplementary S1--S9 contract changed",
    ):
        figures.build_figure_suite(
            mode="final",
            figure_inputs=figure_inputs,
            model_contract_proposal=proposal,
            output=output,
            receipt_paths=receipts,
        )
    assert not output.exists()


def test_final_builder_rejects_multitrait_source_hash_drift(
    monkeypatch, tmp_path: Path
) -> None:
    receipts, figure_inputs, proposal = _fixture(tmp_path / "hash-drift-inputs")
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    clean_path = figure_inputs.parent / manifest["source_inputs"]["clean_traits"][
        "path"
    ]
    clean_path.write_text(clean_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(figures, "save_figure_bundle", _fast_bundle)
    output = tmp_path / "hash-drift-output"
    with pytest.raises(figures.FigureSuiteError, match="SHA-256 mismatch"):
        figures.build_figure_suite(
            mode="final",
            figure_inputs=figure_inputs,
            model_contract_proposal=proposal,
            output=output,
            receipt_paths=receipts,
        )
    assert not output.exists()


def test_final_builder_rejects_resealed_invented_multitrait_effect(
    monkeypatch, tmp_path: Path
) -> None:
    receipts, figure_inputs, proposal = _fixture(tmp_path / "invented-effect-inputs")
    manifest = json.loads(figure_inputs.read_text(encoding="utf-8"))
    atlas_path = figure_inputs.parent / manifest["resources"]["multitrait_atlas"][
        "path"
    ]
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    effect = atlas["descriptors"][1]["cohorts"]["primary_clean261"]["effects"][
        "OE_vs_EV"
    ]
    effect.update(
        {
            "status": "estimated_fixed_15_effect_family",
            "estimate": 1.2,
            "ci95_low": 1.0,
            "ci95_high": 1.4,
            "endpoint_n": 261,
            "effect_scale": "ratio",
            "not_estimable_reason": None,
        }
    )
    atlas.pop("atlas_identity_sha256")
    atlas["atlas_identity_sha256"] = figures.sha256_json(atlas)
    _write_json(atlas_path, atlas)
    manifest["resources"]["multitrait_atlas"]["sha256"] = sha256_file(atlas_path)
    manifest.pop("figure_input_assembly_identity_sha256")
    manifest["figure_input_assembly_identity_sha256"] = figures.sha256_json(
        manifest
    )
    _write_json(figure_inputs, manifest)
    monkeypatch.setattr(figures, "save_figure_bundle", _fast_bundle)
    output = tmp_path / "invented-effect-output"
    with pytest.raises(figures.FigureSuiteError, match="invented effect outside fixed family"):
        figures.build_figure_suite(
            mode="final",
            figure_inputs=figure_inputs,
            model_contract_proposal=proposal,
            output=output,
            receipt_paths=receipts,
        )
    assert not output.exists()
