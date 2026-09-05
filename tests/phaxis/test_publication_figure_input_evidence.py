from __future__ import annotations

import importlib.util
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phaxis.hair_attachment_assurance import build_hair_attachment_assurance
from phaxis.publication_evidence import (
    WT_CONTRAST_MULTIPLICITY_FAMILY,
    WT_META_MULTIPLICITY_FAMILY,
    WT_SECONDARY_COHORT_ROLES,
    WT_SECONDARY_ENDPOINTS,
)
from phaxis.root_continuity_assurance import build_root_continuity_assurance


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = PROJECT_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assembler = _load(
    "phaxis_publication_figure_input_assembler_test",
    "scripts/phaxis/build_publication_figure_inputs.py",
)
assurance = _load(
    "phaxis_measurement_assurance_builder_test",
    "scripts/phaxis/build_measurement_assurance_evidence.py",
)
overlay_producer = _load(
    "phaxis_overlay_evidence_builder_test",
    "scripts/phaxis/build_condition_blinded_overlay_evidence.py",
)


@lru_cache(maxsize=1)
def _component_assurance_fixture() -> tuple[dict, dict]:
    source_units = [f"qc-{index:02d}" for index in range(44)]
    root_records = []
    hair_records = []
    for index, source_unit in enumerate(source_units):
        source_sha = assembler.sha256_json(["qc-image", index])
        endpoint = 100.0 if index % 4 == 0 else 72.0 + index % 7
        root_records.append(
            {
                "pair_type": "primary_root_continuity",
                "source_unit": source_unit,
                "source_image_sha256": source_sha,
                "coordinate_space": "physical_um_xy",
                "reference_axis_definition": assembler._root_continuity.ROOT_CONTINUITY_REFERENCE_DEFINITION,
                "prediction_axis_definition": assembler._root_continuity.ROOT_CONTINUITY_PREDICTION_DEFINITION,
                "reference_axis_artifact_sha256": assembler.sha256_json(
                    ["root-reference", index]
                ),
                "prediction_axis_artifact_sha256": assembler.sha256_json(
                    ["root-prediction", index]
                ),
                "reference_axis_xy_um": [[0.0, 0.0], [100.0, 0.0]],
                "predicted_axis_components_xy_um": [
                    [[0.0, 0.0], [endpoint, 0.0]]
                ],
            }
        )
        offset = float(index % 5)
        hair_records.append(
            {
                "pair_type": "hair_attachment",
                "source_unit": source_unit,
                "source_image_sha256": source_sha,
                "coordinate_space": "physical_um_xy",
                "polyline_orientation": "attachment_to_visible_distal_endpoint",
                "annotation_artifact_sha256": assembler.sha256_json(
                    ["hair-reference", index]
                ),
                "prediction_artifact_sha256": assembler.sha256_json(
                    ["hair-prediction", index]
                ),
                "annotated_polylines_xy_um": [
                    [[0.0, 0.0], [0.0, 30.0]],
                    [[20.0, 0.0], [20.0, 28.0]],
                ],
                "predicted_polylines_xy_um": [
                    [[offset, 0.0], [offset, 30.0]],
                    [[20.0 + offset / 2.0, 0.0], [20.0 + offset / 2.0, 28.0]],
                ],
            }
        )
    authority = assembler.sha256_json({"authority": "synthetic-qcdev44"})
    root = build_root_continuity_assurance(
        records=root_records,
        source_units=source_units,
        reference_authority_sha256=authority,
        prediction_authority_identity_sha256=assembler.sha256_json(
            {"root-prediction": "synthetic"}
        ),
    )
    hair = build_hair_attachment_assurance(
        records=hair_records,
        source_units=source_units,
        annotation_authority_sha256=authority,
        prediction_authority_identity_sha256=assembler.sha256_json(
            {"hair-prediction": "synthetic"}
        ),
    )
    return root, hair


def _component_metric_rows(root: dict, hair: dict) -> list[dict]:
    _, expected_root = assembler._validate_root_continuity_assurance(root)
    _, expected_hair = assembler._validate_hair_attachment_assurance(hair)
    root_specs = {
        "root_continuity_reference_axis_coverage_mean": (
            "Mean union reference-axis coverage",
            "fraction",
            "union support diagnostic across every sealed final-mask skeleton component; not a single-component continuity claim",
        ),
        "root_continuity_maximum_single_component_coverage_mean": (
            "Mean maximum single-component root coverage",
            "fraction",
            "mean per-image coverage from the best one connected final-mask skeleton component",
        ),
        "root_continuity_maximum_single_component_coverage_median": (
            "Median maximum single-component root coverage",
            "fraction",
            "median per-image coverage from the best one connected final-mask skeleton component",
        ),
        "root_continuity_best_component_gap_median_um": (
            "Median longest gap on the best root component",
            "um",
            "median longest unsupported reference-axis gap on the maximum-coverage single connected component",
        ),
        "root_continuity_break_free_rate": (
            "Break-free root image rate",
            "fraction",
            "fraction of source images with at least one single connected component spanning every reference interval",
        ),
        "root_continuity_visible_axis_extent_mae_um": (
            "Visible root-axis extent MAE",
            "um",
            "mean absolute proximal-to-distal projected extent error; internal gaps are scored separately",
        ),
    }
    hair_specs = {
        "hair_attachment_qualified_precision_20um": (
            "Attachment-qualified precision @20 µm",
            "fraction",
            "pooled precision whose true positives are formal biological-presence identities with base error <=20 µm",
        ),
        "hair_attachment_qualified_recall_20um": (
            "Attachment-qualified recall @20 µm",
            "fraction",
            "pooled recall whose true positives are formal biological-presence identities with base error <=20 µm",
        ),
        "hair_attachment_qualified_f1_20um": (
            "Attachment-qualified F1 @20 µm",
            "fraction",
            "pooled F1 from the explicit predicted/annotated denominators and attachment-qualified formal identities",
        ),
        "hair_attachment_error_median_um": (
            "Median base error on formal hair identities",
            "um",
            "median attachment/base error over all formal biological-presence matches; no base-only rematching",
        ),
        "hair_attachment_error_p95_um": (
            "P95 base error on formal hair identities",
            "um",
            "95th-percentile attachment/base error over all formal biological-presence matches; no base-only rematching",
        ),
    }
    qualified = hair["summary"]["formal_matched_attachment_accuracy"][
        "attachment_qualified_identity"
    ]
    match_n = hair["summary"]["formal_matched_attachment_accuracy"][
        "attachment_position_error_on_all_formal_identity_matches"
    ]["n"]
    instances = {
        **{key: 44 for key in root_specs},
        "hair_attachment_qualified_precision_20um": qualified["n_pred"],
        "hair_attachment_qualified_recall_20um": qualified["n_gt"],
        "hair_attachment_qualified_f1_20um": qualified["n_pred"]
        + qualified["n_gt"],
        "hair_attachment_error_median_um": match_n,
        "hair_attachment_error_p95_um": match_n,
    }
    rows = []
    for key, (label, unit, definition) in (root_specs | hair_specs).items():
        expected = (expected_root | expected_hair)[key]
        rows.append(
            {
                "domain": "root_continuity" if key in root_specs else "hair_attachment",
                "metric_key": key,
                "label": label,
                "value": expected["value"],
                "ci_low": expected["ci_low"],
                "ci_high": expected["ci_high"],
                "unit": unit,
                "n": 44,
                "instances": instances[key],
                "definition": definition,
                "ci_method": "image/source-unit nonparametric bootstrap",
                "bootstrap_repetitions": 10000,
                "bootstrap_seed": 20260828,
            }
        )
    return rows


def _phenotype_effect_frame(cohort: str) -> pd.DataFrame:
    rows = []
    for endpoint_index, endpoint in enumerate(assembler.PRIMARY_ENDPOINTS):
        for effect_index, effect in enumerate(assembler.EFFECT_MAP):
            estimate = 0.8 + endpoint_index * 0.05 + effect_index * 0.02
            is_h11 = endpoint == assembler.H11_ENDPOINT
            raw_estimate = (estimate - 1.0) * 100.0
            rows.append(
                {
                    "cohort": cohort,
                    "endpoint": endpoint,
                    "model_component": assembler.PRIMARY_ENDPOINT_COMPONENTS[
                        endpoint
                    ],
                    "effect": effect,
                    "n": 261 if cohort == "primary_clean261" else 283,
                    "estimate": estimate,
                    "ci95_low": estimate - 0.1,
                    "ci95_high": estimate + 0.1,
                    "effect_scale": "ratio",
                    "raw_effect_estimate": raw_estimate,
                    "raw_effect_ci95_low": raw_estimate - 5.0,
                    "raw_effect_ci95_high": raw_estimate + 5.0,
                    "raw_effect_estimand": (
                        assembler.RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
                        if is_h11
                        else assembler.RAW_EFFECT_OLS_MEAN_CONTRAST
                    ),
                    "raw_effect_interval_method": (
                        assembler.RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
                        if is_h11
                        else assembler.RAW_EFFECT_HC3_INTERVAL
                    ),
                    "raw_effect_bootstrap_replicates": (
                        assembler.H11_RAW_BOOTSTRAP_REPLICATES if is_h11 else 0
                    ),
                    "raw_effect_bootstrap_seed": (
                        assembler.raw_median_bootstrap_seed(
                            seed=assembler.H11_RAW_BOOTSTRAP_BASE_SEED,
                            field=assembler.H11_ENDPOINT,
                            component="continuous",
                        )
                        if is_h11
                        else None
                    ),
                    "standardized_effect": raw_estimate / 50.0,
                    "standardized_ci95_low": (raw_estimate - 5.0) / 50.0,
                    "standardized_ci95_high": (raw_estimate + 5.0) / 50.0,
                    "causal_treatment_claim_allowed": False,
                }
            )
    return pd.DataFrame(rows)


def _wt_secondary_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    contrast_rows: list[dict] = []
    meta_rows: list[dict] = []
    flow_rows: list[dict] = []
    experiments = (
        (5, ("D5-A", "D5-B", "D5-C")),
        (7, ("D7-A", "D7-B")),
        (None, ("DAY-UNKNOWN",)),
    )
    for cohort, cohort_role in WT_SECONDARY_COHORT_ROLES.items():
        for day, names in experiments:
            for experiment_index, experiment in enumerate(names):
                for endpoint_index, endpoint in enumerate(
                    WT_SECONDARY_ENDPOINTS
                ):
                    estimate = 1.05 + endpoint_index * 0.04 + experiment_index * 0.01
                    log_effect = float(np.log(estimate))
                    standard_error = 0.08
                    day_status = "known_consistent" if day else "unknown_all_rows"
                    contrast_rows.append(
                        {
                            "cohort": cohort,
                            "cohort_role": cohort_role,
                            "endpoint": endpoint,
                            "endpoint_label": endpoint,
                            "experiment_key": experiment,
                            "developmental_day": day,
                            "developmental_day_status": day_status,
                            "n_total_22C": 6,
                            "n_total_30C": 6,
                            "n_formal_22C": 5,
                            "n_formal_30C": 5,
                            "n_endpoint_22C": 4,
                            "n_endpoint_30C": 4,
                            "effect_scale": "ratio_30C_over_22C",
                            "log_effect_30C_over_22C": log_effect,
                            "log_effect_standard_error": standard_error,
                            "sampling_variance": standard_error**2,
                            "estimate_30C_over_22C": estimate,
                            "ci95_low": float(np.exp(log_effect - 1.96 * standard_error)),
                            "ci95_high": float(np.exp(log_effect + 1.96 * standard_error)),
                            "p_value_model": 0.4,
                            "p_value_model_BH_FDR": 0.5,
                            "reject_model_BH_FDR_0p05": False,
                            "multiplicity_family": WT_CONTRAST_MULTIPLICITY_FAMILY,
                            "analysis_status": "estimated",
                            "not_estimable_reason": "",
                            "meta_eligible": day is not None,
                            "meta_exclusion_reason": (
                                "" if day is not None else "unknown_developmental_day"
                            ),
                            "inference_status": (
                                "secondary_exploratory_within_experiment_association"
                            ),
                        }
                    )
                    flow_rows.append(
                        {
                            "cohort": cohort,
                            "cohort_role": cohort_role,
                            "experiment_key": experiment,
                            "developmental_day": day,
                            "developmental_day_status": day_status,
                            "endpoint": endpoint,
                            "endpoint_label": endpoint,
                            "n_total_22C": 6,
                            "n_total_30C": 6,
                            "n_formal_22C": 5,
                            "n_formal_30C": 5,
                            "n_endpoint_22C": 4,
                            "n_endpoint_30C": 4,
                            "base_gate_pass": True,
                            "endpoint_gate_pass": True,
                            "model_status": "estimated",
                            "not_estimable_reason": "",
                            "phenotype_outlier_filter_applied": False,
                        }
                    )
        for day, names in experiments[:2]:
            for endpoint_index, endpoint in enumerate(
                WT_SECONDARY_ENDPOINTS
            ):
                k = len(names)
                estimated = k >= 3
                estimate = 1.08 + endpoint_index * 0.03
                log_effect = float(np.log(estimate))
                standard_error = 0.06
                meta_rows.append(
                    {
                        "cohort": cohort,
                        "cohort_role": cohort_role,
                        "endpoint": endpoint,
                        "endpoint_label": endpoint,
                        "developmental_day": day,
                        "k_eligible_experiments": k,
                        "eligible_experiments": ";".join(names),
                        "model": "random_effects_REML_Hartung_Knapp",
                        "effect_scale": "ratio_30C_over_22C",
                        "log_effect_30C_over_22C": log_effect if estimated else np.nan,
                        "log_effect_standard_error_hartung_knapp": standard_error if estimated else np.nan,
                        "estimate_30C_over_22C": estimate if estimated else np.nan,
                        "ci95_low": float(np.exp(log_effect - 1.96 * standard_error)) if estimated else np.nan,
                        "ci95_high": float(np.exp(log_effect + 1.96 * standard_error)) if estimated else np.nan,
                        "p_value_hartung_knapp": 0.3 if estimated else np.nan,
                        "p_value_hartung_knapp_BH_FDR": 0.4 if estimated else np.nan,
                        "reject_hartung_knapp_BH_FDR_0p05": False,
                        "multiplicity_family": WT_META_MULTIPLICITY_FAMILY,
                        "tau2_reml_log_scale": 0.0 if estimated else np.nan,
                        "Q": 1.5 if estimated else np.nan,
                        "Q_df": float(k - 1) if estimated else np.nan,
                        "Q_p_value": 0.47 if estimated else np.nan,
                        "I2": 0.0 if estimated else np.nan,
                        "I2_percent": 0.0 if estimated else np.nan,
                        "hartung_knapp_scale": 0.8 if estimated else np.nan,
                        "analysis_status": "estimated" if estimated else "not_estimable",
                        "not_estimable_reason": (
                            "" if estimated else "fewer_than_3_estimable_same_day_experiments"
                        ),
                        "cross_day_pooling_performed": False,
                        "unknown_day_contrasts_included": False,
                        "inference_status": (
                            "secondary_exploratory_same_day_experiment_replication"
                        ),
                    }
                )
    return (
        pd.DataFrame(contrast_rows),
        pd.DataFrame(meta_rows),
        pd.DataFrame(flow_rows),
    )


def test_ccc_uses_identical_sample_covariance_with_nonzero_bias() -> None:
    observed = np.asarray([10.0, 20.0, 35.0, 60.0])
    predicted = np.asarray([13.0, 18.0, 41.0, 67.0])
    covariance = np.cov(observed, predicted, ddof=1)[0, 1]
    expected = 2.0 * covariance / (
        np.var(observed, ddof=1)
        + np.var(predicted, ddof=1)
        + (np.mean(observed) - np.mean(predicted)) ** 2
    )
    assert assurance._ccc(observed, predicted) == pytest.approx(expected, abs=1e-15)
    assert assembler._ccc(observed, predicted) == pytest.approx(expected, abs=1e-15)


def _assurance_fixture():
    root_continuity, hair_attachment = (
        deepcopy(value) for value in _component_assurance_fixture()
    )
    observed = np.asarray([100.0, 180.0, 310.0])
    predicted = np.asarray([112.0, 169.0, 342.0])
    endpoint_error = np.asarray([7.0, 11.0, 19.0])
    continuity = np.asarray([0.90, 0.75, 0.82])
    support = pd.DataFrame(
        {
            "condition_code": list(assembler.GROUP_ORDER),
            "support_fraction": [0.8, 0.75, 0.7, 0.65],
            "supported_hairs": [8, 6, 7, 13],
            "identity_hairs": [10, 8, 10, 20],
            "source_units": [2, 2, 2, 2],
        }
    )
    topology = pd.DataFrame(
        {
            "source_unit": [f"clean-{index:03d}" for index in range(261)],
            "axis_containment_fraction": [0.98] + [1.0] * 260,
            "axis_in_root_coverage_fraction": [0.98] + [1.0] * 260,
            "axis_single_component_coverage_fraction": [0.96] + [1.0] * 260,
            "longest_unsupported_axis_gap_um": [12.0] + [0.0] * 260,
            "root_mask_component_count": [2] + [1] * 260,
            "axis_support_component_label": [1] * 261,
            "unsupported_attachment_n": [1] + [0] * 260,
            "identity_hair_n": [10] * 261,
        }
    )
    pairs = pd.DataFrame(
        [
            {
                "pair_type": "conditional_length",
                "source_unit": f"qc-{index}",
                "observed": observed[index],
                "predicted": predicted[index],
                "unit": "um",
                "endpoint_error_um": endpoint_error[index],
                "trajectory_continuity": continuity[index],
            }
            for index in range(3)
        ]
        + [
            *[
                {
                    "pair_type": "scale",
                    "source_unit": f"qc-scale-{index}",
                    "pair_id": f"qc-scale-{index}:scale",
                    "observed": observed_scale,
                    "predicted": observed_scale * 1.01,
                    "unit": "um_per_px",
                    "relative_error_percent": 1.0,
                    "scale_line_endpoint_error_um": float(2 + 2 * index),
                    "source_image_sha256": f"{index + 1:064x}",
                    "endpoint_error_um": np.nan,
                    "trajectory_continuity": np.nan,
                }
                for index, observed_scale in enumerate((2.0, 2.5, 3.0))
            ],
            {
                "pair_type": "root_trait",
                "source_unit": "qc-root",
                "observed": 1000.0,
                "predicted": 1001.0,
                "unit": "um",
                "endpoint_error_um": np.nan,
                "trajectory_continuity": np.nan,
            },
        ]
    )
    computed = {
        "conditional_length_mae_um": float(np.mean(np.abs(predicted - observed))),
        "conditional_length_bias_um": float(np.mean(predicted - observed)),
        "conditional_length_ccc": float(assurance._ccc(observed, predicted)),
        "matched_endpoint_error_um": float(np.median(endpoint_error)),
        "matched_trajectory_continuity": float(np.mean(continuity)),
        "endpoint_complete_support_fraction": float(
            support["supported_hairs"].sum() / support["identity_hairs"].sum()
        ),
        "axis_containment_median": float(
            np.median(topology["axis_containment_fraction"])
        ),
        "axis_containment_min": 0.98,
        "unsupported_attachment_n": 1.0,
        "scale_geometry_endpoint_error_um": 4.0,
        "scale_relative_error_percent": 1.0,
    }
    required = {
        "root_dice",
        "root_boundary_f1",
        "root_hd95_um",
        "distal_median_error_um",
        "distal_pck",
        "scale_detection_coverage",
        "scale_geometry_endpoint_error_um",
        "scale_relative_error_percent",
        "root_trait_agreement",
        "provider_exact_fraction",
        *computed,
        "root_continuity_reference_axis_coverage_mean",
        *assembler.ROOT_CONTINUITY_FORMAL_METRIC_KEYS,
        *assembler.HAIR_ATTACHMENT_FORMAL_METRIC_KEYS,
    }
    ordinary_required = required - {
        "root_continuity_reference_axis_coverage_mean",
        *assembler.ROOT_CONTINUITY_FORMAL_METRIC_KEYS,
        *assembler.HAIR_ATTACHMENT_FORMAL_METRIC_KEYS,
    }
    metrics = pd.DataFrame(
        [
            {
                "domain": "synthetic",
                "metric_key": key,
                "label": key,
                "value": (
                    3 / 38
                    if key == "scale_detection_coverage"
                    else computed.get(key, 0.9)
                ),
                "ci_low": (
                    0.03
                    if key == "scale_detection_coverage"
                    else computed.get(key, 0.9) - 0.01
                ),
                "ci_high": (
                    0.15
                    if key == "scale_detection_coverage"
                    else computed.get(key, 0.9) + 0.01
                ),
                "unit": "synthetic",
                "n": (
                    38
                    if key == "scale_detection_coverage"
                    else 3
                    if key
                    in {
                        "scale_geometry_endpoint_error_um",
                        "scale_relative_error_percent",
                    }
                    else 44
                ),
                "instances": (
                    3 if key.startswith("scale_") else 44
                ),
                "ci_method": "image/source-unit nonparametric bootstrap",
                "bootstrap_repetitions": 10000,
                "bootstrap_seed": 20260828,
            }
            for key in sorted(ordinary_required)
        ]
        + _component_metric_rows(root_continuity, hair_attachment)
    )
    formal_attachment = hair_attachment["summary"][
        "formal_matched_attachment_accuracy"
    ]
    qualified = formal_attachment["attachment_qualified_identity"]
    formal_presence = formal_attachment["formal_biological_presence"]
    biological_presence_locks = [
        {
            "task_id": row["source_unit"],
            "n_pred": row["formal_matched_attachment_accuracy"][
                "formal_biological_presence"
            ]["n_pred"],
            "n_gt": row["formal_matched_attachment_accuracy"][
                "formal_biological_presence"
            ]["n_gt"],
            "biological_presence_tp_20um": row[
                "formal_matched_attachment_accuracy"
            ]["formal_biological_presence"]["tp"],
            "hair_attachment_row_identity_sha256": row["row_identity_sha256"],
        }
        for row in hair_attachment["per_image"]
    ]
    receipt = {
        "schema_version": assembler.ASSURANCE_RECEIPT_SCHEMA,
        "status": "completed_locked_qc_development_assurance",
        "scope": "QC-development measurement assurance; non-independent",
        "independent_accuracy_claim_allowed": False,
        "metric_evidence_role_by_key": {
            key: "annotated_qc_development_non_independent" for key in required
        },
        "component_receipts": {
            "root_continuity": {
                "audit_copy": "root_continuity_assurance.json",
                "audit_copy_sha256": "a" * 64,
                "identity_field": "root_continuity_assurance_identity_sha256",
                "identity_sha256": root_continuity[
                    "root_continuity_assurance_identity_sha256"
                ],
                "input_contract_audit_copy": "root_continuity_assurance_input.json",
                "input_contract_audit_copy_sha256": "b" * 64,
                "input_contract_identity_sha256": root_continuity[
                    "input_contract_identity_sha256"
                ],
            },
            "hair_attachment": {
                "audit_copy": "hair_attachment_assurance.json",
                "audit_copy_sha256": "c" * 64,
                "identity_field": "hair_attachment_assurance_identity_sha256",
                "identity_sha256": hair_attachment[
                    "hair_attachment_assurance_identity_sha256"
                ],
                "input_contract_audit_copy": "hair_attachment_assurance_input.json",
                "input_contract_audit_copy_sha256": "d" * 64,
                "input_contract_identity_sha256": hair_attachment[
                    "input_contract_identity_sha256"
                ],
            },
        },
        "source_authority_identity_sha256": {
            "root_continuity_assurance": root_continuity[
                "root_continuity_assurance_identity_sha256"
            ],
            "hair_attachment_assurance": hair_attachment[
                "hair_attachment_assurance_identity_sha256"
            ],
            "qcdev_stageb_biological_presence_20um_crosscheck": assembler.sha256_json(
                biological_presence_locks
            ),
        },
        "qcdev_stageb_biological_presence_20um_crosscheck_locks": biological_presence_locks,
        "scale_applicability": {
            "qcdevelopment_images": 44,
            "visible_annotated_scale_bar_cases": 38,
            "trusted_metadata_without_visible_bar_cases": 6,
            "absent_or_untrusted_scale_truth_cases": 0,
            "absence_specificity_status": "not_estimable_no_absent_or_untrusted_scale_cases",
            "fail_closed_evidence_basis": "software_contract_and_unit_tests",
            "empirical_absence_specificity_claimed": False,
        },
        "counts": {
            "qcdevelopment_images": 44,
            "visible_scale_bars": 38,
            "trusted_metadata_without_visible_bar_cases": 6,
            "absent_or_untrusted_scale_truth_cases": 0,
            "detected_scale_bars": 3,
            "scale_localization_pairs": 3,
            "scale_calibration_pairs": 3,
            "root_continuity_source_units": 44,
            "root_continuity_break_free_images": root_continuity["summary"][
                "break_free_images"
            ],
            "root_continuity_union_coverage_hides_fragmentation_images": root_continuity[
                "summary"
            ]["union_coverage_hides_fragmentation_images"],
            "hair_attachment_source_units": 44,
            "hair_attachment_predicted_hairs": qualified["n_pred"],
            "hair_attachment_annotated_hairs": qualified["n_gt"],
            "hair_attachment_formal_identity_matches": formal_presence["tp"],
            "hair_attachment_qualified_true_positives_20um": qualified["tp"],
            "hair_attachment_evaluator_crosschecked_source_units": 44,
        },
        "root_continuity_assurance": root_continuity,
        "hair_attachment_assurance": hair_attachment,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    return receipt, metrics, pairs, support, topology


def test_assurance_normalization_recomputes_every_geometry_cell() -> None:
    receipt, metrics, pairs, support, topology = _assurance_fixture()
    normalized, normalized_pairs, normalized_support = assembler._normalize_assurance(
        receipt, metrics, pairs, support, topology
    )
    assert len(normalized) == len(metrics)
    assert set(normalized["evidence_role"]) == {
        "annotated_qc_development_non_independent"
    }
    assert (
        normalized.set_index("metric_key").loc[
            "root_continuity_reference_axis_coverage_mean",
            "publication_metric_role",
        ]
        == "diagnostic_only_union_coverage"
    )
    assert set(
        normalized[
            normalized["metric_key"].isin(
                [
                    *assembler.ROOT_CONTINUITY_FORMAL_METRIC_KEYS,
                    *assembler.HAIR_ATTACHMENT_FORMAL_METRIC_KEYS,
                ]
            )
        ]["publication_metric_role"]
    ) == {"formal_measurement_assurance"}
    assert normalized_pairs.equals(pairs)
    assert normalized_support.equals(support)


def test_scale_truth_summary_closes_visible_metadata_and_absence_test_cases() -> None:
    task_ids = [f"qc-{index:02d}" for index in range(44)]
    manifest = pd.DataFrame(
        {
            "task_id": task_ids,
            "scale_status": ["visible"] * 38 + ["trusted_metadata"] * 6,
            "scale_bar_count": [1] * 38 + [0] * 6,
        }
    )
    summary = assurance._scale_truth_summary(manifest, task_ids)
    assert summary == {
        "qcdevelopment_images": 44,
        "visible_annotated_scale_bar_cases": 38,
        "trusted_metadata_without_visible_bar_cases": 6,
        "absent_or_untrusted_scale_truth_cases": 0,
        "absence_specificity_status": "not_estimable_no_absent_or_untrusted_scale_cases",
        "fail_closed_evidence_basis": "software_contract_and_unit_tests",
        "empirical_absence_specificity_claimed": False,
    }

    manifest.loc[43, "scale_status"] = "absent"
    with pytest.raises(
        assurance.MeasurementAssuranceError,
        match=r"38 visible \+ 6 trusted metadata \+ 0 absence-test cases",
    ):
        assurance._scale_truth_summary(manifest, task_ids)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_localization", "metric set incomplete"),
        ("bootstrap_drift", "lacks source-image bootstrap CI"),
        ("absence_claim", r"38 visible \+ 6 trusted metadata \+ 0 absence-test cases"),
        ("localization_pair_drift", "differs from sufficient statistics"),
    ),
)
def test_scale_assurance_contract_fails_closed(
    mutation: str, message: str
) -> None:
    receipt, metrics, pairs, support, topology = _assurance_fixture()
    if mutation == "missing_localization":
        metrics = metrics[
            metrics["metric_key"] != "scale_geometry_endpoint_error_um"
        ].copy()
        receipt["metric_evidence_role_by_key"].pop(
            "scale_geometry_endpoint_error_um"
        )
    elif mutation == "bootstrap_drift":
        metrics.loc[
            metrics["metric_key"] == "scale_geometry_endpoint_error_um",
            "ci_method",
        ] = "instance bootstrap"
    elif mutation == "absence_claim":
        receipt["scale_applicability"]["empirical_absence_specificity_claimed"] = (
            True
        )
    else:
        pairs.loc[
            pairs["pair_type"] == "scale", "scale_line_endpoint_error_um"
        ] += 10.0
    with pytest.raises(assembler.FigureInputAssemblyError, match=message):
        assembler._normalize_assurance(
            receipt, metrics, pairs, support, topology
        )


def _reseal_component(payload: dict, identity_field: str) -> None:
    payload.pop(identity_field, None)
    payload[identity_field] = assembler.sha256_json(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("root_seal", "does not seal"),
        ("root_schema", "schema, role, or geometry semantics"),
        ("root_bootstrap", "source-image bootstrap contract drift"),
        ("root_row_identity", "per-image row identity drift"),
        ("hair_role", "schema, role, or polyline semantics"),
        ("hair_proxy_promoted", "base proxy promoted"),
        ("hair_denominator", "attachment-qualified denominator drift"),
        ("metric_value", "differs from embedded source-image sufficient statistics"),
        ("metric_ci", "differs from embedded source-image sufficient statistics"),
        ("metric_n", "semantics/denominators drift"),
        ("metric_instances", "semantics/denominators drift"),
        ("base_proxy_row", "base proxy entered"),
        ("evaluator_crosscheck", "production/evaluator identity crosscheck drift"),
    ),
)
def test_component_assurance_mutations_fail_closed(
    mutation: str, message: str
) -> None:
    receipt, metrics, pairs, support, topology = _assurance_fixture()
    root = receipt["root_continuity_assurance"]
    hair = receipt["hair_attachment_assurance"]
    if mutation == "root_seal":
        root["summary"]["break_free_images"] += 1
    elif mutation == "root_schema":
        root["schema_version"] = "drift"
        _reseal_component(root, "root_continuity_assurance_identity_sha256")
    elif mutation == "root_bootstrap":
        root["bootstrap"]["unit"] = "axis_interval"
        _reseal_component(root, "root_continuity_assurance_identity_sha256")
    elif mutation == "root_row_identity":
        root["per_image"][0]["break_free"] = not root["per_image"][0]["break_free"]
        root["per_image_set_identity_sha256"] = assembler.sha256_json(
            root["per_image"]
        )
        _reseal_component(root, "root_continuity_assurance_identity_sha256")
    elif mutation == "hair_role":
        hair["evidence_role"] = "independent_test"
        _reseal_component(hair, "hair_attachment_assurance_identity_sha256")
    elif mutation == "hair_proxy_promoted":
        hair["metric_contract"]["threshold_selection_used_as_formal_accuracy"] = True
        _reseal_component(hair, "hair_attachment_assurance_identity_sha256")
    elif mutation == "hair_denominator":
        row = hair["per_image"][0]
        row["formal_matched_attachment_accuracy"]["attachment_qualified_identity"][
            "n_pred"
        ] += 1
        row.pop("row_identity_sha256")
        row["row_identity_sha256"] = assembler.sha256_json(row)
        hair["per_image_set_identity_sha256"] = assembler.sha256_json(
            hair["per_image"]
        )
        _reseal_component(hair, "hair_attachment_assurance_identity_sha256")
    elif mutation == "metric_value":
        row = metrics["metric_key"] == "root_continuity_break_free_rate"
        metrics.loc[row, "value"] += 0.01
    elif mutation == "metric_ci":
        row = metrics["metric_key"] == "hair_attachment_qualified_f1_20um"
        metrics.loc[row, "ci_high"] -= 0.01
    elif mutation == "metric_n":
        row = metrics["metric_key"] == "root_continuity_visible_axis_extent_mae_um"
        metrics.loc[row, "n"] = 43
    elif mutation == "metric_instances":
        row = metrics["metric_key"] == "hair_attachment_error_median_um"
        metrics.loc[row, "instances"] += 1
    elif mutation == "base_proxy_row":
        extra = metrics[metrics["metric_key"] == "hair_attachment_qualified_f1_20um"].copy()
        extra["metric_key"] = "hair_attachment_base_proxy_f1_20um"
        metrics = pd.concat([metrics, extra], ignore_index=True)
        receipt["metric_evidence_role_by_key"][
            "hair_attachment_base_proxy_f1_20um"
        ] = "annotated_qc_development_non_independent"
    else:
        lock = receipt[
            "qcdev_stageb_biological_presence_20um_crosscheck_locks"
        ][0]
        lock["biological_presence_tp_20um"] -= 1
        receipt["source_authority_identity_sha256"][
            "qcdev_stageb_biological_presence_20um_crosscheck"
        ] = assembler.sha256_json(
            receipt["qcdev_stageb_biological_presence_20um_crosscheck_locks"]
        )
    with pytest.raises(assembler.FigureInputAssemblyError, match=message):
        assembler._normalize_assurance(
            receipt, metrics, pairs, support, topology
        )


def test_assurance_rejects_a_self_reported_but_wrong_ccc() -> None:
    receipt, metrics, pairs, support, topology = _assurance_fixture()
    row = metrics["metric_key"] == "conditional_length_ccc"
    metrics.loc[row, "value"] = metrics.loc[row, "value"] + 0.01
    with pytest.raises(
        assembler.FigureInputAssemblyError,
        match="conditional_length_ccc differs from sufficient statistics",
    ):
        assembler._normalize_assurance(receipt, metrics, pairs, support, topology)


def test_wt_secondary_figure_evidence_keeps_day_and_D15_gates() -> None:
    contrasts, meta, flow = _wt_secondary_fixture()
    observed = assembler.validate_wt_secondary_evidence(
        contrasts=contrasts.to_dict("records"),
        meta=meta.to_dict("records"),
        flow=flow.to_dict("records"),
    )
    assert observed["estimated_within_day_meta_rows"] == 10
    assert observed["typed_not_estimable_meta_rows"] == 10
    assert observed["unknown_day_experiment_count"] == 2
    assert observed["cross_day_pooling_performed"] is False
    assert observed["D15_fixed_effect_family_changed"] is False

    unknown_promoted = contrasts.copy()
    unknown_promoted.loc[
        unknown_promoted["developmental_day"].isna(), "meta_eligible"
    ] = True
    with pytest.raises(ValueError, match="unknown day entered pooling"):
        assembler.validate_wt_secondary_evidence(
            contrasts=unknown_promoted.to_dict("records"),
            meta=meta.to_dict("records"),
            flow=flow.to_dict("records"),
        )

    false_diamond = meta.copy()
    target = (
        false_diamond["developmental_day"].eq(7)
        & false_diamond["endpoint"].eq(WT_SECONDARY_ENDPOINTS[0])
    )
    false_diamond.loc[target, "analysis_status"] = "estimated"
    false_diamond.loc[target, "not_estimable_reason"] = ""
    false_diamond.loc[target, "log_effect_30C_over_22C"] = np.log(1.1)
    false_diamond.loc[target, "log_effect_standard_error_hartung_knapp"] = 0.1
    false_diamond.loc[target, "estimate_30C_over_22C"] = 1.1
    false_diamond.loc[target, "ci95_low"] = 0.9
    false_diamond.loc[target, "ci95_high"] = 1.3
    false_diamond.loc[target, "p_value_hartung_knapp"] = 0.5
    false_diamond.loc[target, "p_value_hartung_knapp_BH_FDR"] = 0.6
    with pytest.raises(ValueError, match="pooled estimate has k<3"):
        assembler.validate_wt_secondary_evidence(
            contrasts=contrasts.to_dict("records"),
            meta=false_diamond.to_dict("records"),
            flow=flow.to_dict("records"),
        )

    false_heterogeneity = meta.copy()
    false_heterogeneity.loc[
        false_heterogeneity["analysis_status"].eq("not_estimable"),
        "tau2_reml_log_scale",
    ] = 0.0
    with pytest.raises(ValueError, match="non-estimable value is populated"):
        assembler.validate_wt_secondary_evidence(
            contrasts=contrasts.to_dict("records"),
            meta=false_heterogeneity.to_dict("records"),
            flow=flow.to_dict("records"),
        )

    invalid_count_order = contrasts.copy()
    invalid_count_order.loc[0, "n_endpoint_22C"] = 6
    with pytest.raises(ValueError, match="endpoint/formal/total count order changed"):
        assembler.validate_wt_secondary_evidence(
            contrasts=invalid_count_order.to_dict("records"),
            meta=meta.to_dict("records"),
            flow=flow.to_dict("records"),
        )

    count_drift = flow.copy()
    count_drift.loc[0, "n_endpoint_22C"] = 3
    with pytest.raises(ValueError, match="contrast/QC sample counts differ"):
        assembler.validate_wt_secondary_evidence(
            contrasts=contrasts.to_dict("records"),
            meta=meta.to_dict("records"),
            flow=count_drift.to_dict("records"),
        )

    false_gate = flow.copy()
    false_gate.loc[0, "endpoint_gate_pass"] = False
    with pytest.raises(ValueError, match="sample-size gate is inconsistent"):
        assembler.validate_wt_secondary_evidence(
            contrasts=contrasts.to_dict("records"),
            meta=meta.to_dict("records"),
            flow=false_gate.to_dict("records"),
        )


def test_input_contract_has_exact_resources_and_explicit_new_authorities() -> None:
    assert len(assembler.RESOURCE_ROLES) == 25
    assert {
        "qcdev_assignment",
        "overlay_audit",
        "narrative_decision",
    } <= set(assembler.RESOURCE_ROLES)
    assert "multitrait_atlas" in assembler.RESOURCE_ROLES
    assert set(assembler.WT_SECONDARY_RESOURCE_ROLES) <= set(
        assembler.RESOURCE_ROLES
    )
    assert set(assembler.WT_SECONDARY_RESOURCE_ROLES) <= set(
        assembler.FIGURE_SOURCE_INPUT_ROLES
    )
    assert set(assembler.CORE_ROLES) == {
        "train399_evaluation",
        "root_exact283",
        "stageb",
        "fusion",
        "traits",
        "cohorts",
        "analysis",
        "profiles",
    }
    parser_destinations = {action.dest for action in assembler._parser()._actions}
    assert {
        "assurance_topology",
        "full_image_traits",
        "sensitivity_profiles_summary",
        "runtime_latency_summary",
        "runtime_production_summary",
        "baseline_runtime_latency_summary",
        "baseline_runtime_production_summary",
    }.issubset(parser_destinations)
    source = (
        PROJECT_ROOT / "scripts/phaxis/build_publication_figure_inputs.py"
    ).read_text(encoding="utf-8")
    assert "rhaxis_nextgen" not in source.casefold()
    supplementary = assembler.supplementary_figure_contract()
    assert supplementary["ordered_figure_count"] == 9
    assert [record["number"] for record in supplementary["figures"]] == [
        f"S{index}" for index in range(1, 10)
    ]
    assert supplementary["figures"][-1]["stem"].endswith(
        "multitrait_atlas_coverage_effect_heatmap"
    )


def test_phenotype_effects_reject_first_hair_observability_contamination() -> None:
    primary = _phenotype_effect_frame("primary_clean261")
    sensitivity = _phenotype_effect_frame("sensitivity_full283")
    first_hair_observability = primary[
        (primary["endpoint"] == assembler.PRIMARY_ENDPOINTS[2])
        & (primary["effect"] == next(iter(assembler.EFFECT_MAP)))
    ].copy()
    first_hair_observability["model_component"] = "observability"
    contaminated = pd.concat(
        [primary, first_hair_observability], ignore_index=True
    )

    with pytest.raises(
        assembler.FigureInputAssemblyError,
        match="primary_clean261: fixed 15-effect family",
    ):
        assembler._derive_phenotype_effects(contaminated, sensitivity)


def test_phenotype_effects_reject_wrong_model_component() -> None:
    primary = _phenotype_effect_frame("primary_clean261")
    sensitivity = _phenotype_effect_frame("sensitivity_full283")
    wrong_component = (
        (primary["endpoint"] == assembler.PRIMARY_ENDPOINTS[0])
        & (primary["effect"] == next(iter(assembler.EFFECT_MAP)))
    )
    primary.loc[wrong_component, "model_component"] = "continuous"

    with pytest.raises(
        assembler.FigureInputAssemblyError,
        match=(
            "primary_clean261/local_hair_count_1_4mm: model component is outside "
            "the fixed conditional phenotype family"
        ),
    ):
        assembler._derive_phenotype_effects(primary, sensitivity)


def test_phenotype_effects_are_explicitly_reindexed_endpoint_effect_cohort() -> None:
    primary = _phenotype_effect_frame("primary_clean261").sample(
        frac=1.0, random_state=17
    )
    sensitivity = _phenotype_effect_frame("sensitivity_full283").sample(
        frac=1.0, random_state=23
    )
    observed = assembler._derive_phenotype_effects(primary, sensitivity)
    expected = [
        (endpoint, effect, cohort)
        for endpoint in assembler.PRIMARY_ENDPOINTS
        for effect in assembler.EFFECT_ORDER
        for cohort in assembler.PHENOTYPE_EFFECT_COHORT_ORDER
    ]
    assert list(
        observed[["endpoint_key", "effect_key", "cohort"]].itertuples(
            index=False, name=None
        )
    ) == expected
    h11 = observed[observed["endpoint_key"].eq(assembler.H11_ENDPOINT)]
    assert len(h11) == 6
    assert h11["raw_effect_estimand"].eq(
        assembler.RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
    ).all()
    assert h11["raw_effect_interval_method"].eq(
        assembler.RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
    ).all()
    assert h11["raw_effect_bootstrap_replicates"].eq(5000).all()
    assert h11["raw_effect_estimate"].notna().all()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("raw_effect_estimand", "OLS_raw_mean_difference"),
        ("raw_effect_interval_method", "HC3_Wald_confidence_interval"),
        ("raw_effect_bootstrap_replicates", 4999),
        ("raw_effect_bootstrap_seed", 1),
    ],
)
def test_phenotype_effects_reject_h11_raw_median_companion_drift(
    column: str, value: object
) -> None:
    primary = _phenotype_effect_frame("primary_clean261")
    sensitivity = _phenotype_effect_frame("sensitivity_full283")
    target = primary["endpoint"].eq(assembler.H11_ENDPOINT)
    primary.loc[target, column] = value
    with pytest.raises(
        assembler.FigureInputAssemblyError, match="H11 raw-median companion drift"
    ):
        assembler._derive_phenotype_effects(primary, sensitivity)


def test_phenotype_effects_normalize_null_seed_after_csv_roundtrip(
    tmp_path: Path,
) -> None:
    restored = []
    for cohort in ("primary_clean261", "sensitivity_full283"):
        path = tmp_path / f"{cohort}.csv"
        _phenotype_effect_frame(cohort).to_csv(path, index=False)
        restored.append(pd.read_csv(path))
    observed = assembler._derive_phenotype_effects(*restored)
    non_h11 = observed.loc[~observed["endpoint_key"].eq(assembler.H11_ENDPOINT)]
    assert non_h11["raw_effect_bootstrap_seed"].map(
        lambda value: value is None
    ).all()
    assert len(assembler.sha256_json(observed.to_dict("records"))) == 64


@pytest.mark.parametrize("mutation", ["duplicate", "missing"])
def test_phenotype_effects_reject_duplicate_or_missing_fixed_cells(
    mutation: str,
) -> None:
    primary = _phenotype_effect_frame("primary_clean261")
    sensitivity = _phenotype_effect_frame("sensitivity_full283")
    if mutation == "duplicate":
        primary = pd.concat([primary, primary.iloc[[0]]], ignore_index=True)
        pattern = "duplicate cells"
    else:
        primary = primary.iloc[1:].reset_index(drop=True)
        pattern = "incomplete or contains unexpected cells"
    with pytest.raises(assembler.FigureInputAssemblyError, match=pattern):
        assembler._derive_phenotype_effects(primary, sensitivity)


def test_figure_input_staging_does_not_repeat_long_destination_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_mkdtemp(*, prefix: str, dir: Path) -> str:
        parent = Path(dir)
        calls.append((prefix, parent))
        staging = parent / ".figure-inputs-test"
        staging.mkdir()
        return str(staging)

    monkeypatch.setattr(assembler.tempfile, "mkdtemp", fake_mkdtemp)
    destination = tmp_path / ("long-publication-resource-name-" * 6)
    staging = assembler._make_staging_directory(destination)
    assert calls == [(".figure-inputs-", tmp_path)]
    assert destination.name not in staging.name


def _overlay_contract_fixture(tmp_path: Path):
    rows = []
    prediction_sha: dict[str, str] = {}
    trait_rows = []
    for index, role in enumerate(assembler.CASE_ROLES):
        task_id = assembler.OVERLAY_LOCKED_ANCHOR_TASK_IDS.get(
            role, f"overlay-{index}"
        )
        prediction_digest = assembler.sha256_json(["prediction", task_id])
        source_digest = assembler.sha256_json(["source", task_id])
        prediction_sha[task_id] = prediction_digest
        trait_rows.append(
            {"task_id": task_id, "source_image_sha256": source_digest}
        )
        inset_required = role in assembler.OVERLAY_INSET_ROLES
        rows.append(
            {
                "task_id": task_id,
                "prediction_sha256": prediction_digest,
                "raw_source_image_sha256": source_digest,
                "case_id": f"{role}__{task_id}",
                "case_role": role,
                "case_selection_basis": assembler.OVERLAY_CASE_SELECTION_BASIS,
                "random_or_representative_performance_sample": False,
                "experimental_condition_metadata_used_for_rendering": False,
                "experimental_condition_metadata_used_for_evidence_assembly": False,
                "experimental_condition_metadata_used_for_evidence_assembly_scope": (
                    overlay_producer.CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
                ),
                "overlay_sha256": prediction_digest,
                "full_cohort_review_overlay_path": (
                    f"full283_review_overlays/experiment/condition/formal/"
                    f"{task_id}.phaxis_overlay.png"
                ),
                "full_cohort_review_overlay_sha256": prediction_digest,
                "overlay_bytes_reused_from_full_cohort_review_export": True,
                "inset_required": inset_required,
                "inset_rule": (
                    "deterministic_test_axis_window"
                    if inset_required else "not_applicable"
                ),
                "inset_x0": 10 if inset_required else np.nan,
                "inset_y0": 20 if inset_required else np.nan,
                "inset_x1": 80 if inset_required else np.nan,
                "inset_y1": 100 if inset_required else np.nan,
                "inset_geometry_sha256": (
                    assembler.sha256_json(["inset", task_id])
                    if inset_required else np.nan
                ),
            }
        )
    selection = pd.DataFrame(rows)
    selection_path = tmp_path / "overlay_selection.csv"
    selection.to_csv(selection_path, index=False)
    receipt = {
        "schema_version": assembler.OVERLAY_RECEIPT_SCHEMA,
        "status": assembler.OVERLAY_RECEIPT_STATUS,
        "selection_csv_sha256": assembler.sha256_file(selection_path),
        "case_plan_columns": ["case_role", "task_id"],
        "case_selection_basis": assembler.OVERLAY_CASE_SELECTION_BASIS,
        "random_or_representative_performance_sample": False,
        "experimental_condition_metadata_used_for_rendering": False,
        "experimental_condition_metadata_used_for_evidence_assembly": False,
        "experimental_condition_metadata_used_for_evidence_assembly_scope": (
            overlay_producer.CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
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
            "schema_version": assembler.OVERLAY_REVIEW_SCHEMA,
            "status": assembler.OVERLAY_REVIEW_STATUS,
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
            "readme_cn_sha256": "d" * 64,
            "summary_json": "full283_review_summary.json",
            "summary_json_sha256": "9" * 64,
            "review_export_identity_sha256": "a" * 64,
            "ordered_task_set_identity_sha256": "b" * 64,
            "overlay_png_set_identity_sha256": "c" * 64,
            "review_status_on_export": assembler.OVERLAY_REVIEW_PENDING_STATUS,
            "organization_fields": [
                "experiment_key",
                "condition_code",
                "formal_statistics_eligible",
            ],
            "experimental_condition_metadata_used_for_prediction": False,
            "experimental_condition_metadata_used_for_rendering": False,
            "experimental_condition_metadata_used_for_evidence_assembly": False,
            "experimental_condition_metadata_used_for_evidence_assembly_scope": (
                overlay_producer.CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
            ),
            "experimental_condition_metadata_used_for_output_organization": True,
            "create_only": True,
            "canonical_annotations_read": False,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        },
        "inset_contract": {
            "roles": list(assembler.OVERLAY_INSET_ROLES),
            "locked_anchor_task_ids": assembler.OVERLAY_LOCKED_ANCHOR_TASK_IDS,
            "source_and_overlay_use_identical_crop_coordinates": True,
            "whole_image_context_retained": True,
            "performance_based_crop_selection": False,
        },
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    return receipt, selection_path, selection, prediction_sha, pd.DataFrame(trait_rows)


def test_overlay_contract_is_shared_and_avoids_selection_blindness_claims(
    tmp_path: Path,
) -> None:
    assert overlay_producer.SCHEMA_VERSION == assembler.OVERLAY_RECEIPT_SCHEMA
    assert overlay_producer.RECEIPT_STATUS == assembler.OVERLAY_RECEIPT_STATUS
    assert (
        overlay_producer.CASE_SELECTION_BASIS
        == assembler.OVERLAY_CASE_SELECTION_BASIS
    )
    assert overlay_producer.CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE == (
        "overlay_pixels_and_morphology_evidence_cards_before_output_organization"
    )
    receipt, selection_path, selection, prediction_sha, traits = (
        _overlay_contract_fixture(tmp_path)
    )
    assembler._validate_overlay(
        receipt=receipt,
        selection_path=selection_path,
        selection=selection,
        prediction_sha=prediction_sha,
        full_traits=traits,
    )


def test_overlay_insets_are_axis_derived_and_role_deterministic(
    tmp_path: Path,
) -> None:
    axis_path = tmp_path / "axis.npz"
    source_sha = assembler.sha256_json(["inset-source"])
    x = np.linspace(20.0, 180.0, 101)
    y = 80.0 + 35.0 * np.sin(np.linspace(0.0, np.pi, 101))
    axis_xy = np.column_stack((x, y))
    distance = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(axis_xy, axis=0), axis=1)))
    )
    np.savez(
        axis_path,
        path_xy=axis_xy,
        distance_from_tip_px=distance,
        radius_px=np.full(len(axis_xy), 4.0),
        source_image_sha256=np.asarray(source_sha),
    )
    prediction = {
        "task_id": "anchor",
        "source_image_sha256": source_sha,
        "root_axis_geometry_relpath": axis_path.name,
        "root_axis_geometry_sha256": assembler.sha256_file(axis_path),
    }
    low_first = overlay_producer._deterministic_inset(
        role="low_contrast",
        prediction=prediction,
        fusion_root=tmp_path,
        image_shape=(180, 220, 3),
    )
    low_second = overlay_producer._deterministic_inset(
        role="low_contrast",
        prediction=prediction,
        fusion_root=tmp_path,
        image_shape=(180, 220, 3),
    )
    curved = overlay_producer._deterministic_inset(
        role="curved_dense",
        prediction=prediction,
        fusion_root=tmp_path,
        image_shape=(180, 220, 3),
    )
    assert low_first == low_second
    assert low_first["inset_required"] is True
    assert curved["inset_required"] is True
    assert low_first["inset_rule"] != curved["inset_rule"]
    assert low_first["inset_geometry_sha256"] != curved["inset_geometry_sha256"]


def test_fig4_audit_2_reports_counts_continuity_and_eligibility() -> None:
    task_ids = [
        "routine",
        "RHSCU-aa5b6e37df15821f",
        "RHSCU-bbf649822174e0a2",
        "continuity",
        "review",
    ]
    selection_rows = []
    trait_rows = []
    topology_rows = []
    root_lengths = (6000.0, 4500.0, 3500.0, 5200.0, 0.0)
    for role, task_id, root_length in zip(
        assembler.CASE_ROLES, task_ids, root_lengths, strict=True
    ):
        formal = role != "fail_closed"
        selection_rows.append(
            {
                "case_id": f"{role}__{task_id}",
                "case_role": role,
                "task_id": task_id,
                "prediction_sha256": assembler.sha256_json(["prediction", task_id]),
                "formal_statistics_eligible": formal,
            }
        )
        trait_rows.append(
            {
                "task_id": task_id,
                "source_image_sha256": assembler.sha256_json(["source", task_id]),
                "formal_statistics_eligible": formal,
                "exclusion_reason": "root_geometry_fail_closed" if not formal else "",
                "hair_count": 10 if formal else 0,
                "hair_length_measurement_hair_count": 7 if formal else 0,
                "hair_length_measurement_fraction": 0.7 if formal else np.nan,
                "visible_root_axis_length_um": root_length,
                "distal_window_1_4mm_eligible": formal and root_length >= 4000.0,
            }
        )
        if formal:
            topology_rows.append(
                {
                    "source_unit": task_id,
                    "axis_containment_fraction": 0.98,
                    "axis_in_root_coverage_fraction": 0.98,
                    "axis_single_component_coverage_fraction": 0.95,
                    "longest_unsupported_axis_gap_um": 12.0,
                    "identity_hair_n": 10,
                }
            )
    audit = assembler.derive_overlay_audit(
        pd.DataFrame(selection_rows),
        pd.DataFrame(trait_rows),
        pd.DataFrame(topology_rows),
        case_roles=assembler.CASE_ROLES,
    )
    assert set(audit["schema_version"]) == {"PHAxis-Fig4-case-audit-2.0"}
    assert list(audit["case_role"]) == list(assembler.CASE_ROLES)
    assert audit.loc[0, "endpoint_complete_support_count"] == 7
    assert audit.loc[0, "endpoint_complete_support_fraction"] == pytest.approx(0.7)
    assert bool(audit.loc[1, "profile_0_5mm_eligible"]) is False
    assert bool(audit.loc[2, "distal_window_1_4mm_eligible"]) is False
    review = audit.iloc[-1]
    assert review["formal_state"] == "review_only"
    assert pd.isna(review["axis_in_root_coverage_fraction"])
    assert "root_geometry_fail_closed" in review["profile_0_5mm_reason"]


def test_overlay_contract_rejects_performance_or_condition_metadata_use(
    tmp_path: Path,
) -> None:
    receipt, selection_path, selection, prediction_sha, traits = (
        _overlay_contract_fixture(tmp_path)
    )
    unsafe_receipt = deepcopy(receipt)
    unsafe_receipt["random_or_representative_performance_sample"] = True
    with pytest.raises(
        assembler.FigureInputAssemblyError,
        match="preselected non-performance acquisition-challenge gallery",
    ):
        assembler._validate_overlay(
            receipt=unsafe_receipt,
            selection_path=selection_path,
            selection=selection,
            prediction_sha=prediction_sha,
            full_traits=traits,
        )

    unsafe_selection = selection.copy()
    unsafe_selection.loc[0, "experimental_condition_metadata_used_for_rendering"] = True
    with pytest.raises(
        assembler.FigureInputAssemblyError,
        match="experimental_condition_metadata_used_for_rendering=false",
    ):
        assembler._validate_overlay(
            receipt=receipt,
            selection_path=selection_path,
            selection=unsafe_selection,
            prediction_sha=prediction_sha,
            full_traits=traits,
        )


def test_overlay_contract_rejects_incomplete_or_dual_authority_review_export(
    tmp_path: Path,
) -> None:
    receipt, selection_path, selection, prediction_sha, traits = (
        _overlay_contract_fixture(tmp_path)
    )
    incomplete = deepcopy(receipt)
    incomplete["full_cohort_review_export"]["images"] = 282
    with pytest.raises(
        assembler.FigureInputAssemblyError,
        match="exact283 final-fusion review export contract",
    ):
        assembler._validate_overlay(
            receipt=incomplete,
            selection_path=selection_path,
            selection=selection,
            prediction_sha=prediction_sha,
            full_traits=traits,
        )

    dual_authority = selection.copy()
    dual_authority.loc[0, "full_cohort_review_overlay_sha256"] = "f" * 64
    with pytest.raises(
        assembler.FigureInputAssemblyError,
        match="byte-identical to its exact283 review PNG",
    ):
        assembler._validate_overlay(
            receipt=receipt,
            selection_path=selection_path,
            selection=dual_authority,
            prediction_sha=prediction_sha,
            full_traits=traits,
        )


def _evaluation_prediction_authority_fixture() -> dict:
    task_ids = [f"qc-{index:02d}" for index in range(44)]
    stageb_files = [
        {"task_id": task_id, "sha256": assembler.sha256_json(["stageb", task_id])}
        for task_id in task_ids
    ]
    hybrid_files = [
        {"task_id": task_id, "sha256": assembler.sha256_json(["hybrid", task_id])}
        for task_id in task_ids
    ]
    stageb_identity = assembler.sha256_json(stageb_files)
    hybrid_identity = assembler.sha256_json(hybrid_files)
    assembler.LEGACY_HYBRID_QCDEV44_PREDICTION_SET_IDENTITY_SHA256 = hybrid_identity
    authority = {
        "schema_version": assembler.EVALUATION_RUN_SCHEMA,
        "artifact_role": assembler.EVALUATION_ARTIFACT_ROLE,
        "evaluation_detection_schema_version": assembler.EVALUATION_DETECTION_SCHEMA,
        "evaluation_inference_summary_sha256": assembler.sha256_json("summary-file"),
        "evaluation_inference_summary_identity_sha256": assembler.sha256_json(
            "summary-identity"
        ),
        "evaluation_gate_identity_sha256": assembler.sha256_json("evaluation-gate"),
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
    return {
        "per_image": [{"task_id": task_id} for task_id in task_ids],
        "prediction_input_locks": {
            "stageb_detection_files": stageb_files,
            "stageb_detection_set_identity_sha256": stageb_identity,
            "hybrid_prediction_files": hybrid_files,
            "hybrid_prediction_set_identity_sha256": hybrid_identity,
        },
        "evaluation_inference_authority": authority,
        "inputs_sha256": {
            "evaluation_inference_summary": authority[
                "evaluation_inference_summary_sha256"
            ]
        },
        "training_contract": {
            "evaluation_gate_identity_sha256": authority[
                "evaluation_gate_identity_sha256"
            ],
            "evaluation_inference_summary_identity_sha256": authority[
                "evaluation_inference_summary_identity_sha256"
            ],
        },
        "comparator_contract": {
            "hybrid_max": {
                "evidence_role": "locked_legacy_development_comparator",
                "schema_version": assembler.LEGACY_HYBRID_COMPARATOR_SCHEMA,
                "identity_hair_variant": assembler.LEGACY_HYBRID_IDENTITY_VARIANT,
                "count_hair_variant": assembler.LEGACY_HYBRID_IDENTITY_VARIANT,
                "endpoint_complete_identity_layer": True,
                "phaxis_payload_allowed": False,
                "stageb_identity_source_allowed": False,
                "prediction_set_identity_sha256": hybrid_identity,
                "expected_prediction_set_identity_sha256": hybrid_identity,
            }
        },
    }


def test_figure_inputs_record_eval_only_schema_and_reject_production_mislabel() -> None:
    evaluation = _evaluation_prediction_authority_fixture()
    normalized = assembler._validate_train399_prediction_inputs(evaluation)
    assert normalized["stageb_detection_files_schema_version"] == (
        assembler.EVALUATION_DETECTION_SCHEMA
    )
    assert normalized["stageb_evaluation_inference_authority"][
        "artifact_role"
    ] == assembler.EVALUATION_ARTIFACT_ROLE
    assert normalized["stageb_evaluation_inference_authority"][
        "fusion_consumption_allowed"
    ] is False

    production_mislabel = deepcopy(evaluation)
    production_mislabel["evaluation_inference_authority"][
        "evaluation_detection_schema_version"
    ] = "PHAxis-RHAxiscc-StageB-detections-1.0"
    with pytest.raises(
        assembler.FigureInputAssemblyError,
        match="evaluation-only inference authority/schema missing",
    ):
        assembler._validate_train399_prediction_inputs(production_mislabel)

    deployable = deepcopy(evaluation)
    deployable["evaluation_inference_authority"][
        "fusion_consumption_allowed"
    ] = True
    with pytest.raises(
        assembler.FigureInputAssemblyError,
        match="circular, deployable, or tainted",
    ):
        assembler._validate_train399_prediction_inputs(deployable)
