#!/usr/bin/env python
"""Exploratory PHAxis biology models on clean-primary/full-sensitivity cohorts.

The numerical model implementation is owned by the installable PHAxis package.
This wrapper verifies cohort/configuration locks and writes analysis provenance.
It must only be run after the final PHAxis trait export and cohort build are
locked.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.biological_analysis import (  # noqa: E402
    RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
    RAW_EFFECT_HC3_INTERVAL,
    RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
    RAW_EFFECT_OLS_MEAN_CONTRAST,
    WT_CONTRAST_MULTIPLICITY_FAMILY,
    WT_META_MULTIPLICITY_FAMILY,
    coerce_boolean_series,
    count_results,
    group_summaries,
    linear_results,
    raw_median_bootstrap_seed,
    robust_sensitivity,
    wt_temperature_secondary_results,
)
from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402
from phaxis.model_contract_binding import (  # noqa: E402
    read_model_contract_authority,
    require_output_identity,
)


SCHEMA = "PHAxis-exploratory-biological-analysis-1.0"
WT_SECONDARY_SCHEMA = "PHAxis-WT-temperature-secondary-1.0"
MODEL_SPEC_SCHEMA = "PHAxis-biological-model-spec-1.1"
MODEL_SPEC_STATUS = "locked_for_phaxis_v1_0_postresult_exploratory_analysis"
HISTORICAL_MODEL_SPEC_SHA256 = (
    "9ec5d72d70547d43afbf8cacd690e3a0e007b4ab19cb0c672f5d09dbcbd82901"
)
H11_ENDPOINT = "local_median_hair_length_um_1_4mm"


def _verify_model_spec(spec: Mapping[str, Any]) -> None:
    """Fail closed unless the PHAxis-owned spec retains the locked semantics."""
    if spec.get("schema_version") != MODEL_SPEC_SCHEMA:
        raise RuntimeError("unexpected PHAxis biological model-spec schema")
    if spec.get("status") != MODEL_SPEC_STATUS:
        raise RuntimeError("PHAxis biological model spec is not locked")
    provenance = spec.get("historical_provenance", {})
    if (
        provenance.get("source_spec_sha256") != HISTORICAL_MODEL_SPEC_SHA256
        or provenance.get("model_or_inference_semantics_changed") is not True
        or provenance.get("semantic_harmonization")
        != (
            "First-hair observability is reported descriptively by condition and is "
            "not fitted inside the fixed five-endpoint by three-effect family; only "
            "conditional distance enters the 15 model effects."
        )
    ):
        raise RuntimeError("PHAxis biological model-spec provenance mismatch")
    expected_scope = {
        "study_role": "rhd6_factorial_8d_primary",
        "experiment_key": "D15_8d",
        "developmental_day": 8,
        "factors": {
            "construct": ["RHD6-EV", "RHD6-OE"],
            "temperature_c": [22, 30],
        },
        "effects": [
            "construct_OE_minus_EV",
            "temperature_30C_minus_22C",
            "construct_by_temperature_interaction",
        ],
    }
    if spec.get("primary_scope") != expected_scope:
        raise RuntimeError("PHAxis biological primary-scope semantics mismatch")
    endpoint_keys = ("field", "model", "offset", "effect_scale")
    expected_endpoints = [
        {
            "field": "local_hair_count_1_4mm",
            "model": "negative_binomial_log_link",
            "offset": "log_window_length_mm_equals_log_3",
            "effect_scale": "rate_ratio",
        },
        {
            "field": "local_median_hair_length_um_1_4mm",
            "model": "HC3_OLS_log_transformed",
            "offset": None,
            "effect_scale": "geometric_mean_ratio_and_raw_median_difference",
        },
        {
            "field": "first_hair_ge40um_distance_from_distal_point_um",
            "model": "HC3_OLS_log_transformed_conditional_on_observable",
            "offset": None,
            "effect_scale": "geometric_mean_ratio_conditional_distance",
        },
        {
            "field": "median_root_width_um",
            "model": "HC3_OLS_log_transformed",
            "offset": None,
            "effect_scale": "geometric_mean_ratio_and_raw_mean_difference",
        },
        {
            "field": "visible_root_axis_length_um",
            "model": "HC3_OLS_log_transformed",
            "offset": None,
            "effect_scale": "geometric_mean_ratio_and_raw_mean_difference",
        },
    ]
    endpoints = [
        {key: item.get(key) for key in endpoint_keys}
        for item in spec.get("confirmatory_endpoints", [])
    ]
    if endpoints != expected_endpoints:
        raise RuntimeError("PHAxis biological endpoint-model semantics mismatch")
    eligibility = spec.get("eligibility", {})
    expected_eligibility = {
        "phenotype_based_exclusion_allowed": False,
        "fixed_distal_window_um": [1000.0, 4000.0],
        "fixed_distal_window_requires_visible_axis_to_um": 4000.0,
        "first_hair_minimum_centerline_length_um": 40.0,
        "maximum_attachment_boundary_error_um": 40.0,
        "whole_hair_zone_confirmatory_traits_allowed": False,
        "root_cap_region_statistics_allowed": False,
        "root_cap_point_statistics_allowed": True,
    }
    if any(eligibility.get(key) != value for key, value in expected_eligibility.items()):
        raise RuntimeError("PHAxis biological eligibility semantics mismatch")
    inference = spec.get("inference", {})
    expected_inference = {
        "continuous_design_formula": "outcome ~ construct * temperature_c",
        "heteroskedastic_covariance": "HC3",
        "confidence_level": 0.95,
        "factorial_cell_stratified_bootstrap_replicates": 5000,
        "freedman_lane_permutations": 9999,
        "random_seed": 20260823,
        "confirmatory_multiplicity": (
            "Benjamini-Hochberg across the fixed five-endpoint by three-effect family "
            "(15 conditional phenotype effects)"
        ),
        "confirmatory_fdr_q": 0.05,
        "phenotype_outlier_filter": "none",
        "robust_sensitivity": "Huber_M_estimation_and_leave_one_out_sign_stability",
    }
    if any(inference.get(key) != value for key, value in expected_inference.items()):
        raise RuntimeError("PHAxis biological inference semantics mismatch")
    expected_fallback = {
        "only_after_nb2_nonconvergence_or_nonfinite_inference": True,
        "criterion": "Pearson chi-square divided by residual degrees of freedom",
        "pearson_chi2_over_df_maximum": 1.25,
        "fallback_covariance": "HC3",
        "fail_closed_above_threshold": True,
    }
    if inference.get("poisson_fallback_diagnostic") != expected_fallback:
        raise RuntimeError("PHAxis Poisson fallback semantics mismatch")
    expected_wt = {
        "study_role": "wt_temperature_block",
        "status": "secondary_blocked_replication",
        "primary_model": "experiment_fixed_effect_temperature_contrast",
        "endpoint_family": [
            "local_hair_count_1_4mm",
            "local_median_hair_length_um_1_4mm",
            "first_hair_ge40um_distance_from_distal_point_um",
            "median_root_width_um",
            "visible_root_axis_length_um",
        ],
        "within_experiment_estimand": (
            "30C_over_22C_ratio_on_log_or_log_link_scale"
        ),
        "within_experiment_models": {
            "local_hair_count_1_4mm": (
                "negative_binomial_log_link_with_locked_Poisson_HC3_fallback"
            ),
            "positive_continuous_endpoints": "HC3_OLS_log_transformed",
            "first_hair_distance": "conditional_on_observable",
        },
        "per_experiment_minimum_per_temperature": 3,
        "per_endpoint_minimum_per_temperature": 3,
        "minimum_experiments_per_day_meta_analysis": 3,
        "meta_analysis": "random_effects_REML_with_Hartung_Knapp_interval",
        "heterogeneity": ["tau2", "I2", "Q"],
        "developmental_day_handling": (
            "strict_within-day_stratification; cross-day pooled estimate forbidden"
        ),
        "unknown_day_handling": (
            "report_estimable_within_experiment_contrast_but_forbid_meta_analysis"
        ),
        "insufficient_same_day_experiments": (
            "typed_not_estimable_row_without_pooled_estimate"
        ),
        "within_experiment_multiplicity": (
            "Benjamini-Hochberg within each cohort across every estimated "
            "experiment-by-endpoint contrast, including unknown-day contrasts"
        ),
        "within_day_meta_multiplicity": (
            "Benjamini-Hochberg within each cohort across every estimated "
            "developmental-day-by-endpoint meta-analysis"
        ),
        "clean_full_pooling_allowed": False,
    }
    if spec.get("wt_temperature_scope") != expected_wt:
        raise RuntimeError("PHAxis WT temperature-stratification semantics mismatch")
    if spec.get("reporting", {}).get("blind_images_used") != 0:
        raise RuntimeError("model specification is blind-tainted")
    reporting = spec.get("reporting", {})
    if (
        reporting.get("first_hair_observability_in_fixed_effect_family") is not False
        or reporting.get("first_hair_observability_reporting")
        != (
            "descriptive n/formal_n by condition; conditional distance alone enters "
            "the fixed 15-effect family"
        )
    ):
        raise RuntimeError("first-hair observability reporting semantics mismatch")


def _atomic_dataframe(
    path: Path, frame: pd.DataFrame, *, allow_empty: bool = False
) -> None:
    if frame.empty and not allow_empty:
        raise RuntimeError(f"refusing to write empty table: {path}")
    if frame.empty and not len(frame.columns):
        raise RuntimeError(f"empty table has no typed columns: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8-sig", newline=""
        ) as handle:
            frame.to_csv(handle, index=False, lineterminator="\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _verify_cohort_build(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = root / "summary.json"
    lock_path = root / "analysis_contract_lock.json"
    summary = read_json(summary_path)
    lock = read_json(lock_path)
    if (
        summary.get("schema_version") != "PHAxis-biological-cohorts-1.0"
        or summary.get("status")
        != "completed_without_fitting_biological_effect_models"
        or summary.get("blind_images_used") != 0
    ):
        raise RuntimeError("invalid or blind-tainted cohort build")
    if sha256_file(lock_path) != summary["output_sha256"]["analysis_contract_lock"]:
        raise RuntimeError("cohort lock hash mismatch")
    identity = lock.get("cohort_lock_identity_sha256")
    unsigned = dict(lock)
    unsigned.pop("cohort_lock_identity_sha256", None)
    if identity != sha256_json(unsigned):
        raise RuntimeError("cohort lock identity mismatch")
    if lock.get("blind_images_used") != 0:
        raise RuntimeError("cohort lock is blind-tainted")
    for cohort_name in summary["cohort_directories"].values():
        for table, expected in summary["output_sha256"][cohort_name].items():
            path = root / cohort_name / f"{table}.csv"
            if sha256_file(path) != expected:
                raise RuntimeError(f"cohort table hash mismatch: {cohort_name}/{table}")
    return summary, lock


def _adjust_fdr(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    finite_model = pd.to_numeric(frame["p_value_model"], errors="coerce").notna()
    frame["p_value_model_BH_FDR"] = np.nan
    frame["reject_model_BH_FDR_0p05"] = False
    if finite_model.any():
        rejected, adjusted, _alpha_sidak, _alpha_bonf = multipletests(
            frame.loc[finite_model, "p_value_model"].to_numpy(dtype=float),
            alpha=0.05,
            method="fdr_bh",
        )
        frame.loc[finite_model, "p_value_model_BH_FDR"] = adjusted
        frame.loc[finite_model, "reject_model_BH_FDR_0p05"] = rejected
    finite_permutation = pd.to_numeric(
        frame["p_value_freedman_lane"], errors="coerce"
    ).notna()
    frame["p_value_freedman_lane_BH_FDR"] = np.nan
    if finite_permutation.any():
        _rejected, adjusted, _alpha_sidak, _alpha_bonf = multipletests(
            frame.loc[finite_permutation, "p_value_freedman_lane"].to_numpy(
                dtype=float
            ),
            alpha=0.05,
            method="fdr_bh",
        )
        frame.loc[
            finite_permutation, "p_value_freedman_lane_BH_FDR"
        ] = adjusted
    return frame


def _verify_factorial_raw_companions(
    frame: pd.DataFrame, *, bootstrap_replicates: int, seed: int
) -> None:
    """Fail closed on the locked H11 median and other continuous companions."""

    provenance_columns = (
        "raw_effect_estimand",
        "raw_effect_interval_method",
        "raw_effect_bootstrap_replicates",
        "raw_effect_bootstrap_seed",
    )
    missing = [column for column in provenance_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"factorial raw companion provenance is absent: {missing}")
    h11 = frame.loc[frame["endpoint"].eq(H11_ENDPOINT)].copy()
    if len(h11) != 3 or set(h11["effect"].astype(str)) != {
        "construct_OE_minus_EV",
        "temperature_30C_minus_22C",
        "construct_by_temperature_interaction",
    }:
        raise RuntimeError("H11 raw-median companion is not a complete three-effect set")
    expected_seed = raw_median_bootstrap_seed(
        seed=seed, field=H11_ENDPOINT, component="continuous"
    )
    if (
        not h11["raw_effect_estimand"]
        .eq(RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST)
        .all()
        or not h11["raw_effect_interval_method"]
        .eq(RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL)
        .all()
        or not pd.to_numeric(
            h11["raw_effect_bootstrap_replicates"], errors="coerce"
        )
        .eq(bootstrap_replicates)
        .all()
        or not pd.to_numeric(h11["raw_effect_bootstrap_seed"], errors="coerce")
        .eq(expected_seed)
        .all()
    ):
        raise RuntimeError("H11 raw-median companion semantics or RNG lock drifted")
    estimate = pd.to_numeric(h11["raw_effect_estimate"], errors="coerce")
    low = pd.to_numeric(h11["raw_effect_ci95_low"], errors="coerce")
    high = pd.to_numeric(h11["raw_effect_ci95_high"], errors="coerce")
    if not (
        np.isfinite(estimate).all()
        and np.isfinite(low).all()
        and np.isfinite(high).all()
        and low.le(high).all()
    ):
        raise RuntimeError("H11 raw-median point estimate or interval is invalid")
    other_continuous = frame.loc[
        frame["model_component"].eq("continuous")
        & ~frame["endpoint"].eq(H11_ENDPOINT)
    ]
    if len(other_continuous) != 9 or (
        not other_continuous["raw_effect_estimand"]
        .eq(RAW_EFFECT_OLS_MEAN_CONTRAST)
        .all()
        or not other_continuous["raw_effect_interval_method"]
        .eq(RAW_EFFECT_HC3_INTERVAL)
        .all()
        or not pd.to_numeric(
            other_continuous["raw_effect_bootstrap_replicates"], errors="coerce"
        )
        .eq(0)
        .all()
        or not other_continuous["raw_effect_bootstrap_seed"].isna().all()
    ):
        raise RuntimeError("non-H11 continuous raw-mean companion semantics drifted")
    count = frame.loc[frame["model_component"].eq("count_rate")]
    if len(count) != 3 or (
        not count["raw_effect_estimand"].eq(RAW_EFFECT_OLS_MEAN_CONTRAST).all()
        or not count["raw_effect_interval_method"].eq(RAW_EFFECT_HC3_INTERVAL).all()
        or not pd.to_numeric(
            count["raw_effect_bootstrap_replicates"], errors="coerce"
        )
        .eq(0)
        .all()
        or not count["raw_effect_bootstrap_seed"].isna().all()
    ):
        raise RuntimeError("count raw-mean companion semantics drifted")


def _run_factorial(
    frame: pd.DataFrame,
    *,
    model_spec: Mapping[str, Any],
    cohort: str,
    cohort_role: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["formal_statistics_eligible"] = coerce_boolean_series(
        frame["formal_statistics_eligible"]
    )
    scope = model_spec["primary_scope"]
    all_primary = frame[
        frame["study_role"].eq(scope["study_role"])
        & frame["experiment_key"].eq(scope["experiment_key"])
    ].copy()
    primary = all_primary[all_primary["formal_statistics_eligible"]].copy()
    expected_conditions = {
        "RHD6_EV_22C",
        "RHD6_EV_30C",
        "RHD6_OE_22C",
        "RHD6_OE_30C",
    }
    if set(primary["condition_code"]) != expected_conditions:
        raise RuntimeError(f"{cohort}: primary scope lost one or more cells")
    endpoint_labels = {
        item["field"]: item["label"] for item in model_spec["confirmatory_endpoints"]
    }
    inference = model_spec["inference"]
    bootstrap = int(inference["factorial_cell_stratified_bootstrap_replicates"])
    permutations = int(inference["freedman_lane_permutations"])
    seed = int(inference["random_seed"])
    results: list[dict[str, Any]] = []
    results.extend(
        count_results(
            primary,
            field="local_hair_count_1_4mm",
            endpoint_label=endpoint_labels["local_hair_count_1_4mm"],
            bootstrap_replicates=bootstrap,
            permutations=permutations,
            seed=seed,
            poisson_fallback_max_dispersion=float(
                inference["poisson_fallback_diagnostic"]
                ["pearson_chi2_over_df_maximum"]
            ),
        )
    )
    for field in (
        "local_median_hair_length_um_1_4mm",
        "median_root_width_um",
        "visible_root_axis_length_um",
    ):
        results.extend(
            linear_results(
                primary,
                field=field,
                endpoint_label=endpoint_labels[field],
                component="continuous",
                log_transform=True,
                bootstrap_replicates=bootstrap,
                permutations=permutations,
                seed=seed,
                raw_effect_estimand=(
                    RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
                    if field == H11_ENDPOINT
                    else RAW_EFFECT_OLS_MEAN_CONTRAST
                ),
            )
        )
    first = "first_hair_ge40um_distance_from_distal_point_um"
    results.extend(
        linear_results(
            primary,
            field=first,
            endpoint_label=endpoint_labels[first],
            component="continuous",
            log_transform=True,
            bootstrap_replicates=bootstrap,
            permutations=permutations,
            seed=seed,
        )
    )
    result_frame = _adjust_fdr(pd.DataFrame(results))
    if result_frame.empty:
        raise RuntimeError(f"{cohort}: no factorial model produced results")
    if len(result_frame) != 15 or set(result_frame["model_component"].astype(str)) != {
        "count_rate",
        "continuous",
    }:
        raise RuntimeError(
            f"{cohort}: formal factorial output is not the fixed 15 conditional phenotype effects"
        )
    _verify_factorial_raw_companions(
        result_frame, bootstrap_replicates=bootstrap, seed=seed
    )
    result_frame.insert(0, "cohort", cohort)
    result_frame.insert(1, "cohort_role", cohort_role)
    result_frame["inference_status"] = (
        "exploratory_model_based_not_plate_randomization_confirmatory"
    )
    result_frame["freedman_lane_null"] = (
        "unit_residual_permutation_not_biological_plate_randomization"
    )
    result_frame["causal_treatment_claim_allowed"] = False
    robust = robust_sensitivity(primary)
    if robust.empty:
        raise RuntimeError(f"{cohort}: robust sensitivity produced no result")
    robust.insert(0, "cohort", cohort)
    endpoints = [
        (item["field"], item["label"])
        for item in model_spec["confirmatory_endpoints"]
    ]
    groups = group_summaries(primary, endpoints, scope=cohort)
    flow = (
        all_primary.groupby(
            ["condition_code", "formal_statistics_eligible"], dropna=False
        )
        .size()
        .rename("units")
        .reset_index()
    )
    flow.insert(0, "cohort", cohort)
    return result_frame, robust, groups, flow


def _run_wt_secondary(
    frame: pd.DataFrame,
    *,
    model_spec: Mapping[str, Any],
    cohort: str,
    cohort_role: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the secondary WT block without changing the D15 effect family."""

    wt_spec = model_spec["wt_temperature_scope"]
    inference = model_spec["inference"]
    contrasts, meta, flow = wt_temperature_secondary_results(
        frame,
        minimum_per_temperature=int(
            wt_spec["per_experiment_minimum_per_temperature"]
        ),
        minimum_experiments_per_day_meta=int(
            wt_spec["minimum_experiments_per_day_meta_analysis"]
        ),
        poisson_fallback_max_dispersion=float(
            inference["poisson_fallback_diagnostic"]
            ["pearson_chi2_over_df_maximum"]
        ),
    )
    for table in (contrasts, meta, flow):
        table.insert(0, "cohort", cohort)
        table.insert(1, "cohort_role", cohort_role)

    contrasts["p_value_model_BH_FDR"] = np.nan
    contrasts["reject_model_BH_FDR_0p05"] = False
    contrasts["multiplicity_family"] = WT_CONTRAST_MULTIPLICITY_FAMILY
    finite_contrasts = (
        contrasts["analysis_status"].eq("estimated")
        & pd.to_numeric(contrasts["p_value_model"], errors="coerce").notna()
    )
    if finite_contrasts.any():
        rejected, adjusted, _alpha_sidak, _alpha_bonf = multipletests(
            contrasts.loc[finite_contrasts, "p_value_model"].to_numpy(
                dtype=float
            ),
            alpha=0.05,
            method="fdr_bh",
        )
        contrasts.loc[
            finite_contrasts, "p_value_model_BH_FDR"
        ] = adjusted
        contrasts.loc[
            finite_contrasts, "reject_model_BH_FDR_0p05"
        ] = rejected

    meta["p_value_hartung_knapp_BH_FDR"] = np.nan
    meta["reject_hartung_knapp_BH_FDR_0p05"] = False
    meta["multiplicity_family"] = WT_META_MULTIPLICITY_FAMILY
    finite_meta = (
        meta["analysis_status"].eq("estimated")
        & pd.to_numeric(
            meta["p_value_hartung_knapp"], errors="coerce"
        ).notna()
    )
    if finite_meta.any():
        rejected, adjusted, _alpha_sidak, _alpha_bonf = multipletests(
            meta.loc[finite_meta, "p_value_hartung_knapp"].to_numpy(
                dtype=float
            ),
            alpha=0.05,
            method="fdr_bh",
        )
        meta.loc[
            finite_meta, "p_value_hartung_knapp_BH_FDR"
        ] = adjusted
        meta.loc[
            finite_meta, "reject_hartung_knapp_BH_FDR_0p05"
        ] = rejected
    contrasts["inference_status"] = (
        "secondary_exploratory_within_experiment_association"
    )
    meta["inference_status"] = (
        "secondary_exploratory_same_day_experiment_replication"
    )
    flow["phenotype_outlier_filter_applied"] = False
    return contrasts, meta, flow


def _comparison(clean: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    keys = ["endpoint", "model_component", "effect"]
    selected = [
        *keys,
        "n",
        "estimate",
        "ci95_low",
        "ci95_high",
        "p_value_model",
        "p_value_model_BH_FDR",
        "p_value_freedman_lane",
        "p_value_freedman_lane_BH_FDR",
    ]
    merged = clean[selected].merge(
        full[selected], on=keys, suffixes=("_clean", "_full"), validate="one_to_one"
    )
    ratio_fields: list[float] = []
    direction: list[bool] = []
    for clean_estimate, full_estimate in zip(
        merged["estimate_clean"], merged["estimate_full"], strict=True
    ):
        clean_value = float(clean_estimate)
        full_value = float(full_estimate)
        if clean_value > 0 and full_value > 0:
            ratio_fields.append(math.log(clean_value) - math.log(full_value))
            direction.append(
                (clean_value >= 1 and full_value >= 1)
                or (clean_value <= 1 and full_value <= 1)
            )
        else:
            ratio_fields.append(float("nan"))
            direction.append(
                math.copysign(1.0, clean_value) == math.copysign(1.0, full_value)
            )
    merged["clean_minus_full_log_effect"] = ratio_fields
    merged["same_direction_clean_vs_full"] = direction
    merged["interpretation"] = (
        "clean is primary; full is overlap-contaminated sensitivity"
    )
    return merged


def analyze(
    *,
    cohorts: Path,
    analysis_contract: Path,
    model_spec: Path,
    model_contract_proposal: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    proposal_binding = read_model_contract_authority(model_contract_proposal)
    proposal_fields = proposal_binding.receipt_fields()
    public_identity = proposal_binding.public_identity_fields()
    cohort_summary, cohort_lock = _verify_cohort_build(cohorts)
    require_output_identity(
        cohort_summary,
        proposal_binding,
        role="PHAxis biological-cohort summary",
    )
    require_output_identity(
        cohort_lock,
        proposal_binding,
        role="PHAxis biological-cohort lock",
    )
    contract = read_json(analysis_contract)
    spec = read_json(model_spec)
    if contract.get("schema_version") != "PHAxis-biological-analysis-contract-1.0":
        raise RuntimeError("unexpected biological analysis contract")
    if cohort_lock.get("analysis_contract_sha256") != sha256_file(analysis_contract):
        raise RuntimeError("cohort build used a different biological analysis contract")
    _verify_model_spec(spec)
    wt_spec = spec["wt_temperature_scope"]
    contract_endpoints = contract["endpoints"]["primary"]
    model_endpoints = [item["field"] for item in spec["confirmatory_endpoints"]]
    if contract_endpoints != model_endpoints:
        raise RuntimeError("biological endpoint contract/model-spec mismatch")
    clean_name = cohort_summary["cohort_directories"]["primary"]
    full_name = cohort_summary["cohort_directories"]["sensitivity"]
    clean_traits = pd.read_csv(cohorts / clean_name / "traits.csv")
    full_traits = pd.read_csv(cohorts / full_name / "traits.csv")
    clean_result, clean_robust, clean_groups, clean_flow = _run_factorial(
        clean_traits,
        model_spec=spec,
        cohort=clean_name,
        cohort_role="primary_SHA_disjoint",
    )
    full_result, full_robust, full_groups, full_flow = _run_factorial(
        full_traits,
        model_spec=spec,
        cohort=full_name,
        cohort_role="overlap_contaminated_sensitivity",
    )
    clean_wt_contrasts, clean_wt_meta, clean_wt_flow = _run_wt_secondary(
        clean_traits,
        model_spec=spec,
        cohort=clean_name,
        cohort_role="primary_SHA_disjoint",
    )
    full_wt_contrasts, full_wt_meta, full_wt_flow = _run_wt_secondary(
        full_traits,
        model_spec=spec,
        cohort=full_name,
        cohort_role="overlap_contaminated_sensitivity",
    )
    wt_contrasts = pd.concat(
        [clean_wt_contrasts, full_wt_contrasts], ignore_index=True
    )
    wt_meta = pd.concat([clean_wt_meta, full_wt_meta], ignore_index=True)
    wt_flow = pd.concat([clean_wt_flow, full_wt_flow], ignore_index=True)
    if (
        wt_meta.get("cross_day_pooling_performed", pd.Series(False))
        .fillna(False)
        .astype(bool)
        .any()
        or wt_meta.get("unknown_day_contrasts_included", pd.Series(False))
        .fillna(False)
        .astype(bool)
        .any()
    ):
        raise RuntimeError("WT secondary analysis crossed a developmental-day gate")
    estimated_meta = wt_meta.loc[wt_meta["analysis_status"].eq("estimated")]
    if not estimated_meta["k_eligible_experiments"].ge(3).all():
        raise RuntimeError("WT meta-analysis estimated a row with fewer than k=3")
    comparison = _comparison(clean_result, full_result)
    paths = {
        "primary_tests": output / "tables/primary_clean_exploratory_factorial_tests.csv",
        "sensitivity_tests": output
        / "tables/full283_sensitivity_factorial_tests.csv",
        "clean_full_comparison": output
        / "tables/clean_vs_full_effect_stability.csv",
        "robust_sensitivity": output / "tables/robust_sensitivity.csv",
        "group_summaries": output / "tables/primary_group_summaries.csv",
        "model_qc_flow": output / "tables/primary_model_qc_flow.csv",
        "wt_within_experiment_contrasts": output
        / "tables/wt_within_experiment_temperature_contrasts.csv",
        "wt_within_day_meta_analysis": output
        / "tables/wt_within_day_REML_Hartung_Knapp.csv",
        "wt_temperature_qc_flow": output
        / "tables/wt_temperature_model_qc_flow.csv",
    }
    _atomic_dataframe(paths["primary_tests"], clean_result)
    _atomic_dataframe(paths["sensitivity_tests"], full_result)
    _atomic_dataframe(paths["clean_full_comparison"], comparison)
    _atomic_dataframe(
        paths["robust_sensitivity"],
        pd.concat([clean_robust, full_robust], ignore_index=True),
    )
    _atomic_dataframe(
        paths["group_summaries"],
        pd.concat([clean_groups, full_groups], ignore_index=True),
    )
    _atomic_dataframe(
        paths["model_qc_flow"],
        pd.concat([clean_flow, full_flow], ignore_index=True),
    )
    _atomic_dataframe(
        paths["wt_within_experiment_contrasts"],
        wt_contrasts,
        allow_empty=True,
    )
    _atomic_dataframe(
        paths["wt_within_day_meta_analysis"],
        wt_meta,
        allow_empty=True,
    )
    _atomic_dataframe(
        paths["wt_temperature_qc_flow"], wt_flow, allow_empty=True
    )
    result_lock: dict[str, Any] = {
        "schema_version": "PHAxis-biological-analysis-result-lock-1.2",
        "status": "postresult_exploratory_analysis_provenance_lock",
        "preregistration_claimed": False,
        "analysis_contract_sha256": sha256_file(analysis_contract),
        "model_spec_sha256": sha256_file(model_spec),
        "model_spec_schema_version": spec["schema_version"],
        "historical_model_spec_sha256": spec["historical_provenance"][
            "source_spec_sha256"
        ],
        "cohort_summary_sha256": sha256_file(cohorts / "summary.json"),
        "cohort_lock_sha256": sha256_file(cohorts / "analysis_contract_lock.json"),
        "wrapper_implementation_sha256": sha256_file(Path(__file__)),
        "numerical_implementation": "phaxis.biological_analysis",
        "numerical_implementation_sha256": sha256_file(
            PROJECT_ROOT / "src/phaxis/biological_analysis.py"
        ),
        "legacy_model_implementation_imported": False,
        "output_table_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "primary_cohort": clean_name,
        "sensitivity_cohort": full_name,
        "biological_plate_randomization_inference_performed": False,
        "causal_treatment_claim_allowed": False,
        "h11_raw_median_companion": {
            "status": "materialized_contract_conformant",
            "endpoint": H11_ENDPOINT,
            "source_unit": "source_root",
            "cell_summary": "median",
            "construct_effect": (
                "0.5*((median_OE_22C-median_EV_22C)+"
                "(median_OE_30C-median_EV_30C))"
            ),
            "temperature_effect": (
                "0.5*((median_EV_30C-median_EV_22C)+"
                "(median_OE_30C-median_OE_22C))"
            ),
            "interaction_effect": (
                "(median_OE_30C-median_OE_22C)-"
                "(median_EV_30C-median_EV_22C)"
            ),
            "interval_method": RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
            "bootstrap_replicates": int(
                spec["inference"]["factorial_cell_stratified_bootstrap_replicates"]
            ),
            "base_seed": int(spec["inference"]["random_seed"]),
            "effective_seed": raw_median_bootstrap_seed(
                seed=int(spec["inference"]["random_seed"]),
                field=H11_ENDPOINT,
                component="continuous",
            ),
            "separate_hypothesis_test_added": False,
            "D15_fixed_effect_family_changed": False,
        },
        "wt_secondary_analysis": {
            "schema_version": WT_SECONDARY_SCHEMA,
            "status": "materialized_as_separate_secondary_family",
            "endpoint_count": 5,
            "within_experiment_estimand": (
                "30C_over_22C_ratio_on_log_or_log_link_scale"
            ),
            "minimum_per_temperature_base_and_endpoint": 3,
            "minimum_experiments_per_day_meta_analysis": 3,
            "meta_analysis": "random_effects_REML_with_Hartung_Knapp_interval",
            "within_experiment_multiplicity": wt_spec[
                "within_experiment_multiplicity"
            ],
            "within_day_meta_multiplicity": wt_spec[
                "within_day_meta_multiplicity"
            ],
            "cross_day_pooling_performed": False,
            "unknown_day_meta_analysis_performed": False,
            "clean_full_pooling_performed": False,
            "D15_fixed_effect_family_changed": False,
        },
        "root_cap_region_statistics_included": False,
        "whole_hair_zone_confirmatory_traits_included": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        **proposal_fields,
        **public_identity,
    }
    result_lock["analysis_result_lock_identity_sha256"] = sha256_json(result_lock)
    lock_path = output / "analysis_result_lock.json"
    atomic_write_json(lock_path, result_lock)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "completed_exploratory_clean_primary_full_sensitivity",
        "primary_cohort": clean_name,
        "sensitivity_cohort": full_name,
        "primary_scope_units": int(clean_result["n"].max()),
        "sensitivity_scope_units": int(full_result["n"].max()),
        "primary_result_rows": len(clean_result),
        "sensitivity_result_rows": len(full_result),
        "same_direction_clean_vs_full_rows": int(
            comparison["same_direction_clean_vs_full"].sum()
        ),
        "clean_full_comparison_rows": len(comparison),
        "primary_model_BH_FDR_rejections": int(
            clean_result["reject_model_BH_FDR_0p05"].sum()
        ),
        "D15_fixed_effect_rows": len(clean_result),
        "D15_fixed_effect_family_changed_by_WT_secondary": False,
        "h11_raw_median_companion": result_lock["h11_raw_median_companion"],
        "wt_secondary_analysis": {
            "schema_version": WT_SECONDARY_SCHEMA,
            "status": "materialized_as_separate_secondary_family",
            "endpoint_count": 5,
            "within_experiment_estimand": (
                "30C_over_22C_ratio_on_log_or_log_link_scale"
            ),
            "minimum_per_temperature_base_and_endpoint": 3,
            "minimum_experiments_per_day_meta_analysis": 3,
            "meta_analysis": "random_effects_REML_with_Hartung_Knapp_interval",
            "within_experiment_multiplicity": wt_spec[
                "within_experiment_multiplicity"
            ],
            "within_day_meta_multiplicity": wt_spec[
                "within_day_meta_multiplicity"
            ],
            "cross_day_pooling_performed": False,
            "unknown_day_meta_analysis_performed": False,
            "clean_full_pooling_performed": False,
            "D15_fixed_effect_family_changed": False,
        },
        "wt_secondary_within_experiment_rows": len(wt_contrasts),
        "wt_secondary_estimable_within_experiment_rows": int(
            wt_contrasts["analysis_status"].eq("estimated").sum()
        ),
        "wt_secondary_primary_contrast_BH_FDR_rejections": int(
            clean_wt_contrasts["reject_model_BH_FDR_0p05"].sum()
        ),
        "wt_secondary_sensitivity_contrast_BH_FDR_rejections": int(
            full_wt_contrasts["reject_model_BH_FDR_0p05"].sum()
        ),
        "wt_secondary_unknown_day_contrast_rows": int(
            wt_contrasts["developmental_day"].isna().sum()
        ),
        "wt_secondary_within_day_meta_rows": len(wt_meta),
        "wt_secondary_estimable_within_day_meta_rows": int(
            wt_meta["analysis_status"].eq("estimated").sum()
        ),
        "wt_secondary_primary_meta_BH_FDR_rejections": int(
            clean_wt_meta["reject_hartung_knapp_BH_FDR_0p05"].sum()
        ),
        "wt_secondary_sensitivity_meta_BH_FDR_rejections": int(
            full_wt_meta["reject_hartung_knapp_BH_FDR_0p05"].sum()
        ),
        "wt_secondary_typed_not_estimable_meta_rows": int(
            wt_meta["analysis_status"].eq("not_estimable").sum()
        ),
        "wt_secondary_cross_day_pooling_performed": False,
        "wt_secondary_unknown_day_meta_analysis_performed": False,
        "wt_secondary_clean_full_pooling_performed": False,
        "wt_secondary_claim_status": (
            "secondary exploratory blocked replication; pooled estimates require "
            "at least three estimable experiments within one developmental day"
        ),
        "claim_status": (
            "exploratory within-experiment association; not biological-plate "
            "randomization-confirmatory and not causal"
        ),
        "design_identifiability": cohort_summary["design_identifiability"],
        "cohort_build_summary_sha256": sha256_file(cohorts / "summary.json"),
        "analysis_result_lock_sha256": sha256_file(lock_path),
        "output_table_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "root_cap_region_statistics_included": False,
        "whole_hair_zone_confirmatory_traits_included": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        **proposal_fields,
        **public_identity,
    }
    summary["analysis_identity_sha256"] = sha256_json(summary)
    atomic_write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument(
        "--analysis-contract",
        type=Path,
        default=PROJECT_ROOT
        / "configs/phaxis/v1_0/biological_analysis_contract.json",
    )
    parser.add_argument(
        "--model-spec",
        type=Path,
        default=PROJECT_ROOT
        / "configs/phaxis/v1_0/biological_model_spec.json",
    )
    parser.add_argument("--model-contract-proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        cohorts=args.cohorts.resolve(),
        analysis_contract=args.analysis_contract.resolve(),
        model_spec=args.model_spec.resolve(),
        model_contract_proposal=args.model_contract_proposal.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
