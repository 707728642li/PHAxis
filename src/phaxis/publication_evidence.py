"""Canonical logical-identity preimages shared by PHAxis publication gates."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence


SUPPLEMENTARY_FIGURE_CONTRACT_SCHEMA = (
    "PHAxis-publication-supplementary-figure-contract-1.0"
)
MAIN_FIGURE_STEMS = (
    "Figure_01_biological_measurement_design",
    "Figure_02_train399_development_evidence",
    "Figure_03_measurement_assurance",
    "Figure_04_difficult_image_interpretability",
    "Figure_05_exploratory_phenotype_atlas",
    "Figure_06_reproducibility_and_efficiency",
)
MAIN_FIGURE_RESOURCE_ROLES = (
    ("figure1_image", "figure1_geometry", "trait_contract"),
    (
        "development_per_image",
        "development_tolerance",
        "development_threshold",
        "development_strata",
        "assurance_metrics",
        "qcdev_assignment",
    ),
    ("assurance_metrics", "assurance_pairs", "assurance_support"),
    ("overlay_selection", "overlay_audit"),
    (
        "phenotype_points",
        "phenotype_effects",
        "multitrait_atlas",
        "axial_profiles",
        "narrative_decision",
    ),
    ("cohort_flow", "workflow_stages", "runtime_summary", "runtime_per_image"),
)
SUPPLEMENTARY_FIGURE_STEMS = (
    "Supplementary_Figure_S01_stageb_input_architecture_targets",
    "Supplementary_Figure_S02_split_selection_development_strata",
    "Supplementary_Figure_S03_identity_attachment_endpoint_assurance",
    "Supplementary_Figure_S04_primary_root_trait_agreement",
    "Supplementary_Figure_S05_provider_tiling_numerical_equivalence",
    "Supplementary_Figure_S06_expanded_overlay_gallery",
    "Supplementary_Figure_S07_biological_sensitivity_observability",
    "Supplementary_Figure_S08_runtime_memory_io",
    "Supplementary_Figure_S09_multitrait_atlas_coverage_effect_heatmap",
)
SUPPLEMENTARY_FIGURE_TITLES = (
    "Stage-B physical input representation, multihead architecture, and target contract",
    "Family-isolated split, operating-point selection, and development strata",
    "Identity, formal attachment, endpoint, and conditional-length assurance",
    "Agreement of 19 derived primary-root descriptors",
    "Root-provider equivalence, same-component root continuity, formal attachment, and tiled-inference assurance",
    "Expanded acquisition-challenge overlay gallery",
    "Clean-cohort D15 analysis, full-cohort D15 sensitivity, and observability",
    "Direct runtime, memory, utilization, and I/O decomposition",
    (
        "Clean-cohort D15 32-descriptor phenotype map and block/day-stratified "
        "WT temperature secondary evidence"
    ),
)
WT_SECONDARY_RESOURCE_ROLES = (
    "wt_within_experiment_contrasts",
    "wt_within_day_meta_analysis",
    "wt_temperature_qc_flow",
)
WT_SECONDARY_TABLE_FILENAMES = {
    "wt_within_experiment_contrasts": (
        "wt_within_experiment_temperature_contrasts.csv"
    ),
    "wt_within_day_meta_analysis": "wt_within_day_REML_Hartung_Knapp.csv",
    "wt_temperature_qc_flow": "wt_temperature_model_qc_flow.csv",
}
WT_SECONDARY_ENDPOINTS = (
    "local_hair_count_1_4mm",
    "local_median_hair_length_um_1_4mm",
    "first_hair_ge40um_distance_from_distal_point_um",
    "median_root_width_um",
    "visible_root_axis_length_um",
)
WT_SECONDARY_COHORT_ROLES = {
    "primary_clean261": "primary_SHA_disjoint",
    "sensitivity_full283": "overlap_contaminated_sensitivity",
}
WT_CONTRAST_MULTIPLICITY_FAMILY = (
    "within_cohort_all_estimated_WT_experiment_by_endpoint_contrasts_"
    "including_unknown_day"
)
WT_META_MULTIPLICITY_FAMILY = (
    "within_cohort_all_estimated_WT_developmental_day_by_endpoint_"
    "meta_analyses"
)
SUPPLEMENTARY_FIGURE_RESOURCE_ROLES = (
    ("figure1_image", "figure1_geometry", "trait_contract"),
    (
        "development_per_image",
        "development_tolerance",
        "development_threshold",
        "development_strata",
        "cohort_flow",
    ),
    (
        "development_tolerance",
        "assurance_metrics",
        "assurance_pairs",
        "assurance_support",
    ),
    ("trait_contract", "assurance_pairs"),
    ("assurance_metrics", "assurance_pairs", "workflow_stages"),
    ("overlay_selection",),
    (
        "phenotype_points",
        "phenotype_effects",
        "assurance_support",
        "axial_profiles",
    ),
    ("runtime_summary", "runtime_per_image"),
    ("multitrait_atlas", *WT_SECONDARY_RESOURCE_ROLES),
)
SUPPLEMENTARY_FIGURE_RECEIPT_ROLES = (
    ("stageb",),
    ("train399_evaluation",),
    ("train399_evaluation", "fusion", "traits"),
    ("root_exact283", "traits"),
    ("root_exact283", "stageb"),
    ("fusion", "traits"),
    ("cohorts", "analysis", "profiles"),
    (),
    ("traits", "cohorts", "analysis"),
)

# Exact source-table authority copied by the production figure-input assembler.
# The original 19 roles support figures/values; the appended roles close the
# reviewer-facing Table/Data S1--S10 bundle without expanding the exact-eight
# core receipt chain used by the six main figures.
FIGURE_SOURCE_INPUT_ROLES = (
    "split_manifest",
    "historical_oof_per_image",
    "assurance_metrics",
    "assurance_pairs",
    "assurance_support",
    "assurance_topology",
    "clean_traits",
    "full_traits",
    "full_image_traits",
    "analysis_primary_table",
    "analysis_sensitivity_table",
    "profile_analysis_table",
    "sensitivity_profiles_summary",
    "runtime_latency",
    "runtime_production",
    "runtime_per_image",
    "baseline_runtime_latency",
    "baseline_runtime_production",
    "baseline_runtime_per_image",
    "dataset_manifest",
    "image_traits_schema",
    "train399_candidate",
    "train399_selection",
    "model_contract_proposal",
    "training_receipt_seed_2026082801",
    "training_receipt_seed_2026082802",
    "training_receipt_seed_2026082803",
    "training_receipt_seed_2026082804",
    "training_receipt_seed_2026082805",
    "benchmark_same_hardware",
    "benchmark_artifact_inventory",
    "runtime_latency_comparison",
    "runtime_production_comparison",
    *WT_SECONDARY_RESOURCE_ROLES,
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"", "nan", "na", "none", "null"}
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _number(value: Any, role: str, *, finite: bool = True) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{role}: value is not numeric") from error
    if finite and not math.isfinite(result):
        raise ValueError(f"{role}: value is not finite")
    return result


def _integer(value: Any, role: str) -> int:
    number = _number(value, role)
    if number < 0 or not math.isclose(number, round(number)):
        raise ValueError(f"{role}: value is not a non-negative integer")
    return int(round(number))


def _boolean(value: Any, role: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{role}: value is not boolean")


def _day(value: Any, role: str) -> int | None:
    if _is_missing(value):
        return None
    result = _number(value, role)
    if result <= 0 or not math.isclose(result, round(result)):
        raise ValueError(f"{role}: developmental day is not a positive integer")
    return int(round(result))


def _require_columns(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str], role: str
) -> None:
    if not rows:
        return
    missing = sorted(set(columns) - set(rows[0]))
    if missing:
        raise ValueError(f"{role}: required columns missing: {missing}")
    for index, row in enumerate(rows):
        if set(columns) - set(row):
            raise ValueError(f"{role}: row {index} has an inconsistent schema")
        lowered_columns = {str(field).casefold() for field in row}
        if any("root_cap_region" in field or "blind" in field for field in lowered_columns):
            raise ValueError(f"{role}: forbidden blind/root-cap-region column present")
        if any(
            "root_cap_region" in str(value).casefold()
            or "blind_dataset" in str(value).casefold()
            for value in row.values()
        ):
            raise ValueError(f"{role}: forbidden blind/root-cap-region marker present")


def _validate_ratio_row(
    row: Mapping[str, Any], *, role: str, standard_error_field: str
) -> None:
    estimate = _number(row["estimate_30C_over_22C"], f"{role}/estimate")
    low = _number(row["ci95_low"], f"{role}/ci95_low")
    high = _number(row["ci95_high"], f"{role}/ci95_high")
    log_effect = _number(
        row["log_effect_30C_over_22C"], f"{role}/log_effect"
    )
    standard_error = _number(row[standard_error_field], f"{role}/standard_error")
    if not (0 < low <= estimate <= high) or standard_error < 0:
        raise ValueError(f"{role}: invalid positive ratio/interval/standard error")
    if not math.isclose(math.log(estimate), log_effect, rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError(f"{role}: ratio is inconsistent with its log effect")


def validate_wt_secondary_evidence(
    *,
    contrasts: Sequence[Mapping[str, Any]],
    meta: Sequence[Mapping[str, Any]],
    flow: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the independent WT secondary family used by Figure S9.

    The validator is intentionally independent of the numerical modelling
    module.  It prevents a plotting or release step from silently pooling
    experiments across developmental days, treating an unknown day as a known
    block, or drawing a pooled estimate when fewer than three same-day
    experiments are eligible.
    """

    wt_count_columns = (
        "n_total_22C",
        "n_total_30C",
        "n_formal_22C",
        "n_formal_30C",
        "n_endpoint_22C",
        "n_endpoint_30C",
    )
    contrast_columns = (
        "cohort",
        "cohort_role",
        "endpoint",
        "experiment_key",
        "developmental_day",
        "developmental_day_status",
        *wt_count_columns,
        "effect_scale",
        "log_effect_30C_over_22C",
        "log_effect_standard_error",
        "sampling_variance",
        "estimate_30C_over_22C",
        "ci95_low",
        "ci95_high",
        "p_value_model",
        "p_value_model_BH_FDR",
        "reject_model_BH_FDR_0p05",
        "multiplicity_family",
        "analysis_status",
        "not_estimable_reason",
        "meta_eligible",
        "meta_exclusion_reason",
        "inference_status",
    )
    meta_columns = (
        "cohort",
        "cohort_role",
        "endpoint",
        "developmental_day",
        "k_eligible_experiments",
        "eligible_experiments",
        "model",
        "effect_scale",
        "log_effect_30C_over_22C",
        "log_effect_standard_error_hartung_knapp",
        "estimate_30C_over_22C",
        "ci95_low",
        "ci95_high",
        "p_value_hartung_knapp",
        "p_value_hartung_knapp_BH_FDR",
        "reject_hartung_knapp_BH_FDR_0p05",
        "multiplicity_family",
        "tau2_reml_log_scale",
        "Q",
        "Q_df",
        "Q_p_value",
        "I2",
        "I2_percent",
        "hartung_knapp_scale",
        "analysis_status",
        "not_estimable_reason",
        "cross_day_pooling_performed",
        "unknown_day_contrasts_included",
        "inference_status",
    )
    flow_columns = (
        "cohort",
        "cohort_role",
        "experiment_key",
        "developmental_day",
        "developmental_day_status",
        "endpoint",
        *wt_count_columns,
        "base_gate_pass",
        "endpoint_gate_pass",
        "model_status",
        "not_estimable_reason",
        "phenotype_outlier_filter_applied",
    )
    _require_columns(contrasts, contrast_columns, "WT contrasts")
    _require_columns(meta, meta_columns, "WT same-day meta-analysis")
    _require_columns(flow, flow_columns, "WT model-QC flow")
    if not contrasts:
        raise ValueError("WT contrasts: no secondary WT evidence rows")
    if not flow:
        raise ValueError("WT model-QC flow: no secondary WT evidence rows")

    expected_cohorts = set(WT_SECONDARY_COHORT_ROLES)
    contrast_cohorts = {str(row["cohort"]) for row in contrasts}
    flow_cohorts = {str(row["cohort"]) for row in flow}
    if contrast_cohorts != expected_cohorts or flow_cohorts != expected_cohorts:
        raise ValueError("WT evidence: clean/full cohort roles are incomplete")

    contrast_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    contrast_counts_by_key: dict[tuple[str, str, str], dict[str, int]] = {}
    eligible_by_block: dict[tuple[str, int, str], set[str]] = {}
    known_days: set[tuple[str, int]] = set()
    unknown_experiments: set[tuple[str, str]] = set()
    estimated_contrasts = 0
    for index, row in enumerate(contrasts):
        cohort = str(row["cohort"])
        if str(row["cohort_role"]) != WT_SECONDARY_COHORT_ROLES[cohort]:
            raise ValueError(f"WT contrasts row {index}: cohort role changed")
        endpoint = str(row["endpoint"])
        if endpoint not in WT_SECONDARY_ENDPOINTS:
            raise ValueError(f"WT contrasts row {index}: unexpected endpoint")
        experiment = str(row["experiment_key"]).strip()
        if not experiment:
            raise ValueError(f"WT contrasts row {index}: empty experiment key")
        key = (cohort, experiment, endpoint)
        if key in contrast_by_key:
            raise ValueError("WT contrasts: duplicate cohort/experiment/endpoint row")
        contrast_by_key[key] = row
        counts = {
            field: _integer(row[field], f"WT contrasts row {index}/{field}")
            for field in wt_count_columns
        }
        if any(value < 0 for value in counts.values()):
            raise ValueError(f"WT contrasts row {index}: negative sample count")
        for temperature in ("22C", "30C"):
            if not (
                counts[f"n_endpoint_{temperature}"]
                <= counts[f"n_formal_{temperature}"]
                <= counts[f"n_total_{temperature}"]
            ):
                raise ValueError(
                    f"WT contrasts row {index}: endpoint/formal/total count order changed"
                )
        contrast_counts_by_key[key] = counts
        day = _day(row["developmental_day"], f"WT contrasts row {index}")
        day_status = str(row["developmental_day_status"])
        meta_eligible = _boolean(row["meta_eligible"], f"WT contrasts row {index}/meta_eligible")
        status = str(row["analysis_status"])
        if status not in {"estimated", "not_estimable"}:
            raise ValueError(f"WT contrasts row {index}: invalid analysis status")
        if str(row["effect_scale"]) != "ratio_30C_over_22C":
            raise ValueError(f"WT contrasts row {index}: effect scale changed")
        if str(row["multiplicity_family"]) != WT_CONTRAST_MULTIPLICITY_FAMILY:
            raise ValueError(f"WT contrasts row {index}: multiplicity family changed")
        if str(row["inference_status"]) != (
            "secondary_exploratory_within_experiment_association"
        ):
            raise ValueError(f"WT contrasts row {index}: inference status changed")
        if day is None:
            unknown_experiments.add((cohort, experiment))
            if day_status not in {"unknown_all_rows", "unknown_partial_metadata"}:
                raise ValueError(f"WT contrasts row {index}: unknown-day status changed")
            if meta_eligible or (
                status == "estimated"
                and str(row["meta_exclusion_reason"]) != "unknown_developmental_day"
            ):
                raise ValueError(f"WT contrasts row {index}: unknown day entered pooling")
        else:
            known_days.add((cohort, day))
            if day_status != "known_consistent":
                raise ValueError(f"WT contrasts row {index}: known-day status changed")
            if status == "estimated" and not meta_eligible:
                raise ValueError(f"WT contrasts row {index}: estimable known day excluded")
        if status == "estimated":
            if not (
                counts["n_formal_22C"] >= 3
                and counts["n_formal_30C"] >= 3
                and counts["n_endpoint_22C"] >= 3
                and counts["n_endpoint_30C"] >= 3
            ):
                raise ValueError(
                    f"WT contrasts row {index}: estimated model bypassed a sample-size gate"
                )
            estimated_contrasts += 1
            _validate_ratio_row(
                row,
                role=f"WT contrasts row {index}",
                standard_error_field="log_effect_standard_error",
            )
            variance = _number(row["sampling_variance"], f"WT contrasts row {index}/variance")
            standard_error = _number(
                row["log_effect_standard_error"],
                f"WT contrasts row {index}/standard_error",
            )
            if variance <= 0 or not math.isclose(
                variance, standard_error**2, rel_tol=1e-7, abs_tol=1e-10
            ):
                raise ValueError(f"WT contrasts row {index}: variance/SE mismatch")
            for field in ("p_value_model", "p_value_model_BH_FDR"):
                value = _number(row[field], f"WT contrasts row {index}/{field}")
                if not 0 <= value <= 1:
                    raise ValueError(f"WT contrasts row {index}: invalid p value")
            _boolean(
                row["reject_model_BH_FDR_0p05"],
                f"WT contrasts row {index}/BH decision",
            )
            if day is not None and meta_eligible:
                eligible_by_block.setdefault((cohort, day, endpoint), set()).add(
                    experiment
                )
        else:
            if not str(row["not_estimable_reason"]).strip():
                raise ValueError(f"WT contrasts row {index}: missing not-estimable reason")
            if meta_eligible or _boolean(
                row["reject_model_BH_FDR_0p05"],
                f"WT contrasts row {index}/BH decision",
            ):
                raise ValueError(f"WT contrasts row {index}: non-estimable row promoted")
            for field in (
                "log_effect_30C_over_22C",
                "log_effect_standard_error",
                "sampling_variance",
                "estimate_30C_over_22C",
                "ci95_low",
                "ci95_high",
                "p_value_model",
                "p_value_model_BH_FDR",
            ):
                if not _is_missing(row[field]):
                    raise ValueError(f"WT contrasts row {index}: non-estimable value is populated")

    for cohort, experiment in {(key[0], key[1]) for key in contrast_by_key}:
        observed = {
            key[2] for key in contrast_by_key if key[:2] == (cohort, experiment)
        }
        if observed != set(WT_SECONDARY_ENDPOINTS):
            raise ValueError("WT contrasts: an experiment lacks the fixed five endpoints")

    meta_by_key: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    estimated_meta = 0
    not_estimable_meta = 0
    for index, row in enumerate(meta):
        cohort = str(row["cohort"])
        if cohort not in expected_cohorts or str(row["cohort_role"]) != WT_SECONDARY_COHORT_ROLES[cohort]:
            raise ValueError(f"WT meta row {index}: cohort role changed")
        endpoint = str(row["endpoint"])
        if endpoint not in WT_SECONDARY_ENDPOINTS:
            raise ValueError(f"WT meta row {index}: unexpected endpoint")
        day = _day(row["developmental_day"], f"WT meta row {index}")
        if day is None:
            raise ValueError(f"WT meta row {index}: unknown day was pooled")
        key = (cohort, day, endpoint)
        if key in meta_by_key:
            raise ValueError("WT meta-analysis: duplicate cohort/day/endpoint row")
        meta_by_key[key] = row
        if _boolean(row["cross_day_pooling_performed"], f"WT meta row {index}/cross-day"):
            raise ValueError(f"WT meta row {index}: cross-day pooling was performed")
        if _boolean(row["unknown_day_contrasts_included"], f"WT meta row {index}/unknown-day"):
            raise ValueError(f"WT meta row {index}: unknown-day contrast was pooled")
        if str(row["model"]) != "random_effects_REML_Hartung_Knapp":
            raise ValueError(f"WT meta row {index}: model changed")
        if str(row["effect_scale"]) != "ratio_30C_over_22C":
            raise ValueError(f"WT meta row {index}: effect scale changed")
        if str(row["multiplicity_family"]) != WT_META_MULTIPLICITY_FAMILY:
            raise ValueError(f"WT meta row {index}: multiplicity family changed")
        if str(row["inference_status"]) != (
            "secondary_exploratory_same_day_experiment_replication"
        ):
            raise ValueError(f"WT meta row {index}: inference status changed")
        eligible = eligible_by_block.get(key, set())
        k = _integer(row["k_eligible_experiments"], f"WT meta row {index}/k")
        declared_eligible = {
            item for item in str(row["eligible_experiments"]).split(";") if item
        }
        if k != len(eligible) or declared_eligible != eligible:
            raise ValueError(f"WT meta row {index}: eligible experiment set differs")
        status = str(row["analysis_status"])
        if status == "estimated":
            if k < 3:
                raise ValueError(f"WT meta row {index}: pooled estimate has k<3")
            estimated_meta += 1
            _validate_ratio_row(
                row,
                role=f"WT meta row {index}",
                standard_error_field="log_effect_standard_error_hartung_knapp",
            )
            for field in (
                "p_value_hartung_knapp",
                "p_value_hartung_knapp_BH_FDR",
            ):
                value = _number(row[field], f"WT meta row {index}/{field}")
                if not 0 <= value <= 1:
                    raise ValueError(f"WT meta row {index}: invalid p value")
            _boolean(
                row["reject_hartung_knapp_BH_FDR_0p05"],
                f"WT meta row {index}/BH decision",
            )
            tau2 = _number(
                row["tau2_reml_log_scale"], f"WT meta row {index}/tau2"
            )
            q_value = _number(row["Q"], f"WT meta row {index}/Q")
            q_df = _integer(row["Q_df"], f"WT meta row {index}/Q_df")
            q_p_value = _number(
                row["Q_p_value"], f"WT meta row {index}/Q_p_value"
            )
            i2 = _number(row["I2"], f"WT meta row {index}/I2")
            i2_percent = _number(
                row["I2_percent"], f"WT meta row {index}/I2_percent"
            )
            hk_scale = _number(
                row["hartung_knapp_scale"],
                f"WT meta row {index}/Hartung-Knapp scale",
            )
            if (
                tau2 < 0
                or q_value < 0
                or q_df != k - 1
                or not 0 <= q_p_value <= 1
                or not 0 <= i2 <= 1
                or not math.isclose(
                    i2_percent, 100.0 * i2, rel_tol=1e-7, abs_tol=1e-7
                )
                or hk_scale < 0
            ):
                raise ValueError(
                    f"WT meta row {index}: invalid heterogeneity/Hartung-Knapp diagnostics"
                )
        elif status == "not_estimable":
            not_estimable_meta += 1
            if not str(row["not_estimable_reason"]).strip():
                raise ValueError(f"WT meta row {index}: missing not-estimable reason")
            if _boolean(
                row["reject_hartung_knapp_BH_FDR_0p05"],
                f"WT meta row {index}/BH decision",
            ):
                raise ValueError(f"WT meta row {index}: non-estimable row promoted")
            for field in (
                "log_effect_30C_over_22C",
                "log_effect_standard_error_hartung_knapp",
                "estimate_30C_over_22C",
                "ci95_low",
                "ci95_high",
                "p_value_hartung_knapp",
                "p_value_hartung_knapp_BH_FDR",
                "tau2_reml_log_scale",
                "Q",
                "Q_df",
                "Q_p_value",
                "I2",
                "I2_percent",
                "hartung_knapp_scale",
            ):
                if not _is_missing(row[field]):
                    raise ValueError(f"WT meta row {index}: non-estimable value is populated")
        else:
            raise ValueError(f"WT meta row {index}: invalid analysis status")

    expected_meta_keys = {
        (cohort, day, endpoint)
        for cohort, day in known_days
        for endpoint in WT_SECONDARY_ENDPOINTS
    }
    if set(meta_by_key) != expected_meta_keys:
        raise ValueError("WT meta-analysis: same-day five-endpoint grid is incomplete")

    flow_by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for index, row in enumerate(flow):
        cohort = str(row["cohort"])
        if cohort not in expected_cohorts or str(row["cohort_role"]) != WT_SECONDARY_COHORT_ROLES[cohort]:
            raise ValueError(f"WT flow row {index}: cohort role changed")
        key = (cohort, str(row["experiment_key"]), str(row["endpoint"]))
        if key in flow_by_key:
            raise ValueError("WT model-QC flow: duplicate row")
        flow_by_key[key] = row
        if _boolean(
            row["phenotype_outlier_filter_applied"],
            f"WT flow row {index}/outlier filter",
        ):
            raise ValueError(f"WT flow row {index}: phenotype outlier filtering was applied")
        contrast = contrast_by_key.get(key)
        if contrast is None:
            raise ValueError(f"WT flow row {index}: no matching contrast")
        flow_counts = {
            field: _integer(row[field], f"WT flow row {index}/{field}")
            for field in wt_count_columns
        }
        if any(value < 0 for value in flow_counts.values()):
            raise ValueError(f"WT flow row {index}: negative sample count")
        for temperature in ("22C", "30C"):
            if not (
                flow_counts[f"n_endpoint_{temperature}"]
                <= flow_counts[f"n_formal_{temperature}"]
                <= flow_counts[f"n_total_{temperature}"]
            ):
                raise ValueError(
                    f"WT flow row {index}: endpoint/formal/total count order changed"
                )
        if flow_counts != contrast_counts_by_key[key]:
            raise ValueError(
                f"WT flow row {index}: contrast/QC sample counts differ"
            )
        expected_base_gate = (
            flow_counts["n_formal_22C"] >= 3
            and flow_counts["n_formal_30C"] >= 3
        )
        expected_endpoint_gate = (
            flow_counts["n_endpoint_22C"] >= 3
            and flow_counts["n_endpoint_30C"] >= 3
        )
        if (
            _boolean(row["base_gate_pass"], f"WT flow row {index}/base gate")
            != expected_base_gate
            or _boolean(
                row["endpoint_gate_pass"],
                f"WT flow row {index}/endpoint gate",
            )
            != expected_endpoint_gate
        ):
            raise ValueError(f"WT flow row {index}: sample-size gate is inconsistent")
        if (
            str(row["model_status"]) != str(contrast["analysis_status"])
            or str(row["not_estimable_reason"]) != str(contrast["not_estimable_reason"])
            or _day(row["developmental_day"], f"WT flow row {index}")
            != _day(contrast["developmental_day"], f"WT contrast for flow row {index}")
            or str(row["developmental_day_status"])
            != str(contrast["developmental_day_status"])
        ):
            raise ValueError(f"WT flow row {index}: contrast/QC status differs")
    if set(flow_by_key) != set(contrast_by_key):
        raise ValueError("WT model-QC flow: contrast coverage is incomplete")

    return {
        "within_experiment_rows": len(contrasts),
        "estimated_within_experiment_rows": estimated_contrasts,
        "known_day_block_count": len(known_days),
        "unknown_day_experiment_count": len(unknown_experiments),
        "unknown_day_contrast_rows": sum(
            _day(row["developmental_day"], "WT contrast/unknown-day count")
            is None
            for row in contrasts
        ),
        "within_day_meta_rows": len(meta),
        "estimated_within_day_meta_rows": estimated_meta,
        "typed_not_estimable_meta_rows": not_estimable_meta,
        "cross_day_pooling_performed": False,
        "unknown_day_meta_analysis_performed": False,
        "clean_full_pooling_performed": False,
        "D15_fixed_effect_family_changed": False,
    }


def validate_wt_secondary_analysis_binding(
    *,
    analysis_summary: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    table_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Bind the three WT tables to the sealed stage-22 analysis summary."""

    expected_hash_roles = set(WT_SECONDARY_RESOURCE_ROLES)
    if set(table_sha256) != expected_hash_roles:
        raise ValueError("WT secondary table SHA role set is not exact")
    output_hashes = analysis_summary.get("output_table_sha256")
    if not isinstance(output_hashes, Mapping) or any(
        output_hashes.get(role) != table_sha256[role]
        for role in WT_SECONDARY_RESOURCE_ROLES
    ):
        raise ValueError("WT secondary tables are not analysis-receipt hash-bound")
    nested = analysis_summary.get("wt_secondary_analysis")
    if not isinstance(nested, Mapping):
        raise ValueError("WT secondary analysis contract is missing")
    if (
        nested.get("schema_version") != "PHAxis-WT-temperature-secondary-1.0"
        or nested.get("status") != "materialized_as_separate_secondary_family"
        or nested.get("endpoint_count") != 5
        or nested.get("minimum_per_temperature_base_and_endpoint") != 3
        or nested.get("minimum_experiments_per_day_meta_analysis") != 3
        or nested.get("cross_day_pooling_performed") is not False
        or nested.get("unknown_day_meta_analysis_performed") is not False
        or nested.get("clean_full_pooling_performed") is not False
        or nested.get("D15_fixed_effect_family_changed") is not False
    ):
        raise ValueError("WT secondary analysis contract changed")
    expected_counts = {
        "wt_secondary_within_experiment_rows": "within_experiment_rows",
        "wt_secondary_estimable_within_experiment_rows": (
            "estimated_within_experiment_rows"
        ),
        "wt_secondary_unknown_day_contrast_rows": "unknown_day_contrast_rows",
        "wt_secondary_within_day_meta_rows": "within_day_meta_rows",
        "wt_secondary_estimable_within_day_meta_rows": (
            "estimated_within_day_meta_rows"
        ),
        "wt_secondary_typed_not_estimable_meta_rows": (
            "typed_not_estimable_meta_rows"
        ),
    }
    for field, evidence_field in expected_counts.items():
        if analysis_summary.get(field) != evidence_summary.get(evidence_field):
            raise ValueError(f"WT secondary analysis count differs: {field}")
    if (
        analysis_summary.get("D15_fixed_effect_family_changed_by_WT_secondary")
        is not False
        or analysis_summary.get("wt_secondary_cross_day_pooling_performed")
        is not False
        or analysis_summary.get(
            "wt_secondary_unknown_day_meta_analysis_performed"
        )
        is not False
        or analysis_summary.get("wt_secondary_clean_full_pooling_performed")
        is not False
    ):
        raise ValueError("WT secondary analysis escaped its independent family")
    return {
        **dict(evidence_summary),
        "schema_version": str(nested["schema_version"]),
        "status": str(nested["status"]),
        "table_sha256": {
            role: str(table_sha256[role])
            for role in WT_SECONDARY_RESOURCE_ROLES
        },
    }


def supplementary_figure_contract() -> dict[str, Any]:
    """Return the canonical ordered S1--S9 evidence/placeholder contract."""

    figures = []
    for number, (stem, title, resources, receipts) in enumerate(
        zip(
            SUPPLEMENTARY_FIGURE_STEMS,
            SUPPLEMENTARY_FIGURE_TITLES,
            SUPPLEMENTARY_FIGURE_RESOURCE_ROLES,
            SUPPLEMENTARY_FIGURE_RECEIPT_ROLES,
            strict=True,
        ),
        start=1,
    ):
        figures.append(
            {
                "number": f"S{number}",
                "stem": stem,
                "title": title,
                "resource_roles": list(resources),
                "receipt_roles": list(receipts),
            }
        )
    return {
        "schema_version": SUPPLEMENTARY_FIGURE_CONTRACT_SCHEMA,
        "ordered_figure_count": 9,
        "main_figure_count_unchanged": 6,
        "figures": figures,
        "final_policy": (
            "recompute every plotted quantitative cell from the hash-locked "
            "figure resources and named sealed receipts; fail closed"
        ),
        "provisional_policy": (
            "missing final quantitative evidence is rendered only as an "
            "explicit watermarked pending-evidence panel; submission use false"
        ),
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }


def figure_suite_identity_preimage(
    *,
    status: str,
    figure_hashes: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    figure_input_assembly_identity_sha256: str,
    model_contract_proposal_identity_sha256: str,
    model_contract_public_identity: Mapping[str, str],
    train399_prediction_input_provenance: Mapping[str, Any],
    supplementary_table_bundle_identity_sha256: str,
    supplementary_table_bundle_receipt_sha256: str,
) -> dict[str, Any]:
    """Return the sole canonical preimage for a six-figure suite identity."""

    return {
        "status": str(status),
        "figure_hashes": deepcopy(dict(figure_hashes)),
        "source_hashes": dict(source_hashes),
        "figure_input_assembly_identity_sha256": str(
            figure_input_assembly_identity_sha256
        ),
        "model_contract_proposal_identity_sha256": str(
            model_contract_proposal_identity_sha256
        ),
        "model_contract_public_identity": dict(model_contract_public_identity),
        "train399_prediction_input_provenance": deepcopy(
            dict(train399_prediction_input_provenance)
        ),
        "supplementary_table_bundle_identity_sha256": str(
            supplementary_table_bundle_identity_sha256
        ),
        "supplementary_table_bundle_receipt_sha256": str(
            supplementary_table_bundle_receipt_sha256
        ),
    }
