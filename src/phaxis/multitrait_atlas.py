"""Deterministic 32-descriptor publication atlas for PHAxis.

The manuscript's five prespecified model endpoints are only one view of the
canonical PHAxis measurement space.  This module keeps that inferential family
fixed while exposing descriptive clean/full support for every one of the 32
canonical traits.  Missing model effects are represented explicitly; they are
never imputed or estimated post hoc.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from phaxis.biological_analysis import (
    RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
    RAW_EFFECT_HC3_INTERVAL,
    RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
    RAW_EFFECT_OLS_MEAN_CONTRAST,
    raw_median_bootstrap_seed,
)
from phaxis.io import sha256_json


SCHEMA_VERSION = "PHAxis-multitrait-atlas-2.0"
COHORTS = ("primary_clean261", "sensitivity_full283")
EFFECT_KEYS = ("OE_vs_EV", "30C_vs_22C", "interaction")
GROUP_ORDER = ("RHD6_EV_22C", "RHD6_EV_30C", "RHD6_OE_22C", "RHD6_OE_30C")
PRIMARY_ENDPOINTS = (
    "local_hair_count_1_4mm",
    "local_median_hair_length_um_1_4mm",
    "first_hair_ge40um_distance_from_distal_point_um",
    "median_root_width_um",
    "visible_root_axis_length_um",
)
H11_ENDPOINT = "local_median_hair_length_um_1_4mm"
H11_RAW_BOOTSTRAP_REPLICATES = 5000
H11_RAW_BOOTSTRAP_BASE_SEED = 20260823
EFFECT_NAME_TO_KEY = {
    "construct_OE_minus_EV": "OE_vs_EV",
    "temperature_30C_minus_22C": "30C_vs_22C",
    "construct_by_temperature_interaction": "interaction",
}
NOT_ESTIMATED_REASON = "trait_not_in_prespecified_five_endpoint_15_effect_family"
CONDITION_NOT_ESTIMATED_REASON = "no_finite_observations_in_formal_D15_condition"
CONDITION_SUMMARY_STATUS = "estimated_raw_unadjusted_source_unit_summary"
CONDITION_ROW_UNIT = "one formal D15 source image/root"
MEASUREMENT_FAMILY_ORDER = (
    "visible_hair_abundance",
    "conditional_projected_length",
    "axial_deployment",
    "visible_root_extent",
    "root_form_trajectory",
)
MEASUREMENT_FAMILY_TRAIT_IDS = {
    "visible_hair_abundance": ("H01", "H05", "H08", "H09"),
    "conditional_projected_length": (
        "H02",
        "H03",
        "H04",
        "H10",
        "H11",
        "H12",
    ),
    "axial_deployment": ("H06", "H07", "H13"),
    "visible_root_extent": ("R01", "R02", "R05", "R06"),
    "root_form_trajectory": (
        "R03",
        "R04",
        "R07",
        "R08",
        "R09",
        "R10",
        "R11",
        "R12",
        "R13",
        "R14",
        "R15",
        "R16",
        "R17",
        "R18",
        "R19",
    ),
}
LEGACY_V1_FAMILY_VALUES = frozenset(
    {"conditional_elongation", "root_growth_extent"}
)


class MultitraitAtlasError(RuntimeError):
    """The atlas is incomplete, denominator-open, or not source-derived."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultitraitAtlasError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})


def _canonical_descriptors(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    counts = contract.get("counts")
    _require(
        contract.get("schema_version") == "PHAxis-trait-contract-1.0.0"
        and isinstance(counts, Mapping)
        and counts.get("nonredundant_biological_numeric_fields") == 32
        and counts.get("primary_root_fields") == 19
        and counts.get("root_hair_fields") == 13
        and counts.get("root_cap_region_fields") == 0,
        "trait contract is not the canonical 19+13 ontology",
    )
    measurement_family_by_id = {
        trait_id: family
        for family, trait_ids in MEASUREMENT_FAMILY_TRAIT_IDS.items()
        for trait_id in trait_ids
    }
    _require(
        tuple(MEASUREMENT_FAMILY_TRAIT_IDS) == MEASUREMENT_FAMILY_ORDER
        and len(measurement_family_by_id) == 32,
        "measurement-family ontology is incomplete",
    )
    result: list[dict[str, str]] = []
    for family_key, family_name, prefix, expected in (
        ("primary_root_traits", "primary_root", "R", 19),
        ("root_hair_traits", "root_hair", "H", 13),
    ):
        records = contract.get(family_key)
        _require(
            isinstance(records, list) and len(records) == expected,
            f"{family_name}: descriptor count changed",
        )
        for ordinal, record in enumerate(records, start=1):
            _require(isinstance(record, Mapping), f"{family_name}: descriptor malformed")
            expected_id = f"{prefix}{ordinal:02d}"
            _require(record.get("id") == expected_id, f"trait id sequence changed at {expected_id}")
            normalized = {
                "trait_id": expected_id,
                "trait_family": family_name,
                "measurement_family": measurement_family_by_id.get(expected_id, ""),
                "field": str(record.get("field", "")),
                "display_name_cn": str(record.get("display_name_cn", "")),
                "unit": str(record.get("unit", "")),
                "value_type": str(record.get("type", "")),
                "source_definition": str(record.get("source", "")),
            }
            _require(
                all(
                    normalized[key]
                    for key in (
                        "measurement_family",
                        "field",
                        "display_name_cn",
                        "unit",
                        "value_type",
                        "source_definition",
                    )
                ),
                f"{expected_id}: descriptor metadata is incomplete",
            )
            result.append(normalized)
    fields = [record["field"] for record in result]
    _require(len(result) == len(set(fields)) == 32, "canonical trait fields are not unique 32")
    return result


def _numeric_values(table: pd.DataFrame, field: str, cohort: str) -> np.ndarray:
    _require(field in table.columns, f"{cohort}: canonical trait missing: {field}")
    raw = table[field]
    present = raw.notna() & raw.astype(str).str.strip().ne("")
    numeric = pd.to_numeric(raw, errors="coerce")
    _require(
        numeric[present].notna().all(),
        f"{cohort}/{field}: non-null value is not numeric",
    )
    finite = numeric.dropna().to_numpy(dtype=float)
    _require(np.isfinite(finite).all(), f"{cohort}/{field}: non-finite value")
    return finite


def _validate_conditional_total_null_semantics(
    table: pd.DataFrame, *, cohort: str
) -> None:
    """Reject legacy empty-set sums before they enter descriptive statistics."""

    for count_field, total_field in (
        ("hair_count", "total_hair_length_um"),
        ("local_hair_count_1_4mm", "local_total_hair_length_um_per_root_mm_1_4mm"),
    ):
        _require(
            {count_field, total_field}.issubset(table.columns),
            f"{cohort}: conditional-total semantic fields are incomplete",
        )
        counts = pd.to_numeric(table[count_field], errors="coerce")
        totals = pd.to_numeric(table[total_field], errors="coerce")
        count_present = (
            table[count_field].notna()
            & table[count_field].astype(str).str.strip().ne("")
        )
        total_present = (
            table[total_field].notna()
            & table[total_field].astype(str).str.strip().ne("")
        )
        _require(
            counts[count_present].notna().all() and totals[total_present].notna().all(),
            f"{cohort}: conditional-total semantic fields are not numeric",
        )
        fabricated_zero = (
            count_present & total_present & counts.gt(0.0) & totals.eq(0.0)
        )
        if fabricated_zero.any():
            task_ids = (
                table.loc[fabricated_zero, "task_id"].astype(str).head(5).tolist()
            )
            raise MultitraitAtlasError(
                f"{cohort}/{total_field}: zero with positive identity count is a "
                f"missing endpoint-complete measurement, not a biological zero; "
                f"tasks={task_ids}"
            )

    if "hair_length_measurement_fraction" in table.columns:
        counts = pd.to_numeric(table["hair_count"], errors="coerce")
        raw_fraction = table["hair_length_measurement_fraction"]
        fraction_present = (
            raw_fraction.notna() & raw_fraction.astype(str).str.strip().ne("")
        )
        fractions = pd.to_numeric(raw_fraction, errors="coerce")
        _require(
            fractions[fraction_present].notna().all(),
            f"{cohort}/hair_length_measurement_fraction: non-null value is not numeric",
        )
        undefined_fraction = counts.eq(0.0) & fraction_present
        if undefined_fraction.any():
            task_ids = (
                table.loc[undefined_fraction, "task_id"].astype(str).head(5).tolist()
            )
            raise MultitraitAtlasError(
                f"{cohort}/hair_length_measurement_fraction: zero-denominator "
                f"support must be null; tasks={task_ids}"
            )

    if "attachment_axis_valid_fraction" in table.columns:
        counts = pd.to_numeric(table["hair_count"], errors="coerce")
        raw_fraction = table["attachment_axis_valid_fraction"]
        fraction_present = (
            raw_fraction.notna() & raw_fraction.astype(str).str.strip().ne("")
        )
        fractions = pd.to_numeric(raw_fraction, errors="coerce")
        _require(
            fractions[fraction_present].notna().all(),
            f"{cohort}/attachment_axis_valid_fraction: non-null value is not numeric",
        )
        invalid_fraction = fraction_present & (
            ~np.isfinite(fractions) | fractions.lt(0.0) | fractions.gt(1.0)
        )
        if invalid_fraction.any():
            task_ids = (
                table.loc[invalid_fraction, "task_id"].astype(str).head(5).tolist()
            )
            raise MultitraitAtlasError(
                f"{cohort}/attachment_axis_valid_fraction: observed support must be "
                f"finite within [0,1]; tasks={task_ids}"
            )
        undefined_fraction = counts.eq(0.0) & fraction_present
        if undefined_fraction.any():
            task_ids = (
                table.loc[undefined_fraction, "task_id"].astype(str).head(5).tolist()
            )
            raise MultitraitAtlasError(
                f"{cohort}/attachment_axis_valid_fraction: zero-denominator support "
                f"must be null; tasks={task_ids}"
            )
        missing_fraction = counts.gt(0.0) & ~fraction_present
        if missing_fraction.any():
            task_ids = (
                table.loc[missing_fraction, "task_id"].astype(str).head(5).tolist()
            )
            raise MultitraitAtlasError(
                f"{cohort}/attachment_axis_valid_fraction: positive identity count "
                f"requires observed support; tasks={task_ids}"
            )


def _bind_cohort_metadata_to_canonical_traits(
    *,
    cohort_table: pd.DataFrame,
    canonical_image_traits: pd.DataFrame,
    root_descriptor_fields: Sequence[str],
    hair_descriptor_fields: Sequence[str],
    cohort: str,
) -> pd.DataFrame:
    """Join cohort membership/biology metadata to the canonical 32-trait table.

    The sealed cohort tables carry D15 condition and eligibility fields but may
    expose only the prespecified analysis endpoints.  The canonical image-level
    export carries all 19 root traits.  Hair traits remain cohort-table values;
    they are never filled or overwritten from the canonical export.  Both
    sources are required and overlapping root cells must agree exactly before
    the full atlas is materialized.
    """

    metadata_fields = (
        "task_id",
        "source_image_sha256",
        "experiment_key",
        "study_role",
        "condition_code",
        "formal_statistics_eligible",
    )
    _require(
        set(metadata_fields).issubset(cohort_table.columns),
        f"{cohort}: cohort membership/biology metadata is incomplete",
    )
    _require(
        {"task_id", "source_image_sha256", *root_descriptor_fields}.issubset(
            canonical_image_traits.columns
        ),
        "canonical_image_traits does not expose all 19 root traits",
    )
    _require(
        set(hair_descriptor_fields).issubset(cohort_table.columns),
        f"{cohort}: cohort table does not expose all 13 hair traits",
    )
    _validate_conditional_total_null_semantics(cohort_table, cohort=cohort)
    _require(
        cohort_table["task_id"].nunique() == len(cohort_table)
        and canonical_image_traits["task_id"].nunique()
        == len(canonical_image_traits),
        f"{cohort}: cohort/canonical source units are not unique",
    )
    canonical = canonical_image_traits.set_index(
        canonical_image_traits["task_id"].astype(str), drop=False
    )
    cohort_ids = cohort_table["task_id"].astype(str)
    _require(
        set(cohort_ids).issubset(set(canonical.index)),
        f"{cohort}: cohort source units are absent from canonical_image_traits",
    )
    projected = canonical.loc[cohort_ids].reset_index(drop=True)
    _require(
        cohort_table["source_image_sha256"].astype(str).reset_index(drop=True).equals(
            projected["source_image_sha256"].astype(str).reset_index(drop=True)
        ),
        f"{cohort}: source-image identity differs from canonical_image_traits",
    )
    overlapping_fields = [
        field for field in root_descriptor_fields if field in cohort_table.columns
    ]
    for field in overlapping_fields:
        cohort_numeric = pd.to_numeric(cohort_table[field], errors="coerce").reset_index(
            drop=True
        )
        canonical_numeric = pd.to_numeric(projected[field], errors="coerce").reset_index(
            drop=True
        )
        _require(
            cohort_numeric.isna().equals(canonical_numeric.isna())
            and np.array_equal(
                cohort_numeric.fillna(0.0).to_numpy(dtype=float),
                canonical_numeric.fillna(0.0).to_numpy(dtype=float),
            ),
            f"{cohort}/{field}: cohort and canonical trait cells differ",
        )
    bound = cohort_table.loc[:, list(metadata_fields)].reset_index(drop=True).copy()
    for field in root_descriptor_fields:
        bound[field] = projected[field].reset_index(drop=True)
    for field in hair_descriptor_fields:
        bound[field] = cohort_table[field].reset_index(drop=True)
    return bound


def _summary(table: pd.DataFrame, field: str, cohort: str) -> dict[str, Any]:
    values = _numeric_values(table, field, cohort)
    total = int(len(table))
    n = int(values.size)
    _require(total > 0 and 0 <= n <= total, f"{cohort}/{field}: invalid support denominator")
    if n == 0:
        return {
            "source_unit_total": total,
            "non_null_source_unit_n": 0,
            "support_fraction": 0.0,
            "summary_status": "not_estimable_no_finite_source_units",
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "source_unit_total": total,
        "non_null_source_unit_n": n,
        "support_fraction": float(n / total),
        "summary_status": "estimated_descriptive_source_unit_summary",
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _condition_summaries(
    table: pd.DataFrame,
    field: str,
) -> dict[str, dict[str, Any]]:
    """Return raw formal-D15 condition distributions before model fitting."""

    required = {
        "task_id",
        "experiment_key",
        "study_role",
        "condition_code",
        "formal_statistics_eligible",
        field,
    }
    _require(required.issubset(table.columns), f"condition-summary columns missing for {field}")
    selected = table[
        (table["experiment_key"].astype(str) == "D15_8d")
        & (table["study_role"].astype(str) == "rhd6_factorial_8d_primary")
        & table["condition_code"].astype(str).isin(GROUP_ORDER)
        & _bool_series(table["formal_statistics_eligible"])
    ].copy()
    result: dict[str, dict[str, Any]] = {}
    for condition in GROUP_ORDER:
        group = selected[selected["condition_code"].astype(str) == condition]
        total = int(len(group))
        _require(total > 0, f"{field}/{condition}: formal D15 condition is empty")
        values = _numeric_values(group, field, f"formal_D15/{condition}")
        n = int(values.size)
        if n == 0:
            result[condition] = {
                "source_unit_total": total,
                "non_null_source_unit_n": 0,
                "observability_fraction": 0.0,
                "summary_status": "not_estimated_no_finite_source_units",
                "median": None,
                "q25": None,
                "q75": None,
                "iqr": None,
                "minimum": None,
                "maximum": None,
                "not_estimable_reason": CONDITION_NOT_ESTIMATED_REASON,
                "raw_unadjusted": True,
                "unit_of_analysis": CONDITION_ROW_UNIT,
            }
            continue
        q25 = float(np.quantile(values, 0.25))
        q75 = float(np.quantile(values, 0.75))
        result[condition] = {
            "source_unit_total": total,
            "non_null_source_unit_n": n,
            "observability_fraction": float(n / total),
            "summary_status": CONDITION_SUMMARY_STATUS,
            "median": float(np.median(values)),
            "q25": q25,
            "q75": q75,
            "iqr": float(q75 - q25),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "not_estimable_reason": None,
            "raw_unadjusted": True,
            "unit_of_analysis": CONDITION_ROW_UNIT,
        }
    return result


def _analysis_scope(
    condition_summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, int], int]:
    by_condition = {
        condition: int(condition_summaries[condition]["non_null_source_unit_n"])
        for condition in GROUP_ORDER
    }
    return by_condition, int(sum(by_condition.values()))


def _effect_records(
    analysis: pd.DataFrame,
    *,
    cohort: str,
    field: str,
    expected_n: int,
    condition_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    required = {
        "cohort",
        "endpoint",
        "effect",
        "n",
        "estimate",
        "ci95_low",
        "ci95_high",
        "effect_scale",
        "raw_effect_estimate",
        "raw_effect_ci95_low",
        "raw_effect_ci95_high",
        "raw_effect_estimand",
        "raw_effect_interval_method",
        "raw_effect_bootstrap_replicates",
        "raw_effect_bootstrap_seed",
        "standardized_effect",
        "standardized_ci95_low",
        "standardized_ci95_high",
        "causal_treatment_claim_allowed",
    }
    _require(required.issubset(analysis.columns), f"{cohort}: analysis columns incomplete")
    selected = analysis[
        (analysis["cohort"].astype(str) == cohort)
        & (analysis["endpoint"].astype(str) == field)
        & analysis["effect"].astype(str).isin(EFFECT_NAME_TO_KEY)
    ]
    if field not in PRIMARY_ENDPOINTS:
        _require(len(selected) == 0, f"{field}: effect outside the prespecified family was supplied")
        return {
            effect_key: {
                "status": "not_estimated",
                "estimate": None,
                "ci95_low": None,
                "ci95_high": None,
                "endpoint_n": None,
                "effect_scale": None,
                "raw_effect_estimate": None,
                "raw_effect_ci95_low": None,
                "raw_effect_ci95_high": None,
                "raw_effect_estimand": None,
                "raw_effect_interval_method": None,
                "raw_effect_bootstrap_replicates": None,
                "raw_effect_bootstrap_seed": None,
                "standardized_effect": None,
                "standardized_ci95_low": None,
                "standardized_ci95_high": None,
                "not_estimable_reason": NOT_ESTIMATED_REASON,
            }
            for effect_key in EFFECT_KEYS
        }
    _require(len(selected) == 3, f"{cohort}/{field}: fixed three-effect family is incomplete")
    result: dict[str, dict[str, Any]] = {}
    for record in selected.to_dict("records"):
        key = EFFECT_NAME_TO_KEY[str(record["effect"])]
        _require(key not in result, f"{cohort}/{field}: duplicate effect {key}")
        _require(
            record.get("causal_treatment_claim_allowed") in {False, "False", "false", 0, "0"},
            f"{cohort}/{field}/{key}: causal claim is forbidden",
        )
        n = int(record["n"])
        _require(
            n == expected_n,
            f"{cohort}/{field}/{key}: effect denominator does not close to source units",
        )
        values = [float(record[name]) for name in ("estimate", "ci95_low", "ci95_high")]
        _require(all(math.isfinite(value) for value in values), f"{cohort}/{field}/{key}: non-finite effect")
        _require(values[1] <= values[2], f"{cohort}/{field}/{key}: reversed interval")
        raw_values = [
            float(record[name])
            for name in (
                "raw_effect_estimate",
                "raw_effect_ci95_low",
                "raw_effect_ci95_high",
                "standardized_effect",
                "standardized_ci95_low",
                "standardized_ci95_high",
            )
        ]
        _require(
            all(math.isfinite(value) for value in raw_values)
            and raw_values[1] <= raw_values[2]
            and raw_values[4] <= raw_values[5],
            f"{cohort}/{field}/{key}: raw or standardized companion is invalid",
        )
        raw_estimand = str(record["raw_effect_estimand"])
        raw_interval_method = str(record["raw_effect_interval_method"])
        raw_replicates = int(record["raw_effect_bootstrap_replicates"])
        raw_seed_value = record["raw_effect_bootstrap_seed"]
        raw_seed = None if pd.isna(raw_seed_value) else int(raw_seed_value)
        if field == H11_ENDPOINT:
            _require(
                raw_estimand == RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
                and raw_interval_method == RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
                and raw_replicates == H11_RAW_BOOTSTRAP_REPLICATES
                and raw_seed
                == raw_median_bootstrap_seed(
                    seed=H11_RAW_BOOTSTRAP_BASE_SEED,
                    field=H11_ENDPOINT,
                    component="continuous",
                ),
                f"{cohort}/{field}/{key}: H11 raw-median companion drift",
            )
            medians = [
                condition_summaries[condition]["median"]
                for condition in GROUP_ORDER
            ]
            _require(
                all(
                    isinstance(value, (int, float))
                    and math.isfinite(float(value))
                    for value in medians
                ),
                f"{cohort}/{field}: H11 cell median is not estimable",
            )
            ev22, ev30, oe22, oe30 = (float(value) for value in medians)
            expected_raw = {
                "OE_vs_EV": 0.5 * ((oe22 - ev22) + (oe30 - ev30)),
                "30C_vs_22C": 0.5 * ((ev30 - ev22) + (oe30 - oe22)),
                "interaction": (oe30 - oe22) - (ev30 - ev22),
            }
            _require(
                math.isclose(
                    raw_values[0], expected_raw[key], rel_tol=0.0, abs_tol=1e-12
                ),
                f"{cohort}/{field}/{key}: H11 raw effect is not the four-cell median contrast",
            )
        else:
            _require(
                raw_estimand == RAW_EFFECT_OLS_MEAN_CONTRAST
                and raw_interval_method == RAW_EFFECT_HC3_INTERVAL
                and raw_replicates == 0
                and raw_seed is None,
                f"{cohort}/{field}/{key}: raw-mean companion drift",
            )
        result[key] = {
            "status": "estimated_fixed_15_effect_family",
            "estimate": values[0],
            "ci95_low": values[1],
            "ci95_high": values[2],
            "endpoint_n": n,
            "effect_scale": str(record["effect_scale"]),
            "raw_effect_estimate": raw_values[0],
            "raw_effect_ci95_low": raw_values[1],
            "raw_effect_ci95_high": raw_values[2],
            "raw_effect_estimand": raw_estimand,
            "raw_effect_interval_method": raw_interval_method,
            "raw_effect_bootstrap_replicates": raw_replicates,
            "raw_effect_bootstrap_seed": raw_seed,
            "standardized_effect": raw_values[3],
            "standardized_ci95_low": raw_values[4],
            "standardized_ci95_high": raw_values[5],
            "not_estimable_reason": None,
        }
    _require(set(result) == set(EFFECT_KEYS), f"{cohort}/{field}: effect keys changed")
    return result


def build_multitrait_atlas(
    *,
    trait_contract: Mapping[str, Any],
    clean_traits: pd.DataFrame,
    full_traits: pd.DataFrame,
    canonical_image_traits: pd.DataFrame,
    primary_analysis: pd.DataFrame,
    sensitivity_analysis: pd.DataFrame,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build and seal the deterministic 32-trait clean/full atlas."""

    expected_sources = {
        "trait_contract",
        "clean_traits",
        "full_traits",
        "canonical_image_traits",
        "analysis_primary_table",
        "analysis_sensitivity_table",
    }
    _require(set(source_sha256) == expected_sources, "atlas source hash roles changed")
    _require(all(_is_sha256(value) for value in source_sha256.values()), "atlas source SHA-256 is invalid")
    descriptors_contract = _canonical_descriptors(trait_contract)
    root_descriptor_fields = [
        record["field"]
        for record in descriptors_contract
        if record["trait_family"] == "primary_root"
    ]
    hair_descriptor_fields = [
        record["field"]
        for record in descriptors_contract
        if record["trait_family"] == "root_hair"
    ]
    _require(
        len(canonical_image_traits) == 283,
        "canonical_image_traits must contain the exact 283 source units",
    )
    bound_tables: dict[str, pd.DataFrame] = {}
    for cohort, table, expected_total in (
        (COHORTS[0], clean_traits, 261),
        (COHORTS[1], full_traits, 283),
    ):
        _require("task_id" in table.columns, f"{cohort}: task_id missing")
        _require(table["task_id"].nunique() == len(table), f"{cohort}: source units are not unique")
        _require(len(table) == expected_total, f"{cohort}: expected {expected_total} source units")
        bound_tables[cohort] = _bind_cohort_metadata_to_canonical_traits(
            cohort_table=table,
            canonical_image_traits=canonical_image_traits,
            root_descriptor_fields=root_descriptor_fields,
            hair_descriptor_fields=hair_descriptor_fields,
            cohort=cohort,
        )
    _require(
        set(clean_traits["task_id"].astype(str)).issubset(set(full_traits["task_id"].astype(str))),
        "clean261 is not a source-unit subset of full283",
    )

    descriptors: list[dict[str, Any]] = []
    for ordinal, descriptor in enumerate(descriptors_contract, start=1):
        field = descriptor["field"]
        cohorts: dict[str, Any] = {}
        for cohort, table, analysis in (
            (COHORTS[0], bound_tables[COHORTS[0]], primary_analysis),
            (COHORTS[1], bound_tables[COHORTS[1]], sensitivity_analysis),
        ):
            condition_summaries = _condition_summaries(table, field)
            by_condition, analysis_n = _analysis_scope(condition_summaries)
            effects = _effect_records(
                analysis,
                cohort=cohort,
                field=field,
                expected_n=analysis_n,
                condition_summaries=condition_summaries,
            )
            cohorts[cohort] = {
                **_summary(table, field, cohort),
                "effect_source_unit_n_by_condition": by_condition,
                "effect_source_unit_n": analysis_n,
                "condition_summaries": condition_summaries,
                "effects": effects,
            }
        descriptors.append({"ordinal": ordinal, **descriptor, "cohorts": cohorts})

    estimated = sum(
        record["status"] == "estimated_fixed_15_effect_family"
        for descriptor in descriptors
        for cohort in COHORTS
        for record in descriptor["cohorts"][cohort]["effects"].values()
    )
    effect_slots = len(descriptors) * len(COHORTS) * len(EFFECT_KEYS)
    condition_summary_slots = len(descriptors) * len(COHORTS) * len(GROUP_ORDER)
    estimated_condition_summaries = sum(
        summary["summary_status"] == CONDITION_SUMMARY_STATUS
        for descriptor in descriptors
        for cohort in COHORTS
        for summary in descriptor["cohorts"][cohort]["condition_summaries"].values()
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_source_derived_32_trait_atlas",
        "row_unit": "one visible primary root per canonical source image",
        "descriptor_count": 32,
        "root_descriptor_count": 19,
        "hair_descriptor_count": 13,
        "cohort_order": list(COHORTS),
        "effect_order": list(EFFECT_KEYS),
        "condition_order": list(GROUP_ORDER),
        "measurement_family_order": list(MEASUREMENT_FAMILY_ORDER),
        "prespecified_inferential_endpoint_fields": list(PRIMARY_ENDPOINTS),
        "effect_slot_count": effect_slots,
        "estimated_effect_slot_count": int(estimated),
        "not_estimated_effect_slot_count": int(effect_slots - estimated),
        "condition_summary_slot_count": condition_summary_slots,
        "estimated_condition_summary_slot_count": int(estimated_condition_summaries),
        "not_estimated_condition_summary_slot_count": int(
            condition_summary_slots - estimated_condition_summaries
        ),
        "source_sha256": dict(source_sha256),
        "descriptors": descriptors,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    _require(estimated == 30 and effect_slots - estimated == 162, "15-effect clean/full closure failed")
    payload["atlas_identity_sha256"] = sha256_json(payload)
    return payload


def validate_multitrait_atlas_structure(
    payload: Mapping[str, Any],
    *,
    trait_contract: Mapping[str, Any],
    expected_source_sha256: Mapping[str, str],
) -> None:
    """Validate the sealed atlas without trusting its summary counters."""

    _require(payload.get("schema_version") == SCHEMA_VERSION, "multitrait atlas schema changed")
    _require(
        "biological_response_family_order" not in payload,
        "legacy v1 biological_response_family_order key is forbidden",
    )
    declared_families = payload.get("measurement_family_order")
    _require(
        not (
            isinstance(declared_families, Sequence)
            and not isinstance(declared_families, (str, bytes))
            and any(value in LEGACY_V1_FAMILY_VALUES for value in declared_families)
        ),
        "legacy v1 measurement-family value is forbidden",
    )
    identity = payload.get("atlas_identity_sha256")
    _require(_is_sha256(identity), "multitrait atlas identity missing")
    unsigned = deepcopy(dict(payload))
    unsigned.pop("atlas_identity_sha256", None)
    _require(sha256_json(unsigned) == identity, "multitrait atlas sealed hash drift")
    _require(payload.get("source_sha256") == dict(expected_source_sha256), "multitrait atlas source hash drift")
    _require(payload.get("blind_images_used") == 0, "multitrait atlas blind guard changed")
    _require(payload.get("root_cap_region_statistics_included") is False, "root-cap region entered atlas")
    expected = _canonical_descriptors(trait_contract)
    descriptors = payload.get("descriptors")
    _require(isinstance(descriptors, list) and len(descriptors) == 32, "multitrait atlas omits canonical traits")
    observed_ids: list[str] = []
    estimated = 0
    not_estimated = 0
    estimated_condition_summaries = 0
    not_estimated_condition_summaries = 0
    for ordinal, (record, canonical) in enumerate(zip(descriptors, expected, strict=True), start=1):
        _require(isinstance(record, Mapping), f"atlas descriptor {ordinal}: malformed")
        _require(
            "biological_response_family" not in record,
            f"atlas descriptor {ordinal}: legacy v1 biological_response_family key is forbidden",
        )
        _require(
            record.get("measurement_family") not in LEGACY_V1_FAMILY_VALUES,
            f"atlas descriptor {ordinal}: legacy v1 measurement-family value is forbidden",
        )
        _require(record.get("ordinal") == ordinal, f"atlas descriptor {ordinal}: order changed")
        _require(
            all(record.get(key) == value for key, value in canonical.items()),
            f"atlas descriptor {ordinal}: trait contract metadata drift",
        )
        observed_ids.append(str(record.get("trait_id")))
        cohorts = record.get("cohorts")
        _require(isinstance(cohorts, Mapping) and set(cohorts) == set(COHORTS), f"{canonical['trait_id']}: cohort route changed")
        for cohort in COHORTS:
            item = cohorts[cohort]
            _require(isinstance(item, Mapping), f"{canonical['trait_id']}/{cohort}: malformed")
            total = item.get("source_unit_total")
            n = item.get("non_null_source_unit_n")
            fraction = item.get("support_fraction")
            expected_total = 261 if cohort == COHORTS[0] else 283
            _require(total == expected_total and isinstance(n, int) and 0 <= n <= total, f"{canonical['trait_id']}/{cohort}: support denominator changed")
            _require(
                isinstance(fraction, (int, float))
                and math.isfinite(float(fraction))
                and math.isclose(float(fraction), n / total, rel_tol=1e-12, abs_tol=1e-12),
                f"{canonical['trait_id']}/{cohort}: support fraction denominator does not close",
            )
            by_condition = item.get("effect_source_unit_n_by_condition")
            analysis_n = item.get("effect_source_unit_n")
            condition_summaries = item.get("condition_summaries")
            _require(
                isinstance(condition_summaries, Mapping)
                and list(condition_summaries) == list(GROUP_ORDER),
                f"{canonical['trait_id']}/{cohort}: condition-summary slots changed",
            )
            observed_condition_n: dict[str, int] = {}
            for condition in GROUP_ORDER:
                summary = condition_summaries[condition]
                _require(
                    isinstance(summary, Mapping),
                    f"{canonical['trait_id']}/{cohort}/{condition}: condition summary malformed",
                )
                condition_total = summary.get("source_unit_total")
                condition_n = summary.get("non_null_source_unit_n")
                observability = summary.get("observability_fraction")
                _require(
                    isinstance(condition_total, int)
                    and condition_total > 0
                    and isinstance(condition_n, int)
                    and 0 <= condition_n <= condition_total
                    and isinstance(observability, (int, float))
                    and math.isfinite(float(observability))
                    and math.isclose(
                        float(observability),
                        condition_n / condition_total,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                    and summary.get("raw_unadjusted") is True
                    and summary.get("unit_of_analysis") == CONDITION_ROW_UNIT,
                    f"{canonical['trait_id']}/{cohort}/{condition}: observability denominator changed",
                )
                observed_condition_n[condition] = condition_n
                distribution = [summary.get(key) for key in ("median", "q25", "q75", "iqr", "minimum", "maximum")]
                if condition_n == 0:
                    _require(
                        summary.get("summary_status")
                        == "not_estimated_no_finite_source_units"
                        and summary.get("not_estimable_reason")
                        == CONDITION_NOT_ESTIMATED_REASON
                        and all(value is None for value in distribution),
                        f"{canonical['trait_id']}/{cohort}/{condition}: missing-data reason/statistics changed",
                    )
                    not_estimated_condition_summaries += 1
                else:
                    _require(
                        summary.get("summary_status") == CONDITION_SUMMARY_STATUS
                        and summary.get("not_estimable_reason") is None
                        and all(
                            isinstance(value, (int, float))
                            and math.isfinite(float(value))
                            for value in distribution
                        )
                        and float(summary["minimum"])
                        <= float(summary["q25"])
                        <= float(summary["median"])
                        <= float(summary["q75"])
                        <= float(summary["maximum"])
                        and math.isclose(
                            float(summary["iqr"]),
                            float(summary["q75"]) - float(summary["q25"]),
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        ),
                        f"{canonical['trait_id']}/{cohort}/{condition}: raw median/IQR changed",
                    )
                    estimated_condition_summaries += 1
            _require(
                isinstance(by_condition, Mapping)
                and set(by_condition) == set(GROUP_ORDER)
                and all(isinstance(value, int) and value >= 0 for value in by_condition.values())
                and dict(by_condition) == observed_condition_n
                and analysis_n == sum(by_condition.values()),
                f"{canonical['trait_id']}/{cohort}: effect denominator does not close",
            )
            effects = item.get("effects")
            _require(isinstance(effects, Mapping) and set(effects) == set(EFFECT_KEYS), f"{canonical['trait_id']}/{cohort}: effect slots changed")
            for effect_key, effect in effects.items():
                _require(isinstance(effect, Mapping), f"{canonical['trait_id']}/{cohort}/{effect_key}: malformed")
                if canonical["field"] in PRIMARY_ENDPOINTS:
                    _require(
                        effect.get("status") == "estimated_fixed_15_effect_family"
                        and effect.get("endpoint_n") == analysis_n
                        and effect.get("not_estimable_reason") is None
                        and all(
                            isinstance(effect.get(key), (int, float))
                            and math.isfinite(float(effect[key]))
                            for key in ("estimate", "ci95_low", "ci95_high")
                        ),
                        f"{canonical['trait_id']}/{cohort}/{effect_key}: fixed effect is incomplete",
                    )
                    _require(
                        float(effect["ci95_low"]) <= float(effect["ci95_high"])
                        and all(
                            isinstance(effect.get(key), (int, float))
                            and math.isfinite(float(effect[key]))
                            for key in (
                                "raw_effect_estimate",
                                "raw_effect_ci95_low",
                                "raw_effect_ci95_high",
                                "standardized_effect",
                                "standardized_ci95_low",
                                "standardized_ci95_high",
                            )
                        )
                        and float(effect["raw_effect_ci95_low"])
                        <= float(effect["raw_effect_ci95_high"])
                        and float(effect["standardized_ci95_low"])
                        <= float(effect["standardized_ci95_high"]),
                        f"{canonical['trait_id']}/{cohort}/{effect_key}: raw companion is incomplete",
                    )
                    raw_seed = effect.get("raw_effect_bootstrap_seed")
                    if canonical["field"] == H11_ENDPOINT:
                        _require(
                            effect.get("raw_effect_estimand")
                            == RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
                            and effect.get("raw_effect_interval_method")
                            == RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
                            and effect.get("raw_effect_bootstrap_replicates")
                            == H11_RAW_BOOTSTRAP_REPLICATES
                            and raw_seed
                            == raw_median_bootstrap_seed(
                                seed=H11_RAW_BOOTSTRAP_BASE_SEED,
                                field=H11_ENDPOINT,
                                component="continuous",
                            ),
                            f"{canonical['trait_id']}/{cohort}/{effect_key}: H11 raw-median companion drift",
                        )
                    else:
                        _require(
                            effect.get("raw_effect_estimand")
                            == RAW_EFFECT_OLS_MEAN_CONTRAST
                            and effect.get("raw_effect_interval_method")
                            == RAW_EFFECT_HC3_INTERVAL
                            and effect.get("raw_effect_bootstrap_replicates") == 0
                            and raw_seed is None,
                            f"{canonical['trait_id']}/{cohort}/{effect_key}: raw-mean companion drift",
                        )
                    estimated += 1
                else:
                    _require(
                        effect.get("status") == "not_estimated"
                        and effect.get("not_estimable_reason") == NOT_ESTIMATED_REASON
                        and all(
                            effect.get(key) is None
                            for key in (
                                "estimate",
                                "ci95_low",
                                "ci95_high",
                                "endpoint_n",
                                "effect_scale",
                                "raw_effect_estimate",
                                "raw_effect_ci95_low",
                                "raw_effect_ci95_high",
                                "raw_effect_estimand",
                                "raw_effect_interval_method",
                                "raw_effect_bootstrap_replicates",
                                "raw_effect_bootstrap_seed",
                                "standardized_effect",
                                "standardized_ci95_low",
                                "standardized_ci95_high",
                            )
                        ),
                        f"{canonical['trait_id']}/{cohort}/{effect_key}: invented effect outside fixed family",
                    )
                    not_estimated += 1
    _require(len(set(observed_ids)) == 32, "multitrait atlas duplicates canonical traits")
    _require(
        payload.get("descriptor_count") == 32
        and payload.get("root_descriptor_count") == 19
        and payload.get("hair_descriptor_count") == 13
        and payload.get("effect_slot_count") == 192
        and payload.get("estimated_effect_slot_count") == estimated == 30
        and payload.get("not_estimated_effect_slot_count") == not_estimated == 162,
        "multitrait atlas counters do not close",
    )
    _require(
        payload.get("condition_summary_slot_count") == 32 * 2 * 4 == 256
        and payload.get("estimated_condition_summary_slot_count")
        == estimated_condition_summaries
        and payload.get("not_estimated_condition_summary_slot_count")
        == not_estimated_condition_summaries
        and estimated_condition_summaries + not_estimated_condition_summaries
        == 256,
        "multitrait atlas condition-summary counters do not close",
    )
    _require(
        payload.get("cohort_order") == list(COHORTS)
        and payload.get("effect_order") == list(EFFECT_KEYS)
        and payload.get("condition_order") == list(GROUP_ORDER)
        and payload.get("measurement_family_order")
        == list(MEASUREMENT_FAMILY_ORDER)
        and payload.get("prespecified_inferential_endpoint_fields")
        == list(PRIMARY_ENDPOINTS),
        "multitrait atlas declared order/fixed family changed",
    )


def validate_multitrait_atlas_against_sources(
    payload: Mapping[str, Any],
    *,
    trait_contract: Mapping[str, Any],
    clean_traits: pd.DataFrame,
    full_traits: pd.DataFrame,
    canonical_image_traits: pd.DataFrame,
    primary_analysis: pd.DataFrame,
    sensitivity_analysis: pd.DataFrame,
    source_sha256: Mapping[str, str],
) -> None:
    """Recompute the complete atlas and require canonical equality."""

    validate_multitrait_atlas_structure(
        payload,
        trait_contract=trait_contract,
        expected_source_sha256=source_sha256,
    )
    expected = build_multitrait_atlas(
        trait_contract=trait_contract,
        clean_traits=clean_traits,
        full_traits=full_traits,
        canonical_image_traits=canonical_image_traits,
        primary_analysis=primary_analysis,
        sensitivity_analysis=sensitivity_analysis,
        source_sha256=source_sha256,
    )
    _require(dict(payload) == expected, "multitrait atlas differs from source-derived canonical payload")


def heatmap_matrices(payload: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return trait labels, clean/full coverage, and clean/full effect matrices."""

    descriptors = payload["descriptors"]
    labels = [f"{record['trait_id']} {record['field']}" for record in descriptors]
    coverage = np.asarray(
        [
            [record["cohorts"][cohort]["support_fraction"] for cohort in COHORTS]
            for record in descriptors
        ],
        dtype=float,
    )
    effects = np.full((32, len(COHORTS) * len(EFFECT_KEYS)), np.nan, dtype=float)
    for row, record in enumerate(descriptors):
        for cohort_index, cohort in enumerate(COHORTS):
            for effect_index, effect_key in enumerate(EFFECT_KEYS):
                effect = record["cohorts"][cohort]["effects"][effect_key]
                if effect["status"] == "estimated_fixed_15_effect_family":
                    effects[row, cohort_index * len(EFFECT_KEYS) + effect_index] = float(effect["estimate"])
    return labels, coverage, effects


def descriptive_heatmap_matrices(
    payload: Mapping[str, Any],
    *,
    cohort: str = COHORTS[0],
) -> dict[str, Any]:
    """Return publication matrices that foreground all 32 raw trait summaries.

    Native units differ across traits, so condition medians are standardized only
    within each descriptor.  IQR is likewise divided by the largest finite IQR
    within that descriptor.  Coverage remains the observed non-null fraction.
    The inferential panel is restricted to the five prespecified endpoints rather
    than rendering the 162 deliberately unmodeled slots as a dominant grey field.
    """

    _require(cohort in COHORTS, f"unknown descriptive heatmap cohort: {cohort}")
    descriptors = payload["descriptors"]
    _require(len(descriptors) == 32, "descriptive heatmap requires all 32 descriptors")
    labels = [f"{record['trait_id']} {record['field']}" for record in descriptors]
    medians = np.full((32, len(GROUP_ORDER)), np.nan, dtype=float)
    relative_iqrs = np.full_like(medians, np.nan)
    condition_coverage = np.full_like(medians, np.nan)

    for row, record in enumerate(descriptors):
        summaries = record["cohorts"][cohort]["condition_summaries"]
        _require(
            list(summaries) == list(GROUP_ORDER),
            f"{record['trait_id']}: condition order changed",
        )
        iqrs = np.full(len(GROUP_ORDER), np.nan, dtype=float)
        for column, condition in enumerate(GROUP_ORDER):
            summary = summaries[condition]
            condition_coverage[row, column] = float(
                summary["observability_fraction"]
            )
            if summary["summary_status"] != CONDITION_SUMMARY_STATUS:
                continue
            medians[row, column] = float(summary["median"])
            iqrs[column] = float(summary["iqr"])

        finite_iqrs = iqrs[np.isfinite(iqrs)]
        if finite_iqrs.size:
            maximum_iqr = float(np.max(finite_iqrs))
            finite = np.isfinite(iqrs)
            relative_iqrs[row, finite] = (
                iqrs[finite] / maximum_iqr if maximum_iqr > 0 else 0.0
            )

    standardized_medians = np.full_like(medians, np.nan)
    for row in range(medians.shape[0]):
        finite = np.isfinite(medians[row])
        if not finite.any():
            continue
        values = medians[row, finite]
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=0))
        standardized_medians[row, finite] = (
            (values - mean) / standard_deviation
            if standard_deviation > 0
            else 0.0
        )

    by_field = {record["field"]: record for record in descriptors}
    _require(
        set(PRIMARY_ENDPOINTS).issubset(by_field),
        "prespecified endpoint descriptors are incomplete",
    )
    effect_labels: list[str] = []
    effect_matrix = np.full(
        (len(PRIMARY_ENDPOINTS), len(COHORTS) * len(EFFECT_KEYS)),
        np.nan,
        dtype=float,
    )
    for row, field in enumerate(PRIMARY_ENDPOINTS):
        record = by_field[field]
        effect_labels.append(f"{record['trait_id']} {record['field']}")
        for cohort_index, effect_cohort in enumerate(COHORTS):
            for effect_index, effect_key in enumerate(EFFECT_KEYS):
                effect = record["cohorts"][effect_cohort]["effects"][effect_key]
                _require(
                    effect["status"] == "estimated_fixed_15_effect_family",
                    f"{effect_cohort}/{field}/{effect_key}: fixed effect missing",
                )
                effect_matrix[
                    row,
                    cohort_index * len(EFFECT_KEYS) + effect_index,
                ] = float(effect["estimate"])

    return {
        "trait_labels": labels,
        "condition_labels": list(GROUP_ORDER),
        "standardized_medians": standardized_medians,
        "relative_iqrs": relative_iqrs,
        "condition_coverage": condition_coverage,
        "effect_trait_labels": effect_labels,
        "effect_estimates": effect_matrix,
        "descriptive_cohort": cohort,
    }
