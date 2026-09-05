#!/usr/bin/env python
"""Audit the narrowly scoped Stage-22 H11 raw-median correction.

The locked biological model specification names the H11 raw companion as a
balanced contrast of four condition-cell medians.  The historical PHAxis-native
table instead carried the effect-coded raw *mean* contrast.  This CPU-only
producer reruns the fixed 15-effect family directly on the already locked
clean/full trait tables, proves that protected inference is unchanged, and
independently recomputes the corrected H11 point estimates and bootstrap
intervals.  It never reads images, annotations, condition-routing inputs, or
blind/final-validation data.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.biological_analysis import (  # noqa: E402
    RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
    RAW_EFFECT_HC3_INTERVAL,
    RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
    RAW_EFFECT_OLS_MEAN_CONTRAST,
)
from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402


SCHEMA_VERSION = "PHAxis-stage22-H11-raw-median-amendment-audit-1.0"
ARTIFACT_ROLE = "h11_raw_median_contract_amendment_current"
H11_ENDPOINT = "local_median_hair_length_um_1_4mm"
EFFECTS = (
    "construct_OE_minus_EV",
    "temperature_30C_minus_22C",
    "construct_by_temperature_interaction",
)
CELLS = (
    ("RHD6-EV", 22, "EV22"),
    ("RHD6-EV", 30, "EV30"),
    ("RHD6-OE", 22, "OE22"),
    ("RHD6-OE", 30, "OE30"),
)
PRIMARY_TABLE = "primary_clean_exploratory_factorial_tests.csv"
SENSITIVITY_TABLE = "full283_sensitivity_factorial_tests.csv"
UNCHANGED_TABLES = (
    "clean_vs_full_effect_stability.csv",
    "robust_sensitivity.csv",
    "primary_group_summaries.csv",
    "primary_model_qc_flow.csv",
)
TABLES = (PRIMARY_TABLE, SENSITIVITY_TABLE, *UNCHANGED_TABLES)
ADDED_PROVENANCE_COLUMNS = (
    "raw_effect_estimand",
    "raw_effect_interval_method",
    "raw_effect_bootstrap_replicates",
    "raw_effect_bootstrap_seed",
)
H11_CHANGED_EXISTING_COLUMNS = (
    "raw_effect_estimate",
    "raw_effect_ci95_low",
    "raw_effect_ci95_high",
    "standardized_effect",
    "standardized_ci95_low",
    "standardized_ci95_high",
)
ROW_IDENTITY_COLUMNS = ("cohort", "endpoint", "model_component", "effect")


class H11AmendmentAuditError(RuntimeError):
    """The amendment exceeded its whitelist or failed independent validation."""


def _load_analyzer() -> Any:
    path = PROJECT_ROOT / "scripts/phaxis/analyze_biological_cohorts.py"
    spec = importlib.util.spec_from_file_location(
        "_phaxis_h11_amendment_analyzer", path
    )
    if spec is None or spec.loader is None:
        raise H11AmendmentAuditError("cannot load PHAxis biological analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stable_offset(*values: object) -> int:
    token = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") % 100000


def _cell_contrasts(values: Mapping[str, float]) -> dict[str, float]:
    ev22 = float(values["EV22"])
    ev30 = float(values["EV30"])
    oe22 = float(values["OE22"])
    oe30 = float(values["OE30"])
    return {
        EFFECTS[0]: 0.5 * ((oe22 - ev22) + (oe30 - ev30)),
        EFFECTS[1]: 0.5 * ((ev30 - ev22) + (oe30 - oe22)),
        EFFECTS[2]: (oe30 - oe22) - (ev30 - ev22),
    }


def _scoped_h11(frame: pd.DataFrame, model_spec: Mapping[str, Any]) -> pd.DataFrame:
    scope = model_spec["primary_scope"]
    eligible = frame["formal_statistics_eligible"].astype(str).str.casefold().map(
        {"true": True, "false": False}
    )
    values = pd.to_numeric(frame[H11_ENDPOINT], errors="coerce")
    keep = (
        frame["study_role"].eq(scope["study_role"])
        & frame["experiment_key"].eq(scope["experiment_key"])
        & eligible.eq(True)
        & values.notna()
        & np.isfinite(values)
        & values.gt(0)
    )
    scoped = frame.loc[keep].copy().reset_index(drop=True)
    scoped[H11_ENDPOINT] = values.loc[keep].to_numpy(dtype=np.float64)
    identities = scoped.get("task_id")
    if identities is None:
        identities = scoped.get("source_image_sha256")
    if identities is None:
        raise H11AmendmentAuditError("H11 source-root identity is absent")
    identities = identities.astype("string")
    if (
        identities.isna().any()
        or identities.str.strip().eq("").any()
        or identities.duplicated().any()
    ):
        raise H11AmendmentAuditError("H11 source-root identity is missing or duplicated")
    scoped["_audit_source_root_identity"] = identities.astype(str)
    return scoped


def _independent_h11(
    frame: pd.DataFrame,
    *,
    model_spec: Mapping[str, Any],
) -> dict[str, Any]:
    scoped = _scoped_h11(frame, model_spec)
    identity_field = (
        "task_id" if "task_id" in scoped.columns else "source_image_sha256"
    )
    inference = model_spec["inference"]
    replicates = int(inference["factorial_cell_stratified_bootstrap_replicates"])
    base_seed = int(inference["random_seed"])
    effective_seed = base_seed + _stable_offset(
        "continuous", H11_ENDPOINT, "raw_median_bootstrap"
    )
    cell_indices: dict[str, np.ndarray] = {}
    cell_values: dict[str, np.ndarray] = {}
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    cell_counts: dict[str, int] = {}
    for construct, temperature, label in CELLS:
        keep = (
            scoped["genotype_or_construct"].eq(construct)
            & pd.to_numeric(scoped["temperature_c"]).eq(temperature)
        ).to_numpy()
        indices = np.flatnonzero(keep)
        if not len(indices):
            raise H11AmendmentAuditError(f"H11 cell is empty: {label}")
        identity = scoped.iloc[indices]["_audit_source_root_identity"].to_numpy()
        indices = indices[np.argsort(identity, kind="stable")]
        values = scoped.iloc[indices][H11_ENDPOINT].to_numpy(dtype=np.float64)
        cell_indices[label] = indices
        cell_values[label] = values
        medians[label] = float(np.median(values))
        means[label] = float(np.mean(values))
        cell_counts[label] = int(len(values))

    point = _cell_contrasts(medians)
    historical_mean_point = _cell_contrasts(means)
    rng = np.random.default_rng(effective_seed)
    sampled = np.empty((replicates, len(EFFECTS)), dtype=np.float64)
    values_all = scoped[H11_ENDPOINT].to_numpy(dtype=np.float64)
    for replicate in range(replicates):
        sampled_cells: dict[str, float] = {}
        for _construct, _temperature, label in CELLS:
            indices = cell_indices[label]
            drawn = rng.choice(indices, size=len(indices), replace=True)
            sampled_cells[label] = float(np.median(values_all[drawn]))
        contrast = _cell_contrasts(sampled_cells)
        sampled[replicate] = [contrast[effect] for effect in EFFECTS]
    quantiles = np.quantile(
        sampled, (0.025, 0.975), axis=0, method="linear"
    )
    standard_deviation = float(np.std(values_all, ddof=1))
    effects: dict[str, Any] = {}
    for index, effect in enumerate(EFFECTS):
        low = float(quantiles[0, index])
        high = float(quantiles[1, index])
        effects[effect] = {
            "raw_effect_estimate": float(point[effect]),
            "raw_effect_ci95_low": low,
            "raw_effect_ci95_high": high,
            "standardized_effect": float(point[effect] / standard_deviation),
            "standardized_ci95_low": float(low / standard_deviation),
            "standardized_ci95_high": float(high / standard_deviation),
            "historical_raw_mean_contrast": float(historical_mean_point[effect]),
        }
    return {
        "n": int(len(scoped)),
        "unique_source_roots": int(scoped["_audit_source_root_identity"].nunique()),
        "source_root_identity_field": identity_field,
        "source_root_identity_policy": (
            "task_id_preferred_then_source_image_sha256_fallback"
        ),
        "all_four_cells_nonempty": all(value > 0 for value in cell_counts.values()),
        "cell_counts": cell_counts,
        "cell_medians": medians,
        "cell_means": means,
        "sample_standard_deviation": standard_deviation,
        "bootstrap_replicates": replicates,
        "base_seed": base_seed,
        "effective_seed": effective_seed,
        "effects": effects,
    }


def _equal_cell(left: object, right: object, *, tolerance: float) -> bool:
    left_missing = bool(pd.isna(left))
    right_missing = bool(pd.isna(right))
    if left_missing or right_missing:
        return left_missing and right_missing
    if (
        isinstance(left, (int, float, np.integer, np.floating))
        and not isinstance(left, (bool, np.bool_))
        and isinstance(right, (int, float, np.integer, np.floating))
        and not isinstance(right, (bool, np.bool_))
    ):
        return bool(
            np.isclose(
                float(left),
                float(right),
                atol=tolerance,
                rtol=tolerance,
                equal_nan=True,
            )
        )
    return left == right


def _compare_factorial_table(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    independent: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    expected_columns = [*baseline.columns, *ADDED_PROVENANCE_COLUMNS]
    schema_extension_exact = (
        len(candidate.columns) == len(expected_columns)
        and set(candidate.columns) == set(expected_columns)
    )
    row_identity_exact = bool(
        len(baseline) == len(candidate)
        and all(column in baseline.columns and column in candidate.columns for column in ROW_IDENTITY_COLUMNS)
        and baseline[list(ROW_IDENTITY_COLUMNS)].astype(str).equals(
            candidate[list(ROW_IDENTITY_COLUMNS)].astype(str)
        )
    )
    protected_differences = 0
    non_h11_differences = 0
    allowed_h11_differences = 0
    unauthorized_differences: list[dict[str, Any]] = []
    if schema_extension_exact and row_identity_exact:
        for row_index in range(len(baseline)):
            is_h11 = str(candidate.iloc[row_index]["endpoint"]) == H11_ENDPOINT
            for column in baseline.columns:
                equal = _equal_cell(
                    baseline.iloc[row_index][column],
                    candidate.iloc[row_index][column],
                    tolerance=tolerance,
                )
                if equal:
                    continue
                allowed = is_h11 and column in H11_CHANGED_EXISTING_COLUMNS
                if allowed:
                    allowed_h11_differences += 1
                else:
                    protected_differences += 1
                    if not is_h11:
                        non_h11_differences += 1
                    unauthorized_differences.append(
                        {
                            "row": row_index,
                            "endpoint": str(candidate.iloc[row_index]["endpoint"]),
                            "effect": str(candidate.iloc[row_index]["effect"]),
                            "column": str(column),
                        }
                    )

    h11 = candidate.loc[candidate["endpoint"].eq(H11_ENDPOINT)].copy()
    h11_effects_exact = len(h11) == 3 and set(h11["effect"].astype(str)) == set(EFFECTS)
    h11_contract_exact = bool(
        h11_effects_exact
        and h11["raw_effect_estimand"].eq(RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST).all()
        and h11["raw_effect_interval_method"].eq(RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL).all()
        and pd.to_numeric(h11["raw_effect_bootstrap_replicates"], errors="coerce")
        .eq(int(independent["bootstrap_replicates"]))
        .all()
        and pd.to_numeric(h11["raw_effect_bootstrap_seed"], errors="coerce")
        .eq(int(independent["effective_seed"]))
        .all()
    )
    h11_independent_exact = h11_effects_exact
    historical_mean_identity_exact = h11_effects_exact
    if h11_effects_exact:
        baseline_h11 = baseline.loc[baseline["endpoint"].eq(H11_ENDPOINT)].set_index(
            "effect"
        )
        candidate_h11 = h11.set_index("effect")
        for effect in EFFECTS:
            expected = independent["effects"][effect]
            for field in H11_CHANGED_EXISTING_COLUMNS:
                h11_independent_exact = h11_independent_exact and _equal_cell(
                    candidate_h11.loc[effect, field],
                    expected[field],
                    tolerance=tolerance,
                )
            historical_mean_identity_exact = (
                historical_mean_identity_exact
                and _equal_cell(
                    baseline_h11.loc[effect, "raw_effect_estimate"],
                    expected["historical_raw_mean_contrast"],
                    tolerance=tolerance,
                )
            )

    non_h11 = candidate.loc[~candidate["endpoint"].eq(H11_ENDPOINT)]
    non_h11_provenance_exact = bool(
        len(non_h11) == 12
        and non_h11["raw_effect_estimand"].eq(RAW_EFFECT_OLS_MEAN_CONTRAST).all()
        and non_h11["raw_effect_interval_method"].eq(RAW_EFFECT_HC3_INTERVAL).all()
        and pd.to_numeric(
            non_h11["raw_effect_bootstrap_replicates"], errors="coerce"
        ).eq(0).all()
        and non_h11["raw_effect_bootstrap_seed"].isna().all()
    )
    passed = bool(
        schema_extension_exact
        and row_identity_exact
        and protected_differences == 0
        and non_h11_differences == 0
        and allowed_h11_differences > 0
        and h11_contract_exact
        and h11_independent_exact
        and historical_mean_identity_exact
        and non_h11_provenance_exact
    )
    return {
        "policy": "protected_exact_with_independently_validated_H11_companion_amendment",
        "passed": passed,
        "rows": int(len(candidate)),
        "row_identity_exact": row_identity_exact,
        "candidate_schema_extension_exact": schema_extension_exact,
        "protected_primary_inference_exact": protected_differences == 0,
        "protected_differing_cells": protected_differences,
        "non_h11_existing_fields_exact": non_h11_differences == 0,
        "non_h11_differing_cells": non_h11_differences,
        "allowed_h11_changed_cells": allowed_h11_differences,
        "unauthorized_differing_cells": len(unauthorized_differences),
        "unauthorized_differences": unauthorized_differences,
        "H11_raw_median_contract_exact": h11_contract_exact,
        "H11_independent_numeric_recomputation_exact": h11_independent_exact,
        "historical_H11_raw_point_was_mean_contrast": historical_mean_identity_exact,
        "non_H11_raw_mean_provenance_exact": non_h11_provenance_exact,
        "H11": dict(independent),
    }


def _unchanged_table_report(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline_sha = sha256_file(baseline_path)
    candidate_sha = sha256_file(candidate_path)
    return {
        "policy": "byte_identical",
        "passed": baseline_sha == candidate_sha,
        "baseline_sha256": baseline_sha,
        "candidate_sha256": candidate_sha,
        "byte_identical": baseline_sha == candidate_sha,
        "unauthorized_differing_cells": 0 if baseline_sha == candidate_sha else 1,
    }


def audit(
    *,
    baseline: Path,
    pre_amendment_audit: Path,
    cohorts: Path,
    model_spec: Path,
    output: Path,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    baseline = baseline.resolve()
    pre_amendment_audit = pre_amendment_audit.resolve()
    cohorts = cohorts.resolve()
    model_spec = model_spec.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    historical = read_json(pre_amendment_audit)
    if (
        historical.get("schema_version")
        != "PHAxis-biological-analysis-native-equivalence-audit-1.0"
        or historical.get("status") != "passed"
        or historical.get("tables_byte_identical") is not True
        or historical.get("total_differing_cells") != 0
        or historical.get("blind_images_used") != 0
        or historical.get("canonical_annotations_read") is not False
    ):
        raise H11AmendmentAuditError("pre-amendment equivalence authority is invalid")
    if Path(str(historical.get("candidate_analysis", ""))).resolve() != baseline:
        raise H11AmendmentAuditError("baseline differs from pre-amendment authority")
    cohort_summary = read_json(cohorts / "summary.json")
    cohort_lock = read_json(cohorts / "analysis_contract_lock.json")
    if (
        cohort_summary.get("status")
        != "completed_without_fitting_biological_effect_models"
        or cohort_summary.get("blind_images_used") != 0
        or cohort_summary.get("canonical_annotations_read") is not False
    ):
        raise H11AmendmentAuditError("locked cohort authority is invalid")
    spec_payload = read_json(model_spec)
    inference = spec_payload.get("inference", {})
    if (
        spec_payload.get("reporting", {}).get("blind_images_used") != 0
        or int(inference.get("factorial_cell_stratified_bootstrap_replicates", 0))
        != 5000
        or int(inference.get("random_seed", 0)) != 20260823
        or [item.get("field") for item in spec_payload.get("confirmatory_endpoints", [])]
        != [
            "local_hair_count_1_4mm",
            H11_ENDPOINT,
            "first_hair_ge40um_distance_from_distal_point_um",
            "median_root_width_um",
            "visible_root_axis_length_um",
        ]
    ):
        raise H11AmendmentAuditError("biological model specification drifted")

    analyzer = _load_analyzer()
    clean_name = str(cohort_summary["cohort_directories"]["primary"])
    full_name = str(cohort_summary["cohort_directories"]["sensitivity"])
    clean_traits_path = cohorts / clean_name / "traits.csv"
    full_traits_path = cohorts / full_name / "traits.csv"
    clean_traits = pd.read_csv(clean_traits_path)
    full_traits = pd.read_csv(full_traits_path)
    clean, clean_robust, clean_groups, clean_flow = analyzer._run_factorial(
        clean_traits,
        model_spec=spec_payload,
        cohort=clean_name,
        cohort_role="primary_SHA_disjoint",
    )
    full, full_robust, full_groups, full_flow = analyzer._run_factorial(
        full_traits,
        model_spec=spec_payload,
        cohort=full_name,
        cohort_role="overlap_contaminated_sensitivity",
    )
    candidate_tables = {
        PRIMARY_TABLE: clean,
        SENSITIVITY_TABLE: full,
        "clean_vs_full_effect_stability.csv": analyzer._comparison(clean, full),
        "robust_sensitivity.csv": pd.concat(
            [clean_robust, full_robust], ignore_index=True
        ),
        "primary_group_summaries.csv": pd.concat(
            [clean_groups, full_groups], ignore_index=True
        ),
        "primary_model_qc_flow.csv": pd.concat(
            [clean_flow, full_flow], ignore_index=True
        ),
    }
    candidate_paths: dict[str, Path] = {}
    for name in TABLES:
        path = output / "recomputed_tables" / name
        analyzer._atomic_dataframe(path, candidate_tables[name])
        candidate_paths[name] = path

    independent_by_table = {
        PRIMARY_TABLE: _independent_h11(clean_traits, model_spec=spec_payload),
        SENSITIVITY_TABLE: _independent_h11(full_traits, model_spec=spec_payload),
    }
    reports: dict[str, Any] = {}
    for name in (PRIMARY_TABLE, SENSITIVITY_TABLE):
        baseline_path = baseline / "tables" / name
        baseline_frame = pd.read_csv(baseline_path)
        candidate_frame = pd.read_csv(candidate_paths[name])
        report = _compare_factorial_table(
            baseline_frame,
            candidate_frame,
            independent=independent_by_table[name],
            tolerance=tolerance,
        )
        report.update(
            {
                "baseline_sha256": sha256_file(baseline_path),
                "candidate_sha256": sha256_file(candidate_paths[name]),
            }
        )
        reports[name] = report
    for name in UNCHANGED_TABLES:
        reports[name] = _unchanged_table_report(
            baseline / "tables" / name, candidate_paths[name]
        )

    protected_pass = all(
        reports[name].get("protected_primary_inference_exact") is True
        and reports[name].get("non_h11_existing_fields_exact") is True
        and reports[name].get("unauthorized_differing_cells") == 0
        for name in (PRIMARY_TABLE, SENSITIVITY_TABLE)
    )
    h11_pass = all(
        reports[name].get("H11_raw_median_contract_exact") is True
        and reports[name].get("H11_independent_numeric_recomputation_exact") is True
        and reports[name].get("historical_H11_raw_point_was_mean_contrast") is True
        for name in (PRIMARY_TABLE, SENSITIVITY_TABLE)
    )
    unaffected_pass = all(
        reports[name].get("byte_identical") is True for name in UNCHANGED_TABLES
    )
    unauthorized = sum(
        int(report.get("unauthorized_differing_cells", 0))
        for report in reports.values()
    )
    all_pass = protected_pass and h11_pass and unaffected_pass and unauthorized == 0

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all_pass else "failed",
        "artifact_role": ARTIFACT_ROLE,
        "scope": (
            "CPU-only current-source rerun of the fixed D15 five-endpoint by "
            "three-effect family on locked clean/full trait tables"
        ),
        "pre_amendment_baseline": {
            "authority_path": pre_amendment_audit.relative_to(PROJECT_ROOT).as_posix(),
            "authority_sha256": sha256_file(pre_amendment_audit),
            "authority_schema_version": historical["schema_version"],
            "authority_status": historical["status"],
            "candidate_analysis": baseline.relative_to(PROJECT_ROOT).as_posix(),
        },
        "locked_inputs": {
            "cohort_summary_sha256": sha256_file(cohorts / "summary.json"),
            "cohort_lock_sha256": sha256_file(
                cohorts / "analysis_contract_lock.json"
            ),
            "primary_traits_sha256": sha256_file(clean_traits_path),
            "sensitivity_traits_sha256": sha256_file(full_traits_path),
            "model_spec_sha256": sha256_file(model_spec),
            "analysis_contract_sha256": str(cohort_lock["analysis_contract_sha256"]),
        },
        "implementation_sha256": {
            "biological_analysis": sha256_file(
                PROJECT_ROOT / "src/phaxis/biological_analysis.py"
            ),
            "biological_analysis_wrapper": sha256_file(
                PROJECT_ROOT / "scripts/phaxis/analyze_biological_cohorts.py"
            ),
            "audit_producer": sha256_file(Path(__file__)),
            "publication_figure_input_builder": sha256_file(
                PROJECT_ROOT / "scripts/phaxis/build_publication_figure_inputs.py"
            ),
            "multitrait_atlas": sha256_file(
                PROJECT_ROOT / "src/phaxis/multitrait_atlas.py"
            ),
            "audit_test": sha256_file(
                PROJECT_ROOT
                / "tests/phaxis/test_h11_raw_median_amendment_audit.py"
            ),
        },
        "change_contract": {
            "endpoint": H11_ENDPOINT,
            "raw_effect_estimand": RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
            "raw_effect_interval_method": RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
            "bootstrap_replicates": 5000,
            "base_seed": 20260823,
            "effective_seed": int(
                independent_by_table[PRIMARY_TABLE]["effective_seed"]
            ),
            "stable_seed_offset_token": "raw_median_bootstrap",
            "source_unit": "source_root",
            "source_root_identity_policy": (
                "task_id_preferred_then_source_image_sha256_fallback"
            ),
            "cell_summary": "median",
            "percentile_interval": [0.025, 0.975],
            "numpy_quantile_method": "linear",
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
            "changed_existing_columns_whitelist": list(
                H11_CHANGED_EXISTING_COLUMNS
            ),
            "added_provenance_columns": list(ADDED_PROVENANCE_COLUMNS),
            "separate_hypothesis_test_added": False,
            "D15_fixed_effect_family_changed": False,
        },
        "tables": reports,
        "protected_primary_inference_equivalent": protected_pass,
        "unaffected_tables_byte_identical": unaffected_pass,
        "non_h11_existing_fields_equivalent": all(
            reports[name].get("non_h11_existing_fields_exact") is True
            for name in (PRIMARY_TABLE, SENSITIVITY_TABLE)
        ),
        "candidate_schema_extension_exact": all(
            reports[name].get("candidate_schema_extension_exact") is True
            for name in (PRIMARY_TABLE, SENSITIVITY_TABLE)
        ),
        "H11_raw_median_companion": {
            "validated": h11_pass,
            "cohort_tables": [PRIMARY_TABLE, SENSITIVITY_TABLE],
            "effect_rows": 6,
            "independent_point_and_interval_recomputation": True,
        },
        "unauthorized_differing_cells": unauthorized,
        "separate_hypothesis_test_added": False,
        "new_hypothesis_tests_added": 0,
        "D15_fixed_effect_family_changed": False,
        "gpu_programs_started": 0,
        "canonical_annotations_read": False,
        "condition_metadata_used_for_routing": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    payload["amendment_audit_identity_sha256"] = sha256_json(payload)
    atomic_write_json(output / "amendment_audit.json", payload)
    if not all_pass:
        raise H11AmendmentAuditError(
            "H11 amendment exceeded its whitelist or failed independent validation"
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pre-amendment-audit", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = audit(
        baseline=args.baseline,
        pre_amendment_audit=args.pre_amendment_audit,
        cohorts=args.cohorts,
        model_spec=args.model_spec,
        output=args.output,
        tolerance=float(args.tolerance),
    )
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
