from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from statsmodels.stats.multitest import multipletests

from phaxis.biological_analysis import (
    RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
    RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
    RAW_EFFECT_OLS_MEAN_CONTRAST,
    WT_CONTRAST_MULTIPLICITY_FAMILY,
    WT_META_MULTIPLICITY_FAMILY,
    coerce_boolean_series,
    count_results,
    factorial_design,
    group_summaries,
    linear_results,
    observability_results,
    random_effects_reml_hartung_knapp,
    raw_median_bootstrap_seed,
    robust_sensitivity,
    wt_temperature_secondary_results,
)


def _factorial_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cells = (
        ("RHD6_EV_22C", "RHD6-EV", 22),
        ("RHD6_EV_30C", "RHD6-EV", 30),
        ("RHD6_OE_22C", "RHD6-OE", 22),
        ("RHD6_OE_30C", "RHD6-OE", 30),
    )
    for condition, construct, temperature in cells:
        oe = int(construct == "RHD6-OE")
        warm = int(temperature == 30)
        for replicate in range(8):
            modulation = 1.0 + 0.015 * (replicate - 3.5)
            first_hair = (
                np.nan
                if replicate in {0, 5}
                else (1450.0 - 180.0 * oe + 120.0 * warm) * modulation
            )
            rows.append(
                {
                    "task_id": f"{condition}-{replicate:02d}",
                    "experiment_key": "D15_8d",
                    "condition_code": condition,
                    "genotype_or_construct": construct,
                    "temperature_c": temperature,
                    "formal_statistics_eligible": "TRUE"
                    if replicate % 2
                    else "true",
                    "local_hair_count_1_4mm": (
                        8 + 7 * oe - 3 * warm + 2 * oe * warm + replicate % 4
                    ),
                    "local_median_hair_length_um_1_4mm": (
                        90.0 * (1.35**oe) * (0.82**warm) * (1.18 ** (oe * warm))
                        * modulation
                    ),
                    "first_hair_ge40um_distance_from_distal_point_um": first_hair,
                    "median_root_width_um": (
                        118.0 * (0.96**oe) * (0.93**warm) * (1.12 ** (oe * warm))
                        * modulation
                    ),
                    "visible_root_axis_length_um": (
                        12_000.0
                        * (0.9**oe)
                        * (1.2**warm)
                        * (0.95 ** (oe * warm))
                        * modulation
                    ),
                }
            )
    return pd.DataFrame(rows)


def _wt_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    experiments = (
        ("WT5_A", 5.0, 0),
        ("WT5_B", 5.0, 1),
        ("WT5_C", 5.0, 2),
        ("WT7_A", 7.0, 3),
        ("WT7_B", 7.0, 4),
        ("WT_UNKNOWN", np.nan, 5),
    )
    for experiment, day, experiment_index in experiments:
        warm_ratio = 1.16 + 0.035 * experiment_index
        for temperature in (22, 30):
            warm = int(temperature == 30)
            for replicate in range(6):
                modulation = 1.0 + 0.018 * (replicate - 2.5)
                count_base = 5 + experiment_index + replicate % 3
                rows.append(
                    {
                        "task_id": (
                            f"{experiment}-{temperature}-{replicate:02d}"
                        ),
                        "experiment_key": experiment,
                        "study_role": "wt_temperature_block",
                        "developmental_day": day,
                        "condition_code": f"WT_{temperature}C",
                        "genotype_or_construct": "WT",
                        "temperature_c": temperature,
                        "formal_statistics_eligible": "true",
                        "distal_window_1_4mm_eligible": "true",
                        "local_hair_count_1_4mm": (
                            count_base + warm * (3 + experiment_index % 2)
                        ),
                        "local_median_hair_length_um_1_4mm": (
                            95.0
                            * (1.0 + 0.025 * experiment_index)
                            * (warm_ratio**warm)
                            * modulation
                        ),
                        "first_hair_ge40um_distance_from_distal_point_um": (
                            1650.0
                            * (1.0 + 0.02 * experiment_index)
                            * ((warm_ratio + 0.04) ** warm)
                            * modulation
                        ),
                        "median_root_width_um": (
                            112.0
                            * (1.0 + 0.01 * experiment_index)
                            * ((1.04 + 0.01 * experiment_index) ** warm)
                            * modulation
                        ),
                        "visible_root_axis_length_um": (
                            12_500.0
                            * (1.0 + 0.015 * experiment_index)
                            * ((1.10 + 0.02 * experiment_index) ** warm)
                            * modulation
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_boolean_coercion_and_locked_factorial_design() -> None:
    converted = coerce_boolean_series(
        pd.Series([True, False, "TRUE", "false", "unknown", None])
    )
    assert converted.iloc[:4].tolist() == [True, False, True, False]
    assert converted.iloc[4:].isna().all()

    frame = pd.DataFrame(
        {
            "genotype_or_construct": [
                "RHD6-EV",
                "RHD6-EV",
                "RHD6-OE",
                "RHD6-OE",
            ],
            "temperature_c": [22, 30, 22, 30],
        }
    )
    assert np.array_equal(
        factorial_design(frame),
        np.asarray(
            [
                [1.0, -0.5, -0.5, 0.25],
                [1.0, -0.5, 0.5, -0.25],
                [1.0, 0.5, -0.5, -0.25],
                [1.0, 0.5, 0.5, 0.25],
            ]
        ),
    )


def test_synthetic_models_are_deterministic_and_preserve_effect_semantics() -> None:
    frame = _factorial_fixture()
    count_kwargs = {
        "field": "local_hair_count_1_4mm",
        "endpoint_label": "Local hair count",
        "bootstrap_replicates": 64,
        "permutations": 63,
        "seed": 20260823,
        "poisson_fallback_max_dispersion": 100.0,
    }
    first_count = pd.DataFrame(count_results(frame, **count_kwargs))
    second_count = pd.DataFrame(count_results(frame, **count_kwargs))
    assert_frame_equal(first_count, second_count, check_exact=True)
    assert len(first_count) == 3
    assert first_count.loc[
        first_count["effect"].eq("construct_OE_minus_EV"), "estimate"
    ].iat[0] > 1.0
    assert first_count.loc[
        first_count["effect"].eq("temperature_30C_minus_22C"), "estimate"
    ].iat[0] < 1.0

    linear_kwargs = {
        "field": "local_median_hair_length_um_1_4mm",
        "endpoint_label": "Conditional length",
        "component": "continuous",
        "log_transform": True,
        "bootstrap_replicates": 64,
        "permutations": 63,
        "seed": 20260823,
    }
    first_linear = pd.DataFrame(linear_results(frame, **linear_kwargs))
    second_linear = pd.DataFrame(linear_results(frame, **linear_kwargs))
    assert_frame_equal(first_linear, second_linear, check_exact=True)
    assert len(first_linear) == 3
    assert first_linear.loc[
        first_linear["effect"].eq("construct_OE_minus_EV"), "estimate"
    ].iat[0] > 1.0


def test_h11_raw_companion_uses_cell_medians_and_preserves_primary_inference() -> None:
    cell_values = (
        ("RHD6_EV_22C", "RHD6-EV", 22, (1.0, 2.0, 3.0, 1000.0)),
        ("RHD6_EV_30C", "RHD6-EV", 30, (2.0, 3.0, 4.0, 2000.0)),
        ("RHD6_OE_22C", "RHD6-OE", 22, (11.0, 12.0, 13.0, 3000.0)),
        ("RHD6_OE_30C", "RHD6-OE", 30, (22.0, 23.0, 24.0, 5000.0)),
    )
    rows: list[dict[str, object]] = []
    for condition, construct, temperature, values in cell_values:
        for replicate, value in enumerate(values):
            rows.append(
                {
                    "task_id": f"{condition}-{replicate}",
                    "genotype_or_construct": construct,
                    "temperature_c": temperature,
                    "local_median_hair_length_um_1_4mm": value,
                }
            )
    frame = pd.DataFrame(rows)
    kwargs = {
        "field": "local_median_hair_length_um_1_4mm",
        "endpoint_label": "Median hair length",
        "component": "continuous",
        "log_transform": True,
        "bootstrap_replicates": 128,
        "permutations": 63,
        "seed": 20260823,
    }
    legacy_mean = pd.DataFrame(linear_results(frame, **kwargs))
    median = pd.DataFrame(
        linear_results(
            frame,
            **kwargs,
            raw_effect_estimand=RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
        )
    )
    repeated = pd.DataFrame(
        linear_results(
            frame.sample(frac=1.0, random_state=731).reset_index(drop=True),
            **kwargs,
            raw_effect_estimand=RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
        )
    )

    main_columns = [
        "estimate",
        "ci95_low",
        "ci95_high",
        "p_value_model",
        "p_value_freedman_lane",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
    ]
    assert_frame_equal(
        legacy_mean[main_columns], median[main_columns], check_exact=True
    )
    expected = {
        "construct_OE_minus_EV": 15.0,
        "temperature_30C_minus_22C": 6.0,
        "construct_by_temperature_interaction": 10.0,
    }
    assert median.set_index("effect")["raw_effect_estimate"].to_dict() == expected
    assert legacy_mean.set_index("effect")["raw_effect_estimate"].to_dict() != expected
    assert median["raw_effect_estimand"].eq(
        RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
    ).all()
    assert median["raw_effect_interval_method"].eq(
        RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
    ).all()
    assert median["raw_effect_bootstrap_replicates"].eq(128).all()
    assert median["raw_effect_bootstrap_seed"].eq(
        raw_median_bootstrap_seed(
            seed=20260823,
            field="local_median_hair_length_um_1_4mm",
            component="continuous",
        )
    ).all()
    assert np.isfinite(median["raw_effect_ci95_low"]).all()
    assert np.isfinite(median["raw_effect_ci95_high"]).all()
    assert median["raw_effect_ci95_low"].le(
        median["raw_effect_ci95_high"]
    ).all()
    raw_sd = float(
        np.std(frame["local_median_hair_length_um_1_4mm"], ddof=1)
    )
    np.testing.assert_allclose(
        median["standardized_effect"].to_numpy(dtype=float) * raw_sd,
        median["raw_effect_estimate"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        median["standardized_ci95_low"].to_numpy(dtype=float) * raw_sd,
        median["raw_effect_ci95_low"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        median["standardized_ci95_high"].to_numpy(dtype=float) * raw_sd,
        median["raw_effect_ci95_high"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    raw_columns = [
        "effect",
        "raw_effect_estimate",
        "raw_effect_ci95_low",
        "raw_effect_ci95_high",
        "raw_effect_estimand",
        "raw_effect_interval_method",
        "raw_effect_bootstrap_replicates",
        "raw_effect_bootstrap_seed",
    ]
    assert_frame_equal(median[raw_columns], repeated[raw_columns], check_exact=True)

    assert legacy_mean["raw_effect_estimand"].eq(
        RAW_EFFECT_OLS_MEAN_CONTRAST
    ).all()
    assert legacy_mean["raw_effect_bootstrap_replicates"].eq(0).all()
    assert legacy_mean["raw_effect_bootstrap_seed"].isna().all()


def test_h11_raw_median_companion_fails_closed_on_source_root_identity() -> None:
    frame = _factorial_fixture()
    kwargs = {
        "field": "local_median_hair_length_um_1_4mm",
        "endpoint_label": "Median hair length",
        "component": "continuous",
        "log_transform": True,
        "bootstrap_replicates": 16,
        "permutations": 15,
        "seed": 20260823,
        "raw_effect_estimand": RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
    }
    duplicated = frame.copy()
    duplicated.loc[duplicated.index[1], "task_id"] = duplicated.loc[
        duplicated.index[0], "task_id"
    ]
    try:
        linear_results(duplicated, **kwargs)
    except RuntimeError as error:
        assert "identity is duplicated" in str(error)
    else:
        raise AssertionError("duplicated source-root identity must fail closed")

    missing_cell = frame.loc[~frame["condition_code"].eq("RHD6_OE_30C")]
    assert linear_results(missing_cell, **kwargs) == []


def test_raw_mean_null_seed_survives_h08_csv_roundtrip(tmp_path: Path) -> None:
    frame = _factorial_fixture()
    h08 = pd.DataFrame(
        count_results(
            frame,
            field="local_hair_count_1_4mm",
            endpoint_label="Local hair count",
            bootstrap_replicates=16,
            permutations=15,
            seed=20260823,
            poisson_fallback_max_dispersion=100.0,
        )
    )
    path = tmp_path / "h08.csv"
    h08.to_csv(path, index=False)
    restored = pd.read_csv(path)
    assert restored["raw_effect_bootstrap_replicates"].eq(0).all()
    assert restored["raw_effect_bootstrap_seed"].isna().all()
    assert restored["raw_effect_estimand"].eq(
        RAW_EFFECT_OLS_MEAN_CONTRAST
    ).all()

    observed = pd.DataFrame(
        observability_results(
            frame,
            field="first_hair_ge40um_distance_from_distal_point_um",
            endpoint_label="First hair observability",
        )
    )
    assert len(observed) == 3
    assert observed["model_component"].eq("observability").all()
    assert observed["p_value_freedman_lane"].isna().all()


def test_synthetic_robust_and_group_summaries_keep_source_unit_denominators() -> None:
    frame = _factorial_fixture()
    robust = robust_sensitivity(frame)
    assert len(robust) == 15
    assert set(robust["effect"]) == {
        "construct_OE_minus_EV",
        "temperature_30C_minus_22C",
        "construct_by_temperature_interaction",
    }
    assert robust["n"].min() == 24
    assert robust["n"].max() == 32

    endpoints = [
        ("local_hair_count_1_4mm", "Local count"),
        (
            "first_hair_ge40um_distance_from_distal_point_um",
            "First hair",
        ),
    ]
    summaries = group_summaries(frame, endpoints, scope="synthetic")
    assert len(summaries) == 8
    assert summaries["n_total"].eq(8).all()
    first_hair = summaries[
        summaries["endpoint"].eq(
            "first_hair_ge40um_distance_from_distal_point_um"
        )
    ]
    assert first_hair["n_nonmissing"].eq(6).all()
    assert first_hair["n_missing"].eq(2).all()


def test_wt_secondary_is_experiment_blocked_and_strictly_within_day() -> None:
    contrasts, meta, flow = wt_temperature_secondary_results(
        _wt_fixture(),
        minimum_per_temperature=3,
        minimum_experiments_per_day_meta=3,
        poisson_fallback_max_dispersion=1.25,
    )
    assert len(contrasts) == 30  # six experiments x five locked endpoints
    assert contrasts["analysis_status"].eq("estimated").all()
    assert len(flow) == len(contrasts)
    unknown = contrasts[contrasts["experiment_key"].eq("WT_UNKNOWN")]
    assert len(unknown) == 5
    assert unknown["developmental_day"].isna().all()
    assert ~unknown["meta_eligible"].all()
    assert unknown["meta_exclusion_reason"].eq(
        "unknown_developmental_day"
    ).all()
    assert meta["developmental_day"].notna().all()
    assert set(meta["developmental_day"]) == {5.0, 7.0}
    assert len(meta) == 10  # two known days x five endpoints
    day5 = meta[meta["developmental_day"].eq(5.0)]
    assert len(day5) == 5
    assert day5["analysis_status"].eq("estimated").all()
    assert day5["k_eligible_experiments"].eq(3).all()
    assert day5["estimate_30C_over_22C"].gt(1.0).all()
    day7 = meta[meta["developmental_day"].eq(7.0)]
    assert len(day7) == 5
    assert day7["analysis_status"].eq("not_estimable").all()
    assert day7["k_eligible_experiments"].eq(2).all()
    assert day7["estimate_30C_over_22C"].isna().all()
    assert meta["cross_day_pooling_performed"].eq(False).all()
    assert meta["unknown_day_contrasts_included"].eq(False).all()


def test_wt_secondary_base_and_endpoint_gates_are_independent() -> None:
    frame = _wt_fixture()
    warm_a = frame["experiment_key"].eq("WT5_A") & frame["temperature_c"].eq(30)
    missing_length = frame.index[warm_a][:4]
    frame.loc[
        missing_length, "local_median_hair_length_um_1_4mm"
    ] = np.nan
    contrasts, meta, flow = wt_temperature_secondary_results(
        frame,
        minimum_per_temperature=3,
        minimum_experiments_per_day_meta=3,
        poisson_fallback_max_dispersion=1.25,
    )
    failed_length = contrasts[
        contrasts["experiment_key"].eq("WT5_A")
        & contrasts["endpoint"].eq(
            "local_median_hair_length_um_1_4mm"
        )
    ].iloc[0]
    retained_count = contrasts[
        contrasts["experiment_key"].eq("WT5_A")
        & contrasts["endpoint"].eq("local_hair_count_1_4mm")
    ].iloc[0]
    assert failed_length["analysis_status"] == "not_estimable"
    assert failed_length["n_formal_30C"] == 6
    assert failed_length["n_endpoint_30C"] == 2
    assert failed_length["not_estimable_reason"] == "endpoint_30C_n_below_3"
    assert retained_count["analysis_status"] == "estimated"
    meta_length = meta[
        meta["developmental_day"].eq(5.0)
        & meta["endpoint"].eq("local_median_hair_length_um_1_4mm")
    ].iloc[0]
    assert meta_length["k_eligible_experiments"] == 2
    assert meta_length["analysis_status"] == "not_estimable"
    flow_length = flow[
        flow["experiment_key"].eq("WT5_A")
        & flow["endpoint"].eq("local_median_hair_length_um_1_4mm")
    ].iloc[0]
    assert bool(flow_length["base_gate_pass"])
    assert not bool(flow_length["endpoint_gate_pass"])

    base_failed = _wt_fixture()
    cold_a = base_failed["experiment_key"].eq("WT5_A") & base_failed[
        "temperature_c"
    ].eq(22)
    base_failed.loc[base_failed.index[cold_a][:4], "formal_statistics_eligible"] = (
        "false"
    )
    base_contrasts, _base_meta, _base_flow = wt_temperature_secondary_results(
        base_failed,
        minimum_per_temperature=3,
        minimum_experiments_per_day_meta=3,
        poisson_fallback_max_dispersion=1.25,
    )
    experiment_rows = base_contrasts[
        base_contrasts["experiment_key"].eq("WT5_A")
    ]
    assert len(experiment_rows) == 5
    assert experiment_rows["analysis_status"].eq("not_estimable").all()
    assert experiment_rows["not_estimable_reason"].eq(
        "base_22C_n_below_3"
    ).all()


def test_wt_secondary_is_deterministic_under_input_row_order() -> None:
    kwargs = {
        "minimum_per_temperature": 3,
        "minimum_experiments_per_day_meta": 3,
        "poisson_fallback_max_dispersion": 1.25,
    }
    first = wt_temperature_secondary_results(_wt_fixture(), **kwargs)
    shuffled = _wt_fixture().sample(frac=1.0, random_state=981).reset_index(
        drop=True
    )
    second = wt_temperature_secondary_results(shuffled, **kwargs)
    for left, right in zip(first, second, strict=True):
        assert_frame_equal(left, right, check_exact=True)


def test_wt_secondary_fails_closed_on_blind_rows_and_conflicting_days() -> None:
    kwargs = {
        "minimum_per_temperature": 3,
        "minimum_experiments_per_day_meta": 3,
        "poisson_fallback_max_dispersion": 1.25,
    }
    blind_tainted = _wt_fixture()
    blind_tainted["blind_images_used"] = 0
    blind_tainted.loc[blind_tainted.index[0], "blind_images_used"] = 1
    try:
        wt_temperature_secondary_results(blind_tainted, **kwargs)
    except RuntimeError as error:
        assert "blind-tainted" in str(error)
    else:
        raise AssertionError("blind-tainted WT rows must fail closed")

    conflicting_day = _wt_fixture()
    experiment_a = conflicting_day["experiment_key"].eq("WT5_A")
    conflicting_day.loc[conflicting_day.index[experiment_a][0], "developmental_day"] = 6
    try:
        wt_temperature_secondary_results(conflicting_day, **kwargs)
    except RuntimeError as error:
        assert "multiple developmental days" in str(error)
    else:
        raise AssertionError("one experiment spanning known days must fail closed")


def test_wt_reml_hartung_knapp_requires_three_experiments() -> None:
    try:
        random_effects_reml_hartung_knapp(
            np.asarray([0.1, 0.2]), np.asarray([0.02, 0.03])
        )
    except RuntimeError as error:
        assert "at least three experiments" in str(error)
    else:
        raise AssertionError("two experiments must not produce a WT meta-estimate")

    result = random_effects_reml_hartung_knapp(
        np.asarray([0.10, 0.24, 0.16]),
        np.asarray([0.020, 0.035, 0.028]),
    )
    assert result["k"] == 3
    assert result["ci95_low"] < result["estimate"] < result["ci95_high"]
    assert result["tau2"] >= 0
    assert 0 <= result["I2"] <= 1


def test_wt_secondary_bh_families_include_unknown_day_and_stay_per_cohort() -> None:
    from scripts.phaxis.analyze_biological_cohorts import _run_wt_secondary

    spec_path = (
        Path(__file__).resolve().parents[2]
        / "configs/phaxis/v1_0/biological_model_spec.json"
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    contrasts, meta, _flow = _run_wt_secondary(
        _wt_fixture(),
        model_spec=spec,
        cohort="clean_fixture",
        cohort_role="primary_SHA_disjoint",
    )

    estimated = contrasts["analysis_status"].eq("estimated")
    expected_rejected, expected_q, _sidak, _bonferroni = multipletests(
        contrasts.loc[estimated, "p_value_model"].to_numpy(dtype=float),
        alpha=0.05,
        method="fdr_bh",
    )
    np.testing.assert_allclose(
        contrasts.loc[estimated, "p_value_model_BH_FDR"].to_numpy(
            dtype=float
        ),
        expected_q,
        rtol=0.0,
        atol=0.0,
    )
    assert np.array_equal(
        contrasts.loc[estimated, "reject_model_BH_FDR_0p05"].to_numpy(
            dtype=bool
        ),
        expected_rejected,
    )
    assert contrasts["multiplicity_family"].eq(
        WT_CONTRAST_MULTIPLICITY_FAMILY
    ).all()
    unknown = contrasts[contrasts["developmental_day"].isna()]
    assert len(unknown) == 5
    assert unknown["p_value_model_BH_FDR"].notna().all()
    assert unknown["meta_eligible"].eq(False).all()

    estimated_meta = meta["analysis_status"].eq("estimated")
    _meta_rejected, expected_meta_q, _sidak, _bonferroni = multipletests(
        meta.loc[estimated_meta, "p_value_hartung_knapp"].to_numpy(
            dtype=float
        ),
        alpha=0.05,
        method="fdr_bh",
    )
    np.testing.assert_allclose(
        meta.loc[
            estimated_meta, "p_value_hartung_knapp_BH_FDR"
        ].to_numpy(dtype=float),
        expected_meta_q,
        rtol=0.0,
        atol=0.0,
    )
    assert meta["multiplicity_family"].eq(
        WT_META_MULTIPLICITY_FAMILY
    ).all()
    assert meta.loc[
        ~estimated_meta, "p_value_hartung_knapp_BH_FDR"
    ].isna().all()
    assert meta.loc[
        ~estimated_meta, "reject_hartung_knapp_BH_FDR_0p05"
    ].eq(False).all()


def test_production_wrapper_has_no_frozen_hybrid_max_import() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/phaxis/analyze_biological_cohorts.py"
    ).read_text(encoding="utf-8")
    assert "analyze_six_condition_hybrid_max" not in script
    assert "from scripts import" not in script
    assert "configs/rhaxis_nextgen" not in script
    assert "configs/phaxis/v1_0/biological_model_spec.json" in script
    assert "phaxis.biological_analysis" in script
    assert "observability_results(" not in script
    assert "wt_temperature_secondary_results" in script
    assert "D15_fixed_effect_family_changed" in script
    assert script.count('"schema_version": WT_SECONDARY_SCHEMA') == 2
    assert '--model-contract-proposal' in script
    assert "require_output_identity" in script
    assert "model_contract_proposal_sha256" not in script  # emitted by the shared helper


def test_phaxis_model_spec_locks_semantics_without_predecessor_runtime_path() -> None:
    root = Path(__file__).resolve().parents[2]
    phaxis_path = root / "configs/phaxis/v1_0/biological_model_spec.json"
    phaxis_spec = json.loads(phaxis_path.read_text(encoding="utf-8"))
    assert phaxis_spec["schema_version"] == "PHAxis-biological-model-spec-1.1"
    assert phaxis_spec["status"] == (
        "locked_for_phaxis_v1_0_postresult_exploratory_analysis"
    )
    assert phaxis_spec["historical_provenance"]["source_spec_sha256"] == (
        "9ec5d72d70547d43afbf8cacd690e3a0e007b4ab19cb0c672f5d09dbcbd82901"
    )
    assert "rhaxis_nextgen" not in phaxis_path.read_text(encoding="utf-8").lower()
    assert [item["field"] for item in phaxis_spec["confirmatory_endpoints"]] == [
        "local_hair_count_1_4mm",
        "local_median_hair_length_um_1_4mm",
        "first_hair_ge40um_distance_from_distal_point_um",
        "median_root_width_um",
        "visible_root_axis_length_um",
    ]
    first_hair = phaxis_spec["confirmatory_endpoints"][2]
    assert first_hair["model"] == "HC3_OLS_log_transformed_conditional_on_observable"
    assert first_hair["effect_scale"] == "geometric_mean_ratio_conditional_distance"
    assert phaxis_spec["historical_provenance"][
        "model_or_inference_semantics_changed"
    ] is True
    assert phaxis_spec["reporting"][
        "first_hair_observability_in_fixed_effect_family"
    ] is False
    assert "fixed 15-effect family" in phaxis_spec["reporting"][
        "first_hair_observability_reporting"
    ]
    inference = phaxis_spec["inference"]
    assert inference["factorial_cell_stratified_bootstrap_replicates"] == 5000
    assert inference["freedman_lane_permutations"] == 9999
    assert inference["random_seed"] == 20260823
    assert inference["poisson_fallback_diagnostic"] == {
        "only_after_nb2_nonconvergence_or_nonfinite_inference": True,
        "criterion": "Pearson chi-square divided by residual degrees of freedom",
        "pearson_chi2_over_df_maximum": 1.25,
        "fallback_covariance": "HC3",
        "fail_closed_above_threshold": True,
    }
    assert phaxis_spec["wt_temperature_scope"][
        "developmental_day_handling"
    ] == "strict_within-day_stratification; cross-day pooled estimate forbidden"
    wt = phaxis_spec["wt_temperature_scope"]
    assert wt["endpoint_family"] == [
        "local_hair_count_1_4mm",
        "local_median_hair_length_um_1_4mm",
        "first_hair_ge40um_distance_from_distal_point_um",
        "median_root_width_um",
        "visible_root_axis_length_um",
    ]
    assert wt["per_experiment_minimum_per_temperature"] == 3
    assert wt["per_endpoint_minimum_per_temperature"] == 3
    assert wt["minimum_experiments_per_day_meta_analysis"] == 3
    assert wt["unknown_day_handling"] == (
        "report_estimable_within_experiment_contrast_but_forbid_meta_analysis"
    )
    assert wt["insufficient_same_day_experiments"] == (
        "typed_not_estimable_row_without_pooled_estimate"
    )
    assert wt["within_experiment_multiplicity"] == (
        "Benjamini-Hochberg within each cohort across every estimated "
        "experiment-by-endpoint contrast, including unknown-day contrasts"
    )
    assert wt["within_day_meta_multiplicity"] == (
        "Benjamini-Hochberg within each cohort across every estimated "
        "developmental-day-by-endpoint meta-analysis"
    )
    assert wt["clean_full_pooling_allowed"] is False


def test_phaxis_model_spec_guard_fails_closed_on_semantic_drift() -> None:
    from scripts.phaxis.analyze_biological_cohorts import (
        SCHEMA,
        WT_SECONDARY_SCHEMA,
        _verify_model_spec,
    )

    assert SCHEMA == "PHAxis-exploratory-biological-analysis-1.0"
    assert WT_SECONDARY_SCHEMA == "PHAxis-WT-temperature-secondary-1.0"

    path = (
        Path(__file__).resolve().parents[2]
        / "configs/phaxis/v1_0/biological_model_spec.json"
    )
    spec = json.loads(path.read_text(encoding="utf-8"))
    _verify_model_spec(spec)

    altered = deepcopy(spec)
    altered["inference"]["random_seed"] += 1
    try:
        _verify_model_spec(altered)
    except RuntimeError as error:
        assert "inference semantics mismatch" in str(error)
    else:
        raise AssertionError("model-spec guard accepted a changed random seed")

    altered = deepcopy(spec)
    altered["wt_temperature_scope"]["developmental_day_handling"] = (
        "cross-day pooling allowed"
    )
    try:
        _verify_model_spec(altered)
    except RuntimeError as error:
        assert "temperature-stratification semantics mismatch" in str(error)
    else:
        raise AssertionError("model-spec guard accepted cross-day WT pooling")
