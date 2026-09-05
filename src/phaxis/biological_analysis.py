"""PHAxis-owned numerical models for the exploratory biological use case.

This module intentionally contains only the numerical routines consumed by
``scripts/phaxis/analyze_biological_cohorts.py``.  It has no project-path
discovery, figure generation, model inference, annotation loading, or blind
data access.  The implementations preserve the locked predecessor estimands
while giving the installable PHAxis package an independent runtime boundary.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm, t as student_t
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial


EFFECTS: tuple[tuple[str, int], ...] = (
    ("construct_OE_minus_EV", 1),
    ("temperature_30C_minus_22C", 2),
    ("construct_by_temperature_interaction", 3),
)

FACTORIAL_CELLS: tuple[tuple[str, int, str], ...] = (
    ("RHD6-EV", 22, "EV 22°C"),
    ("RHD6-EV", 30, "EV 30°C"),
    ("RHD6-OE", 22, "OE 22°C"),
    ("RHD6-OE", 30, "OE 30°C"),
)

WT_ENDPOINTS: tuple[tuple[str, str, str], ...] = (
    (
        "local_hair_count_1_4mm",
        "Local hair count, 1-4 mm distal window",
        "count_rate",
    ),
    (
        "local_median_hair_length_um_1_4mm",
        "Median hair length, 1-4 mm distal window",
        "continuous_log",
    ),
    (
        "first_hair_ge40um_distance_from_distal_point_um",
        "First >=40 um hair distance from distal root-cap point",
        "conditional_continuous_log",
    ),
    (
        "median_root_width_um",
        "Median main-root width",
        "continuous_log",
    ),
    (
        "visible_root_axis_length_um",
        "Visible point-anchored main-root length",
        "continuous_log",
    ),
)

WT_CONTRAST_MULTIPLICITY_FAMILY = (
    "within_cohort_all_estimated_WT_experiment_by_endpoint_contrasts_"
    "including_unknown_day"
)
WT_META_MULTIPLICITY_FAMILY = (
    "within_cohort_all_estimated_WT_developmental_day_by_endpoint_"
    "meta_analyses"
)

RAW_EFFECT_OLS_MEAN_CONTRAST = "effect_coded_2x2_factorial_OLS_raw_mean_difference"
RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST = (
    "equal_margin_2x2_factorial_cell_raw_median_difference"
)
RAW_EFFECT_HC3_INTERVAL = "HC3_Wald_confidence_interval"
RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL = (
    "source_root_within_cell_stratified_bootstrap_percentile_"
    "2p5_97p5_numpy_linear"
)

WT_CONTRAST_COLUMNS: tuple[str, ...] = (
    "endpoint",
    "endpoint_label",
    "model_component",
    "experiment_key",
    "developmental_day",
    "developmental_day_status",
    "n_total_22C",
    "n_total_30C",
    "n_formal_22C",
    "n_formal_30C",
    "n_endpoint_22C",
    "n_endpoint_30C",
    "mean_22C",
    "mean_30C",
    "median_22C",
    "median_30C",
    "raw_difference_30C_minus_22C",
    "model",
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
    "nb2_alpha",
    "poisson_pearson_dispersion_diagnostic",
    "poisson_fallback_used",
    "analysis_status",
    "not_estimable_reason",
    "meta_eligible",
    "meta_exclusion_reason",
)

WT_META_COLUMNS: tuple[str, ...] = (
    "endpoint",
    "endpoint_label",
    "model_component",
    "developmental_day",
    "k_eligible_experiments",
    "eligible_experiments",
    "excluded_experiments",
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
)

WT_FLOW_COLUMNS: tuple[str, ...] = (
    "experiment_key",
    "developmental_day",
    "developmental_day_status",
    "endpoint",
    "endpoint_label",
    "n_total_22C",
    "n_total_30C",
    "n_formal_22C",
    "n_formal_30C",
    "n_endpoint_22C",
    "n_endpoint_30C",
    "base_gate_pass",
    "endpoint_gate_pass",
    "model_status",
    "not_estimable_reason",
)


def coerce_boolean_series(values: pd.Series) -> pd.Series:
    """Map case-insensitive CSV boolean strings to booleans.

    Unrecognised or missing values remain missing so callers can fail closed
    rather than treating an arbitrary non-empty string as true.
    """

    return values.astype(str).str.casefold().map({"true": True, "false": False})


def factorial_design(frame: pd.DataFrame) -> np.ndarray:
    """Return the locked effect-coded 2 x 2 design matrix."""

    construct = np.where(
        frame["genotype_or_construct"].eq("RHD6-OE"), 0.5, -0.5
    )
    temperature = np.where(
        pd.to_numeric(frame["temperature_c"]).eq(30), 0.5, -0.5
    )
    return np.column_stack(
        (
            np.ones(len(frame), dtype=np.float64),
            construct,
            temperature,
            construct * temperature,
        )
    )


def _stable_offset(*values: object) -> int:
    token = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big") % 100000


def _stratified_bootstrap(
    frame: pd.DataFrame,
    y: np.ndarray,
    *,
    transform_back: Callable[[np.ndarray], np.ndarray],
    replicates: int,
    seed: int,
) -> dict[int, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    cells: list[np.ndarray] = []
    for construct, temperature, _label in FACTORIAL_CELLS:
        keep = (
            frame["genotype_or_construct"].eq(construct)
            & pd.to_numeric(frame["temperature_c"]).eq(temperature)
        ).to_numpy()
        indices = np.flatnonzero(keep)
        if not len(indices):
            raise RuntimeError("factorial bootstrap encountered an empty cell")
        cells.append(indices)
    coefficients = np.empty((replicates, 3), dtype=np.float64)
    for index in range(replicates):
        sampled = np.concatenate(
            [rng.choice(cell, size=len(cell), replace=True) for cell in cells]
        )
        beta = np.linalg.lstsq(
            factorial_design(frame.iloc[sampled]), y[sampled], rcond=None
        )[0]
        coefficients[index] = beta[1:4]
    returned = transform_back(coefficients)
    return {
        column + 1: (
            float(np.quantile(returned[:, column], 0.025)),
            float(np.quantile(returned[:, column], 0.975)),
        )
        for column in range(3)
    }


def raw_median_bootstrap_seed(*, seed: int, field: str, component: str) -> int:
    """Return the locked independent RNG seed for the raw-median companion."""

    return seed + _stable_offset(component, field, "raw_median_bootstrap")


def _factorial_cell_median_contrasts(
    frame: pd.DataFrame, raw: np.ndarray
) -> np.ndarray:
    """Return construct, temperature, and interaction contrasts of cell medians.

    The first two contrasts use equal 0.5/0.5 margins across the other factor;
    the interaction is the usual difference-in-differences.  These definitions
    exactly match the coefficient orientation of :func:`factorial_design` while
    replacing cell means with cell medians.
    """

    if len(frame) != len(raw):
        raise RuntimeError("raw-median contrast frame/value length mismatch")
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("raw-median contrast received non-finite values")
    medians: list[float] = []
    for construct, temperature, _label in FACTORIAL_CELLS:
        keep = (
            frame["genotype_or_construct"].eq(construct)
            & pd.to_numeric(frame["temperature_c"]).eq(temperature)
        ).to_numpy()
        cell = raw[keep]
        if not len(cell):
            raise RuntimeError("raw-median contrast encountered an empty cell")
        medians.append(float(np.median(cell)))
    ev22, ev30, oe22, oe30 = medians
    return np.asarray(
        (
            0.5 * ((oe22 - ev22) + (oe30 - ev30)),
            0.5 * ((ev30 - ev22) + (oe30 - oe22)),
            (oe30 - oe22) - (ev30 - ev22),
        ),
        dtype=np.float64,
    )


def _source_root_sorted_cell_indices(frame: pd.DataFrame) -> list[np.ndarray]:
    """Return reproducibly ordered source-root indices for the four cells."""

    identity_field = next(
        (
            field
            for field in ("task_id", "source_image_sha256")
            if field in frame.columns
        ),
        None,
    )
    if identity_field is None:
        raise RuntimeError(
            "raw-median bootstrap requires source_image_sha256 or task_id"
        )
    identities = frame[identity_field].astype("string")
    if identities.isna().any() or identities.str.strip().eq("").any():
        raise RuntimeError("raw-median bootstrap source-root identity is missing")
    if identities.duplicated().any():
        raise RuntimeError("raw-median bootstrap source-root identity is duplicated")
    cells: list[np.ndarray] = []
    for construct, temperature, _label in FACTORIAL_CELLS:
        keep = (
            frame["genotype_or_construct"].eq(construct)
            & pd.to_numeric(frame["temperature_c"]).eq(temperature)
        ).to_numpy()
        indices = np.flatnonzero(keep)
        if not len(indices):
            raise RuntimeError("raw-median bootstrap encountered an empty cell")
        order = np.argsort(
            identities.iloc[indices].astype(str).to_numpy(), kind="stable"
        )
        cells.append(indices[order])
    return cells


def _stratified_raw_median_bootstrap(
    frame: pd.DataFrame,
    raw: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[int, tuple[float, float]]:
    """Percentile CIs for factorial cell-median contrasts at source-root level."""

    if replicates <= 0:
        raise RuntimeError("raw-median bootstrap replicates must be positive")
    cells = _source_root_sorted_cell_indices(frame)
    rng = np.random.default_rng(seed)
    coefficients = np.empty((replicates, 3), dtype=np.float64)
    for index in range(replicates):
        sampled = np.concatenate(
            [rng.choice(cell, size=len(cell), replace=True) for cell in cells]
        )
        coefficients[index] = _factorial_cell_median_contrasts(
            frame.iloc[sampled].reset_index(drop=True), raw[sampled]
        )
    quantiles = np.quantile(
        coefficients, (0.025, 0.975), axis=0, method="linear"
    )
    return {
        column + 1: (float(quantiles[0, column]), float(quantiles[1, column]))
        for column in range(3)
    }


def _freedman_lane(
    x: np.ndarray,
    y: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> dict[int, float]:
    rng = np.random.default_rng(seed)
    result: dict[int, float] = {}
    full_beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for column in (1, 2, 3):
        reduced = np.delete(x, column, axis=1)
        fitted = reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
        residual = y - fitted
        observed = abs(float(full_beta[column]))
        exceed = 0
        for _ in range(permutations):
            permuted = fitted + residual[rng.permutation(len(residual))]
            coefficient = np.linalg.lstsq(x, permuted, rcond=None)[0][column]
            exceed += abs(float(coefficient)) >= observed - 1e-15
        result[column] = (exceed + 1.0) / (permutations + 1.0)
    return result


def linear_results(
    frame: pd.DataFrame,
    *,
    field: str,
    endpoint_label: str,
    component: str,
    log_transform: bool,
    bootstrap_replicates: int,
    permutations: int,
    seed: int,
    raw_effect_estimand: str = RAW_EFFECT_OLS_MEAN_CONTRAST,
) -> list[dict[str, Any]]:
    """Fit the locked HC3 model and its explicitly typed raw companion."""

    if raw_effect_estimand not in {
        RAW_EFFECT_OLS_MEAN_CONTRAST,
        RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
    }:
        raise ValueError(f"unsupported raw-effect estimand: {raw_effect_estimand}")

    values = pd.to_numeric(frame[field], errors="coerce")
    keep = values.notna() & np.isfinite(values) & (
        values > 0 if log_transform else True
    )
    scoped = frame.loc[keep].reset_index(drop=True)
    raw = values.loc[keep].to_numpy(dtype=np.float64)
    if len(scoped) < 12 or any(
        not np.any(
            scoped["genotype_or_construct"].eq(construct)
            & pd.to_numeric(scoped["temperature_c"]).eq(temperature)
        )
        for construct, temperature, _label in FACTORIAL_CELLS
    ):
        return []
    y = np.log(raw) if log_transform else raw
    x = factorial_design(scoped)
    fitted = sm.OLS(y, x).fit(cov_type="HC3")
    raw_fitted = sm.OLS(raw, x).fit(cov_type="HC3")
    permutation_p = _freedman_lane(
        x,
        y,
        permutations=permutations,
        seed=seed + _stable_offset(field, component, "permutation"),
    )
    transform_back = np.exp if log_transform else (lambda result: result)
    bootstrap = _stratified_bootstrap(
        scoped,
        y,
        transform_back=transform_back,
        replicates=bootstrap_replicates,
        seed=seed + _stable_offset(component, field, "bootstrap"),
    )
    standard_deviation = float(np.std(raw, ddof=1))
    rows: list[dict[str, Any]] = []
    confidence = np.asarray(fitted.conf_int(alpha=0.05), dtype=np.float64)
    raw_confidence = np.asarray(
        raw_fitted.conf_int(alpha=0.05), dtype=np.float64
    )
    raw_effects = {
        column: float(raw_fitted.params[column]) for _effect, column in EFFECTS
    }
    raw_intervals = {
        column: (
            float(raw_confidence[column, 0]),
            float(raw_confidence[column, 1]),
        )
        for _effect, column in EFFECTS
    }
    raw_interval_method = RAW_EFFECT_HC3_INTERVAL
    raw_bootstrap_replicates = 0
    raw_bootstrap_seed: int | None = None
    if raw_effect_estimand == RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST:
        raw_point = _factorial_cell_median_contrasts(scoped, raw)
        raw_effects = {
            column: float(raw_point[column - 1]) for _effect, column in EFFECTS
        }
        raw_bootstrap_replicates = bootstrap_replicates
        raw_bootstrap_seed = raw_median_bootstrap_seed(
            seed=seed, field=field, component=component
        )
        raw_intervals = _stratified_raw_median_bootstrap(
            scoped,
            raw,
            replicates=raw_bootstrap_replicates,
            seed=raw_bootstrap_seed,
        )
        raw_interval_method = RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
    for effect, column in EFFECTS:
        estimate = float(fitted.params[column])
        low, high = confidence[column]
        raw_estimate = raw_effects[column]
        raw_low, raw_high = raw_intervals[column]
        rows.append(
            {
                "endpoint": field,
                "endpoint_label": endpoint_label,
                "model_component": component,
                "effect": effect,
                "model": "HC3_OLS_log" if log_transform else "HC3_OLS_raw",
                "n": len(scoped),
                "estimate": math.exp(estimate) if log_transform else estimate,
                "ci95_low": math.exp(low) if log_transform else low,
                "ci95_high": math.exp(high) if log_transform else high,
                "effect_scale": "ratio" if log_transform else "raw_difference",
                "raw_effect_estimate": raw_estimate,
                "raw_effect_ci95_low": raw_low,
                "raw_effect_ci95_high": raw_high,
                "raw_effect_estimand": raw_effect_estimand,
                "raw_effect_interval_method": raw_interval_method,
                "raw_effect_bootstrap_replicates": raw_bootstrap_replicates,
                "raw_effect_bootstrap_seed": raw_bootstrap_seed,
                "standardized_effect": raw_estimate / standard_deviation
                if standard_deviation > 0
                else float("nan"),
                "standardized_ci95_low": raw_low / standard_deviation
                if standard_deviation > 0
                else float("nan"),
                "standardized_ci95_high": raw_high / standard_deviation
                if standard_deviation > 0
                else float("nan"),
                "p_value_model": float(fitted.pvalues[column]),
                "p_value_freedman_lane": permutation_p[column],
                "bootstrap_ci95_low": bootstrap[column][0],
                "bootstrap_ci95_high": bootstrap[column][1],
            }
        )
    return rows


def count_results(
    frame: pd.DataFrame,
    *,
    field: str,
    endpoint_label: str,
    bootstrap_replicates: int,
    permutations: int,
    seed: int,
    poisson_fallback_max_dispersion: float,
) -> list[dict[str, Any]]:
    """Fit the locked NB2 count model with its fail-closed Poisson fallback."""

    values = pd.to_numeric(frame[field], errors="coerce")
    keep = values.notna() & np.isfinite(values) & (values >= 0)
    scoped = frame.loc[keep].reset_index(drop=True)
    y = values.loc[keep].to_numpy(dtype=np.float64)
    if len(scoped) < 12:
        return []
    x = factorial_design(scoped)
    offset = np.full(len(scoped), math.log(3.0), dtype=np.float64)
    model_name = "negative_binomial_NB2_HC0"
    poisson_diagnostic = sm.GLM(
        y, x, offset=offset, family=sm.families.Poisson()
    ).fit()
    poisson_pearson_dispersion = float(
        np.sum(
            np.asarray(poisson_diagnostic.resid_pearson, dtype=np.float64) ** 2
        )
        / max(float(poisson_diagnostic.df_resid), 1.0)
    )
    poisson_fallback_used = False
    nb2_alpha = float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = NegativeBinomial(
                y, x, offset=offset, loglike_method="nb2"
            ).fit(disp=0, maxiter=500, cov_type="HC0")
        params = np.asarray(fitted.params[:4], dtype=np.float64)
        standard_errors = np.asarray(fitted.bse[:4], dtype=np.float64)
        pvalues = np.asarray(fitted.pvalues[:4], dtype=np.float64)
        if not bool(getattr(fitted, "mle_retvals", {}).get("converged", False)):
            raise RuntimeError("NB2 did not converge")
        if not (
            np.all(np.isfinite(params))
            and np.all(np.isfinite(standard_errors))
            and np.all(np.isfinite(pvalues))
        ):
            raise RuntimeError("NB2 inference is non-finite")
        nb2_alpha = float(np.asarray(fitted.params, dtype=np.float64)[-1])
        if not math.isfinite(nb2_alpha) or nb2_alpha <= 0:
            raise RuntimeError("NB2 dispersion is non-positive")
    except Exception as error:
        if not (
            math.isfinite(poisson_fallback_max_dispersion)
            and poisson_fallback_max_dispersion > 0
            and poisson_pearson_dispersion <= poisson_fallback_max_dispersion
        ):
            raise RuntimeError(
                "preregistered NB2 failed and Poisson dispersion exceeded the "
                "locked fallback threshold"
            ) from error
        model_name = "Poisson_log_link_HC3_locked_dispersion_fallback"
        fitted = sm.GLM(
            y, x, offset=offset, family=sm.families.Poisson()
        ).fit(cov_type="HC3")
        params = np.asarray(fitted.params, dtype=np.float64)
        standard_errors = np.asarray(fitted.bse, dtype=np.float64)
        pvalues = np.asarray(fitted.pvalues, dtype=np.float64)
        if not (
            np.all(np.isfinite(params))
            and np.all(np.isfinite(standard_errors))
            and np.all(np.isfinite(pvalues))
        ):
            raise RuntimeError("locked Poisson fallback inference is non-finite")
        poisson_fallback_used = True
    transformed = np.log1p(y)
    permutation_p = _freedman_lane(
        x,
        transformed,
        permutations=permutations,
        seed=seed + _stable_offset(field, "count", "permutation"),
    )
    bootstrap = _stratified_bootstrap(
        scoped,
        np.log((y + 0.5) / 3.0),
        transform_back=np.exp,
        replicates=bootstrap_replicates,
        seed=seed + _stable_offset(field, "count", "bootstrap"),
    )
    raw_fit = sm.OLS(y, x).fit(cov_type="HC3")
    raw_ci = np.asarray(raw_fit.conf_int(alpha=0.05), dtype=np.float64)
    standard_deviation = float(np.std(y, ddof=1))
    rows: list[dict[str, Any]] = []
    for effect, column in EFFECTS:
        low = params[column] - norm.ppf(0.975) * standard_errors[column]
        high = params[column] + norm.ppf(0.975) * standard_errors[column]
        rows.append(
            {
                "endpoint": field,
                "endpoint_label": endpoint_label,
                "model_component": "count_rate",
                "effect": effect,
                "model": model_name,
                "n": len(scoped),
                "estimate": math.exp(params[column]),
                "ci95_low": math.exp(low),
                "ci95_high": math.exp(high),
                "effect_scale": "rate_ratio",
                "raw_effect_estimate": float(raw_fit.params[column]),
                "raw_effect_ci95_low": float(raw_ci[column, 0]),
                "raw_effect_ci95_high": float(raw_ci[column, 1]),
                "raw_effect_estimand": RAW_EFFECT_OLS_MEAN_CONTRAST,
                "raw_effect_interval_method": RAW_EFFECT_HC3_INTERVAL,
                "raw_effect_bootstrap_replicates": 0,
                "raw_effect_bootstrap_seed": None,
                "standardized_effect": float(raw_fit.params[column])
                / standard_deviation
                if standard_deviation > 0
                else float("nan"),
                "standardized_ci95_low": float(raw_ci[column, 0])
                / standard_deviation
                if standard_deviation > 0
                else float("nan"),
                "standardized_ci95_high": float(raw_ci[column, 1])
                / standard_deviation
                if standard_deviation > 0
                else float("nan"),
                "p_value_model": float(pvalues[column]),
                "p_value_freedman_lane": permutation_p[column],
                "bootstrap_ci95_low": bootstrap[column][0],
                "bootstrap_ci95_high": bootstrap[column][1],
                "nb2_alpha": nb2_alpha,
                "poisson_pearson_dispersion_diagnostic": (
                    poisson_pearson_dispersion
                ),
                "poisson_fallback_max_dispersion_locked": (
                    poisson_fallback_max_dispersion
                ),
                "poisson_fallback_used": poisson_fallback_used,
            }
        )
    return rows


def _temperature_design(frame: pd.DataFrame) -> np.ndarray:
    """Return an intercept plus the within-experiment 30 C indicator."""

    temperature = pd.to_numeric(frame["temperature_c"], errors="coerce")
    if temperature.isna().any() or not temperature.isin([22, 30]).all():
        raise RuntimeError("WT temperature contrast contains a non-22/30 C value")
    return np.column_stack(
        (
            np.ones(len(frame), dtype=np.float64),
            temperature.eq(30).to_numpy(dtype=np.float64),
        )
    )


def _wt_count_temperature_fit(
    frame: pd.DataFrame,
    y: np.ndarray,
    *,
    poisson_fallback_max_dispersion: float,
) -> dict[str, Any]:
    """Fit the two-arm WT count contrast with the locked NB2 fallback path."""

    x = _temperature_design(frame)
    offset = np.full(len(frame), math.log(3.0), dtype=np.float64)
    poisson_diagnostic = sm.GLM(
        y, x, offset=offset, family=sm.families.Poisson()
    ).fit()
    poisson_dispersion = float(
        np.sum(
            np.asarray(poisson_diagnostic.resid_pearson, dtype=np.float64) ** 2
        )
        / max(float(poisson_diagnostic.df_resid), 1.0)
    )
    model_name = "negative_binomial_NB2_HC0"
    fallback_used = False
    nb2_alpha = float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = NegativeBinomial(
                y, x, offset=offset, loglike_method="nb2"
            ).fit(disp=0, maxiter=500, cov_type="HC0")
        params = np.asarray(fitted.params[:2], dtype=np.float64)
        standard_errors = np.asarray(fitted.bse[:2], dtype=np.float64)
        pvalues = np.asarray(fitted.pvalues[:2], dtype=np.float64)
        if not bool(getattr(fitted, "mle_retvals", {}).get("converged", False)):
            raise RuntimeError("NB2 did not converge")
        if not (
            np.all(np.isfinite(params))
            and np.all(np.isfinite(standard_errors))
            and np.all(np.isfinite(pvalues))
        ):
            raise RuntimeError("NB2 inference is non-finite")
        nb2_alpha = float(np.asarray(fitted.params, dtype=np.float64)[-1])
        if not math.isfinite(nb2_alpha) or nb2_alpha <= 0:
            raise RuntimeError("NB2 dispersion is non-positive")
    except Exception as error:
        if not (
            math.isfinite(poisson_fallback_max_dispersion)
            and poisson_fallback_max_dispersion > 0
            and poisson_dispersion <= poisson_fallback_max_dispersion
        ):
            raise RuntimeError(
                "WT NB2 failed and Poisson dispersion exceeded the locked "
                "fallback threshold"
            ) from error
        model_name = "Poisson_log_link_HC3_locked_dispersion_fallback"
        fitted = sm.GLM(
            y, x, offset=offset, family=sm.families.Poisson()
        ).fit(cov_type="HC3")
        params = np.asarray(fitted.params, dtype=np.float64)
        standard_errors = np.asarray(fitted.bse, dtype=np.float64)
        pvalues = np.asarray(fitted.pvalues, dtype=np.float64)
        if not (
            np.all(np.isfinite(params))
            and np.all(np.isfinite(standard_errors))
            and np.all(np.isfinite(pvalues))
        ):
            raise RuntimeError("WT locked Poisson fallback inference is non-finite")
        fallback_used = True
    coefficient = float(params[1])
    standard_error = float(standard_errors[1])
    if not math.isfinite(standard_error) or standard_error <= 0:
        raise RuntimeError("WT count contrast has a non-positive standard error")
    critical = float(norm.ppf(0.975))
    return {
        "model": model_name,
        "log_effect": coefficient,
        "standard_error": standard_error,
        "p_value": float(pvalues[1]),
        "ci95_low": coefficient - critical * standard_error,
        "ci95_high": coefficient + critical * standard_error,
        "nb2_alpha": nb2_alpha,
        "poisson_dispersion": poisson_dispersion,
        "poisson_fallback_used": fallback_used,
    }


def _wt_continuous_temperature_fit(
    frame: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, Any]:
    """Fit one positive WT endpoint as a log-scale HC3 two-arm contrast."""

    x = _temperature_design(frame)
    if np.linalg.matrix_rank(x) != 2:
        raise RuntimeError("WT continuous temperature design is rank deficient")
    fitted = sm.OLS(np.log(y), x).fit(cov_type="HC3")
    params = np.asarray(fitted.params, dtype=np.float64)
    standard_errors = np.asarray(fitted.bse, dtype=np.float64)
    pvalues = np.asarray(fitted.pvalues, dtype=np.float64)
    if not (
        np.all(np.isfinite(params))
        and np.all(np.isfinite(standard_errors))
        and np.all(np.isfinite(pvalues))
    ):
        raise RuntimeError("WT HC3 log-OLS inference is non-finite")
    coefficient = float(params[1])
    standard_error = float(standard_errors[1])
    if standard_error <= 0:
        raise RuntimeError("WT continuous contrast has a non-positive standard error")
    critical = float(norm.ppf(0.975))
    return {
        "model": "HC3_OLS_log_within_experiment_temperature",
        "log_effect": coefficient,
        "standard_error": standard_error,
        "p_value": float(pvalues[1]),
        "ci95_low": coefficient - critical * standard_error,
        "ci95_high": coefficient + critical * standard_error,
        "nb2_alpha": float("nan"),
        "poisson_dispersion": float("nan"),
        "poisson_fallback_used": False,
    }


def random_effects_reml_hartung_knapp(
    effect: np.ndarray,
    variance: np.ndarray,
) -> dict[str, float]:
    """Synthesize at least three same-day log effects by REML and HK.

    The minimum of three experiments is inherited from the locked WT study
    contract.  Callers must never use this routine to pool developmental days.
    """

    effect = np.asarray(effect, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    keep = np.isfinite(effect) & np.isfinite(variance) & (variance > 0)
    y, v = effect[keep], variance[keep]
    k = len(y)
    if k < 3:
        raise RuntimeError("REML/Hartung-Knapp requires at least three experiments")

    def objective(tau2: float) -> float:
        weights = 1.0 / (v + tau2)
        mean = float(np.sum(weights * y) / np.sum(weights))
        return 0.5 * float(
            np.sum(np.log(v + tau2))
            + math.log(float(np.sum(weights)))
            + np.sum(weights * (y - mean) ** 2)
        )

    empirical_variance = float(np.var(y, ddof=1))
    upper = max(empirical_variance, float(np.max(v)), 1e-9) * 100.0
    optimized = minimize_scalar(
        objective,
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    if not optimized.success or not math.isfinite(float(optimized.fun)):
        raise RuntimeError("WT REML optimization did not converge")
    tau2 = max(0.0, float(optimized.x))
    if objective(0.0) <= float(optimized.fun) + 1e-12:
        tau2 = 0.0
    weights = 1.0 / (v + tau2)
    estimate = float(np.sum(weights * y) / np.sum(weights))
    residual_sum = float(np.sum(weights * (y - estimate) ** 2))
    hk_scale = residual_sum / float(k - 1)
    hk_variance = hk_scale / float(np.sum(weights))
    if not math.isfinite(hk_variance) or hk_variance <= 0:
        raise RuntimeError("WT Hartung-Knapp variance is non-positive")
    standard_error = math.sqrt(hk_variance)
    critical = float(student_t.ppf(0.975, df=k - 1))
    statistic = estimate / standard_error
    p_value = float(2.0 * student_t.sf(abs(statistic), df=k - 1))

    fixed_weights = 1.0 / v
    fixed_mean = float(np.sum(fixed_weights * y) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * (y - fixed_mean) ** 2))
    q_df = k - 1
    i2 = max(0.0, (q - q_df) / q) if q > 0 else 0.0
    return {
        "k": float(k),
        "estimate": estimate,
        "standard_error_hartung_knapp": standard_error,
        "ci95_low": estimate - critical * standard_error,
        "ci95_high": estimate + critical * standard_error,
        "p_value_hartung_knapp": p_value,
        "tau2": tau2,
        "Q": q,
        "Q_df": float(q_df),
        "Q_p_value": float(chi2.sf(q, df=q_df)),
        "I2": i2,
        "hartung_knapp_scale": hk_scale,
    }


def _wt_not_estimable_reason(
    *,
    n_formal_22: int,
    n_formal_30: int,
    n_endpoint_22: int,
    n_endpoint_30: int,
    minimum: int,
) -> str:
    reasons: list[str] = []
    if n_formal_22 < minimum:
        reasons.append(f"base_22C_n_below_{minimum}")
    if n_formal_30 < minimum:
        reasons.append(f"base_30C_n_below_{minimum}")
    if not reasons:
        if n_endpoint_22 < minimum:
            reasons.append(f"endpoint_22C_n_below_{minimum}")
        if n_endpoint_30 < minimum:
            reasons.append(f"endpoint_30C_n_below_{minimum}")
    return ";".join(reasons)


def wt_temperature_secondary_results(
    frame: pd.DataFrame,
    *,
    minimum_per_temperature: int,
    minimum_experiments_per_day_meta: int,
    poisson_fallback_max_dispersion: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build WT experiment contrasts and strictly within-day REML/HK rows."""

    if minimum_per_temperature < 3:
        raise ValueError("WT per-arm minimum cannot be below three")
    if minimum_experiments_per_day_meta < 3:
        raise ValueError("WT same-day meta-analysis minimum cannot be below three")
    required = {
        "task_id",
        "experiment_key",
        "study_role",
        "developmental_day",
        "condition_code",
        "genotype_or_construct",
        "temperature_c",
        "formal_statistics_eligible",
        *(field for field, _label, _component in WT_ENDPOINTS),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"WT secondary analysis is missing fields: {missing}")
    wt = frame.loc[frame["study_role"].eq("wt_temperature_block")].copy()
    if wt.empty:
        return (
            pd.DataFrame(columns=WT_CONTRAST_COLUMNS),
            pd.DataFrame(columns=WT_META_COLUMNS),
            pd.DataFrame(columns=WT_FLOW_COLUMNS),
        )
    if "blind_images_used" in wt and not pd.to_numeric(
        wt["blind_images_used"], errors="coerce"
    ).eq(0).all():
        raise RuntimeError("WT secondary analysis input is blind-tainted")
    if "root_cap_region_output" in wt and not coerce_boolean_series(
        wt["root_cap_region_output"]
    ).eq(False).all():
        raise RuntimeError("WT secondary analysis contains root-cap regions")
    if wt["task_id"].astype(str).duplicated().any():
        raise RuntimeError("WT secondary analysis contains duplicate task IDs")
    if "source_image_sha256" in wt and wt["source_image_sha256"].astype(
        str
    ).str.casefold().duplicated().any():
        raise RuntimeError("WT secondary analysis contains duplicate source images")
    formal = coerce_boolean_series(wt["formal_statistics_eligible"])
    if formal.isna().any():
        raise RuntimeError("WT formal_statistics_eligible contains unknown values")
    wt["_formal"] = formal.astype(bool)
    temperature = pd.to_numeric(wt["temperature_c"], errors="coerce")
    if temperature.isna().any() or not temperature.isin([22, 30]).all():
        raise RuntimeError("WT secondary scope contains a non-22/30 C temperature")
    wt["_temperature"] = temperature.astype(int)
    if not wt["genotype_or_construct"].astype(str).eq("WT").all():
        raise RuntimeError("WT secondary scope contains a non-WT construct")
    expected_condition = wt["_temperature"].map({22: "WT_22C", 30: "WT_30C"})
    if not wt["condition_code"].astype(str).eq(expected_condition).all():
        raise RuntimeError("WT condition code and temperature disagree")
    if wt["experiment_key"].isna().any() or wt["experiment_key"].astype(
        str
    ).str.strip().eq("").any():
        raise RuntimeError("WT secondary scope contains an empty experiment key")

    contrast_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    for experiment_key, experiment in wt.groupby(
        "experiment_key", sort=True, dropna=False
    ):
        experiment = experiment.sort_values(
            ["_temperature", "task_id"], kind="mergesort"
        ).reset_index(drop=True)
        day_values = pd.to_numeric(
            experiment["developmental_day"], errors="coerce"
        )
        finite_days = sorted(
            {float(value) for value in day_values[np.isfinite(day_values)]}
        )
        if any(
            value <= 0 or not math.isclose(value, round(value))
            for value in finite_days
        ):
            raise RuntimeError(
                f"{experiment_key}: WT developmental day is not a positive integer"
            )
        if len(finite_days) > 1:
            raise RuntimeError(
                f"{experiment_key}: WT experiment maps to multiple developmental days"
            )
        if len(finite_days) == 1 and day_values.notna().all():
            developmental_day = finite_days[0]
            day_status = "known_consistent"
        else:
            developmental_day = float("nan")
            day_status = (
                "unknown_all_rows"
                if day_values.isna().all()
                else "unknown_partial_metadata"
            )
        total_counts = experiment["_temperature"].value_counts()
        formal_experiment = experiment.loc[experiment["_formal"]].copy()
        formal_counts = formal_experiment["_temperature"].value_counts()
        n_total_22 = int(total_counts.get(22, 0))
        n_total_30 = int(total_counts.get(30, 0))
        n_formal_22 = int(formal_counts.get(22, 0))
        n_formal_30 = int(formal_counts.get(30, 0))

        for field, label, component in WT_ENDPOINTS:
            values = pd.to_numeric(formal_experiment[field], errors="coerce")
            valid = values.notna() & np.isfinite(values)
            if component == "count_rate":
                valid &= values.ge(0) & np.isclose(values, np.round(values))
                if "distal_window_1_4mm_eligible" in formal_experiment:
                    window = coerce_boolean_series(
                        formal_experiment["distal_window_1_4mm_eligible"]
                    )
                    valid &= window.eq(True)
            else:
                valid &= values.gt(0)
            endpoint = formal_experiment.loc[valid].copy()
            endpoint["_value"] = values.loc[valid].to_numpy(dtype=np.float64)
            endpoint_counts = endpoint["_temperature"].value_counts()
            n_endpoint_22 = int(endpoint_counts.get(22, 0))
            n_endpoint_30 = int(endpoint_counts.get(30, 0))
            reason = _wt_not_estimable_reason(
                n_formal_22=n_formal_22,
                n_formal_30=n_formal_30,
                n_endpoint_22=n_endpoint_22,
                n_endpoint_30=n_endpoint_30,
                minimum=minimum_per_temperature,
            )
            base_gate = (
                n_formal_22 >= minimum_per_temperature
                and n_formal_30 >= minimum_per_temperature
            )
            endpoint_gate = (
                n_endpoint_22 >= minimum_per_temperature
                and n_endpoint_30 >= minimum_per_temperature
            )
            cold = endpoint.loc[endpoint["_temperature"].eq(22), "_value"]
            hot = endpoint.loc[endpoint["_temperature"].eq(30), "_value"]
            row: dict[str, Any] = {
                "endpoint": field,
                "endpoint_label": label,
                "model_component": component,
                "experiment_key": str(experiment_key),
                "developmental_day": developmental_day,
                "developmental_day_status": day_status,
                "n_total_22C": n_total_22,
                "n_total_30C": n_total_30,
                "n_formal_22C": n_formal_22,
                "n_formal_30C": n_formal_30,
                "n_endpoint_22C": n_endpoint_22,
                "n_endpoint_30C": n_endpoint_30,
                "mean_22C": float(cold.mean()) if len(cold) else float("nan"),
                "mean_30C": float(hot.mean()) if len(hot) else float("nan"),
                "median_22C": float(cold.median()) if len(cold) else float("nan"),
                "median_30C": float(hot.median()) if len(hot) else float("nan"),
                "raw_difference_30C_minus_22C": (
                    float(hot.mean() - cold.mean())
                    if len(cold) and len(hot)
                    else float("nan")
                ),
                "model": "",
                "effect_scale": "ratio_30C_over_22C",
                "log_effect_30C_over_22C": float("nan"),
                "log_effect_standard_error": float("nan"),
                "sampling_variance": float("nan"),
                "estimate_30C_over_22C": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
                "p_value_model": float("nan"),
                "p_value_model_BH_FDR": float("nan"),
                "reject_model_BH_FDR_0p05": False,
                "multiplicity_family": WT_CONTRAST_MULTIPLICITY_FAMILY,
                "nb2_alpha": float("nan"),
                "poisson_pearson_dispersion_diagnostic": float("nan"),
                "poisson_fallback_used": False,
                "analysis_status": "not_estimable",
                "not_estimable_reason": reason,
                "meta_eligible": False,
                "meta_exclusion_reason": reason,
            }
            if not reason:
                try:
                    y = endpoint["_value"].to_numpy(dtype=np.float64)
                    fitted = (
                        _wt_count_temperature_fit(
                            endpoint,
                            y,
                            poisson_fallback_max_dispersion=(
                                poisson_fallback_max_dispersion
                            ),
                        )
                        if component == "count_rate"
                        else _wt_continuous_temperature_fit(endpoint, y)
                    )
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                    reason = "model_failure:" + str(error).replace(";", ",")
                    row["not_estimable_reason"] = reason
                    row["meta_exclusion_reason"] = reason
                else:
                    coefficient = float(fitted["log_effect"])
                    standard_error = float(fitted["standard_error"])
                    row.update(
                        {
                            "model": fitted["model"],
                            "log_effect_30C_over_22C": coefficient,
                            "log_effect_standard_error": standard_error,
                            "sampling_variance": standard_error**2,
                            "estimate_30C_over_22C": math.exp(coefficient),
                            "ci95_low": math.exp(float(fitted["ci95_low"])),
                            "ci95_high": math.exp(float(fitted["ci95_high"])),
                            "p_value_model": float(fitted["p_value"]),
                            "nb2_alpha": float(fitted["nb2_alpha"]),
                            "poisson_pearson_dispersion_diagnostic": float(
                                fitted["poisson_dispersion"]
                            ),
                            "poisson_fallback_used": bool(
                                fitted["poisson_fallback_used"]
                            ),
                            "analysis_status": "estimated",
                            "not_estimable_reason": "",
                            "meta_eligible": math.isfinite(developmental_day),
                            "meta_exclusion_reason": (
                                ""
                                if math.isfinite(developmental_day)
                                else "unknown_developmental_day"
                            ),
                        }
                    )
            contrast_rows.append(row)
            flow_rows.append(
                {
                    "experiment_key": str(experiment_key),
                    "developmental_day": developmental_day,
                    "developmental_day_status": day_status,
                    "endpoint": field,
                    "endpoint_label": label,
                    "n_total_22C": n_total_22,
                    "n_total_30C": n_total_30,
                    "n_formal_22C": n_formal_22,
                    "n_formal_30C": n_formal_30,
                    "n_endpoint_22C": n_endpoint_22,
                    "n_endpoint_30C": n_endpoint_30,
                    "base_gate_pass": base_gate,
                    "endpoint_gate_pass": endpoint_gate,
                    "model_status": row["analysis_status"],
                    "not_estimable_reason": row["not_estimable_reason"],
                }
            )

    endpoint_order = {
        field: index for index, (field, _label, _component) in enumerate(WT_ENDPOINTS)
    }
    contrast_frame = pd.DataFrame(contrast_rows, columns=WT_CONTRAST_COLUMNS)
    contrast_frame["_endpoint_order"] = contrast_frame["endpoint"].map(
        endpoint_order
    )
    contrast_frame["_day_missing"] = contrast_frame["developmental_day"].isna()
    contrast_frame = (
        contrast_frame.sort_values(
            [
                "_day_missing",
                "developmental_day",
                "experiment_key",
                "_endpoint_order",
            ],
            kind="mergesort",
        )
        .drop(columns=["_endpoint_order", "_day_missing"])
        .reset_index(drop=True)
    )
    flow_frame = pd.DataFrame(flow_rows, columns=WT_FLOW_COLUMNS)
    flow_frame["_endpoint_order"] = flow_frame["endpoint"].map(endpoint_order)
    flow_frame["_day_missing"] = flow_frame["developmental_day"].isna()
    flow_frame = (
        flow_frame.sort_values(
            [
                "_day_missing",
                "developmental_day",
                "experiment_key",
                "_endpoint_order",
            ],
            kind="mergesort",
        )
        .drop(columns=["_endpoint_order", "_day_missing"])
        .reset_index(drop=True)
    )

    known_days = sorted(
        {
            float(value)
            for value in contrast_frame["developmental_day"]
            if math.isfinite(float(value))
        }
    )
    meta_rows: list[dict[str, Any]] = []
    for day in known_days:
        for field, label, component in WT_ENDPOINTS:
            scoped = contrast_frame.loc[
                contrast_frame["endpoint"].eq(field)
                & contrast_frame["developmental_day"].eq(day)
            ].copy()
            eligible = scoped.loc[
                scoped["analysis_status"].eq("estimated")
                & scoped["meta_eligible"].eq(True)
            ].copy()
            eligible_experiments = sorted(eligible["experiment_key"].astype(str))
            excluded_experiments = sorted(
                scoped.loc[~scoped.index.isin(eligible.index), "experiment_key"]
                .astype(str)
                .unique()
            )
            k = len(eligible)
            meta_row: dict[str, Any] = {
                "endpoint": field,
                "endpoint_label": label,
                "model_component": component,
                "developmental_day": day,
                "k_eligible_experiments": k,
                "eligible_experiments": ";".join(eligible_experiments),
                "excluded_experiments": ";".join(excluded_experiments),
                "model": "random_effects_REML_Hartung_Knapp",
                "effect_scale": "ratio_30C_over_22C",
                "log_effect_30C_over_22C": float("nan"),
                "log_effect_standard_error_hartung_knapp": float("nan"),
                "estimate_30C_over_22C": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
                "p_value_hartung_knapp": float("nan"),
                "p_value_hartung_knapp_BH_FDR": float("nan"),
                "reject_hartung_knapp_BH_FDR_0p05": False,
                "multiplicity_family": WT_META_MULTIPLICITY_FAMILY,
                "tau2_reml_log_scale": float("nan"),
                "Q": float("nan"),
                "Q_df": float("nan"),
                "Q_p_value": float("nan"),
                "I2": float("nan"),
                "I2_percent": float("nan"),
                "hartung_knapp_scale": float("nan"),
                "analysis_status": "not_estimable",
                "not_estimable_reason": (
                    f"fewer_than_{minimum_experiments_per_day_meta}_"
                    "estimable_same_day_experiments"
                ),
                "cross_day_pooling_performed": False,
                "unknown_day_contrasts_included": False,
            }
            if k >= minimum_experiments_per_day_meta:
                try:
                    synthesized = random_effects_reml_hartung_knapp(
                        eligible["log_effect_30C_over_22C"].to_numpy(
                            dtype=np.float64
                        ),
                        eligible["sampling_variance"].to_numpy(dtype=np.float64),
                    )
                except (RuntimeError, ValueError) as error:
                    meta_row["not_estimable_reason"] = (
                        "meta_model_failure:" + str(error).replace(";", ",")
                    )
                else:
                    coefficient = float(synthesized["estimate"])
                    meta_row.update(
                        {
                            "log_effect_30C_over_22C": coefficient,
                            "log_effect_standard_error_hartung_knapp": float(
                                synthesized["standard_error_hartung_knapp"]
                            ),
                            "estimate_30C_over_22C": math.exp(coefficient),
                            "ci95_low": math.exp(float(synthesized["ci95_low"])),
                            "ci95_high": math.exp(float(synthesized["ci95_high"])),
                            "p_value_hartung_knapp": float(
                                synthesized["p_value_hartung_knapp"]
                            ),
                            "tau2_reml_log_scale": float(synthesized["tau2"]),
                            "Q": float(synthesized["Q"]),
                            "Q_df": float(synthesized["Q_df"]),
                            "Q_p_value": float(synthesized["Q_p_value"]),
                            "I2": float(synthesized["I2"]),
                            "I2_percent": 100.0 * float(synthesized["I2"]),
                            "hartung_knapp_scale": float(
                                synthesized["hartung_knapp_scale"]
                            ),
                            "analysis_status": "estimated",
                            "not_estimable_reason": "",
                        }
                    )
            meta_rows.append(meta_row)
    meta_frame = pd.DataFrame(meta_rows, columns=WT_META_COLUMNS)
    if not meta_frame.empty:
        meta_frame["_endpoint_order"] = meta_frame["endpoint"].map(endpoint_order)
        meta_frame = (
            meta_frame.sort_values(
                ["developmental_day", "_endpoint_order"], kind="mergesort"
            )
            .drop(columns=["_endpoint_order"])
            .reset_index(drop=True)
        )
    return contrast_frame, meta_frame, flow_frame


def observability_results(
    frame: pd.DataFrame,
    *,
    field: str,
    endpoint_label: str,
) -> list[dict[str, Any]]:
    """Fit the observability component of the locked two-part endpoint."""

    observed = (
        pd.to_numeric(frame[field], errors="coerce")
        .notna()
        .astype(float)
        .to_numpy()
    )
    x = factorial_design(frame)
    if len(np.unique(observed)) < 2:
        return []
    try:
        fitted = sm.GLM(
            observed, x, family=sm.families.Binomial()
        ).fit(cov_type="HC3")
    except Exception:
        return []
    confidence = np.asarray(fitted.conf_int(alpha=0.05), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for effect, column in EFFECTS:
        rows.append(
            {
                "endpoint": field,
                "endpoint_label": endpoint_label,
                "model_component": "observability",
                "effect": effect,
                "model": "binomial_logit_HC3",
                "n": len(frame),
                "estimate": math.exp(float(fitted.params[column])),
                "ci95_low": math.exp(float(confidence[column, 0])),
                "ci95_high": math.exp(float(confidence[column, 1])),
                "effect_scale": "odds_ratio",
                "raw_effect_estimate": float("nan"),
                "raw_effect_ci95_low": float("nan"),
                "raw_effect_ci95_high": float("nan"),
                "standardized_effect": float("nan"),
                "standardized_ci95_low": float("nan"),
                "standardized_ci95_high": float("nan"),
                "p_value_model": float(fitted.pvalues[column]),
                "p_value_freedman_lane": float("nan"),
                "bootstrap_ci95_low": float("nan"),
                "bootstrap_ci95_high": float("nan"),
            }
        )
    return rows


def robust_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """Run Huber and leave-one-unit-out checks on the locked analysis scales."""

    specifications = (
        (
            "local_hair_count_1_4mm",
            "count_rate_sensitivity",
            lambda values: np.log((values + 0.5) / 3.0),
            True,
        ),
        (
            "local_median_hair_length_um_1_4mm",
            "continuous_log",
            np.log,
            False,
        ),
        (
            "first_hair_ge40um_distance_from_distal_point_um",
            "conditional_continuous_log",
            np.log,
            False,
        ),
        ("median_root_width_um", "continuous_log", np.log, False),
        ("visible_root_axis_length_um", "continuous_log", np.log, False),
    )
    rows: list[dict[str, Any]] = []
    for field, component, transform, allow_zero in specifications:
        values = pd.to_numeric(frame[field], errors="coerce")
        keep = values.notna() & np.isfinite(values) & (
            values >= 0 if allow_zero else values > 0
        )
        scoped = frame.loc[keep].reset_index(drop=True)
        raw = values.loc[keep].to_numpy(dtype=np.float64)
        if len(scoped) < 12:
            continue
        x = factorial_design(scoped)
        y = np.asarray(transform(raw), dtype=np.float64)
        full = np.linalg.lstsq(x, y, rcond=None)[0]
        try:
            huber = np.asarray(
                sm.RLM(y, x, M=sm.robust.norms.HuberT())
                .fit(maxiter=500)
                .params,
                dtype=np.float64,
            )
        except Exception:
            huber = np.full(x.shape[1], np.nan, dtype=np.float64)
        leave_one_out = np.empty((len(scoped), 3), dtype=np.float64)
        for index in range(len(scoped)):
            retained = np.arange(len(scoped)) != index
            leave_one_out[index] = np.linalg.lstsq(
                x[retained], y[retained], rcond=None
            )[0][1:4]
        for effect, column in EFFECTS:
            coefficient = float(full[column])
            leave_one_out_effect = leave_one_out[:, column - 1]
            reference_sign = (
                0.0
                if abs(coefficient) < 1e-12
                else math.copysign(1.0, coefficient)
            )
            leave_one_out_sign = np.where(
                np.abs(leave_one_out_effect) < 1e-12,
                0.0,
                np.sign(leave_one_out_effect),
            )
            rows.append(
                {
                    "endpoint": field,
                    "model_component": component,
                    "effect": effect,
                    "n": len(scoped),
                    "analysis_scale": "log_rate" if allow_zero else "log",
                    "ols_coefficient": coefficient,
                    "huber_coefficient": float(huber[column]),
                    "huber_same_direction": bool(
                        math.isfinite(float(huber[column]))
                        and (
                            reference_sign == 0.0
                            or math.copysign(1.0, float(huber[column]))
                            == reference_sign
                        )
                    ),
                    "leave_one_out_min": float(np.min(leave_one_out_effect)),
                    "leave_one_out_max": float(np.max(leave_one_out_effect)),
                    "leave_one_out_same_direction_fraction": float(
                        np.mean(leave_one_out_sign == reference_sign)
                    ),
                    "leave_one_out_all_same_direction": bool(
                        np.all(leave_one_out_sign == reference_sign)
                    ),
                }
            )
    return pd.DataFrame(rows)


def group_summaries(
    frame: pd.DataFrame,
    fields: Sequence[tuple[str, str]],
    *,
    scope: str,
) -> pd.DataFrame:
    """Summarize endpoint availability and distribution within each cell."""

    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        [
            "experiment_key",
            "condition_code",
            "genotype_or_construct",
            "temperature_c",
        ],
        dropna=False,
    ):
        for field, label in fields:
            values = (
                pd.to_numeric(group[field], errors="coerce")
                if field in group
                else pd.Series(dtype=float)
            )
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "scope": scope,
                    "experiment_key": keys[0],
                    "condition_code": keys[1],
                    "genotype_or_construct": keys[2],
                    "temperature_c": keys[3],
                    "endpoint": field,
                    "endpoint_label": label,
                    "n_total": len(group),
                    "n_nonmissing": len(finite),
                    "n_missing": len(group) - len(finite),
                    "mean": float(finite.mean())
                    if len(finite)
                    else float("nan"),
                    "sd": float(finite.std(ddof=1))
                    if len(finite) > 1
                    else float("nan"),
                    "median": float(finite.median())
                    if len(finite)
                    else float("nan"),
                    "q25": float(finite.quantile(0.25))
                    if len(finite)
                    else float("nan"),
                    "q75": float(finite.quantile(0.75))
                    if len(finite)
                    else float("nan"),
                    "minimum": float(finite.min())
                    if len(finite)
                    else float("nan"),
                    "maximum": float(finite.max())
                    if len(finite)
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "EFFECTS",
    "FACTORIAL_CELLS",
    "WT_CONTRAST_COLUMNS",
    "WT_CONTRAST_MULTIPLICITY_FAMILY",
    "WT_ENDPOINTS",
    "WT_FLOW_COLUMNS",
    "WT_META_COLUMNS",
    "WT_META_MULTIPLICITY_FAMILY",
    "coerce_boolean_series",
    "count_results",
    "factorial_design",
    "group_summaries",
    "linear_results",
    "observability_results",
    "random_effects_reml_hartung_knapp",
    "robust_sensitivity",
    "wt_temperature_secondary_results",
]
