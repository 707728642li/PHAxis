"""Sealed, trait-wise assurance for the 19 PHAxis primary-root descriptors.

This module deliberately separates annotated measurement agreement from
portable-provider equivalence.  The former compares each derived prediction
with a reference recomputed from the canonical vector-derived root mask and
the annotated distal point.  The latter only demonstrates that two providers
emit identical files and is never accepted here as accuracy evidence.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .io import sha256_json
from .traits import ROOT_TRAIT_FIELDS


ROOT_TRAIT_ASSURANCE_SCHEMA = "PHAxis-root-derived-trait-assurance-1.0"
ROOT_TRAIT_ACCURACY_EVIDENCE_ROLE = "annotated_qc_development_non_independent"
ROOT_TRAIT_REFERENCE_DEFINITION = (
    "trait recomputed from the canonical vector-derived primary-root mask and "
    "the annotated distal/root-cap point with the deterministic PHAxis 1.0.0 "
    "measurement geometry"
)
ROOT_TRAIT_PREDICTION_DEFINITION = (
    "matching detailed_root_statistics field recomputed from the sealed PHAxis "
    "Hybrid-Max root mask, distal point, physical scale, and ordered axis"
)

# These tokens are optional until the manuscript elects to cite them.  The
# compiler derives every token that appears in the master and ignores no
# requested token; keeping the registry here makes future manuscript edits
# stable and typo-resistant.
ROOT_TRAIT_ASSURANCE_TOKENS = (
    "FINAL_ROOT_TRAIT_VALIDATED_N",
    "FINAL_ROOT_TRAIT_VALIDATION_IMAGE_N",
    "FINAL_ROOT_TRAIT_ELIGIBLE_N_RANGE",
    "FINAL_ROOT_TRAIT_OBSERVABILITY_RANGE_PERCENT",
    "FINAL_ROOT_TRAIT_CCC_ESTIMABLE_N",
    "FINAL_ROOT_TRAIT_CCC_MEDIAN",
    "FINAL_ROOT_TRAIT_CCC_RANGE",
    "FINAL_ROOT_TRAIT_AGREEMENT_SUMMARY",
    "FINAL_ROOT_TRAIT_FAMILY_SUMMARY",
)

ROOT_TRAIT_FAMILY_BY_FIELD = {
    "visible_root_axis_length_um": "axis_extent",
    "root_axis_chord_um": "axis_extent",
    "root_centerline_chord_tortuosity": "axis_shape",
    "root_straightness": "axis_shape",
    "root_projected_area_um2": "projected_area",
    "root_projected_area_um2_per_root_mm": "projected_area",
    "median_root_width_um": "global_width_distribution",
    "root_width_p10_um": "global_width_distribution",
    "root_width_q25_um": "global_width_distribution",
    "root_width_q75_um": "global_width_distribution",
    "root_width_p90_um": "global_width_distribution",
    "root_width_cv": "global_width_distribution",
    "root_width_tip_third_median_um": "axial_width_pattern",
    "root_width_middle_third_median_um": "axial_width_pattern",
    "root_width_shootward_third_median_um": "axial_width_pattern",
    "root_width_shootward_to_tip_ratio": "axial_width_pattern",
    "root_width_axial_slope_um_per_mm": "axial_width_pattern",
    "root_centerline_curvature_median_rad_per_mm": "centerline_curvature",
    "root_centerline_curvature_p95_rad_per_mm": "centerline_curvature",
}

ROOT_TRAIT_FAMILY_ORDER = (
    "axis_extent",
    "axis_shape",
    "projected_area",
    "global_width_distribution",
    "axial_width_pattern",
    "centerline_curvature",
)


class RootTraitAssuranceError(RuntimeError):
    """The 19-trait truth, denominator, statistic, or identity contract failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RootTraitAssuranceError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _boolean(value: Any, role: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().casefold()
    _require(normalized in {"true", "false", "1", "0", "yes", "no"}, f"{role}: invalid boolean")
    return normalized in {"true", "1", "yes"}


def _ccc(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    if len(observed) < 2:
        return None
    covariance = float(np.cov(observed, predicted, ddof=1)[0, 1])
    denominator = float(
        np.var(observed, ddof=1)
        + np.var(predicted, ddof=1)
        + (np.mean(observed) - np.mean(predicted)) ** 2
    )
    if not math.isfinite(denominator) or denominator <= 0:
        return None
    value = 2.0 * covariance / denominator
    return float(value) if math.isfinite(value) else None


def _interval(values: np.ndarray) -> tuple[float | None, float | None, int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, None, 0
    low, high = np.quantile(finite, (0.025, 0.975))
    return float(low), float(high), int(len(finite))


def _first_difference(observed: Any, expected: Any, path: str = "$") -> str | None:
    if type(observed) is not type(expected):
        return f"{path}: type {type(observed).__name__} != {type(expected).__name__}"
    if isinstance(observed, Mapping):
        observed_keys, expected_keys = set(observed), set(expected)
        if observed_keys != expected_keys:
            return f"{path}: keys missing={sorted(expected_keys-observed_keys)} extra={sorted(observed_keys-expected_keys)}"
        for key in sorted(observed_keys):
            difference = _first_difference(observed[key], expected[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(observed, list):
        if len(observed) != len(expected):
            return f"{path}: length {len(observed)} != {len(expected)}"
        for index, (left, right) in enumerate(zip(observed, expected, strict=True)):
            difference = _first_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if observed != expected:
        return f"{path}: {observed!r} != {expected!r}"
    return None


def _trait_contract_rows(trait_contract: Mapping[str, Any]) -> list[dict[str, str]]:
    _require(
        trait_contract.get("schema_version") == "PHAxis-trait-contract-1.0.0",
        "root-trait assurance requires the canonical PHAxis 1.0.0 trait contract",
    )
    counts = trait_contract.get("counts")
    _require(
        isinstance(counts, Mapping)
        and counts.get("primary_root_fields") == 19
        and counts.get("root_cap_region_fields") == 0,
        "trait contract is not the canonical 19-root/no-root-cap-region contract",
    )
    records = trait_contract.get("primary_root_traits")
    _require(isinstance(records, list) and len(records) == 19, "trait contract does not enumerate 19 root traits")
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(records, start=1):
        _require(isinstance(raw, Mapping), f"root trait contract row {index} is malformed")
        field = str(raw.get("field", ""))
        expected_id = f"R{index:02d}"
        _require(raw.get("id") == expected_id, f"root trait ID/order drift at {expected_id}")
        _require(field == ROOT_TRAIT_FIELDS[index - 1], f"root trait field/order drift at {expected_id}")
        unit = str(raw.get("unit", ""))
        _require(
            unit
            and field in ROOT_TRAIT_FAMILY_BY_FIELD
            and raw.get("source") == f"detailed_root_statistics.{field}"
            and raw.get("type") == "number_or_null",
            f"{expected_id}: unit/family/source/type contract drift",
        )
        normalized.append(
            {
                "trait_id": expected_id,
                "trait_key": field,
                "trait_family": ROOT_TRAIT_FAMILY_BY_FIELD[field],
                "unit": unit,
            }
        )
    _require(
        {row["trait_key"] for row in normalized} == set(ROOT_TRAIT_FIELDS),
        "root trait contract has missing or duplicate fields",
    )
    return normalized


def _normalized_pairs(
    pairs: Sequence[Mapping[str, Any]],
    *,
    contract_rows: Sequence[Mapping[str, str]],
    source_units: Sequence[str],
) -> list[dict[str, Any]]:
    expected_units = tuple(sorted(str(item) for item in source_units))
    _require(len(expected_units) >= 2 and len(expected_units) == len(set(expected_units)), "root-trait source-unit denominator is invalid")
    by_field = {row["trait_key"]: row for row in contract_rows}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    image_sha_by_unit: dict[str, str] = {}
    for raw in pairs:
        _require(isinstance(raw, Mapping), "root-trait pair row is malformed")
        _require(str(raw.get("pair_type")) == "root_trait", "non-root pair entered root-trait assurance")
        field = str(raw.get("trait_key", ""))
        source_unit = str(raw.get("source_unit", ""))
        _require(field in by_field and source_unit in expected_units, "unknown root trait or source unit")
        key = (field, source_unit)
        _require(key not in seen, f"duplicate root-trait pair: {source_unit}/{field}")
        seen.add(key)
        contract = by_field[field]
        _require(str(raw.get("trait_id")) == contract["trait_id"], f"{source_unit}/{field}: trait ID drift")
        _require(str(raw.get("trait_family")) == contract["trait_family"], f"{source_unit}/{field}: trait family drift")
        _require(str(raw.get("unit")) == contract["unit"], f"{source_unit}/{field}: unit drift")
        _require(str(raw.get("pair_id")) == f"{source_unit}:{field}", f"{source_unit}/{field}: pair identity drift")
        image_sha = str(raw.get("source_image_sha256", ""))
        _require(_is_sha256(image_sha), f"{source_unit}/{field}: image SHA-256 missing")
        prior_sha = image_sha_by_unit.setdefault(source_unit, image_sha)
        _require(prior_sha == image_sha, f"{source_unit}: source image identity differs across traits")
        reference_observable = _boolean(raw.get("reference_observable"), f"{source_unit}/{field} reference_observable")
        prediction_observable = _boolean(raw.get("prediction_observable"), f"{source_unit}/{field} prediction_observable")
        agreement_eligible = _boolean(raw.get("agreement_eligible"), f"{source_unit}/{field} agreement_eligible")
        _require(
            agreement_eligible == (reference_observable and prediction_observable),
            f"{source_unit}/{field}: eligibility differs from observability",
        )
        observed = _finite_or_none(raw.get("observed"))
        predicted = _finite_or_none(raw.get("predicted"))
        reason_value = raw.get("ineligibility_reason", "")
        reason = "" if reason_value is None or str(reason_value).casefold() == "nan" else str(reason_value).strip()
        if agreement_eligible:
            _require(observed is not None and predicted is not None and not reason, f"{source_unit}/{field}: eligible pair has missing value/reason")
        else:
            _require(reason and (observed is None or predicted is None), f"{source_unit}/{field}: ineligible pair lacks explicit missing support")
        _require(
            str(raw.get("reference_definition")) == ROOT_TRAIT_REFERENCE_DEFINITION,
            f"{source_unit}/{field}: truth/reference definition drift",
        )
        _require(
            str(raw.get("prediction_definition")) == ROOT_TRAIT_PREDICTION_DEFINITION,
            f"{source_unit}/{field}: prediction definition drift",
        )
        normalized.append(
            {
                "pair_type": "root_trait",
                "source_unit": source_unit,
                "pair_id": f"{source_unit}:{field}",
                "trait_id": contract["trait_id"],
                "trait_key": field,
                "trait_family": contract["trait_family"],
                "unit": contract["unit"],
                "observed": observed,
                "predicted": predicted,
                "reference_observable": reference_observable,
                "prediction_observable": prediction_observable,
                "agreement_eligible": agreement_eligible,
                "ineligibility_reason": reason,
                "reference_definition": ROOT_TRAIT_REFERENCE_DEFINITION,
                "prediction_definition": ROOT_TRAIT_PREDICTION_DEFINITION,
                "source_image_sha256": image_sha,
            }
        )
    expected_pairs = len(contract_rows) * len(expected_units)
    _require(len(normalized) == expected_pairs, f"root-trait pair denominator drift: expected {expected_pairs}, found {len(normalized)}")
    for field in ROOT_TRAIT_FIELDS:
        observed_units = tuple(sorted(row["source_unit"] for row in normalized if row["trait_key"] == field))
        _require(observed_units == expected_units, f"{field}: missing or drifted source-unit denominator")
    return sorted(normalized, key=lambda row: (row["trait_id"], row["source_unit"]))


def build_root_trait_assurance(
    *,
    pairs: Sequence[Mapping[str, Any]],
    trait_contract: Mapping[str, Any],
    source_units: Sequence[str],
    trait_contract_file_sha256: str,
    reference_authority_sha256: str,
    prediction_authority_identity_sha256: str,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 20_260_828,
) -> dict[str, Any]:
    """Recompute and seal all 19 trait-wise agreement rows from sufficient statistics."""

    _require(_is_sha256(trait_contract_file_sha256), "trait-contract file SHA-256 missing")
    _require(_is_sha256(reference_authority_sha256), "annotated reference authority SHA-256 missing")
    _require(_is_sha256(prediction_authority_identity_sha256), "prediction authority identity missing")
    _require(isinstance(bootstrap_repetitions, int) and bootstrap_repetitions >= 100, "root-trait bootstrap repetitions are insufficient")
    _require(isinstance(bootstrap_seed, int) and bootstrap_seed >= 0, "root-trait bootstrap seed is invalid")
    contract_rows = _trait_contract_rows(trait_contract)
    normalized_pairs = _normalized_pairs(
        pairs,
        contract_rows=contract_rows,
        source_units=source_units,
    )
    source_unit_rows = sorted(
        {
            (row["source_unit"], row["source_image_sha256"])
            for row in normalized_pairs
        }
    )
    _require(len(source_unit_rows) == len(source_units), "root-trait image/source-unit identity denominator drift")
    source_unit_identity = sha256_json(
        [
            {"source_unit": source_unit, "source_image_sha256": image_sha}
            for source_unit, image_sha in source_unit_rows
        ]
    )

    trait_rows: list[dict[str, Any]] = []
    generator = np.random.default_rng(bootstrap_seed)
    for contract in contract_rows:
        selected = [row for row in normalized_pairs if row["trait_key"] == contract["trait_key"]]
        eligible = [row for row in selected if row["agreement_eligible"]]
        _require(eligible, f"{contract['trait_id']}: no observable reference/prediction pairs")
        observed = np.asarray([row["observed"] for row in eligible], dtype=np.float64)
        predicted = np.asarray([row["predicted"] for row in eligible], dtype=np.float64)
        error = predicted - observed
        mae = float(np.mean(np.abs(error)))
        bias = float(np.mean(error))
        ccc = _ccc(observed, predicted)
        indices = generator.integers(0, len(eligible), size=(bootstrap_repetitions, len(eligible)))
        boot_error = error[indices]
        mae_low, mae_high, _ = _interval(np.mean(np.abs(boot_error), axis=1))
        bias_low, bias_high, _ = _interval(np.mean(boot_error, axis=1))
        boot_ccc = np.full(bootstrap_repetitions, np.nan, dtype=np.float64)
        if len(eligible) >= 2:
            boot_observed = observed[indices]
            boot_predicted = predicted[indices]
            mean_observed = np.mean(boot_observed, axis=1)
            mean_predicted = np.mean(boot_predicted, axis=1)
            centred_observed = boot_observed - mean_observed[:, None]
            centred_predicted = boot_predicted - mean_predicted[:, None]
            divisor = len(eligible) - 1
            covariance = np.sum(
                centred_observed * centred_predicted, axis=1
            ) / divisor
            denominator = (
                np.sum(centred_observed**2, axis=1) / divisor
                + np.sum(centred_predicted**2, axis=1) / divisor
                + (mean_observed - mean_predicted) ** 2
            )
            estimable = np.isfinite(denominator) & (denominator > 0)
            boot_ccc[estimable] = (
                2.0 * covariance[estimable] / denominator[estimable]
            )
        ccc_low, ccc_high, finite_ccc_bootstrap_n = _interval(boot_ccc)
        if ccc is None:
            ccc_status = "not_estimable_zero_total_variance"
            agreement_statistic = "mae_native_unit"
            agreement_value = mae
            agreement_higher_is_better = False
            ccc_low = ccc_high = None
        else:
            ccc_status = (
                "estimable_with_percentile_interval"
                if finite_ccc_bootstrap_n >= max(100, int(0.5 * bootstrap_repetitions))
                else "estimable_point_interval_not_stable"
            )
            if ccc_status != "estimable_with_percentile_interval":
                ccc_low = ccc_high = None
            agreement_statistic = "ccc"
            agreement_value = ccc
            agreement_higher_is_better = True
        pair_payload = [
            {
                "source_unit": row["source_unit"],
                "pair_id": row["pair_id"],
                "source_image_sha256": row["source_image_sha256"],
                "observed": row["observed"],
                "predicted": row["predicted"],
                "reference_observable": row["reference_observable"],
                "prediction_observable": row["prediction_observable"],
                "agreement_eligible": row["agreement_eligible"],
                "ineligibility_reason": row["ineligibility_reason"],
            }
            for row in selected
        ]
        row: dict[str, Any] = {
            **contract,
            "truth_reference": ROOT_TRAIT_REFERENCE_DEFINITION,
            "prediction_reference": ROOT_TRAIT_PREDICTION_DEFINITION,
            "evidence_role": ROOT_TRAIT_ACCURACY_EVIDENCE_ROLE,
            "independent_accuracy_claim_allowed": False,
            "provider_equivalence_used_as_accuracy": False,
            "total_source_units": len(selected),
            "eligible_source_units": len(eligible),
            "reference_observable_n": sum(row["reference_observable"] for row in selected),
            "prediction_observable_n": sum(row["prediction_observable"] for row in selected),
            "observability_fraction": len(eligible) / len(selected),
            "support_status": "fully_observable" if len(eligible) == len(selected) else "partially_observable",
            "mae": mae,
            "mae_ci_low": mae_low,
            "mae_ci_high": mae_high,
            "bias": bias,
            "bias_ci_low": bias_low,
            "bias_ci_high": bias_high,
            "ccc": ccc,
            "ccc_ci_low": ccc_low,
            "ccc_ci_high": ccc_high,
            "ccc_status": ccc_status,
            "agreement_statistic": agreement_statistic,
            "agreement_value": agreement_value,
            "agreement_higher_is_better": agreement_higher_is_better,
            "bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_seed": bootstrap_seed,
            "finite_ccc_bootstrap_n": finite_ccc_bootstrap_n,
            "source_unit_set_identity_sha256": source_unit_identity,
            "pair_set_identity_sha256": sha256_json(pair_payload),
            "reference_authority_sha256": reference_authority_sha256,
            "prediction_authority_identity_sha256": prediction_authority_identity_sha256,
            "trait_contract_file_sha256": trait_contract_file_sha256,
        }
        row["row_identity_sha256"] = sha256_json(row)
        trait_rows.append(row)

    family_rows: list[dict[str, Any]] = []
    for family in ROOT_TRAIT_FAMILY_ORDER:
        selected = [row for row in trait_rows if row["trait_family"] == family]
        _require(selected, f"empty root-trait family: {family}")
        estimable = [float(row["ccc"]) for row in selected if row["ccc"] is not None]
        family_row: dict[str, Any] = {
            "trait_family": family,
            "trait_ids": [row["trait_id"] for row in selected],
            "trait_count": len(selected),
            "eligible_source_units_min": min(int(row["eligible_source_units"]) for row in selected),
            "eligible_source_units_max": max(int(row["eligible_source_units"]) for row in selected),
            "observability_fraction_min": min(float(row["observability_fraction"]) for row in selected),
            "observability_fraction_max": max(float(row["observability_fraction"]) for row in selected),
            "ccc_estimable_traits": len(estimable),
            "median_ccc": float(np.median(estimable)) if estimable else None,
            "evidence_role": ROOT_TRAIT_ACCURACY_EVIDENCE_ROLE,
            "provider_equivalence_used_as_accuracy": False,
        }
        family_row["family_identity_sha256"] = sha256_json(family_row)
        family_rows.append(family_row)

    payload: dict[str, Any] = {
        "schema_version": ROOT_TRAIT_ASSURANCE_SCHEMA,
        "scope": "QC-development measurement assurance; non-independent",
        "evidence_role": ROOT_TRAIT_ACCURACY_EVIDENCE_ROLE,
        "truth_reference": ROOT_TRAIT_REFERENCE_DEFINITION,
        "prediction_reference": ROOT_TRAIT_PREDICTION_DEFINITION,
        "independent_accuracy_claim_allowed": False,
        "provider_equivalence_used_as_accuracy": False,
        "trait_count": len(trait_rows),
        "family_count": len(family_rows),
        "source_unit_total": len(source_units),
        "source_unit_set_identity_sha256": source_unit_identity,
        "trait_contract_file_sha256": trait_contract_file_sha256,
        "trait_contract_identity_sha256": sha256_json(trait_contract),
        "reference_authority_sha256": reference_authority_sha256,
        "prediction_authority_identity_sha256": prediction_authority_identity_sha256,
        "bootstrap": {
            "method": "image/source-unit nonparametric percentile bootstrap",
            "repetitions": bootstrap_repetitions,
            "seed": bootstrap_seed,
        },
        "trait_rows": trait_rows,
        "family_rows": family_rows,
        "trait_row_set_identity_sha256": sha256_json(trait_rows),
        "family_row_set_identity_sha256": sha256_json(family_rows),
    }
    payload["root_trait_assurance_identity_sha256"] = sha256_json(payload)
    return payload


def validate_root_trait_assurance(
    payload: Mapping[str, Any],
    *,
    pairs: Sequence[Mapping[str, Any]],
    trait_contract: Mapping[str, Any],
    source_units: Sequence[str],
    trait_contract_file_sha256: str,
    reference_authority_sha256: str,
    prediction_authority_identity_sha256: str,
) -> dict[str, Any]:
    """Recompute every statistic and identity and return a canonical copy."""

    _require(payload.get("schema_version") == ROOT_TRAIT_ASSURANCE_SCHEMA, "root-trait assurance schema drift")
    _require(payload.get("evidence_role") == ROOT_TRAIT_ACCURACY_EVIDENCE_ROLE, "provider equivalence or another role cannot masquerade as trait accuracy")
    _require(payload.get("provider_equivalence_used_as_accuracy") is False, "provider equivalence cannot satisfy root-trait accuracy")
    _require(payload.get("independent_accuracy_claim_allowed") is False, "QC-development root-trait assurance was mislabelled independent")
    bootstrap = payload.get("bootstrap")
    _require(isinstance(bootstrap, Mapping), "root-trait bootstrap contract missing")
    expected = build_root_trait_assurance(
        pairs=pairs,
        trait_contract=trait_contract,
        source_units=source_units,
        trait_contract_file_sha256=trait_contract_file_sha256,
        reference_authority_sha256=reference_authority_sha256,
        prediction_authority_identity_sha256=prediction_authority_identity_sha256,
        bootstrap_repetitions=int(bootstrap.get("repetitions", -1)),
        bootstrap_seed=int(bootstrap.get("seed", -1)),
    )
    difference = _first_difference(dict(payload), expected)
    _require(
        difference is None,
        f"root-trait assurance values, denominator, or identity drift ({difference})",
    )
    return deepcopy(expected)


__all__ = [
    "ROOT_TRAIT_ACCURACY_EVIDENCE_ROLE",
    "ROOT_TRAIT_ASSURANCE_SCHEMA",
    "ROOT_TRAIT_ASSURANCE_TOKENS",
    "ROOT_TRAIT_FAMILY_BY_FIELD",
    "ROOT_TRAIT_FAMILY_ORDER",
    "ROOT_TRAIT_PREDICTION_DEFINITION",
    "ROOT_TRAIT_REFERENCE_DEFINITION",
    "RootTraitAssuranceError",
    "build_root_trait_assurance",
    "validate_root_trait_assurance",
]
