"""Deterministic PHAxis biological-narrative decision authority.

The decision is made once, at publication figure-input assembly, from the
sealed five-sentinel by three-effect clean/full evidence family.  Figures and
manuscript values consume the same hash-sealed object; distal profiles may
localise an interpretation but can never select or veto a branch.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Iterable, Mapping, Sequence

from .io import sha256_json


SCHEMA_VERSION = "PHAxis-narrative-decision-1.0"
STATUS = "completed_deterministic_branch_decision"

ENDPOINT_ORDER = (
    "local_hair_count_1_4mm",
    "local_median_hair_length_um_1_4mm",
    "first_hair_ge40um_distance_from_distal_point_um",
    "median_root_width_um",
    "visible_root_axis_length_um",
)
EFFECT_ORDER = ("OE_vs_EV", "30C_vs_22C", "interaction")
COHORT_ORDER = ("primary_clean261", "sensitivity_full283")

ENDPOINT_CONTRACT = {
    ENDPOINT_ORDER[0]: {"sentinel": "H08", "badge": "N", "layer": "visible population", "group": "hair_spatial"},
    ENDPOINT_ORDER[1]: {"sentinel": "H11", "badge": "L", "layer": "supported morphology", "group": "hair_spatial"},
    ENDPOINT_ORDER[2]: {"sentinel": "H07", "badge": "F", "layer": "deployment boundary", "group": "hair_spatial"},
    ENDPOINT_ORDER[3]: {"sentinel": "R07", "badge": "W", "layer": "carrying-root calibre", "group": "root"},
    ENDPOINT_ORDER[4]: {"sentinel": "R01", "badge": "A", "layer": "visible organ extent", "group": "root"},
}


class NarrativeDecisionError(ValueError):
    """The fixed evidence family cannot support a publication decision."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NarrativeDecisionError(message)


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise NarrativeDecisionError(f"{label} is not numeric") from error
    _require(math.isfinite(result), f"{label} is not finite")
    return result


def _null_for_scale(effect_scale: str) -> float:
    normalized = str(effect_scale).strip().casefold()
    if normalized in {"ratio", "rate_ratio", "ratio_of_ratios", "multiplicative"}:
        return 1.0
    if normalized in {"difference", "additive", "identity"}:
        return 0.0
    raise NarrativeDecisionError(f"unsupported effect scale: {effect_scale!r}")


def _direction(value: float, null: float) -> str:
    if value > null:
        return "higher"
    if value < null:
        return "lower"
    return "null"


def _normalise_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw in rows:
        endpoint = str(raw.get("endpoint_key", raw.get("endpoint", "")))
        effect = str(raw.get("effect_key", raw.get("effect", "")))
        cohort = str(raw.get("cohort", ""))
        key = (endpoint, effect, cohort)
        _require(endpoint in ENDPOINT_ORDER, f"unexpected narrative endpoint: {endpoint!r}")
        _require(effect in EFFECT_ORDER, f"unexpected narrative effect: {effect!r}")
        _require(cohort in COHORT_ORDER, f"unexpected narrative cohort: {cohort!r}")
        _require(key not in observed, f"duplicate narrative cell: {key}")
        estimate = _finite(raw.get("estimate"), f"{key}.estimate")
        low = _finite(raw.get("ci_low", raw.get("ci95_low")), f"{key}.ci_low")
        high = _finite(raw.get("ci_high", raw.get("ci95_high")), f"{key}.ci_high")
        _require(low <= high, f"{key}: reversed confidence interval")
        n = raw.get("endpoint_n", raw.get("n"))
        try:
            endpoint_n = int(n)
        except (TypeError, ValueError) as error:
            raise NarrativeDecisionError(f"{key}: endpoint n is invalid") from error
        _require(endpoint_n > 0, f"{key}: endpoint n must be positive")
        scale = str(raw.get("effect_scale", "ratio"))
        null = _null_for_scale(scale)
        observed[key] = {
            "estimate": estimate,
            "ci_low": low,
            "ci_high": high,
            "endpoint_n": endpoint_n,
            "effect_scale": scale,
            "null": null,
        }
    expected = {
        (endpoint, effect, cohort)
        for endpoint in ENDPOINT_ORDER
        for effect in EFFECT_ORDER
        for cohort in COHORT_ORDER
    }
    _require(set(observed) == expected, "narrative evidence is not the exact 5 x 3 x 2 family")
    return observed


def build_narrative_decision(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Return the unique A/B/C decision for a sealed clean/full evidence family."""

    sources = {str(key): str(value) for key, value in source_sha256.items()}
    _require(bool(sources), "narrative decision source hashes are missing")
    _require(
        all(len(value) == 64 and all(character in "0123456789abcdef" for character in value) for value in sources.values()),
        "narrative decision source SHA-256 is invalid",
    )
    observed = _normalise_rows(rows)
    cells: list[dict[str, Any]] = []
    support_mask: list[bool] = []
    for endpoint in ENDPOINT_ORDER:
        contract = ENDPOINT_CONTRACT[endpoint]
        for effect in EFFECT_ORDER:
            clean = observed[(endpoint, effect, COHORT_ORDER[0])]
            full = observed[(endpoint, effect, COHORT_ORDER[1])]
            _require(clean["effect_scale"] == full["effect_scale"], f"{endpoint}/{effect}: clean/full scale differs")
            null = clean["null"]
            clean_direction = _direction(clean["estimate"], null)
            full_direction = _direction(full["estimate"], null)
            interval_excludes_null = clean["ci_high"] < null or clean["ci_low"] > null
            direction_retained = clean_direction != "null" and full_direction == clean_direction
            headline = bool(interval_excludes_null and direction_retained)
            support_mask.append(headline)
            cells.append(
                {
                    "endpoint_key": endpoint,
                    "effect_key": effect,
                    **contract,
                    "clean": {key: clean[key] for key in ("estimate", "ci_low", "ci_high", "endpoint_n", "effect_scale")},
                    "full283": {key: full[key] for key in ("estimate", "ci_low", "ci_high", "endpoint_n", "effect_scale")},
                    "clean_direction": clean_direction,
                    "full283_point_direction": full_direction,
                    "clean_interval_excludes_null": interval_excludes_null,
                    "full283_point_retains_clean_direction": direction_retained,
                    "headline_supported": headline,
                }
            )

    b_effects = []
    for effect in EFFECT_ORDER:
        supported = [cell for cell in cells if cell["effect_key"] == effect and cell["headline_supported"]]
        if any(cell["group"] == "hair_spatial" for cell in supported) and any(cell["group"] == "root" for cell in supported):
            b_effects.append(effect)
    if b_effects:
        branch = "B"
        rationale = "same_effect_supported_across_hair_or_spatial_and_carrying_root_layers"
    elif any(support_mask):
        branch = "A"
        rationale = "one_or_more_layer_specific_headlines_without_cross_layer_same_effect_alignment"
    else:
        branch = "C"
        rationale = "no_clean_full_direction_concordant_headline_in_fixed_15_effect_family"

    supported_layers = [
        f"{ENDPOINT_CONTRACT[endpoint]['sentinel']}/{ENDPOINT_CONTRACT[endpoint]['badge']}"
        for endpoint in ENDPOINT_ORDER
        if any(cell["endpoint_key"] == endpoint and cell["headline_supported"] for cell in cells)
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "branch_id": branch,
        "decision_rule": {
            "B": "same formal effect has >=1 supported H08/H11/H07 cell and >=1 supported R07/R01 cell",
            "A": "otherwise at least one of the fixed 15 cells is supported",
            "C": "none of the fixed 15 cells is supported",
            "headline": "clean interval excludes null and Full283 point estimate retains that endpoint's clean direction",
            "profiles_select_or_veto_branch": False,
        },
        "rationale_code": rationale,
        "endpoint_order": list(ENDPOINT_ORDER),
        "effect_order": list(EFFECT_ORDER),
        "support_mask": support_mask,
        "support_mask_bits": "".join("1" if value else "0" for value in support_mask),
        "supported_layers": supported_layers,
        "branch_b_effects": b_effects,
        "cells": cells,
        "source_sha256": sources,
        "blind_images_used": 0,
        "independent_accuracy_claim_allowed": False,
    }
    payload["narrative_decision_identity_sha256"] = sha256_json(payload)
    return payload


def validate_narrative_decision(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate identity and deterministically recompute the declared decision."""

    _require(payload.get("schema_version") == SCHEMA_VERSION, "narrative decision schema changed")
    _require(payload.get("status") == STATUS, "narrative decision status changed")
    _require(payload.get("blind_images_used") == 0, "blind images entered narrative decision")
    unsigned = deepcopy(dict(payload))
    identity = unsigned.pop("narrative_decision_identity_sha256", None)
    _require(isinstance(identity, str) and sha256_json(unsigned) == identity, "narrative decision identity mismatch")
    rows: list[dict[str, Any]] = []
    cells = payload.get("cells")
    _require(isinstance(cells, Sequence) and len(cells) == 15, "narrative decision cells are incomplete")
    for cell in cells:
        _require(isinstance(cell, Mapping), "narrative decision cell is invalid")
        for cohort, field in zip(COHORT_ORDER, ("clean", "full283"), strict=True):
            values = cell.get(field)
            _require(isinstance(values, Mapping), f"narrative decision {field} evidence missing")
            rows.append({"endpoint_key": cell.get("endpoint_key"), "effect_key": cell.get("effect_key"), "cohort": cohort, **values})
    expected = build_narrative_decision(rows, source_sha256=payload.get("source_sha256", {}))
    _require(expected == dict(payload), "narrative decision differs from fixed decision rule")
    return deepcopy(expected)


__all__ = [
    "COHORT_ORDER",
    "EFFECT_ORDER",
    "ENDPOINT_CONTRACT",
    "ENDPOINT_ORDER",
    "NarrativeDecisionError",
    "SCHEMA_VERSION",
    "STATUS",
    "build_narrative_decision",
    "validate_narrative_decision",
]
