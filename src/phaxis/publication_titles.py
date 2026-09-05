"""Single title authority for PHAxis main figures and tables."""

from __future__ import annotations

from typing import Any, Mapping

from .narrative_decision import validate_narrative_decision
from .io import sha256_json


FIGURE_STATIC_TITLES = {
    1: "PHAxis anchors visible-hair population, supported morphology and primary-root form to one physical axis",
    2: "PHAxis recovers visible root-hair populations at individual-hair resolution",
    3: "Continuity, calibration and conditional geometry make organ-anchored traits physically interpretable",
    4: "PHAxis exposes interpretable measurement support across challenging image contexts",
    6: "PHAxis carries raw images to a reusable, benchmarked root-hair phenotype atlas",
}
TABLE_STATIC_TITLES = {
    1: "The PHAxis ontology preserves five plant questions and every descriptor’s observation state",
    2: "Measurement assurance validates each plant-facing layer in its native biological unit",
}


def _supported_layer_text(decision: Mapping[str, Any]) -> str:
    layers = decision.get("supported_layers")
    if not isinstance(layers, list) or not layers:
        return "the supported measurement layer"
    return ", ".join(str(value) for value in layers)


def figure_title(number: int, decision: Mapping[str, Any] | None = None) -> str:
    if number in FIGURE_STATIC_TITLES:
        return FIGURE_STATIC_TITLES[number]
    if number != 5 or decision is None:
        raise ValueError(f"Figure {number} requires a narrative decision")
    selected = validate_narrative_decision(decision)
    branch = selected["branch_id"]
    if branch == "A":
        return "The D15 atlas distinguishes layer-specific phenotype associations along the root–hair interface"
    if branch == "B":
        return "The D15 atlas resolves effect-aligned, endpoint-specific associations across hair and carrying-root layers"
    return "The D15 atlas maps five complementary dimensions of the root–hair interface"


def table_title(number: int, decision: Mapping[str, Any] | None = None) -> str:
    if number in TABLE_STATIC_TITLES:
        return TABLE_STATIC_TITLES[number]
    if number != 3 or decision is None:
        raise ValueError(f"Table {number} requires a narrative decision")
    selected = validate_narrative_decision(decision)
    branch = selected["branch_id"]
    if branch == "A":
        return "Five sentinel traits distinguish layer-specific D15 phenotype associations"
    if branch == "B":
        return "Five sentinel traits resolve effect-aligned D15 associations across hair and carrying-root layers"
    return "Five sentinel traits resolve complementary dimensions of the D15 root–hair interface"


def title_contract(decision: Mapping[str, Any]) -> dict[str, Any]:
    selected = validate_narrative_decision(decision)
    contract: dict[str, Any] = {
        "narrative_decision_identity_sha256": selected["narrative_decision_identity_sha256"],
        "branch_id": selected["branch_id"],
        "figures": {str(number): figure_title(number, selected) for number in range(1, 7)},
        "tables": {str(number): table_title(number, selected) for number in range(1, 4)},
    }
    contract["title_contract_identity_sha256"] = sha256_json(contract)
    return contract


__all__ = ["FIGURE_STATIC_TITLES", "TABLE_STATIC_TITLES", "figure_title", "table_title", "title_contract"]
