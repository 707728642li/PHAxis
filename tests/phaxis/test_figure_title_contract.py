from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from phaxis.narrative_decision import EFFECT_ORDER, ENDPOINT_ORDER, build_narrative_decision
from phaxis.publication_titles import title_contract


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_MANUSCRIPT = (
    PROJECT_ROOT / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
)
SUPPLEMENTARY_MANUSCRIPT = (
    PROJECT_ROOT / "docs/phaxis/PHAXIS_SUPPLEMENTARY_MASTER_DRAFT_20260830.md"
)
FIGURE_BUILDER = PROJECT_ROOT / "scripts/phaxis/build_publication_figures.py"

SPEC = importlib.util.spec_from_file_location(
    "phaxis_figure_title_contract_builder", FIGURE_BUILDER
)
assert SPEC and SPEC.loader
figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figures)


def _titles(text: str, *, supplementary: bool) -> dict[str, str]:
    identifier = r"S[1-9]" if supplementary else r"[1-6]"
    pattern = re.compile(
        rf"^(?:###|##) (?:Supplementary )?Figure ({identifier})\.\s+(.+?)\s*$",
        re.MULTILINE,
    )
    rows = pattern.findall(text)
    result = {key: title for key, title in rows}
    assert len(result) == len(rows), "duplicate figure title identifier"
    return result


def _decision(supported: set[tuple[str, str]]) -> dict:
    rows = []
    for endpoint in ENDPOINT_ORDER:
        for effect in EFFECT_ORDER:
            for cohort in ("primary_clean261", "sensitivity_full283"):
                headline = (endpoint, effect) in supported
                rows.append(
                    {
                        "endpoint_key": endpoint,
                        "effect_key": effect,
                        "cohort": cohort,
                        "estimate": 1.25 if headline else 1.0,
                        "ci_low": 1.10 if headline else 0.90,
                        "ci_high": 1.40 if headline else 1.10,
                        "endpoint_n": 8,
                        "effect_scale": "ratio",
                    }
                )
    return build_narrative_decision(rows, source_sha256={"fixture": "a" * 64})


def _branch_b_decision() -> dict:
    return _decision(
        {
            (ENDPOINT_ORDER[0], EFFECT_ORDER[0]),
            (ENDPOINT_ORDER[3], EFFECT_ORDER[0]),
        }
    )


def _opposite_direction_branch_b_decision() -> dict:
    """Branch B does not require hair and carrying-root signs to match."""

    supported = {
        (ENDPOINT_ORDER[0], EFFECT_ORDER[0]): (1.25, 1.10, 1.40),
        (ENDPOINT_ORDER[3], EFFECT_ORDER[0]): (0.75, 0.60, 0.90),
    }
    rows = []
    for endpoint in ENDPOINT_ORDER:
        for effect in EFFECT_ORDER:
            estimate, ci_low, ci_high = supported.get(
                (endpoint, effect), (1.0, 0.90, 1.10)
            )
            for cohort in ("primary_clean261", "sensitivity_full283"):
                rows.append(
                    {
                        "endpoint_key": endpoint,
                        "effect_key": effect,
                        "cohort": cohort,
                        "estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "endpoint_n": 8,
                        "effect_scale": "ratio",
                    }
                )
    return build_narrative_decision(
        rows, source_sha256={"fixture": "b" * 64}
    )


def _resolve_main_title_slots(text: str, contract: dict) -> str:
    for kind, maximum in (("figures", 6), ("tables", 3)):
        for number in range(1, maximum + 1):
            text = text.replace(
                f"⟦RESULT SLOT → publication_title_contract.{kind}.{number}⟧",
                contract[kind][str(number)],
            )
    return text


def _table_titles(text: str) -> dict[str, str]:
    rows = re.findall(r"^### Table ([1-3])\.\s+(.+?)\s*$", text, flags=re.MULTILINE)
    result = {key: value for key, value in rows}
    assert len(result) == len(rows), "duplicate table title identifier"
    return result


def test_manuscript_and_generated_figure_titles_are_identical() -> None:
    """Prevent a caption contract from silently drifting from its rendered figure."""

    decision = _branch_b_decision()
    contract = title_contract(decision)
    generated = figures._legends_and_alt_text(
        provisional=True,
        runtime={},
        decision=decision,
    )
    resolved_main = _resolve_main_title_slots(
        MAIN_MANUSCRIPT.read_text(encoding="utf-8"),
        contract,
    )
    main = _titles(resolved_main, supplementary=False)
    supplement = _titles(
        SUPPLEMENTARY_MANUSCRIPT.read_text(encoding="utf-8"), supplementary=True
    )
    generated_main = _titles(generated, supplementary=False)
    generated_supplement = _titles(generated, supplementary=True)

    assert tuple(main) == tuple(str(index) for index in range(1, 7))
    assert tuple(supplement) == tuple(f"S{index}" for index in range(1, 10))
    assert tuple(generated_main) == tuple(str(index) for index in range(1, 7))
    assert tuple(generated_supplement) == tuple(
        f"S{index}" for index in range(1, 10)
    )
    assert main == generated_main
    assert supplement == generated_supplement
    assert _table_titles(resolved_main) == contract["tables"]


def test_all_three_branches_use_plant_facing_titles_without_internal_decision_terms() -> None:
    decisions = {
        "A": _decision({(ENDPOINT_ORDER[0], EFFECT_ORDER[0])}),
        "B": _branch_b_decision(),
        "C": _decision(set()),
    }
    expected = {
        "A": (
            "The D15 atlas distinguishes layer-specific phenotype associations along "
            "the root–hair interface",
            "Five sentinel traits distinguish layer-specific D15 phenotype associations",
        ),
        "B": (
            "The D15 atlas resolves effect-aligned, endpoint-specific associations "
            "across hair and carrying-root layers",
            "Five sentinel traits resolve effect-aligned D15 associations across hair "
            "and carrying-root layers",
        ),
        "C": (
            "The D15 atlas maps five complementary dimensions of the root–hair interface",
            "Five sentinel traits resolve complementary dimensions of the D15 root–hair interface",
        ),
    }
    forbidden = (
        "headline",
        "fixed decision",
        "sensitivity-unstable",
        "clean/full-concordant",
        "coordinated",
        "remodeling",
    )
    for branch, decision in decisions.items():
        assert decision["branch_id"] == branch
        contract = title_contract(decision)
        assert contract["figures"]["5"] == expected[branch][0]
        assert contract["tables"]["3"] == expected[branch][1]
        reader_text = " ".join(
            [*contract["figures"].values(), *contract["tables"].values()]
        ).casefold()
        assert not any(term in reader_text for term in forbidden)


def test_branch_b_titles_do_not_imply_same_direction_across_layers() -> None:
    decision = _opposite_direction_branch_b_decision()
    assert decision["branch_id"] == "B"
    supported = [
        cell for cell in decision["cells"] if cell["headline_supported"]
    ]
    assert {cell["clean_direction"] for cell in supported} == {"higher", "lower"}

    contract = title_contract(decision)
    text = f"{contract['figures']['5']} {contract['tables']['3']}".casefold()
    assert "effect-aligned" in text
    assert "endpoint-specific" in contract["figures"]["5"].casefold()
    assert "coordinated" not in text
    assert "remodeling" not in text
