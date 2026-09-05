from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN = PROJECT_ROOT / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
SUPPLEMENT = (
    PROJECT_ROOT / "docs/phaxis/PHAXIS_SUPPLEMENTARY_MASTER_DRAFT_20260830.md"
)
CONTRACT = (
    PROJECT_ROOT / "docs/phaxis/PHAXIS_MEASUREMENT_ASSURANCE_CONTRACT_CN_20260829.md"
)
HAIR_PROTOCOL = (
    PROJECT_ROOT / "docs/phaxis/PHAXIS_ROOT_HAIR_BIOLOGICAL_METRIC_PROTOCOL_20260828.md"
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_formal_attachment_reuses_biological_presence_matches() -> None:
    main = _text(MAIN)
    supplement = _text(SUPPLEMENT)
    contract = _text(CONTRACT)
    protocol = _text(HAIR_PROTOCOL)

    assert "Attachment coordinate supporting H07 (F)" in main
    assert "same formal one-to-one biological-presence matches" in main
    assert "no second base-only assignment is allowed" in supplement
    assert "不允许另做更有利的 base-only" in contract
    assert "no base-only rematching" in main
    assert "selected-development evidence" in main
    assert "QC-development44 also selected the operating point" in main
    assert "rather than external-test performance" in main
    assert "not formal attachment accuracy" in supplement
    assert "禁止为附着结果另做一个更有利的 base-only rematch" in protocol
    assert "不得称为正式附着准确率" in protocol


def test_root_continuity_is_single_component_and_evaluator_does_not_bridge() -> None:
    main = _text(MAIN)
    supplement = _text(SUPPLEMENT)
    contract = _text(CONTRACT)

    assert "sealed final fused root foreground actually delivered to trait extraction" in main
    assert "skeletonized every 8-connected foreground component without adding a bridge" in main
    assert "break-free status requires one and the same predicted component" in supplement
    assert "评价器只对最终前景" in contract
    assert "不在评分阶段插值、桥接或补线" in contract
    assert "不得算作连续" in contract


def test_scale_applicability_and_root_cap_scope_are_explicit() -> None:
    main = _text(MAIN)
    supplement = _text(SUPPLEMENT)
    contract = _text(CONTRACT)

    assert "38 QC-development images with a visible annotated scale bar" in main
    assert "six images with trusted metadata calibration" in main
    assert "contains no absent-scale or untrusted-metadata case" in main
    assert "empirical absence specificity was not estimable" in main
    assert "38 visible annotated scale bars and six trusted-metadata calibrations" in supplement
    assert "not_estimable_no_absent_or_untrusted_scale_cases" in contract
    assert "A root-cap region is neither segmented nor quantified" in main
    assert "no root-cap region or area is produced" in supplement


def test_formal_assurance_is_closed_in_results_and_table_two() -> None:
    main = _text(MAIN)
    supplement = _text(SUPPLEMENT)

    assert "Attachment coordinate supporting H07 (F)" in main
    assert "FINAL_HAIR_ATTACHMENT_PREDICTED_N" in main
    assert "FINAL_HAIR_ATTACHMENT_ANNOTATED_N" in main
    assert "no base-only rematching" in main
    assert "Continuous carrying-root coordinate supporting R01/R07 (A/W)" in main
    assert "no evaluator-side bridging" in main
    assert "Union coverage cannot convert jointly covering fragments" in main
    assert "Source-image and matched-curve intervals" not in main
    assert "unsupported length is null, not zero" in main
    assert "Distribution of hair-curve trajectory-continuity support" in supplement
    assert "performs no base-only rematching" in supplement
    assert "complete annotated same-component root-continuity family" in supplement
    assert "evaluator adds no bridge" in supplement


def test_scale_result_separates_applicability_and_absence_specificity() -> None:
    main = _text(MAIN)

    assert "38 visible annotated bars and six trusted-metadata calibrations" in main
    assert "no absent or untrusted-scale case" in main
    assert "FINAL_SCALE_APPLICABILITY_STATEMENT" in main
    assert "FINAL_SCALE_ABSENCE_SPECIFICITY_STATUS" in main
    assert "software fail-closed tests" in main
