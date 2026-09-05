from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER = (
    PROJECT_ROOT / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
)
STORY_OVERLAY = (
    PROJECT_ROOT / "docs/phaxis/PHAXIS_FIGURE_TABLE_STORY_OVERLAY_20260830.md"
)


def _master() -> str:
    return MASTER.read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


def test_results_headlines_do_not_overstate_or_duplicate_supplementary_detail() -> None:
    manuscript = _master()
    section_32 = _between(manuscript, "### 3.2 ", "### 3.3 ")
    assert "Against {{FINAL_QCDEV_ANNOTATED_HAIR_N}} annotated identities" in section_32
    assert "recovered `{{FINAL_QCDEV_ANNOTATED_HAIR_N}}` annotated" not in section_32
    for routed_token in (
        "FINAL_HAIR_PRECISION_20UM",
        "FINAL_HAIR_RECALL_20UM",
        "FINAL_HAIR_COUNT_BIAS",
        "FINAL_HAIR_COUNT_CCC",
        "HISTORICAL_OOF_VERY_DENSE_F1",
    ):
        assert routed_token not in section_32

    section_33 = _between(manuscript, "### 3.3 ", "### 3.4 ")
    assert "FINAL_ENDPOINT_COMPLETE_IDENTITY_PERCENT" in section_33
    assert "FINAL_MATCHED_LENGTH_MAE_UM" in section_33
    for headline_token in (
        "FINAL_ROOT_CONTINUITY_BREAK_FREE_RATE",
        "FINAL_ROOT_CONTINUITY_MAXIMUM_SINGLE_COMPONENT_COVERAGE_MEDIAN",
        "FINAL_DISTAL_MEDIAN_ERROR_UM",
        "FINAL_SCALE_DETECTION_COVERAGE",
        "FINAL_ROOT_TRAIT_CCC_MEDIAN",
    ):
        assert headline_token in section_33
    for routed_token in (
        "FINAL_MATCHED_LENGTH_BIAS_UM",
        "FINAL_MATCHED_LENGTH_CCC",
        "FINAL_MATCHED_ENDPOINT_ERROR_UM",
        "FINAL_MATCHED_TRAJECTORY_CONTINUITY",
    ):
        assert routed_token not in section_33

    section_34 = _between(manuscript, "### 3.4 ", "### 3.5 ")
    for headline_token in (
        "FINAL_D15_ABSTRACT_SYNTHESIS",
        "FINAL_D15_CLEAN_FORMAL_N",
        "FINAL_D15_FULL_FORMAL_N",
        "FINAL_MULTITRAIT_ATLAS_SUMMARY",
    ):
        assert headline_token in section_34
    for routed_token in ("FINAL_ROOT_DICE", "FINAL_ROOT_BOUNDARY_F1", "FINAL_ROOT_HD95_UM"):
        assert routed_token not in section_34


def test_main_figure_legends_match_the_current_four_panel_and_audit_contracts() -> None:
    manuscript = _master()
    figure2 = _between(manuscript, "### Figure 2.", "### Figure 3.")
    assert all(f"({panel})" in figure2 for panel in "abcd")
    assert "(e)" not in figure2 and "(f)" not in figure2
    assert "One-to-one 20-µm assignments" in figure2
    assert "selected-development rather than external-test performance" in figure2

    figure4 = _between(manuscript, "### Figure 4.", "### Figure 5.")
    for anchor in ("RHSCU-aa5b6e37df15821f", "RHSCU-bbf649822174e0a2"):
        assert anchor in figure4
    for semantic in (
        "axis-in-root coverage",
        "maximum single-component root support",
        "longest unsupported axis gap",
        "formal identity n",
        "endpoint-complete n/fraction",
        "[1,4)-mm window",
        "[0,5)-mm profile eligibility",
        "Review-only metric fields remain `NA`, never zero",
    ):
        assert semantic in figure4
    assert "axis-guided insets" in figure4
    assert "cohort-level performance is quantified in Figs. 2–3" in figure4
    assert "orange point on an amber vector marks only the Stage-B vector terminus" in figure4
    assert "only after pixels were fixed" in figure4


def test_figure_four_story_overlay_uses_the_same_metadata_and_endpoint_scope() -> None:
    story = STORY_OVERLAY.read_text(encoding="utf-8")
    assert "orange point on an amber vector is only the Stage-B vector terminus" in story
    assert "only the terminus of a green one-to-one matched curve" in story
    assert "Condition metadata entered neither prediction, overlay pixels" in story
    assert "only after pixels were fixed" in story


def test_table_two_has_exactly_six_plant_facing_layers() -> None:
    table = _between(_master(), "### Table 2.", "### Table 3.")
    rows = [
        line
        for line in table.splitlines()
        if line.startswith("| ")
        and not line.startswith("| Biological layer")
        and not re.match(r"^\|[-:| ]+\|$", line)
    ]
    assert len(rows) == 6
    expected = (
        "Visible-hair identity/count supporting H08 (N)",
        "Attachment coordinate supporting H07 (F)",
        "Endpoint-complete projected morphology supporting H11 (L)",
        "Continuous carrying-root coordinate supporting R01/R07 (A/W)",
        "Distal landmark and physical scale",
        "Derived primary-root context R01–R19",
    )
    assert tuple(line.split("|")[1].strip() for line in rows) == expected
    assert "Portable-provider exact equivalence" not in table


def test_figure_five_and_table_three_preserve_endpoint_specific_support_semantics() -> None:
    manuscript = _master()
    figure5 = _between(manuscript, "### Figure 5.", "### Figure 6.")
    for semantic in (
        "fixed N→L→F→W→A order",
        "H11 displays non-null/formal source roots and endpoint-complete/accepted hair support",
        "H07 displays observable/formal source roots",
        "same five sentinels on a common log-ratio axis with null 1",
        "filled points show the clean cohort, hollow points show full283 sensitivity",
    ):
        assert semantic in figure5

    table3 = _between(manuscript, "### Table 3.", "## 15. References")
    assert "Support / observability and boundary" in table3
    assert "Endpoint-specific clean / Full283 direction evidence" in table3
    for token in (
        "FINAL_D15_ABUNDANCE_PATTERN",
        "FINAL_D15_LENGTH_PATTERN",
        "FINAL_FIRST_HAIR_PATTERN",
        "FINAL_ROOT_WIDTH_PATTERN",
        "FINAL_ROOT_LENGTH_PATTERN",
    ):
        assert f"{{{{{token}}}}}" in table3
    assert "coordinated phenotype associations" not in manuscript.casefold()
    assert "coordinated d15 remodeling" not in manuscript.casefold()


def test_root_cap_is_point_only_and_figure_six_is_an_external_lab_journey() -> None:
    manuscript = _master()
    assert "A root-cap region is neither segmented nor quantified" in manuscript
    assert "Point-only distal landmark; no root-cap region statistic" in manuscript
    assert "root-cap area" not in manuscript.casefold()
    figure6 = _between(manuscript, "### Figure 6.", "## 14. Main Table Legends")
    assert "reusable journey from a raw image and calibration manifest" in figure6
    assert "32 image-level descriptors" in figure6
    assert "later release or clean-install gates" not in figure6
    assert "another laboratory" in figure6
    assert "a new experiment" in figure6
