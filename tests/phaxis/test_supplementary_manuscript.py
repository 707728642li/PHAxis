from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPLEMENT = (
    PROJECT_ROOT
    / "docs"
    / "phaxis"
    / "PHAXIS_SUPPLEMENTARY_MASTER_DRAFT_20260830.md"
)
MASTER = (
    PROJECT_ROOT
    / "docs"
    / "phaxis"
    / "PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
)
ACTIVE_SUPPLEMENT = (
    PROJECT_ROOT
    / "docs"
    / "phaxis"
    / "PHAXIS_SUPPLEMENTARY_MASTER_DRAFT_20260830.md"
)
ACTIVE_MASTER = (
    PROJECT_ROOT
    / "docs"
    / "phaxis"
    / "PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
)
COMPILER_PATH = PROJECT_ROOT / "scripts" / "phaxis" / "compile_supplementary_manuscript.py"
COMPILER_SPEC = importlib.util.spec_from_file_location(
    "phaxis_compile_supplementary_manuscript_title_test", COMPILER_PATH
)
assert COMPILER_SPEC is not None and COMPILER_SPEC.loader is not None
COMPILER = importlib.util.module_from_spec(COMPILER_SPEC)
COMPILER_SPEC.loader.exec_module(COMPILER)
TRAIN_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "phaxis"
    / "v1_0"
    / "stageb_train399_training_config.json"
)
RUNTIME_TRAIN_CONFIG = (
    PROJECT_ROOT
    / "models"
    / "phaxis_stageb_train399_v1_0_20260828"
    / "seed_2026082801"
    / "config.json"
)


def test_supplement_has_complete_reviewer_package() -> None:
    text = SUPPLEMENT.read_text(encoding="utf-8")
    methods = re.findall(r"^### S([1-9])\.", text, flags=re.MULTILINE)
    figures = re.findall(r"^### Figure S([1-9])\.", text, flags=re.MULTILINE)
    tables = re.findall(r"^### Table S([0-9]+)\.", text, flags=re.MULTILINE)
    assert methods == [str(index) for index in range(1, 10)]
    assert figures == [str(index) for index in range(1, 10)]
    assert tables == [str(index) for index in range(1, 11)]
    for phrase in (
        "DOME data, optimization, model, and evaluation checklist",
        "Complete 32-trait dictionary and 82-column export schema",
        "Per-trait primary-root agreement",
        "implementation-choice ledger",
        "Clean-cohort D15 32-descriptor phenotype map and block/day-stratified WT temperature secondary evidence",
    ):
        assert phrase.casefold() in text.casefold()


def test_active_supplement_titles_match_current_main_manuscript() -> None:
    main_text = ACTIVE_MASTER.read_text(encoding="utf-8")
    supplement_text = ACTIVE_SUPPLEMENT.read_text(encoding="utf-8")
    assert COMPILER._validate_companion_titles(
        main_text=main_text,
        master_text=supplement_text,
    ) == "PHAxis enables organ-anchored spatial phenomics of the Arabidopsis root–hair interface"


@pytest.mark.parametrize("title_location", ["first_line", "companion_line"])
def test_compiler_rejects_either_stale_supplement_title(title_location: str) -> None:
    main_text = ACTIVE_MASTER.read_text(encoding="utf-8")
    supplement_text = ACTIVE_SUPPLEMENT.read_text(encoding="utf-8")
    current_title = (
        "PHAxis enables organ-anchored spatial phenomics of the Arabidopsis "
        "root–hair interface"
    )
    lines = supplement_text.splitlines(keepends=True)
    line_index = (
        0
        if title_location == "first_line"
        else next(
            index
            for index, line in enumerate(lines)
            if line.startswith("**Companion main manuscript:**")
        )
    )
    assert current_title in lines[line_index]
    lines[line_index] = lines[line_index].replace(current_title, "Stale companion title")
    with pytest.raises(COMPILER.SupplementaryCompileError, match="differs"):
        COMPILER._validate_companion_titles(
            main_text=main_text,
            master_text="".join(lines),
        )


def test_s7_s9_name_the_d15_analysis_cohorts_and_separate_sensitivity_views() -> None:
    text = SUPPLEMENT.read_text(encoding="utf-8")
    s7_s9 = text.split("### Figure S7.", maxsplit=1)[1].split(
        "## Supplementary Tables and Data Files", maxsplit=1
    )[0]
    for phrase in (
        "Clean-cohort D15 analysis, full-cohort D15 sensitivity",
        "Clean-cohort D15 primary ratios",
        "full-cohort D15 overlap-inclusion sensitivity ratios",
        "Figure S9. Clean-cohort D15 32-descriptor phenotype map",
        "clean-cohort D15 raw medians",
        "full-cohort D15 sensitivity",
        "Clean261 and overlap-inclusive Full283 sensitivity estimates remain separate",
    ):
        assert phrase in s7_s9


def test_table_s9_is_a_complete_biological_multitrait_ledger() -> None:
    text = SUPPLEMENT.read_text(encoding="utf-8")
    assert (
        "### Table S9. Complete 32-trait D15 atlas and block/day-stratified WT "
        "temperature secondary analysis"
    ) in text
    for phrase in (
        "256 descriptor–cohort–condition rows",
        "192 descriptor–cohort–contrast slots",
        "EV-22°C",
        "EV-30°C",
        "OE-labelled-22°C",
        "OE-labelled-30°C",
        "non_null_source_unit_n/source_unit_total",
        "median, Q25, Q75, and IQR",
        "no_finite_observations_in_formal_D15_condition",
        "unadjusted source-unit descriptions before model fitting",
        "estimated_fixed_15_effect_family",
        "trait_not_in_prespecified_five_endpoint_15_effect_family",
        "Not_estimated",
        "must not be read as evidence for no biological response",
        "22 byte-identical HumanCurated443–application pairs",
    ):
        assert phrase in text
    assert "clean and full D15 views retain separate raw" in text
    assert "A measured zero remains part of the raw distribution" in text
    assert "an unobservable value remains null" in text
    assert "missing plotting value" in text


def test_table_s9_machine_fill_policy_rejects_semantic_drift() -> None:
    text = SUPPLEMENT.read_text(encoding="utf-8")
    policy = text.split("## Supplementary machine-fill policy", maxsplit=1)[1]
    for phrase in (
        "any canonical descriptor lacks either cohort",
        "one of the four ordered condition summaries",
        "observability does not equal its non-null numerator divided by its source-unit denominator",
        "IQR does not equal Q75−Q25",
        "an effect slot is absent",
        "a null effect lacks its declared reason",
    ):
        assert phrase in policy


def test_active_table_s9_adds_a_separate_fail_closed_wt_secondary_family() -> None:
    text = ACTIVE_SUPPLEMENT.read_text(encoding="utf-8")
    assert (
        "### Table S9. Complete 32-trait D15 atlas and block/day-stratified WT "
        "temperature secondary analysis"
    ) in text
    table_s9 = text.split("### Table S9.", maxsplit=1)[1].split(
        "### Table S10.", maxsplit=1
    )[0]
    for phrase in (
        "six identity-linked long-format blocks",
        "The original D15 atlas remains intact in Blocks A–C",
        "`wt_gate_flow`",
        "`wt_experiment_contrasts`",
        "`wt_same_day_meta`",
        "unknown_developmental_day",
        "at least three eligible experiments",
        "No cross-day pooling",
        "do not enter, enlarge, or select the D15 five-endpoint/15-effect family",
    ):
        assert phrase in table_s9
    policy = text.split("## Supplementary machine-fill policy", maxsplit=1)[1]
    for phrase in (
        "three source-table hashes differ from the sealed analysis summary",
        "an unknown-day contrast is meta-eligible",
        "a pooled row has fewer than three eligible same-day experiments",
        "a `not_estimable` row contains a pooled statistic",
        "clean/full or developmental-day boundaries are crossed",
    ):
        assert phrase in policy
    assert "{{FINAL_WT" not in text

    figure_s9 = text.split("### Figure S9.", maxsplit=1)[1].split(
        "## Supplementary Tables and Data Files", maxsplit=1
    )[0]
    for phrase in (
        "unchanged five-endpoint/15-effect family",
        "neither selects nor alters this D15 family or its narrative branch",
        "developmental-day-specific random-effects REML/Hartung–Knapp diamonds",
        "at least three same-day experiments",
        "A typed `Not estimable` label",
        "unknown developmental day remain descriptive within-experiment contrasts",
        "produce no pooled diamond",
        "Clean261 and overlap-inclusive Full283 sensitivity estimates remain separate",
    ):
        assert phrase in figure_s9


def test_active_main_reports_wt_as_a_separate_secondary_result() -> None:
    text = ACTIVE_MASTER.read_text(encoding="utf-8")
    paragraph = next(
        paragraph
        for paragraph in text.split("\n\n")
        if paragraph.startswith("WT temperature comparisons")
    )
    for phrase in (
        "within compatible archived experiments",
        "reported separately in Supplementary Fig. S9 and Table S9",
        "each known developmental day was represented by only one archived experiment",
        "same-day synthesis remained not estimable",
        "did not alter the D15 factorial analysis",
    ):
        assert phrase in paragraph
    assert "{{" not in paragraph
    assert "FINAL_WT" not in text


def test_main_supplement_cross_references_resolve() -> None:
    main = MASTER.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    cited = set(re.findall(r"Supplementary Fig\. S([0-9]+)", main))
    available = set(re.findall(r"^### Figure S([0-9]+)\.", supplement, re.MULTILINE))
    assert cited
    assert cited <= available


def test_static_training_methods_match_formal_seed_config() -> None:
    config = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
    text = SUPPLEMENT.read_text(encoding="utf-8")
    expected = {
        "crop": 768,
        "crops_per_image": 8,
        "batch_size": 8,
        "workers": 8,
        "epochs": 60,
        "out_stride": 2,
        "in_channels": 3,
        "encoder": "resnet34",
        "imagenet_source": "timm/resnet34.a1_in1k",
    }
    assert {key: config[key] for key in expected} == expected
    for phrase in (
        "2 µm px⁻¹",
        "eight deterministic 768 × 768 crops",
        "batch size 8",
        "eight data-loader workers",
        "fixed 60-epoch schedule",
        "output stride 2",
        "three input channels",
        "ResNet34",
        "timm/resnet34.a1_in1k",
    ):
        assert phrase in text


def test_public_training_config_matches_runtime_seed_when_available() -> None:
    """Keep the portable publication config byte-for-value aligned with training."""

    canonical = json.loads(TRAIN_CONFIG.read_text(encoding="utf-8"))
    if RUNTIME_TRAIN_CONFIG.is_file():
        runtime = json.loads(RUNTIME_TRAIN_CONFIG.read_text(encoding="utf-8"))
        assert canonical == runtime


def test_supplement_does_not_invent_prohibited_scope() -> None:
    text = SUPPLEMENT.read_text(encoding="utf-8").lower()
    for prohibited in (
        "root-cap area metric",
        "independent accuracy",
        "three-dimensional hair length",
        "tensorrt was used",
        "pyroothair was used",
    ):
        assert prohibited not in text
    assert "no root-cap region or area is produced" in text
    assert "one-to-one" in text
    assert "20 µm" in text
