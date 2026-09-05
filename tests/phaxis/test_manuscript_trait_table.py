from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_manuscript_table_one_lists_the_machine_contract_exactly_once() -> None:
    contract = json.loads(
        (PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json").read_text(
            encoding="utf-8"
        )
    )
    expected = [
        item["id"]
        for family in ("primary_root_traits", "root_hair_traits")
        for item in contract[family]
    ]
    manuscript = (
        PROJECT_ROOT
        / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
    ).read_text(encoding="utf-8")
    table_matches = re.findall(
        r"^### Table 1\.[^\n]*\n(.*?)(?=^### Table 2\.)",
        manuscript,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert len(table_matches) == 1
    table = table_matches[0]
    observed = re.findall(r"^\| ([RH]\d{2}) \|", table, flags=re.MULTILINE)

    assert expected == observed
    assert len(observed) == 32
    assert len(set(observed)) == 32
    table_rows = {
        cells[0]: cells
        for line in table.splitlines()
        if re.match(r"^\| [RH]\d{2} \|", line)
        for cells in [[cell.strip() for cell in line.strip("|").split("|")]]
    }
    assert all(len(cells) == 7 for cells in table_rows.values())
    expected_modules = {
        **{key: "Visible-organ observation frame" for key in ("R01", "R02", "R05", "R06")},
        **{key: "Primary-root trajectory" for key in ("R03", "R04", "R18", "R19")},
        **{key: "Primary-root radial state" for key in ("R07", "R08", "R09", "R10", "R11", "R12")},
        **{key: "Longitudinal caliber pattern" for key in ("R13", "R14", "R15", "R16", "R17")},
        **{key: "Visible-hair population" for key in ("H01", "H05", "H08", "H09")},
        **{key: "Supported individual morphology" for key in ("H02", "H03", "H04", "H10", "H11", "H12")},
        **{key: "Axial deployment" for key in ("H06", "H07", "H13")},
    }
    assert set(expected_modules) == set(table_rows)
    assert all(
        table_rows[trait_id][1] == module
        for trait_id, module in expected_modules.items()
    )
    assert "| ID | Plant-observation module | Descriptor role |" in table
    expected_roles = {
        **{key: "Sentinel endpoint" for key in ("R01", "R07", "H07", "H08", "H11")},
        **{key: "Normalized derivative" for key in ("R03", "R04", "R06", "R12", "R16", "H05", "H09", "H12")},
        **{key: "Distributional/spatial summary" for key in ("R08", "R09", "R10", "R11", "R13", "R14", "R15", "R17", "R18", "R19", "H02", "H03", "H06", "H10", "H13")},
        **{key: "Composite/descriptive context" for key in ("R02", "R05", "H01", "H04")},
    }
    assert set(expected_roles) == set(table_rows)
    for trait_id, expected_role in expected_roles.items():
        assert table_rows[trait_id][2].startswith(expected_role)
    assert table_rows["R01"][2].endswith("(A)")
    assert table_rows["R07"][2].endswith("(W)")
    assert table_rows["H07"][2].endswith("(F)")
    assert table_rows["H08"][2].endswith("(N)")
    assert table_rows["H11"][2].endswith("(L)")
    assert "root-cap region area" not in table.casefold()
    assert "endpoint-complete" in table
    assert "Zero only when H01=0" in table
    assert "H01>0 but no endpoint-complete curve is observed" in table
    assert "zero only when an eligible window has H08=0" in table
    assert "H08>0 with zero length support" in table

    methods = manuscript.split("## 2. Materials and Methods", maxsplit=1)[1].split(
        "## 3. Results", maxsplit=1
    )[0]
    assert "measured totals are zero only in an observed zero-identity scope" in methods
    assert "null when identities exist but no curve is supported" in methods
    assert "explicitly partial when support is incomplete" in methods


def test_numeric_references_are_first_cited_in_strict_order_and_all_are_used() -> None:
    manuscript = (
        PROJECT_ROOT
        / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
    ).read_text(encoding="utf-8")
    body, references = manuscript.split("## 15. References", maxsplit=1)
    first_citation_order: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"\[([0-9,–\-\s]+)\]", body):
        for part in match.group(1).split(","):
            token = part.strip()
            interval = re.fullmatch(r"(\d+)[–-](\d+)", token)
            numbers = (
                range(int(interval.group(1)), int(interval.group(2)) + 1)
                if interval
                else (int(token),)
            )
            for number in numbers:
                if number not in seen:
                    seen.add(number)
                    first_citation_order.append(number)

    assert first_citation_order == list(range(1, 21))
    assert len(re.findall(r"^\d+\. ", references, flags=re.MULTILINE)) == 20
    assert "et al." not in references
    assert "{{PHAXIS_RELEASE_DOI}}` under" in body
    assert "{{PHAXIS_RELEASE_DOI}}` under `{{PHAXIS_SOFTWARE_LICENSE}}` [20]" in body

    discussion = body.split("## 4. Discussion", maxsplit=1)[1].split(
        "## 5. Data Availability", maxsplit=1
    )[0]
    first_paragraph = next(
        paragraph.strip()
        for paragraph in discussion.split("\n\n")
        if paragraph.strip()
    )
    assert "{{FINAL_D15_ABSTRACT_SYNTHESIS}}" not in first_paragraph
    assert "{{FINAL_DISCUSSION_BIOLOGICAL_SYNTHESIS}}" in first_paragraph
    abstract = body.split("## Abstract", maxsplit=1)[1].split(
        "## 1. Introduction", maxsplit=1
    )[0]
    assert "{{FINAL_D15_ABSTRACT_SYNTHESIS}}" in abstract
    assert "fixed endpoint/effect priority" not in first_paragraph
    methods = body.split("## 2. Materials and Methods", maxsplit=1)[1].split(
        "## 3. Results", maxsplit=1
    )[0]
    assert "fixed endpoint/effect priority" in methods
    assert "controlled narrative selection only" in methods


def test_biology_framing_acquisition_fields_and_table_three_order_are_locked() -> None:
    manuscript = (
        PROJECT_ROOT
        / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
    ).read_text(encoding="utf-8")
    abstract = manuscript.split("## Abstract", maxsplit=1)[1].split(
        "## 1. Introduction", maxsplit=1
    )[0]
    assert "283 application images" in abstract
    assert "In the archived D15 RHD6 construct-label × temperature use case" in abstract
    assert "per-root count mean absolute error" in abstract
    assert "genotype-by-temperature" not in abstract

    introduction = manuscript.split("## 1. Introduction", maxsplit=1)[1].split(
        "## 2. Materials and Methods", maxsplit=1
    )[0]
    for phrase in (
        "first visible position",
        "separable biological dimensions",
        "supported morphology",
        "the carrying organ",
        "seven plant-observation modules and five sentinels",
    ):
        assert phrase in introduction

    methods = manuscript.split("## 2. Materials and Methods", maxsplit=1)[1].split(
        "## 3. Results", maxsplit=1
    )[0]
    required_acquisition_tokens = {
        "FINAL_BIOLOGICAL_ACCESSION",
        "FINAL_BIOLOGICAL_CONSTRUCT_CONTROL_IDENTITY_AND_SOURCE",
        "FINAL_BIOLOGICAL_GROWTH_MEDIUM",
        "FINAL_BIOLOGICAL_PHOTOPERIOD",
        "FINAL_BIOLOGICAL_TEMPERATURE_EXPOSURE_ONSET",
        "FINAL_BIOLOGICAL_TEMPERATURE_EXPOSURE_DURATION",
        "FINAL_BIOLOGICAL_PLATE_BLOCK_AND_PLANT_UNIT",
        "FINAL_BIOLOGICAL_IMAGING_DEVICE",
        "FINAL_BIOLOGICAL_IMAGING_OBJECTIVE",
        "FINAL_BIOLOGICAL_NATIVE_PIXEL_SAMPLING",
        "FINAL_BIOLOGICAL_FIELD_SAMPLING_AND_STITCHING",
        "FINAL_BIOLOGICAL_EXCLUSION_RULES",
    }
    assert "FINAL_BIOLOGICAL_ACQUISITION_METHODS" not in methods
    assert all(f"{{{{{token}}}}}" in methods for token in required_acquisition_tokens)
    assert "blocks a formal build rather than being inferred" in methods

    table_three = manuscript.split("### Table 3.", maxsplit=1)[1].split(
        "## 15. References", maxsplit=1
    )[0]
    assert "five linked plant questions" in table_three
    assert "Sentinel / plant question and measured endpoint" in table_three
    assert "Support / observability and boundary" in table_three
    assert "Endpoint-specific clean / Full283 direction evidence" in table_three
    table_rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in table_three.splitlines()
        if re.match(r"^\| \*\*(?:H08/N|H11/L|H07/F|R07/W|R01/A)", line)
    ]
    assert len(table_rows) == 5
    assert all(len(row) == 8 for row in table_rows)
    assert "{{FINAL_D15_LENGTH_PATTERN}}" in table_rows[1][-1]
    assert "{{FINAL_D15_FIRST_HAIR_OBSERVABILITY_BY_CELL}}" in table_rows[2][-2]
    assert "{{FINAL_FIRST_HAIR_PATTERN}}" in table_rows[2][-1]
    assert "{{FINAL_D15_VISIBLE_AXIS_CENSORING_BY_CELL}}" in table_rows[4][-2]
    assert "{{FINAL_ROOT_LENGTH_PATTERN}}" in table_rows[4][-1]
    assert table_three.index("Does the visible hair population differ?") < table_three.index(
        "Is the elongation-qualified boundary displaced along the root?"
    ) < table_three.index("Does carrying-root caliber differ?")

    results = manuscript.split("## 3. Results", maxsplit=1)[1].split(
        "## 4. Discussion", maxsplit=1
    )[0]
    assert "Figure 5 places raw source-unit distributions before construct-label" in results
    assert "temperature, and interaction effects" in results
    assert "(Fig. 5c–e)" in results

    figure_five = manuscript.split("### Figure 5.", maxsplit=1)[1].split(
        "### Figure 6.", maxsplit=1
    )[0]
    assert all(f"({panel})" in figure_five for panel in "abcde")
    assert "(f)" not in figure_five
    assert "same five sentinels on a common log-ratio axis with null 1" in figure_five
    assert "complete four-condition map and coverage of all 32 descriptors" in figure_five
    assert "Supplementary Figs. S7 and S9" in figure_five


def test_successor_master_preserves_exact_machine_token_collection_and_reader_language() -> None:
    manuscript = (
        PROJECT_ROOT
        / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
    ).read_text(encoding="utf-8")
    tokens = re.findall(r"\{\{([A-Z0-9_]+)\}\}", manuscript)
    unique_tokens = sorted(set(tokens))
    token_identity = hashlib.sha256("\n".join(unique_tokens).encode("utf-8")).hexdigest()
    assert len(tokens) == 293
    assert len(unique_tokens) == 229
    assert token_identity == "30639329886205ba7dff2f554085f8f9ec8eb81b00bf66a3cdf6ba25a697b154"

    reader_sections = manuscript.split("## 3. Results", maxsplit=1)[1]
    forbidden = (
        "sealed a/b/c",
        "headline rule",
        "fixed decision",
        "sensitivity-unstable",
        "clean/full-concordant",
        "headline-supported",
        "audit-2.0",
        "actual hash-bound",
        "prelocked",
    )
    assert not any(term in reader_sections.casefold() for term in forbidden)
    assert "coordinated phenotype associations" not in reader_sections.casefold()
    assert "coordinated d15 remodeling" not in reader_sections.casefold()
    assert "fixed N→L→F→W→A" in manuscript


def test_results_modules_and_clean_full_direction_language_are_exact() -> None:
    manuscript = (
        PROJECT_ROOT
        / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
    ).read_text(encoding="utf-8")
    headings = re.findall(r"^### (3\.\d) (.+)$", manuscript, flags=re.MULTILINE)
    assert [number for number, _title in headings] == [f"3.{index}" for index in range(1, 7)]
    assert len(headings) == 6
    for module in (
        "visible-organ observation frame",
        "primary-root trajectory",
        "primary-root radial state",
        "longitudinal caliber pattern",
        "visible-hair population",
        "supported individual morphology",
        "axial deployment",
    ):
        assert module in manuscript
    assert (
        "clean/Full283 point-estimate direction disagreements were "
        "{{FINAL_CLEAN_FULL_UNSTABLE_EFFECTS}}"
    ) in manuscript
    assert "materially different direction or interval support" not in manuscript
    assert "effects with materially different direction" not in manuscript
