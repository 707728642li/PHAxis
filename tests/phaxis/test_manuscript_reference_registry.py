from __future__ import annotations

import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY = (
    PROJECT_ROOT
    / "configs/phaxis/v1_0/manuscript_reference_registry_2_0.json"
)
MANUSCRIPT = (
    PROJECT_ROOT
    / "docs/phaxis/PHAXIS_MANUSCRIPT_MASTER_DRAFT_20260830.md"
)


def test_phaxis_reference_registry_is_current_and_matches_master() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "PHAxis-manuscript-reference-registry-2.0"
    assert payload["product"] == "PHAxis"
    assert payload["product_version"] == "1.0.0"
    assert payload["status"] == "current_biology_forward_primary_source_registry"
    assert payload["journal_reference_limit"] == 20
    assert payload["manuscript"] == MANUSCRIPT.relative_to(PROJECT_ROOT).as_posix()
    assert "frozen RHAxis provenance" in payload["legacy_registry_policy"]

    references = payload["references"]
    assert [record["number"] for record in references] == list(range(1, 21))
    assert len({record["id"] for record in references}) == 20
    assert [record["id"] for record in references[:10]] == [
        "du_2025_mild_heat_rhd6",
        "lee_2002_root_epidermal_pattern",
        "masucci_1994_rhd6_initiation",
        "pires_2013_ancient_root_hair_grn",
        "yi_2010_bhlh_root_hair_growth",
        "datta_2015_rsl4_pulse",
        "ma_2001_phosphorus_density",
        "stetter_2015_root_hair_diversity",
        "bahamonde_2026_boron_rhd6_rsl4",
        "yang_2026_phr1_rsl2",
    ]
    assert [record["id"] for record in references[14:18]] == [
        "berrigan_2024_root_pose",
        "shoaib_2025_algorithmic_root_traits",
        "walsh_2021_dome",
        "shit_2021_cldice",
    ]
    plant_phenomics = [
        record for record in references if record["container"] == "Plant Phenomics"
    ]
    assert [(record["year"], record["doi"]) for record in plant_phenomics] == [
        (2024, "10.34133/plantphenomics.0175"),
        (2025, "10.1016/j.plaphe.2025.100088"),
    ]
    registered_ids = {record["id"] for record in references}
    assert {"ronneberger_2015_unet", "he_2016_resnet"}.isdisjoint(registered_ids)
    assert "validated traits" in plant_phenomics[0]["role"]
    assert "biological discrimination" in plant_phenomics[1]["role"]
    assert all(record["doi"] for record in references[:19])
    assert references[-1]["doi"] is None
    assert references[-1]["doi_authority_token"] == "PHAXIS_RELEASE_DOI"
    assert references[-1]["status"] == "pending_human_release_authority"

    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    reference_block = manuscript.split("## 15. References", maxsplit=1)[1]
    manuscript_lines = {
        int(match.group(1)): match.group(2)
        for match in re.finditer(
            r"^(\d+)\. (.+)$",
            reference_block,
            flags=re.MULTILINE,
        )
    }
    assert set(manuscript_lines) == set(range(1, 21))
    for record in references[:19]:
        assert record["doi"] in manuscript_lines[record["number"]]
        # Scientific binomials retain Markdown italics in the manuscript while
        # the JSON registry remains a plain-text bibliographic authority.
        plain_reference_line = manuscript_lines[record["number"]].replace("*", "")
        assert record["title"] in plain_reference_line
    assert "{{PHAXIS_RELEASE_DOI}}" in manuscript_lines[20]

    serialized = json.dumps(payload, ensure_ascii=False).casefold()
    assert "segment anything" not in serialized
    assert "rootquant" not in serialized
