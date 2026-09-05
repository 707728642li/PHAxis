from __future__ import annotations

import json
from pathlib import Path
import re

from phaxis.traits import HAIR_TRAIT_FIELDS, IMAGE_TRAIT_FIELDS, ROOT_TRAIT_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json"
SCHEMA_PATH = PROJECT_ROOT / "configs/phaxis/v1_0/image_traits.schema.json"
CATALOG_PATH = PROJECT_ROOT / "docs/phaxis/TRAIT_CONTRACT_CN.md"


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _unit_cell(unit: str) -> str:
    return {
        "um": "µm (`um`)",
        "um2": "µm² (`um2`)",
        "um2_per_mm": "µm²/mm (`um2_per_mm`)",
        "um_per_mm": "µm/mm (`um_per_mm`)",
        "rad_per_mm": "rad/mm (`rad_per_mm`)",
        "count_per_mm": "count/mm",
        "count": "count",
        "ratio": "ratio",
    }[unit]


def test_bilingual_catalog_is_exactly_the_machine_contract_19_plus_13() -> None:
    contract = _payload(CONTRACT_PATH)
    catalog = CATALOG_PATH.read_text(encoding="utf-8")
    compact_catalog = " ".join(catalog.split())
    roots = contract["primary_root_traits"]
    hairs = contract["root_hair_traits"]
    rows = [*roots, *hairs]

    assert contract["counts"] == {
        "nonredundant_biological_numeric_fields": 32,
        "primary_root_fields": 19,
        "root_hair_fields": 13,
        "root_cap_region_fields": 0,
        "distal_root_cap_point_geometry_fields": 2,
    }
    assert [row["id"] for row in roots] == [f"R{index:02d}" for index in range(1, 20)]
    assert [row["id"] for row in hairs] == [f"H{index:02d}" for index in range(1, 14)]
    assert tuple(row["field"] for row in roots) == ROOT_TRAIT_FIELDS
    assert tuple(row["field"] for row in hairs) == HAIR_TRAIT_FIELDS
    assert len(rows) == len({row["field"] for row in rows}) == 32

    for row in rows:
        assert row["display_name_cn"].strip()
        assert row["display_name_en"].strip()
        marker = (
            f"| {row['id']} | {row['display_name_cn']}<br>{row['display_name_en']} | "
            f"`{row['field']}` | {_unit_cell(row['unit'])} |"
        )
        assert catalog.count(marker) == 1, row["id"]

    observed_ids = re.findall(r"^\| ([RH]\d{2}) \|", catalog, flags=re.MULTILINE)
    assert observed_ids == [row["id"] for row in rows]
    assert "single authoritative, human-readable" in compact_catalog
    assert "逐字节复制" in catalog
    assert "never an independently edited second catalogue" in compact_catalog


def test_catalog_and_flat_schema_have_no_id_field_or_unit_drift() -> None:
    contract = _payload(CONTRACT_PATH)
    schema = _payload(SCHEMA_PATH)
    rows = [*contract["primary_root_traits"], *contract["root_hair_traits"]]

    assert len(IMAGE_TRAIT_FIELDS) == len(set(IMAGE_TRAIT_FIELDS)) == 82
    for row in rows:
        property_contract = schema["properties"][row["field"]]
        assert property_contract["x-trait-id"] == row["id"]
        assert property_contract["x-unit"] == row["unit"]

    trait_ids = [
        value["x-trait-id"]
        for value in schema["properties"].values()
        if isinstance(value, dict) and "x-trait-id" in value
    ]
    assert trait_ids == [row["id"] for row in rows]


def test_catalog_locks_root_cap_cross_expert_window_and_censoring_semantics() -> None:
    contract = _payload(CONTRACT_PATH)
    catalog = CATALOG_PATH.read_text(encoding="utf-8")
    invariants = contract["invariants"]
    hairs = {row["id"]: row for row in contract["root_hair_traits"]}

    assert contract["plant_facing_catalog"] == {
        "path": "docs/phaxis/TRAIT_CONTRACT_CN.md",
        "languages": ["zh-CN", "en"],
        "handover_copy": "PHENOTYPE_CAPABILITIES_CN.md",
        "handover_copy_semantics": (
            "byte-identical copy of the catalog path at handover build time; "
            "never an independently maintained catalog"
        ),
    }
    assert invariants["root_cap_region_output"] is False
    assert invariants["root_cap_area_used"] is False
    assert invariants["distal_window_um"] == [1000.0, 4000.0]
    assert invariants["first_long_hair_minimum_length_um"] == 40.0
    assert invariants["maximum_identity_length_base_match_um"] == 20.0
    assert contract["length_link_contract"]["cardinality"] == (
        "one-to-zero-or-one in both directions"
    )
    assert contract["length_link_contract"]["stageb_predicted_length_role"] == (
        "diagnostic_only"
    )
    assert hairs["H13"]["censoring"] == "right-censored descriptive value"

    for token in (
        "The root-cap representation is exactly one distal/root-cap point",
        "no root-cap region",
        "H06、H07、H13",
        "`[1,4) mm`",
        "left-closed and right-open",
        "H07 为 null",
        "whole_hair_zone_confirmatory_allowed=false",
        "right-censored",
        "numeric partial",
        "five-member root-hair identity/count expert",
        "does not report 82 phenotypes",
        "--model-contract <official-contract.json>",
        'read_model_contract_authority("<official-contract.json>")',
        "model_contract_proposal=authority.receipt_fields()",
        "model_contract_public_identity=authority.public_identity_fields()",
    ):
        assert token in catalog
    folded = catalog.casefold()
    for forbidden in (
        "rhaxiscc",
        "rhaxis_nextgen",
        "hybrid-max",
        "stage-b",
        "stage b",
        "v2.0",
        "outputs/",
        "models/",
        "scripts/phaxis/",
    ):
        assert forbidden not in folded


def test_public_cards_call_82_a_schema_not_82_phenotypes() -> None:
    for relative in (
        "README.md",
        "MODEL_CARD.md",
        "DATA_CARD.md",
        "docs/phaxis/USER_GUIDE.md",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        folded = " ".join(text.casefold().split())
        assert "32 canonical" in folded, relative
        assert "82-column" in folded or "82 columns" in folded, relative
        assert "not" in folded and "82 phenotypes" in folded, relative
        assert "reports 82 phenotypes" not in folded, relative
        assert "82 项表型" not in text, relative
        assert "rhaxiscc" not in folded, relative
        assert "rhaxis_nextgen" not in folded, relative
        assert "hybrid-max" not in folded, relative
        assert "stage-b" not in folded, relative
        assert "stage b" not in folded, relative
        assert "v2.0" not in folded, relative
