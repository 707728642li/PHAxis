from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phaxis.biological_analysis import (
    RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
    RAW_EFFECT_HC3_INTERVAL,
    RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
    RAW_EFFECT_OLS_MEAN_CONTRAST,
    raw_median_bootstrap_seed,
)
from phaxis.io import sha256_json
from phaxis.multitrait_atlas import (
    EFFECT_NAME_TO_KEY,
    GROUP_ORDER,
    MEASUREMENT_FAMILY_ORDER,
    MEASUREMENT_FAMILY_TRAIT_IDS,
    MultitraitAtlasError,
    SCHEMA_VERSION,
    build_multitrait_atlas,
    descriptive_heatmap_matrices,
    validate_multitrait_atlas_against_sources,
    validate_multitrait_atlas_structure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_MEASUREMENT_FAMILY_TRAIT_IDS = {
    "visible_hair_abundance": ("H01", "H05", "H08", "H09"),
    "conditional_projected_length": (
        "H02",
        "H03",
        "H04",
        "H10",
        "H11",
        "H12",
    ),
    "axial_deployment": ("H06", "H07", "H13"),
    "visible_root_extent": ("R01", "R02", "R05", "R06"),
    "root_form_trajectory": (
        "R03",
        "R04",
        "R07",
        "R08",
        "R09",
        "R10",
        "R11",
        "R12",
        "R13",
        "R14",
        "R15",
        "R16",
        "R17",
        "R18",
        "R19",
    ),
}


def _hash(label: str) -> str:
    return sha256_json({"fixture": label})


def _fixture():
    contract = json.loads(
        (PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json").read_text(
            encoding="utf-8"
        )
    )
    fields = [
        record["field"]
        for family in ("primary_root_traits", "root_hair_traits")
        for record in contract[family]
    ]

    def traits(count: int) -> pd.DataFrame:
        rows = []
        for index in range(count):
            row = {
                "task_id": f"unit-{index:03d}",
                "source_image_sha256": _hash(f"image-{index}"),
                "experiment_key": "D15_8d",
                "study_role": "rhd6_factorial_8d_primary",
                "condition_code": GROUP_ORDER[index % 4],
                "formal_statistics_eligible": True,
            }
            for field_index, field in enumerate(fields):
                row[field] = float(1 + field_index + index / 100.0)
            rows.append(row)
        return pd.DataFrame(rows)

    full_image = traits(283)
    # Mirror the sealed application contract: cohort tables carry biology
    # metadata, all 13 hair traits, and the two prespecified root endpoints;
    # the canonical image export supplies the other 17 root descriptors.
    metadata_fields = [
        "task_id",
        "source_image_sha256",
        "experiment_key",
        "study_role",
        "condition_code",
        "formal_statistics_eligible",
    ]
    cohort_trait_fields = [
        record["field"] for record in contract["root_hair_traits"]
    ] + ["visible_root_axis_length_um", "median_root_width_um"]
    full = full_image.loc[:, metadata_fields + cohort_trait_fields].copy()
    clean = full.iloc[:261].copy()
    primary_endpoints = {
        "local_hair_count_1_4mm",
        "local_median_hair_length_um_1_4mm",
        "first_hair_ge40um_distance_from_distal_point_um",
        "median_root_width_um",
        "visible_root_axis_length_um",
    }

    def analysis(cohort: str, table: pd.DataFrame) -> pd.DataFrame:
        rows = []
        n = len(table)
        h11_medians = {
            condition: float(
                table.loc[
                    table["condition_code"].astype(str).eq(condition),
                    "local_median_hair_length_um_1_4mm",
                ].median()
            )
            for condition in GROUP_ORDER
        }
        ev22, ev30, oe22, oe30 = (
            h11_medians[condition] for condition in GROUP_ORDER
        )
        h11_raw = {
            "construct_OE_minus_EV": 0.5
            * ((oe22 - ev22) + (oe30 - ev30)),
            "temperature_30C_minus_22C": 0.5
            * ((ev30 - ev22) + (oe30 - oe22)),
            "construct_by_temperature_interaction": (oe30 - oe22)
            - (ev30 - ev22),
        }
        for endpoint_index, endpoint in enumerate(sorted(primary_endpoints)):
            for effect_index, effect in enumerate(EFFECT_NAME_TO_KEY):
                estimate = 0.8 + endpoint_index * 0.03 + effect_index * 0.02
                is_h11 = endpoint == "local_median_hair_length_um_1_4mm"
                raw_estimate = (
                    h11_raw[effect] if is_h11 else (estimate - 1.0) * 100.0
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "endpoint": endpoint,
                        "effect": effect,
                        "n": n,
                        "estimate": estimate,
                        "ci95_low": estimate - 0.1,
                        "ci95_high": estimate + 0.1,
                        "effect_scale": "ratio",
                        "raw_effect_estimate": raw_estimate,
                        "raw_effect_ci95_low": raw_estimate - 5.0,
                        "raw_effect_ci95_high": raw_estimate + 5.0,
                        "raw_effect_estimand": (
                            RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
                            if is_h11
                            else RAW_EFFECT_OLS_MEAN_CONTRAST
                        ),
                        "raw_effect_interval_method": (
                            RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
                            if is_h11
                            else RAW_EFFECT_HC3_INTERVAL
                        ),
                        "raw_effect_bootstrap_replicates": 5000 if is_h11 else 0,
                        "raw_effect_bootstrap_seed": (
                            raw_median_bootstrap_seed(
                                seed=20260823,
                                field="local_median_hair_length_um_1_4mm",
                                component="continuous",
                            )
                            if is_h11
                            else None
                        ),
                        "standardized_effect": raw_estimate / 50.0,
                        "standardized_ci95_low": (raw_estimate - 5.0) / 50.0,
                        "standardized_ci95_high": (raw_estimate + 5.0) / 50.0,
                        "causal_treatment_claim_allowed": False,
                    }
                )
        return pd.DataFrame(rows)

    primary = analysis("primary_clean261", clean)
    sensitivity = analysis("sensitivity_full283", full)
    hashes = {
        "trait_contract": _hash("contract"),
        "clean_traits": _hash("clean"),
        "full_traits": _hash("full"),
        "canonical_image_traits": _hash("canonical-image-traits"),
        "analysis_primary_table": _hash("primary"),
        "analysis_sensitivity_table": _hash("sensitivity"),
    }
    atlas = build_multitrait_atlas(
        trait_contract=contract,
        clean_traits=clean,
        full_traits=full,
        canonical_image_traits=full_image,
        primary_analysis=primary,
        sensitivity_analysis=sensitivity,
        source_sha256=hashes,
    )
    return contract, clean, full, full_image, primary, sensitivity, hashes, atlas


def _reseal(payload: dict) -> None:
    payload.pop("atlas_identity_sha256", None)
    payload["atlas_identity_sha256"] = sha256_json(payload)


def test_complete_32_trait_atlas_is_source_recomputable() -> None:
    contract, clean, full, full_image, primary, sensitivity, hashes, atlas = _fixture()
    assert SCHEMA_VERSION == "PHAxis-multitrait-atlas-2.0"
    assert atlas["schema_version"] == SCHEMA_VERSION
    assert atlas["descriptor_count"] == 32
    assert atlas["estimated_effect_slot_count"] == 30
    assert atlas["not_estimated_effect_slot_count"] == 162
    assert atlas["condition_summary_slot_count"] == 256
    assert atlas["estimated_condition_summary_slot_count"] == 256
    assert atlas["not_estimated_condition_summary_slot_count"] == 0
    assert atlas["measurement_family_order"] == list(MEASUREMENT_FAMILY_ORDER)
    assert "biological_response_family_order" not in atlas
    observed_measurement_families = {
        descriptor["measurement_family"]
        for descriptor in atlas["descriptors"]
    }
    assert observed_measurement_families == set(MEASUREMENT_FAMILY_ORDER)
    assert MEASUREMENT_FAMILY_TRAIT_IDS == EXPECTED_MEASUREMENT_FAMILY_TRAIT_IDS
    assert sum(map(len, MEASUREMENT_FAMILY_TRAIT_IDS.values())) == 32
    for family, expected_trait_ids in EXPECTED_MEASUREMENT_FAMILY_TRAIT_IDS.items():
        assert tuple(
            descriptor["trait_id"]
            for descriptor in atlas["descriptors"]
            if descriptor["measurement_family"] == family
        ) == expected_trait_ids
    assert all(
        "biological_response_family" not in descriptor
        for descriptor in atlas["descriptors"]
    )
    first = atlas["descriptors"][0]["cohorts"]["primary_clean261"]
    assert list(first["condition_summaries"]) == list(GROUP_ORDER)
    source = clean[
        clean["condition_code"].astype(str) == GROUP_ORDER[0]
    ]["visible_root_axis_length_um"].astype(float)
    summary = first["condition_summaries"][GROUP_ORDER[0]]
    assert summary["source_unit_total"] == summary["non_null_source_unit_n"] == len(source)
    assert summary["observability_fraction"] == 1.0
    assert summary["median"] == pytest.approx(source.median())
    assert summary["q25"] == pytest.approx(source.quantile(0.25))
    assert summary["q75"] == pytest.approx(source.quantile(0.75))
    assert summary["iqr"] == pytest.approx(source.quantile(0.75) - source.quantile(0.25))
    assert summary["raw_unadjusted"] is True
    assert summary["not_estimable_reason"] is None
    validate_multitrait_atlas_against_sources(
        atlas,
        trait_contract=contract,
        clean_traits=clean,
        full_traits=full,
        canonical_image_traits=full_image,
        primary_analysis=primary,
        sensitivity_analysis=sensitivity,
        source_sha256=hashes,
    )
    h11 = next(
        descriptor
        for descriptor in atlas["descriptors"]
        if descriptor["field"] == "local_median_hair_length_um_1_4mm"
    )
    for cohort in ("primary_clean261", "sensitivity_full283"):
        for effect in h11["cohorts"][cohort]["effects"].values():
            assert effect["raw_effect_estimand"] == (
                RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
            )
            assert effect["raw_effect_interval_method"] == (
                RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
            )
            assert effect["raw_effect_bootstrap_replicates"] == 5000
            assert effect["raw_effect_bootstrap_seed"] > 0


def test_h11_atlas_rejects_labels_without_four_cell_median_value() -> None:
    contract, clean, full, full_image, primary, sensitivity, hashes, _ = _fixture()
    primary = primary.copy()
    target = (
        primary["endpoint"].eq("local_median_hair_length_um_1_4mm")
        & primary["effect"].eq("construct_OE_minus_EV")
    )
    primary.loc[target, "raw_effect_estimate"] += 1.0
    with pytest.raises(
        MultitraitAtlasError, match="not the four-cell median contrast"
    ):
        build_multitrait_atlas(
            trait_contract=contract,
            clean_traits=clean,
            full_traits=full,
            canonical_image_traits=full_image,
            primary_analysis=primary,
            sensitivity_analysis=sensitivity,
            source_sha256=hashes,
        )


def test_h11_atlas_structure_rejects_resealed_companion_drift() -> None:
    contract, *_, hashes, atlas = _fixture()
    broken = deepcopy(atlas)
    h11 = next(
        descriptor
        for descriptor in broken["descriptors"]
        if descriptor["field"] == "local_median_hair_length_um_1_4mm"
    )
    h11["cohorts"]["primary_clean261"]["effects"]["OE_vs_EV"][
        "raw_effect_bootstrap_replicates"
    ] = 4999
    _reseal(broken)
    with pytest.raises(MultitraitAtlasError, match="H11 raw-median companion drift"):
        validate_multitrait_atlas_structure(
            broken, trait_contract=contract, expected_source_sha256=hashes
        )


def test_atlas_accepts_null_seed_csv_roundtrip_and_rejects_zero_seed(
    tmp_path: Path,
) -> None:
    contract, clean, full, full_image, primary, sensitivity, hashes, _ = _fixture()
    restored = []
    for name, table in (("primary", primary), ("sensitivity", sensitivity)):
        path = tmp_path / f"{name}.csv"
        table.to_csv(path, index=False)
        restored.append(pd.read_csv(path))
    build_multitrait_atlas(
        trait_contract=contract,
        clean_traits=clean,
        full_traits=full,
        canonical_image_traits=full_image,
        primary_analysis=restored[0],
        sensitivity_analysis=restored[1],
        source_sha256=hashes,
    )
    wrong = restored[0].copy()
    non_h11 = ~wrong["endpoint"].eq("local_median_hair_length_um_1_4mm")
    wrong.loc[non_h11, "raw_effect_bootstrap_seed"] = 0
    with pytest.raises(MultitraitAtlasError, match="raw-mean companion drift"):
        build_multitrait_atlas(
            trait_contract=contract,
            clean_traits=clean,
            full_traits=full,
            canonical_image_traits=full_image,
            primary_analysis=wrong,
            sensitivity_analysis=restored[1],
            source_sha256=hashes,
        )


def test_descriptive_heatmap_foregrounds_all_32_traits_and_only_fixed_effects() -> None:
    *_, atlas = _fixture()
    matrices = descriptive_heatmap_matrices(atlas)
    assert matrices["descriptive_cohort"] == "primary_clean261"
    assert len(matrices["trait_labels"]) == 32
    assert matrices["condition_labels"] == list(GROUP_ORDER)
    assert matrices["standardized_medians"].shape == (32, 4)
    assert matrices["relative_iqrs"].shape == (32, 4)
    assert matrices["condition_coverage"].shape == (32, 4)
    assert len(matrices["effect_trait_labels"]) == 5
    assert matrices["effect_estimates"].shape == (5, 6)
    assert np.isfinite(matrices["effect_estimates"]).all()
    assert np.nanmin(matrices["condition_coverage"]) >= 0
    assert np.nanmax(matrices["condition_coverage"]) <= 1
    assert np.nanmin(matrices["relative_iqrs"]) >= 0
    assert np.nanmax(matrices["relative_iqrs"]) <= 1
    for row in matrices["standardized_medians"]:
        finite = row[np.isfinite(row)]
        assert finite.size > 0
        assert float(np.mean(finite)) == pytest.approx(0.0, abs=1e-12)


def test_rejects_omitted_canonical_trait_even_when_resealed() -> None:
    contract, *_, hashes, atlas = _fixture()
    broken = deepcopy(atlas)
    broken["descriptors"].pop()
    _reseal(broken)
    with pytest.raises(MultitraitAtlasError, match="omits canonical traits"):
        validate_multitrait_atlas_structure(
            broken, trait_contract=contract, expected_source_sha256=hashes
        )


def test_rejects_measurement_family_drift_even_when_resealed() -> None:
    contract, *_, hashes, atlas = _fixture()
    broken = deepcopy(atlas)
    broken["descriptors"][0]["measurement_family"] = "root_form_trajectory"
    _reseal(broken)
    with pytest.raises(MultitraitAtlasError, match="trait contract metadata drift"):
        validate_multitrait_atlas_structure(
            broken, trait_contract=contract, expected_source_sha256=hashes
        )


def test_rejects_legacy_v1_family_keys_even_when_resealed() -> None:
    contract, *_, hashes, atlas = _fixture()
    broken = deepcopy(atlas)
    broken["biological_response_family_order"] = broken.pop(
        "measurement_family_order"
    )
    _reseal(broken)
    with pytest.raises(
        MultitraitAtlasError, match="legacy v1 biological_response_family_order"
    ):
        validate_multitrait_atlas_structure(
            broken, trait_contract=contract, expected_source_sha256=hashes
        )

    broken = deepcopy(atlas)
    descriptor = broken["descriptors"][0]
    descriptor["biological_response_family"] = descriptor.pop(
        "measurement_family"
    )
    _reseal(broken)
    with pytest.raises(
        MultitraitAtlasError, match="legacy v1 biological_response_family key"
    ):
        validate_multitrait_atlas_structure(
            broken, trait_contract=contract, expected_source_sha256=hashes
        )


def test_rejects_legacy_v1_schema_even_when_resealed() -> None:
    contract, *_, hashes, atlas = _fixture()
    broken = deepcopy(atlas)
    broken["schema_version"] = "PHAxis-multitrait-atlas-1.0"
    _reseal(broken)
    with pytest.raises(MultitraitAtlasError, match="multitrait atlas schema changed"):
        validate_multitrait_atlas_structure(
            broken, trait_contract=contract, expected_source_sha256=hashes
        )


@pytest.mark.parametrize(
    "legacy_value", ("conditional_elongation", "root_growth_extent")
)
def test_rejects_legacy_v1_family_values_even_when_resealed(
    legacy_value: str,
) -> None:
    contract, *_, hashes, atlas = _fixture()
    broken = deepcopy(atlas)
    broken["descriptors"][0]["measurement_family"] = legacy_value
    _reseal(broken)
    with pytest.raises(
        MultitraitAtlasError, match="legacy v1 measurement-family value"
    ):
        validate_multitrait_atlas_structure(
            broken, trait_contract=contract, expected_source_sha256=hashes
        )


def test_rejects_support_and_effect_denominators_that_do_not_close() -> None:
    contract, *_, hashes, atlas = _fixture()
    broken = deepcopy(atlas)
    clean = broken["descriptors"][0]["cohorts"]["primary_clean261"]
    clean["non_null_source_unit_n"] -= 1
    _reseal(broken)
    with pytest.raises(MultitraitAtlasError, match="support fraction denominator"):
        validate_multitrait_atlas_structure(
            broken, trait_contract=contract, expected_source_sha256=hashes
        )

    broken = deepcopy(atlas)
    condition = broken["descriptors"][0]["cohorts"]["primary_clean261"][
        "condition_summaries"
    ][GROUP_ORDER[0]]
    condition["observability_fraction"] -= 0.1
    _reseal(broken)
    with pytest.raises(MultitraitAtlasError, match="observability denominator"):
        validate_multitrait_atlas_structure(
            broken, trait_contract=contract, expected_source_sha256=hashes
        )

    broken = deepcopy(atlas)
    condition = broken["descriptors"][0]["cohorts"]["primary_clean261"][
        "condition_summaries"
    ][GROUP_ORDER[0]]
    condition["iqr"] += 1.0
    _reseal(broken)
    with pytest.raises(MultitraitAtlasError, match="raw median/IQR"):
        validate_multitrait_atlas_structure(
            broken, trait_contract=contract, expected_source_sha256=hashes
        )

    broken = deepcopy(atlas)
    clean = broken["descriptors"][0]["cohorts"]["primary_clean261"]
    clean["effect_source_unit_n"] -= 1
    _reseal(broken)
    with pytest.raises(MultitraitAtlasError, match="effect denominator does not close"):
        validate_multitrait_atlas_structure(
            broken, trait_contract=contract, expected_source_sha256=hashes
        )


def test_rejects_source_hash_drift() -> None:
    contract, *_, hashes, atlas = _fixture()
    changed = dict(hashes)
    changed["clean_traits"] = _hash("different-clean")
    with pytest.raises(MultitraitAtlasError, match="source hash drift"):
        validate_multitrait_atlas_structure(
            atlas, trait_contract=contract, expected_source_sha256=changed
        )


def test_rejects_invented_effect_for_unmodelled_descriptor() -> None:
    contract, *_, hashes, atlas = _fixture()
    broken = deepcopy(atlas)
    # R02 is descriptive and outside the five-endpoint fixed family.
    effect = broken["descriptors"][1]["cohorts"]["primary_clean261"]["effects"][
        "OE_vs_EV"
    ]
    effect.update(
        {
            "status": "estimated_fixed_15_effect_family",
            "estimate": 1.2,
            "ci95_low": 1.0,
            "ci95_high": 1.4,
            "endpoint_n": 261,
            "effect_scale": "ratio",
            "not_estimable_reason": None,
        }
    )
    _reseal(broken)
    with pytest.raises(MultitraitAtlasError, match="invented effect outside fixed family"):
        validate_multitrait_atlas_structure(
            broken, trait_contract=contract, expected_source_sha256=hashes
        )


def test_builder_rejects_analysis_n_that_disagrees_with_source_units() -> None:
    contract, clean, full, full_image, primary, sensitivity, hashes, _ = _fixture()
    primary = primary.copy()
    primary.loc[0, "n"] = 260
    with pytest.raises(MultitraitAtlasError, match="effect denominator does not close"):
        build_multitrait_atlas(
            trait_contract=contract,
            clean_traits=clean,
            full_traits=full,
            canonical_image_traits=full_image,
            primary_analysis=primary,
            sensitivity_analysis=sensitivity,
            source_sha256=hashes,
        )


def test_zero_observability_condition_is_explicit_not_a_fabricated_zero() -> None:
    contract, clean, full, full_image, primary, sensitivity, hashes, _ = _fixture()
    field = "root_axis_chord_um"  # descriptive R02, outside the fixed effect family
    full_image = full_image.copy()
    full_image.loc[
        full_image["condition_code"].astype(str) == GROUP_ORDER[0], field
    ] = None
    atlas = build_multitrait_atlas(
        trait_contract=contract,
        clean_traits=clean,
        full_traits=full,
        canonical_image_traits=full_image,
        primary_analysis=primary,
        sensitivity_analysis=sensitivity,
        source_sha256=hashes,
    )
    validate_multitrait_atlas_against_sources(
        atlas,
        trait_contract=contract,
        clean_traits=clean,
        full_traits=full,
        canonical_image_traits=full_image,
        primary_analysis=primary,
        sensitivity_analysis=sensitivity,
        source_sha256=hashes,
    )
    descriptor = next(row for row in atlas["descriptors"] if row["field"] == field)
    for cohort in ("primary_clean261", "sensitivity_full283"):
        summary = descriptor["cohorts"][cohort]["condition_summaries"][
            GROUP_ORDER[0]
        ]
        assert summary["non_null_source_unit_n"] == 0
        assert summary["observability_fraction"] == 0.0
        assert summary["median"] is summary["q25"] is summary["q75"] is summary["iqr"] is None
        assert summary["not_estimable_reason"] == (
            "no_finite_observations_in_formal_D15_condition"
        )


def test_measured_zero_is_observed_and_remains_in_raw_distribution() -> None:
    contract, clean, full, full_image, primary, sensitivity, hashes, _ = _fixture()
    field = "root_axis_chord_um"  # descriptive R02, outside the fixed effect family
    full_image = full_image.copy()
    full_image.loc[
        full_image["condition_code"].astype(str) == GROUP_ORDER[0], field
    ] = 0.0
    atlas = build_multitrait_atlas(
        trait_contract=contract,
        clean_traits=clean,
        full_traits=full,
        canonical_image_traits=full_image,
        primary_analysis=primary,
        sensitivity_analysis=sensitivity,
        source_sha256=hashes,
    )
    descriptor = next(row for row in atlas["descriptors"] if row["field"] == field)
    for cohort in ("primary_clean261", "sensitivity_full283"):
        summary = descriptor["cohorts"][cohort]["condition_summaries"][
            GROUP_ORDER[0]
        ]
        assert summary["non_null_source_unit_n"] == summary["source_unit_total"]
        assert summary["observability_fraction"] == 1.0
        assert summary["median"] == summary["q25"] == summary["q75"] == 0.0
        assert summary["iqr"] == 0.0
        assert summary["not_estimable_reason"] is None


@pytest.mark.parametrize(
    ("count_field", "total_field"),
    (
        ("hair_count", "total_hair_length_um"),
        (
            "local_hair_count_1_4mm",
            "local_total_hair_length_um_per_root_mm_1_4mm",
        ),
    ),
)
def test_atlas_rejects_empty_set_length_zero_with_positive_identity_count(
    count_field: str, total_field: str
) -> None:
    contract, clean, full, full_image, primary, sensitivity, hashes, _ = _fixture()
    assert float(clean.loc[0, count_field]) > 0.0
    clean.loc[0, total_field] = 0.0
    with pytest.raises(
        MultitraitAtlasError,
        match="zero with positive identity count is a missing endpoint-complete measurement",
    ):
        build_multitrait_atlas(
            trait_contract=contract,
            clean_traits=clean,
            full_traits=full,
            canonical_image_traits=full_image,
            primary_analysis=primary,
            sensitivity_analysis=sensitivity,
            source_sha256=hashes,
        )


def test_atlas_rejects_one_as_zero_denominator_length_support() -> None:
    contract, clean, full, full_image, primary, sensitivity, hashes, _ = _fixture()
    clean.loc[0, "hair_count"] = 0.0
    clean.loc[0, "total_hair_length_um"] = 0.0
    clean["hair_length_measurement_fraction"] = 0.5
    clean.loc[0, "hair_length_measurement_fraction"] = 1.0
    with pytest.raises(
        MultitraitAtlasError, match="zero-denominator support must be null"
    ):
        build_multitrait_atlas(
            trait_contract=contract,
            clean_traits=clean,
            full_traits=full,
            canonical_image_traits=full_image,
            primary_analysis=primary,
            sensitivity_analysis=sensitivity,
            source_sha256=hashes,
        )


def test_atlas_rejects_attachment_support_denominator_drift() -> None:
    contract, clean, full, full_image, primary, sensitivity, hashes, _ = _fixture()
    clean["attachment_axis_valid_fraction"] = 0.5
    clean.loc[0, "hair_count"] = 0.0
    clean.loc[0, "total_hair_length_um"] = 0.0
    clean.loc[0, "attachment_axis_valid_fraction"] = 1.0
    with pytest.raises(
        MultitraitAtlasError, match="zero-denominator support must be null"
    ):
        build_multitrait_atlas(
            trait_contract=contract,
            clean_traits=clean,
            full_traits=full,
            canonical_image_traits=full_image,
            primary_analysis=primary,
            sensitivity_analysis=sensitivity,
            source_sha256=hashes,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (None, "requires observed support"),
        (float("nan"), "requires observed support"),
        (-0.1, "finite within"),
        (1.1, "finite within"),
    ),
)
def test_atlas_requires_finite_bounded_attachment_support_for_positive_count(
    value: float | None,
    expected: str,
) -> None:
    contract, clean, full, full_image, primary, sensitivity, hashes, _ = _fixture()
    clean["attachment_axis_valid_fraction"] = 0.5
    clean.loc[0, "attachment_axis_valid_fraction"] = value
    with pytest.raises(MultitraitAtlasError, match=expected):
        build_multitrait_atlas(
            trait_contract=contract,
            clean_traits=clean,
            full_traits=full,
            canonical_image_traits=full_image,
            primary_analysis=primary,
            sensitivity_analysis=sensitivity,
            source_sha256=hashes,
        )


def test_atlas_counts_missing_h12_as_unobserved_instead_of_zero() -> None:
    contract, clean, full, full_image, primary, sensitivity, hashes, _ = _fixture()
    field = "local_total_hair_length_um_per_root_mm_1_4mm"
    target_condition = str(clean.loc[0, "condition_code"])
    clean.loc[0, field] = None
    atlas = build_multitrait_atlas(
        trait_contract=contract,
        clean_traits=clean,
        full_traits=full,
        canonical_image_traits=full_image,
        primary_analysis=primary,
        sensitivity_analysis=sensitivity,
        source_sha256=hashes,
    )
    descriptor = next(row for row in atlas["descriptors"] if row["field"] == field)
    cohort = descriptor["cohorts"]["primary_clean261"]
    assert cohort["non_null_source_unit_n"] == len(clean) - 1
    condition = cohort["condition_summaries"][target_condition]
    expected_total = int((clean["condition_code"].astype(str) == target_condition).sum())
    assert condition["source_unit_total"] == expected_total
    assert condition["non_null_source_unit_n"] == expected_total - 1
    assert condition["observability_fraction"] == pytest.approx(
        (expected_total - 1) / expected_total
    )
