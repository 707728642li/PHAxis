from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from phaxis.io import sha256_file, sha256_json
from phaxis.root_trait_assurance import (
    ROOT_TRAIT_ACCURACY_EVIDENCE_ROLE,
    ROOT_TRAIT_ASSURANCE_TOKENS,
    ROOT_TRAIT_FAMILY_BY_FIELD,
    ROOT_TRAIT_PREDICTION_DEFINITION,
    ROOT_TRAIT_REFERENCE_DEFINITION,
    RootTraitAssuranceError,
    build_root_trait_assurance,
    validate_root_trait_assurance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json"
REFERENCE_AUTHORITY = sha256_json({"reference": "canonical-qcdevelopment44"})
PREDICTION_AUTHORITY = sha256_json({"prediction": "sealed-hybrid-max-qcdevelopment44"})


def _fixture(*, constant_trait: str | None = None, partial_trait: str | None = None):
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    source_units = [f"qc-{index:02d}" for index in range(44)]
    pairs = []
    for trait_index, trait in enumerate(contract["primary_root_traits"]):
        field = str(trait["field"])
        for index, source_unit in enumerate(source_units):
            if field == constant_trait:
                observed = predicted = 5.0
            else:
                observed = 5.0 + trait_index * 2.0 + index * (0.15 + trait_index / 200)
                predicted = observed * (0.99 + trait_index / 5000) + (index % 4 - 1.5) * 0.02
            prediction_observable = not (field == partial_trait and index == 0)
            pairs.append(
                {
                    "pair_type": "root_trait",
                    "source_unit": source_unit,
                    "pair_id": f"{source_unit}:{field}",
                    "trait_id": str(trait["id"]),
                    "trait_key": field,
                    "trait_family": ROOT_TRAIT_FAMILY_BY_FIELD[field],
                    "unit": str(trait["unit"]),
                    "observed": observed,
                    "predicted": predicted if prediction_observable else None,
                    "reference_observable": True,
                    "prediction_observable": prediction_observable,
                    "agreement_eligible": prediction_observable,
                    "ineligibility_reason": "" if prediction_observable else "prediction_not_observable",
                    "reference_definition": ROOT_TRAIT_REFERENCE_DEFINITION,
                    "prediction_definition": ROOT_TRAIT_PREDICTION_DEFINITION,
                    "source_image_sha256": sha256_json(["qc-image", index]),
                }
            )
    payload = build_root_trait_assurance(
        pairs=pairs,
        trait_contract=contract,
        source_units=source_units,
        trait_contract_file_sha256=sha256_file(CONTRACT_PATH),
        reference_authority_sha256=REFERENCE_AUTHORITY,
        prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
        bootstrap_repetitions=100,
        bootstrap_seed=20_260_828,
    )
    return contract, source_units, pairs, payload


def _validate(contract, source_units, pairs, payload):
    return validate_root_trait_assurance(
        payload,
        pairs=pairs,
        trait_contract=contract,
        source_units=source_units,
        trait_contract_file_sha256=sha256_file(CONTRACT_PATH),
        reference_authority_sha256=REFERENCE_AUTHORITY,
        prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
    )


def test_seals_all_19_traits_six_families_truth_support_and_statistics() -> None:
    contract, source_units, pairs, payload = _fixture()
    assert _validate(contract, source_units, pairs, payload) == payload
    assert payload["trait_count"] == 19
    assert payload["family_count"] == 6
    assert payload["evidence_role"] == ROOT_TRAIT_ACCURACY_EVIDENCE_ROLE
    assert payload["provider_equivalence_used_as_accuracy"] is False
    assert payload["truth_reference"] == ROOT_TRAIT_REFERENCE_DEFINITION
    assert [row["trait_id"] for row in payload["trait_rows"]] == [
        f"R{index:02d}" for index in range(1, 20)
    ]
    for row in payload["trait_rows"]:
        assert row["total_source_units"] == row["eligible_source_units"] == 44
        assert row["observability_fraction"] == 1.0
        assert row["support_status"] == "fully_observable"
        assert row["mae"] >= 0
        assert row["bias_ci_low"] <= row["bias"] <= row["bias_ci_high"]
        assert row["ccc"] is not None
        assert row["row_identity_sha256"] == sha256_json(
            {key: value for key, value in row.items() if key != "row_identity_sha256"}
        )


def test_partial_observability_retains_denominator_and_explicit_eligible_n() -> None:
    field = "root_width_axial_slope_um_per_mm"
    contract, source_units, pairs, payload = _fixture(partial_trait=field)
    assert _validate(contract, source_units, pairs, payload) == payload
    row = next(item for item in payload["trait_rows"] if item["trait_key"] == field)
    assert row["total_source_units"] == 44
    assert row["eligible_source_units"] == 43
    assert row["prediction_observable_n"] == 43
    assert row["observability_fraction"] == pytest.approx(43 / 44)
    assert row["support_status"] == "partially_observable"


def test_constant_trait_uses_native_unit_mae_instead_of_fabricated_ccc() -> None:
    field = "root_centerline_curvature_median_rad_per_mm"
    contract, source_units, pairs, payload = _fixture(constant_trait=field)
    assert _validate(contract, source_units, pairs, payload) == payload
    row = next(item for item in payload["trait_rows"] if item["trait_key"] == field)
    assert row["ccc"] is None
    assert row["ccc_status"] == "not_estimable_zero_total_variance"
    assert row["agreement_statistic"] == "mae_native_unit"
    assert row["agreement_value"] == row["mae"] == 0.0
    assert row["agreement_higher_is_better"] is False


@pytest.mark.parametrize("mutation", ["missing_trait", "denominator_drift"])
def test_rejects_missing_trait_or_source_unit_denominator_drift(mutation: str) -> None:
    contract, source_units, pairs, _payload = _fixture()
    if mutation == "missing_trait":
        pairs = [row for row in pairs if row["trait_id"] != "R19"]
    else:
        pairs = pairs[:-1]
    with pytest.raises(RootTraitAssuranceError, match="denominator|missing"):
        build_root_trait_assurance(
            pairs=pairs,
            trait_contract=contract,
            source_units=source_units,
            trait_contract_file_sha256=sha256_file(CONTRACT_PATH),
            reference_authority_sha256=REFERENCE_AUTHORITY,
            prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
            bootstrap_repetitions=100,
        )


def test_rejects_pair_value_or_embedded_summary_hash_drift() -> None:
    contract, source_units, pairs, payload = _fixture()
    changed_pairs = deepcopy(pairs)
    changed_pairs[0]["predicted"] += 1.0
    with pytest.raises(RootTraitAssuranceError, match="identity drift"):
        _validate(contract, source_units, changed_pairs, payload)
    changed_payload = deepcopy(payload)
    changed_payload["trait_rows"][0]["eligible_source_units"] = 43
    changed_payload["root_trait_assurance_identity_sha256"] = sha256_json(
        {
            key: value
            for key, value in changed_payload.items()
            if key != "root_trait_assurance_identity_sha256"
        }
    )
    with pytest.raises(RootTraitAssuranceError, match="denominator, or identity drift"):
        _validate(contract, source_units, pairs, changed_payload)


def test_rejects_provider_equivalence_masquerading_as_accuracy() -> None:
    contract, source_units, pairs, payload = _fixture()
    changed = deepcopy(payload)
    changed["evidence_role"] = "exact_portable_provider_equivalence"
    changed["provider_equivalence_used_as_accuracy"] = True
    changed["root_trait_assurance_identity_sha256"] = sha256_json(
        {
            key: value
            for key, value in changed.items()
            if key != "root_trait_assurance_identity_sha256"
        }
    )
    with pytest.raises(RootTraitAssuranceError, match="cannot masquerade"):
        _validate(contract, source_units, pairs, changed)


def test_stable_manuscript_token_registry_is_unique_and_root_trait_scoped() -> None:
    assert len(ROOT_TRAIT_ASSURANCE_TOKENS) == len(set(ROOT_TRAIT_ASSURANCE_TOKENS)) == 9
    assert all(token.startswith("FINAL_ROOT_TRAIT_") for token in ROOT_TRAIT_ASSURANCE_TOKENS)
