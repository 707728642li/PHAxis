from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from phaxis.axial_profiles import export_distal_axis_profiles
from phaxis.contracts import ContractError
from phaxis.io import sha256_json


PROPOSAL_BINDING = {
    "model_contract_proposal_sha256": "1" * 64,
    "model_contract_proposal_identity_sha256": "2" * 64,
}
PUBLIC_IDENTITY = {
    "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
    "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _contract(path: Path) -> Path:
    payload = {
        "schema_version": "PHAxis-distal-axis-profile-contract-1.0.0",
        "bins_um": [[0, 1000], [1000, 2000]],
        "root_cap_region_output": False,
        "stageb_two_point_vector_used_as_length": False,
        "blind_images_used": 0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    image_sha = "a" * 64
    traits = tmp_path / "traits.csv"
    hairs = tmp_path / "hair_instances.csv"
    _write_csv(
        traits,
        [
            {
                "task_id": "T1",
                "source_image_sha256": image_sha,
                "experiment_key": "E",
                "condition_code": "C",
                "study_role": "primary",
                "developmental_day": "15",
                "genotype_or_construct": "WT",
                "temperature_c": "22",
                "formal_statistics_eligible": "True",
                "visible_root_axis_length_um": "2000",
                "attachment_axis_valid_fraction": "0.75",
                "distal_window_1_4mm_eligible": "False",
                "local_hair_count_1_4mm": "",
                "local_hair_density_per_mm_1_4mm": "",
                "local_mean_hair_length_um_1_4mm": "",
                "local_median_hair_length_um_1_4mm": "",
                "local_total_hair_length_um_per_root_mm_1_4mm": "",
            }
        ],
    )
    rows = []
    for hair_id, axis, eligible, length in (
        ("h1", "0", "True", "100"),
        ("h2", "999.999", "False", ""),
        ("h3", "1000", "True", "200"),
        ("h4", "", "False", ""),
    ):
        rows.append(
            {
                "task_id": "T1",
                "source_image_sha256": image_sha,
                "hair_id": hair_id,
                "attachment_valid_within_40um": str(bool(axis)),
                "attachment_distance_from_distal_point_um": axis,
                "complete_length_measurement_eligible": eligible,
                "length_um": length,
            }
        )
    _write_csv(hairs, rows)
    (traits.parent / "summary.json").write_text(
        json.dumps({**PROPOSAL_BINDING, **PUBLIC_IDENTITY}), encoding="utf-8"
    )
    return traits, hairs, _contract(tmp_path / "contract.json")


def test_fixed_bins_and_conditional_length_semantics(tmp_path: Path) -> None:
    traits, hairs, contract = _inputs(tmp_path)
    summary = export_distal_axis_profiles(
        traits_csv=traits,
        hair_instances_csv=hairs,
        contract_json=contract,
        output=tmp_path / "out",
        model_contract_proposal=PROPOSAL_BINDING,
        model_contract_public_identity=PUBLIC_IDENTITY,
    )
    assert summary["rows"] == 2
    with (tmp_path / "out" / "distal_axis_profiles.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["attached_identity_count"] == "2"
    assert rows[0]["endpoint_complete_length_count"] == "1"
    assert rows[0]["endpoint_complete_length_support_fraction"] == "0.5"
    assert rows[0]["conditional_mean_hair_length_um"] == "100.0"
    assert rows[1]["attached_identity_count"] == "1"
    assert rows[1]["conditional_median_hair_length_um"] == "200.0"
    assert rows[0]["stageb_two_point_vector_used_as_length"] == "False"


def test_profile_total_is_null_for_positive_identity_without_length_and_zero_for_none(
    tmp_path: Path,
) -> None:
    traits, hairs, contract = _inputs(tmp_path)
    with traits.open("r", encoding="utf-8", newline="") as handle:
        trait_rows = list(csv.DictReader(handle))
    trait_rows[0].update(
        {
            "visible_root_axis_length_um": "5000",
            "distal_window_1_4mm_eligible": "True",
            "local_hair_count_1_4mm": "1",
            "local_hair_density_per_mm_1_4mm": str(1.0 / 3.0),
            "local_mean_hair_length_um_1_4mm": "",
            "local_median_hair_length_um_1_4mm": "",
            "local_total_hair_length_um_per_root_mm_1_4mm": "",
        }
    )
    _write_csv(traits, trait_rows)

    with hairs.open("r", encoding="utf-8", newline="") as handle:
        hair_rows = list(csv.DictReader(handle))
    hair_rows[2]["complete_length_measurement_eligible"] = "False"
    hair_rows[2]["length_um"] = ""
    _write_csv(hairs, hair_rows)

    contract_payload = json.loads(contract.read_text(encoding="utf-8"))
    contract_payload["bins_um"].append([2000, 3000])
    contract.write_text(json.dumps(contract_payload), encoding="utf-8")

    export_distal_axis_profiles(
        traits_csv=traits,
        hair_instances_csv=hairs,
        contract_json=contract,
        output=tmp_path / "null-total",
        model_contract_proposal=PROPOSAL_BINDING,
        model_contract_public_identity=PUBLIC_IDENTITY,
    )
    with (tmp_path / "null-total" / "distal_axis_profiles.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["attached_identity_count"] == "1"
    assert rows[1]["endpoint_complete_length_count"] == "0"
    assert rows[1]["endpoint_complete_length_support_fraction"] == "0.0"
    assert rows[1]["measured_total_hair_length_um"] == ""
    assert rows[1]["measured_total_hair_length_per_root_mm"] == ""
    assert rows[2]["attached_identity_count"] == "0"
    assert rows[2]["endpoint_complete_length_support_fraction"] == ""
    assert rows[2]["measured_total_hair_length_um"] == "0.0"
    assert rows[2]["measured_total_hair_length_per_root_mm"] == "0.0"


def test_profile_attachment_support_is_null_for_zero_identities_and_fail_closed(
    tmp_path: Path,
) -> None:
    traits, hairs, contract = _inputs(tmp_path)
    with traits.open("r", encoding="utf-8", newline="") as handle:
        trait_rows = list(csv.DictReader(handle))
    trait_rows[0]["attachment_axis_valid_fraction"] = ""
    _write_csv(traits, trait_rows)

    with hairs.open("r", encoding="utf-8", newline="") as handle:
        hair_reader = csv.DictReader(handle)
        hair_fields = list(hair_reader.fieldnames or ())
    with hairs.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=hair_fields).writeheader()

    export_distal_axis_profiles(
        traits_csv=traits,
        hair_instances_csv=hairs,
        contract_json=contract,
        output=tmp_path / "zero-identities",
        model_contract_proposal=PROPOSAL_BINDING,
        model_contract_public_identity=PUBLIC_IDENTITY,
    )
    with (tmp_path / "zero-identities" / "distal_axis_profiles.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["attachment_axis_valid_fraction"] == "" for row in rows)
    assert all(row["attached_identity_count"] == "0" for row in rows)

    trait_rows[0]["attachment_axis_valid_fraction"] = "1.0"
    _write_csv(traits, trait_rows)
    with pytest.raises(
        ContractError, match="zero hair identities require null attachment fraction"
    ):
        export_distal_axis_profiles(
            traits_csv=traits,
            hair_instances_csv=hairs,
            contract_json=contract,
            output=tmp_path / "legacy-one",
            model_contract_proposal=PROPOSAL_BINDING,
            model_contract_public_identity=PUBLIC_IDENTITY,
        )


def test_profile_requires_attachment_support_for_positive_identity_count(
    tmp_path: Path,
) -> None:
    traits, hairs, contract = _inputs(tmp_path)
    with traits.open("r", encoding="utf-8", newline="") as handle:
        trait_rows = list(csv.DictReader(handle))
    trait_rows[0]["attachment_axis_valid_fraction"] = ""
    _write_csv(traits, trait_rows)
    with pytest.raises(
        ContractError, match="positive hair identity count requires attachment fraction"
    ):
        export_distal_axis_profiles(
            traits_csv=traits,
            hair_instances_csv=hairs,
            contract_json=contract,
            output=tmp_path / "missing-positive-support",
            model_contract_proposal=PROPOSAL_BINDING,
            model_contract_public_identity=PUBLIC_IDENTITY,
        )


def test_profile_summary_natively_inherits_sealed_traits_proposal_binding(
    tmp_path: Path,
) -> None:
    traits, hairs, contract = _inputs(tmp_path)
    binding = {
        "model_contract_proposal_sha256": "b" * 64,
        "model_contract_proposal_identity_sha256": "c" * 64,
    }
    public_identity = {
        "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
        "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
    }
    (traits.parent / "summary.json").write_text(
        json.dumps({**binding, **public_identity}), encoding="utf-8"
    )
    summary = export_distal_axis_profiles(
        traits_csv=traits,
        hair_instances_csv=hairs,
        contract_json=contract,
        output=tmp_path / "bound",
        model_contract_proposal=binding,
        model_contract_public_identity=public_identity,
    )
    assert all(
        summary[field] == value
        for field, value in {**binding, **public_identity}.items()
    )
    unsigned = dict(summary)
    identity = unsigned.pop("export_identity_sha256")
    assert identity == sha256_json(unsigned)


def test_profile_can_validate_an_explicit_upstream_traits_summary(
    tmp_path: Path,
) -> None:
    traits, hairs, contract = _inputs(tmp_path)
    binding = {
        "model_contract_proposal_sha256": "d" * 64,
        "model_contract_proposal_identity_sha256": "e" * 64,
    }
    public_identity = {
        "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
        "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
    }
    upstream = tmp_path / "cohort-root" / "traits-summary.json"
    upstream.parent.mkdir()
    upstream.write_text(
        json.dumps({**binding, **public_identity}), encoding="utf-8"
    )
    summary = export_distal_axis_profiles(
        traits_csv=traits,
        hair_instances_csv=hairs,
        contract_json=contract,
        output=tmp_path / "bound-explicit",
        model_contract_proposal=binding,
        model_contract_public_identity=public_identity,
        traits_summary_json=upstream,
    )
    assert all(
        summary[field] == value
        for field, value in {**binding, **public_identity}.items()
    )


def test_profile_rejects_partial_or_mismatched_public_identity(tmp_path: Path) -> None:
    traits, hairs, contract = _inputs(tmp_path)
    binding = {
        "model_contract_proposal_sha256": "b" * 64,
        "model_contract_proposal_identity_sha256": "c" * 64,
    }
    (traits.parent / "summary.json").write_text(
        json.dumps(binding), encoding="utf-8"
    )
    with pytest.raises(ContractError):
        export_distal_axis_profiles(
            traits_csv=traits,
            hair_instances_csv=hairs,
            contract_json=contract,
            output=tmp_path / "partial",
            model_contract_proposal=binding,
            model_contract_public_identity=PUBLIC_IDENTITY,
        )


def test_short_root_marks_later_bin_ineligible(tmp_path: Path) -> None:
    traits, hairs, contract = _inputs(tmp_path)
    rows = list(csv.DictReader(traits.open(encoding="utf-8", newline="")))
    rows[0]["visible_root_axis_length_um"] = "1500"
    _write_csv(traits, rows)
    export_distal_axis_profiles(
        traits_csv=traits,
        hair_instances_csv=hairs,
        contract_json=contract,
        output=tmp_path / "out",
        model_contract_proposal=PROPOSAL_BINDING,
        model_contract_public_identity=PUBLIC_IDENTITY,
    )
    output_rows = list(
        csv.DictReader(
            (tmp_path / "out" / "distal_axis_profiles.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    assert output_rows[1]["bin_eligible"] == "False"
    assert output_rows[1]["attached_identity_count"] == ""
    assert output_rows[1]["bin_ineligibility_reason"] == "visible_root_axis_shorter_than_bin_end"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attachment_valid_within_40um", "False"),
        ("complete_length_measurement_eligible", "False"),
    ],
)
def test_hair_flag_value_drift_fails_closed(
    tmp_path: Path, field: str, value: str
) -> None:
    traits, hairs, contract = _inputs(tmp_path)
    rows = list(csv.DictReader(hairs.open(encoding="utf-8", newline="")))
    rows[0][field] = value
    _write_csv(hairs, rows)
    with pytest.raises(ContractError):
        export_distal_axis_profiles(
            traits_csv=traits,
            hair_instances_csv=hairs,
            contract_json=contract,
            output=tmp_path / "out",
            model_contract_proposal=PROPOSAL_BINDING,
            model_contract_public_identity=PUBLIC_IDENTITY,
        )


def test_blind_tainted_contract_fails_closed(tmp_path: Path) -> None:
    traits, hairs, contract = _inputs(tmp_path)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    payload["blind_images_used"] = 1
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError):
        export_distal_axis_profiles(
            traits_csv=traits,
            hair_instances_csv=hairs,
            contract_json=contract,
            output=tmp_path / "out",
            model_contract_proposal=PROPOSAL_BINDING,
            model_contract_public_identity=PUBLIC_IDENTITY,
        )
