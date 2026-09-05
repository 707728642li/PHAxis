from __future__ import annotations

import csv

import pytest

from phaxis.contracts import ContractError
from phaxis.fusion import fuse_hybrid_root_with_stageb_hairs
from phaxis.io import atomic_write_json, sha256_json
from phaxis.traits import (
    IMAGE_TRAIT_FIELDS,
    ROOT_TRAIT_FIELDS,
    _one_prediction,
    export_traits,
)


def _direct_trait_case(identity_axes, length_identity_indices=()):
    identities = [
        {
            "source_instance_id": f"PHSB-{index + 1:04d}",
            "points_xy": [[10.0, float(index)], [20.0, float(index)]],
            "root_attachment_valid": True,
            "root_axis_distance_from_tip_um": float(axis_um),
        }
        for index, axis_um in enumerate(identity_axes)
    ]
    lengths = [
        {
            "source_instance_id": f"HML-{order + 1:04d}",
            "identity_source_instance_id": identities[index]["source_instance_id"],
            "points_xy": [[10.0, float(index)], [20.0, float(index)]],
            "complete_length_measurement_eligible": True,
        }
        for order, index in enumerate(length_identity_indices)
    ]
    prediction = {
        "task_id": "T-DIRECT",
        "source_image_sha256": "a" * 64,
        "root_cap_region_output": False,
        "formal_phenotype_eligible": True,
        "scale": {"um_per_px": 1.0, "fail_closed": False},
        "detailed_root_statistics": {
            **{field: 1.0 for field in ROOT_TRAIT_FIELDS},
            "visible_root_axis_length_um": 5000.0,
        },
        "identity_hairs": identities,
        "length_hairs": lengths,
        "phaxis": {
            "blind_images_used": 0,
            "formal_stageb_identity_count": len(identities),
            "hair_identity_count_expert": "PHAxis-StageB-train399-unit-test",
            "length_identity_association": {
                "matched_length_identities": len(lengths)
            },
        },
        "blind_images_used": 0,
    }
    metadata = {
        "task_id": "T-DIRECT",
        "image_sha256": "a" * 64,
        "um_per_px": "1.0",
        "experiment_key": "synthetic",
        "condition_code": "synthetic",
        "study_role": "unit_test",
        "developmental_day": "",
        "genotype_or_construct": "",
        "temperature_c": "",
        "qc_disposition": "eligible",
    }
    return prediction, metadata


def test_trait_export_keeps_stageb_identity_but_uses_hybrid_length_curve(
    phaxis_case, tmp_path
):
    hybrid, stageb, artifact_root = phaxis_case
    root_statistics = {field: 1.0 for field in ROOT_TRAIT_FIELDS}
    root_statistics.update(
        {
            "visible_root_axis_length_um": 5000.0,
            "median_root_width_um": 12.0,
        }
    )
    hybrid["detailed_root_statistics"] = root_statistics
    proposal_binding = {
        "model_contract_proposal_sha256": "b" * 64,
        "model_contract_proposal_identity_sha256": "c" * 64,
    }
    public_identity = {
        "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
        "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
    }
    stageb.update({**proposal_binding, **public_identity})
    stageb.pop("detection_identity_sha256", None)
    stageb["detection_identity_sha256"] = sha256_json(stageb)
    fused = fuse_hybrid_root_with_stageb_hairs(
        hybrid,
        stageb,
        hybrid_artifact_root=artifact_root,
        model_contract_proposal=proposal_binding,
        model_contract_public_identity=public_identity,
    )
    # Final train399 candidates carry their selected expert identity inside the
    # fused prediction; trait export must never fall back to the legacy constant.
    selected_expert = "PHAxis-StageB-train399-five-seed-unit-test"
    fused["phaxis"]["hair_identity_count_expert"] = selected_expert
    fused["phaxis"].update(proposal_binding)
    fused["phaxis"]["model_bundle_id"] = (
        "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC"
    )
    fused["phaxis"]["root_expert"] = "PHAxis-root-provider-SYNTHETIC"
    prediction_root = artifact_root / "predictions"
    atomic_write_json(prediction_root / "T1.json", fused)
    metadata = artifact_root / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task_id",
                "image_sha256",
                "um_per_px",
                "experiment_key",
                "condition_code",
                "study_role",
                "developmental_day",
                "genotype_or_construct",
                "temperature_c",
                "qc_disposition",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "T1",
                "image_sha256": "a" * 64,
                "um_per_px": "1.0",
                "experiment_key": "synthetic",
                "condition_code": "synthetic",
                "study_role": "unit_test",
                "developmental_day": "",
                "genotype_or_construct": "",
                "temperature_c": "",
                "qc_disposition": "eligible",
            }
        )
    output = tmp_path / "traits"
    summary = export_traits(
        prediction_root=prediction_root,
        metadata_csv=metadata,
        output=output,
        model_contract_proposal=proposal_binding,
        model_contract_public_identity=public_identity,
    )
    with (output / "traits.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    with (output / "image_traits.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        canonical = next(csv.DictReader(handle))
    with (output / "hair_instances.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        hair_instance = next(csv.DictReader(handle))
    assert summary["hair_identities"] == 2
    assert summary["endpoint_complete_length_identities"] == 1
    assert row["hair_count"] == "2"
    assert float(row["mean_hair_length_um"]) == 14.0
    assert float(row["median_hair_length_um"]) == 14.0
    assert float(row["hair_length_measurement_fraction"]) == 0.5
    # The Stage-B two-point identity vector is 9 um; it must never be the
    # formal length when a 14-um Hybrid endpoint-complete curve is associated.
    assert float(row["mean_hair_length_um"]) != 9.0
    assert tuple(canonical) == IMAGE_TRAIT_FIELDS
    assert len(canonical) == 82
    assert canonical["root_cap_region_output"] == "False"
    assert canonical["hair_identity_count_expert_id"] == selected_expert
    assert hair_instance["identity_expert"] == selected_expert
    assert summary["hair_identity_count_expert"] == selected_expert
    assert summary["model_bundle_id"] == "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC"
    assert summary["root_expert_id"] == "PHAxis-root-provider-SYNTHETIC"
    assert all(summary[field] == value for field, value in proposal_binding.items())


def test_trait_null_semantics_distinguish_zero_identity_from_missing_length():
    positive_prediction, metadata = _direct_trait_case([1500.0, 2500.0])
    traits, _root, _hairs = _one_prediction(positive_prediction, metadata)
    assert traits["hair_count"] == 2
    assert traits["hair_length_measurement_hair_count"] == 0
    assert traits["hair_length_measurement_fraction"] == 0.0
    assert traits["mean_hair_length_um"] is None
    assert traits["median_hair_length_um"] is None
    assert traits["total_hair_length_um"] is None
    assert traits["local_hair_count_1_4mm"] == 2
    assert traits["local_mean_hair_length_um_1_4mm"] is None
    assert traits["local_median_hair_length_um_1_4mm"] is None
    assert traits["local_total_hair_length_um_per_root_mm_1_4mm"] is None
    assert traits["total_hair_length_is_partial"] is True

    zero_prediction, metadata = _direct_trait_case([])
    traits, _root, _hairs = _one_prediction(zero_prediction, metadata)
    assert traits["hair_count"] == 0
    assert traits["hair_length_measurement_hair_count"] == 0
    assert traits["hair_length_measurement_fraction"] is None
    assert traits["attachment_axis_valid_fraction"] is None
    assert traits["mean_hair_length_um"] is None
    assert traits["median_hair_length_um"] is None
    assert traits["total_hair_length_um"] == 0.0
    assert traits["local_hair_count_1_4mm"] == 0
    assert traits["local_total_hair_length_um_per_root_mm_1_4mm"] == 0.0
    assert traits["total_hair_length_is_partial"] is False


def test_trait_partial_total_requires_at_least_one_positive_length_curve():
    prediction, metadata = _direct_trait_case([1500.0, 2500.0], [0])
    traits, _root, _hairs = _one_prediction(prediction, metadata)
    assert traits["hair_length_measurement_fraction"] == 0.5
    assert traits["total_hair_length_um"] == 10.0
    assert traits["local_total_hair_length_um_per_root_mm_1_4mm"] == pytest.approx(
        10.0 / 3.0
    )
    assert traits["total_hair_length_is_partial"] is True

    prediction["length_hairs"][0]["points_xy"] = [[10.0, 0.0], [10.0, 0.0]]
    with pytest.raises(ContractError, match="must have positive length"):
        _one_prediction(prediction, metadata)
