from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from phaxis.contracts import ContractError
from phaxis.phenotypes import (
    associate_endpoint_complete_lengths,
    attach_hairs_to_axis,
    load_axis_geometry,
    recompose_hair_phenotypes,
)


def _axis():
    return {
        "path_xy": np.asarray([[10.0, 0.0], [10.0, 10.0], [10.0, 20.0]]),
        "distance_from_tip_px": np.asarray([0.0, 10.0, 20.0]),
        "radius_px": np.asarray([3.0, 3.0, 3.0]),
    }


def test_attachment_projects_to_axis_without_deleting_invalid_identity():
    hairs = [
        {"points_xy": [[13.0, 0.0], [20.0, 0.0]]},
        {"points_xy": [[100.0, 20.0], [110.0, 20.0]]},
    ]
    summary = attach_hairs_to_axis(
        hairs, _axis(), um_per_px=2.0, maximum_boundary_error_um=5.0
    )
    assert len(hairs) == 2
    assert hairs[0]["root_attachment_valid"] is True
    assert hairs[1]["root_attachment_valid"] is False
    assert summary["valid_fraction"] == pytest.approx(0.5)
    assert summary["first_hair_distance_from_tip_um"] == pytest.approx(0.0)
    assert summary["hair_zone_length_um"] == pytest.approx(0.0)


def test_empty_attachment_has_undefined_support_and_no_spatial_extent():
    assert attach_hairs_to_axis([], _axis(), um_per_px=2.0) == {
        "valid_fraction": None,
        "first_hair_distance_from_tip_um": None,
        "hair_zone_length_um": None,
    }


def test_length_association_is_one_to_one_and_does_not_union_identities():
    identities = [
        {"source_instance_id": "S1", "points_xy": [[10.0, 10.0], [20.0, 10.0]]},
        {"source_instance_id": "S2", "points_xy": [[10.5, 10.0], [20.5, 10.0]]},
    ]
    candidate = {
        "points_xy": [[10.1, 10.0], [30.1, 10.0]],
        "complete_length_measurement_eligible": True,
    }
    matched, audit = associate_endpoint_complete_lengths(
        identities, [candidate], um_per_px=2.0, maximum_base_distance_um=20.0
    )
    assert len(matched) == 1
    assert audit["matched_length_identities"] == 1
    assert sum(hair.get("complete_length_measurement_eligible", False) for hair in identities) == 1
    assert matched[0]["identity_source_instance_id"] in {"S1", "S2"}
    assert candidate == {
        "points_xy": [[10.1, 10.0], [30.1, 10.0]],
        "complete_length_measurement_eligible": True,
    }


def test_recomposition_preserves_root_traits_and_marks_partial_length_scope(phaxis_case):
    prediction, _, _ = phaxis_case
    before_root = deepcopy(prediction["phenotypes"]["identity_tier"])
    prediction["identity_hairs"] = [
        {"points_xy": [[1.0, 1.0], [2.0, 1.0]]},
        {"points_xy": [[2.0, 2.0], [3.0, 2.0]]},
    ]
    prediction["length_hairs"] = [
        {"points_xy": [[0.0, 0.0], [3.0, 4.0]]}
    ]
    recompose_hair_phenotypes(
        prediction,
        attachment={
            "valid_fraction": 0.5,
            "first_hair_distance_from_tip_um": 20.0,
            "hair_zone_length_um": 100.0,
        },
        um_per_px=2.0,
    )
    tier = prediction["phenotypes"]["identity_tier"]
    for name in ("root_area_um2", "main_root_length_um", "main_root_width_um"):
        assert tier[name] == before_root[name]
    assert tier["hair_count"] == 2.0
    assert tier["mean_hair_length_um"] == pytest.approx(10.0)
    assert tier["hair_length_measurement_fraction"] == pytest.approx(0.5)
    assert tier["total_hair_length_is_partial"] is True


def test_recomposition_distinguishes_zero_identities_from_missing_lengths(phaxis_case):
    prediction, _, _ = phaxis_case
    prediction["identity_hairs"] = [
        {"points_xy": [[1.0, 1.0], [2.0, 1.0]]},
        {"points_xy": [[2.0, 2.0], [3.0, 2.0]]},
    ]
    prediction["length_hairs"] = []
    recompose_hair_phenotypes(
        prediction,
        attachment={
            "valid_fraction": 0.5,
            "first_hair_distance_from_tip_um": 20.0,
            "hair_zone_length_um": 100.0,
        },
        um_per_px=2.0,
    )
    tier = prediction["phenotypes"]["identity_tier"]
    assert tier["hair_count"] == 2.0
    assert tier["mean_hair_length_um"] is None
    assert tier["median_hair_length_um"] is None
    assert tier["total_hair_length_um"] is None
    assert tier["hair_length_measurement_hair_count"] == 0.0
    assert tier["hair_length_measurement_fraction"] == 0.0
    assert tier["total_hair_length_is_partial"] is True

    prediction["identity_hairs"] = []
    recompose_hair_phenotypes(
        prediction,
        attachment={
            "valid_fraction": None,
            "first_hair_distance_from_tip_um": None,
            "hair_zone_length_um": None,
        },
        um_per_px=2.0,
    )
    tier = prediction["phenotypes"]["identity_tier"]
    assert tier["hair_count"] == 0.0
    assert tier["mean_hair_length_um"] is None
    assert tier["median_hair_length_um"] is None
    assert tier["total_hair_length_um"] == 0.0
    assert tier["hair_length_measurement_fraction"] is None
    assert tier["attachment_axis_valid_fraction"] is None
    assert tier["total_hair_length_is_partial"] is False


def test_recomposition_rejects_attachment_support_denominator_drift(phaxis_case):
    prediction, _, _ = phaxis_case
    prediction["identity_hairs"] = []
    prediction["length_hairs"] = []
    with pytest.raises(ContractError, match="zero hair identities require null"):
        recompose_hair_phenotypes(
            prediction,
            attachment={
                "valid_fraction": 1.0,
                "first_hair_distance_from_tip_um": None,
                "hair_zone_length_um": None,
            },
            um_per_px=2.0,
        )

    prediction["identity_hairs"] = [
        {"points_xy": [[1.0, 1.0], [2.0, 1.0]]},
    ]
    with pytest.raises(ContractError, match="requires numeric"):
        recompose_hair_phenotypes(
            prediction,
            attachment={
                "valid_fraction": None,
                "first_hair_distance_from_tip_um": None,
                "hair_zone_length_um": None,
            },
            um_per_px=2.0,
        )


def test_recomposition_rejects_degenerate_endpoint_complete_curve(phaxis_case):
    prediction, _, _ = phaxis_case
    prediction["identity_hairs"] = [
        {"points_xy": [[1.0, 1.0], [2.0, 1.0]]},
    ]
    prediction["length_hairs"] = [
        {"points_xy": [[1.0, 1.0], [1.0, 1.0]]},
    ]
    with pytest.raises(ContractError, match="must have positive length"):
        recompose_hair_phenotypes(
            prediction,
            attachment={
                "valid_fraction": 1.0,
                "first_hair_distance_from_tip_um": 0.0,
                "hair_zone_length_um": 0.0,
            },
            um_per_px=2.0,
        )


def test_axis_geometry_rejects_source_identity_mismatch(tmp_path: Path):
    path = tmp_path / "axis.npz"
    np.savez_compressed(
        path,
        path_xy=np.asarray([[0.0, 0.0], [0.0, 1.0]]),
        distance_from_tip_px=np.asarray([0.0, 1.0]),
        radius_px=np.asarray([1.0, 1.0]),
        source_image_sha256=np.asarray("a" * 64),
    )
    with pytest.raises(ContractError, match="axis/source hash mismatch"):
        load_axis_geometry(path, expected_image_sha256="b" * 64)
