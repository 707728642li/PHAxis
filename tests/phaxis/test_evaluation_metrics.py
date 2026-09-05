from __future__ import annotations

import numpy as np
import pytest

from phaxis.evaluation_metrics import (
    evaluate_image_instances,
    match_biological_hair_presence,
    match_points,
    precision_recall_f1,
    strict_presence_matches,
)


def test_point_matching_is_one_to_one_and_tolerance_bounded():
    predicted = np.asarray([[0.0, 0.0], [0.9, 0.0], [50.0, 0.0]])
    annotated = np.asarray([[0.2, 0.0], [1.1, 0.0]])
    pred_indices, gt_indices, distances = match_points(predicted, annotated, 0.5)
    assert pred_indices.tolist() == [0, 1]
    assert gt_indices.tolist() == [0, 1]
    assert distances.tolist() == pytest.approx([0.2, 0.2])
    assert precision_recall_f1(2, 3, 2) == pytest.approx(
        {
            "tp": 2,
            "n_pred": 3,
            "n_gt": 2,
            "precision": 2 / 3,
            "recall": 1.0,
            "f1": 0.8,
        }
    )


def test_point_matching_cardinality_is_not_limited_by_fixed_cost_sentinel():
    predicted = np.asarray([[0.0, 0.0], [3_000_000.0, 0.0]])
    annotated = np.asarray([[1_500_000.0, 0.0], [4_500_000.0, 0.0]])
    pred_indices, gt_indices, distances = match_points(
        predicted, annotated, 2_000_000.0
    )
    assert len(pred_indices) == len(gt_indices) == len(distances) == 2


def test_image_evaluation_preserves_base_tip_and_length_roles():
    prediction = {
        "base": np.asarray([[1.0, 0.0], [100.0, 0.0]]),
        "tip": np.asarray([[11.0, 0.0], [110.0, 0.0]]),
        "length_um": np.asarray([20.0, 20.0]),
    }
    annotation = {
        "base": np.asarray([[0.0, 0.0]]),
        "tip": np.asarray([[10.0, 0.0]]),
        "length_um": np.asarray([18.0]),
    }
    result = evaluate_image_instances(prediction, annotation, 2.0, (5.0,))
    metrics = result["tol"][5.0]
    assert (result["n_pred"], result["n_gt"], metrics["tp"]) == (2, 1, 1)
    assert metrics["base_err_um_mean"] == pytest.approx(2.0)
    assert metrics["tip_err_um_mean"] == pytest.approx(2.0)
    assert metrics["length_mae_um"] == pytest.approx(2.0)


def _strict(
    predicted_base,
    predicted_tip,
    polyline,
    *,
    tolerance=2.0,
):
    polyline = np.asarray(polyline, dtype=np.float64)
    return strict_presence_matches(
        np.asarray([predicted_base], dtype=np.float64),
        np.asarray([predicted_tip], dtype=np.float64),
        [polyline],
        np.asarray([polyline[0]], dtype=np.float64),
        np.asarray([polyline[-1]], dtype=np.float64),
        1.0,
        tolerance,
    )


def test_strict_presence_accepts_same_hair_with_small_lateral_offset():
    assert _strict(
        [0.0, 1.0],
        [20.0, 1.0],
        [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
        tolerance=2.0,
    ) == 1


@pytest.mark.parametrize(
    ("predicted_base", "predicted_tip"),
    [
        ([20.0, 0.0], [0.0, 0.0]),  # reversed polarity
        ([10.0, 0.0], [10.0, 0.0]),  # zero-length midpoint
        ([0.0, 0.0], [4.0, 0.0]),  # length ratio > 2.5
        ([0.0, 0.0], [20.0, 20.0]),  # geometric deviation/Hausdorff guard
    ],
)
def test_strict_presence_rejects_direction_length_and_geometry_holes(
    predicted_base, predicted_tip
):
    assert _strict(
        predicted_base,
        predicted_tip,
        [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]],
        tolerance=2.0,
    ) == 0


def test_strict_presence_is_one_to_one():
    polyline = np.asarray([[0.0, 0.0], [20.0, 0.0]])
    matched = strict_presence_matches(
        np.asarray([[0.0, 0.0], [0.0, 0.5]]),
        np.asarray([[20.0, 0.0], [20.0, 0.5]]),
        [polyline],
        np.asarray([[0.0, 0.0]]),
        np.asarray([[20.0, 0.0]]),
        1.0,
        2.0,
    )
    assert matched == 1


def test_biological_presence_accepts_partial_trunk_without_endpoint_gate():
    metrics, matches = match_biological_hair_presence(
        [np.asarray([[0.0, 0.5], [8.0, 0.5]])],
        [np.asarray([[0.0, 0.0], [20.0, 0.0]])],
        1.0,
        1.0,
        minimum_truth_coverage=0.25,
        minimum_prediction_coverage=0.25,
    )
    assert metrics["tp"] == 1
    assert metrics["f1"] == pytest.approx(1.0)
    assert matches[0]["truth_coverage"] >= 0.25


def test_biological_presence_is_one_to_one_and_rejects_reversed_polarity():
    truth = [np.asarray([[0.0, 0.0], [20.0, 0.0]])]
    metrics, _matches = match_biological_hair_presence(
        [
            np.asarray([[0.0, 0.0], [20.0, 0.0]]),
            np.asarray([[0.0, 0.5], [20.0, 0.5]]),
        ],
        truth,
        1.0,
        1.0,
    )
    assert metrics["tp"] == 1
    assert metrics["n_pred"] == 2
    reversed_metrics, _matches = match_biological_hair_presence(
        [np.asarray([[20.0, 0.0], [0.0, 0.0]])],
        truth,
        1.0,
        1.0,
    )
    assert reversed_metrics["tp"] == 0


def test_biological_presence_validates_physical_metric_contract():
    line = [np.asarray([[0.0, 0.0], [1.0, 0.0]])]
    with pytest.raises(ValueError, match="finite and positive"):
        match_biological_hair_presence(line, line, 0.0, 1.0)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        match_biological_hair_presence(
            line, line, 1.0, 1.0, minimum_truth_coverage=1.1
        )


def test_metric_inputs_fail_closed_on_invalid_geometry():
    with pytest.raises(ValueError, match="finite N x 2"):
        match_points([[np.nan, 0.0]], [[0.0, 0.0]], 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        match_points([[0.0, 0.0]], [[0.0, 0.0]], -1.0)


@pytest.mark.parametrize("malformed_on_prediction_side", [False, True])
@pytest.mark.parametrize(
    "malformed",
    [np.empty((0, 2)), np.asarray([[np.nan, 0.0], [1.0, 0.0]])],
)
def test_biological_presence_validates_nonempty_side_before_empty_return(
    malformed_on_prediction_side, malformed
):
    predicted = [malformed] if malformed_on_prediction_side else []
    annotated = [] if malformed_on_prediction_side else [malformed]
    with pytest.raises(ValueError):
        match_biological_hair_presence(predicted, annotated, 1.0, 20.0)


@pytest.mark.parametrize("spacing_um", [5.0, 10.0, 19.9, 20.0, 20.1, 30.0])
@pytest.mark.parametrize("duplicate_offset_um", [0.0, 2.5, 5.0])
def test_biological_presence_duplicate_resolution_boundary(
    spacing_um, duplicate_offset_um
):
    duplicated_prediction = [
        np.asarray([[0.0, duplicate_offset_um], [100.0, duplicate_offset_um]]),
        np.asarray([[0.0, duplicate_offset_um], [100.0, duplicate_offset_um]]),
    ]
    one_truth = [np.asarray([[0.0, 0.0], [100.0, 0.0]])]
    one_metrics, _ = match_biological_hair_presence(
        duplicated_prediction, one_truth, 1.0, 20.0
    )
    assert one_metrics["tp"] == 1

    two_truth = [
        one_truth[0],
        np.asarray([[0.0, spacing_um], [100.0, spacing_um]]),
    ]
    two_metrics, _ = match_biological_hair_presence(
        duplicated_prediction, two_truth, 1.0, 20.0
    )
    both_within_tolerance = (
        duplicate_offset_um <= 20.0
        and abs(spacing_um - duplicate_offset_um) <= 20.0
    )
    assert two_metrics["tp"] == (2 if both_within_tolerance else 1)
