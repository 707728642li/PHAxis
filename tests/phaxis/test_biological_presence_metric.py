from __future__ import annotations

import inspect

import numpy as np

from phaxis.evaluation_metrics import (
    biological_hair_presence_matcher_contract,
    match_biological_hair_presence,
)


def _match(predicted, truth, tolerance: float = 20.0):
    return match_biological_hair_presence(
        predicted,
        truth,
        1.0,
        tolerance,
        minimum_truth_coverage=0.25,
        minimum_prediction_coverage=0.25,
        minimum_direction_cosine=0.0,
        proximal_arc_fraction=0.25,
        resample_points=32,
    )


def test_translation_uses_physical_tolerance_not_pixel_overlap() -> None:
    truth = [np.asarray([[0.0, 0.0], [100.0, 0.0]])]
    within = [np.asarray([[0.0, 19.0], [100.0, 19.0]])]
    outside = [np.asarray([[0.0, 21.0], [100.0, 21.0]])]
    assert _match(within, truth)[0]["tp"] == 1
    assert _match(outside, truth)[0]["tp"] == 0


def test_truncated_same_hair_and_distal_error_are_not_hard_gates() -> None:
    truth = [np.asarray([[0.0, 0.0], [100.0, 0.0]])]
    # Both annotated endpoints are wrong by 30 um, and the attachment proxy is
    # outside 20 um.  The central visible trunk still provides bidirectional
    # partial support in the correct direction.
    truncated = [np.asarray([[30.0, 1.0], [70.0, 1.0]])]
    metrics, matches = _match(truncated, truth)
    assert metrics["tp"] == metrics["n_pred"] == metrics["n_gt"] == 1
    assert matches[0]["prediction_coverage"] == 1.0
    assert matches[0]["truth_coverage"] >= 0.25


def test_vertex_sampling_density_does_not_change_identity() -> None:
    predicted = [np.asarray([[20.0, 2.0], [80.0, 2.0]])]
    sparse_truth = [np.asarray([[0.0, 0.0], [100.0, 0.0]])]
    dense_truth = [
        np.column_stack((np.linspace(0.0, 100.0, 1001), np.zeros(1001)))
    ]
    sparse_metrics, sparse_matches = _match(predicted, sparse_truth)
    dense_metrics, dense_matches = _match(predicted, dense_truth)
    assert sparse_metrics == dense_metrics
    for field in (
        "prediction_coverage",
        "truth_coverage",
        "curve_f1",
        "proximal_direction_cosine",
    ):
        assert sparse_matches[0][field] == dense_matches[0][field]


def test_crossing_branch_interference_lacks_bidirectional_support() -> None:
    truth = [np.asarray([[-200.0, 0.0], [200.0, 0.0]])]
    crossing_branch = [np.asarray([[0.0, -200.0], [0.0, 200.0]])]
    metrics, matches = _match(crossing_branch, truth)
    assert metrics["tp"] == 0
    assert matches == []


def test_duplicate_predictions_cannot_double_match_one_manual_hair() -> None:
    truth = [np.asarray([[0.0, 0.0], [100.0, 0.0]])]
    duplicate = np.asarray([[0.0, 1.0], [100.0, 1.0]])
    metrics, matches = _match([duplicate, duplicate.copy()], truth)
    assert metrics["tp"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert len(matches) == 1


def test_opposite_polarity_is_rejected_without_an_endpoint_gate() -> None:
    truth = [np.asarray([[0.0, 0.0], [100.0, 0.0]])]
    reversed_prediction = [np.asarray([[100.0, 0.0], [0.0, 0.0]])]
    metrics, _matches = _match(reversed_prediction, truth)
    assert metrics["tp"] == 0


def test_matcher_contract_excludes_intensity_width_and_length_inputs() -> None:
    contract = biological_hair_presence_matcher_contract()
    assert contract["image_intensity_or_colour_is_matcher_input"] is False
    assert contract["manual_hair_width_assumed"] is False
    assert contract["distal_endpoint_is_identity_gate"] is False
    assert contract["complete_centreline_overlap_is_identity_gate"] is False
    assert contract["length_error_is_identity_gate"] is False
    parameters = set(inspect.signature(match_biological_hair_presence).parameters)
    assert not parameters.intersection(
        {"image", "intensity", "colour", "hair_width", "length_error"}
    )
