"""Self-contained instance and centreline metrics used by PHAxis evaluation.

The primary root-hair endpoint is tolerant one-to-one *biological presence*
against the annotated single-trunk centreline.  It requires bidirectional
partial curve support and a weak proximal-direction guard, but it does not
require the distal endpoints, the full curves, or a fictitious hair width to
coincide.  Attachment/base matching and ``strict_presence_matches`` are
secondary geometric diagnostics.  All coordinates passed to these functions
use one explicit unit; formal PHAxis evaluation passes micrometres and
therefore uses ``units_per_coordinate=1``.

This module deliberately has no model, dataset, or external-checkout imports.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


INFEASIBLE_COST = 1_000_000.0

PRIMARY_HAIR_PRESENCE_TOLERANCE_UM = 20.0
PRIMARY_HAIR_PRESENCE_MINIMUM_TRUTH_COVERAGE = 0.25
PRIMARY_HAIR_PRESENCE_MINIMUM_PREDICTION_COVERAGE = 0.25
PRIMARY_HAIR_PRESENCE_MINIMUM_DIRECTION_COSINE = 0.0
PRIMARY_HAIR_PRESENCE_PROXIMAL_ARC_FRACTION = 0.25
PRIMARY_HAIR_PRESENCE_RESAMPLE_POINTS = 32
PRIMARY_HAIR_PRESENCE_MINIMUM_DIRECTION_NORM = 1e-9
PRIMARY_HAIR_PRESENCE_SUPPORTED_DISTANCE_COST_WEIGHT = 0.10
PRIMARY_HAIR_PRESENCE_DIRECTION_COST_WEIGHT = 0.05
PRIMARY_HAIR_PRESENCE_NORMALIZED_DISTANCE_CAP = 2.0


def biological_hair_presence_matcher_contract() -> dict[str, Any]:
    """Return the publication/runtime identity of the primary hair matcher.

    Keeping the matcher contract next to its implementation prevents threshold
    selection, formal evaluation, and measurement assurance from silently
    assigning different meanings to ``F1@20 um``.
    """

    return {
        "schema_version": "PHAxis-biological-hair-presence-matcher-1.0",
        "target": "one_manual_single_trunk_centreline_per_visible_root_hair",
        "coordinate_space": "physical_um_xy",
        "curve_tolerance_um": PRIMARY_HAIR_PRESENCE_TOLERANCE_UM,
        "minimum_truth_coverage": (
            PRIMARY_HAIR_PRESENCE_MINIMUM_TRUTH_COVERAGE
        ),
        "minimum_prediction_coverage": (
            PRIMARY_HAIR_PRESENCE_MINIMUM_PREDICTION_COVERAGE
        ),
        "minimum_direction_cosine": (
            PRIMARY_HAIR_PRESENCE_MINIMUM_DIRECTION_COSINE
        ),
        "proximal_arc_fraction": PRIMARY_HAIR_PRESENCE_PROXIMAL_ARC_FRACTION,
        "resample_points": PRIMARY_HAIR_PRESENCE_RESAMPLE_POINTS,
        "assignment": (
            "per_source_image_maximum_cardinality_one_to_one_Hungarian_then_"
            "minimum_supported_curve_cost"
        ),
        "coverage": "bidirectional_arc_length_resampled_point_support",
        "stageB_predicted_geometry_proxy": "straight_base_to_tip",
        "manual_hair_width_assumed": False,
        "distal_endpoint_is_identity_gate": False,
        "complete_centreline_overlap_is_identity_gate": False,
        "length_error_is_identity_gate": False,
        "image_intensity_or_colour_is_matcher_input": False,
    }


def _points(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite N x 2 array")
    return result


def precision_recall_f1(true_positive: int, predicted: int, annotated: int) -> dict[str, float | int]:
    """Return micro precision/recall/F1 with explicit empty-set semantics."""

    tp = int(true_positive)
    n_pred = int(predicted)
    n_gt = int(annotated)
    if tp < 0 or n_pred < 0 or n_gt < 0 or tp > min(n_pred, n_gt):
        raise ValueError("invalid true-positive/predicted/annotated counts")
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {
        "tp": tp,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def match_points(
    predicted_xy: Any, annotated_xy: Any, tolerance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maximum-cardinality one-to-one point matching under ``tolerance``.

    Hungarian assignment receives a cost much larger than any feasible pair,
    so it first maximizes accepted cardinality and then minimizes total
    accepted distance.
    """

    predicted = _points(predicted_xy, name="predicted_xy")
    annotated = _points(annotated_xy, name="annotated_xy")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    if len(predicted) == 0 or len(annotated) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.float64)
    distances = np.linalg.norm(
        predicted[:, None, :] - annotated[None, :, :], axis=-1
    )
    feasible = distances <= tolerance
    # Normalizing every allowed edge to [0, 1] and charging more than the
    # entire possible allowed-edge sum for a forbidden edge proves the stated
    # maximum-cardinality-first semantics for every finite tolerance, rather
    # than only for tolerances smaller than a fixed sentinel.
    normalized = distances / max(tolerance, 1.0)
    forbidden = float(min(len(predicted), len(annotated)) + 1)
    costs = np.where(feasible, normalized, forbidden)
    pred_indices, gt_indices = linear_sum_assignment(costs)
    accepted_distances = distances[pred_indices, gt_indices]
    keep = accepted_distances <= tolerance
    return (
        pred_indices[keep].astype(np.int64, copy=False),
        gt_indices[keep].astype(np.int64, copy=False),
        accepted_distances[keep],
    )


def concordance_correlation(first: Any, second: Any) -> float:
    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape or first_array.ndim != 1:
        raise ValueError("CCC inputs must be equal-length vectors")
    if len(first_array) < 2:
        return float("nan")
    first_variance = first_array.var()
    second_variance = second_array.var()
    covariance = (
        (first_array - first_array.mean())
        * (second_array - second_array.mean())
    ).mean()
    denominator = (
        first_variance
        + second_variance
        + (first_array.mean() - second_array.mean()) ** 2
        + 1e-12
    )
    return float(2.0 * covariance / denominator)


def evaluate_image_instances(
    prediction: dict[str, Any],
    annotation: dict[str, Any],
    units_per_coordinate: float,
    tolerances: Iterable[float] = (5.0, 10.0, 20.0),
) -> dict[str, Any]:
    """Evaluate base identity and matched endpoint/length geometry for one image."""

    scale = float(units_per_coordinate)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("units_per_coordinate must be finite and positive")
    predicted_base = _points(prediction["base"], name="prediction.base") * scale
    annotated_base = _points(annotation["base"], name="annotation.base") * scale
    predicted_tip = _points(prediction["tip"], name="prediction.tip")
    annotated_tip = _points(annotation["tip"], name="annotation.tip")
    predicted_length = np.asarray(prediction["length_um"], dtype=np.float64)
    annotated_length = np.asarray(annotation["length_um"], dtype=np.float64)
    if len(predicted_tip) != len(predicted_base) or len(predicted_length) != len(predicted_base):
        raise ValueError("prediction base/tip/length counts differ")
    if len(annotated_tip) != len(annotated_base) or len(annotated_length) != len(annotated_base):
        raise ValueError("annotation base/tip/length counts differ")
    result: dict[str, Any] = {
        "n_pred": len(predicted_base),
        "n_gt": len(annotated_base),
        "tol": {},
    }
    for tolerance_value in tolerances:
        tolerance = float(tolerance_value)
        pred_indices, gt_indices, base_errors = match_points(
            predicted_base, annotated_base, tolerance
        )
        metrics = precision_recall_f1(
            len(pred_indices), len(predicted_base), len(annotated_base)
        )
        metrics["base_err_um_mean"] = (
            float(base_errors.mean()) if len(base_errors) else float("nan")
        )
        if len(pred_indices):
            endpoint_errors = np.linalg.norm(
                predicted_tip[pred_indices] * scale
                - annotated_tip[gt_indices] * scale,
                axis=1,
            )
            matched_predicted_length = predicted_length[pred_indices]
            matched_annotated_length = annotated_length[gt_indices]
            metrics.update(
                {
                    "tip_err_um_mean": float(endpoint_errors.mean()),
                    "tip_err_um_median": float(np.median(endpoint_errors)),
                    "length_mae_um": float(
                        np.abs(matched_predicted_length - matched_annotated_length).mean()
                    ),
                    "length_rel_err": float(
                        (
                            np.abs(matched_predicted_length - matched_annotated_length)
                            / np.maximum(matched_annotated_length, 1e-6)
                        ).mean()
                    ),
                    "_pred_len": matched_predicted_length,
                    "_gt_len": matched_annotated_length,
                    "_tip_err": endpoint_errors,
                }
            )
        result["tol"][tolerance] = metrics
    return result


def resample_polyline(points_xy: Any, step: float) -> np.ndarray:
    """Resample a polyline at approximately uniform coordinate spacing."""

    points = _points(points_xy, name="polyline")
    if len(points) < 2:
        return points
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = cumulative[-1]
    if total <= 0.0:
        return points[:1]
    sample_count = max(2, int(np.ceil(total / max(float(step), 1e-6))) + 1)
    locations = np.linspace(0.0, total, sample_count)
    return np.column_stack(
        (
            np.interp(locations, cumulative, points[:, 0]),
            np.interp(locations, cumulative, points[:, 1]),
        )
    )


def _resample_polyline_count(points_xy: Any, count: int) -> np.ndarray:
    """Resample a finite polyline to an exact number of points."""

    points = _points(points_xy, name="polyline")
    count = int(count)
    if len(points) < 2:
        raise ValueError("polyline must contain at least two points")
    if count < 2:
        raise ValueError("resample count must be at least two")
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if cumulative[-1] <= 1e-12:
        return np.repeat(points[:1], count, axis=0)
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-12))
    cumulative = cumulative[keep]
    points = points[keep]
    locations = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        (
            np.interp(locations, cumulative, points[:, 0]),
            np.interp(locations, cumulative, points[:, 1]),
        )
    )


def match_biological_hair_presence(
    predicted_polylines_xy: Sequence[Any],
    annotated_polylines_xy: Sequence[Any],
    units_per_coordinate: float,
    tolerance: float,
    *,
    minimum_truth_coverage: float = (
        PRIMARY_HAIR_PRESENCE_MINIMUM_TRUTH_COVERAGE
    ),
    minimum_prediction_coverage: float = (
        PRIMARY_HAIR_PRESENCE_MINIMUM_PREDICTION_COVERAGE
    ),
    minimum_direction_cosine: float = (
        PRIMARY_HAIR_PRESENCE_MINIMUM_DIRECTION_COSINE
    ),
    proximal_arc_fraction: float = PRIMARY_HAIR_PRESENCE_PROXIMAL_ARC_FRACTION,
    resample_points: int = PRIMARY_HAIR_PRESENCE_RESAMPLE_POINTS,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Match biological root-hair presence without an endpoint hard gate.

    The manual target is one centreline polyline per visible root hair, not a
    dense-width mask.  A feasible pair therefore needs only bidirectional
    partial centreline support within the physical tolerance and a weak
    non-opposing proximal-direction guard.  Attachment, complete-centreline,
    distal-endpoint, and length errors remain separate diagnostics.  Hungarian
    assignment makes every prediction and annotation contribute at most once.
    """

    scale = float(units_per_coordinate)
    tolerance = float(tolerance)
    minimum_truth_coverage = float(minimum_truth_coverage)
    minimum_prediction_coverage = float(minimum_prediction_coverage)
    minimum_direction_cosine = float(minimum_direction_cosine)
    proximal_arc_fraction = float(proximal_arc_fraction)
    resample_points = int(resample_points)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("units_per_coordinate must be finite and positive")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    for name, value in (
        ("minimum_truth_coverage", minimum_truth_coverage),
        ("minimum_prediction_coverage", minimum_prediction_coverage),
        ("proximal_arc_fraction", proximal_arc_fraction),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if (
        not np.isfinite(minimum_direction_cosine)
        or not -1.0 <= minimum_direction_cosine <= 1.0
    ):
        raise ValueError("minimum_direction_cosine must be finite and in [-1, 1]")
    if resample_points < 2:
        raise ValueError("resample_points must be at least two")

    # Validate both geometry collections before applying empty-set semantics.
    # Otherwise an empty collection on one side could make a malformed NaN or
    # empty polyline on the other side silently count as a valid unmatched
    # instance.  Degenerate but finite two-point predictions remain allowed and
    # are rejected as matches by ``predicted_valid_direction`` below.
    predicted_dense_rows = [
        _resample_polyline_count(polyline, resample_points) * scale
        for polyline in predicted_polylines_xy
    ]
    annotated_dense_rows = [
        _resample_polyline_count(polyline, resample_points) * scale
        for polyline in annotated_polylines_xy
    ]
    predicted_count = len(predicted_dense_rows)
    annotated_count = len(annotated_dense_rows)
    if predicted_count == 0 or annotated_count == 0:
        metrics = precision_recall_f1(0, predicted_count, annotated_count)
        metrics.update(
            {
                "curve_tolerance": tolerance,
                "minimum_truth_coverage": minimum_truth_coverage,
                "minimum_prediction_coverage": minimum_prediction_coverage,
                "minimum_direction_cosine": minimum_direction_cosine,
            }
        )
        return metrics, []

    predicted_dense = np.stack(predicted_dense_rows)
    annotated_dense = np.stack(annotated_dense_rows)
    proximal_index = max(
        1,
        min(
            resample_points - 1,
            int(round((resample_points - 1) * proximal_arc_fraction)),
        ),
    )
    predicted_vectors = predicted_dense[:, proximal_index] - predicted_dense[:, 0]
    annotated_vectors = annotated_dense[:, proximal_index] - annotated_dense[:, 0]
    predicted_norms = np.linalg.norm(predicted_vectors, axis=1)
    annotated_norms = np.linalg.norm(annotated_vectors, axis=1)
    predicted_valid_direction = (
        predicted_norms > PRIMARY_HAIR_PRESENCE_MINIMUM_DIRECTION_NORM
    )
    annotated_valid_direction = (
        annotated_norms > PRIMARY_HAIR_PRESENCE_MINIMUM_DIRECTION_NORM
    )
    predicted_directions = predicted_vectors / np.maximum(
        predicted_norms[:, None], PRIMARY_HAIR_PRESENCE_MINIMUM_DIRECTION_NORM
    )
    annotated_directions = annotated_vectors / np.maximum(
        annotated_norms[:, None], PRIMARY_HAIR_PRESENCE_MINIMUM_DIRECTION_NORM
    )
    direction_cosine = predicted_directions @ annotated_directions.T

    curve_precision = np.zeros((predicted_count, annotated_count), dtype=np.float64)
    curve_recall = np.zeros_like(curve_precision)
    curve_f1 = np.zeros_like(curve_precision)
    supported_mean_distance = np.full_like(curve_precision, np.inf)
    annotated_tree = cKDTree(annotated_dense.reshape(-1, 2))
    for predicted_index, dense in enumerate(predicted_dense):
        nearby = annotated_tree.query_ball_point(dense, tolerance)
        candidate_annotations = sorted(
            {
                point_index // resample_points
                for neighbours in nearby
                for point_index in neighbours
            }
        )
        for annotated_index in candidate_annotations:
            distances = np.linalg.norm(
                dense[:, None, :] - annotated_dense[annotated_index][None, :, :],
                axis=-1,
            )
            prediction_nearest = distances.min(axis=1)
            annotation_nearest = distances.min(axis=0)
            prediction_support = prediction_nearest <= tolerance
            annotation_support = annotation_nearest <= tolerance
            precision = float(prediction_support.mean())
            recall = float(annotation_support.mean())
            harmonic = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0.0
                else 0.0
            )
            curve_precision[predicted_index, annotated_index] = precision
            curve_recall[predicted_index, annotated_index] = recall
            curve_f1[predicted_index, annotated_index] = harmonic
            supported = np.concatenate(
                (
                    prediction_nearest[prediction_support],
                    annotation_nearest[annotation_support],
                )
            )
            if len(supported):
                supported_mean_distance[predicted_index, annotated_index] = float(
                    supported.mean()
                )

    feasible = (
        (curve_precision >= minimum_prediction_coverage)
        & (curve_recall >= minimum_truth_coverage)
        & predicted_valid_direction[:, None]
        & annotated_valid_direction[None, :]
        & (direction_cosine >= minimum_direction_cosine)
    )
    normalized_distance = np.minimum(
        supported_mean_distance
        / max(tolerance, PRIMARY_HAIR_PRESENCE_MINIMUM_DIRECTION_NORM),
        PRIMARY_HAIR_PRESENCE_NORMALIZED_DISTANCE_CAP,
    )
    direction_penalty = PRIMARY_HAIR_PRESENCE_DIRECTION_COST_WEIGHT * (
        1.0 - np.clip(direction_cosine, -1.0, 1.0)
    ) / 2.0
    cost = (
        1.0
        - curve_f1
        + PRIMARY_HAIR_PRESENCE_SUPPORTED_DISTANCE_COST_WEIGHT
        * normalized_distance
        + direction_penalty
    )
    assignment_size = min(predicted_count, annotated_count)
    feasible_cost_max = float(np.max(cost[feasible])) if np.any(feasible) else 1.0
    forbidden_cost = (assignment_size + 1.0) * (feasible_cost_max + 1.0)
    predicted_indices, annotated_indices = linear_sum_assignment(
        np.where(feasible, cost, forbidden_cost)
    )
    keep = feasible[predicted_indices, annotated_indices]
    predicted_indices = predicted_indices[keep]
    annotated_indices = annotated_indices[keep]
    matches: list[dict[str, float | int]] = []
    for predicted_index, annotated_index in zip(
        predicted_indices, annotated_indices, strict=True
    ):
        matches.append(
            {
                "predicted_index": int(predicted_index),
                "annotated_index": int(annotated_index),
                "prediction_coverage": float(
                    curve_precision[predicted_index, annotated_index]
                ),
                "truth_coverage": float(
                    curve_recall[predicted_index, annotated_index]
                ),
                "curve_f1": float(curve_f1[predicted_index, annotated_index]),
                "supported_mean_distance": float(
                    supported_mean_distance[predicted_index, annotated_index]
                ),
                "proximal_direction_cosine": float(
                    direction_cosine[predicted_index, annotated_index]
                ),
            }
        )
    metrics = precision_recall_f1(
        len(matches), predicted_count, annotated_count
    )
    metrics.update(
        {
            "curve_tolerance": tolerance,
            "minimum_truth_coverage": minimum_truth_coverage,
            "minimum_prediction_coverage": minimum_prediction_coverage,
            "minimum_direction_cosine": minimum_direction_cosine,
        }
    )
    return metrics, matches


def strict_presence_matches(
    predicted_base_xy: Any,
    predicted_tip_xy: Any,
    annotated_polylines_xy: Sequence[Any],
    annotated_base_xy: Any,
    annotated_tip_xy: Any,
    units_per_coordinate: float,
    tolerance: float,
    *,
    maximum_length_ratio: float = 2.5,
    minimum_direction_cosine: float = 0.5,
    hausdorff_tolerance_multiplier: float = 2.5,
    predicted_samples: int = 15,
    centroid_gate: float = 250.0,
) -> int:
    """Count strict one-to-one whole-centreline correspondences.

    This is intentionally secondary to attachment/base identity.  A feasible
    pair must have the same polarity, a bounded bidirectional length ratio,
    symmetric mean-of-nearest-point distance within ``tolerance``, and maximum
    bidirectional nearest-point distance within the Hausdorff guard.
    """

    predicted_base = _points(predicted_base_xy, name="predicted_base_xy")
    predicted_tip = _points(predicted_tip_xy, name="predicted_tip_xy")
    annotated_base = _points(annotated_base_xy, name="annotated_base_xy")
    annotated_tip = _points(annotated_tip_xy, name="annotated_tip_xy")
    if len(predicted_base) != len(predicted_tip):
        raise ValueError("predicted base/tip counts differ")
    if len(annotated_base) != len(annotated_tip) or len(annotated_base) != len(annotated_polylines_xy):
        raise ValueError("annotated base/tip/polyline counts differ")
    scale = float(units_per_coordinate)
    tolerance = float(tolerance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("units_per_coordinate must be finite and positive")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    if len(predicted_base) == 0 or len(annotated_base) == 0:
        return 0

    interpolation = np.linspace(0.0, 1.0, int(predicted_samples))
    predicted_lines = (
        predicted_base[:, None, :]
        + interpolation[None, :, None]
        * (predicted_tip - predicted_base)[:, None, :]
    )
    annotated_lines = [
        resample_polyline(polyline, max(len(np.asarray(polyline)), predicted_samples))
        for polyline in annotated_polylines_xy
    ]
    if any(len(polyline) == 0 for polyline in annotated_lines):
        raise ValueError("annotated polylines cannot be empty")
    annotated_centroids = np.asarray(
        [polyline.mean(axis=0) for polyline in annotated_lines], dtype=np.float64
    )
    predicted_lengths = np.linalg.norm(predicted_tip - predicted_base, axis=1) * scale
    annotated_lengths = np.asarray(
        [np.linalg.norm(np.diff(polyline, axis=0), axis=1).sum() for polyline in annotated_lines],
        dtype=np.float64,
    ) * scale
    predicted_direction = predicted_tip - predicted_base
    annotated_direction = annotated_tip - annotated_base
    centroid_distances = np.linalg.norm(
        predicted_lines.mean(axis=1)[:, None, :] - annotated_centroids[None, :, :],
        axis=-1,
    ) * scale
    costs = np.full(
        (len(predicted_base), len(annotated_base)),
        INFEASIBLE_COST,
        dtype=np.float64,
    )
    for pred_index in range(len(predicted_base)):
        candidate_indices = np.flatnonzero(
            centroid_distances[pred_index] <= float(centroid_gate)
        )
        for gt_index in candidate_indices:
            dot_product = float(
                predicted_direction[pred_index] @ annotated_direction[gt_index]
            )
            norm_product = float(
                np.linalg.norm(predicted_direction[pred_index])
                * np.linalg.norm(annotated_direction[gt_index])
            )
            if norm_product <= 1e-9 or dot_product / norm_product < minimum_direction_cosine:
                continue
            ratio = max(predicted_lengths[pred_index], annotated_lengths[gt_index]) / max(
                min(predicted_lengths[pred_index], annotated_lengths[gt_index]), 1e-6
            )
            if ratio > maximum_length_ratio:
                continue
            pairwise = np.linalg.norm(
                predicted_lines[pred_index][:, None, :]
                - annotated_lines[gt_index][None, :, :],
                axis=-1,
            )
            pred_to_annotation = pairwise.min(axis=1)
            annotation_to_pred = pairwise.min(axis=0)
            symmetric_mean = 0.5 * (
                pred_to_annotation.mean() + annotation_to_pred.mean()
            ) * scale
            hausdorff = max(
                pred_to_annotation.max(), annotation_to_pred.max()
            ) * scale
            if (
                symmetric_mean <= tolerance
                and hausdorff <= tolerance * hausdorff_tolerance_multiplier
            ):
                costs[pred_index, gt_index] = symmetric_mean
    predicted_indices, annotated_indices = linear_sum_assignment(costs)
    return int((costs[predicted_indices, annotated_indices] <= tolerance).sum())


# Concise compatibility aliases used by the evaluation script.  They remain
# explicit functions rather than an external module import.
prf = precision_recall_f1
evaluate_image = evaluate_image_instances
presence_match_strict = strict_presence_matches
