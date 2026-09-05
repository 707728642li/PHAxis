"""Leak-aware operating-point selection on train399-only OOF predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..evaluation_metrics import (
    biological_hair_presence_matcher_contract,
    match_biological_hair_presence,
)


def match_points_within_tolerance(
    predicted: np.ndarray, ground_truth: np.ndarray, tolerance_um: float
) -> int:
    """Maximum-cardinality one-to-one point matching under a hard tolerance."""

    tolerance_um = float(tolerance_um)
    if not np.isfinite(tolerance_um) or tolerance_um < 0.0:
        raise ValueError("tolerance_um must be finite and nonnegative")
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    ground_truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(ground_truth)):
        raise ValueError("point coordinates must be finite")
    if not len(predicted) or not len(ground_truth):
        return 0
    distance = np.linalg.norm(
        predicted[:, None, :] - ground_truth[None, :, :], axis=-1
    )
    feasible = distance <= tolerance_um
    normalized = distance / max(tolerance_um, 1.0)
    forbidden = float(min(len(predicted), len(ground_truth)) + 1)
    cost = np.where(feasible, normalized, forbidden)
    predicted_indices, truth_indices = linear_sum_assignment(cost)
    return int(
        np.count_nonzero(
            distance[predicted_indices, truth_indices] <= tolerance_um
        )
    )


def score_threshold(
    rows: Sequence[dict[str, Any]], threshold: float, tolerance_um: float
) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("threshold scoring requires at least one row")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1]")
    matcher = biological_hair_presence_matcher_contract()
    if not np.isclose(
        float(tolerance_um),
        float(matcher["curve_tolerance_um"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("threshold selection must use the locked 20-um matcher")
    biological_true_positive = 0
    attachment_true_positive = 0
    total_predicted = 0
    total_truth = 0
    per_image_absolute_count_error: list[float] = []
    per_image_count_bias: list[float] = []
    for row in rows:
        scores = np.asarray(row["pred"]["score"], dtype=np.float64)
        predicted_base = np.asarray(row["pred"]["base"], dtype=np.float64)
        predicted_tip = np.asarray(row["pred"]["tip"], dtype=np.float64)
        truth_base = np.asarray(row["gt"]["base"], dtype=np.float64)
        truth_polylines = [
            np.asarray(polyline, dtype=np.float64)
            for polyline in row["gt"]["polys"]
        ]
        if (
            predicted_base.shape != (len(scores), 2)
            or predicted_tip.shape != predicted_base.shape
            or truth_base.ndim != 2
            or truth_base.shape[1:] != (2,)
            or len(truth_polylines) != len(truth_base)
            or not np.all(np.isfinite(scores))
            or not np.all((0.0 <= scores) & (scores <= 1.0))
            or not np.all(np.isfinite(predicted_base))
            or not np.all(np.isfinite(predicted_tip))
            or not np.all(np.isfinite(truth_base))
        ):
            raise RuntimeError(f"invalid OOF geometry for {row['task_id']}")
        keep = scores >= threshold
        microns_per_pixel = float(row["um_per_px"])
        if not np.isfinite(microns_per_pixel) or microns_per_pixel <= 0.0:
            raise RuntimeError(f"invalid OOF physical scale for {row['task_id']}")
        kept_base_um = predicted_base[keep] * microns_per_pixel
        kept_tip_um = predicted_tip[keep] * microns_per_pixel
        truth_base_um = truth_base * microns_per_pixel
        truth_polylines_um = [
            polyline * microns_per_pixel for polyline in truth_polylines
        ]
        presence, _matches = match_biological_hair_presence(
            [
                np.stack((base, tip))
                for base, tip in zip(kept_base_um, kept_tip_um, strict=True)
            ],
            truth_polylines_um,
            1.0,
            matcher["curve_tolerance_um"],
            minimum_truth_coverage=matcher["minimum_truth_coverage"],
            minimum_prediction_coverage=matcher["minimum_prediction_coverage"],
            minimum_direction_cosine=matcher["minimum_direction_cosine"],
            proximal_arc_fraction=matcher["proximal_arc_fraction"],
            resample_points=matcher["resample_points"],
        )
        attachment_matches = match_points_within_tolerance(
            kept_base_um, truth_base_um, tolerance_um
        )
        biological_true_positive += int(presence["tp"])
        attachment_true_positive += attachment_matches
        total_predicted += len(kept_base_um)
        total_truth += len(truth_base_um)
        difference = len(kept_base_um) - len(truth_base_um)
        per_image_absolute_count_error.append(abs(difference))
        per_image_count_bias.append(difference)
    precision = (
        biological_true_positive / total_predicted if total_predicted else 0.0
    )
    recall = biological_true_positive / total_truth if total_truth else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "threshold": threshold,
        "tolerance_um": float(tolerance_um),
        "images": len(rows),
        "true_positive": int(biological_true_positive),
        "biological_presence_true_positive": int(biological_true_positive),
        "attachment_proxy_true_positive": int(attachment_true_positive),
        "predicted": int(total_predicted),
        "ground_truth": int(total_truth),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "count_mae": float(np.mean(per_image_absolute_count_error)),
        "count_bias": float(np.mean(per_image_count_bias)),
        "primary_matcher_contract": matcher,
    }


def _selection_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    # The primary biological-presence F1 is optimized first.  Count agreement
    # then controls the operating point without turning endpoint or length
    # geometry into a detection gate.
    return (
        float(metrics["f1"]),
        -float(metrics["count_mae"]),
        -abs(float(metrics["count_bias"])),
        float(metrics["threshold"]),
    )


def select_train399_threshold(
    rows: Sequence[dict[str, Any]],
    train_task_family_rows: Sequence[Sequence[str]],
    *,
    thresholds: Iterable[float],
    tolerance_um: float = 20.0,
) -> dict[str, Any]:
    """Select one robust threshold without consulting locked dev44 labels.

    Every input prediction must be out-of-fold for its own image, and each
    biological family must occur in only one source fold. The deployment lock
    is the median of the five leave-one-fold-out train399 optima; this avoids
    choosing an operating point on the very fold to which it is applied while
    still yielding one fixed threshold for the final five-seed ensemble.
    """

    rows = list(rows)
    threshold_grid = sorted({float(value) for value in thresholds})
    if len(rows) != 399 or not threshold_grid:
        raise RuntimeError("threshold selection requires 399 rows and a non-empty grid")
    if any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for value in threshold_grid
    ):
        raise RuntimeError("threshold grid must be finite and in [0, 1]")
    row_by_id = {str(row["task_id"]): row for row in rows}
    if len(row_by_id) != 399:
        raise RuntimeError("OOF train399 task IDs are not unique")
    family_rows = [
        (str(task), str(family)) for task, family in train_task_family_rows
    ]
    family_by_id = {task: family for task, family in family_rows}
    if (
        len(family_rows) != 399
        or len(family_by_id) != 399
        or any(not task or not family for task, family in family_rows)
    ):
        raise RuntimeError("locked family table is incomplete or duplicated")
    if set(row_by_id) != set(family_by_id):
        raise RuntimeError("OOF train399 IDs differ from the locked family table")
    family_folds: dict[str, set[int]] = defaultdict(set)
    for task_id, row in row_by_id.items():
        family_folds[family_by_id[task_id]].add(int(row["fold"]))
    leaking_families = {
        family: sorted(folds)
        for family, folds in family_folds.items()
        if len(folds) != 1
    }
    if leaking_families:
        raise RuntimeError(f"OOF source folds split biological families: {leaking_families}")
    fold_ids = sorted({int(row["fold"]) for row in rows})
    if fold_ids != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"expected five source OOF folds, found {fold_ids}")

    sweep = [score_threshold(rows, value, tolerance_um) for value in threshold_grid]
    global_best = max(sweep, key=_selection_key)
    nested_optima: dict[str, dict[str, Any]] = {}
    for held_out_fold in fold_ids:
        selection_rows = [
            row for row in rows if int(row["fold"]) != held_out_fold
        ]
        candidates = [
            score_threshold(selection_rows, value, tolerance_um)
            for value in threshold_grid
        ]
        nested_optima[str(held_out_fold)] = {
            "held_out_fold": held_out_fold,
            "selection_images": len(selection_rows),
            "held_out_images": len(rows) - len(selection_rows),
            "best": max(candidates, key=_selection_key),
        }
    nested_thresholds = [
        float(nested_optima[str(fold)]["best"]["threshold"])
        for fold in fold_ids
    ]
    locked_threshold = float(np.median(nested_thresholds))
    if locked_threshold not in threshold_grid:
        locked_threshold = min(
            threshold_grid, key=lambda value: abs(value - locked_threshold)
        )
    locked_metrics = score_threshold(rows, locked_threshold, tolerance_um)
    return {
        "locked_threshold": locked_threshold,
        "selection_rule": (
            "median_of_five_leave_one_source_fold_out_train399_optima"
        ),
        "primary_metric": "tolerant_biological_presence_f1_at_20um",
        "primary_matcher_contract": biological_hair_presence_matcher_contract(),
        "tie_break_rule": (
            "primary_micro_f1_then_count_mae_then_abs_count_bias_then_higher_threshold"
        ),
        "family_grouped_oof_verified": True,
        "fold_ids": fold_ids,
        "nested_thresholds": nested_thresholds,
        "nested_optima": nested_optima,
        "global_train399_best": global_best,
        "locked_threshold_train399_oof_metrics": locked_metrics,
        "sweep": sweep,
    }
