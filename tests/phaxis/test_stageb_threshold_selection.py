from __future__ import annotations

import numpy as np
import pytest

from phaxis.hair_stageb.threshold_selection import (
    match_points_within_tolerance,
    score_threshold,
    select_train399_threshold,
)


def test_matcher_maximizes_cardinality_under_tolerance() -> None:
    predicted = np.asarray([[0.0, 0.0], [2.0, 0.0]])
    truth = np.asarray([[0.9, 0.0], [2.9, 0.0]])
    assert match_points_within_tolerance(predicted, truth, 1.1) == 2
    assert match_points_within_tolerance(predicted, truth, 0.5) == 0


def test_threshold_selection_is_family_grouped_and_count_aware() -> None:
    rows = []
    families = []
    for index in range(399):
        task_id = f"T{index:03d}"
        fold = index % 5
        # Family is intentionally unique in this compact contract test.
        families.append((task_id, f"F{index:03d}"))
        rows.append(
            {
                "task_id": task_id,
                "fold": fold,
                "um_per_px": 1.0,
                "pred": {
                    "base": np.asarray([[0.0, 0.0], [100.0, 100.0]]),
                    "tip": np.asarray([[40.0, 0.0], [140.0, 100.0]]),
                    "score": np.asarray([0.30, 0.10]),
                },
                "gt": {
                    "base": np.asarray([[0.0, 0.0]]),
                    "polys": [np.asarray([[0.0, 0.0], [40.0, 0.0]])],
                },
            }
        )
    result = select_train399_threshold(
        rows, families, thresholds=[0.05, 0.20, 0.40], tolerance_um=20.0
    )
    assert result["locked_threshold"] == 0.20
    assert result["family_grouped_oof_verified"] is True
    metrics = score_threshold(rows, 0.20, 20.0)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["primary_matcher_contract"]["distal_endpoint_is_identity_gate"] is False


def test_threshold_score_uses_biological_presence_not_base_proxy() -> None:
    # The prediction starts 30 um distal to the manual attachment, so the
    # attachment proxy fails at 20 um.  It nevertheless covers the same visible
    # single-trunk hair in the same direction and is a valid biological identity.
    rows = [
        {
            "task_id": "T0",
            "fold": 0,
            "um_per_px": 1.0,
            "pred": {
                "base": np.asarray([[30.0, 0.0]]),
                "tip": np.asarray([[70.0, 0.0]]),
                "score": np.asarray([0.8]),
                # Irrelevant appearance metadata must not enter the matcher.
                "gray_white_intensity": np.asarray([245.0]),
            },
            "gt": {
                "base": np.asarray([[0.0, 0.0]]),
                "polys": [np.asarray([[0.0, 0.0], [100.0, 0.0]])],
            },
        }
    ]
    metrics = score_threshold(rows, 0.5, 20.0)
    assert metrics["biological_presence_true_positive"] == 1
    assert metrics["attachment_proxy_true_positive"] == 0
    assert metrics["f1"] == 1.0


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, np.nan, np.inf])
def test_historical_threshold_prior_rejects_invalid_physical_scale(bad_scale) -> None:
    rows = [
        {
            "task_id": "T0",
            "fold": 0,
            "um_per_px": bad_scale,
            "pred": {
                "base": np.asarray([[0.0, 0.0]]),
                "tip": np.asarray([[10.0, 0.0]]),
                "score": np.asarray([0.5]),
            },
            "gt": {
                "base": np.asarray([[0.0, 0.0]]),
                "polys": [np.asarray([[0.0, 0.0], [10.0, 0.0]])],
            },
        }
    ]
    with pytest.raises(RuntimeError, match="physical scale"):
        score_threshold(rows, 0.5, 20.0)


def test_historical_threshold_prior_rejects_nonfinite_scores_and_points() -> None:
    with pytest.raises(ValueError, match="coordinates must be finite"):
        match_points_within_tolerance(
            np.asarray([[np.nan, 0.0]]), np.asarray([[0.0, 0.0]]), 20.0
        )
    rows = [
        {
            "task_id": "T0",
            "fold": 0,
            "um_per_px": 1.0,
            "pred": {
                "base": np.asarray([[0.0, 0.0]]),
                "tip": np.asarray([[10.0, 0.0]]),
                "score": np.asarray([np.nan]),
            },
            "gt": {
                "base": np.asarray([[0.0, 0.0]]),
                "polys": [np.asarray([[0.0, 0.0], [10.0, 0.0]])],
            },
        }
    ]
    with pytest.raises(RuntimeError, match="invalid OOF geometry"):
        score_threshold(rows, 0.5, 20.0)
