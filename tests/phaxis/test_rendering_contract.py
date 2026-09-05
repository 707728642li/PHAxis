from __future__ import annotations

from copy import deepcopy

import cv2
import numpy as np
from PIL import Image
import pytest

from phaxis.contracts import ContractError
from phaxis.rendering import render_prediction_overlay


def _case(tmp_path):
    mask_dir = tmp_path / "masks"
    axis_dir = tmp_path / "axis_geometry"
    mask_dir.mkdir()
    axis_dir.mkdir()
    mask = np.zeros((64, 64), np.uint8)
    mask[8:56, 28:36] = 255
    Image.fromarray(mask).save(mask_dir / "T1.root.png")
    np.savez_compressed(
        axis_dir / "T1.npz",
        path_xy=np.asarray([[32.0, 8.0], [32.0, 56.0]], np.float32),
        distance_from_tip_px=np.asarray([0.0, 48.0], np.float32),
        radius_px=np.asarray([4.0, 4.0], np.float32),
        reference_radius_px=np.asarray(4.0, np.float32),
        point_xy=np.asarray([32.0, 56.0], np.float32),
        source_image_sha256=np.asarray("a" * 64),
    )
    prediction = {
        "task_id": "T1",
        "source_image_sha256": "a" * 64,
        "root_mask_relpath": "masks/T1.root.png",
        "root_axis_geometry_relpath": "axis_geometry/T1.npz",
        "root_cap_point_xy": [32.0, 56.0],
        "identity_hairs": [
            {
                "source_instance_id": "I1",
                "points_xy": [[28.0, 20.0], [18.0, 20.0]],
                "complete_length_measurement_eligible": True,
            },
            {
                "source_instance_id": "I2",
                "points_xy": [[28.0, 30.0], [18.0, 30.0]],
                "complete_length_measurement_eligible": False,
            },
        ],
        "length_hairs": [
            {
                "identity_source_instance_id": "I1",
                "points_xy": [[28.0, 20.0], [24.0, 16.0], [18.0, 20.0]],
            }
        ],
    }
    return prediction


def test_overlay_draws_matched_curve_not_stageb_identity_vector(
    tmp_path, monkeypatch
):
    prediction = _case(tmp_path)
    calls = []
    original = cv2.polylines

    def capture(image, points, closed, color, thickness, line_type):
        calls.append((np.asarray(points[0]).copy(), tuple(color)))
        return original(image, points, closed, color, thickness, line_type)

    monkeypatch.setattr(cv2, "polylines", capture)
    render_prediction_overlay(
        np.zeros((64, 64), np.uint16),
        prediction,
        artifact_root=tmp_path,
        include_text=False,
    )

    assert len(calls) == 3  # ordered axis, matched curve, unmatched vector
    np.testing.assert_array_equal(
        calls[1][0].reshape(-1, 2),
        np.asarray([[28, 20], [24, 16], [18, 20]]),
    )
    assert calls[1][1] == (90, 245, 115)
    np.testing.assert_array_equal(
        calls[2][0].reshape(-1, 2), np.asarray([[28, 30], [18, 30]])
    )
    assert calls[2][1] == (20, 205, 255)


def test_overlay_rejects_length_flag_association_mismatch(tmp_path):
    prediction = deepcopy(_case(tmp_path))
    prediction["identity_hairs"][0]["complete_length_measurement_eligible"] = False
    with pytest.raises(ContractError, match="flag/association mismatch"):
        render_prediction_overlay(
            np.zeros((64, 64), np.uint16),
            prediction,
            artifact_root=tmp_path,
            include_text=False,
        )
