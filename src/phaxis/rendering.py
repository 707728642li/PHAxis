"""Lossless review overlays in the PHAxis source-pixel coordinate frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .contracts import ContractError
from .phenotypes import load_axis_geometry


def render_display_background(
    image: np.ndarray,
    *,
    lower_percentile: float = 0.2,
    upper_percentile: float = 99.8,
    gamma: float = 0.72,
) -> np.ndarray:
    """Return the exact global source display used beneath an overlay."""

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] >= 3:
        array = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    elif array.ndim == 3:
        array = array[..., 0]
    array = np.squeeze(array).astype(np.float32)
    lower, upper = np.percentile(array, [lower_percentile, upper_percentile])
    normalized = np.clip((array - lower) / max(float(upper - lower), 1e-6), 0.0, 1.0)
    inverted = np.power(1.0 - normalized, float(gamma)) * 165.0
    gray = inverted.astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _integer_polyline(points_xy: Any) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ContractError("overlay polyline must be N x 2")
    return np.rint(points).astype(np.int32).reshape(-1, 1, 2)


def render_prediction_overlay(
    image: np.ndarray,
    prediction: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    include_text: bool = True,
    display_gamma: float = 0.72,
) -> np.ndarray:
    canvas = render_display_background(image, gamma=display_gamma)
    height, width = canvas.shape[:2]
    root_path = Path(artifact_root) / str(prediction["root_mask_relpath"])
    # Root masks are locally generated, hash-locked PHAxis artifacts.  OpenCV
    # avoids Pillow's generic decompression-bomb heuristic, which rejects
    # legitimate stitched microscopy masks above ~179 MP.  Geometry is still
    # bounded by the source image through the exact shape check below.
    root_image = cv2.imread(str(root_path), cv2.IMREAD_GRAYSCALE)
    if root_image is None:
        raise ContractError(f"cannot read root-mask artifact: {root_path}")
    root = root_image > 0
    if root.shape != (height, width):
        raise ContractError(
            f"root/image shape mismatch: root={root.shape}, image={(height, width)}"
        )
    line_width = max(1, int(round(min(height, width) / 900.0)))
    root_boundary = cv2.morphologyEx(
        root.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    canvas[root_boundary] = (220, 170, 25)  # cyan in BGR

    axis_path = Path(artifact_root) / str(prediction["root_axis_geometry_relpath"])
    axis = load_axis_geometry(
        axis_path, expected_image_sha256=str(prediction["source_image_sha256"])
    )
    cv2.polylines(
        canvas,
        [_integer_polyline(axis["path_xy"])],
        False,
        (230, 230, 230),
        line_width,
        cv2.LINE_AA,
    )

    identities = prediction.get("identity_hairs", ())
    length_hairs = prediction.get("length_hairs", ())
    length_identity_ids = [
        str(hair["identity_source_instance_id"]) for hair in length_hairs
    ]
    if len(set(length_identity_ids)) != len(length_identity_ids):
        raise ContractError("duplicate length-to-identity association in overlay")
    lengths_by_identity = dict(zip(length_identity_ids, length_hairs, strict=True))
    for hair in identities:
        source_instance_id = str(hair["source_instance_id"])
        matched = lengths_by_identity.get(source_instance_id)
        matched_length = matched is not None
        if bool(hair.get("complete_length_measurement_eligible", False)) != matched_length:
            raise ContractError("identity length-eligibility flag/association mismatch")
        points = _integer_polyline(
            matched["points_xy"] if matched_length else hair["points_xy"]
        )
        color = (90, 245, 115) if matched_length else (20, 205, 255)
        cv2.polylines(canvas, [points], False, color, line_width, cv2.LINE_AA)
        identity_points = _integer_polyline(hair["points_xy"])
        base = tuple(identity_points[0, 0].tolist())
        tip = tuple(points[-1, 0].tolist())
        cv2.circle(canvas, base, max(2, line_width + 1), (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, tip, max(1, line_width), (25, 105, 255), -1, cv2.LINE_AA)

    distal = tuple(
        np.rint(np.asarray(prediction["root_cap_point_xy"], dtype=float)).astype(int).tolist()
    )
    cv2.circle(
        canvas, distal, max(5, line_width * 3), (255, 60, 220), 2, cv2.LINE_AA
    )

    if include_text:
        task_id = str(prediction["task_id"])
        length_count = len(prediction.get("length_hairs", ()))
        lines = (
            f"PHAxis 1.0.0 | {task_id}",
            f"Stage-B identities: {len(identities)} | endpoint-complete lengths: {length_count}",
            "green=matched complete centreline  amber=presence vector (not a length)",
            "cyan=root boundary  white=ordered axis  yellow dot=hair base  magenta=distal point",
        )
        font_scale = max(0.45, min(width, height) / 2200.0)
        thickness = max(1, int(round(font_scale * 2)))
        padding = max(10, int(round(16 * font_scale)))
        line_height = max(22, int(round(30 * font_scale)))
        box_height = padding * 2 + line_height * len(lines)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width, box_height), (0, 0, 0), -1)
        canvas = cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0)
        for index, text in enumerate(lines):
            cv2.putText(
                canvas,
                text,
                (padding, padding + line_height * (index + 1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (245, 245, 245),
                thickness,
                cv2.LINE_AA,
            )
    return canvas
