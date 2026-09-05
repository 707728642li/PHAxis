"""Single-source geometry contract for Stage-B candidate-pool decoding.

The QC-development score sweep is allowed to filter candidates by the base
score only.  Every option that can alter a candidate's base or straight
base-to-tip proxy therefore belongs in this small, torch-free contract.
"""

from __future__ import annotations

import math
import operator
from typing import Any

from ..constants import (
    HAIR_MAX_INSTANCES,
    HAIR_NMS_KERNEL,
    HAIR_OUT_STRIDE,
    HAIR_ROOT_GATE_UM,
    HAIR_WORKING_UM_PER_PX,
)


CANDIDATE_DECODER_CONTRACT_SCHEMA = (
    "PHAxis-StageB-biological-presence-candidate-decoder-1.0"
)
TIP_HEATMAP_SCORE_FLOOR = 0.15
TIP_SNAP_RADIUS_UM = 30.0
TIP_SNAP_MINIMUM_DIRECTION_COSINE = 0.85
LENGTH_SCALE_UM = 100.0
BASE_LENGTH_LOG_CLIP = (-4.0, 3.0)
OFFSET_CLIP = (-0.5, 1.5)
DIRECTION_NORMALIZATION_EPSILON = 1e-6
PRESENCE_PROXY_MINIMUM_LENGTH_WORKING_PX = 1e-6
ROOT_FOREGROUND_PROBABILITY_THRESHOLD = 0.5
BILINEAR_UPPER_BORDER_EPSILON_PX = 1e-3


def _strict_integer(value: Any, *, name: str) -> int:
    """Return an integer parameter without silently truncating a float."""

    if isinstance(value, bool):
        raise ValueError(f"candidate {name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"candidate {name} must be an integer") from error
    return int(result)


def biological_presence_candidate_decoder_contract(
    *,
    working_um_per_px: float,
    out_stride: int,
    nms_kernel: int,
    max_instances: int,
    root_gate_um: tuple[float, float] | None,
    tip_score_floor: float,
    tip_snap_radius_um: float,
) -> dict[str, Any]:
    """Materialize every parameter that can alter candidate-pool geometry."""

    working_um_per_px = float(working_um_per_px)
    out_stride = _strict_integer(out_stride, name="out_stride")
    nms_kernel = _strict_integer(nms_kernel, name="nms_kernel")
    max_instances = _strict_integer(max_instances, name="max_instances")
    tip_score_floor = float(tip_score_floor)
    tip_snap_radius_um = float(tip_snap_radius_um)
    if not math.isfinite(working_um_per_px) or working_um_per_px <= 0.0:
        raise ValueError("candidate working_um_per_px must be finite and positive")
    if out_stride <= 0:
        raise ValueError("candidate out_stride must be positive")
    if nms_kernel <= 0 or nms_kernel % 2 != 1:
        raise ValueError("candidate nms_kernel must be a positive odd integer")
    if max_instances <= 0:
        raise ValueError("candidate max_instances must be positive")
    if not math.isfinite(tip_score_floor) or not 0.0 < tip_score_floor < 1.0:
        raise ValueError("candidate tip_score_floor must be finite and in (0, 1)")
    if not math.isfinite(tip_snap_radius_um) or tip_snap_radius_um < 0.0:
        raise ValueError("candidate tip_snap_radius_um must be finite and nonnegative")
    if root_gate_um is None:
        serialized_root_gate = None
    else:
        if len(root_gate_um) != 2:
            raise ValueError("candidate root_gate_um must have two bounds")
        serialized_root_gate = [float(value) for value in root_gate_um]
        if (
            not all(math.isfinite(value) for value in serialized_root_gate)
            or serialized_root_gate[0] > serialized_root_gate[1]
        ):
            raise ValueError("candidate root_gate_um bounds are invalid")
    return {
        "schema_version": CANDIDATE_DECODER_CONTRACT_SCHEMA,
        "candidate_score_field": "sigmoid_base_heatmap_after_NMS",
        "candidate_score_comparison": "inclusive_greater_than_or_equal",
        "threshold_operation": "base_score_filter_only_after_one_decode",
        "tip_heatmap_role": "fixed_floor_geometry_snap_only_not_candidate_score",
        "working_um_per_px": working_um_per_px,
        "out_stride": out_stride,
        "nms_kernel": nms_kernel,
        "max_instances": max_instances,
        "root_gate_um": serialized_root_gate,
        "tip_score_floor": tip_score_floor,
        "tip_snap_radius_um": tip_snap_radius_um,
        "tip_snap_minimum_direction_cosine_strict_gt": (
            TIP_SNAP_MINIMUM_DIRECTION_COSINE
        ),
        "length_scale_um": LENGTH_SCALE_UM,
        "base_length_log_clip": list(BASE_LENGTH_LOG_CLIP),
        "base_and_tip_offset_clip": list(OFFSET_CLIP),
        "direction_normalization_epsilon": DIRECTION_NORMALIZATION_EPSILON,
        "presence_proxy_minimum_length_working_px_strict_gt": (
            PRESENCE_PROXY_MINIMUM_LENGTH_WORKING_PX
        ),
        "root_foreground_probability_threshold_strict_gt": (
            ROOT_FOREGROUND_PROBABILITY_THRESHOLD
        ),
        "root_gate_empty_foreground_policy": "do_not_filter_candidates",
        "bilinear_upper_border_epsilon_px": BILINEAR_UPPER_BORDER_EPSILON_PX,
        "base_candidate_order": (
            "descending_base_score_then_stable_heatmap_row_major_yx"
        ),
        "tip_snap_nearest_tie_order": "tip_heatmap_row_major_yx",
    }


def locked_biological_presence_candidate_decoder_contract() -> dict[str, Any]:
    """Return the only decoder contract accepted for formal pool generation."""

    return biological_presence_candidate_decoder_contract(
        working_um_per_px=HAIR_WORKING_UM_PER_PX,
        out_stride=HAIR_OUT_STRIDE,
        nms_kernel=HAIR_NMS_KERNEL,
        max_instances=HAIR_MAX_INSTANCES,
        root_gate_um=HAIR_ROOT_GATE_UM,
        tip_score_floor=TIP_HEATMAP_SCORE_FLOOR,
        tip_snap_radius_um=TIP_SNAP_RADIUS_UM,
    )


__all__ = [
    "BASE_LENGTH_LOG_CLIP",
    "BILINEAR_UPPER_BORDER_EPSILON_PX",
    "CANDIDATE_DECODER_CONTRACT_SCHEMA",
    "DIRECTION_NORMALIZATION_EPSILON",
    "LENGTH_SCALE_UM",
    "OFFSET_CLIP",
    "PRESENCE_PROXY_MINIMUM_LENGTH_WORKING_PX",
    "ROOT_FOREGROUND_PROBABILITY_THRESHOLD",
    "TIP_HEATMAP_SCORE_FLOOR",
    "TIP_SNAP_MINIMUM_DIRECTION_COSINE",
    "TIP_SNAP_RADIUS_UM",
    "biological_presence_candidate_decoder_contract",
    "locked_biological_presence_candidate_decoder_contract",
]
