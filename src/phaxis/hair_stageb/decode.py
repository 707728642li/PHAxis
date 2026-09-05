"""Base-anchored decoding for the locked Stage B operating point."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from scipy.ndimage import distance_transform_edt

from .candidate_pool_contract import (
    BASE_LENGTH_LOG_CLIP,
    BILINEAR_UPPER_BORDER_EPSILON_PX,
    DIRECTION_NORMALIZATION_EPSILON,
    LENGTH_SCALE_UM,
    OFFSET_CLIP,
    PRESENCE_PROXY_MINIMUM_LENGTH_WORKING_PX,
    ROOT_FOREGROUND_PROBABILITY_THRESHOLD,
    TIP_HEATMAP_SCORE_FLOOR,
    TIP_SNAP_MINIMUM_DIRECTION_COSINE,
    TIP_SNAP_RADIUS_UM,
    biological_presence_candidate_decoder_contract,
)


def _nms(heatmap: torch.Tensor, kernel: int) -> torch.Tensor:
    keep = functional.max_pool2d(
        heatmap, kernel, stride=1, padding=kernel // 2
    ) == heatmap
    return heatmap * keep


def _bilinear_sample(array: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    channels, height, width = array.shape
    x = np.clip(
        points_xy[:, 0], 0, width - 1.0 - BILINEAR_UPPER_BORDER_EPSILON_PX
    )
    y = np.clip(
        points_xy[:, 1], 0, height - 1.0 - BILINEAR_UPPER_BORDER_EPSILON_PX
    )
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    wx, wy = x - x0, y - y0
    output = (
        array[:, y0, x0] * (1 - wx) * (1 - wy)
        + array[:, y0, x1] * wx * (1 - wy)
        + array[:, y1, x0] * (1 - wx) * wy
        + array[:, y1, x1] * wx * wy
    )
    return output.T.reshape(len(points_xy), channels)


def decode_instances(
    heads: dict[str, np.ndarray],
    *,
    um_per_px: float,
    out_stride: int,
    score_threshold: float,
    nms_kernel: int,
    max_instances: int,
    root_gate_um: tuple[float, float] | None,
    tip_snap_radius_um: float = TIP_SNAP_RADIUS_UM,
    tip_score_floor: float = TIP_HEATMAP_SCORE_FLOOR,
) -> dict[str, np.ndarray | int | str]:
    if not 0.0 < float(tip_score_floor) < 1.0:
        raise ValueError("tip_score_floor must be between zero and one")
    base_heatmap = _nms(
        torch.from_numpy(heads["base_hm"]).sigmoid()[None], nms_kernel
    )[0, 0].numpy()
    ys, xs = np.nonzero(base_heatmap >= score_threshold)
    if not len(ys):
        empty_points = np.zeros((0, 2), dtype=np.float32)
        return {
            "base": empty_points,
            "tip": empty_points.copy(),
            "score": np.zeros(0, dtype=np.float32),
            "length_um": np.zeros(0, dtype=np.float32),
            "tip_snapped": np.zeros(0, dtype=bool),
            "length_semantics": "regressed_polyline_arc_length_um_diagnostic_only",
            "n": 0,
        }
    scores = base_heatmap[ys, xs]
    # Stable sorting preserves the row-major heatmap order for exact score
    # ties, including at the max-instance truncation boundary.
    order = np.argsort(-scores, kind="stable")[:max_instances]
    ys, xs, scores = ys[order], xs[order], scores[order]
    offsets = heads["base_off"]
    base_output = np.column_stack(
        (
            xs + np.clip(offsets[0, ys, xs], *OFFSET_CLIP),
            ys + np.clip(offsets[1, ys, xs], *OFFSET_CLIP),
        )
    )
    direction = np.column_stack(
        (heads["base_dir"][0, ys, xs], heads["base_dir"][1, ys, xs])
    )
    direction /= np.maximum(
        np.linalg.norm(direction, axis=1, keepdims=True),
        DIRECTION_NORMALIZATION_EPSILON,
    )
    length_um = (
        np.exp(np.clip(heads["base_len"][0, ys, xs], *BASE_LENGTH_LOG_CLIP))
        * LENGTH_SCALE_UM
    )
    output_um_per_px = um_per_px * out_stride
    tip_output = base_output + direction * (length_um / output_um_per_px)[:, None]
    tip_snapped = np.zeros(len(tip_output), dtype=bool)

    tip_heatmap = _nms(
        torch.from_numpy(heads["tip_hm"]).sigmoid()[None], nms_kernel
    )[0, 0].numpy()
    # The tip-proxy decoder is deliberately independent of the base-score
    # operating point.  This makes a threshold sweep a pure filter over one
    # immutable base->tip candidate pool rather than silently changing distal
    # geometry at the highest base threshold.
    tip_ys, tip_xs = np.nonzero(tip_heatmap >= float(tip_score_floor))
    if len(tip_ys):
        tip_offsets = heads["tip_off"]
        tip_candidates = np.column_stack(
            (
                tip_xs + np.clip(tip_offsets[0, tip_ys, tip_xs], *OFFSET_CLIP),
                tip_ys + np.clip(tip_offsets[1, tip_ys, tip_xs], *OFFSET_CLIP),
            )
        )
        distances = np.linalg.norm(
            tip_output[:, None, :] - tip_candidates[None, :, :], axis=-1
        )
        nearest = np.argmin(distances, axis=1)
        accepted = distances[np.arange(len(tip_output)), nearest] <= (
            tip_snap_radius_um / output_um_per_px
        )
        vectors = tip_candidates[nearest] - base_output
        vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        cosine = (
            vectors
            / np.maximum(vector_norms, DIRECTION_NORMALIZATION_EPSILON)
            * direction
        ).sum(axis=1)
        accepted &= cosine > TIP_SNAP_MINIMUM_DIRECTION_COSINE
        tip_output[accepted] = tip_candidates[nearest][accepted]
        tip_snapped[accepted] = True

    keep = np.ones(len(base_output), dtype=bool)
    if root_gate_um is not None and "root" in heads:
        root = (
            1.0 / (1.0 + np.exp(-heads["root"][0]))
            > ROOT_FOREGROUND_PROBABILITY_THRESHOLD
        )
        if root.any():
            signed_distance = (
                distance_transform_edt(~root) - distance_transform_edt(root)
            ) * output_um_per_px
            sampled = _bilinear_sample(
                signed_distance[None].astype(np.float32), base_output
            )[:, 0]
            keep = (sampled >= root_gate_um[0]) & (sampled <= root_gate_um[1])
    return {
        "base": (base_output[keep] * out_stride).astype(np.float32),
        "tip": (tip_output[keep] * out_stride).astype(np.float32),
        "score": scores[keep].astype(np.float32),
        "length_um": length_um[keep].astype(np.float32),
        "tip_snapped": tip_snapped[keep],
        "length_semantics": "regressed_polyline_arc_length_um_diagnostic_only",
        "n": int(keep.sum()),
    }


def decode_biological_presence_candidates(
    heads: dict[str, np.ndarray],
    *,
    um_per_px: float,
    out_stride: int,
    score_floor: float,
    nms_kernel: int,
    max_instances: int,
    root_gate_um: tuple[float, float] | None,
    tip_snap_radius_um: float = TIP_SNAP_RADIUS_UM,
    tip_score_floor: float = TIP_HEATMAP_SCORE_FLOOR,
) -> dict[str, np.ndarray | int | float | str]:
    """Decode one immutable pool for biological-presence threshold selection.

    The straight base-to-tip segment is only the current Stage-B presence
    proxy.  Its distal error, complete overlap and regressed length are never
    selection gates.  A degenerate predicted direction remains a counted
    candidate but is marked invalid, so the matcher cannot invent geometry.
    """

    decoded = decode_instances(
        heads,
        um_per_px=um_per_px,
        out_stride=out_stride,
        score_threshold=score_floor,
        nms_kernel=nms_kernel,
        max_instances=max_instances,
        root_gate_um=root_gate_um,
        tip_snap_radius_um=tip_snap_radius_um,
        tip_score_floor=tip_score_floor,
    )
    base = np.asarray(decoded["base"], dtype=np.float32).reshape(-1, 2)
    tip = np.asarray(decoded["tip"], dtype=np.float32).reshape(-1, 2)
    proxy_valid = (
        np.linalg.norm(tip - base, axis=1)
        > PRESENCE_PROXY_MINIMUM_LENGTH_WORKING_PX
    )
    decoder_contract = biological_presence_candidate_decoder_contract(
        working_um_per_px=um_per_px,
        out_stride=out_stride,
        nms_kernel=nms_kernel,
        max_instances=max_instances,
        root_gate_um=root_gate_um,
        tip_score_floor=tip_score_floor,
        tip_snap_radius_um=tip_snap_radius_um,
    )
    return {
        "base": base,
        "tip": tip,
        "score": np.asarray(decoded["score"], dtype=np.float32),
        "presence_proxy_valid": proxy_valid,
        "n": int(decoded["n"]),
        "score_floor": float(score_floor),
        "candidate_pool_decode_scope": (
            "base_score_plus_straight_base_to_tip_biological_presence_proxy"
        ),
        "presence_proxy_kind": "straight_base_to_tip",
        "distal_endpoint_or_length_used_as_selection_gate": False,
        "candidate_decoder_contract": decoder_contract,
        "network_forward_passes": 1,
    }
