"""Phenotype recomposition across the locked root and hair experts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from .constants import PUBLIC_HAIR_LENGTH_SEMANTICS
from .contracts import ContractError


def load_axis_geometry(
    path: str | Path, *, expected_image_sha256: str
) -> dict[str, np.ndarray | float]:
    with np.load(Path(path), allow_pickle=False) as payload:
        stored_sha = str(np.asarray(payload["source_image_sha256"]).item())
        if stored_sha.casefold() != expected_image_sha256.casefold():
            raise ContractError(f"axis/source hash mismatch: {path}")
        path_xy = np.asarray(payload["path_xy"], dtype=np.float64)
        distance = np.asarray(payload["distance_from_tip_px"], dtype=np.float64)
        radius = np.asarray(payload["radius_px"], dtype=np.float64)
    if path_xy.ndim != 2 or path_xy.shape[1] != 2 or len(path_xy) < 2:
        raise ContractError(f"invalid axis path: {path}")
    if distance.shape != (len(path_xy),) or radius.shape != (len(path_xy),):
        raise ContractError(f"invalid axis arrays: {path}")
    return {"path_xy": path_xy, "distance_from_tip_px": distance, "radius_px": radius}


def attach_hairs_to_axis(
    hairs: Sequence[dict[str, Any]],
    axis: Mapping[str, Any],
    *,
    um_per_px: float,
    maximum_boundary_error_um: float = 40.0,
) -> dict[str, Any]:
    """Project Stage-B bases onto the immutable Hybrid-Max root axis.

    The projection measures attachment-derived phenotypes; it does not delete a
    Stage-B identity. Stage B remains the formally validated identity/count
    expert, while the Hybrid axis supplies the common biological coordinate frame.
    """

    if not hairs:
        return {
            "valid_fraction": None,
            "first_hair_distance_from_tip_um": None,
            "hair_zone_length_um": None,
        }
    points = np.stack([np.asarray(hair["points_xy"][0], dtype=np.float64) for hair in hairs])
    path_xy = np.asarray(axis["path_xy"], dtype=np.float64)
    radius_px = np.asarray(axis["radius_px"], dtype=np.float64)
    axial_px = np.asarray(axis["distance_from_tip_px"], dtype=np.float64)
    distances_px, indices = cKDTree(path_xy).query(points, k=1)
    boundary_error_um = np.abs(distances_px - radius_px[indices]) * float(um_per_px)
    valid = np.isfinite(axial_px[indices]) & (boundary_error_um <= maximum_boundary_error_um)
    axial_um = axial_px[indices] * float(um_per_px)
    for hair, index, centre_distance, boundary_error, distance_um, accepted in zip(
        hairs,
        indices,
        distances_px,
        boundary_error_um,
        axial_um,
        valid,
        strict=True,
    ):
        hair.update(
            {
                "root_axis_projection_xy": path_xy[int(index)].tolist(),
                "root_axis_centre_distance_um": float(centre_distance * um_per_px),
                "root_boundary_attachment_error_um": float(boundary_error),
                "root_axis_distance_from_tip_um": float(distance_um),
                "root_attachment_valid": bool(accepted),
                "root_attachment_projection_mode": "hybrid_axis_local_radius",
            }
        )
    valid_axial = axial_um[valid]
    return {
        "valid_fraction": float(valid.mean()),
        "first_hair_distance_from_tip_um": (
            float(valid_axial.min()) if valid_axial.size else None
        ),
        "hair_zone_length_um": (
            float(valid_axial.max() - valid_axial.min())
            if valid_axial.size >= 2
            else 0.0 if valid_axial.size == 1 else None
        ),
    }


def associate_endpoint_complete_lengths(
    identity_hairs: list[dict[str, Any]],
    hybrid_length_hairs: Sequence[Mapping[str, Any]],
    *,
    um_per_px: float,
    maximum_base_distance_um: float = 20.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One-to-one link endpoint-complete Hybrid curves to Stage-B identities."""

    if not identity_hairs or not hybrid_length_hairs:
        return [], {
            "stageb_identities": len(identity_hairs),
            "hybrid_endpoint_complete_candidates": len(hybrid_length_hairs),
            "matched_length_identities": 0,
            "maximum_base_distance_um": float(maximum_base_distance_um),
        }
    identity_bases = np.stack(
        [np.asarray(hair["points_xy"][0], dtype=np.float64) for hair in identity_hairs]
    )
    length_bases = np.stack(
        [
            np.asarray(candidate["points_xy"][0], dtype=np.float64)
            for candidate in hybrid_length_hairs
        ]
    )
    distances_um = (
        np.linalg.norm(identity_bases[:, None, :] - length_bases[None, :, :], axis=-1)
        * float(um_per_px)
    )
    infeasible = 1e9
    costs = np.where(distances_um <= maximum_base_distance_um, distances_um, infeasible)
    identity_indices, length_indices = linear_sum_assignment(costs)
    accepted = costs[identity_indices, length_indices] <= maximum_base_distance_um
    identity_indices = identity_indices[accepted]
    length_indices = length_indices[accepted]
    order = np.argsort(identity_indices)
    matched: list[dict[str, Any]] = []
    for identity_index, length_index in zip(
        identity_indices[order], length_indices[order], strict=True
    ):
        identity = identity_hairs[int(identity_index)]
        distance_um = float(distances_um[int(identity_index), int(length_index)])
        identity["complete_length_measurement_eligible"] = True
        identity["length_measurement_source_instance_id"] = f"HML-{int(length_index) + 1:04d}"
        identity["length_identity_base_match_error_um"] = distance_um
        candidate = deepcopy(dict(hybrid_length_hairs[int(length_index)]))
        candidate.update(
            {
                "phaxis_length_source": "hybrid_max_endpoint_complete_centerline",
                "source_instance_id": f"HML-{int(length_index) + 1:04d}",
                "identity_source_instance_id": identity["source_instance_id"],
                "identity_base_match_error_um": distance_um,
                "complete_length_measurement_eligible": True,
            }
        )
        matched.append(candidate)
    return matched, {
        "stageb_identities": len(identity_hairs),
        "hybrid_endpoint_complete_candidates": len(hybrid_length_hairs),
        "matched_length_identities": len(matched),
        "maximum_base_distance_um": float(maximum_base_distance_um),
        "matching": "one_to_one_hungarian_on_predicted_base_distance",
    }


def _polyline_lengths_um(
    length_hairs: Sequence[Mapping[str, Any]], *, um_per_px: float
) -> np.ndarray:
    lengths = []
    for hair in length_hairs:
        points = np.asarray(hair["points_xy"], dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.all(np.isfinite(points))
        ):
            raise ContractError("invalid endpoint-complete hair polyline")
        length_um = float(
            np.linalg.norm(np.diff(points, axis=0), axis=1).sum() * um_per_px
        )
        if not np.isfinite(length_um) or length_um <= 0.0:
            raise ContractError("endpoint-complete hair polyline must have positive length")
        lengths.append(length_um)
    return np.asarray(lengths, dtype=np.float64)


def _update_tier(
    tier: Mapping[str, Any],
    *,
    hair_count: int,
    attachment: Mapping[str, Any],
    length_values_um: np.ndarray,
) -> dict[str, Any]:
    result = deepcopy(dict(tier))
    root_length_um = result.get("main_root_length_um")
    root_length_um = float(root_length_um) if root_length_um is not None else float("nan")
    result["hair_count"] = float(hair_count)
    result["hair_density_per_mm"] = (
        float(hair_count / (root_length_um / 1000.0))
        if np.isfinite(root_length_um) and root_length_um > 0
        else None
    )
    result["first_hair_distance_from_tip_um"] = attachment[
        "first_hair_distance_from_tip_um"
    ]
    result["hair_zone_length_um"] = attachment["hair_zone_length_um"]
    attachment_fraction = attachment["valid_fraction"]
    if hair_count == 0:
        if attachment_fraction is not None:
            raise ContractError(
                "zero hair identities require null attachment-axis support"
            )
    else:
        try:
            attachment_fraction = float(attachment_fraction)
        except (TypeError, ValueError) as error:
            raise ContractError(
                "positive hair identity count requires numeric attachment-axis support"
            ) from error
        if not np.isfinite(attachment_fraction) or not 0.0 <= attachment_fraction <= 1.0:
            raise ContractError(
                "attachment-axis support must be finite within [0,1]"
            )
    result["attachment_axis_valid_fraction"] = attachment_fraction
    result["attachment_axis_projection_mode"] = "hybrid_axis_local_radius"
    length_measurement_count = int(len(length_values_um))
    if length_measurement_count > hair_count:
        raise ContractError("endpoint-complete length support exceeds identity count")
    result["mean_hair_length_um"] = (
        float(length_values_um.mean()) if length_measurement_count else None
    )
    result["median_hair_length_um"] = (
        float(np.median(length_values_um)) if length_measurement_count else None
    )
    result["total_hair_length_um"] = (
        float(length_values_um.sum())
        if length_measurement_count
        else 0.0
        if hair_count == 0
        else None
    )
    result["hair_length_measurement_hair_count"] = float(length_measurement_count)
    result["hair_length_measurement_fraction"] = (
        float(length_measurement_count / hair_count) if hair_count else None
    )
    result["hair_length_semantics"] = PUBLIC_HAIR_LENGTH_SEMANTICS
    result["total_hair_length_is_partial"] = bool(length_measurement_count != hair_count)
    result["hair_identity_semantics"] = "rhaxiscc_stage_b_base_anchored_presence"
    return result


def recompose_hair_phenotypes(
    prediction: dict[str, Any],
    *,
    attachment: Mapping[str, Any],
    um_per_px: float,
) -> None:
    target_name = "phenotypes" if isinstance(prediction.get("phenotypes"), Mapping) else "phenotypes_review_only"
    container = prediction.get(target_name)
    if not isinstance(container, Mapping):
        return
    updated = deepcopy(dict(container))
    hair_count = len(prediction["identity_hairs"])
    length_values_um = _polyline_lengths_um(
        prediction.get("length_hairs", ()), um_per_px=um_per_px
    )
    for tier_name in ("identity_tier", "count_tier"):
        tier = updated.get(tier_name)
        if isinstance(tier, Mapping):
            updated[tier_name] = _update_tier(
                tier,
                hair_count=hair_count,
                attachment=attachment,
                length_values_um=length_values_um,
            )
    prediction[target_name] = updated
