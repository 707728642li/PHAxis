"""PHAxis 1.0.0 cross-expert fusion without root-geometry regression."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .constants import (
    BLIND_IMAGES_USED,
    LEGACY_HAIR_EXPERT_ID,
    PREDICTION_SCHEMA,
    PRODUCT_NAME,
    PRODUCT_VERSION,
    PUBLIC_HAIR_LENGTH_EXPERT_ID,
)
from .contracts import (
    ContractError,
    assert_root_lock_unchanged,
    root_lock_sha256,
    validate_hybrid_prediction,
    validate_stageb_detection_payload,
)
from .phenotypes import (
    associate_endpoint_complete_lengths,
    attach_hairs_to_axis,
    load_axis_geometry,
    recompose_hair_phenotypes,
)
from .public_identity import MODEL_BUNDLE_PREFIX, ROOT_EXPERT_PREFIX


TRAIN399_STAGEB_POLICY = "five_seed_train399_last_epoch_60"


def _stageb_hairs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    coordinate_space = payload["coordinate_space"]
    working_um_per_px = float(coordinate_space["working_um_per_px"])
    source_um_per_px = float(coordinate_space["source_um_per_px"])
    scale_xy_value = coordinate_space.get("source_to_working_scale_xy")
    if scale_xy_value is not None:
        source_to_working_xy = np.asarray(scale_xy_value, dtype=np.float64)
        if source_to_working_xy.shape != (2,) or np.any(source_to_working_xy <= 0):
            raise ContractError("invalid Stage-B per-axis coordinate scale")
        working_to_source_xy = 1.0 / source_to_working_xy
    else:
        working_to_source_xy = np.full(
            2, working_um_per_px / source_um_per_px, dtype=np.float64
        )
    model = payload.get("model", {})
    source_name = (
        "phaxis_stage_b_train399"
        if model.get("checkpoint_policy") == TRAIN399_STAGEB_POLICY
        else "rhaxiscc_stage_b"
    )
    instance_prefix = (
        "PHSB"
        if model.get("checkpoint_policy") == TRAIN399_STAGEB_POLICY
        else "RHCCB"
    )
    result: list[dict[str, Any]] = []
    for index, detection in enumerate(payload["detections"], start=1):
        base_working = np.asarray(detection["base_xy_working"], dtype=np.float64)
        tip_working = np.asarray(detection["tip_xy_working"], dtype=np.float64)
        base_source = base_working * working_to_source_xy
        tip_source = tip_working * working_to_source_xy
        result.append(
            {
                "points_xy": [base_source.tolist(), tip_source.tolist()],
                "source": source_name,
                "source_instance_id": f"{instance_prefix}-{index:04d}",
                "stageb_score": float(detection["score"]),
                "stageb_predicted_length_um": float(detection["predicted_length_um"]),
                "stageb_tip_snapped": detection.get("tip_snapped"),
                "stageb_predicted_length_semantics": detection.get(
                    "predicted_length_semantics"
                ),
                "stageb_base_xy_working": base_working.tolist(),
                "stageb_tip_xy_working": tip_working.tolist(),
                "complete_length_measurement_eligible": False,
            }
        )
    return result


def fuse_hybrid_root_with_stageb_hairs(
    hybrid_prediction: Mapping[str, Any],
    stageb_detections: Mapping[str, Any],
    *,
    hybrid_artifact_root: str | Path,
    model_contract_proposal: Mapping[str, str],
    model_contract_public_identity: Mapping[str, str],
    maximum_attachment_boundary_error_um: float = 40.0,
    physical_scale_contract: str = "strict_root_provider",
) -> dict[str, Any]:
    """Create one PHAxis prediction while enforcing the expert boundary.

    Hybrid-Max remains authoritative for root, distal point, scale, continuity,
    and detailed root statistics. RHAxiscc Stage B becomes authoritative only
    for root-hair identity/count. The existing endpoint-complete Hybrid hair set
    remains the PHAxis 1.0.0 length expert.
    """

    proposal_fields = dict(model_contract_proposal)
    public_identity = dict(model_contract_public_identity)
    if set(proposal_fields) != {
        "model_contract_proposal_sha256",
        "model_contract_proposal_identity_sha256",
    } or set(public_identity) != {"model_bundle_id", "root_expert_id"}:
        raise ContractError("fusion requires the complete model-contract identity")
    try:
        valid_proposal = all(
            len(value) == 64 and int(value, 16) >= 0
            for value in proposal_fields.values()
        )
    except (TypeError, ValueError):
        valid_proposal = False
    if (
        not valid_proposal
        or not str(public_identity["model_bundle_id"]).startswith(
            MODEL_BUNDLE_PREFIX
        )
        or not str(public_identity["root_expert_id"]).startswith(ROOT_EXPERT_PREFIX)
    ):
        raise ContractError("fusion model-contract identity is invalid")
    validate_hybrid_prediction(hybrid_prediction, artifact_root=hybrid_artifact_root)
    validate_stageb_detection_payload(
        stageb_detections,
        expected_task_id=str(hybrid_prediction["task_id"]),
        expected_image_sha256=str(hybrid_prediction["source_image_sha256"]),
    )
    for field, expected in {**proposal_fields, **public_identity}.items():
        if stageb_detections.get(field) != expected:
            raise ContractError(f"Stage-B/fusion model-contract mismatch: {field}")
    if physical_scale_contract not in {
        "strict_root_provider",
        "stageb_reference_evaluation",
    }:
        raise ContractError("unsupported fusion physical-scale contract")
    hybrid_scale = hybrid_prediction.get("scale")
    stageb_um_per_px = float(
        stageb_detections["coordinate_space"]["source_um_per_px"]
    )
    hybrid_um_per_px: float | None = None
    relative_scale_difference: float | None = None
    if isinstance(hybrid_scale, Mapping) and hybrid_scale.get("fail_closed") is False:
        hybrid_scale_value = hybrid_scale.get(
            "predicted_um_per_px", hybrid_scale.get("um_per_px")
        )
        if hybrid_scale_value is not None:
            hybrid_um_per_px = float(hybrid_scale_value)
            relative_scale_difference = abs(
                hybrid_um_per_px - stageb_um_per_px
            ) / stageb_um_per_px
        if physical_scale_contract == "strict_root_provider" and (
            hybrid_um_per_px is None
            or not np.isclose(
            hybrid_um_per_px,
            stageb_um_per_px,
            rtol=1e-8,
            atol=1e-10,
            )
        ):
            raise ContractError(
                "Hybrid-Max/Stage-B physical scale mismatch; refusing coordinate fusion"
            )
    root_lock_before = root_lock_sha256(hybrid_prediction)
    result = deepcopy(dict(hybrid_prediction))
    previous_identity_count = len(result.get("identity_hairs", ()))
    if "length_hairs" in result:
        hybrid_length_hairs = deepcopy(result.get("length_hairs", ()))
        hybrid_length_source_field = "length_hairs"
    else:
        hybrid_length_hairs = deepcopy(
            [
                hair
                for hair in result.get("identity_hairs", ())
                if hair.get("complete_length_measurement_eligible", True) is True
            ]
        )
        hybrid_length_source_field = "legacy_identity_hairs_complete_by_contract"
    hairs = _stageb_hairs(stageb_detections)
    axis_path = Path(hybrid_artifact_root) / str(result["root_axis_geometry_relpath"])
    axis = load_axis_geometry(
        axis_path, expected_image_sha256=str(result["source_image_sha256"])
    )
    source_um_per_px = stageb_um_per_px
    attachment = attach_hairs_to_axis(
        hairs,
        axis,
        um_per_px=source_um_per_px,
        maximum_boundary_error_um=maximum_attachment_boundary_error_um,
    )
    stageb_model = stageb_detections.get("model", {})
    train399_stageb = (
        stageb_model.get("checkpoint_policy") == TRAIN399_STAGEB_POLICY
    )
    identity_variant = (
        "phaxis_stage_b_train399_five_seed_identity"
        if train399_stageb
        else "rhaxiscc_stage_b_5fold_formal_identity"
    )
    count_variant = (
        "phaxis_stage_b_train399_five_seed_count"
        if train399_stageb
        else "rhaxiscc_stage_b_5fold_formal_count"
    )
    stageb_boundary_key = (
        "phaxis_stage_b_train399" if train399_stageb else "rhaxiscc_stage_b"
    )
    result["schema_version"] = PREDICTION_SCHEMA
    result["identity_hair_variant"] = identity_variant
    result["identity_hairs"] = hairs
    result["count_hair_variant"] = count_variant
    result["count_hairs"] = deepcopy(hairs)
    matched_length_hairs, length_association = associate_endpoint_complete_lengths(
        hairs,
        hybrid_length_hairs,
        um_per_px=source_um_per_px,
        maximum_base_distance_um=20.0,
    )
    result["identity_hairs"] = hairs
    result["count_hairs"] = deepcopy(hairs)
    result["length_hairs"] = matched_length_hairs
    result["length_hair_variant"] = (
        "hybrid_max_endpoint_complete_matched_to_"
        + ("phaxis_stage_b_train399_identity" if train399_stageb else "rhaxiscc_stage_b_identity")
    )
    result["length_hair_scope"] = (
        "Hybrid-Max endpoint-complete centreline expert; RHAxiscc Stage B "
        "or PHAxis train399 Stage-B identity vectors are excluded from length summaries"
    )
    recompose_hair_phenotypes(
        result, attachment=attachment, um_per_px=source_um_per_px
    )
    result["phaxis"] = {
        "product_name": PRODUCT_NAME,
        "version": PRODUCT_VERSION,
        "model_bundle_id": public_identity["model_bundle_id"],
        "root_expert": public_identity["root_expert_id"],
        "hair_identity_count_expert": stageb_model.get(
            "expert_id", LEGACY_HAIR_EXPERT_ID
        ),
        "hair_identity_count_checkpoint_policy": stageb_model.get(
            "checkpoint_policy"
        ),
        "hair_identity_count_candidate_bundle_identity_sha256": stageb_model.get(
            "candidate_bundle_identity_sha256"
        ),
        "hair_length_expert": PUBLIC_HAIR_LENGTH_EXPERT_ID,
        "expert_boundary": {
            "hybrid_max": [
                "main_root",
                "distal_root_cap_point",
                "scale",
                "continuity",
                "detailed_root_statistics",
                "endpoint_complete_hair_length",
            ],
            stageb_boundary_key: ["hair_identity", "hair_count"],
        },
        "root_lock_sha256": root_lock_before,
        "source_hybrid_schema_version": hybrid_prediction.get("schema_version"),
        "stageb_detection_identity_sha256": stageb_detections.get(
            "detection_identity_sha256"
        ),
        "previous_hybrid_identity_count": previous_identity_count,
        "formal_stageb_identity_count": len(hairs),
        "attachment_valid_fraction": attachment["valid_fraction"],
        "length_identity_association": length_association,
        "hybrid_length_source_field": hybrid_length_source_field,
        "physical_scale_contract": {
            "mode": physical_scale_contract,
            "root_provider_scale_um_per_px": hybrid_um_per_px,
            "stageb_source_scale_um_per_px": stageb_um_per_px,
            "relative_scale_difference": relative_scale_difference,
            "geometry_scale_authority": (
                "stageb_locked_reference_acquisition_scale"
                if physical_scale_contract == "stageb_reference_evaluation"
                else "root_provider_stageb_strict_equal"
            ),
            "root_provider_scale_output_unchanged": True,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "blind_images_used": BLIND_IMAGES_USED,
        **proposal_fields,
    }
    result["canonical_annotations_read_during_inference"] = False
    result["condition_metadata_used_for_routing"] = False
    result["blind_images_used"] = BLIND_IMAGES_USED
    assert_root_lock_unchanged(hybrid_prediction, result)
    return result
