"""Runtime contracts that prevent accidental cross-expert regression."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .constants import (
    BLIND_IMAGES_USED,
    HAIR_BATCH,
    LEGACY_HAIR_EXPERT_ID,
    HAIR_MAX_INSTANCES,
    HAIR_NMS_KERNEL,
    HAIR_OVERLAP,
    HAIR_ROOT_GATE_UM,
    HAIR_SCORE_THRESHOLD,
    HAIR_WINDOW,
    HAIR_WORKING_UM_PER_PX,
    ROOT_CAP_REGION_OUTPUT,
    ROOT_LOCK_TOP_LEVEL_FIELDS,
    ROOT_PHENOTYPE_FIELDS,
    STAGEB_SCHEMA,
)
from .io import sha256_file, sha256_json
from .errors import ContractError
from .hair_stageb.candidate_bundle import (
    CandidateBundleError,
    TRAIN399_CHECKPOINT_POLICY,
    validate_train399_detection_model_metadata,
)


def _existing_fields(payload: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: deepcopy(payload[name]) for name in names if name in payload}


def root_lock_payload(prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Return every root/scale value that Stage B is forbidden to alter."""

    locked = _existing_fields(prediction, ROOT_LOCK_TOP_LEVEL_FIELDS)
    phenotype_payload: dict[str, Any] = {}
    for container_name in ("phenotypes", "phenotypes_review_only"):
        container = prediction.get(container_name)
        if not isinstance(container, Mapping):
            continue
        tiers: dict[str, Any] = {}
        for tier_name, tier in container.items():
            if isinstance(tier, Mapping):
                tiers[str(tier_name)] = _existing_fields(tier, ROOT_PHENOTYPE_FIELDS)
        phenotype_payload[container_name] = tiers
    locked["phenotype_root_fields"] = phenotype_payload
    return locked


def root_lock_sha256(prediction: Mapping[str, Any]) -> str:
    return sha256_json(root_lock_payload(prediction))


def assert_root_lock_unchanged(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    expected = root_lock_sha256(before)
    observed = root_lock_sha256(after)
    if observed != expected:
        raise ContractError(
            f"root/point/scale/statistics lock changed: expected {expected}, got {observed}"
        )


def validate_hybrid_prediction(
    prediction: Mapping[str, Any], *, artifact_root: str | Path | None = None
) -> None:
    task_id = prediction.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ContractError("Hybrid-Max prediction has no task_id")
    if int(prediction.get("blind_images_used", -1)) != BLIND_IMAGES_USED:
        raise ContractError(f"{task_id}: blind_images_used must be 0")
    if prediction.get("root_cap_region_output") is not ROOT_CAP_REGION_OUTPUT:
        raise ContractError(f"{task_id}: root-cap region output is forbidden")
    point = np.asarray(prediction.get("root_cap_point_xy"), dtype=np.float64)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ContractError(f"{task_id}: invalid distal/root-cap point")
    for relpath_name, digest_name in (
        ("root_mask_relpath", "root_mask_sha256"),
        ("root_axis_geometry_relpath", "root_axis_geometry_sha256"),
    ):
        if not prediction.get(relpath_name) or not prediction.get(digest_name):
            raise ContractError(f"{task_id}: missing {relpath_name}/{digest_name}")
        if artifact_root is not None:
            path = Path(artifact_root) / str(prediction[relpath_name])
            if not path.is_file():
                raise ContractError(f"{task_id}: missing locked artifact: {path}")
            observed = sha256_file(path)
            if observed.casefold() != str(prediction[digest_name]).casefold():
                raise ContractError(f"{task_id}: artifact hash mismatch: {path}")


def validate_stageb_detection_payload(
    payload: Mapping[str, Any],
    *,
    expected_task_id: str,
    expected_image_sha256: str,
    expected_model_metadata: Mapping[str, Any] | None = None,
) -> None:
    if payload.get("schema_version") != STAGEB_SCHEMA:
        raise ContractError("unsupported Stage-B detection schema")
    if payload.get("task_id") != expected_task_id:
        raise ContractError("Stage-B/Hybrid task_id mismatch")
    if str(payload.get("source_image_sha256", "")).casefold() != expected_image_sha256.casefold():
        raise ContractError("Stage-B/Hybrid source image hash mismatch")
    if int(payload.get("blind_images_used", -1)) != BLIND_IMAGES_USED:
        raise ContractError("Stage-B blind_images_used must be 0")
    coordinate_space = payload.get("coordinate_space")
    if not isinstance(coordinate_space, Mapping):
        raise ContractError("Stage-B coordinate_space is missing")
    for field in ("working_um_per_px", "source_um_per_px"):
        value = float(coordinate_space.get(field, float("nan")))
        if not np.isfinite(value) or value <= 0:
            raise ContractError(f"invalid Stage-B coordinate field: {field}")
    if not np.isclose(
        float(coordinate_space["working_um_per_px"]),
        HAIR_WORKING_UM_PER_PX,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ContractError("Stage-B working_um_per_px differs from the locked model")
    source_to_working = float(
        coordinate_space.get(
            "source_to_working_scale",
            float(coordinate_space["source_um_per_px"]) / HAIR_WORKING_UM_PER_PX,
        )
    )
    expected_scale = float(coordinate_space["source_um_per_px"]) / HAIR_WORKING_UM_PER_PX
    if not np.isfinite(source_to_working) or not np.isclose(
        source_to_working, expected_scale, rtol=1e-8, atol=1e-10
    ):
        raise ContractError("invalid Stage-B source_to_working_scale")
    realized_extension_fields = (
        "source_shape",
        "source_to_working_scale_xy",
        "realized_um_per_px_xy",
    )
    realized_present = [
        coordinate_space.get(field) is not None for field in realized_extension_fields
    ]
    if any(realized_present) and not all(realized_present):
        raise ContractError("Stage-B realized resize geometry is only partially present")
    has_realized_geometry = all(realized_present)
    if has_realized_geometry:
        if coordinate_space.get("working_shape") is None:
            raise ContractError("Stage-B realized resize geometry has no working_shape")
        source_shape = np.asarray(coordinate_space["source_shape"], dtype=np.float64)
        working_shape = np.asarray(coordinate_space["working_shape"], dtype=np.float64)
        scale_xy = np.asarray(
            coordinate_space["source_to_working_scale_xy"], dtype=np.float64
        )
        realized_um_per_px_xy = np.asarray(
            coordinate_space["realized_um_per_px_xy"], dtype=np.float64
        )
        for name, value in (
            ("source_shape", source_shape),
            ("working_shape", working_shape),
            ("source_to_working_scale_xy", scale_xy),
            ("realized_um_per_px_xy", realized_um_per_px_xy),
        ):
            if value.shape != (2,) or not np.all(np.isfinite(value)) or np.any(value <= 0):
                raise ContractError(f"invalid Stage-B realized geometry field: {name}")
        if not np.all(source_shape == np.floor(source_shape)) or not np.all(
            working_shape == np.floor(working_shape)
        ):
            raise ContractError("Stage-B source/working shapes must contain integers")
        expected_scale_xy = np.asarray(
            [
                working_shape[1] / source_shape[1],
                working_shape[0] / source_shape[0],
            ],
            dtype=np.float64,
        )
        if not np.allclose(scale_xy, expected_scale_xy, rtol=0.0, atol=1e-12):
            raise ContractError(
                "Stage-B source_to_working_scale_xy differs from source/working shapes"
            )
        expected_realized = float(coordinate_space["source_um_per_px"]) / scale_xy
        if not np.allclose(
            realized_um_per_px_xy, expected_realized, rtol=1e-12, atol=1e-12
        ):
            raise ContractError(
                "Stage-B realized_um_per_px_xy differs from the realized resize"
            )

    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise ContractError("Stage-B model lock is missing")
    checkpoint_policy = model.get("checkpoint_policy")
    if checkpoint_policy == "five_fold_last_epoch_60":
        expected_model = {
            "expert_id": LEGACY_HAIR_EXPERT_ID,
            "ensemble_members": 5,
            "checkpoint_policy": "five_fold_last_epoch_60",
        }
        for field, expected in expected_model.items():
            if model.get(field) != expected:
                raise ContractError(f"Stage-B model lock mismatch: {field}")
        selected_score_threshold = HAIR_SCORE_THRESHOLD
    elif checkpoint_policy == TRAIN399_CHECKPOINT_POLICY:
        try:
            validate_train399_detection_model_metadata(model)
        except CandidateBundleError as error:
            raise ContractError(f"invalid train399 Stage-B model metadata: {error}") from error
        selected_score_threshold = float(model["selected_score_threshold"])
        if not has_realized_geometry:
            raise ContractError("Stage-B train399 payload lacks realized resize geometry")
    else:
        raise ContractError("Stage-B model lock mismatch: checkpoint_policy")
    if model.get("precision_mode") not in {"fp32_locked", "fp32_locked_oof"}:
        raise ContractError("Stage-B precision_mode is not locked FP32")
    if expected_model_metadata is not None:
        for field, expected in expected_model_metadata.items():
            if model.get(field) != expected:
                raise ContractError(f"Stage-B expected model metadata mismatch: {field}")

    operating_point = payload.get("operating_point")
    if not isinstance(operating_point, Mapping):
        raise ContractError("Stage-B operating point is missing")
    expected_operating_point = {
        "score_threshold": selected_score_threshold,
        "nms_kernel": HAIR_NMS_KERNEL,
        "horizontal_flip_tta": True,
        "use_trace": False,
        "root_gate_um": list(HAIR_ROOT_GATE_UM),
        "window": HAIR_WINDOW,
        "overlap": HAIR_OVERLAP,
        "batch": HAIR_BATCH,
        "max_instances": HAIR_MAX_INSTANCES,
    }
    for field, expected in expected_operating_point.items():
        observed = operating_point.get(field)
        if isinstance(expected, float):
            matches = np.isclose(float(observed), expected, rtol=0.0, atol=1e-12)
        else:
            matches = observed == expected
        if not matches:
            raise ContractError(f"Stage-B operating point mismatch: {field}")
    detections = payload.get("detections")
    if not isinstance(detections, list):
        raise ContractError("Stage-B detections must be a list")
    if int(payload.get("n", -1)) != len(detections):
        raise ContractError("Stage-B n does not match the detection list")
    for index, detection in enumerate(detections):
        if not isinstance(detection, Mapping):
            raise ContractError(f"Stage-B detection {index} is not an object")
        for field in ("base_xy_working", "tip_xy_working"):
            point = np.asarray(detection.get(field), dtype=np.float64)
            if point.shape != (2,) or not np.all(np.isfinite(point)):
                raise ContractError(f"Stage-B detection {index} has invalid {field}")
        score = float(detection.get("score", float("nan")))
        if not np.isfinite(score) or score < 0.0 or score > 1.0:
            raise ContractError(f"Stage-B detection {index} has invalid score")
        length = float(detection.get("predicted_length_um", float("nan")))
        if not np.isfinite(length) or length < 0.0:
            raise ContractError(
                f"Stage-B detection {index} has invalid predicted_length_um"
            )
        if "tip_snapped" in detection and not isinstance(
            detection["tip_snapped"], bool
        ):
            raise ContractError(f"Stage-B detection {index} has invalid tip_snapped")
        semantics = detection.get("predicted_length_semantics")
        if semantics is not None and semantics != (
            "regressed_polyline_arc_length_um_diagnostic_only"
        ):
            raise ContractError(
                f"Stage-B detection {index} has invalid predicted-length semantics"
            )
        if checkpoint_policy == TRAIN399_CHECKPOINT_POLICY and (
            "tip_snapped" not in detection or semantics is None
        ):
            raise ContractError(
                f"Stage-B train399 detection {index} lacks geometry semantics"
            )
    unsigned = deepcopy(dict(payload))
    observed_identity = str(unsigned.pop("detection_identity_sha256", ""))
    if len(observed_identity) != 64 or sha256_json(unsigned) != observed_identity:
        raise ContractError("Stage-B detection_identity_sha256 mismatch")
