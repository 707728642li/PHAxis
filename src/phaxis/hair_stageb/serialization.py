"""Dependency-light Stage B detection interchange serialization."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..constants import (
    HAIR_BATCH,
    LEGACY_HAIR_EXPERT_ID,
    HAIR_MAX_INSTANCES,
    HAIR_NMS_KERNEL,
    HAIR_OVERLAP,
    HAIR_ROOT_GATE_UM,
    HAIR_SCORE_THRESHOLD,
    HAIR_WINDOW,
    HAIR_WORKING_UM_PER_PX,
    STAGEB_SCHEMA,
)
from ..io import sha256_json
from .candidate_bundle import (
    PREREGISTERED_SCORE_THRESHOLDS,
    TRAIN399_CHECKPOINT_POLICY,
    validate_train399_detection_model_metadata,
)


def make_detection_payload(
    *,
    task_id: str,
    source_image_sha256: str,
    source_um_per_px: float,
    prediction: dict[str, Any],
    precision_mode: str,
    model_metadata: Mapping[str, Any] | None = None,
    score_threshold: float | None = None,
    operating_point_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    detections = []
    has_tip_snapped = "tip_snapped" in prediction
    length_semantics = prediction.get("length_semantics")
    if length_semantics is not None and length_semantics != (
        "regressed_polyline_arc_length_um_diagnostic_only"
    ):
        raise ValueError("unsupported Stage-B predicted-length semantics")
    for index, (base, tip, score, length) in enumerate(
        zip(
            prediction["base"],
            prediction["tip"],
            prediction["score"],
            prediction["length_um"],
            strict=True,
        )
    ):
        detection = {
            "base_xy_working": np.asarray(base, dtype=float).tolist(),
            "tip_xy_working": np.asarray(tip, dtype=float).tolist(),
            "score": float(score),
            "predicted_length_um": float(length),
        }
        if has_tip_snapped:
            detection["tip_snapped"] = bool(prediction["tip_snapped"][index])
        if length_semantics is not None:
            detection["predicted_length_semantics"] = str(length_semantics)
        detections.append(detection)
    if model_metadata is None:
        if score_threshold is not None and not np.isclose(
            float(score_threshold), HAIR_SCORE_THRESHOLD, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "legacy 443CV detection payloads are locked to score threshold 0.225"
            )
        if operating_point_metadata is not None:
            raise ValueError(
                "legacy 443CV detection payloads do not accept new operating-point metadata"
            )
        model = {
            "expert_id": LEGACY_HAIR_EXPERT_ID,
            "ensemble_members": 5,
            "checkpoint_policy": "five_fold_last_epoch_60",
            "precision_mode": precision_mode,
        }
        selected_threshold = HAIR_SCORE_THRESHOLD
    else:
        model = dict(model_metadata)
        recorded_precision = model.get("precision_mode")
        if recorded_precision is not None and recorded_precision != precision_mode:
            raise ValueError("model metadata precision_mode conflicts with inference")
        model["precision_mode"] = precision_mode
        validate_train399_detection_model_metadata(model)
        if score_threshold is None:
            raise ValueError(
                "train399 detection payloads require an explicit selected score threshold"
            )
        selected_threshold = float(score_threshold)
        if not any(
            abs(selected_threshold - value) <= 1e-12
            for value in PREREGISTERED_SCORE_THRESHOLDS
        ):
            raise ValueError("train399 score threshold is outside the preregistered grid")
        if not np.isclose(
            selected_threshold,
            float(model["selected_score_threshold"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("payload threshold differs from selected model metadata")

    operating_point: dict[str, Any] = {
        "score_threshold": selected_threshold,
        "nms_kernel": HAIR_NMS_KERNEL,
        "horizontal_flip_tta": True,
        "use_trace": False,
        "root_gate_um": list(HAIR_ROOT_GATE_UM),
        "window": HAIR_WINDOW,
        "overlap": HAIR_OVERLAP,
        "batch": HAIR_BATCH,
        "max_instances": HAIR_MAX_INSTANCES,
    }
    if operating_point_metadata is not None:
        protected = set(operating_point) & set(operating_point_metadata)
        if protected:
            raise ValueError(
                "operating_point_metadata cannot override runtime locks: "
                + ", ".join(sorted(protected))
            )
        operating_point.update(dict(operating_point_metadata))

    coordinate_space: dict[str, Any] = {
        "working_um_per_px": HAIR_WORKING_UM_PER_PX,
        "source_um_per_px": float(source_um_per_px),
        "source_to_working_scale": float(
            prediction.get(
                "source_to_working_scale",
                source_um_per_px / HAIR_WORKING_UM_PER_PX,
            )
        ),
        "working_shape": prediction.get("working_shape"),
    }
    realized_fields = (
        "source_shape",
        "source_to_working_scale_xy",
        "realized_um_per_px_xy",
    )
    present_realized = [field in prediction for field in realized_fields]
    if any(present_realized) and not all(present_realized):
        raise ValueError("realized resize geometry must be supplied as one complete lock")
    if all(present_realized):
        coordinate_space.update(
            {
                field: np.asarray(prediction[field]).tolist()
                for field in realized_fields
            }
        )

    payload: dict[str, Any] = {
        "schema_version": STAGEB_SCHEMA,
        "task_id": task_id,
        "source_image_sha256": source_image_sha256,
        "coordinate_space": coordinate_space,
        "model": model,
        "operating_point": operating_point,
        "detections": detections,
        "n": len(detections),
        "blind_images_used": 0,
    }
    payload["detection_identity_sha256"] = sha256_json(payload)
    return payload
