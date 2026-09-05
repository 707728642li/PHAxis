"""Information-limited QC-development operating-point selection for Stage B."""

from __future__ import annotations

from copy import deepcopy
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..evaluation_metrics import (
    biological_hair_presence_matcher_contract,
    match_biological_hair_presence,
)
from ..io import atomic_write_json, read_json, sha256_file, sha256_json
from .canonical_ground_truth import (
    CANONICAL_GT_AUTHORITY,
    CanonicalGroundTruthError,
    load_canonical_qcdev_ground_truth,
)
from .candidate_bundle import (
    CANDIDATE_POOL_SCORE_FLOOR,
    PREREGISTERED_SCORE_THRESHOLDS,
    CandidateBundleError,
    bind_selected_operating_point,
    operating_point_selection_contract,
    read_candidate_manifest,
    validate_candidate_manifest,
    validate_train399_detection_model_metadata,
)
from .candidate_pool_contract import (
    PRESENCE_PROXY_MINIMUM_LENGTH_WORKING_PX,
    locked_biological_presence_candidate_decoder_contract,
)


CANDIDATE_POOL_SCHEMA = (
    "PHAxis-StageB-train399-QCdev44-biological-presence-candidate-pool-1.0"
)
SELECTION_RECEIPT_SCHEMA = "PHAxis-StageB-train399-QCdev44-selection-receipt-1.3"


class SelectionGateError(RuntimeError):
    """The preregistered QC-development selection contract was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionGateError(message)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.casefold()


def _coordinate_space_from_prediction(
    prediction: Mapping[str, Any], *, source_um_per_px: float
) -> dict[str, Any]:
    required = (
        "source_shape",
        "working_shape",
        "source_to_working_scale_xy",
        "realized_um_per_px_xy",
    )
    _require(
        all(field in prediction for field in required),
        "candidate pool lacks complete realized resize geometry",
    )
    coordinate = {
        "working_um_per_px": 2.0,
        "source_um_per_px": float(source_um_per_px),
        "source_shape": np.asarray(prediction["source_shape"]).tolist(),
        "working_shape": np.asarray(prediction["working_shape"]).tolist(),
        "source_to_working_scale": float(
            prediction.get("source_to_working_scale", source_um_per_px / 2.0)
        ),
        "source_to_working_scale_xy": np.asarray(
            prediction["source_to_working_scale_xy"]
        ).tolist(),
        "realized_um_per_px_xy": np.asarray(
            prediction["realized_um_per_px_xy"]
        ).tolist(),
    }
    _validate_coordinate_space(coordinate)
    return coordinate


def _validate_coordinate_space(coordinate: Mapping[str, Any]) -> None:
    source_shape = np.asarray(coordinate.get("source_shape"), dtype=np.float64)
    working_shape = np.asarray(coordinate.get("working_shape"), dtype=np.float64)
    scale_xy = np.asarray(
        coordinate.get("source_to_working_scale_xy"), dtype=np.float64
    )
    realized = np.asarray(coordinate.get("realized_um_per_px_xy"), dtype=np.float64)
    for name, value in (
        ("source_shape", source_shape),
        ("working_shape", working_shape),
        ("source_to_working_scale_xy", scale_xy),
        ("realized_um_per_px_xy", realized),
    ):
        _require(
            value.shape == (2,) and np.all(np.isfinite(value)) and np.all(value > 0),
            f"invalid candidate pool coordinate field: {name}",
        )
    _require(
        np.all(source_shape == np.floor(source_shape))
        and np.all(working_shape == np.floor(working_shape)),
        "candidate pool shapes are not integer-valued",
    )
    expected_scale = np.asarray(
        [working_shape[1] / source_shape[1], working_shape[0] / source_shape[0]]
    )
    _require(
        np.allclose(scale_xy, expected_scale, rtol=0.0, atol=1e-12),
        "candidate pool scale_xy differs from source/working shapes",
    )
    source_um_per_px = float(coordinate.get("source_um_per_px", np.nan))
    _require(
        np.isfinite(source_um_per_px) and source_um_per_px > 0,
        "invalid candidate pool source physical scale",
    )
    requested_scale = float(coordinate.get("source_to_working_scale", np.nan))
    _require(
        np.isfinite(requested_scale)
        and np.isclose(
            requested_scale,
            source_um_per_px / 2.0,
            rtol=0.0,
            atol=1e-12,
        ),
        "candidate pool requested resize scale is invalid",
    )
    _require(
        np.allclose(realized, source_um_per_px / scale_xy, rtol=1e-12, atol=1e-12),
        "candidate pool realized physical scale is invalid",
    )
    _require(
        np.isclose(
            float(coordinate.get("working_um_per_px", np.nan)),
            2.0,
            rtol=0.0,
            atol=1e-12,
        ),
        "candidate pool nominal working scale changed",
    )


def make_biological_candidate_pool_payload(
    *,
    task_id: str,
    source_image_sha256: str,
    source_um_per_px: float,
    prediction: Mapping[str, Any],
    pending_model_metadata: Mapping[str, Any],
    precision_mode: str = "fp32_locked",
) -> dict[str, Any]:
    """Serialize immutable base->tip presence proxies from one floor-0.10 pass."""

    try:
        validate_train399_detection_model_metadata(
            pending_model_metadata, allow_pending=True
        )
    except CandidateBundleError as error:
        raise SelectionGateError(str(error)) from error
    _require(isinstance(task_id, str) and bool(task_id), "candidate pool has no task ID")
    _require(_is_sha256(source_image_sha256), "candidate pool image hash is invalid")
    _require(
        pending_model_metadata.get("operating_point_status")
        == "pending_QCdevelopment44_selection",
        "candidate-pool model metadata is not pending selection",
    )
    _require(
        prediction.get("candidate_pool_decode_scope")
        == "base_score_plus_straight_base_to_tip_biological_presence_proxy",
        "candidate pool was not produced by the biological-presence decoder",
    )
    _require(
        prediction.get("network_forward_passes") == 1,
        "candidate pool was not produced by exactly one network forward pass",
    )
    _require(
        float(prediction.get("score_floor", np.nan))
        == CANDIDATE_POOL_SCORE_FLOOR,
        "candidate pool decoder used the wrong score floor",
    )
    _require(
        prediction.get("presence_proxy_kind") == "straight_base_to_tip"
        and prediction.get("distal_endpoint_or_length_used_as_selection_gate")
        is False,
        "candidate pool presence-proxy semantics changed",
    )
    expected_decoder_contract = (
        locked_biological_presence_candidate_decoder_contract()
    )
    _require(
        prediction.get("candidate_decoder_contract")
        == expected_decoder_contract,
        "candidate pool decoder parameters differ from the locked contract",
    )
    _require(
        not any(field in prediction for field in ("length_um", "length_semantics")),
        "candidate pool input contains a prohibited length-selection carrier",
    )
    _require(precision_mode == "fp32_locked", "candidate pool precision must be fp32")
    bases = np.asarray(prediction.get("base"), dtype=np.float64).reshape(-1, 2)
    tips = np.asarray(prediction.get("tip"), dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(prediction.get("score"), dtype=np.float64).reshape(-1)
    proxy_valid = np.asarray(
        prediction.get("presence_proxy_valid"), dtype=bool
    ).reshape(-1)
    _require(
        len(bases) == len(tips) == len(scores) == len(proxy_valid),
        "candidate base/tip/score/valid lengths differ",
    )
    _require(
        np.all(np.isfinite(bases)) and np.all(np.isfinite(tips)),
        "candidate presence proxies contain non-finite values",
    )
    expected_proxy_valid = (
        np.linalg.norm(tips - bases, axis=1)
        > PRESENCE_PROXY_MINIMUM_LENGTH_WORKING_PX
    )
    _require(
        np.array_equal(proxy_valid, expected_proxy_valid),
        "candidate presence-proxy validity flag is inconsistent",
    )
    _require(
        np.all(np.isfinite(scores))
        and np.all(scores >= CANDIDATE_POOL_SCORE_FLOOR)
        and np.all(scores <= 1.0),
        "candidate scores violate the preregistered floor",
    )
    coordinate = _coordinate_space_from_prediction(
        prediction, source_um_per_px=source_um_per_px
    )
    payload: dict[str, Any] = {
        "schema_version": CANDIDATE_POOL_SCHEMA,
        "artifact_role": "candidate_pool_not_final_prediction",
        "fusion_or_traits_consumption_allowed": False,
        "task_id": task_id,
        "source_image_sha256": source_image_sha256,
        "coordinate_space": coordinate,
        "model": deepcopy(dict(pending_model_metadata)),
        "candidate_pool_score_floor": CANDIDATE_POOL_SCORE_FLOOR,
        "precision_mode": precision_mode,
        "network_forward_passes_for_pool": 1,
        "threshold_operation": (
            "filter_the_same_base_tip_presence_proxy_pool_by_base_score_only"
        ),
        "presence_proxy_kind": "straight_base_to_tip",
        "distal_endpoint_or_length_used_as_selection_gate": False,
        "manual_hair_width_assumed": False,
        "primary_matcher_contract": biological_hair_presence_matcher_contract(),
        "candidate_decoder_contract": deepcopy(expected_decoder_contract),
        "candidate_decoder_contract_sha256": sha256_json(
            expected_decoder_contract
        ),
        "candidates": [
            {
                "base_xy_working": base.tolist(),
                "tip_xy_working": tip.tolist(),
                "base_score": float(score),
                "presence_proxy_valid": bool(valid),
            }
            for base, tip, score, valid in zip(
                bases, tips, scores, proxy_valid, strict=True
            )
        ],
        "n": len(scores),
        "blind_images_used": 0,
    }
    payload["candidate_pool_payload_identity_sha256"] = sha256_json(payload)
    return payload


def validate_biological_candidate_pool_payload(
    payload: Mapping[str, Any],
    *,
    expected_task_id: str,
    expected_image_sha256: str,
    expected_pending_model_metadata: Mapping[str, Any],
) -> None:
    _require(payload.get("schema_version") == CANDIDATE_POOL_SCHEMA, "wrong pool schema")
    _require(
        payload.get("artifact_role") == "candidate_pool_not_final_prediction"
        and payload.get("fusion_or_traits_consumption_allowed") is False,
        "candidate pool is not explicitly barred from final prediction consumers",
    )
    _require(payload.get("task_id") == expected_task_id, "candidate pool task mismatch")
    _require(
        str(payload.get("source_image_sha256", "")).casefold()
        == expected_image_sha256.casefold(),
        "candidate pool image hash mismatch",
    )
    _require(payload.get("blind_images_used") == 0, "candidate pool is blind-tainted")
    _require(
        payload.get("model") == expected_pending_model_metadata,
        "candidate pool model metadata differs from the candidate Gate",
    )
    _require(
        payload.get("candidate_pool_score_floor") == CANDIDATE_POOL_SCORE_FLOOR,
        "candidate pool floor changed",
    )
    _require(payload.get("precision_mode") == "fp32_locked", "candidate pool is not fp32")
    _require(
        payload.get("network_forward_passes_for_pool") == 1,
        "candidate pool was not declared as one forward pass",
    )
    _require(
        payload.get("threshold_operation")
        == "filter_the_same_base_tip_presence_proxy_pool_by_base_score_only",
        "candidate pool threshold operation changed",
    )
    _require(
        payload.get("presence_proxy_kind") == "straight_base_to_tip"
        and payload.get("distal_endpoint_or_length_used_as_selection_gate") is False
        and payload.get("manual_hair_width_assumed") is False,
        "candidate pool biological-presence semantics changed",
    )
    _require(
        payload.get("primary_matcher_contract")
        == biological_hair_presence_matcher_contract(),
        "candidate pool primary matcher contract changed",
    )
    expected_decoder_contract = (
        locked_biological_presence_candidate_decoder_contract()
    )
    _require(
        payload.get("candidate_decoder_contract") == expected_decoder_contract
        and payload.get("candidate_decoder_contract_sha256")
        == sha256_json(expected_decoder_contract),
        "candidate pool decoder contract changed",
    )
    coordinate = payload.get("coordinate_space")
    _require(isinstance(coordinate, Mapping), "candidate pool coordinate space is missing")
    _validate_coordinate_space(coordinate)
    candidates = payload.get("candidates")
    _require(isinstance(candidates, list), "candidate pool candidates are not a list")
    _require(payload.get("n") == len(candidates), "candidate pool n is inconsistent")
    for index, candidate in enumerate(candidates):
        _require(isinstance(candidate, Mapping), f"candidate {index} is not an object")
        base = np.asarray(candidate.get("base_xy_working"), dtype=np.float64)
        tip = np.asarray(candidate.get("tip_xy_working"), dtype=np.float64)
        score = float(candidate.get("base_score", np.nan))
        _require(
            base.shape == (2,)
            and tip.shape == (2,)
            and np.all(np.isfinite(base))
            and np.all(np.isfinite(tip)),
            f"candidate {index} has an invalid presence proxy",
        )
        _require(
            type(candidate.get("presence_proxy_valid")) is bool
            and candidate["presence_proxy_valid"]
            == bool(
                np.linalg.norm(tip - base)
                > PRESENCE_PROXY_MINIMUM_LENGTH_WORKING_PX
            ),
            f"candidate {index} has an inconsistent proxy-valid flag",
        )
        _require(
            np.isfinite(score)
            and CANDIDATE_POOL_SCORE_FLOOR <= score <= 1.0,
            f"candidate {index} has an invalid score",
        )
    unsigned = deepcopy(dict(payload))
    identity = unsigned.pop("candidate_pool_payload_identity_sha256", None)
    _require(_is_sha256(identity), "candidate pool payload identity is missing")
    _require(sha256_json(unsigned) == identity, "candidate pool payload identity is invalid")


def _one_to_one_match_count_um(
    predicted_xy_um: np.ndarray,
    truth_xy_um: np.ndarray,
    *,
    tolerance_um: float = 20.0,
) -> int:
    """Count tolerant one-to-one matches after both inputs enter physical µm."""

    predicted = np.asarray(predicted_xy_um, dtype=np.float64).reshape(-1, 2)
    truth = np.asarray(truth_xy_um, dtype=np.float64).reshape(-1, 2)
    if not len(predicted) or not len(truth):
        return 0
    _require(
        np.all(np.isfinite(predicted)) and np.all(np.isfinite(truth)),
        "physical attachment coordinates contain non-finite values",
    )
    distances = np.linalg.norm(predicted[:, None] - truth[None], axis=2)
    # A disallowed edge costs more than every possible sum of allowed edges,
    # so the assignment first maximizes the number of tolerant matches and only
    # then minimizes distance among those matches.
    forbidden = (max(len(predicted), len(truth)) + 1) * (tolerance_um + 1.0)
    cost = np.where(distances <= tolerance_um, distances, forbidden)
    rows, columns = linear_sum_assignment(cost)
    return int(np.sum(distances[rows, columns] <= tolerance_um))


def _prf(true_positive: int, predicted: int, truth: int) -> dict[str, float | int]:
    precision = true_positive / predicted if predicted else (1.0 if truth == 0 else 0.0)
    recall = true_positive / truth if truth else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": int(true_positive),
        "predicted": int(predicted),
        "ground_truth": int(truth),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def select_operating_point(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict], dict]:
    """Select on tolerant biological presence from one immutable pool/image."""

    _require(len(rows) == 44, "operating-point selection requires exactly 44 images")
    matcher = biological_hair_presence_matcher_contract()
    metrics: list[dict[str, Any]] = []
    for threshold in PREREGISTERED_SCORE_THRESHOLDS:
        biological_true_positive = 0
        attachment_true_positive = 0
        predicted_total = 0
        truth_total = 0
        count_errors: list[float] = []
        per_image: list[dict[str, Any]] = []
        for row in rows:
            bases = np.asarray(
                row["candidate_base_xy_um"], dtype=np.float64
            ).reshape(-1, 2)
            tips = np.asarray(
                row["candidate_tip_xy_um"], dtype=np.float64
            ).reshape(-1, 2)
            scores = np.asarray(row["candidate_scores"], dtype=np.float64)
            _require(
                scores.shape == (len(bases),)
                and tips.shape == bases.shape
                and np.all(np.isfinite(scores))
                and np.all(np.isfinite(bases))
                and np.all(np.isfinite(tips)),
                f"{row.get('task_id')}: candidate presence-proxy arrays are invalid",
            )
            keep = scores >= threshold
            predicted_bases = bases[keep]
            predicted_tips = tips[keep]
            predicted_polylines = [
                np.stack((base, tip))
                for base, tip in zip(
                    predicted_bases, predicted_tips, strict=True
                )
            ]
            truth_bases = np.asarray(
                row["truth_base_xy_um"], dtype=np.float64
            ).reshape(-1, 2)
            truth_polylines = [
                np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
                for polyline in row["truth_polylines_xy_um"]
            ]
            _require(
                len(truth_bases) == len(truth_polylines),
                f"{row.get('task_id')}: truth base/polyline counts differ",
            )
            biological, matches = match_biological_hair_presence(
                predicted_polylines,
                truth_polylines,
                1.0,
                matcher["curve_tolerance_um"],
                minimum_truth_coverage=matcher["minimum_truth_coverage"],
                minimum_prediction_coverage=matcher[
                    "minimum_prediction_coverage"
                ],
                minimum_direction_cosine=matcher["minimum_direction_cosine"],
                proximal_arc_fraction=matcher["proximal_arc_fraction"],
                resample_points=matcher["resample_points"],
            )
            attachment_matched = _one_to_one_match_count_um(
                predicted_bases,
                truth_bases,
                tolerance_um=float(matcher["curve_tolerance_um"]),
            )
            biological_true_positive += int(biological["tp"])
            attachment_true_positive += attachment_matched
            predicted_total += len(predicted_bases)
            truth_total += len(truth_bases)
            error = len(predicted_bases) - len(truth_bases)
            count_errors.append(float(error))
            per_image.append(
                {
                    "task_id": row["task_id"],
                    "predicted": len(predicted_bases),
                    "ground_truth": len(truth_bases),
                    "biological_presence_true_positive_20um": int(
                        biological["tp"]
                    ),
                    "attachment_proxy_true_positive_20um": attachment_matched,
                    "biological_presence_matched_pairs": [
                        {
                            "predicted_index_after_threshold": int(
                                match["predicted_index"]
                            ),
                            "annotated_index": int(match["annotated_index"]),
                            "prediction_coverage": float(
                                match["prediction_coverage"]
                            ),
                            "truth_coverage": float(match["truth_coverage"]),
                            "proximal_direction_cosine": float(
                                match["proximal_direction_cosine"]
                            ),
                        }
                        for match in matches
                    ],
                    "count_error": error,
                }
            )
        biological_identity = _prf(
            biological_true_positive, predicted_total, truth_total
        )
        attachment_identity = _prf(
            attachment_true_positive, predicted_total, truth_total
        )
        errors = np.asarray(count_errors, dtype=np.float64)
        metrics.append(
            {
                "threshold": float(threshold),
                "tolerant_biological_presence_20um": biological_identity,
                "identity_attachment_proxy_20um": attachment_identity,
                "count_mae": float(np.mean(np.abs(errors))),
                "count_bias": float(np.mean(errors)),
                "per_image": per_image,
            }
        )
    selected = min(
        metrics,
        key=lambda item: (
            -float(item["tolerant_biological_presence_20um"]["f1"]),
            float(item["count_mae"]),
            abs(float(item["count_bias"])),
            -float(item["threshold"]),
        ),
    )
    return metrics, deepcopy(selected)


def _validate_metric_row(
    row: Mapping[str, Any],
    *,
    threshold: float,
    task_ids: Sequence[str],
    ground_truth_by_task: Mapping[str, int],
    previous_predicted_by_task: Mapping[str, int] | None,
) -> dict[str, int]:
    matcher = biological_hair_presence_matcher_contract()
    _require(
        np.isclose(float(row.get("threshold", np.nan)), threshold, rtol=0.0, atol=1e-12),
        "selection threshold row is out of order",
    )
    per_image = row.get("per_image")
    _require(
        isinstance(per_image, list)
        and [item.get("task_id") for item in per_image] == list(task_ids),
        f"selection per-image rows are incomplete at threshold {threshold}",
    )
    biological_true_positive = 0
    attachment_true_positive = 0
    predicted_total = 0
    truth_total = 0
    errors: list[float] = []
    predicted_by_task: dict[str, int] = {}
    for item in per_image:
        task_id = str(item["task_id"])
        predicted = item.get("predicted")
        ground_truth = item.get("ground_truth")
        biological_matched = item.get(
            "biological_presence_true_positive_20um"
        )
        attachment_matched = item.get("attachment_proxy_true_positive_20um")
        matched_pairs = item.get("biological_presence_matched_pairs")
        error = item.get("count_error")
        _require(
            isinstance(predicted, int)
            and not isinstance(predicted, bool)
            and predicted >= 0,
            f"{task_id}: invalid predicted count in selection receipt",
        )
        _require(
            isinstance(ground_truth, int)
            and not isinstance(ground_truth, bool)
            and ground_truth == ground_truth_by_task[task_id],
            f"{task_id}: ground-truth count changed across threshold rows",
        )
        _require(
            isinstance(biological_matched, int)
            and not isinstance(biological_matched, bool)
            and 0 <= biological_matched <= min(predicted, ground_truth),
            f"{task_id}: invalid biological-presence match count",
        )
        _require(
            isinstance(attachment_matched, int)
            and not isinstance(attachment_matched, bool)
            and 0 <= attachment_matched <= min(predicted, ground_truth),
            f"{task_id}: invalid attachment-proxy match count",
        )
        pair_rows_valid = isinstance(matched_pairs, list) and all(
            isinstance(match, Mapping)
            and isinstance(match.get("predicted_index_after_threshold"), int)
            and not isinstance(match.get("predicted_index_after_threshold"), bool)
            and 0 <= match["predicted_index_after_threshold"] < predicted
            and isinstance(match.get("annotated_index"), int)
            and not isinstance(match.get("annotated_index"), bool)
            and 0 <= match["annotated_index"] < ground_truth
            and np.isfinite(float(match.get("prediction_coverage", np.nan)))
            and 0.0 <= float(match["prediction_coverage"]) <= 1.0
            and np.isfinite(float(match.get("truth_coverage", np.nan)))
            and 0.0 <= float(match["truth_coverage"]) <= 1.0
            and np.isfinite(
                float(match.get("proximal_direction_cosine", np.nan))
            )
            and -1.0 <= float(match["proximal_direction_cosine"]) <= 1.0
            and float(match["prediction_coverage"])
            >= float(matcher["minimum_prediction_coverage"])
            and float(match["truth_coverage"])
            >= float(matcher["minimum_truth_coverage"])
            and float(match["proximal_direction_cosine"])
            >= float(matcher["minimum_direction_cosine"])
            for match in (matched_pairs if isinstance(matched_pairs, list) else ())
        )
        _require(
            pair_rows_valid
            and len(matched_pairs) == biological_matched
            and len(
                {
                    int(match["predicted_index_after_threshold"])
                    for match in matched_pairs
                }
            )
            == biological_matched
            and len(
                {int(match["annotated_index"]) for match in matched_pairs}
            )
            == biological_matched,
            f"{task_id}: biological-presence matched-pair sufficiency is invalid",
        )
        _require(error == predicted - ground_truth, f"{task_id}: invalid count error")
        if previous_predicted_by_task is not None:
            _require(
                predicted <= previous_predicted_by_task[task_id],
                f"{task_id}: predicted count increases as threshold rises",
            )
        predicted_by_task[task_id] = predicted
        biological_true_positive += biological_matched
        attachment_true_positive += attachment_matched
        predicted_total += predicted
        truth_total += ground_truth
        errors.append(float(error))

    expected_biological_identity = _prf(
        biological_true_positive, predicted_total, truth_total
    )
    expected_attachment_identity = _prf(
        attachment_true_positive, predicted_total, truth_total
    )
    _require(
        row.get("tolerant_biological_presence_20um")
        == expected_biological_identity,
        f"selection biological-presence metrics are inconsistent at {threshold}",
    )
    _require(
        row.get("identity_attachment_proxy_20um")
        == expected_attachment_identity,
        f"selection attachment-proxy metrics are inconsistent at {threshold}",
    )
    error_array = np.asarray(errors, dtype=np.float64)
    _require(
        float(row.get("count_mae", np.nan)) == float(np.mean(np.abs(error_array))),
        f"selection count MAE is internally inconsistent at {threshold}",
    )
    _require(
        float(row.get("count_bias", np.nan)) == float(np.mean(error_array)),
        f"selection count bias is internally inconsistent at {threshold}",
    )
    return predicted_by_task


def validate_selection_receipt(receipt: Mapping[str, Any]) -> None:
    _require(
        receipt.get("schema_version") == SELECTION_RECEIPT_SCHEMA,
        "unsupported selection receipt schema",
    )
    _require(receipt.get("status") == "completed", "selection receipt is incomplete")
    _require(receipt.get("images") == 44, "selection receipt does not cover 44 images")
    _require(receipt.get("blind_images_used") == 0, "selection receipt is blind-tainted")
    _require(
        receipt.get("independent_accuracy_claim_allowed") is False,
        "selection receipt overclaims independent accuracy",
    )
    _require(
        receipt.get("selection_contract") == operating_point_selection_contract(),
        "selection receipt changed the preregistered protocol",
    )
    _require(
        receipt.get("metric_coordinate_space")
        == "physical_um_xy_after_axis_specific_realized_resize_conversion",
        "selection receipt did not evaluate centreline proxies in physical µm",
    )
    _require(
        receipt.get("canonical_ground_truth_authority") == CANONICAL_GT_AUTHORITY,
        "selection receipt does not use canonical annotation authority",
    )
    _require(
        receipt.get("candidate_pool_score_floor") == CANDIDATE_POOL_SCORE_FLOOR,
        "selection receipt candidate floor changed",
    )
    expected_decoder_contract = (
        locked_biological_presence_candidate_decoder_contract()
    )
    _require(
        receipt.get("candidate_decoder_contract") == expected_decoder_contract
        and receipt.get("candidate_decoder_contract_sha256")
        == sha256_json(expected_decoder_contract),
        "selection receipt candidate decoder contract changed",
    )
    _require(
        receipt.get("straight_base_to_tip_presence_proxy_evaluated_during_selection")
        is True
        and receipt.get("distal_endpoint_error_used_as_selection_gate") is False
        and receipt.get("complete_line_overlap_used_as_selection_gate") is False
        and receipt.get("length_error_used_as_selection_gate") is False
        and receipt.get("manual_hair_width_assumed") is False,
        "selection receipt changed biological-presence proxy/Gate semantics",
    )
    matcher = biological_hair_presence_matcher_contract()
    _require(
        receipt.get("primary_matcher_contract") == matcher
        and receipt.get("primary_matcher_contract_sha256") == sha256_json(matcher),
        "selection receipt primary matcher identity changed",
    )
    _require(
        receipt.get("validation_labels_used_for_gradient_or_early_stopping") is False,
        "selection receipt reports validation leakage into training",
    )
    for field in (
        "candidate_bundle_identity_sha256",
        "candidate_pool_identity_sha256",
        "dataset_manifest_sha256",
        "split_manifest_sha256",
        "dataset_split_identity_sha256",
        "integrity_manifest_sha256",
        "canonical_ground_truth_lock_identity_sha256",
        "task_image_lock_identity_sha256",
    ):
        _require(_is_sha256(receipt.get(field)), f"selection receipt lacks {field}")
    task_locks = receipt.get("task_image_locks")
    _require(
        isinstance(task_locks, list)
        and len(task_locks) == 44
        and len({row.get("task_id") for row in task_locks}) == 44,
        "selection task/image locks are incomplete",
    )
    _require(
        sha256_json(task_locks) == receipt["task_image_lock_identity_sha256"],
        "selection task/image lock identity is invalid",
    )
    task_ids = [str(row["task_id"]) for row in task_locks]
    ground_truth_by_task: dict[str, int] = {}
    canonical_locks: list[dict[str, Any]] = []
    for row in task_locks:
        task_id = str(row["task_id"])
        _require(_is_sha256(row.get("source_image_sha256")), f"{task_id}: bad image hash")
        _require(
            _is_sha256(row.get("candidate_pool_payload_identity_sha256")),
            f"{task_id}: bad candidate-pool payload identity",
        )
        _require(
            _is_sha256(row.get("candidate_pool_payload_file_sha256")),
            f"{task_id}: bad candidate-pool file hash",
        )
        for field in (
            "candidate_count_at_floor",
            "valid_presence_proxy_count_at_floor",
            "ground_truth_count",
            "canonical_annotation_size_bytes",
            "root_hair_count",
            "vertex_orders_geometrically_flipped",
        ):
            value = row.get(field)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{task_id}: invalid {field}",
            )
        _require(
            row["canonical_annotation_size_bytes"] > 0,
            f"{task_id}: canonical annotation byte size is empty",
        )
        _require(
            row["vertex_orders_geometrically_flipped"] <= row["root_hair_count"],
            f"{task_id}: canonical orientation-flip count is impossible",
        )
        _require(
            row["valid_presence_proxy_count_at_floor"]
            <= row["candidate_count_at_floor"],
            f"{task_id}: valid presence-proxy count exceeds candidate count",
        )
        ground_truth_by_task[task_id] = int(row["ground_truth_count"])
        for field in (
            "canonical_annotation_sha256",
            "source_image_sha256",
            "canonical_embedded_raw_annotation_sha256",
            "root_polygon_source_geometry_identity_sha256",
            "root_hair_instance_id_order_sha256",
            "oriented_source_geometry_identity_sha256",
            "physical_geometry_identity_sha256",
        ):
            _require(_is_sha256(row.get(field)), f"{task_id}: invalid canonical {field}")
        source_shape = row.get("source_image_shape_hw")
        _require(
            isinstance(source_shape, list)
            and len(source_shape) == 2
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in source_shape
            ),
            f"{task_id}: invalid canonical source image shape",
        )
        source_um_per_px = float(row.get("source_um_per_px", np.nan))
        _require(
            np.isfinite(source_um_per_px) and source_um_per_px > 0,
            f"{task_id}: invalid canonical physical scale",
        )
        _require(
            isinstance(row.get("canonical_annotation_relpath"), str)
            and bool(row["canonical_annotation_relpath"]),
            f"{task_id}: canonical annotation path is missing",
        )
        canonical_locks.append(
            {
                key: row[key]
                for key in (
                    "task_id",
                    "canonical_annotation_relpath",
                    "canonical_annotation_sha256",
                    "canonical_annotation_size_bytes",
                    "source_image_sha256",
                    "canonical_embedded_raw_annotation_sha256",
                    "source_image_shape_hw",
                    "source_um_per_px",
                    "root_hair_count",
                    "vertex_orders_geometrically_flipped",
                    "root_polygon_source_geometry_identity_sha256",
                    "root_hair_instance_id_order_sha256",
                    "oriented_source_geometry_identity_sha256",
                    "physical_geometry_identity_sha256",
                )
            }
        )
        _require(
            row["root_hair_count"] == row["ground_truth_count"],
            f"{task_id}: canonical and selection ground-truth counts differ",
        )
    _require(
        sha256_json(canonical_locks)
        == receipt["canonical_ground_truth_lock_identity_sha256"],
        "selection canonical ground-truth lock identity is invalid",
    )
    expected_pool_identity = sha256_json(
        {
            "schema_version": CANDIDATE_POOL_SCHEMA,
            "candidate_bundle_identity_sha256": receipt[
                "candidate_bundle_identity_sha256"
            ],
            "candidate_pool_score_floor": CANDIDATE_POOL_SCORE_FLOOR,
            "candidate_decoder_contract_sha256": sha256_json(
                locked_biological_presence_candidate_decoder_contract()
            ),
            "split_manifest_sha256": receipt["split_manifest_sha256"],
            "task_pool_payload_identities": [
                [row["task_id"], row["candidate_pool_payload_identity_sha256"]]
                for row in task_locks
            ],
        }
    )
    _require(
        receipt["candidate_pool_identity_sha256"] == expected_pool_identity,
        "selection candidate-pool identity is invalid",
    )

    rows = receipt.get("threshold_metrics")
    _require(isinstance(rows, list) and len(rows) == 10, "selection grid is incomplete")
    previous: dict[str, int] | None = None
    for row, threshold in zip(rows, PREREGISTERED_SCORE_THRESHOLDS, strict=True):
        _require(isinstance(row, Mapping), "selection metric row is not an object")
        current = _validate_metric_row(
            row,
            threshold=threshold,
            task_ids=task_ids,
            ground_truth_by_task=ground_truth_by_task,
            previous_predicted_by_task=previous,
        )
        if previous is None:
            _require(
                all(
                    current[task_id]
                    == int(task_locks[index]["candidate_count_at_floor"])
                    for index, task_id in enumerate(task_ids)
                ),
                "floor-threshold counts differ from the sealed candidate pool",
            )
        previous = current
    expected = min(
        rows,
        key=lambda item: (
            -float(item["tolerant_biological_presence_20um"]["f1"]),
            float(item["count_mae"]),
            abs(float(item["count_bias"])),
            -float(item["threshold"]),
        ),
    )
    _require(receipt.get("selected") == expected, "selected operating point violates tie-breaks")
    unsigned = deepcopy(dict(receipt))
    identity = unsigned.pop("selection_receipt_identity_sha256", None)
    _require(_is_sha256(identity), "selection receipt identity is missing")
    _require(sha256_json(unsigned) == identity, "selection receipt identity is invalid")


def read_selection_receipt(path: str | Path) -> dict[str, Any]:
    receipt = read_json(path)
    validate_selection_receipt(receipt)
    return receipt


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(bool(rows), f"empty CSV: {path}")
    _require(
        len({row["task_id"] for row in rows}) == len(rows),
        f"duplicate task_id in {path}",
    )
    return rows


def build_selection_receipt_from_paths(
    *,
    candidate_manifest_path: str | Path,
    candidate_pool_dir: str | Path,
    dataset_root: str | Path,
    dataset_manifest: str | Path,
    split_manifest: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the locked 44-image inputs and construct a deterministic receipt."""

    candidate_manifest_path = Path(candidate_manifest_path).resolve()
    pool_dir = Path(candidate_pool_dir).resolve()
    dataset_root_path = Path(dataset_root).resolve()
    dataset_manifest_path = Path(dataset_manifest).resolve()
    split_manifest_path = Path(split_manifest).resolve()
    manifest = read_candidate_manifest(candidate_manifest_path)
    pending_metadata = manifest["detection_model_metadata"]
    training_lock = manifest["identity_payload"]["training_lock"]
    _require(
        sha256_file(dataset_manifest_path) == training_lock["dataset_manifest_sha256"],
        "dataset manifest differs from the candidate training lock",
    )
    _require(
        sha256_file(split_manifest_path) == training_lock["split_manifest_sha256"],
        "split manifest differs from the candidate training lock",
    )
    dataset_rows = _csv_rows(dataset_manifest_path)
    metadata = {row["task_id"]: row for row in dataset_rows}
    split_rows = _csv_rows(split_manifest_path)
    task_ids = [row["task_id"] for row in split_rows if row.get("split") == "val"]
    _require(len(task_ids) == 44 and len(set(task_ids)) == 44, "split does not contain val44")
    _require(set(task_ids).issubset(metadata), "val44 tasks are missing from dataset manifest")
    try:
        canonical_gt, canonical_provenance = load_canonical_qcdev_ground_truth(
            dataset_root=dataset_root_path,
            dataset_manifest=dataset_manifest_path,
            split_manifest=split_manifest_path,
            expected_task_ids=task_ids,
        )
    except CanonicalGroundTruthError as error:
        raise SelectionGateError(str(error)) from error
    _require(
        canonical_provenance["dataset_manifest_sha256"]
        == training_lock["dataset_manifest_sha256"],
        "canonical GT dataset manifest differs from the candidate training lock",
    )
    _require(
        canonical_provenance["split_manifest_sha256"]
        == training_lock["split_manifest_sha256"],
        "canonical GT split manifest differs from the candidate training lock",
    )
    _require(
        canonical_provenance["integrity_manifest_sha256"]
        == training_lock["integrity_manifest_sha256"],
        "canonical GT integrity manifest differs from the candidate training lock",
    )
    canonical_lock_by_task = {
        row["task_id"]: row
        for row in canonical_provenance["canonical_annotation_locks"]
    }

    selection_rows: list[dict[str, Any]] = []
    task_locks: list[dict[str, Any]] = []
    for task_id in task_ids:
        row = metadata[task_id]
        image_sha256 = row.get("image_sha256") or row.get("source_image_sha256")
        _require(_is_sha256(image_sha256), f"{task_id}: dataset image hash is invalid")
        payload_path = pool_dir / f"{task_id}.json"
        if not payload_path.is_file():
            payload_path = pool_dir / "candidate_pools" / f"{task_id}.json"
        _require(payload_path.is_file(), f"{task_id}: candidate pool payload is missing")
        payload = read_json(payload_path)
        validate_biological_candidate_pool_payload(
            payload,
            expected_task_id=task_id,
            expected_image_sha256=image_sha256,
            expected_pending_model_metadata=pending_metadata,
        )
        coordinate = payload["coordinate_space"]
        candidates = payload["candidates"]
        bases = np.asarray(
            [candidate["base_xy_working"] for candidate in candidates],
            dtype=np.float64,
        ).reshape(-1, 2)
        tips = np.asarray(
            [candidate["tip_xy_working"] for candidate in candidates],
            dtype=np.float64,
        ).reshape(-1, 2)
        proxy_valid = np.asarray(
            [candidate["presence_proxy_valid"] for candidate in candidates],
            dtype=bool,
        )
        scores = np.asarray(
            [candidate["base_score"] for candidate in candidates], dtype=np.float64
        )
        source_um_per_px = float(row["source_um_per_px"])
        canonical_lock = canonical_lock_by_task[task_id]
        _require(
            canonical_lock["source_image_sha256"] == image_sha256,
            f"{task_id}: canonical and candidate source-image locks differ",
        )
        _require(
            np.isfinite(source_um_per_px)
            and source_um_per_px > 0
            and np.isclose(
                float(coordinate["source_um_per_px"]),
                source_um_per_px,
                rtol=0.0,
                atol=1e-12,
            ),
            f"{task_id}: candidate-pool scale differs from dataset metadata",
        )
        _require(
            np.isclose(
                source_um_per_px,
                float(canonical_lock["source_um_per_px"]),
                rtol=0.0,
                atol=1e-12,
            ),
            f"{task_id}: canonical and candidate physical scales differ",
        )
        if row.get("image_height") and row.get("image_width"):
            _require(
                coordinate["source_shape"]
                == [int(row["image_height"]), int(row["image_width"])],
                f"{task_id}: candidate-pool source shape differs from dataset metadata",
            )
        _require(
            coordinate["source_shape"] == canonical_lock["source_image_shape_hw"],
            f"{task_id}: canonical and candidate source shapes differ",
        )
        realized_um_per_px_xy = np.asarray(
            coordinate["realized_um_per_px_xy"], dtype=np.float64
        )
        candidate_bases_um = bases * realized_um_per_px_xy
        candidate_tips_um = tips * realized_um_per_px_xy
        truth_um = np.asarray(canonical_gt[task_id]["base"], dtype=np.float64)
        truth_polylines_um = [
            np.asarray(polyline, dtype=np.float64)
            for polyline in canonical_gt[task_id]["polys"]
        ]
        _require(
            np.all(np.isfinite(truth_um))
            and all(
                polyline.ndim == 2
                and polyline.shape[1:] == (2,)
                and len(polyline) >= 2
                and np.all(np.isfinite(polyline))
                for polyline in truth_polylines_um
            )
            and len(truth_um) == len(truth_polylines_um),
            f"{task_id}: canonical physical centreline geometry is invalid",
        )
        selection_rows.append(
            {
                "task_id": task_id,
                "candidate_base_xy_um": candidate_bases_um,
                "candidate_tip_xy_um": candidate_tips_um,
                "candidate_scores": scores,
                "truth_base_xy_um": truth_um,
                "truth_polylines_xy_um": truth_polylines_um,
            }
        )
        task_locks.append(
            {
                "task_id": task_id,
                "source_image_sha256": image_sha256,
                "candidate_pool_payload_identity_sha256": payload[
                    "candidate_pool_payload_identity_sha256"
                ],
                "candidate_pool_payload_file_sha256": sha256_file(payload_path),
                "candidate_count_at_floor": len(candidates),
                "valid_presence_proxy_count_at_floor": int(proxy_valid.sum()),
                "ground_truth_count": len(truth_um),
                **{
                    key: canonical_lock_by_task[task_id][key]
                    for key in (
                        "canonical_annotation_relpath",
                        "canonical_annotation_sha256",
                        "canonical_annotation_size_bytes",
                        "source_image_sha256",
                        "canonical_embedded_raw_annotation_sha256",
                        "source_image_shape_hw",
                        "source_um_per_px",
                        "root_hair_count",
                        "vertex_orders_geometrically_flipped",
                        "root_polygon_source_geometry_identity_sha256",
                        "root_hair_instance_id_order_sha256",
                        "oriented_source_geometry_identity_sha256",
                        "physical_geometry_identity_sha256",
                    )
                },
            }
        )
    candidate_pool_identity = sha256_json(
        {
            "schema_version": CANDIDATE_POOL_SCHEMA,
            "candidate_bundle_identity_sha256": manifest[
                "candidate_bundle_identity_sha256"
            ],
            "candidate_pool_score_floor": CANDIDATE_POOL_SCORE_FLOOR,
            "candidate_decoder_contract_sha256": sha256_json(
                locked_biological_presence_candidate_decoder_contract()
            ),
            "split_manifest_sha256": sha256_file(split_manifest_path),
            "task_pool_payload_identities": [
                [row["task_id"], row["candidate_pool_payload_identity_sha256"]]
                for row in task_locks
            ],
        }
    )
    threshold_metrics, selected = select_operating_point(selection_rows)
    receipt: dict[str, Any] = {
        "schema_version": SELECTION_RECEIPT_SCHEMA,
        "status": "completed",
        "scope": "locked_overlay_visible_QCdevelopment44_model_selection_only",
        "images": 44,
        "candidate_bundle_identity_sha256": manifest[
            "candidate_bundle_identity_sha256"
        ],
        "candidate_pool_identity_sha256": candidate_pool_identity,
        "candidate_pool_score_floor": CANDIDATE_POOL_SCORE_FLOOR,
        "candidate_decoder_contract": (
            locked_biological_presence_candidate_decoder_contract()
        ),
        "candidate_decoder_contract_sha256": sha256_json(
            locked_biological_presence_candidate_decoder_contract()
        ),
        "metric_coordinate_space": (
            "physical_um_xy_after_axis_specific_realized_resize_conversion"
        ),
        "canonical_ground_truth_authority": CANONICAL_GT_AUTHORITY,
        "canonical_ground_truth_lock_identity_sha256": canonical_provenance[
            "canonical_ground_truth_lock_identity_sha256"
        ],
        "selection_contract": operating_point_selection_contract(),
        "primary_matcher_contract": biological_hair_presence_matcher_contract(),
        "primary_matcher_contract_sha256": sha256_json(
            biological_hair_presence_matcher_contract()
        ),
        "threshold_metrics": threshold_metrics,
        "selected": selected,
        "task_image_locks": task_locks,
        "task_image_lock_identity_sha256": sha256_json(task_locks),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "dataset_split_identity_sha256": training_lock[
            "dataset_split_identity_sha256"
        ],
        "integrity_manifest_sha256": canonical_provenance[
            "integrity_manifest_sha256"
        ],
        "straight_base_to_tip_presence_proxy_evaluated_during_selection": True,
        "distal_endpoint_error_used_as_selection_gate": False,
        "complete_line_overlap_used_as_selection_gate": False,
        "length_error_used_as_selection_gate": False,
        "manual_hair_width_assumed": False,
        "validation_labels_used_for_gradient_or_early_stopping": False,
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
    }
    receipt["selection_receipt_identity_sha256"] = sha256_json(receipt)
    validate_selection_receipt(receipt)
    return receipt, deepcopy(pending_metadata)


def validate_selected_operating_point_binding(
    *,
    candidate_manifest: Mapping[str, Any],
    selected_model_metadata: Mapping[str, Any],
    selection_receipt: Mapping[str, Any],
    selection_receipt_file_sha256: str,
) -> None:
    """Cross-check the immutable candidate, QC receipt and runtime metadata."""

    validate_candidate_manifest(candidate_manifest)
    validate_train399_detection_model_metadata(selected_model_metadata)
    validate_selection_receipt(selection_receipt)
    pending = candidate_manifest["detection_model_metadata"]
    for field in (
        "expert_id",
        "ensemble_members",
        "checkpoint_policy",
        "training_images",
        "validation_images",
        "validation_labels_used_for_gradient_or_early_stopping",
        "blind_images_used",
        "seeds",
        "member_ids",
        "candidate_bundle_identity_sha256",
        "training_lock_identity_sha256",
        "checkpoint_sha256",
        "model_state_sha256",
        "training_task_ids_sha256",
        "split_manifest_sha256",
        "operating_point_selection_contract_sha256",
    ):
        _require(
            selected_model_metadata.get(field) == pending.get(field),
            f"selected metadata differs from candidate Gate: {field}",
        )
    _require(
        selection_receipt.get("candidate_bundle_identity_sha256")
        == candidate_manifest.get("candidate_bundle_identity_sha256"),
        "selection receipt belongs to a different candidate bundle",
    )
    training_lock = candidate_manifest["identity_payload"]["training_lock"]
    for receipt_field, lock_field in (
        ("dataset_manifest_sha256", "dataset_manifest_sha256"),
        ("split_manifest_sha256", "split_manifest_sha256"),
        ("dataset_split_identity_sha256", "dataset_split_identity_sha256"),
    ):
        _require(
            selection_receipt.get(receipt_field) == training_lock.get(lock_field),
            f"selection receipt differs from candidate training lock: {receipt_field}",
        )
    _require(
        sha256_json(selection_receipt["selection_contract"])
        == selected_model_metadata["operating_point_selection_contract_sha256"],
        "selection receipt protocol differs from selected metadata",
    )
    _require(
        _is_sha256(selection_receipt_file_sha256)
        and selected_model_metadata.get("selection_receipt_sha256")
        == selection_receipt_file_sha256,
        "selection receipt file hash differs from selected metadata",
    )
    _require(
        selected_model_metadata.get("selection_receipt_identity_sha256")
        == selection_receipt.get("selection_receipt_identity_sha256"),
        "selection receipt logical identity differs from selected metadata",
    )
    _require(
        selected_model_metadata.get("candidate_pool_identity_sha256")
        == selection_receipt.get("candidate_pool_identity_sha256"),
        "candidate-pool identity differs from selected metadata",
    )
    _require(
        np.isclose(
            float(selected_model_metadata["selected_score_threshold"]),
            float(selection_receipt["selected"]["threshold"]),
            rtol=0.0,
            atol=1e-12,
        ),
        "selected threshold differs between receipt and model metadata",
    )


def write_selection_receipt_and_metadata(
    *,
    receipt: Mapping[str, Any],
    pending_model_metadata: Mapping[str, Any],
    receipt_path: str | Path,
    selected_model_metadata_path: str | Path,
) -> dict[str, Any]:
    """Atomically seal the receipt, bind its file hash, then write selected metadata."""

    validate_selection_receipt(receipt)
    receipt_path = Path(receipt_path)
    metadata_path = Path(selected_model_metadata_path)
    if receipt_path.exists() or metadata_path.exists():
        raise FileExistsError("refusing to overwrite selection receipt or selected metadata")
    atomic_write_json(receipt_path, dict(receipt))
    receipt_file_sha256 = sha256_file(receipt_path)
    selected = bind_selected_operating_point(
        pending_model_metadata,
        selected_score_threshold=float(receipt["selected"]["threshold"]),
        selection_receipt_sha256=receipt_file_sha256,
        selection_receipt_identity_sha256=str(
            receipt["selection_receipt_identity_sha256"]
        ),
        candidate_pool_identity_sha256=str(receipt["candidate_pool_identity_sha256"]),
    )
    atomic_write_json(metadata_path, selected)
    return selected
