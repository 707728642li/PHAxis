#!/usr/bin/env python3
"""Build sealed, source-unit measurement-assurance evidence for PHAxis.

The command recomputes every publication value from explicit QC-development44
predictions, canonical vector-derived masks/points, and a named clean261 trait
table.  It never accepts a normalized metric CSV, discovers a newest run, or
reads blind data.  The output is development measurement assurance rather than
an independent-accuracy claim.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.contracts import (  # noqa: E402
    ContractError,
    validate_hybrid_prediction,
    validate_stageb_detection_payload,
)
from phaxis.fusion import (  # noqa: E402
    TRAIN399_STAGEB_POLICY,
    _stageb_hairs,
)
from phaxis.hair_attachment_assurance import (  # noqa: E402
    FORMAL_ATTACHMENT_TOLERANCE_UM,
    FORMAL_MATCHER_CONFIG,
    HAIR_ATTACHMENT_COORDINATE_SPACE,
    HAIR_ATTACHMENT_INPUT_SCHEMA,
    HAIR_POLYLINE_ORIENTATION,
    PROXY_TOLERANCES_UM,
    SELECTED_PROXY_TOLERANCE_UM,
    build_from_input_contract as build_hair_attachment_from_input_contract,
)
from phaxis.hair_stageb.canonical_ground_truth import (  # noqa: E402
    load_canonical_qcdev_ground_truth,
)
from phaxis.hair_stageb.candidate_bundle import (  # noqa: E402
    CandidateBundleError,
    validate_train399_detection_model_metadata,
)
from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402
from phaxis.root_continuity_assurance import (  # noqa: E402
    ROOT_CONTINUITY_COORDINATE_SPACE,
    ROOT_CONTINUITY_INPUT_SCHEMA,
    ROOT_CONTINUITY_PREDICTION_DEFINITION,
    ROOT_CONTINUITY_REFERENCE_DEFINITION,
    build_from_input_contract as build_root_continuity_from_input_contract,
)
from phaxis.root_trait_assurance import (  # noqa: E402
    ROOT_TRAIT_FAMILY_BY_FIELD,
    ROOT_TRAIT_PREDICTION_DEFINITION,
    ROOT_TRAIT_REFERENCE_DEFINITION,
    build_root_trait_assurance,
)
from phaxis.traits import ROOT_TRAIT_FIELDS  # noqa: E402


SCHEMA_VERSION = "PHAxis-measurement-assurance-receipt-1.0"
EVALUATION_SCHEMA = "PHAxis-StageB-train399-QCdev44-development-evaluation-1.2"
STAGEB_SCHEMA = "PHAxis-StageB-inference-run-1.1"
FUSION_SCHEMA = "PHAxis-fusion-run-1.1"
ROOT_EXACT_SCHEMA = "PHAxis-root-provider-fresh-reference283-audit-1.0"
COHORT_SCHEMA = "PHAxis-biological-cohorts-1.0"
GROUP_ORDER = ("RHD6_EV_22C", "RHD6_EV_30C", "RHD6_OE_22C", "RHD6_OE_30C")
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20260828
PCK_THRESHOLD_UM = 25.0
BOUNDARY_TOLERANCE_UM = 5.0
SCALE_ABSENCE_SPECIFICITY_STATUS = (
    "not_estimable_no_absent_or_untrusted_scale_cases"
)
SCALE_FAIL_CLOSED_EVIDENCE_BASIS = "software_contract_and_unit_tests"


class MeasurementAssuranceError(RuntimeError):
    """An authority or recomputed measurement failed its evidence contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MeasurementAssuranceError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sealed(payload: Mapping[str, Any], field: str, role: str) -> str:
    identity = payload.get(field)
    _require(_is_sha256(identity), f"{role}: missing {field}")
    unsigned = deepcopy(dict(payload))
    unsigned.pop(field, None)
    _require(sha256_json(unsigned) == identity, f"{role}: sealed identity mismatch")
    return str(identity)


def _read_csv(path: Path, columns: Sequence[str], role: str) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        raise MeasurementAssuranceError(f"{role}: unreadable CSV") from error
    missing = [column for column in columns if column not in frame.columns]
    _require(not missing and len(frame) > 0, f"{role}: missing columns {missing} or empty")
    return frame


def _read_mask(path: Path, expected_shape: tuple[int, int], role: str) -> np.ndarray:
    values = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    _require(values is not None, f"{role}: unreadable mask")
    unique = set(int(value) for value in np.unique(values))
    _require(
        bool(unique) and unique.issubset({0, 1, 255}),
        f"{role}: mask raster is not binary",
    )
    mask = np.asarray(values > 0, dtype=bool)
    _require(mask.shape == expected_shape and mask.any(), f"{role}: invalid mask geometry")
    return mask


def _locked_regular_file(root: Path, relative: str | Path, role: str) -> Path:
    """Resolve one named artifact inside ``root`` without symlink traversal."""

    base = root.resolve()
    value = Path(relative)
    _require(not value.is_absolute(), f"{role}: absolute artifact path refused")
    candidate = base / value
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"{role}: missing/non-regular artifact",
    )
    resolved = candidate.resolve()
    _require(
        resolved == base or base in resolved.parents,
        f"{role}: artifact escapes its locked root",
    )
    return resolved


def _normalise_ordered_file_locks(
    value: Any, task_ids: Sequence[str], role: str
) -> list[dict[str, str]]:
    _require(isinstance(value, list), f"{role}: ordered file locks missing")
    result: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        _require(
            isinstance(raw, Mapping) and set(raw) == {"task_id", "sha256"},
            f"{role}[{index}]: lock fields drift",
        )
        task_id, digest = str(raw["task_id"]), str(raw["sha256"])
        _require(_is_sha256(digest), f"{role}[{index}]: invalid SHA-256")
        result.append({"task_id": task_id, "sha256": digest})
    _require(
        [row["task_id"] for row in result] == list(task_ids),
        f"{role}: task order drift",
    )
    return result


def _validate_stageb_authority(
    evaluation: Mapping[str, Any],
    stageb: Mapping[str, Any],
    task_ids: Sequence[str],
    stageb_root: Path | None = None,
) -> dict[str, Any]:
    """Separate eval-only bytes from production bytes and close shared authority."""

    locks = evaluation.get("prediction_input_locks")
    _require(isinstance(locks, Mapping), "evaluator Stage-B input locks missing")
    evaluation_files = _normalise_ordered_file_locks(
        locks.get("stageb_detection_files"),
        task_ids,
        "QCdev evaluation-only Stage-B detections",
    )
    evaluation_set_identity = sha256_json(evaluation_files)
    _require(
        locks.get("stageb_detection_set_identity_sha256")
        == evaluation_set_identity,
        "QCdev evaluation-only Stage-B ordered-set identity drift",
    )

    records = stageb.get("records")
    _require(
        isinstance(records, list)
        and len(records) == len(task_ids)
        and [str(record.get("task_id")) for record in records] == list(task_ids),
        "QCdev production Stage-B task order changed",
    )
    production_files: list[dict[str, str]] = []
    production_identities: list[dict[str, str]] = []
    detection_total = 0
    for index, record in enumerate(records):
        logical_digest = str(record.get("detection_identity_sha256", ""))
        count = record.get("detections")
        task_id = str(record["task_id"])
        recorded_file_digest = record.get("detection_file_sha256")
        if stageb_root is not None:
            detection_path = stageb_root / "detections" / f"{task_id}.json"
            _require(
                detection_path.is_file() and not detection_path.is_symlink(),
                f"QCdev production Stage-B record {index}: detection file missing",
            )
            observed_file_digest = sha256_file(detection_path)
            if recorded_file_digest is not None:
                _require(
                    str(recorded_file_digest) == observed_file_digest,
                    f"QCdev production Stage-B record {index}: file identity drift",
                )
            detection_payload = read_json(detection_path)
            payload_identity_matches = (
                detection_payload.get("task_id") == task_id
                and detection_payload.get("detection_identity_sha256")
                == logical_digest
            )
        else:
            observed_file_digest = str(recorded_file_digest or "")
            payload_identity_matches = True
        _require(
            _is_sha256(observed_file_digest)
            and _is_sha256(logical_digest)
            and payload_identity_matches,
            f"QCdev production Stage-B record {index}: file/logical identity missing or changed",
        )
        _require(
            isinstance(count, int) and not isinstance(count, bool) and count >= 0,
            f"QCdev production Stage-B record {index}: detection count invalid",
        )
        production_files.append(
            {"task_id": task_id, "sha256": observed_file_digest}
        )
        production_identities.append(
            {"task_id": task_id, "detection_identity_sha256": logical_digest}
        )
        detection_total += int(count)
    _require(
        stageb.get("images") == len(task_ids)
        and stageb.get("detections") == detection_total,
        "QCdev production Stage-B image/detection denominator drift",
    )

    training = evaluation.get("training_contract")
    model = stageb.get("detection_model_metadata")
    _require(
        isinstance(training, Mapping) and isinstance(model, Mapping),
        "QCdev shared Stage-B model authority missing",
    )
    try:
        validate_train399_detection_model_metadata(model)
    except CandidateBundleError as error:
        raise MeasurementAssuranceError(
            f"QCdev production Stage-B model metadata invalid: {error}"
        ) from error
    _require(
        training.get("training_images") == 399
        and training.get("validation_images") == len(task_ids) == 44
        and training.get("validation_labels_used_for_gradient_or_early_stopping")
        is False,
        "QCdev evaluation training authority changed",
    )
    checkpoint_sha256 = list(model.get("checkpoint_sha256", ()))
    _require(
        checkpoint_sha256
        == list(stageb.get("checkpoint_sha256", ()))
        == list(training.get("checkpoint_sha256", ())),
        "QCdev evaluation/production Stage-B checkpoint authority drift",
    )
    for field in (
        "candidate_bundle_identity_sha256",
        "selected_model_metadata_identity_sha256",
        "selection_receipt_identity_sha256",
    ):
        _require(
            _is_sha256(model.get(field))
            and model.get(field) == training.get(field),
            f"QCdev evaluation/production Stage-B {field} drift",
        )
    selected_threshold = float(model.get("selected_score_threshold", float("nan")))
    _require(
        math.isfinite(selected_threshold)
        and math.isclose(
            float(stageb.get("score_threshold", float("nan"))),
            selected_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and model.get("checkpoint_policy") == TRAIN399_STAGEB_POLICY
        and model.get("expert_id") == "PHAxis-StageB-train399-five-seed",
        "QCdev production Stage-B selected threshold/policy drift",
    )
    _require(
        stageb.get("condition_metadata_used_for_routing") in (None, False)
        and stageb.get("canonical_annotations_read") in (None, False)
        and stageb.get("root_cap_region_output") in (None, False)
        and stageb.get("blind_images_used") == 0,
        "QCdev production Stage-B guard drift",
    )
    return {
        "evaluation_detection_files": evaluation_files,
        "evaluation_detection_ordered_file_set_identity_sha256": (
            evaluation_set_identity
        ),
        "production_detection_files": production_files,
        "production_detection_ordered_file_set_identity_sha256": sha256_json(
            production_files
        ),
        "production_detection_identities": production_identities,
        "production_detection_ordered_identity_set_sha256": sha256_json(
            production_identities
        ),
        "shared_model_authority": {
            "expert_id": str(model["expert_id"]),
            "checkpoint_policy": str(model["checkpoint_policy"]),
            "checkpoint_sha256": checkpoint_sha256,
            "selected_score_threshold": selected_threshold,
            "candidate_bundle_identity_sha256": str(
                model["candidate_bundle_identity_sha256"]
            ),
            "selected_model_metadata_identity_sha256": str(
                model["selected_model_metadata_identity_sha256"]
            ),
            "selection_receipt_identity_sha256": str(
                model["selection_receipt_identity_sha256"]
            ),
        },
    }


def _validate_fused_stageb_identity(
    prediction: Mapping[str, Any], stageb_detection: Mapping[str, Any]
) -> list[np.ndarray]:
    """Return authoritative physical hair vectors after exact fusion cross-checks."""

    task_id = str(stageb_detection.get("task_id", ""))
    expected_hairs = _stageb_hairs(stageb_detection)
    identity_hairs = prediction.get("identity_hairs")
    count_hairs = prediction.get("count_hairs")
    phaxis = prediction.get("phaxis")
    _require(
        prediction.get("identity_hair_variant")
        == "phaxis_stage_b_train399_five_seed_identity"
        and prediction.get("count_hair_variant")
        == "phaxis_stage_b_train399_five_seed_count",
        f"{task_id}: fused Stage-B identity/count variant drift",
    )
    _require(
        isinstance(identity_hairs, list)
        and isinstance(count_hairs, list)
        and identity_hairs == count_hairs
        and len(identity_hairs) == len(expected_hairs) == int(stageb_detection["n"]),
        f"{task_id}: fused Stage-B identity/count/order denominator drift",
    )
    _require(isinstance(phaxis, Mapping), f"{task_id}: fused PHAxis provenance missing")
    model = stageb_detection["model"]
    _require(
        phaxis.get("hair_identity_count_expert") == model.get("expert_id")
        and phaxis.get("hair_identity_count_checkpoint_policy")
        == TRAIN399_STAGEB_POLICY
        and phaxis.get("hair_identity_count_candidate_bundle_identity_sha256")
        == model.get("candidate_bundle_identity_sha256")
        and phaxis.get("stageb_detection_identity_sha256")
        == stageb_detection.get("detection_identity_sha256")
        and phaxis.get("formal_stageb_identity_count") == len(identity_hairs)
        and prediction.get("blind_images_used") == 0,
        f"{task_id}: fused Stage-B authority/count provenance drift",
    )
    expert_boundary = phaxis.get("expert_boundary")
    _require(
        isinstance(expert_boundary, Mapping)
        and expert_boundary.get("phaxis_stage_b_train399")
        == ["hair_identity", "hair_count"],
        f"{task_id}: fused Stage-B expert boundary drift",
    )

    array_fields = (
        "points_xy",
        "stageb_base_xy_working",
        "stageb_tip_xy_working",
    )
    numeric_fields = ("stageb_score", "stageb_predicted_length_um")
    exact_fields = (
        "source",
        "source_instance_id",
        "stageb_tip_snapped",
        "stageb_predicted_length_semantics",
    )
    for index, (observed, expected) in enumerate(
        zip(identity_hairs, expected_hairs, strict=True)
    ):
        _require(
            isinstance(observed, Mapping),
            f"{task_id}: fused Stage-B hair {index} is malformed",
        )
        for field in array_fields:
            left = np.asarray(observed.get(field), dtype=np.float64)
            right = np.asarray(expected.get(field), dtype=np.float64)
            _require(
                left.shape == right.shape
                and np.all(np.isfinite(left))
                and np.allclose(left, right, rtol=0.0, atol=1e-12),
                f"{task_id}: fused Stage-B hair {index} {field} drift",
            )
        for field in numeric_fields:
            _require(
                math.isclose(
                    float(observed.get(field, float("nan"))),
                    float(expected[field]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"{task_id}: fused Stage-B hair {index} {field} drift",
            )
        for field in exact_fields:
            _require(
                observed.get(field) == expected.get(field),
                f"{task_id}: fused Stage-B hair {index} {field} drift",
            )

    source_um_per_px = float(stageb_detection["coordinate_space"]["source_um_per_px"])
    return [
        np.asarray(hair["points_xy"], dtype=np.float64) * source_um_per_px
        for hair in expected_hairs
    ]


def _evaluation_stageb_presence_counts(
    row: Mapping[str, Any], *, role: str
) -> dict[str, int]:
    stageb = row.get("stageb_train399")
    _require(isinstance(stageb, Mapping), f"{role}: Stage-B evaluator row missing")
    presence = stageb.get("biological_presence_tp")
    _require(
        isinstance(presence, Mapping),
        f"{role}: biological-presence sufficient statistics missing",
    )
    candidates = (presence.get("20"), presence.get("20.0"), presence.get(20), presence.get(20.0))
    values = [value for value in candidates if value is not None]
    _require(
        values and len({int(value) for value in values}) == 1,
        f"{role}: biological-presence TP@20um is missing or ambiguous",
    )
    result = {
        "n_pred": int(stageb.get("n_pred", -1)),
        "n_gt": int(stageb.get("n_gt", -1)),
        "biological_presence_tp_20um": int(values[0]),
    }
    _require(
        result["n_pred"] >= 0
        and result["n_gt"] >= 0
        and 0
        <= result["biological_presence_tp_20um"]
        <= min(result["n_pred"], result["n_gt"]),
        f"{role}: impossible Stage-B biological-presence counts",
    )
    return result


def _crosscheck_hair_biological_presence(
    evaluation: Mapping[str, Any],
    hair_attachment_assurance: Mapping[str, Any],
    task_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], str]:
    """Require production/fused formal matches to reproduce evaluator counts."""

    evaluation_rows = evaluation.get("per_image")
    measured_rows = hair_attachment_assurance.get("per_image")
    _require(
        isinstance(evaluation_rows, list)
        and isinstance(measured_rows, list)
        and [str(row.get("task_id")) for row in evaluation_rows]
        == list(task_ids)
        and [str(row.get("source_unit")) for row in measured_rows]
        == list(task_ids),
        "hair biological-presence crosscheck task order drift",
    )
    locks: list[dict[str, Any]] = []
    for task_id, evaluation_row, measured in zip(
        task_ids, evaluation_rows, measured_rows, strict=True
    ):
        expected = _evaluation_stageb_presence_counts(
            evaluation_row, role=f"{task_id} evaluator"
        )
        observed_formal = measured["formal_matched_attachment_accuracy"][
            "formal_biological_presence"
        ]
        observed = {
            "n_pred": int(observed_formal["n_pred"]),
            "n_gt": int(observed_formal["n_gt"]),
            "biological_presence_tp_20um": int(observed_formal["tp"]),
        }
        _require(
            observed == expected,
            f"{task_id}: production/fused biological-presence counts differ from evaluator Stage-B@20um",
        )
        locks.append(
            {
                "task_id": task_id,
                **observed,
                "hair_attachment_row_identity_sha256": measured[
                    "row_identity_sha256"
                ],
            }
        )

    evaluation_overall = evaluation.get("overall", {}).get("stageb_train399")
    _require(
        isinstance(evaluation_overall, Mapping),
        "evaluator Stage-B overall biological-presence summary missing",
    )
    evaluation_overall_presence = evaluation_overall.get(
        "tolerant_biological_presence", {}
    ).get("20")
    production_overall_presence = hair_attachment_assurance["summary"][
        "formal_matched_attachment_accuracy"
    ]["formal_biological_presence"]
    _require(
        isinstance(evaluation_overall_presence, Mapping)
        and {
            "tp": int(evaluation_overall_presence.get("tp", -1)),
            "n_pred": int(evaluation_overall_presence.get("n_pred", -1)),
            "n_gt": int(evaluation_overall_presence.get("n_gt", -1)),
        }
        == {
            "tp": int(production_overall_presence["tp"]),
            "n_pred": int(production_overall_presence["n_pred"]),
            "n_gt": int(production_overall_presence["n_gt"]),
        }
        and evaluation_overall.get("predicted_hairs")
        == production_overall_presence["n_pred"]
        and evaluation_overall.get("ground_truth_hairs")
        == production_overall_presence["n_gt"],
        "production/fused overall biological-presence counts differ from evaluator Stage-B@20um",
    )
    return locks, sha256_json(locks)


def _skeleton_components_xy_um(
    mask: np.ndarray, *, um_per_px: float
) -> list[list[list[float]]]:
    """Encode every 8-connected skeleton component without adding bridge edges."""

    skeleton = np.asarray(skeletonize(np.asarray(mask, dtype=bool), method="lee"), dtype=bool)
    labels, count = label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    _require(count > 0, "sealed final fused root mask produced an empty skeleton")
    components: list[list[list[float]]] = []
    neighbours = tuple(
        (dy, dx)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )
    for component_index in range(1, count + 1):
        nodes = {
            (int(y), int(x))
            for y, x in np.argwhere(labels == component_index)
        }
        # A one-pixel skeleton island has no axis segment and therefore cannot
        # represent a primary-root fragment.  Exclude this raster debris while
        # retaining every component with a measurable (non-zero) axis; no
        # interpolation or bridge is introduced.
        if len(nodes) < 2:
            continue
        adjacency = {
            node: sorted(
                (
                    (node[0] + dy, node[1] + dx)
                    for dy, dx in neighbours
                    if (node[0] + dy, node[1] + dx) in nodes
                )
            )
            for node in nodes
        }
        _require(
            all(adjacency[node] for node in nodes),
            "sealed final fused root skeleton contains an isolated node",
        )
        expected_edges = {
            tuple(sorted((node, neighbour)))
            for node, values in adjacency.items()
            for neighbour in values
        }
        start = min(nodes)
        walk = [start]
        seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        stack: list[tuple[tuple[int, int], int]] = [(start, 0)]
        while stack:
            node, next_index = stack[-1]
            values = adjacency[node]
            while next_index < len(values):
                neighbour = values[next_index]
                next_index += 1
                stack[-1] = (node, next_index)
                edge = tuple(sorted((node, neighbour)))
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                walk.append(neighbour)
                stack.append((neighbour, 0))
                break
            else:
                stack.pop()
                if stack:
                    walk.append(stack[-1][0])
        _require(
            seen_edges == expected_edges,
            "sealed final fused root skeleton edge traversal is incomplete",
        )
        walk_yx = np.asarray(walk, dtype=np.int64)
        _require(
            np.all(
                np.max(np.abs(np.diff(walk_yx, axis=0)), axis=1) <= 1
            ),
            "root skeleton traversal attempted evaluator-side bridging",
        )
        xy_um = (
            np.asarray([(x, y) for y, x in walk], dtype=np.float64)
            * float(um_per_px)
        )
        components.append(xy_um.tolist())
    _require(
        components,
        "sealed final fused root mask has no measurable skeleton component",
    )
    return components


def _seal_portable_input_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    _require(
        "input_contract_identity_sha256" not in result,
        "portable input contract was already sealed",
    )
    result["input_contract_identity_sha256"] = sha256_json(result)
    return result


def _largest_component(mask: np.ndarray) -> np.ndarray:
    components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    _require(count > 0, "root mask has no foreground component")
    sizes = np.bincount(components.ravel())
    return np.asarray(components == int(np.argmax(sizes[1:]) + 1), dtype=bool)


def _root_axis(mask: np.ndarray, tip_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return distal-oriented main skeleton path, arc distance and local radius."""

    canonical = _largest_component(np.asarray(mask, dtype=bool))
    yy, xx = np.nonzero(canonical)
    margin = 4
    y0, y1 = max(0, int(yy.min()) - margin), min(mask.shape[0], int(yy.max()) + margin + 1)
    x0, x1 = max(0, int(xx.min()) - margin), min(mask.shape[1], int(xx.max()) + margin + 1)
    roi = np.ascontiguousarray(canonical[y0:y1, x0:x1])
    skeleton = np.asarray(skeletonize(roi, method="lee"), dtype=bool)
    sk_components, count = label(skeleton, structure=np.ones((3, 3), dtype=np.uint8))
    _require(count > 0, "root skeleton is empty")
    sizes = np.bincount(sk_components.ravel())
    coordinates_yx = np.argwhere(sk_components == int(np.argmax(sizes[1:]) + 1))
    _require(len(coordinates_yx) >= 2, "root skeleton is too short")
    index_map = np.full(roi.shape, -1, dtype=np.int32)
    index_map[coordinates_yx[:, 0], coordinates_yx[:, 1]] = np.arange(
        len(coordinates_yx), dtype=np.int32
    )
    source_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    degree = np.zeros(len(coordinates_yx), dtype=np.uint8)
    source = np.arange(len(coordinates_yx), dtype=np.int32)
    for dy, dx, weight in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0))):
        target_y = coordinates_yx[:, 0] + dy
        target_x = coordinates_yx[:, 1] + dx
        inside = (
            (target_y >= 0)
            & (target_y < roi.shape[0])
            & (target_x >= 0)
            & (target_x < roi.shape[1])
        )
        neighbour = np.full(len(coordinates_yx), -1, dtype=np.int32)
        neighbour[inside] = index_map[target_y[inside], target_x[inside]]
        keep = neighbour >= 0
        starts, stops = source[keep], neighbour[keep]
        degree[starts] += 1
        degree[stops] += 1
        source_blocks.extend((starts, stops))
        target_blocks.extend((stops, starts))
        weight_blocks.extend(
            (
                np.full(len(starts), weight, dtype=np.float64),
                np.full(len(starts), weight, dtype=np.float64),
            )
        )
    _require(source_blocks, "root skeleton graph has no edges")
    graph = coo_matrix(
        (
            np.concatenate(weight_blocks),
            (np.concatenate(source_blocks), np.concatenate(target_blocks)),
        ),
        shape=(len(coordinates_yx), len(coordinates_yx)),
    ).tocsr()
    coordinates_xy = coordinates_yx[:, ::-1].astype(np.float64)
    local_tip = np.asarray(tip_xy, dtype=np.float64) - np.asarray([x0, y0], dtype=np.float64)
    start = int(cKDTree(coordinates_xy).query(local_tip, k=1)[1])
    distances, predecessors = dijkstra(
        graph, directed=False, indices=start, return_predecessors=True
    )
    endpoints = np.flatnonzero(degree <= 1)
    candidates = endpoints[np.isfinite(distances[endpoints])]
    candidates = candidates[candidates != start]
    if not candidates.size:
        candidates = np.flatnonzero(np.isfinite(distances))
    far = int(candidates[int(np.argmax(distances[candidates]))])
    nodes = [far]
    cursor = far
    while cursor != start and len(nodes) <= len(coordinates_yx):
        cursor = int(predecessors[cursor])
        _require(cursor >= 0, "root geodesic predecessor chain is incomplete")
        nodes.append(cursor)
    _require(nodes[-1] == start, "root geodesic did not reach distal anchor")
    nodes_array = np.asarray(nodes[::-1], dtype=np.int32)
    path_xy = coordinates_xy[nodes_array] + np.asarray([x0, y0], dtype=np.float64)
    increments = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    distance_px = np.concatenate(([0.0], np.cumsum(increments)))
    radius_map = distance_transform_edt(roi)
    path_local_yx = coordinates_yx[nodes_array]
    radius_px = radius_map[path_local_yx[:, 0], path_local_yx[:, 1]].astype(np.float64)
    return path_xy, distance_px, radius_px


def _root_statistics(mask: np.ndarray, tip_xy: np.ndarray, um_per_px: float) -> dict[str, float]:
    path_xy, axis_s_px, radius_px = _root_axis(mask, tip_xy)
    bridge_px = min(
        float(np.linalg.norm(np.asarray(tip_xy, dtype=float) - path_xy[0])),
        2.5 * float(np.clip(np.median(radius_px[: min(len(radius_px), 80)]), 2.0, 250.0)),
    )
    s_um = (axis_s_px + bridge_px) * um_per_px
    axis_length_um = float(s_um[-1])
    chord_um = float(np.linalg.norm(np.asarray(tip_xy, dtype=float) - path_xy[-1]) * um_per_px)
    width_um = 2.0 * radius_px * um_per_px
    end_exclusion_um = max(2.0 * float(np.median(radius_px)) * um_per_px, 0.02 * axis_length_um)
    central = (s_um >= end_exclusion_um) & (s_um <= axis_length_um - end_exclusion_um)
    if int(central.sum()) < 6:
        central = np.isfinite(width_um) & (width_um > 0)
    x, widths = s_um[central], width_um[central]
    _require(len(widths) >= 3, "root axis has insufficient width support")
    normalized = (x - float(x.min())) / max(float(np.ptp(x)), 1e-9)
    thirds = (
        normalized < 1 / 3,
        (normalized >= 1 / 3) & (normalized < 2 / 3),
        normalized >= 2 / 3,
    )
    _require(all(np.any(part) for part in thirds), "root width thirds are incomplete")
    tip_width, middle_width, shootward_width = (
        float(np.median(widths[part])) for part in thirds
    )
    weights = np.gradient(x) if len(x) > 1 else np.ones_like(x)
    design = np.column_stack((x, np.ones_like(x)))
    root_weight = np.sqrt(np.clip(weights, 1e-9, None))
    slope = float(
        np.linalg.lstsq(
            design * root_weight[:, None], widths * root_weight, rcond=None
        )[0][0]
        * 1000.0
    )

    complete_path = np.vstack((np.asarray(tip_xy, dtype=float), path_xy))
    complete_s = np.concatenate(([0.0], s_um))
    keep = np.concatenate(([True], np.diff(complete_s) > 1e-9))
    complete_path, complete_s = complete_path[keep], complete_s[keep]
    targets = np.arange(0.0, complete_s[-1], 25.0)
    if not len(targets) or targets[-1] < complete_s[-1]:
        targets = np.append(targets, complete_s[-1])
    points_um = np.column_stack(
        [np.interp(targets, complete_s, complete_path[:, axis]) for axis in range(2)]
    ) * um_per_px
    half_window_um = 250.0
    curvature_targets = np.arange(
        half_window_um, targets[-1] - half_window_um + 62.5, 125.0
    )
    if len(curvature_targets) >= 3:
        def interpolate(query: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [np.interp(query, targets, points_um[:, axis]) for axis in range(2)]
            )

        left = interpolate(curvature_targets - half_window_um)
        centre = interpolate(curvature_targets)
        right = interpolate(curvature_targets + half_window_um)
        incoming, outgoing = centre - left, right - centre
        dot = np.einsum("ij,ij->i", incoming, outgoing)
        cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
        curvature = np.abs(np.arctan2(cross, dot)) / half_window_um * 1000.0
        curvature_median = float(np.median(curvature))
        curvature_p95 = float(np.quantile(curvature, 0.95))
    else:
        curvature_median = curvature_p95 = 0.0
    area_um2 = float(np.count_nonzero(mask) * um_per_px**2)
    return {
        "visible_root_axis_length_um": axis_length_um,
        "root_axis_chord_um": chord_um,
        "root_centerline_chord_tortuosity": axis_length_um / chord_um,
        "root_straightness": chord_um / axis_length_um,
        "root_projected_area_um2": area_um2,
        "root_projected_area_um2_per_root_mm": area_um2 / (axis_length_um / 1000.0),
        "median_root_width_um": float(np.median(widths)),
        "root_width_p10_um": float(np.quantile(widths, 0.10)),
        "root_width_q25_um": float(np.quantile(widths, 0.25)),
        "root_width_q75_um": float(np.quantile(widths, 0.75)),
        "root_width_p90_um": float(np.quantile(widths, 0.90)),
        "root_width_cv": float(np.std(widths, ddof=1) / np.mean(widths)),
        "root_width_tip_third_median_um": tip_width,
        "root_width_middle_third_median_um": middle_width,
        "root_width_shootward_third_median_um": shootward_width,
        "root_width_shootward_to_tip_ratio": shootward_width / tip_width,
        "root_width_axial_slope_um_per_mm": slope,
        "root_centerline_curvature_median_rad_per_mm": curvature_median,
        "root_centerline_curvature_p95_rad_per_mm": curvature_p95,
    }


def _axis_root_support_metrics(
    root_mask: np.ndarray,
    axis_xy: np.ndarray,
    *,
    um_per_px: float,
) -> dict[str, float | int]:
    """Measure ordered-axis support by the root-mask union and one component.

    ``axis_in_root_coverage_fraction`` is the nearest-pixel fraction supported
    by *any* 8-connected root-mask component.  The single-component fraction
    uses the one component supporting the largest number of ordered-axis
    samples (ties resolve to the smaller component label).  The unsupported
    gap is the longest contiguous arc-length interval for which at least one
    segment endpoint is not supported by that winning component.  This keeps
    union support from disguising a fragmented carrying-root coordinate.
    """

    mask = np.asarray(root_mask, dtype=bool)
    path = np.asarray(axis_xy, dtype=np.float64)
    _require(mask.ndim == 2 and mask.any(), "axis support requires a non-empty 2D root mask")
    _require(
        path.ndim == 2
        and path.shape[1] == 2
        and len(path) >= 2
        and np.all(np.isfinite(path)),
        "axis support requires a finite ordered xy path",
    )
    _require(math.isfinite(float(um_per_px)) and float(um_per_px) > 0.0, "axis support requires a positive physical scale")

    component_labels, component_count = label(
        mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    rounded = np.rint(path).astype(np.int64)
    inside_frame = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < mask.shape[0])
    )
    axis_labels = np.zeros(len(rounded), dtype=np.int32)
    axis_labels[inside_frame] = component_labels[
        rounded[inside_frame, 1], rounded[inside_frame, 0]
    ]
    positive_labels, positive_counts = np.unique(
        axis_labels[axis_labels > 0], return_counts=True
    )
    _require(len(positive_labels) > 0, "ordered axis has no root-mask support")
    best_count = int(positive_counts.max())
    winning_label = int(positive_labels[positive_counts == best_count].min())
    in_root = axis_labels > 0
    on_winning_component = axis_labels == winning_label

    segment_lengths_um = (
        np.linalg.norm(np.diff(path, axis=0), axis=1) * float(um_per_px)
    )
    unsupported_segments = ~(
        on_winning_component[:-1] & on_winning_component[1:]
    )
    longest_gap_um = 0.0
    current_gap_um = 0.0
    for segment_length_um, unsupported in zip(
        segment_lengths_um, unsupported_segments, strict=True
    ):
        if unsupported:
            current_gap_um += float(segment_length_um)
            longest_gap_um = max(longest_gap_um, current_gap_um)
        else:
            current_gap_um = 0.0

    return {
        "axis_in_root_coverage_fraction": float(np.mean(in_root)),
        "axis_single_component_coverage_fraction": float(
            np.mean(on_winning_component)
        ),
        "longest_unsupported_axis_gap_um": float(longest_gap_um),
        "root_mask_component_count": int(component_count),
        "axis_support_component_label": winning_label,
    }


def _root_overlap_metrics(predicted: np.ndarray, truth: np.ndarray, um_per_px: float) -> tuple[float, float, float]:
    intersection = int(np.count_nonzero(predicted & truth))
    dice = 2.0 * intersection / (int(predicted.sum()) + int(truth.sum()))
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_u8, truth_u8 = predicted.astype(np.uint8), truth.astype(np.uint8)
    pred_boundary = pred_u8.astype(bool) & ~cv2.erode(pred_u8, kernel, iterations=1).astype(bool)
    truth_boundary = truth_u8.astype(bool) & ~cv2.erode(truth_u8, kernel, iterations=1).astype(bool)
    pred_distance = distance_transform_edt(~pred_boundary)
    truth_distance = distance_transform_edt(~truth_boundary)
    tolerance_px = BOUNDARY_TOLERANCE_UM / um_per_px
    precision = float(np.mean(truth_distance[pred_boundary] <= tolerance_px))
    recall = float(np.mean(pred_distance[truth_boundary] <= tolerance_px))
    boundary_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    forward = truth_distance[pred_boundary] * um_per_px
    reverse = pred_distance[truth_boundary] * um_per_px
    hd95 = max(float(np.quantile(forward, 0.95)), float(np.quantile(reverse, 0.95)))
    return float(dice), float(boundary_f1), float(hd95)


def _ccc(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    keep = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[keep], predicted[keep]
    if len(observed) < 2:
        return float("nan")
    covariance = float(np.cov(observed, predicted, ddof=1)[0, 1])
    denominator = float(
        np.var(observed, ddof=1)
        + np.var(predicted, ddof=1)
        + (np.mean(observed) - np.mean(predicted)) ** 2
    )
    return 2.0 * covariance / denominator if denominator > 0 else float("nan")


def _bootstrap(
    source_units: Sequence[str], statistic: Callable[[list[str]], float], *, seed_offset: int
) -> tuple[float, float]:
    values = list(source_units)
    _require(len(values) >= 2, "bootstrap requires at least two source units")
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    estimates = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index in range(BOOTSTRAP_REPETITIONS):
        selected = generator.integers(0, len(values), len(values))
        estimates[index] = statistic([values[item] for item in selected])
    finite = estimates[np.isfinite(estimates)]
    _require(len(finite) >= BOOTSTRAP_REPETITIONS * 0.95, "bootstrap produced too many undefined replicates")
    low, high = np.quantile(finite, (0.025, 0.975))
    return float(low), float(high)


def _metric_row(
    *, domain: str, key: str, label_text: str, value: float, interval: tuple[float, float], unit: str, n: int, definition: str, instances: int | None = None
) -> dict[str, Any]:
    _require(math.isfinite(value) and all(math.isfinite(item) for item in interval), f"{key}: non-finite metric")
    return {
        "domain": domain,
        "metric_key": key,
        "label": label_text,
        "value": float(value),
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
        "unit": unit,
        "n": int(n),
        "instances": int(instances if instances is not None else n),
        "definition": definition,
        "ci_method": "image/source-unit nonparametric bootstrap",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _component_metric_row(
    *,
    domain: str,
    key: str,
    label_text: str,
    point: Any,
    bootstrap_interval: Mapping[str, Any],
    unit: str,
    n: int,
    instances: int,
    definition: str,
) -> dict[str, Any]:
    """Copy one component point/CI into the shared metrics table without drift."""

    _require(
        isinstance(bootstrap_interval, Mapping)
        and bootstrap_interval.get("estimable_replicates")
        == BOOTSTRAP_REPETITIONS,
        f"{key}: component bootstrap is incomplete",
    )
    point_value = float(point)
    _require(
        math.isclose(
            float(bootstrap_interval.get("point_estimate", float("nan"))),
            point_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{key}: component point estimate differs from its bootstrap receipt",
    )
    return _metric_row(
        domain=domain,
        key=key,
        label_text=label_text,
        value=point_value,
        interval=(
            float(bootstrap_interval["ci_low_2_5"]),
            float(bootstrap_interval["ci_high_97_5"]),
        ),
        unit=unit,
        n=n,
        instances=instances,
        definition=definition,
    )


def _scale_truth_summary(
    manifest: pd.DataFrame, task_ids: Sequence[str]
) -> dict[str, Any]:
    """Recompute the QC-development scale applicability denominator.

    A trusted metadata calibration is usable for physical measurements, but it
    is not a negative visible-scale example.  Consequently the seven metadata-
    only QC-development images cannot estimate scale-absence specificity.
    """

    _require(
        {"task_id", "scale_status", "scale_bar_count"}.issubset(manifest.columns),
        "dataset manifest scale-truth columns are incomplete",
    )
    selected = manifest[
        manifest["task_id"].astype(str).isin(map(str, task_ids))
    ].copy()
    _require(
        len(selected) == selected["task_id"].nunique() == len(task_ids),
        "QC-development scale applicability rows do not close the task set",
    )
    status = selected["scale_status"].astype(str).str.strip().str.casefold()
    count = pd.to_numeric(selected["scale_bar_count"], errors="coerce")
    _require(
        np.isfinite(count).all(),
        "QC-development scale-bar counts are non-finite",
    )
    visible = status == "visible"
    trusted = status == "trusted_metadata"
    absent_or_untrusted = ~(visible | trusted)
    _require(
        bool((count[visible] == 1).all())
        and bool((count[trusted] == 0).all()),
        "visible/trusted-metadata scale status disagrees with scale-bar count",
    )
    summary = {
        "qcdevelopment_images": int(len(selected)),
        "visible_annotated_scale_bar_cases": int(visible.sum()),
        "trusted_metadata_without_visible_bar_cases": int(trusted.sum()),
        "absent_or_untrusted_scale_truth_cases": int(absent_or_untrusted.sum()),
        "absence_specificity_status": SCALE_ABSENCE_SPECIFICITY_STATUS,
        "fail_closed_evidence_basis": SCALE_FAIL_CLOSED_EVIDENCE_BASIS,
        "empirical_absence_specificity_claimed": False,
    }
    _require(
        summary["qcdevelopment_images"] == 44
        and summary["visible_annotated_scale_bar_cases"] == 37
        and summary["trusted_metadata_without_visible_bar_cases"] == 7
        and summary["absent_or_untrusted_scale_truth_cases"] == 0
        and sum(
            summary[key]
            for key in (
                "visible_annotated_scale_bar_cases",
                "trusted_metadata_without_visible_bar_cases",
                "absent_or_untrusted_scale_truth_cases",
            )
        )
        == summary["qcdevelopment_images"],
        "QC-development scale applicability must close as 37 visible + 7 trusted metadata + 0 absence-test cases",
    )
    return summary


def _shape_points(annotation: Mapping[str, Any], labels: set[str]) -> list[np.ndarray]:
    return [
        np.asarray(shape.get("points"), dtype=np.float64)
        for shape in annotation.get("shapes", [])
        if str(shape.get("label")) in labels
    ]


def _polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _resample_polyline(points: np.ndarray, step_um: float = 2.0) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    _require(
        values.ndim == 2
        and values.shape[1] == 2
        and len(values) >= 2
        and np.all(np.isfinite(values)),
        "trajectory polyline is invalid",
    )
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    _require(cumulative[-1] > 0.0, "trajectory polyline has zero length")
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-12))
    cumulative = cumulative[keep]
    values = values[keep]
    positions = np.linspace(
        0.0,
        cumulative[-1],
        max(2, int(math.ceil(cumulative[-1] / step_um)) + 1),
    )
    return np.column_stack(
        (
            np.interp(positions, cumulative, values[:, 0]),
            np.interp(positions, cumulative, values[:, 1]),
        )
    )


def _trajectory_continuity(
    predicted_um: np.ndarray, truth_um: np.ndarray, tolerance_um: float = 20.0
) -> float:
    """Return the smaller bidirectional covered-arc fraction.

    This is a secondary geometry-assurance outcome, not an identity gate: both
    curves are resampled every 2 um and the less-covered direction determines
    the score.  A missing distal segment therefore lowers continuity without
    reclassifying an otherwise valid biological-hair identity.
    """

    predicted = _resample_polyline(predicted_um)
    truth = _resample_polyline(truth_um)
    prediction_coverage = float(
        np.mean(cKDTree(truth).query(predicted, k=1)[0] <= tolerance_um)
    )
    truth_coverage = float(
        np.mean(cKDTree(predicted).query(truth, k=1)[0] <= tolerance_um)
    )
    return min(prediction_coverage, truth_coverage)


def _match_lengths(
    prediction: Mapping[str, Any], truth_polylines_um: Sequence[np.ndarray], um_per_px: float
) -> tuple[list[tuple[int, int, float]], list[np.ndarray]]:
    predicted = []
    for hair in prediction.get("length_hairs", []):
        points = np.asarray(hair.get("points_xy"), dtype=np.float64)
        if points.ndim == 2 and points.shape[1] == 2 and len(points) >= 2 and np.all(np.isfinite(points)):
            predicted.append(points * um_per_px)
    if not predicted or not truth_polylines_um:
        return [], predicted
    cost = np.linalg.norm(
        np.asarray([points[0] for points in predicted])[:, None, :]
        - np.asarray([points[0] for points in truth_polylines_um])[None, :, :],
        axis=2,
    )
    rows, columns = linear_sum_assignment(np.where(cost <= 20.0, cost, 1e9))
    return [
        (int(row), int(column), float(cost[row, column]))
        for row, column in zip(rows, columns, strict=True)
        if cost[row, column] <= 20.0
    ], predicted


def build_measurement_assurance(
    *,
    output: str | Path,
    train399_evaluation: str | Path,
    qcdev_stageb_summary: str | Path,
    qcdev_fusion_summary: str | Path,
    qcdev_fusion_root: str | Path,
    application_fusion_summary: str | Path,
    application_fusion_root: str | Path,
    dataset_root: str | Path,
    dataset_manifest: str | Path,
    split_manifest: str | Path,
    clean_traits: str | Path,
    clean_image_traits: str | Path,
    cohorts_receipt: str | Path,
    root_exact283_receipt: str | Path,
    trait_contract: str | Path = PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json",
) -> dict[str, Any]:
    destination = Path(output).resolve()
    _require(not destination.exists(), f"refusing to overwrite {destination}")
    paths = {
        "train399_evaluation": Path(train399_evaluation).resolve(),
        "qcdev_stageb_summary": Path(qcdev_stageb_summary).resolve(),
        "qcdev_fusion_summary": Path(qcdev_fusion_summary).resolve(),
        "application_fusion_summary": Path(application_fusion_summary).resolve(),
        "dataset_manifest": Path(dataset_manifest).resolve(),
        "split_manifest": Path(split_manifest).resolve(),
        "clean_traits": Path(clean_traits).resolve(),
        "clean_image_traits": Path(clean_image_traits).resolve(),
        "cohorts_receipt": Path(cohorts_receipt).resolve(),
        "root_exact283_receipt": Path(root_exact283_receipt).resolve(),
        "trait_contract": Path(trait_contract).resolve(),
    }
    for role, path in paths.items():
        _require(path.is_file() and not path.is_symlink(), f"{role}: missing/non-regular file")
        _require("blind" not in str(path).casefold(), f"{role}: blind-labelled path refused")
    data_root = Path(dataset_root).resolve()
    stageb_root = paths["qcdev_stageb_summary"].parent.resolve()
    fusion_root = Path(qcdev_fusion_root).resolve()
    application_root = Path(application_fusion_root).resolve()
    _require(
        data_root.is_dir()
        and stageb_root.is_dir()
        and fusion_root.is_dir()
        and application_root.is_dir(),
        "dataset/QC-development Stage-B/fusion/application fusion root missing",
    )

    evaluation = read_json(paths["train399_evaluation"])
    stageb = read_json(paths["qcdev_stageb_summary"])
    fusion = read_json(paths["qcdev_fusion_summary"])
    application_fusion = read_json(paths["application_fusion_summary"])
    cohorts = read_json(paths["cohorts_receipt"])
    exact = read_json(paths["root_exact283_receipt"])
    trait_contract_payload = read_json(paths["trait_contract"])
    _require(evaluation.get("schema_version") == EVALUATION_SCHEMA and evaluation.get("status") == "completed", "train399 evaluator schema/status changed")
    _require(stageb.get("schema_version") == STAGEB_SCHEMA and stageb.get("status") == "completed", "QCdev Stage-B summary schema/status changed")
    _require(fusion.get("schema_version") == FUSION_SCHEMA and fusion.get("status") == "completed", "QCdev fusion summary schema/status changed")
    _require(
        application_fusion.get("schema_version") == FUSION_SCHEMA
        and application_fusion.get("status") == "completed"
        and application_fusion.get("images") == 283,
        "application fusion summary is not a completed exact283 run",
    )
    _sealed(stageb, "summary_identity_sha256", "QCdev Stage-B summary")
    _sealed(fusion, "summary_identity_sha256", "QCdev fusion summary")
    _sealed(
        application_fusion,
        "summary_identity_sha256",
        "application fusion summary",
    )
    _require(cohorts.get("schema_version") == COHORT_SCHEMA, "cohort receipt schema changed")
    _require(exact.get("schema_version") == ROOT_EXACT_SCHEMA and exact.get("status") == "pass_exact_283", "root exact283 receipt did not pass")
    for role, payload in (("evaluation", evaluation), ("stageb", stageb), ("fusion", fusion), ("application_fusion", application_fusion), ("cohorts", cohorts), ("root_exact283", exact)):
        _require(payload.get("blind_images_used") == 0, f"{role}: blind guard changed")
    _require(
        evaluation.get("inputs_sha256", {}).get("dataset_manifest") == sha256_file(paths["dataset_manifest"])
        and evaluation.get("inputs_sha256", {}).get("split_manifest") == sha256_file(paths["split_manifest"]),
        "evaluator does not bind named canonical manifests",
    )
    rows = evaluation.get("per_image")
    _require(isinstance(rows, list) and len(rows) == 44, "evaluation is not QC44")
    task_ids = [str(row["task_id"]) for row in rows]
    _require(
        len(task_ids) == len(set(task_ids))
        and all(
            Path(f"{task_id}.json").name == f"{task_id}.json"
            and not any(separator in task_id for separator in ("/", "\\"))
            for task_id in task_ids
        ),
        "QCdev task IDs are duplicated or unsafe as artifact basenames",
    )
    stageb_authority = _validate_stageb_authority(
        evaluation, stageb, task_ids, stageb_root
    )
    _require(
        fusion.get("source_stageb_summary_sha256") == sha256_file(paths["qcdev_stageb_summary"]),
        "QCdev fusion does not bind the named Stage-B summary",
    )
    fusion_records = fusion.get("records")
    _require(isinstance(fusion_records, list) and [str(row.get("task_id")) for row in fusion_records] == task_ids, "QCdev fusion task order changed")
    _require(fusion.get("images") == 44, "QCdev fusion is not 44 images")
    for field in (
        "model_contract_proposal_sha256",
        "model_contract_proposal_identity_sha256",
        "model_bundle_id",
    ):
        _require(
            stageb.get(field) == fusion.get(field),
            f"QCdev Stage-B/fusion {field} authority drift",
        )
    _require(
        stageb.get("root_expert_id") == fusion.get("root_expert")
        and fusion.get("hair_identity_count_expert")
        == stageb_authority["shared_model_authority"]["expert_id"]
        and fusion.get("condition_metadata_used_for_routing") is False
        and fusion.get("canonical_annotations_read") is False
        and fusion.get("root_cap_region_output") is False
        and fusion.get("blind_images_used") == 0,
        "QCdev Stage-B/fusion expert or guard authority drift",
    )
    detection_directory = stageb_root / "detections"
    _require(
        detection_directory.is_dir() and not detection_directory.is_symlink(),
        "QCdev production Stage-B detection directory is missing or symlinked",
    )
    expected_detection_names = {f"{task_id}.json" for task_id in task_ids}
    actual_detection_paths = list(detection_directory.glob("*.json"))
    _require(
        len(actual_detection_paths) == len(task_ids)
        and {path.name for path in actual_detection_paths}
        == expected_detection_names,
        "QCdev production Stage-B detection directory has missing/extra JSON files",
    )

    manifest = _read_csv(
        paths["dataset_manifest"],
        (
            "task_id", "split", "image_sha256", "image_width", "image_height",
            "source_um_per_px", "root_mask_relpath", "canonical_annotation_relpath",
            "scale_status", "scale_bar_count", "scale_bar_value_um",
        ),
        "dataset manifest",
    )
    by_task = {
        str(row.task_id): row for row in manifest.itertuples(index=False)
    }
    _require(
        set(task_ids).issubset(by_task),
        "QCdev tasks are missing from the canonical dataset manifest",
    )
    split_manifest = _read_csv(
        paths["split_manifest"],
        ("task_id", "split"),
        "locked QC-development split manifest",
    )
    locked_split_by_task = {
        str(row.task_id): str(row.split)
        for row in split_manifest.itertuples(index=False)
    }
    _require(
        set(task_ids).issubset(locked_split_by_task)
        and all(locked_split_by_task[task] == "val" for task in task_ids),
        "QCdev tasks are not the locked validation split authority",
    )
    scale_truth_summary = _scale_truth_summary(manifest, task_ids)
    canonical_gt, canonical_provenance = load_canonical_qcdev_ground_truth(
        dataset_root=data_root,
        dataset_manifest=paths["dataset_manifest"],
        split_manifest=paths["split_manifest"],
        expected_task_ids=task_ids,
    )
    root_trait_contract_rows = trait_contract_payload.get("primary_root_traits")
    _require(
        trait_contract_payload.get("schema_version")
        == "PHAxis-trait-contract-1.0.0"
        and isinstance(root_trait_contract_rows, list)
        and len(root_trait_contract_rows) == 19,
        "canonical 19-trait contract is missing or changed",
    )
    root_trait_contract_by_field = {
        str(record.get("field")): record
        for record in root_trait_contract_rows
        if isinstance(record, Mapping)
    }
    _require(
        tuple(root_trait_contract_by_field) == ROOT_TRAIT_FIELDS,
        "root trait contract field/order drift",
    )

    root_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    root_trait_rows: list[dict[str, Any]] = []
    prediction_file_locks: list[dict[str, str]] = []
    fused_assurance_input_locks: list[dict[str, Any]] = []
    root_continuity_records: list[dict[str, Any]] = []
    hair_attachment_records: list[dict[str, Any]] = []
    fusion_by_task = {str(row["task_id"]): row for row in fusion_records}
    stageb_by_task = {
        str(row["task_id"]): row for row in stageb.get("records", [])
    }
    stageb_file_sha256_by_task = {
        str(row["task_id"]): str(row["sha256"])
        for row in stageb_authority["production_detection_files"]
    }
    evaluation_by_task = {str(row["task_id"]): row for row in rows}
    for task_id in task_ids:
        meta = by_task[task_id]
        prediction_path = _locked_regular_file(
            fusion_root,
            Path("predictions") / f"{task_id}.json",
            f"{task_id} fused prediction",
        )
        expected_prediction_sha = str(fusion_by_task[task_id].get("prediction_sha256"))
        _require(
            _is_sha256(expected_prediction_sha)
            and sha256_file(prediction_path) == expected_prediction_sha,
            f"{task_id}: fused prediction hash mismatch",
        )
        prediction_file_locks.append({"task_id": task_id, "sha256": expected_prediction_sha})
        prediction = read_json(prediction_path)
        validate_hybrid_prediction(prediction, artifact_root=fusion_root)
        _require(prediction.get("source_image_sha256") == str(meta.image_sha256), f"{task_id}: source image identity mismatch")
        height, width = int(meta.image_height), int(meta.image_width)

        stageb_record = stageb_by_task[task_id]
        detection_path = _locked_regular_file(
            stageb_root,
            Path("detections") / f"{task_id}.json",
            f"{task_id} production Stage-B detection",
        )
        detection_file_sha256 = stageb_file_sha256_by_task[task_id]
        _require(
            sha256_file(detection_path) == detection_file_sha256,
            f"{task_id}: production Stage-B detection file hash mismatch",
        )
        stageb_detection = read_json(detection_path)
        try:
            validate_stageb_detection_payload(
                stageb_detection,
                expected_task_id=task_id,
                expected_image_sha256=str(meta.image_sha256),
                expected_model_metadata=stageb["detection_model_metadata"],
            )
        except ContractError as error:
            raise MeasurementAssuranceError(
                f"{task_id}: invalid production Stage-B detection: {error}"
            ) from error
        _require(
            stageb_detection.get("detection_identity_sha256")
            == stageb_record.get("detection_identity_sha256")
            and stageb_detection.get("n") == stageb_record.get("detections"),
            f"{task_id}: production Stage-B record/payload identity or count drift",
        )
        for field in (
            "model_contract_proposal_sha256",
            "model_contract_proposal_identity_sha256",
            "model_bundle_id",
            "root_expert_id",
        ):
            _require(
                stageb_detection.get(field) == stageb.get(field),
                f"{task_id}: production Stage-B {field} drift",
            )
        coordinate_space = stageb_detection["coordinate_space"]
        _require(
            tuple(int(value) for value in coordinate_space.get("source_shape", ()))
            == (height, width)
            and math.isclose(
                float(coordinate_space["source_um_per_px"]),
                float(meta.source_um_per_px),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{task_id}: production Stage-B source geometry/scale drift",
        )
        predicted_hair_polylines_um = _validate_fused_stageb_identity(
            prediction, stageb_detection
        )
        phaxis = prediction["phaxis"]
        for field in (
            "model_contract_proposal_sha256",
            "model_contract_proposal_identity_sha256",
            "model_bundle_id",
        ):
            _require(
                phaxis.get(field) == stageb.get(field),
                f"{task_id}: fused PHAxis {field} drift",
            )
        _require(
            phaxis.get("root_expert") == stageb.get("root_expert_id"),
            f"{task_id}: fused PHAxis root expert drift",
        )

        truth_mask_path = _locked_regular_file(
            data_root, str(meta.root_mask_relpath), f"{task_id} truth root"
        )
        truth_mask = _read_mask(truth_mask_path, (height, width), f"{task_id} truth root")
        predicted_mask_path = _locked_regular_file(
            fusion_root,
            str(prediction["root_mask_relpath"]),
            f"{task_id} predicted root",
        )
        _require(sha256_file(predicted_mask_path) == prediction["root_mask_sha256"], f"{task_id}: predicted root-mask hash mismatch")
        predicted_mask = _read_mask(predicted_mask_path, (height, width), f"{task_id} predicted root")
        um_per_px = float(meta.source_um_per_px)
        annotation_path = _locked_regular_file(
            data_root,
            str(meta.canonical_annotation_relpath),
            f"{task_id} canonical annotation",
        )
        annotation = read_json(annotation_path)
        truth_tip_shapes = _shape_points(annotation, {"root_tip"})
        _require(
            len(truth_tip_shapes) == 1 and truth_tip_shapes[0].shape == (1, 2),
            f"{task_id}: canonical annotation must contain exactly one root_tip point",
        )
        truth_tip = truth_tip_shapes[0][0]
        reference_axis_xy_px, _reference_s_px, _reference_radius_px = _root_axis(
            truth_mask, truth_tip
        )
        reference_axis_xy_um = reference_axis_xy_px * um_per_px
        predicted_axis_components_xy_um = _skeleton_components_xy_um(
            predicted_mask, um_per_px=um_per_px
        )
        reference_axis_artifact_sha256 = sha256_json(
            {
                "canonical_root_mask_file_sha256": sha256_file(truth_mask_path),
                "canonical_annotation_file_sha256": sha256_file(annotation_path),
                "annotated_distal_point_xy_px": truth_tip.tolist(),
                "ordered_reference_axis_xy_um": reference_axis_xy_um.tolist(),
            }
        )
        root_continuity_records.append(
            {
                "pair_type": "primary_root_continuity",
                "source_unit": task_id,
                "source_image_sha256": str(meta.image_sha256),
                "coordinate_space": ROOT_CONTINUITY_COORDINATE_SPACE,
                "reference_axis_definition": ROOT_CONTINUITY_REFERENCE_DEFINITION,
                "prediction_axis_definition": ROOT_CONTINUITY_PREDICTION_DEFINITION,
                "reference_axis_artifact_sha256": reference_axis_artifact_sha256,
                "prediction_axis_artifact_sha256": str(
                    prediction["root_mask_sha256"]
                ),
                "reference_axis_xy_um": reference_axis_xy_um.tolist(),
                "predicted_axis_components_xy_um": predicted_axis_components_xy_um,
            }
        )
        annotated_hair_polylines_um = [
            np.asarray(polyline, dtype=np.float64)
            for polyline in canonical_gt[task_id]["polys"]
        ]
        hair_attachment_records.append(
            {
                "pair_type": "hair_attachment",
                "source_unit": task_id,
                "source_image_sha256": str(meta.image_sha256),
                "coordinate_space": HAIR_ATTACHMENT_COORDINATE_SPACE,
                "polyline_orientation": HAIR_POLYLINE_ORIENTATION,
                "annotation_artifact_sha256": sha256_file(annotation_path),
                "prediction_artifact_sha256": detection_file_sha256,
                "predicted_polylines_xy_um": [
                    polyline.tolist() for polyline in predicted_hair_polylines_um
                ],
                "annotated_polylines_xy_um": [
                    polyline.tolist() for polyline in annotated_hair_polylines_um
                ],
            }
        )
        fused_assurance_input_locks.append(
            {
                "task_id": task_id,
                "source_image_sha256": str(meta.image_sha256),
                "fused_prediction_sha256": expected_prediction_sha,
                "final_fused_root_mask_sha256": str(
                    prediction["root_mask_sha256"]
                ),
                "production_stageb_detection_sha256": detection_file_sha256,
                "production_stageb_detection_identity_sha256": str(
                    stageb_detection["detection_identity_sha256"]
                ),
                "stageb_identity_hairs_identity_sha256": sha256_json(
                    prediction["identity_hairs"]
                ),
                "stageb_identity_hair_count": len(prediction["identity_hairs"]),
                "evaluator_stageb_biological_presence_20um": (
                    _evaluation_stageb_presence_counts(
                        evaluation_by_task[task_id],
                        role=f"{task_id} evaluator",
                    )
                ),
            }
        )
        predicted_tip = np.asarray(prediction["root_cap_point_xy"], dtype=np.float64)
        dice, boundary_f1, hd95 = _root_overlap_metrics(predicted_mask, truth_mask, um_per_px)
        point_error = float(np.linalg.norm(predicted_tip - truth_tip) * um_per_px)
        root_rows.append(
            {
                "source_unit": task_id,
                "root_dice": dice,
                "root_boundary_f1": boundary_f1,
                "root_hd95_um": hd95,
                "distal_error_um": point_error,
            }
        )
        truth_statistics = _root_statistics(truth_mask, truth_tip, um_per_px)
        predicted_statistics = prediction.get("detailed_root_statistics")
        _require(isinstance(predicted_statistics, Mapping), f"{task_id}: detailed root statistics missing")
        for field in ROOT_TRAIT_FIELDS:
            observed = float(truth_statistics[field])
            predicted_value = float(predicted_statistics[field])
            _require(math.isfinite(observed) and math.isfinite(predicted_value), f"{task_id}/{field}: root trait is non-finite")
            contract_row = root_trait_contract_by_field[field]
            root_trait_rows.append(
                {
                    "pair_type": "root_trait",
                    "source_unit": task_id,
                    "pair_id": f"{task_id}:{field}",
                    "trait_id": str(contract_row["id"]),
                    "trait_key": field,
                    "trait_family": ROOT_TRAIT_FAMILY_BY_FIELD[field],
                    "observed": observed,
                    "predicted": predicted_value,
                    "unit": str(contract_row["unit"]),
                    "reference_observable": True,
                    "prediction_observable": True,
                    "agreement_eligible": True,
                    "ineligibility_reason": "",
                    "reference_definition": ROOT_TRAIT_REFERENCE_DEFINITION,
                    "prediction_definition": ROOT_TRAIT_PREDICTION_DEFINITION,
                    "source_image_sha256": str(meta.image_sha256),
                }
            )

        scale_count = int(meta.scale_bar_count)
        scale_status = str(meta.scale_status).strip().casefold()
        _require(
            (scale_status == "visible" and scale_count == 1)
            or (scale_status == "trusted_metadata" and scale_count == 0),
            f"{task_id}: scale applicability status/count drift",
        )
        scale = prediction.get("scale", {})
        detected = bool(
            isinstance(scale, Mapping)
            and scale.get("predicted_visible") is True
            and scale.get("fail_closed") is False
            and scale.get("predicted_um_per_px") is not None
            and math.isfinite(float(scale["predicted_um_per_px"]))
        )
        if scale_status == "visible":
            scale_record: dict[str, Any] = {
                "source_unit": task_id,
                "detected": detected,
                "observed_um_per_px": um_per_px,
                "predicted_um_per_px": float(scale["predicted_um_per_px"]) if detected else np.nan,
                "relative_error_percent": (
                    abs(float(scale["predicted_um_per_px"]) - um_per_px) / um_per_px * 100.0
                    if detected else np.nan
                ),
                "geometry_endpoint_error_um": np.nan,
                "source_image_sha256": str(meta.image_sha256),
            }
            reference_lines = _shape_points(annotation, {"scale_bar", "scale"})
            _require(
                len(reference_lines) == 1 and reference_lines[0].shape == (2, 2),
                f"{task_id}: visible scale truth must contain exactly one two-endpoint line",
            )
            predicted_points = np.asarray(scale.get("predicted_points_xy"), dtype=np.float64) if detected else np.empty((0, 2))
            if detected:
                _require(
                    predicted_points.shape == (2, 2)
                    and np.isfinite(predicted_points).all(),
                    f"{task_id}: detected scale lacks finite two-endpoint geometry",
                )
                direct = np.linalg.norm(reference_lines[0] - predicted_points, axis=1).mean()
                flipped = np.linalg.norm(reference_lines[0] - predicted_points[::-1], axis=1).mean()
                scale_record["geometry_endpoint_error_um"] = min(float(direct), float(flipped)) * um_per_px
            scale_rows.append(scale_record)

        match_result = _match_lengths(
            prediction,
            [np.asarray(polyline, dtype=np.float64) for polyline in canonical_gt[task_id]["polys"]],
            um_per_px,
        )
        matches, predicted_polylines = match_result
        for index, (predicted_index, truth_index, base_error) in enumerate(matches):
            predicted_curve = predicted_polylines[predicted_index]
            truth_curve = np.asarray(
                canonical_gt[task_id]["polys"][truth_index], dtype=np.float64
            )
            length_rows.append(
                {
                    "pair_type": "conditional_length",
                    "source_unit": task_id,
                    "pair_id": f"{task_id}:length:{index:04d}",
                    "trait_key": "endpoint_complete_hair_length_um",
                    "observed": _polyline_length(truth_curve),
                    "predicted": _polyline_length(predicted_curve),
                    "unit": "um",
                    "base_match_error_um": base_error,
                    "endpoint_error_um": float(
                        np.linalg.norm(predicted_curve[-1] - truth_curve[-1])
                    ),
                    "trajectory_continuity": _trajectory_continuity(
                        predicted_curve, truth_curve
                    ),
                    "source_image_sha256": str(meta.image_sha256),
                }
            )

    _require(
        len(root_continuity_records)
        == len(hair_attachment_records)
        == len(fused_assurance_input_locks)
        == len(task_ids)
        == 44,
        "QCdev component-assurance input denominator drift",
    )
    canonical_authority_identity = canonical_provenance[
        "canonical_ground_truth_lock_identity_sha256"
    ]
    root_prediction_authority_locks = [
        {
            "task_id": row["task_id"],
            "fused_prediction_sha256": row["fused_prediction_sha256"],
            "final_fused_root_mask_sha256": row[
                "final_fused_root_mask_sha256"
            ],
        }
        for row in fused_assurance_input_locks
    ]
    hair_prediction_authority_locks = [
        {
            "task_id": row["task_id"],
            "fused_prediction_sha256": row["fused_prediction_sha256"],
            "production_stageb_detection_sha256": row[
                "production_stageb_detection_sha256"
            ],
            "production_stageb_detection_identity_sha256": row[
                "production_stageb_detection_identity_sha256"
            ],
            "stageb_identity_hairs_identity_sha256": row[
                "stageb_identity_hairs_identity_sha256"
            ],
            "stageb_identity_hair_count": row["stageb_identity_hair_count"],
        }
        for row in fused_assurance_input_locks
    ]
    root_prediction_authority_identity = sha256_json(
        root_prediction_authority_locks
    )
    hair_prediction_authority_identity = sha256_json(
        hair_prediction_authority_locks
    )
    root_continuity_input = _seal_portable_input_contract(
        {
            "schema_version": ROOT_CONTINUITY_INPUT_SCHEMA,
            "source_units": list(task_ids),
            "reference_authority_sha256": canonical_authority_identity,
            "prediction_authority_identity_sha256": (
                root_prediction_authority_identity
            ),
            "metric_config": {
                "support_tolerance_um": 5.0,
                "sampling_step_um": 2.0,
            },
            "records": root_continuity_records,
            "independent_accuracy_claim_allowed": False,
            "blind_images_used": 0,
        }
    )
    hair_attachment_input = _seal_portable_input_contract(
        {
            "schema_version": HAIR_ATTACHMENT_INPUT_SCHEMA,
            "source_units": list(task_ids),
            "annotation_authority_sha256": canonical_authority_identity,
            "prediction_authority_identity_sha256": (
                hair_prediction_authority_identity
            ),
            "metric_config": {
                "proxy_tolerances_um": list(PROXY_TOLERANCES_UM),
                "selected_proxy_tolerance_um": SELECTED_PROXY_TOLERANCE_UM,
                "formal_attachment_tolerance_um": (
                    FORMAL_ATTACHMENT_TOLERANCE_UM
                ),
                "formal_matcher": dict(FORMAL_MATCHER_CONFIG),
            },
            "records": hair_attachment_records,
            "independent_accuracy_claim_allowed": False,
            "blind_images_used": 0,
        }
    )
    root_continuity_assurance = build_root_continuity_from_input_contract(
        root_continuity_input
    )
    hair_attachment_assurance = build_hair_attachment_from_input_contract(
        hair_attachment_input
    )
    (
        hair_biological_presence_crosscheck_locks,
        hair_biological_presence_crosscheck_identity,
    ) = _crosscheck_hair_biological_presence(
        evaluation, hair_attachment_assurance, task_ids
    )

    root_frame = pd.DataFrame(root_rows)
    scale_frame = pd.DataFrame(scale_rows)
    length_frame = pd.DataFrame(length_rows)
    root_trait_frame = pd.DataFrame(root_trait_rows)
    _require(len(root_frame) == 44, "root assurance did not produce 44 rows")
    _require(
        len(scale_frame)
        == scale_frame["source_unit"].nunique()
        == scale_frame["source_image_sha256"].nunique()
        == scale_truth_summary["visible_annotated_scale_bar_cases"]
        == 37
        and scale_frame["detected"].sum() >= 2,
        "scale assurance does not contain the 37 unique visible-bar source images",
    )
    _require(len(length_frame) >= 2 and length_frame["source_unit"].nunique() >= 2, "conditional-length assurance has insufficient matches")

    root_ids = list(root_frame["source_unit"].astype(str))
    root_trait_assurance = build_root_trait_assurance(
        pairs=root_trait_frame.to_dict("records"),
        trait_contract=trait_contract_payload,
        source_units=root_ids,
        trait_contract_file_sha256=sha256_file(paths["trait_contract"]),
        reference_authority_sha256=canonical_provenance[
            "canonical_ground_truth_lock_identity_sha256"
        ],
        prediction_authority_identity_sha256=sha256_json(prediction_file_locks),
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    root_lookup = root_frame.set_index("source_unit")
    metrics: list[dict[str, Any]] = []
    for offset, (key, label_text, reducer, unit, definition) in enumerate(
        (
            ("root_dice", "Mean root Dice", np.mean, "fraction", "macro mean across annotated QC-development images"),
            ("root_boundary_f1", "Mean boundary F1 @5 µm", np.mean, "fraction", "macro bidirectional boundary F1 at 5-µm tolerance"),
            ("root_hd95_um", "Median root HD95", np.median, "um", "median of per-image symmetric Hausdorff-95 distances"),
        ),
        start=1,
    ):
        statistic = lambda sampled, column=key, function=reducer: float(function([root_lookup.loc[item, column] for item in sampled]))
        point = float(reducer(root_frame[key]))
        metrics.append(
            _metric_row(
                domain="root", key=key, label_text=label_text, value=point,
                interval=_bootstrap(root_ids, statistic, seed_offset=offset), unit=unit,
                n=len(root_ids), definition=definition,
            )
        )
    distal_values = root_frame["distal_error_um"].to_numpy(dtype=float)
    for offset, (key, label_text, point, statistic, unit, definition) in enumerate(
        (
            ("distal_median_error_um", "Median distal-point error", float(np.median(distal_values)), lambda sampled: float(np.median([root_lookup.loc[item, "distal_error_um"] for item in sampled])), "um", "single visible distal/root-cap point physical error"),
            ("distal_pck", f"PCK @{PCK_THRESHOLD_UM:g} µm", float(np.mean(distal_values <= PCK_THRESHOLD_UM)), lambda sampled: float(np.mean([root_lookup.loc[item, "distal_error_um"] <= PCK_THRESHOLD_UM for item in sampled])), "fraction", f"fraction with distal-point error <= {PCK_THRESHOLD_UM:g} µm"),
            ("distal_pck_10um", "PCK @10 µm", float(np.mean(distal_values <= 10.0)), lambda sampled: float(np.mean([root_lookup.loc[item, "distal_error_um"] <= 10.0 for item in sampled])), "fraction", "fraction with distal-point error <=10 µm"),
            ("distal_pck_50um", "PCK @50 µm", float(np.mean(distal_values <= 50.0)), lambda sampled: float(np.mean([root_lookup.loc[item, "distal_error_um"] <= 50.0 for item in sampled])), "fraction", "fraction with distal-point error <=50 µm"),
        ),
        start=10,
    ):
        metrics.append(_metric_row(domain="distal", key=key, label_text=label_text, value=point, interval=_bootstrap(root_ids, statistic, seed_offset=offset), unit=unit, n=len(root_ids), definition=definition))

    scale_ids = list(scale_frame["source_unit"].astype(str))
    scale_lookup = scale_frame.set_index("source_unit")
    coverage = float(scale_frame["detected"].mean())
    coverage_stat = lambda sampled: float(np.mean([bool(scale_lookup.loc[item, "detected"]) for item in sampled]))
    detected_scale = scale_frame[scale_frame["detected"]].copy()
    detected_ids = list(detected_scale["source_unit"].astype(str))
    detected_lookup = detected_scale.set_index("source_unit")
    scale_error = float(np.median(detected_scale["relative_error_percent"]))
    scale_error_stat = lambda sampled: float(np.median([detected_lookup.loc[item, "relative_error_percent"] for item in sampled]))
    metrics.append(_metric_row(domain="scale", key="scale_detection_coverage", label_text="Visible scale detection coverage", value=coverage, interval=_bootstrap(scale_ids, coverage_stat, seed_offset=20), unit="fraction", n=len(scale_ids), definition="visible annotated scale bars detected without fail-closed output", instances=int(scale_frame["detected"].sum())))
    metrics.append(_metric_row(domain="scale", key="scale_relative_error_percent", label_text="Median |µm px⁻¹ relative error|", value=scale_error, interval=_bootstrap(detected_ids, scale_error_stat, seed_offset=21), unit="percent", n=len(detected_ids), definition="conditional on a non-fail-closed detection of a visible annotated scale bar"))
    geometry_scale = detected_scale[np.isfinite(detected_scale["geometry_endpoint_error_um"])].copy()
    _require(
        len(geometry_scale) == len(detected_scale) >= 2,
        "every detected visible scale bar must support line-endpoint localization assurance",
    )
    geometry_ids = list(geometry_scale["source_unit"].astype(str))
    geometry_lookup = geometry_scale.set_index("source_unit")
    geometry_stat = lambda sampled: float(np.median([geometry_lookup.loc[item, "geometry_endpoint_error_um"] for item in sampled]))
    metrics.append(_metric_row(domain="scale", key="scale_geometry_endpoint_error_um", label_text="Median scale-line endpoint error", value=float(np.median(geometry_scale["geometry_endpoint_error_um"])), interval=_bootstrap(geometry_ids, geometry_stat, seed_offset=22), unit="um", n=len(geometry_ids), definition="conditional on detection; best endpoint assignment, averaged within each visible annotated scale bar"))

    length_ids = sorted(set(length_frame["source_unit"].astype(str)))
    length_groups = {task: group for task, group in length_frame.groupby("source_unit")}
    def length_sample(sampled: list[str]) -> tuple[np.ndarray, np.ndarray]:
        observed, predicted = [], []
        for task in sampled:
            group = length_groups[task]
            observed.extend(group["observed"].astype(float))
            predicted.extend(group["predicted"].astype(float))
        return np.asarray(observed), np.asarray(predicted)
    length_observed = length_frame["observed"].to_numpy(dtype=float)
    length_predicted = length_frame["predicted"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(length_predicted - length_observed)))
    ccc = float(_ccc(length_observed, length_predicted))
    bias = float(np.mean(length_predicted - length_observed))
    metrics.append(_metric_row(domain="conditional_length", key="conditional_length_mae_um", label_text="Matched length MAE", value=mae, interval=_bootstrap(length_ids, lambda sampled: float(np.mean(np.abs(length_sample(sampled)[1] - length_sample(sampled)[0]))), seed_offset=30), unit="um", n=len(length_ids), instances=len(length_frame), definition="one-to-one attachment-matched endpoint-complete curves"))
    metrics.append(_metric_row(domain="conditional_length", key="conditional_length_ccc", label_text="Matched length CCC", value=ccc, interval=_bootstrap(length_ids, lambda sampled: float(_ccc(*length_sample(sampled))), seed_offset=31), unit="ccc", n=len(length_ids), instances=len(length_frame), definition="one-to-one attachment-matched endpoint-complete curves"))
    metrics.append(_metric_row(domain="conditional_length", key="conditional_length_bias_um", label_text="Matched length bias", value=bias, interval=_bootstrap(length_ids, lambda sampled: float(np.mean(length_sample(sampled)[1] - length_sample(sampled)[0])), seed_offset=32), unit="um", n=len(length_ids), instances=len(length_frame), definition="predicted minus annotated centreline length"))
    metrics.append(
        _metric_row(
            domain="conditional_length",
            key="matched_endpoint_error_um",
            label_text="Median matched distal-end error",
            value=float(np.median(length_frame["endpoint_error_um"])),
            interval=_bootstrap(
                length_ids,
                lambda sampled: float(
                    np.median(
                        pd.concat(
                            [length_groups[task] for task in sampled],
                            ignore_index=True,
                        )["endpoint_error_um"]
                    )
                ),
                seed_offset=33,
            ),
            unit="um",
            n=len(length_ids),
            instances=len(length_frame),
            definition="distal-end Euclidean error after one-to-one 20-um attachment matching; not an identity gate",
        )
    )
    metrics.append(
        _metric_row(
            domain="conditional_length",
            key="matched_trajectory_continuity",
            label_text="Mean matched trajectory continuity",
            value=float(np.mean(length_frame["trajectory_continuity"])),
            interval=_bootstrap(
                length_ids,
                lambda sampled: float(
                    np.mean(
                        pd.concat(
                            [length_groups[task] for task in sampled],
                            ignore_index=True,
                        )["trajectory_continuity"]
                    )
                ),
                seed_offset=34,
            ),
            unit="fraction",
            n=len(length_ids),
            instances=len(length_frame),
            definition="smaller bidirectional 20-um covered-arc fraction after 2-um resampling; secondary geometry assurance",
        )
    )

    root_trait_pairs = root_trait_frame.copy()
    trait_ids = root_ids
    trait_groups = {task: group for task, group in root_trait_pairs.groupby("source_unit")}
    def trait_agreement(sampled: list[str]) -> float:
        frames = [trait_groups[task] for task in sampled]
        combined = pd.concat(frames, ignore_index=True)
        values = []
        for field in ROOT_TRAIT_FIELDS:
            selected = combined[combined["trait_key"] == field]
            value = _ccc(selected["observed"].to_numpy(float), selected["predicted"].to_numpy(float))
            if math.isfinite(value):
                values.append(value)
        return float(np.median(values)) if values else float("nan")
    trait_ccc_values = [
        float(row["ccc"])
        for row in root_trait_assurance["trait_rows"]
        if row["ccc"] is not None
    ]
    _require(trait_ccc_values, "none of the 19 root-trait CCC values is estimable")
    root_trait_value = float(np.median(trait_ccc_values))
    _require(
        math.isclose(root_trait_value, trait_agreement(trait_ids), abs_tol=1e-12, rel_tol=0),
        "root-trait aggregate differs from sealed per-trait evidence",
    )
    metrics.append(_metric_row(domain="root_trait", key="root_trait_agreement", label_text="Median trait-wise CCC across 19 root traits", value=root_trait_value, interval=_bootstrap(trait_ids, trait_agreement, seed_offset=40), unit="median_ccc", n=len(trait_ids), instances=len(root_trait_pairs), definition=f"median of {len(trait_ccc_values)}/19 estimable per-trait CCC values against the canonical-mask-plus-distal-point reference; the sealed trait table also reports MAE, bias, observability, and non-CCC equivalent statistics"))

    root_continuity_summary = root_continuity_assurance["summary"]
    root_continuity_ci = root_continuity_summary["bootstrap_95ci"]
    root_metric_contract = (
        (
            "root_continuity_reference_axis_coverage_mean",
            "Mean union reference-axis coverage",
            "reference_axis_coverage_mean",
            "reference_axis_coverage_mean",
            "fraction",
            "union support diagnostic across every sealed final-mask skeleton component; not a single-component continuity claim",
        ),
        (
            "root_continuity_maximum_single_component_coverage_mean",
            "Mean maximum single-component root coverage",
            "maximum_single_component_coverage_mean",
            "maximum_single_component_coverage_mean",
            "fraction",
            "mean per-image coverage from the best one connected final-mask skeleton component",
        ),
        (
            "root_continuity_maximum_single_component_coverage_median",
            "Median maximum single-component root coverage",
            "maximum_single_component_coverage_median",
            "maximum_single_component_coverage_median",
            "fraction",
            "median per-image coverage from the best one connected final-mask skeleton component",
        ),
        (
            "root_continuity_best_component_gap_median_um",
            "Median longest gap on the best root component",
            "longest_unsupported_gap_um_on_best_component_median",
            "longest_unsupported_gap_um_on_best_component_median",
            "um",
            "median longest unsupported reference-axis gap on the maximum-coverage single connected component",
        ),
        (
            "root_continuity_break_free_rate",
            "Break-free root image rate",
            "break_free_image_rate",
            "break_free_image_rate",
            "fraction",
            "fraction of source images with at least one single connected component spanning every reference interval",
        ),
        (
            "root_continuity_visible_axis_extent_mae_um",
            "Visible root-axis extent MAE",
            "visible_axis_extent_error_um_mae",
            "visible_axis_extent_error_um_mae",
            "um",
            "mean absolute proximal-to-distal projected extent error; internal gaps are scored separately",
        ),
    )
    for key, label_text, summary_key, ci_key, unit, definition in root_metric_contract:
        metrics.append(
            _component_metric_row(
                domain="root_continuity",
                key=key,
                label_text=label_text,
                point=root_continuity_summary[summary_key],
                bootstrap_interval=root_continuity_ci[ci_key],
                unit=unit,
                n=len(task_ids),
                instances=len(task_ids),
                definition=definition,
            )
        )

    formal_attachment = hair_attachment_assurance["summary"][
        "formal_matched_attachment_accuracy"
    ]
    attachment_identity = formal_attachment["attachment_qualified_identity"]
    formal_presence = formal_attachment["formal_biological_presence"]
    attachment_errors = formal_attachment[
        "attachment_position_error_on_all_formal_identity_matches"
    ]
    attachment_ci = formal_attachment["bootstrap_95ci"]
    _require(
        attachment_identity["n_pred"]
        == formal_presence["n_pred"]
        == hair_attachment_assurance["summary"]["predicted_hairs"]
        and attachment_identity["n_gt"]
        == formal_presence["n_gt"]
        == hair_attachment_assurance["summary"]["annotated_hairs"]
        and attachment_errors["n"] == formal_presence["tp"]
        and attachment_identity["tp"] <= formal_presence["tp"],
        "formal hair-attachment denominators do not close biological-presence identities",
    )
    hair_metric_contract = (
        (
            "hair_attachment_qualified_precision_20um",
            "Attachment-qualified precision @20 µm",
            "precision",
            "formal_attachment_precision",
            int(attachment_identity["n_pred"]),
            "pooled precision whose true positives are formal biological-presence identities with base error <=20 µm",
        ),
        (
            "hair_attachment_qualified_recall_20um",
            "Attachment-qualified recall @20 µm",
            "recall",
            "formal_attachment_recall",
            int(attachment_identity["n_gt"]),
            "pooled recall whose true positives are formal biological-presence identities with base error <=20 µm",
        ),
        (
            "hair_attachment_qualified_f1_20um",
            "Attachment-qualified F1 @20 µm",
            "f1",
            "formal_attachment_f1",
            int(attachment_identity["n_pred"] + attachment_identity["n_gt"]),
            "pooled F1 from the explicit predicted/annotated denominators and attachment-qualified formal identities",
        ),
    )
    for key, label_text, point_key, ci_key, instances, definition in hair_metric_contract:
        metrics.append(
            _component_metric_row(
                domain="hair_attachment",
                key=key,
                label_text=label_text,
                point=attachment_identity[point_key],
                bootstrap_interval=attachment_ci[ci_key],
                unit="fraction",
                n=len(task_ids),
                instances=instances,
                definition=definition,
            )
        )
    for key, label_text, point_key, ci_key, definition in (
        (
            "hair_attachment_error_median_um",
            "Median base error on formal hair identities",
            "median_um",
            "formal_attachment_error_median_um",
            "median attachment/base error over all formal biological-presence matches; no base-only rematching",
        ),
        (
            "hair_attachment_error_p95_um",
            "P95 base error on formal hair identities",
            "p95_um",
            "formal_attachment_error_p95_um",
            "95th-percentile attachment/base error over all formal biological-presence matches; no base-only rematching",
        ),
    ):
        metrics.append(
            _component_metric_row(
                domain="hair_attachment",
                key=key,
                label_text=label_text,
                point=attachment_errors[point_key],
                bootstrap_interval=attachment_ci[ci_key],
                unit="um",
                n=len(task_ids),
                instances=int(attachment_errors["n"]),
                definition=definition,
            )
        )

    clean_biology = _read_csv(
        paths["clean_traits"],
        (
            "task_id", "source_image_sha256",
            "condition_code", "study_role", "formal_statistics_eligible",
            "hair_count", "hair_length_measurement_hair_count",
        ),
        "clean261 traits",
    )
    clean_image = _read_csv(
        paths["clean_image_traits"],
        (
            "task_id", "source_image_sha256", "prediction_sha256",
            "formal_statistics_eligible", "um_per_px",
        ),
        "clean261 image traits",
    )
    _require(
        cohorts.get("output_sha256", {}).get("primary_clean261", {}).get("traits") == sha256_file(paths["clean_traits"]),
        "cohort receipt does not bind the named clean261 trait table",
    )
    _require(
        cohorts.get("output_sha256", {}).get("primary_clean261", {}).get("image_traits")
        == sha256_file(paths["clean_image_traits"]),
        "cohort receipt does not bind the named clean261 image-trait table",
    )
    _require(
        len(clean_biology)
        == clean_biology["task_id"].nunique()
        == len(clean_image)
        == clean_image["task_id"].nunique()
        == 261,
        "clean261 biological/image trait tables do not close one-to-one",
    )
    clean = clean_biology.merge(
        clean_image[
            [
                "task_id",
                "source_image_sha256",
                "prediction_sha256",
                "formal_statistics_eligible",
                "um_per_px",
            ]
        ],
        on=("task_id", "source_image_sha256"),
        how="inner",
        validate="one_to_one",
        suffixes=("", "_image"),
    )
    _require(
        len(clean) == 261
        and (
            clean["formal_statistics_eligible"].astype(str).str.casefold()
            == clean["formal_statistics_eligible_image"]
            .astype(str)
            .str.casefold()
        ).all(),
        "clean261 biological/image trait eligibility or identity drift",
    )
    clean = clean.drop(columns=["formal_statistics_eligible_image"])
    formal = clean[clean["formal_statistics_eligible"].astype(str).str.casefold().isin({"true", "1", "yes"})].copy()
    expected_formal_clean = int(
        cohorts.get("primary_clean_formal_statistics_eligible", -1)
    )
    _require(
        expected_formal_clean == 238
        and len(formal)
        == formal["task_id"].nunique()
        == expected_formal_clean,
        "application assurance requires the exact 238 formal source units within clean261",
    )
    application_records = application_fusion.get("records")
    _require(
        isinstance(application_records, list)
        and len(application_records) == 283
        and len({str(record.get("task_id")) for record in application_records}) == 283,
        "application fusion records are not exact283",
    )
    application_by_task = {
        str(record["task_id"]): record for record in application_records
    }
    application_audit_rows: list[dict[str, Any]] = []
    application_prediction_locks: list[dict[str, str]] = []
    for trait_row in formal.itertuples(index=False):
        task_id = str(trait_row.task_id)
        _require(task_id in application_by_task, f"{task_id}: absent from application fusion")
        record = application_by_task[task_id]
        expected_sha = str(record.get("prediction_sha256"))
        _require(
            _is_sha256(expected_sha)
            and expected_sha == str(trait_row.prediction_sha256),
            f"{task_id}: clean trait/application fusion prediction identity mismatch",
        )
        prediction_path = application_root / "predictions" / f"{task_id}.json"
        _require(
            prediction_path.is_file() and sha256_file(prediction_path) == expected_sha,
            f"{task_id}: application prediction file hash mismatch",
        )
        prediction = read_json(prediction_path)
        validate_hybrid_prediction(prediction, artifact_root=application_root)
        _require(
            prediction.get("task_id") == task_id
            and prediction.get("source_image_sha256")
            == str(trait_row.source_image_sha256)
            and prediction.get("formal_phenotype_eligible") is True
            and prediction.get("automatic_measurement_fail_closed") is False,
            f"{task_id}: application prediction is not the named formal source unit",
        )
        root_mask_path = application_root / str(prediction["root_mask_relpath"])
        _require(
            root_mask_path.is_file()
            and sha256_file(root_mask_path) == prediction["root_mask_sha256"],
            f"{task_id}: application root-mask hash mismatch",
        )
        mask_values = cv2.imread(str(root_mask_path), cv2.IMREAD_GRAYSCALE)
        _require(mask_values is not None, f"{task_id}: unreadable application root mask")
        root_mask = np.asarray(mask_values > 0, dtype=bool)
        _require(root_mask.any(), f"{task_id}: empty application root mask")
        axis_path = application_root / str(prediction["root_axis_geometry_relpath"])
        _require(
            axis_path.is_file()
            and sha256_file(axis_path) == prediction["root_axis_geometry_sha256"],
            f"{task_id}: application root-axis hash mismatch",
        )
        with np.load(axis_path, allow_pickle=False) as geometry:
            _require(
                "path_xy" in geometry.files
                and "source_image_sha256" in geometry.files,
                f"{task_id}: application root-axis fields missing",
            )
            axis_xy = np.asarray(geometry["path_xy"], dtype=np.float64)
            geometry_source_sha = str(np.asarray(geometry["source_image_sha256"]).item())
        _require(
            axis_xy.ndim == 2
            and axis_xy.shape[1] == 2
            and len(axis_xy) >= 2
            and np.all(np.isfinite(axis_xy))
            and geometry_source_sha == str(trait_row.source_image_sha256),
            f"{task_id}: invalid/source-mismatched application root axis",
        )
        axis_support = _axis_root_support_metrics(
            root_mask,
            axis_xy,
            um_per_px=float(trait_row.um_per_px),
        )
        unsupported_attachments = 0
        identity_hairs = prediction.get("identity_hairs")
        _require(
            isinstance(identity_hairs, list)
            and len(identity_hairs) == int(trait_row.hair_count),
            f"{task_id}: clean trait hair count differs from prediction",
        )
        for hair in identity_hairs:
            projection = np.asarray(
                hair.get("root_axis_projection_xy"), dtype=np.float64
            )
            valid_projection = bool(
                projection.shape == (2,) and np.all(np.isfinite(projection))
            )
            supported = False
            if valid_projection:
                x, y = np.rint(projection).astype(np.int64)
                supported = bool(
                    0 <= x < root_mask.shape[1]
                    and 0 <= y < root_mask.shape[0]
                    and root_mask[y, x]
                )
            if hair.get("root_attachment_valid") is not True or not supported:
                unsupported_attachments += 1
        application_audit_rows.append(
            {
                "source_unit": task_id,
                # Compatibility field retained for cohort-level values; the
                # Fig. 4 audit labels the same quantity explicitly as
                # axis-in-root coverage.
                "axis_containment_fraction": axis_support[
                    "axis_in_root_coverage_fraction"
                ],
                **axis_support,
                "unsupported_attachment_n": unsupported_attachments,
                "identity_hair_n": len(identity_hairs),
            }
        )
        application_prediction_locks.append(
            {"task_id": task_id, "sha256": expected_sha}
        )
    application_audit = pd.DataFrame(application_audit_rows)
    _require(
        len(application_audit) == expected_formal_clean
        and np.all(
            np.isfinite(application_audit["axis_containment_fraction"])
        ),
        "application topology audit is incomplete",
    )
    _require(
        np.allclose(
            application_audit["axis_containment_fraction"],
            application_audit["axis_in_root_coverage_fraction"],
            rtol=0.0,
            atol=1e-12,
        )
        and (
            application_audit["axis_single_component_coverage_fraction"]
            <= application_audit["axis_in_root_coverage_fraction"] + 1e-12
        ).all()
        and (
            application_audit["longest_unsupported_axis_gap_um"] >= 0.0
        ).all(),
        "application topology union/single-component/gap semantics are inconsistent",
    )
    # Support fractions reported for the four RHD6 x temperature cells belong
    # to the prespecified D15_8d primary experiment.  A technical-
    # generalization image in B7 intentionally reuses the RHD6_EV_30C label;
    # selecting on condition_code alone would silently contaminate the primary
    # denominator with that source unit.
    support_scope = formal[
        (formal["experiment_key"].astype(str) == "D15_8d")
        & (formal["study_role"].astype(str) == "rhd6_factorial_8d_primary")
    ].copy()
    _require(
        len(support_scope) == 47,
        "D15_8d formal support scope is not the locked 47 source units",
    )
    support_rows = []
    for group in GROUP_ORDER:
        selected = support_scope[
            support_scope["condition_code"].astype(str) == group
        ]
        _require(len(selected) > 0, f"{group}: no formal clean261 source units")
        identity = int(pd.to_numeric(selected["hair_count"]).sum())
        supported = int(pd.to_numeric(selected["hair_length_measurement_hair_count"]).sum())
        _require(0 <= supported <= identity and identity > 0, f"{group}: impossible length support")
        support_rows.append(
            {
                "condition_code": group,
                "support_fraction": supported / identity,
                "supported_hairs": supported,
                "identity_hairs": identity,
                "source_units": len(selected),
                "support_semantics": "endpoint-complete matched subset; absent length is not zero",
            }
        )
    support_frame = pd.DataFrame(support_rows)
    support_ids = list(support_scope["task_id"].astype(str))
    support_lookup = support_scope.set_index("task_id")
    def support_stat(sampled: list[str]) -> float:
        supported = sum(float(support_lookup.loc[item, "hair_length_measurement_hair_count"]) for item in sampled)
        identity = sum(float(support_lookup.loc[item, "hair_count"]) for item in sampled)
        return supported / identity if identity > 0 else float("nan")
    total_identity = int(pd.to_numeric(support_scope["hair_count"]).sum())
    total_supported = int(
        pd.to_numeric(
            support_scope["hair_length_measurement_hair_count"]
        ).sum()
    )
    metrics.append(_metric_row(domain="conditional_length", key="endpoint_complete_support_fraction", label_text="Endpoint-complete support", value=total_supported / total_identity, interval=_bootstrap(support_ids, support_stat, seed_offset=50), unit="fraction", n=len(support_ids), instances=total_identity, definition="D15_8d primary formal identities supporting conditional length"))
    topology_ids = list(application_audit["source_unit"].astype(str))
    topology_lookup = application_audit.set_index("source_unit")
    metrics.append(
        _metric_row(
            domain="application_topology",
            key="axis_containment_median",
            label_text="Median formal axis containment",
            value=float(np.median(application_audit["axis_containment_fraction"])),
            interval=_bootstrap(
                topology_ids,
                lambda sampled: float(
                    np.median(
                        [
                            topology_lookup.loc[item, "axis_containment_fraction"]
                            for item in sampled
                        ]
                    )
                ),
                seed_offset=51,
            ),
            unit="fraction",
            n=len(topology_ids),
            definition="nearest-pixel fraction of each sealed ordered axis contained by its sealed final root mask",
        )
    )
    metrics.append(
        _metric_row(
            domain="application_topology",
            key="axis_containment_min",
            label_text="Minimum formal axis containment",
            value=float(np.min(application_audit["axis_containment_fraction"])),
            interval=_bootstrap(
                topology_ids,
                lambda sampled: float(
                    np.min(
                        [
                            topology_lookup.loc[item, "axis_containment_fraction"]
                            for item in sampled
                        ]
                    )
                ),
                seed_offset=52,
            ),
            unit="fraction",
            n=len(topology_ids),
            definition="minimum source-unit ordered-axis containment among formal clean261 outputs",
        )
    )
    total_unsupported_attachments = int(
        application_audit["unsupported_attachment_n"].sum()
    )
    metrics.append(
        _metric_row(
            domain="application_topology",
            key="unsupported_attachment_n",
            label_text="Formal attachments on unsupported axis",
            value=float(total_unsupported_attachments),
            interval=(
                float(total_unsupported_attachments),
                float(total_unsupported_attachments),
            ),
            unit="count",
            n=len(topology_ids),
            instances=int(application_audit["identity_hair_n"].sum()),
            definition="enumerated formal identity attachments whose validity flag or sealed-mask axis projection lacks support",
        )
    )

    layers = exact.get("layers")
    _require(isinstance(layers, Mapping) and all(record.get("exact") == 283 and record.get("mismatch_count") == 0 and record.get("gate_pass") is True for record in layers.values()), "portable provider exact283 layers are incomplete")
    metrics.append(_metric_row(domain="provider_equivalence", key="provider_exact_fraction", label_text="Fresh provider exact equivalence", value=1.0, interval=(1.0, 1.0), unit="fraction", n=283, definition="fresh portable raw-image rerun exact to all frozen root layers"))

    metrics_frame = pd.DataFrame(metrics)
    scale_pairs = detected_scale.assign(
        pair_type="scale",
        pair_id=lambda frame: frame["source_unit"].astype(str) + ":scale",
        trait_key="um_per_px",
        observed=lambda frame: frame["observed_um_per_px"],
        predicted=lambda frame: frame["predicted_um_per_px"],
        unit="um_per_px",
        relative_error_percent=lambda frame: frame["relative_error_percent"],
        scale_line_endpoint_error_um=lambda frame: frame[
            "geometry_endpoint_error_um"
        ],
    )[
        [
            "pair_type", "source_unit", "pair_id", "trait_key", "observed",
            "predicted", "unit", "relative_error_percent",
            "scale_line_endpoint_error_um", "source_image_sha256",
        ]
    ]
    pairs_frame = pd.concat(
        [scale_pairs, length_frame, root_trait_pairs], ignore_index=True, sort=False
    )
    role_map = {
        str(key): (
            "application_observability_non_accuracy"
            if key
            in {
                "endpoint_complete_support_fraction",
                "axis_containment_median",
                "axis_containment_min",
                "unsupported_attachment_n",
            }
            else "exact_portable_provider_equivalence"
            if key == "provider_exact_fraction"
            else "annotated_qc_development_non_independent"
        )
        for key in metrics_frame["metric_key"]
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        metrics_path = staging / "assurance_metrics.csv"
        pairs_path = staging / "assurance_pairs.csv"
        support_path = staging / "assurance_support.csv"
        topology_path = staging / "assurance_topology.csv"
        root_traits_path = staging / "assurance_root_traits.csv"
        root_continuity_input_path = (
            staging / "root_continuity_assurance_input.json"
        )
        root_continuity_path = staging / "root_continuity_assurance.json"
        hair_attachment_input_path = (
            staging / "hair_attachment_assurance_input.json"
        )
        hair_attachment_path = staging / "hair_attachment_assurance.json"
        metrics_frame.to_csv(metrics_path, index=False, encoding="utf-8", lineterminator="\n")
        pairs_frame.to_csv(pairs_path, index=False, encoding="utf-8", lineterminator="\n")
        support_frame.to_csv(support_path, index=False, encoding="utf-8", lineterminator="\n")
        application_audit.to_csv(
            topology_path, index=False, encoding="utf-8", lineterminator="\n"
        )
        pd.DataFrame(root_trait_assurance["trait_rows"]).to_csv(
            root_traits_path,
            index=False,
            encoding="utf-8",
            lineterminator="\n",
        )
        atomic_write_json(root_continuity_input_path, root_continuity_input)
        atomic_write_json(root_continuity_path, root_continuity_assurance)
        atomic_write_json(hair_attachment_input_path, hair_attachment_input)
        atomic_write_json(hair_attachment_path, hair_attachment_assurance)
        component_receipts = {
            "root_continuity": {
                "embedded_payload_field": "root_continuity_assurance",
                "audit_copy": root_continuity_path.name,
                "audit_copy_sha256": sha256_file(root_continuity_path),
                "identity_field": (
                    "root_continuity_assurance_identity_sha256"
                ),
                "identity_sha256": root_continuity_assurance[
                    "root_continuity_assurance_identity_sha256"
                ],
                "input_contract_audit_copy": root_continuity_input_path.name,
                "input_contract_audit_copy_sha256": sha256_file(
                    root_continuity_input_path
                ),
                "input_contract_identity_sha256": root_continuity_input[
                    "input_contract_identity_sha256"
                ],
            },
            "hair_attachment": {
                "embedded_payload_field": "hair_attachment_assurance",
                "audit_copy": hair_attachment_path.name,
                "audit_copy_sha256": sha256_file(hair_attachment_path),
                "identity_field": "hair_attachment_assurance_identity_sha256",
                "identity_sha256": hair_attachment_assurance[
                    "hair_attachment_assurance_identity_sha256"
                ],
                "input_contract_audit_copy": hair_attachment_input_path.name,
                "input_contract_audit_copy_sha256": sha256_file(
                    hair_attachment_input_path
                ),
                "input_contract_identity_sha256": hair_attachment_input[
                    "input_contract_identity_sha256"
                ],
            },
        }
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_locked_qc_development_assurance",
            "scope": "QC-development measurement assurance; non-independent",
            "metric_evidence_role_by_key": role_map,
            "source_table_sha256": {
                "metrics": sha256_file(metrics_path),
                "pairs": sha256_file(pairs_path),
                "support": sha256_file(support_path),
                "topology": sha256_file(topology_path),
                "root_traits": sha256_file(root_traits_path),
            },
            "source_authority_sha256": {
                role: sha256_file(path) for role, path in paths.items()
            },
            "source_authority_identity_sha256": {
                "qcdev_stageb_summary": stageb["summary_identity_sha256"],
                "qcdev_fusion_summary": fusion["summary_identity_sha256"],
                "application_fusion_summary": application_fusion[
                    "summary_identity_sha256"
                ],
                "canonical_ground_truth": canonical_provenance[
                    "canonical_ground_truth_lock_identity_sha256"
                ],
                "stageb_detection_ordered_file_set": stageb_authority[
                    "evaluation_detection_ordered_file_set_identity_sha256"
                ],
                "qcdev_evaluation_stageb_detection_ordered_file_set": (
                    stageb_authority[
                        "evaluation_detection_ordered_file_set_identity_sha256"
                    ]
                ),
                "qcdev_production_stageb_detection_ordered_file_set": (
                    stageb_authority[
                        "production_detection_ordered_file_set_identity_sha256"
                    ]
                ),
                "qcdev_production_stageb_detection_ordered_identity_set": (
                    stageb_authority[
                        "production_detection_ordered_identity_set_sha256"
                    ]
                ),
                "qcdev_fusion_prediction_ordered_file_set": sha256_json(
                    prediction_file_locks
                ),
                "qcdev_task_order_identity": sha256_json(task_ids),
                "qcdev_fused_assurance_input_set_identity": sha256_json(
                    fused_assurance_input_locks
                ),
                "root_continuity_prediction_authority": (
                    root_prediction_authority_identity
                ),
                "hair_attachment_prediction_authority": (
                    hair_prediction_authority_identity
                ),
                "root_continuity_assurance": root_continuity_assurance[
                    "root_continuity_assurance_identity_sha256"
                ],
                "hair_attachment_assurance": hair_attachment_assurance[
                    "hair_attachment_assurance_identity_sha256"
                ],
                "qcdev_stageb_biological_presence_20um_crosscheck": (
                    hair_biological_presence_crosscheck_identity
                ),
                "application_formal_prediction_ordered_file_set": sha256_json(
                    application_prediction_locks
                ),
            },
            "shared_stageb_authority": stageb_authority[
                "shared_model_authority"
            ],
            "qcdev_evaluation_stageb_detection_file_locks": stageb_authority[
                "evaluation_detection_files"
            ],
            "qcdev_production_stageb_detection_file_locks": stageb_authority[
                "production_detection_files"
            ],
            "qcdev_production_stageb_detection_identity_locks": (
                stageb_authority["production_detection_identities"]
            ),
            "prediction_file_locks": prediction_file_locks,
            "qcdev_fused_assurance_input_locks": fused_assurance_input_locks,
            "qcdev_stageb_biological_presence_20um_crosscheck_locks": (
                hair_biological_presence_crosscheck_locks
            ),
            "application_prediction_file_locks": application_prediction_locks,
            "component_receipts": component_receipts,
            "bootstrap": {
                "method": "image/source-unit nonparametric bootstrap",
                "repetitions": BOOTSTRAP_REPETITIONS,
                "seed": BOOTSTRAP_SEED,
            },
            "measurement_contract": {
                "root_boundary_tolerance_um": BOUNDARY_TOLERANCE_UM,
                "distal_pck_threshold_um": PCK_THRESHOLD_UM,
                "conditional_length_base_match_tolerance_um": 20.0,
                "root_trait_count": len(ROOT_TRAIT_FIELDS),
                "root_trait_reference": "canonical root mask plus annotated distal point under one deterministic PHAxis measurement geometry",
                "root_trait_truth_authority": "canonical vector-derived root mask plus annotated distal/root-cap point; never provider equivalence",
                "root_trait_prediction_authority": "sealed QC-development Hybrid-Max detailed_root_statistics",
                "root_trait_accuracy_evidence_role": "annotated_qc_development_non_independent",
                "root_trait_provider_equivalence_used_as_accuracy": False,
                "matched_trajectory_tolerance_um": 20.0,
                "matched_trajectory_resample_step_um": 2.0,
                "axis_containment_sampling": "nearest integer pixel on sealed ordered axis versus sealed final root mask",
                "application_axis_single_component_policy": "8-connected root-mask component supporting the largest number of sealed ordered-axis samples; ties use the smaller component label",
                "application_longest_unsupported_axis_gap_definition": "longest contiguous ordered-axis arc-length run whose segment endpoints are not both supported by the winning root-mask component",
                "fig4_case_audit_schema": "PHAxis-Fig4-case-audit-2.0",
                "fig4_profile_0_5mm_eligibility": "formal_statistics_eligible and visible_root_axis_length_um at least 5000",
                "unsupported_attachment_definition": "invalid attachment flag or unsupported nearest-pixel sealed axis projection",
                "root_cap_region_output": False,
                "qcdev_evaluation_and_production_stageb_file_sha256_sets_are_distinct_authorities": True,
                "root_continuity_reference_source": "canonical vector-derived root mask plus annotated distal/root-cap point",
                "root_continuity_prediction_source": "sealed final fused prediction root_mask_sha256",
                "root_continuity_connected_component_policy": "all 8-connected skeleton components encoded without evaluator-side interpolation, bridging, or gap completion",
                "hair_attachment_prediction_source": "validated production Stage-B detection geometry cross-checked against final fused identity_hairs/count_hairs",
                "hair_attachment_formal_identity_matcher": dict(
                    FORMAL_MATCHER_CONFIG
                ),
                "hair_attachment_formal_tolerance_um": (
                    FORMAL_ATTACHMENT_TOLERANCE_UM
                ),
                "hair_attachment_base_proxy_tolerances_um": list(
                    PROXY_TOLERANCES_UM
                ),
                "hair_attachment_base_proxy_role": "development-only threshold sensitivity; never formal matched attachment accuracy",
                "hair_attachment_threshold_selection_used_as_formal_accuracy": False,
                "hair_attachment_evaluator_crosscheck": "per-source and pooled n_pred, n_gt, and tolerant biological-presence TP@20um recomputed from production Stage-B geometry must exactly equal the sealed evaluation-only Stage-B results",
                "scale_coverage_denominator": "visible_annotated_scale_bar_cases",
                "scale_localization_denominator": "detected_visible_scale_bars",
                "scale_calibration_denominator": "detected_visible_scale_bars",
                "scale_absence_specificity_status": SCALE_ABSENCE_SPECIFICITY_STATUS,
                "scale_fail_closed_evidence_basis": SCALE_FAIL_CLOSED_EVIDENCE_BASIS,
            },
            "scale_applicability": scale_truth_summary,
            "counts": {
                "qcdevelopment_images": len(root_frame),
                "visible_scale_bars": len(scale_frame),
                "trusted_metadata_without_visible_bar_cases": scale_truth_summary[
                    "trusted_metadata_without_visible_bar_cases"
                ],
                "absent_or_untrusted_scale_truth_cases": scale_truth_summary[
                    "absent_or_untrusted_scale_truth_cases"
                ],
                "detected_scale_bars": int(scale_frame["detected"].sum()),
                "scale_localization_pairs": len(geometry_scale),
                "scale_calibration_pairs": len(detected_scale),
                "conditional_length_pairs": len(length_frame),
                "conditional_length_source_units": len(length_ids),
                "root_trait_pairs": len(root_trait_pairs),
                "root_trait_summary_rows": len(root_trait_assurance["trait_rows"]),
                "root_trait_families": len(root_trait_assurance["family_rows"]),
                "root_trait_eligible_source_units_min": min(
                    int(row["eligible_source_units"])
                    for row in root_trait_assurance["trait_rows"]
                ),
                "root_trait_eligible_source_units_max": max(
                    int(row["eligible_source_units"])
                    for row in root_trait_assurance["trait_rows"]
                ),
                "root_continuity_source_units": int(
                    root_continuity_assurance["source_unit_total"]
                ),
                "root_continuity_break_free_images": int(
                    root_continuity_summary["break_free_images"]
                ),
                "root_continuity_union_coverage_hides_fragmentation_images": int(
                    root_continuity_summary[
                        "union_coverage_hides_fragmentation_images"
                    ]
                ),
                "hair_attachment_source_units": int(
                    hair_attachment_assurance["source_unit_total"]
                ),
                "hair_attachment_predicted_hairs": int(
                    attachment_identity["n_pred"]
                ),
                "hair_attachment_annotated_hairs": int(
                    attachment_identity["n_gt"]
                ),
                "hair_attachment_formal_identity_matches": int(
                    formal_presence["tp"]
                ),
                "hair_attachment_qualified_true_positives_20um": int(
                    attachment_identity["tp"]
                ),
                "hair_attachment_evaluator_crosschecked_source_units": len(
                    hair_biological_presence_crosscheck_locks
                ),
                "application_support_source_units": len(support_ids),
                "application_topology_source_units": len(application_audit),
                "application_formal_identity_hairs": int(
                    application_audit["identity_hair_n"].sum()
                ),
                "unsupported_application_attachments": total_unsupported_attachments,
                "provider_exact_images": 283,
            },
            "independent_accuracy_claim_allowed": False,
            "root_trait_assurance": root_trait_assurance,
            "root_continuity_assurance": root_continuity_assurance,
            "hair_attachment_assurance": hair_attachment_assurance,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        }
        receipt["measurement_assurance_identity_sha256"] = sha256_json(receipt)
        atomic_write_json(staging / "measurement_assurance_receipt.json", receipt)
        _require(not destination.exists(), "measurement-assurance output appeared during build")
        os.replace(staging, destination)
        return receipt
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train399-evaluation", required=True)
    parser.add_argument("--qcdev-stageb-summary", required=True)
    parser.add_argument("--qcdev-fusion-summary", required=True)
    parser.add_argument("--qcdev-fusion-root", required=True)
    parser.add_argument("--application-fusion-summary", required=True)
    parser.add_argument("--application-fusion-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--clean-traits", required=True)
    parser.add_argument("--clean-image-traits", required=True)
    parser.add_argument("--cohorts-receipt", required=True)
    parser.add_argument("--root-exact283-receipt", required=True)
    parser.add_argument(
        "--trait-contract",
        default=str(PROJECT_ROOT / "configs/phaxis/v1_0/trait_contract.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_measurement_assurance(
        output=args.output,
        train399_evaluation=args.train399_evaluation,
        qcdev_stageb_summary=args.qcdev_stageb_summary,
        qcdev_fusion_summary=args.qcdev_fusion_summary,
        qcdev_fusion_root=args.qcdev_fusion_root,
        application_fusion_summary=args.application_fusion_summary,
        application_fusion_root=args.application_fusion_root,
        dataset_root=args.dataset_root,
        dataset_manifest=args.dataset_manifest,
        split_manifest=args.split_manifest,
        clean_traits=args.clean_traits,
        clean_image_traits=args.clean_image_traits,
        cohorts_receipt=args.cohorts_receipt,
        root_exact283_receipt=args.root_exact283_receipt,
        trait_contract=args.trait_contract,
    )
    print(receipt["measurement_assurance_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
