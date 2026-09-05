"""Sealed, non-deployable Stage-B inference artifacts for QC-development44.

The selected train399 operating point must be evaluated before a PHAxis model
contract can be proposed.  Production Stage-B inference intentionally requires
that proposal, so it cannot be used to create the evidence that the proposal
itself consumes.  This module defines the narrow exception: an exact-QCdev44,
evaluation-only artifact that is bound to the candidate/selection receipts and
contains complete geometry, but has a different schema and is explicitly
barred from fusion, traits, and deployment.

Nothing in this module imports torch or CUDA.  GPU preflight and model loading
belong to the dedicated evaluation-inference entry point and happen only after
these receipt checks have passed.
"""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import ContractError, validate_stageb_detection_payload
from ..io import read_json, sha256_file, sha256_json
from .candidate_bundle import read_candidate_manifest
from .selection import (
    read_selection_receipt,
    validate_selected_operating_point_binding,
)
from .serialization import make_detection_payload


EVALUATION_GATE_SCHEMA = (
    "PHAxis-StageB-train399-QCdev44-evaluation-inference-gate-1.0"
)
EVALUATION_DETECTION_SCHEMA = (
    "PHAxis-StageB-train399-QCdev44-evaluation-only-full-geometry-1.0"
)
EVALUATION_RUN_SCHEMA = (
    "PHAxis-StageB-train399-QCdev44-evaluation-inference-run-1.0"
)
EVALUATION_ARTIFACT_ROLE = (
    "locked_qcdevelopment44_full_geometry_evaluation_only_not_deployable"
)
EVALUATION_IMAGE_COUNT = 44

_PRODUCTION_AUTHORITY_FIELDS = frozenset(
    {
        "model_contract_proposal_sha256",
        "model_contract_proposal_identity_sha256",
        "model_bundle_id",
        "root_expert_id",
    }
)


class EvaluationInferenceError(RuntimeError):
    """The selected-candidate evaluation-only inference contract was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationInferenceError(message)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.casefold()


def _unsigned_identity(payload: Mapping[str, Any], field: str) -> str:
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field, None)
    _require(_is_sha256(observed), f"{field} is absent or invalid")
    _require(sha256_json(unsigned) == observed, f"{field} mismatch")
    return str(observed)


def _candidate_checkpoint_locks(
    candidate_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    members = candidate_manifest.get("identity_payload", {}).get("members")
    _require(isinstance(members, list) and len(members) == 5, "candidate must have five members")
    locks: list[dict[str, Any]] = []
    for index, member in enumerate(members):
        _require(isinstance(member, Mapping), f"candidate member {index} is invalid")
        member_id = member.get("member_id")
        seed = member.get("seed")
        digest = member.get("checkpoint_sha256")
        _require(isinstance(member_id, str) and member_id, f"candidate member {index} has no ID")
        _require(isinstance(seed, int) and not isinstance(seed, bool), f"candidate member {index} has no seed")
        _require(_is_sha256(digest), f"candidate member {index} checkpoint hash is invalid")
        locks.append(
            {
                "member_id": member_id,
                "seed": seed,
                "checkpoint_sha256": digest,
            }
        )
    _require(
        len({row["checkpoint_sha256"] for row in locks}) == 5,
        "candidate checkpoint hashes are not unique",
    )
    return locks


def build_evaluation_gate_binding(
    *,
    candidate_manifest_path: str | Path,
    selected_model_metadata_path: str | Path,
    selection_receipt_path: str | Path,
    checkpoint_paths: Sequence[str | Path] | None = None,
    checkpoint_locks: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and cross-bind the selected candidate without requiring a proposal.

    Runtime callers supply ``checkpoint_paths`` so the actual five files are
    hashed before GPU preflight.  Downstream validators may instead supply the
    already sealed ``checkpoint_locks`` from an evaluation-run summary; those
    locks must exactly equal the candidate member order and hashes.
    """

    _require(
        (checkpoint_paths is None) != (checkpoint_locks is None),
        "provide exactly one of checkpoint_paths or checkpoint_locks",
    )
    candidate_path = Path(candidate_manifest_path)
    selected_path = Path(selected_model_metadata_path)
    receipt_path = Path(selection_receipt_path)
    candidate_manifest = read_candidate_manifest(candidate_path)
    selected_model_metadata = read_json(selected_path)
    selection_receipt = read_selection_receipt(receipt_path)
    validate_selected_operating_point_binding(
        candidate_manifest=candidate_manifest,
        selected_model_metadata=selected_model_metadata,
        selection_receipt=selection_receipt,
        selection_receipt_file_sha256=sha256_file(receipt_path),
    )

    expected_checkpoint_locks = _candidate_checkpoint_locks(candidate_manifest)
    if checkpoint_paths is not None:
        _require(len(checkpoint_paths) == 5, "evaluation inference requires five checkpoints")
        observed_checkpoint_locks = []
        for expected, raw_path in zip(
            expected_checkpoint_locks, checkpoint_paths, strict=True
        ):
            path = Path(raw_path)
            _require(path.is_file(), f"evaluation checkpoint is missing: {path}")
            observed_checkpoint_locks.append(
                {
                    "member_id": expected["member_id"],
                    "seed": expected["seed"],
                    "checkpoint_sha256": sha256_file(path),
                }
            )
    else:
        observed_checkpoint_locks = [deepcopy(dict(row)) for row in checkpoint_locks or ()]
    _require(
        observed_checkpoint_locks == expected_checkpoint_locks,
        "evaluation checkpoint files/order differ from the selected candidate",
    )
    _require(
        selected_model_metadata.get("checkpoint_sha256")
        == [row["checkpoint_sha256"] for row in expected_checkpoint_locks],
        "selected metadata checkpoint order differs from the candidate",
    )

    task_locks = selection_receipt.get("task_image_locks")
    _require(
        isinstance(task_locks, list) and len(task_locks) == EVALUATION_IMAGE_COUNT,
        "selection receipt is not exact QCdevelopment44",
    )
    task_ids = [row.get("task_id") if isinstance(row, Mapping) else None for row in task_locks]
    _require(
        all(isinstance(task_id, str) and task_id for task_id in task_ids)
        and len(set(task_ids)) == EVALUATION_IMAGE_COUNT,
        "selection receipt QCdevelopment44 task IDs are invalid",
    )
    _require(
        selection_receipt.get("task_image_lock_identity_sha256")
        == sha256_json(task_locks),
        "selection receipt task-image lock identity mismatch",
    )

    gate: dict[str, Any] = {
        "schema_version": EVALUATION_GATE_SCHEMA,
        "artifact_role": EVALUATION_ARTIFACT_ROLE,
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "candidate_bundle_identity_sha256": candidate_manifest[
            "candidate_bundle_identity_sha256"
        ],
        "selected_model_metadata_sha256": sha256_file(selected_path),
        "selected_model_metadata_identity_sha256": selected_model_metadata[
            "selected_model_metadata_identity_sha256"
        ],
        "selection_receipt_sha256": sha256_file(receipt_path),
        "selection_receipt_identity_sha256": selection_receipt[
            "selection_receipt_identity_sha256"
        ],
        "candidate_pool_identity_sha256": selection_receipt[
            "candidate_pool_identity_sha256"
        ],
        "checkpoint_locks": expected_checkpoint_locks,
        "checkpoint_set_identity_sha256": sha256_json(expected_checkpoint_locks),
        "qcdevelopment44_task_ids": task_ids,
        "qcdevelopment44_task_order_identity_sha256": sha256_json(task_ids),
        "task_image_lock_identity_sha256": selection_receipt[
            "task_image_lock_identity_sha256"
        ],
        "images": EVALUATION_IMAGE_COUNT,
        "model_contract_proposal_required_for_artifact": False,
        "model_contract_proposal_present": False,
        "production_consumption_allowed": False,
        "fusion_consumption_allowed": False,
        "traits_consumption_allowed": False,
        "canonical_annotations_read_during_inference": False,
        "condition_metadata_used_for_routing": False,
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
    }
    gate["evaluation_gate_identity_sha256"] = sha256_json(gate)
    validate_evaluation_gate_binding(gate)
    return candidate_manifest, selected_model_metadata, selection_receipt, gate


def validate_evaluation_gate_binding(
    gate: Mapping[str, Any], *, expected_gate: Mapping[str, Any] | None = None
) -> None:
    _require(gate.get("schema_version") == EVALUATION_GATE_SCHEMA, "wrong evaluation gate schema")
    _require(gate.get("artifact_role") == EVALUATION_ARTIFACT_ROLE, "wrong evaluation artifact role")
    _require(gate.get("images") == EVALUATION_IMAGE_COUNT, "evaluation gate is not exact44")
    for field in (
        "candidate_manifest_sha256",
        "candidate_bundle_identity_sha256",
        "selected_model_metadata_sha256",
        "selected_model_metadata_identity_sha256",
        "selection_receipt_sha256",
        "selection_receipt_identity_sha256",
        "candidate_pool_identity_sha256",
        "checkpoint_set_identity_sha256",
        "qcdevelopment44_task_order_identity_sha256",
        "task_image_lock_identity_sha256",
    ):
        _require(_is_sha256(gate.get(field)), f"evaluation gate {field} is invalid")
    checkpoint_locks = gate.get("checkpoint_locks")
    _require(isinstance(checkpoint_locks, list) and len(checkpoint_locks) == 5, "evaluation gate checkpoint locks are not exact5")
    _require(
        all(
            isinstance(row, Mapping)
            and set(row) == {"member_id", "seed", "checkpoint_sha256"}
            and isinstance(row.get("member_id"), str)
            and bool(row.get("member_id"))
            and isinstance(row.get("seed"), int)
            and not isinstance(row.get("seed"), bool)
            and _is_sha256(row.get("checkpoint_sha256"))
            for row in checkpoint_locks
        )
        and len({row["member_id"] for row in checkpoint_locks}) == 5
        and len({row["seed"] for row in checkpoint_locks}) == 5
        and len({row["checkpoint_sha256"] for row in checkpoint_locks}) == 5,
        "evaluation gate checkpoint member locks are invalid",
    )
    _require(
        gate.get("checkpoint_set_identity_sha256") == sha256_json(checkpoint_locks),
        "evaluation gate checkpoint-set identity mismatch",
    )
    task_ids = gate.get("qcdevelopment44_task_ids")
    _require(
        isinstance(task_ids, list)
        and len(task_ids) == EVALUATION_IMAGE_COUNT
        and len(set(task_ids)) == EVALUATION_IMAGE_COUNT
        and all(isinstance(value, str) and value for value in task_ids),
        "evaluation gate task IDs are not exact44",
    )
    _require(
        gate.get("qcdevelopment44_task_order_identity_sha256") == sha256_json(task_ids),
        "evaluation gate task-order identity mismatch",
    )
    _require(
        gate.get("model_contract_proposal_required_for_artifact") is False
        and gate.get("model_contract_proposal_present") is False,
        "evaluation gate is circularly bound to a model-contract proposal",
    )
    _require(
        gate.get("production_consumption_allowed") is False
        and gate.get("fusion_consumption_allowed") is False
        and gate.get("traits_consumption_allowed") is False,
        "evaluation gate is not barred from production consumers",
    )
    _require(
        gate.get("canonical_annotations_read_during_inference") is False
        and gate.get("condition_metadata_used_for_routing") is False
        and gate.get("independent_accuracy_claim_allowed") is False
        and gate.get("blind_images_used") == 0,
        "evaluation gate violates information-boundary locks",
    )
    _require(
        not (_PRODUCTION_AUTHORITY_FIELDS & set(gate)),
        "evaluation gate contains production authority fields",
    )
    _unsigned_identity(gate, "evaluation_gate_identity_sha256")
    if expected_gate is not None:
        _require(dict(gate) == dict(expected_gate), "evaluation gate differs from source receipts")


def make_evaluation_detection_payload(
    *,
    task_id: str,
    source_image_sha256: str,
    source_um_per_px: float,
    prediction: dict[str, Any],
    selected_model_metadata: Mapping[str, Any],
    evaluation_gate: Mapping[str, Any],
    precision_mode: str = "fp32_locked",
) -> dict[str, Any]:
    """Wrap a complete selected-model detection in a non-production schema."""

    validate_evaluation_gate_binding(evaluation_gate)
    _require(
        task_id in evaluation_gate["qcdevelopment44_task_ids"],
        f"{task_id}: task is outside locked QCdevelopment44",
    )
    core = make_detection_payload(
        task_id=task_id,
        source_image_sha256=source_image_sha256,
        source_um_per_px=source_um_per_px,
        prediction=prediction,
        precision_mode=precision_mode,
        model_metadata=selected_model_metadata,
        score_threshold=float(selected_model_metadata["selected_score_threshold"]),
    )
    try:
        validate_stageb_detection_payload(
            core,
            expected_task_id=task_id,
            expected_image_sha256=source_image_sha256,
            expected_model_metadata=selected_model_metadata,
        )
    except ContractError as error:
        raise EvaluationInferenceError(str(error)) from error
    _require(
        not (_PRODUCTION_AUTHORITY_FIELDS & set(core)),
        "evaluation-only core unexpectedly contains production authority",
    )
    payload: dict[str, Any] = {
        "schema_version": EVALUATION_DETECTION_SCHEMA,
        "status": "completed",
        "artifact_role": EVALUATION_ARTIFACT_ROLE,
        "task_id": task_id,
        "source_image_sha256": source_image_sha256,
        "evaluation_gate_identity_sha256": evaluation_gate[
            "evaluation_gate_identity_sha256"
        ],
        "candidate_bundle_identity_sha256": evaluation_gate[
            "candidate_bundle_identity_sha256"
        ],
        "selected_model_metadata_identity_sha256": evaluation_gate[
            "selected_model_metadata_identity_sha256"
        ],
        "selection_receipt_identity_sha256": evaluation_gate[
            "selection_receipt_identity_sha256"
        ],
        "model_contract_proposal_required_for_artifact": False,
        "model_contract_proposal_present": False,
        "production_consumption_allowed": False,
        "fusion_consumption_allowed": False,
        "traits_consumption_allowed": False,
        "canonical_annotations_read_during_inference": False,
        "condition_metadata_used_for_routing": False,
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
        "stageb_detection_payload": core,
    }
    payload["evaluation_detection_identity_sha256"] = sha256_json(payload)
    validate_evaluation_detection_payload(
        payload,
        expected_task_id=task_id,
        expected_image_sha256=source_image_sha256,
        expected_model_metadata=selected_model_metadata,
        expected_evaluation_gate=evaluation_gate,
    )
    return payload


def validate_evaluation_detection_payload(
    payload: Mapping[str, Any],
    *,
    expected_task_id: str,
    expected_image_sha256: str,
    expected_model_metadata: Mapping[str, Any],
    expected_evaluation_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an eval-only wrapper and return its complete geometry payload."""

    validate_evaluation_gate_binding(expected_evaluation_gate)
    _require(payload.get("schema_version") == EVALUATION_DETECTION_SCHEMA, "wrong evaluation detection schema")
    _require(payload.get("status") == "completed", "evaluation detection is incomplete")
    _require(payload.get("artifact_role") == EVALUATION_ARTIFACT_ROLE, "wrong evaluation detection role")
    _require(payload.get("task_id") == expected_task_id, "evaluation detection task mismatch")
    _require(
        str(payload.get("source_image_sha256", "")).casefold()
        == expected_image_sha256.casefold(),
        "evaluation detection image hash mismatch",
    )
    for field in (
        "evaluation_gate_identity_sha256",
        "candidate_bundle_identity_sha256",
        "selected_model_metadata_identity_sha256",
        "selection_receipt_identity_sha256",
    ):
        _require(
            payload.get(field) == expected_evaluation_gate.get(field),
            f"evaluation detection {field} mismatch",
        )
    _require(
        payload.get("model_contract_proposal_required_for_artifact") is False
        and payload.get("model_contract_proposal_present") is False,
        "evaluation detection is circularly proposal-bound",
    )
    _require(
        payload.get("production_consumption_allowed") is False
        and payload.get("fusion_consumption_allowed") is False
        and payload.get("traits_consumption_allowed") is False,
        "evaluation detection is not barred from production consumers",
    )
    _require(
        payload.get("canonical_annotations_read_during_inference") is False
        and payload.get("condition_metadata_used_for_routing") is False
        and payload.get("independent_accuracy_claim_allowed") is False
        and payload.get("blind_images_used") == 0,
        "evaluation detection violates information-boundary locks",
    )
    _require(
        not (_PRODUCTION_AUTHORITY_FIELDS & set(payload)),
        "evaluation detection contains production authority fields",
    )
    _unsigned_identity(payload, "evaluation_detection_identity_sha256")
    core = payload.get("stageb_detection_payload")
    _require(isinstance(core, Mapping), "evaluation detection has no full-geometry payload")
    _require(
        not (_PRODUCTION_AUTHORITY_FIELDS & set(core)),
        "evaluation detection core contains production authority fields",
    )
    try:
        validate_stageb_detection_payload(
            core,
            expected_task_id=expected_task_id,
            expected_image_sha256=expected_image_sha256,
            expected_model_metadata=expected_model_metadata,
        )
    except ContractError as error:
        raise EvaluationInferenceError(str(error)) from error
    detections = core.get("detections")
    _require(
        isinstance(detections, list)
        and all(
            isinstance(row, Mapping)
            and "tip_xy_working" in row
            and "predicted_length_um" in row
            and "tip_snapped" in row
            and row.get("predicted_length_semantics")
            == "regressed_polyline_arc_length_um_diagnostic_only"
            for row in detections
        ),
        "evaluation detection lacks complete tip/length geometry",
    )
    return deepcopy(dict(core))


def make_evaluation_inference_summary(
    *,
    evaluation_gate: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    runtime_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_evaluation_gate_binding(evaluation_gate)
    copied_records = [deepcopy(dict(row)) for row in records]
    task_ids = [row.get("task_id") for row in copied_records]
    _require(
        task_ids == evaluation_gate["qcdevelopment44_task_ids"],
        "evaluation inference record order differs from locked QCdevelopment44",
    )
    file_locks: list[dict[str, Any]] = []
    for index, record in enumerate(copied_records):
        for field in (
            "source_image_sha256",
            "evaluation_detection_file_sha256",
            "evaluation_detection_identity_sha256",
        ):
            _require(_is_sha256(record.get(field)), f"evaluation record {index} {field} is invalid")
        _require(
            isinstance(record.get("detections"), int)
            and not isinstance(record.get("detections"), bool)
            and record["detections"] >= 0,
            f"evaluation record {index} detection count is invalid",
        )
        seconds = record.get("wall_seconds_including_io")
        _require(
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and math.isfinite(float(seconds))
            and float(seconds) >= 0.0,
            f"evaluation record {index} wall time is invalid",
        )
        _require(isinstance(record.get("resumed"), bool), f"evaluation record {index} resume flag is invalid")
        file_locks.append(
            {
                "task_id": record["task_id"],
                "sha256": record["evaluation_detection_file_sha256"],
            }
        )
    summary: dict[str, Any] = {
        "schema_version": EVALUATION_RUN_SCHEMA,
        "status": "completed",
        "artifact_role": EVALUATION_ARTIFACT_ROLE,
        "images": EVALUATION_IMAGE_COUNT,
        "evaluation_gate_binding": deepcopy(dict(evaluation_gate)),
        "evaluation_gate_identity_sha256": evaluation_gate[
            "evaluation_gate_identity_sha256"
        ],
        "evaluation_detection_schema_version": EVALUATION_DETECTION_SCHEMA,
        "evaluation_detection_files": file_locks,
        "evaluation_detection_set_identity_sha256": sha256_json(file_locks),
        "records": copied_records,
        "model_contract_proposal_required_for_artifact": False,
        "model_contract_proposal_present": False,
        "production_consumption_allowed": False,
        "fusion_consumption_allowed": False,
        "traits_consumption_allowed": False,
        "canonical_annotations_read_during_inference": False,
        "condition_metadata_used_for_routing": False,
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
    }
    if runtime_metadata is not None:
        protected = set(summary) & set(runtime_metadata)
        _require(not protected, "runtime metadata cannot override evaluation summary locks")
        summary.update(deepcopy(dict(runtime_metadata)))
    summary["evaluation_inference_summary_identity_sha256"] = sha256_json(summary)
    validate_evaluation_inference_summary(summary, expected_evaluation_gate=evaluation_gate)
    return summary


def validate_evaluation_inference_summary(
    summary: Mapping[str, Any], *, expected_evaluation_gate: Mapping[str, Any]
) -> None:
    validate_evaluation_gate_binding(expected_evaluation_gate)
    _require(summary.get("schema_version") == EVALUATION_RUN_SCHEMA, "wrong evaluation inference summary schema")
    _require(summary.get("status") == "completed", "evaluation inference summary is incomplete")
    _require(summary.get("artifact_role") == EVALUATION_ARTIFACT_ROLE, "wrong evaluation inference role")
    _require(summary.get("images") == EVALUATION_IMAGE_COUNT, "evaluation inference summary is not exact44")
    gate = summary.get("evaluation_gate_binding")
    _require(isinstance(gate, Mapping), "evaluation inference summary has no gate binding")
    validate_evaluation_gate_binding(gate, expected_gate=expected_evaluation_gate)
    _require(
        summary.get("evaluation_gate_identity_sha256")
        == expected_evaluation_gate["evaluation_gate_identity_sha256"],
        "evaluation inference summary gate identity mismatch",
    )
    _require(
        summary.get("evaluation_detection_schema_version")
        == EVALUATION_DETECTION_SCHEMA,
        "evaluation inference summary detection schema mismatch",
    )
    records = summary.get("records")
    file_locks = summary.get("evaluation_detection_files")
    _require(
        isinstance(records, list) and len(records) == EVALUATION_IMAGE_COUNT,
        "evaluation inference records are not exact44",
    )
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"evaluation record {index} is invalid")
        for field in (
            "source_image_sha256",
            "evaluation_detection_file_sha256",
            "evaluation_detection_identity_sha256",
        ):
            _require(
                _is_sha256(record.get(field)),
                f"evaluation record {index} {field} is invalid",
            )
        _require(
            isinstance(record.get("detections"), int)
            and not isinstance(record.get("detections"), bool)
            and record["detections"] >= 0,
            f"evaluation record {index} detection count is invalid",
        )
        seconds = record.get("wall_seconds_including_io")
        _require(
            isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and math.isfinite(float(seconds))
            and float(seconds) >= 0.0,
            f"evaluation record {index} wall time is invalid",
        )
        _require(
            isinstance(record.get("resumed"), bool),
            f"evaluation record {index} resume flag is invalid",
        )
    expected_file_locks = [
        {"task_id": row.get("task_id"), "sha256": row.get("evaluation_detection_file_sha256")}
        for row in records
    ]
    _require(file_locks == expected_file_locks, "evaluation inference file locks differ from records")
    _require(
        [row["task_id"] for row in expected_file_locks]
        == expected_evaluation_gate["qcdevelopment44_task_ids"],
        "evaluation inference task order differs from QCdevelopment44",
    )
    _require(
        summary.get("evaluation_detection_set_identity_sha256")
        == sha256_json(expected_file_locks),
        "evaluation inference detection-set identity mismatch",
    )
    _require(
        summary.get("model_contract_proposal_required_for_artifact") is False
        and summary.get("model_contract_proposal_present") is False,
        "evaluation inference summary is circularly proposal-bound",
    )
    _require(
        summary.get("production_consumption_allowed") is False
        and summary.get("fusion_consumption_allowed") is False
        and summary.get("traits_consumption_allowed") is False,
        "evaluation inference summary is not barred from production consumers",
    )
    _require(
        summary.get("canonical_annotations_read_during_inference") is False
        and summary.get("condition_metadata_used_for_routing") is False
        and summary.get("independent_accuracy_claim_allowed") is False
        and summary.get("blind_images_used") == 0,
        "evaluation inference summary violates information-boundary locks",
    )
    _require(
        not (_PRODUCTION_AUTHORITY_FIELDS & set(summary)),
        "evaluation inference summary contains production authority fields",
    )
    _unsigned_identity(summary, "evaluation_inference_summary_identity_sha256")
