"""Fail-closed promotion gate for strict train399-only Stage-B ensembles.

This module deliberately separates *training-contract acceptance* from product
promotion.  A passing manifest is an immutable candidate receipt; it never
changes :mod:`phaxis.constants` or the public model contract.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..evaluation_metrics import biological_hair_presence_matcher_contract
from ..io import atomic_write_json, read_json, sha256_file, sha256_json
from .candidate_pool_contract import (
    locked_biological_presence_candidate_decoder_contract,
)


CHECKPOINT_SCHEMA = "PHAxis-StageB-train399-checkpoint-1.0"
CANDIDATE_MANIFEST_SCHEMA = "PHAxis-StageB-train399-candidate-bundle-1.0"
TRAINING_RECEIPT_SCHEMA = "PHAxis-StageB-train399-training-receipt-1.0"
AMP_AMENDMENT_SCHEMA = "PHAxis-StageB-train399-AMP-backward-amendment-1.0"
TRAINING_FAILURE_SCHEMA = "PHAxis-StageB-training-failure-1.0"
TRAIN399_CHECKPOINT_POLICY = "five_seed_train399_last_epoch_60"
CANDIDATE_STATUS = "candidate_gate_passed_not_promoted"
FORMAL_TRAIN399_SEEDS = (
    2026082801,
    2026082802,
    2026082803,
    2026082804,
    2026082805,
)
EXPECTED_TRAIN_IMAGES = 399
EXPECTED_VALIDATION_IMAGES = 44
EXPECTED_EPOCH = 60
IMAGENET_SOURCE = "timm/resnet34.a1_in1k"
CANDIDATE_POOL_SCORE_FLOOR = 0.10
PREREGISTERED_SCORE_THRESHOLDS = (
    0.100,
    0.125,
    0.150,
    0.175,
    0.200,
    0.225,
    0.250,
    0.275,
    0.300,
    0.325,
)


def amp_backward_retry_policy_lock() -> dict[str, Any]:
    """Return the immutable numerical interpretation added after seed-3 overflow."""

    return {
        "contract_policy_string": "fail_closed_no_optimizer_step_skip",
        "interpretation": (
            "A scaled-backward overflow may be retried on the identical retained "
            "forward graph after GradScaler backoff; the batch and optimizer update "
            "are never skipped, and exhaustion remains a hard failure."
        ),
        "initial_scale": 1024.0,
        "backoff_factor": 0.5,
        "maximum_backward_retries_per_batch": 16,
        "same_forward_graph_replayed": True,
        "forward_recomputed": False,
        "batchnorm_buffers_updated_again": False,
        "rng_or_data_order_advanced": False,
        "optimizer_step_before_finite_unscaled_gradient": False,
        "optimizer_steps_skipped_due_nonfinite_gradients": 0,
        "failure_after_retry_exhaustion": True,
    }


def operating_point_selection_contract() -> dict[str, Any]:
    """Return the preregistered, information-limited QC-development protocol."""

    matcher = biological_hair_presence_matcher_contract()
    candidate_decoder = locked_biological_presence_candidate_decoder_contract()
    return {
        "scope": "locked_overlay_visible_QCdevelopment44_model_selection_only",
        "independent_accuracy_claim_allowed": False,
        "network_forward_passes_per_image": 1,
        "candidate_pool_score_field": "base_score",
        "candidate_pool_score_floor": CANDIDATE_POOL_SCORE_FLOOR,
        "threshold_grid": list(PREREGISTERED_SCORE_THRESHOLDS),
        "threshold_operation": (
            "filter_the_same_base_tip_presence_proxy_pool_by_base_score_only"
        ),
        "candidate_decoder_contract": candidate_decoder,
        "candidate_decoder_contract_sha256": sha256_json(candidate_decoder),
        "primary_selection_metric": (
            "one_to_one_tolerant_biological_presence_F1_at_20um"
        ),
        "primary_matcher_contract": matcher,
        "primary_matcher_contract_sha256": sha256_json(matcher),
        "metric_coordinate_space": (
            "physical_um_xy_after_axis_specific_realized_resize_conversion"
        ),
        "ground_truth_authority": (
            "HumanCurated443_canonical_vectors_with_per_file_annotation_SHA_"
            "source_image_SHA_shape_explicit_um_per_px_physical_geometry_SHA_and_"
            "training_identical_root_polygon_endpoint_orientation"
        ),
        "tie_break_order": [
            "maximum_primary_biological_presence_F1_at_20um",
            "minimum_per_image_count_MAE",
            "minimum_absolute_count_bias",
            "higher_score_threshold",
        ],
        "tie_definition": "exact_after_metrics_are_computed_in_float64",
        "straight_base_to_tip_presence_proxy_during_threshold_selection": True,
        "distal_endpoint_error_used_as_selection_gate": False,
        "complete_line_overlap_used_as_selection_gate": False,
        "length_error_used_as_selection_gate": False,
        "manual_hair_width_assumed": False,
        "validation_labels_used_for_gradient_or_early_stopping": False,
        "blind_images_used": 0,
    }


class CandidateBundleError(RuntimeError):
    """A candidate checkpoint or manifest failed a mandatory provenance gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateBundleError(message)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.casefold()


def _require_sha256(value: Any, field: str) -> str:
    _require(_is_sha256(value), f"{field} must be a lowercase SHA-256 digest")
    return str(value)


def _resolve_audit_path(value: Any, *, audit_path: Path, field: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"dataset audit has no {field}")
    path = Path(value)
    if not path.is_absolute():
        path = audit_path.parent / path
    return path.resolve()


def _load_and_validate_dataset_audit(path: str | Path) -> tuple[Path, dict[str, Any]]:
    audit_path = Path(path).resolve()
    _require(audit_path.is_file(), f"dataset audit does not exist: {audit_path}")
    audit = read_json(audit_path)
    _require(audit.get("status") == "passed", "dataset audit is not a pass")
    _require(
        audit.get("schema_version") == "PHAxis-StageB-train399-dataset-audit-1.0",
        "unsupported dataset audit schema",
    )
    _require(
        int(audit.get("train_records", -1)) == EXPECTED_TRAIN_IMAGES,
        "dataset audit is not train399",
    )
    _require(
        int(audit.get("excluded_val_records", -1)) == EXPECTED_VALIDATION_IMAGES,
        "dataset audit does not exclude exactly val44",
    )
    _require(audit.get("family_key_overlap") == [], "dataset audit has family leakage")
    _require(audit.get("blind_images_used") == 0, "dataset audit is blind-tainted")
    _require(
        audit.get("pyRootHair_called_or_copied") is False,
        "dataset audit does not prohibit pyRootHair",
    )
    _require(
        audit.get("validation_labels_used_for_gradient") is False
        and audit.get("validation_labels_used_for_early_stopping") is False,
        "dataset audit allowed validation labels into optimization",
    )

    train_ids = audit.get("train_ids")
    val_ids = audit.get("excluded_val_ids")
    _require(
        isinstance(train_ids, list)
        and len(train_ids) == EXPECTED_TRAIN_IMAGES
        and len(set(train_ids)) == EXPECTED_TRAIN_IMAGES,
        "dataset audit train IDs are not 399 unique tasks",
    )
    _require(
        isinstance(val_ids, list)
        and len(val_ids) == EXPECTED_VALIDATION_IMAGES
        and len(set(val_ids)) == EXPECTED_VALIDATION_IMAGES,
        "dataset audit validation IDs are not 44 unique tasks",
    )
    _require(not (set(train_ids) & set(val_ids)), "train399 and val44 task IDs overlap")
    _require(
        sha256_json(train_ids) == audit.get("train_ids_sha256"),
        "dataset audit train_ids_sha256 is invalid",
    )
    _require(
        sha256_json(val_ids) == audit.get("excluded_val_ids_sha256"),
        "dataset audit excluded_val_ids_sha256 is invalid",
    )
    for field in (
        "dataset_split_identity_sha256",
        "dataset_manifest_sha256",
        "split_manifest_sha256",
        "split_lock_sha256",
        "integrity_manifest_sha256",
        "train_task_family_sha256",
        "train_families_sha256",
        "excluded_val_families_sha256",
    ):
        _require_sha256(audit.get(field), f"dataset audit {field}")

    split_manifest = _resolve_audit_path(
        audit.get("selected_split_manifest"),
        audit_path=audit_path,
        field="selected_split_manifest",
    )
    split_lock = _resolve_audit_path(
        audit.get("selected_split_lock"),
        audit_path=audit_path,
        field="selected_split_lock",
    )
    _require(split_manifest.is_file(), f"locked split manifest is missing: {split_manifest}")
    _require(split_lock.is_file(), f"locked split receipt is missing: {split_lock}")
    _require(
        sha256_file(split_manifest) == audit["split_manifest_sha256"],
        "locked split manifest changed after the dataset audit",
    )
    _require(
        sha256_file(split_lock) == audit["split_lock_sha256"],
        "locked split receipt changed after the dataset audit",
    )
    return audit_path, audit


def _model_state_digests(state_dict: Mapping[str, Any]) -> tuple[str, str]:
    """Return content and tensor-schema digests without pickle reserialization."""

    import torch

    _require(bool(state_dict), "checkpoint model state_dict is empty")
    content = hashlib.sha256()
    schema: list[tuple[str, str, list[int]]] = []
    for name in sorted(state_dict):
        tensor = state_dict[name]
        _require(torch.is_tensor(tensor), f"model state {name!r} is not a tensor")
        tensor = tensor.detach().cpu().contiguous()
        shape = list(tensor.shape)
        dtype = str(tensor.dtype)
        schema.append((str(name), dtype, shape))
        content.update(str(name).encode("utf-8"))
        content.update(dtype.encode("ascii"))
        content.update(bytes(str(shape), "ascii"))
        content.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return content.hexdigest(), sha256_json(schema)


def _validate_initialization(
    initialization: Mapping[str, Any], *, checkpoint_path: Path
) -> dict[str, Any]:
    _require(
        initialization.get("source") == IMAGENET_SOURCE,
        f"{checkpoint_path}: encoder initialization is not the locked ImageNet source",
    )
    revision = initialization.get("huggingface_revision")
    _require(
        isinstance(revision, str) and len(revision) >= 12,
        f"{checkpoint_path}: ImageNet source revision is missing",
    )
    weight_sha256 = _require_sha256(
        initialization.get("cached_weight_sha256"),
        f"{checkpoint_path}: ImageNet cached_weight_sha256",
    )
    _require(
        int(initialization.get("cached_weight_size_bytes") or 0) > 0,
        f"{checkpoint_path}: ImageNet cached weight size is missing",
    )
    cached_path_value = initialization.get("cached_weight_path")
    _require(
        isinstance(cached_path_value, str) and bool(cached_path_value),
        f"{checkpoint_path}: ImageNet cached weight path is missing",
    )
    cached_path = Path(cached_path_value)
    _require(
        cached_path.is_file(),
        f"{checkpoint_path}: recorded ImageNet source file is unavailable: {cached_path}",
    )
    _require(
        cached_path.stat().st_size == int(initialization["cached_weight_size_bytes"]),
        f"{checkpoint_path}: recorded ImageNet source size changed",
    )
    _require(
        sha256_file(cached_path) == weight_sha256,
        f"{checkpoint_path}: recorded ImageNet source hash changed",
    )
    encoder_sha256 = _require_sha256(
        initialization.get("initial_encoder_state_sha256"),
        f"{checkpoint_path}: initial encoder state",
    )
    model_sha256 = _require_sha256(
        initialization.get("initial_complete_model_state_sha256"),
        f"{checkpoint_path}: initial complete model state",
    )
    _require(
        initialization.get("historical_stageb_checkpoint_loaded") is False,
        f"{checkpoint_path}: historical 443CV state may have been loaded",
    )
    unsigned = dict(initialization)
    recorded_identity = unsigned.pop("initialization_sha256", None)
    _require_sha256(recorded_identity, f"{checkpoint_path}: initialization_sha256")
    _require(
        sha256_json(unsigned) == recorded_identity,
        f"{checkpoint_path}: initialization provenance identity is invalid",
    )
    return {
        "imagenet_source": IMAGENET_SOURCE,
        "imagenet_revision": revision,
        "imagenet_weight_sha256": weight_sha256,
        "imagenet_weight_size_bytes": int(initialization["cached_weight_size_bytes"]),
        "initial_encoder_state_sha256": encoder_sha256,
        "initial_complete_model_state_sha256": model_sha256,
        "initialization_sha256": recorded_identity,
    }


def _resolve_local_sidecar(
    value: Any, *, directory: Path, field: str, checkpoint_path: Path
) -> Path:
    _require(
        isinstance(value, str) and bool(value),
        f"{checkpoint_path}: {field} path is missing",
    )
    path = Path(value)
    if not path.is_absolute():
        path = directory / path
    resolved = path.resolve()
    _require(
        resolved.parent == directory.resolve(),
        f"{checkpoint_path}: {field} is outside the seed directory",
    )
    return resolved


def _validate_completion_receipt(
    checkpoint_path: Path,
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, str | None]:
    """Require exactly one authoritative completion receipt for one seed."""

    directory = checkpoint_path.parent
    receipt_paths = sorted(directory.glob("training_receipt*.json"))
    _require(
        len(receipt_paths) == 1,
        f"{checkpoint_path}: expected exactly one completion receipt, found "
        f"{len(receipt_paths)}",
    )
    receipt_path = receipt_paths[0]
    receipt = read_json(receipt_path)
    suffix = receipt.get("invocation_artifact_suffix")
    _require(
        isinstance(suffix, str)
        and re.fullmatch(r"(?:|_resume_[0-9]{3})", suffix) is not None,
        f"{checkpoint_path}: completion receipt invocation suffix is invalid",
    )
    _require(
        receipt_path.name == f"training_receipt{suffix}.json",
        f"{checkpoint_path}: completion receipt filename/suffix mismatch",
    )
    expected = {
        "schema_version": TRAINING_RECEIPT_SCHEMA,
        "status": "completed",
        "formal_training": True,
        "seed": checkpoint["seed"],
        "epochs": EXPECTED_EPOCH,
        "steps_per_epoch": 399,
        "global_steps": 23_940,
        "checkpoint_sha256": checkpoint_sha256,
        "validation_evaluated_during_training": False,
        "blind_images_used": 0,
        "nvidia_smi_preflight_status": "passed",
        "nvidia_smi_training_monitor_status": "passed",
        "internal_device": "cuda:0",
        "physical_device_mapping_note": (
            "cuda:0 maps to the first entry of CUDA_VISIBLE_DEVICES"
        ),
    }
    for field, value in expected.items():
        _require(
            receipt.get(field) == value,
            f"{checkpoint_path}: training receipt mismatch for {field}",
        )
    recorded_checkpoint = _resolve_local_sidecar(
        receipt.get("checkpoint"),
        directory=directory,
        field="completion receipt checkpoint",
        checkpoint_path=checkpoint_path,
    )
    _require(
        recorded_checkpoint == checkpoint_path.resolve(),
        f"{checkpoint_path}: completion receipt points to another checkpoint",
    )
    _require(
        isinstance(receipt.get("cuda_visible_devices"), str)
        and re.fullmatch(r"[01]", receipt["cuda_visible_devices"]) is not None,
        f"{checkpoint_path}: completion receipt has no single physical GPU mapping",
    )
    _require(
        isinstance(receipt.get("gpu_name"), str) and bool(receipt["gpu_name"]),
        f"{checkpoint_path}: completion receipt GPU name is missing",
    )
    for field in (
        "parameter_count",
        "total_wall_seconds_this_invocation",
        "median_epoch_wall_seconds",
        "peak_allocated_mib",
        "peak_reserved_mib",
    ):
        value = receipt.get(field)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0.0,
            f"{checkpoint_path}: completion receipt {field} is invalid",
        )

    checkpoint_retry_events = checkpoint.get("amp_backward_retry_events", [])
    normalized_retry_count = len(checkpoint_retry_events)
    retry_audit_sha256: str | None = None
    if "amp_backward_retry_count" in receipt:
        _require(
            receipt.get("amp_backward_retry_count") == normalized_retry_count
            and receipt.get("optimizer_steps_skipped_due_nonfinite_gradients") == 0,
            f"{checkpoint_path}: training receipt AMP retry summary mismatch",
        )
        scaler = checkpoint.get("scaler")
        _require(
            isinstance(scaler, Mapping)
            and receipt.get("amp_final_scale") == scaler.get("scale"),
            f"{checkpoint_path}: training receipt final AMP scale mismatch",
        )
        expected_minimum_scale = (
            min(
                [1024.0]
                + [float(event["scale_after_backoff"]) for event in checkpoint_retry_events]
            )
            if checkpoint_retry_events
            else 1024.0
        )
        _require(
            receipt.get("amp_min_scale") == expected_minimum_scale,
            f"{checkpoint_path}: training receipt minimum AMP scale mismatch",
        )
        retry_audit_path = _resolve_local_sidecar(
            receipt.get("amp_backward_retry_audit"),
            directory=directory,
            field="AMP retry audit",
            checkpoint_path=checkpoint_path,
        )
        _require(
            retry_audit_path.is_file(),
            f"{checkpoint_path}: AMP retry audit is absent",
        )
        retry_audit_sha256 = sha256_file(retry_audit_path)
        _require(
            retry_audit_sha256 == receipt.get("amp_backward_retry_audit_sha256"),
            f"{checkpoint_path}: AMP retry audit hash mismatch",
        )
        retry_audit = read_json(retry_audit_path)
        _require(
            retry_audit.get("schema_version")
            == "PHAxis-StageB-AMP-backward-retry-audit-1.0"
            and retry_audit.get("status") == "completed_through_epoch"
            and retry_audit.get("seed") == checkpoint["seed"]
            and retry_audit.get("completed_epoch") == EXPECTED_EPOCH
            and retry_audit.get("event_count") == normalized_retry_count
            and retry_audit.get("events") == checkpoint_retry_events
            and retry_audit.get("optimizer_steps_skipped_due_nonfinite_gradients")
            == 0
            and retry_audit.get("same_forward_graph_replayed") is True
            and retry_audit.get("blind_images_used") == 0,
            f"{checkpoint_path}: AMP retry audit differs from checkpoint provenance",
        )
    else:
        _require(
            normalized_retry_count == 0,
            f"{checkpoint_path}: legacy receipt cannot omit nonzero AMP retries",
        )
    return {
        "training_receipt_filename": receipt_path.name,
        "training_receipt_sha256": sha256_file(receipt_path),
        "amp_backward_retry_audit_sha256": retry_audit_sha256,
    }


def _validate_required_sidecars(
    checkpoint_path: Path,
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_sha256: str,
) -> dict[str, str | None]:
    """Require the four reconstructable training sidecars and validate their bytes."""

    sidecar_hashes: dict[str, str | None] = {
        "history_sha256": None,
        "training_contract_sha256": None,
        "config_sidecar_sha256": None,
        "initialization_sidecar_sha256": None,
    }
    directory = checkpoint_path.parent
    history_path = directory / "history.json"
    _require(
        history_path.is_file(),
        f"{checkpoint_path}: required history.json sidecar is missing",
    )
    history = json.loads(history_path.read_text(encoding="utf-8"))
    _require(
        isinstance(history, list) and len(history) == EXPECTED_EPOCH,
        f"{checkpoint_path}: history does not contain 60 epochs",
    )
    for index, record in enumerate(history, start=1):
        _require(isinstance(record, Mapping), f"{checkpoint_path}: invalid history row")
        _require(
            record.get("epoch") == index
            and record.get("batches") == 399
            and record.get("global_step") == index * 399
            and record.get("validation_evaluated") is False,
            f"{checkpoint_path}: history contract mismatch at epoch {index}",
        )
    _require(
        checkpoint.get("history") == history,
        f"{checkpoint_path}: history.json differs from embedded checkpoint history",
    )
    sidecar_hashes["history_sha256"] = sha256_file(history_path)

    for filename, checkpoint_field, result_field in (
        ("training_contract.json", "contract", "training_contract_sha256"),
        ("config.json", "cfg", "config_sidecar_sha256"),
        ("initialization.json", "initialization", "initialization_sidecar_sha256"),
    ):
        sidecar_path = directory / filename
        _require(
            sidecar_path.is_file(),
            f"{checkpoint_path}: required {filename} sidecar is missing",
        )
        sidecar = read_json(sidecar_path)
        _require(
            sha256_json(sidecar) == sha256_json(checkpoint[checkpoint_field]),
            f"{checkpoint_path}: {filename} differs from checkpoint",
        )
        sidecar_hashes[result_field] = sha256_file(sidecar_path)
    return sidecar_hashes


def _validate_amp_retry_provenance(
    checkpoint_path: Path, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize legacy zero-retry and amended same-graph retry checkpoints."""

    raw_events = checkpoint.get("amp_backward_retry_events", [])
    _require(
        isinstance(raw_events, list),
        f"{checkpoint_path}: AMP backward retry events are not a list",
    )
    events: list[dict[str, Any]] = []
    retry_counts_by_step: dict[tuple[int, int], int] = {}
    previous_position: tuple[int, int] | None = None
    previous_scale_after: float | None = None
    for index, raw_event in enumerate(raw_events, start=1):
        _require(
            isinstance(raw_event, Mapping),
            f"{checkpoint_path}: AMP retry event {index} is not a mapping",
        )
        event = dict(raw_event)
        before = event.get("scale_before_backoff")
        after = event.get("scale_after_backoff")
        _require(
            isinstance(before, (int, float))
            and not isinstance(before, bool)
            and isinstance(after, (int, float))
            and not isinstance(after, bool)
            and float(before) > 0.0
            and float(after) > 0.0
            and abs(float(after) - float(before) * 0.5)
            <= max(1e-12, abs(float(before)) * 1e-12),
            f"{checkpoint_path}: AMP retry event {index} is not one 0.5 backoff",
        )
        _require(
            event.get("same_forward_graph_replayed") is True
            and event.get("optimizer_step_skipped") is False,
            f"{checkpoint_path}: AMP retry event {index} changed the batch/step contract",
        )
        _require(
            isinstance(event.get("epoch"), int)
            and 1 <= int(event["epoch"]) <= EXPECTED_EPOCH
            and isinstance(event.get("global_step"), int)
            and 0 <= int(event["global_step"]) < 23_940,
            f"{checkpoint_path}: AMP retry event {index} has invalid training position",
        )
        position = (int(event["epoch"]), int(event["global_step"]))
        _require(
            (position[0] - 1) * 399 <= position[1] < position[0] * 399,
            f"{checkpoint_path}: AMP retry event {index} epoch/step disagree",
        )
        _require(
            previous_position is None or position >= previous_position,
            f"{checkpoint_path}: AMP retry events are not chronological",
        )
        if previous_scale_after is None:
            _require(
                float(before) == 1024.0,
                f"{checkpoint_path}: first AMP retry did not start at scale 1024",
            )
        else:
            _require(
                abs(float(before) - previous_scale_after)
                <= max(1e-12, abs(previous_scale_after) * 1e-12),
                f"{checkpoint_path}: AMP retry scale sequence is discontinuous",
            )
        expected_retry_index = retry_counts_by_step.get(position, 0) + 1
        _require(
            event.get("retry_index") == expected_retry_index,
            f"{checkpoint_path}: AMP retry sequence is not contiguous within a step",
        )
        _require(
            expected_retry_index <= 16,
            f"{checkpoint_path}: AMP retry count exceeds the 16-retry limit",
        )
        retry_counts_by_step[position] = expected_retry_index
        _require(
            isinstance(event.get("nonfinite_parameters"), list)
            and bool(event["nonfinite_parameters"]),
            f"{checkpoint_path}: AMP retry event {index} lacks the failed-gradient audit",
        )
        events.append(event)
        previous_position = position
        previous_scale_after = float(after)

    embedded_history = checkpoint.get("history")
    if isinstance(embedded_history, list):
        history_retry_count = sum(
            int(record.get("amp_backward_retry_count", 0))
            for record in embedded_history
            if isinstance(record, Mapping)
        )
        _require(
            history_retry_count == len(events),
            f"{checkpoint_path}: history AMP retry count differs from checkpoint events",
        )
        _require(
            all(
                int(record.get("optimizer_steps_skipped_due_nonfinite_gradients", 0))
                == 0
                for record in embedded_history
                if isinstance(record, Mapping)
            ),
            f"{checkpoint_path}: history reports a skipped optimizer step",
        )

    scaler_state = checkpoint.get("scaler")
    _require(
        isinstance(scaler_state, Mapping),
        f"{checkpoint_path}: AMP scaler state is missing",
    )
    required_scaler_keys = {
        "scale",
        "growth_factor",
        "backoff_factor",
        "growth_interval",
        "_growth_tracker",
    }
    _require(
        set(scaler_state) == required_scaler_keys,
        f"{checkpoint_path}: AMP scaler state is incomplete or has drifted",
    )
    config = checkpoint.get("cfg")
    _require(isinstance(config, Mapping), f"{checkpoint_path}: AMP config is missing")
    expected_scaler_values = {
        "growth_factor": config.get("amp_growth_factor"),
        "backoff_factor": config.get("amp_backoff_factor"),
        "growth_interval": config.get("amp_growth_interval"),
    }
    for field, expected in expected_scaler_values.items():
        _require(
            scaler_state.get(field) == expected,
            f"{checkpoint_path}: AMP scaler {field} differs from config",
        )
    final_scale_value = scaler_state.get("scale")
    _require(
        isinstance(final_scale_value, (int, float))
        and not isinstance(final_scale_value, bool)
        and math.isfinite(float(final_scale_value))
        and float(final_scale_value) > 0.0,
        f"{checkpoint_path}: final AMP scale is invalid",
    )
    final_scale = float(final_scale_value)
    expected_final_scale = (
        float(events[-1]["scale_after_backoff"])
        if events
        else float(config["amp_initial_scale"])
    )
    _require(
        final_scale == expected_final_scale,
        f"{checkpoint_path}: final AMP scale differs from retry provenance",
    )
    growth_tracker = scaler_state.get("_growth_tracker")
    _require(
        isinstance(growth_tracker, int) and not isinstance(growth_tracker, bool),
        f"{checkpoint_path}: AMP growth tracker is invalid",
    )
    expected_growth_tracker = 23_940 - (
        int(events[-1]["global_step"]) if events else 0
    )
    _require(
        growth_tracker == expected_growth_tracker,
        f"{checkpoint_path}: AMP growth tracker differs from retry provenance",
    )
    return {
        "amp_backward_retry_count": len(events),
        "amp_backward_retry_mode": (
            "same_forward_graph_backoff" if events else "legacy_or_amended_zero_retry"
        ),
        "optimizer_steps_skipped_due_nonfinite_gradients": 0,
        "amp_final_scale": final_scale,
        "amp_growth_tracker": growth_tracker,
    }


def _validate_adamw_optimizer_state(
    checkpoint_path: Path, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    """Require a complete fixed-horizon AdamW state for every parameter id."""

    import torch

    optimizer = checkpoint.get("optimizer")
    _require(
        isinstance(optimizer, Mapping),
        f"{checkpoint_path}: AdamW optimizer state is missing",
    )
    _require(
        set(optimizer) == {"state", "param_groups"},
        f"{checkpoint_path}: AdamW optimizer state structure drifted",
    )
    state = optimizer.get("state")
    param_groups = optimizer.get("param_groups")
    _require(
        isinstance(state, Mapping) and bool(state),
        f"{checkpoint_path}: AdamW per-parameter state is empty",
    )
    _require(
        isinstance(param_groups, list) and bool(param_groups),
        f"{checkpoint_path}: AdamW parameter groups are missing",
    )
    config = checkpoint.get("cfg")
    _require(isinstance(config, Mapping), f"{checkpoint_path}: optimizer config is missing")
    parameter_ids: list[int] = []
    for index, group in enumerate(param_groups):
        _require(
            isinstance(group, Mapping),
            f"{checkpoint_path}: AdamW parameter group {index} is invalid",
        )
        ids = group.get("params")
        _require(
            isinstance(ids, list)
            and bool(ids)
            and all(isinstance(value, int) and not isinstance(value, bool) for value in ids),
            f"{checkpoint_path}: AdamW parameter group {index} has invalid ids",
        )
        parameter_ids.extend(ids)
        _require(
            tuple(group.get("betas", ())) == (0.9, 0.999)
            and group.get("eps") == 1e-8
            and group.get("weight_decay") == config.get("weight_decay")
            and group.get("amsgrad") is False
            and group.get("maximize") is False
            and group.get("decoupled_weight_decay") is True,
            f"{checkpoint_path}: optimizer parameter group {index} is not locked AdamW",
        )
        learning_rate = group.get("lr")
        _require(
            isinstance(learning_rate, (int, float))
            and not isinstance(learning_rate, bool)
            and math.isfinite(float(learning_rate))
            and 0.0 <= float(learning_rate) <= float(config.get("lr", -1.0)),
            f"{checkpoint_path}: AdamW parameter group {index} learning rate is invalid",
        )
    _require(
        len(parameter_ids) == len(set(parameter_ids)),
        f"{checkpoint_path}: AdamW parameter ids occur in multiple groups",
    )
    _require(
        set(state) == set(parameter_ids),
        f"{checkpoint_path}: AdamW state does not cover exactly the param-group ids",
    )
    for parameter_id in parameter_ids:
        parameter_state = state[parameter_id]
        _require(
            isinstance(parameter_state, Mapping)
            and set(parameter_state) == {"step", "exp_avg", "exp_avg_sq"},
            f"{checkpoint_path}: AdamW state for parameter {parameter_id} is incomplete",
        )
        step = parameter_state["step"]
        if torch.is_tensor(step):
            _require(
                step.numel() == 1,
                f"{checkpoint_path}: AdamW step for parameter {parameter_id} is not scalar",
            )
            step = step.item()
        _require(
            isinstance(step, (int, float))
            and not isinstance(step, bool)
            and float(step) == 23_940.0,
            f"{checkpoint_path}: AdamW step for parameter {parameter_id} is not 23940",
        )
        exp_avg = parameter_state["exp_avg"]
        exp_avg_sq = parameter_state["exp_avg_sq"]
        _require(
            torch.is_tensor(exp_avg)
            and torch.is_tensor(exp_avg_sq)
            and exp_avg.shape == exp_avg_sq.shape,
            f"{checkpoint_path}: AdamW moments for parameter {parameter_id} are invalid",
        )
    return {
        "optimizer_name": "AdamW",
        "optimizer_parameter_state_count": len(parameter_ids),
        "optimizer_parameter_step": 23_940,
    }


def _validate_checkpoint(
    path: Path, *, audit: Mapping[str, Any], dataset_audit_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    _require(path.is_file(), f"checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # pragma: no cover - exact torch exception is version-specific
        raise CandidateBundleError(f"cannot safely load checkpoint {path}: {error}") from error
    _require(isinstance(checkpoint, dict), f"{path}: checkpoint is not a mapping")
    _require(
        checkpoint.get("schema_version") == CHECKPOINT_SCHEMA,
        f"{path}: unsupported checkpoint schema",
    )
    contract = checkpoint.get("contract")
    config = checkpoint.get("cfg")
    initialization = checkpoint.get("initialization")
    _require(isinstance(contract, Mapping), f"{path}: training contract is missing")
    _require(isinstance(config, Mapping), f"{path}: training config is missing")
    _require(isinstance(initialization, Mapping), f"{path}: initialization is missing")

    seed = checkpoint.get("seed")
    _require(
        isinstance(seed, int) and not isinstance(seed, bool),
        f"{path}: seed is not an integer",
    )
    member_id = checkpoint.get("member_id")
    _require(member_id == f"seed_{seed}", f"{path}: member_id does not bind its seed")
    _require(seed in FORMAL_TRAIN399_SEEDS, f"{path}: seed is not in the fixed five-seed lock")
    _require(int(checkpoint.get("epoch", -1)) == EXPECTED_EPOCH, f"{path}: epoch is not 60")
    _require(int(config.get("epochs", -1)) == EXPECTED_EPOCH, f"{path}: cfg.epochs is not 60")
    _require(
        config.get("fixed_last_epoch_policy") is True,
        f"{path}: fixed last-epoch policy is disabled",
    )
    _require(
        int(config.get("crops_per_image", -1)) == 8,
        f"{path}: formal crops_per_image lock changed",
    )
    _require(
        config.get("imagenet_source") == IMAGENET_SOURCE,
        f"{path}: config ImageNet source differs from provenance",
    )
    expected_amp_policy = {
        "dtype": "float16",
        "enabled": True,
        "initial_scale": 1024.0,
        "growth_interval": 1_000_000,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "nonfinite_step_policy": "fail_closed_no_optimizer_step_skip",
    }
    expected_amp_config = {
        "amp": True,
        "amp_initial_scale": 1024.0,
        "amp_growth_interval": 1_000_000,
        "amp_growth_factor": 2.0,
        "amp_backoff_factor": 0.5,
    }
    for field, expected in expected_amp_config.items():
        _require(config.get(field) == expected, f"{path}: AMP config mismatch for {field}")
    batch_size = int(config.get("batch_size", -1))
    _require(batch_size == 8, f"{path}: formal preregistered batch_size is not 8")
    expected_global_steps = 23_940
    _require(
        int(checkpoint.get("global_step", -1)) == expected_global_steps,
        f"{path}: global_step does not represent 60 complete fixed-horizon epochs",
    )
    _require(
        int(checkpoint.get("training_images", -1)) == EXPECTED_TRAIN_IMAGES,
        f"{path}: top-level training_images is not 399",
    )
    _require(
        int(checkpoint.get("validation_images", -1)) == EXPECTED_VALIDATION_IMAGES,
        f"{path}: top-level validation_images is not 44",
    )
    _require(
        checkpoint.get("validation_labels_used_for_gradient_or_early_stopping") is False,
        f"{path}: top-level validation optimization lock is not false",
    )

    expected_contract_values = {
        "formal_training": True,
        "training_policy": "all399_fixed_60_epoch_last_checkpoint",
        "model_selection_policy": "none_during_training",
        "seed": seed,
        "member_id": member_id,
        "training_images": EXPECTED_TRAIN_IMAGES,
        "validation_images": EXPECTED_VALIDATION_IMAGES,
        "validation_labels_used_for_gradient": False,
        "validation_labels_used_for_early_stopping": False,
        "validation_labels_used_for_gradient_or_early_stopping": False,
        "validation_metrics_observed_during_training": False,
        "blind_images_used": 0,
        "pyRootHair_called_or_copied": False,
    }
    for field, expected in expected_contract_values.items():
        _require(contract.get(field) == expected, f"{path}: contract mismatch for {field}")
    _require(contract.get("amp_policy") == expected_amp_policy, f"{path}: AMP policy changed")
    _require(
        "ImageNet" in str(contract.get("initialization_policy", ""))
        and "random" in str(contract.get("initialization_policy", "")).casefold(),
        f"{path}: contract does not declare ImageNet plus random head initialization",
    )
    _require(
        "443" in str(contract.get("prohibited_initialization", ""))
        and "val44" in str(contract.get("prohibited_initialization", "")),
        f"{path}: prohibited legacy initialization is not explicitly locked",
    )

    train_ids = contract.get("train_ids")
    excluded_val_ids = contract.get("excluded_val_ids")
    _require(train_ids == audit["train_ids"], f"{path}: train IDs differ from the audit")
    _require(
        excluded_val_ids == audit["excluded_val_ids"],
        f"{path}: excluded val IDs differ from the audit",
    )
    audit_bindings = {
        "dataset_audit_sha256": dataset_audit_sha256,
        "training_task_ids_sha256": audit["train_ids_sha256"],
        "train_ids_sha256": audit["train_ids_sha256"],
        "train_task_family_sha256": audit["train_task_family_sha256"],
        "train_families_sha256": audit["train_families_sha256"],
        "excluded_val_ids_sha256": audit["excluded_val_ids_sha256"],
        "excluded_val_families_sha256": audit["excluded_val_families_sha256"],
        "dataset_split_identity_sha256": audit["dataset_split_identity_sha256"],
        "dataset_manifest_sha256": audit["dataset_manifest_sha256"],
        "split_manifest_sha256": audit["split_manifest_sha256"],
        "integrity_manifest_sha256": audit["integrity_manifest_sha256"],
    }
    for field, expected in audit_bindings.items():
        _require(contract.get(field) == expected, f"{path}: audit lock mismatch for {field}")
    _require(
        checkpoint.get("training_task_ids_sha256") == audit["train_ids_sha256"],
        f"{path}: top-level training task identity differs from the audit",
    )
    _require(
        checkpoint.get("split_manifest_sha256") == audit["split_manifest_sha256"],
        f"{path}: top-level split identity differs from the audit",
    )
    config_sha256 = sha256_json(dict(config))
    _require(
        contract.get("config_sha256") == config_sha256,
        f"{path}: config_sha256 is invalid",
    )
    _require_sha256(contract.get("cache_identity_sha256"), f"{path}: cache identity")

    initialization_summary = _validate_initialization(initialization, checkpoint_path=path)
    _require(
        checkpoint.get("initialization_sha256")
        == initialization_summary["initialization_sha256"],
        f"{path}: top-level initialization identity is invalid",
    )
    state_sha256, state_schema_sha256 = _model_state_digests(checkpoint.get("model", {}))
    checkpoint_sha256 = sha256_file(path)
    retry_summary = _validate_amp_retry_provenance(path, checkpoint)
    optimizer_summary = _validate_adamw_optimizer_state(path, checkpoint)
    completion_receipt_hashes = _validate_completion_receipt(
        path,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    sidecar_hashes = _validate_required_sidecars(
        path,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
    )
    member = {
        "seed": seed,
        "member_id": member_id,
        "epoch": EXPECTED_EPOCH,
        "global_step": int(checkpoint["global_step"]),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": path.stat().st_size,
        "model_state_sha256": state_sha256,
        "model_state_schema_sha256": state_schema_sha256,
        **retry_summary,
        **optimizer_summary,
        **completion_receipt_hashes,
        **sidecar_hashes,
        **initialization_summary,
    }
    invariant = {
        "dataset_audit_sha256": contract["dataset_audit_sha256"],
        "training_task_ids_sha256": contract["training_task_ids_sha256"],
        "train_task_family_sha256": contract["train_task_family_sha256"],
        "train_families_sha256": contract["train_families_sha256"],
        "excluded_val_ids_sha256": contract["excluded_val_ids_sha256"],
        "excluded_val_families_sha256": contract["excluded_val_families_sha256"],
        "dataset_split_identity_sha256": contract["dataset_split_identity_sha256"],
        "dataset_manifest_sha256": contract["dataset_manifest_sha256"],
        "split_manifest_sha256": contract["split_manifest_sha256"],
        "integrity_manifest_sha256": contract["integrity_manifest_sha256"],
        "cache_identity_sha256": contract["cache_identity_sha256"],
        "config_sha256": contract["config_sha256"],
        "amp_policy": expected_amp_policy,
        "imagenet_source": initialization_summary["imagenet_source"],
        "imagenet_revision": initialization_summary["imagenet_revision"],
        "imagenet_weight_sha256": initialization_summary["imagenet_weight_sha256"],
        "initial_encoder_state_sha256": initialization_summary[
            "initial_encoder_state_sha256"
        ],
        "model_state_schema_sha256": state_schema_sha256,
    }
    return member, invariant


def _detection_metadata(identity: Mapping[str, Any], candidate_identity: str) -> dict[str, Any]:
    members = identity["members"]
    return {
        "expert_id": "PHAxis-StageB-train399-five-seed",
        "ensemble_members": 5,
        "checkpoint_policy": TRAIN399_CHECKPOINT_POLICY,
        "deployment_role": CANDIDATE_STATUS,
        "operating_point_status": "pending_QCdevelopment44_selection",
        "selected_score_threshold": None,
        "selection_receipt_sha256": None,
        "selection_receipt_identity_sha256": None,
        "candidate_pool_identity_sha256": None,
        "selected_model_metadata_identity_sha256": None,
        "training_images": EXPECTED_TRAIN_IMAGES,
        "validation_images": EXPECTED_VALIDATION_IMAGES,
        "validation_labels_used_for_gradient_or_early_stopping": False,
        "blind_images_used": 0,
        "seeds": [member["seed"] for member in members],
        "member_ids": [member["member_id"] for member in members],
        "checkpoint_sha256": [member["checkpoint_sha256"] for member in members],
        "model_state_sha256": [member["model_state_sha256"] for member in members],
        "training_task_ids_sha256": identity["training_lock"][
            "training_task_ids_sha256"
        ],
        "split_manifest_sha256": identity["training_lock"]["split_manifest_sha256"],
        "training_lock_identity_sha256": identity["training_lock_identity_sha256"],
        "candidate_bundle_identity_sha256": candidate_identity,
        "operating_point_selection_contract_sha256": sha256_json(
            identity["operating_point_selection_contract"]
        ),
    }


def validate_train399_detection_model_metadata(
    model: Mapping[str, Any], *, allow_pending: bool = False
) -> None:
    """Validate the self-contained model block used by new detection payloads."""

    _require(
        model.get("checkpoint_policy") == TRAIN399_CHECKPOINT_POLICY,
        "train399 detection model has the wrong checkpoint policy",
    )
    _require(
        isinstance(model.get("expert_id"), str) and bool(model.get("expert_id")),
        "train399 detection model has no expert_id",
    )
    _require(model.get("ensemble_members") == 5, "train399 detection ensemble is not five")
    _require(
        model.get("deployment_role") in {CANDIDATE_STATUS, "formally_promoted"},
        "train399 detection deployment role is invalid",
    )
    _require(model.get("training_images") == 399, "train399 detection training count changed")
    _require(model.get("validation_images") == 44, "train399 detection val count changed")
    _require(
        model.get("validation_labels_used_for_gradient_or_early_stopping") is False,
        "train399 detection reports validation leakage",
    )
    _require(model.get("blind_images_used") == 0, "train399 detection model is blind-tainted")
    seeds = model.get("seeds")
    member_ids = model.get("member_ids")
    _require(
        tuple(seeds or ()) == FORMAL_TRAIN399_SEEDS,
        "train399 detection seeds differ from the five-seed lock",
    )
    _require(
        member_ids == [f"seed_{seed}" for seed in FORMAL_TRAIN399_SEEDS],
        "train399 detection member IDs do not bind the seeds",
    )
    for field in ("checkpoint_sha256", "model_state_sha256"):
        values = model.get(field)
        _require(
            isinstance(values, list)
            and len(values) == 5
            and len(set(values)) == 5
            and all(_is_sha256(value) for value in values),
            f"train399 detection {field} is not five distinct hashes",
        )
    for field in (
        "training_task_ids_sha256",
        "split_manifest_sha256",
        "training_lock_identity_sha256",
        "candidate_bundle_identity_sha256",
        "operating_point_selection_contract_sha256",
    ):
        _require_sha256(model.get(field), f"train399 detection {field}")
    operating_point_status = model.get("operating_point_status")
    if operating_point_status == "pending_QCdevelopment44_selection":
        _require(allow_pending, "train399 operating point is still pending selection")
        _require(
            model.get("selected_score_threshold") is None
            and model.get("selection_receipt_sha256") is None
            and model.get("selection_receipt_identity_sha256") is None
            and model.get("candidate_pool_identity_sha256") is None
            and model.get("selected_model_metadata_identity_sha256") is None,
            "pending train399 metadata contains a partially bound operating point",
        )
        return
    _require(
        operating_point_status == "selected_on_locked_QCdevelopment44",
        "train399 operating point status is invalid",
    )
    selected = model.get("selected_score_threshold")
    _require(
        isinstance(selected, (int, float))
        and not isinstance(selected, bool)
        and any(abs(float(selected) - value) <= 1e-12 for value in PREREGISTERED_SCORE_THRESHOLDS),
        "train399 selected score threshold is outside the preregistered grid",
    )
    _require_sha256(model.get("selection_receipt_sha256"), "selection receipt")
    _require_sha256(
        model.get("selection_receipt_identity_sha256"), "selection receipt identity"
    )
    _require_sha256(model.get("candidate_pool_identity_sha256"), "candidate pool identity")
    metadata_identity = _require_sha256(
        model.get("selected_model_metadata_identity_sha256"),
        "selected model metadata identity",
    )
    unsigned_metadata = deepcopy(dict(model))
    unsigned_metadata.pop("precision_mode", None)
    unsigned_metadata.pop("selected_model_metadata_identity_sha256", None)
    _require(
        sha256_json(unsigned_metadata) == metadata_identity,
        "selected model metadata identity is invalid",
    )


def bind_selected_operating_point(
    pending_model_metadata: Mapping[str, Any],
    *,
    selected_score_threshold: float,
    selection_receipt_sha256: str,
    selection_receipt_identity_sha256: str,
    candidate_pool_identity_sha256: str,
) -> dict[str, Any]:
    """Bind a preregistered QC-development choice without promoting the model."""

    validate_train399_detection_model_metadata(
        pending_model_metadata, allow_pending=True
    )
    _require(
        pending_model_metadata.get("operating_point_status")
        == "pending_QCdevelopment44_selection",
        "operating point was already bound",
    )
    selected = deepcopy(dict(pending_model_metadata))
    selected.update(
        {
            "operating_point_status": "selected_on_locked_QCdevelopment44",
            "selected_score_threshold": float(selected_score_threshold),
            "selection_receipt_sha256": selection_receipt_sha256,
            "selection_receipt_identity_sha256": selection_receipt_identity_sha256,
            "candidate_pool_identity_sha256": candidate_pool_identity_sha256,
        }
    )
    unsigned_metadata = deepcopy(selected)
    unsigned_metadata.pop("selected_model_metadata_identity_sha256", None)
    selected["selected_model_metadata_identity_sha256"] = sha256_json(
        unsigned_metadata
    )
    validate_train399_detection_model_metadata(selected)
    return selected


def _resolve_amp_amendment_path(
    checkpoint_paths: Sequence[Path], explicit_path: str | Path | None
) -> tuple[Path, Path]:
    roots = {path.parent.parent.resolve() for path in checkpoint_paths}
    _require(
        len(roots) == 1,
        "the five seed directories do not share one candidate model root",
    )
    model_root = next(iter(roots))
    if explicit_path is None:
        candidates = sorted(model_root.glob("AMP_BACKWARD_RETRY_AMENDMENT*.json"))
        _require(
            len(candidates) == 1,
            "cannot safely infer exactly one AMP backward-retry amendment; "
            "pass it explicitly",
        )
        amendment_path = candidates[0].resolve()
    else:
        amendment_path = Path(explicit_path).resolve()
    _require(
        amendment_path.is_file(),
        f"AMP backward-retry amendment does not exist: {amendment_path}",
    )
    _require(
        amendment_path.parent == model_root,
        "AMP backward-retry amendment is not in the shared candidate model root",
    )
    return amendment_path, model_root


def _resolve_amendment_reference(
    value: Any, *, amendment_path: Path, field: str
) -> Path:
    _require(
        isinstance(value, str) and bool(value),
        f"AMP amendment has no {field}",
    )
    path = Path(value)
    if not path.is_absolute():
        path = amendment_path.parent / path
    return path.resolve()


def _validate_amp_amendment(
    amendment_path: Path,
    *,
    model_root: Path,
    members: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], Path]:
    amendment = read_json(amendment_path)
    _require(
        amendment.get("schema_version") == AMP_AMENDMENT_SCHEMA,
        "unsupported AMP backward-retry amendment schema",
    )
    _require(
        amendment.get("status")
        == "applied_before_authoritative_seed3_optimizer_trajectory",
        "AMP backward-retry amendment status is not authoritative",
    )
    fixed_policy = amp_backward_retry_policy_lock()
    _require(
        amendment.get("amended_numeric_policy") == fixed_policy,
        "AMP backward-retry amendment policy drifted",
    )
    scientific_contract = amendment.get("unchanged_scientific_contract")
    expected_scientific_contract = {
        "training_images": 399,
        "excluded_qcdevelopment_images": 44,
        "family_key_overlap": 0,
        "epochs": 60,
        "steps_per_epoch": 399,
        "global_steps_per_seed": 23_940,
        "batch_size": 8,
        "crops_per_image": 8,
        "architecture_changed": False,
        "loss_objective_changed": False,
        "augmentation_or_sampler_changed": False,
        "validation_used_for_gradient_early_stopping_or_retry": False,
        "blind_images_used": 0,
        "pyRootHair_called_or_copied": False,
    }
    _require(
        scientific_contract == expected_scientific_contract,
        "AMP amendment changed the scientific training contract",
    )

    failed_attempt = amendment.get("superseded_failed_attempt")
    _require(
        isinstance(failed_attempt, Mapping)
        and failed_attempt.get("seed") == 2026082803
        and failed_attempt.get("completed_epoch") == 0
        and failed_attempt.get("global_step_at_failure") == 6
        and failed_attempt.get("authoritative_checkpoint_created") is False
        and failed_attempt.get("blind_images_used") == 0,
        "AMP amendment superseded-failure lock drifted",
    )
    failure_path = _resolve_amendment_reference(
        failed_attempt.get("failure_receipt"),
        amendment_path=amendment_path,
        field="superseded failure receipt",
    )
    try:
        failure_path.relative_to(model_root)
    except ValueError:
        _require(False, "AMP amendment failure receipt is outside the model root")
    _require(failure_path.is_file(), "AMP amendment failure receipt is missing")
    failure_sha256 = sha256_file(failure_path)
    _require(
        failure_sha256 == failed_attempt.get("failure_receipt_sha256"),
        "AMP amendment failure receipt hash mismatch",
    )
    failure = read_json(failure_path)
    _require(
        failure.get("schema_version") == TRAINING_FAILURE_SCHEMA
        and failure.get("status") == "failed"
        and failure.get("completed_epoch") == 0
        and failure.get("global_step") == 6
        and failure.get("exception_type") == "builtins.FloatingPointError"
        and failure.get("exception_swallowed") is False
        and failure.get("nvidia_smi_preflight_status") == "passed"
        and failure.get("blind_images_used") == 0,
        "AMP amendment referenced failure receipt fields drifted",
    )
    failed_loss = failure.get("last_finite_loss_total")
    _require(
        isinstance(failed_loss, (int, float))
        and not isinstance(failed_loss, bool)
        and math.isfinite(float(failed_loss)),
        "AMP amendment failure receipt has no finite pre-overflow loss",
    )
    root_cause = amendment.get("root_cause")
    _require(
        isinstance(root_cause, Mapping)
        and root_cause.get("failure_class") == "fp16_scaled_backward_overflow"
        and root_cause.get("loss_was_finite") is True
        and root_cause.get("loss_total") == failed_loss
        and root_cause.get("oom") is False
        and root_cause.get("data_or_target_nonfinite") is False
        and root_cause.get("gpu_preflight_passed") is True,
        "AMP amendment root-cause evidence drifted",
    )
    failure_message = str(failure.get("exception_message", ""))
    parameter_groups = root_cause.get("nonfinite_parameter_groups")
    _require(
        isinstance(parameter_groups, list)
        and bool(parameter_groups)
        and all(str(group) in failure_message for group in parameter_groups),
        "AMP amendment overflow parameters differ from the failure receipt",
    )

    implementation = amendment.get("implementation")
    _require(isinstance(implementation, Mapping), "AMP amendment implementation is missing")
    training_source = _resolve_amendment_reference(
        implementation.get("training_source"),
        amendment_path=amendment_path,
        field="training source",
    )
    _require(training_source.is_file(), "AMP amendment training source is unavailable")
    training_source_sha256 = sha256_file(training_source)
    _require(
        training_source_sha256 == implementation.get("training_source_sha256"),
        "AMP amendment training source hash mismatch",
    )
    _require(
        implementation.get("finite_path_gradient_identity_tested") is True
        and implementation.get("same_graph_backoff_tested") is True
        and implementation.get("retry_exhaustion_fail_closed_tested") is True
        and implementation.get("candidate_retry_tamper_gate_tested") is True,
        "AMP amendment implementation test evidence is incomplete",
    )

    legacy = amendment.get("legacy_zero_retry_normalization")
    _require(
        isinstance(legacy, Mapping)
        and legacy.get("seeds") == [2026082801, 2026082802]
        and legacy.get("completed_epochs") == [60, 60]
        and legacy.get("completed_global_steps") == [23_940, 23_940]
        and legacy.get("final_scaler_scale") == [1024.0, 1024.0]
        and legacy.get("scaler_growth_tracker") == [23_940, 23_940]
        and legacy.get("retry_event_field_present") == [False, False]
        and legacy.get("normalized_amp_backward_retry_count") == [0, 0],
        "AMP amendment legacy zero-retry normalization drifted",
    )
    restart = amendment.get("authoritative_seed3_restart")
    _require(
        isinstance(restart, Mapping)
        and restart.get("physical_gpu") == 1
        and restart.get("cuda_visible_devices") == "1"
        and restart.get("internal_device") == "cuda:0"
        and restart.get("launch_mode") == "fresh"
        and restart.get("deterministic_initialization_reproduced") is True,
        "AMP amendment authoritative seed-3 restart provenance drifted",
    )
    for field in (
        "failed_and_restart_initialization_file_sha256",
        "initialization_identity_sha256",
        "initial_complete_model_state_sha256",
    ):
        _require_sha256(restart.get(field), f"AMP amendment seed-3 {field}")

    by_seed = {int(member["seed"]): member for member in members}
    for seed in (2026082801, 2026082802):
        member = by_seed[seed]
        _require(
            member.get("amp_backward_retry_count") == 0
            and member.get("amp_final_scale") == 1024.0
            and member.get("amp_growth_tracker") == 23_940,
            f"AMP amendment legacy normalization differs from seed {seed}",
        )
    seed3 = by_seed[2026082803]
    _require(
        int(seed3.get("amp_backward_retry_count", 0)) >= 1
        and float(seed3.get("amp_final_scale", 1024.0)) < 1024.0
        and seed3.get("initialization_sha256")
        == restart["initialization_identity_sha256"]
        and seed3.get("initial_complete_model_state_sha256")
        == restart["initial_complete_model_state_sha256"],
        "AMP amendment does not bind the authoritative seed-3 restart/checkpoint",
    )

    evidence = {
        "schema_version": AMP_AMENDMENT_SCHEMA,
        "status": amendment["status"],
        "amendment_sha256": sha256_file(amendment_path),
        "failure_receipt_schema_version": TRAINING_FAILURE_SCHEMA,
        "failure_receipt_sha256": failure_sha256,
        "fixed_numeric_policy": fixed_policy,
        "unchanged_scientific_contract_sha256": sha256_json(scientific_contract),
        "training_source_sha256": training_source_sha256,
    }
    return evidence, failure_path


def build_candidate_manifest(
    checkpoint_paths: Sequence[str | Path],
    *,
    dataset_audit_path: str | Path,
    amp_amendment_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit exactly five checkpoints and return a non-promoting candidate manifest."""

    _require(len(checkpoint_paths) == 5, "exactly five checkpoints are required")
    resolved = [Path(path).resolve() for path in checkpoint_paths]
    _require(len(set(resolved)) == 5, "checkpoint paths must be distinct")
    amendment_path, model_root = _resolve_amp_amendment_path(
        resolved, amp_amendment_path
    )
    audit_path, audit = _load_and_validate_dataset_audit(dataset_audit_path)
    dataset_audit_sha256 = sha256_file(audit_path)
    validated = [
        _validate_checkpoint(
            path,
            audit=audit,
            dataset_audit_sha256=dataset_audit_sha256,
        )
        for path in resolved
    ]
    members = [member for member, _invariant in validated]
    invariants = [invariant for _member, invariant in validated]
    _require(
        len({sha256_json(invariant) for invariant in invariants}) == 1,
        "the five checkpoints do not share one training lock",
    )
    members.sort(key=lambda member: member["seed"])
    _require(
        tuple(member["seed"] for member in members) == FORMAL_TRAIN399_SEEDS,
        "the checkpoints do not contain the exact fixed five seeds",
    )
    _require(
        len({member["member_id"] for member in members}) == 5,
        "member IDs are not distinct",
    )
    _require(
        len({member["checkpoint_sha256"] for member in members}) == 5,
        "checkpoint files are not distinct",
    )
    _require(
        len({member["model_state_sha256"] for member in members}) == 5,
        "trained model state_dict values are not distinct",
    )
    _require(
        len({member["initial_complete_model_state_sha256"] for member in members}) == 5,
        "independently randomized decoder/head initial states are not distinct",
    )
    _require(
        len({member["initial_encoder_state_sha256"] for member in members}) == 1,
        "the five members do not share one verified ImageNet encoder initialization",
    )
    amendment_evidence, failure_receipt_path = _validate_amp_amendment(
        amendment_path,
        model_root=model_root,
        members=members,
    )
    training_lock = invariants[0]
    training_lock_identity = sha256_json(training_lock)
    for index, member in enumerate(members):
        member["member_index"] = index
    identity_payload = {
        "checkpoint_policy": TRAIN399_CHECKPOINT_POLICY,
        "ensemble_members": 5,
        "training_images": EXPECTED_TRAIN_IMAGES,
        "validation_images": EXPECTED_VALIDATION_IMAGES,
        "validation_labels_used_for_gradient_or_early_stopping": False,
        "blind_images_used": 0,
        "training_lock": training_lock,
        "training_lock_identity_sha256": training_lock_identity,
        "amp_backward_retry_amendment_lock": amendment_evidence,
        "operating_point_selection_contract": operating_point_selection_contract(),
        "members": members,
    }
    candidate_identity = sha256_json(identity_payload)
    source_by_seed = {
        member["seed"]: str(path)
        for path, (member, _invariant) in zip(resolved, validated, strict=True)
    }
    manifest: dict[str, Any] = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA,
        "status": CANDIDATE_STATUS,
        "candidate_only": True,
        "official_constants_modified": False,
        "official_model_contract_modified": False,
        "automatic_promotion_performed": False,
        "dataset_audit_path": str(audit_path),
        "dataset_audit_sha256": dataset_audit_sha256,
        "amp_backward_retry_amendment_path": str(amendment_path),
        "superseded_failure_receipt_path": str(failure_receipt_path),
        "source_checkpoint_paths_in_member_order": [
            source_by_seed[member["seed"]] for member in members
        ],
        "identity_payload": identity_payload,
        "candidate_bundle_identity_sha256": candidate_identity,
        "detection_model_metadata": _detection_metadata(
            identity_payload, candidate_identity
        ),
        "blind_images_used": 0,
    }
    manifest["candidate_manifest_identity_sha256"] = sha256_json(manifest)
    validate_candidate_manifest(manifest)
    return manifest


def validate_candidate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate a candidate manifest without requiring its source paths to exist."""

    _require(
        manifest.get("schema_version") == CANDIDATE_MANIFEST_SCHEMA,
        "unsupported candidate manifest schema",
    )
    _require(manifest.get("status") == CANDIDATE_STATUS, "candidate status is not locked")
    for field in (
        "candidate_only",
        "official_constants_modified",
        "official_model_contract_modified",
        "automatic_promotion_performed",
    ):
        expected = field == "candidate_only"
        _require(manifest.get(field) is expected, f"candidate promotion guard changed: {field}")
    _require(manifest.get("blind_images_used") == 0, "candidate manifest is blind-tainted")
    identity = manifest.get("identity_payload")
    _require(isinstance(identity, Mapping), "candidate identity payload is missing")
    candidate_identity = _require_sha256(
        manifest.get("candidate_bundle_identity_sha256"), "candidate bundle identity"
    )
    _require(
        sha256_json(identity) == candidate_identity,
        "candidate bundle identity does not match its logical content",
    )
    members = identity.get("members")
    _require(isinstance(members, list) and len(members) == 5, "candidate members are not five")
    _require(
        [member.get("member_index") for member in members] == list(range(5)),
        "candidate member order is not canonical",
    )
    _require(
        tuple(member.get("seed") for member in members) == FORMAL_TRAIN399_SEEDS,
        "candidate seed order changed",
    )
    for member in members:
        _require_sha256(
            member.get("training_receipt_sha256"),
            f"seed {member.get('seed')} completion receipt",
        )
        receipt_filename = member.get("training_receipt_filename")
        _require(
            isinstance(receipt_filename, str)
            and re.fullmatch(
                r"training_receipt(?:_resume_[0-9]{3})?\.json",
                receipt_filename,
            )
            is not None,
            f"seed {member.get('seed')} completion receipt filename is invalid",
        )
        _require(
            member.get("optimizer_name") == "AdamW"
            and isinstance(member.get("optimizer_parameter_state_count"), int)
            and int(member["optimizer_parameter_state_count"]) > 0
            and member.get("optimizer_parameter_step") == 23_940,
            f"seed {member.get('seed')} AdamW completion evidence is invalid",
        )
        _require(
            isinstance(member.get("amp_final_scale"), (int, float))
            and float(member["amp_final_scale"]) > 0.0
            and isinstance(member.get("amp_growth_tracker"), int)
            and int(member["amp_growth_tracker"]) >= 0
            and member.get("optimizer_steps_skipped_due_nonfinite_gradients") == 0,
            f"seed {member.get('seed')} AMP completion evidence is invalid",
        )
        retry_count = member.get("amp_backward_retry_count")
        _require(
            isinstance(retry_count, int)
            and not isinstance(retry_count, bool)
            and retry_count >= 0
            and member.get("amp_backward_retry_mode")
            == (
                "same_forward_graph_backoff"
                if retry_count
                else "legacy_or_amended_zero_retry"
            ),
            f"seed {member.get('seed')} AMP retry mode/count is invalid",
        )
        if retry_count == 0:
            _require(
                member.get("amp_final_scale") == 1024.0
                and member.get("amp_growth_tracker") == 23_940,
                f"seed {member.get('seed')} zero-retry scaler evidence drifted",
            )
    for field in ("checkpoint_sha256", "model_state_sha256"):
        hashes = [member.get(field) for member in members]
        _require(
            len(set(hashes)) == 5 and all(_is_sha256(value) for value in hashes),
            f"candidate {field} values are not five distinct hashes",
        )
    completion_receipt_hashes = [
        member.get("training_receipt_sha256") for member in members
    ]
    _require(
        len(set(completion_receipt_hashes)) == 5,
        "candidate completion receipts are not five distinct seed receipts",
    )
    _require(
        len({member.get("optimizer_parameter_state_count") for member in members}) == 1,
        "candidate AdamW state coverage differs across ensemble members",
    )
    training_lock = identity.get("training_lock")
    _require(isinstance(training_lock, Mapping), "candidate training lock is missing")
    _require(
        sha256_json(training_lock) == identity.get("training_lock_identity_sha256"),
        "candidate training lock identity is invalid",
    )
    amendment_lock = identity.get("amp_backward_retry_amendment_lock")
    _require(
        isinstance(amendment_lock, Mapping),
        "candidate AMP backward-retry amendment lock is missing",
    )
    _require(
        amendment_lock.get("schema_version") == AMP_AMENDMENT_SCHEMA
        and amendment_lock.get("status")
        == "applied_before_authoritative_seed3_optimizer_trajectory"
        and amendment_lock.get("failure_receipt_schema_version")
        == TRAINING_FAILURE_SCHEMA
        and amendment_lock.get("fixed_numeric_policy")
        == amp_backward_retry_policy_lock(),
        "candidate AMP backward-retry amendment policy is invalid",
    )
    for field in (
        "amendment_sha256",
        "failure_receipt_sha256",
        "unchanged_scientific_contract_sha256",
        "training_source_sha256",
    ):
        _require_sha256(amendment_lock.get(field), f"candidate AMP amendment {field}")
    _require(
        int(members[2].get("amp_backward_retry_count", 0)) >= 1,
        "candidate seed 2026082803 has no amended same-graph retry evidence",
    )
    _require_sha256(
        members[2].get("amp_backward_retry_audit_sha256"),
        "candidate seed 2026082803 AMP retry audit",
    )
    _require(
        identity.get("operating_point_selection_contract")
        == operating_point_selection_contract(),
        "candidate operating-point selection protocol changed",
    )
    expected_metadata = _detection_metadata(identity, candidate_identity)
    _require(
        manifest.get("detection_model_metadata") == expected_metadata,
        "candidate detection metadata differs from the gated identity",
    )
    _require(
        isinstance(manifest.get("amp_backward_retry_amendment_path"), str)
        and bool(manifest["amp_backward_retry_amendment_path"])
        and isinstance(manifest.get("superseded_failure_receipt_path"), str)
        and bool(manifest["superseded_failure_receipt_path"]),
        "candidate source AMP amendment paths are missing",
    )
    validate_train399_detection_model_metadata(expected_metadata, allow_pending=True)
    unsigned = deepcopy(dict(manifest))
    manifest_identity = unsigned.pop("candidate_manifest_identity_sha256", None)
    _require_sha256(manifest_identity, "candidate manifest identity")
    _require(
        sha256_json(unsigned) == manifest_identity,
        "candidate manifest identity does not match the complete receipt",
    )


def read_candidate_manifest(path: str | Path) -> dict[str, Any]:
    manifest = read_json(path)
    validate_candidate_manifest(manifest)
    return manifest


def detection_model_metadata_from_candidate_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validate_candidate_manifest(manifest)
    return deepcopy(dict(manifest["detection_model_metadata"]))


def write_candidate_manifest(
    path: str | Path, manifest: Mapping[str, Any], *, allow_identical_existing: bool = False
) -> None:
    """Atomically write once, or explicitly accept an already-identical receipt."""

    validate_candidate_manifest(manifest)
    destination = Path(path)
    if destination.exists():
        if allow_identical_existing and read_json(destination) == dict(manifest):
            return
        raise FileExistsError(f"refusing to overwrite candidate manifest: {destination}")
    atomic_write_json(destination, dict(manifest))
