from __future__ import annotations

from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from phaxis.contracts import ContractError, validate_stageb_detection_payload
from phaxis.hair_stageb.candidate_bundle import (
    CANDIDATE_STATUS,
    FORMAL_TRAIN399_SEEDS,
    PREREGISTERED_SCORE_THRESHOLDS,
    CandidateBundleError,
    amp_backward_retry_policy_lock,
    bind_selected_operating_point,
    build_candidate_manifest,
    validate_candidate_manifest,
    validate_train399_detection_model_metadata,
    write_candidate_manifest,
)
from phaxis.hair_stageb.candidate_pool_contract import (
    locked_biological_presence_candidate_decoder_contract,
)
from phaxis.hair_stageb.canonical_ground_truth import (
    CanonicalGroundTruthError,
    load_canonical_qcdev_ground_truth,
)
from phaxis.hair_stageb.evaluation_inference import (
    EVALUATION_ARTIFACT_ROLE,
    EVALUATION_DETECTION_SCHEMA,
    EvaluationInferenceError,
    build_evaluation_gate_binding,
    make_evaluation_detection_payload,
    make_evaluation_inference_summary,
    validate_evaluation_detection_payload,
    validate_evaluation_inference_summary,
)
from phaxis.fusion import fuse_hybrid_root_with_stageb_hairs
from phaxis.hair_stageb.runtime import StageBEnsemble
import phaxis.hair_stageb.runtime as stageb_runtime
from phaxis.hair_stageb.selection import (
    SelectionGateError,
    build_selection_receipt_from_paths,
    make_biological_candidate_pool_payload,
    read_selection_receipt,
    select_operating_point,
    validate_biological_candidate_pool_payload,
    validate_selected_operating_point_binding,
    write_selection_receipt_and_metadata,
)
from phaxis.hair_stageb.serialization import make_detection_payload
from phaxis.io import sha256_file, sha256_json


def _hash(character: str) -> str:
    return character * 64


def _adamw_state() -> dict:
    return {
        "state": {
            0: {
                "step": torch.tensor(23_940.0),
                "exp_avg": torch.zeros(2),
                "exp_avg_sq": torch.zeros(2),
            }
        },
        "param_groups": [
            {
                "lr": 1.0e-12,
                "betas": (0.9, 0.999),
                "eps": 1.0e-8,
                "weight_decay": 1.0e-4,
                "amsgrad": False,
                "maximize": False,
                "decoupled_weight_decay": True,
                "params": [0],
            }
        ],
    }


def _refresh_completion_receipt(checkpoint_path: Path) -> Path:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    seed = int(payload["seed"])
    suffix = "_resume_001" if seed == FORMAL_TRAIN399_SEEDS[0] else ""
    legacy = seed in FORMAL_TRAIN399_SEEDS[:2]
    directory = checkpoint_path.parent
    for old in directory.glob("training_receipt*.json"):
        old.unlink()
    retry_events = list(payload.get("amp_backward_retry_events", []))
    receipt = {
        "schema_version": "PHAxis-StageB-train399-training-receipt-1.0",
        "status": "completed",
        "formal_training": True,
        "seed": seed,
        "epochs": 60,
        "steps_per_epoch": 399,
        "global_steps": 23_940,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_evaluated_during_training": False,
        "blind_images_used": 0,
        "nvidia_smi_preflight_status": "passed",
        "nvidia_smi_training_monitor_status": "passed",
        "cuda_visible_devices": "1",
        "internal_device": "cuda:0",
        "physical_device_mapping_note": (
            "cuda:0 maps to the first entry of CUDA_VISIBLE_DEVICES"
        ),
        "gpu_name": "synthetic RTX 3090",
        "parameter_count": 2,
        "total_wall_seconds_this_invocation": 1.0,
        "median_epoch_wall_seconds": 1.0,
        "peak_allocated_mib": 1.0,
        "peak_reserved_mib": 1.0,
        "invocation_artifact_suffix": suffix,
    }
    if not legacy:
        retry_audit_path = directory / f"amp_backward_retries{suffix}.json"
        retry_audit = {
            "schema_version": "PHAxis-StageB-AMP-backward-retry-audit-1.0",
            "status": "completed_through_epoch",
            "seed": seed,
            "completed_epoch": 60,
            "event_count": len(retry_events),
            "events": retry_events,
            "same_forward_graph_replayed": True,
            "optimizer_steps_skipped_due_nonfinite_gradients": 0,
            "blind_images_used": 0,
        }
        retry_audit_path.write_text(
            json.dumps(retry_audit, sort_keys=True), encoding="utf-8"
        )
        receipt.update(
            {
                "amp_backward_retry_count": len(retry_events),
                "amp_min_scale": min(
                    [1024.0]
                    + [float(event["scale_after_backoff"]) for event in retry_events]
                ),
                "amp_final_scale": payload["scaler"]["scale"],
                "amp_backward_retry_audit": str(retry_audit_path.resolve()),
                "amp_backward_retry_audit_sha256": sha256_file(retry_audit_path),
                "optimizer_steps_skipped_due_nonfinite_gradients": 0,
            }
        )
    receipt_path = directory / f"training_receipt{suffix}.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return receipt_path


def _write_fixture(tmp_path: Path) -> tuple[list[Path], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifests = tmp_path / "manifests"
    annotations = tmp_path / "annotations" / "rhaxis_canonical"
    images = tmp_path / "images"
    manifests.mkdir()
    annotations.mkdir(parents=True)
    images.mkdir()
    split_manifest = tmp_path / "split_manifest.csv"
    train_ids = [f"T{index:03d}" for index in range(399)]
    val_ids = [f"V{index:03d}" for index in range(44)]
    split_rows = [
        {
            "task_id": task_id,
            "split": "train" if task_id.startswith("T") else "val",
            "family_key": f"family-{task_id}",
        }
        for task_id in (*train_ids, *val_ids)
    ]

    def write_split(path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("task_id", "split", "family_key")
            )
            writer.writeheader()
            writer.writerows(split_rows)

    write_split(split_manifest)
    write_split(manifests / "split_manifest.csv")
    (tmp_path / "verification_report.json").write_text(
        json.dumps({"status": "passed", "blind_images_used": 0}),
        encoding="utf-8",
    )

    roles = (
        "image",
        "raw_annotation",
        "canonical_annotation",
        "labelme_annotation",
        "root_mask",
        "root_cap_mask",
        "root_hair_centerline",
        "root_tip_keypoint",
        "scale_bar_centerline",
    )
    dataset_rows = []
    integrity_rows = []
    for task_id in (*train_ids, *val_ids):
        annotation_relpath = f"annotations/rhaxis_canonical/{task_id}.json"
        annotation_path = tmp_path / annotation_relpath
        image_path = images / f"{task_id}.tif"
        image_path.write_bytes(f"immutable-{task_id}".encode())
        image_sha = sha256_file(image_path)
        raw_sha = sha256_json([task_id, "raw_annotation"])
        annotation = {
            "schema_version": "RHAxis-human-curated-vector-1.0",
            "dataset_version": "synthetic-HumanCurated443",
            "task_id": task_id,
            "image": {
                "file_name": f"{task_id}.tif",
                "width": 127,
                "height": 63,
                "sha256": raw_sha,
            },
            "calibration": {"source_um_per_px": 1.0},
            "shapes": [
                {
                    "label": "root",
                    "shape_type": "polygon",
                    "points": [[0.0, 0.0], [8.0, 0.0], [8.0, 63.0], [0.0, 63.0]],
                },
                {
                    "label": "root_hair",
                    "shape_type": "polyline",
                    "instance_id": "H001",
                    "points": [[10.0, 10.0], [20.0, 10.0]],
                },
            ],
        }
        annotation_path.write_text(
            json.dumps(annotation, sort_keys=True), encoding="utf-8"
        )
        dataset_rows.append(
            {
                "task_id": task_id,
                "dataset_version": "synthetic-HumanCurated443",
                "family_key": f"family-{task_id}",
                "image_relpath": f"images/{task_id}.tif",
                "canonical_annotation_relpath": annotation_relpath,
                "image_sha256": image_sha,
                "raw_annotation_sha256": raw_sha,
                "source_um_per_px": "1.0",
                "image_width": "127",
                "image_height": "63",
                "root_hair_count": "1",
                "model_prediction_shapes_retained": "0",
            }
        )
        for role in roles:
            if role == "canonical_annotation":
                relative_path = annotation_relpath
                digest = sha256_file(annotation_path)
                size = annotation_path.stat().st_size
            elif role == "image":
                relative_path = f"images/{task_id}.tif"
                digest, size = image_sha, image_path.stat().st_size
            elif role == "raw_annotation":
                relative_path = f"annotations/raw/{task_id}.json"
                digest, size = raw_sha, 1
            else:
                relative_path = f"synthetic/{role}/{task_id}"
                digest, size = sha256_json([task_id, role]), 1
            integrity_rows.append(
                {
                    "task_id": task_id,
                    "role": role,
                    "relative_path": relative_path,
                    "sha256": digest,
                    "size_bytes": size,
                }
            )

    dataset_manifest = manifests / "dataset_manifest.csv"
    with dataset_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(dataset_rows[0]),
        )
        writer.writeheader()
        writer.writerows(dataset_rows)
    integrity_manifest = manifests / "integrity_sha256.csv"
    with integrity_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(integrity_rows[0]))
        writer.writeheader()
        writer.writerows(integrity_rows)
    split_lock = tmp_path / "split_lock.json"
    split_lock.write_text('{"locked":true}\n', encoding="utf-8")
    audit = {
        "schema_version": "PHAxis-StageB-train399-dataset-audit-1.0",
        "status": "passed",
        "dataset_root": str(tmp_path),
        "train_records": 399,
        "excluded_val_records": 44,
        "train_ids": train_ids,
        "train_ids_sha256": sha256_json(train_ids),
        "excluded_val_ids": val_ids,
        "excluded_val_ids_sha256": sha256_json(val_ids),
        "family_key_overlap": [],
        "dataset_split_identity_sha256": _hash("1"),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "split_lock_sha256": sha256_file(split_lock),
        "integrity_manifest_sha256": sha256_file(integrity_manifest),
        "train_task_family_sha256": _hash("4"),
        "train_families_sha256": _hash("5"),
        "excluded_val_families_sha256": _hash("6"),
        "selected_split_manifest": str(split_manifest),
        "selected_split_lock": str(split_lock),
        "validation_labels_used_for_gradient": False,
        "validation_labels_used_for_early_stopping": False,
        "blind_images_used": 0,
        "pyRootHair_called_or_copied": False,
    }
    audit_path = tmp_path / "dataset_audit.json"
    audit_path.write_text(json.dumps(audit, sort_keys=True), encoding="utf-8")
    audit_sha256 = sha256_file(audit_path)

    imagenet_weight = tmp_path / "model.safetensors"
    imagenet_weight.write_bytes(b"synthetic immutable ImageNet source")
    imagenet_sha256 = sha256_file(imagenet_weight)
    config = {
        "epochs": 60,
        "fixed_last_epoch_policy": True,
        "crops_per_image": 8,
        "batch_size": 8,
        "amp": True,
        "amp_initial_scale": 1024.0,
        "amp_growth_interval": 1_000_000,
        "amp_growth_factor": 2.0,
        "amp_backoff_factor": 0.5,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "imagenet_source": "timm/resnet34.a1_in1k",
        "encoder": "resnet34",
        "in_channels": 3,
        "out_stride": 2,
        "decoder_channels": (256, 128, 96, 64),
        "context": True,
        "stem_stride1": False,
    }
    checkpoints: list[Path] = []
    initializations: dict[int, dict] = {}
    for index, seed in enumerate(FORMAL_TRAIN399_SEEDS):
        initialization = {
            "source": "timm/resnet34.a1_in1k",
            "huggingface_revision": "0" * 40,
            "cached_weight_path": str(imagenet_weight),
            "cached_weight_sha256": imagenet_sha256,
            "cached_weight_size_bytes": imagenet_weight.stat().st_size,
            "initial_encoder_state_sha256": _hash("a"),
            "initial_complete_model_state_sha256": f"{index + 10:064x}",
            "historical_stageb_checkpoint_loaded": False,
        }
        initialization["initialization_sha256"] = sha256_json(initialization)
        initializations[seed] = initialization
        contract = {
            "formal_training": True,
            "training_policy": "all399_fixed_60_epoch_last_checkpoint",
            "model_selection_policy": "none_during_training",
            "initialization_policy": "ImageNet encoder plus newly randomized decoder/heads",
            "prohibited_initialization": (
                "all RHAxiscc 443-fold checkpoints and any state exposed to locked val44"
            ),
            "seed": seed,
            "member_id": f"seed_{seed}",
            "training_images": 399,
            "validation_images": 44,
            "train_ids": train_ids,
            "excluded_val_ids": val_ids,
            "dataset_audit_sha256": audit_sha256,
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
            "cache_identity_sha256": _hash("7"),
            "config_sha256": sha256_json(config),
            "amp_policy": {
                "dtype": "float16",
                "enabled": True,
                "initial_scale": 1024.0,
                "growth_interval": 1_000_000,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "nonfinite_step_policy": "fail_closed_no_optimizer_step_skip",
            },
            "validation_labels_used_for_gradient": False,
            "validation_labels_used_for_early_stopping": False,
            "validation_labels_used_for_gradient_or_early_stopping": False,
            "validation_metrics_observed_during_training": False,
            "blind_images_used": 0,
            "pyRootHair_called_or_copied": False,
        }
        retry_events = (
            [
                {
                    "epoch": 1,
                    "global_step": 6,
                    "retry_index": 1,
                    "scale_before_backoff": 1024.0,
                    "scale_after_backoff": 512.0,
                    "nonfinite_parameters": [
                        "heads.base_hm.body.1.weight[nonfinite=20/64]"
                    ],
                    "optimizer_step_skipped": False,
                    "same_forward_graph_replayed": True,
                }
            ]
            if seed == 2026082803
            else []
        )
        payload = {
            "schema_version": "PHAxis-StageB-train399-checkpoint-1.0",
            "model": {
                "weight": torch.tensor([float(index), float(seed % 97)]),
                "counter": torch.tensor(index, dtype=torch.int64),
            },
            "optimizer": _adamw_state(),
            "scaler": {
                "scale": 512.0 if retry_events else 1024.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 1_000_000,
                "_growth_tracker": 23_934 if retry_events else 23_940,
            },
            "cfg": config,
            "contract": contract,
            "initialization": initialization,
            "initialization_sha256": initialization["initialization_sha256"],
            "seed": seed,
            "member_id": f"seed_{seed}",
            "training_images": 399,
            "training_task_ids_sha256": audit["train_ids_sha256"],
            "split_manifest_sha256": audit["split_manifest_sha256"],
            "validation_images": 44,
            "validation_labels_used_for_gradient_or_early_stopping": False,
            "epoch": 60,
            "global_step": 23940,
            "rng": {},
        }
        if retry_events:
            payload["amp_backward_retry_events"] = retry_events
        seed_dir = tmp_path / f"seed_{seed}"
        seed_dir.mkdir()
        history = [
            {
                "epoch": epoch,
                "batches": 399,
                "global_step": epoch * 399,
                "validation_evaluated": False,
                "amp_backward_retry_count": (
                    1 if seed == 2026082803 and epoch == 1 else 0
                ),
                "optimizer_steps_skipped_due_nonfinite_gradients": 0,
            }
            for epoch in range(1, 61)
        ]
        payload["history"] = history
        for sidecar_name, sidecar_payload in (
            ("history.json", history),
            ("config.json", config),
            ("training_contract.json", contract),
            ("initialization.json", initialization),
        ):
            (seed_dir / sidecar_name).write_text(
                json.dumps(sidecar_payload, sort_keys=True), encoding="utf-8"
            )
        path = seed_dir / "last.pt"
        torch.save(payload, path)
        _refresh_completion_receipt(path)
        checkpoints.append(path)

    failure_dir = (
        tmp_path
        / "failed_attempts"
        / "seed_2026082803_amp_overflow_20260829T013208Z"
    )
    failure_dir.mkdir(parents=True)
    failed_groups = ["heads.base_hm.body.1.weight[nonfinite=20/64]"]
    failure = {
        "schema_version": "PHAxis-StageB-training-failure-1.0",
        "status": "failed",
        "completed_epoch": 0,
        "global_step": 6,
        "exception_type": "builtins.FloatingPointError",
        "exception_message": "non-finite gradients: " + ", ".join(failed_groups),
        "exception_swallowed": False,
        "nvidia_smi_preflight_status": "passed",
        "last_finite_loss_total": 39.0,
        "blind_images_used": 0,
    }
    failure_path = failure_dir / "training_failure.json"
    failure_path.write_text(json.dumps(failure, sort_keys=True), encoding="utf-8")
    synthetic_training_source = tmp_path / "synthetic_training.py"
    synthetic_training_source.write_text("# immutable synthetic source\n", encoding="utf-8")
    seed3_initialization = initializations[2026082803]
    amendment = {
        "schema_version": "PHAxis-StageB-train399-AMP-backward-amendment-1.0",
        "status": "applied_before_authoritative_seed3_optimizer_trajectory",
        "superseded_failed_attempt": {
            "seed": 2026082803,
            "completed_epoch": 0,
            "global_step_at_failure": 6,
            "failure_receipt": str(failure_path.relative_to(tmp_path)),
            "failure_receipt_sha256": sha256_file(failure_path),
            "authoritative_checkpoint_created": False,
            "blind_images_used": 0,
        },
        "root_cause": {
            "failure_class": "fp16_scaled_backward_overflow",
            "loss_was_finite": True,
            "loss_total": 39.0,
            "nonfinite_parameter_groups": failed_groups,
            "oom": False,
            "data_or_target_nonfinite": False,
            "gpu_preflight_passed": True,
        },
        "amended_numeric_policy": amp_backward_retry_policy_lock(),
        "unchanged_scientific_contract": {
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
        },
        "legacy_zero_retry_normalization": {
            "seeds": [2026082801, 2026082802],
            "completed_epochs": [60, 60],
            "completed_global_steps": [23_940, 23_940],
            "final_scaler_scale": [1024.0, 1024.0],
            "scaler_growth_tracker": [23_940, 23_940],
            "retry_event_field_present": [False, False],
            "normalized_amp_backward_retry_count": [0, 0],
        },
        "authoritative_seed3_restart": {
            "physical_gpu": 1,
            "cuda_visible_devices": "1",
            "internal_device": "cuda:0",
            "launch_mode": "fresh",
            "failed_and_restart_initialization_file_sha256": _hash("d"),
            "initialization_identity_sha256": seed3_initialization[
                "initialization_sha256"
            ],
            "initial_complete_model_state_sha256": seed3_initialization[
                "initial_complete_model_state_sha256"
            ],
            "deterministic_initialization_reproduced": True,
        },
        "implementation": {
            "training_source": synthetic_training_source.name,
            "training_source_sha256": sha256_file(synthetic_training_source),
            "finite_path_gradient_identity_tested": True,
            "same_graph_backoff_tested": True,
            "retry_exhaustion_fail_closed_tested": True,
            "candidate_retry_tamper_gate_tested": True,
        },
    }
    (tmp_path / "AMP_BACKWARD_RETRY_AMENDMENT_TEST.json").write_text(
        json.dumps(amendment, sort_keys=True), encoding="utf-8"
    )
    return checkpoints, audit_path


def _prediction() -> dict:
    return {
        "base": np.asarray([[1.0, 2.0]], np.float32),
        "tip": np.asarray([[3.0, 4.0]], np.float32),
        "score": np.asarray([0.75], np.float32),
        "length_um": np.asarray([12.5], np.float32),
        "working_shape": [32, 64],
        "source_shape": [64, 128],
        "source_to_working_scale": 0.5,
        "source_to_working_scale_xy": [0.5, 0.5],
        "realized_um_per_px_xy": [2.0, 2.0],
        "tip_snapped": np.asarray([True]),
        "length_semantics": "regressed_polyline_arc_length_um_diagnostic_only",
    }


def _biological_pool_prediction() -> dict:
    scale_xy = np.asarray([64 / 127, 32 / 63], dtype=np.float64)
    return {
        "base": np.asarray([[10.0, 10.0] * scale_xy, [50.0, 50.0]], np.float32),
        "tip": np.asarray([[20.0, 10.0] * scale_xy, [60.0, 50.0]], np.float32),
        "score": np.asarray([0.25, 0.15], np.float32),
        "presence_proxy_valid": np.asarray([True, True]),
        "n": 2,
        "score_floor": 0.10,
        "candidate_pool_decode_scope": (
            "base_score_plus_straight_base_to_tip_biological_presence_proxy"
        ),
        "presence_proxy_kind": "straight_base_to_tip",
        "distal_endpoint_or_length_used_as_selection_gate": False,
        "candidate_decoder_contract": (
            locked_biological_presence_candidate_decoder_contract()
        ),
        "network_forward_passes": 1,
        "working_shape": [32, 64],
        "source_shape": [63, 127],
        "source_to_working_scale": 0.5,
        "source_to_working_scale_xy": scale_xy.tolist(),
        "realized_um_per_px_xy": (1.0 / scale_xy).tolist(),
    }


def _evaluation_prediction() -> dict:
    prediction = _prediction()
    scale_xy = np.asarray([64 / 127, 32 / 63], dtype=np.float64)
    prediction.update(
        {
            "source_shape": [63, 127],
            "source_to_working_scale_xy": scale_xy.tolist(),
            "realized_um_per_px_xy": (1.0 / scale_xy).tolist(),
        }
    )
    return prediction


def _write_selection_fixture(
    tmp_path: Path,
) -> tuple[list[Path], dict, Path, dict, Path]:
    checkpoints, audit_path = _write_fixture(tmp_path)
    manifest = build_candidate_manifest(checkpoints, dataset_audit_path=audit_path)
    manifest_path = tmp_path / "candidate_manifest.json"
    write_candidate_manifest(manifest_path, manifest)
    pending = manifest["detection_model_metadata"]
    pool = tmp_path / "candidate_pool" / "candidate_pools"
    for index in range(44):
        task_id = f"V{index:03d}"
        payload = make_biological_candidate_pool_payload(
            task_id=task_id,
            source_image_sha256=sha256_file(tmp_path / "images" / f"{task_id}.tif"),
            source_um_per_px=1.0,
            prediction=_biological_pool_prediction(),
            pending_model_metadata=pending,
        )
        from phaxis.io import atomic_write_json

        atomic_write_json(pool / f"{task_id}.json", payload)
    receipt, returned_pending = build_selection_receipt_from_paths(
        candidate_manifest_path=manifest_path,
        candidate_pool_dir=pool.parent,
        dataset_root=tmp_path,
        dataset_manifest=tmp_path / "manifests" / "dataset_manifest.csv",
        split_manifest=tmp_path / "split_manifest.csv",
    )
    assert returned_pending == pending
    receipt_path = tmp_path / "selection_receipt.json"
    metadata_path = tmp_path / "selected_model_metadata.json"
    selected = write_selection_receipt_and_metadata(
        receipt=receipt,
        pending_model_metadata=pending,
        receipt_path=receipt_path,
        selected_model_metadata_path=metadata_path,
    )
    return checkpoints, manifest, receipt_path, selected, pool.parent


def test_candidate_gate_builds_deterministic_non_promoting_receipt(tmp_path: Path) -> None:
    checkpoints, audit = _write_fixture(tmp_path)
    manifest = build_candidate_manifest(checkpoints[::-1], dataset_audit_path=audit)
    validate_candidate_manifest(manifest)
    assert manifest["status"] == CANDIDATE_STATUS
    assert manifest["candidate_only"] is True
    assert manifest["automatic_promotion_performed"] is False
    assert [member["seed"] for member in manifest["identity_payload"]["members"]] == list(
        FORMAL_TRAIN399_SEEDS
    )
    assert len(set(
        member["model_state_sha256"]
        for member in manifest["identity_payload"]["members"]
    )) == 5
    assert {
        member["amp_backward_retry_count"]
        for member in manifest["identity_payload"]["members"]
    } == {0, 1}
    assert {
        member["amp_backward_retry_mode"]
        for member in manifest["identity_payload"]["members"]
    } == {"legacy_or_amended_zero_retry", "same_forward_graph_backoff"}
    members = manifest["identity_payload"]["members"]
    assert members[0]["training_receipt_filename"] == (
        "training_receipt_resume_001.json"
    )
    amendment_lock = manifest["identity_payload"][
        "amp_backward_retry_amendment_lock"
    ]
    assert amendment_lock["fixed_numeric_policy"] == amp_backward_retry_policy_lock()
    assert len(amendment_lock["amendment_sha256"]) == 64
    assert len(amendment_lock["failure_receipt_sha256"]) == 64
    selection = manifest["identity_payload"]["operating_point_selection_contract"]
    assert selection["network_forward_passes_per_image"] == 1
    assert selection["candidate_pool_score_floor"] == 0.10
    assert selection["threshold_grid"] == list(PREREGISTERED_SCORE_THRESHOLDS)
    with pytest.raises(CandidateBundleError, match="pending selection"):
        validate_train399_detection_model_metadata(manifest["detection_model_metadata"])

    output = tmp_path / "candidate_manifest.json"
    write_candidate_manifest(output, manifest)
    write_candidate_manifest(output, manifest, allow_identical_existing=True)
    with pytest.raises(FileExistsError):
        write_candidate_manifest(output, manifest)


def test_candidate_gate_normalizes_and_audits_same_graph_amp_retry(
    tmp_path: Path,
) -> None:
    checkpoints, audit = _write_fixture(tmp_path)
    target = checkpoints[2]
    manifest = build_candidate_manifest(checkpoints, dataset_audit_path=audit)
    member = next(
        row
        for row in manifest["identity_payload"]["members"]
        if row["seed"] == 2026082803
    )
    assert member["amp_backward_retry_count"] == 1
    assert member["amp_backward_retry_mode"] == "same_forward_graph_backoff"
    assert member["optimizer_steps_skipped_due_nonfinite_gradients"] == 0
    assert member["amp_final_scale"] == 512.0

    payload = torch.load(target, map_location="cpu", weights_only=True)
    payload["amp_backward_retry_events"][0]["optimizer_step_skipped"] = True
    torch.save(payload, target)
    with pytest.raises(CandidateBundleError, match="batch/step contract"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)


def test_gate_rejects_audit_hash_drift_and_duplicate_model_state(tmp_path: Path) -> None:
    checkpoints, audit = _write_fixture(tmp_path)
    payload = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    payload["contract"]["dataset_audit_sha256"] = _hash("f")
    torch.save(payload, checkpoints[0])
    with pytest.raises(CandidateBundleError, match="dataset_audit_sha256"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)

    checkpoints, audit = _write_fixture(tmp_path / "second")
    first = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    second = torch.load(checkpoints[1], map_location="cpu", weights_only=True)
    second["model"] = first["model"]
    torch.save(second, checkpoints[1])
    _refresh_completion_receipt(checkpoints[1])
    with pytest.raises(CandidateBundleError, match="state_dict values are not distinct"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)


def test_gate_rejects_inconsistent_optional_history_sidecar(tmp_path: Path) -> None:
    checkpoints, audit = _write_fixture(tmp_path)
    history = [
        {
            "epoch": epoch,
            "batches": 399,
            "global_step": epoch * 399,
            "validation_evaluated": epoch == 17,
        }
        for epoch in range(1, 61)
    ]
    (checkpoints[0].parent / "history.json").write_text(
        json.dumps(history), encoding="utf-8"
    )
    with pytest.raises(CandidateBundleError, match="history contract mismatch at epoch 17"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)


def test_gate_requires_all_four_training_sidecars(tmp_path: Path) -> None:
    checkpoints, audit = _write_fixture(tmp_path)
    directory = checkpoints[0].parent
    for filename in (
        "history.json",
        "config.json",
        "training_contract.json",
        "initialization.json",
    ):
        path = directory / filename
        original = path.read_bytes()
        path.unlink()
        with pytest.raises(CandidateBundleError, match=rf"required {filename} sidecar"):
            build_candidate_manifest(checkpoints, dataset_audit_path=audit)
        path.write_bytes(original)


def test_gate_requires_one_hash_and_field_exact_completion_receipt(
    tmp_path: Path,
) -> None:
    checkpoints, audit = _write_fixture(tmp_path)
    target = checkpoints[0]
    receipt_path = next(target.parent.glob("training_receipt*.json"))
    original_bytes = receipt_path.read_bytes()

    receipt_path.unlink()
    with pytest.raises(CandidateBundleError, match="exactly one completion receipt, found 0"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)
    receipt_path.write_bytes(original_bytes)

    duplicate = target.parent / "training_receipt_resume_002.json"
    duplicate.write_bytes(original_bytes)
    with pytest.raises(CandidateBundleError, match="exactly one completion receipt, found 2"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)
    duplicate.unlink()

    receipt = json.loads(original_bytes)
    receipt["checkpoint_sha256"] = _hash("f")
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(CandidateBundleError, match="mismatch for checkpoint_sha256"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)

    receipt = json.loads(original_bytes)
    receipt["seed"] = 1
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(CandidateBundleError, match="mismatch for seed"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)


def test_gate_requires_complete_scaler_and_fixed_horizon_adamw_state(
    tmp_path: Path,
) -> None:
    checkpoints, audit = _write_fixture(tmp_path)
    target = checkpoints[3]
    original = torch.load(target, map_location="cpu", weights_only=True)

    payload = deepcopy(original)
    del payload["scaler"]["_growth_tracker"]
    torch.save(payload, target)
    with pytest.raises(CandidateBundleError, match="scaler state is incomplete"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)

    payload = deepcopy(original)
    payload["scaler"]["_growth_tracker"] = 23_939
    torch.save(payload, target)
    with pytest.raises(CandidateBundleError, match="growth tracker differs"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)

    payload = deepcopy(original)
    payload["optimizer"]["state"] = {}
    torch.save(payload, target)
    with pytest.raises(CandidateBundleError, match="per-parameter state is empty"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)

    payload = deepcopy(original)
    payload["optimizer"]["state"][0]["step"] = torch.tensor(23_939.0)
    torch.save(payload, target)
    with pytest.raises(CandidateBundleError, match="is not 23940"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)

    payload = deepcopy(original)
    payload["optimizer"]["param_groups"][0]["params"] = [0, 1]
    torch.save(payload, target)
    with pytest.raises(CandidateBundleError, match="does not cover exactly"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)


def test_gate_validates_explicit_amendment_and_referenced_failure_receipt(
    tmp_path: Path,
) -> None:
    checkpoints, audit = _write_fixture(tmp_path)
    amendment_path = tmp_path / "AMP_BACKWARD_RETRY_AMENDMENT_TEST.json"
    inferred = build_candidate_manifest(checkpoints, dataset_audit_path=audit)
    explicit = build_candidate_manifest(
        checkpoints,
        dataset_audit_path=audit,
        amp_amendment_path=amendment_path,
    )
    assert explicit == inferred

    amendment_bytes = amendment_path.read_bytes()
    amendment = json.loads(amendment_bytes)
    failure_path = tmp_path / amendment["superseded_failed_attempt"]["failure_receipt"]
    failure_bytes = failure_path.read_bytes()
    failure = json.loads(failure_bytes)
    failure["global_step"] = 7
    failure_path.write_text(json.dumps(failure, sort_keys=True), encoding="utf-8")
    with pytest.raises(CandidateBundleError, match="failure receipt hash mismatch"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)
    failure_path.write_bytes(failure_bytes)

    amendment["amended_numeric_policy"]["maximum_backward_retries_per_batch"] = 15
    amendment_path.write_text(json.dumps(amendment, sort_keys=True), encoding="utf-8")
    with pytest.raises(CandidateBundleError, match="amendment policy drifted"):
        build_candidate_manifest(checkpoints, dataset_audit_path=audit)
    amendment_path.write_bytes(amendment_bytes)

    tampered = deepcopy(inferred)
    tampered["identity_payload"]["amp_backward_retry_amendment_lock"][
        "fixed_numeric_policy"
    ]["maximum_backward_retries_per_batch"] = 15
    candidate_identity = sha256_json(tampered["identity_payload"])
    tampered["candidate_bundle_identity_sha256"] = candidate_identity
    tampered["detection_model_metadata"][
        "candidate_bundle_identity_sha256"
    ] = candidate_identity
    unsigned = deepcopy(tampered)
    unsigned.pop("candidate_manifest_identity_sha256")
    tampered["candidate_manifest_identity_sha256"] = sha256_json(unsigned)
    with pytest.raises(CandidateBundleError, match="amendment policy is invalid"):
        validate_candidate_manifest(tampered)


def test_selected_metadata_binds_payload_threshold_and_receipts(tmp_path: Path) -> None:
    checkpoints, audit = _write_fixture(tmp_path)
    manifest = build_candidate_manifest(checkpoints, dataset_audit_path=audit)
    selected = bind_selected_operating_point(
        manifest["detection_model_metadata"],
        selected_score_threshold=0.225,
        selection_receipt_sha256=_hash("b"),
        selection_receipt_identity_sha256=_hash("f"),
        candidate_pool_identity_sha256=_hash("c"),
    )
    payload = make_detection_payload(
        task_id="T1",
        source_image_sha256=_hash("d"),
        source_um_per_px=1.0,
        prediction=_prediction(),
        precision_mode="fp32_locked",
        model_metadata=selected,
        score_threshold=0.225,
    )
    assert payload["model"]["selection_receipt_sha256"] == _hash("b")
    assert payload["operating_point"]["score_threshold"] == 0.225
    validate_stageb_detection_payload(
        payload,
        expected_task_id="T1",
        expected_image_sha256=_hash("d"),
        expected_model_metadata=selected,
    )
    wrong_geometry = deepcopy(payload)
    wrong_geometry["coordinate_space"]["source_to_working_scale_xy"][0] = 0.49
    unsigned = deepcopy(wrong_geometry)
    unsigned.pop("detection_identity_sha256")
    wrong_geometry["detection_identity_sha256"] = sha256_json(unsigned)
    with pytest.raises(ContractError, match="differs from source/working shapes"):
        validate_stageb_detection_payload(
            wrong_geometry,
            expected_task_id="T1",
            expected_image_sha256=_hash("d"),
        )
    with pytest.raises(ValueError, match="differs from selected"):
        make_detection_payload(
            task_id="T1",
            source_image_sha256=_hash("d"),
            source_um_per_px=1.0,
            prediction=_prediction(),
            precision_mode="fp32_locked",
            model_metadata=selected,
            score_threshold=0.200,
        )
    tampered = deepcopy(payload)
    tampered["model"]["selection_receipt_sha256"] = None
    unsigned = deepcopy(tampered)
    unsigned.pop("detection_identity_sha256")
    tampered["detection_identity_sha256"] = sha256_json(unsigned)
    with pytest.raises(ContractError, match="selection receipt"):
        validate_stageb_detection_payload(
            tampered,
            expected_task_id="T1",
            expected_image_sha256=_hash("d"),
        )


def test_legacy_serialization_remains_locked_to_original_policy() -> None:
    payload = make_detection_payload(
        task_id="LEGACY",
        source_image_sha256=_hash("e"),
        source_um_per_px=1.0,
        prediction=_prediction(),
        precision_mode="fp32_locked",
    )
    assert payload["model"] == {
        "expert_id": "RHAxiscc-StageB-5fold-last-e60-hflip-20260827",
        "ensemble_members": 5,
        "checkpoint_policy": "five_fold_last_epoch_60",
        "precision_mode": "fp32_locked",
    }
    assert payload["operating_point"]["score_threshold"] == 0.225
    validate_stageb_detection_payload(
        payload,
        expected_task_id="LEGACY",
        expected_image_sha256=_hash("e"),
    )
    with pytest.raises(ValueError, match="legacy 443CV"):
        make_detection_payload(
            task_id="LEGACY",
            source_image_sha256=_hash("e"),
            source_um_per_px=1.0,
            prediction=_prediction(),
            precision_mode="fp32_locked",
            score_threshold=0.20,
        )


def test_qcdev_selection_uses_physical_um_and_exact_preregistered_tiebreak(
    tmp_path: Path,
) -> None:
    _checkpoints, manifest, receipt_path, selected, _pool = _write_selection_fixture(
        tmp_path
    )
    receipt = read_selection_receipt(receipt_path)
    assert receipt["metric_coordinate_space"] == (
        "physical_um_xy_after_axis_specific_realized_resize_conversion"
    )
    # 0.175, 0.200, 0.225 and 0.250 are equally perfect; the exact final
    # preregistered tie-break selects the higher threshold.
    assert receipt["selected"]["threshold"] == 0.25
    assert receipt["selected"]["tolerant_biological_presence_20um"]["f1"] == 1.0
    assert receipt["selected"]["identity_attachment_proxy_20um"]["f1"] == 1.0
    assert receipt["selected"]["count_mae"] == 0.0
    assert selected["selected_score_threshold"] == 0.25
    decoder_contract = locked_biological_presence_candidate_decoder_contract()
    assert receipt["candidate_decoder_contract"] == decoder_contract
    assert receipt["candidate_decoder_contract_sha256"] == sha256_json(
        decoder_contract
    )
    first_lock = receipt["task_image_locks"][0]
    assert first_lock["source_image_shape_hw"] == [63, 127]
    assert first_lock["source_um_per_px"] == 1.0
    assert first_lock["source_image_sha256"] == sha256_file(
        tmp_path / "images" / "V000.tif"
    )
    assert len(first_lock["root_polygon_source_geometry_identity_sha256"]) == 64
    assert len(first_lock["root_hair_instance_id_order_sha256"]) == 64
    validate_selected_operating_point_binding(
        candidate_manifest=manifest,
        selected_model_metadata=selected,
        selection_receipt=receipt,
        selection_receipt_file_sha256=sha256_file(receipt_path),
    )

    tampered = deepcopy(receipt)
    tampered["threshold_metrics"][0]["per_image"][0]["predicted"] += 1
    tampered["selection_receipt_identity_sha256"] = sha256_json(
        {
            key: value
            for key, value in tampered.items()
            if key != "selection_receipt_identity_sha256"
        }
    )
    with pytest.raises(SelectionGateError, match="count error"):
        from phaxis.hair_stageb.selection import validate_selection_receipt

        validate_selection_receipt(tampered)


def test_candidate_pool_and_selection_receipt_fail_closed_on_decoder_drift(
    tmp_path: Path,
) -> None:
    checkpoints, audit = _write_fixture(tmp_path)
    manifest = build_candidate_manifest(checkpoints, dataset_audit_path=audit)
    pending = manifest["detection_model_metadata"]
    prediction = _biological_pool_prediction()
    prediction["candidate_decoder_contract"] = deepcopy(
        prediction["candidate_decoder_contract"]
    )
    prediction["candidate_decoder_contract"]["tip_score_floor"] = 0.25
    with pytest.raises(SelectionGateError, match="decoder parameters"):
        make_biological_candidate_pool_payload(
            task_id="V000",
            source_image_sha256=sha256_file(tmp_path / "images" / "V000.tif"),
            source_um_per_px=1.0,
            prediction=prediction,
            pending_model_metadata=pending,
        )

    _checkpoints, _manifest, receipt_path, _selected, pool_dir = (
        _write_selection_fixture(tmp_path / "sealed")
    )
    receipt = read_selection_receipt(receipt_path)
    changed_receipt = deepcopy(receipt)
    changed_receipt["candidate_decoder_contract"]["tip_snap_radius_um"] = 31.0
    changed_receipt["candidate_decoder_contract_sha256"] = sha256_json(
        changed_receipt["candidate_decoder_contract"]
    )
    unsigned_receipt = deepcopy(changed_receipt)
    unsigned_receipt.pop("selection_receipt_identity_sha256")
    changed_receipt["selection_receipt_identity_sha256"] = sha256_json(
        unsigned_receipt
    )
    from phaxis.hair_stageb.selection import validate_selection_receipt

    with pytest.raises(SelectionGateError, match="decoder contract changed"):
        validate_selection_receipt(changed_receipt)

    pool_path = pool_dir / "candidate_pools" / "V000.json"
    pool_payload = json.loads(pool_path.read_text(encoding="utf-8"))
    pool_payload["candidate_decoder_contract"]["tip_score_floor"] = 0.25
    pool_payload["candidate_decoder_contract_sha256"] = sha256_json(
        pool_payload["candidate_decoder_contract"]
    )
    unsigned_pool = deepcopy(pool_payload)
    unsigned_pool.pop("candidate_pool_payload_identity_sha256")
    pool_payload["candidate_pool_payload_identity_sha256"] = sha256_json(
        unsigned_pool
    )
    with pytest.raises(SelectionGateError, match="decoder contract changed"):
        validate_biological_candidate_pool_payload(
            pool_payload,
            expected_task_id="V000",
            expected_image_sha256=sha256_file(
                tmp_path / "sealed" / "images" / "V000.tif"
            ),
            expected_pending_model_metadata=_manifest["detection_model_metadata"],
        )


def test_selection_receipt_rejects_rehashed_infeasible_matched_pair(
    tmp_path: Path,
) -> None:
    _checkpoints, _manifest, receipt_path, _selected, _pool_dir = (
        _write_selection_fixture(tmp_path)
    )
    receipt = read_selection_receipt(receipt_path)
    tampered = deepcopy(receipt)
    matched_pair = next(
        pair
        for threshold_row in tampered["threshold_metrics"]
        for image_row in threshold_row["per_image"]
        for pair in image_row["biological_presence_matched_pairs"]
    )
    matched_pair["truth_coverage"] = 0.0
    unsigned = deepcopy(tampered)
    unsigned.pop("selection_receipt_identity_sha256")
    tampered["selection_receipt_identity_sha256"] = sha256_json(unsigned)
    from phaxis.hair_stageb.selection import validate_selection_receipt

    with pytest.raises(SelectionGateError, match="matched-pair sufficiency"):
        validate_selection_receipt(tampered)


def test_qcdev_selection_primary_identity_can_beat_attachment_proxy() -> None:
    """The real candidate selector must not silently optimize base localization."""

    rows = [
        {
            "task_id": f"V{index:03d}",
            # Candidate 0 is the same biological trunk but begins 30 um distal
            # to the manual base. Candidate 1 has a perfect base but follows a
            # perpendicular distractor and therefore lacks curve support.
            "candidate_base_xy_um": np.asarray([[30.0, 0.0], [0.0, 0.0]]),
            "candidate_tip_xy_um": np.asarray([[70.0, 0.0], [0.0, 100.0]]),
            "candidate_scores": np.asarray([0.25, 0.15]),
            "candidate_presence_proxy_valid": np.asarray([True, True]),
            "truth_base_xy_um": np.asarray([[0.0, 0.0]]),
            "truth_polylines_xy_um": [
                np.asarray([[0.0, 0.0], [100.0, 0.0]])
            ],
        }
        for index in range(44)
    ]
    metrics, selected = select_operating_point(rows)
    assert selected["threshold"] == 0.25
    assert selected["tolerant_biological_presence_20um"]["f1"] == 1.0
    assert selected["identity_attachment_proxy_20um"]["f1"] == 0.0
    low_threshold = next(row for row in metrics if row["threshold"] == 0.15)
    assert low_threshold["identity_attachment_proxy_20um"]["f1"] > 0.0


def test_selection_count_agreement_exposes_duplicate_inflation_unless_truths_are_within_resolution() -> None:
    duplicate_bases = np.asarray([[0.0, 0.0], [0.0, 0.0]])
    duplicate_tips = np.asarray([[100.0, 0.0], [100.0, 0.0]])
    scores = np.asarray([0.25, 0.25])

    one_truth_rows = [
        {
            "task_id": f"S{index:03d}",
            "candidate_base_xy_um": duplicate_bases,
            "candidate_tip_xy_um": duplicate_tips,
            "candidate_scores": scores,
            "truth_base_xy_um": np.asarray([[0.0, 0.0]]),
            "truth_polylines_xy_um": [
                np.asarray([[0.0, 0.0], [100.0, 0.0]])
            ],
        }
        for index in range(44)
    ]
    _metrics, one_selected = select_operating_point(one_truth_rows)
    assert one_selected["threshold"] == 0.25
    assert one_selected["tolerant_biological_presence_20um"][
        "true_positive"
    ] == 44
    assert one_selected["count_mae"] == 1.0
    assert one_selected["count_bias"] == 1.0

    close_two_truth_rows = [
        {
            **row,
            "task_id": f"D{index:03d}",
            "truth_base_xy_um": np.asarray([[0.0, 0.0], [0.0, 12.0]]),
            "truth_polylines_xy_um": [
                np.asarray([[0.0, 0.0], [100.0, 0.0]]),
                np.asarray([[0.0, 12.0], [100.0, 12.0]]),
            ],
        }
        for index, row in enumerate(one_truth_rows)
    ]
    _metrics, close_selected = select_operating_point(close_two_truth_rows)
    assert close_selected["tolerant_biological_presence_20um"]["f1"] == 1.0
    assert close_selected["count_mae"] == 0.0
    assert close_selected["count_bias"] == 0.0


def test_runtime_requires_and_cross_checks_selection_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints, manifest, receipt_path, selected, _pool = _write_selection_fixture(
        tmp_path
    )

    class FakeModel:
        def to(self, _device):
            return self

        def load_state_dict(self, _state):
            return None

        def eval(self):
            return self

    monkeypatch.setattr(stageb_runtime, "MultiHeadUNet", lambda *_a, **_k: FakeModel())
    with pytest.raises(ValueError, match="requires the bound selection receipt"):
        StageBEnsemble(
            checkpoints,
            device="cpu",
            candidate_manifest=manifest,
            selected_model_metadata=selected,
        )
    ensemble = StageBEnsemble(
        checkpoints,
        device="cpu",
        candidate_manifest=manifest,
        selected_model_metadata=selected,
        selection_receipt=receipt_path,
    )
    assert ensemble.score_threshold == 0.25
    assert ensemble.detection_model_metadata == selected

    pending = StageBEnsemble(
        checkpoints,
        device="cpu",
        candidate_manifest=manifest,
        candidate_pool_mode=True,
    )
    with pytest.raises(RuntimeError, match="cannot emit final"):
        pending.predict(np.zeros((8, 8), np.uint8), source_um_per_px=1.0)


def test_formal_evaluator_requires_exact_candidate_metadata_receipt_binding(
    tmp_path: Path,
) -> None:
    _checkpoints, manifest, receipt_path, selected, _pool = _write_selection_fixture(
        tmp_path
    )
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "phaxis"
        / "evaluate_stageb_train399_qcdev44.py"
    )
    spec = importlib.util.spec_from_file_location("formal_stageb_eval_gate_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed_manifest, observed_selected, observed_receipt = module._load_formal_model_gate(
        candidate_manifest_path=tmp_path / "candidate_manifest.json",
        selected_model_metadata_path=tmp_path / "selected_model_metadata.json",
        selection_receipt_path=receipt_path,
    )
    assert observed_manifest == manifest
    assert observed_selected == selected
    assert observed_receipt["selection_receipt_identity_sha256"] == selected[
        "selection_receipt_identity_sha256"
    ]

    tampered = deepcopy(selected)
    tampered["expert_id"] = "unbound-expert-name"
    unsigned = deepcopy(tampered)
    unsigned.pop("selected_model_metadata_identity_sha256")
    tampered["selected_model_metadata_identity_sha256"] = sha256_json(unsigned)
    tampered_path = tmp_path / "tampered_selected_model_metadata.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(SelectionGateError, match="candidate Gate: expert_id"):
        module._load_formal_model_gate(
            candidate_manifest_path=tmp_path / "candidate_manifest.json",
            selected_model_metadata_path=tampered_path,
            selection_receipt_path=receipt_path,
        )


def test_evaluation_only_full_geometry_breaks_proposal_cycle_and_is_nonproduction(
    tmp_path: Path, phaxis_case
) -> None:
    checkpoints, _manifest, receipt_path, selected, _pool = _write_selection_fixture(
        tmp_path
    )
    candidate_path = tmp_path / "candidate_manifest.json"
    selected_path = tmp_path / "selected_model_metadata.json"
    _candidate, _selected, _receipt, gate = build_evaluation_gate_binding(
        candidate_manifest_path=candidate_path,
        selected_model_metadata_path=selected_path,
        selection_receipt_path=receipt_path,
        checkpoint_paths=checkpoints,
    )
    assert gate["model_contract_proposal_required_for_artifact"] is False
    assert gate["model_contract_proposal_present"] is False
    assert "model_contract_proposal_sha256" not in gate
    image_sha256 = sha256_file(tmp_path / "images" / "V000.tif")
    payload = make_evaluation_detection_payload(
        task_id="V000",
        source_image_sha256=image_sha256,
        source_um_per_px=1.0,
        prediction=_evaluation_prediction(),
        selected_model_metadata=selected,
        evaluation_gate=gate,
    )
    assert payload["schema_version"] == EVALUATION_DETECTION_SCHEMA
    assert payload["artifact_role"] == EVALUATION_ARTIFACT_ROLE
    assert payload["fusion_consumption_allowed"] is False
    core = validate_evaluation_detection_payload(
        payload,
        expected_task_id="V000",
        expected_image_sha256=image_sha256,
        expected_model_metadata=selected,
        expected_evaluation_gate=gate,
    )
    assert core["detections"][0]["tip_snapped"] is True
    assert core["detections"][0]["predicted_length_semantics"] == (
        "regressed_polyline_arc_length_um_diagnostic_only"
    )
    with pytest.raises(ContractError, match="unsupported Stage-B detection schema"):
        validate_stageb_detection_payload(
            payload,
            expected_task_id="V000",
            expected_image_sha256=image_sha256,
        )

    hybrid, _production_stageb, artifact_root = phaxis_case
    proposal = {
        "model_contract_proposal_sha256": "b" * 64,
        "model_contract_proposal_identity_sha256": "c" * 64,
    }
    public = {
        "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
        "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
    }
    with pytest.raises(ContractError, match="unsupported Stage-B detection schema"):
        fuse_hybrid_root_with_stageb_hairs(
            hybrid,
            payload,
            hybrid_artifact_root=artifact_root,
            model_contract_proposal=proposal,
            model_contract_public_identity=public,
        )

    # Even deliberately extracting the inner normal-schema geometry cannot
    # bypass production: it carries no proposal/public authority fields.
    extracted = deepcopy(core)
    extracted["task_id"] = hybrid["task_id"]
    extracted["source_image_sha256"] = hybrid["source_image_sha256"]
    extracted.pop("detection_identity_sha256")
    extracted["detection_identity_sha256"] = sha256_json(extracted)
    with pytest.raises(ContractError, match="model-contract mismatch"):
        fuse_hybrid_root_with_stageb_hairs(
            hybrid,
            extracted,
            hybrid_artifact_root=artifact_root,
            model_contract_proposal=proposal,
            model_contract_public_identity=public,
        )

    tampered = deepcopy(payload)
    tampered["selection_receipt_identity_sha256"] = "f" * 64
    with pytest.raises(EvaluationInferenceError, match="selection_receipt_identity"):
        validate_evaluation_detection_payload(
            tampered,
            expected_task_id="V000",
            expected_image_sha256=image_sha256,
            expected_model_metadata=selected,
            expected_evaluation_gate=gate,
        )


def test_evaluation_only_cli_seals_exact44_without_proposal_and_validates_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints, _manifest, receipt_path, selected, _pool = _write_selection_fixture(
        tmp_path
    )
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "phaxis"
        / "run_stageb_evaluation_inference.py"
    )
    spec = importlib.util.spec_from_file_location(
        "stageb_evaluation_inference_cli_test", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "model_contract_proposal" not in {
        action.dest for action in module._parser()._actions
    }

    task_ids = [f"V{index:03d}" for index in range(44)]
    locked_ids = tmp_path / "locked_val_ids.txt"
    locked_ids.write_text("\n".join(task_ids) + "\n", encoding="utf-8")
    inference_manifest = tmp_path / "evaluation_inference.csv"
    with inference_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_id", "image_path", "image_sha256", "um_per_px"),
        )
        writer.writeheader()
        for task_id in task_ids:
            image_path = tmp_path / "images" / f"{task_id}.tif"
            writer.writerow(
                {
                    "task_id": task_id,
                    "image_path": str(image_path),
                    "image_sha256": sha256_file(image_path),
                    "um_per_px": "1.0",
                }
            )

    base_argv = [
        "--manifest",
        str(inference_manifest),
        "--locked-val-ids",
        str(locked_ids),
        "--candidate-manifest",
        str(tmp_path / "candidate_manifest.json"),
        "--selected-model-metadata",
        str(tmp_path / "selected_model_metadata.json"),
        "--selection-receipt",
        str(receipt_path),
        "--output",
        str(tmp_path / "evaluation_output"),
    ]
    for checkpoint in checkpoints:
        base_argv.extend(("--checkpoint", str(checkpoint)))

    monkeypatch.setattr(
        module,
        "_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("GPU preflight must not run")),
    )
    missing_checkpoint_argv = list(base_argv)
    missing_checkpoint_argv[-1] = str(tmp_path / "missing-checkpoint.pt")
    with pytest.raises(EvaluationInferenceError, match="checkpoint is missing"):
        module.main(missing_checkpoint_argv)

    class FakeEnsemble:
        def __init__(self, paths, **kwargs):
            assert len(paths) == 5
            assert kwargs["use_amp"] is False
            assert kwargs["shared_input_acceleration"] is False
            self.detection_model_metadata = selected
            self.checkpoint_sha256 = tuple(selected["checkpoint_sha256"])
            self.last_shared_input_audit = None

        def predict(self, _image, *, source_um_per_px):
            assert source_um_per_px == 1.0
            self.last_shared_input_audit = {
                "requested": False,
                "used": False,
                "runtime_path": "synthetic_cpu_oracle",
                "fallback_reason": "not_requested",
            }
            return _evaluation_prediction()

    monkeypatch.setattr(stageb_runtime, "StageBEnsemble", FakeEnsemble)
    monkeypatch.setattr(module, "_preflight", lambda: "synthetic nvidia-smi")
    monkeypatch.setattr(module.tifffile, "imread", lambda _path: np.zeros((63, 127)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    assert module.main(base_argv) == 0

    summary_path = tmp_path / "evaluation_output" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _candidate, _selected, _receipt, expected_gate = build_evaluation_gate_binding(
        candidate_manifest_path=tmp_path / "candidate_manifest.json",
        selected_model_metadata_path=tmp_path / "selected_model_metadata.json",
        selection_receipt_path=receipt_path,
        checkpoint_paths=checkpoints,
    )
    validate_evaluation_inference_summary(
        summary, expected_evaluation_gate=expected_gate
    )
    assert summary["images"] == 44
    assert summary["model_contract_proposal_present"] is False
    assert summary["production_consumption_allowed"] is False
    assert summary["records"][0]["shared_input_runtime_audit"]["runtime_path"] == (
        "synthetic_cpu_oracle"
    )

    tampered = deepcopy(summary)
    tampered["records"][0]["evaluation_detection_file_sha256"] = "f" * 64
    with pytest.raises(EvaluationInferenceError, match="file locks differ"):
        validate_evaluation_inference_summary(
            tampered, expected_evaluation_gate=expected_gate
        )

    evaluator_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "phaxis"
        / "evaluate_stageb_train399_qcdev44.py"
    )
    evaluator_spec = importlib.util.spec_from_file_location(
        "stageb_eval_only_authority_test", evaluator_path
    )
    assert evaluator_spec is not None and evaluator_spec.loader is not None
    evaluator = importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator)
    loaded_summary, loaded_gate, record_map = (
        evaluator._load_evaluation_inference_authority(
            summary_path=summary_path,
            detections_dir=tmp_path / "evaluation_output" / "detections",
            candidate_manifest_path=tmp_path / "candidate_manifest.json",
            selected_model_metadata_path=tmp_path / "selected_model_metadata.json",
            selection_receipt_path=receipt_path,
        )
    )
    assert loaded_summary["evaluation_inference_summary_identity_sha256"] == summary[
        "evaluation_inference_summary_identity_sha256"
    ]
    assert loaded_gate == expected_gate
    assert list(record_map) == task_ids

    first_detection = tmp_path / "evaluation_output" / "detections" / "V000.json"
    drifted = json.loads(first_detection.read_text(encoding="utf-8"))
    drifted["stageb_detection_payload"]["detections"][0]["tip_xy_working"][0] += 1.0
    first_detection.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(RuntimeError, match="file hash drift"):
        evaluator._load_evaluation_inference_authority(
            summary_path=summary_path,
            detections_dir=tmp_path / "evaluation_output" / "detections",
            candidate_manifest_path=tmp_path / "candidate_manifest.json",
            selected_model_metadata_path=tmp_path / "selected_model_metadata.json",
            selection_receipt_path=receipt_path,
        )


def test_runtime_recognizes_formal_train399_checkpoint_and_requires_gate_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints, _audit = _write_fixture(tmp_path)
    monkeypatch.setattr(
        stageb_runtime,
        "HAIR_CHECKPOINT_SHA256",
        tuple(sha256_file(path) for path in checkpoints),
    )
    with pytest.raises(ValueError, match="formal train399 checkpoints require"):
        StageBEnsemble(checkpoints, device="cpu")


def test_canonical_ground_truth_locks_bytes_geometry_and_physical_scale(
    tmp_path: Path,
) -> None:
    _checkpoints, _audit = _write_fixture(tmp_path)
    truth, provenance = load_canonical_qcdev_ground_truth(
        dataset_root=tmp_path,
        dataset_manifest=tmp_path / "manifests" / "dataset_manifest.csv",
        split_manifest=tmp_path / "split_manifest.csv",
        expected_task_ids=[f"V{index:03d}" for index in range(44)],
    )
    assert len(truth) == 44
    np.testing.assert_allclose(truth["V000"]["base"], [[10.0, 10.0]])
    lock = provenance["canonical_annotation_locks"][0]
    assert lock["source_um_per_px"] == 1.0
    assert lock["source_image_shape_hw"] == [63, 127]
    assert lock["root_hair_count"] == 1
    assert len(lock["canonical_annotation_sha256"]) == 64
    assert len(lock["oriented_source_geometry_identity_sha256"]) == 64
    assert len(lock["physical_geometry_identity_sha256"]) == 64

    path = tmp_path / "annotations" / "rhaxis_canonical" / "V000.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["shapes"][1]["points"][0][0] += 1.0
    path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(CanonicalGroundTruthError, match="SHA-256 mismatch"):
        load_canonical_qcdev_ground_truth(
            dataset_root=tmp_path,
            dataset_manifest=tmp_path / "manifests" / "dataset_manifest.csv",
            split_manifest=tmp_path / "split_manifest.csv",
        )


def test_candidate_pool_cli_emits_only_nonfinal_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoints, audit_path = _write_fixture(tmp_path)
    manifest = build_candidate_manifest(checkpoints, dataset_audit_path=audit_path)
    manifest_path = tmp_path / "candidate_manifest.json"
    write_candidate_manifest(manifest_path, manifest)
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    inference_manifest = tmp_path / "inference.csv"
    with inference_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_id", "image_path", "image_sha256", "um_per_px"),
        )
        writer.writeheader()
        for index in range(44):
            task_id = f"V{index:03d}"
            image_path = images / f"{task_id}.tif"
            image_path.write_bytes(f"immutable-{task_id}".encode())
            writer.writerow(
                {
                    "task_id": task_id,
                    "image_path": str(image_path),
                    "image_sha256": sha256_file(image_path),
                    "um_per_px": "1.0",
                }
            )

    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "phaxis"
        / "build_stageb_train399_qcdev44_candidate_pool.py"
    )
    spec = importlib.util.spec_from_file_location("candidate_pool_cli_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeEnsemble:
        def __init__(self, paths, **kwargs):
            assert len(paths) == 5
            assert kwargs["candidate_pool_mode"] is True
            assert kwargs["use_amp"] is False
            assert kwargs["shared_input_acceleration"] is False
            self.checkpoint_sha256 = tuple(
                manifest["detection_model_metadata"]["checkpoint_sha256"]
            )
            self.shared_input_max_host_bytes = 2 * 1024**3
            self.shared_input_max_device_bytes = 1 * 1024**3
            self.shared_input_device_reserve_bytes = 2 * 1024**3
            self.last_shared_input_audit = None

        def predict_biological_candidate_pool(self, _image, *, source_um_per_px):
            assert source_um_per_px == 1.0
            self.last_shared_input_audit = {
                "requested": False,
                "used": False,
                "runtime_path": "legacy",
                "fallback_reason": "not_requested",
            }
            return _biological_pool_prediction()

    monkeypatch.setattr(stageb_runtime, "StageBEnsemble", FakeEnsemble)
    monkeypatch.setattr(module, "_preflight", lambda: "synthetic nvidia-smi")
    monkeypatch.setattr(module.tifffile, "imread", lambda _path: np.zeros((64, 128)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    output = tmp_path / "pool_output"
    argv = [
        str(script_path),
        "--manifest",
        str(inference_manifest),
        "--candidate-manifest",
        str(manifest_path),
        "--output",
        str(output),
    ]
    for checkpoint in checkpoints:
        argv.extend(("--checkpoint", str(checkpoint)))
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    payload = json.loads(
        (output / "candidate_pools" / "V000.json").read_text(encoding="utf-8")
    )
    validate_biological_candidate_pool_payload(
        payload,
        expected_task_id="V000",
        expected_image_sha256=sha256_file(images / "V000.tif"),
        expected_pending_model_metadata=manifest["detection_model_metadata"],
    )
    assert payload["artifact_role"] == "candidate_pool_not_final_prediction"
    assert "detections" not in payload
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["shared_input_acceleration"] == {
        "requested": False,
        "default_enabled": False,
        "effective_max_host_bytes": 2 * 1024**3,
        "effective_max_device_bytes": 1 * 1024**3,
        "effective_device_reserve_bytes": 2 * 1024**3,
        "executed_images": 44,
        "resumed_images_not_executed": 0,
        "runtime_path_counts": {"legacy": 44},
        "fallback_reason_counts": {"not_requested": 44},
    }
    assert summary["records"][0]["shared_input_runtime_audit"] == {
        "requested": False,
        "used": False,
        "runtime_path": "legacy",
        "fallback_reason": "not_requested",
    }
    with pytest.raises(ContractError, match="schema"):
        validate_stageb_detection_payload(
            payload,
            expected_task_id="V000",
            expected_image_sha256=sha256_file(images / "V000.tif"),
        )


def test_candidate_pool_is_rejected_by_actual_fusion_consumer(
    tmp_path: Path, phaxis_case
) -> None:
    checkpoints, audit = _write_fixture(tmp_path)
    manifest = build_candidate_manifest(checkpoints, dataset_audit_path=audit)
    hybrid, _stageb, artifact_root = phaxis_case
    pool = make_biological_candidate_pool_payload(
        task_id=hybrid["task_id"],
        source_image_sha256=hybrid["source_image_sha256"],
        source_um_per_px=1.0,
        prediction=_biological_pool_prediction(),
        pending_model_metadata=manifest["detection_model_metadata"],
    )
    with pytest.raises(ContractError, match="unsupported Stage-B detection schema"):
        fuse_hybrid_root_with_stageb_hairs(
            hybrid,
            pool,
            hybrid_artifact_root=artifact_root,
            model_contract_proposal={
                "model_contract_proposal_sha256": "b" * 64,
                "model_contract_proposal_identity_sha256": "c" * 64,
            },
            model_contract_public_identity={
                "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
                "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
            },
        )


def test_batch_inference_cli_rejects_partial_train399_gate_before_gpu_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "phaxis"
        / "run_stageb_inference.py"
    )
    spec = importlib.util.spec_from_file_location("stageb_batch_cli_gate_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("GPU preflight must not run")),
    )
    argv = [
        str(script_path),
        "--manifest",
        str(tmp_path / "images.csv"),
        "--output",
        str(tmp_path / "output"),
        "--candidate-manifest",
        str(tmp_path / "candidate.json"),
    ]
    for index in range(5):
        argv.extend(("--checkpoint", str(tmp_path / f"member{index}.pt")))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="requires --model-contract-proposal"):
        module.main()


def test_batch_inference_cli_rejects_amp_formal_gate_before_gpu_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "phaxis"
        / "run_stageb_inference.py"
    )
    spec = importlib.util.spec_from_file_location("stageb_batch_cli_amp_gate_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("GPU preflight must not run")),
    )
    argv = [
        str(script_path),
        "--manifest",
        str(tmp_path / "images.csv"),
        "--output",
        str(tmp_path / "output"),
        "--model-contract-proposal",
        str(tmp_path / "proposal.json"),
        "--candidate-manifest",
        str(tmp_path / "candidate.json"),
        "--selected-model-metadata",
        str(tmp_path / "selected.json"),
        "--selection-receipt",
        str(tmp_path / "selection.json"),
        "--amp",
    ]
    for index in range(5):
        argv.extend(("--checkpoint", str(tmp_path / f"member{index}.pt")))
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(RuntimeError, match="locked to FP32"):
        module.main()


def test_batch_inference_cli_records_explicit_shared_input_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "phaxis"
        / "run_stageb_inference.py"
    )
    spec = importlib.util.spec_from_file_location("stageb_batch_cli_audit_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image_path = tmp_path / "image.tif"
    image_path.write_bytes(b"immutable-image")
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_id", "image_path", "image_sha256", "um_per_px"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "T001",
                "image_path": str(image_path),
                "image_sha256": sha256_file(image_path),
                "um_per_px": "1.0",
            }
        )

    class FakeEnsemble:
        def __init__(self, paths, **kwargs):
            assert len(paths) == 5
            assert kwargs["shared_input_acceleration"] is True
            assert kwargs["shared_input_max_host_bytes"] == 123_456_789
            assert kwargs["shared_input_max_device_bytes"] == 234_567_891
            assert kwargs["shared_input_device_reserve_bytes"] == 345_678_912
            self.shared_input_max_host_bytes = kwargs[
                "shared_input_max_host_bytes"
            ]
            self.shared_input_max_device_bytes = kwargs[
                "shared_input_max_device_bytes"
            ]
            self.shared_input_device_reserve_bytes = kwargs[
                "shared_input_device_reserve_bytes"
            ]
            self.checkpoint_sha256 = tuple(_hash(str(index + 1)) for index in range(5))
            self.detection_model_metadata = None
            self.score_threshold = 0.225
            self.last_shared_input_audit = None

        def predict(self, _image, *, source_um_per_px):
            assert source_um_per_px == 1.0
            self.last_shared_input_audit = {
                "requested": True,
                "used": True,
                "runtime_path": "shared_input_acceleration",
                "fallback_reason": "none",
            }
            return _prediction()

    monkeypatch.setattr(stageb_runtime, "StageBEnsemble", FakeEnsemble)

    class FakeProposalBinding:
        def receipt_fields(self):
            return {
                "model_contract_proposal_sha256": _hash("a"),
                "model_contract_proposal_identity_sha256": _hash("b"),
            }

        def public_identity_fields(self):
            return {
                "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
                "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
            }

    monkeypatch.setattr(
        module,
        "read_model_contract_authority",
        lambda _path: FakeProposalBinding(),
    )
    monkeypatch.setattr(
        module,
        "validate_stageb_proposal_binding",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(module, "_preflight", lambda: "synthetic nvidia-smi")
    monkeypatch.setattr(module.tifffile, "imread", lambda _path: np.zeros((64, 128)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    output = tmp_path / "output"
    proposal_path = tmp_path / "proposal.json"
    candidate_path = tmp_path / "candidate.json"
    selected_path = tmp_path / "selected.json"
    selection_path = tmp_path / "selection.json"
    for path in (proposal_path, candidate_path, selected_path, selection_path):
        path.write_text("{}\n", encoding="utf-8")
    argv = [
        str(script_path),
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
        "--model-contract-proposal",
        str(proposal_path),
        "--candidate-manifest",
        str(candidate_path),
        "--selected-model-metadata",
        str(selected_path),
        "--selection-receipt",
        str(selection_path),
        "--shared-input-acceleration",
        "--shared-input-max-host-bytes",
        "123456789",
        "--shared-input-max-device-bytes",
        "234567891",
        "--shared-input-device-reserve-bytes",
        "345678912",
    ]
    for index in range(5):
        argv.extend(("--checkpoint", str(tmp_path / f"member{index}.pt")))
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["shared_input_acceleration"] == {
        "requested": True,
        "default_enabled": False,
        "effective_max_host_bytes": 123_456_789,
        "effective_max_device_bytes": 234_567_891,
        "effective_device_reserve_bytes": 345_678_912,
        "executed_images": 1,
        "resumed_images_not_executed": 0,
        "runtime_path_counts": {"shared_input_acceleration": 1},
        "fallback_reason_counts": {"none": 1},
    }
    assert summary["records"][0]["shared_input_runtime_audit"]["used"] is True
