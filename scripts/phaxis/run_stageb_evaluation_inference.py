"""Create exact-QCdev44 full-geometry Stage-B evidence before promotion.

This entry point is deliberately separate from production inference.  It
accepts the selected candidate/selection receipts but no model-contract
proposal, and writes only the non-deployable schemas defined by
``phaxis.hair_stageb.evaluation_inference``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.hair_stageb.evaluation_inference import (  # noqa: E402
    EVALUATION_IMAGE_COUNT,
    build_evaluation_gate_binding,
    make_evaluation_detection_payload,
    make_evaluation_inference_summary,
    validate_evaluation_detection_payload,
    validate_evaluation_inference_summary,
)
from phaxis.io import atomic_write_json, read_json, sha256_file  # noqa: E402


def _preflight() -> str:
    result = subprocess.run(
        ["nvidia-smi"], check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected train399 Stage-B on locked QCdev44 for formal "
            "evaluation only; outputs cannot be consumed by production."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--locked-val-ids", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--selected-model-metadata", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--shared-input-acceleration",
        action="store_true",
        help="opt in to exact shared-input five-member inference",
    )
    parser.add_argument("--shared-input-max-host-bytes", type=int, default=None)
    parser.add_argument("--shared-input-max-device-bytes", type=int, default=None)
    parser.add_argument("--shared-input-device-reserve-bytes", type=int, default=None)
    return parser


def _locked_ids(path: Path) -> list[str]:
    values = [value.strip() for value in path.read_text(encoding="utf-8").splitlines()]
    task_ids = [value for value in values if value]
    if len(task_ids) != EVALUATION_IMAGE_COUNT or len(set(task_ids)) != EVALUATION_IMAGE_COUNT:
        raise RuntimeError("locked QC-development set must contain 44 unique task IDs")
    return task_ids


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    task_ids = [row.get("task_id") for row in rows]
    if len(rows) != EVALUATION_IMAGE_COUNT or len(set(task_ids)) != EVALUATION_IMAGE_COUNT:
        raise RuntimeError("evaluation inference manifest must contain exact QCdev44")
    return rows


def _validate_source_rows(
    rows: list[dict[str, str]],
    *,
    locked_task_ids: list[str],
    selection_receipt: dict,
) -> list[dict[str, object]]:
    if [row["task_id"] for row in rows] != locked_task_ids:
        raise RuntimeError("evaluation manifest order differs from locked QCdev44")
    task_locks = selection_receipt["task_image_locks"]
    if [row["task_id"] for row in task_locks] != locked_task_ids:
        raise RuntimeError("locked QCdev44 differs from selection receipt task order")
    prepared: list[dict[str, object]] = []
    for row, task_lock in zip(rows, task_locks, strict=True):
        task_id = row["task_id"]
        image_path = Path(row.get("image_path") or "")
        if not image_path.is_file():
            raise FileNotFoundError(f"{task_id}: source image is missing: {image_path}")
        observed_sha256 = sha256_file(image_path)
        declared_sha256 = row.get("image_sha256") or row.get("source_image_sha256")
        if not declared_sha256 or observed_sha256.casefold() != declared_sha256.casefold():
            raise RuntimeError(f"{task_id}: inference manifest image hash mismatch")
        if observed_sha256.casefold() != task_lock["source_image_sha256"].casefold():
            raise RuntimeError(f"{task_id}: source image differs from selection receipt")
        raw_scale = row.get("um_per_px") or row.get("source_um_per_px")
        if raw_scale is None:
            raise RuntimeError(f"{task_id}: physical scale is absent")
        source_um_per_px = float(raw_scale)
        if not np.isclose(
            source_um_per_px,
            float(task_lock["source_um_per_px"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"{task_id}: physical scale differs from selection receipt")
        if row.get("image_height") and row.get("image_width"):
            declared_shape = [int(row["image_height"]), int(row["image_width"])]
            if declared_shape != task_lock["source_image_shape_hw"]:
                raise RuntimeError(f"{task_id}: source shape differs from selection receipt")
        prepared.append(
            {
                "task_id": task_id,
                "image_path": image_path,
                "source_image_sha256": observed_sha256,
                "source_um_per_px": source_um_per_px,
                "source_image_shape_hw": list(task_lock["source_image_shape_hw"]),
            }
        )
    return prepared


def _validate_existing_detection(
    path: Path,
    *,
    source: dict[str, object],
    selected_model_metadata: dict,
    evaluation_gate: dict,
) -> tuple[dict, dict]:
    payload = read_json(path)
    core = validate_evaluation_detection_payload(
        payload,
        expected_task_id=str(source["task_id"]),
        expected_image_sha256=str(source["source_image_sha256"]),
        expected_model_metadata=selected_model_metadata,
        expected_evaluation_gate=evaluation_gate,
    )
    return payload, core


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.checkpoint) != 5:
        raise RuntimeError("exactly five --checkpoint arguments are required")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not str(args.device).startswith("cuda"):
        raise RuntimeError("evaluation-only Stage-B inference requires a CUDA device")
    for name in ("shared_input_max_host_bytes", "shared_input_max_device_bytes"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if (
        args.shared_input_device_reserve_bytes is not None
        and args.shared_input_device_reserve_bytes < 0
    ):
        raise ValueError("shared-input-device-reserve-bytes cannot be negative")

    (
        candidate_manifest,
        selected_model_metadata,
        selection_receipt,
        evaluation_gate,
    ) = build_evaluation_gate_binding(
        candidate_manifest_path=args.candidate_manifest,
        selected_model_metadata_path=args.selected_model_metadata,
        selection_receipt_path=args.selection_receipt,
        checkpoint_paths=args.checkpoint,
    )
    locked_task_ids = _locked_ids(args.locked_val_ids)
    if locked_task_ids != evaluation_gate["qcdevelopment44_task_ids"]:
        raise RuntimeError("locked QCdev44 differs from the sealed selection receipt")
    sources = _validate_source_rows(
        _manifest_rows(args.manifest),
        locked_task_ids=locked_task_ids,
        selection_receipt=selection_receipt,
    )

    summary_path = args.output / "summary.json"
    if args.resume and summary_path.is_file():
        summary = read_json(summary_path)
        validate_evaluation_inference_summary(
            summary, expected_evaluation_gate=evaluation_gate
        )
        if summary.get("inference_manifest_sha256") != sha256_file(args.manifest):
            raise RuntimeError("completed evaluation run used a different inference manifest")
        if summary.get("locked_val_ids_sha256") != sha256_file(args.locked_val_ids):
            raise RuntimeError("completed evaluation run used a different QCdev44 lock")
        records = summary["records"]
        for source, record in zip(sources, records, strict=True):
            path = args.output / "detections" / f"{source['task_id']}.json"
            if not path.is_file() or sha256_file(path) != record["evaluation_detection_file_sha256"]:
                raise RuntimeError(f"{source['task_id']}: completed evaluation artifact drifted")
            _validate_existing_detection(
                path,
                source=source,
                selected_model_metadata=selected_model_metadata,
                evaluation_gate=evaluation_gate,
            )
        print("completed evaluation-only QCdev44 output is hash-valid; nothing to run")
        return 0

    records: list[dict[str, object]] = []
    pending_sources: list[dict[str, object]] = []
    for source in sources:
        task_id = str(source["task_id"])
        detection_path = args.output / "detections" / f"{task_id}.json"
        if args.resume and detection_path.is_file():
            payload, core = _validate_existing_detection(
                detection_path,
                source=source,
                selected_model_metadata=selected_model_metadata,
                evaluation_gate=evaluation_gate,
            )
            records.append(
                {
                    "task_id": task_id,
                    "source_image_sha256": source["source_image_sha256"],
                    "detections": int(core["n"]),
                    "wall_seconds_including_io": 0.0,
                    "resumed": True,
                    "evaluation_detection_identity_sha256": payload[
                        "evaluation_detection_identity_sha256"
                    ],
                    "evaluation_detection_file_sha256": sha256_file(detection_path),
                    "shared_input_runtime_audit": None,
                }
            )
        else:
            pending_sources.append(source)

    nvidia_smi = "not_run_all_images_resumed"
    model_load_seconds = 0.0
    run_started = time.perf_counter()
    ensemble = None
    torch = None
    if pending_sources:
        nvidia_smi = _preflight()

        # Keep torch/CUDA imports strictly after receipt, checkpoint, task-order,
        # source-image, and physical-scale preflight.
        import torch as torch_module

        from phaxis.hair_stageb.runtime import StageBEnsemble

        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        shared_options: dict[str, object] = {
            "shared_input_acceleration": bool(args.shared_input_acceleration)
        }
        for name in (
            "shared_input_max_host_bytes",
            "shared_input_max_device_bytes",
            "shared_input_device_reserve_bytes",
        ):
            value = getattr(args, name)
            if value is not None:
                shared_options[name] = value
        load_started = time.perf_counter()
        ensemble = StageBEnsemble(
            args.checkpoint,
            device=args.device,
            use_amp=False,
            candidate_manifest=args.candidate_manifest,
            selected_model_metadata=args.selected_model_metadata,
            selection_receipt=args.selection_receipt,
            **shared_options,
        )
        if ensemble.detection_model_metadata != selected_model_metadata:
            raise RuntimeError("runtime model metadata differs from the selected candidate")
        if list(ensemble.checkpoint_sha256) != selected_model_metadata["checkpoint_sha256"]:
            raise RuntimeError("runtime checkpoint order differs from the selected candidate")
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started
        torch.cuda.reset_peak_memory_stats()

    prior_records = {str(row["task_id"]): row for row in records}
    for source in sources:
        task_id = str(source["task_id"])
        if task_id in prior_records:
            continue
        assert ensemble is not None and torch is not None
        started = time.perf_counter()
        image = tifffile.imread(Path(source["image_path"]))
        if list(image.shape[:2]) != source["source_image_shape_hw"]:
            raise RuntimeError(f"{task_id}: decoded image shape differs from selection receipt")
        prediction = ensemble.predict(
            image, source_um_per_px=float(source["source_um_per_px"])
        )
        observed_audit = getattr(ensemble, "last_shared_input_audit", None)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        payload = make_evaluation_detection_payload(
            task_id=task_id,
            source_image_sha256=str(source["source_image_sha256"]),
            source_um_per_px=float(source["source_um_per_px"]),
            prediction=prediction,
            selected_model_metadata=selected_model_metadata,
            evaluation_gate=evaluation_gate,
        )
        detection_path = args.output / "detections" / f"{task_id}.json"
        atomic_write_json(detection_path, payload)
        prior_records[task_id] = {
            "task_id": task_id,
            "source_image_sha256": source["source_image_sha256"],
            "detections": int(payload["stageb_detection_payload"]["n"]),
            "wall_seconds_including_io": seconds,
            "resumed": False,
            "evaluation_detection_identity_sha256": payload[
                "evaluation_detection_identity_sha256"
            ],
            "evaluation_detection_file_sha256": sha256_file(detection_path),
            "shared_input_runtime_audit": (
                dict(observed_audit) if observed_audit is not None else None
            ),
        }
        print(f"{task_id}: {payload['stageb_detection_payload']['n']} evaluation-only hairs", flush=True)

    ordered_records = [prior_records[task_id] for task_id in locked_task_ids]
    executed_audits = [
        row["shared_input_runtime_audit"]
        for row in ordered_records
        if row["shared_input_runtime_audit"] is not None
    ]
    path_counts = Counter(row["runtime_path"] for row in executed_audits)
    fallback_counts = Counter(row["fallback_reason"] for row in executed_audits)
    run_seconds = time.perf_counter() - run_started
    runtime_metadata = {
        "inference_manifest_sha256": sha256_file(args.manifest),
        "locked_val_ids_sha256": sha256_file(args.locked_val_ids),
        "model_load_seconds": model_load_seconds,
        "batch_wall_seconds_including_io": run_seconds,
        "resumed_images": sum(bool(row["resumed"]) for row in ordered_records),
        "peak_allocated_vram_mib": (
            float(torch.cuda.max_memory_allocated() / 2**20) if torch is not None else 0.0
        ),
        "peak_reserved_vram_mib": (
            float(torch.cuda.max_memory_reserved() / 2**20) if torch is not None else 0.0
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "internal_device": args.device,
        "precision_mode": "fp32_locked",
        "shared_input_acceleration": {
            "requested": bool(args.shared_input_acceleration),
            "default_enabled": False,
            "executed_images": len(executed_audits),
            "runtime_path_counts": dict(sorted(path_counts.items())),
            "fallback_reason_counts": dict(sorted(fallback_counts.items())),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__ if torch is not None else None,
        "cuda_runtime": torch.version.cuda if torch is not None else None,
        "nvidia_smi_preflight": nvidia_smi,
        "candidate_status": candidate_manifest.get("status"),
    }
    summary = make_evaluation_inference_summary(
        evaluation_gate=evaluation_gate,
        records=ordered_records,
        runtime_metadata=runtime_metadata,
    )
    atomic_write_json(summary_path, summary)
    print(f"completed non-deployable QCdev44 evaluation inference in {run_seconds:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
