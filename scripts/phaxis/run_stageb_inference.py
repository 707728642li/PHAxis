"""Run the locked PHAxis Stage-B expert on a manifest of source images."""

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

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json
from phaxis.model_contract_binding import (
    read_model_contract_authority,
    require_receipt_binding,
    validate_stageb_proposal_binding,
)


def _preflight() -> str:
    result = subprocess.run(
        ["nvidia-smi"], check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--model-contract-proposal",
        type=Path,
        default=None,
        help=(
            "sealed PHAxis 1.0.0 model-contract authority "
            "(unapplied proposal or applied official contract)"
        ),
    )
    parser.add_argument(
        "--candidate-manifest",
        type=Path,
        default=None,
        help="strict train399 candidate Gate receipt; requires --selected-model-metadata",
    )
    parser.add_argument(
        "--selected-model-metadata",
        type=Path,
        default=None,
        help="operating-point-bound metadata derived from the candidate receipt",
    )
    parser.add_argument(
        "--selection-receipt",
        type=Path,
        default=None,
        help="required for train399; file/logical/pool/threshold identities are rechecked",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only hash-valid detections already present in the output directory",
    )
    parser.add_argument(
        "--shared-input-acceleration",
        action="store_true",
        help=(
            "opt in to exact shared-input five-member inference; default is legacy"
        ),
    )
    parser.add_argument(
        "--shared-input-max-host-bytes",
        type=int,
        default=None,
        help="override the runtime safe host-array limit",
    )
    parser.add_argument(
        "--shared-input-max-device-bytes",
        type=int,
        default=None,
        help="override the deterministic device-staging limit",
    )
    parser.add_argument(
        "--shared-input-device-reserve-bytes",
        type=int,
        default=None,
        help="override the free-device-memory reserve (default 2 GiB)",
    )
    args = parser.parse_args()
    if len(args.checkpoint) != 5:
        raise RuntimeError("exactly five --checkpoint arguments are required")
    gate_values = (
        args.model_contract_proposal,
        args.candidate_manifest,
        args.selected_model_metadata,
        args.selection_receipt,
    )
    if not all(value is not None for value in gate_values):
        raise RuntimeError(
            "formal train399 batch inference requires --model-contract-proposal, "
            "--candidate-manifest, --selected-model-metadata and --selection-receipt; "
            "legacy 443CV fallback is forbidden"
        )
    if args.amp:
        raise RuntimeError("formal PHAxis Stage-B inference is locked to FP32")
    proposal_binding = read_model_contract_authority(args.model_contract_proposal)
    candidate_manifest = read_json(args.candidate_manifest)
    selected_model_metadata = read_json(args.selected_model_metadata)
    selection_receipt = read_json(args.selection_receipt)
    validate_stageb_proposal_binding(
        proposal_binding,
        candidate_manifest_path=args.candidate_manifest,
        candidate_manifest=candidate_manifest,
        selected_model_metadata_path=args.selected_model_metadata,
        selected_model_metadata=selected_model_metadata,
        selection_receipt_path=args.selection_receipt,
        selection_receipt=selection_receipt,
        checkpoints=args.checkpoint,
    )
    public_identity = proposal_binding.public_identity_fields()
    for name in (
        "shared_input_max_host_bytes",
        "shared_input_max_device_bytes",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if (
        args.shared_input_device_reserve_bytes is not None
        and args.shared_input_device_reserve_bytes < 0
    ):
        raise ValueError("shared-input-device-reserve-bytes cannot be negative")

    nvidia_smi = _preflight()
    if not args.device.startswith("cuda"):
        raise RuntimeError("the locked Stage-B ensemble requires a CUDA device")

    import torch

    from phaxis.hair_stageb.runtime import StageBEnsemble
    from phaxis.hair_stageb.serialization import make_detection_payload

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("empty inference manifest")

    load_started = time.perf_counter()
    shared_input_options = {
        "shared_input_acceleration": bool(args.shared_input_acceleration),
    }
    for name in (
        "shared_input_max_host_bytes",
        "shared_input_max_device_bytes",
        "shared_input_device_reserve_bytes",
    ):
        value = getattr(args, name)
        if value is not None:
            shared_input_options[name] = value
    ensemble = StageBEnsemble(
        args.checkpoint,
        device=args.device,
        use_amp=bool(args.amp),
        candidate_manifest=args.candidate_manifest,
        selected_model_metadata=args.selected_model_metadata,
        selection_receipt=args.selection_receipt,
        **shared_input_options,
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()
    records = []
    resumed_images = 0
    batch_started = time.perf_counter()
    for row in rows:
        task_id = row["task_id"]
        image_path = Path(row["image_path"])
        observed_image_sha256 = sha256_file(image_path)
        expected_image_sha256 = row["image_sha256"]
        if observed_image_sha256.casefold() != expected_image_sha256.casefold():
            raise RuntimeError(f"{task_id}: source image hash mismatch")
        detection_path = args.output / "detections" / f"{task_id}.json"
        if args.resume and detection_path.is_file():
            from phaxis.contracts import validate_stageb_detection_payload

            existing = read_json(detection_path)
            require_receipt_binding(
                existing,
                proposal_binding,
                role=f"resumed Stage-B detection {task_id}",
            )
            if (
                existing.get("model_bundle_id") != public_identity["model_bundle_id"]
                or existing.get("root_expert_id") != public_identity["root_expert_id"]
            ):
                raise RuntimeError(f"{task_id}: resumed detection public identity mismatch")
            validate_stageb_detection_payload(
                existing,
                expected_task_id=task_id,
                expected_image_sha256=observed_image_sha256,
                expected_model_metadata=ensemble.detection_model_metadata,
            )
            records.append(
                {
                    "task_id": task_id,
                    "source_megapixels": float(row.get("source_megapixels") or 0.0),
                    "detections": int(existing["n"]),
                    "wall_seconds_including_io": 0.0,
                    "detection_identity_sha256": existing["detection_identity_sha256"],
                    "resumed": True,
                    "shared_input_runtime_audit": None,
                }
            )
            resumed_images += 1
            print(f"{task_id}: reused {existing['n']} hash-valid hairs", flush=True)
            continue
        image_started = time.perf_counter()
        image = tifffile.imread(image_path)
        source_um_per_px = float(row["um_per_px"])
        prediction = ensemble.predict(image, source_um_per_px=source_um_per_px)
        observed_audit = getattr(ensemble, "last_shared_input_audit", None)
        shared_input_audit = (
            dict(observed_audit) if observed_audit is not None else None
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - image_started
        payload = make_detection_payload(
            task_id=task_id,
            source_image_sha256=observed_image_sha256,
            source_um_per_px=source_um_per_px,
            prediction=prediction,
            precision_mode="amp_experimental" if args.amp else "fp32_locked",
            model_metadata=ensemble.detection_model_metadata,
            score_threshold=(
                ensemble.score_threshold
                if ensemble.detection_model_metadata is not None
                else None
            ),
        )
        payload.pop("detection_identity_sha256", None)
        payload.update(proposal_binding.receipt_fields())
        payload.update(public_identity)
        payload["detection_identity_sha256"] = sha256_json(payload)
        from phaxis.contracts import validate_stageb_detection_payload

        validate_stageb_detection_payload(
            payload,
            expected_task_id=task_id,
            expected_image_sha256=observed_image_sha256,
            expected_model_metadata=ensemble.detection_model_metadata,
        )
        atomic_write_json(detection_path, payload)
        records.append(
            {
                "task_id": task_id,
                "source_megapixels": float(np.prod(image.shape[:2]) / 1e6),
                "detections": int(payload["n"]),
                "wall_seconds_including_io": seconds,
                "detection_identity_sha256": payload["detection_identity_sha256"],
                "resumed": False,
                "shared_input_runtime_audit": shared_input_audit,
            }
        )
        print(f"{task_id}: {payload['n']} hairs in {seconds:.3f} s", flush=True)
    torch.cuda.synchronize()
    batch_seconds = time.perf_counter() - batch_started
    executed_audits = [
        record["shared_input_runtime_audit"]
        for record in records
        if record["shared_input_runtime_audit"] is not None
    ]
    path_counts = Counter(audit["runtime_path"] for audit in executed_audits)
    fallback_counts = Counter(
        audit["fallback_reason"] for audit in executed_audits
    )
    summary = {
        "schema_version": "PHAxis-StageB-inference-run-1.1",
        "status": "completed",
        "images": len(records),
        "detections": sum(record["detections"] for record in records),
        "model_load_seconds": model_load_seconds,
        "batch_wall_seconds_including_io": batch_seconds,
        "median_seconds_per_image": float(np.median([
            record["wall_seconds_including_io"] for record in records if not record["resumed"]
        ])) if resumed_images < len(records) else 0.0,
        "p95_seconds_per_image": float(np.quantile([
            record["wall_seconds_including_io"] for record in records if not record["resumed"]
        ], 0.95)) if resumed_images < len(records) else 0.0,
        "peak_allocated_vram_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_vram_mib": float(torch.cuda.max_memory_reserved() / 2**20),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "internal_device": args.device,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "precision_mode": "amp_experimental" if args.amp else "fp32_locked",
        "shared_input_acceleration": {
            "requested": bool(args.shared_input_acceleration),
            "default_enabled": False,
            "effective_max_host_bytes": int(
                getattr(
                    ensemble,
                    "shared_input_max_host_bytes",
                    args.shared_input_max_host_bytes or 2 * 1024**3,
                )
            ),
            "effective_max_device_bytes": int(
                getattr(
                    ensemble,
                    "shared_input_max_device_bytes",
                    args.shared_input_max_device_bytes or 1 * 1024**3,
                )
            ),
            "effective_device_reserve_bytes": int(
                getattr(
                    ensemble,
                    "shared_input_device_reserve_bytes",
                    (
                        args.shared_input_device_reserve_bytes
                        if args.shared_input_device_reserve_bytes is not None
                        else 2 * 1024**3
                    ),
                )
            ),
            "executed_images": len(executed_audits),
            "resumed_images_not_executed": resumed_images,
            "runtime_path_counts": dict(sorted(path_counts.items())),
            "fallback_reason_counts": dict(sorted(fallback_counts.items())),
        },
        "resumed_images": resumed_images,
        "checkpoint_sha256": list(ensemble.checkpoint_sha256),
        "detection_model_metadata": ensemble.detection_model_metadata,
        "score_threshold": ensemble.score_threshold,
        "records": records,
        "nvidia_smi_preflight": nvidia_smi,
        "blind_images_used": 0,
        **proposal_binding.receipt_fields(),
        **public_identity,
    }
    summary["summary_identity_sha256"] = sha256_json(summary)
    atomic_write_json(args.output / "summary.json", summary)
    print(f"completed {len(records)} images in {batch_seconds:.3f} s")


if __name__ == "__main__":
    main()
