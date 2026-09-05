"""Generate the non-final QCdev44 biological-presence pool with one forward/image.

This is the only runtime entry point that accepts a train399 candidate manifest
whose operating point is still pending.  It fixes the base-score floor at 0.10,
decodes one immutable straight base-to-tip presence proxy per candidate, and
emits a schema that final fusion and phenotype consumers do not accept.  The
proxy supports tolerant presence selection; distal endpoint and length errors
are not selection gates.
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

from phaxis.hair_stageb.candidate_bundle import read_candidate_manifest  # noqa: E402
from phaxis.hair_stageb.selection import (  # noqa: E402
    make_biological_candidate_pool_payload,
    validate_biological_candidate_pool_payload,
)
from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402


def _preflight() -> str:
    result = subprocess.run(
        ["nvidia-smi"], check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 44 or len({row.get("task_id") for row in rows}) != 44:
        raise RuntimeError("candidate-pool inference manifest must contain exact QCdev44")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
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
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not args.device.startswith("cuda"):
        raise RuntimeError("candidate-pool inference requires a CUDA device")
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
    import torch

    from phaxis.hair_stageb.runtime import StageBEnsemble

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    manifest = read_candidate_manifest(args.candidate_manifest)
    pending_metadata = manifest["detection_model_metadata"]
    audit_path = Path(manifest["dataset_audit_path"])
    if not audit_path.is_file() or sha256_file(audit_path) != manifest["dataset_audit_sha256"]:
        raise RuntimeError("candidate dataset audit is missing or hash-drifted")
    audit = read_json(audit_path)
    expected_ids = list(audit["excluded_val_ids"])
    dataset_root = Path(audit["dataset_root"])
    dataset_manifest = dataset_root / "manifests" / "dataset_manifest.csv"
    if (
        not dataset_manifest.is_file()
        or sha256_file(dataset_manifest)
        != manifest["identity_payload"]["training_lock"]["dataset_manifest_sha256"]
    ):
        raise RuntimeError("candidate canonical dataset manifest is missing or hash-drifted")
    with dataset_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        canonical_rows = {row["task_id"]: row for row in csv.DictReader(handle)}
    if not set(expected_ids).issubset(canonical_rows):
        raise RuntimeError("candidate-excluded QCdev44 is missing from the canonical manifest")
    rows = _rows(args.manifest)
    by_task = {row["task_id"]: row for row in rows}
    if set(by_task) != set(expected_ids):
        raise RuntimeError("inference manifest differs from the candidate-excluded QCdev44")

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
        use_amp=False,
        candidate_manifest=manifest,
        candidate_pool_mode=True,
        **shared_input_options,
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()
    records = []
    resumed = 0
    run_started = time.perf_counter()
    for task_id in expected_ids:
        row = by_task[task_id]
        image_path = Path(row["image_path"])
        image_sha256 = sha256_file(image_path)
        if image_sha256.casefold() != row["image_sha256"].casefold():
            raise RuntimeError(f"{task_id}: source image hash mismatch")
        if image_sha256.casefold() != canonical_rows[task_id]["image_sha256"].casefold():
            raise RuntimeError(f"{task_id}: source image differs from canonical dataset bytes")
        declared_scale = float(row.get("um_per_px") or row["source_um_per_px"])
        if not np.isclose(
            declared_scale,
            float(canonical_rows[task_id]["source_um_per_px"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"{task_id}: physical scale differs from canonical metadata")
        output_path = args.output / "candidate_pools" / f"{task_id}.json"
        if args.resume and output_path.is_file():
            payload = read_json(output_path)
            validate_biological_candidate_pool_payload(
                payload,
                expected_task_id=task_id,
                expected_image_sha256=image_sha256,
                expected_pending_model_metadata=pending_metadata,
            )
            resumed += 1
            seconds = 0.0
            shared_input_audit = None
        else:
            started = time.perf_counter()
            image = tifffile.imread(image_path)
            source_um_per_px = declared_scale
            prediction = ensemble.predict_biological_candidate_pool(
                image, source_um_per_px=source_um_per_px
            )
            observed_audit = getattr(ensemble, "last_shared_input_audit", None)
            shared_input_audit = (
                dict(observed_audit) if observed_audit is not None else None
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            payload = make_biological_candidate_pool_payload(
                task_id=task_id,
                source_image_sha256=image_sha256,
                source_um_per_px=source_um_per_px,
                prediction=prediction,
                pending_model_metadata=pending_metadata,
                precision_mode="fp32_locked",
            )
            atomic_write_json(output_path, payload)
        records.append(
            {
                "task_id": task_id,
                "candidate_count_at_floor": int(payload["n"]),
                "candidate_pool_payload_identity_sha256": payload[
                    "candidate_pool_payload_identity_sha256"
                ],
                "candidate_pool_payload_file_sha256": sha256_file(output_path),
                "wall_seconds_including_io": seconds,
                "resumed": bool(seconds == 0.0),
                "shared_input_runtime_audit": shared_input_audit,
            }
        )
        print(f"{task_id}: {payload['n']} biological-presence candidates", flush=True)
    torch.cuda.synchronize()
    run_seconds = time.perf_counter() - run_started
    run_identity = sha256_json(
        {
            "candidate_bundle_identity_sha256": manifest[
                "candidate_bundle_identity_sha256"
            ],
            "records": [
                [row["task_id"], row["candidate_pool_payload_identity_sha256"]]
                for row in records
            ],
        }
    )
    executed_audits = [
        row["shared_input_runtime_audit"]
        for row in records
        if row["shared_input_runtime_audit"] is not None
    ]
    path_counts = Counter(audit["runtime_path"] for audit in executed_audits)
    fallback_counts = Counter(
        audit["fallback_reason"] for audit in executed_audits
    )
    summary = {
        "schema_version": "PHAxis-StageB-train399-QCdev44-candidate-pool-run-1.0",
        "status": "completed",
        "artifact_role": "candidate_pool_not_final_prediction",
        "fusion_or_traits_consumption_allowed": False,
        "images": 44,
        "candidate_bundle_identity_sha256": manifest[
            "candidate_bundle_identity_sha256"
        ],
        "candidate_pool_run_identity_sha256": run_identity,
        "model_load_seconds": model_load_seconds,
        "batch_wall_seconds_including_io": run_seconds,
        "median_seconds_per_image": float(
            np.median([row["wall_seconds_including_io"] for row in records if not row["resumed"]])
        ) if resumed < 44 else 0.0,
        "peak_allocated_vram_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_vram_mib": float(torch.cuda.max_memory_reserved() / 2**20),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "internal_device": args.device,
        "precision_mode": "fp32_locked",
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
            "resumed_images_not_executed": resumed,
            "runtime_path_counts": dict(sorted(path_counts.items())),
            "fallback_reason_counts": dict(sorted(fallback_counts.items())),
        },
        "resumed_images": resumed,
        "checkpoint_sha256": list(ensemble.checkpoint_sha256),
        "records": records,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvidia_smi_preflight": nvidia_smi,
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
    }
    atomic_write_json(args.output / "summary.json", summary)
    print(f"completed candidate pool in {run_seconds:.3f} s")


if __name__ == "__main__":
    main()
