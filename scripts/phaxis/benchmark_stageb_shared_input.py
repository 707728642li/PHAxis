"""Reproducible GPU A/B benchmark for Stage-B shared-input inference.

This tool is intentionally separate from production inference.  It never
promotes a runtime path: any fallback or numerical difference produces a
fail-closed receipt and a non-zero exit.  Run only after the target GPU has
enough capacity for the already-loaded five-member ensemble.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.constants import (  # noqa: E402
    HAIR_MAX_INSTANCES,
    HAIR_NMS_KERNEL,
    HAIR_OUT_STRIDE,
    HAIR_ROOT_GATE_UM,
    HAIR_WORKING_UM_PER_PX,
)
from phaxis.hair_stageb.decode import decode_instances  # noqa: E402
from phaxis.io import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    sha256_json,
)


RECEIPT_SCHEMA = "PHAxis-StageB-shared-input-GPU-AB-benchmark-1.0"


def _nvidia_smi_preflight() -> dict[str, str]:
    full = subprocess.run(
        ["nvidia-smi"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    return {"full": full, "query": query}


def _parse_gpu_query(text: str) -> list[dict[str, Any]]:
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4:
            raise RuntimeError(f"unexpected nvidia-smi query row: {line}")
        rows.append(
            {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": int(fields[3]),
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return rows


def _resolve_device_mapping(
    device: str,
    cuda_visible_devices: str | None,
    physical_gpus: list[dict[str, Any]],
) -> dict[str, Any]:
    match = re.fullmatch(r"cuda:(\d+)", str(device))
    if match is None:
        raise ValueError("benchmark device must be an explicit cuda:<index>")
    internal_index = int(match.group(1))
    by_index = {row["physical_index"]: row for row in physical_gpus}
    by_uuid = {row["uuid"]: row for row in physical_gpus}

    if cuda_visible_devices is None or not cuda_visible_devices.strip():
        visible = [
            str(row["physical_index"])
            for row in sorted(
                physical_gpus, key=lambda item: item["physical_index"]
            )
        ]
        mapping_source = "nvidia_smi_physical_order_no_CVD"
    else:
        visible = [token.strip() for token in cuda_visible_devices.split(",")]
        if any(not token for token in visible) or visible == ["-1"]:
            raise RuntimeError("CUDA_VISIBLE_DEVICES exposes no usable GPU")
        mapping_source = "CUDA_VISIBLE_DEVICES"
    if internal_index >= len(visible):
        raise RuntimeError("internal CUDA index is outside the visible-device list")
    token = visible[internal_index]
    if token.isdecimal():
        physical = by_index.get(int(token))
    else:
        physical = by_uuid.get(token)
        if physical is None:
            matches = [row for uuid, row in by_uuid.items() if uuid.startswith(token)]
            physical = matches[0] if len(matches) == 1 else None
    if physical is None:
        raise RuntimeError(f"cannot resolve visible GPU token {token!r}")
    return {
        "internal_device": str(device),
        "internal_index": internal_index,
        "cuda_visible_devices": cuda_visible_devices,
        "visible_tokens": visible,
        "selected_visible_token": token,
        "mapping_source": mapping_source,
        **physical,
    }


def _selected_rows(manifest: Path, task_ids: list[str]) -> list[dict[str, Any]]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_task = {row.get("task_id"): row for row in rows}
    if len(by_task) != len(rows):
        raise RuntimeError("benchmark manifest contains duplicate task IDs")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("benchmark task IDs must be unique")
    selected = []
    for task_id in task_ids:
        if task_id not in by_task:
            raise RuntimeError(f"benchmark task is absent from manifest: {task_id}")
        row = by_task[task_id]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = manifest.parent / image_path
        image_path = image_path.resolve()
        observed_sha256 = sha256_file(image_path)
        expected_sha256 = row["image_sha256"]
        if observed_sha256.casefold() != expected_sha256.casefold():
            raise RuntimeError(f"{task_id}: source image hash mismatch")
        source_um_per_px = float(
            row.get("um_per_px") or row.get("source_um_per_px") or "nan"
        )
        if not math.isfinite(source_um_per_px) or source_um_per_px <= 0:
            raise RuntimeError(f"{task_id}: source physical scale is invalid")
        selected.append(
            {
                "task_id": task_id,
                "image_path": image_path,
                "image_sha256": observed_sha256,
                "source_um_per_px": source_um_per_px,
            }
        )
    return selected


def _array_difference(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    left = np.asarray(first)
    right = np.asarray(second)
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    exact = same_shape and same_dtype and bool(np.array_equal(left, right))
    if same_shape and left.size and np.issubdtype(left.dtype, np.number):
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        maximum = float(np.nanmax(difference))
    elif same_shape and left.size == 0:
        maximum = 0.0
    else:
        maximum = None
    return {
        "exact": exact,
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "shape_legacy": list(left.shape),
        "shape_shared": list(right.shape),
        "dtype_legacy": str(left.dtype),
        "dtype_shared": str(right.dtype),
        "max_abs_difference": maximum,
    }


def _mapping_difference(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    same_keys = set(first) == set(second)
    fields: dict[str, Any] = {}
    exact = same_keys
    maximum = 0.0
    for name in sorted(set(first) | set(second)):
        if name not in first or name not in second:
            fields[name] = {"exact": False, "missing": True}
            exact = False
            continue
        left, right = first[name], second[name]
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            difference = _array_difference(np.asarray(left), np.asarray(right))
            fields[name] = difference
            exact &= bool(difference["exact"])
            if difference["max_abs_difference"] is not None:
                maximum = max(maximum, difference["max_abs_difference"])
        else:
            equal = left == right
            fields[name] = {"exact": bool(equal), "legacy": left, "shared": right}
            exact &= bool(equal)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                maximum = max(maximum, abs(float(left) - float(right)))
    return {
        "exact": bool(exact),
        "same_keys": same_keys,
        "max_abs_difference": float(maximum),
        "fields": fields,
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}

    def statistics(name: str) -> dict[str, float]:
        values = np.asarray([run[name] for run in runs], dtype=np.float64)
        return {
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    return {
        "runs": len(runs),
        "images_per_run": int(runs[0]["images"]),
        "source_megapixels_per_run": float(runs[0]["source_megapixels"]),
        "io_decode_seconds": statistics("io_decode_seconds"),
        "inference_seconds": statistics("inference_seconds"),
        "end_to_end_seconds": statistics("end_to_end_seconds"),
        "inference_megapixels_per_second": statistics(
            "inference_megapixels_per_second"
        ),
        "end_to_end_megapixels_per_second": statistics(
            "end_to_end_megapixels_per_second"
        ),
        "peak_allocated_vram_mib": statistics("peak_allocated_vram_mib"),
        "peak_reserved_vram_mib": statistics("peak_reserved_vram_mib"),
    }


def _run_mode(ensemble, rows, *, mode: str, repetition: int, order_position: int):
    import torch

    ensemble.shared_input_acceleration = mode == "shared_input"
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    records = []
    source_megapixels = 0.0
    io_seconds = 0.0
    inference_seconds = 0.0
    started = time.perf_counter()
    for row in rows:
        io_started = time.perf_counter()
        image = tifffile.imread(row["image_path"])
        io_elapsed = time.perf_counter() - io_started
        megapixels = float(np.prod(image.shape[:2]) / 1e6)
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        prediction = ensemble.predict(
            image, source_um_per_px=row["source_um_per_px"]
        )
        torch.cuda.synchronize()
        inference_elapsed = time.perf_counter() - inference_started
        audit = dict(ensemble.last_shared_input_audit)
        records.append(
            {
                "task_id": row["task_id"],
                "source_megapixels": megapixels,
                "io_decode_seconds": io_elapsed,
                "inference_seconds": inference_elapsed,
                "detections": int(prediction["n"]),
                "shared_input_runtime_audit": audit,
            }
        )
        source_megapixels += megapixels
        io_seconds += io_elapsed
        inference_seconds += inference_elapsed
    torch.cuda.synchronize()
    end_to_end_seconds = time.perf_counter() - started
    return {
        "mode": mode,
        "repetition": repetition,
        "order_position": order_position,
        "images": len(rows),
        "source_megapixels": source_megapixels,
        "io_decode_seconds": io_seconds,
        "inference_seconds": inference_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "inference_megapixels_per_second": source_megapixels / inference_seconds,
        "end_to_end_megapixels_per_second": (
            source_megapixels / end_to_end_seconds
        ),
        "peak_allocated_vram_mib": float(
            torch.cuda.max_memory_allocated() / 2**20
        ),
        "peak_reserved_vram_mib": float(
            torch.cuda.max_memory_reserved() / 2**20
        ),
        "records": records,
    }


def _equivalence(ensemble, rows) -> list[dict[str, Any]]:
    import torch

    results = []
    for row in rows:
        image = tifffile.imread(row["image_path"])
        ensemble.shared_input_acceleration = False
        legacy_heads, legacy_geometry = ensemble._predict_heads_and_geometry(
            image, source_um_per_px=row["source_um_per_px"]
        )
        torch.cuda.synchronize()
        ensemble.shared_input_acceleration = True
        shared_heads, shared_geometry = ensemble._predict_heads_and_geometry(
            image, source_um_per_px=row["source_um_per_px"]
        )
        torch.cuda.synchronize()
        shared_audit = dict(ensemble.last_shared_input_audit)
        head_difference = _mapping_difference(legacy_heads, shared_heads)
        legacy_decoded = decode_instances(
            legacy_heads,
            um_per_px=HAIR_WORKING_UM_PER_PX,
            out_stride=HAIR_OUT_STRIDE,
            score_threshold=ensemble.score_threshold,
            nms_kernel=HAIR_NMS_KERNEL,
            max_instances=HAIR_MAX_INSTANCES,
            root_gate_um=HAIR_ROOT_GATE_UM,
        )
        shared_decoded = decode_instances(
            shared_heads,
            um_per_px=HAIR_WORKING_UM_PER_PX,
            out_stride=HAIR_OUT_STRIDE,
            score_threshold=ensemble.score_threshold,
            nms_kernel=HAIR_NMS_KERNEL,
            max_instances=HAIR_MAX_INSTANCES,
            root_gate_um=HAIR_ROOT_GATE_UM,
        )
        decoded_difference = _mapping_difference(legacy_decoded, shared_decoded)
        results.append(
            {
                "task_id": row["task_id"],
                "shared_input_runtime_audit": shared_audit,
                "geometry_exact": legacy_geometry == shared_geometry,
                "legacy_geometry": legacy_geometry,
                "shared_geometry": shared_geometry,
                "heads": head_difference,
                "decoded": decoded_difference,
            }
        )
    return results


def _receipt_identity(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_identity_sha256"] = sha256_json(result)
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--selected-model-metadata", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--expected-physical-gpu", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--shared-input-max-host-bytes", type=int, default=2 * 1024**3
    )
    parser.add_argument(
        "--shared-input-max-device-bytes", type=int, default=1 * 1024**3
    )
    parser.add_argument(
        "--shared-input-device-reserve-bytes", type=int, default=2 * 1024**3
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoint) != 5:
        raise ValueError("benchmark requires exactly five checkpoints")
    if not 1 <= len(args.task_id) <= 8:
        raise ValueError("benchmark requires an explicit 1--8 image subset")
    if args.warmup < 1 or args.repetitions < 3:
        raise ValueError("benchmark requires >=1 warmup and >=3 repetitions")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark receipt: {args.output}")
    if args.shared_input_max_host_bytes <= 0:
        raise ValueError("shared-input-max-host-bytes must be positive")
    if args.shared_input_max_device_bytes <= 0:
        raise ValueError("shared-input-max-device-bytes must be positive")
    if args.shared_input_device_reserve_bytes < 0:
        raise ValueError("shared-input-device-reserve-bytes cannot be negative")
    return args


def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    rows = _selected_rows(args.manifest.resolve(), list(args.task_id))
    preflight = _nvidia_smi_preflight()
    physical_gpus = _parse_gpu_query(preflight["query"])
    mapping = _resolve_device_mapping(
        args.device,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        physical_gpus,
    )
    if mapping["physical_index"] != args.expected_physical_gpu:
        raise RuntimeError(
            "resolved physical GPU differs from --expected-physical-gpu"
        )

    import torch

    from phaxis.hair_stageb.runtime import StageBEnsemble

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(args.device)
    device_properties = torch.cuda.get_device_properties(args.device)
    mapping["torch_visible_name"] = device_properties.name
    mapping["torch_visible_total_memory_bytes"] = int(
        device_properties.total_memory
    )
    device_free_before_load, device_total = torch.cuda.mem_get_info(args.device)
    load_started = time.perf_counter()
    ensemble = StageBEnsemble(
        args.checkpoint,
        device=args.device,
        use_amp=False,
        candidate_manifest=args.candidate_manifest,
        selected_model_metadata=args.selected_model_metadata,
        selection_receipt=args.selection_receipt,
        shared_input_acceleration=False,
        shared_input_max_host_bytes=args.shared_input_max_host_bytes,
        shared_input_max_device_bytes=args.shared_input_max_device_bytes,
        shared_input_device_reserve_bytes=(
            args.shared_input_device_reserve_bytes
        ),
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    device_free_after_load, _total_after = torch.cuda.mem_get_info(args.device)

    warmups = []
    warmup_image = tifffile.imread(rows[0]["image_path"])
    for warmup_index in range(args.warmup):
        for mode in ("legacy", "shared_input"):
            ensemble.shared_input_acceleration = mode == "shared_input"
            torch.cuda.synchronize()
            started = time.perf_counter()
            ensemble.predict(
                warmup_image,
                source_um_per_px=rows[0]["source_um_per_px"],
            )
            torch.cuda.synchronize()
            warmups.append(
                {
                    "warmup_index": warmup_index,
                    "mode": mode,
                    "seconds": time.perf_counter() - started,
                    "shared_input_runtime_audit": dict(
                        ensemble.last_shared_input_audit
                    ),
                }
            )
    if any(
        entry["mode"] == "shared_input"
        and not entry["shared_input_runtime_audit"]["used"]
        for entry in warmups
    ):
        raise RuntimeError("shared-input warmup fell back; A/B timing is invalid")

    runs = {"legacy": [], "shared_input": []}
    schedule = []
    for repetition in range(args.repetitions):
        modes = (
            ("legacy", "shared_input")
            if repetition % 2 == 0
            else ("shared_input", "legacy")
        )
        for order_position, mode in enumerate(modes):
            run = _run_mode(
                ensemble,
                rows,
                mode=mode,
                repetition=repetition,
                order_position=order_position,
            )
            runs[mode].append(run)
            schedule.append({"repetition": repetition, "mode": mode})

    equivalence = _equivalence(ensemble, rows)
    shared_audits = [
        record["shared_input_runtime_audit"]
        for run in runs["shared_input"]
        for record in run["records"]
    ] + [row["shared_input_runtime_audit"] for row in equivalence]
    shared_used = all(audit["used"] for audit in shared_audits)
    heads_exact = all(row["heads"]["exact"] for row in equivalence)
    decoded_exact = all(row["decoded"]["exact"] for row in equivalence)
    geometry_exact = all(row["geometry_exact"] for row in equivalence)
    if not shared_used:
        status = "failed_shared_path_fallback"
    elif not (heads_exact and decoded_exact and geometry_exact):
        status = "failed_numerical_equivalence"
    else:
        status = "passed_exact"

    aggregate = {
        mode: _aggregate_runs(mode_runs) for mode, mode_runs in runs.items()
    }
    legacy_aggregate = aggregate["legacy"]
    shared_aggregate = aggregate["shared_input"]
    comparison = {
        "median_inference_speedup_legacy_over_shared": (
            legacy_aggregate["inference_seconds"]["median"]
            / shared_aggregate["inference_seconds"]["median"]
        ),
        "median_end_to_end_speedup_legacy_over_shared": (
            legacy_aggregate["end_to_end_seconds"]["median"]
            / shared_aggregate["end_to_end_seconds"]["median"]
        ),
        "median_inference_megapixels_per_second_ratio_shared_over_legacy": (
            shared_aggregate["inference_megapixels_per_second"]["median"]
            / legacy_aggregate["inference_megapixels_per_second"]["median"]
        ),
        "median_peak_allocated_vram_delta_mib_shared_minus_legacy": (
            shared_aggregate["peak_allocated_vram_mib"]["median"]
            - legacy_aggregate["peak_allocated_vram_mib"]["median"]
        ),
        "median_peak_reserved_vram_delta_mib_shared_minus_legacy": (
            shared_aggregate["peak_reserved_vram_mib"]["median"]
            - legacy_aggregate["peak_reserved_vram_mib"]["median"]
        ),
    }

    input_receipt = [
        {
            "task_id": row["task_id"],
            "image_path": str(row["image_path"]),
            "image_sha256": row["image_sha256"],
            "source_um_per_px": row["source_um_per_px"],
        }
        for row in rows
    ]
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "promotion_authorized": False,
        "benchmark_contract": {
            "same_loaded_ensemble_for_both_paths": True,
            "same_ordered_input_subset": True,
            "timed_io_scope": "tifffile.imread_for_every_image_and_path",
            "model_load_excluded_from_repeated_end_to_end_runs": True,
            "inference_scope": (
                "preprocess_transfer_model_hflip_tta_stitch_and_decode"
            ),
            "cuda_synchronize_before_and_after_each_timed_inference": True,
            "alternating_ab_ba_order": True,
            "fp32": True,
            "tf32": False,
            "horizontal_flip_tta": True,
            "warmup_runs_per_path": args.warmup,
            "timed_repetitions_per_path": args.repetitions,
        },
        "input_manifest": str(args.manifest.resolve()),
        "input_manifest_sha256": sha256_file(args.manifest),
        "inputs": input_receipt,
        "checkpoint_sha256": list(ensemble.checkpoint_sha256),
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "selected_model_metadata_sha256": sha256_file(
            args.selected_model_metadata
        ),
        "selection_receipt_sha256": sha256_file(args.selection_receipt),
        "runtime_source_sha256": sha256_file(
            PROJECT_ROOT / "src/phaxis/hair_stageb/runtime.py"
        ),
        "benchmark_source_sha256": sha256_file(Path(__file__)),
        "device_mapping": mapping,
        "physical_gpu_inventory": physical_gpus,
        "nvidia_smi_preflight": preflight["full"],
        "device_memory": {
            "free_before_model_load_bytes": int(device_free_before_load),
            "free_after_model_load_bytes": int(device_free_after_load),
            "total_bytes": int(device_total),
            "shared_input_max_host_bytes": args.shared_input_max_host_bytes,
            "shared_input_max_device_bytes": args.shared_input_max_device_bytes,
            "shared_input_device_reserve_bytes": (
                args.shared_input_device_reserve_bytes
            ),
        },
        "model_load_seconds": model_load_seconds,
        "warmups": warmups,
        "schedule": schedule,
        "runs": runs,
        "aggregate": aggregate,
        "comparison": comparison,
        "equivalence": {
            "heads_exact": heads_exact,
            "decoded_exact": decoded_exact,
            "geometry_exact": geometry_exact,
            "max_head_abs_difference": max(
                row["heads"]["max_abs_difference"] for row in equivalence
            ),
            "max_decoded_abs_difference": max(
                row["decoded"]["max_abs_difference"] for row in equivalence
            ),
            "per_image": equivalence,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "blind_images_used": 0,
    }


def main() -> None:
    args = _arguments()
    try:
        receipt = _benchmark(args)
    except Exception as error:
        failure = {
            "schema_version": RECEIPT_SCHEMA,
            "status": "failed_before_complete_receipt",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "promotion_authorized": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "output": str(args.output.resolve()),
        }
        atomic_write_json(args.output, _receipt_identity(failure))
        raise
    receipt = _receipt_identity(receipt)
    atomic_write_json(args.output, receipt)
    print(receipt["receipt_identity_sha256"])
    if receipt["status"] != "passed_exact":
        raise RuntimeError(
            f"Stage-B shared-input benchmark failed closed: {receipt['status']}"
        )


if __name__ == "__main__":
    main()
