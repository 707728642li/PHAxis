#!/usr/bin/env python3
"""Execute one hash-locked, raw-to-final direct benchmark provider.

All four formal modes use this adapter and one normalized ordered source
manifest. ``--check`` is CPU-only. ``--execute`` is the sole GPU path.
Frozen-v1 code is invoked read-only in fresh output directories.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.benchmark import (  # noqa: E402
    COLD_LATENCY_MODE,
    FROZEN_V1_BENCHMARK_SYSTEM,
    GpuTelemetry,
    LATENCY_SCHEMA,
    MEASUREMENT_SCOPE,
    PHAXIS_BENCHMARK_SYSTEM,
    PRODUCTION_MODE,
    PRODUCTION_SCHEMA,
    capture_hardware_preflight,
    run_production_batch_benchmark,
)
from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json  # noqa: E402
from phaxis.workflow import build_analysis_plan, load_analysis_manifest  # noqa: E402


INTERFACE_SCHEMA = "PHAxis-formal-direct-benchmark-provider-descriptor-1.0"
CHECK_SCHEMA = "PHAxis-formal-direct-benchmark-provider-preflight-1.0"
MODES = frozenset(
    {
        "phaxis_production",
        "frozen_v1_production",
        "phaxis_sequential",
        "frozen_v1_sequential",
    }
)
PRODUCTION_MODES = frozenset({"phaxis_production", "frozen_v1_production"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TASK = re.compile(r"^[A-Za-z0-9_.-]+$")
_TRACE_FIELDS = (
    "source_unit",
    "wall_seconds",
    "megapixels",
    "io_seconds",
    "preprocess_seconds",
    "inference_seconds",
    "postprocess_seconds",
)
STRICT_PHYSICAL_GPU_ENV = "PHAXIS_REQUIRE_EXACT_PHYSICAL_GPU"


class ProviderError(RuntimeError):
    """A provider descriptor, input, execution, or receipt is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProviderError(message)


def _sealed(payload: Mapping[str, Any], field: str, *, role: str) -> str:
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field, None)
    _require(
        isinstance(observed, str)
        and _SHA256.fullmatch(observed) is not None
        and sha256_json(unsigned) == observed,
        f"{role}: {field} does not seal the complete object",
    )
    return str(observed)


def _within(root: Path, path: Path, *, role: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ProviderError(f"{role} leaves the approved root: {resolved}") from error
    return resolved


def _locked_project_file(root: Path, record: Mapping[str, Any], *, role: str) -> Path:
    relative = record.get("path")
    digest = record.get("sha256")
    _require(
        isinstance(relative, str)
        and relative
        and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts,
        f"{role}: path is invalid",
    )
    _require(
        isinstance(digest, str) and _SHA256.fullmatch(digest) is not None,
        f"{role}: SHA-256 is invalid",
    )
    path = _within(root, root / relative, role=role)
    _require(path.is_file() and not path.is_symlink(), f"{role}: file is absent or a symlink")
    _require(sha256_file(path) == digest, f"{role}: file hash drifted")
    return path


def _read_descriptor(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    descriptor_path = _within(root, path, role="provider descriptor")
    _require(
        descriptor_path.is_file() and not descriptor_path.is_symlink(),
        "provider descriptor is absent or a symlink",
    )
    payload = read_json(descriptor_path)
    _require(
        payload.get("schema_version") == INTERFACE_SCHEMA
        and payload.get("status") == "ready_hash_locked_direct_execution"
        and payload.get("measurement_scope") == MEASUREMENT_SCOPE
        and payload.get("exact_images") == 283
        and payload.get("formal_result_receipts_emitted") is True
        and payload.get("blind_images_used") == 0,
        "provider descriptor is not a ready exact283 direct interface",
    )
    _sealed(payload, "descriptor_identity_sha256", role="provider descriptor")
    entrypoints = payload.get("entrypoints")
    _require(
        isinstance(entrypoints, Mapping) and set(entrypoints) == MODES,
        "provider descriptor does not bind all four modes",
    )
    gpu_mappings: set[tuple[int, ...]] = set()
    for mode in sorted(MODES):
        binding = entrypoints[mode]
        _require(isinstance(binding, Mapping), f"{mode}: descriptor binding is invalid")
        _locked_project_file(root, binding, role=f"{mode} entrypoint")
        _require(
            binding.get("warmup_runs") == 0
            and binding.get("measured_repeats") == 1,
            f"{mode}: formal exact283 contract must be warmup=0/repeats=1",
        )
        expected_system = (
            PHAXIS_BENCHMARK_SYSTEM
            if mode.startswith("phaxis_")
            else FROZEN_V1_BENCHMARK_SYSTEM
        )
        expected_mode = PRODUCTION_MODE if mode in PRODUCTION_MODES else COLD_LATENCY_MODE
        _require(binding.get("benchmark_system") == expected_system, f"{mode}: system identity drifted")
        _require(binding.get("benchmark_mode") == expected_mode, f"{mode}: benchmark mode drifted")
        physical_gpus = binding.get("physical_gpus")
        _require(
            isinstance(physical_gpus, list)
            and physical_gpus
            and all(isinstance(value, int) and value >= 0 for value in physical_gpus)
            and len(set(physical_gpus)) == len(physical_gpus),
            f"{mode}: physical GPU mapping is invalid",
        )
        gpu_mappings.add(tuple(physical_gpus))
        closure = binding.get("implementation_closure")
        _require(isinstance(closure, list) and closure, f"{mode}: implementation closure is absent")
        for index, record in enumerate(closure):
            _require(isinstance(record, Mapping), f"{mode}: invalid closure record {index}")
            _locked_project_file(root, record, role=f"{mode} closure[{index}]")
    _require(
        payload.get("same_physical_gpu_required_for_all_modes") is True
        and gpu_mappings == {(0,)},
        "provider descriptor must bind all four modes to frozen-v1 physical GPU0",
    )
    inputs: dict[str, Path] = {}
    static_inputs = payload.get("static_inputs")
    _require(isinstance(static_inputs, Mapping), "provider static_inputs are absent")
    for name, record in static_inputs.items():
        _require(
            isinstance(name, str) and name and isinstance(record, Mapping),
            "provider static input record is invalid",
        )
        inputs[name] = _locked_project_file(root, record, role=f"static input {name}")
    return payload, inputs


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), [dict(row) for row in reader]


def _source_manifest_lock(path: Path, image_root: Path) -> dict[str, Any]:
    manifest = path.resolve()
    approved = image_root.resolve()
    _require(manifest.is_file() and not manifest.is_symlink(), "source manifest is absent or a symlink")
    _require(approved.is_dir(), "approved image root is absent")
    fields, raw_rows = _read_csv(manifest)
    required = {"task_id", "image_path", "image_sha256", "um_per_px", "source_megapixels"}
    _require(required <= set(fields), f"source manifest columns are incomplete: {sorted(required - set(fields))}")
    _require(len(raw_rows) == 283, "formal direct benchmark source manifest is not exact283")
    rows: list[dict[str, Any]] = []
    seen_tasks: set[str] = set()
    seen_images: set[str] = set()
    for index, raw in enumerate(raw_rows, start=2):
        task = str(raw.get("task_id", "")).strip()
        digest = str(raw.get("image_sha256", "")).strip().casefold()
        _require(
            _SAFE_TASK.fullmatch(task) is not None and task not in seen_tasks,
            f"invalid or duplicate task at source row {index}",
        )
        _require(
            _SHA256.fullmatch(digest) is not None and digest not in seen_images,
            f"invalid or duplicate image hash at source row {index}",
        )
        source = Path(str(raw.get("image_path", "")))
        if not source.is_absolute():
            source = manifest.parent / source
        source = _within(approved, source, role=f"source image {task}")
        _require(source.is_file() and not source.is_symlink(), f"source image is absent: {task}")
        _require(sha256_file(source) == digest, f"source image byte hash differs: {task}")
        try:
            scale = float(raw["um_per_px"])
            megapixels = float(raw["source_megapixels"])
        except (TypeError, ValueError) as error:
            raise ProviderError(f"source geometry is invalid: {task}") from error
        _require(
            math.isfinite(scale)
            and scale > 0
            and math.isfinite(megapixels)
            and megapixels > 0,
            f"source geometry is non-positive: {task}",
        )
        seen_tasks.add(task)
        seen_images.add(digest)
        rows.append(
            {
                "task_id": task,
                "image_path": str(source),
                "image_sha256": digest,
                "um_per_px": scale,
                "source_megapixels": megapixels,
                "raw": raw,
            }
        )
    tasks = [str(row["task_id"]) for row in rows]
    # The production manifest order is itself the sealed authority.  It need
    # not be lexicographic (the existing exact283 manifest preserves the
    # preregistered analysis order); every formal mode must preserve this
    # identical order and its ordered-set identity.
    image_locks = [
        {
            "task_id": row["task_id"],
            "image_sha256": row["image_sha256"],
            "um_per_px": row["um_per_px"],
        }
        for row in rows
    ]
    return {
        "path": manifest,
        "sha256": sha256_file(manifest),
        "rows": rows,
        "source_image_lock_identity_sha256": sha256_json(image_locks),
        "source_units_in_order": tasks,
        "source_unit_ordered_set_identity_sha256": sha256_json(tasks),
        "megapixels": sum(float(row["source_megapixels"]) for row in rows),
    }


def _analysis_lock(path: Path, source: Mapping[str, Any], workflow_output: Path) -> dict[str, Any]:
    manifest = path.resolve()
    load_analysis_manifest(manifest)
    plan = build_analysis_plan(manifest, output=workflow_output, review_overlays=False)
    _require(plan.get("tasks") == 283, "analysis workflow manifest is not exact283")
    _require(
        plan.get("task_ids") == source["source_units_in_order"],
        "analysis workflow/source task order differs",
    )
    _require(
        plan.get("task_identity_sha256")
        == source["source_image_lock_identity_sha256"],
        "analysis workflow/source image lock differs",
    )
    return {"path": manifest, "sha256": sha256_file(manifest), "plan": plan}


def _write_csv(
    path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_root: Path,
    name: str,
) -> float:
    log_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        check=False,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    wall = time.perf_counter() - started
    (log_root / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (log_root / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    _require(
        completed.returncode == 0,
        f"provider subprocess failed: {name}; exit={completed.returncode}; "
        f"stderr={completed.stderr[-1000:]}",
    )
    return wall


def _hash_existing(paths: Mapping[str, Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, path in sorted(paths.items()):
        _require(path.is_file(), f"provider final artifact is absent: {name}: {path}")
        result[name] = sha256_file(path)
    return result


def _frozen_raw_manifest(rows: Sequence[Mapping[str, Any]], destination: Path) -> None:
    _write_csv(
        destination,
        (
            "image_id",
            "input_path",
            "analysis_mode",
            "source_um_per_px",
            "source_scale_provenance",
        ),
        [
            {
                "image_id": row["task_id"],
                "input_path": row["image_path"],
                "analysis_mode": "sparse_instance",
                "source_um_per_px": repr(float(row["um_per_px"])),
                "source_scale_provenance": "raw_image_classical_train399_locked",
            }
            for row in rows
        ],
    )


def _run_frozen_unit(
    *,
    rows: Sequence[Mapping[str, Any]],
    root: Path,
    python: Path,
    acquisition_gate: Path,
    runtime_config: Path,
    env: Mapping[str, str],
    name: str,
) -> tuple[dict[str, float], dict[str, str]]:
    root.mkdir(parents=True, exist_ok=False)
    raw = root / "raw_manifest.csv"
    prepared = root / "prepared"
    final = root / "final"
    _frozen_raw_manifest(rows, raw)
    preprocess_seconds = _run(
        [
            str(python),
            "-m",
            "rhizoweave.cli",
            "prepare",
            "--input-manifest",
            str(raw),
            "--output",
            str(prepared),
            "--acquisition-gate",
            str(acquisition_gate),
            "--no-ocr",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        log_root=root / "logs",
        name="prepare",
    )
    prepared_manifest = prepared / "manifests" / "sparse_instance.csv"
    quality = prepared / "acquisition_quality.csv"
    _require(
        prepared_manifest.is_file() and quality.is_file(),
        f"{name}: frozen-v1 preparation output is incomplete",
    )
    inference_seconds = _run(
        [
            str(python),
            str(PROJECT_ROOT / "scripts/run_rhpheno_dual_mode_v5.py"),
            "--mode",
            "sparse_instance",
            "--manifest",
            str(prepared_manifest),
            "--output",
            str(final),
            "--runtime-config",
            str(runtime_config),
            "--quality-csv",
            str(quality),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        log_root=root / "logs",
        name="analyze",
    )
    report = read_json(final / "rhpheno_run.json")
    _require(
        report.get("status") == "complete"
        and report.get("blind_test_images_used") == 0,
        f"{name}: frozen-v1 final report is incomplete or blind-tainted",
    )
    hashes = _hash_existing(
        {
            "rhpheno_run": final / "rhpheno_run.json",
            "summary": final / "summary.json",
            "per_image_traits": final / "per_image.csv",
            "distal_profiles": final / "distance_bins.csv",
        }
    )
    return (
        {
            "io_seconds": 0.0,
            "preprocess_seconds": preprocess_seconds,
            "inference_seconds": inference_seconds,
            "postprocess_seconds": 0.0,
        },
        hashes,
    )


def _ref_path(ref: Mapping[str, Any], base: Path) -> Path:
    path = Path(str(ref["path"]))
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _filter_one_csv(source: Path, destination: Path, task_id: str) -> None:
    fields, rows = _read_csv(source)
    matches = [
        row
        for row in rows
        if task_id
        in {
            str(row.get(name, "")).strip()
            for name in ("task_id", "image_id", "biological_unit_id", "sample_code")
        }
    ]
    _require(
        len(matches) == 1,
        f"cannot derive exactly one source row for {task_id}: {source}",
    )
    _write_csv(destination, fields, matches)


def _derived_deployment_lock(
    source_lock: Path, manifest: Path, destination: Path
) -> None:
    payload = read_json(source_lock)
    _fields, rows = _read_csv(manifest)
    _require(len(rows) == 1, "derived deployment manifest is not one source")
    row = rows[0]
    projection = [
        {
            "task_id": row["task_id"],
            "image_relpath": Path(row["image_relpath"]).as_posix(),
            "image_sha256": row["image_sha256"].casefold(),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "um_per_px": float(row["um_per_px"]),
        }
    ]
    manifest_sha = sha256_file(manifest)
    payload.update(
        {
            "samples": 1,
            "manifest": manifest.name,
            "manifest_sha256": manifest_sha,
            "deployment_identity_sha256": sha256_json(
                {
                    "schema_version": "RHAxis-NextGen-deployment-identity-1.0",
                    "manifest_sha256": manifest_sha,
                    "source_qc_lock_identity_sha256": payload[
                        "source_qc_lock_identity_sha256"
                    ],
                    "samples": projection,
                }
            ),
            "benchmark_derived_single_source": True,
            "blind_images_used": 0,
        }
    )
    atomic_write_json(destination, payload)


def _single_source_analysis_manifest(
    full_manifest: Path,
    task_id: str,
    destination: Path,
    physical_gpu: int,
) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    payload = read_json(full_manifest)
    payload.pop("manifest_identity_sha256", None)
    base = full_manifest.parent
    root = payload["root_provider"]
    for name in ("input_manifest", "deployment_metadata"):
        source = _ref_path(root[name], base)
        target = destination / f"root_{name}.csv"
        _filter_one_csv(source, target, task_id)
        root[name] = {"path": str(target), "sha256": sha256_file(target)}
    source_deployment = _ref_path(root["deployment_manifest"], base)
    target_deployment = destination / "deployment_manifest.csv"
    _filter_one_csv(source_deployment, target_deployment, task_id)
    target_lock = destination / "deployment_manifest_lock.json"
    _derived_deployment_lock(
        _ref_path(root["deployment_lock"], base), target_deployment, target_lock
    )
    root["deployment_manifest"] = {
        "path": str(target_deployment),
        "sha256": sha256_file(target_deployment),
    }
    root["deployment_lock"] = {
        "path": str(target_lock),
        "sha256": sha256_file(target_lock),
    }
    root.pop("reference_registry", None)
    root.update(
        {
            "v1_physical_gpus": [physical_gpu],
            "q8_physical_gpus": [physical_gpu],
            "v1_shards": 1,
            "v20_shards": 1,
            "q8_shards": 1,
            "v1_concurrency": 1,
            "v20_concurrency": 1,
            "q8_concurrency": 1,
        }
    )
    stageb_source = _ref_path(payload["stageb"]["input_manifest"], base)
    stageb_target = destination / "stageb_manifest.csv"
    _filter_one_csv(stageb_source, stageb_target, task_id)
    payload["stageb"]["input_manifest"] = {
        "path": str(stageb_target),
        "sha256": sha256_file(stageb_target),
    }
    payload["stageb"]["physical_gpu"] = physical_gpu
    payload["stageb"]["internal_device"] = "cuda:0"
    traits_source = _ref_path(payload["traits"]["metadata_csv"], base)
    traits_target = destination / "traits_metadata.csv"
    _filter_one_csv(traits_source, traits_target, task_id)
    payload["traits"]["metadata_csv"] = {
        "path": str(traits_target),
        "sha256": sha256_file(traits_target),
    }
    payload["status"] = "ready_hash_locked_single_source_benchmark_derivative"
    payload["benchmark_derivation"] = {
        "source_exact283_workflow_manifest_sha256": sha256_file(full_manifest),
        "task_id": task_id,
        "cold_cli": True,
        "warmup_runs": 0,
        "measured_repeats": 1,
    }
    payload["manifest_identity_sha256"] = sha256_json(payload)
    path = destination / "analysis_workflow_manifest.json"
    atomic_write_json(path, payload)
    return path


def _run_phaxis_sequential(
    *,
    analysis_manifest: Path,
    rows: Sequence[Mapping[str, Any]],
    root: Path,
    python: Path,
    env: Mapping[str, str],
    physical_gpu: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    trace: list[dict[str, Any]] = []
    output_hashes: dict[str, str] = {}
    root.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(rows):
        task = str(row["task_id"])
        unit_started = time.perf_counter()
        unit = root / f"{index:03d}_{task}"
        unit.mkdir()
        derived = _single_source_analysis_manifest(
            analysis_manifest, task, unit / "inputs", physical_gpu
        )
        preprocess_seconds = time.perf_counter() - unit_started
        final = unit / "analysis"
        subprocess_wall = _run(
            [
                str(python),
                "-m",
                "phaxis.cli",
                "analyze",
                "--manifest",
                str(derived),
                "--output",
                str(final),
                "--execute",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            log_root=unit / "logs",
            name="phaxis_analyze",
        )
        state = read_json(final / "workflow_state.json")
        attempts = state.get("execution_attempts")
        _require(
            state.get("status") == "completed"
            and isinstance(attempts, list)
            and attempts,
            f"{task}: PHAxis workflow state is incomplete",
        )
        latest = attempts[-1]
        _require(
            latest.get("fresh_direct_benchmark_eligible") is True
            and latest.get("resume_or_cache_used") is False,
            f"{task}: PHAxis sequential run is not fresh",
        )
        profile_summary = final / "distal_axis_profiles" / "summary.json"
        _require(profile_summary.is_file(), f"{task}: final profile is absent")
        output_hashes[f"{task}.profile_summary"] = sha256_file(profile_summary)
        wall = time.perf_counter() - unit_started
        _require(
            preprocess_seconds + subprocess_wall <= wall * 1.02,
            f"{task}: PHAxis component walls exceed raw-to-final wall",
        )
        trace.append(
            {
                "source_unit": task,
                "wall_seconds": wall,
                "megapixels": float(row["source_megapixels"]),
                "io_seconds": 0.0,
                "preprocess_seconds": preprocess_seconds,
                "inference_seconds": subprocess_wall,
                # Include validation, final-profile visibility, hashing and
                # orchestration in the same outer wall used by frozen-v1.
                "postprocess_seconds": max(
                    0.0, wall - preprocess_seconds - subprocess_wall
                ),
            }
        )
    return trace, output_hashes


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _common_summary(
    *,
    mode: str,
    source: Mapping[str, Any],
    hardware: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    descriptor_sha256: str,
    physical_gpus: Sequence[int],
    cvd: str,
) -> dict[str, Any]:
    return {
        "benchmark_system": (
            PHAXIS_BENCHMARK_SYSTEM
            if mode.startswith("phaxis_")
            else FROZEN_V1_BENCHMARK_SYSTEM
        ),
        "status": "completed_direct_full283",
        "benchmark_scope_class": "full_workflow",
        "measurement_scope": MEASUREMENT_SCOPE,
        "images": 283,
        "n_images": 283,
        "pixels": int(round(float(source["megapixels"]) * 1_000_000)),
        "megapixels": float(source["megapixels"]),
        "includes_io": True,
        "includes_preprocess": True,
        "includes_stitching_fusion_traits_profiles": True,
        "peak_vram_mib": float(telemetry["peak_vram_mib"]),
        "mean_gpu_utilization_pct": float(telemetry["mean_gpu_utilization_pct"]),
        "gpu_telemetry": dict(telemetry),
        "hardware": dict(hardware["hardware"]),
        "hardware_identity_sha256": hardware["hardware_identity_sha256"],
        "nvidia_smi_preflight": dict(hardware["nvidia_smi_preflight"]),
        "nvidia_smi_preflight_identity_sha256": hardware[
            "nvidia_smi_preflight_identity_sha256"
        ],
        "source_manifest_sha256": source["sha256"],
        "source_image_lock_identity_sha256": source[
            "source_image_lock_identity_sha256"
        ],
        "source_units_in_order": list(source["source_units_in_order"]),
        "source_unit_ordered_set_identity_sha256": source[
            "source_unit_ordered_set_identity_sha256"
        ],
        "provider_descriptor_sha256": descriptor_sha256,
        "physical_gpu_mapping": list(physical_gpus),
        "cuda_visible_devices_by_stage": {"direct_provider": cvd},
        "warmup_runs": 0,
        "measured_repeats": 1,
        "fresh_direct_run": True,
        "resume_or_cache_used": False,
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        "rootcap_region_metric": False,
    }


def _production_summary(
    *,
    mode: str,
    source: Mapping[str, Any],
    hardware: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    descriptor_sha256: str,
    wall: float,
    stages: Sequence[Mapping[str, Any]],
    output_hashes: Mapping[str, str],
    input_hashes: Mapping[str, str],
    config_hashes: Mapping[str, str],
    physical_gpus: Sequence[int],
    cvd: str,
) -> dict[str, Any]:
    stage_rows = [dict(row) for row in stages]
    attributed = sum(float(row["wall_seconds"]) for row in stage_rows)
    _require(
        attributed <= wall * 1.02,
        "direct production stage walls exceed outer batch wall",
    )
    for row in stage_rows:
        row["fraction_of_batch_wall"] = float(row["wall_seconds"]) / wall
    payload = {
        "schema_version": PRODUCTION_SCHEMA,
        **_common_summary(
            mode=mode,
            source=source,
            hardware=hardware,
            telemetry=telemetry,
            descriptor_sha256=descriptor_sha256,
            physical_gpus=physical_gpus,
            cvd=cvd,
        ),
        "benchmark_mode": PRODUCTION_MODE,
        "timing_granularity": "direct_batch_stage_wall",
        "per_image_latency_reported": False,
        "per_image_latency_reason": (
            "batch/full-workflow wall is not divided across images"
        ),
        "batch_wall_seconds": wall,
        "images_per_min": 283 * 60.0 / wall,
        "megapixels_per_second": float(source["megapixels"]) / wall,
        "stage_timing_semantics": "nonoverlapping_wall_components",
        "stage_timings": stage_rows,
        "attributed_stage_wall_seconds": attributed,
        "unattributed_orchestration_wall_seconds": max(0.0, wall - attributed),
        "unattributed_orchestration_fraction": max(0.0, wall - attributed) / wall,
        "input_hashes": dict(input_hashes),
        "config_hashes": dict(config_hashes),
        "output_hashes": dict(output_hashes),
        "official_output_hashes": dict(output_hashes),
    }
    payload["summary_identity_sha256"] = sha256_json(payload)
    return payload


def _sequential_summary(
    *,
    mode: str,
    source: Mapping[str, Any],
    hardware: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    descriptor_sha256: str,
    trace: Sequence[Mapping[str, Any]],
    trace_path: Path,
    output_hashes: Mapping[str, str],
    input_hashes: Mapping[str, str],
    config_hashes: Mapping[str, str],
    physical_gpus: Sequence[int],
    cvd: str,
) -> dict[str, Any]:
    wall_values = [float(row["wall_seconds"]) for row in trace]
    component_totals = {
        field: sum(float(row[field]) for row in trace)
        for field in _TRACE_FIELDS[3:]
    }
    measured = sum(wall_values)
    payload = {
        "schema_version": LATENCY_SCHEMA,
        **_common_summary(
            mode=mode,
            source=source,
            hardware=hardware,
            telemetry=telemetry,
            descriptor_sha256=descriptor_sha256,
            physical_gpus=physical_gpus,
            cvd=cvd,
        ),
        "benchmark_mode": COLD_LATENCY_MODE,
        "latency_mode": COLD_LATENCY_MODE,
        "timing_granularity": "direct_per_source_raw_to_final_profile",
        "stage_timing_semantics": "nonoverlapping_wall_components",
        "measurement_method": (
            "direct_per_source_perf_counter_start_to_final_profile_visibility"
        ),
        "persistent_worker_and_models": False,
        "model_or_process_startup_per_source": True,
        "startup_included_in_per_image_wall": True,
        "one_time_startup_seconds": 0.0,
        "median_seconds_per_image": float(statistics.median(wall_values)),
        "p95_seconds_per_image": _quantile(wall_values, 0.95),
        "direct_sequential_wall_seconds": measured,
        "images_per_min": 283 * 60.0 / measured,
        "megapixels_per_second": float(source["megapixels"]) / measured,
        "component_total_seconds": component_totals,
        "component_fraction_of_sequential_source_wall": {
            field: value / measured for field, value in component_totals.items()
        },
        "component_boundary_note": (
            "Read-only monolithic model entrypoints keep their full subprocess "
            "wall in inference_seconds; the frozen public preparer is timed "
            "separately when it exposes a real boundary."
        ),
        "per_image_csv_sha256": sha256_file(trace_path),
        "input_hashes": dict(input_hashes),
        "config_hashes": dict(config_hashes),
        "output_hashes": dict(output_hashes),
    }
    payload["summary_identity_sha256"] = sha256_json(payload)
    return payload


def preflight_provider(
    *,
    project_root: Path,
    producer_interface: Path,
    mode: str | None = None,
    source_manifest: Path | None = None,
    image_root: Path | None = None,
    analysis_manifest: Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    descriptor, static_inputs = _read_descriptor(root, producer_interface)
    if mode is not None:
        _require(mode in MODES, f"unsupported direct benchmark mode: {mode}")
    source = None
    analysis = None
    if source_manifest is not None:
        _require(image_root is not None, "source preflight requires --image-root")
        source = _source_manifest_lock(source_manifest, image_root)
    if analysis_manifest is not None:
        _require(source is not None, "analysis preflight requires source preflight")
        analysis = _analysis_lock(
            analysis_manifest, source, Path(".provider-plan-output").resolve()
        )
    payload: dict[str, Any] = {
        "schema_version": CHECK_SCHEMA,
        "status": "ready_cpu_preflight_only",
        "descriptor_sha256": sha256_file(producer_interface),
        "descriptor_identity_sha256": descriptor["descriptor_identity_sha256"],
        "mode": mode,
        "source_manifest_sha256": source["sha256"] if source else None,
        "analysis_manifest_sha256": analysis["sha256"] if analysis else None,
        "static_input_sha256": {
            name: sha256_file(path) for name, path in sorted(static_inputs.items())
        },
        "gpu_program_started": False,
        "nvidia_smi_called": False,
        "formal_result_receipt": False,
        "blind_images_used": 0,
    }
    payload["preflight_identity_sha256"] = sha256_json(payload)
    return payload


def _validate_q8_exact_device_bindings(
    workflow_root: Path,
    *,
    physical_gpus: Sequence[int],
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every PHAxis Q8 shard to the outer benchmark GPU UUID.

    The root provider normally permits a capacity-safe fallback.  Formal
    same-hardware benchmarks deliberately do not: an index-only outer receipt
    must never conceal a Q8 shard that actually ran on another physical card.
    """

    gpu_rows = hardware.get("gpus")
    _require(isinstance(gpu_rows, list), "benchmark hardware GPU inventory is absent")
    expected = {
        int(row["physical_index"]): str(row["uuid"])
        for row in gpu_rows
        if isinstance(row, Mapping)
        and row.get("physical_index") is not None
        and row.get("uuid") is not None
    }
    _require(
        set(expected) == set(map(int, physical_gpus)),
        "benchmark hardware inventory differs from the declared physical GPUs",
    )
    paths = sorted(workflow_root.rglob("q8_device_selection.json"))
    _require(paths, "PHAxis workflow published no Q8 device-selection receipts")
    bindings: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        requested = payload.get("requested_physical_gpu")
        selected = payload.get("selected_physical_gpu")
        _require(
            payload.get("exact_physical_gpu_required") is True,
            f"Q8 exact-device mode was not active: {path}",
        )
        _require(
            isinstance(requested, int)
            and selected == requested
            and requested in expected,
            f"Q8 shard left the formal physical GPU mapping: {path}",
        )
        snapshot = payload.get("gpu_snapshot")
        _require(isinstance(snapshot, list), f"Q8 GPU snapshot is absent: {path}")
        matches = [
            row
            for row in snapshot
            if isinstance(row, Mapping) and row.get("index") == requested
        ]
        _require(len(matches) == 1, f"Q8 selected GPU snapshot is ambiguous: {path}")
        observed_uuid = str(matches[0].get("uuid", ""))
        _require(
            observed_uuid == expected[requested],
            f"Q8 selected GPU UUID differs from outer benchmark preflight: {path}",
        )
        bindings.append(
            {
                "selection_receipt_sha256": sha256_file(path),
                "requested_physical_gpu": requested,
                "selected_physical_gpu": selected,
                "physical_gpu_uuid": observed_uuid,
            }
        )
    result = {
        "status": "passed_exact_physical_gpu_and_uuid",
        "selection_receipts": len(bindings),
        "bindings": bindings,
    }
    result["binding_identity_sha256"] = sha256_json(result)
    return result


def run_provider(
    *,
    project_root: Path,
    producer_interface: Path,
    mode: str,
    source_manifest: Path,
    image_root: Path,
    analysis_manifest: Path,
    workflow_output: Path,
    output: Path,
    python_executable: Path,
    cuda_visible_devices: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    _require(root == PROJECT_ROOT.resolve(), "provider must run from the sealed project root")
    _require(mode in MODES, f"unsupported direct benchmark mode: {mode}")
    _require(
        re.fullmatch(r"\d+(?:,\d+)*", cuda_visible_devices) is not None,
        "CUDA_VISIBLE_DEVICES mapping is invalid",
    )
    physical_gpus = tuple(int(value) for value in cuda_visible_devices.split(","))
    _require(
        len(physical_gpus) == len(set(physical_gpus)),
        "CUDA_VISIBLE_DEVICES contains duplicate physical GPUs",
    )
    python = python_executable.resolve()
    _require(python.is_file(), "provider Python executable is absent")
    destination = _within(root, output, role="benchmark output")
    workflow_destination = _within(root, workflow_output, role="workflow output")
    _require(
        not destination.exists() and not workflow_destination.exists(),
        "direct benchmark requires new output/workflow directories",
    )
    descriptor, static_inputs = _read_descriptor(root, producer_interface)
    mode_binding = descriptor["entrypoints"][mode]
    allowed = mode_binding.get("physical_gpus")
    _require(
        isinstance(allowed, list) and list(physical_gpus) == allowed,
        f"{mode}: GPU mapping differs from the sealed descriptor",
    )
    source = _source_manifest_lock(source_manifest, image_root)
    analysis = _analysis_lock(analysis_manifest, source, workflow_destination)
    if mode.startswith("phaxis_"):
        plan_gpus: set[int] = set()
        for stage in analysis["plan"].get("stages", []):
            estimated = stage.get("estimated_gpu", {}) if isinstance(stage, Mapping) else {}
            for field in ("v1_physical_gpus", "q8_physical_gpus"):
                values = estimated.get(field, [])
                if isinstance(values, list):
                    plan_gpus.update(int(value) for value in values)
            if estimated.get("physical_gpu") is not None:
                plan_gpus.add(int(estimated["physical_gpu"]))
        _require(
            tuple(sorted(plan_gpus)) == tuple(sorted(physical_gpus)),
            f"{mode}: analysis manifest leaves the sealed physical GPU mapping",
        )
    descriptor_sha = sha256_file(producer_interface)
    # This adapter is the formal direct provider, so its physical mapping is
    # part of the benchmark authority rather than a dynamic scheduling hint.
    # Child root-provider stages inherit this before importing CUDA.
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        STRICT_PHYSICAL_GPU_ENV: "1",
    }

    workflow_attempt = workflow_destination.parent / (
        f".{workflow_destination.name}.provider-attempt-{os.getpid()}"
    )
    output_attempt = destination.parent / (
        f".{destination.name}.provider-attempt-{os.getpid()}"
    )
    _require(
        not workflow_attempt.exists() and not output_attempt.exists(),
        "provider attempt directory already exists",
    )

    if mode == "phaxis_production":
        old_strict = os.environ.get(STRICT_PHYSICAL_GPU_ENV)
        os.environ[STRICT_PHYSICAL_GPU_ENV] = "1"
        try:
            summary = run_production_batch_benchmark(
                manifest=analysis["path"],
                workflow_output=workflow_attempt,
                benchmark_output=output_attempt,
            )
        finally:
            if old_strict is None:
                os.environ.pop(STRICT_PHYSICAL_GPU_ENV, None)
            else:
                os.environ[STRICT_PHYSICAL_GPU_ENV] = old_strict
        _require(
            summary.get("source_image_lock_identity_sha256")
            == source["source_image_lock_identity_sha256"],
            "PHAxis production source identity differs from the shared manifest",
        )
        rewritten = deepcopy(summary)
        rewritten.pop("summary_identity_sha256", None)
        rewritten["provider_native_source_manifest_sha256"] = summary.get(
            "source_manifest_sha256"
        )
        rewritten["source_manifest_sha256"] = source["sha256"]
        rewritten["provider_descriptor_sha256"] = descriptor_sha
        rewritten["warmup_runs"] = 0
        rewritten["measured_repeats"] = 1
        rewritten["q8_exact_device_binding"] = _validate_q8_exact_device_bindings(
            workflow_attempt,
            physical_gpus=physical_gpus,
            hardware=rewritten["hardware"],
        )
        rewritten["summary_identity_sha256"] = sha256_json(rewritten)
        atomic_write_json(output_attempt / "runtime_summary.json", rewritten)
        workflow_destination.parent.mkdir(parents=True, exist_ok=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(workflow_attempt, workflow_destination)
        os.replace(output_attempt, destination)
        return rewritten

    workflow_attempt.mkdir(parents=True)
    output_attempt.mkdir(parents=True)
    hardware = capture_hardware_preflight(physical_gpus)
    telemetry = GpuTelemetry(physical_gpus)
    telemetry.start()
    started = time.perf_counter()
    try:
        if mode == "frozen_v1_production":
            gate = static_inputs.get("frozen_v1_acquisition_gate")
            runtime = static_inputs.get("frozen_v1_runtime_config")
            _require(gate is not None and runtime is not None, "frozen-v1 inputs absent")
            components, output_hashes = _run_frozen_unit(
                rows=source["rows"],
                root=workflow_attempt / "frozen_v1_full283",
                python=python,
                acquisition_gate=gate,
                runtime_config=runtime,
                env=environment,
                name=mode,
            )
            wall = time.perf_counter() - started
            stage_rows = [
                {
                    "stage": "raw_image_prepare",
                    "wall_seconds": components["preprocess_seconds"],
                    "cache_status": "executed_fresh",
                },
                {
                    "stage": "frozen_v1_inference_traits_profiles",
                    "wall_seconds": components["inference_seconds"],
                    "cache_status": "executed_fresh",
                },
            ]
            trace = None
        elif mode == "phaxis_sequential":
            trace, output_hashes = _run_phaxis_sequential(
                analysis_manifest=analysis["path"],
                rows=source["rows"],
                root=workflow_attempt / "phaxis_sequential",
                python=python,
                env=environment,
                physical_gpu=physical_gpus[0],
            )
            wall = time.perf_counter() - started
            stage_rows = []
        else:
            gate = static_inputs.get("frozen_v1_acquisition_gate")
            runtime = static_inputs.get("frozen_v1_runtime_config")
            _require(gate is not None and runtime is not None, "frozen-v1 inputs absent")
            trace = []
            output_hashes = {}
            seq_root = workflow_attempt / "frozen_v1_sequential"
            seq_root.mkdir()
            for index, row in enumerate(source["rows"]):
                unit_started = time.perf_counter()
                components, hashes = _run_frozen_unit(
                    rows=[row],
                    root=seq_root / f"{index:03d}_{row['task_id']}",
                    python=python,
                    acquisition_gate=gate,
                    runtime_config=runtime,
                    env=environment,
                    name=str(row["task_id"]),
                )
                # The latency authority is the complete raw-manifest-to-final-
                # profile interval, not the sum of two convenient subprocess
                # timers.  This outer wall also includes manifest creation,
                # validation, hashing and orchestration between the exposed
                # prepare/analyse boundaries.
                row_wall = time.perf_counter() - unit_started
                _require(
                    sum(components.values()) <= row_wall * 1.02,
                    f"{row['task_id']}: frozen-v1 component walls exceed raw-to-final wall",
                )
                trace.append(
                    {
                        "source_unit": row["task_id"],
                        "wall_seconds": row_wall,
                        "megapixels": row["source_megapixels"],
                        **components,
                    }
                )
                output_hashes.update(
                    {
                        f"{row['task_id']}.{name}": digest
                        for name, digest in hashes.items()
                    }
                )
            wall = time.perf_counter() - started
            stage_rows = []
    finally:
        telemetry_receipt = telemetry.stop()

    input_hashes = {
        "shared_source_manifest": source["sha256"],
        "analysis_workflow_manifest": analysis["sha256"],
        "provider_descriptor": descriptor_sha,
    }
    config_hashes = {
        name: sha256_file(path) for name, path in sorted(static_inputs.items())
    }
    q8_binding = None
    if mode == "phaxis_sequential":
        q8_binding = _validate_q8_exact_device_bindings(
            workflow_attempt,
            physical_gpus=physical_gpus,
            hardware=hardware["hardware"],
        )
    if mode in PRODUCTION_MODES:
        summary = _production_summary(
            mode=mode,
            source=source,
            hardware=hardware,
            telemetry=telemetry_receipt,
            descriptor_sha256=descriptor_sha,
            wall=wall,
            stages=stage_rows,
            output_hashes=output_hashes,
            input_hashes=input_hashes,
            config_hashes=config_hashes,
            physical_gpus=physical_gpus,
            cvd=cuda_visible_devices,
        )
    else:
        _require(trace is not None and len(trace) == 283, "sequential trace is not exact283")
        trace_path = output_attempt / "runtime_per_image.csv"
        _write_csv(trace_path, _TRACE_FIELDS, trace)
        summary = _sequential_summary(
            mode=mode,
            source=source,
            hardware=hardware,
            telemetry=telemetry_receipt,
            descriptor_sha256=descriptor_sha,
            trace=trace,
            trace_path=trace_path,
            output_hashes=output_hashes,
            input_hashes=input_hashes,
            config_hashes=config_hashes,
            physical_gpus=physical_gpus,
            cvd=cuda_visible_devices,
        )
    telemetry_path = output_attempt / "gpu_telemetry.json"
    hardware_path = output_attempt / "hardware_preflight.json"
    atomic_write_json(telemetry_path, dict(telemetry_receipt))
    atomic_write_json(hardware_path, dict(hardware["nvidia_smi_preflight"]))
    summary.pop("summary_identity_sha256", None)
    summary["gpu_telemetry_artifact"] = {
        "path": telemetry_path.name,
        "sha256": sha256_file(telemetry_path),
    }
    summary["hardware_preflight_artifact"] = {
        "path": hardware_path.name,
        "sha256": sha256_file(hardware_path),
    }
    if q8_binding is not None:
        summary["q8_exact_device_binding"] = q8_binding
    summary["summary_identity_sha256"] = sha256_json(summary)
    atomic_write_json(output_attempt / "runtime_summary.json", summary)
    os.replace(workflow_attempt, workflow_destination)
    os.replace(output_attempt, destination)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--producer-interface", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(MODES))
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--analysis-manifest", type=Path)
    parser.add_argument("--workflow-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cuda-visible-devices")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.check:
            _require(
                not any((args.workflow_output, args.output, args.cuda_visible_devices)),
                "--check forbids workflow/output/CUDA arguments",
            )
            result = preflight_provider(
                project_root=args.project_root,
                producer_interface=args.producer_interface,
                mode=args.mode,
                source_manifest=args.source_manifest,
                image_root=args.image_root,
                analysis_manifest=args.analysis_manifest,
            )
        else:
            required = {
                "mode": args.mode,
                "source_manifest": args.source_manifest,
                "image_root": args.image_root,
                "analysis_manifest": args.analysis_manifest,
                "workflow_output": args.workflow_output,
                "output": args.output,
                "cuda_visible_devices": args.cuda_visible_devices,
            }
            missing = [name for name, value in required.items() if value is None]
            _require(not missing, "--execute requires: " + ", ".join(missing))
            result = run_provider(
                project_root=args.project_root,
                producer_interface=args.producer_interface,
                mode=args.mode,
                source_manifest=args.source_manifest,
                image_root=args.image_root,
                analysis_manifest=args.analysis_manifest,
                workflow_output=args.workflow_output,
                output=args.output,
                python_executable=args.python,
                cuda_visible_devices=args.cuda_visible_devices,
            )
    except (
        ProviderError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
