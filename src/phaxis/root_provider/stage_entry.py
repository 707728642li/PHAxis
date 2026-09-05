"""Isolated, portable stage runner for the frozen root-provider closure.

Every public stage receives explicit bundle/input/output arguments.  GPU stages
run ``nvidia-smi`` immediately before importing a CUDA consumer, bind the
caller-selected physical device through ``CUDA_VISIBLE_DEVICES``, and retain
``cuda:0`` inside the frozen program.  Successful stage directories are
published atomically; a failed attempt remains beside the destination with a
machine-readable failure record and can never be mistaken for completed data.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import ModuleType
from typing import Any, Iterator, Sequence

from phaxis.io import atomic_write_json, read_json, sha256_file
from phaxis.root_provider.bundle import BundleError, verify_bundle
from phaxis.root_provider.identity import install_deployment_identity_adapter


SCHEMA = "PHAxis-root-provider-portable-stage-1.0"
# Same-hardware end-to-end Q8 runs reached 4,384 MiB at the NVIDIA process
# level (the lower 2,984 MiB value is only torch peak allocation).  Capacity
# sharing must budget the process-level peak, then retain the independent
# workspace reserve below.
Q8_OBSERVED_PEAK_MIB = 4384
GPU_SAFETY_RESERVE_MIB = 2048
GPU_UTILIZATION_SHARED_LIMIT_PERCENT = 80
# Nine samples separated by two seconds span sixteen seconds from first to
# last observation.  In addition to being bounded, this is long enough that a
# checkpoint/epoch-boundary dip cannot masquerade as sustained spare compute.
GPU_UTILIZATION_SAMPLE_COUNT = 9
GPU_UTILIZATION_SAMPLE_INTERVAL_SECONDS = 2.0
Q8_WAIT_SCHEMA = "PHAxis-Q8-waiting-for-gpu-capacity-1.0"
Q8_CAPACITY_RETRY_SECONDS = 60.0
STRICT_PHYSICAL_GPU_ENV = "PHAXIS_REQUIRE_EXACT_PHYSICAL_GPU"


class Q8CapacityUnavailableError(RuntimeError):
    """A transient, explicitly retryable Q8 capacity/utilization condition."""


def _model_root(bundle: Path) -> Path:
    return bundle / "hybrid_candidate" / "model"


def _legacy_root(bundle: Path) -> Path:
    return bundle / "legacy_project"


def _prepend_sys_path(paths: Sequence[Path]) -> None:
    for path in reversed(paths):
        value = str(path.resolve())
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load portable stage script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _path_has_protected_component(path: Path) -> bool:
    normalized = {
        part.casefold().replace("_", "-") for part in path.resolve().parts
    }
    return "blind" in normalized or "final-validation" in normalized


def _reject_protected_inputs(paths: Sequence[Path | None]) -> None:
    protected = [str(path) for path in paths if path and _path_has_protected_component(path)]
    if protected:
        raise RuntimeError(f"refusing protected blind/final-validation input: {protected}")


def _completion_is_valid(
    output: Path, stage: str, *, require_wrapper_sentinel: bool = True
) -> bool:
    if require_wrapper_sentinel:
        sentinel = output / "PHAXIS_STAGE_COMPLETE.json"
        if not sentinel.is_file():
            return False
        sentinel_payload = read_json(sentinel)
        if sentinel_payload.get("status") != "completed" or sentinel_payload.get(
            "blind_images_used"
        ) != 0:
            return False
    candidates = (output / "summary.json", output / "rhpheno_run.json")
    payloads = [read_json(path) for path in candidates if path.is_file()]
    if not payloads:
        return False
    if any(payload.get("blind_images_used", payload.get("blind_test_images_used", 0)) != 0 for payload in payloads):
        return False
    statuses = [str(payload.get("status", "")) for payload in payloads]
    if stage == "v1":
        return any(status == "complete" for status in statuses)
    return any(status.startswith("complete") or status in {"completed", "locked"} for status in statuses)


@contextmanager
def _atomic_output(destination: Path, stage: str, resume: bool) -> Iterator[Path | None]:
    destination = destination.resolve()
    if destination.exists():
        if resume and _completion_is_valid(destination, stage):
            yield None
            return
        raise FileExistsError(f"stage output already exists and is not resumable: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt = destination.parent / f".{destination.name}.{stage}.attempt-{stamp}-{os.getpid()}"
    try:
        # Frozen outputs contain absolute self-references.  They must therefore
        # be written at their final path; renaming a successful directory would
        # invalidate manifests.  Publication is the final atomic sentinel.
        # On failure the entire partial directory is atomically quarantined, so
        # the official path disappears and cannot contaminate downstream work.
        yield destination
        if not _completion_is_valid(
            destination, stage, require_wrapper_sentinel=False
        ):
            raise RuntimeError(f"{stage} did not produce a valid completion record")
        atomic_write_json(
            destination / "PHAXIS_STAGE_COMPLETE.json",
            {
                "schema_version": SCHEMA,
                "status": "completed",
                "stage": stage,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "official_output": str(destination),
                "absolute_output_paths_are_stable": True,
                "blind_images_used": 0,
            },
        )
    except BaseException as error:
        if destination.exists():
            os.replace(destination, attempt)
        else:
            attempt.mkdir()
        atomic_write_json(
            attempt / "PHAXIS_STAGE_FAILURE.json",
            {
                "schema_version": SCHEMA,
                "status": "failed_recoverable_new_attempt_required",
                "stage": stage,
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
                "official_output_published": False,
                "blind_images_used": 0,
            },
        )
        raise


def _gpu_preflight(physical_gpu: int, output: Path) -> None:
    if physical_gpu < 0:
        raise ValueError("physical GPU index must be non-negative")
    snapshot = subprocess.run(
        ["nvidia-smi"], check=True, capture_output=True, text=True
    )
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    indices = {
        int(line.split(",", 1)[0].strip())
        for line in query.stdout.splitlines()
        if line.strip()
    }
    if physical_gpu not in indices:
        raise RuntimeError(f"physical GPU {physical_gpu} absent from nvidia-smi")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        snapshot.stdout + "\nQUERY\n" + query.stdout,
        encoding="utf-8",
    )
    # The frozen program must continue to address the selected physical card as
    # logical cuda:0.  No program is imported before this assignment.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)


def _q8_device_from_snapshot(
    requested_gpu: int,
    gpu_rows: Sequence[dict[str, Any]],
    active_gpu_uuids: set[str],
    *,
    allow_fallback: bool = True,
) -> tuple[int, str]:
    """Choose by peak-plus-reserve capacity and sustained utilization.

    An active compute process is retained in the audit record but is not an
    automatic veto.  This matches the workstation policy: safe sharing is
    decided by memory headroom and multi-sample utilization, never by killing
    or suspending another process.
    """

    by_index = {int(row["index"]): row for row in gpu_rows}
    if not by_index or len(by_index) != len(gpu_rows):
        raise RuntimeError("Q8 GPU snapshot is empty or contains duplicate indices")
    if requested_gpu not in by_index:
        raise RuntimeError(f"requested Q8 physical GPU {requested_gpu} is absent")

    def integer_samples(
        row: dict[str, Any], key: str, fallback_key: str
    ) -> tuple[int, ...]:
        raw = row.get(key)
        if raw is None:
            if fallback_key not in row:
                raise RuntimeError(f"Q8 GPU snapshot lacks {key}: {row}")
            raw = (row[fallback_key],)
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise RuntimeError(f"Q8 GPU snapshot has invalid {key}: {row}")
        values = tuple(int(value) for value in raw)
        if not values:
            raise RuntimeError(f"Q8 GPU snapshot has empty {key}: {row}")
        return values

    def memory_used_mib(row: dict[str, Any]) -> int:
        values = integer_samples(
            row, "memory_used_samples_mib", "memory_used_mib"
        )
        if any(value < 0 for value in values):
            raise RuntimeError(f"Q8 GPU snapshot has negative memory use: {row}")
        return max(values)

    def sustained_utilization_percent(row: dict[str, Any]) -> float:
        values = integer_samples(
            row, "utilization_samples_percent", "utilization_percent"
        )
        if any(value < 0 or value > 100 for value in values):
            raise RuntimeError(f"Q8 GPU snapshot has invalid utilization: {row}")
        # "Sustained <80%" is deliberately fail-closed: every observation in
        # the bounded window must be below the limit.  A median would accept a
        # short checkpoint/epoch gap surrounded by fully occupied samples.
        return float(max(values))

    def is_eligible(row: dict[str, Any]) -> bool:
        total = int(row["memory_total_mib"])
        if total <= 0:
            raise RuntimeError(f"Q8 GPU snapshot has invalid total memory: {row}")
        free_after_q8 = (
            total - memory_used_mib(row) - Q8_OBSERVED_PEAK_MIB
        )
        return (
            free_after_q8 >= GPU_SAFETY_RESERVE_MIB
            and sustained_utilization_percent(row)
            < GPU_UTILIZATION_SHARED_LIMIT_PERCENT
        )

    requested = by_index[requested_gpu]
    requested_active = str(requested["uuid"]) in active_gpu_uuids
    if is_eligible(requested):
        return requested_gpu, (
            "explicit_requested_gpu_shared_capacity_available"
            if requested_active
            else "explicit_requested_gpu_available"
        )

    if not allow_fallback:
        raise Q8CapacityUnavailableError(
            "Q8 GPU safety gate failed: the exact requested physical GPU "
            "violates peak-plus-reserve memory or sustained-utilization policy; "
            "formal exact-device mode forbids fallback"
        )

    candidates = [
        row
        for index, row in by_index.items()
        if index != requested_gpu
        and is_eligible(row)
    ]
    if not candidates:
        raise Q8CapacityUnavailableError(
            "Q8 GPU safety gate failed: requested GPU and all alternatives "
            "violate peak-plus-reserve memory or sustained-utilization policy; "
            "no capacity-and-utilization-safe fallback"
        )
    selected = min(
        candidates,
        key=lambda row: (
            memory_used_mib(row),
            sustained_utilization_percent(row),
            int(row["index"]),
        ),
    )
    return int(selected["index"]), "requested_gpu_ineligible_safe_shared_fallback"


def _sample_q8_gpu_rows() -> list[dict[str, Any]]:
    """Take a bounded multi-sample snapshot before Q8 imports CUDA."""

    aggregate: dict[int, dict[str, Any]] = {}
    expected_indices: set[int] | None = None
    for sample_index in range(GPU_UTILIZATION_SAMPLE_COUNT):
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        observed_indices: set[int] = set()
        for line in query.stdout.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6:
                raise RuntimeError(f"unexpected nvidia-smi GPU row: {line}")
            index = int(fields[0])
            observed_indices.add(index)
            uuid = fields[1]
            total = int(fields[2])
            memory_used = int(fields[3])
            utilization = int(fields[4])
            temperature = int(fields[5])
            if index not in aggregate:
                aggregate[index] = {
                    "index": index,
                    "uuid": uuid,
                    "memory_total_mib": total,
                    "memory_used_samples_mib": [],
                    "utilization_samples_percent": [],
                    "temperature_samples_c": [],
                }
            row = aggregate[index]
            if row["uuid"] != uuid or row["memory_total_mib"] != total:
                raise RuntimeError(
                    f"nvidia-smi GPU identity changed during Q8 sampling: {index}"
                )
            row["memory_used_samples_mib"].append(memory_used)
            row["utilization_samples_percent"].append(utilization)
            row["temperature_samples_c"].append(temperature)
        if not observed_indices:
            raise RuntimeError("nvidia-smi returned no GPUs during Q8 sampling")
        if expected_indices is None:
            expected_indices = observed_indices
        elif observed_indices != expected_indices:
            raise RuntimeError("nvidia-smi GPU set changed during Q8 sampling")
        if sample_index + 1 < GPU_UTILIZATION_SAMPLE_COUNT:
            time.sleep(GPU_UTILIZATION_SAMPLE_INTERVAL_SECONDS)

    rows: list[dict[str, Any]] = []
    for index in sorted(aggregate):
        row = aggregate[index]
        memory_samples = [int(value) for value in row["memory_used_samples_mib"]]
        utilization_samples = [
            int(value) for value in row["utilization_samples_percent"]
        ]
        temperature_samples = [
            int(value) for value in row["temperature_samples_c"]
        ]
        rows.append(
            {
                **row,
                "memory_used_mib": max(memory_samples),
                "utilization_percent": int(utilization_samples[-1]),
                "sustained_utilization_percent": float(max(utilization_samples)),
                "temperature_c": max(temperature_samples),
            }
        )
    return rows


def _select_q8_physical_gpu(requested_gpu: int, output: Path) -> dict[str, Any]:
    rows = _sample_q8_gpu_rows()
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    process_rows = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    active_gpu_uuids = {line.split(",", 1)[0].strip() for line in process_rows}
    strict_value = os.environ.get(STRICT_PHYSICAL_GPU_ENV, "0").strip()
    if strict_value not in {"0", "1"}:
        raise RuntimeError(
            f"{STRICT_PHYSICAL_GPU_ENV} must be exactly '0' or '1'"
        )
    exact_physical_gpu_required = strict_value == "1"
    selected_gpu, reason = _q8_device_from_snapshot(
        requested_gpu,
        rows,
        active_gpu_uuids,
        allow_fallback=not exact_physical_gpu_required,
    )
    record = {
        "schema_version": "PHAxis-Q8-device-selection-1.0",
        "selected_utc": datetime.now(timezone.utc).isoformat(),
        "requested_physical_gpu": requested_gpu,
        "selected_physical_gpu": selected_gpu,
        "reason": reason,
        "exact_physical_gpu_required": exact_physical_gpu_required,
        "exact_physical_gpu_environment": STRICT_PHYSICAL_GPU_ENV,
        "q8_observed_peak_mib": Q8_OBSERVED_PEAK_MIB,
        "q8_peak_basis": "same_hardware_nvidia_process_peak_not_torch_allocator_peak",
        "required_safety_reserve_mib": GPU_SAFETY_RESERVE_MIB,
        "sustained_utilization_limit_percent": GPU_UTILIZATION_SHARED_LIMIT_PERCENT,
        "utilization_sample_count": GPU_UTILIZATION_SAMPLE_COUNT,
        "utilization_sample_interval_seconds": GPU_UTILIZATION_SAMPLE_INTERVAL_SECONDS,
        "sustained_utilization_statistic": "maximum_all_samples_must_be_below_limit",
        "active_compute_process_is_not_automatic_veto": True,
        "gpu_snapshot": rows,
        "active_compute_process_rows": process_rows,
        "blind_images_used": 0,
    }
    atomic_write_json(output, record)
    return record


def _q8_control_paths(output: Path) -> tuple[Path, Path, Path]:
    """Return sidecar paths which can never contaminate candidate output."""

    output = output.resolve()
    control_root = output.parent / f".{output.name}.q8_control"
    return (
        control_root,
        control_root / "waiting_for_gpu_capacity.json",
        control_root / "q8_device_selection.json",
    )


def _wait_for_q8_physical_gpu(
    requested_gpu: int, output: Path
) -> dict[str, Any]:
    """Wait durably without CUDA imports or creating candidate output."""

    output = output.resolve()
    _control_root, wait_path, selection_path = _q8_control_paths(output)
    # Releases before the sidecar fix wrote only the wait receipt beside the
    # shard.  It remains immutable evidence and may seed a resumed wait, but
    # every new receipt is written to the canonical control directory.
    legacy_wait_path = (
        output.parent / f".{output.name}.waiting_for_gpu_capacity.json"
    )
    previous_path = wait_path if wait_path.is_file() else legacy_wait_path
    previous: dict[str, Any] | None = None
    if previous_path.is_file():
        previous = read_json(previous_path)
        if previous.get("schema_version") != Q8_WAIT_SCHEMA:
            raise RuntimeError(f"Q8 wait receipt schema mismatch: {previous_path}")
        if previous.get("requested_physical_gpu") != requested_gpu:
            raise RuntimeError(f"Q8 wait receipt device mismatch: {previous_path}")
        if previous.get("blind_images_used") != 0:
            raise RuntimeError(f"Q8 wait receipt is blind-tainted: {previous_path}")
        if previous.get("status") not in {
            "waiting_for_gpu_capacity",
            "capacity_available_q8_starting",
        }:
            raise RuntimeError(f"Q8 wait receipt status is invalid: {previous_path}")

    wait_started_utc = (
        str(previous["wait_started_utc"])
        if previous is not None and previous.get("wait_started_utc")
        else datetime.now(timezone.utc).isoformat()
    )
    capacity_checks = int(previous.get("capacity_checks", 0)) if previous else 0
    scheduled_sleep_seconds = (
        float(previous.get("scheduled_sleep_seconds", 0.0)) if previous else 0.0
    )
    resumed = previous is not None

    while True:
        try:
            selected = _select_q8_physical_gpu(requested_gpu, selection_path)
        except Q8CapacityUnavailableError as error:
            capacity_checks += 1
            scheduled_sleep_seconds += Q8_CAPACITY_RETRY_SECONDS
            atomic_write_json(
                wait_path,
                {
                    "schema_version": Q8_WAIT_SCHEMA,
                    "status": "waiting_for_gpu_capacity",
                    "wait_started_utc": wait_started_utc,
                    "last_checked_utc": datetime.now(timezone.utc).isoformat(),
                    "requested_physical_gpu": requested_gpu,
                    "capacity_checks": capacity_checks,
                    "retry_seconds": Q8_CAPACITY_RETRY_SECONDS,
                    "scheduled_sleep_seconds": scheduled_sleep_seconds,
                    "last_capacity_error": str(error),
                    "q8_observed_peak_mib": Q8_OBSERVED_PEAK_MIB,
                    "required_safety_reserve_mib": GPU_SAFETY_RESERVE_MIB,
                    "sustained_utilization_limit_percent": GPU_UTILIZATION_SHARED_LIMIT_PERCENT,
                    "utilization_sample_count_per_check": GPU_UTILIZATION_SAMPLE_COUNT,
                    "utilization_sample_interval_seconds": GPU_UTILIZATION_SAMPLE_INTERVAL_SECONDS,
                    "wait_occurs_before_cuda_import": True,
                    "low_cpu_wait": True,
                    "resumed_from_prior_wait_receipt": resumed,
                    "no_process_killed_or_suspended": True,
                    "blind_images_used": 0,
                },
            )
            time.sleep(Q8_CAPACITY_RETRY_SECONDS)
            continue

        if previous is not None or wait_path.is_file():
            atomic_write_json(
                wait_path,
                {
                    "schema_version": Q8_WAIT_SCHEMA,
                    "status": "capacity_available_q8_starting",
                    "wait_started_utc": wait_started_utc,
                    "capacity_available_utc": datetime.now(timezone.utc).isoformat(),
                    "requested_physical_gpu": requested_gpu,
                    "selected_physical_gpu": selected["selected_physical_gpu"],
                    "selection_reason": selected["reason"],
                    "capacity_checks": capacity_checks,
                    "retry_seconds": Q8_CAPACITY_RETRY_SECONDS,
                    "scheduled_sleep_seconds": scheduled_sleep_seconds,
                    "selection_record": str(selection_path),
                    "selection_record_sha256": sha256_file(selection_path),
                    "wait_occurs_before_cuda_import": True,
                    "resumed_from_prior_wait_receipt": resumed,
                    "no_process_killed_or_suspended": True,
                    "blind_images_used": 0,
                },
            )
        if output.exists():
            raise RuntimeError(
                "Q8 control-plane metadata contaminated candidate output: "
                f"{output}"
            )
        return selected


def _legacy_paths(bundle: Path) -> tuple[Path, Path, Path]:
    legacy = _legacy_root(bundle)
    model_runtime = _model_root(bundle) / "runtime"
    return legacy, legacy / "src", model_runtime / "src"


def _run_prepare(args: argparse.Namespace, attempt: Path) -> None:
    _reject_protected_inputs((args.input_manifest, args.acquisition_gate))
    legacy, legacy_src, runtime_src = _legacy_paths(args.bundle)
    _prepend_sys_path((legacy_src, runtime_src, legacy / "scripts"))
    from rhizoweave.prepare import prepare_dataset

    result = prepare_dataset(
        args.input_manifest,
        output=attempt,
        acquisition_gate=args.acquisition_gate,
        declared_scale_value_um=args.declared_scale_value_um,
        use_ocr=not args.no_ocr,
        gate_max_side=args.gate_max_side,
        rootguide_max_side=args.rootguide_max_side,
        rootguide_min_confidence=args.rootguide_min_confidence,
        half_width_um=args.half_width_um,
        longitudinal_step_um=args.longitudinal_step_um,
        transverse_step_um=args.transverse_step_um,
        frame_smoothing_um=args.frame_smoothing_um,
    )
    if result.get("blind_images_used") != 0:
        raise RuntimeError("prepare stage blind guard failed")


def _run_v1(args: argparse.Namespace, attempt: Path) -> None:
    _reject_protected_inputs((args.prepared_manifest, args.quality_csv))
    preflight_path = attempt.parent / f".{attempt.name}.nvidia_smi_preflight.txt"
    _gpu_preflight(args.physical_gpu, preflight_path)
    legacy, legacy_src, runtime_src = _legacy_paths(args.bundle)
    _prepend_sys_path((legacy_src, runtime_src, legacy / "scripts"))
    module = _load_script(
        legacy / "scripts/run_rhpheno_dual_mode_v5.py",
        "_phaxis_frozen_v1_entrypoint",
    )
    # Frozen v3 writes a historical physical-GPU-0 environment into the
    # backend subprocess.  Intercept only that subprocess boundary so the
    # caller-selected physical card remains mapped to logical cuda:0.  The
    # numerical entrypoint and all hash-locked artifacts remain byte-exact.
    original_subprocess_run = module.runner.subprocess.run

    def portable_backend_run(command: Any, *positional: Any, **keyword: Any) -> Any:
        environment = dict(keyword.get("env") or os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
        keyword["env"] = environment
        return original_subprocess_run(command, *positional, **keyword)

    module.runner.subprocess.run = portable_backend_run
    old_argv = sys.argv[:]
    sys.argv = [
        str(Path(module.__file__).resolve()),
        "--mode",
        "sparse_instance",
        "--manifest",
        str(args.prepared_manifest.resolve()),
        "--output",
        str(attempt),
        "--runtime-config",
        str(args.runtime_config.resolve()),
        "--quality-csv",
        str(args.quality_csv.resolve()),
    ]
    for task_id in args.task_id:
        sys.argv.extend(("--image-id", task_id))
    try:
        module.main()
    finally:
        module.runner.subprocess.run = original_subprocess_run
        sys.argv = old_argv
        if attempt.is_dir() and preflight_path.is_file():
            os.replace(preflight_path, attempt / "nvidia_smi_preflight.txt")
    report_path = attempt / "rhpheno_run.json"
    report = read_json(report_path)
    report["portable_phaxis_device_mapping"] = {
        "physical_gpu": args.physical_gpu,
        "cuda_visible_devices": str(args.physical_gpu),
        "logical_device": "cuda:0",
        "historical_gpu0_assignment_intercepted_at_subprocess_boundary": True,
        "frozen_entrypoint_modified": False,
    }
    atomic_write_json(report_path, report)


def _run_script_main(
    *, path: Path, module_name: str, argv: Sequence[str], import_paths: Sequence[Path]
) -> None:
    _prepend_sys_path(import_paths)
    module = _load_script(path, module_name)
    old_argv = sys.argv[:]
    sys.argv = [str(path.resolve()), *map(str, argv)]
    try:
        module.main()
    finally:
        sys.argv = old_argv


def _run_v1_merge(args: argparse.Namespace, attempt: Path) -> None:
    legacy, legacy_src, runtime_src = _legacy_paths(args.bundle)
    _run_script_main(
        path=legacy / "scripts/merge_six_condition_v1_shards.py",
        module_name="_phaxis_frozen_v1_merge",
        argv=("--orchestration-root", args.orchestration_root, "--output", attempt),
        import_paths=(legacy_src, runtime_src, legacy / "scripts"),
    )


def _run_materialize(args: argparse.Namespace, attempt: Path) -> None:
    _reject_protected_inputs(
        (args.prepared_manifest, args.deployment_metadata, args.canonical_manifest)
    )
    legacy, legacy_src, runtime_src = _legacy_paths(args.bundle)
    path = legacy / "scripts/materialize_six_condition_v1_prefill_adapter.py"
    _prepend_sys_path((legacy_src, runtime_src, legacy / "scripts"))
    module = _load_script(path, "_phaxis_frozen_v1_materializer")
    original_export = module.export_assisted_prefills

    def portable_public_task_namespaces(*positional: Any, **keyword: Any) -> Any:
        """Add the public RHAUD namespace without changing frozen numerics."""

        declared = tuple(keyword.get("allowed_sample_prefixes", ()))
        if declared == ("RHSCU-",):
            keyword["allowed_sample_prefixes"] = ("RHSCU-", "RHAUD-")
        return original_export(*positional, **keyword)

    module.export_assisted_prefills = portable_public_task_namespaces
    old_argv = sys.argv[:]
    sys.argv = [
        str(path.resolve()),
        "--prepared-manifest", str(args.prepared_manifest.resolve()),
        "--deployment-metadata", str(args.deployment_metadata.resolve()),
        "--canonical-manifest", str(args.canonical_manifest.resolve()),
        "--v1-run-root", str(args.v1_run_root.resolve()),
        "--runtime-config", str(args.runtime_config.resolve()),
        "--output", str(attempt.resolve()),
    ]
    try:
        module.main()
    finally:
        module.export_assisted_prefills = original_export
        sys.argv = old_argv
    summary_path = attempt / "summary.json"
    summary = read_json(summary_path)
    summary["portable_public_task_namespace_adapter"] = {
        "accepted_prefixes": ["RHSCU-", "RHAUD-"],
        "frozen_materializer_modified": False,
        "prediction_geometry_or_numerics_changed": False,
    }
    atomic_write_json(summary_path, summary)


def _run_v20(args: argparse.Namespace, attempt: Path) -> None:
    _reject_protected_inputs((args.selection_manifest, args.compatibility_data_root))
    legacy, legacy_src, runtime_src = _legacy_paths(args.bundle)
    argv: list[Any] = [
        "--selection-manifest", args.selection_manifest,
        "--compatibility-data-root", args.compatibility_data_root,
        "--config", args.config,
        "--output", attempt,
    ]
    for task_id in args.task_id:
        argv.extend(("--only-task", task_id))
    _run_script_main(
        path=legacy / "scripts/run_six_condition_v20_12_readonly_adapter.py",
        module_name="_phaxis_frozen_v20_adapter",
        argv=argv,
        import_paths=(legacy_src, runtime_src, legacy / "scripts"),
    )


def _run_v20_merge(args: argparse.Namespace, attempt: Path) -> None:
    legacy, legacy_src, runtime_src = _legacy_paths(args.bundle)
    _run_script_main(
        path=legacy / "scripts/merge_six_condition_v20_12_shards.py",
        module_name="_phaxis_frozen_v20_merge",
        argv=("--orchestration-root", args.orchestration_root, "--output", attempt),
        import_paths=(legacy_src, runtime_src, legacy / "scripts"),
    )


def _run_q8(args: argparse.Namespace, attempt: Path) -> None:
    _reject_protected_inputs(
        (args.deployment_manifest, args.deployment_lock, args.deployment_image_root)
    )
    requested_gpu = args.physical_gpu
    device_selection = _wait_for_q8_physical_gpu(requested_gpu, attempt)
    _control_root, wait_path, selection_path = _q8_control_paths(attempt)
    args.physical_gpu = int(device_selection["selected_physical_gpu"])
    preflight_path = attempt.parent / f".{attempt.name}.nvidia_smi_preflight.txt"
    _gpu_preflight(args.physical_gpu, preflight_path)
    model = _model_root(args.bundle)
    runtime_src = model / "runtime/src"
    _prepend_sys_path((runtime_src,))
    import rhaxis_nextgen.candidate_benchmark as candidate

    install_deployment_identity_adapter(
        candidate, args.bundle, bundle_already_verified=True
    )
    old_argv = sys.argv[:]
    sys.argv = [
        str(Path(candidate.__file__).resolve()),
        "--semantic-run", str(model / "weights/q8/semantic_e0"),
        "--topology-run", str(model / "weights/q8/topology_e2"),
        "--query-run", str(model / "weights/q8/query_q6b"),
        "--verifier-run", str(model / "weights/q8/verifier_q7"),
        "--scale-calibration", str(model / "weights/q8/scale_calibration"),
        "--root-enhancement-lock", str(model / "weights/q8/root_enhancement/root_enhancement_lock.json"),
        "--root-run", str(model / "weights/q8/root_e4"),
        "--point-module-lock", str(model / "weights/q8/point_module/point_module_lock.json"),
        "--operating-points", str(model / "weights/q8/verifier_q7/operating_points_recall_sensitivity.json"),
        "--operating-point", "recall_0p85",
        "--dataset", str(args.bundle / "identity_only_training_payload_not_packaged"),
        "--split-override", str(model / "runtime/configs/rhaxis_nextgen/splits/qc_development_v1_0/split_lock.json"),
        "--deployment-manifest", str(args.deployment_manifest.resolve()),
        "--deployment-lock", str(args.deployment_lock.resolve()),
        "--deployment-image-root", str(args.deployment_image_root.resolve()),
        "--shard-index", str(args.shard_index),
        "--num-shards", str(args.num_shards),
        "--field-batch-size", str(args.field_batch_size),
        "--query-batch-size", str(args.query_batch_size),
        "--output", str(attempt),
        "--compact-summary",
    ]
    for task_id in args.task_id:
        sys.argv.extend(("--task-id", task_id))
    try:
        candidate.main()
    except BaseException:
        # candidate has already observed an absent output at this point.  If it
        # fails before creating the directory, create it only now so the outer
        # atomic wrapper can quarantine a complete selection/preflight bundle.
        attempt.mkdir(parents=True, exist_ok=True)
        if selection_path.is_file():
            atomic_write_json(
                attempt / "q8_device_selection.json", read_json(selection_path)
            )
        raise
    finally:
        sys.argv = old_argv
        if attempt.is_dir() and preflight_path.is_file():
            os.replace(
                preflight_path, attempt / "nvidia_smi_preflight_outer.txt"
            )
    if not selection_path.is_file():
        raise RuntimeError(f"Q8 selection receipt is missing: {selection_path}")
    # candidate_benchmark owns an initially absent/empty directory.  Only
    # after it returns successfully do we publish self-contained audit copies.
    internal_selection_path = attempt / "q8_device_selection.json"
    atomic_write_json(internal_selection_path, device_selection)
    summary_path = attempt / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        summary["portable_phaxis_device_mapping"] = {
            "requested_physical_gpu": requested_gpu,
            "selected_physical_gpu": args.physical_gpu,
            "selection_reason": device_selection["reason"],
            "logical_device": "cuda:0",
            "selection_receipt": str(internal_selection_path),
            "selection_receipt_sha256": sha256_file(internal_selection_path),
            "control_selection_receipt": str(selection_path),
            "control_selection_receipt_sha256": sha256_file(selection_path),
            "wait_receipt": str(wait_path) if wait_path.is_file() else None,
        }
        atomic_write_json(summary_path, summary)


def _run_q8_merge(args: argparse.Namespace, attempt: Path) -> None:
    _reject_protected_inputs(
        (args.deployment_manifest, args.deployment_lock, args.deployment_image_root)
    )
    model = _model_root(args.bundle)
    _run_script_main(
        path=model / "runtime/scripts/merge_rhaxis_nextgen_deployment_shards.py",
        module_name="_phaxis_q8_merge",
        argv=(
            "--shard-root", args.shard_root,
            "--deployment-manifest", args.deployment_manifest,
            "--deployment-lock", args.deployment_lock,
            "--image-root", args.deployment_image_root,
            "--expected-shards", str(args.expected_shards),
            "--output", attempt,
        ),
        import_paths=(model / "runtime/src", model / "runtime/scripts"),
    )


def _run_hybrid(args: argparse.Namespace, attempt: Path) -> None:
    _reject_protected_inputs(
        (args.deployment_manifest, args.deployment_lock, args.deployment_image_root)
    )
    model = _model_root(args.bundle)
    _prepend_sys_path((model / "runtime/src",))
    import rhaxis_nextgen.hybrid_max_deployment as hybrid

    install_deployment_identity_adapter(
        hybrid, args.bundle, bundle_already_verified=True
    )
    old_argv = sys.argv[:]
    sys.argv = [
        str(Path(hybrid.__file__).resolve()),
        "--training-dataset", str(args.bundle / "identity_only_training_payload_not_packaged"),
        "--training-split-override", str(model / "runtime/configs/rhaxis_nextgen/splits/qc_development_v1_0/split_lock.json"),
        "--deployment-manifest", str(args.deployment_manifest.resolve()),
        "--deployment-lock", str(args.deployment_lock.resolve()),
        "--deployment-image-root", str(args.deployment_image_root.resolve()),
        "--v20-output", str(args.v20_output.resolve()),
        "--q8-output", str(args.q8_output.resolve()),
        "--v1-verifier-run", str(model / "weights/hybrid/v1_hair_verifier"),
        "--q7-operating-points", str(model / "weights/q8/verifier_q7/operating_points.json"),
        "--geometry-gate-lock", str(model / "weights/hybrid/geometry_gate/hybrid_max_geometry_gate_lock.json"),
        "--method-amendment-lock", str(model / "contracts/final_method_lock.json"),
        "--hair-curvature-verifier", str(model / "weights/hybrid/curvature_verifier"),
        "--hair-polarity-verifier", str(model / "weights/hybrid/polarity_verifier"),
        "--hair-polarity-qc-lock", str(model / "runtime_locks/hair_polarity_qcdev44_evaluation.json"),
        "--hair-axis-ridge-verifier", str(model / "weights/hybrid/axis_ridge_verifier"),
        "--output", str(attempt),
    ]
    for task_id in args.task_id:
        sys.argv.extend(("--task-id", task_id))
    try:
        hybrid.main()
    finally:
        sys.argv = old_argv


def _common(subparser: argparse.ArgumentParser, *, gpu: bool = False) -> None:
    subparser.add_argument("--bundle", type=Path, required=True)
    subparser.add_argument("--output", type=Path, required=True)
    subparser.add_argument("--resume", action="store_true")
    if gpu:
        subparser.add_argument("--physical-gpu", type=int, required=True)


def _deployment(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--deployment-manifest", type=Path, required=True)
    subparser.add_argument("--deployment-lock", type=Path, required=True)
    subparser.add_argument("--deployment-image-root", type=Path, required=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="stage", required=True)

    prepare = commands.add_parser("prepare")
    _common(prepare)
    prepare.add_argument("--input-manifest", type=Path, required=True)
    prepare.add_argument("--acquisition-gate", type=Path, required=True)
    prepare.add_argument("--declared-scale-value-um", type=float)
    prepare.add_argument("--no-ocr", action="store_true")
    prepare.add_argument("--gate-max-side", type=int, default=3072)
    prepare.add_argument("--rootguide-max-side", type=int, default=1536)
    prepare.add_argument("--rootguide-min-confidence", type=float, default=0.45)
    prepare.add_argument("--half-width-um", type=float, default=250.0)
    prepare.add_argument("--longitudinal-step-um", type=float, default=2.0)
    prepare.add_argument("--transverse-step-um", type=float, default=2.0)
    prepare.add_argument("--frame-smoothing-um", type=float, default=30.0)

    v1 = commands.add_parser("v1")
    _common(v1, gpu=True)
    v1.add_argument("--prepared-manifest", type=Path, required=True)
    v1.add_argument("--quality-csv", type=Path, required=True)
    v1.add_argument("--runtime-config", type=Path, required=True)
    v1.add_argument("--task-id", action="append", default=[])

    merge_v1 = commands.add_parser("merge-v1")
    _common(merge_v1)
    merge_v1.add_argument("--orchestration-root", type=Path, required=True)

    materialize = commands.add_parser("materialize-v1")
    _common(materialize)
    materialize.add_argument("--prepared-manifest", type=Path, required=True)
    materialize.add_argument("--deployment-metadata", type=Path, required=True)
    materialize.add_argument("--canonical-manifest", type=Path, required=True)
    materialize.add_argument("--v1-run-root", type=Path, required=True)
    materialize.add_argument("--runtime-config", type=Path, required=True)

    v20 = commands.add_parser("v20")
    _common(v20)
    v20.add_argument("--selection-manifest", type=Path, required=True)
    v20.add_argument("--compatibility-data-root", type=Path, required=True)
    v20.add_argument("--config", type=Path, required=True)
    v20.add_argument("--task-id", action="append", default=[])

    merge_v20 = commands.add_parser("merge-v20")
    _common(merge_v20)
    merge_v20.add_argument("--orchestration-root", type=Path, required=True)

    q8 = commands.add_parser("q8")
    _common(q8, gpu=True)
    _deployment(q8)
    q8.add_argument("--shard-index", type=int, default=0)
    q8.add_argument("--num-shards", type=int, default=1)
    q8.add_argument("--field-batch-size", type=int, default=10)
    q8.add_argument("--query-batch-size", type=int, default=32)
    q8.add_argument("--task-id", action="append", default=[])

    merge_q8 = commands.add_parser("merge-q8")
    _common(merge_q8)
    _deployment(merge_q8)
    merge_q8.add_argument("--shard-root", type=Path, required=True)
    merge_q8.add_argument("--expected-shards", type=int, required=True)

    hybrid = commands.add_parser("hybrid")
    _common(hybrid)
    _deployment(hybrid)
    hybrid.add_argument("--v20-output", type=Path, required=True)
    hybrid.add_argument("--q8-output", type=Path, required=True)
    hybrid.add_argument("--task-id", action="append", default=[])
    return result


RUNNERS = {
    "prepare": _run_prepare,
    "v1": _run_v1,
    "merge-v1": _run_v1_merge,
    "materialize-v1": _run_materialize,
    "v20": _run_v20,
    "merge-v20": _run_v20_merge,
    "q8": _run_q8,
    "merge-q8": _run_q8_merge,
    "hybrid": _run_hybrid,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.bundle = args.bundle.resolve()
    verification = verify_bundle(args.bundle)
    if verification.get("blind_images_used") != 0:
        raise BundleError("root-provider bundle verification is blind-tainted")
    with _atomic_output(args.output, args.stage, args.resume) as attempt:
        if attempt is None:
            print(json.dumps({"status": "already_complete", "stage": args.stage, "output": str(args.output.resolve())}, indent=2))
            return 0
        RUNNERS[args.stage](args, attempt)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA,
                "status": "completed",
                "stage": args.stage,
                "output": str(args.output.resolve()),
                "bundle_identity_sha256": verification["bundle_identity_sha256"],
                "python_executable": sys.executable,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "blind_images_used": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
