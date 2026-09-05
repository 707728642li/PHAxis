"""Direct, fail-closed PHAxis full-workflow benchmark receipts.

Two timing modes are intentionally separate:

* ``production_batch_full283`` measures one fresh production batch invocation
  and reports only batch wall/throughput plus directly observed stage walls.
* sequential modes consume a sealed per-source trace whose 283 rows were timed
  at real raw-image-to-final-profile boundaries.  They alone may report
  per-image median/P95 latency.

No batch duration is divided by 283 to manufacture a latency distribution.
This module imports neither torch nor any CUDA runtime.
"""

from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .contracts import ContractError
from .io import atomic_write_json, read_json, sha256_file, sha256_json
from .workflow import build_analysis_plan, run_analysis


MEASUREMENT_SCOPE = "raw_image_to_final_traits_and_profiles_direct"
PRODUCTION_MODE = "production_batch_full283"
PERSISTENT_LATENCY_MODE = "sequential_persistent_full283"
COLD_LATENCY_MODE = "sequential_cold_cli_full283"
LATENCY_MODES = frozenset({PERSISTENT_LATENCY_MODE, COLD_LATENCY_MODE})
PRODUCTION_SCHEMA = "PHAxis-full-workflow-production-batch-benchmark-1.0"
LATENCY_TRACE_SCHEMA = "PHAxis-full-workflow-per-source-trace-1.0"
LATENCY_SCHEMA = "PHAxis-full-workflow-sequential-latency-benchmark-1.0"
COMPARISON_SCHEMA = "PHAxis-full-workflow-benchmark-comparison-1.0"
SAME_HARDWARE_RECEIPT_SCHEMA = "PHAxis-same-hardware-benchmark-receipt-1.0"
SAME_HARDWARE_PLAN_SCHEMA = "PHAxis-same-hardware-benchmark-plan-1.0"
FROZEN_V1_PRODUCER_INTERFACE_SCHEMA = (
    "PHAxis-frozen-v1-exact283-benchmark-producer-interface-1.0"
)
FROZEN_V1_PRODUCER_GATE_SCHEMA = (
    "PHAxis-frozen-v1-exact283-benchmark-producer-gate-1.0"
)
PHAXIS_BENCHMARK_SYSTEM = "PHAxis-1.0.0"
FROZEN_V1_BENCHMARK_SYSTEM = "frozen-v1-exact283"
PER_IMAGE_FIELDS = (
    "source_unit",
    "wall_seconds",
    "megapixels",
    "io_seconds",
    "preprocess_seconds",
    "inference_seconds",
    "postprocess_seconds",
)
_CORE_STAGES = (
    "root_provider",
    "stageb_train399",
    "fusion",
    "traits",
    "distal_axis_profiles",
)
_BINDING_FIELDS = (
    "model_contract_proposal_sha256",
    "model_contract_proposal_identity_sha256",
)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _sealed(payload: Mapping[str, Any], field: str, *, role: str) -> str:
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field, None)
    if not _is_sha256(observed):
        raise ContractError(f"{role}: {field} is absent or invalid")
    expected = sha256_json(unsigned)
    if str(observed).casefold() != expected.casefold():
        raise ContractError(f"{role}: {field} does not seal the complete receipt")
    return expected


def _finite_number(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{field} must be numeric") from error
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        raise ContractError(f"{field} must be finite and {'positive' if positive else 'non-negative'}")
    return number


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ContractError("cannot calculate a percentile from an empty trace")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _validated_hash_map(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ContractError(f"{field} must be a non-empty hash map")
    result = {str(key): str(digest) for key, digest in value.items()}
    if any(not key or not _is_sha256(digest) for key, digest in result.items()):
        raise ContractError(f"{field} contains an invalid SHA-256 binding")
    return dict(sorted(result.items()))


def _ordered_source_units(
    payload: Mapping[str, Any], *, role: str, exact283: bool
) -> tuple[list[str], str]:
    values = payload.get("source_units_in_order")
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise ContractError(f"{role}: ordered source-unit list is absent")
    ordered = [str(value) for value in values]
    if len(ordered) != len(set(ordered)):
        raise ContractError(f"{role}: ordered source-unit list contains duplicates")
    if exact283 and len(ordered) != 283:
        raise ContractError(f"{role}: ordered source-unit list is not exact283")
    identity = payload.get("source_unit_ordered_set_identity_sha256")
    expected = sha256_json(ordered)
    if identity != expected:
        raise ContractError(f"{role}: ordered source-unit identity mismatch")
    return ordered, expected


def _hardware_uuid_driver_identity(
    payload: Mapping[str, Any], *, role: str
) -> tuple[dict[str, Any], str, str]:
    hardware = payload.get("hardware")
    if not isinstance(hardware, Mapping):
        raise ContractError(f"{role}: hardware inventory is absent")
    hardware_object = deepcopy(dict(hardware))
    hardware_identity = payload.get("hardware_identity_sha256")
    if hardware_identity != sha256_json(hardware_object):
        raise ContractError(f"{role}: hardware identity mismatch")
    gpus = hardware_object.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise ContractError(f"{role}: GPU UUID/driver inventory is absent")
    uuid_driver: list[dict[str, str]] = []
    for index, gpu in enumerate(gpus):
        if not isinstance(gpu, Mapping):
            raise ContractError(f"{role}: GPU inventory row {index} is invalid")
        uuid = gpu.get("uuid")
        driver = gpu.get("driver_version")
        if not isinstance(uuid, str) or not uuid.strip() or not isinstance(driver, str) or not driver.strip():
            raise ContractError(f"{role}: GPU UUID/driver is missing at row {index}")
        uuid_driver.append({"uuid": uuid.strip(), "driver_version": driver.strip()})
    if len({row["uuid"] for row in uuid_driver}) != len(uuid_driver):
        raise ContractError(f"{role}: GPU UUID inventory contains duplicates")
    uuid_driver.sort(key=lambda row: row["uuid"])
    return hardware_object, str(hardware_identity), sha256_json(uuid_driver)


def _publish_directory(destination: Path, producer: Callable[[Path], Any]) -> Any:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite benchmark output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = destination.parent / f".{destination.name}.benchmark-attempt-{os.getpid()}"
    if attempt.exists():
        raise FileExistsError(f"benchmark attempt already exists: {attempt}")
    attempt.mkdir()
    try:
        result = producer(attempt)
        os.replace(attempt, destination)
        return result
    except BaseException:
        # The attempt is deliberately retained as non-official diagnostic state.
        raise


def benchmark_plan(
    *, manifest: str | Path, workflow_output: str | Path, benchmark_output: str | Path
) -> dict[str, Any]:
    """Return a side-effect-free benchmark plan; no nvidia-smi call is made."""

    analysis_plan = build_analysis_plan(manifest, output=workflow_output)
    result: dict[str, Any] = {
        "schema_version": "PHAxis-full-workflow-benchmark-plan-1.0",
        "status": "planned_not_executed",
        "default_dry_run": True,
        "execute_requires_explicit_flag": True,
        "production_mode": PRODUCTION_MODE,
        "latency_modes": sorted(LATENCY_MODES),
        "measurement_scope": MEASUREMENT_SCOPE,
        "workflow_output": str(Path(workflow_output).resolve()),
        "benchmark_output": str(Path(benchmark_output).resolve()),
        "analysis_plan_identity_sha256": analysis_plan["plan_identity_sha256"],
        "tasks": analysis_plan["tasks"],
        "batch_latency_is_never_derived_per_image": True,
        "blind_images_used": 0,
        "root_cap_region_metric": False,
    }
    result["plan_identity_sha256"] = sha256_json(result)
    return result


def _physical_gpus(plan: Mapping[str, Any]) -> tuple[int, ...]:
    result: set[int] = set()
    for stage in plan.get("stages", []):
        gpu = stage.get("estimated_gpu", {}) if isinstance(stage, Mapping) else {}
        for field in ("v1_physical_gpus", "q8_physical_gpus"):
            values = gpu.get(field, [])
            if isinstance(values, list):
                result.update(int(value) for value in values)
        if gpu.get("physical_gpu") is not None:
            result.add(int(gpu["physical_gpu"]))
    if not result:
        raise ContractError("benchmark plan contains no explicit physical GPUs")
    return tuple(sorted(result))


def _source_manifest_sha256(plan: Mapping[str, Any]) -> str:
    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise ContractError("analysis plan stages are absent")
    root_stage = next(
        (stage for stage in stages if stage.get("name") == "root_provider"), None
    )
    inputs = root_stage.get("input_hashes") if isinstance(root_stage, Mapping) else None
    digest = inputs.get("root_provider.input_manifest") if isinstance(inputs, Mapping) else None
    if not _is_sha256(digest):
        raise ContractError("analysis plan does not lock the raw-image source manifest")
    return str(digest)


def _parse_nvidia_rows(text: str, *, expected_columns: int) -> list[list[str]]:
    rows = [
        [part.strip() for part in line.split(",")]
        for line in text.splitlines()
        if line.strip()
    ]
    if not rows or any(len(row) != expected_columns for row in rows):
        raise ContractError("nvidia-smi returned an invalid benchmark inventory")
    return rows


def capture_hardware_preflight(physical_gpus: Sequence[int]) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    parsed = _parse_nvidia_rows(completed.stdout, expected_columns=5)
    selected: list[dict[str, Any]] = []
    requested = set(int(value) for value in physical_gpus)
    for row in parsed:
        index = int(row[0])
        if index in requested:
            selected.append(
                {
                    "physical_index": index,
                    "uuid": row[1],
                    "name": row[2],
                    "memory_total_mib": int(row[3]),
                    "driver_version": row[4],
                }
            )
    if {row["physical_index"] for row in selected} != requested:
        raise ContractError("one or more planned physical GPUs are absent")
    hardware = {
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpus": sorted(selected, key=lambda row: int(row["physical_index"])),
    }
    nvidia_preflight = {
        "command": command,
        "stdout_sha256": sha256_json(completed.stdout.splitlines()),
        "physical_gpus": list(sorted(requested)),
    }
    return {
        "hardware": hardware,
        "hardware_identity_sha256": sha256_json(hardware),
        "nvidia_smi_preflight": nvidia_preflight,
        "nvidia_smi_preflight_identity_sha256": sha256_json(nvidia_preflight),
    }


@dataclass
class GpuTelemetry:
    physical_gpus: tuple[int, ...]
    interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        self._samples: list[dict[str, float | int | str]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def _sample_once(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        wanted = set(self.physical_gpus)
        for row in _parse_nvidia_rows(completed.stdout, expected_columns=4):
            index = int(row[0])
            if index in wanted:
                self._samples.append(
                    {
                        "monotonic_seconds": time.perf_counter(),
                        "physical_index": index,
                        "uuid": row[1],
                        "utilization_pct": float(row[2]),
                        "memory_used_mib": float(row[3]),
                    }
                )

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                self._sample_once()
                self._stop.wait(self.interval_seconds)
        except BaseException as error:  # surfaced synchronously by stop()
            self._error = error

    def start(self) -> None:
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 3.0))
        if self._error is not None:
            raise RuntimeError("GPU telemetry sampling failed") from self._error
        if not self._samples:
            raise ContractError("GPU telemetry contains no direct samples")
        utilization = [float(row["utilization_pct"]) for row in self._samples]
        memory = [float(row["memory_used_mib"]) for row in self._samples]
        return {
            "samples": len(self._samples),
            "sample_interval_seconds": self.interval_seconds,
            "mean_gpu_utilization_pct": float(statistics.fmean(utilization)),
            "peak_vram_mib": float(max(memory)),
            "sample_identity_sha256": sha256_json(self._samples),
        }


def _latest_fresh_attempt(state: Mapping[str, Any]) -> Mapping[str, Any]:
    if state.get("status") != "completed":
        raise ContractError("workflow state is not completed")
    attempts = state.get("execution_attempts")
    latest = state.get("latest_execution_attempt_id")
    if not isinstance(attempts, list) or not isinstance(latest, int) or latest < 1:
        raise ContractError("workflow state has no execution-attempt timing")
    try:
        attempt = attempts[latest - 1]
    except IndexError as error:
        raise ContractError("workflow latest execution attempt is absent") from error
    if (
        not isinstance(attempt, Mapping)
        or attempt.get("fresh_direct_benchmark_eligible") is not True
        or attempt.get("resume_or_cache_used") is not False
        or attempt.get("resume_requested") is not False
    ):
        raise ContractError("resume/cached workflow cannot be reported as a fresh direct benchmark")
    records = attempt.get("stages")
    if not isinstance(records, list):
        raise ContractError("workflow attempt stage timing is absent")
    by_stage = {str(row.get("stage")): row for row in records if isinstance(row, Mapping)}
    if set(_CORE_STAGES) - set(by_stage):
        raise ContractError("workflow attempt is missing a required stage timing")
    if any(by_stage[name].get("execution_status") != "executed_fresh" for name in _CORE_STAGES):
        raise ContractError("workflow stage cache status is not fresh")
    return attempt


def _production_summary(
    *,
    plan: Mapping[str, Any],
    workflow_output: Path,
    outer_wall_seconds: float,
    hardware_preflight: Mapping[str, Any],
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    state_path = workflow_output / "workflow_state.json"
    state = read_json(state_path)
    _sealed(state, "state_identity_sha256", role="workflow state")
    attempt = _latest_fresh_attempt(state)
    analysis_plan_path = workflow_output / "analysis_plan.json"
    if sha256_file(analysis_plan_path) != state.get("analysis_plan_sha256"):
        raise ContractError("workflow state/analysis plan file SHA mismatch")
    published_plan = read_json(analysis_plan_path)
    if published_plan.get("plan_identity_sha256") != plan.get("plan_identity_sha256"):
        raise ContractError("benchmark and executed workflow plan identities differ")
    stageb_path = workflow_output / "stageb" / "summary.json"
    stageb = read_json(stageb_path)
    _sealed(stageb, "summary_identity_sha256", role="Stage-B summary")
    images = int(stageb.get("images", -1))
    if images != int(plan.get("tasks", -2)):
        raise ContractError("workflow/Stage-B image count mismatch")
    records = stageb.get("records")
    if not isinstance(records, list) or len(records) != images:
        raise ContractError("Stage-B per-source records are incomplete")
    if any(record.get("resumed") is not False for record in records):
        raise ContractError("Stage-B resumed records invalidate a direct benchmark")
    expected_task_ids = plan.get("task_ids")
    if (
        not isinstance(expected_task_ids, list)
        or len(expected_task_ids) != images
        or [str(record.get("task_id")) for record in records]
        != list(map(str, expected_task_ids))
    ):
        raise ContractError("Stage-B task identities differ from the locked plan")
    source_units_in_order = list(map(str, expected_task_ids))
    if stageb.get("resumed_images") != 0:
        raise ContractError("Stage-B resumed image count invalidates a direct benchmark")
    checkpoints = stageb.get("checkpoint_sha256")
    if (
        not isinstance(checkpoints, list)
        or len(checkpoints) != 5
        or len(set(checkpoints)) != 5
        or not all(_is_sha256(value) for value in checkpoints)
    ):
        raise ContractError("production Stage-B checkpoint binding is invalid")
    if not _is_sha256(stageb.get("nvidia_smi_preflight_sha256")):
        raise ContractError("production Stage-B preflight hash is invalid")
    megapixels = sum(
        _finite_number(record.get("source_megapixels"), field="source_megapixels", positive=True)
        for record in records
    )
    wall = _finite_number(outer_wall_seconds, field="outer_wall_seconds", positive=True)
    stage_timings = []
    for record in attempt["stages"]:
        if record.get("stage") not in _CORE_STAGES:
            continue
        stage_timings.append(
            {
                "stage": record["stage"],
                "wall_seconds": _finite_number(
                    record.get("wall_seconds_including_stage_io_and_evidence_hashing"),
                    field=f"{record.get('stage')}.wall_seconds",
                ),
                "cache_status": record["execution_status"],
                "evidence_tree_identity_sha256": record.get(
                    "evidence_tree_identity_sha256"
                ),
            }
        )
    if sum(row["wall_seconds"] for row in stage_timings) > wall * 1.02:
        raise ContractError("non-overlapping workflow stage walls exceed outer batch wall")
    for row in stage_timings:
        row["fraction_of_batch_wall"] = float(row["wall_seconds"] / wall)
    attributed_stage_wall = sum(row["wall_seconds"] for row in stage_timings)
    for field in _BINDING_FIELDS:
        if state.get(field) != plan.get(field) or stageb.get(field) != plan.get(field):
            raise ContractError(f"production benchmark model-contract drift: {field}")
    downstream_receipts = (
        (workflow_output / "fusion/fusion_summary.json", "summary_identity_sha256", "fusion"),
        (workflow_output / "traits/summary.json", "export_identity_sha256", "traits"),
        (
            workflow_output / "distal_axis_profiles/summary.json",
            "export_identity_sha256",
            "profiles",
        ),
    )
    for path, identity_field, role in downstream_receipts:
        payload = read_json(path)
        _sealed(payload, identity_field, role=f"{role} summary")
        for field in _BINDING_FIELDS:
            if payload.get(field) != plan.get(field):
                raise ContractError(f"{role} summary model-contract drift: {field}")
        if payload.get("blind_images_used") != 0:
            raise ContractError(f"{role} summary is blind-tainted")
    if int(telemetry.get("samples", 0)) <= 0:
        raise ContractError("production benchmark has no GPU telemetry samples")
    hardware_payload = hardware_preflight.get("hardware")
    if not isinstance(hardware_payload, Mapping) or hardware_preflight.get(
        "hardware_identity_sha256"
    ) != sha256_json(hardware_payload):
        raise ContractError("production benchmark hardware identity mismatch")
    preflight_payload = hardware_preflight.get("nvidia_smi_preflight")
    if not isinstance(preflight_payload, Mapping) or hardware_preflight.get(
        "nvidia_smi_preflight_identity_sha256"
    ) != sha256_json(preflight_payload):
        raise ContractError("production benchmark nvidia-smi preflight identity mismatch")
    official_hashes = {
        "workflow_state_sha256": sha256_file(state_path),
        "analysis_plan_sha256": sha256_file(analysis_plan_path),
        "stageb_summary_sha256": sha256_file(stageb_path),
        "fusion_summary_sha256": sha256_file(workflow_output / "fusion/fusion_summary.json"),
        "traits_summary_sha256": sha256_file(workflow_output / "traits/summary.json"),
        "profiles_summary_sha256": sha256_file(
            workflow_output / "distal_axis_profiles/summary.json"
        ),
    }
    summary: dict[str, Any] = {
        "schema_version": PRODUCTION_SCHEMA,
        "benchmark_system": PHAXIS_BENCHMARK_SYSTEM,
        "status": (
            "completed_direct_full283"
            if images == 283
            else "completed_direct_synthetic_cpu"
        ),
        "benchmark_mode": PRODUCTION_MODE,
        "benchmark_scope_class": "full_workflow",
        "measurement_scope": MEASUREMENT_SCOPE,
        "images": images,
        "n_images": images,
        "pixels": int(round(megapixels * 1_000_000)),
        "megapixels": megapixels,
        "batch_wall_seconds": wall,
        "images_per_min": images * 60.0 / wall,
        "megapixels_per_second": megapixels / wall,
        "includes_io": True,
        "includes_preprocess": True,
        "includes_stitching_fusion_traits_profiles": True,
        "stage_timing_semantics": "nonoverlapping_wall_components",
        "timing_granularity": "direct_batch_stage_wall",
        "stage_timings": stage_timings,
        "attributed_stage_wall_seconds": attributed_stage_wall,
        "unattributed_orchestration_wall_seconds": max(
            0.0, wall - attributed_stage_wall
        ),
        "unattributed_orchestration_fraction": max(
            0.0, wall - attributed_stage_wall
        ) / wall,
        "per_image_latency_reported": False,
        "per_image_latency_reason": "parallel/batch stage wall is not divided across images",
        "fresh_direct_run": True,
        "resume_or_cache_used": False,
        "peak_vram_mib": _finite_number(telemetry.get("peak_vram_mib"), field="peak_vram_mib"),
        "mean_gpu_utilization_pct": _finite_number(
            telemetry.get("mean_gpu_utilization_pct"), field="mean_gpu_utilization_pct"
        ),
        "gpu_telemetry": dict(telemetry),
        "hardware": hardware_preflight["hardware"],
        "hardware_identity_sha256": hardware_preflight["hardware_identity_sha256"],
        "nvidia_smi_preflight": hardware_preflight["nvidia_smi_preflight"],
        "nvidia_smi_preflight_identity_sha256": hardware_preflight[
            "nvidia_smi_preflight_identity_sha256"
        ],
        "physical_gpu_mapping": list(_physical_gpus(plan)),
        "cuda_visible_devices_by_stage": {
            "root_provider_v1": ",".join(
                map(str, plan["stages"][0]["estimated_gpu"]["v1_physical_gpus"])
            ),
            "root_provider_q8": ",".join(
                map(str, plan["stages"][0]["estimated_gpu"]["q8_physical_gpus"])
            ),
            "stageb_train399": str(
                plan["stages"][1]["estimated_gpu"]["physical_gpu"]
            ),
        },
        "stageb_nvidia_smi_preflight_sha256": stageb.get(
            "nvidia_smi_preflight_sha256"
        ),
        "checkpoint_sha256": checkpoints,
        "source_manifest_sha256": _source_manifest_sha256(plan),
        "workflow_manifest_sha256": plan["manifest_sha256"],
        "workflow_manifest_identity_sha256": plan["manifest_identity_sha256"],
        "source_image_lock_identity_sha256": plan["task_identity_sha256"],
        "source_units_in_order": source_units_in_order,
        "source_unit_ordered_set_identity_sha256": sha256_json(
            source_units_in_order
        ),
        "analysis_plan_identity_sha256": plan["plan_identity_sha256"],
        "stage_input_hashes": {
            str(stage["name"]): dict(stage.get("input_hashes", {}))
            for stage in plan["stages"]
        },
        "input_hashes": {
            "raw_image_manifest": _source_manifest_sha256(plan),
            "workflow_manifest": plan["manifest_sha256"],
            "workflow_manifest_identity": plan["manifest_identity_sha256"],
            "source_image_lock_identity": plan["task_identity_sha256"],
            "model_contract_proposal": plan[
                "model_contract_proposal_sha256"
            ],
            "model_contract_proposal_identity": plan[
                "model_contract_proposal_identity_sha256"
            ],
        },
        "config_hashes": {
            "analysis_plan_identity": plan["plan_identity_sha256"],
            "workflow_module": plan["stages"][0]["input_hashes"][
                "workflow_module"
            ],
        },
        "output_hashes": official_hashes,
        **{field: plan[field] for field in _BINDING_FIELDS},
        "official_output_hashes": official_hashes,
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        "rootcap_region_metric": False,
    }
    summary["summary_identity_sha256"] = sha256_json(summary)
    return summary


def run_production_batch_benchmark(
    *,
    manifest: str | Path,
    workflow_output: str | Path,
    benchmark_output: str | Path,
    workflow_runner: Callable[..., Mapping[str, Any]] = run_analysis,
    hardware_capture: Callable[[Sequence[int]], Mapping[str, Any]] = capture_hardware_preflight,
    telemetry_factory: Callable[[tuple[int, ...]], Any] = GpuTelemetry,
) -> dict[str, Any]:
    """Execute and atomically publish a fresh production-batch receipt."""

    workflow_path = Path(workflow_output).resolve()
    benchmark_path = Path(benchmark_output).resolve()
    if workflow_path.exists():
        raise FileExistsError("production benchmark requires a new workflow output")
    if benchmark_path.exists():
        raise FileExistsError("refusing to overwrite benchmark output")
    plan = build_analysis_plan(manifest, output=workflow_path, review_overlays=False)
    physical_gpus = _physical_gpus(plan)
    hardware = dict(hardware_capture(physical_gpus))
    telemetry = telemetry_factory(physical_gpus)
    telemetry.start()
    started = time.perf_counter()
    try:
        workflow_runner(
            manifest,
            output=workflow_path,
            resume=False,
            review_overlays=False,
        )
    finally:
        outer_wall = time.perf_counter() - started
        telemetry_receipt = telemetry.stop()
    summary = _production_summary(
        plan=plan,
        workflow_output=workflow_path,
        outer_wall_seconds=outer_wall,
        hardware_preflight=hardware,
        telemetry=telemetry_receipt,
    )

    def produce(attempt: Path) -> None:
        atomic_write_json(attempt / "runtime_summary.json", summary)

    _publish_directory(benchmark_path, produce)
    return summary


def _read_latency_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PER_IMAGE_FIELDS:
            raise ContractError(
                "runtime_per_image CSV columns/order must be " + ",".join(PER_IMAGE_FIELDS)
            )
        raw_rows = list(reader)
    if not raw_rows:
        raise ContractError("runtime_per_image CSV is empty")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows, start=2):
        source = str(raw.get("source_unit", "")).strip()
        if not source or source in seen:
            raise ContractError(f"invalid or duplicate source_unit at row {index}")
        seen.add(source)
        row: dict[str, Any] = {"source_unit": source}
        for field in PER_IMAGE_FIELDS[1:]:
            row[field] = _finite_number(
                raw.get(field), field=f"row {index}.{field}", positive=field in {"wall_seconds", "megapixels"}
            )
        components = sum(row[field] for field in PER_IMAGE_FIELDS[3:])
        wall = row["wall_seconds"]
        if components > wall * 1.02 or components < wall * 0.98:
            raise ContractError(
                f"row {index}: non-overlapping components do not reconstruct wall within 2%"
            )
        result.append(row)
    return result


def compile_sequential_latency_benchmark(
    *,
    manifest: str | Path,
    workflow_output: str | Path,
    trace_csv: str | Path,
    trace_receipt: str | Path,
    benchmark_output: str | Path,
) -> dict[str, Any]:
    """Validate and publish a real per-source sequential latency trace."""

    plan = build_analysis_plan(manifest, output=workflow_output, review_overlays=False)
    csv_path = Path(trace_csv).resolve()
    receipt_path = Path(trace_receipt).resolve()
    output_path = Path(benchmark_output).resolve()
    receipt = read_json(receipt_path)
    if receipt.get("schema_version") != LATENCY_TRACE_SCHEMA:
        raise ContractError("unsupported sequential latency trace schema")
    _sealed(receipt, "trace_identity_sha256", role="sequential latency trace")
    mode = receipt.get("latency_mode")
    if mode not in LATENCY_MODES:
        raise ContractError("unsupported sequential latency mode")
    if receipt.get("measurement_scope") != MEASUREMENT_SCOPE:
        raise ContractError("sequential latency scope is not raw-image-to-final")
    if receipt.get("per_image_csv_sha256") != sha256_file(csv_path):
        raise ContractError("sequential latency CSV hash mismatch")
    if (
        receipt.get("stage_timing_semantics") != "nonoverlapping_wall_components"
        or receipt.get("measurement_method")
        != "direct_per_source_perf_counter_start_to_final_profile_visibility"
        or receipt.get("includes_io") is not True
        or receipt.get("includes_preprocess") is not True
        or receipt.get("includes_stitching_fusion_traits_profiles") is not True
        or receipt.get("resume_or_cache_used") is not False
        or receipt.get("blind_images_used") != 0
        or receipt.get("rootcap_region_metric") is not False
    ):
        raise ContractError("sequential latency trace guards or timing semantics changed")
    if mode == PERSISTENT_LATENCY_MODE:
        if (
            receipt.get("persistent_worker_and_models") is not True
            or receipt.get("model_or_process_startup_per_source") is not False
            or receipt.get("startup_included_in_per_image_wall") is not False
        ):
            raise ContractError("persistent latency trace startup semantics are invalid")
    else:
        if (
            receipt.get("persistent_worker_and_models") is not False
            or receipt.get("model_or_process_startup_per_source") is not True
            or receipt.get("startup_included_in_per_image_wall") is not True
        ):
            raise ContractError("cold CLI latency trace startup semantics are invalid")
    expected_plan = {
        "source_manifest_sha256": _source_manifest_sha256(plan),
        "workflow_manifest_sha256": plan["manifest_sha256"],
        "workflow_manifest_identity_sha256": plan["manifest_identity_sha256"],
        "source_image_lock_identity_sha256": plan["task_identity_sha256"],
        **{field: plan[field] for field in _BINDING_FIELDS},
    }
    for field, expected in expected_plan.items():
        if receipt.get(field) != expected:
            raise ContractError(f"sequential latency trace input binding mismatch: {field}")
    rows = _read_latency_rows(csv_path)
    if len(rows) != int(plan["tasks"]):
        raise ContractError("sequential latency trace task count differs from manifest")
    expected_task_ids = plan.get("task_ids")
    if not isinstance(expected_task_ids, list) or [
        str(row["source_unit"]) for row in rows
    ] != list(map(str, expected_task_ids)):
        raise ContractError("sequential latency source units differ from the locked manifest")
    source_units_in_order = [str(row["source_unit"]) for row in rows]
    wall_values = [float(row["wall_seconds"]) for row in rows]
    megapixels = sum(float(row["megapixels"]) for row in rows)
    measured_total = sum(wall_values)
    component_totals = {
        field: sum(float(row[field]) for row in rows)
        for field in PER_IMAGE_FIELDS[3:]
    }
    startup = _finite_number(receipt.get("one_time_startup_seconds", 0.0), field="one_time_startup_seconds")
    if mode == COLD_LATENCY_MODE and startup != 0.0:
        raise ContractError("cold CLI startup is per-source and cannot also be counted once")
    elapsed_for_throughput = measured_total + startup
    images = len(rows)
    hardware = receipt.get("hardware")
    if not isinstance(hardware, Mapping) or receipt.get(
        "hardware_identity_sha256"
    ) != sha256_json(hardware):
        raise ContractError("sequential latency hardware identity mismatch")
    nvidia_preflight = receipt.get("nvidia_smi_preflight")
    if not isinstance(nvidia_preflight, Mapping) or receipt.get(
        "nvidia_smi_preflight_identity_sha256"
    ) != sha256_json(nvidia_preflight):
        raise ContractError("sequential latency nvidia-smi preflight identity mismatch")
    checkpoints = receipt.get("checkpoint_sha256")
    if (
        not isinstance(checkpoints, list)
        or len(checkpoints) != 5
        or len(set(checkpoints)) != 5
        or not all(_is_sha256(value) for value in checkpoints)
    ):
        raise ContractError("sequential latency checkpoint binding is invalid")
    if images == 283 and nvidia_preflight.get("synthetic_cpu_oracle") is True:
        raise ContractError("synthetic CPU telemetry cannot publish a final 283 latency receipt")
    input_hashes = _validated_hash_map(receipt.get("input_hashes"), field="input_hashes")
    config_hashes = _validated_hash_map(
        receipt.get("config_hashes"), field="config_hashes"
    )
    output_hashes = _validated_hash_map(
        receipt.get("output_hashes"), field="output_hashes"
    )
    summary: dict[str, Any] = {
        "schema_version": LATENCY_SCHEMA,
        "benchmark_system": PHAXIS_BENCHMARK_SYSTEM,
        "status": (
            "completed_direct_full283"
            if images == 283
            else "completed_direct_synthetic_cpu"
        ),
        "benchmark_mode": mode,
        "benchmark_scope_class": "full_workflow",
        "latency_mode": mode,
        "measurement_scope": MEASUREMENT_SCOPE,
        "images": images,
        "n_images": images,
        "pixels": int(round(megapixels * 1_000_000)),
        "megapixels": megapixels,
        "median_seconds_per_image": float(statistics.median(wall_values)),
        "p95_seconds_per_image": _quantile(wall_values, 0.95),
        "direct_sequential_wall_seconds": measured_total,
        "one_time_startup_seconds": startup,
        "startup_included_in_per_image_wall": receipt[
            "startup_included_in_per_image_wall"
        ],
        "images_per_min": images * 60.0 / elapsed_for_throughput,
        "megapixels_per_second": megapixels / elapsed_for_throughput,
        "includes_io": True,
        "includes_preprocess": True,
        "includes_stitching_fusion_traits_profiles": True,
        "stage_timing_semantics": "nonoverlapping_wall_components",
        "measurement_method": "direct_per_source_perf_counter_start_to_final_profile_visibility",
        "timing_granularity": "direct_per_source_raw_to_final_profile",
        "persistent_worker_and_models": receipt["persistent_worker_and_models"],
        "model_or_process_startup_per_source": receipt[
            "model_or_process_startup_per_source"
        ],
        "component_total_seconds": component_totals,
        "component_fraction_of_sequential_source_wall": {
            field: value / measured_total
            for field, value in component_totals.items()
        },
        "per_image_csv_sha256": sha256_file(csv_path),
        "peak_vram_mib": _finite_number(receipt.get("peak_vram_mib"), field="peak_vram_mib"),
        "mean_gpu_utilization_pct": _finite_number(
            receipt.get("mean_gpu_utilization_pct"), field="mean_gpu_utilization_pct"
        ),
        "hardware": dict(hardware),
        "hardware_identity_sha256": receipt["hardware_identity_sha256"],
        "nvidia_smi_preflight": dict(nvidia_preflight),
        "nvidia_smi_preflight_identity_sha256": receipt[
            "nvidia_smi_preflight_identity_sha256"
        ],
        "physical_gpu_mapping": receipt.get("physical_gpu_mapping"),
        "cuda_visible_devices_by_stage": receipt.get("cuda_visible_devices_by_stage"),
        "checkpoint_sha256": checkpoints,
        "input_hashes": input_hashes,
        "config_hashes": config_hashes,
        "output_hashes": output_hashes,
        **expected_plan,
        "trace_receipt_sha256": sha256_file(receipt_path),
        "trace_identity_sha256": receipt["trace_identity_sha256"],
        "source_units_in_order": source_units_in_order,
        "source_unit_ordered_set_identity_sha256": sha256_json(
            source_units_in_order
        ),
        "fresh_direct_run": True,
        "resume_or_cache_used": False,
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        "rootcap_region_metric": False,
    }
    summary["summary_identity_sha256"] = sha256_json(summary)

    def produce(attempt: Path) -> None:
        copied = attempt / "runtime_per_image.csv"
        shutil.copy2(csv_path, copied)
        if sha256_file(copied) != summary["per_image_csv_sha256"]:
            raise ContractError("copied sequential timing CSV hash changed")
        atomic_write_json(attempt / "runtime_summary.json", summary)

    _publish_directory(output_path, produce)
    return summary


def compare_benchmarks(
    *,
    phaxis_summary: str | Path,
    baseline_summary: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Publish a speedup only for directly comparable full-scope receipts."""

    candidate_path = Path(phaxis_summary).resolve()
    baseline_path = Path(baseline_summary).resolve()
    destination = Path(output).resolve()
    candidate = read_json(candidate_path)
    baseline = read_json(baseline_path)
    for role, payload in (("PHAxis", candidate), ("baseline", baseline)):
        _sealed(payload, "summary_identity_sha256", role=f"{role} benchmark")
    fields = (
        "status",
        "benchmark_mode",
        "measurement_scope",
        "images",
        "source_manifest_sha256",
        "source_image_lock_identity_sha256",
        "source_unit_ordered_set_identity_sha256",
        "hardware_identity_sha256",
        "includes_io",
        "includes_preprocess",
        "includes_stitching_fusion_traits_profiles",
    )
    mismatches = [field for field in fields if candidate.get(field) != baseline.get(field)]
    if candidate.get("benchmark_system") != PHAXIS_BENCHMARK_SYSTEM:
        mismatches.append("PHAxis.benchmark_system")
    if baseline.get("benchmark_system") != FROZEN_V1_BENCHMARK_SYSTEM:
        mismatches.append("baseline.benchmark_system")
    for role, payload in (("PHAxis", candidate), ("baseline", baseline)):
        if payload.get("status") != "completed_direct_full283":
            mismatches.append(f"{role}.status_not_direct_full283")
        if payload.get("measurement_scope") != MEASUREMENT_SCOPE:
            mismatches.append(f"{role}.scope_not_direct_full")
        if payload.get("benchmark_scope_class", "full_workflow") != "full_workflow":
            mismatches.append(f"{role}.component_only_noncomparable")
        if payload.get("resume_or_cache_used") is not False:
            mismatches.append(f"{role}.resume_or_cache_used")
        try:
            _ordered_source_units(payload, role=role, exact283=True)
        except ContractError:
            mismatches.append(f"{role}.ordered_source_set_invalid")
    mismatches = sorted(set(mismatches))
    comparable = not mismatches
    comparison: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA,
        "status": (
            "comparable_direct_full283" if comparable else "not_comparable"
        ),
        "comparable": comparable,
        "noncomparability_reasons": mismatches,
        "phaxis_summary_sha256": sha256_file(candidate_path),
        "baseline_summary_sha256": sha256_file(baseline_path),
        "phaxis_summary_identity_sha256": candidate.get("summary_identity_sha256"),
        "baseline_summary_identity_sha256": baseline.get("summary_identity_sha256"),
        "phaxis_benchmark_system": candidate.get("benchmark_system"),
        "baseline_benchmark_system": baseline.get("benchmark_system"),
        "benchmark_mode": candidate.get("benchmark_mode"),
        "measurement_scope": candidate.get("measurement_scope"),
        "source_unit_ordered_set_identity_sha256": candidate.get(
            "source_unit_ordered_set_identity_sha256"
        ),
        "same_283_source_manifest_hardware_and_io_scope": comparable,
        "historical_component_runtime_used_as_full_baseline": False,
        "blind_images_used": 0,
        "rootcap_region_metric": False,
    }
    if comparable:
        if candidate["benchmark_mode"] == PRODUCTION_MODE:
            candidate_wall = _finite_number(
                candidate.get("batch_wall_seconds"), field="PHAxis.batch_wall_seconds", positive=True
            )
            baseline_wall = _finite_number(
                baseline.get("batch_wall_seconds"), field="baseline.batch_wall_seconds", positive=True
            )
            comparison["batch_wall_speedup_frozen_v1_over_phaxis"] = baseline_wall / candidate_wall
        elif candidate["benchmark_mode"] in LATENCY_MODES:
            candidate_median = _finite_number(
                candidate.get("median_seconds_per_image"), field="PHAxis.median", positive=True
            )
            baseline_median = _finite_number(
                baseline.get("median_seconds_per_image"), field="baseline.median", positive=True
            )
            comparison["median_latency_speedup_frozen_v1_over_phaxis"] = (
                baseline_median / candidate_median
            )
        else:  # guarded by same-mode comparison but still fail closed
            raise ContractError("unsupported comparable benchmark mode")
    comparison["comparison_identity_sha256"] = sha256_json(comparison)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite comparison receipt: {destination}")
    atomic_write_json(destination, comparison)
    return comparison


def _formal_summary(
    path: Path, *, role: str, expected_system: str, expected_kind: str
) -> dict[str, Any]:
    payload = read_json(path)
    identity = _sealed(payload, "summary_identity_sha256", role=role)
    if payload.get("benchmark_system") != expected_system:
        raise ContractError(f"{role}: benchmark system identity changed")
    if payload.get("status") != "completed_direct_full283" or payload.get("images") != 283:
        raise ContractError(f"{role}: benchmark is not completed direct exact283")
    if payload.get("n_images") != 283:
        raise ContractError(f"{role}: n_images is not exact283")
    if (
        payload.get("measurement_scope") != MEASUREMENT_SCOPE
        or payload.get("benchmark_scope_class") != "full_workflow"
        or payload.get("includes_io") is not True
        or payload.get("includes_preprocess") is not True
        or payload.get("includes_stitching_fusion_traits_profiles") is not True
        or payload.get("fresh_direct_run") is not True
        or payload.get("resume_or_cache_used") is not False
        or payload.get("condition_metadata_used_for_routing") is not False
        or payload.get("canonical_annotations_read") is not False
    ):
        raise ContractError(f"{role}: benchmark is not fresh/no-cache I/O-inclusive full workflow")
    if payload.get("blind_images_used") != 0 or payload.get("rootcap_region_metric") is not False:
        raise ContractError(f"{role}: benchmark violates blind/root-cap red lines")
    for field in ("source_manifest_sha256", "source_image_lock_identity_sha256"):
        if not _is_sha256(payload.get(field)):
            raise ContractError(f"{role}: {field} is invalid")
    ordered, ordered_identity = _ordered_source_units(payload, role=role, exact283=True)
    hardware, hardware_identity, uuid_driver_identity = _hardware_uuid_driver_identity(
        payload, role=role
    )
    if expected_kind == "production":
        if (
            payload.get("schema_version") != PRODUCTION_SCHEMA
            or payload.get("benchmark_mode") != PRODUCTION_MODE
            or payload.get("timing_granularity") != "direct_batch_stage_wall"
            or payload.get("per_image_latency_reported") is not False
            or "median_seconds_per_image" in payload
            or "p95_seconds_per_image" in payload
        ):
            raise ContractError(f"{role}: production timing semantics changed")
        _finite_number(payload.get("batch_wall_seconds"), field=f"{role}.batch_wall_seconds", positive=True)
        stages = payload.get("stage_timings")
        if not isinstance(stages, list) or not stages:
            raise ContractError(f"{role}: direct production stage evidence is absent")
        for index, stage in enumerate(stages):
            if not isinstance(stage, Mapping):
                raise ContractError(f"{role}: stage timing row {index} is invalid")
            _finite_number(stage.get("wall_seconds"), field=f"{role}.stage[{index}].wall_seconds")
            if stage.get("cache_status", "executed_fresh") != "executed_fresh":
                raise ContractError(f"{role}: production stage is not fresh")
        output_hashes = payload.get("official_output_hashes", payload.get("output_hashes"))
        _validated_hash_map(output_hashes, field=f"{role}.official_output_hashes")
    elif expected_kind == "latency":
        if (
            payload.get("schema_version") != LATENCY_SCHEMA
            or payload.get("benchmark_mode") not in LATENCY_MODES
            or payload.get("latency_mode") != payload.get("benchmark_mode")
            or payload.get("timing_granularity") != "direct_per_source_raw_to_final_profile"
            or payload.get("stage_timing_semantics") != "nonoverlapping_wall_components"
            or payload.get("measurement_method")
            != "direct_per_source_perf_counter_start_to_final_profile_visibility"
        ):
            raise ContractError(f"{role}: sequential latency semantics changed")
        persistent = payload.get("benchmark_mode") == PERSISTENT_LATENCY_MODE
        if (
            payload.get("persistent_worker_and_models") is not persistent
            or payload.get("model_or_process_startup_per_source") is persistent
            or payload.get("startup_included_in_per_image_wall") is persistent
        ):
            raise ContractError(f"{role}: sequential latency startup semantics changed")
        _finite_number(payload.get("median_seconds_per_image"), field=f"{role}.median", positive=True)
        _finite_number(payload.get("p95_seconds_per_image"), field=f"{role}.p95", positive=True)
        if not _is_sha256(payload.get("per_image_csv_sha256")):
            raise ContractError(f"{role}: per-image trace hash is invalid")
        _validated_hash_map(payload.get("output_hashes"), field=f"{role}.output_hashes")
    else:
        raise AssertionError(expected_kind)
    return {
        "payload": payload,
        "summary_identity_sha256": identity,
        "source_units_in_order": ordered,
        "source_unit_ordered_set_identity_sha256": ordered_identity,
        "hardware": hardware,
        "hardware_identity_sha256": hardware_identity,
        "hardware_uuid_driver_identity_sha256": uuid_driver_identity,
    }


def _formal_comparison(
    path: Path,
    *,
    role: str,
    phaxis_path: Path,
    baseline_path: Path,
    phaxis: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    payload = read_json(path)
    identity = _sealed(payload, "comparison_identity_sha256", role=role)
    mode = phaxis.get("benchmark_mode")
    if (
        payload.get("schema_version") != COMPARISON_SCHEMA
        or payload.get("status") != "comparable_direct_full283"
        or payload.get("comparable") is not True
        or payload.get("noncomparability_reasons") != []
        or payload.get("benchmark_mode") != mode
        or payload.get("measurement_scope") != MEASUREMENT_SCOPE
        or payload.get("same_283_source_manifest_hardware_and_io_scope") is not True
        or payload.get("historical_component_runtime_used_as_full_baseline") is not False
        or payload.get("blind_images_used") != 0
        or payload.get("rootcap_region_metric") is not False
    ):
        raise ContractError(f"{role}: comparison is not a direct comparable full283 receipt")
    expected_bindings = {
        "phaxis_summary_sha256": sha256_file(phaxis_path),
        "baseline_summary_sha256": sha256_file(baseline_path),
        "phaxis_summary_identity_sha256": phaxis.get("summary_identity_sha256"),
        "baseline_summary_identity_sha256": baseline.get("summary_identity_sha256"),
        "phaxis_benchmark_system": PHAXIS_BENCHMARK_SYSTEM,
        "baseline_benchmark_system": FROZEN_V1_BENCHMARK_SYSTEM,
        "source_unit_ordered_set_identity_sha256": phaxis.get(
            "source_unit_ordered_set_identity_sha256"
        ),
    }
    for field, expected in expected_bindings.items():
        if payload.get(field) != expected:
            raise ContractError(f"{role}: comparison binding mismatch: {field}")
    if mode == PRODUCTION_MODE:
        field = "batch_wall_speedup_frozen_v1_over_phaxis"
        expected_speedup = _finite_number(
            baseline.get("batch_wall_seconds"), field="baseline.batch_wall_seconds", positive=True
        ) / _finite_number(phaxis.get("batch_wall_seconds"), field="PHAxis.batch_wall_seconds", positive=True)
    elif mode in LATENCY_MODES:
        field = "median_latency_speedup_frozen_v1_over_phaxis"
        expected_speedup = _finite_number(
            baseline.get("median_seconds_per_image"), field="baseline.median", positive=True
        ) / _finite_number(phaxis.get("median_seconds_per_image"), field="PHAxis.median", positive=True)
    else:
        raise ContractError(f"{role}: unsupported comparison mode")
    observed = _finite_number(payload.get(field), field=f"{role}.{field}", positive=True)
    if not math.isclose(observed, expected_speedup, rel_tol=1e-12, abs_tol=1e-12):
        raise ContractError(f"{role}: speedup was not recomputed from bound summaries")
    return {"payload": payload, "comparison_identity_sha256": identity, "speedup_field": field}


def aggregate_same_hardware_benchmark_receipt(
    *,
    phaxis_production_summary: str | Path,
    phaxis_sequential_summary: str | Path,
    frozen_v1_production_summary: str | Path,
    frozen_v1_sequential_summary: str | Path,
    production_comparison: str | Path,
    sequential_comparison: str | Path,
) -> dict[str, Any]:
    """Recompute and seal the formal exact283 same-hardware handover receipt."""

    paths = {
        "phaxis_production": Path(phaxis_production_summary).resolve(),
        "phaxis_sequential": Path(phaxis_sequential_summary).resolve(),
        "frozen_v1_production": Path(frozen_v1_production_summary).resolve(),
        "frozen_v1_sequential": Path(frozen_v1_sequential_summary).resolve(),
        "production_comparison": Path(production_comparison).resolve(),
        "sequential_comparison": Path(sequential_comparison).resolve(),
    }
    if len(set(paths.values())) != len(paths) or any(not path.is_file() for path in paths.values()):
        raise ContractError("same-hardware aggregation requires six distinct explicit files")
    summaries = {
        "phaxis_production": _formal_summary(paths["phaxis_production"], role="PHAxis production", expected_system=PHAXIS_BENCHMARK_SYSTEM, expected_kind="production"),
        "phaxis_sequential": _formal_summary(paths["phaxis_sequential"], role="PHAxis sequential", expected_system=PHAXIS_BENCHMARK_SYSTEM, expected_kind="latency"),
        "frozen_v1_production": _formal_summary(paths["frozen_v1_production"], role="frozen v1 production", expected_system=FROZEN_V1_BENCHMARK_SYSTEM, expected_kind="production"),
        "frozen_v1_sequential": _formal_summary(paths["frozen_v1_sequential"], role="frozen v1 sequential", expected_system=FROZEN_V1_BENCHMARK_SYSTEM, expected_kind="latency"),
    }
    payloads = {role: record["payload"] for role, record in summaries.items()}
    ordered_lists = {tuple(record["source_units_in_order"]) for record in summaries.values()}
    if len(ordered_lists) != 1:
        raise ContractError("PHAxis/frozen-v1 modes did not use the identical ordered exact283 source set")
    for field in (
        "source_manifest_sha256",
        "source_image_lock_identity_sha256",
        "source_unit_ordered_set_identity_sha256",
        "hardware_identity_sha256",
        "hardware_uuid_driver_identity_sha256",
    ):
        values = {
            record[field] if field in record else record["payload"].get(field)
            for record in summaries.values()
        }
        if len(values) != 1:
            raise ContractError(f"same-hardware exact283 cross-mode mismatch: {field}")
    if len({sha256_json(record["hardware"]) for record in summaries.values()}) != 1:
        raise ContractError("same-hardware summaries contain different hardware inventories")
    latency_modes = {
        payloads["phaxis_sequential"].get("benchmark_mode"),
        payloads["frozen_v1_sequential"].get("benchmark_mode"),
    }
    if len(latency_modes) != 1:
        raise ContractError("PHAxis and frozen v1 used different sequential latency modes")
    comparisons = {
        "production": _formal_comparison(
            paths["production_comparison"], role="production comparison",
            phaxis_path=paths["phaxis_production"], baseline_path=paths["frozen_v1_production"],
            phaxis=payloads["phaxis_production"], baseline=payloads["frozen_v1_production"],
        ),
        "sequential": _formal_comparison(
            paths["sequential_comparison"], role="sequential comparison",
            phaxis_path=paths["phaxis_sequential"], baseline_path=paths["frozen_v1_sequential"],
            phaxis=payloads["phaxis_sequential"], baseline=payloads["frozen_v1_sequential"],
        ),
    }
    hardware_identity = summaries["phaxis_production"]["hardware_identity_sha256"]
    ordered_identity = summaries["phaxis_production"]["source_unit_ordered_set_identity_sha256"]
    runs = []
    for role in ("phaxis_production", "phaxis_sequential", "frozen_v1_production", "frozen_v1_sequential"):
        item = summaries[role]
        run = item["payload"]
        runs.append(
            {
                "role": role,
                "benchmark_system": run["benchmark_system"],
                "mode": run["benchmark_mode"],
                "summary_sha256": sha256_file(paths[role]),
                "summary_identity_sha256": item["summary_identity_sha256"],
                "hardware_identity_sha256": hardware_identity,
                "source_unit_ordered_set_identity_sha256": ordered_identity,
                "fresh_direct_run": True,
                "resume_or_cache_used": False,
                "full_workflow_io_included": True,
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": SAME_HARDWARE_RECEIPT_SCHEMA,
        "status": "passed",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "images": 283,
        "measurement_scope": MEASUREMENT_SCOPE,
        "latency_mode": next(iter(latency_modes)),
        "source_manifest_sha256": payloads["phaxis_production"]["source_manifest_sha256"],
        "source_image_lock_identity_sha256": payloads["phaxis_production"]["source_image_lock_identity_sha256"],
        "source_unit_ordered_set_identity_sha256": ordered_identity,
        "hardware": summaries["phaxis_production"]["hardware"],
        "hardware_identity_sha256": hardware_identity,
        "hardware_uuid_driver_identity_sha256": summaries["phaxis_production"]["hardware_uuid_driver_identity_sha256"],
        "runs": runs,
        "comparisons": {
            role: {
                "comparison_sha256": sha256_file(paths[f"{role}_comparison"]),
                "comparison_identity_sha256": record["comparison_identity_sha256"],
                record["speedup_field"]: record["payload"][record["speedup_field"]],
            }
            for role, record in comparisons.items()
        },
        "same_ordered_exact283_sources": True,
        "same_hardware_uuid_and_driver": True,
        "same_io_and_full_workflow_scope": True,
        "fresh_no_cache": True,
        "historical_98_47_min_component_receipt_used": False,
        "forward_only_runtime_used": False,
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }
    receipt["receipt_identity_sha256"] = sha256_json(receipt)
    return receipt


def same_hardware_benchmark_plan(**paths: str | Path) -> dict[str, Any]:
    receipt = aggregate_same_hardware_benchmark_receipt(**paths)
    plan: dict[str, Any] = {
        "schema_version": SAME_HARDWARE_PLAN_SCHEMA,
        "status": "validated_not_published",
        "default_check_only": True,
        "publish_requires_explicit_execute": True,
        "candidate_receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "images": 283,
        "latency_mode": receipt["latency_mode"],
        "hardware_identity_sha256": receipt["hardware_identity_sha256"],
        "source_unit_ordered_set_identity_sha256": receipt[
            "source_unit_ordered_set_identity_sha256"
        ],
        "blind_images_used": 0,
    }
    plan["plan_identity_sha256"] = sha256_json(plan)
    return plan


def publish_same_hardware_benchmark_receipt(*, output: str | Path, **paths: str | Path) -> dict[str, Any]:
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite same-hardware receipt: {destination}")
    receipt = aggregate_same_hardware_benchmark_receipt(**paths)
    atomic_write_json(destination, receipt)
    return receipt


def _frozen_v1_source_lock(path: Path) -> dict[str, str]:
    """Validate the legacy benchmark source scope without executing a workflow."""

    if not path.is_file() or path.is_symlink():
        raise ContractError("frozen-v1 producer source manifest is absent or a symlink")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 283:
        raise ContractError("frozen-v1 producer source manifest is not exact283")
    ordered: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        task = str(row.get("task_id", "")).strip()
        image_hash = str(row.get("image_sha256", "")).strip().lower()
        image_path = str(row.get("image_path", "")).strip()
        if not task or task in seen:
            raise ContractError(f"frozen-v1 source row {index}: duplicate/absent task_id")
        if not _is_sha256(image_hash):
            raise ContractError(f"frozen-v1 source row {index}: invalid image SHA-256")
        if "blind" in image_path.casefold() or "blind" in task.casefold():
            raise ContractError("blind/final-validation source refused by legacy producer gate")
        seen.add(task)
        ordered.append({"task_id": task, "image_sha256": image_hash})
    return {
        "source_manifest_sha256": sha256_file(path),
        "ordered_source_lock_identity_sha256": sha256_json(ordered),
    }


def inspect_frozen_v1_exact283_benchmark_producer(
    *,
    project_root: str | Path,
    source_manifest: str | Path,
    producer_interface: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect the missing legacy benchmark producer without running CUDA.

    This diagnostic can establish whether a future adapter implements the
    required *interface*.  It can never stand in for either measured legacy
    summary and intentionally uses neither a formal summary schema nor a
    ``summary_identity_sha256`` field.
    """

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ContractError("frozen-v1 producer project root is absent")
    manifest_path = Path(source_manifest).resolve()
    source_lock = _frozen_v1_source_lock(manifest_path)
    partial_specs = (
        (
            "sharded_inference",
            "scripts/run_six_condition_v1_sharded.py",
            "restartable component inference; no fresh raw-to-final receipt",
        ),
        (
            "shard_merge",
            "scripts/merge_six_condition_v1_shards.py",
            "merge utility; concurrent shard elapsed times are not sequential latency",
        ),
        (
            "prefill_adapter",
            "scripts/materialize_six_condition_v1_prefill_adapter.py",
            "materialisation utility; no traits/profiles full-workflow timing authority",
        ),
    )
    partial: list[dict[str, Any]] = []
    for role, relative, limitation in partial_specs:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ContractError("legacy partial component escapes project root") from error
        if path.is_file() and not path.is_symlink():
            partial.append(
                {
                    "role": role,
                    "path": relative,
                    "sha256": sha256_file(path),
                    "formal_eligible": False,
                    "limitation": limitation,
                }
            )

    requirement_names = (
        "fresh_full_workflow_production_adapter",
        "cold_cli_sequential_adapter",
        "exact283_source_order_and_image_hash_lock",
        "hardware_uuid_driver_and_preflight_binding",
        "nonoverlapping_stage_timing_hooks",
    )
    interface_lock: dict[str, Any] | None = None
    satisfied: set[str] = {"exact283_source_order_and_image_hash_lock"}
    if producer_interface is not None:
        interface_path = Path(producer_interface).resolve()
        interface = read_json(interface_path)
        _sealed(
            interface,
            "interface_identity_sha256",
            role="frozen-v1 exact283 producer interface",
        )
        if interface.get("schema_version") != FROZEN_V1_PRODUCER_INTERFACE_SCHEMA:
            raise ContractError("unsupported frozen-v1 producer interface schema")
        if interface.get("target_system_id") != FROZEN_V1_BENCHMARK_SYSTEM:
            raise ContractError("frozen-v1 producer interface target system changed")
        if interface.get("source_manifest_sha256") != source_lock["source_manifest_sha256"]:
            raise ContractError("frozen-v1 producer interface source manifest drifted")
        if (
            interface.get("ordered_source_lock_identity_sha256")
            != source_lock["ordered_source_lock_identity_sha256"]
        ):
            raise ContractError("frozen-v1 producer interface source order drifted")
        capabilities = interface.get("capabilities")
        if not isinstance(capabilities, Mapping):
            raise ContractError("frozen-v1 producer interface capabilities are absent")
        satisfied.update(
            name for name in requirement_names if capabilities.get(name) is True
        )
        entrypoints = interface.get("entrypoints")
        if not (
            isinstance(entrypoints, Mapping)
            and set(entrypoints)
            == {
                "execute_fresh_production_exact283",
                "execute_fresh_sequential_exact283",
            }
            and all(isinstance(value, Mapping) for value in entrypoints.values())
        ):
            raise ContractError("frozen-v1 producer interface entrypoints are incomplete")
        for role, record in entrypoints.items():
            relative_raw = record.get("path")
            expected_sha = record.get("sha256")
            if not (
                isinstance(relative_raw, str)
                and relative_raw
                and _is_sha256(expected_sha)
            ):
                raise ContractError(
                    f"frozen-v1 producer interface entrypoint is invalid: {role}"
                )
            relative = Path(relative_raw)
            if relative.is_absolute() or ".." in relative.parts:
                raise ContractError(
                    f"frozen-v1 producer interface entrypoint path is unsafe: {role}"
                )
            entrypoint = (root / relative).resolve()
            try:
                entrypoint.relative_to(root)
            except ValueError as error:
                raise ContractError(
                    f"frozen-v1 producer interface entrypoint escapes project: {role}"
                ) from error
            if (
                not entrypoint.is_file()
                or entrypoint.is_symlink()
                or sha256_file(entrypoint) != expected_sha
            ):
                raise ContractError(
                    f"frozen-v1 producer interface entrypoint drifted: {role}"
                )
        interface_lock = {
            "path": str(interface_path),
            "sha256": sha256_file(interface_path),
            "interface_identity_sha256": interface["interface_identity_sha256"],
        }

    blocker_by_requirement = {
        "fresh_full_workflow_production_adapter": "FROZEN_V1_PRODUCTION_PRODUCER_MISSING",
        "cold_cli_sequential_adapter": "FROZEN_V1_SEQUENTIAL_PRODUCER_MISSING",
        "exact283_source_order_and_image_hash_lock": "FROZEN_V1_SOURCE_LOCK_MISSING",
        "hardware_uuid_driver_and_preflight_binding": "FROZEN_V1_FORMAL_HARDWARE_BINDING_MISSING",
        "nonoverlapping_stage_timing_hooks": "FROZEN_V1_NONOVERLAPPING_STAGE_TIMING_HOOKS_MISSING",
    }
    missing = [name for name in requirement_names if name not in satisfied]
    payload: dict[str, Any] = {
        "schema_version": FROZEN_V1_PRODUCER_GATE_SCHEMA,
        "status": (
            "ready_interface_only_non_formal"
            if not missing
            else "blocked_missing_real_producer"
        ),
        "target_system_id": FROZEN_V1_BENCHMARK_SYSTEM,
        "source_scope_contract": "exact283_locked_ordered_sources",
        **source_lock,
        "producer_interface": interface_lock,
        "requirements": [
            {"name": name, "satisfied": name in satisfied}
            for name in requirement_names
        ],
        "blocker_codes": [blocker_by_requirement[name] for name in missing],
        "discovered_partial_components": partial,
        "cpu_only_inspection": True,
        "benchmark_execution_performed": False,
        "performance_measurements_present": False,
        "formal_summary_schema_emitted": False,
        "formal_aggregation_allowed": False,
        "formal_result_receipt": False,
        "blind_images_used": 0,
        "rootcap_region_metric": False,
    }
    payload["gate_identity_sha256"] = sha256_json(payload)
    return payload


__all__ = [
    "COLD_LATENCY_MODE",
    "COMPARISON_SCHEMA",
    "FROZEN_V1_BENCHMARK_SYSTEM",
    "FROZEN_V1_PRODUCER_GATE_SCHEMA",
    "FROZEN_V1_PRODUCER_INTERFACE_SCHEMA",
    "GpuTelemetry",
    "LATENCY_MODES",
    "LATENCY_SCHEMA",
    "LATENCY_TRACE_SCHEMA",
    "MEASUREMENT_SCOPE",
    "PHAXIS_BENCHMARK_SYSTEM",
    "PER_IMAGE_FIELDS",
    "PERSISTENT_LATENCY_MODE",
    "PRODUCTION_MODE",
    "PRODUCTION_SCHEMA",
    "SAME_HARDWARE_PLAN_SCHEMA",
    "SAME_HARDWARE_RECEIPT_SCHEMA",
    "aggregate_same_hardware_benchmark_receipt",
    "benchmark_plan",
    "capture_hardware_preflight",
    "compare_benchmarks",
    "compile_sequential_latency_benchmark",
    "inspect_frozen_v1_exact283_benchmark_producer",
    "run_production_batch_benchmark",
    "same_hardware_benchmark_plan",
    "publish_same_hardware_benchmark_receipt",
]
