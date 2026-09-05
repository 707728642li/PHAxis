"""Restartable orchestration for raw images -> frozen Hybrid-Max root output.

Importing this module performs no filesystem, subprocess, CUDA, or model work.
All effects are behind :func:`run_pipeline` and every executable command is
constructed from explicit caller paths plus verified bundle-relative assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json
from phaxis.root_provider.bundle import verify_bundle
from phaxis.root_provider.reference import audit_fresh_reference


PIPELINE_SCHEMA = "PHAxis-root-provider-portable-pipeline-1.0"
STRICT_PHYSICAL_GPU_ENV = "PHAXIS_REQUIRE_EXACT_PHYSICAL_GPU"


@dataclass(frozen=True)
class PipelineConfig:
    project: Path
    bundle: Path
    input_manifest: Path
    acquisition_gate: Path
    deployment_metadata: Path
    canonical_manifest: Path
    deployment_manifest: Path
    deployment_lock: Path
    image_root: Path
    output: Path
    v1_physical_gpus: tuple[int, ...]
    q8_physical_gpus: tuple[int, ...]
    python_executable: Path = Path(sys.executable)
    v1_shards: int = 4
    v1_concurrency: int = 2
    v20_shards: int = 8
    v20_concurrency: int = 8
    q8_shards: int = 8
    q8_concurrency: int = 1
    field_batch_size: int = 10
    query_batch_size: int = 32
    reference_registry: Path | None = None
    strict_physical_gpu: bool = False

    def resolved(self) -> "PipelineConfig":
        return PipelineConfig(
            **{
                **self.__dict__,
                **{
                    field: Path(getattr(self, field)).resolve()
                    for field in (
                        "project",
                        "bundle",
                        "input_manifest",
                        "acquisition_gate",
                        "deployment_metadata",
                        "canonical_manifest",
                        "deployment_manifest",
                        "deployment_lock",
                        "image_root",
                        "output",
                        "python_executable",
                    )
                },
                "reference_registry": (
                    Path(self.reference_registry).resolve()
                    if self.reference_registry is not None
                    else None
                ),
            }
        )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _partition(values: Sequence[str], count: int) -> list[list[str]]:
    if count <= 0 or count > len(values):
        raise ValueError("shard count must be in [1, task count]")
    width = math.ceil(len(values) / count)
    return [list(values[start : start + width]) for start in range(0, len(values), width)]


def _stage_command(config: PipelineConfig, stage: str, *arguments: Any) -> list[str]:
    return [
        str(config.python_executable),
        "-m",
        "phaxis.root_provider.stage_entry",
        stage,
        "--bundle",
        str(config.bundle),
        *map(str, arguments),
    ]


def _gpu_for(devices: Sequence[int], shard_index: int) -> int:
    if not devices:
        raise ValueError("at least one physical GPU must be explicit")
    if any(index < 0 for index in devices):
        raise ValueError("physical GPU indices must be non-negative")
    return devices[shard_index % len(devices)]


def _fixed_paths(config: PipelineConfig) -> dict[str, Path]:
    output = config.output
    return {
        "prepare": output / "prepare",
        "v1_orchestration": output / "v1_orchestration",
        "v1_merged": output / "v1_merged",
        "v1_materialized": output / "v1_materialized",
        "v20_orchestration": output / "v20_orchestration",
        "v20_merged": output / "v20_merged",
        "q8_shards": output / "q8_shards",
        "q8_merged": output / "q8_merged",
        "hybrid": output / "hybrid",
        "audit": output / "fresh_reference_audit.json",
    }


def _common_commands(config: PipelineConfig) -> dict[str, list[str]]:
    paths = _fixed_paths(config)
    legacy = config.bundle / "legacy_project"
    runtime_config = legacy / "configs/rhpheno_dual_mode_runtime_v6_v19_prospective.json"
    v20_config = legacy / "configs/rhaxis_v20_assisted_review_rootcap_recall_dev_v20.json"
    return {
        "prepare": _stage_command(
            config,
            "prepare",
            "--input-manifest", config.input_manifest,
            "--acquisition-gate", config.acquisition_gate,
            "--output", paths["prepare"],
        ),
        "merge_v1": _stage_command(
            config,
            "merge-v1",
            "--orchestration-root", paths["v1_orchestration"],
            "--output", paths["v1_merged"],
        ),
        "materialize_v1": _stage_command(
            config,
            "materialize-v1",
            "--prepared-manifest", paths["prepare"] / "manifests/sparse_instance.csv",
            "--deployment-metadata", config.deployment_metadata,
            "--canonical-manifest", config.canonical_manifest,
            "--v1-run-root", paths["v1_merged"],
            "--runtime-config", runtime_config,
            "--output", paths["v1_materialized"],
        ),
        "merge_v20": _stage_command(
            config,
            "merge-v20",
            "--orchestration-root", paths["v20_orchestration"],
            "--output", paths["v20_merged"],
        ),
        "merge_q8": _stage_command(
            config,
            "merge-q8",
            "--shard-root", paths["q8_shards"],
            "--deployment-manifest", config.deployment_manifest,
            "--deployment-lock", config.deployment_lock,
            "--deployment-image-root", config.image_root,
            "--expected-shards", config.q8_shards,
            "--output", paths["q8_merged"],
        ),
        "hybrid": _stage_command(
            config,
            "hybrid",
            "--deployment-manifest", config.deployment_manifest,
            "--deployment-lock", config.deployment_lock,
            "--deployment-image-root", config.image_root,
            "--v20-output", paths["v20_merged"],
            "--q8-output", paths["q8_merged"],
            "--output", paths["hybrid"],
        ),
    }


def build_execution_plan(config: PipelineConfig) -> dict[str, Any]:
    """Return a side-effect-free plan; no path is implicitly discovered."""

    config = config.resolved()
    paths = _fixed_paths(config)
    commands = _common_commands(config)
    legacy = config.bundle / "legacy_project"
    runtime_config = legacy / "configs/rhpheno_dual_mode_runtime_v6_v19_prospective.json"
    v20_config = legacy / "configs/rhaxis_v20_assisted_review_rootcap_recall_dev_v20.json"
    v1_template = _stage_command(
        config,
        "v1",
        "--prepared-manifest", paths["prepare"] / "manifests/sparse_instance.csv",
        "--quality-csv", paths["prepare"] / "acquisition_quality.csv",
        "--runtime-config", runtime_config,
        "--physical-gpu", "<explicit-shard-device>",
        "--task-id", "<partitioned-task-id>...",
        "--output", paths["v1_orchestration"] / "shardNN_attempt01",
    )
    v20_template = _stage_command(
        config,
        "v20",
        "--selection-manifest", paths["v1_materialized"] / "v19_package_selection_manifest.csv",
        "--compatibility-data-root", paths["v1_materialized"] / "compatibility_data_root",
        "--config", v20_config,
        "--task-id", "<partitioned-task-id>...",
        "--output", paths["v20_orchestration"] / "shardNN_attempt01",
    )
    q8_template = _stage_command(
        config,
        "q8",
        "--deployment-manifest", config.deployment_manifest,
        "--deployment-lock", config.deployment_lock,
        "--deployment-image-root", config.image_root,
        "--physical-gpu", "<explicit-shard-device>",
        "--shard-index", "<0..N-1>",
        "--num-shards", config.q8_shards,
        "--field-batch-size", config.field_batch_size,
        "--query-batch-size", config.query_batch_size,
        "--output", paths["q8_shards"] / "shardNN_gpuG",
    )
    return {
        "schema_version": "PHAxis-root-provider-execution-plan-1.0",
        "status": "planned_not_executed",
        "project": str(config.project),
        "bundle": str(config.bundle),
        "python_executable": str(config.python_executable),
        "input_manifest": str(config.input_manifest),
        "output": str(config.output),
        "v1_physical_gpus": list(config.v1_physical_gpus),
        "q8_physical_gpus": list(config.q8_physical_gpus),
        "strict_physical_gpu": config.strict_physical_gpu,
        "stages": [
            {"name": "prepare", "gpu": False, "command": commands["prepare"]},
            {"name": "v1_shards", "gpu": True, "template": v1_template, "shards": config.v1_shards, "concurrency": config.v1_concurrency},
            {"name": "merge_v1", "gpu": False, "command": commands["merge_v1"]},
            {"name": "materialize_v1", "gpu": False, "command": commands["materialize_v1"]},
            {"name": "v20_shards", "gpu": False, "template": v20_template, "shards": config.v20_shards, "concurrency": config.v20_concurrency},
            {"name": "merge_v20", "gpu": False, "command": commands["merge_v20"]},
            {"name": "q8_shards", "gpu": True, "template": q8_template, "shards": config.q8_shards, "concurrency": config.q8_concurrency},
            {"name": "merge_q8", "gpu": False, "command": commands["merge_q8"]},
            {"name": "hybrid", "gpu": False, "command": commands["hybrid"]},
            {"name": "fresh_three_layer_audit", "gpu": False, "reference": str(config.reference_registry) if config.reference_registry else None},
        ],
        "hardcoded_legacy_project_root": False,
        "hardcoded_project_environment": False,
        "hardcoded_physical_gpu0": False,
        "v20_z_drive_execution_dependency": False,
        "training_payload_opened_at_deployment": False,
        "fresh_raw_image_execution_completed": False,
        "blind_images_used": 0,
    }


def _environment(config: PipelineConfig, *, numeric_threads_one: bool = False) -> dict[str, str]:
    environment = os.environ.copy()
    if config.strict_physical_gpu:
        environment[STRICT_PHYSICAL_GPU_ENV] = "1"
    # A source checkout remains useful during development, but an installed
    # wheel must not require a sibling ``scripts`` tree (or even a ``src``
    # directory) merely to launch the package-owned stage entry point.
    project_src_path = (config.project / "src").resolve()
    if (project_src_path / "phaxis").is_dir():
        project_src = str(project_src_path)
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = project_src + (
            os.pathsep + existing if existing else ""
        )
    if numeric_threads_one:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
    return environment


def _run_checked(
    config: PipelineConfig,
    name: str,
    command: Sequence[str],
    *,
    numeric_threads_one: bool = False,
) -> None:
    logs = config.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{name}.stdout.log"
    stderr_path = logs / f"{name}.stderr.log"
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        completed = subprocess.run(
            list(command),
            cwd=config.project,
            env=_environment(config, numeric_threads_one=numeric_threads_one),
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"portable root-provider stage failed: {name}; exit={completed.returncode}; stderr={stderr_path}"
        )


def _run_wave(
    config: PipelineConfig,
    jobs: Sequence[tuple[str, list[str]]],
    concurrency: int,
    *,
    numeric_threads_one: bool = False,
) -> None:
    if not 1 <= concurrency <= len(jobs):
        raise ValueError("stage concurrency must be in [1, job count]")
    logs = config.output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(jobs), concurrency):
        active: list[tuple[str, subprocess.Popen[str], Any, Any, Path]] = []
        for name, command in jobs[start : start + concurrency]:
            stdout_path = logs / f"{name}.stdout.log"
            stderr_path = logs / f"{name}.stderr.log"
            stdout = stdout_path.open("a", encoding="utf-8")
            stderr = stderr_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=config.project,
                env=_environment(config, numeric_threads_one=numeric_threads_one),
                stdout=stdout,
                stderr=stderr,
                text=True,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
            )
            active.append((name, process, stdout, stderr, stderr_path))
        failures: list[str] = []
        for name, process, stdout, stderr, stderr_path in active:
            returncode = process.wait()
            stdout.close()
            stderr.close()
            if returncode != 0:
                failures.append(f"{name}:exit={returncode}:stderr={stderr_path}")
        if failures:
            raise RuntimeError("portable shard wave failed: " + "; ".join(failures))


def _input_identity(config: PipelineConfig, bundle_identity: str) -> dict[str, Any]:
    files = {
        name: sha256_file(path)
        for name, path in {
            "input_manifest": config.input_manifest,
            "acquisition_gate": config.acquisition_gate,
            "deployment_metadata": config.deployment_metadata,
            "canonical_manifest": config.canonical_manifest,
            "deployment_manifest": config.deployment_manifest,
            "deployment_lock": config.deployment_lock,
        }.items()
    }
    return {
        "schema_version": PIPELINE_SCHEMA,
        "bundle_identity_sha256": bundle_identity,
        "files": files,
        "v1_physical_gpus": list(config.v1_physical_gpus),
        "q8_physical_gpus": list(config.q8_physical_gpus),
        "strict_physical_gpu": config.strict_physical_gpu,
        "v1_shards": config.v1_shards,
        "v20_shards": config.v20_shards,
        "q8_shards": config.q8_shards,
        "field_batch_size": config.field_batch_size,
        "query_batch_size": config.query_batch_size,
    }


def _legacy_resume_identity(
    *,
    state: Mapping[str, Any],
    identity_payload: Mapping[str, Any],
    strict_physical_gpu: bool,
) -> str | None:
    """Return the exact pre-strict-GPU identity accepted for a legacy resume."""

    if "strict_physical_gpu" in state or strict_physical_gpu:
        return None
    legacy_payload = dict(identity_payload)
    legacy_payload.pop("strict_physical_gpu", None)
    legacy_identity = sha256_json(legacy_payload)
    if state.get("pipeline_identity_sha256") != legacy_identity:
        return None
    return legacy_identity


def _orchestration_summary(
    *,
    schema: str,
    tasks: Sequence[str],
    attempts: Mapping[int, Path],
    shards: int,
    concurrency: int,
    wall_seconds: float,
    physical_gpus: Sequence[int] | None,
    numeric_threads_one: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": schema,
        "status": "completed",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": len(tasks),
        "shards": shards,
        "concurrency": concurrency,
        "completed_attempts": {str(index): str(path.resolve()) for index, path in sorted(attempts.items())},
        "orchestration_wall_seconds_this_invocation": wall_seconds,
        "preflight_nvidia_smi_before_every_gpu_program": physical_gpus is not None,
        "physical_gpus": list(physical_gpus or ()),
        "per_process_numeric_threads": 1 if numeric_threads_one else None,
        "frozen_implementation_modified": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        "pyRootHair_called_or_copied": False,
    }
    payload["orchestration_identity_sha256"] = sha256_json(
        {key: value for key, value in payload.items() if key not in {"completed_utc", "orchestration_wall_seconds_this_invocation"}}
    )
    return payload


def _write_or_verify_summary(path: Path, payload: Mapping[str, Any]) -> None:
    if path.is_file():
        previous = read_json(path)
        comparable = (
            "schema_version", "tasks", "shards", "concurrency", "completed_attempts",
            "physical_gpus", "per_process_numeric_threads", "blind_images_used",
        )
        if any(previous.get(key) != payload.get(key) for key in comparable):
            raise RuntimeError(f"resume orchestration summary drift: {path}")
        return
    atomic_write_json(path, payload)


def _strict_physical_gpu_required(config: PipelineConfig) -> bool:
    inherited = os.environ.get(STRICT_PHYSICAL_GPU_ENV, "0").strip()
    if inherited not in {"0", "1"}:
        raise RuntimeError(
            f"{STRICT_PHYSICAL_GPU_ENV} must be exactly '0' or '1'"
        )
    return config.strict_physical_gpu or inherited == "1"


def _validate_q8_shard_device_bindings(
    config: PipelineConfig,
    shard_root: Path,
) -> dict[str, Any]:
    """Fail before Q8 merge unless every formal shard stayed on its planned UUID."""

    exact_required = _strict_physical_gpu_required(config)
    if not exact_required:
        return {
            "schema_version": "PHAxis-Q8-shard-device-binding-1.0",
            "status": "dynamic_device_policy_not_a_formal_exact_device_run",
            "exact_physical_gpu_required": False,
            "shards": config.q8_shards,
            "blind_images_used": 0,
        }
    records: list[dict[str, Any]] = []
    for index in range(config.q8_shards):
        planned = _gpu_for(config.q8_physical_gpus, index)
        shard = shard_root / f"shard{index:02d}_gpu{planned}"
        selection_path = shard / "q8_device_selection.json"
        if not selection_path.is_file():
            raise RuntimeError(
                f"Q8 shard exact-device selection receipt is missing: {selection_path}"
            )
        selection = read_json(selection_path)
        requested = selection.get("requested_physical_gpu")
        selected = selection.get("selected_physical_gpu")
        if (
            selection.get("schema_version") != "PHAxis-Q8-device-selection-1.0"
            or selection.get("exact_physical_gpu_required") is not True
            or requested != planned
            or selected != planned
        ):
            raise RuntimeError(
                f"Q8 shard left its strict planned physical GPU: shard={index}, "
                f"planned={planned}, requested={requested}, selected={selected}"
            )
        snapshot = selection.get("gpu_snapshot")
        if not isinstance(snapshot, list):
            raise RuntimeError(f"Q8 shard GPU snapshot is absent: {selection_path}")
        matches = [
            row
            for row in snapshot
            if isinstance(row, Mapping) and row.get("index") == planned
        ]
        if len(matches) != 1 or not str(matches[0].get("uuid", "")):
            raise RuntimeError(
                f"Q8 shard selected physical UUID is absent or ambiguous: {selection_path}"
            )
        records.append(
            {
                "shard_index": index,
                "planned_physical_gpu": planned,
                "requested_physical_gpu": requested,
                "selected_physical_gpu": selected,
                "physical_gpu_uuid": str(matches[0]["uuid"]),
                "selection_receipt": str(selection_path.resolve()),
                "selection_receipt_sha256": sha256_file(selection_path),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "PHAxis-Q8-shard-device-binding-1.0",
        "status": "passed_before_q8_merge",
        "exact_physical_gpu_required": True,
        "planned_physical_gpus": list(config.q8_physical_gpus),
        "shards": config.q8_shards,
        "records": records,
        "requested_equals_selected_equals_planned": True,
        "uuid_bound_to_each_selection_receipt": True,
        "merge_started": False,
        "blind_images_used": 0,
    }
    payload["binding_identity_sha256"] = sha256_json(payload)
    return payload


def _append_task_ids(command: list[str], option: str, task_ids: Iterable[str]) -> list[str]:
    result = list(command)
    for task_id in task_ids:
        result.extend((option, task_id))
    return result


def _validate_config(config: PipelineConfig) -> None:
    if not config.project.is_dir():
        raise FileNotFoundError(f"explicit project/cwd is absent: {config.project}")
    if not config.python_executable.is_file():
        raise FileNotFoundError(config.python_executable)
    for path in (
        config.input_manifest,
        config.acquisition_gate,
        config.deployment_metadata,
        config.canonical_manifest,
        config.deployment_manifest,
        config.deployment_lock,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not config.image_root.is_dir():
        raise FileNotFoundError(config.image_root)
    if not config.v1_physical_gpus or not config.q8_physical_gpus:
        raise ValueError("explicit V1 and Q8 physical GPU selections are required")
    if any(index < 0 for index in (*config.v1_physical_gpus, *config.q8_physical_gpus)):
        raise ValueError("physical GPU indices must be non-negative")
    for shards, concurrency, name in (
        (config.v1_shards, config.v1_concurrency, "V1"),
        (config.v20_shards, config.v20_concurrency, "V20"),
        (config.q8_shards, config.q8_concurrency, "Q8"),
    ):
        if shards <= 0 or not 1 <= concurrency <= shards:
            raise ValueError(f"invalid {name} shard/concurrency setting")


def run_pipeline(config: PipelineConfig, *, resume: bool = False) -> dict[str, Any]:
    config = config.resolved()
    _validate_config(config)
    bundle_verification = verify_bundle(config.bundle)
    identity_payload = _input_identity(
        config, bundle_verification["bundle_identity_sha256"]
    )
    pipeline_identity = sha256_json(identity_payload)
    state_path = config.output / "pipeline_state.json"
    if config.output.exists():
        if not resume or not state_path.is_file():
            raise FileExistsError(f"pipeline output exists; pass --resume with valid state: {config.output}")
        state = read_json(state_path)
        if state.get("pipeline_identity_sha256") != pipeline_identity:
            # ``strict_physical_gpu`` was added to the identity after the first
            # long exact283 run had already published six hash-locked stages.
            # Permit that one schema-compatible legacy state to resume only
            # when the caller keeps the historical non-strict configuration
            # and every earlier identity field still hashes exactly.  A true
            # configuration drift continues to fail closed.
            legacy_pipeline_identity = _legacy_resume_identity(
                state=state,
                identity_payload=identity_payload,
                strict_physical_gpu=config.strict_physical_gpu,
            )
            if legacy_pipeline_identity is None:
                raise RuntimeError("pipeline resume identity mismatch")
            state["resume_identity_compatibility"] = {
                "status": "accepted_legacy_identity_missing_strict_physical_gpu",
                "legacy_pipeline_identity_sha256": legacy_pipeline_identity,
                "current_pipeline_identity_sha256": pipeline_identity,
                "strict_physical_gpu": False,
            }
            atomic_write_json(state_path, state)
    else:
        config.output.mkdir(parents=True)
        state = {
            **identity_payload,
            "pipeline_identity_sha256": pipeline_identity,
            "status": "running",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "completed_stages": [],
            "raw_image_provider": True,
            "canonical_annotations_read": False,
            "blind_images_used": 0,
        }
        atomic_write_json(state_path, state)

    paths = _fixed_paths(config)
    commands = _common_commands(config)

    def complete(name: str, evidence_root: Path | None = None) -> None:
        if name not in state["completed_stages"]:
            state["completed_stages"].append(name)
        if evidence_root is not None:
            evidence_root = evidence_root.resolve()
            evidence_files = {}
            for filename in ("summary.json", "rhpheno_run.json"):
                candidate = evidence_root / filename
                if candidate.is_file():
                    evidence_files[filename] = sha256_file(candidate)
            if not evidence_files:
                raise RuntimeError(f"stage evidence has no completion record: {evidence_root}")
            state.setdefault("stage_evidence", {})[name] = {
                "output": str(evidence_root),
                "files": evidence_files,
            }
        state["last_completed_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(state_path, state)

    resume_flag = ("--resume",) if resume else ()
    _run_checked(config, "prepare", [*commands["prepare"], *resume_flag])
    complete("prepare", paths["prepare"])

    prepared_manifest = paths["prepare"] / "manifests/sparse_instance.csv"
    task_ids = [row["image_id"] for row in _read_csv(prepared_manifest)]
    if len(task_ids) != len(set(task_ids)) or not task_ids:
        raise RuntimeError("prepared V1 task IDs are empty or duplicated")
    v1_parts = _partition(task_ids, config.v1_shards)
    v1_jobs: list[tuple[str, list[str]]] = []
    v1_attempts: dict[int, Path] = {}
    runtime_config = config.bundle / "legacy_project/configs/rhpheno_dual_mode_runtime_v6_v19_prospective.json"
    for index, selected in enumerate(v1_parts):
        output = paths["v1_orchestration"] / f"shard{index:02d}_attempt01"
        v1_attempts[index] = output
        command = _stage_command(
            config, "v1",
            "--prepared-manifest", prepared_manifest,
            "--quality-csv", paths["prepare"] / "acquisition_quality.csv",
            "--runtime-config", runtime_config,
            "--physical-gpu", _gpu_for(config.v1_physical_gpus, index),
            "--output", output,
            *resume_flag,
        )
        v1_jobs.append((f"v1_shard{index:02d}", _append_task_ids(command, "--task-id", selected)))
    started = time.perf_counter()
    _run_wave(config, v1_jobs, config.v1_concurrency)
    v1_wall = time.perf_counter() - started
    _write_or_verify_summary(
        paths["v1_orchestration"] / "summary.json",
        _orchestration_summary(
            schema="PHAxis-portable-frozen-V1-sharded-orchestration-1.0",
            tasks=task_ids, attempts=v1_attempts, shards=len(v1_parts),
            concurrency=config.v1_concurrency, wall_seconds=v1_wall,
            physical_gpus=config.v1_physical_gpus, numeric_threads_one=False,
        ),
    )
    complete("v1_shards", paths["v1_orchestration"])
    _run_checked(config, "merge_v1", [*commands["merge_v1"], *resume_flag])
    complete("merge_v1", paths["v1_merged"])
    _run_checked(config, "materialize_v1", [*commands["materialize_v1"], *resume_flag])
    complete("materialize_v1", paths["v1_materialized"])

    selection = paths["v1_materialized"] / "v19_package_selection_manifest.csv"
    v20_task_ids = [row["task_id"] for row in _read_csv(selection)]
    if set(v20_task_ids) != set(task_ids):
        raise RuntimeError("V20 selection does not preserve V1 task set")
    v20_parts = _partition(v20_task_ids, config.v20_shards)
    v20_jobs: list[tuple[str, list[str]]] = []
    v20_attempts: dict[int, Path] = {}
    v20_config = config.bundle / "legacy_project/configs/rhaxis_v20_assisted_review_rootcap_recall_dev_v20.json"
    for index, selected in enumerate(v20_parts):
        output = paths["v20_orchestration"] / f"shard{index:02d}_attempt01"
        v20_attempts[index] = output
        command = _stage_command(
            config, "v20",
            "--selection-manifest", selection,
            "--compatibility-data-root", paths["v1_materialized"] / "compatibility_data_root",
            "--config", v20_config,
            "--output", output,
            *resume_flag,
        )
        v20_jobs.append((f"v20_shard{index:02d}", _append_task_ids(command, "--task-id", selected)))
    started = time.perf_counter()
    _run_wave(config, v20_jobs, config.v20_concurrency, numeric_threads_one=True)
    v20_wall = time.perf_counter() - started
    _write_or_verify_summary(
        paths["v20_orchestration"] / "summary.json",
        _orchestration_summary(
            schema="PHAxis-portable-frozen-V20.12-sharded-orchestration-1.0",
            tasks=v20_task_ids, attempts=v20_attempts, shards=len(v20_parts),
            concurrency=config.v20_concurrency, wall_seconds=v20_wall,
            physical_gpus=None, numeric_threads_one=True,
        ),
    )
    complete("v20_shards", paths["v20_orchestration"])
    _run_checked(config, "merge_v20", [*commands["merge_v20"], *resume_flag])
    complete("merge_v20", paths["v20_merged"])

    q8_jobs: list[tuple[str, list[str]]] = []
    for index in range(config.q8_shards):
        physical_gpu = _gpu_for(config.q8_physical_gpus, index)
        output = paths["q8_shards"] / f"shard{index:02d}_gpu{physical_gpu}"
        command = _stage_command(
            config, "q8",
            "--deployment-manifest", config.deployment_manifest,
            "--deployment-lock", config.deployment_lock,
            "--deployment-image-root", config.image_root,
            "--physical-gpu", physical_gpu,
            "--shard-index", index,
            "--num-shards", config.q8_shards,
            "--field-batch-size", config.field_batch_size,
            "--query-batch-size", config.query_batch_size,
            "--output", output,
            *resume_flag,
        )
        q8_jobs.append((f"q8_shard{index:02d}_gpu{physical_gpu}", command))
    _run_wave(config, q8_jobs, config.q8_concurrency)
    # The exact-device authority is materialized and sealed before merge.  A
    # formal run cannot hide a safe-but-different fallback behind a merged Q8
    # directory or an outer stage receipt that still names the planned card.
    q8_device_binding = _validate_q8_shard_device_bindings(
        config, paths["q8_shards"]
    )
    q8_device_binding_path = paths["q8_shards"] / "exact_device_binding.json"
    if q8_device_binding_path.is_file():
        if read_json(q8_device_binding_path) != q8_device_binding:
            raise RuntimeError("Q8 exact-device binding drifted on resume")
    else:
        atomic_write_json(q8_device_binding_path, q8_device_binding)
    state["q8_exact_device_binding"] = {
        "path": str(q8_device_binding_path.resolve()),
        "sha256": sha256_file(q8_device_binding_path),
        "identity_sha256": q8_device_binding.get("binding_identity_sha256"),
        "status": q8_device_binding["status"],
    }
    atomic_write_json(state_path, state)
    complete("q8_shards")
    _run_checked(config, "merge_q8", [*commands["merge_q8"], *resume_flag])
    complete("merge_q8", paths["q8_merged"])
    _run_checked(config, "hybrid", [*commands["hybrid"], *resume_flag])
    complete("hybrid", paths["hybrid"])

    if config.reference_registry is not None:
        audit = audit_fresh_reference(
            reference_registry=config.reference_registry,
            fresh_v1_root=paths["v1_merged"],
            fresh_v20_root=paths["v20_merged"],
            fresh_final_root=paths["hybrid"],
            pipeline_state=state_path,
            output=paths["audit"],
        )
        if audit["status"] != "pass_exact_283":
            raise RuntimeError("fresh raw-image three-layer equivalence gate failed")
        complete("fresh_three_layer_audit")
    state.update(
        {
            "status": (
                "completed_fresh_283_exact"
                if config.reference_registry is not None
                else "completed_uncompared"
            ),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "official_hybrid_output": str(paths["hybrid"].resolve()),
            "fresh_reference_audit": str(paths["audit"].resolve()) if config.reference_registry else None,
            "canonical_annotations_read": False,
            "blind_images_used": 0,
        }
    )
    atomic_write_json(state_path, state)
    return state


__all__ = ["PipelineConfig", "build_execution_plan", "run_pipeline"]
