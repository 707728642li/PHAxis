from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import pytest

import phaxis.benchmark as benchmark
from phaxis.contracts import ContractError
from phaxis.io import sha256_file, sha256_json


def _plan(tasks: int = 2) -> dict:
    binding = {
        "model_contract_proposal_sha256": "a" * 64,
        "model_contract_proposal_identity_sha256": "b" * 64,
    }
    stages = [
        {
            "name": "root_provider",
            "input_hashes": {
                "root_provider.input_manifest": "1" * 64,
                "workflow_module": "2" * 64,
            },
            "estimated_gpu": {
                "v1_physical_gpus": [0],
                "q8_physical_gpus": [1],
            },
        },
        {
            "name": "stageb_train399",
            "estimated_gpu": {"physical_gpu": 1},
        },
        {"name": "fusion", "estimated_gpu": {}},
        {"name": "traits", "estimated_gpu": {}},
        {"name": "distal_axis_profiles", "estimated_gpu": {}},
    ]
    return {
        "tasks": tasks,
        "task_ids": [f"T{index}" for index in range(1, tasks + 1)],
        "manifest_sha256": "c" * 64,
        "manifest_identity_sha256": "d" * 64,
        "task_identity_sha256": "e" * 64,
        "plan_identity_sha256": "f" * 64,
        "stages": stages,
        **binding,
    }


def _write_json(path: Path, payload: dict, identity: str | None = None) -> Path:
    data = dict(payload)
    if identity is not None:
        data[identity] = sha256_json(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_benchmark_default_plan_never_queries_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(benchmark, "build_analysis_plan", lambda *_a, **_k: _plan())
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("dry-run queried GPU")),
    )
    result = benchmark.benchmark_plan(
        manifest=tmp_path / "manifest.json",
        workflow_output=tmp_path / "workflow",
        benchmark_output=tmp_path / "benchmark",
    )
    assert result["status"] == "planned_not_executed"
    assert result["batch_latency_is_never_derived_per_image"] is True
    assert not (tmp_path / "benchmark").exists()


def _latency_fixture(tmp_path: Path, *, mode: str) -> tuple[Path, Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    timing = tmp_path / "trace.csv"
    with timing.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=benchmark.PER_IMAGE_FIELDS)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_unit": "T1",
                    "wall_seconds": 10.0,
                    "megapixels": 1.0,
                    "io_seconds": 1.0,
                    "preprocess_seconds": 2.0,
                    "inference_seconds": 5.0,
                    "postprocess_seconds": 2.0,
                },
                {
                    "source_unit": "T2",
                    "wall_seconds": 20.0,
                    "megapixels": 2.0,
                    "io_seconds": 2.0,
                    "preprocess_seconds": 4.0,
                    "inference_seconds": 10.0,
                    "postprocess_seconds": 4.0,
                },
            ]
        )
    plan = _plan()
    hardware = {"host": "synthetic", "gpu": "CPU oracle"}
    persistent = mode == benchmark.PERSISTENT_LATENCY_MODE
    nvidia_preflight = {"synthetic_cpu_oracle": True}
    receipt = {
        "schema_version": benchmark.LATENCY_TRACE_SCHEMA,
        "latency_mode": mode,
        "measurement_scope": benchmark.MEASUREMENT_SCOPE,
        "per_image_csv_sha256": sha256_file(timing),
        "stage_timing_semantics": "nonoverlapping_wall_components",
        "measurement_method": "direct_per_source_perf_counter_start_to_final_profile_visibility",
        "includes_io": True,
        "includes_preprocess": True,
        "includes_stitching_fusion_traits_profiles": True,
        "resume_or_cache_used": False,
        "persistent_worker_and_models": persistent,
        "model_or_process_startup_per_source": not persistent,
        "startup_included_in_per_image_wall": not persistent,
        "one_time_startup_seconds": 3.0 if persistent else 0.0,
        "peak_vram_mib": 0.0,
        "mean_gpu_utilization_pct": 0.0,
        "hardware": hardware,
        "hardware_identity_sha256": sha256_json(hardware),
        "nvidia_smi_preflight": nvidia_preflight,
        "nvidia_smi_preflight_identity_sha256": sha256_json(nvidia_preflight),
        "physical_gpu_mapping": [],
        "cuda_visible_devices_by_stage": {},
        "checkpoint_sha256": [str(index) * 64 for index in range(1, 6)],
        "input_hashes": {"raw_image_manifest": "1" * 64},
        "config_hashes": {"workflow": "2" * 64},
        "output_hashes": {"final_profiles": "3" * 64},
        "source_manifest_sha256": "1" * 64,
        "workflow_manifest_sha256": plan["manifest_sha256"],
        "workflow_manifest_identity_sha256": plan["manifest_identity_sha256"],
        "source_image_lock_identity_sha256": plan["task_identity_sha256"],
        "model_contract_proposal_sha256": plan[
            "model_contract_proposal_sha256"
        ],
        "model_contract_proposal_identity_sha256": plan[
            "model_contract_proposal_identity_sha256"
        ],
        "blind_images_used": 0,
        "rootcap_region_metric": False,
    }
    receipt["trace_identity_sha256"] = sha256_json(receipt)
    receipt_path = _write_json(tmp_path / "trace.json", receipt)
    return timing, receipt_path, plan


@pytest.mark.parametrize(
    "mode",
    [benchmark.PERSISTENT_LATENCY_MODE, benchmark.COLD_LATENCY_MODE],
)
def test_sequential_trace_is_direct_atomic_and_mode_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    timing, receipt, plan = _latency_fixture(tmp_path, mode=mode)
    monkeypatch.setattr(benchmark, "build_analysis_plan", lambda *_a, **_k: plan)
    output = tmp_path / "published"
    summary = benchmark.compile_sequential_latency_benchmark(
        manifest=tmp_path / "manifest.json",
        workflow_output=tmp_path / "workflow",
        trace_csv=timing,
        trace_receipt=receipt,
        benchmark_output=output,
    )
    assert summary["latency_mode"] == mode
    assert summary["median_seconds_per_image"] == 15.0
    assert summary["p95_seconds_per_image"] == pytest.approx(19.5)
    assert summary["status"] == "completed_direct_synthetic_cpu"
    assert sha256_file(output / "runtime_per_image.csv") == summary[
        "per_image_csv_sha256"
    ]
    assert not any(output.parent.glob(".*benchmark-attempt-*"))


def test_sequential_trace_rejects_tamper_and_fake_component_decomposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    timing, receipt, plan = _latency_fixture(
        tmp_path, mode=benchmark.PERSISTENT_LATENCY_MODE
    )
    monkeypatch.setattr(benchmark, "build_analysis_plan", lambda *_a, **_k: plan)
    timing.write_text(timing.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="CSV hash mismatch"):
        benchmark.compile_sequential_latency_benchmark(
            manifest="unused",
            workflow_output=tmp_path / "workflow",
            trace_csv=timing,
            trace_receipt=receipt,
            benchmark_output=tmp_path / "tampered",
        )

    timing, receipt, plan = _latency_fixture(
        tmp_path / "components", mode=benchmark.PERSISTENT_LATENCY_MODE
    )
    rows = timing.read_text(encoding="utf-8").replace(",5.0,2.0", ",1.0,2.0", 1)
    timing.write_text(rows, encoding="utf-8")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["per_image_csv_sha256"] = sha256_file(timing)
    payload.pop("trace_identity_sha256")
    payload["trace_identity_sha256"] = sha256_json(payload)
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="do not reconstruct"):
        benchmark.compile_sequential_latency_benchmark(
            manifest="unused",
            workflow_output=tmp_path / "workflow2",
            trace_csv=timing,
            trace_receipt=receipt,
            benchmark_output=tmp_path / "fake-components",
        )


def _production_fixture(tmp_path: Path, *, cached: bool = False) -> tuple[dict, Path]:
    plan = _plan()
    output = tmp_path / "workflow"
    stage_records = []
    for name in benchmark._CORE_STAGES:
        stage_records.append(
            {
                "stage": name,
                "execution_status": (
                    "cached_completed_evidence_validated"
                    if cached and name == "fusion"
                    else "executed_fresh"
                ),
                "wall_seconds_including_stage_io_and_evidence_hashing": 1.0,
                "evidence_tree_identity_sha256": sha256_json(name),
            }
        )
    state = {
        "status": "completed",
        "latest_execution_attempt_id": 1,
        "execution_attempts": [
            {
                "fresh_direct_benchmark_eligible": not cached,
                "resume_or_cache_used": cached,
                "resume_requested": False,
                "stages": stage_records,
            }
        ],
        **{field: plan[field] for field in benchmark._BINDING_FIELDS},
    }
    _write_json(output / "workflow_state.json", state, "state_identity_sha256")
    analysis_plan_path = _write_json(output / "analysis_plan.json", plan)
    state_path = output / "workflow_state.json"
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    state_payload.pop("state_identity_sha256")
    state_payload["analysis_plan_sha256"] = sha256_file(analysis_plan_path)
    state_payload["state_identity_sha256"] = sha256_json(state_payload)
    state_path.write_text(json.dumps(state_payload), encoding="utf-8")
    stageb = {
        "images": 2,
        "records": [
            {"task_id": "T1", "source_megapixels": 1.0, "resumed": False},
            {"task_id": "T2", "source_megapixels": 2.0, "resumed": False},
        ],
        "checkpoint_sha256": [str(index) * 64 for index in range(1, 6)],
        "nvidia_smi_preflight_sha256": "9" * 64,
        "resumed_images": 0,
        **{field: plan[field] for field in benchmark._BINDING_FIELDS},
    }
    _write_json(output / "stageb/summary.json", stageb, "summary_identity_sha256")
    _write_json(
        output / "fusion/fusion_summary.json",
        {
            "fixture": "fusion",
            "blind_images_used": 0,
            **{field: plan[field] for field in benchmark._BINDING_FIELDS},
        },
        "summary_identity_sha256",
    )
    for path in (
        output / "traits/summary.json",
        output / "distal_axis_profiles/summary.json",
    ):
        _write_json(
            path,
            {
                "fixture": path.parent.name,
                "blind_images_used": 0,
                **{field: plan[field] for field in benchmark._BINDING_FIELDS},
            },
            "export_identity_sha256",
        )
    return plan, output


def test_production_summary_reports_batch_only_and_rejects_cached_workflow(
    tmp_path: Path,
) -> None:
    plan, output = _production_fixture(tmp_path)
    hardware = {"host": "synthetic"}
    nvidia_preflight = {"synthetic": True}
    summary = benchmark._production_summary(
        plan=plan,
        workflow_output=output,
        outer_wall_seconds=10.0,
        hardware_preflight={
            "hardware": hardware,
            "hardware_identity_sha256": sha256_json(hardware),
            "nvidia_smi_preflight": nvidia_preflight,
            "nvidia_smi_preflight_identity_sha256": sha256_json(
                nvidia_preflight
            ),
        },
        telemetry={
            "peak_vram_mib": 0.0,
            "mean_gpu_utilization_pct": 0.0,
            "samples": 1,
        },
    )
    assert summary["benchmark_mode"] == benchmark.PRODUCTION_MODE
    assert summary["per_image_latency_reported"] is False
    assert "median_seconds_per_image" not in summary
    assert summary["images_per_min"] == 12.0

    cached_plan, cached_output = _production_fixture(tmp_path / "cached", cached=True)
    with pytest.raises(ContractError, match="resume/cached"):
        cached_preflight = {}
        benchmark._production_summary(
            plan=cached_plan,
            workflow_output=cached_output,
            outer_wall_seconds=10.0,
            hardware_preflight={
                "hardware": hardware,
                "hardware_identity_sha256": sha256_json(hardware),
                "nvidia_smi_preflight": cached_preflight,
                "nvidia_smi_preflight_identity_sha256": sha256_json(
                    cached_preflight
                ),
            },
            telemetry={"peak_vram_mib": 0, "mean_gpu_utilization_pct": 0},
        )


def _benchmark_summary(
    *, wall: float, system: str, component_only: bool = False
) -> dict:
    source_units = [f"image-{index:03d}" for index in range(283)]
    payload = {
        "benchmark_system": system,
        "status": "completed_direct_full283",
        "benchmark_mode": benchmark.PRODUCTION_MODE,
        "measurement_scope": benchmark.MEASUREMENT_SCOPE,
        "images": 283,
        "source_manifest_sha256": "a" * 64,
        "source_image_lock_identity_sha256": "b" * 64,
        "source_units_in_order": source_units,
        "source_unit_ordered_set_identity_sha256": sha256_json(source_units),
        "hardware_identity_sha256": "c" * 64,
        "includes_io": True,
        "includes_preprocess": True,
        "includes_stitching_fusion_traits_profiles": True,
        "resume_or_cache_used": False,
        "batch_wall_seconds": wall,
        "benchmark_scope_class": "component_only" if component_only else "full_workflow",
    }
    payload["summary_identity_sha256"] = sha256_json(payload)
    return payload


def test_baseline_speedup_requires_same_full_direct_scope(tmp_path: Path) -> None:
    candidate = _write_json(
        tmp_path / "candidate.json",
        _benchmark_summary(wall=10.0, system=benchmark.PHAXIS_BENCHMARK_SYSTEM),
    )
    baseline = _write_json(
        tmp_path / "baseline.json",
        _benchmark_summary(
            wall=20.0, system=benchmark.FROZEN_V1_BENCHMARK_SYSTEM
        ),
    )
    comparable = benchmark.compare_benchmarks(
        phaxis_summary=candidate,
        baseline_summary=baseline,
        output=tmp_path / "comparison.json",
    )
    assert comparable["comparable"] is True
    assert comparable["batch_wall_speedup_frozen_v1_over_phaxis"] == 2.0

    component = _write_json(
        tmp_path / "historical-98min.json",
        _benchmark_summary(
            wall=98.47 * 60.0,
            system=benchmark.FROZEN_V1_BENCHMARK_SYSTEM,
            component_only=True,
        ),
    )
    rejected = benchmark.compare_benchmarks(
        phaxis_summary=candidate,
        baseline_summary=component,
        output=tmp_path / "not-comparable.json",
    )
    assert rejected["comparable"] is False
    assert "baseline.component_only_noncomparable" in rejected[
        "noncomparability_reasons"
    ]
    assert "batch_wall_speedup_frozen_v1_over_phaxis" not in rejected
