from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import phaxis.benchmark as benchmark
from phaxis.contracts import ContractError
from phaxis.io import sha256_json


def _write_sealed(path: Path, payload: dict, identity_field: str) -> Path:
    data = dict(payload)
    data[identity_field] = sha256_json(data)
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    return path


def _summary(
    *, system: str, kind: str, value: float, latency_mode: str
) -> dict:
    source_units = [f"image-{index:03d}.png" for index in range(283)]
    hardware = {
        "host": "synthetic-same-host",
        "platform": "windows-test",
        "gpus": [
            {
                "physical_index": 0,
                "uuid": "GPU-SYNTHETIC-3090-0",
                "driver_version": "555.42",
                "name": "RTX 3090",
                "memory_total_mib": 24576,
            }
        ],
    }
    payload = {
        "benchmark_system": system,
        "status": "completed_direct_full283",
        "measurement_scope": benchmark.MEASUREMENT_SCOPE,
        "benchmark_scope_class": "full_workflow",
        "images": 283,
        "n_images": 283,
        "includes_io": True,
        "includes_preprocess": True,
        "includes_stitching_fusion_traits_profiles": True,
        "fresh_direct_run": True,
        "resume_or_cache_used": False,
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read": False,
        "source_manifest_sha256": "a" * 64,
        "source_image_lock_identity_sha256": "b" * 64,
        "source_units_in_order": source_units,
        "source_unit_ordered_set_identity_sha256": sha256_json(source_units),
        "hardware": hardware,
        "hardware_identity_sha256": sha256_json(hardware),
        "blind_images_used": 0,
        "rootcap_region_metric": False,
    }
    if kind == "production":
        payload.update(
            {
                "schema_version": benchmark.PRODUCTION_SCHEMA,
                "benchmark_mode": benchmark.PRODUCTION_MODE,
                "timing_granularity": "direct_batch_stage_wall",
                "per_image_latency_reported": False,
                "batch_wall_seconds": value,
                "stage_timings": [
                    {
                        "stage": "complete_workflow",
                        "wall_seconds": value,
                        "cache_status": "executed_fresh",
                    }
                ],
                "official_output_hashes": {"final_outputs": "c" * 64},
            }
        )
    else:
        persistent = latency_mode == benchmark.PERSISTENT_LATENCY_MODE
        payload.update(
            {
                "schema_version": benchmark.LATENCY_SCHEMA,
                "benchmark_mode": latency_mode,
                "latency_mode": latency_mode,
                "timing_granularity": "direct_per_source_raw_to_final_profile",
                "stage_timing_semantics": "nonoverlapping_wall_components",
                "measurement_method": "direct_per_source_perf_counter_start_to_final_profile_visibility",
                "persistent_worker_and_models": persistent,
                "model_or_process_startup_per_source": not persistent,
                "startup_included_in_per_image_wall": not persistent,
                "median_seconds_per_image": value,
                "p95_seconds_per_image": value * 1.2,
                "per_image_csv_sha256": "d" * 64,
                "output_hashes": {"final_outputs": "e" * 64},
            }
        )
    return payload


def _formal_inputs(tmp_path: Path) -> dict[str, Path]:
    latency_mode = benchmark.PERSISTENT_LATENCY_MODE
    summary_paths = {
        "phaxis_production_summary": _write_sealed(
            tmp_path / "phaxis-production.json",
            _summary(
                system=benchmark.PHAXIS_BENCHMARK_SYSTEM,
                kind="production",
                value=60.0,
                latency_mode=latency_mode,
            ),
            "summary_identity_sha256",
        ),
        "phaxis_sequential_summary": _write_sealed(
            tmp_path / "phaxis-sequential.json",
            _summary(
                system=benchmark.PHAXIS_BENCHMARK_SYSTEM,
                kind="latency",
                value=2.0,
                latency_mode=latency_mode,
            ),
            "summary_identity_sha256",
        ),
        "frozen_v1_production_summary": _write_sealed(
            tmp_path / "frozen-production.json",
            _summary(
                system=benchmark.FROZEN_V1_BENCHMARK_SYSTEM,
                kind="production",
                value=120.0,
                latency_mode=latency_mode,
            ),
            "summary_identity_sha256",
        ),
        "frozen_v1_sequential_summary": _write_sealed(
            tmp_path / "frozen-sequential.json",
            _summary(
                system=benchmark.FROZEN_V1_BENCHMARK_SYSTEM,
                kind="latency",
                value=4.0,
                latency_mode=latency_mode,
            ),
            "summary_identity_sha256",
        ),
    }
    summary_paths["production_comparison"] = tmp_path / "production-comparison.json"
    benchmark.compare_benchmarks(
        phaxis_summary=summary_paths["phaxis_production_summary"],
        baseline_summary=summary_paths["frozen_v1_production_summary"],
        output=summary_paths["production_comparison"],
    )
    summary_paths["sequential_comparison"] = tmp_path / "sequential-comparison.json"
    benchmark.compare_benchmarks(
        phaxis_summary=summary_paths["phaxis_sequential_summary"],
        baseline_summary=summary_paths["frozen_v1_sequential_summary"],
        output=summary_paths["sequential_comparison"],
    )
    return summary_paths


def _reseal(path: Path, field: str, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop(field)
    mutate(payload)
    payload[field] = sha256_json(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_exact283_aggregate_is_sealed_deterministic_and_cli_check_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs = _formal_inputs(tmp_path)
    first = benchmark.aggregate_same_hardware_benchmark_receipt(**inputs)
    second = benchmark.aggregate_same_hardware_benchmark_receipt(**inputs)
    assert first == second
    assert first["status"] == "passed"
    assert first["images"] == 283
    assert first["same_ordered_exact283_sources"] is True
    assert first["same_hardware_uuid_and_driver"] is True
    assert first["fresh_no_cache"] is True
    unsigned = dict(first)
    assert unsigned.pop("receipt_identity_sha256") == sha256_json(unsigned)

    cli_path = Path(__file__).parents[2] / "scripts/phaxis/benchmark_full_workflow.py"
    spec = importlib.util.spec_from_file_location("benchmark_full_workflow_cli", cli_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / "formal-receipt.json"
    argv = ["--aggregate-same-hardware", "--output", str(output)]
    for key, value in inputs.items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    assert cli.main(argv) == 0
    assert not output.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "validated_not_published"

    published = benchmark.publish_same_hardware_benchmark_receipt(
        output=output, **inputs
    )
    assert json.loads(output.read_text(encoding="utf-8")) == published


@pytest.mark.parametrize(
    ("target", "mutation", "match"),
    [
        (
            "frozen_v1_production_summary",
            lambda value: value.update(
                {
                    "benchmark_scope_class": "component_only",
                    "batch_wall_seconds": 98.47 * 60,
                }
            ),
            "full workflow",
        ),
        (
            "frozen_v1_sequential_summary",
            lambda value: value["source_units_in_order"].reverse(),
            "ordered source-unit identity mismatch",
        ),
        (
            "frozen_v1_sequential_summary",
            lambda value: value.update({"resume_or_cache_used": True}),
            "fresh/no-cache",
        ),
        (
            "frozen_v1_sequential_summary",
            lambda value: value["hardware"]["gpus"][0].update(
                {"driver_version": "tampered-driver"}
            ),
            "hardware identity mismatch",
        ),
        (
            "phaxis_production_summary",
            lambda value: value.update({"status": "forward_only"}),
            "not completed direct exact283",
        ),
    ],
)
def test_aggregate_rejects_component_forward_cache_order_and_hardware_tamper(
    tmp_path: Path, target: str, mutation, match: str
) -> None:
    inputs = _formal_inputs(tmp_path)
    _reseal(inputs[target], "summary_identity_sha256", mutation)
    with pytest.raises(ContractError, match=match):
        benchmark.aggregate_same_hardware_benchmark_receipt(**inputs)


def test_aggregate_recomputes_comparison_speedup(tmp_path: Path) -> None:
    inputs = _formal_inputs(tmp_path)

    def mutate(payload: dict) -> None:
        payload["batch_wall_speedup_frozen_v1_over_phaxis"] = 98.47

    _reseal(inputs["production_comparison"], "comparison_identity_sha256", mutate)
    with pytest.raises(ContractError, match="speedup was not recomputed"):
        benchmark.aggregate_same_hardware_benchmark_receipt(**inputs)
