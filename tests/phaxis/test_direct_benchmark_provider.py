from __future__ import annotations

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from phaxis import benchmark
from phaxis.io import atomic_write_json, sha256_file, sha256_json
from scripts.phaxis import build_direct_benchmark_provider_descriptor as builder
from scripts.phaxis import run_external_direct_benchmark as provider


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_descriptor_builder_seals_real_four_mode_closures_and_is_create_only(
    tmp_path: Path,
) -> None:
    payload = builder.build_descriptor(PROJECT_ROOT, physical_gpu=0)
    assert payload["schema_version"] == provider.INTERFACE_SCHEMA
    assert payload["status"] == "ready_hash_locked_direct_execution"
    assert set(payload["entrypoints"]) == provider.MODES
    assert payload["blind_images_used"] == 0
    assert payload["descriptor_identity_sha256"] == sha256_json(
        {
            key: value
            for key, value in payload.items()
            if key != "descriptor_identity_sha256"
        }
    )
    for mode, binding in payload["entrypoints"].items():
        assert binding["path"] == builder.ADAPTER
        assert binding["sha256"] == sha256_file(PROJECT_ROOT / builder.ADAPTER)
        assert binding["physical_gpus"] == [0]
        assert binding["warmup_runs"] == 0
        assert binding["measured_repeats"] == 1
        assert binding["implementation_closure"]
        assert all(
            sha256_file(PROJECT_ROOT / record["path"]) == record["sha256"]
            for record in binding["implementation_closure"]
        ), mode

    destination = tmp_path / "provider.json"
    assert builder.write_descriptor_create_only(destination, payload) == destination
    before = destination.read_bytes()
    with pytest.raises(builder.DescriptorBuildError, match="already exists"):
        builder.write_descriptor_create_only(destination, payload)
    assert destination.read_bytes() == before


def test_descriptor_builder_check_subprocess_is_explicitly_cpu_only() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "scripts/phaxis/build_direct_benchmark_provider_descriptor.py"
            ),
            "--project-root",
            str(PROJECT_ROOT),
            "--check",
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ready_cpu_preflight_only"
    assert payload["gpu_program_started"] is False
    assert payload["nvidia_smi_called"] is False
    assert payload["blind_images_used"] == 0


def _exact283_source(tmp_path: Path) -> tuple[Path, Path]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    manifest = tmp_path / "source283.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task_id",
                "image_path",
                "image_sha256",
                "um_per_px",
                "source_megapixels",
            ),
        )
        writer.writeheader()
        for index in range(283):
            source = image_root / f"source-{index:03d}.bin"
            source.write_bytes(f"synthetic-source-{index:03d}".encode("ascii"))
            writer.writerow(
                {
                    "task_id": f"SOURCE-{index:03d}",
                    "image_path": source,
                    "image_sha256": sha256_file(source),
                    "um_per_px": "0.5",
                    "source_megapixels": "1.25",
                }
            )
    return manifest, image_root


def test_source_lock_rehashes_exact283_bytes_and_fails_closed_on_drift(
    tmp_path: Path,
) -> None:
    manifest, image_root = _exact283_source(tmp_path)
    lock = provider._source_manifest_lock(manifest, image_root)
    assert len(lock["rows"]) == 283
    assert lock["source_units_in_order"][0] == "SOURCE-000"
    assert lock["source_units_in_order"][-1] == "SOURCE-282"
    assert lock["megapixels"] == pytest.approx(283 * 1.25)

    (image_root / "source-117.bin").write_bytes(b"tampered")
    with pytest.raises(provider.ProviderError, match="byte hash differs"):
        provider._source_manifest_lock(manifest, image_root)


def test_source_lock_preserves_the_sealed_nonlexicographic_manifest_order(
    tmp_path: Path,
) -> None:
    manifest, image_root = _exact283_source(tmp_path)
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    rows[0], rows[1] = rows[1], rows[0]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lock = provider._source_manifest_lock(manifest, image_root)
    assert lock["source_units_in_order"][:2] == ["SOURCE-001", "SOURCE-000"]
    assert lock["source_unit_ordered_set_identity_sha256"] == sha256_json(
        lock["source_units_in_order"]
    )


def test_provider_subprocess_adapter_propagates_cpu_environment_and_logs(
    tmp_path: Path,
) -> None:
    stub = tmp_path / "stub.py"
    stub.write_text(
        "import os\n"
        "assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'\n"
        "print('synthetic-subprocess-ok')\n",
        encoding="utf-8",
    )
    log_root = tmp_path / "logs"
    elapsed = provider._run(
        [sys.executable, str(stub)],
        cwd=tmp_path,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        log_root=log_root,
        name="cpu-contract",
    )
    assert elapsed >= 0
    assert (log_root / "cpu-contract.stdout.log").read_text(
        encoding="utf-8"
    ).strip() == "synthetic-subprocess-ok"
    assert (log_root / "cpu-contract.stderr.log").read_text(encoding="utf-8") == ""


def test_phaxis_sequential_times_the_same_outer_raw_to_final_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter((100.0, 102.0, 108.0))
    monkeypatch.setattr(provider.time, "perf_counter", lambda: next(clock))

    def derive(_manifest, _task, destination, _physical_gpu):
        destination.mkdir(parents=True)
        derived = destination / "analysis.json"
        derived.write_text("{}\n", encoding="utf-8")
        return derived

    def run(command, *, cwd, env, log_root, name):
        del cwd, env, log_root, name
        final = Path(command[command.index("--output") + 1])
        (final / "distal_axis_profiles").mkdir(parents=True)
        provider.atomic_write_json(
            final / "workflow_state.json",
            {
                "status": "completed",
                "execution_attempts": [
                    {
                        "fresh_direct_benchmark_eligible": True,
                        "resume_or_cache_used": False,
                    }
                ],
            },
        )
        provider.atomic_write_json(
            final / "distal_axis_profiles" / "summary.json", {"status": "completed"}
        )
        return 3.0

    monkeypatch.setattr(provider, "_single_source_analysis_manifest", derive)
    monkeypatch.setattr(provider, "_run", run)
    trace, hashes = provider._run_phaxis_sequential(
        analysis_manifest=tmp_path / "unused.json",
        rows=[
            {
                "task_id": "SOURCE-000",
                "source_megapixels": 1.25,
            }
        ],
        root=tmp_path / "workflow",
        python=Path(sys.executable),
        env={"CUDA_VISIBLE_DEVICES": "-1"},
        physical_gpu=0,
    )
    assert len(hashes) == 1
    assert trace == [
        {
            "source_unit": "SOURCE-000",
            "wall_seconds": 8.0,
            "megapixels": 1.25,
            "io_seconds": 0.0,
            "preprocess_seconds": 2.0,
            "inference_seconds": 3.0,
            "postprocess_seconds": 3.0,
        }
    ]


def test_q8_binding_requires_exact_index_and_outer_preflight_uuid(
    tmp_path: Path,
) -> None:
    workflow = tmp_path / "workflow"
    selection = workflow / "q8" / "q8_device_selection.json"
    provider.atomic_write_json(
        selection,
        {
            "schema_version": "PHAxis-Q8-device-selection-1.0",
            "requested_physical_gpu": 0,
            "selected_physical_gpu": 0,
            "exact_physical_gpu_required": True,
            "gpu_snapshot": [{"index": 0, "uuid": "GPU-SYNTHETIC-0"}],
        },
    )
    hardware = {
        "gpus": [
            {
                "physical_index": 0,
                "uuid": "GPU-SYNTHETIC-0",
                "name": "synthetic",
                "memory_total_mib": 24576,
            }
        ]
    }
    receipt = provider._validate_q8_exact_device_bindings(
        workflow, physical_gpus=(0,), hardware=hardware
    )
    assert receipt["status"] == "passed_exact_physical_gpu_and_uuid"

    tampered = json.loads(selection.read_text(encoding="utf-8"))
    tampered["selected_physical_gpu"] = 1
    provider.atomic_write_json(selection, tampered)
    with pytest.raises(provider.ProviderError, match="left the formal physical GPU"):
        provider._validate_q8_exact_device_bindings(
            workflow, physical_gpus=(0,), hardware=hardware
        )


def test_provider_object_seal_rejects_any_descriptor_tamper() -> None:
    payload = builder.build_descriptor(PROJECT_ROOT, physical_gpu=0)
    provider._sealed(payload, "descriptor_identity_sha256", role="test descriptor")
    tampered = deepcopy(payload)
    tampered["entrypoints"]["phaxis_production"]["measured_repeats"] = 2
    with pytest.raises(provider.ProviderError, match="does not seal"):
        provider._sealed(
            tampered, "descriptor_identity_sha256", role="test descriptor"
        )


def test_all_four_provider_receipts_close_the_formal_same_hardware_contract(
    tmp_path: Path,
) -> None:
    tasks = [f"SOURCE-{index:03d}" for index in range(283)]
    source = {
        "sha256": "a" * 64,
        "source_image_lock_identity_sha256": "b" * 64,
        "source_units_in_order": tasks,
        "source_unit_ordered_set_identity_sha256": sha256_json(tasks),
        "megapixels": 283.0,
    }
    hardware_object = {
        "host": "synthetic-host",
        "platform": "synthetic-platform",
        "processor": "synthetic-cpu",
        "gpus": [
            {
                "physical_index": 0,
                "uuid": "GPU-SYNTHETIC-0000",
                "name": "synthetic-gpu",
                "memory_total_mib": 24576,
                "driver_version": "synthetic-driver",
            }
        ],
    }
    preflight = {
        "command": ["nvidia-smi", "synthetic-not-executed"],
        "stdout_sha256": "c" * 64,
        "physical_gpus": [0],
    }
    hardware = {
        "hardware": hardware_object,
        "hardware_identity_sha256": sha256_json(hardware_object),
        "nvidia_smi_preflight": preflight,
        "nvidia_smi_preflight_identity_sha256": sha256_json(preflight),
    }
    telemetry = {
        "samples": 1,
        "peak_vram_mib": 1024.0,
        "mean_gpu_utilization_pct": 50.0,
    }
    common = {
        "source": source,
        "hardware": hardware,
        "telemetry": telemetry,
        "descriptor_sha256": "d" * 64,
        "input_hashes": {"source": "a" * 64},
        "config_hashes": {"provider": "d" * 64},
        "physical_gpus": (0,),
        "cvd": "0",
    }
    production_paths: dict[str, Path] = {}
    for mode, wall in (("phaxis_production", 10.0), ("frozen_v1_production", 20.0)):
        summary = provider._production_summary(
            mode=mode,
            wall=wall,
            stages=(
                {
                    "stage": "raw_to_final",
                    "wall_seconds": wall - 1.0,
                    "cache_status": "executed_fresh",
                },
            ),
            output_hashes={"final": "e" * 64},
            **common,
        )
        path = tmp_path / f"{mode}.json"
        atomic_write_json(path, summary)
        production_paths[mode] = path
        validated = benchmark._formal_summary(  # type: ignore[attr-defined]
            path,
            role=mode,
            expected_system=(
                benchmark.PHAXIS_BENCHMARK_SYSTEM
                if mode.startswith("phaxis_")
                else benchmark.FROZEN_V1_BENCHMARK_SYSTEM
            ),
            expected_kind="production",
        )
        assert validated["source_units_in_order"] == tasks
        assert summary["physical_gpu_mapping"] == [0]

    trace_path = tmp_path / "runtime_per_image.csv"
    trace = [
        {
            "source_unit": task,
            "wall_seconds": 1.0,
            "megapixels": 1.0,
            "io_seconds": 0.1,
            "preprocess_seconds": 0.2,
            "inference_seconds": 0.6,
            "postprocess_seconds": 0.1,
        }
        for task in tasks
    ]
    provider._write_csv(trace_path, provider._TRACE_FIELDS, trace)
    sequential_paths: dict[str, Path] = {}
    for mode in ("phaxis_sequential", "frozen_v1_sequential"):
        summary = provider._sequential_summary(
            mode=mode,
            trace=trace,
            trace_path=trace_path,
            output_hashes={"final": "f" * 64},
            **common,
        )
        path = tmp_path / f"{mode}.json"
        atomic_write_json(path, summary)
        sequential_paths[mode] = path
        benchmark._formal_summary(  # type: ignore[attr-defined]
            path,
            role=mode,
            expected_system=(
                benchmark.PHAXIS_BENCHMARK_SYSTEM
                if mode.startswith("phaxis_")
                else benchmark.FROZEN_V1_BENCHMARK_SYSTEM
            ),
            expected_kind="latency",
        )

    production_comparison = tmp_path / "production_comparison.json"
    sequential_comparison = tmp_path / "sequential_comparison.json"
    benchmark.compare_benchmarks(
        phaxis_summary=production_paths["phaxis_production"],
        baseline_summary=production_paths["frozen_v1_production"],
        output=production_comparison,
    )
    benchmark.compare_benchmarks(
        phaxis_summary=sequential_paths["phaxis_sequential"],
        baseline_summary=sequential_paths["frozen_v1_sequential"],
        output=sequential_comparison,
    )
    receipt = benchmark.aggregate_same_hardware_benchmark_receipt(
        phaxis_production_summary=production_paths["phaxis_production"],
        phaxis_sequential_summary=sequential_paths["phaxis_sequential"],
        frozen_v1_production_summary=production_paths["frozen_v1_production"],
        frozen_v1_sequential_summary=sequential_paths["frozen_v1_sequential"],
        production_comparison=production_comparison,
        sequential_comparison=sequential_comparison,
    )
    assert receipt["status"] == "passed"
    assert receipt["same_hardware_uuid_and_driver"] is True
    assert receipt["same_ordered_exact283_sources"] is True
    assert receipt["fresh_no_cache"] is True
