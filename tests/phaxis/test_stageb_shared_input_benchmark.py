from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "phaxis"
    / "benchmark_stageb_shared_input.py"
)
SPEC = importlib.util.spec_from_file_location(
    "benchmark_stageb_shared_input_test", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_physical_gpu_mapping_respects_cuda_visible_devices_order():
    query = "\n".join(
        (
            "0, GPU-aaaa, NVIDIA RTX 3090, 24576",
            "1, GPU-bbbb, NVIDIA RTX 3090, 24576",
        )
    )
    inventory = benchmark._parse_gpu_query(query)
    mapping = benchmark._resolve_device_mapping("cuda:0", "1,0", inventory)
    assert mapping["internal_index"] == 0
    assert mapping["selected_visible_token"] == "1"
    assert mapping["physical_index"] == 1
    assert mapping["uuid"] == "GPU-bbbb"
    uuid_mapping = benchmark._resolve_device_mapping(
        "cuda:1", "GPU-bbbb,GPU-aaaa", inventory
    )
    assert uuid_mapping["physical_index"] == 0


def test_benchmark_equivalence_oracle_reports_exact_and_maximum_difference():
    exact = benchmark._mapping_difference(
        {
            "head": np.asarray([[1.0, 2.0]], np.float32),
            "n": 1,
        },
        {
            "head": np.asarray([[1.0, 2.0]], np.float32),
            "n": 1,
        },
    )
    assert exact["exact"] is True
    assert exact["max_abs_difference"] == 0.0
    changed = benchmark._mapping_difference(
        {"head": np.asarray([1.0, 2.0], np.float32)},
        {"head": np.asarray([1.0, 2.25], np.float32)},
    )
    assert changed["exact"] is False
    assert changed["max_abs_difference"] == 0.25


def test_benchmark_aggregates_latency_throughput_and_peak_vram():
    runs = []
    for index, seconds in enumerate((1.0, 2.0, 3.0)):
        runs.append(
            {
                "images": 2,
                "source_megapixels": 10.0,
                "io_decode_seconds": seconds / 10,
                "inference_seconds": seconds,
                "end_to_end_seconds": seconds + 0.1,
                "inference_megapixels_per_second": 10.0 / seconds,
                "end_to_end_megapixels_per_second": 10.0 / (seconds + 0.1),
                "peak_allocated_vram_mib": 1000.0 + index,
                "peak_reserved_vram_mib": 1200.0 + index,
            }
        )
    aggregate = benchmark._aggregate_runs(runs)
    assert aggregate["runs"] == 3
    assert aggregate["images_per_run"] == 2
    assert aggregate["inference_seconds"]["median"] == 2.0
    assert aggregate["inference_seconds"]["p95"] == 2.9
    assert aggregate["peak_allocated_vram_mib"]["maximum"] == 1002.0
