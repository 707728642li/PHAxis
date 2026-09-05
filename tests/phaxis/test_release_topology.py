from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from phaxis import benchmark
from phaxis.contracts import ContractError
from phaxis.io import sha256_file, sha256_json
from phaxis.release_topology import (
    MANDATORY_STAGE_ORDER,
    ReleaseTopologyError,
    STAGE_DEPENDENCIES,
    require_manifest_stage_dependencies,
    validate_release_topology,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _exact283_manifest(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("task_id", "image_path", "image_sha256"),
        )
        writer.writeheader()
        for index in range(283):
            writer.writerow(
                {
                    "task_id": f"SOURCE-{index:03d}",
                    "image_path": f"source-{index:03d}.tif",
                    "image_sha256": f"{index + 1:064x}",
                }
            )
    return path


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_real_producer_sources_define_a_reachable_release_topology() -> None:
    report = validate_release_topology(project_root=PROJECT_ROOT)
    assert report["status"] == "structurally_valid_non_formal"
    assert report["declared_capability_gaps"] == []
    assert report["formal_release_allowed"] is False
    assert report["scientific_or_performance_results_present"] is False
    assert report["producer_stage_order"] == list(MANDATORY_STAGE_ORDER)
    order = report["producer_stage_order"]
    assert order.index("figures") < order.index("evidence")
    assert order.index("production_manifest") < order.index("root_provider_exact283")
    assert order.index("authority_pin") < order.index("analysis_workflow_manifest")
    assert order.index("analysis_workflow_manifest") < order.index(
        "benchmark_phaxis_production"
    )
    assert order.index("benchmark_phaxis_production") < order.index(
        "benchmark_same_hardware"
    )
    assert order.index("benchmark_frozen_v1_sequential") < order.index(
        "benchmark_same_hardware"
    )
    assert order.index("figure_inputs") < order.index("figures")
    assert order.index("evidence") < order.index("official_apply")
    assert order.index("official_apply") < order.index("source_release")
    assert order.index("source_release") < order.index("clean_install")
    assert order.index("distributions") < order.index("clean_install")
    assert order.index("clean_install") < order.index("values")
    assert order[-1] == "release_finalize"

    checked = report["real_producer_source_checks"]
    assert checked
    by_stage = {record["stage"]: record for record in checked}
    assert "--expected-task-count" in by_stage["overlay_evidence"][
        "required_cli_options"
    ]
    for record in checked:
        source = PROJECT_ROOT / record["producer"]
        assert source.is_file()
        assert record["producer_sha256"] == sha256_file(source)
        assert record["required_cli_options"]
    unsigned = dict(report)
    assert unsigned.pop("topology_audit_identity_sha256") == sha256_json(unsigned)

    # The test exercises the canonical edge validator, not fabricated receipt
    # dictionaries.  Removing the real figures -> evidence edge must fail.
    stage_inputs = {
        name: set(dependencies) for name, dependencies in STAGE_DEPENDENCIES.items()
    }
    stage_inputs["evidence"].remove("figures")
    with pytest.raises(ReleaseTopologyError, match="evidence.*figures"):
        require_manifest_stage_dependencies(stage_inputs)


def test_topology_cli_accepts_the_closed_producer_graph_without_claiming_results() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/phaxis/check_post_training_release_topology.py"),
            "--project-root",
            str(PROJECT_ROOT),
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["declared_capability_gaps"] == []
    assert payload["formal_release_allowed"] is False
    assert payload["scientific_or_performance_results_present"] is False


def test_repository_legacy_partial_chain_yields_only_a_nonformal_blocked_gate(
    tmp_path: Path,
) -> None:
    manifest = _exact283_manifest(tmp_path / "source283.csv")
    gate = benchmark.inspect_frozen_v1_exact283_benchmark_producer(
        project_root=PROJECT_ROOT,
        source_manifest=manifest,
    )
    assert gate["schema_version"] == benchmark.FROZEN_V1_PRODUCER_GATE_SCHEMA
    assert gate["status"] == "blocked_missing_real_producer"
    assert gate["formal_aggregation_allowed"] is False
    assert gate["benchmark_execution_performed"] is False
    assert gate["performance_measurements_present"] is False
    assert gate["formal_summary_schema_emitted"] is False
    assert gate["blind_images_used"] == 0
    assert set(gate["blocker_codes"]) == {
        "FROZEN_V1_PRODUCTION_PRODUCER_MISSING",
        "FROZEN_V1_SEQUENTIAL_PRODUCER_MISSING",
        "FROZEN_V1_FORMAL_HARDWARE_BINDING_MISSING",
        "FROZEN_V1_NONOVERLAPPING_STAGE_TIMING_HOOKS_MISSING",
    }
    expected_partial = {
        role
        for role, relative in (
            ("sharded_inference", "scripts/run_six_condition_v1_sharded.py"),
            ("shard_merge", "scripts/merge_six_condition_v1_shards.py"),
            (
                "prefill_adapter",
                "scripts/materialize_six_condition_v1_prefill_adapter.py",
            ),
        )
        if (PROJECT_ROOT / relative).is_file()
    }
    assert {
        record["role"] for record in gate["discovered_partial_components"]
    } == expected_partial
    assert all(
        record["formal_eligible"] is False
        and record["sha256"] == sha256_file(PROJECT_ROOT / record["path"])
        for record in gate["discovered_partial_components"]
    )
    forbidden_result_keys = {
        "benchmark_system",
        "images",
        "n_images",
        "batch_wall_seconds",
        "median_seconds_per_image",
        "p95_seconds_per_image",
        "images_per_minute",
        "megapixels_per_second",
        "peak_vram_mib",
        "mean_gpu_utilization_pct",
        "speedup",
        "summary_identity_sha256",
    }
    assert not (set(_walk_keys(gate)) & forbidden_result_keys)
    unsigned = dict(gate)
    assert unsigned.pop("gate_identity_sha256") == sha256_json(unsigned)

    gate_path = tmp_path / "legacy-producer-gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError):
        benchmark._formal_summary(  # type: ignore[attr-defined]
            gate_path,
            role="frozen v1 production",
            expected_system=benchmark.FROZEN_V1_BENCHMARK_SYSTEM,
            expected_kind="production",
        )

    fake_interface = {
        "schema_version": benchmark.FROZEN_V1_PRODUCER_INTERFACE_SCHEMA,
        "target_system_id": benchmark.FROZEN_V1_BENCHMARK_SYSTEM,
        "source_manifest_sha256": gate["source_manifest_sha256"],
        "ordered_source_lock_identity_sha256": gate[
            "ordered_source_lock_identity_sha256"
        ],
        "capabilities": {
            record["name"]: True for record in gate["requirements"]
        },
        "entrypoints": {
            role: {"path": f"scripts/{role}.py", "sha256": "f" * 64}
            for role in (
                "execute_fresh_production_exact283",
                "execute_fresh_sequential_exact283",
            )
        },
    }
    fake_interface["interface_identity_sha256"] = sha256_json(fake_interface)
    fake_path = tmp_path / "fake-interface.json"
    fake_path.write_text(
        json.dumps(fake_interface, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ContractError, match="entrypoint drifted"):
        benchmark.inspect_frozen_v1_exact283_benchmark_producer(
            project_root=PROJECT_ROOT,
            source_manifest=manifest,
            producer_interface=fake_path,
        )


def test_legacy_producer_cli_is_cpu_only_check_only_and_writes_nothing(
    tmp_path: Path,
) -> None:
    manifest = _exact283_manifest(tmp_path / "source283.csv")
    script = PROJECT_ROOT / "scripts/phaxis/benchmark_full_workflow.py"
    command = [
        sys.executable,
        str(script),
        "--inspect-frozen-v1-producer",
        "--project-root",
        str(PROJECT_ROOT),
        "--manifest",
        str(manifest),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 3
    gate = json.loads(completed.stdout)
    assert gate["status"] == "blocked_missing_real_producer"
    assert gate["benchmark_execution_performed"] is False

    forbidden_output = tmp_path / "must-not-exist.json"
    refused = subprocess.run(
        [*command, "--output", str(forbidden_output)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 2
    assert "check-only" in refused.stderr
    assert not forbidden_output.exists()
