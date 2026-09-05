from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
from types import ModuleType

import pytest

from phaxis.io import atomic_write_json, read_json
from phaxis.root_provider.identity import IdentityOnlyDatasetContract
from phaxis.root_provider.runtime import PipelineConfig, build_execution_plan
import phaxis.root_provider.identity as identity_module
import phaxis.root_provider.runtime as runtime_module
import phaxis.root_provider.stage_entry as stage_module


def test_legacy_pipeline_identity_without_strict_gpu_field_is_exactly_recoverable(
    tmp_path: Path,
) -> None:
    identity = {
        "schema_version": runtime_module.PIPELINE_SCHEMA,
        "bundle_identity_sha256": "a" * 64,
        "files": {"input_manifest": "c" * 64},
        "v1_physical_gpus": [0],
        "q8_physical_gpus": [1],
        "strict_physical_gpu": False,
        "v1_shards": 4,
        "v20_shards": 8,
        "q8_shards": 8,
        "field_batch_size": 10,
        "query_batch_size": 32,
    }
    legacy = dict(identity)
    legacy.pop("strict_physical_gpu")
    legacy_sha = runtime_module.sha256_json(legacy)
    state = {"pipeline_identity_sha256": legacy_sha}

    assert runtime_module._legacy_resume_identity(
        state=state,
        identity_payload=identity,
        strict_physical_gpu=False,
    ) == legacy_sha
    assert runtime_module._legacy_resume_identity(
        state=state,
        identity_payload=identity,
        strict_physical_gpu=True,
    ) is None
    assert runtime_module._legacy_resume_identity(
        state={**state, "strict_physical_gpu": False},
        identity_payload=identity,
        strict_physical_gpu=False,
    ) is None
    assert runtime_module._legacy_resume_identity(
        state={"pipeline_identity_sha256": "b" * 64},
        identity_payload=identity,
        strict_physical_gpu=False,
    ) is None


def _config(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        project=tmp_path / "portable_project",
        bundle=tmp_path / "model_bundle",
        input_manifest=tmp_path / "inputs.csv",
        acquisition_gate=tmp_path / "gate.json",
        deployment_metadata=tmp_path / "metadata.csv",
        canonical_manifest=tmp_path / "canonical.csv",
        deployment_manifest=tmp_path / "deployment.csv",
        deployment_lock=tmp_path / "deployment.lock.json",
        image_root=tmp_path / "images",
        output=tmp_path / "fresh_run",
        v1_physical_gpus=(0,),
        q8_physical_gpus=(1,),
        python_executable=tmp_path / "runtime" / "python.exe",
        reference_registry=tmp_path / "reference283.json",
    )


def test_execution_plan_has_no_implicit_legacy_machine_paths(tmp_path: Path) -> None:
    payload = build_execution_plan(_config(tmp_path))
    encoded = json.dumps(payload)
    assert payload["status"] == "planned_not_executed"
    assert payload["v1_physical_gpus"] == [0]
    assert payload["q8_physical_gpus"] == [1]
    windows_host_path_marker = "Z:" + "\\\\"
    assert windows_host_path_marker not in encoded
    assert "envs\\\\rhpheno\\\\python.exe" not in encoded
    assert "CUDA_VISIBLE_DEVICES=0" not in encoded
    assert "--physical-gpu" in encoded
    assert "identity_only_training_payload_not_packaged" not in encoded
    assert payload["training_payload_opened_at_deployment"] is False


def test_q8_strict_premerge_binding_locks_every_shard_index_and_uuid(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path)
    config = PipelineConfig(
        **{
            **base.__dict__,
            "q8_physical_gpus": (1,),
            "q8_shards": 2,
            "strict_physical_gpu": True,
        }
    )
    shard_root = tmp_path / "q8_shards"
    for index in range(2):
        path = shard_root / f"shard{index:02d}_gpu1" / "q8_device_selection.json"
        path.parent.mkdir(parents=True)
        atomic_write_json(
            path,
            {
                "schema_version": "PHAxis-Q8-device-selection-1.0",
                "requested_physical_gpu": 1,
                "selected_physical_gpu": 1,
                "exact_physical_gpu_required": True,
                "gpu_snapshot": [{"index": 1, "uuid": "GPU-SYNTHETIC-1"}],
            },
        )
    binding = runtime_module._validate_q8_shard_device_bindings(
        config, shard_root
    )
    assert binding["status"] == "passed_before_q8_merge"
    assert len(binding["records"]) == 2
    assert {row["physical_gpu_uuid"] for row in binding["records"]} == {
        "GPU-SYNTHETIC-1"
    }

    tampered_path = shard_root / "shard01_gpu1" / "q8_device_selection.json"
    tampered = read_json(tampered_path)
    tampered["selected_physical_gpu"] = 0
    atomic_write_json(tampered_path, tampered)
    with pytest.raises(RuntimeError, match="left its strict planned physical GPU"):
        runtime_module._validate_q8_shard_device_bindings(config, shard_root)


def test_importing_runtime_modules_has_no_process_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def forbidden(*_args: object, **_kwargs: object) -> None:
        calls.append("called")
        raise AssertionError("subprocess side effect during import")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    importlib.reload(runtime_module)
    importlib.reload(stage_module)
    assert calls == []


def test_q8_device_selection_honors_idle_explicit_preference() -> None:
    rows = [
        {"index": 0, "uuid": "GPU-0", "memory_total_mib": 24576, "memory_used_mib": 0, "utilization_samples_percent": [0, 0, 0, 0, 0]},
        {"index": 1, "uuid": "GPU-1", "memory_total_mib": 24576, "memory_used_mib": 0, "utilization_samples_percent": [0, 0, 0, 0, 0]},
    ]
    selected, reason = stage_module._q8_device_from_snapshot(1, rows, set())
    assert selected == 1
    assert reason == "explicit_requested_gpu_available"


def test_q8_device_selection_allows_active_process_when_capacity_and_sustained_utilization_are_safe() -> None:
    rows = [
        {"index": 0, "uuid": "GPU-0", "memory_total_mib": 24576, "memory_used_mib": 0, "utilization_samples_percent": [0, 0, 0, 0, 0]},
        {"index": 1, "uuid": "GPU-1", "memory_total_mib": 24576, "memory_used_mib": 12000, "utilization_samples_percent": [61, 65, 68, 70, 73]},
    ]
    selected, reason = stage_module._q8_device_from_snapshot(
        1, rows, {"GPU-1"}
    )
    assert selected == 1
    assert reason == "explicit_requested_gpu_shared_capacity_available"


def test_q8_device_selection_fails_closed_when_peak_plus_reserve_does_not_fit() -> None:
    rows = [
        {"index": 0, "uuid": "GPU-0", "memory_total_mib": 24576, "memory_used_mib": 21000, "utilization_samples_percent": [0, 0, 0, 0, 0]},
        {"index": 1, "uuid": "GPU-1", "memory_total_mib": 24576, "memory_used_mib": 19000, "utilization_samples_percent": [0, 0, 0, 0, 0]},
    ]
    with pytest.raises(RuntimeError, match="no capacity-and-utilization-safe fallback"):
        stage_module._q8_device_from_snapshot(1, rows, {"GPU-0", "GPU-1"})


def test_q8_device_selection_fails_closed_for_sustained_utilization_at_or_above_80_percent() -> None:
    rows = [
        {"index": 0, "uuid": "GPU-0", "memory_total_mib": 24576, "memory_used_mib": 0, "utilization_samples_percent": [80, 82, 84, 86, 88]},
        {"index": 1, "uuid": "GPU-1", "memory_total_mib": 24576, "memory_used_mib": 0, "utilization_samples_percent": [91, 95, 97, 99, 100]},
    ]
    with pytest.raises(RuntimeError, match="no capacity-and-utilization-safe fallback"):
        stage_module._q8_device_from_snapshot(1, rows, {"GPU-0", "GPU-1"})


def test_q8_device_selection_rejects_transient_epoch_gap() -> None:
    rows = [
        {"index": 0, "uuid": "GPU-0", "memory_total_mib": 24576, "memory_used_mib": 21000, "utilization_samples_percent": [0] * 9},
        # Five low observations would pass the former median gate even though
        # the same training process is fully occupied around the brief gap.
        {"index": 1, "uuid": "GPU-1", "memory_total_mib": 24576, "memory_used_mib": 12000, "utilization_samples_percent": [100, 100, 5, 3, 2, 4, 6, 100, 100]},
    ]
    with pytest.raises(RuntimeError, match="no capacity-and-utilization-safe fallback"):
        stage_module._q8_device_from_snapshot(1, rows, {"GPU-1"})


def test_q8_device_selection_accepts_genuinely_sustained_low_utilization() -> None:
    rows = [
        {"index": 0, "uuid": "GPU-0", "memory_total_mib": 24576, "memory_used_mib": 0, "utilization_samples_percent": [0] * 9},
        {"index": 1, "uuid": "GPU-1", "memory_total_mib": 24576, "memory_used_mib": 12000, "utilization_samples_percent": [61, 65, 68, 70, 73, 72, 69, 71, 74]},
    ]
    selected, reason = stage_module._q8_device_from_snapshot(
        1, rows, {"GPU-1"}
    )
    assert selected == 1
    assert reason == "explicit_requested_gpu_shared_capacity_available"


def test_q8_device_selection_uses_safe_fallback_when_requested_gpu_is_not_eligible() -> None:
    rows = [
        {"index": 0, "uuid": "GPU-0", "memory_total_mib": 24576, "memory_used_mib": 5000, "utilization_samples_percent": [18, 21, 23, 25, 29]},
        {"index": 1, "uuid": "GPU-1", "memory_total_mib": 24576, "memory_used_mib": 19000, "utilization_samples_percent": [65, 68, 70, 72, 75]},
    ]
    selected, reason = stage_module._q8_device_from_snapshot(
        1, rows, {"GPU-0", "GPU-1"}
    )
    assert selected == 0
    assert reason == "requested_gpu_ineligible_safe_shared_fallback"


def test_q8_formal_exact_device_mode_forbids_safe_fallback() -> None:
    rows = [
        {"index": 0, "uuid": "GPU-0", "memory_total_mib": 24576, "memory_used_mib": 5000, "utilization_samples_percent": [18, 21, 23, 25, 29]},
        {"index": 1, "uuid": "GPU-1", "memory_total_mib": 24576, "memory_used_mib": 19000, "utilization_samples_percent": [65, 68, 70, 72, 75]},
    ]
    with pytest.raises(
        stage_module.Q8CapacityUnavailableError,
        match="formal exact-device mode forbids fallback",
    ):
        stage_module._q8_device_from_snapshot(
            1,
            rows,
            {"GPU-0", "GPU-1"},
            allow_fallback=False,
        )


def test_q8_runtime_selection_uses_multi_sample_utilization_and_keeps_process_rows_for_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    utilization = [60, 62, 64, 66, 68, 65, 63, 67, 69]
    gpu_query_calls = 0
    sleeps: list[float] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal gpu_query_calls
        if any(str(value).startswith("--query-gpu=") for value in command):
            value = utilization[gpu_query_calls]
            gpu_query_calls += 1
            stdout = (
                "0, GPU-0, 24576, 0, 0, 35\n"
                f"1, GPU-1, 24576, 12000, {value}, 68\n"
            )
            return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
        if any(str(value).startswith("--query-compute-apps=") for value in command):
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="GPU-1, 1234, trainer.exe, 12000\n",
                stderr="",
            )
        raise AssertionError(f"unexpected subprocess command: {command}")

    monkeypatch.setattr(stage_module.subprocess, "run", fake_run)
    monkeypatch.setattr(stage_module.time, "sleep", sleeps.append)
    record_path = tmp_path / "q8_device_selection.json"
    record = stage_module._select_q8_physical_gpu(1, record_path)

    assert gpu_query_calls == stage_module.GPU_UTILIZATION_SAMPLE_COUNT
    assert sleeps == [stage_module.GPU_UTILIZATION_SAMPLE_INTERVAL_SECONDS] * 8
    assert record["selected_physical_gpu"] == 1
    assert record["reason"] == "explicit_requested_gpu_shared_capacity_available"
    assert record["gpu_snapshot"][1]["utilization_samples_percent"] == utilization
    assert record["gpu_snapshot"][1]["sustained_utilization_percent"] == 69.0
    assert record["sustained_utilization_statistic"] == "maximum_all_samples_must_be_below_limit"
    assert record["active_compute_process_is_not_automatic_veto"] is True
    assert record_path.is_file()


def test_q8_capacity_wait_is_low_frequency_atomic_and_resumes_prior_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "shard00_gpu1"
    _control_root, wait_path, selection_path = stage_module._q8_control_paths(output)
    atomic_write_json(
        wait_path,
        {
            "schema_version": stage_module.Q8_WAIT_SCHEMA,
            "status": "waiting_for_gpu_capacity",
            "requested_physical_gpu": 1,
            "wait_started_utc": "2026-08-28T00:00:00+00:00",
            "capacity_checks": 3,
            "scheduled_sleep_seconds": 180.0,
            "blind_images_used": 0,
        },
    )
    outcomes: list[object] = [
        stage_module.Q8CapacityUnavailableError("both lanes sustained >=80%"),
        {
            "requested_physical_gpu": 1,
            "selected_physical_gpu": 0,
            "reason": "requested_gpu_ineligible_safe_shared_fallback",
            "blind_images_used": 0,
        },
    ]

    def fake_select(_requested: int, receipt: Path) -> dict[str, object]:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        atomic_write_json(receipt, outcome)
        return outcome  # type: ignore[return-value]

    sleeps: list[float] = []
    monkeypatch.setattr(stage_module, "_select_q8_physical_gpu", fake_select)
    monkeypatch.setattr(stage_module.time, "sleep", sleeps.append)
    selected = stage_module._wait_for_q8_physical_gpu(1, output)

    receipt = json.loads(wait_path.read_text(encoding="utf-8"))
    assert selected["selected_physical_gpu"] == 0
    assert sleeps == [stage_module.Q8_CAPACITY_RETRY_SECONDS]
    assert receipt["status"] == "capacity_available_q8_starting"
    assert receipt["wait_started_utc"] == "2026-08-28T00:00:00+00:00"
    assert receipt["capacity_checks"] == 4
    assert receipt["scheduled_sleep_seconds"] == 240.0
    assert receipt["no_process_killed_or_suspended"] is True
    assert receipt["blind_images_used"] == 0
    assert not output.exists()
    assert selection_path.is_file()


def test_q8_resume_migrates_legacy_wait_receipt_without_overwriting_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "shard00_gpu1"
    legacy_wait = tmp_path / ".shard00_gpu1.waiting_for_gpu_capacity.json"
    legacy_payload = {
        "schema_version": stage_module.Q8_WAIT_SCHEMA,
        "status": "capacity_available_q8_starting",
        "requested_physical_gpu": 1,
        "wait_started_utc": "2026-08-28T00:00:00+00:00",
        "capacity_checks": 7,
        "scheduled_sleep_seconds": 420.0,
        "blind_images_used": 0,
    }
    atomic_write_json(legacy_wait, legacy_payload)
    legacy_bytes = legacy_wait.read_bytes()

    def safe_selection(_requested: int, receipt: Path) -> dict[str, object]:
        selected: dict[str, object] = {
            "requested_physical_gpu": 1,
            "selected_physical_gpu": 1,
            "reason": "explicit_requested_gpu_available",
            "blind_images_used": 0,
        }
        atomic_write_json(receipt, selected)
        return selected

    monkeypatch.setattr(stage_module, "_select_q8_physical_gpu", safe_selection)
    selected = stage_module._wait_for_q8_physical_gpu(1, output)
    _control_root, wait_path, selection_path = stage_module._q8_control_paths(output)

    assert selected["selected_physical_gpu"] == 1
    assert legacy_wait.read_bytes() == legacy_bytes
    assert read_json(wait_path)["resumed_from_prior_wait_receipt"] is True
    assert read_json(wait_path)["capacity_checks"] == 7
    assert selection_path.is_file()
    assert not output.exists()


def test_q8_capacity_wait_does_not_retry_noncapacity_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def invalid_snapshot(_requested: int, _output: Path) -> dict[str, object]:
        raise RuntimeError("GPU UUID changed during sampling")

    monkeypatch.setattr(stage_module, "_select_q8_physical_gpu", invalid_snapshot)
    with pytest.raises(RuntimeError, match="UUID changed"):
        stage_module._wait_for_q8_physical_gpu(1, tmp_path / "shard00_gpu1")
    assert not list(tmp_path.glob("*.waiting_for_gpu_capacity.json"))


def test_atomic_stage_publish_and_failed_attempt_isolation(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    with stage_module._atomic_output(destination, "prepare", False) as attempt:
        assert attempt is not None
        atomic_write_json(
            attempt / "summary.json",
            {"status": "complete", "blind_images_used": 0},
        )
    assert (destination / "summary.json").is_file()

    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="intentional"):
        with stage_module._atomic_output(failed, "prepare", False):
            raise RuntimeError("intentional")
    assert not failed.exists()
    attempts = list(tmp_path.glob(".failed.prepare.attempt-*"))
    assert len(attempts) == 1
    record = json.loads(
        (attempts[0] / "PHAXIS_STAGE_FAILURE.json").read_text(encoding="utf-8")
    )
    assert record["official_output_published"] is False


def test_q8_fresh_and_resume_preserve_candidate_empty_directory_contract_and_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "shard00_gpu1"

    def safe_selection(_requested: int, receipt: Path) -> dict[str, object]:
        selected: dict[str, object] = {
            "requested_physical_gpu": 1,
            "selected_physical_gpu": 1,
            "reason": "explicit_requested_gpu_available",
            "blind_images_used": 0,
        }
        atomic_write_json(receipt, selected)
        return selected

    monkeypatch.setattr(stage_module, "_select_q8_physical_gpu", safe_selection)
    with pytest.raises(RuntimeError, match="candidate failed after taking ownership"):
        with stage_module._atomic_output(destination, "q8", False) as attempt:
            assert attempt is not None
            stage_module._wait_for_q8_physical_gpu(1, attempt)
            # This is the frozen candidate_benchmark empty-output contract.
            assert not attempt.exists()
            attempt.mkdir()
            atomic_write_json(attempt / "candidate_partial.json", {"kept": True})
            raise RuntimeError("candidate failed after taking ownership")

    failed_attempts = list(tmp_path.glob(".shard00_gpu1.q8.attempt-*"))
    assert len(failed_attempts) == 1
    failed_attempt = failed_attempts[0]
    assert (failed_attempt / "candidate_partial.json").is_file()
    assert (failed_attempt / "PHAXIS_STAGE_FAILURE.json").is_file()
    failed_record_bytes = (failed_attempt / "PHAXIS_STAGE_FAILURE.json").read_bytes()

    # Resume starts a new clean official destination without altering the
    # quarantined failure evidence.
    with stage_module._atomic_output(destination, "q8", True) as attempt:
        assert attempt is not None
        assert not attempt.exists()
        attempt.mkdir()
        atomic_write_json(
            attempt / "summary.json",
            {"status": "complete", "blind_images_used": 0},
        )
    assert (destination / "PHAXIS_STAGE_COMPLETE.json").is_file()
    assert (failed_attempt / "PHAXIS_STAGE_FAILURE.json").read_bytes() == failed_record_bytes

    complete_bytes = (destination / "summary.json").read_bytes()
    with stage_module._atomic_output(destination, "q8", True) as attempt:
        assert attempt is None
    assert (destination / "summary.json").read_bytes() == complete_bytes


def test_run_q8_keeps_candidate_output_absent_until_candidate_takes_ownership(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "shard00_gpu1"
    model = tmp_path / "model"
    observed: dict[str, object] = {}

    def safe_selection(_requested: int, receipt: Path) -> dict[str, object]:
        selected: dict[str, object] = {
            "schema_version": "PHAxis-Q8-device-selection-1.0",
            "requested_physical_gpu": 1,
            "selected_physical_gpu": 1,
            "reason": "explicit_requested_gpu_available",
            "blind_images_used": 0,
        }
        atomic_write_json(receipt, selected)
        return selected

    def fake_preflight(_physical_gpu: int, receipt: Path) -> None:
        receipt.write_text("CPU mock preflight\n", encoding="utf-8")

    candidate = ModuleType("rhaxis_nextgen.candidate_benchmark")
    candidate.__file__ = str(model / "runtime/src/rhaxis_nextgen/candidate_benchmark.py")

    def candidate_main() -> None:
        output = Path(stage_module.sys.argv[stage_module.sys.argv.index("--output") + 1])
        observed["output_absent_on_entry"] = not output.exists()
        output.mkdir(parents=True)
        atomic_write_json(
            output / "summary.json",
            {"status": "complete", "blind_images_used": 0},
        )

    candidate.main = candidate_main  # type: ignore[attr-defined]
    package = ModuleType("rhaxis_nextgen")
    package.__path__ = []  # type: ignore[attr-defined]
    package.candidate_benchmark = candidate  # type: ignore[attr-defined]
    monkeypatch.setitem(stage_module.sys.modules, "rhaxis_nextgen", package)
    monkeypatch.setitem(
        stage_module.sys.modules, "rhaxis_nextgen.candidate_benchmark", candidate
    )
    monkeypatch.setattr(stage_module, "_select_q8_physical_gpu", safe_selection)
    monkeypatch.setattr(stage_module, "_gpu_preflight", fake_preflight)
    monkeypatch.setattr(stage_module, "_model_root", lambda _bundle: model)
    monkeypatch.setattr(stage_module, "_prepend_sys_path", lambda _paths: None)
    monkeypatch.setattr(
        stage_module, "install_deployment_identity_adapter", lambda *_args, **_kwargs: None
    )
    args = stage_module.argparse.Namespace(
        bundle=tmp_path / "bundle",
        deployment_manifest=tmp_path / "deployment.csv",
        deployment_lock=tmp_path / "deployment.lock.json",
        deployment_image_root=tmp_path / "images",
        physical_gpu=1,
        shard_index=0,
        num_shards=8,
        field_batch_size=10,
        query_batch_size=32,
        task_id=[],
    )

    with stage_module._atomic_output(destination, "q8", False) as attempt:
        assert attempt is not None
        stage_module._run_q8(args, attempt)

    assert observed["output_absent_on_entry"] is True
    selection_copy = read_json(destination / "q8_device_selection.json")
    assert selection_copy["selected_physical_gpu"] == 1
    summary = read_json(destination / "summary.json")
    mapping = summary["portable_phaxis_device_mapping"]
    assert mapping["selection_receipt_sha256"] == stage_module.sha256_file(
        Path(mapping["selection_receipt"])
    )
    assert Path(mapping["selection_receipt"]).parent == destination.resolve()
    assert mapping["control_selection_receipt_sha256"] == stage_module.sha256_file(
        Path(mapping["control_selection_receipt"])
    )

    failed_destination = tmp_path / "shard01_gpu1"

    def candidate_fails_before_output_creation() -> None:
        output = Path(stage_module.sys.argv[stage_module.sys.argv.index("--output") + 1])
        assert not output.exists()
        raise RuntimeError("CPU mock candidate startup failure")

    candidate.main = candidate_fails_before_output_creation  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="CPU mock candidate startup failure"):
        with stage_module._atomic_output(failed_destination, "q8", True) as attempt:
            assert attempt is not None
            stage_module._run_q8(args, attempt)

    failed_attempts = list(tmp_path.glob(".shard01_gpu1.q8.attempt-*"))
    assert len(failed_attempts) == 1
    assert (failed_attempts[0] / "q8_device_selection.json").is_file()
    assert (failed_attempts[0] / "nvidia_smi_preflight_outer.txt").is_file()
    assert (failed_attempts[0] / "PHAXIS_STAGE_FAILURE.json").is_file()


def test_identity_adapter_patches_only_explicit_deployment_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = IdentityOnlyDatasetContract(
        root=tmp_path / "not_packaged",
        samples=(),
        train=(),
        val=(),
        file_sha256={},
        identity_sha256="d" * 64,
        split_file_sha256={},
        split_identity_sha256="s" * 64,
        split_override_path=tmp_path / "split.lock.json",
    )
    monkeypatch.setattr(
        identity_module, "load_identity_only_contract", lambda *_args, **_kwargs: contract
    )
    deployment = ModuleType("deployment")
    deployment.load_dataset_contract = lambda *_args: object()
    observed = identity_module.install_deployment_identity_adapter(
        deployment, tmp_path / "bundle"
    )
    assert observed is contract
    assert deployment.load_dataset_contract(tmp_path) is contract
    assert contract.samples == contract.train == contract.val == ()
