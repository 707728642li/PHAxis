from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUEUE_SCRIPT = PROJECT_ROOT / "scripts" / "phaxis" / "run_stageb_train399_gpu_queue.ps1"
PWSH = shutil.which("pwsh")


def _run_pwsh(*arguments: str) -> subprocess.CompletedProcess[str]:
    if PWSH is None:
        pytest.skip("PowerShell 7 is unavailable")
    return subprocess.run(
        [
            PWSH,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(QUEUE_SCRIPT),
            *arguments,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    ("lane", "gpu", "expected_seeds", "first_policy"),
    [
        (
            "odd-resume",
            "1",
            [2026082801, 2026082803, 2026082805],
            "resume-required-if-incomplete",
        ),
        (
            "even-fresh",
            "0",
            [2026082802, 2026082804],
            "fresh-or-resume-if-incomplete",
        ),
        (
            "pending-single-gpu",
            "1",
            [2026082803, 2026082804, 2026082805],
            "fresh-or-resume-if-incomplete",
        ),
    ],
)
def test_cpu_plan_locks_physical_to_internal_device(
    lane: str, gpu: str, expected_seeds: list[int], first_policy: str
) -> None:
    completed = _run_pwsh("-Lane", lane, "-PhysicalGpu", gpu, "-PlanOnly")
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["physical_gpu"] == int(gpu)
    assert payload["cuda_visible_devices"] == gpu
    assert payload["cuda_device_order"] == "PCI_BUS_ID"
    assert [row["seed"] for row in payload["members"]] == expected_seeds
    assert payload["members"][0]["initial_policy"] == first_policy
    assert {row["internal_device"] for row in payload["members"]} == {"cuda:0"}
    assert payload["expected_epochs"] == 60
    assert payload["expected_steps_per_epoch"] == 399
    assert payload["expected_global_steps"] == 23940
    assert payload["failure_policy"] == "stop-entire-lane-on-first-failure"


def _write_completed_fixture(root: Path, seed: int, gpu: int) -> Path:
    run = root / f"seed_{seed}"
    run.mkdir(parents=True)
    checkpoint = run / "last.pt"
    checkpoint.write_bytes(b"CPU-only fake checkpoint for queue receipt validation\n")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    history = [
        {
            "epoch": epoch,
            "batches": 399,
            "global_step": epoch * 399,
            "train_loss_total": 10.0 / epoch,
            "validation_evaluated": False,
        }
        for epoch in range(1, 61)
    ]
    (run / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (run / "config.json").write_text(
        json.dumps(
            {
                "epochs": 60,
                "batch_size": 8,
                "crops_per_image": 8,
                "fixed_last_epoch_policy": True,
                "amp": True,
                "amp_initial_scale": 1024.0,
                "amp_growth_interval": 1_000_000,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run / "training_contract.json").write_text(
        json.dumps(
            {
                "schema_version": "PHAxis-StageB-train399-checkpoint-contract-1.0",
                "formal_training": True,
                "seed": seed,
                "member_id": f"seed_{seed}",
                "blind_images_used": 0,
                "pyRootHair_called_or_copied": False,
                "validation_labels_used_for_gradient_or_early_stopping": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run / "initialization.json").write_text(
        json.dumps({"historical_stageb_checkpoint_loaded": False}, indent=2),
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "PHAxis-StageB-train399-training-receipt-1.0",
        "status": "completed",
        "formal_training": True,
        "seed": seed,
        "epochs": 60,
        "steps_per_epoch": 399,
        "global_steps": 23940,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "cuda_visible_devices": str(gpu),
        "internal_device": "cuda:0",
        "nvidia_smi_preflight_status": "passed",
        "nvidia_smi_training_monitor_status": "passed",
        "validation_evaluated_during_training": False,
        "blind_images_used": 0,
    }
    (run / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return run


def test_cpu_completion_verifier_binds_receipt_history_and_checkpoint_hash(
    tmp_path: Path,
) -> None:
    seed = 2026082802
    run = _write_completed_fixture(tmp_path, seed, gpu=0)
    completed = _run_pwsh(
        "-VerifySeedDirectory",
        str(run),
        "-VerifySeed",
        str(seed),
        "-ExpectedPhysicalGpu",
        "0",
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "verified_complete"
    assert payload["epochs"] == 60
    assert payload["global_steps"] == 23940
    assert payload["checkpoint_sha256"] == hashlib.sha256(
        (run / "last.pt").read_bytes()
    ).hexdigest()


def test_cpu_completion_verifier_handles_explicit_resume_freshness_boundary(
    tmp_path: Path,
) -> None:
    """Exercise the exact post-training path that receives a bound DateTime."""

    seed = 2026082803
    run = _write_completed_fixture(tmp_path, seed, gpu=1)
    earlier = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    completed = _run_pwsh(
        "-VerifySeedDirectory",
        str(run),
        "-VerifySeed",
        str(seed),
        "-ExpectedPhysicalGpu",
        "1",
        "-VerifyReceiptNotBeforeUtc",
        earlier,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "verified_complete"

    future = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
    stale = _run_pwsh(
        "-VerifySeedDirectory",
        str(run),
        "-VerifySeed",
        str(seed),
        "-ExpectedPhysicalGpu",
        "1",
        "-VerifyReceiptNotBeforeUtc",
        future,
    )
    assert stale.returncode != 0
    combined = " ".join((stale.stdout + stale.stderr).split())
    assert "receipt predates this training" in combined
    assert "invocation" in combined


def test_cpu_completion_verifier_fails_closed_on_checkpoint_tamper(
    tmp_path: Path,
) -> None:
    seed = 2026082804
    run = _write_completed_fixture(tmp_path, seed, gpu=0)
    with (run / "last.pt").open("ab") as handle:
        handle.write(b"tampered")
    completed = _run_pwsh(
        "-VerifySeedDirectory",
        str(run),
        "-VerifySeed",
        str(seed),
        "-ExpectedPhysicalGpu",
        "0",
    )
    assert completed.returncode != 0
    assert "checkpoint SHA-256 mismatch" in (completed.stdout + completed.stderr)


def test_cpu_completion_verifier_fails_closed_on_incomplete_horizon(
    tmp_path: Path,
) -> None:
    seed = 2026082805
    run = _write_completed_fixture(tmp_path, seed, gpu=1)
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    (run / "history.json").write_text(
        json.dumps(history[:-1], indent=2), encoding="utf-8"
    )
    completed = _run_pwsh(
        "-VerifySeedDirectory",
        str(run),
        "-VerifySeed",
        str(seed),
        "-ExpectedPhysicalGpu",
        "1",
    )
    assert completed.returncode != 0
    assert "history has 59 rows instead of 60" in (
        completed.stdout + completed.stderr
    )


def test_queue_uses_hidden_direct_logs_exclusive_locks_and_per_seed_preflight() -> None:
    source = QUEUE_SCRIPT.read_text(encoding="utf-8")
    assert "[System.IO.FileShare]::None" in source
    assert "WindowStyle Hidden" in source
    assert "RedirectStandardOutput" in source
    assert "RedirectStandardError" in source
    assert "Invoke-GpuPreflight" in source
    assert "Assert-NoDuplicateSeedProcess" in source
    assert r'train_stageb_train399\.py"?\s+train' in source
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "--batch-size" not in source
    assert "--workers" not in source
    assert "WaitForExit()" in source
    assert "Stop-Process" not in source
    assert ".Kill(" not in source


def _write_attach_fixture(
    tmp_path: Path, *, seed: int, gpu: int, command_line: str | None = None
) -> Path:
    seed_directory = tmp_path / f"seed_{seed}"
    seed_directory.mkdir()
    created = datetime.now(timezone.utc) - timedelta(minutes=1)
    preflight = {
        "captured_utc": (created + timedelta(seconds=15)).isoformat(),
        "cuda_visible_devices": str(gpu),
        "existing_processes_killed_or_suspended": False,
        "internal_device": "cuda:0",
        "schema_version": "PHAxis-nvidia-smi-preflight-1.0",
        "status": "passed",
        "torch_cuda_initialized_before_preflight": False,
    }
    (seed_directory / "nvidia_smi_preflight_resume_001.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    python = PROJECT_ROOT / "envs" / "rhpheno" / "python.exe"
    if command_line is None:
        command_line = (
            f'"{python}" scripts\\phaxis\\train_stageb_train399.py train '
            f"--seed {seed} --device cuda:0 --resume"
        )
    fixture = {
        "seed_directory": str(seed_directory),
        "process": {
            "process_id": 424242,
            "executable_path": str(python),
            "command_line": command_line,
            "creation_utc": created.isoformat(),
        },
    }
    fixture_path = tmp_path / "attach_identity_fixture.json"
    fixture_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    return fixture_path


def test_cpu_attach_identity_binds_exact_command_seed_device_and_gpu_preflight(
    tmp_path: Path,
) -> None:
    seed = 2026082801
    fixture = _write_attach_fixture(tmp_path, seed=seed, gpu=1)
    completed = _run_pwsh(
        "-AuditAttachFixture",
        str(fixture),
        "-AuditAttachSeed",
        str(seed),
        "-AuditAttachPhysicalGpu",
        "1",
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "verified_attach_identity"
    assert payload["process_id"] == 424242
    assert payload["seed"] == seed
    assert payload["physical_gpu"] == 1
    assert payload["internal_device"] == "cuda:0"
    assert payload["trainer_preflight"]["artifact_suffix"] == "_resume_001"
    assert payload["trainer_preflight"]["expected_completion_receipt"].endswith(
        "training_receipt_resume_001.json"
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "--seed 2026082802 --device cuda:0 --resume",
        "--seed 2026082801 --device cuda:1 --resume",
        "--seed 2026082801 --device cuda:0 --resume --workers 0",
    ],
)
def test_cpu_attach_identity_rejects_non_exact_resume_command(
    tmp_path: Path, tamper: str
) -> None:
    seed = 2026082801
    python = PROJECT_ROOT / "envs" / "rhpheno" / "python.exe"
    command = (
        f'"{python}" scripts\\phaxis\\train_stageb_train399.py train {tamper}'
    )
    fixture = _write_attach_fixture(
        tmp_path, seed=seed, gpu=1, command_line=command
    )
    completed = _run_pwsh(
        "-AuditAttachFixture",
        str(fixture),
        "-AuditAttachSeed",
        str(seed),
        "-AuditAttachPhysicalGpu",
        "1",
    )
    assert completed.returncode != 0
    assert "not the exact formal resume command" in (
        completed.stdout + completed.stderr
    )


def test_cpu_attach_identity_rejects_wrong_physical_gpu_receipt(
    tmp_path: Path,
) -> None:
    seed = 2026082801
    fixture = _write_attach_fixture(tmp_path, seed=seed, gpu=0)
    completed = _run_pwsh(
        "-AuditAttachFixture",
        str(fixture),
        "-AuditAttachSeed",
        str(seed),
        "-AuditAttachPhysicalGpu",
        "1",
    )
    assert completed.returncode != 0
    assert "requires exactly one post-creation trainer preflight" in (
        completed.stdout + completed.stderr
    )
