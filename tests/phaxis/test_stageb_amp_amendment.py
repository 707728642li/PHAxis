from __future__ import annotations

import json
from pathlib import Path

import torch

from phaxis.io import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = (
    PROJECT_ROOT / "models" / "phaxis_stageb_train399_v1_0_20260828"
)
AMENDMENT = MODEL_ROOT / "AMP_BACKWARD_RETRY_AMENDMENT_20260829.json"


def test_amp_amendment_binds_failure_source_restart_and_legacy_zero_retry() -> None:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    assert payload["schema_version"] == (
        "PHAxis-StageB-train399-AMP-backward-amendment-1.0"
    )
    assert payload["status"] == (
        "applied_before_authoritative_seed3_optimizer_trajectory"
    )
    assert payload["unchanged_scientific_contract"]["blind_images_used"] == 0
    assert payload["unchanged_scientific_contract"][
        "validation_used_for_gradient_early_stopping_or_retry"
    ] is False

    failed = MODEL_ROOT / payload["superseded_failed_attempt"]["failure_receipt"]
    assert failed.is_file()
    assert sha256_file(failed) == payload["superseded_failed_attempt"][
        "failure_receipt_sha256"
    ]
    assert not (failed.parent / "last.pt").exists()

    source = (MODEL_ROOT / payload["implementation"]["training_source"]).resolve()
    assert source == (PROJECT_ROOT / "src/phaxis/hair_stageb/training.py").resolve()
    assert sha256_file(source) == payload["implementation"][
        "training_source_sha256"
    ]

    archived_initialization = failed.parent / "initialization.json"
    restarted_initialization = MODEL_ROOT / "seed_2026082803" / "initialization.json"
    assert sha256_file(archived_initialization) == sha256_file(
        restarted_initialization
    )
    restarted = json.loads(restarted_initialization.read_text(encoding="utf-8"))
    assert restarted["initialization_sha256"] == payload[
        "authoritative_seed3_restart"
    ]["initialization_identity_sha256"]
    assert restarted["initial_complete_model_state_sha256"] == payload[
        "authoritative_seed3_restart"
    ]["initial_complete_model_state_sha256"]

    normalized = payload["legacy_zero_retry_normalization"]
    assert normalized["normalized_amp_backward_retry_count"] == [0, 0]
    for seed in normalized["seeds"]:
        checkpoint = torch.load(
            MODEL_ROOT / f"seed_{seed}" / "last.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert checkpoint["epoch"] == 60
        assert checkpoint["global_step"] == 23_940
        assert checkpoint["scaler"]["scale"] == 1024.0
        assert checkpoint["scaler"]["_growth_tracker"] == 23_940
        assert "amp_backward_retry_events" not in checkpoint
