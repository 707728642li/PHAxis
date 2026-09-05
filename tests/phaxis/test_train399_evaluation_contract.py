from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "phaxis" / "evaluate_stageb_train399_qcdev44.py"
SPEC = importlib.util.spec_from_file_location("evaluate_stageb_train399_qcdev44", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _prf(tp: int, predicted: int, truth: int) -> dict[str, float | int]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / truth if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "n_pred": predicted,
        "n_gt": truth,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _metadata() -> dict[str, str]:
    return {
        "image_width": "101",
        "image_height": "103",
        "source_um_per_px": "2.3",
    }


def _payload(policy: str = "five_seed_train399_last_epoch_60", training_images: int = 399):
    scale_xy = [116 / 101, 118 / 103]
    return {
        "task_id": "RHAUD-test",
        "blind_images_used": 0,
        "model": {
            "checkpoint_policy": policy,
            "training_images": training_images,
        },
        "coordinate_space": {
            "working_um_per_px": 2.0,
            "source_um_per_px": 2.3,
            "working_shape": [118, 116],
            "source_to_working_scale_xy": scale_xy,
        },
        "n": 1,
        "detections": [
            {
                "base_xy_working": [1.0, 2.0],
                "tip_xy_working": [4.0, 6.0],
                "score": 0.8,
                "predicted_length_um": 10.0,
            }
        ],
    }


def test_train399_detection_contract_accepts_only_formal_policy():
    prediction = MODULE._stageb_prediction(_payload(), _metadata())
    scale_xy = np.asarray([116 / 101, 118 / 103])
    np.testing.assert_allclose(
        prediction["base"], np.asarray([[1.0, 2.0]]) / scale_xy * 2.3
    )
    np.testing.assert_allclose(
        prediction["tip"], np.asarray([[4.0, 6.0]]) / scale_xy * 2.3
    )

    with pytest.raises(RuntimeError, match="not a strict train399-only"):
        MODULE._stageb_prediction(
            _payload(policy="five_fold_last_epoch_60"), _metadata()
        )
    with pytest.raises(RuntimeError, match="training_images is not 399"):
        MODULE._stageb_prediction(_payload(training_images=443), _metadata())

    bad_scale = _payload()
    bad_scale["coordinate_space"]["source_to_working_scale_xy"][0] += 1e-3
    with pytest.raises(RuntimeError, match="realized source/working scale mismatch"):
        MODULE._stageb_prediction(bad_scale, _metadata())

    bad_physical_scale = _payload()
    bad_physical_scale["coordinate_space"]["source_um_per_px"] = 1.0
    with pytest.raises(RuntimeError, match="physical scale differs"):
        MODULE._stageb_prediction(bad_physical_scale, _metadata())


def test_pooled_identity_and_bootstrap_are_deterministic():
    def expert(n_pred: int, n_gt: int, tp: int):
        return {
            "n_pred": n_pred,
            "n_gt": n_gt,
            "base_tp": {5.0: tp, 10.0: tp, 20.0: tp},
            "biological_presence_tp": {5.0: tp, 10.0: tp, 20.0: tp},
            "strict_tp": {5.0: tp, 10.0: tp, 20.0: tp},
        }

    rows = [
        {"stageb_train399": expert(3, 2, 2), "hybrid_max": expert(1, 2, 1)},
        {"stageb_train399": expert(4, 5, 4), "hybrid_max": expert(3, 5, 3)},
        {"stageb_train399": expert(2, 2, 2), "hybrid_max": expert(2, 2, 2)},
    ]
    pooled = MODULE._pool(rows, "stageb_train399", prf=_prf)
    assert pooled["predicted_hairs"] == 9
    assert pooled["ground_truth_hairs"] == 9
    assert pooled["tolerant_biological_presence"]["20"]["f1"] == pytest.approx(
        8 / 9
    )
    assert pooled["identity_attachment_proxy"]["20"]["f1"] == pytest.approx(8 / 9)
    assert pooled["count"]["mae"] == pytest.approx(2 / 3)

    first = MODULE._bootstrap(rows, prf=_prf, repetitions=200, seed=7)
    second = MODULE._bootstrap(rows, prf=_prf, repetitions=200, seed=7)
    assert first == second


def test_blind_and_count_mismatches_fail_closed():
    payload = _payload()
    payload["blind_images_used"] = 1
    with pytest.raises(RuntimeError, match="blind_images_used"):
        MODULE._stageb_prediction(payload, _metadata())

    payload = _payload()
    payload["n"] = 2
    with pytest.raises(RuntimeError, match="detection count mismatch"):
        MODULE._stageb_prediction(payload, _metadata())
