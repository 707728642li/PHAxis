from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from phaxis.evaluation_metrics import precision_recall_f1


def _module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "phaxis"
        / "evaluate_stageb_train399_qcdev44.py"
    )
    spec = importlib.util.spec_from_file_location("stageb_formal_evaluator_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_evaluator_separates_tolerant_presence_from_strict_geometry():
    module = _module()
    prediction = {
        "base": np.asarray([[0.0, 0.5]]),
        "tip": np.asarray([[8.0, 0.5]]),
        "length_um": np.asarray([8.0]),
    }
    truth = {
        "base": np.asarray([[0.0, 0.0]]),
        "tip": np.asarray([[40.0, 0.0]]),
        "polys": [np.asarray([[0.0, 0.0], [40.0, 0.0]])],
        "length_um": np.asarray([40.0]),
    }
    result = module._evaluate_prediction(prediction, truth)
    assert result["biological_presence_tp"][5.0] == 1
    assert result["base_tp"][5.0] == 1
    assert result["strict_tp"][5.0] == 0


def test_formal_pool_derives_primary_presence_from_per_image_counts():
    module = _module()
    expert = {
        "n_pred": 2,
        "n_gt": 1,
        "base_tp": {5.0: 0, 10.0: 0, 20.0: 1},
        "biological_presence_tp": {5.0: 1, 10.0: 1, 20.0: 1},
        "strict_tp": {5.0: 0, 10.0: 0, 20.0: 0},
    }
    pooled = module._pool(
        [{"stageb_train399": expert}],
        "stageb_train399",
        prf=precision_recall_f1,
    )
    assert pooled["tolerant_biological_presence"]["20"]["f1"] == 2 / 3
    assert pooled["identity_attachment_proxy"]["5"]["f1"] == 0.0
    assert pooled["strict_whole_line_correspondence"]["20"]["f1"] == 0.0


def _legacy_hybrid_payload() -> dict:
    return {
        "schema_version": "RHAxis-NextGen-Hybrid-Max-qc-development-evaluation-1.0",
        "task_id": "RHAUD-test",
        "source_image_sha256": "a" * 64,
        "blind_images_used": 0,
        "canonical_annotations_read_during_inference": False,
        "identity_hair_variant": "hybrid_verified_increment",
        "count_hair_variant": "hybrid_verified_increment",
        "identity_hairs": [
            {"source": "frozen_v1", "points_xy": [[1.0, 2.0], [3.0, 4.0]]}
        ],
    }


def test_formal_evaluator_locks_legacy_hybrid_comparator_semantics():
    module = _module()
    payload = _legacy_hybrid_payload()
    module._validate_legacy_hybrid_comparator(
        payload,
        task_id="RHAUD-test",
        expected_image_sha256="a" * 64,
    )
    prediction = module._hybrid_prediction(payload, 2.0)
    np.testing.assert_array_equal(
        prediction["polys"][0], np.asarray([[2.0, 4.0], [6.0, 8.0]])
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "PHAxis-prediction-1.0"),
        ("identity_hair_variant", "rhaxiscc_stage_b_5fold_formal_identity"),
        ("count_hair_variant", "rhaxiscc_stage_b_5fold_formal_count"),
        ("blind_images_used", 1),
    ),
)
def test_formal_evaluator_rejects_nonlegacy_hybrid_comparator(field, value):
    module = _module()
    payload = _legacy_hybrid_payload()
    payload[field] = value
    with pytest.raises(RuntimeError, match="legacy Hybrid comparator"):
        module._validate_legacy_hybrid_comparator(
            payload,
            task_id="RHAUD-test",
            expected_image_sha256="a" * 64,
        )


def test_formal_evaluator_rejects_phaxis_fused_comparator():
    module = _module()
    payload = _legacy_hybrid_payload()
    payload["phaxis"] = {"hair_identity_count_expert": "train399"}
    with pytest.raises(RuntimeError, match="legacy Hybrid comparator"):
        module._validate_legacy_hybrid_comparator(
            payload,
            task_id="RHAUD-test",
            expected_image_sha256="a" * 64,
        )


def test_formal_bootstrap_includes_primary_biological_presence_delta():
    module = _module()

    def expert(n_pred, n_gt, presence_tp, base_tp):
        return {
            "n_pred": n_pred,
            "n_gt": n_gt,
            "base_tp": {5.0: base_tp, 10.0: base_tp, 20.0: base_tp},
            "biological_presence_tp": {
                5.0: presence_tp,
                10.0: presence_tp,
                20.0: presence_tp,
            },
            "strict_tp": {5.0: 0, 10.0: 0, 20.0: 0},
        }

    rows = [
        {
            "stageb_train399": expert(2, 2, 2, 1),
            "hybrid_max": expert(1, 2, 1, 1),
        },
        {
            "stageb_train399": expert(3, 2, 2, 2),
            "hybrid_max": expert(2, 2, 1, 1),
        },
        {
            "stageb_train399": expert(4, 3, 3, 2),
            "hybrid_max": expert(2, 3, 1, 1),
        },
    ]
    result = module._bootstrap(
        rows,
        prf=precision_recall_f1,
        repetitions=20,
        seed=7,
    )
    interval = result["delta_stageb_train399_minus_hybrid"][
        "biological_presence_f1_20um"
    ]
    assert interval["lower_2_5"] > 0.0
    assert interval["upper_97_5"] > 0.0
