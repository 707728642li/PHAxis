from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "phaxis"
    / "compare_hair_experts_biological_presence.py"
)
SPEC = importlib.util.spec_from_file_location(
    "historical_biological_presence_comparison", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_legacy_oof_and_source_curves_share_physical_um_coordinates() -> None:
    record = {
        "thresh": 0.5,
        "pred": {
            "score": np.asarray([0.9, 0.1]),
            "base": np.asarray([[1.0, 2.0], [50.0, 50.0]]),
            "tip": np.asarray([[3.0, 4.0], [60.0, 60.0]]),
        },
    }
    stageb = MODULE._straight_stageb_polylines(record)
    hybrid = MODULE._hybrid_polylines(
        {
            "task_id": "synthetic",
            "blind_images_used": 0,
            "identity_hairs": [
                {"points_xy": [[4.0, 8.0], [12.0, 16.0]]}
            ],
        },
        0.5,
    )

    expected_um = np.asarray([[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_array_equal(stageb[0], expected_um)
    np.testing.assert_array_equal(hybrid[0], expected_um)
    assert MODULE._presence(stageb, hybrid)["5"] == 1


def test_hybrid_input_rejects_blind_usage() -> None:
    payload = {
        "task_id": "synthetic",
        "blind_images_used": 1,
        "identity_hairs": [],
    }
    try:
        MODULE._hybrid_polylines(payload, 1.0)
    except RuntimeError as error:
        assert "blind_images_used" in str(error)
    else:
        raise AssertionError("blind-marked predictions must be rejected")
