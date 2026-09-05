from __future__ import annotations

import numpy as np
import pytest

from phaxis.hair_stageb.candidate_pool_contract import (
    biological_presence_candidate_decoder_contract,
    locked_biological_presence_candidate_decoder_contract,
)
from phaxis.hair_stageb.decode import decode_biological_presence_candidates
from phaxis.io import sha256_json


def _heads() -> dict[str, np.ndarray]:
    height = width = 16
    channels = {
        "base_hm": 1,
        "base_off": 2,
        "base_dir": 2,
        "base_len": 1,
        "tip_hm": 1,
        "tip_off": 2,
    }
    heads = {
        name: np.zeros((count, height, width), dtype=np.float32)
        for name, count in channels.items()
    }
    heads["base_hm"][:] = -20.0
    heads["tip_hm"][:] = -20.0
    heads["base_hm"][0, 5, 2] = 10.0
    heads["base_dir"][0, 5, 2] = 1.0
    heads["base_len"][0, 5, 2] = np.log(0.04)
    heads["tip_hm"][0, 5, 10] = np.log(0.2 / 0.8)
    return heads


def _decode(*, tip_score_floor: float, tip_snap_radius_um: float = 30.0) -> dict:
    return decode_biological_presence_candidates(
        _heads(),
        um_per_px=2.0,
        out_stride=2,
        score_floor=0.10,
        nms_kernel=5,
        max_instances=4000,
        root_gate_um=(-90.0, 25.0),
        tip_score_floor=tip_score_floor,
        tip_snap_radius_um=tip_snap_radius_um,
    )


def test_tip_score_parameter_changes_geometry_and_is_declared_in_contract() -> None:
    locked = _decode(tip_score_floor=0.15)
    drifted = _decode(tip_score_floor=0.25)
    assert not np.array_equal(locked["tip"], drifted["tip"])
    assert (
        locked["candidate_decoder_contract"]
        == locked_biological_presence_candidate_decoder_contract()
    )
    assert drifted["candidate_decoder_contract"]["tip_score_floor"] == 0.25
    assert (
        drifted["candidate_decoder_contract"]
        != locked["candidate_decoder_contract"]
    )
    assert sha256_json(drifted["candidate_decoder_contract"]) != sha256_json(
        locked["candidate_decoder_contract"]
    )


def test_tip_snap_radius_changes_geometry_and_contract_identity() -> None:
    locked = _decode(tip_score_floor=0.15, tip_snap_radius_um=30.0)
    drifted = _decode(tip_score_floor=0.15, tip_snap_radius_um=20.0)
    assert not np.array_equal(locked["tip"], drifted["tip"])
    assert drifted["candidate_decoder_contract"]["tip_snap_radius_um"] == 20.0
    assert sha256_json(drifted["candidate_decoder_contract"]) != sha256_json(
        locked["candidate_decoder_contract"]
    )


@pytest.mark.parametrize("field", ["out_stride", "nms_kernel", "max_instances"])
def test_contract_fails_closed_instead_of_truncating_fractional_integer_fields(
    field: str,
) -> None:
    values = {
        "working_um_per_px": 2.0,
        "out_stride": 2,
        "nms_kernel": 5,
        "max_instances": 4000,
        "root_gate_um": (-90.0, 25.0),
        "tip_score_floor": 0.15,
        "tip_snap_radius_um": 30.0,
    }
    values[field] = float(values[field]) + 0.25
    with pytest.raises(ValueError, match="must be an integer"):
        biological_presence_candidate_decoder_contract(**values)


def test_threshold_subsets_do_not_mutate_candidate_geometry() -> None:
    pool = _decode(tip_score_floor=0.15)
    original_base = pool["base"].copy()
    original_tip = pool["tip"].copy()
    original_score = pool["score"].copy()
    for threshold in (0.10, 0.225, 0.325):
        keep = original_score >= threshold
        assert np.array_equal(pool["base"][keep], original_base[keep])
        assert np.array_equal(pool["tip"][keep], original_tip[keep])
    assert np.array_equal(pool["base"], original_base)
    assert np.array_equal(pool["tip"], original_tip)
