from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

# The core wheel intentionally does not force a CUDA/PyTorch installation.
# These decoder checks run when the inference dependencies are available and
# skip cleanly in the lightweight packaging job.
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from phaxis.constants import (
    LEGACY_HAIR_EXPERT_ID,
    HAIR_SCORE_THRESHOLD,
    STAGEB_SCHEMA,
)
from phaxis.hair_stageb.decode import (
    decode_biological_presence_candidates,
    decode_instances,
)
from phaxis.hair_stageb.preprocess import (
    make_input_channels,
    resample_to_physical_scale,
    to_gray,
)
from phaxis.hair_stageb.serialization import make_detection_payload
import phaxis.hair_stageb.runtime as stageb_runtime
from phaxis.hair_stageb.runtime import StageBEnsemble, _tile_origins
from phaxis.io import sha256_json


def _empty_heads(height: int = 16, width: int = 16):
    return {
        "base_hm": np.full((1, height, width), -20.0, np.float32),
        "base_off": np.zeros((2, height, width), np.float32),
        "base_dir": np.zeros((2, height, width), np.float32),
        "base_len": np.zeros((1, height, width), np.float32),
        "tip_hm": np.full((1, height, width), -20.0, np.float32),
        "tip_off": np.zeros((2, height, width), np.float32),
    }


def test_decode_single_base_uses_offset_direction_and_physical_length():
    heads = _empty_heads()
    y, x = 4, 5
    heads["base_hm"][0, y, x] = 10.0
    heads["base_off"][:, y, x] = [0.25, 0.5]
    heads["base_dir"][:, y, x] = [1.0, 0.0]
    heads["base_len"][0, y, x] = np.log(0.2)
    decoded = decode_instances(
        heads,
        um_per_px=2.0,
        out_stride=2,
        score_threshold=HAIR_SCORE_THRESHOLD,
        nms_kernel=5,
        max_instances=100,
        root_gate_um=None,
    )
    assert decoded["n"] == 1
    np.testing.assert_allclose(decoded["base"][0], [10.5, 9.0], atol=1e-5)
    np.testing.assert_allclose(decoded["tip"][0], [20.5, 9.0], atol=1e-5)
    assert decoded["length_um"][0] == pytest.approx(20.0, rel=1e-5)


def test_decode_empty_heatmap_fails_closed_to_no_instances():
    decoded = decode_instances(
        _empty_heads(),
        um_per_px=2.0,
        out_stride=2,
        score_threshold=HAIR_SCORE_THRESHOLD,
        nms_kernel=5,
        max_instances=100,
        root_gate_um=None,
    )
    assert decoded["n"] == 0
    assert decoded["base"].shape == (0, 2)
    assert decoded["tip"].shape == (0, 2)


def test_candidate_pool_decoder_emits_presence_proxy_at_fixed_floor():
    heads = _empty_heads(height=8, width=8)
    heads["base_hm"][0, 3, 4] = 10.0
    heads["base_off"][:, 3, 4] = [0.25, 0.5]
    heads["base_dir"][:, 3, 4] = [1.0, 0.0]
    heads["base_len"][0, 3, 4] = np.log(0.2)
    decoded = decode_biological_presence_candidates(
        heads,
        um_per_px=2.0,
        out_stride=2,
        score_floor=0.10,
        nms_kernel=5,
        max_instances=4000,
        root_gate_um=(-90.0, 25.0),
    )
    assert decoded["n"] == 1
    np.testing.assert_allclose(decoded["base"][0], [8.5, 7.0], atol=1e-5)
    np.testing.assert_allclose(decoded["tip"][0], [18.5, 7.0], atol=1e-5)
    assert bool(decoded["presence_proxy_valid"][0]) is True
    assert decoded["score"][0] > 0.99
    assert decoded["score_floor"] == 0.10
    assert decoded["network_forward_passes"] == 1
    assert decoded["candidate_pool_decode_scope"] == (
        "base_score_plus_straight_base_to_tip_biological_presence_proxy"
    )
    assert decoded["presence_proxy_kind"] == "straight_base_to_tip"
    assert decoded["distal_endpoint_or_length_used_as_selection_gate"] is False
    assert not {"length_um", "tip_snapped"} & set(decoded)


def test_tip_snap_preserves_regressed_polyline_arc_length_semantics():
    heads = _empty_heads(height=32, width=40)
    base_y, base_x = 4, 5
    heads["base_hm"][0, base_y, base_x] = 10.0
    heads["base_dir"][:, base_y, base_x] = [1.0, 0.0]
    heads["base_len"][0, base_y, base_x] = np.log(0.8)  # 80-um arc length
    # The nearest valid endpoint is a 72-um chord from the base.  A curved
    # annotated centerline can legitimately have an 80-um arc length.
    heads["tip_hm"][0, base_y, 23] = 10.0
    decoded = decode_instances(
        heads,
        um_per_px=2.0,
        out_stride=2,
        score_threshold=0.225,
        nms_kernel=5,
        max_instances=100,
        root_gate_um=None,
    )
    assert decoded["n"] == 1
    assert bool(decoded["tip_snapped"][0]) is True
    chord_um = (
        np.linalg.norm(decoded["tip"][0] - decoded["base"][0]) * 2.0
    )
    assert chord_um == pytest.approx(72.0)
    assert decoded["length_um"][0] == pytest.approx(80.0, rel=1e-5)
    assert decoded["length_um"][0] != pytest.approx(chord_um)
    assert decoded["length_semantics"] == (
        "regressed_polyline_arc_length_um_diagnostic_only"
    )


def test_tip_proxy_is_immutable_across_base_threshold_sweep():
    heads = _empty_heads(height=32, width=40)
    y, x = 4, 5
    heads["base_hm"][0, y, x] = 10.0
    heads["base_dir"][:, y, x] = [1.0, 0.0]
    heads["base_len"][0, y, x] = np.log(0.8)
    # This tip is above the fixed 0.15 tip floor but below 0.325 / 2.
    tip_probability = 0.155
    heads["tip_hm"][0, y, 23] = np.log(
        tip_probability / (1.0 - tip_probability)
    )
    low = decode_instances(
        heads,
        um_per_px=2.0,
        out_stride=2,
        score_threshold=0.10,
        nms_kernel=5,
        max_instances=100,
        root_gate_um=None,
    )
    high = decode_instances(
        heads,
        um_per_px=2.0,
        out_stride=2,
        score_threshold=0.325,
        nms_kernel=5,
        max_instances=100,
        root_gate_um=None,
    )
    np.testing.assert_array_equal(low["base"], high["base"])
    np.testing.assert_array_equal(low["tip"], high["tip"])
    np.testing.assert_array_equal(low["tip_snapped"], high["tip_snapped"])
    assert bool(low["tip_snapped"][0]) is True


def test_detection_payload_is_canonically_self_identifying():
    prediction = {
        "base": np.asarray([[1.0, 2.0]], np.float32),
        "tip": np.asarray([[3.0, 4.0]], np.float32),
        "score": np.asarray([0.75], np.float32),
        "length_um": np.asarray([12.5], np.float32),
        "working_shape": [32, 64],
        "source_to_working_scale": 0.5,
        "tip_snapped": np.asarray([True]),
        "length_semantics": "regressed_polyline_arc_length_um_diagnostic_only",
    }
    payload = make_detection_payload(
        task_id="T1",
        source_image_sha256="a" * 64,
        source_um_per_px=1.0,
        prediction=prediction,
        precision_mode="fp32",
    )
    unsigned = deepcopy(payload)
    identity = unsigned.pop("detection_identity_sha256")
    assert identity == sha256_json(unsigned)
    assert payload["schema_version"] == STAGEB_SCHEMA
    assert payload["model"]["expert_id"] == LEGACY_HAIR_EXPERT_ID
    assert payload["operating_point"]["score_threshold"] == HAIR_SCORE_THRESHOLD
    assert payload["n"] == len(payload["detections"]) == 1
    assert payload["detections"][0]["tip_snapped"] is True
    assert payload["detections"][0]["predicted_length_semantics"] == (
        "regressed_polyline_arc_length_um_diagnostic_only"
    )
    assert payload["blind_images_used"] == 0


def test_preprocess_is_deterministic_and_physical_scale_aware():
    rgb = np.zeros((16, 20, 3), np.uint8)
    rgb[..., 0] = np.arange(20, dtype=np.uint8)[None]
    gray = to_gray(rgb)
    resized, scale = resample_to_physical_scale(gray, 1.0, 2.0)
    first = make_input_channels(resized, 2.0, 3)
    second = make_input_channels(resized, 2.0, 3)
    assert gray.shape == (16, 20)
    assert resized.shape == (8, 10)
    assert scale == pytest.approx(0.5)
    assert first.shape == (3, 8, 10)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_preprocess_rejects_non_image_tensor():
    with pytest.raises(ValueError, match="expected a 2-D image"):
        to_gray(np.zeros((2, 3, 4, 5), np.uint8))


def test_odd_height_width_tile_origins_are_stride_aligned_and_cover_oracle():
    height, width = 1793, 2561
    stride, window, overlap = 2, 1024, 256
    ys = _tile_origins(
        height, window=window, overlap=overlap, out_stride=stride
    )
    xs = _tile_origins(width, window=window, overlap=overlap, out_stride=stride)
    assert max(0, height - window) % stride == 1  # reproduces the old bug
    assert max(0, width - window) % stride == 1
    assert ys[-1] == ((height - window) // stride) * stride == 768
    assert xs[-1] == ((width - window) // stride) * stride == 1536
    assert all(origin % stride == 0 for origin in (*ys, *xs))

    # Independent 2-D oracle: every floor(H/s) x floor(W/s) output cell must
    # be covered by at least one tile after aligning both axes.
    covered = np.zeros((height // stride, width // stride), dtype=bool)
    for y in ys:
        for x in xs:
            y0, x0 = y // stride, x // stride
            y1 = min(covered.shape[0], (y + window) // stride)
            x1 = min(covered.shape[1], (x + window) // stride)
            covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_runtime_records_realized_xy_scale_after_noninteger_resize(monkeypatch):
    ensemble = object.__new__(StageBEnsemble)
    ensemble.device = "cpu"
    ensemble.use_amp = False
    ensemble.models = [object()]
    ensemble.config = {"in_channels": 3}
    ensemble.score_threshold = 0.225
    monkeypatch.setattr(
        stageb_runtime,
        "_predict_image",
        lambda *_args, **_kwargs: {
            "base_hm": np.zeros((1, 1, 1), dtype=np.float32)
        },
    )
    monkeypatch.setattr(
        stageb_runtime,
        "decode_instances",
        lambda *_args, **_kwargs: {
            "base": np.zeros((0, 2), np.float32),
            "tip": np.zeros((0, 2), np.float32),
            "score": np.zeros(0, np.float32),
            "length_um": np.zeros(0, np.float32),
            "tip_snapped": np.zeros(0, dtype=bool),
            "length_semantics": "regressed_polyline_arc_length_um_diagnostic_only",
            "n": 0,
        },
    )
    result = ensemble.predict(np.zeros((7, 11), np.uint8), source_um_per_px=1.3)
    assert result["source_shape"] == [7, 11]
    assert result["working_shape"] == [5, 7]
    np.testing.assert_allclose(
        result["source_to_working_scale_xy"], [7 / 11, 5 / 7], rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        result["realized_um_per_px_xy"],
        [1.3 / (7 / 11), 1.3 / (5 / 7)],
        rtol=0,
        atol=1e-15,
    )


def test_shared_input_path_is_exact_for_odd_overlapping_tta_tiles(monkeypatch):
    monkeypatch.setattr(stageb_runtime, "HAIR_WINDOW", 8)
    monkeypatch.setattr(stageb_runtime, "HAIR_OVERLAP", 4)
    monkeypatch.setattr(stageb_runtime, "HAIR_BATCH", 2)

    class FakeModel:
        def __init__(self, member_index: int, calls: list[int]):
            self.member_index = member_index
            self.calls = calls

        def eval(self):
            return self

        def __call__(self, tensor):
            self.calls.append(self.member_index)
            scalar = tensor[:, :1, ::2, ::2]
            outputs = {}
            for head_index, (name, channels) in enumerate(
                stageb_runtime.HEADS.items()
            ):
                if name in ("base_dir", "flow"):
                    x = torch.full_like(
                        scalar, 0.5 + self.member_index / 16.0
                    )
                    y = torch.full_like(
                        scalar, -0.25 + self.member_index / 32.0
                    )
                    outputs[name] = torch.cat((x, y), dim=1)
                elif name in ("base_off", "tip_off"):
                    x = torch.full_like(scalar, 0.25)
                    y = torch.full_like(
                        scalar, 0.125 + self.member_index / 32.0
                    )
                    outputs[name] = torch.cat((x, y), dim=1)
                else:
                    outputs[name] = (
                        scalar * 0.125
                        + 1.5
                        + self.member_index / 16.0
                        + head_index / 32.0
                    )
                    assert channels == 1
            return outputs

    def ensemble(shared: bool, calls: list[int]) -> StageBEnsemble:
        result = object.__new__(StageBEnsemble)
        result.device = "cpu"
        result.use_amp = False
        result.models = [FakeModel(index, calls) for index in range(5)]
        result.config = {"in_channels": 3}
        result.score_threshold = 0.225
        result.shared_input_acceleration = shared
        result.shared_input_max_host_bytes = 1024**3
        result.last_shared_input_audit = None
        return result

    counts = {"channels": 0, "from_numpy": 0}
    original_make_input_channels = stageb_runtime.make_input_channels
    original_from_numpy = torch.from_numpy

    def counted_make_input_channels(*args, **kwargs):
        counts["channels"] += 1
        return original_make_input_channels(*args, **kwargs)

    def counted_from_numpy(array):
        counts["from_numpy"] += 1
        return original_from_numpy(array)

    monkeypatch.setattr(
        stageb_runtime, "make_input_channels", counted_make_input_channels
    )
    monkeypatch.setattr(torch, "from_numpy", counted_from_numpy)
    image = (np.arange(13 * 15).reshape(13, 15) % 251).astype(np.uint8)

    legacy_calls: list[int] = []
    legacy_heads, legacy_geometry = ensemble(False, legacy_calls)._predict_heads_and_geometry(
        image, source_um_per_px=2.3
    )
    legacy_counts = dict(counts)
    counts.update(channels=0, from_numpy=0)

    shared_calls: list[int] = []
    shared_ensemble = ensemble(True, shared_calls)
    shared_heads, shared_geometry = shared_ensemble._predict_heads_and_geometry(
        image, source_um_per_px=2.3
    )
    shared_counts = dict(counts)
    monkeypatch.setattr(torch, "from_numpy", original_from_numpy)

    tile_count = 9  # 3 odd-height origins x 3 odd-width origins
    batches = 5
    assert legacy_counts == {"channels": 5 * tile_count, "from_numpy": 5 * batches}
    assert shared_counts == {"channels": tile_count, "from_numpy": batches}
    assert legacy_calls == [
        member
        for member in range(5)
        for _call in range(2 * batches)
    ]
    assert shared_calls == [
        member
        for _batch in range(batches)
        for member in range(5)
        for _original_then_flip in range(2)
    ]
    assert legacy_geometry == shared_geometry
    assert shared_geometry["working_shape"] == [15, 17]
    np.testing.assert_allclose(
        shared_geometry["source_to_working_scale_xy"],
        [17 / 15, 15 / 13],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        shared_geometry["realized_um_per_px_xy"],
        [2.3 / (17 / 15), 2.3 / (15 / 13)],
        rtol=0,
        atol=1e-15,
    )
    max_abs_difference = 0.0
    for name in stageb_runtime.HEADS:
        np.testing.assert_array_equal(shared_heads[name], legacy_heads[name])
        max_abs_difference = max(
            max_abs_difference,
            float(np.max(np.abs(shared_heads[name] - legacy_heads[name]))),
        )
    assert max_abs_difference == 0.0

    # Independent invariants for horizontal TTA: x direction changes sign and
    # x offset changes coordinate origin under mirroring.
    np.testing.assert_array_equal(
        shared_heads["base_dir"][0], np.zeros_like(shared_heads["base_dir"][0])
    )
    np.testing.assert_array_equal(
        shared_heads["flow"][0], np.zeros_like(shared_heads["flow"][0])
    )
    np.testing.assert_array_equal(
        shared_heads["base_off"][0],
        np.full_like(shared_heads["base_off"][0], 0.5),
    )
    np.testing.assert_array_equal(
        shared_heads["tip_off"][0],
        np.full_like(shared_heads["tip_off"][0], 0.5),
    )
    assert shared_ensemble.last_shared_input_audit["used"] is True
    assert shared_ensemble.last_shared_input_audit["fallback_reason"] == "none"

    decode_kwargs = {
        "um_per_px": 2.0,
        "out_stride": 2,
        "score_threshold": 0.225,
        "nms_kernel": 3,
        "max_instances": 4000,
        "root_gate_um": None,
    }
    legacy_decoded = decode_instances(legacy_heads, **decode_kwargs)
    shared_decoded = decode_instances(shared_heads, **decode_kwargs)
    assert shared_decoded["n"] > 0
    assert shared_decoded.keys() == legacy_decoded.keys()
    for name, legacy_value in legacy_decoded.items():
        shared_value = shared_decoded[name]
        if isinstance(legacy_value, np.ndarray):
            np.testing.assert_array_equal(shared_value, legacy_value)
        else:
            assert shared_value == legacy_value


def test_shared_input_memory_estimate_gates_180_megapixel_accumulators():
    estimate = stageb_runtime._shared_input_memory_estimate(
        12_000,
        15_000,
        in_channels=3,
        out_stride=2,
        ensemble_members=5,
    )
    assert estimate["output_cells"] == 45_000_000
    assert estimate["head_channels"] == 13
    assert estimate["member_accumulators_bytes"] == 11_700_000_000
    assert estimate["device_full_image_accumulators_bytes"] == 0
    assert estimate["estimated_peak_device_array_bytes"] == 184_549_376
    assert (
        estimate["estimated_peak_device_array_bytes"]
        < stageb_runtime._DEFAULT_SHARED_INPUT_MAX_DEVICE_BYTES
    )
    assert (
        estimate["estimated_peak_host_array_bytes"]
        > estimate["member_accumulators_bytes"]
        > stageb_runtime._DEFAULT_SHARED_INPUT_MAX_HOST_BYTES
    )


def test_shared_input_request_falls_back_before_over_limit_allocation(monkeypatch):
    monkeypatch.setattr(stageb_runtime, "HAIR_WINDOW", 8)
    monkeypatch.setattr(stageb_runtime, "HAIR_OVERLAP", 4)
    monkeypatch.setattr(stageb_runtime, "HAIR_BATCH", 2)
    accelerated_calls = 0

    def forbidden_accelerated(*_args, **_kwargs):
        nonlocal accelerated_calls
        accelerated_calls += 1
        raise AssertionError("over-limit accelerated allocation was attempted")

    def fake_legacy(_model, gray, *, out_stride, **_kwargs):
        output_height = gray.shape[0] // out_stride
        output_width = gray.shape[1] // out_stride
        return {
            name: np.zeros((channels, output_height, output_width), np.float32)
            for name, channels in stageb_runtime.HEADS.items()
        }

    monkeypatch.setattr(
        stageb_runtime,
        "_predict_ensemble_image_shared_input",
        forbidden_accelerated,
    )
    monkeypatch.setattr(stageb_runtime, "_predict_image", fake_legacy)
    ensemble = object.__new__(StageBEnsemble)
    ensemble.device = "cpu"
    ensemble.use_amp = False
    ensemble.models = [object() for _index in range(5)]
    ensemble.config = {"in_channels": 3}
    ensemble.shared_input_acceleration = True
    ensemble.shared_input_max_host_bytes = 1
    heads, _geometry = ensemble._predict_heads_and_geometry(
        np.zeros((13, 15), np.uint8), source_um_per_px=2.0
    )
    assert set(heads) == set(stageb_runtime.HEADS)
    assert accelerated_calls == 0
    assert ensemble.last_shared_input_audit["requested"] is True
    assert ensemble.last_shared_input_audit["used"] is False
    assert ensemble.last_shared_input_audit["runtime_path"] == "legacy"
    assert (
        ensemble.last_shared_input_audit["fallback_reason"]
        == "estimated_peak_exceeds_limit"
    )
    assert (
        ensemble.last_shared_input_audit["memory_estimate"][
            "estimated_peak_array_bytes"
        ]
        > ensemble.last_shared_input_audit["max_host_bytes"]
    )


def test_shared_input_request_falls_back_on_device_staging_limit(monkeypatch):
    monkeypatch.setattr(stageb_runtime, "HAIR_WINDOW", 8)
    monkeypatch.setattr(stageb_runtime, "HAIR_OVERLAP", 4)
    monkeypatch.setattr(stageb_runtime, "HAIR_BATCH", 2)

    def forbidden_accelerated(*_args, **_kwargs):
        raise AssertionError("over-limit device staging was attempted")

    def fake_legacy(_model, gray, *, out_stride, **_kwargs):
        output_height = gray.shape[0] // out_stride
        output_width = gray.shape[1] // out_stride
        return {
            name: np.zeros((channels, output_height, output_width), np.float32)
            for name, channels in stageb_runtime.HEADS.items()
        }

    monkeypatch.setattr(
        stageb_runtime,
        "_predict_ensemble_image_shared_input",
        forbidden_accelerated,
    )
    monkeypatch.setattr(stageb_runtime, "_predict_image", fake_legacy)
    ensemble = object.__new__(StageBEnsemble)
    ensemble.device = "cpu"
    ensemble.use_amp = False
    ensemble.models = [object() for _index in range(5)]
    ensemble.config = {"in_channels": 3}
    ensemble.shared_input_acceleration = True
    ensemble.shared_input_max_host_bytes = 1024**3
    ensemble.shared_input_max_device_bytes = 1
    ensemble.shared_input_device_reserve_bytes = 0
    ensemble._predict_heads_and_geometry(
        np.zeros((13, 15), np.uint8), source_um_per_px=2.0
    )
    audit = ensemble.last_shared_input_audit
    assert audit["used"] is False
    assert audit["runtime_path"] == "legacy"
    assert audit["fallback_reason"] == "device_estimated_peak_exceeds_limit"
    assert (
        audit["memory_estimate"]["estimated_peak_device_array_bytes"]
        > audit["max_device_bytes"]
    )
