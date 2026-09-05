"""Whole-image, five-fold Stage B ensemble inference."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..constants import (
    HAIR_BATCH,
    HAIR_CHECKPOINT_SHA256,
    HAIR_MAX_INSTANCES,
    HAIR_NMS_KERNEL,
    HAIR_OUT_STRIDE,
    HAIR_OVERLAP,
    HAIR_ROOT_GATE_UM,
    HAIR_SCORE_THRESHOLD,
    HAIR_WINDOW,
    HAIR_WORKING_UM_PER_PX,
)
from ..io import read_json, sha256_file
from .candidate_bundle import (
    CANDIDATE_POOL_SCORE_FLOOR,
    detection_model_metadata_from_candidate_manifest,
    read_candidate_manifest,
    validate_candidate_manifest,
    validate_train399_detection_model_metadata,
)
from .decode import decode_biological_presence_candidates, decode_instances
from .selection import (
    read_selection_receipt,
    validate_selected_operating_point_binding,
)
from .model import HEADS, MultiHeadUNet
from .preprocess import make_input_channels, resample_to_physical_scale, to_gray


_DEFAULT_SHARED_INPUT_MAX_HOST_BYTES = 2 * 1024**3
_DEFAULT_SHARED_INPUT_MAX_DEVICE_BYTES = 1 * 1024**3
_DEFAULT_SHARED_INPUT_DEVICE_RESERVE_BYTES = 2 * 1024**3


def _window_weight(size: int, taper: float = 0.25, floor: float = 0.05) -> np.ndarray:
    length = int(size * taper)
    weights = np.ones(size, dtype=np.float32)
    if length > 1:
        ramp = 0.5 * (
            1 - np.cos(np.linspace(0, np.pi, length, dtype=np.float32))
        )
        weights[:length] = ramp
        weights[-length:] = ramp[::-1]
    weights = np.maximum(weights, floor)
    return np.outer(weights, weights).astype(np.float32)


def _tile_origins(
    length: int, *, window: int, overlap: int, out_stride: int
) -> list[int]:
    """Return stride-aligned origins covering the complete valid output domain.

    For an odd input extent and an even output stride, the final input row/column
    has no corresponding ``floor(length / stride)`` output cell.  Aligning the
    final window down therefore preserves every valid output while preventing a
    one-working-pixel coordinate shift at stitching time.
    """

    if length <= 0 or window <= 0 or not 0 <= overlap < window or out_stride <= 0:
        raise ValueError("invalid tiled-inference geometry")
    if window % out_stride or (window - overlap) % out_stride:
        raise ValueError("window and step must be divisible by output stride")
    step = window - overlap
    final_origin = (max(0, length - window) // out_stride) * out_stride
    origins = sorted(
        {
            min(origin, final_origin)
            for origin in range(0, max(1, length - overlap), step)
        }
    )
    if not origins or origins[0] != 0 or origins[-1] != final_origin:
        raise AssertionError("tile origins do not include both valid-domain boundaries")
    if any(origin % out_stride for origin in origins):
        raise AssertionError("tile origin is not aligned to output stride")

    output_length = length // out_stride
    if output_length:
        covered = np.zeros(output_length, dtype=bool)
        for origin in origins:
            start = origin // out_stride
            stop = min(output_length, (origin + window) // out_stride)
            covered[start:stop] = True
        if not bool(np.all(covered)):
            missing = np.flatnonzero(~covered)
            raise AssertionError(
                f"tiled inference leaves {len(missing)} valid output cells uncovered"
            )
    return origins


def _predict_image(
    model: Any,
    gray: np.ndarray,
    *,
    device: str,
    in_channels: int,
    out_stride: int,
    use_amp: bool,
    horizontal_flip_tta: bool,
) -> dict[str, np.ndarray]:
    import torch

    model.eval()
    height, width = gray.shape
    ys = _tile_origins(
        height,
        window=HAIR_WINDOW,
        overlap=HAIR_OVERLAP,
        out_stride=out_stride,
    )
    xs = _tile_origins(
        width,
        window=HAIR_WINDOW,
        overlap=HAIR_OVERLAP,
        out_stride=out_stride,
    )
    output_height, output_width = height // out_stride, width // out_stride
    accumulated = {
        name: np.zeros((channels, output_height, output_width), dtype=np.float32)
        for name, channels in HEADS.items()
    }
    accumulated_weight = np.zeros((output_height, output_width), dtype=np.float32)
    full_weight = _window_weight(HAIR_WINDOW // out_stride)
    buffer: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []

    def flush() -> None:
        if not buffer:
            return
        tensor = torch.from_numpy(np.stack(buffer)).to(device, non_blocking=True)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp and str(device).startswith("cuda")
            else nullcontext()
        )
        with torch.no_grad(), amp_context:
            outputs = model(tensor)
            if horizontal_flip_tta:
                mirrored = model(torch.flip(tensor, dims=[3]))
                for name in outputs:
                    inverse = torch.flip(mirrored[name], dims=[3])
                    if name in ("base_dir", "flow"):
                        inverse[:, 0] = -inverse[:, 0]
                    elif name in ("base_off", "tip_off"):
                        inverse[:, 0] = 1.0 - inverse[:, 0]
                    outputs[name] = 0.5 * (outputs[name] + inverse)
        numpy_outputs = {
            name: tensor.float().cpu().numpy() for name, tensor in outputs.items()
        }
        for batch_index, (y, x) in enumerate(positions):
            output_y, output_x = y // out_stride, x // out_stride
            tile_height = min(full_weight.shape[0], output_height - output_y)
            tile_width = min(full_weight.shape[1], output_width - output_x)
            if tile_height <= 0 or tile_width <= 0:
                continue
            weight = full_weight[:tile_height, :tile_width]
            for name in accumulated:
                accumulated[name][
                    :, output_y : output_y + tile_height, output_x : output_x + tile_width
                ] += numpy_outputs[name][batch_index, :, :tile_height, :tile_width] * weight
            accumulated_weight[
                output_y : output_y + tile_height, output_x : output_x + tile_width
            ] += weight
        buffer.clear()
        positions.clear()

    for y in ys:
        for x in xs:
            patch = gray[y : y + HAIR_WINDOW, x : x + HAIR_WINDOW]
            patch_height, patch_width = patch.shape
            if patch_height < HAIR_WINDOW or patch_width < HAIR_WINDOW:
                patch = np.pad(
                    patch,
                    (
                        (0, HAIR_WINDOW - patch_height),
                        (0, HAIR_WINDOW - patch_width),
                    ),
                    mode="edge",
                )
            buffer.append(
                make_input_channels(patch, HAIR_WORKING_UM_PER_PX, in_channels)
            )
            positions.append((y, x))
            if len(buffer) >= HAIR_BATCH:
                flush()
    flush()
    covered = accumulated_weight > 1e-3
    divisor = np.maximum(accumulated_weight, 1e-6)
    for name in accumulated:
        accumulated[name] /= divisor[None]
    for name in ("base_hm", "tip_hm", "line", "root"):
        accumulated[name][:, ~covered] = -12.0
    return accumulated


def _shared_input_memory_estimate(
    height: int,
    width: int,
    *,
    in_channels: int,
    out_stride: int,
    ensemble_members: int,
) -> dict[str, int]:
    """Estimate deterministic ndarray buffers for shared-input inference.

    The estimate deliberately excludes the source/working grayscale images and
    framework/model allocations, which are also present in the legacy path.  It
    includes every shape-determined host array owned by the accelerated path and
    uses the conservative CPU lifetime where separate tile-channel arrays, their
    stacked batch and one member's output staging can coexist.  Device staging
    is reported separately; model parameters and activation workspaces cannot be
    derived from image geometry and are measured by the benchmark receipt.
    """

    if height <= 0 or width <= 0 or in_channels <= 0 or ensemble_members <= 0:
        raise ValueError("invalid shared-input memory-estimate geometry")
    if out_stride <= 0 or HAIR_WINDOW % out_stride:
        raise ValueError("invalid shared-input output stride")
    ys = _tile_origins(
        height,
        window=HAIR_WINDOW,
        overlap=HAIR_OVERLAP,
        out_stride=out_stride,
    )
    xs = _tile_origins(
        width,
        window=HAIR_WINDOW,
        overlap=HAIR_OVERLAP,
        out_stride=out_stride,
    )
    output_cells = (height // out_stride) * (width // out_stride)
    head_channels = sum(HEADS.values())
    tiles_in_largest_batch = min(HAIR_BATCH, len(ys) * len(xs))
    float32_bytes = np.dtype(np.float32).itemsize
    float64_bytes = np.dtype(np.float64).itemsize
    bool_bytes = np.dtype(bool).itemsize

    member_accumulators = (
        ensemble_members * head_channels * output_cells * float32_bytes
    )
    shared_weight = output_cells * float32_bytes
    coverage_mask = output_cells * bool_bytes
    window_weight = (HAIR_WINDOW // out_stride) ** 2 * float32_bytes
    tile_channel_buffers = (
        tiles_in_largest_batch
        * in_channels
        * HAIR_WINDOW
        * HAIR_WINDOW
        * float32_bytes
    )
    stacked_input_batch = tile_channel_buffers
    member_output_staging = (
        tiles_in_largest_batch
        * head_channels
        * (HAIR_WINDOW // out_stride) ** 2
        * float32_bytes
    )
    largest_head_output = (
        tiles_in_largest_batch
        * max(HEADS.values())
        * (HAIR_WINDOW // out_stride) ** 2
        * float32_bytes
    )
    device_second_forward_staging = (
        2 * stacked_input_batch + member_output_staging
    )
    device_tta_transform_staging = (
        stacked_input_batch
        + 2 * member_output_staging
        + 3 * largest_head_output
    )
    estimated_peak_device_arrays = max(
        device_second_forward_staging,
        device_tta_transform_staging,
    )

    # Finalization consumes member arrays head by head.  Calculate the exact
    # high-water mark for the retained averaged heads plus the current float64
    # aggregate instead of pessimistically retaining a full float64 ensemble.
    finalized_channels = 0
    finalization_peak = 0
    for channels in HEADS.values():
        remaining_member_channels = head_channels - finalized_channels
        before_first_member_release = (
            remaining_member_channels
            * ensemble_members
            * output_cells
            * float32_bytes
        )
        retained_averaged = finalized_channels * output_cells * float32_bytes
        current_float64_aggregate = channels * output_cells * float64_bytes
        finalization_peak = max(
            finalization_peak,
            before_first_member_release
            + retained_averaged
            + current_float64_aggregate,
        )
        finalized_channels += channels

    batch_phase_peak = (
        member_accumulators
        + shared_weight
        + window_weight
        + tile_channel_buffers
        + stacked_input_batch
        + member_output_staging
    )
    finalization_phase_peak = (
        finalization_peak + shared_weight + coverage_mask + window_weight
    )
    return {
        "working_height": int(height),
        "working_width": int(width),
        "output_cells": int(output_cells),
        "head_channels": int(head_channels),
        "ensemble_members": int(ensemble_members),
        "tiles": int(len(ys) * len(xs)),
        "tiles_in_largest_batch": int(tiles_in_largest_batch),
        "member_accumulators_bytes": int(member_accumulators),
        "shared_weight_bytes": int(shared_weight),
        "coverage_mask_bytes": int(coverage_mask),
        "window_weight_bytes": int(window_weight),
        "tile_channel_buffers_bytes": int(tile_channel_buffers),
        "stacked_input_batch_bytes": int(stacked_input_batch),
        "member_output_staging_bytes": int(member_output_staging),
        "device_full_image_accumulators_bytes": 0,
        "device_input_batch_bytes": int(stacked_input_batch),
        "device_member_output_bytes_fp32": int(member_output_staging),
        "device_largest_head_output_bytes_fp32": int(largest_head_output),
        "device_second_forward_staging_bytes": int(
            device_second_forward_staging
        ),
        "device_tta_transform_staging_bytes": int(
            device_tta_transform_staging
        ),
        "estimated_peak_device_array_bytes": int(
            estimated_peak_device_arrays
        ),
        "batch_phase_peak_bytes": int(batch_phase_peak),
        "finalization_phase_peak_bytes": int(finalization_phase_peak),
        "estimated_peak_host_array_bytes": int(
            max(batch_phase_peak, finalization_phase_peak)
        ),
        "estimated_peak_array_bytes": int(
            max(batch_phase_peak, finalization_phase_peak)
        ),
    }


def _predict_ensemble_image_shared_input(
    models: Sequence[Any],
    gray: np.ndarray,
    *,
    device: str,
    in_channels: int,
    out_stride: int,
    use_amp: bool,
    horizontal_flip_tta: bool,
) -> dict[str, np.ndarray]:
    """Run members in canonical order on one transferred tensor per tile batch.

    Five independent float32 stitching buffers preserve the legacy per-member
    accumulation and rounding order exactly.  This costs more host memory than
    the legacy model-outer loop; callers must apply ``_shared_input_memory_estimate``
    before entering this function.
    """

    import torch

    if not models:
        raise ValueError("shared-input inference requires at least one model")
    for model in models:
        model.eval()
    height, width = gray.shape
    ys = _tile_origins(
        height,
        window=HAIR_WINDOW,
        overlap=HAIR_OVERLAP,
        out_stride=out_stride,
    )
    xs = _tile_origins(
        width,
        window=HAIR_WINDOW,
        overlap=HAIR_OVERLAP,
        out_stride=out_stride,
    )
    output_height, output_width = height // out_stride, width // out_stride
    member_accumulated = [
        {
            name: np.zeros(
                (channels, output_height, output_width), dtype=np.float32
            )
            for name, channels in HEADS.items()
        }
        for _model in models
    ]
    accumulated_weight = np.zeros(
        (output_height, output_width), dtype=np.float32
    )
    full_weight = _window_weight(HAIR_WINDOW // out_stride)
    buffer: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []

    def flush() -> None:
        if not buffer:
            return
        tensor = torch.from_numpy(np.stack(buffer)).to(device, non_blocking=True)
        placements: list[tuple[int, int, int, int, int, np.ndarray]] = []
        for batch_index, (y, x) in enumerate(positions):
            output_y, output_x = y // out_stride, x // out_stride
            tile_height = min(full_weight.shape[0], output_height - output_y)
            tile_width = min(full_weight.shape[1], output_width - output_x)
            if tile_height <= 0 or tile_width <= 0:
                continue
            placements.append(
                (
                    batch_index,
                    output_y,
                    output_x,
                    tile_height,
                    tile_width,
                    full_weight[:tile_height, :tile_width],
                )
            )

        # The member order and each member's original-then-flipped TTA call
        # order are the locked legacy order within a shared tile batch.
        for model, accumulated in zip(models, member_accumulated, strict=True):
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp and str(device).startswith("cuda")
                else nullcontext()
            )
            with torch.no_grad(), amp_context:
                outputs = model(tensor)
                if horizontal_flip_tta:
                    mirrored = model(torch.flip(tensor, dims=[3]))
                    for name in outputs:
                        inverse = torch.flip(mirrored[name], dims=[3])
                        if name in ("base_dir", "flow"):
                            inverse[:, 0] = -inverse[:, 0]
                        elif name in ("base_off", "tip_off"):
                            inverse[:, 0] = 1.0 - inverse[:, 0]
                        outputs[name] = 0.5 * (outputs[name] + inverse)
            numpy_outputs = {
                name: output.float().cpu().numpy()
                for name, output in outputs.items()
            }
            for (
                batch_index,
                output_y,
                output_x,
                tile_height,
                tile_width,
                weight,
            ) in placements:
                for name in accumulated:
                    accumulated[name][
                        :,
                        output_y : output_y + tile_height,
                        output_x : output_x + tile_width,
                    ] += (
                        numpy_outputs[name][
                            batch_index, :, :tile_height, :tile_width
                        ]
                        * weight
                    )

        for (
            _batch_index,
            output_y,
            output_x,
            tile_height,
            tile_width,
            weight,
        ) in placements:
            accumulated_weight[
                output_y : output_y + tile_height,
                output_x : output_x + tile_width,
            ] += weight
        buffer.clear()
        positions.clear()

    for y in ys:
        for x in xs:
            patch = gray[y : y + HAIR_WINDOW, x : x + HAIR_WINDOW]
            patch_height, patch_width = patch.shape
            if patch_height < HAIR_WINDOW or patch_width < HAIR_WINDOW:
                patch = np.pad(
                    patch,
                    (
                        (0, HAIR_WINDOW - patch_height),
                        (0, HAIR_WINDOW - patch_width),
                    ),
                    mode="edge",
                )
            buffer.append(
                make_input_channels(patch, HAIR_WORKING_UM_PER_PX, in_channels)
            )
            positions.append((y, x))
            if len(buffer) >= HAIR_BATCH:
                flush()
    flush()

    covered = accumulated_weight > 1e-3
    np.maximum(accumulated_weight, 1e-6, out=accumulated_weight)
    averaged: dict[str, np.ndarray] = {}
    for name in HEADS:
        aggregate: np.ndarray | None = None
        for accumulated in member_accumulated:
            value = accumulated.pop(name)
            value /= accumulated_weight[None]
            if name in ("base_hm", "tip_hm", "line", "root"):
                value[:, ~covered] = -12.0
            if aggregate is None:
                aggregate = value.astype(np.float64)
            else:
                aggregate += value
            del value
        assert aggregate is not None
        aggregate /= len(models)
        averaged[name] = aggregate.astype(np.float32)
    return averaged


class StageBEnsemble:
    """Hash-verifiable legacy or explicitly gated train399 Stage-B ensemble."""

    def __init__(
        self,
        checkpoint_paths: Sequence[str | Path],
        *,
        device: str,
        use_amp: bool = False,
        candidate_manifest: str | Path | Mapping[str, Any] | None = None,
        selected_model_metadata: str | Path | Mapping[str, Any] | None = None,
        selection_receipt: str | Path | None = None,
        candidate_pool_mode: bool = False,
        shared_input_acceleration: bool = False,
        shared_input_max_host_bytes: int = _DEFAULT_SHARED_INPUT_MAX_HOST_BYTES,
        shared_input_max_device_bytes: int = (
            _DEFAULT_SHARED_INPUT_MAX_DEVICE_BYTES
        ),
        shared_input_device_reserve_bytes: int = (
            _DEFAULT_SHARED_INPUT_DEVICE_RESERVE_BYTES
        ),
    ):
        import torch

        if len(checkpoint_paths) != 5:
            raise ValueError("PHAxis 1.0.0 requires exactly five Stage-B checkpoints")
        self.device = str(device)
        self.use_amp = bool(use_amp)
        self.models: list[Any] = []
        self.config: dict[str, Any] | None = None
        self.detection_model_metadata: dict[str, Any] | None = None
        self.candidate_pool_mode = bool(candidate_pool_mode)
        self.shared_input_acceleration = bool(shared_input_acceleration)
        if isinstance(shared_input_max_host_bytes, bool) or int(
            shared_input_max_host_bytes
        ) <= 0:
            raise ValueError("shared-input host-memory limit must be positive")
        self.shared_input_max_host_bytes = int(shared_input_max_host_bytes)
        if isinstance(shared_input_max_device_bytes, bool) or int(
            shared_input_max_device_bytes
        ) <= 0:
            raise ValueError("shared-input device-memory limit must be positive")
        self.shared_input_max_device_bytes = int(shared_input_max_device_bytes)
        if isinstance(shared_input_device_reserve_bytes, bool) or int(
            shared_input_device_reserve_bytes
        ) < 0:
            raise ValueError("shared-input device-memory reserve cannot be negative")
        self.shared_input_device_reserve_bytes = int(
            shared_input_device_reserve_bytes
        )
        self.last_shared_input_audit: dict[str, Any] | None = None
        if self.candidate_pool_mode and self.use_amp:
            raise ValueError("candidate-pool model selection is locked to fp32")
        self.score_threshold = HAIR_SCORE_THRESHOLD

        ordered_paths = [Path(path) for path in checkpoint_paths]
        expected_checkpoint_sha256 = list(HAIR_CHECKPOINT_SHA256)
        expected_members: list[Mapping[str, Any]] | None = None
        if candidate_manifest is not None:
            if isinstance(candidate_manifest, Mapping):
                manifest = dict(candidate_manifest)
                validate_candidate_manifest(manifest)
            else:
                manifest = read_candidate_manifest(candidate_manifest)
            pending = detection_model_metadata_from_candidate_manifest(manifest)
            if self.candidate_pool_mode:
                if selected_model_metadata is not None or selection_receipt is not None:
                    raise ValueError(
                        "candidate-pool runtime requires pending metadata and cannot "
                        "accept a selected operating point"
                    )
                self.detection_model_metadata = pending
                self.score_threshold = CANDIDATE_POOL_SCORE_FLOOR
            elif selected_model_metadata is None:
                raise ValueError(
                    "train399 runtime requires model metadata with an explicitly "
                    "selected QC-development operating point"
                )
            else:
                if isinstance(selected_model_metadata, Mapping):
                    metadata = dict(selected_model_metadata)
                else:
                    metadata = read_json(selected_model_metadata)
                validate_train399_detection_model_metadata(metadata)
                for field in (
                    "candidate_bundle_identity_sha256",
                    "training_lock_identity_sha256",
                    "checkpoint_sha256",
                    "model_state_sha256",
                    "training_task_ids_sha256",
                    "split_manifest_sha256",
                    "operating_point_selection_contract_sha256",
                ):
                    if metadata.get(field) != pending.get(field):
                        raise ValueError(
                            f"selected model metadata differs from candidate gate: {field}"
                        )
                if selection_receipt is None:
                    raise ValueError(
                        "formal train399 runtime requires the bound selection receipt"
                    )
                receipt = read_selection_receipt(selection_receipt)
                validate_selected_operating_point_binding(
                    candidate_manifest=manifest,
                    selected_model_metadata=metadata,
                    selection_receipt=receipt,
                    selection_receipt_file_sha256=sha256_file(selection_receipt),
                )
                self.detection_model_metadata = metadata
                self.score_threshold = float(metadata["selected_score_threshold"])
            expected_members = list(manifest["identity_payload"]["members"])
            expected_checkpoint_sha256 = [
                member["checkpoint_sha256"] for member in expected_members
            ]
            path_by_hash: dict[str, Path] = {}
            for path in ordered_paths:
                digest = sha256_file(path)
                if digest in path_by_hash:
                    raise ValueError("duplicate Stage-B checkpoint content")
                path_by_hash[digest] = path
            if set(path_by_hash) != set(expected_checkpoint_sha256):
                raise ValueError("Stage-B checkpoints differ from the candidate manifest")
            ordered_paths = [path_by_hash[digest] for digest in expected_checkpoint_sha256]
        elif (
            selected_model_metadata is not None
            or selection_receipt is not None
            or self.candidate_pool_mode
        ):
            raise ValueError(
                "selected metadata, selection receipt and candidate-pool mode require "
                "a candidate_manifest"
            )
        self.checkpoint_sha256 = tuple(expected_checkpoint_sha256)

        for fold, checkpoint_path in enumerate(ordered_paths):
            checkpoint_path = Path(checkpoint_path)
            observed_sha256 = sha256_file(checkpoint_path)
            if observed_sha256 != expected_checkpoint_sha256[fold]:
                raise ValueError(
                    f"Stage-B fold{fold} checkpoint hash mismatch: {observed_sha256}"
                )
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            checkpoint_contract = checkpoint.get("contract")
            is_formal_train399 = (
                checkpoint.get("schema_version")
                == "PHAxis-StageB-train399-checkpoint-1.0"
                or (
                    isinstance(checkpoint_contract, Mapping)
                    and checkpoint_contract.get("formal_training") is True
                    and checkpoint_contract.get("training_images") == 399
                )
            )
            if is_formal_train399 and candidate_manifest is None:
                raise ValueError(
                    "formal train399 checkpoints require a candidate manifest, "
                    "selected model metadata and the bound selection receipt"
                )
            config = dict(checkpoint["cfg"])
            if expected_members is not None:
                expected_member = expected_members[fold]
                if checkpoint.get("seed") != expected_member["seed"] or checkpoint.get(
                    "member_id"
                ) != expected_member["member_id"]:
                    raise ValueError(
                        f"Stage-B candidate member{fold} seed/member identity mismatch"
                    )
            model = MultiHeadUNet(
                HEADS,
                encoder=config.get("encoder", "resnet34"),
                in_channels=int(config.get("in_channels", 3)),
                out_stride=int(config.get("out_stride", HAIR_OUT_STRIDE)),
                decoder_channels=tuple(
                    config.get("decoder_channels", (256, 128, 96, 64))
                ),
                pretrained=False,
                context=bool(config.get("context", True)),
                stem_stride1=bool(config.get("stem_stride1", False)),
            ).to(self.device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            self.models.append(model)
            if self.config is None:
                self.config = config
        assert self.config is not None
        if float(self.config.get("um_per_px", HAIR_WORKING_UM_PER_PX)) != HAIR_WORKING_UM_PER_PX:
            raise ValueError("Stage-B checkpoint physical scale mismatch")
        if int(self.config.get("out_stride", HAIR_OUT_STRIDE)) != HAIR_OUT_STRIDE:
            raise ValueError("Stage-B checkpoint output stride mismatch")

    def _predict_heads_and_geometry(
        self, image: np.ndarray, *, source_um_per_px: float
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        gray = to_gray(image)
        working, scale = resample_to_physical_scale(
            gray, source_um_per_px, HAIR_WORKING_UM_PER_PX
        )
        source_height, source_width = gray.shape
        working_height, working_width = working.shape
        scale_xy = np.asarray(
            [working_width / source_width, working_height / source_height],
            dtype=np.float64,
        )
        realized_um_per_px_xy = np.asarray(
            [source_um_per_px / scale_xy[0], source_um_per_px / scale_xy[1]],
            dtype=np.float64,
        )
        in_channels = int(self.config.get("in_channels", 3))
        requested = bool(getattr(self, "shared_input_acceleration", False))
        max_host_bytes = int(
            getattr(
                self,
                "shared_input_max_host_bytes",
                _DEFAULT_SHARED_INPUT_MAX_HOST_BYTES,
            )
        )
        max_device_bytes = int(
            getattr(
                self,
                "shared_input_max_device_bytes",
                _DEFAULT_SHARED_INPUT_MAX_DEVICE_BYTES,
            )
        )
        device_reserve_bytes = int(
            getattr(
                self,
                "shared_input_device_reserve_bytes",
                _DEFAULT_SHARED_INPUT_DEVICE_RESERVE_BYTES,
            )
        )
        averaged: dict[str, np.ndarray] | None = None
        estimate: dict[str, int] | None = None
        fallback_reason = "not_requested"
        device_free_bytes: int | None = None
        device_total_bytes: int | None = None
        device_required_free_bytes: int | None = None
        if requested:
            estimate = _shared_input_memory_estimate(
                working_height,
                working_width,
                in_channels=in_channels,
                out_stride=HAIR_OUT_STRIDE,
                ensemble_members=len(self.models),
            )
            if estimate["estimated_peak_array_bytes"] > max_host_bytes:
                fallback_reason = "estimated_peak_exceeds_limit"
            elif (
                estimate["estimated_peak_device_array_bytes"]
                > max_device_bytes
            ):
                fallback_reason = "device_estimated_peak_exceeds_limit"
            else:
                if str(self.device).startswith("cuda"):
                    import torch

                    device_required_free_bytes = (
                        estimate["estimated_peak_device_array_bytes"]
                        + device_reserve_bytes
                    )
                    try:
                        free_bytes, total_bytes = torch.cuda.mem_get_info(
                            self.device
                        )
                    except (RuntimeError, ValueError):
                        fallback_reason = "device_memory_query_failed"
                    else:
                        device_free_bytes = int(free_bytes)
                        device_total_bytes = int(total_bytes)
                        if device_free_bytes < device_required_free_bytes:
                            fallback_reason = (
                                "device_free_memory_below_required_reserve"
                            )
                if fallback_reason == "not_requested":
                    try:
                        averaged = _predict_ensemble_image_shared_input(
                            self.models,
                            working,
                            device=self.device,
                            in_channels=in_channels,
                            out_stride=HAIR_OUT_STRIDE,
                            use_amp=self.use_amp,
                            horizontal_flip_tta=True,
                        )
                    except MemoryError:
                        # The estimate gates deterministic ndarrays, but retain
                        # a fail-safe for fragmentation/system pressure.
                        import gc

                        gc.collect()
                        fallback_reason = "host_allocation_failed"
                    else:
                        fallback_reason = "none"

        used_shared_input = averaged is not None
        self.last_shared_input_audit = {
            "requested": requested,
            "used": used_shared_input,
            "runtime_path": (
                "shared_input_acceleration" if used_shared_input else "legacy"
            ),
            "fallback_reason": fallback_reason,
            "max_host_bytes": max_host_bytes,
            "max_device_bytes": max_device_bytes,
            "device_reserve_bytes": device_reserve_bytes,
            "device_required_free_bytes": device_required_free_bytes,
            "device_free_bytes_before": device_free_bytes,
            "device_total_bytes": device_total_bytes,
            "memory_estimate": estimate,
        }
        if averaged is None:
            ensemble: dict[str, np.ndarray] | None = None
            for model in self.models:
                heads = _predict_image(
                    model,
                    working,
                    device=self.device,
                    in_channels=in_channels,
                    out_stride=HAIR_OUT_STRIDE,
                    use_amp=self.use_amp,
                    horizontal_flip_tta=True,
                )
                if ensemble is None:
                    ensemble = {
                        name: value.astype(np.float64)
                        for name, value in heads.items()
                    }
                else:
                    for name in ensemble:
                        ensemble[name] += heads[name]
            assert ensemble is not None
            averaged = {
                name: (value / len(self.models)).astype(np.float32)
                for name, value in ensemble.items()
            }
        geometry = {
            "working_shape": list(working.shape),
            "source_shape": list(gray.shape),
            "source_to_working_scale": float(scale),
            "source_to_working_scale_xy": scale_xy.tolist(),
            "realized_um_per_px_xy": realized_um_per_px_xy.tolist(),
        }
        return averaged, geometry

    def predict(self, image: np.ndarray, *, source_um_per_px: float) -> dict[str, Any]:
        if getattr(self, "candidate_pool_mode", False):
            raise RuntimeError(
                "candidate-pool runtime cannot emit final Stage-B detections"
            )
        averaged, geometry = self._predict_heads_and_geometry(
            image, source_um_per_px=source_um_per_px
        )
        decoded = decode_instances(
            averaged,
            um_per_px=HAIR_WORKING_UM_PER_PX,
            out_stride=HAIR_OUT_STRIDE,
            score_threshold=self.score_threshold,
            nms_kernel=HAIR_NMS_KERNEL,
            max_instances=HAIR_MAX_INSTANCES,
            root_gate_um=HAIR_ROOT_GATE_UM,
        )
        decoded.update(geometry)
        return decoded

    def predict_biological_candidate_pool(
        self, image: np.ndarray, *, source_um_per_px: float
    ) -> dict[str, Any]:
        if not getattr(self, "candidate_pool_mode", False):
            raise RuntimeError(
                "biological-presence candidate pools require explicit candidate_pool_mode"
            )
        averaged, geometry = self._predict_heads_and_geometry(
            image, source_um_per_px=source_um_per_px
        )
        decoded = decode_biological_presence_candidates(
            averaged,
            um_per_px=HAIR_WORKING_UM_PER_PX,
            out_stride=HAIR_OUT_STRIDE,
            score_floor=CANDIDATE_POOL_SCORE_FLOOR,
            nms_kernel=HAIR_NMS_KERNEL,
            max_instances=HAIR_MAX_INSTANCES,
            root_gate_um=HAIR_ROOT_GATE_UM,
        )
        decoded.update(geometry)
        return decoded
