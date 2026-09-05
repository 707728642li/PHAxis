from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import stat

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from phaxis.hair_stageb.training_data import (
    DeterministicEpochSampler,
    HairRecord,
    StageBImageRecord,
    Train399HairCropDataset,
    build_training_targets,
    deterministic_worker_init,
    materialize_image_cache,
)
from phaxis.hair_stageb.training import (
    _gaussian_focal_loss,
    backward_with_loss_scale_retries,
    execute_with_failure_receipt,
    require_finite_unscaled_gradients,
    require_finite_loss,
    resolve_resume_history,
    validate_initialization_identity,
    validate_resume_progress,
)
from phaxis.io import sha256_file, sha256_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _tensor_digest(batch: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(batch):
        tensor = batch[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _synthetic_dataset(tmp_path: Path, seed: int) -> Train399HairCropDataset:
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    task_id = "SYNTHETIC-001"
    array = np.full((96, 96), 180, dtype=np.uint8)
    array[40:43, 10:80] = 80
    np.save(cache / f"{task_id}.npy", array, allow_pickle=False)
    (cache / f"{task_id}.meta.json").write_text(
        json.dumps(
            {
                "source_image_sha256": "a" * 64,
                "source_to_cached_scale": 1.0,
                "source_to_cached_scale_xy": [1.0, 1.0],
                "realized_um_per_px_xy": [2.0, 2.0],
                "cached_shape": [96, 96],
            }
        ),
        encoding="utf-8",
    )
    hair = HairRecord(
        instance_id="H1",
        points=((40.0, 45.0), (20.0, 45.0)),
        length_um=40.0,
        vertex_order_flipped=False,
    )
    record = StageBImageRecord(
        task_id=task_id,
        split="train",
        family_key="SYNTHETIC",
        image_path="unused",
        image_sha256="a" * 64,
        raw_annotation_sha256="b" * 64,
        canonical_annotation_sha256="c" * 64,
        source_um_per_px=2.0,
        width=96,
        height=96,
        root_polygon=((40.0, 5.0), (55.0, 5.0), (55.0, 90.0), (40.0, 90.0)),
        hairs=(hair,),
    )
    # The real gate requires 399 records.  Repeating this immutable synthetic
    # record keeps the unit test tiny while exercising that gate.
    return Train399HairCropDataset(
        [record] * 399,
        cache,
        crop=64,
        out_stride=2,
        crops_per_image=1,
        seed=seed,
    )


def _epoch_digests(dataset: Train399HairCropDataset, epoch: int) -> list[str]:
    # Eight composite indices are enough to prove epoch propagation through
    # persistent workers without processing all 399 synthetic records.
    sampler = DeterministicEpochSampler(8, seed=12345)
    sampler.set_epoch(epoch)
    generator = torch.Generator().manual_seed(991)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=2,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=deterministic_worker_init,
        generator=generator,
    )
    returned = [_tensor_digest(batch) for batch in loader]
    if loader._iterator is not None:  # deterministic cleanup on Windows
        loader._iterator._shutdown_workers()
    del loader
    gc.collect()
    return returned


def test_composite_epoch_index_is_reproducible_across_persistent_workers(
    tmp_path: Path,
) -> None:
    first = _synthetic_dataset(tmp_path, seed=2026082801)
    epoch0_first = _epoch_digests(first, 0)
    epoch1_first = _epoch_digests(first, 1)
    second = _synthetic_dataset(tmp_path, seed=2026082801)
    epoch0_second = _epoch_digests(second, 0)
    assert epoch0_first == epoch0_second
    assert epoch0_first != epoch1_first


def test_locked_audit_uses_qc_family_swap_and_excludes_val_from_gradient() -> None:
    audit_path = (
        PROJECT_ROOT
        / "outputs"
        / "phaxis_stageb_train399_dataset_audit_run1"
        / "dataset_audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["train_records"] == 399
    assert audit["excluded_val_records"] == 44
    assert audit["family_key_overlap"] == []
    assert "RHAUD-358" in audit["train_ids"]
    assert "RHAUD-358" not in audit["excluded_val_ids"]
    assert audit["excluded_val_root_hairs"] == 3800
    assert audit["validation_labels_used_for_gradient"] is False
    assert audit["validation_labels_used_for_early_stopping"] is False
    assert audit["blind_images_used"] == 0


def test_nonfinite_loss_fails_closed() -> None:
    require_finite_loss(torch.tensor(1.0), {"finite": 2.0}, epoch=1, global_step=0)
    with pytest.raises(FloatingPointError, match="non-finite"):
        require_finite_loss(
            torch.tensor(float("inf")), {"finite": 2.0}, epoch=1, global_step=0
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        require_finite_loss(
            torch.tensor(1.0), {"bad": float("nan")}, epoch=1, global_step=0
        )


def test_nonfinite_unscaled_gradient_reports_parameter_and_fails_closed() -> None:
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    model.bias.grad = torch.ones_like(model.bias)
    with pytest.raises(FloatingPointError, match=r"weight\[nonfinite=2/2\]"):
        require_finite_unscaled_gradients(model, epoch=1, global_step=0)


def test_finite_amp_backward_path_is_gradient_identical_without_retry() -> None:
    reference = torch.nn.Linear(3, 1, bias=False)
    candidate = torch.nn.Linear(3, 1, bias=False)
    candidate.load_state_dict(reference.state_dict())
    values = torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float32)

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1e-3)
    reference_scaler = torch.amp.GradScaler(
        "cpu", init_scale=8.0, growth_interval=1_000_000
    )
    reference_loss = reference(values).square().sum()
    reference_optimizer.zero_grad(set_to_none=True)
    reference_scaler.scale(reference_loss).backward()
    reference_scaler.unscale_(reference_optimizer)
    expected = reference.weight.grad.detach().clone()

    candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=1e-3)
    candidate_scaler = torch.amp.GradScaler(
        "cpu", init_scale=8.0, growth_interval=1_000_000
    )
    result = backward_with_loss_scale_retries(
        candidate(values).square().sum(),
        model=candidate,
        optimizer=candidate_optimizer,
        scaler=candidate_scaler,
        epoch=1,
        global_step=0,
    )
    assert result["retry_count"] == 0
    assert result["optimizer_step_skipped"] is False
    assert torch.equal(candidate.weight.grad, expected)


def test_amp_overflow_replays_same_graph_after_backoff_without_batch_skip() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    model = torch.nn.ParameterList([parameter])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-39)
    scaler = torch.amp.GradScaler(
        "cpu",
        init_scale=8.0,
        growth_interval=1_000_000,
        backoff_factor=0.5,
    )
    progress: dict[str, object] = {}
    finite_loss_with_scaled_overflow = parameter.sum() * 1e38

    result = backward_with_loss_scale_retries(
        finite_loss_with_scaled_overflow,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=1,
        global_step=6,
        max_retries=4,
        progress=progress,
    )
    assert result["retry_count"] == 2
    assert result["finite_scale"] == 2.0
    assert result["optimizer_step_skipped"] is False
    assert [event["scale_after_backoff"] for event in result["events"]] == [
        4.0,
        2.0,
    ]
    assert progress["amp_backward_retry_count"] == 2
    assert torch.isfinite(parameter.grad).all()
    value_before = parameter.detach().clone()
    scaler.step(optimizer)
    scaler.update()
    assert torch.isfinite(parameter).all()
    assert not torch.equal(parameter.detach(), value_before)


def test_amp_retry_limit_remains_fail_closed_without_optimizer_step() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    model = torch.nn.ParameterList([parameter])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-39)
    scaler = torch.amp.GradScaler(
        "cpu", init_scale=8.0, growth_interval=1_000_000, backoff_factor=0.5
    )
    value_before = parameter.detach().clone()
    with pytest.raises(FloatingPointError, match="after 1 loss-scale retries"):
        backward_with_loss_scale_retries(
            parameter.sum() * 1e38,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=1,
            global_step=6,
            max_retries=1,
        )
    assert torch.equal(parameter.detach(), value_before)


def test_production_shape_focal_head_overflow_is_recovered_at_scale_512() -> None:
    # Mirrors the seed3 failure mechanism: a heatmap head reduces 8×384×384
    # focal contributions through an FP16 final convolution while only 74
    # exact peaks normalize the loss.  No RNG is involved.
    inputs = torch.full((8, 1, 384, 384), 2.0, dtype=torch.float32)
    target = torch.zeros((8, 1, 384, 384), dtype=torch.float32)
    target.reshape(-1)[:74] = 1.0
    model = torch.nn.Conv2d(1, 1, kernel_size=1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.constant_(model.bias, np.log(0.1 / 0.9))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(
        "cpu",
        init_scale=1024.0,
        growth_interval=1_000_000,
        growth_factor=2.0,
        backoff_factor=0.5,
    )
    with torch.autocast("cpu", dtype=torch.float16):
        loss = _gaussian_focal_loss(model(inputs), target)

    assert float(loss.detach()) == pytest.approx(18.6579056, rel=1e-6)
    result = backward_with_loss_scale_retries(
        loss,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=1,
        global_step=6,
    )
    assert [event["scale_after_backoff"] for event in result["events"]] == [
        512.0
    ]
    assert float(model.weight.grad) == pytest.approx(90.125, rel=1e-5)
    assert torch.isfinite(model.weight.grad).all()
    weight_before = model.weight.detach().clone()
    scaler.step(optimizer)
    scaler.update()
    assert not torch.equal(model.weight.detach(), weight_before)


def test_resume_identity_and_progress_fail_closed() -> None:
    initialization = {
        "source": "timm/resnet34.a1_in1k",
        "huggingface_revision": "abc",
        "cached_weight_sha256": "d" * 64,
        "historical_stageb_checkpoint_loaded": False,
    }
    initialization["initialization_sha256"] = sha256_json(initialization)
    assert (
        validate_initialization_identity(
            initialization, initialization["initialization_sha256"]
        )
        == initialization["initialization_sha256"]
    )
    corrupted = dict(initialization)
    corrupted["source"] = "changed"
    with pytest.raises(RuntimeError, match="initialization"):
        validate_initialization_identity(
            corrupted, initialization["initialization_sha256"]
        )
    checkpoint = {"epoch": 2, "global_step": 20}
    history = [
        {"epoch": 1, "global_step": 10},
        {"epoch": 2, "global_step": 20},
    ]
    assert validate_resume_progress(
        checkpoint, history, steps_per_epoch=10, configured_epochs=60
    ) == (2, 20)
    with pytest.raises(RuntimeError, match="global_step"):
        validate_resume_progress(
            {"epoch": 2, "global_step": 19},
            history,
            steps_per_epoch=10,
            configured_epochs=60,
        )


def test_resume_history_transaction_recovers_sidecar_ahead_or_behind() -> None:
    history = [
        {"epoch": 1, "global_step": 10},
        {"epoch": 2, "global_step": 20},
        {"epoch": 3, "global_step": 30},
    ]
    legacy_checkpoint = {"epoch": 2, "global_step": 20}
    recovered, changed = resolve_resume_history(
        legacy_checkpoint,
        history,
        steps_per_epoch=10,
        configured_epochs=60,
    )
    assert changed is True
    assert recovered == history[:2]

    embedded_checkpoint = {
        "epoch": 2,
        "global_step": 20,
        "history": history[:2],
    }
    recovered, changed = resolve_resume_history(
        embedded_checkpoint,
        history[:1],
        steps_per_epoch=10,
        configured_epochs=60,
    )
    assert changed is True
    assert recovered == history[:2]


def test_existing_cache_validates_shape_dtype_scale_hash_and_readonly(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache" / "2umpx"
    cache.mkdir(parents=True)
    task_id = "CACHE-001"
    array_path = cache / f"{task_id}.npy"
    array = np.zeros((32, 48), dtype=np.uint8)
    np.save(array_path, array, allow_pickle=False)
    array_sha256 = sha256_file(array_path)
    record = StageBImageRecord(
        task_id=task_id,
        split="train",
        family_key="CACHE",
        image_path="unused",
        image_sha256="a" * 64,
        raw_annotation_sha256="b" * 64,
        canonical_annotation_sha256="c" * 64,
        source_um_per_px=2.0,
        width=48,
        height=32,
        root_polygon=((10.0, 0.0), (20.0, 0.0), (20.0, 31.0), (10.0, 31.0)),
        hairs=(),
    )
    metadata_path = cache / f"{task_id}.meta.json"
    metadata = {
        "task_id": record.task_id,
        "source_image_sha256": record.image_sha256,
        "raw_annotation_sha256": record.raw_annotation_sha256,
        "canonical_annotation_sha256": record.canonical_annotation_sha256,
        "target_um_per_px": 2.0,
        "source_to_cached_scale": 1.0,
        "source_shape": [32, 48],
        "cached_shape": [32, 48],
        "dtype": "uint8",
        "array_sha256": array_sha256,
        "materialization": "canonical_image_decode_and_resample",
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    array_path.chmod(stat.S_IREAD)
    result = materialize_image_cache([record], cache, hash_arrays=True)
    assert result["status"] == "passed"
    assert result["entries"][0]["array_sha256"] == array_sha256
    first_identity = result["cache_identity_sha256"]
    validated_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validated_metadata["audit_note"] = "identity-binds-complete-metadata"
    metadata_path.write_text(json.dumps(validated_metadata), encoding="utf-8")
    changed = materialize_image_cache([record], cache, hash_arrays=True)
    assert changed["cache_identity_sha256"] != first_identity
    metadata["source_to_cached_scale"] = 2.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale cache provenance"):
        materialize_image_cache([record], cache, hash_arrays=True)


def test_noninteger_resize_uses_realized_per_axis_vector_scale(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "noninteger" / "2umpx"
    cache.mkdir(parents=True)
    task_id = "CACHE-NONINTEGER"
    width, height = 101, 103
    source_um_per_px, target_um_per_px = 2.3, 2.0
    requested_scale = source_um_per_px / target_um_per_px
    cached_shape = (
        round(height * requested_scale),
        round(width * requested_scale),
    )
    array_path = cache / f"{task_id}.npy"
    np.save(array_path, np.zeros(cached_shape, dtype=np.uint8), allow_pickle=False)
    array_sha256 = sha256_file(array_path)
    hair = HairRecord(
        instance_id="H1",
        points=((10.25, 20.75), (87.625, 91.125)),
        length_um=40.0,
        vertex_order_flipped=False,
    )
    record = StageBImageRecord(
        task_id=task_id,
        split="train",
        family_key="CACHE-NONINTEGER",
        image_path="unused",
        image_sha256="a" * 64,
        raw_annotation_sha256="b" * 64,
        canonical_annotation_sha256="c" * 64,
        source_um_per_px=source_um_per_px,
        width=width,
        height=height,
        root_polygon=((9.25, 2.5), (21.75, 2.5), (21.75, 99.5), (9.25, 99.5)),
        hairs=(hair,),
    )
    metadata_path = cache / f"{task_id}.meta.json"
    # Exercise the safe metadata-only migration from the scalar v1.0 cache.
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "PHAxis-StageB-physical-cache-entry-1.0",
                "task_id": record.task_id,
                "source_image_sha256": record.image_sha256,
                "raw_annotation_sha256": record.raw_annotation_sha256,
                "canonical_annotation_sha256": record.canonical_annotation_sha256,
                "source_um_per_px": source_um_per_px,
                "target_um_per_px": target_um_per_px,
                "source_shape": [height, width],
                "cached_shape": list(cached_shape),
                "source_to_cached_scale": requested_scale,
                "dtype": "uint8",
                "array_sha256": array_sha256,
                "materialization": "canonical_image_decode_and_resample",
            }
        ),
        encoding="utf-8",
    )
    array_path.chmod(stat.S_IREAD)

    audit = materialize_image_cache(
        [record], cache, target_um_per_px=target_um_per_px, hash_arrays=True
    )
    migrated = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_scale_xy = np.asarray(
        [cached_shape[1] / width, cached_shape[0] / height], dtype=np.float64
    )
    assert migrated["schema_version"] == "PHAxis-StageB-physical-cache-entry-1.1"
    assert migrated["source_to_cached_scale"] == pytest.approx(requested_scale)
    assert migrated["source_to_cached_scale_xy"] == pytest.approx(expected_scale_xy)
    assert migrated["scale_xy"] == pytest.approx(expected_scale_xy)
    assert migrated["realized_um_per_px_xy"] == pytest.approx(
        source_um_per_px / expected_scale_xy
    )
    assert audit["vector_coordinate_mapping"] == (
        "per_axis_realized_source_to_cached_scale_xy"
    )

    dataset = Train399HairCropDataset(
        [record] * 399,
        cache,
        crop=64,
        out_stride=2,
        crops_per_image=1,
        seed=2026082801,
    )
    geometry = dataset.geometry[task_id]
    assert geometry["root"] == pytest.approx(
        np.asarray(record.root_polygon) * expected_scale_xy
    )
    assert geometry["hairs"][0]["points"] == pytest.approx(
        np.asarray(hair.points) * expected_scale_xy
    )
    assert geometry["hairs"][0]["length_um"] == hair.length_um


def test_training_failure_receipt_is_atomic_and_exception_is_rethrown(
    tmp_path: Path,
) -> None:
    finalized = {"called": False}

    def fail() -> None:
        raise ValueError("deliberate unit-test failure")

    def finalize() -> dict[str, str]:
        finalized["called"] = True
        return {"status": "passed"}

    with pytest.raises(ValueError, match="deliberate unit-test failure"):
        execute_with_failure_receipt(
            fail,
            finalize,
            output=tmp_path,
            progress={"completed_epoch": 3, "global_step": 123},
            preflight={"status": "passed"},
        )
    assert finalized["called"] is True
    failure = json.loads((tmp_path / "training_failure.json").read_text("utf-8"))
    assert failure["status"] == "failed"
    assert failure["exception_type"] == "builtins.ValueError"
    assert failure["completed_epoch"] == 3
    assert failure["global_step"] == 123
    assert failure["nvidia_smi_preflight_status"] == "passed"
    assert failure["nvidia_smi_training_monitor_status"] == "passed"
    assert failure["exception_swallowed"] is False


def test_continuous_crop_coordinate_augmentations_are_peak_equivariant() -> None:
    crop = 128
    stride = 2
    root_mask = np.zeros((crop, crop), dtype=np.uint8)
    generator = np.random.default_rng(20260828)
    mismatches = 0
    comparisons = 0

    def anchor_cell_and_offset(
        target: dict[str, np.ndarray], anchor: str
    ) -> tuple[tuple[int, int], np.ndarray]:
        mask = target[f"_{anchor}_mask"][0]
        cells = np.argwhere(mask > 0.5)
        assert cells.shape == (1, 2)
        iy, ix = (int(cells[0, 0]), int(cells[0, 1]))
        heatmap_peak = np.unravel_index(
            int(np.argmax(target[f"{anchor}_hm"][0])), mask.shape
        )
        assert heatmap_peak == (iy, ix)
        return (iy, ix), target[f"{anchor}_off"][:, iy, ix]

    def mean_flow_direction(target: dict[str, np.ndarray]) -> np.ndarray:
        flow = target["flow"]
        valid = np.linalg.norm(flow, axis=0) > 0.5
        assert valid.any()
        returned = flow[:, valid].mean(axis=1)
        return returned / np.linalg.norm(returned)

    def targets(points: np.ndarray) -> dict[str, np.ndarray]:
        return build_training_targets(
            [{"points": points.astype(np.float32), "length_um": 40.0}],
            root_mask,
            crop=crop,
            out_stride=stride,
            um_per_px=2.0,
            base_sigma_um=6.0,
            tip_sigma_um=8.0,
            line_halfwidth_um=3.0,
        )

    for _ in range(100):
        # Explicitly avoid integer and stride-boundary coordinates; those hide
        # the old crop-1 convention bug in roughly half the samples.
        base = generator.uniform(20.05, 107.95, size=2)
        base += np.where(np.mod(base, 1.0) < 0.05, 0.17, 0.0)
        delta = generator.uniform(-12.0, 12.0, size=2)
        tip = np.clip(base + delta, 8.05, 119.95)
        points = np.stack((base, tip)).astype(np.float32)
        original = targets(points)

        horizontal_points = points.copy()
        horizontal_points[:, 0] = crop - horizontal_points[:, 0]
        horizontal = targets(horizontal_points)
        vertical_points = points.copy()
        vertical_points[:, 1] = crop - vertical_points[:, 1]
        vertical = targets(vertical_points)
        rotation_points = np.column_stack(
            (points[:, 1], crop - points[:, 0])
        ).astype(np.float32)
        rotation = targets(rotation_points)

        for head in ("base_hm", "tip_hm"):
            checks = (
                np.array_equal(horizontal[head], original[head][:, :, ::-1]),
                np.array_equal(vertical[head], original[head][:, ::-1, :]),
                np.array_equal(
                    rotation[head], np.rot90(original[head], 1, axes=(-2, -1))
                ),
            )
            mismatches += sum(not value for value in checks)
            comparisons += len(checks)

        output_size = crop // stride
        transforms = (
            (
                horizontal,
                lambda iy, ix: (iy, output_size - 1 - ix),
                lambda offset: np.asarray([1.0 - offset[0], offset[1]]),
                np.asarray([[-1.0, 0.0], [0.0, 1.0]]),
            ),
            (
                vertical,
                lambda iy, ix: (output_size - 1 - iy, ix),
                lambda offset: np.asarray([offset[0], 1.0 - offset[1]]),
                np.asarray([[1.0, 0.0], [0.0, -1.0]]),
            ),
            (
                rotation,
                lambda iy, ix: (output_size - 1 - ix, iy),
                lambda offset: np.asarray([offset[1], 1.0 - offset[0]]),
                np.asarray([[0.0, 1.0], [-1.0, 0.0]]),
            ),
        )
        for anchor in ("base", "tip"):
            original_cell, original_offset = anchor_cell_and_offset(original, anchor)
            for transformed, cell_transform, offset_transform, _matrix in transforms:
                transformed_cell, transformed_offset = anchor_cell_and_offset(
                    transformed, anchor
                )
                assert transformed_cell == cell_transform(*original_cell)
                assert transformed_offset == pytest.approx(
                    offset_transform(original_offset), abs=2e-6
                )

        original_base_cell, _ = anchor_cell_and_offset(original, "base")
        original_base_direction = original["base_dir"][:, *original_base_cell]
        original_flow_direction = mean_flow_direction(original)
        for transformed, cell_transform, _offset_transform, matrix in transforms:
            transformed_cell = cell_transform(*original_base_cell)
            assert transformed["base_dir"][:, *transformed_cell] == pytest.approx(
                matrix @ original_base_direction, abs=1e-5
            )
            assert mean_flow_direction(transformed) == pytest.approx(
                matrix @ original_flow_direction, abs=1e-5
            )
    assert comparisons == 600
    assert mismatches == 0
