"""Leak-free fixed-horizon training for the PHAxis Stage-B hair expert."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from ..io import atomic_write_json, sha256_file, sha256_json
from .model import HEADS, MultiHeadUNet
from .training_data import (
    DeterministicEpochSampler,
    EXPECTED_TRAIN,
    StageBImageRecord,
    Train399HairCropDataset,
    deterministic_worker_init,
)


FORMAL_SEEDS = (2026082801, 2026082802, 2026082803, 2026082804, 2026082805)
AMP_BACKWARD_RETRY_LIMIT = 16


@dataclass(frozen=True)
class StageBTrain399Config:
    um_per_px: float = 2.0
    out_stride: int = 2
    crop: int = 768
    base_sigma_um: float = 6.0
    tip_sigma_um: float = 8.0
    line_halfwidth_um: float = 3.0
    encoder: str = "resnet34"
    imagenet_source: str = "timm/resnet34.a1_in1k"
    in_channels: int = 3
    decoder_channels: tuple[int, ...] = (256, 128, 96, 64)
    context: bool = True
    stem_stride1: bool = False
    w_base_hm: float = 1.0
    w_base_off: float = 1.0
    w_base_vec: float = 1.0
    w_tip_hm: float = 1.0
    w_tip_off: float = 1.0
    w_line: float = 1.0
    w_cldice: float = 0.5
    w_dir: float = 0.5
    w_root: float = 0.5
    epochs: int = 60
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    amp: bool = True
    # Default 65,536 overflowed the first unscaled backward pass on RTX 3090
    # despite a finite FP32 loss.  A fixed conservative scale preserves FP16
    # underflow protection without relying on silently skipped optimizer steps.
    amp_initial_scale: float = 1024.0
    amp_growth_interval: int = 1_000_000
    amp_growth_factor: float = 2.0
    amp_backoff_factor: float = 0.5
    crops_per_image: int = 8
    background_fraction: float = 0.12
    workers: int = 8
    prefetch_factor: int = 4
    gradient_clip_norm: float = 5.0
    channels_last: bool = True
    deterministic_algorithms: bool = True
    fixed_last_epoch_policy: bool = True
    score_thresh: float = 0.225
    nms_kernel: int = 5
    max_instances: int = 4000


def _gaussian_focal_loss(
    prediction_logit: torch.Tensor,
    ground_truth: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    prediction_logit = prediction_logit.float()
    ground_truth = ground_truth.float()
    probability = torch.sigmoid(prediction_logit).clamp(1e-4, 1 - 1e-4)
    positive = ground_truth.ge(1.0 - 1e-6).float()
    negative = 1.0 - positive
    negative_weight = torch.pow(1.0 - ground_truth, beta)
    positive_loss = (
        -torch.log(probability) * torch.pow(1 - probability, alpha) * positive
    )
    negative_loss = (
        -torch.log(1 - probability)
        * torch.pow(probability, alpha)
        * negative_weight
        * negative
    )
    count = positive.sum()
    if count < 1:
        return negative_loss.sum() / max(prediction_logit.numel() / 1e4, 1.0)
    return (positive_loss.sum() + negative_loss.sum()) / count


def _masked_l1(
    prediction: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    expanded = mask.expand_as(prediction)
    count = expanded.sum()
    if count < 1:
        return prediction.sum() * 0.0
    return (torch.abs(prediction - ground_truth) * expanded).sum() / count


def _masked_cosine(
    prediction: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    normalized_prediction = functional.normalize(prediction, dim=1, eps=1e-6)
    valid = (ground_truth.norm(dim=1, keepdim=True) > 1e-3).float() * mask
    count = valid.sum()
    if count < 1:
        return prediction.sum() * 0.0
    cosine = (
        normalized_prediction
        * functional.normalize(ground_truth, dim=1, eps=1e-6)
    ).sum(dim=1, keepdim=True)
    return ((1.0 - cosine) * valid).sum() / count


def _soft_dice(
    prediction_logit: torch.Tensor, ground_truth: torch.Tensor, epsilon: float = 1.0
) -> torch.Tensor:
    probability = torch.sigmoid(prediction_logit.float())
    ground_truth = ground_truth.float()
    dimensions = (0, 2, 3)
    intersection = (probability * ground_truth).sum(dimensions)
    denominator = probability.sum(dimensions) + ground_truth.sum(dimensions)
    return (1.0 - (2 * intersection + epsilon) / (denominator + epsilon)).mean()


def _soft_erode(tensor: torch.Tensor) -> torch.Tensor:
    vertical = -functional.max_pool2d(-tensor, (3, 1), (1, 1), (1, 0))
    horizontal = -functional.max_pool2d(-tensor, (1, 3), (1, 1), (0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_open(tensor: torch.Tensor) -> torch.Tensor:
    return functional.max_pool2d(_soft_erode(tensor), (3, 3), (1, 1), (1, 1))


def _soft_skeleton(tensor: torch.Tensor, iterations: int = 6) -> torch.Tensor:
    opened = _soft_open(tensor)
    skeleton = functional.relu(tensor - opened)
    for _ in range(iterations):
        tensor = _soft_erode(tensor)
        opened = _soft_open(tensor)
        delta = functional.relu(tensor - opened)
        skeleton = skeleton + functional.relu(delta - skeleton * delta)
    return skeleton


def _cldice(
    prediction_logit: torch.Tensor,
    ground_truth: torch.Tensor,
    iterations: int = 6,
    epsilon: float = 1.0,
) -> torch.Tensor:
    probability = torch.sigmoid(prediction_logit.float())
    ground_truth = ground_truth.float()
    prediction_skeleton = _soft_skeleton(probability, iterations)
    target_skeleton = _soft_skeleton(ground_truth, iterations)
    topology_precision = (prediction_skeleton * ground_truth).sum() + epsilon
    topology_precision /= prediction_skeleton.sum() + epsilon
    topology_sensitivity = (target_skeleton * probability).sum() + epsilon
    topology_sensitivity /= target_skeleton.sum() + epsilon
    return 1.0 - 2.0 * topology_precision * topology_sensitivity / (
        topology_precision + topology_sensitivity
    )


class StageBTrainingLoss(nn.Module):
    def __init__(self, config: StageBTrain399Config):
        super().__init__()
        self.config = config

    def forward(
        self, output: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        config = self.config
        output = {name: value.float() for name, value in output.items()}
        base_mask = target["_base_mask"]
        tip_mask = target["_tip_mask"]
        parts: dict[str, torch.Tensor] = {}
        parts["base_hm"] = (
            _gaussian_focal_loss(output["base_hm"], target["base_hm"])
            * config.w_base_hm
        )
        parts["base_off"] = (
            _masked_l1(output["base_off"], target["base_off"], base_mask)
            * config.w_base_off
        )
        parts["base_dir"] = (
            _masked_cosine(output["base_dir"], target["base_dir"], base_mask)
            * config.w_base_vec
        )
        parts["base_len"] = (
            _masked_l1(output["base_len"], target["base_len"], base_mask)
            * config.w_base_vec
        )
        parts["tip_hm"] = (
            _gaussian_focal_loss(output["tip_hm"], target["tip_hm"])
            * config.w_tip_hm
        )
        parts["tip_off"] = (
            _masked_l1(output["tip_off"], target["tip_off"], tip_mask)
            * config.w_tip_off
        )
        line_target = target["line"]
        positive_weight = torch.tensor(20.0, device=line_target.device)
        binary_cross_entropy = functional.binary_cross_entropy_with_logits(
            output["line"], line_target, pos_weight=positive_weight
        )
        parts["line"] = (
            binary_cross_entropy + _soft_dice(output["line"], line_target)
        ) * config.w_line
        if config.w_cldice > 0:
            parts["cldice"] = (
                _cldice(output["line"], line_target) * config.w_cldice
            )
        line_mask = (line_target > 0.35).float()
        parts["flow"] = (
            _masked_cosine(output["flow"], target["flow"], line_mask)
            * config.w_dir
        )
        root_target = target["root"]
        parts["root"] = (
            functional.binary_cross_entropy_with_logits(output["root"], root_target)
            + _soft_dice(output["root"], root_target)
        ) * config.w_root
        total = sum(parts.values())
        return total, {name: float(value.detach()) for name, value in parts.items()}


def require_finite_loss(
    loss: torch.Tensor, parts: dict[str, float], *, epoch: int, global_step: int
) -> None:
    if not bool(torch.isfinite(loss).item()) or not all(
        math.isfinite(value) for value in parts.values()
    ):
        raise FloatingPointError(
            f"non-finite Stage-B loss at epoch={epoch} step={global_step}"
        )


def _nonfinite_unscaled_gradient_failures(model: nn.Module) -> list[str]:
    failures: list[str] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        if not bool(finite.all().item()):
            failures.append(
                f"{name}[nonfinite={gradient.numel() - int(finite.sum().item())}/"
                f"{gradient.numel()}]"
            )
            if len(failures) >= 8:
                break
    return failures


def require_finite_unscaled_gradients(
    model: nn.Module, *, epoch: int, global_step: int
) -> None:
    """Identify non-finite gradients after AMP unscale and fail immediately."""

    failures = _nonfinite_unscaled_gradient_failures(model)
    if failures:
        raise FloatingPointError(
            f"non-finite unscaled Stage-B gradients at epoch={epoch} "
            f"step={global_step}: " + ", ".join(failures)
        )


def backward_with_loss_scale_retries(
    loss: torch.Tensor,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    max_retries: int = AMP_BACKWARD_RETRY_LIMIT,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Back off AMP loss scale and replay one graph without skipping its batch.

    GradScaler normally handles overflow by skipping the optimizer step.  That
    would violate the fixed 399-step-per-epoch contract.  Instead, the forward
    graph is retained and the same loss is replayed after each scaler backoff;
    parameters, optimizer state, data order, RNG state and BatchNorm buffers are
    therefore unchanged until one finite unscaled gradient is obtained.  A
    bounded failure remains fail-closed.
    """

    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")
    retry_events: list[dict[str, Any]] = []
    while True:
        scale_before = float(scaler.get_scale())
        if progress is not None:
            progress["amp_scale_before_backward"] = scale_before
            progress["amp_min_scale"] = min(
                float(progress.get("amp_min_scale", scale_before)), scale_before
            )
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(retain_graph=scaler.is_enabled())
        scaler.unscale_(optimizer)
        failures = _nonfinite_unscaled_gradient_failures(model)
        if not failures:
            return {
                "retry_count": len(retry_events),
                "minimum_scale": min(
                    [scale_before]
                    + [float(event["scale_after_backoff"]) for event in retry_events]
                ),
                "finite_scale": scale_before,
                "events": retry_events,
                "optimizer_step_skipped": False,
            }
        if not scaler.is_enabled() or len(retry_events) >= max_retries:
            raise FloatingPointError(
                f"non-finite unscaled Stage-B gradients at epoch={epoch} "
                f"step={global_step} after {len(retry_events)} loss-scale "
                f"retries: " + ", ".join(failures)
            )

        # unscale_ has populated GradScaler's found-inf state.  update() applies
        # the configured backoff and resets the scaler for the same graph.
        scaler.update()
        scale_after = float(scaler.get_scale())
        if (
            not math.isfinite(scale_after)
            or scale_after <= 0.0
            or scale_after >= scale_before
        ):
            raise FloatingPointError(
                "AMP loss-scale backoff did not produce a smaller positive scale"
            )
        event = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "retry_index": len(retry_events) + 1,
            "scale_before_backoff": scale_before,
            "scale_after_backoff": scale_after,
            "nonfinite_parameters": failures,
            "optimizer_step_skipped": False,
            "same_forward_graph_replayed": True,
        }
        retry_events.append(event)
        if progress is not None:
            stored = progress.setdefault("amp_backward_retry_events", [])
            stored.append(event)
            progress["amp_backward_retry_count"] = len(stored)
            progress["amp_min_scale"] = min(
                float(progress.get("amp_min_scale", scale_after)), scale_after
            )


def validate_initialization_identity(
    initialization: dict[str, Any], checkpoint_initialization_sha256: str
) -> str:
    stored = initialization.get("initialization_sha256")
    recomputed = sha256_json(
        {
            key: value
            for key, value in initialization.items()
            if key != "initialization_sha256"
        }
    )
    if stored != recomputed or checkpoint_initialization_sha256 != stored:
        raise RuntimeError("resume initialization identity is invalid")
    return str(stored)


def validate_resume_progress(
    checkpoint: dict[str, Any],
    history: Sequence[dict[str, Any]],
    *,
    steps_per_epoch: int,
    configured_epochs: int,
) -> tuple[int, int]:
    epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint["global_step"])
    if epoch < 0 or epoch > configured_epochs:
        raise RuntimeError("resume epoch is outside the configured horizon")
    if len(history) != epoch:
        raise RuntimeError("resume history length differs from checkpoint epoch")
    for index, record in enumerate(history, start=1):
        if int(record.get("epoch", -1)) != index:
            raise RuntimeError("resume history epoch sequence is not contiguous")
        if int(record.get("global_step", -1)) != index * steps_per_epoch:
            raise RuntimeError("resume history step sequence is not contiguous")
    expected_global_step = epoch * steps_per_epoch
    if global_step != expected_global_step:
        raise RuntimeError(
            f"resume global_step {global_step} != {expected_global_step}"
        )
    if epoch and int(history[-1].get("global_step", -1)) != global_step:
        raise RuntimeError("resume history global_step differs from checkpoint")
    return epoch, global_step


def resolve_resume_history(
    checkpoint: dict[str, Any],
    sidecar_history: Sequence[dict[str, Any]],
    *,
    steps_per_epoch: int,
    configured_epochs: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Recover an interrupted history/checkpoint two-file transaction.

    New checkpoints embed their authoritative history. For the first formal
    member, which may have been launched before this transaction hardening was
    loaded, an older checkpoint can safely trim a sidecar that is one or more
    rows ahead: no model/optimizer state exists for those uncommitted rows.
    """

    epoch = int(checkpoint["epoch"])
    embedded = checkpoint.get("history")
    recovered = False
    if embedded is not None:
        if not isinstance(embedded, list):
            raise RuntimeError("resume checkpoint history is not a list")
        authoritative = list(embedded)
        recovered = list(sidecar_history) != authoritative
    else:
        if len(sidecar_history) < epoch:
            raise RuntimeError("legacy resume history is behind its checkpoint")
        authoritative = list(sidecar_history[:epoch])
        recovered = len(sidecar_history) != epoch
    validate_resume_progress(
        checkpoint,
        authoritative,
        steps_per_epoch=steps_per_epoch,
        configured_epochs=configured_epochs,
    )
    return authoritative, recovered


def _cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _set_reproducibility(seed: int, deterministic: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=False)


def _state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _discover_imagenet_source() -> dict[str, Any]:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repository = cache_root / "models--timm--resnet34.a1_in1k"
    revision = None
    revision_path = repository / "refs" / "main"
    if revision_path.is_file():
        revision = revision_path.read_text(encoding="utf-8").strip()
    candidates = sorted(repository.glob("snapshots/*/model.safetensors"))
    selected = next((path for path in candidates if revision in path.parts), None)
    if selected is None and len(candidates) == 1:
        selected = candidates[0]
    return {
        "source": "timm/resnet34.a1_in1k",
        "huggingface_revision": revision,
        "cached_weight_path": str(selected) if selected else None,
        "cached_weight_sha256": sha256_file(selected) if selected else None,
        "cached_weight_size_bytes": selected.stat().st_size if selected else None,
    }


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(arguments: list[str]) -> tuple[int, str]:
        process = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return process.returncode, process.stdout.strip()

    head_code, head = run(["rev-parse", "HEAD"])
    branch_code, branch = run(["branch", "--show-current"])
    status_code, status = run(["status", "--porcelain=v1", "--untracked-files=all"])
    return {
        "head": head if head_code == 0 else None,
        "head_error": None if head_code == 0 else head,
        "branch": branch if branch_code == 0 else None,
        "status_porcelain_sha256": sha256_json(status.splitlines()),
        "status_entries": len(status.splitlines()) if status_code == 0 else None,
        "repository_has_no_commit": head_code != 0,
    }


def _environment_state() -> dict[str, Any]:
    try:
        import timm

        timm_version = timm.__version__
    except Exception:
        timm_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "timm": timm_version,
        "numpy": np.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _nvidia_smi_preflight(
    output: Path, device: str, *, artifact_suffix: str = ""
) -> dict[str, Any]:
    """Capture GPU occupancy before any call that initializes torch CUDA."""

    if not str(device).startswith("cuda"):
        payload = {
            "status": "not_applicable_cpu",
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "torch_cuda_initialized_before_preflight": torch.cuda.is_initialized(),
        }
        atomic_write_json(
            output / f"nvidia_smi_preflight{artifact_suffix}.json", payload
        )
        return payload
    initialized_before = torch.cuda.is_initialized()
    if initialized_before:
        raise RuntimeError("CUDA was initialized before mandatory nvidia-smi preflight")

    def capture(arguments: list[str]) -> dict[str, Any]:
        process = subprocess.run(
            ["nvidia-smi", *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return {
            "command": ["nvidia-smi", *arguments],
            "returncode": process.returncode,
            "stdout": process.stdout,
        }

    inventory = capture(
        [
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = capture(
        [
            "--query-compute-apps=pid,gpu_uuid,used_memory,name",
            "--format=csv,noheader,nounits",
        ]
    )
    if inventory["returncode"] != 0 or processes["returncode"] != 0:
        raise RuntimeError("mandatory nvidia-smi preflight failed")
    payload = {
        "schema_version": "PHAxis-nvidia-smi-preflight-1.0",
        "status": "passed",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "internal_device": str(device),
        "torch_cuda_initialized_before_preflight": initialized_before,
        "inventory": inventory,
        "compute_processes": processes,
        "existing_processes_killed_or_suspended": False,
    }
    atomic_write_json(
        output / f"nvidia_smi_preflight{artifact_suffix}.json", payload
    )
    return payload


class _NvidiaSmiMonitor:
    def __init__(
        self,
        output: Path,
        interval_seconds: float = 5.0,
        *,
        artifact_suffix: str = "",
    ):
        self.output = output
        self.interval_seconds = float(interval_seconds)
        self.artifact_suffix = artifact_suffix
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []

    def start(self) -> None:
        if self.thread is not None:
            raise RuntimeError("GPU monitor already started")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            process = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            self.samples.append(
                {
                    "captured_utc": datetime.now(timezone.utc).isoformat(),
                    "returncode": process.returncode,
                    "stdout": process.stdout,
                }
            )
            self.stop_event.wait(self.interval_seconds)

    @staticmethod
    def _parse_rows(sample: dict[str, Any]) -> list[dict[str, float | int | str]]:
        returned = []
        if sample["returncode"] != 0:
            return returned
        for line in sample["stdout"].strip().splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 7:
                continue
            try:
                returned.append(
                    {
                        "index": int(fields[0]),
                        "uuid": fields[1],
                        "memory_used_mib": float(fields[2]),
                        "memory_total_mib": float(fields[3]),
                        "utilization_percent": float(fields[4]),
                        "temperature_c": float(fields[5]),
                        "power_w": float(fields[6]),
                    }
                )
            except ValueError:
                continue
        return returned

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(10.0, self.interval_seconds * 2))
        physical = None
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            try:
                physical = int(visible.split(",")[0].strip())
            except ValueError:
                physical = None
        parsed = [row for sample in self.samples for row in self._parse_rows(sample)]
        selected = [row for row in parsed if physical is None or row["index"] == physical]
        summary: dict[str, Any] = {
            "schema_version": "PHAxis-nvidia-smi-training-monitor-1.0",
            "status": "passed" if self.samples and all(s["returncode"] == 0 for s in self.samples) else "failed",
            "interval_seconds": self.interval_seconds,
            "cuda_visible_devices": visible,
            "physical_gpu_index": physical,
            "samples_all_gpus": len(self.samples),
            "selected_samples": len(selected),
            "samples": self.samples,
        }
        if selected:
            for field, output_field, reducer in (
                ("utilization_percent", "mean_utilization_percent", np.mean),
                ("utilization_percent", "p95_utilization_percent", lambda x: np.percentile(x, 95)),
                ("memory_used_mib", "peak_memory_used_mib", np.max),
                ("temperature_c", "peak_temperature_c", np.max),
                ("power_w", "mean_power_w", np.mean),
            ):
                summary[output_field] = float(reducer([row[field] for row in selected]))
        atomic_write_json(
            self.output
            / f"nvidia_smi_training_monitor{self.artifact_suffix}.json",
            summary,
        )
        return summary


def _rng_payload() -> dict[str, Any]:
    return {
        "python_pickle": torch.tensor(
            list(pickle.dumps(random.getstate())), dtype=torch.uint8
        ),
        "numpy_pickle": torch.tensor(
            list(pickle.dumps(np.random.get_state())), dtype=torch.uint8
        ),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(payload: dict[str, Any]) -> None:
    random.setstate(pickle.loads(bytes(payload["python_pickle"].tolist())))
    np.random.set_state(pickle.loads(bytes(payload["numpy_pickle"].tolist())))
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and payload.get("torch_cuda"):
        torch.cuda.set_rng_state_all(payload["torch_cuda"])


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_contract(
    records: Sequence[StageBImageRecord],
    dataset_audit: dict[str, Any],
    cache_audit: dict[str, Any],
    seed: int,
    config: StageBTrain399Config,
    *,
    formal: bool,
) -> dict[str, Any]:
    train_ids = [record.task_id for record in records]
    train_rows = sorted((record.task_id, record.family_key) for record in records)
    if len(train_ids) != EXPECTED_TRAIN or train_ids != dataset_audit["train_ids"]:
        raise RuntimeError("checkpoint training IDs differ from locked dataset audit")
    return {
        "schema_version": "PHAxis-StageB-train399-checkpoint-contract-1.0",
        "formal_training": bool(formal),
        "training_policy": "all399_fixed_60_epoch_last_checkpoint",
        "model_selection_policy": "none_during_training",
        "initialization_policy": "ImageNet encoder plus newly randomized decoder/heads",
        "prohibited_initialization": (
            "all RHAxiscc 443-fold checkpoints and any state exposed to locked val44"
        ),
        "seed": int(seed),
        "member_id": f"seed_{seed}",
        "training_images": EXPECTED_TRAIN,
        "train_ids": train_ids,
        "train_ids_sha256": sha256_json(train_ids),
        "training_task_ids_sha256": sha256_json(train_ids),
        "train_task_family_rows": train_rows,
        "train_task_family_sha256": sha256_json(train_rows),
        "train_families_sha256": dataset_audit["train_families_sha256"],
        "excluded_val_ids": dataset_audit["excluded_val_ids"],
        "excluded_val_ids_sha256": dataset_audit["excluded_val_ids_sha256"],
        "validation_images": len(dataset_audit["excluded_val_ids"]),
        "excluded_val_families_sha256": dataset_audit[
            "excluded_val_families_sha256"
        ],
        "dataset_split_identity_sha256": dataset_audit[
            "dataset_split_identity_sha256"
        ],
        "dataset_audit_sha256": dataset_audit.get("dataset_audit_sha256"),
        "dataset_manifest_sha256": dataset_audit["dataset_manifest_sha256"],
        "split_manifest_sha256": dataset_audit["split_manifest_sha256"],
        "integrity_manifest_sha256": dataset_audit["integrity_manifest_sha256"],
        "cache_identity_sha256": cache_audit["cache_identity_sha256"],
        "cache_audit_sha256": cache_audit.get("cache_audit_sha256"),
        "config_sha256": sha256_json(asdict(config)),
        "amp_policy": {
            "dtype": "float16",
            "enabled": bool(config.amp),
            "initial_scale": config.amp_initial_scale,
            "growth_interval": config.amp_growth_interval,
            "growth_factor": config.amp_growth_factor,
            "backoff_factor": config.amp_backoff_factor,
            "nonfinite_step_policy": "fail_closed_no_optimizer_step_skip",
        },
        "validation_labels_used_for_gradient": False,
        "validation_labels_used_for_early_stopping": False,
        "validation_labels_used_for_gradient_or_early_stopping": False,
        "validation_metrics_observed_during_training": False,
        "blind_images_used": 0,
        "pyRootHair_called_or_copied": False,
    }


def execute_with_failure_receipt(
    operation: Callable[[], Any],
    finalize_monitor: Callable[[], dict[str, Any] | None],
    *,
    output: Path,
    progress: dict[str, Any],
    preflight: dict[str, Any],
    artifact_suffix: str = "",
) -> tuple[Any, dict[str, Any] | None]:
    """Always finalize monitoring, atomically report failure, and rethrow."""

    result: Any = None
    error: BaseException | None = None
    traceback = None
    monitor_summary: dict[str, Any] | None = None
    monitor_finalize_error: str | None = None
    try:
        result = operation()
    except BaseException as caught:
        error = caught
        traceback = caught.__traceback__
    finally:
        try:
            monitor_summary = finalize_monitor()
        except BaseException as monitor_error:  # never mask the training error
            monitor_finalize_error = (
                f"{type(monitor_error).__module__}.{type(monitor_error).__name__}: "
                f"{monitor_error}"
            )
        if error is not None:
            torch_peak_allocated_mib: float | None = None
            torch_peak_reserved_mib: float | None = None
            if torch.cuda.is_initialized():
                current_device = torch.cuda.current_device()
                torch_peak_allocated_mib = (
                    torch.cuda.max_memory_allocated(current_device) / 1024**2
                )
                torch_peak_reserved_mib = (
                    torch.cuda.max_memory_reserved(current_device) / 1024**2
                )
            failure = {
                "schema_version": "PHAxis-StageB-training-failure-1.0",
                "status": "failed",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "exception_type": f"{type(error).__module__}.{type(error).__name__}",
                "exception_message": str(error),
                "completed_epoch": int(progress.get("completed_epoch", 0)),
                "global_step": int(progress.get("global_step", 0)),
                "last_finite_loss_total": progress.get("last_finite_loss_total"),
                "last_finite_loss_parts": progress.get("last_finite_loss_parts"),
                "amp_scale_before_backward": progress.get(
                    "amp_scale_before_backward"
                ),
                "amp_backward_retry_count": int(
                    progress.get("amp_backward_retry_count", 0)
                ),
                "amp_min_scale": progress.get("amp_min_scale"),
                "amp_backward_retry_events": list(
                    progress.get("amp_backward_retry_events", [])
                ),
                "optimizer_steps_skipped_due_nonfinite_gradients": 0,
                "torch_peak_allocated_mib": torch_peak_allocated_mib,
                "torch_peak_reserved_mib": torch_peak_reserved_mib,
                "nvidia_smi_preflight_status": preflight.get("status"),
                "nvidia_smi_training_monitor_status": (
                    monitor_summary.get("status") if monitor_summary else "finalize_failed"
                ),
                "monitor_finalize_error": monitor_finalize_error,
                "exception_swallowed": False,
                "blind_images_used": 0,
            }
            atomic_write_json(
                output / f"training_failure{artifact_suffix}.json", failure
            )
    if error is not None:
        raise error.with_traceback(traceback)
    if monitor_finalize_error is not None:
        raise RuntimeError(f"GPU monitor finalization failed: {monitor_finalize_error}")
    return result, monitor_summary


def _run_training_epochs(
    *,
    model: nn.Module,
    loader: DataLoader,
    sampler: DeterministicEpochSampler,
    criterion: StageBTrainingLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    config: StageBTrain399Config,
    start_epoch: int,
    global_step: int,
    history: list[dict[str, Any]],
    total_steps: int,
    warmup_steps: int,
    output: Path,
    checkpoint_path: Path,
    contract: dict[str, Any],
    initialization: dict[str, Any],
    seed: int,
    progress: dict[str, Any],
    artifact_suffix: str,
) -> tuple[list[dict[str, Any]], int]:
    for epoch in range(start_epoch, config.epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_started = time.perf_counter()
        aggregate: dict[str, float] = {}
        batches = 0
        maximum_gradient_norm = 0.0
        retry_count_at_epoch_start = int(
            progress.get("amp_backward_retry_count", 0)
        )
        epoch_minimum_amp_scale = float(scaler.get_scale())
        for batch in loader:
            learning_rate = _cosine_lr(
                global_step, total_steps, warmup_steps, config.lr
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            images = batch["image"].to(device, non_blocking=True)
            if config.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            targets = {
                name: value.to(device, non_blocking=True)
                for name, value in batch.items()
                if name != "image"
            }
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=config.amp and str(device).startswith("cuda"),
            ):
                prediction = model(images)
                loss, parts = criterion(prediction, targets)
            require_finite_loss(
                loss, parts, epoch=epoch + 1, global_step=global_step
            )
            progress["last_finite_loss_total"] = float(loss.detach())
            progress["last_finite_loss_parts"] = parts
            retry_result = backward_with_loss_scale_retries(
                loss,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch + 1,
                global_step=global_step,
                progress=progress,
            )
            epoch_minimum_amp_scale = min(
                epoch_minimum_amp_scale,
                float(retry_result["minimum_scale"]),
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            if not bool(torch.isfinite(gradient_norm).item()):
                raise FloatingPointError(
                    f"non-finite Stage-B gradient at epoch={epoch + 1} step={global_step}"
                )
            maximum_gradient_norm = max(
                maximum_gradient_norm, float(gradient_norm.detach())
            )
            scaler.step(optimizer)
            scaler.update()
            # backward_with_loss_scale_retries retains the graph so the exact
            # same forward can be replayed after overflow.  Release it before
            # constructing the next batch graph.
            del prediction, loss
            for name, value in parts.items():
                aggregate[name] = aggregate.get(name, 0.0) + value
            batches += 1
            global_step += 1
            progress["global_step"] = global_step
        elapsed = time.perf_counter() - epoch_started
        means = {
            name: value / max(batches, 1) for name, value in aggregate.items()
        }
        epoch_record = {
            "epoch": epoch + 1,
            "batches": batches,
            "global_step": global_step,
            "learning_rate_last": learning_rate,
            "wall_seconds": elapsed,
            "train_loss_parts": means,
            "train_loss_total": sum(means.values()),
            "maximum_unscaled_gradient_norm": maximum_gradient_norm,
            "amp_backward_retry_count": int(
                progress.get("amp_backward_retry_count", 0)
                - retry_count_at_epoch_start
            ),
            "amp_minimum_scale": epoch_minimum_amp_scale,
            "amp_scale_end": float(scaler.get_scale()),
            "optimizer_steps_skipped_due_nonfinite_gradients": 0,
            "validation_evaluated": False,
        }
        history.append(epoch_record)
        checkpoint_payload = {
            "schema_version": "PHAxis-StageB-train399-checkpoint-1.0",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": asdict(config),
            "contract": contract,
            "initialization": initialization,
            "initialization_sha256": initialization["initialization_sha256"],
            "seed": int(seed),
            "member_id": f"seed_{seed}",
            "training_images": EXPECTED_TRAIN,
            "training_task_ids_sha256": contract["training_task_ids_sha256"],
            "split_manifest_sha256": contract["split_manifest_sha256"],
            "validation_images": contract["validation_images"],
            "validation_labels_used_for_gradient_or_early_stopping": False,
            "epoch": epoch + 1,
            "global_step": global_step,
            "history": history,
            "amp_backward_retry_events": list(
                progress.get("amp_backward_retry_events", [])
            ),
            "rng": _rng_payload(),
        }
        _atomic_torch_save(checkpoint_path, checkpoint_payload)
        # The atomic checkpoint is authoritative. If power is lost before this
        # sidecar replace, resume reconstructs the sidecar from embedded history.
        atomic_write_json(output / "history.json", history)
        retry_events = list(progress.get("amp_backward_retry_events", []))
        atomic_write_json(
            output / f"amp_backward_retries{artifact_suffix}.json",
            {
                "schema_version": "PHAxis-StageB-AMP-backward-retry-audit-1.0",
                "status": "completed_through_epoch",
                "seed": int(seed),
                "completed_epoch": epoch + 1,
                "event_count": len(retry_events),
                "events": retry_events,
                "same_forward_graph_replayed": True,
                "optimizer_steps_skipped_due_nonfinite_gradients": 0,
                "blind_images_used": 0,
            },
        )
        progress["completed_epoch"] = epoch + 1
        peak_mib = (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if str(device).startswith("cuda")
            else 0.0
        )
        print(
            f"seed {seed} epoch {epoch + 1:02d}/{config.epochs} "
            f"{elapsed:.1f}s loss={epoch_record['train_loss_total']:.5f} "
            f"lr={learning_rate:.3e} peak={peak_mib:.0f}MiB",
            flush=True,
        )
    return history, global_step


def train_one_seed(
    *,
    records: Sequence[StageBImageRecord],
    dataset_audit: dict[str, Any],
    cache_audit: dict[str, Any],
    cache_root: str | Path,
    run_dir: str | Path,
    project_root: str | Path,
    seed: int,
    device: str,
    config: StageBTrain399Config,
    formal: bool,
    resume: bool,
) -> dict[str, Any]:
    """Train one independent seed.  The function never constructs val data."""

    if formal and (
        config.epochs != 60
        or config.crops_per_image != 8
        or not config.fixed_last_epoch_policy
    ):
        raise RuntimeError("formal Stage-B members require fixed 60 epochs and 8 crops/image")
    if len(records) != EXPECTED_TRAIN or any(record.split != "train" for record in records):
        raise RuntimeError("gradient input is not the locked train399 set")
    if formal:
        cache_methods = {
            entry.get("materialization") for entry in cache_audit.get("entries", [])
        }
        if cache_methods != {"canonical_image_decode_and_resample"}:
            raise RuntimeError(
                "formal Stage-B training requires a cache rebuilt directly from canonical images"
            )
        if not dataset_audit.get("dataset_audit_sha256"):
            raise RuntimeError("formal Stage-B training requires a hash-locked dataset audit")
        if not cache_audit.get("cache_audit_sha256"):
            raise RuntimeError("formal Stage-B training requires a hash-locked cache audit")
    output = Path(run_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "last.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {checkpoint_path}")

    artifact_suffix = ""
    if resume:
        resume_index = 1
        while (
            output / f"nvidia_smi_preflight_resume_{resume_index:03d}.json"
        ).exists():
            resume_index += 1
        artifact_suffix = f"_resume_{resume_index:03d}"
    preflight = _nvidia_smi_preflight(
        output, device, artifact_suffix=artifact_suffix
    )
    _set_reproducibility(seed, config.deterministic_algorithms)
    contract = _checkpoint_contract(
        records, dataset_audit, cache_audit, seed, config, formal=formal
    )
    atomic_write_json(output / "training_contract.json", contract)
    atomic_write_json(output / "config.json", asdict(config))
    atomic_write_json(output / "git_state.json", _git_state(Path(project_root).resolve()))
    atomic_write_json(output / "environment.json", _environment_state())

    dataset = Train399HairCropDataset(
        records,
        cache_root,
        crop=config.crop,
        out_stride=config.out_stride,
        target_um_per_px=config.um_per_px,
        crops_per_image=config.crops_per_image,
        background_fraction=config.background_fraction,
        input_channels=config.in_channels,
        base_sigma_um=config.base_sigma_um,
        tip_sigma_um=config.tip_sigma_um,
        line_halfwidth_um=config.line_halfwidth_um,
        seed=seed,
    )
    sampler = DeterministicEpochSampler(len(dataset), seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader_arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "sampler": sampler,
        "num_workers": config.workers,
        "pin_memory": str(device).startswith("cuda"),
        "drop_last": True,
        "persistent_workers": config.workers > 0,
        "worker_init_fn": deterministic_worker_init,
        "generator": loader_generator,
    }
    if config.workers > 0:
        loader_arguments["prefetch_factor"] = config.prefetch_factor
    loader = DataLoader(**loader_arguments)
    steps_per_epoch = len(loader)

    start_epoch = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    checkpoint: dict[str, Any] | None = None
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["contract"] != contract:
            raise RuntimeError("resume checkpoint contract differs from current lock")
        expected_top_level = {
            "seed": int(seed),
            "member_id": f"seed_{seed}",
            "training_images": EXPECTED_TRAIN,
            "training_task_ids_sha256": contract["training_task_ids_sha256"],
            "split_manifest_sha256": contract["split_manifest_sha256"],
            "validation_images": contract["validation_images"],
            "validation_labels_used_for_gradient_or_early_stopping": False,
        }
        for field, expected in expected_top_level.items():
            if checkpoint.get(field) != expected:
                raise RuntimeError(f"resume checkpoint mismatch at {field}")
        if checkpoint.get("cfg") != asdict(config):
            raise RuntimeError("resume checkpoint config differs from current config")
        initialization = dict(checkpoint["initialization"])
        validate_initialization_identity(
            initialization, str(checkpoint.get("initialization_sha256"))
        )
        existing_initialization = json.loads(
            (output / "initialization.json").read_text(encoding="utf-8")
        )
        if existing_initialization != initialization:
            raise RuntimeError("resume initialization.json differs from checkpoint")
        model = MultiHeadUNet(
            HEADS,
            encoder=config.encoder,
            in_channels=config.in_channels,
            out_stride=config.out_stride,
            decoder_channels=config.decoder_channels,
            pretrained=False,
            context=config.context,
            stem_stride1=config.stem_stride1,
        )
        model.load_state_dict(checkpoint["model"])
    else:
        model = MultiHeadUNet(
            HEADS,
            encoder=config.encoder,
            in_channels=config.in_channels,
            out_stride=config.out_stride,
            decoder_channels=config.decoder_channels,
            pretrained=True,
            context=config.context,
            stem_stride1=config.stem_stride1,
        )
        initial_encoder_sha256 = _state_dict_sha256(model.encoder.state_dict())
        initial_model_sha256 = _state_dict_sha256(model.state_dict())
        imagenet = _discover_imagenet_source()
        if (
            imagenet.get("source") != config.imagenet_source
            or not imagenet.get("huggingface_revision")
            or not imagenet.get("cached_weight_sha256")
            or not imagenet.get("cached_weight_size_bytes")
        ):
            raise RuntimeError("ImageNet initialization provenance is incomplete")
        initialization = {
            **imagenet,
            "initial_encoder_state_sha256": initial_encoder_sha256,
            "initial_complete_model_state_sha256": initial_model_sha256,
            "historical_stageb_checkpoint_loaded": False,
        }
        initialization["initialization_sha256"] = sha256_json(initialization)
        atomic_write_json(output / "initialization.json", initialization)

    model = model.to(device)
    if config.channels_last:
        model = model.to(memory_format=torch.channels_last)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = StageBTrainingLoss(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=config.amp and str(device).startswith("cuda"),
        init_scale=config.amp_initial_scale,
        growth_interval=config.amp_growth_interval,
        growth_factor=config.amp_growth_factor,
        backoff_factor=config.amp_backoff_factor,
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        sidecar_history = json.loads(
            (output / "history.json").read_text(encoding="utf-8")
        )
        history, history_recovered = resolve_resume_history(
            checkpoint,
            sidecar_history,
            steps_per_epoch=steps_per_epoch,
            configured_epochs=config.epochs,
        )
        start_epoch, global_step = validate_resume_progress(
            checkpoint,
            history,
            steps_per_epoch=steps_per_epoch,
            configured_epochs=config.epochs,
        )
        if history_recovered:
            atomic_write_json(output / "history.json", history)
            atomic_write_json(
                output / f"history_recovery{artifact_suffix}.json",
                {
                    "schema_version": "PHAxis-StageB-history-recovery-1.0",
                    "status": "recovered_from_atomic_checkpoint",
                    "checkpoint_epoch": start_epoch,
                    "checkpoint_global_step": global_step,
                    "sidecar_rows_before": len(sidecar_history),
                    "sidecar_rows_after": len(history),
                    "checkpoint_embedded_history_available": (
                        checkpoint.get("history") is not None
                    ),
                },
            )
        _restore_rng(checkpoint["rng"])
        print(f"[resume] seed={seed} at completed epoch {start_epoch}", flush=True)

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs
    started = time.perf_counter()
    gpu_monitor: _NvidiaSmiMonitor | None = None
    if str(device).startswith("cuda"):
        gpu_monitor = _NvidiaSmiMonitor(output, artifact_suffix=artifact_suffix)
        gpu_monitor.start()
    prior_retry_events = (
        list(checkpoint.get("amp_backward_retry_events", []))
        if checkpoint is not None
        else []
    )
    progress = {
        "completed_epoch": start_epoch,
        "global_step": global_step,
        "amp_backward_retry_events": prior_retry_events,
        "amp_backward_retry_count": len(prior_retry_events),
        "amp_min_scale": float(scaler.get_scale()),
    }

    def operation() -> tuple[list[dict[str, Any]], int]:
        return _run_training_epochs(
            model=model,
            loader=loader,
            sampler=sampler,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            start_epoch=start_epoch,
            global_step=global_step,
            history=history,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            output=output,
            checkpoint_path=checkpoint_path,
            contract=contract,
            initialization=initialization,
            seed=seed,
            progress=progress,
            artifact_suffix=artifact_suffix,
        )

    (history, global_step), gpu_monitor_summary = execute_with_failure_receipt(
        operation,
        (gpu_monitor.stop if gpu_monitor is not None else lambda: None),
        output=output,
        progress=progress,
        preflight=preflight,
        artifact_suffix=artifact_suffix,
    )
    total_wall = time.perf_counter() - started
    checkpoint_sha256 = sha256_file(checkpoint_path)
    retry_audit_path = output / f"amp_backward_retries{artifact_suffix}.json"
    if not retry_audit_path.is_file():
        raise RuntimeError("AMP backward retry audit was not written")
    receipt = {
        "schema_version": "PHAxis-StageB-train399-training-receipt-1.0",
        "status": "completed",
        "formal_training": bool(formal),
        "seed": int(seed),
        "epochs": config.epochs,
        "steps_per_epoch": steps_per_epoch,
        "global_steps": global_step,
        "parameter_count": parameter_count,
        "total_wall_seconds_this_invocation": total_wall,
        "median_epoch_wall_seconds": float(
            np.median([record["wall_seconds"] for record in history])
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if str(device).startswith("cuda")
            else 0.0
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if str(device).startswith("cuda")
            else 0.0
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "internal_device": str(device),
        "physical_device_mapping_note": (
            "cuda:0 maps to the first entry of CUDA_VISIBLE_DEVICES"
        ),
        "nvidia_smi_preflight_status": preflight["status"],
        "nvidia_smi_training_monitor_status": (
            gpu_monitor_summary["status"] if gpu_monitor_summary else None
        ),
        "amp_backward_retry_count": int(
            progress.get("amp_backward_retry_count", 0)
        ),
        "amp_min_scale": float(progress.get("amp_min_scale", scaler.get_scale())),
        "amp_final_scale": float(scaler.get_scale()),
        "amp_backward_retry_audit": str(retry_audit_path),
        "amp_backward_retry_audit_sha256": sha256_file(retry_audit_path),
        "optimizer_steps_skipped_due_nonfinite_gradients": 0,
        "validation_evaluated_during_training": False,
        "blind_images_used": 0,
        "invocation_artifact_suffix": artifact_suffix,
    }
    if str(device).startswith("cuda"):
        receipt["gpu_name"] = torch.cuda.get_device_name(device)
    atomic_write_json(
        output / f"training_receipt{artifact_suffix}.json", receipt
    )
    return receipt
