"""Physical-scale normalisation and engineered grayscale channels."""

from __future__ import annotations

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] >= 3:
        array = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    elif array.ndim == 3:
        array = array[..., 0]
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D image after grayscale conversion, got {array.shape}")
    if array.dtype != np.uint8:
        lower, upper = np.percentile(array, [0.1, 99.9])
        array = np.clip((array - lower) / max(float(upper - lower), 1e-6) * 255.0, 0, 255)
    return array.astype(np.uint8)


def resample_to_physical_scale(
    gray: np.ndarray, source_um_per_px: float, target_um_per_px: float
) -> tuple[np.ndarray, float]:
    scale = float(source_um_per_px) / float(target_um_per_px)
    if abs(scale - 1.0) < 1e-3:
        return np.asarray(gray).copy(), 1.0
    height, width = gray.shape
    output_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(gray, output_size, interpolation=interpolation), scale


def make_input_channels(
    gray: np.ndarray, um_per_px: float, n_channels: int = 3
) -> np.ndarray:
    intensity = gray.astype(np.float32)
    median = float(np.median(intensity))
    iqr = float(np.percentile(intensity, 84) - np.percentile(intensity, 16)) or 1.0
    normalized = (intensity - median) / (iqr * 2.0)
    background_sigma = max(2.0, 40.0 / um_per_px)
    background = cv2.GaussianBlur(intensity, (0, 0), background_sigma)
    local_contrast = (intensity - background) / iqr
    channels = [normalized, local_contrast]
    if n_channels >= 3:
        sigma_a = max(0.8, 2.5 / um_per_px)
        sigma_b = max(1.6, 7.0 / um_per_px)
        dark_ridge = (
            cv2.GaussianBlur(intensity, (0, 0), sigma_b)
            - cv2.GaussianBlur(intensity, (0, 0), sigma_a)
        ) / (iqr * 0.5)
        channels.append(dark_ridge)
    output = np.stack(channels[:n_channels]).astype(np.float32)
    return np.tanh(output / 4.0).astype(np.float32) * 4.0

