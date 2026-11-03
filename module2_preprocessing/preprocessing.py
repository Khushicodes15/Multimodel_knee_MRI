"""Simple preprocessing helpers shared by the CNN training pipeline."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def crop_roi(volume: np.ndarray, row: dict) -> np.ndarray:
    """Crop the annotated [z, y, x] region from a 3D volume."""
    x, y, z = (int(row[k]) for k in ("roiX", "roiY", "roiZ"))
    width, height, depth = (int(row[k]) for k in ("roiWidth", "roiHeight", "roiDepth"))
    roi = volume[z:z + depth, y:y + height, x:x + width]
    if roi.ndim != 3 or min(roi.shape) <= 0:
        raise ValueError(f"Invalid ROI for {row.get('volumeFilename', 'volume')}: {roi.shape}")
    return roi


def normalize_resize_roi(volume: np.ndarray, row: dict, image_size: int = 160) -> np.ndarray:
    """Return normalized ROI slices as uint8 images of a fixed size."""
    roi = crop_roi(volume, row).astype(np.float32)
    low, high = np.percentile(roi, (1, 99))
    if high > low:
        roi = np.clip((roi - low) / (high - low), 0, 1)
    else:
        roi = np.zeros_like(roi)
    result = []
    for slice_ in roi:
        image = Image.fromarray(slice_.astype(np.float32), mode="F")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        result.append(np.rint(np.asarray(image).clip(0, 1) * 255).astype(np.uint8))
    return np.stack(result)


def augment_intensity(images: torch.Tensor) -> torch.Tensor:
    """Apply mild brightness/contrast/noise changes; preserve MRI orientation."""
    if torch.rand(()) < 0.5:
        brightness = 0.9 + 0.2 * torch.rand((), device=images.device)
        contrast = 0.9 + 0.2 * torch.rand((), device=images.device)
        mean = images.mean(dim=(-2, -1), keepdim=True)
        images = ((images - mean) * contrast + mean) * brightness
    if torch.rand(()) < 0.3:
        images = images + torch.randn_like(images) * 0.015
    return images.clamp(0, 1)
