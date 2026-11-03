"""Shared data loading, preprocessing, CNN, and metric helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from module2_preprocessing.preprocessing import augment_intensity, normalize_resize_roi


CLASS_NAMES = ("Healthy", "Partial tear", "Complete rupture")
SPLIT_NAMES = {"train", "validation", "test"}


def read_split(path: Path, volumes_dir: Path) -> list[dict[str, Any]]:
    """Read split manifest and enforce labels, paths, and exam-level isolation."""
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"examId", "aclDiagnosis", "volumeFilename", "split", "roiX", "roiY", "roiZ", "roiHeight", "roiWidth", "roiDepth"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Split file must contain {sorted(required)}; found {reader.fieldnames}")
        rows = []
        for source in reader:
            label = int(source["aclDiagnosis"])
            split = source["split"].strip()
            if label not in (0, 1, 2):
                raise ValueError(f"Expected aclDiagnosis 0, 1, or 2; got {label}")
            if split not in SPLIT_NAMES:
                raise ValueError(f"Unexpected split name {split!r}")
            volume_path = volumes_dir / source["volumeFilename"]
            if not volume_path.is_file():
                raise FileNotFoundError(f"Missing MRI volume: {volume_path}")
            row = dict(source)
            row.update({"label": label, "split": split, "path": volume_path})
            for key in ("roiX", "roiY", "roiZ", "roiHeight", "roiWidth", "roiDepth"):
                row[key] = int(source[key])
            rows.append(row)
    group_to_split: dict[str, str] = {}
    for row in rows:
        group = row["examId"].strip()
        previous = group_to_split.setdefault(group, row["split"])
        if previous != row["split"]:
            raise ValueError(f"Exam {group} occurs in multiple splits ({previous}, {row['split']})")
    if not rows:
        raise ValueError(f"No MRI rows found in {path}")
    return rows


def load_or_create_roi_cache(row: dict[str, Any], cache_dir: Path, image_size: int) -> Path:
    """Crop annotated ROI slices, normalize, resize, and cache as uint8 NPY."""
    cache_path = cache_dir / f"{row['volumeFilename']}.size{image_size}.npy"
    if cache_path.exists():
        return cache_path
    # Dataset files are pickle objects from the official archive.
    with row["path"].open("rb") as stream:
        import pickle
        volume = np.asarray(pickle.load(stream))
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume in {row['volumeFilename']}; got {volume.shape}")
    processed = normalize_resize_roi(volume, row, image_size)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, processed, allow_pickle=False)
    temporary.replace(cache_path)
    return cache_path


class MRIVolumes(Dataset):
    """One item is one MRI volume with a fixed number of ROI slices."""

    def __init__(self, rows: list[dict[str, Any]], split: str, cache_dir: Path, image_size: int, slices_per_volume: int, augmentation: bool = False):
        self.rows = [row for row in rows if row["split"] == split]
        self.split = split
        self.cache_dir = cache_dir
        self.image_size = image_size
        self.slices_per_volume = slices_per_volume
        self.augmentation = augmentation
        if not self.rows:
            raise ValueError(f"No rows found for {split} split")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        cache_path = load_or_create_roi_cache(row, self.cache_dir, self.image_size)
        volume_slices = np.load(cache_path, allow_pickle=False)
        n_slices = len(volume_slices)
        if self.split == "train":
            selected = np.random.randint(0, n_slices, size=self.slices_per_volume)
        else:
            n_selected = min(n_slices, self.slices_per_volume)
            selected = np.unique(np.linspace(0, n_slices - 1, n_selected, dtype=int))
            if len(selected) < self.slices_per_volume:
                selected = np.resize(selected, self.slices_per_volume)
        images = torch.from_numpy(volume_slices[selected].copy()).float().unsqueeze(1) / 255.0
        if self.split == "train" and self.augmentation:
            images = augment_intensity(images)
        return images, torch.tensor(row["label"], dtype=torch.long), row["volumeFilename"]


class SmallCNN(nn.Module):
    """Compact scratch CNN accepting normalized, single-channel ROI slices."""

    def __init__(self, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1, bias=False), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(192, 3))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


class RGBTransferModel(nn.Module):
    """Wrap an ImageNet model with grayscale replication and ImageNet scaling."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        rgb = images.repeat(1, 3, 1, 1)
        rgb = (rgb - self.mean) / self.std
        return self.backbone(rgb)


def build_model(architecture: str, pretrained: bool = False, dropout: float = 0.35) -> nn.Module:
    """Build the scratch CNN or a torchvision transfer-learning classifier."""
    if architecture == "small_cnn":
        if pretrained:
            raise ValueError("The small_cnn architecture has no pretrained weights")
        return SmallCNN(dropout=dropout)
    try:
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("Transfer learning requires torchvision. Install a torchvision build compatible with your PyTorch version.") from exc

    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        backbone.fc = nn.Linear(backbone.fc.in_features, 3)
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        backbone.classifier[-1] = nn.Linear(backbone.classifier[-1].in_features, 3)
    elif architecture == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, 3)
    else:
        raise ValueError(f"Unknown architecture {architecture!r}")
    return RGBTransferModel(backbone)


def predict_volumes(model: nn.Module, loader: Any, device: torch.device) -> tuple[list[int], list[int], list[str], list[list[float]]]:
    """Aggregate per-slice logits into per-volume probabilities and predictions."""
    model.eval()
    truths: list[int] = []
    predictions: list[int] = []
    filenames: list[str] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for images, labels, names in loader:
            batch, slices, channels, height, width = images.shape
            images = images.reshape(batch * slices, channels, height, width).to(device)
            slice_logits = model(images).reshape(batch, slices, 3)
            volume_probabilities = torch.softmax(slice_logits.mean(dim=1), dim=1)
            predictions.extend(volume_probabilities.argmax(dim=1).cpu().tolist())
            probabilities.extend(volume_probabilities.cpu().tolist())
            truths.extend(labels.tolist())
            filenames.extend(names)
    return truths, predictions, filenames, probabilities


def metrics_for(truths: list[int], predictions: list[int]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

    return {
        "accuracy": float(accuracy_score(truths, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truths, predictions)),
        "macro_f1": float(f1_score(truths, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truths, predictions, labels=[0, 1, 2]).tolist(),
        "per_class": classification_report(truths, predictions, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0),
    }
