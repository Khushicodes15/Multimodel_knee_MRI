"""Train a 3-class GRU sequence classifier from ordered sagittal MRI slices.

Module 4 predicts the ACL diagnosis for ONE MRI volume:
0 = Healthy, 1 = Partial tear, 2 = Complete rupture.

Slice embeddings are extracted with the project's trained ResNet-18 checkpoint.
The checkpoint is loaded strictly using the same model builder as Module 4 inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from acl_mri_classification.model import build_model
from module2_preprocessing.preprocessing import normalize_resize_roi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_gru")

CLASSES = ["Healthy", "Partial tear", "Complete rupture"]
NUM_CLASSES = 3
FEATURE_SIZE = 512


# ---------------------------------------------------------------------------
# GRU architecture
# ---------------------------------------------------------------------------

class GRUSliceSequenceClassifier(nn.Module):
    """Classify one ordered sequence of per-slice feature vectors."""

    def __init__(
        self,
        input_size: int = FEATURE_SIZE,
        hidden_size: int = 64,
        num_classes: int = NUM_CLASSES,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x shape: [batch, padded_num_slices, feature_size]
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
            final_hidden = hidden[-1]
        else:
            _, hidden = self.gru(x)
            final_hidden = hidden[-1]
        return self.classifier(final_hidden)


# ---------------------------------------------------------------------------
# ResNet-18 slice feature extraction
# ---------------------------------------------------------------------------

class ResNet18SliceFeatureExtractor(nn.Module):
    """Extract 512-dimensional slice embeddings using the trained project ResNet."""

    def __init__(
        self,
        weights_path: Path | str,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        weights_path = Path(weights_path)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"Required trained ResNet-18 checkpoint not found: {weights_path}. "
                "Refusing to silently use random or ImageNet-only weights."
            )

        logger.info("Loading trained ResNet-18 checkpoint: %s", weights_path)
        checkpoint = torch.load(
            weights_path,
            map_location=self.device,
            weights_only=False,
        )
        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "ResNet checkpoint does not contain 'model_state_dict'. "
                f"Found keys: {list(checkpoint.keys())}"
            )

        # IMPORTANT: use the same model builder and checkpoint structure as the
        # production Module 4 adapter. strict=True prevents silent weight mismatch.
        resnet_model = build_model("resnet18", pretrained=False).to(self.device)
        resnet_model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        if not hasattr(resnet_model, "backbone"):
            raise AttributeError(
                "build_model('resnet18') does not expose a .backbone attribute."
            )

        self.backbone = resnet_model.backbone
        mean = getattr(
            resnet_model,
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        std = getattr(
            resnet_model,
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())

        self.to(self.device)
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

        logger.info("ResNet-18 checkpoint loaded strictly; slice embeddings enabled.")

    @torch.no_grad()
    def extract_volume_slices(
        self,
        volume: np.ndarray,
        row: dict[str, Any],
        image_size: int = 160,
    ) -> torch.Tensor:
        """Return ordered features with shape [num_slices, 512]."""
        norm_slices = normalize_resize_roi(
            volume,
            row,
            image_size=image_size,
        )
        batch = (
            torch.from_numpy(norm_slices.copy())
            .float()
            .unsqueeze(1)
            .to(self.device)
            / 255.0
        )
        rgb = batch.repeat(1, 3, 1, 1)
        rgb = (rgb - self.mean) / self.std

        # This is the same ResNet backbone and normalization used by the adapter.
        x = self.backbone.conv1(rgb)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)

        if features.ndim != 2 or features.shape[1] != FEATURE_SIZE:
            raise RuntimeError(
                f"Expected [num_slices, {FEATURE_SIZE}] features; "
                f"received {tuple(features.shape)}"
            )
        return features.cpu()


# ---------------------------------------------------------------------------
# Dataset, split loading, and feature cache
# ---------------------------------------------------------------------------

class VolumeSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[torch.Tensor],
        labels: list[int],
        filenames: list[str],
    ) -> None:
        self.sequences = sequences
        self.labels = labels
        self.filenames = filenames

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        sequence = self.sequences[index]
        return sequence, self.labels[index], len(sequence)


def collate_sequences(
    batch: list[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels, lengths = zip(*batch)
    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0,
    )
    return (
        padded,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )


def load_splits(
    splits_path: Path,
    volumes_dir: Path,
) -> list[dict[str, Any]]:
    with splits_path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    existing: list[dict[str, Any]] = []
    for row in rows:
        volume_path = volumes_dir / row["volumeFilename"]
        if not volume_path.is_file():
            continue

        item = dict(row)
        item["path"] = volume_path
        item["label"] = int(row["aclDiagnosis"])
        existing.append(item)

    return existing


def extract_or_load_features(
    rows: list[dict[str, Any]],
    cache_path: Path,
    resnet_weights: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load a cache or extract missing sequences using the verified ResNet."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.is_file():
        logger.info("Loading slice-feature cache: %s", cache_path)
        cache = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(cache, dict):
            raise TypeError("Slice-feature cache must contain a dictionary.")
    else:
        cache = {}

    missing_rows = [
        row for row in rows
        if row["volumeFilename"] not in cache
    ]

    if missing_rows:
        extractor = ResNet18SliceFeatureExtractor(
            weights_path=resnet_weights,
            device=device,
        )
        logger.info(
            "Extracting features for %d volumes with trained ResNet-18...",
            len(missing_rows),
        )

        for index, row in enumerate(missing_rows, start=1):
            with row["path"].open("rb") as file:
                volume = np.asarray(pickle.load(file), dtype=np.float32)

            sequence_features = extractor.extract_volume_slices(volume, row)
            if sequence_features.shape[0] == 0:
                raise RuntimeError(
                    f"No slices were produced for {row['volumeFilename']}"
                )
            if not torch.isfinite(sequence_features).all():
                raise RuntimeError(
                    f"Non-finite slice features in {row['volumeFilename']}"
                )

            cache[row["volumeFilename"]] = sequence_features

            if index % 50 == 0 or index == len(missing_rows):
                logger.info(
                    "Extracted %d/%d volumes",
                    index,
                    len(missing_rows),
                )

        torch.save(cache, cache_path)
        logger.info("Saved slice-feature cache: %s", cache_path)

    # Validate cached entries as well; stale or malformed entries should fail early.
    for row in rows:
        key = row["volumeFilename"]
        if key not in cache:
            raise KeyError(f"Missing cached features for {key}")
        value = cache[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32)
            cache[key] = value
        if value.ndim != 2 or value.shape[1] != FEATURE_SIZE:
            raise ValueError(
                f"Invalid feature shape for {key}: {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite cached features for {key}")

    return cache


# ---------------------------------------------------------------------------
# Evaluation and training
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    targets_all: list[int] = []

    with torch.no_grad():
        for sequences, targets, lengths in loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            logits = model(sequences, lengths)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets_all.extend(targets.tolist())

    if not targets_all:
        return {}

    y_true = np.asarray(targets_all)
    y_pred = np.asarray(predictions)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2],
        ).tolist(),
        "per_class": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_gru(
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    test_loader: DataLoader | None,
    hidden_size: int = 64,
    num_layers: int = 1,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
) -> tuple[GRUSliceSequenceClassifier, dict[str, Any]]:
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = GRUSliceSequenceClassifier(
        input_size=FEATURE_SIZE,
        hidden_size=hidden_size,
        num_classes=NUM_CLASSES,
        num_layers=num_layers,
    ).to(device)

    training_labels: list[int] = []
    for _, batch_labels, _ in train_loader:
        training_labels.extend(batch_labels.tolist())

    counts = np.bincount(training_labels, minlength=NUM_CLASSES)
    if np.any(counts == 0):
        raise ValueError(
            "Training split must contain all 3 classes. "
            f"Class counts are {counts.tolist()}."
        )

    class_weights = torch.tensor(
        [
            len(training_labels) / (NUM_CLASSES * float(count))
            for count in counts
        ],
        dtype=torch.float32,
        device=device,
    )
    logger.info("Training class counts: %s", counts.tolist())
    logger.info("Loss class weights: %s", class_weights.cpu().tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    logger.info(
        "Training GRU: epochs=%d hidden=%d device=%s",
        epochs,
        hidden_size,
        device,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for sequences, targets, lengths in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(sequences, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()

        if val_loader is not None:
            validation_metrics = evaluate_model(
                model,
                val_loader,
                device,
            )
            current_f1 = validation_metrics["macro_f1"]

            if current_f1 > best_macro_f1:
                best_macro_f1 = current_f1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

            if epoch % 5 == 0 or epoch == epochs:
                logger.info(
                    "Epoch %02d/%02d | loss %.4f | val macro-F1 %.4f | "
                    "val balanced acc %.4f",
                    epoch,
                    epochs,
                    total_loss / max(len(train_loader), 1),
                    current_f1,
                    validation_metrics["balanced_accuracy"],
                )
        else:
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            if epoch % 5 == 0 or epoch == epochs:
                logger.info(
                    "Epoch %02d/%02d | loss %.4f",
                    epoch,
                    epochs,
                    total_loss / max(len(train_loader), 1),
                )

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics: dict[str, Any] = {}
    if val_loader is not None:
        metrics["validation"] = evaluate_model(
            model,
            val_loader,
            device,
        )
    if test_loader is not None:
        metrics["test"] = evaluate_model(
            model,
            test_loader,
            device,
        )
        logger.info(
            "Test accuracy %.4f | balanced accuracy %.4f | macro-F1 %.4f",
            metrics["test"]["accuracy"],
            metrics["test"]["balanced_accuracy"],
            metrics["test"]["macro_f1"],
        )

    return model, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        type=Path,
        default=ROOT / "acl_mri_classification/artifacts/splits.csv",
    )
    parser.add_argument(
        "--volumes-dir",
        type=Path,
        default=ROOT / "data/KneeMRI/volumetric_data",
    )
    parser.add_argument(
        "--resnet-weights",
        type=Path,
        default=ROOT / "acl_mri_classification/artifacts/deploy/resnet18.pt",
    )
    parser.add_argument(
        "--cache-features",
        type=Path,
        default=ROOT / "module4_sequence/artifacts/slice_features_cache.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "module4_sequence/artifacts",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not args.splits.is_file():
        raise FileNotFoundError(f"Splits CSV not found: {args.splits}")
    if not args.resnet_weights.is_file():
        raise FileNotFoundError(
            f"Trained ResNet-18 checkpoint not found: {args.resnet_weights}"
        )
    if not args.volumes_dir.is_dir():
        raise FileNotFoundError(
            f"MRI volumes directory not found: {args.volumes_dir}"
        )

    rows = load_splits(args.splits, args.volumes_dir)
    logger.info(
        "Found %d accessible MRI volumes in %s",
        len(rows),
        args.volumes_dir,
    )
    if not rows:
        raise RuntimeError("No accessible MRI volumes were found.")

    feature_cache = extract_or_load_features(
        rows=rows,
        cache_path=args.cache_features,
        resnet_weights=args.resnet_weights,
        device=device,
    )

    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [
        row for row in rows if row["split"] == "validation"
    ]
    test_rows = [row for row in rows if row["split"] == "test"]

    if not train_rows:
        raise RuntimeError(
            "No training rows were found. Refusing to train on validation/test "
            "data as a fallback."
        )

    def make_dataset(
        subset: list[dict[str, Any]],
    ) -> VolumeSequenceDataset:
        return VolumeSequenceDataset(
            sequences=[
                feature_cache[row["volumeFilename"]]
                for row in subset
            ],
            labels=[row["label"] for row in subset],
            filenames=[row["volumeFilename"] for row in subset],
        )

    train_dataset = make_dataset(train_rows)
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_sequences,
    )

    validation_loader = None
    if validation_rows:
        validation_dataset = make_dataset(validation_rows)
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=min(args.batch_size, len(validation_dataset)),
            shuffle=False,
            collate_fn=collate_sequences,
        )

    test_loader = None
    if test_rows:
        test_dataset = make_dataset(test_rows)
        test_loader = DataLoader(
            test_dataset,
            batch_size=min(args.batch_size, len(test_dataset)),
            shuffle=False,
            collate_fn=collate_sequences,
        )

    model, metrics = train_gru(
        train_loader=train_loader,
        val_loader=validation_loader,
        test_loader=test_loader,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "gru_checkpoint.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "input_size": FEATURE_SIZE,
            "hidden_size": args.hidden_size,
            "num_classes": NUM_CLASSES,
            "num_layers": 1,
        },
        "classes": CLASSES,
        "metrics": metrics,
        "seed": args.seed,
        "note": (
            "GRU sequence classifier over ordered ResNet-18 slice embeddings "
            "from one MRI volume; 3-class ACL diagnosis."
        ),
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info("Saved GRU checkpoint: %s", checkpoint_path)

    metrics_path = args.output_dir / f"metrics_seed_{args.seed}.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
