"""Train GRU Sequence Classifier for 3-Class ACL Diagnosis from Ordered MRI Slices.

Module 4 sequence model:
- Operates on ordered sagittal slices from a SINGLE MRI volume.
- Slices are passed through ResNet-18 to obtain 512-dim slice feature embeddings.
- A GRU network models the slice sequence to predict volume-level ACL diagnosis:
  0: Healthy, 1: Partial tear, 2: Complete rupture.
- Saves trained checkpoint to module4_recovery/artifacts/gru_checkpoint.pt
  for seamless loading by adapters.module4_adapter.Module4Adapter.
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
from torchvision import models

from module2_preprocessing.preprocessing import normalize_resize_roi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_gru")

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Healthy", "Partial tear", "Complete rupture"]


# ---------------------------------------------------------------------------
# Architecture (matches adapters.module4_adapter.GRUSliceSequenceClassifier)
# ---------------------------------------------------------------------------

class GRUSliceSequenceClassifier(nn.Module):
    """GRU sequence classifier over ordered MRI slice embeddings."""

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 64,
        num_classes: int = 3,
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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: [batch, num_slices, input_size]
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
            # hidden: [num_layers, batch, hidden_size]
            last_step = hidden[-1]
        else:
            output, hidden = self.gru(x)
            last_step = output[:, -1, :]
        return self.classifier(last_step)


# ---------------------------------------------------------------------------
# Slice Feature Extractor
# ---------------------------------------------------------------------------

class ResNet18SliceFeatureExtractor(nn.Module):
    """Extracts 512-dim pooled representations from 2D slices."""

    def __init__(self, weights_path: Path | str | None = None, device: torch.device | None = None) -> None:
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if weights_path and Path(weights_path).is_file():
            logger.info("Loading ResNet-18 weights from %s", weights_path)
            cp = torch.load(weights_path, map_location=self.device, weights_only=False)
            base = models.resnet18(weights=None)
            base.fc = nn.Linear(512, 3)
            if "model_state_dict" in cp:
                base.load_state_dict(cp["model_state_dict"], strict=False)
            else:
                base.load_state_dict(cp, strict=False)
        else:
            logger.info("Using standard ImageNet pre-trained ResNet-18")
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove fc, keep feature backbone
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(self.device).eval()

    @torch.no_grad()
    def extract_volume_slices(self, volume: np.ndarray, row: dict[str, Any], image_size: int = 160) -> torch.Tensor:
        """Extract ordered [num_slices, 512] embeddings for a single 3D volume."""
        norm_slices = normalize_resize_roi(volume, row, image_size=image_size)
        # norm_slices: [num_slices, H, W] uint8
        batch = torch.from_numpy(norm_slices.copy()).float().unsqueeze(1).to(self.device) / 255.0
        rgb = batch.repeat(1, 3, 1, 1)
        rgb = (rgb - self.mean) / self.std
        feats = self.backbone(rgb)  # [num_slices, 512, 1, 1]
        return torch.flatten(feats, 1).cpu()  # [num_slices, 512]


# ---------------------------------------------------------------------------
# Dataset & Caching
# ---------------------------------------------------------------------------

class VolumeSequenceDataset(Dataset):
    """Dataset of ordered slice feature sequences for MRI volumes."""

    def __init__(self, sequences: list[torch.Tensor], labels: list[int], filenames: list[str]) -> None:
        self.sequences = sequences
        self.labels = labels
        self.filenames = filenames

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        seq = self.sequences[idx]
        return seq, self.labels[idx], len(seq)


def collate_sequences(batch: list[tuple[torch.Tensor, int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length slice sequences with padding."""
    sequences, labels, lengths = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def load_splits(splits_path: Path, volumes_dir: Path) -> list[dict[str, Any]]:
    """Load splits CSV, filtering to volumes that exist on disk."""
    with splits_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    existing = []
    for r in rows:
        vpath = volumes_dir / r["volumeFilename"]
        if vpath.is_file():
            r_copy = dict(r)
            r_copy["path"] = vpath
            r_copy["label"] = int(r["aclDiagnosis"])
            existing.append(r_copy)
    return existing


def extract_or_load_features(
    rows: list[dict[str, Any]],
    volumes_dir: Path,
    cache_path: Path | None,
    resnet_weights: Path | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Extract ResNet-18 slice features or load from cache."""
    cache: dict[str, torch.Tensor] = {}
    if cache_path and cache_path.is_file():
        logger.info("Loading precomputed slice features from cache: %s", cache_path)
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    missing_rows = [r for r in rows if r["volumeFilename"] not in cache]
    if missing_rows:
        logger.info("Extracting slice features for %d volumes using ResNet-18...", len(missing_rows))
        extractor = ResNet18SliceFeatureExtractor(resnet_weights, device=device)
        for i, r in enumerate(missing_rows, 1):
            with r["path"].open("rb") as f:
                vol = np.asarray(pickle.load(f), dtype=np.float32)
            seq_feats = extractor.extract_volume_slices(vol, r)
            cache[r["volumeFilename"]] = seq_feats
            if i % 50 == 0 or i == len(missing_rows):
                logger.info("Extracted %d / %d volumes", i, len(missing_rows))

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cache, cache_path)
            logger.info("Cached slice features to %s", cache_path)

    return cache


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate model on a dataset loader."""
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    with torch.no_grad():
        for seqs, targets, lengths in loader:
            seqs, lengths = seqs.to(device), lengths.to(device)
            logits = model(seqs, lengths)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_targets.extend(targets.numpy().tolist())

    if not all_targets:
        return {}

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_preds)
    y_prob = np.asarray(all_probs)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "per_class": classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASSES, output_dict=True, zero_division=0),
    }
    return metrics


def train_gru(
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    test_loader: DataLoader | None,
    hidden_size: int = 64,
    num_layers: int = 1,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
) -> tuple[GRUSliceSequenceClassifier, dict[str, Any]]:
    """Train GRU model and evaluate on validation/test sets."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUSliceSequenceClassifier(
        input_size=512,
        hidden_size=hidden_size,
        num_classes=3,
        num_layers=num_layers,
    ).to(device)

    # Class weights to handle class imbalance (Healthy vs Partial vs Rupture)
    labels = []
    for _, batch_labels, _ in train_loader:
        labels.extend(batch_labels.tolist())
    counts = np.bincount(labels, minlength=3)
    weights = torch.tensor(
        [len(labels) / (3.0 * max(c, 1)) for c in counts], dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1 = -1.0
    best_state: dict[str, Any] | None = None

    logger.info("Training GRU model (epochs=%d, hidden=%d, device=%s)...", epochs, hidden_size, device)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for seqs, targets, lengths in train_loader:
            seqs, targets, lengths = seqs.to(device), targets.to(device), lengths.to(device)
            optimizer.zero_grad()
            logits = model(seqs, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if val_loader:
            val_metrics = evaluate_model(model, val_loader, device)
            val_f1 = val_metrics.get("macro_f1", 0.0)
            if val_f1 >= best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0 or epoch == epochs:
                logger.info("Epoch %2d/%2d | Loss: %.4f | Val F1: %.3f | Val Acc: %.3f",
                            epoch, epochs, total_loss / len(train_loader), val_f1, val_metrics.get("accuracy", 0.0))
        else:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 5 == 0 or epoch == epochs:
                logger.info("Epoch %2d/%2d | Loss: %.4f", epoch, epochs, total_loss / len(train_loader))

    if best_state is not None:
        model.load_state_dict(best_state)

    results: dict[str, Any] = {}
    if val_loader:
        results["validation"] = evaluate_model(model, val_loader, device)
    if test_loader:
        results["test"] = evaluate_model(model, test_loader, device)
        logger.info("Test Accuracy: %.3f | Test Macro-F1: %.3f",
                    results["test"].get("accuracy", 0.0), results["test"].get("macro_f1", 0.0))

    return model, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=ROOT / "acl_mri_classification/artifacts/splits.csv")
    parser.add_argument("--volumes-dir", type=Path, default=ROOT / "data/KneeMRI/volumetric_data")
    parser.add_argument("--resnet-weights", type=Path, default=ROOT / "acl_mri_classification/artifacts/deploy/resnet18.pt")
    parser.add_argument("--cache-features", type=Path, default=ROOT / "module4_recovery/artifacts/slice_features_cache.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "module4_recovery/artifacts")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = load_splits(args.splits, args.volumes_dir)
    logger.info("Found %d accessible MRI volumes in %s", len(rows), args.volumes_dir)
    if not rows:
        raise RuntimeError(f"No volume files found in {args.volumes_dir}")

    # Extract or load cached slice features
    feat_cache = extract_or_load_features(
        rows=rows,
        volumes_dir=args.volumes_dir,
        cache_path=args.cache_features,
        resnet_weights=args.resnet_weights if args.resnet_weights.is_file() else None,
        device=device,
    )

    # Partition by split
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]

    # Fallback if working with sample dataset (e.g. only test split available)
    if not train_rows:
        logger.warning("No train split rows found in available volumes. Using all available %d volumes for demo training.", len(rows))
        train_rows = rows
        val_rows = []
        test_rows = rows

    def make_dataset(subset: list[dict[str, Any]]) -> VolumeSequenceDataset:
        seqs = [feat_cache[r["volumeFilename"]] for r in subset]
        labels = [r["label"] for r in subset]
        fns = [r["volumeFilename"] for r in subset]
        return VolumeSequenceDataset(seqs, labels, fns)

    train_ds = make_dataset(train_rows)
    train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True, collate_fn=collate_sequences)

    val_loader = None
    if val_rows:
        val_ds = make_dataset(val_rows)
        val_loader = DataLoader(val_ds, batch_size=min(args.batch_size, len(val_ds)), shuffle=False, collate_fn=collate_sequences)

    test_loader = None
    if test_rows:
        test_ds = make_dataset(test_rows)
        test_loader = DataLoader(test_ds, batch_size=min(args.batch_size, len(test_ds)), shuffle=False, collate_fn=collate_sequences)

    model, metrics = train_gru(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    # Save artifacts in format expected by Module4Adapter
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "gru_checkpoint.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "input_size": 512,
            "hidden_size": args.hidden_size,
            "num_classes": 3,
            "num_layers": 1,
        },
        "classes": CLASSES,
        "metrics": metrics,
        "note": "GRU sequence classifier over ordered MRI slice embeddings (3-class volume diagnosis)",
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info("Saved GRU checkpoint to %s", checkpoint_path)

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
