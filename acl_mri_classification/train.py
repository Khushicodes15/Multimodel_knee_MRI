"""Train the CNN, select a checkpoint using validation data, and save history."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .model import CLASS_NAMES, MRIVolumes, build_model, metrics_for, read_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("acl_mri_classification/artifacts/splits.csv"))
    parser.add_argument("--volumes-dir", type=Path, default=Path("data/KneeMRI/volumetric_data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("acl_mri_classification/artifacts/cache_160"))
    parser.add_argument("--output-dir", type=Path, default=Path("acl_mri_classification/artifacts/model"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4, help="MRI volumes per batch")
    parser.add_argument("--slices-per-volume", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--architecture", choices=("small_cnn", "resnet18", "efficientnet_b0", "densenet121"), default="small_cnn")
    parser.add_argument("--pretrained", action="store_true", help="Initialize transfer-learning backbone with ImageNet weights")
    parser.add_argument("--augmentation", action="store_true", help="Use mild brightness/contrast/noise changes on training ROIs")
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--class-weighting", choices=("balanced", "sqrt", "none"), default="balanced", help="Class loss weights: inverse-frequency, square-root inverse-frequency, or none")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.slices_per_volume < 1 or args.image_size < 32:
        raise SystemExit("epochs, batch size, slices per volume, and image size must be positive (image size >= 32)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but this PyTorch installation cannot access a CUDA GPU.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    rows = read_split(args.splits, args.volumes_dir)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows or not val_rows:
        raise ValueError("The split manifest must contain training and validation cases")
    train_set = MRIVolumes(rows, "train", args.cache_dir, args.image_size, args.slices_per_volume, augmentation=args.augmentation)
    val_set = MRIVolumes(rows, "validation", args.cache_dir, args.image_size, args.slices_per_volume)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    class_counts = np.bincount([row["label"] for row in train_rows], minlength=3)
    if np.any(class_counts == 0):
        raise ValueError(f"Training split is missing a class: {class_counts.tolist()}")
    if args.class_weighting == "balanced":
        class_weights = len(train_rows) / (len(CLASS_NAMES) * class_counts.astype(np.float32))
    elif args.class_weighting == "sqrt":
        class_weights = np.sqrt(len(train_rows) / (len(CLASS_NAMES) * class_counts.astype(np.float32)))
        class_weights /= class_weights.mean()
    else:
        class_weights = None
    weight_tensor = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    model = build_model(args.architecture, pretrained=args.pretrained, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_macro_f1, best_val_loss, stale_epochs, history = -1.0, float("inf"), 0, []
    print(f"Device: {device}; train volumes: {len(train_rows)}; validation volumes: {len(val_rows)}")
    print(f"Training class counts (0/1/2): {class_counts.tolist()}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for images, labels, _ in train_loader:
            batch, slices, channels, height, width = images.shape
            images = images.reshape(batch * slices, channels, height, width).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            volume_logits = model(images).reshape(batch, slices, 3).mean(dim=1)
            loss = loss_fn(volume_logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_truths, val_predictions, val_losses = [], [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                batch, slices, channels, height, width = images.shape
                images = images.reshape(batch * slices, channels, height, width).to(device)
                labels = labels.to(device)
                logits = model(images).reshape(batch, slices, 3).mean(dim=1)
                val_losses.append(float(nn.functional.cross_entropy(logits, labels).cpu()))
                val_truths.extend(labels.cpu().tolist())
                val_predictions.extend(logits.argmax(dim=1).cpu().tolist())
        # The checkpoint criterion is unweighted cross-entropy on deterministic
        # validation slices; class balance is also reflected in macro-F1.
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        val_metrics = metrics_for(val_truths, val_predictions)
        record = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": val_loss, "validation_metrics": val_metrics}
        history.append(record)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={record['train_loss']:.4f} | val_loss={val_loss:.4f} | val_macro_F1={val_metrics['macro_f1']:.3f}")

        val_macro_f1 = val_metrics["macro_f1"]
        improved = val_macro_f1 > best_macro_f1 + 1e-8 or (abs(val_macro_f1 - best_macro_f1) <= 1e-8 and val_loss < best_val_loss)
        if improved:
            best_macro_f1, best_val_loss, stale_epochs = val_macro_f1, val_loss, 0
            config = {
                "image_size": args.image_size,
                "slices_per_volume": args.slices_per_volume,
                "architecture": args.architecture,
                "pretrained": args.pretrained,
                "augmentation": args.augmentation,
                "dropout": args.dropout,
                "class_weighting": args.class_weighting,
                "class_weights": None if class_weights is None else class_weights.tolist(),
                "class_names": list(CLASS_NAMES),
                "seed": args.seed,
            }
            torch.save({"model_state_dict": model.state_dict(), "config": config, "best_validation_loss": best_val_loss, "best_validation_macro_f1": best_macro_f1, "best_epoch": epoch}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    history_path = args.output_dir / "training_history.json"
    saved_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    history_path.write_text(json.dumps({"config": saved_args | {"device_used": str(device)}, "history": history}, indent=2), encoding="utf-8")
    print(f"Best checkpoint saved to {checkpoint_path.resolve()}")
    print(f"Training history saved to {history_path.resolve()}")


if __name__ == "__main__":
    main()
