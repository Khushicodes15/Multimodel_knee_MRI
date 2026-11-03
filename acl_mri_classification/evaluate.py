"""Evaluate a saved CNN once on the held-out test exams."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .model import CLASS_NAMES, MRIVolumes, build_model, metrics_for, predict_volumes, read_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("acl_mri_classification/artifacts/splits.csv"))
    parser.add_argument("--volumes-dir", type=Path, default=Path("data/KneeMRI/volumetric_data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("acl_mri_classification/artifacts/cache"))
    parser.add_argument("--checkpoint", type=Path, default=Path("acl_mri_classification/artifacts/model/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("acl_mri_classification/artifacts/evaluation"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}; train the model first")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but this PyTorch installation cannot access a CUDA GPU.")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    # This checkpoint is produced by this project. Do not load checkpoints from unknown sources.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    rows = read_split(args.splits, args.volumes_dir)
    test_set = MRIVolumes(rows, "test", args.cache_dir, int(config["image_size"]), int(config["slices_per_volume"]))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    architecture = config.get("architecture", "small_cnn")
    # The checkpoint already contains trained weights, so do not redownload
    # ImageNet weights during evaluation.
    model = build_model(architecture, pretrained=False, dropout=float(config.get("dropout", 0.35))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    truths, predictions, filenames, probabilities = predict_volumes(model, test_loader, device)
    metrics = metrics_for(truths, predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "test_metrics.json"
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "test_volumes": len(truths),
        "class_names": list(CLASS_NAMES),
        "label_mapping": {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1], "2": CLASS_NAMES[2]},
        **metrics,
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions_path = args.output_dir / "test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(("volumeFilename", "true_label", "predicted_label", "true_class", "predicted_class", "confidence", "p_healthy", "p_partial", "p_complete"))
        for name, truth, prediction, probs in zip(filenames, truths, predictions, probabilities):
            writer.writerow((name, truth, prediction, CLASS_NAMES[truth], CLASS_NAMES[prediction], f"{max(probs):.6f}", *(f"{p:.6f}" for p in probs)))
    print(f"Device: {device}; held-out test volumes: {len(truths)}")
    print(f"Accuracy={metrics['accuracy']:.3f}; balanced accuracy={metrics['balanced_accuracy']:.3f}; macro-F1={metrics['macro_f1']:.3f}")
    print(f"Metrics: {metrics_path.resolve()}")
    print(f"Predictions: {predictions_path.resolve()}")


if __name__ == "__main__":
    main()
