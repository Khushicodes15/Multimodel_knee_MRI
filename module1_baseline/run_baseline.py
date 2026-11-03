"""Train a simple MLP baseline from MRI ROI statistics and metadata."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["Healthy", "Partial tear", "Complete rupture"]
FEATURE_NAMES = [
    "roi_mean", "roi_std", "roi_min", "roi_max", "roi_median", "roi_p10", "roi_p90",
    "slice_mean", "slice_mean_std", "roi_width", "roi_height", "roi_depth",
    "roi_x", "roi_y", "roi_z", "knee_side",
]


def get_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def extract_features(row: dict[str, str], volumes_dir: Path) -> list[float]:
    # The official dataset stores each 3D MRI volume as a pickle object.
    # Only load files from the trusted KneeMRI archive.
    path = volumes_dir / row["volumeFilename"]
    with path.open("rb") as f:
        volume = np.asarray(pickle.load(f), dtype=np.float32)
    x, y, z = (int(row[k]) for k in ("roiX", "roiY", "roiZ"))
    w, h, d = (int(row[k]) for k in ("roiWidth", "roiHeight", "roiDepth"))
    roi = volume[z:z + d, y:y + h, x:x + w]
    if roi.size == 0:
        raise ValueError(f"Empty ROI for {row['volumeFilename']}")
    values = roi.reshape(-1)
    slice_means = roi.mean(axis=(1, 2))
    # Nine simple intensity measures plus ROI size/location and knee side.
    return [
        float(values.mean()), float(values.std()), float(values.min()), float(values.max()),
        float(np.median(values)), float(np.percentile(values, 10)), float(np.percentile(values, 90)),
        float(slice_means.mean()), float(slice_means.std()), float(w), float(h), float(d),
        float(x), float(y), float(z), float(int(row.get("kneeLR", "0"))),
    ]


def balance_training_data(x: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Simple random oversampling of training examples only."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=3)
    largest = int(counts.max())
    indices = []
    for label in range(3):
        label_indices = np.flatnonzero(y == label)
        indices.extend(rng.choice(label_indices, size=largest, replace=len(label_indices) < largest).tolist())
    rng.shuffle(indices)
    return x[indices], y[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/KneeMRI/metadata.csv")
    parser.add_argument("--splits", type=Path, default=ROOT / "acl_mri_classification/artifacts/splits.csv")
    parser.add_argument("--volumes-dir", type=Path, default=ROOT / "data/KneeMRI/volumetric_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "module1_baseline/artifacts")
    args = parser.parse_args()

    rows = get_rows(args.splits)
    print(f"Extracting ROI features for {len(rows)} MRI volumes...")
    features = np.asarray([extract_features(row, args.volumes_dir) for row in rows], dtype=np.float32)
    labels = np.asarray([int(row["aclDiagnosis"]) for row in rows], dtype=np.int64)
    splits = np.asarray([row["split"] for row in rows])
    train_mask, val_mask, test_mask = splits == "train", splits == "validation", splits == "test"
    x_train, y_train = balance_training_data(features[train_mask], labels[train_mask])

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    )
    model.fit(x_train, y_train)
    validation_predictions = model.predict(features[val_mask])
    predictions = model.predict(features[test_mask])
    probabilities = model.predict_proba(features[test_mask])
    truth = labels[test_mask]
    metrics = {
        "test_volumes": int(test_mask.sum()),
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1, 2]).tolist(),
        "per_class": classification_report(truth, predictions, labels=[0, 1, 2], target_names=CLASSES, output_dict=True, zero_division=0),
        "class_counts": {str(k): int(v) for k, v in Counter(labels[test_mask]).items()},
        "note": "Handcrafted-feature baseline. All splits use the shared examId split file.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "baseline_mlp.joblib")
    metrics["validation"] = {
        "accuracy": float(accuracy_score(labels[val_mask], validation_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels[val_mask], validation_predictions)),
        "macro_f1": float(f1_score(labels[val_mask], validation_predictions, average="macro", zero_division=0)),
    }
    with (args.output_dir / "handcrafted_features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["volumeFilename", "aclDiagnosis", "split", *FEATURE_NAMES])
        for row, values in zip(rows, features):
            writer.writerow([row["volumeFilename"], row["aclDiagnosis"], row["split"], *values.tolist()])
    eda = {
        "volumes": len(rows),
        "class_counts": {str(k): int(v) for k, v in Counter(labels).items()},
        "split_counts": {split: int((splits == split).sum()) for split in ("train", "validation", "test")},
        "laterality_counts": {str(k): int(v) for k, v in Counter(row.get("kneeLR", "unknown") for row in rows).items()},
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
    }
    (args.output_dir / "eda.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["volumeFilename", "true_label", "predicted_label", "p_healthy", "p_partial", "p_complete"])
        test_rows = [row for row in rows if row["split"] == "test"]
        for row, pred, probs in zip(test_rows, predictions, probabilities):
            writer.writerow([row["volumeFilename"], row["aclDiagnosis"], int(pred), *[f"{p:.6f}" for p in probs]])
    print(f"Test accuracy: {metrics['accuracy']:.3f}; macro-F1: {metrics['macro_f1']:.3f}")
    print(f"Saved MLP, metrics, and predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
