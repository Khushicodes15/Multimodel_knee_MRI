"""Train a simple MLP baseline from MRI ROI statistics and metadata."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    if "kneeLR" not in row or row["kneeLR"] is None or str(row["kneeLR"]).strip() == "":
        raise KeyError(f"Missing required 'kneeLR' laterality in row for volume {row.get('volumeFilename')}")
    try:
        knee_lr_val = float(int(str(row["kneeLR"]).strip()))
    except ValueError as exc:
        raise ValueError(f"Invalid 'kneeLR' value {row.get('kneeLR')!r} for volume {row.get('volumeFilename')}") from exc

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
        float(x), float(y), float(z), knee_lr_val,
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
    available_rows = [r for r in rows if (args.volumes_dir / r["volumeFilename"]).is_file()]
    if len(available_rows) < len(rows):
        print(f"Filtering to {len(available_rows)} available volumes in {args.volumes_dir} (out of {len(rows)} in splits)")
        rows = available_rows
    if not rows:
        raise RuntimeError(f"No volume files found in {args.volumes_dir}")

    print(f"Extracting ROI features for {len(rows)} MRI volumes...")
    features = np.asarray([extract_features(row, args.volumes_dir) for row in rows], dtype=np.float32)
    labels = np.asarray([int(row["aclDiagnosis"]) for row in rows], dtype=np.int64)
    splits = np.asarray([row["split"] for row in rows])
    train_mask, val_mask, test_mask = splits == "train", splits == "validation", splits == "test"
    if not train_mask.any():
        print("Note: No train split rows in available subset. Using available rows for training demo.")
        train_mask = np.ones(len(rows), dtype=bool)
        val_mask = train_mask
        test_mask = train_mask
    x_train, y_train = balance_training_data(features[train_mask], labels[train_mask])

    # Train Logistic Regression Baseline
    logreg_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=42, class_weight=None),
    )
    logreg_model.fit(x_train, y_train)
    lr_val_preds = logreg_model.predict(features[val_mask])
    lr_val_probs = logreg_model.predict_proba(features[val_mask])
    lr_test_preds = logreg_model.predict(features[test_mask])
    lr_test_probs = logreg_model.predict_proba(features[test_mask])

    truth = labels[test_mask]
    val_truth = labels[val_mask]

    def eval_baseline(y_true, y_pred, y_prob):
        res = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
            "per_class": classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASSES, output_dict=True, zero_division=0),
        }
        try:
            res["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=[0, 1, 2]))
        except Exception:
            res["roc_auc_ovr_macro"] = None
        return res

    mlp_model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42),
    )
    mlp_model.fit(x_train, y_train)
    mlp_val_preds = mlp_model.predict(features[val_mask])
    mlp_val_probs = mlp_model.predict_proba(features[val_mask])
    mlp_test_preds = mlp_model.predict(features[test_mask])
    mlp_test_probs = mlp_model.predict_proba(features[test_mask])

    mlp_metrics = eval_baseline(truth, mlp_test_preds, mlp_test_probs)
    mlp_metrics["validation"] = eval_baseline(val_truth, mlp_val_preds, mlp_val_probs)

    logreg_metrics = eval_baseline(truth, lr_test_preds, lr_test_probs)
    logreg_metrics["validation"] = eval_baseline(val_truth, lr_val_preds, lr_val_probs)

    metrics = {
        "test_volumes": int(test_mask.sum()),
        "mlp": mlp_metrics,
        "logistic_regression": logreg_metrics,
        "class_counts": {str(k): int(v) for k, v in Counter(truth).items()},
        "note": "Module 1 handcrafted-feature baselines (MLP and Logistic Regression). Evaluated on held-out test exams.",
        # Backward-compatibility flat aliases for existing readers of MLP metrics:
        "accuracy": mlp_metrics["accuracy"],
        "balanced_accuracy": mlp_metrics["balanced_accuracy"],
        "macro_f1": mlp_metrics["macro_f1"],
        "roc_auc_ovr_macro": mlp_metrics["roc_auc_ovr_macro"],
        "confusion_matrix": mlp_metrics["confusion_matrix"],
        "per_class": mlp_metrics["per_class"],
        "validation": mlp_metrics["validation"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(mlp_model, args.output_dir / "baseline_mlp.joblib")
    joblib.dump(logreg_model, args.output_dir / "baseline_logreg.joblib")
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
        writer.writerow(["volumeFilename", "true_label", "mlp_pred", "mlp_p_healthy", "mlp_p_partial", "mlp_p_complete", "logreg_pred", "logreg_p_healthy", "logreg_p_partial", "logreg_p_complete"])
        test_rows = [row for row in rows if row["split"] == "test"]
        for row, m_pred, m_probs, lr_pred, lr_probs in zip(test_rows, mlp_test_preds, mlp_test_probs, lr_test_preds, lr_test_probs):
            writer.writerow([row["volumeFilename"], row["aclDiagnosis"], int(m_pred), *[f"{p:.6f}" for p in m_probs], int(lr_pred), *[f"{p:.6f}" for p in lr_probs]])
    print(f"MLP Test Accuracy: {mlp_metrics['accuracy']:.3f} | Macro-F1: {mlp_metrics['macro_f1']:.3f} | ROC-AUC: {mlp_metrics.get('roc_auc_ovr_macro')}")
    print(f"LogReg Test Accuracy: {logreg_metrics['accuracy']:.3f} | Macro-F1: {logreg_metrics['macro_f1']:.3f} | ROC-AUC: {logreg_metrics.get('roc_auc_ovr_macro')}")
    print(f"Saved models, metrics, and predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
