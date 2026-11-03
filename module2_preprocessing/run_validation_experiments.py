"""Compare CNN imbalance and regularization choices on validation only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("module2_preprocessing/validation_experiments"))
    args = parser.parse_args()
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Identical split, seed, architecture and training budget. Every trial is
    # ranked only by validation macro-F1, then validation loss. Test is never read.
    settings = [
        {"name": "balanced_weights", "weight": "balanced"},
        {"name": "sqrt_weights", "weight": "sqrt"},
        {"name": "no_class_weights", "weight": "none"},
        {"name": "augmentation_balanced", "weight": "balanced", "augmentation": True},
        {"name": "augmentation_sqrt", "weight": "sqrt", "augmentation": True},
        {"name": "more_dropout_balanced", "weight": "balanced", "dropout": 0.50},
        {"name": "more_weight_decay_balanced", "weight": "balanced", "weight_decay": 1e-3},
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="validation_trials_", dir=out) as temp:
        for item in settings:
            run_dir = Path(temp) / item["name"]
            command = [
                sys.executable, "-m", "acl_mri_classification.train",
                "--architecture", "small_cnn", "--epochs", str(args.epochs), "--device", args.device,
                "--output-dir", str(run_dir), "--class-weighting", item["weight"],
                "--dropout", str(item.get("dropout", 0.35)),
                "--weight-decay", str(item.get("weight_decay", 1e-4)),
                "--seed", "42",
            ]
            if item.get("augmentation", False):
                command.append("--augmentation")
            print(f"\n=== Validation-only trial: {item['name']} ===", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            history_doc = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
            best = max(history_doc["history"], key=lambda row: (row["validation_metrics"]["macro_f1"], -row["validation_loss"]))
            results.append({
                "name": item["name"], "class_weighting": item["weight"],
                "augmentation": bool(item.get("augmentation", False)),
                "dropout": float(item.get("dropout", 0.35)),
                "weight_decay": float(item.get("weight_decay", 1e-4)),
                "best_epoch": int(best["epoch"]),
                "validation_macro_f1": float(best["validation_metrics"]["macro_f1"]),
                "validation_balanced_accuracy": float(best["validation_metrics"]["balanced_accuracy"]),
                "validation_accuracy": float(best["validation_metrics"]["accuracy"]),
                "validation_loss": float(best["validation_loss"]),
                "validation_confusion_matrix": best["validation_metrics"]["confusion_matrix"],
                "validation_per_class": best["validation_metrics"]["per_class"],
            })

    results.sort(key=lambda row: (row["validation_macro_f1"], -row["validation_loss"]), reverse=True)
    report = {
        "purpose": "Validation-only CNN experiment; the held-out test partition was not read.",
        "selection": "highest validation macro-F1, then lowest validation loss",
        "training_budget_epochs": args.epochs,
        "device": args.device,
        "architecture": "small_cnn",
        "trials": results,
        "winner": results[0]["name"] if results else None,
        "note": "Treat close rankings cautiously: this is one fixed split and seed. Do not use these results as test performance.",
    }
    report_path = out / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nValidation-only comparison saved to {report_path.resolve()}")
    print(f"Best validation trial: {report['winner']}")


if __name__ == "__main__":
    main()
