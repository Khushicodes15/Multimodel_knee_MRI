"""Compare a few safe preprocessing and regularization settings."""

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
    args = parser.parse_args()
    output_dir = ROOT / "module2_preprocessing" / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = [
        ("baseline", False, 0.35, 1e-4),
        ("intensity_augmentation", True, 0.35, 1e-4),
        ("more_dropout", False, 0.50, 1e-4),
        ("more_weight_decay", False, 0.35, 1e-3),
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="module2_runs_", dir=output_dir) as temp:
        for name, augmentation, dropout, weight_decay in settings:
            run_dir = Path(temp) / name
            command = [
                sys.executable, "-m", "acl_mri_classification.train",
                "--architecture", "small_cnn", "--epochs", str(args.epochs), "--device", args.device,
                "--output-dir", str(run_dir), "--dropout", str(dropout), "--weight-decay", str(weight_decay),
            ]
            if augmentation:
                command.append("--augmentation")
            print(f"\n=== Module 2 experiment: {name} ===", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))["history"]
            best = max(history, key=lambda row: (row["validation_metrics"]["macro_f1"], -row["validation_loss"]))
            results.append({
                "name": name,
                "augmentation": augmentation,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "best_epoch": best["epoch"],
                "validation_macro_f1": best["validation_metrics"]["macro_f1"],
                "validation_loss": best["validation_loss"],
            })
    report = {
        "purpose": "Module 2 preprocessing and regularization comparison",
        "selection": "validation macro-F1, then lower validation loss",
        "runs": results,
    }
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Comparison saved to {report_path}")


if __name__ == "__main__":
    main()
