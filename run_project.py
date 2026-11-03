"""Run the academic KneeMRI project modules in a reproducible order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--module2-experiments", action="store_true", help="Also train four small-CNN regularization variants")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    # Module 3 creates/validates the shared exam-level split and trains/evaluates
    # the selected image model. Restrict the default end-to-end run to the
    # scratch CNN so it works without optional torchvision installations.
    run(
        "acl_mri_classification.run_pipeline", "--epochs", str(args.epochs),
        "--batch-size", "4", "--slices-per-volume", "4", "--image-size", "160",
        "--device", args.device, "--architectures", "small_cnn",
    )
    run("module1_baseline.run_baseline")
    run("module2_preprocessing.run_preprocessing")
    if args.module2_experiments:
        run("module2_preprocessing.compare_training", "--epochs", str(max(1, min(args.epochs, 10))), "--device", args.device)
    run("module4_recovery.generate_simulated_data")
    run("module4_recovery.train_lstm")
    print("\nProject pipeline complete. See PROJECT_GUIDE.md for outputs and interpretation.")


if __name__ == "__main__":
    main()
