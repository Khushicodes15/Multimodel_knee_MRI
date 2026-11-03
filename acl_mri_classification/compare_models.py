"""Train candidate CNNs and keep only the strongest validation checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ARCHITECTURES = ("small_cnn", "resnet18", "efficientnet_b0", "densenet121")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=Path, default=Path("acl_mri_classification/artifacts/splits.csv"))
    parser.add_argument("--volumes-dir", type=Path, default=Path("data/KneeMRI/volumetric_data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("acl_mri_classification/artifacts/cache_160"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("acl_mri_classification/artifacts"))
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--slices-per-volume", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    transfer_architectures = [name for name in args.architectures if name != "small_cnn"]
    if transfer_architectures:
        if importlib.util.find_spec("torchvision") is None:
            raise SystemExit("The selected transfer models require torchvision. Install a torchvision build compatible with PyTorch first.")
        try:
            import torchvision  # noqa: F401
        except Exception as exc:
            raise SystemExit(f"Could not load torchvision; check that it matches the installed PyTorch build: {exc}") from exc

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.artifact_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    winner_score: tuple[float, float] | None = None
    winner_name: str | None = None
    results = []

    # Trial checkpoints and histories live only in this temporary directory.
    # The experiment's persistent model output is a single best_model.pt.
    with tempfile.TemporaryDirectory(prefix="architecture_trials_", dir=args.artifact_dir) as trial_root:
        for architecture in args.architectures:
            trial_dir = Path(trial_root) / architecture
            checkpoint_path = trial_dir / "best_model.pt"
            history_path = trial_dir / "training_history.json"
            command = [
                sys.executable, "-m", "acl_mri_classification.train",
                "--splits", str(args.splits), "--volumes-dir", str(args.volumes_dir),
                "--cache-dir", str(args.cache_dir), "--output-dir", str(trial_dir),
                "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                "--slices-per-volume", str(args.slices_per_volume), "--image-size", str(args.image_size),
                "--learning-rate", str(args.learning_rate), "--patience", str(args.patience),
                "--seed", str(args.seed), "--device", args.device, "--architecture", architecture,
            ]
            if architecture != "small_cnn":
                command.append("--pretrained")
            print(f"\n=== Training candidate: {architecture} ===", flush=True)
            subprocess.run(command, check=True)

            history = json.loads(history_path.read_text(encoding="utf-8"))["history"]
            best_epoch = max(history, key=lambda row: (row["validation_metrics"]["macro_f1"], -row["validation_loss"]))
            macro_f1 = float(best_epoch["validation_metrics"]["macro_f1"])
            val_loss = float(best_epoch["validation_loss"])
            result = {
                "architecture": architecture,
                "pretrained_imagenet": architecture != "small_cnn",
                "best_epoch": int(best_epoch["epoch"]),
                "validation_macro_f1": macro_f1,
                "validation_loss": val_loss,
                "validation_metrics": best_epoch["validation_metrics"],
            }
            results.append(result)
            score = (macro_f1, -val_loss)
            if winner_score is None or score > winner_score:
                winner_score, winner_name = score, architecture
                shutil.copy2(checkpoint_path, model_dir / "best_model.pt")

    if winner_name is None:
        raise RuntimeError("No architecture produced a checkpoint")
    comparison = {
        "selection_metric": "validation_macro_f1",
        "tie_breaker": "lower_validation_loss",
        "winner": winner_name,
        "architectures": results,
        "persistent_model_checkpoint": str((model_dir / "best_model.pt").resolve()),
    }
    comparison_path = args.artifact_dir / "model_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"\nBest architecture: {winner_name}")
    print(f"Only the selected checkpoint remains at: {(model_dir / 'best_model.pt').resolve()}")
    print(f"Validation comparison: {comparison_path.resolve()}")


if __name__ == "__main__":
    main()
