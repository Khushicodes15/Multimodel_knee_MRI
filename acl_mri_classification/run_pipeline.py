"""Run ingestion -> split -> architecture comparison -> held-out evaluation."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("small_cnn", "resnet18", "efficientnet_b0", "densenet121")


def rooted(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_stage(module: str, arguments: list[str]) -> None:
    command = [sys.executable, "-m", f"acl_mri_classification.{module}", *arguments]
    print(f"\n=== {module.upper()} ===", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/KneeMRI"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("acl_mri_classification/artifacts"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--slices-per-volume", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    args = parser.parse_args()

    if any(architecture != "small_cnn" for architecture in args.architectures):
        if importlib.util.find_spec("torchvision") is None:
            raise SystemExit("The selected transfer models require torchvision. Install a torchvision build compatible with PyTorch first.")
        try:
            import torchvision  # noqa: F401
        except Exception as exc:
            raise SystemExit(f"Could not load torchvision; check that it matches the installed PyTorch build: {exc}") from exc

    data_dir, artifact_dir = rooted(args.data_dir), rooted(args.artifact_dir)
    metadata, volumes = data_dir / "metadata.csv", data_dir / "volumetric_data"
    manifest, split_csv = artifact_dir / "manifest.csv", artifact_dir / "splits.csv"
    cache_dir = artifact_dir / f"cache_{args.image_size}"
    model_dir, eval_dir = artifact_dir / "model", artifact_dir / "evaluation"
    if not metadata.is_file() or not volumes.is_dir():
        raise SystemExit(f"Expected metadata.csv and volumetric_data under {data_dir}")

    common_ingest = ["--metadata", str(metadata), "--volumes-dir", str(volumes), "--output-dir", str(artifact_dir)]
    common_preview = ["--metadata", str(metadata), "--volumes-dir", str(volumes), "--output", str(artifact_dir / "roi_preview.png")]
    common_split = ["--manifest", str(manifest), "--output", str(split_csv), "--seed", str(args.seed)]
    common_experiment = [
        "--splits", str(split_csv), "--volumes-dir", str(volumes), "--cache-dir", str(cache_dir), "--artifact-dir", str(artifact_dir),
        "--epochs", str(args.epochs), "--batch-size", str(args.batch_size), "--slices-per-volume", str(args.slices_per_volume),
        "--image-size", str(args.image_size), "--learning-rate", str(args.learning_rate), "--patience", str(args.patience),
        "--seed", str(args.seed), "--device", args.device, "--architectures", *args.architectures,
    ]
    common_eval = [
        "--splits", str(split_csv), "--volumes-dir", str(volumes), "--cache-dir", str(cache_dir),
        "--checkpoint", str(model_dir / "best_model.pt"), "--output-dir", str(eval_dir),
        "--batch-size", str(args.batch_size), "--device", args.device,
    ]
    run_stage("check_data", common_ingest)
    run_stage("preview_rois", common_preview)
    run_stage("make_split", common_split)
    run_stage("compare_models", common_experiment)
    run_stage("evaluate", common_eval)
    print(f"\nPipeline complete. Model and reports are under {artifact_dir}")


if __name__ == "__main__":
    main()
