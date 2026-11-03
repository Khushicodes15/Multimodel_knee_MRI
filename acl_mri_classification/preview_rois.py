"""Create a visual contact sheet with one annotated ACL ROI per class."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABEL_NAMES = {"0": "Healthy", "1": "Partial tear", "2": "Complete rupture"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/KneeMRI/metadata.csv"))
    parser.add_argument("--volumes-dir", type=Path, default=Path("data/KneeMRI/volumetric_data"))
    parser.add_argument("--output", type=Path, default=Path("acl_mri_classification/artifacts/roi_preview.png"))
    parser.add_argument("--size", type=int, default=320)
    args = parser.parse_args()

    examples = {}
    with args.metadata.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            label = row["aclDiagnosis"].strip()
            if label in LABEL_NAMES and label not in examples:
                examples[label] = row
    if set(examples) != set(LABEL_NAMES):
        raise SystemExit(f"Expected labels 0, 1, 2; found {sorted(examples)}")

    panels = []
    for label in ("0", "1", "2"):
        row = examples[label]
        path = args.volumes_dir / row["volumeFilename"]
        # Only load pickle objects from the trusted official dataset.
        with path.open("rb") as stream:
            volume = np.asarray(pickle.load(stream))
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {volume.shape} for {path.name}")
        x, y, z, width, height, depth = [int(row[key]) for key in ("roiX", "roiY", "roiZ", "roiWidth", "roiHeight", "roiDepth")]
        z = min(max(z + max(depth, 1) // 2, 0), volume.shape[0] - 1)
        crop = volume[z, max(y, 0):min(y + height, volume.shape[1]), max(x, 0):min(x + width, volume.shape[2])].astype(np.float32)
        if crop.size == 0:
            raise ValueError(f"ROI is empty for {path.name}")
        low, high = np.percentile(crop, (1, 99))
        crop = np.clip((crop - low) / (high - low), 0, 1) if high > low else np.zeros_like(crop)
        image = Image.fromarray(np.rint(crop * 255).astype(np.uint8)).resize((args.size, args.size), Image.Resampling.BILINEAR)
        panels.append((label, row["volumeFilename"], image))

    canvas = Image.new("RGB", (args.size * 3, args.size + 52), "white")
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    for column, (label, filename, image) in enumerate(panels):
        x = column * args.size
        canvas.paste(image.convert("RGB"), (x, 0))
        draw.text((x + 8, args.size + 5), f"{LABEL_NAMES[label]} (label {label})", fill="black", font=font)
        draw.text((x + 8, args.size + 23), filename, fill="black", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved ROI preview: {args.output}")


if __name__ == "__main__":
    main()
