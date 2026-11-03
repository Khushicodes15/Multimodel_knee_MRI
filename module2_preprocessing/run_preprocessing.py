"""Create the reusable ROI cache used by the training scripts."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from .preprocessing import normalize_resize_roi


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/KneeMRI/metadata.csv")
    parser.add_argument("--volumes-dir", type=Path, default=ROOT / "data/KneeMRI/volumetric_data")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "acl_mri_classification/artifacts/cache_160")
    parser.add_argument("--image-size", type=int, default=160)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    shapes = Counter()
    for index, row in enumerate(rows, start=1):
        cache_file = args.cache_dir / f"{row['volumeFilename']}.size{args.image_size}.npy"
        if not cache_file.exists():
            with (args.volumes_dir / row["volumeFilename"]).open("rb") as f:
                volume = np.asarray(pickle.load(f))
            slices = normalize_resize_roi(volume, row, args.image_size)
            with cache_file.open("wb") as f:
                np.save(f, slices, allow_pickle=False)
        shapes[str(np.load(cache_file, allow_pickle=False).shape)] += 1
        if index % 100 == 0 or index == len(rows):
            print(f"Processed {index}/{len(rows)} volumes")

    report = {
        "source": "KneeMRI ROI annotations",
        "volumes": len(rows),
        "image_size": args.image_size,
        "cached_slice_shapes": dict(shapes),
        "cache_dir": str(args.cache_dir.resolve()),
        "augmentation": "optional mild intensity/noise only; no orientation flips",
    }
    report_path = ROOT / "module2_preprocessing" / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Preprocessing report saved to {report_path}")


if __name__ == "__main__":
    main()
