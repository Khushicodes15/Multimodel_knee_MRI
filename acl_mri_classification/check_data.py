"""Validate raw KneeMRI inputs and write a normalized manifest and audit report."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = ("examId", "seriesNo", "aclDiagnosis", "roiX", "roiY", "roiZ", "roiHeight", "roiWidth", "roiDepth", "volumeFilename")
LABEL_NAMES = {0: "Healthy", 1: "Partial tear", 2: "Complete rupture"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/KneeMRI/metadata.csv"))
    parser.add_argument("--volumes-dir", type=Path, default=Path("data/KneeMRI/volumetric_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("acl_mri_classification/artifacts"))
    args = parser.parse_args()

    with args.metadata.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise SystemExit(f"Metadata has no header: {args.metadata}")
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing_columns:
            raise SystemExit(f"Missing required columns {missing_columns}; found {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise SystemExit("Metadata contains no records")

    errors: list[str] = []
    labels: Counter[int] = Counter()
    group_labels: dict[str, set[int]] = defaultdict(set)
    seen_files: set[str] = set()
    duplicate_files: list[str] = []
    shape_counts: Counter[str] = Counter()
    invalid_rois: list[str] = []

    for row_num, row in enumerate(rows, start=2):
        try:
            label = int(row["aclDiagnosis"])
            if label not in LABEL_NAMES:
                raise ValueError(f"invalid label {label}; expected 0, 1, or 2")
            group = row["examId"].strip()
            filename = row["volumeFilename"].strip()
            if not group or not filename:
                raise ValueError("examId or volumeFilename is blank")
            if filename in seen_files:
                duplicate_files.append(filename)
                raise ValueError(f"duplicate volumeFilename {filename}")
            seen_files.add(filename)
            group_labels[group].add(label)
            labels[label] += 1
            volume_path = args.volumes_dir / filename
            if not volume_path.is_file():
                raise FileNotFoundError(f"missing volume file {volume_path}")
            # These are pickle objects from the official KneeMRI data archive.
            with volume_path.open("rb") as stream:
                volume = np.asarray(pickle.load(stream))
            if volume.ndim != 3:
                raise ValueError(f"expected a 3D volume, got shape {volume.shape}")
            shape_counts[str(tuple(int(v) for v in volume.shape))] += 1
            x, y, z = (int(row[k]) for k in ("roiX", "roiY", "roiZ"))
            width, height, depth = (int(row[k]) for k in ("roiWidth", "roiHeight", "roiDepth"))
            if min(x, y, z) < 0 or min(width, height, depth) <= 0:
                raise ValueError("ROI has a negative origin or non-positive dimensions")
            if x + width > volume.shape[2] or y + height > volume.shape[1] or z + depth > volume.shape[0]:
                invalid_rois.append(filename)
        except Exception as exc:
            errors.append(f"CSV row {row_num} ({row.get('volumeFilename', '')}): {exc}")

    conflicting_groups = {group: sorted(values) for group, values in group_labels.items() if len(values) > 1}
    if conflicting_groups:
        errors.append(f"Exams with conflicting labels: {list(conflicting_groups.items())[:10]}")
    if invalid_rois:
        errors.append(f"ROIs outside volume bounds ({len(invalid_rois)}): {invalid_rois[:10]}")
    expected_files = {row["volumeFilename"].strip() for row in rows}
    available_files = {path.name for path in args.volumes_dir.glob("*.pck")}
    unreferenced_files = sorted(available_files - expected_files)
    missing_files = sorted(expected_files - available_files)
    if missing_files:
        errors.append(f"Missing volume files ({len(missing_files)}): {missing_files[:10]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata_path": str(args.metadata.resolve()),
        "volumes_dir": str(args.volumes_dir.resolve()),
        "metadata_rows": len(rows),
        "unique_exams": len(group_labels),
        "label_mapping": {str(k): v for k, v in LABEL_NAMES.items()},
        "class_counts": {str(k): labels[k] for k in sorted(LABEL_NAMES)},
        "volume_shapes": dict(shape_counts),
        "duplicate_volume_filenames": sorted(set(duplicate_files)),
        "missing_volume_files": missing_files,
        "unreferenced_pck_files": unreferenced_files,
        "invalid_rois": invalid_rois,
        "errors": errors,
        "valid": not errors,
    }
    (args.output_dir / "ingestion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise SystemExit(f"Ingestion failed with {len(errors)} issue(s). See {args.output_dir / 'ingestion_report.json'}")

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Ingestion valid: {len(rows)} volumes, {len(group_labels)} exams")
    print(f"Class counts (0/1/2): {dict(sorted(labels.items()))}")
    print(f"Manifest: {manifest_path}")
    print(f"Audit report: {args.output_dir / 'ingestion_report.json'}")
    if unreferenced_files:
        print(f"Note: {len(unreferenced_files)} unreferenced .pck file(s) found; review the ingestion report.")


if __name__ == "__main__":
    main()
