"""Create deterministic stratified train/validation/test splits by exam ID."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("acl_mri_classification/artifacts/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("acl_mri_classification/artifacts/splits.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    test_fraction = 1.0 - args.train_fraction - args.validation_fraction
    if not (0 < args.train_fraction < 1 and 0 < args.validation_fraction < 1 and test_fraction > 0):
        raise SystemExit("Fractions must be positive and sum to less than 1")

    with args.manifest.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"examId", "aclDiagnosis"}.issubset(reader.fieldnames):
            raise SystemExit("Manifest must contain examId and aclDiagnosis")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    if not rows:
        raise SystemExit("Manifest contains no records")

    groups: dict[str, set[int]] = defaultdict(set)
    groups_by_class: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        groups[row["examId"].strip()].add(int(row["aclDiagnosis"]))
    conflicting = {group: labels for group, labels in groups.items() if len(labels) != 1}
    if conflicting:
        raise SystemExit(f"Exams with conflicting labels: {list(conflicting.items())[:10]}")
    for group, labels in groups.items():
        groups_by_class[next(iter(labels))].append(group)

    rng = random.Random(args.seed)
    assignment: dict[str, str] = {}
    for label in sorted(groups_by_class):
        class_groups = list(groups_by_class[label])
        rng.shuffle(class_groups)
        n_train = round(len(class_groups) * args.train_fraction)
        n_val = round(len(class_groups) * args.validation_fraction)
        # Keep at least one exam in every split when a class has >=3 exams.
        if len(class_groups) >= 3:
            n_train = max(1, min(n_train, len(class_groups) - 2))
            n_val = max(1, min(n_val, len(class_groups) - n_train - 1))
        for group in class_groups[:n_train]:
            assignment[group] = "train"
        for group in class_groups[n_train:n_train + n_val]:
            assignment[group] = "validation"
        for group in class_groups[n_train + n_val:]:
            assignment[group] = "test"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames + ["split"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "split": assignment[row["examId"].strip()]})

    print(f"Wrote {len(rows)} volumes across {len(groups)} exam groups to {args.output}")
    for split in ("train", "validation", "test"):
        subset = [row for row in rows if assignment[row["examId"].strip()] == split]
        volume_counts = Counter(int(row["aclDiagnosis"]) for row in subset)
        exam_count = sum(1 for group, assigned in assignment.items() if assigned == split)
        print(f"{split:10s} exams={exam_count:3d} volumes={len(subset):3d} labels={dict(sorted(volume_counts.items()))}")


if __name__ == "__main__":
    main()
