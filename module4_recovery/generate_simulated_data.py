"""Create fictional recovery sequences for demonstrating an LSTM."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TIMES = (0, 3, 6, 12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "module4_recovery/data/simulated_recovery.csv")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["record_type", "synthetic_id", "time_months", "acl_severity", "treatment", "recovery_score"])
        for person in range(1, args.patients + 1):
            # Severity frequencies are illustrative and do not represent actual patients.
            severity = rng.choices([0, 1, 2], weights=[0.55, 0.30, 0.15])[0]
            treatment = rng.choices(["rehab", "surgery"], weights=([0.75, 0.25] if severity == 0 else [0.45, 0.55] if severity == 1 else [0.20, 0.80]))[0]
            start = 95 - 20 * severity + rng.uniform(-8, 8)
            treatment_effect = 4 if treatment == "surgery" and severity > 0 else 0
            for month in TIMES:
                improvement = (1 - pow(2.7182818, -month / 5.5)) * (91 - start + treatment_effect)
                score = max(0, min(100, start + improvement + rng.gauss(0, 2.5)))
                writer.writerow(["SIMULATED", f"SYN-{person:04d}", month, severity, treatment, round(score, 1)])
    print(f"Created {args.patients} fictional sequences at {args.output}")
    print("This file is simulated teaching data and is not linked to the KneeMRI exams.")


if __name__ == "__main__":
    main()
