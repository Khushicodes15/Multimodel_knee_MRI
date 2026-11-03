"""Compare a simple regression baseline with an LSTM on simulated sequences."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
TIMES = [0, 3, 6, 12]


def read_sequences(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    groups: dict[str, list[dict[str, str]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["record_type"] != "SIMULATED":
                raise ValueError("Module 4 accepts only records marked SIMULATED in this demo.")
            groups.setdefault(row["synthetic_id"], []).append(row)
    x_values, targets, ids = [], [], []
    for patient_id, rows in sorted(groups.items()):
        rows.sort(key=lambda item: int(item["time_months"]))
        if [int(row["time_months"]) for row in rows] != TIMES:
            continue
        sequence = []
        for row in rows[:3]:
            severity = int(row["acl_severity"])
            treatment = row["treatment"]
            sequence.append([
                float(severity == 0), float(severity == 1), float(severity == 2),
                float(treatment == "rehab"), float(treatment == "surgery"),
                int(row["time_months"]) / 12.0, float(row["recovery_score"]) / 100.0,
            ])
        x_values.append(sequence)
        targets.append(float(rows[3]["recovery_score"]) / 100.0)
        ids.append(patient_id)
    return np.asarray(x_values, dtype=np.float32), np.asarray(targets, dtype=np.float32), ids


class RecoveryLSTM(nn.Module):
    def __init__(self, input_size: int = 7, hidden_size: int = 16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(x)
        return torch.sigmoid(self.output(sequence[:, -1])).squeeze(1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_recovery_points": float(mean_absolute_error(y_true * 100, y_pred * 100)),
        "rmse_recovery_points": float(np.sqrt(mean_squared_error(y_true * 100, y_pred * 100))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "module4_recovery/data/simulated_recovery.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "module4_recovery/artifacts")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    x, y, ids = read_sequences(args.data)
    order = np.arange(len(y))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(order)
    n_train, n_val = int(len(y) * 0.70), int(len(y) * 0.15)
    train_idx, val_idx, test_idx = order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]
    x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # Baseline: use the three observed recovery scores to estimate month 12.
    baseline_scaler = StandardScaler()
    x_train_flat = x_train[:, :, -1].reshape(len(x_train), -1)
    x_test_flat = x_test[:, :, -1].reshape(len(x_test), -1)
    baseline = Ridge(alpha=1.0).fit(baseline_scaler.fit_transform(x_train_flat), y_train)
    baseline_pred = np.clip(baseline.predict(baseline_scaler.transform(x_test_flat)), 0, 1)

    model = RecoveryLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    best_val, best_state, stale = float("inf"), None, 0
    train_tensor = torch.from_numpy(x_train)
    target_tensor = torch.from_numpy(y_train)
    for _epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(train_tensor), target_tensor)
        train_loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(torch.from_numpy(x_val)), torch.from_numpy(y_val)))
        if val_loss < best_val:
            best_val, stale = val_loss, 0
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 12:
                break
    if best_state is None:
        raise RuntimeError("LSTM did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        lstm_pred = model(torch.from_numpy(x_test)).numpy().clip(0, 1)

    results = {
        "record_type": "SIMULATED only",
        "patients": len(y),
        "train_patients": len(train_idx),
        "validation_patients": len(val_idx),
        "test_patients": len(test_idx),
        "baseline_ridge": metrics(y_test, baseline_pred),
        "lstm": metrics(y_test, lstm_pred),
        "note": "These scores describe only the generated teaching data. They are not clinical recovery estimates.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_size": 7, "record_type": "SIMULATED"}, args.output_dir / "simulated_lstm.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["synthetic_id", "true_recovery_score", "ridge_prediction", "lstm_prediction"])
        for i, patient_index in enumerate(test_idx):
            writer.writerow([ids[patient_index], round(float(y_test[i] * 100), 1), round(float(baseline_pred[i] * 100), 1), round(float(lstm_pred[i] * 100), 1)])
    print(f"Simulated-data LSTM test MAE: {results['lstm']['mae_recovery_points']:.2f} points")
    print(f"No real KneeMRI follow-up data was used. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
