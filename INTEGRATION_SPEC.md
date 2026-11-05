# Integration Specification — Module 1 & Module 4

**Scope:** Integration-owner work only. No model retraining, no frontend rebuilding, no changes to Modules 2, 3, or 5.

---

## 1. Adapter Contracts

### 1.1 `adapters/module1_adapter.py` — Module1Adapter

**Identity:** Image-derived ROI/intensity feature baseline using MLP and Logistic Regression.  
**NOT** a clinical tabular model. Does **not** use Age, Sex, BMI, WOMAC, or any demographic input.

#### Required Artifacts (provided by model owner)
| File | Location |
|---|---|
| `baseline_mlp.joblib` | `module1_baseline/artifacts/` |
| `baseline_logreg.joblib` | `module1_baseline/artifacts/` |

#### Input
```python
Module1Adapter().predict(
    row: dict | None,       # metadata row from splits.csv (must include ROI bounding box keys)
    volume: np.ndarray | None,   # 3-D MRI volume [slices, H, W], float32
    volumes_dir: Path | None,    # alternative: path to .pck dir; adapter calls owner's extract_features()
)
```

Required metadata keys for feature extraction: `kneeLR`, `roiX`, `roiY`, `roiZ`, `roiWidth`, `roiHeight`, `roiDepth`.

#### Output — Available (artifacts present)
```json
{
  "status": "available",
  "model_identity": "Module 1 Image-Derived ROI Feature Baselines",
  "is_clinical_tabular": false,
  "mlp": {
    "model_name": "MLP (64, 32)",
    "predicted_label": 0,
    "predicted_class": "Healthy",
    "confidence": 0.8842,
    "probabilities": {"Healthy": 0.8842, "Partial tear": 0.0815, "Complete rupture": 0.0343}
  },
  "logistic_regression": {
    "model_name": "Logistic Regression Baseline",
    "predicted_label": 0,
    "predicted_class": "Healthy",
    "confidence": 0.8124,
    "probabilities": {"Healthy": 0.8124, "Partial tear": 0.1245, "Complete rupture": 0.0631}
  },
  "features": { "roi_mean": 184.25, ... },   // 16-feature dict, exact FEATURE_NAMES ordering
  "feature_names": ["roi_mean", "roi_std", ...],
  "feature_values": [184.25, ...]
}
```

#### Output — Unavailable (artifacts missing)
```json
{
  "status": "unavailable",
  "model_identity": "Module 1 Image-Derived ROI Feature Baselines",
  "reason": "...",
  "is_clinical_tabular": false,
  "features_extracted": true,
  "features": { "roi_mean": ..., ... },     // still computed if volume + ROI metadata provided
  "benchmark_reference": {
    "mlp_test_accuracy": 0.662,
    "mlp_test_balanced_accuracy": 0.431,
    "mlp_test_macro_f1": 0.451,
    "logreg_test_accuracy": 0.612
  }
}
```

**Invariant:** `is_clinical_tabular` is always `false`.

---

### 1.2 `adapters/module4_adapter.py` — Module4Adapter

**Identity:** GRU slice-sequence classifier — 3-class ACL diagnosis from ordered slices of **one** MRI volume.  
**NOT** longitudinal recovery prediction.

#### Required Artifacts (provided by model owner)
| File | Location |
|---|---|
| `gru_checkpoint.pt` | `module4_recovery/artifacts/` |
| `resnet18.pt` (feature extractor) | `acl_mri_classification/artifacts/deploy/` ✅ already present |

#### Input
```python
Module4Adapter().predict(
    volume: np.ndarray | None,          # 3-D MRI volume [slices, H, W]
    row: dict | None,                   # metadata (used for ROI-aware normalization)
    image_size: int = 160,
    existing_resnet_model=None,         # optional: pass already-loaded ResNet-18
    precomputed_slice_features=None,    # optional: pre-extracted [num_slices, 512] tensor
)
```

#### Output — Available
```json
{
  "status": "available",
  "model_identity": "Module 4 GRU Slice Sequence Classifier",
  "architecture": "ResNet-18 Slice Embeddings + GRU(hidden=64, layers=1)",
  "predicted_label": 1,
  "predicted_class": "Partial tear",
  "confidence": 0.712,
  "probabilities": {"Healthy": 0.183, "Partial tear": 0.712, "Complete rupture": 0.105},
  "slices_evaluated": 20,
  "input_scope": "Ordered sagittal slices from a single MRI volume",
  "is_longitudinal_recovery": false
}
```

#### Output — Unavailable
```json
{
  "status": "unavailable",
  "model_identity": "Module 4 GRU Slice Sequence Classifier",
  "reason": "GRU sequence classifier weights (gru_checkpoint.pt) not found ...",
  "is_longitudinal_recovery": false,
  "slices_available": 28
}
```

**Invariant:** `is_longitudinal_recovery` is always `false`.

---

### 1.3 `Module4Adapter.get_simulated_recovery_trajectory()` — Simulated Only

This is a **static method**. Its output is **strictly separate** from GRU prediction output.

```python
Module4Adapter.get_simulated_recovery_trajectory(
    acl_severity: int = 0,   # 0=Healthy, 1=Partial tear, 2=Complete rupture
    treatment: str = "rehab"
)
```

Output:
```json
{
  "status": "simulated",
  "record_type": "SIMULATED",
  "label": "SIMULATED",
  "is_trained_prediction": false,
  "trajectory": [
    {"month": 0,  "score": 70.0},
    {"month": 3,  "score": 85.0},
    {"month": 6,  "score": 93.0},
    {"month": 12, "score": 98.0}
  ],
  "disclaimer": "SIMULATED TEACHING DATA ONLY. Illustrative trajectory, NOT a trained model prediction or clinical prognosis."
}
```

**Hard rules:**
- `record_type` is always `"SIMULATED"`.
- `is_trained_prediction` is always `false`.
- This object must never be merged with GRU `predict()` output.
- The UI must display a persistent `SIMULATED` badge whenever this data is shown.

---

## 2. Findings Schema

`build_structured_findings()` in `api/app.py` now returns:

```json
{
  "volume_id": "...",
  "volume_filename": "...",
  "laterality": "right",
  "roi": { ... },
  "preprocessing": { ... },
  "cnn": {
    "consensus_class": "...",
    "confidence": 0.871,
    "probabilities": { ... },
    "per_model": { ... },
    "models_evaluated": [...],
    "models_disagree": false
  },
  "tabular": null,              // Module1Adapter output or null
  "risk": null,                 // Module4Adapter output or null
  "recovery_simulation": null,  // get_simulated_recovery_trajectory() output or null
  "ground_truth": "Healthy"
}
```

**Backward compatibility:** All three new keys (`tabular`, `risk`, `recovery_simulation`) default to `null`. The `validate()` function in `module5_reporting/report.py` passes them through unchanged. Legacy payloads without these keys do not crash.

---

## 3. UI Integration

### Module 1 Card (`#mod1-card` / `#mod1-body`)
- Shows MLP and LogReg results in a 3-column table when `findings.tabular.status === 'available'`.
- Shows a muted "Not run" note with optional benchmark reference when unavailable.
- `#chip-mod1` switches to `status-run` class when available.

### Module 4 Card (`#mod4-card` / `#mod4-body`)
- Shows GRU predicted class, confidence, and per-class probabilities when `findings.risk.status === 'available'`.
- Shows a muted "Not run" note when unavailable.
- `#chip-mod4` switches to `status-run` class when available.

### Recovery Simulation Card (`#recovery-sim-card`)
- Rendered **inside** `#mod4-card` but as a visually separate section.
- Only shown when `findings.recovery_simulation.record_type === 'SIMULATED'`.
- Persistent yellow `SIMULATED` badge is always present.
- Table displays `M{month}` / score rows from the trajectory array.
- Hidden by default (`display: none`) when no simulation data is present.

---

## 4. Report Template

`render_template()` in `module5_reporting/report.py` now dynamically renders:

- **Module 1:** `AVAILABLE` with MLP/LogReg class+confidence if `tabular.status == 'available'`; otherwise "Not run (artifacts unavailable)" with benchmark reference.
- **Module 4:** `AVAILABLE` with GRU class+confidence+slices if `risk.status == 'available'`; otherwise "Not run (checkpoint unavailable)".
- **Recovery simulation:** A `[SIMULATED — NOT A TRAINED PREDICTION]` prefixed trajectory line, only when `recovery_simulation.record_type == 'SIMULATED'`.

---

## 5. Test Fixtures

| File | Purpose |
|---|---|
| `tests/fixtures/module1_available.json` | Module 1 available-state response (MLP + LogReg) |
| `tests/fixtures/module1_unavailable.json` | Module 1 unavailable-state response |
| `tests/fixtures/module1_failure.json` | Module 1 error-state response |
| `tests/fixtures/module4_available.json` | Module 4 available-state response (GRU 3-class) |
| `tests/fixtures/module4_unavailable.json` | Module 4 unavailable-state response |
| `tests/fixtures/module4_failure.json` | Module 4 error-state response |
| `tests/fixtures/recovery_simulation.json` | Simulated trajectory (SIMULATED, 4 timepoints) |
| `tests/fixtures/complete_findings.json` | Full 5-module findings object (all unavailable) |

> [!CAUTION]
> Fixtures are **test-only**. They must never be used as production model outputs or presented as genuine predictions in the UI.

---

## 6. Running Tests

```bash
cd "C:\Users\saanv\OneDrive\Desktop\Multimodel_knee_MRI-main (1)\Multimodel_knee_MRI-main\Multimodel_knee_MRI-main"
python -m pytest tests/test_integration.py -v
```

Expected: **19 passed** (all adapters return `unavailable` since model artifacts not yet provided by model owner — this is correct and expected).

---

## 7. What Remains Blocked (Model Owner Deliverables)

| Artifact | Needed by | Status |
|---|---|---|
| `module1_baseline/artifacts/baseline_mlp.joblib` | Module1Adapter `status: available` | ❌ Not provided |
| `module1_baseline/artifacts/baseline_logreg.joblib` | Module1Adapter `status: available` | ❌ Not provided |
| `module4_recovery/artifacts/gru_checkpoint.pt` | Module4Adapter `status: available` | ❌ Not provided |

When these are dropped in, the adapters will automatically switch from `unavailable` to `available` and the UI cards/chips/report will update accordingly — no further code changes required.
