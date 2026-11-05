# Audit and Implementation Plan: Knee MRI Multi-Modal Framework

**Project Deck Reference:** `project_v2_pptx.pptx` (11 slides)  
**Audit Date:** November 5, 2026  
**Auditor:** Antigravity AI Engineer  

---

## A. Summary Table

| Module | Deck Requirements | Status | DONE | PARTIAL | MISSING | BLOCKED |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Module 1: Baseline ACL Classification** (Slide 3, 7) | Tabular baseline (MLP & Logistic Regression), 3-class classification, demographic/clinical/scores/CNN context features, metrics (Precision, Recall, F1, ROC-AUC, Confusion Matrix) | **PARTIAL / BLOCKED** | Metrics artifact exists (`metrics.json`), feature extractor script implemented | Code uses 16 MRI ROI intensity/geometric features, not clinical (Age/Sex/BMI/KL/WOMAC) | `baseline_mlp.joblib` and `baseline_logreg.joblib` model weights missing; not wired into pipeline | Retraining Module 1 needs all 917 .pck volumes (only 6 present) |
| **Module 2: MRI Preprocessing** (Slide 3, 8) | Slice extraction across 3 planes (Sagittal, Coronal, Axial), ROI localization/cropping, intensity normalization (1st-99th percentile), resizing to uniform tensor (160x160), training regularization (dropout, weight decay, batch norm, early stopping) | **PARTIAL / BLOCKED** | Sagittal ROI crop, percentile normalization (1-99), 160x160 resizing, batch norm, dropout, weight decay, early stopping implemented | Preprocessing supports Sagittal plane only; no coronal or axial slice extraction | Caching full dataset (`cache_160`) missing | Building full cache or running ablation experiments blocked by missing 911 volumes |
| **Module 3: CNN / Transfer Learning** (Slide 3, 8) | Deep CNN backbones, transfer learning (ResNet-50, DenseNet-121, EfficientNet-B0), multi-slice logit pooling to volume prediction, 3 classes, Grad-CAM saliency mapping | **DONE / PARTIAL** | Checkpoints for ResNet-18, DenseNet-121, EfficientNet-B0 deployed; volume logit pooling; Grad-CAM overlay; real API inference works | ResNet-50 specified in Slide 8, but ResNet-18 implemented and trained; Small CNN checkpoint missing | None | Full retraining of CNNs blocked by missing 911 volumes |
| **Module 4: Longitudinal Risk / Recovery** (Slide 3, 9) | Sequence modeling (LSTM or GRU) across temporal visits ($t_0, t_1, t_2 \to t_n$), joint degradation / functional recovery trajectory, transparent simulation layer (OAI / synthetic) | **PARTIAL** | Simulated sequence generator (`generate_simulated_data.py`) and LSTM trainer (`train_lstm.py`) implemented | GRU variant missing; simulation data and artifacts (`simulated_lstm.pt`, `metrics.json`) uncommitted; Module 4 output not wired into API findings (`risk = None`) | GRU model code, committed metrics file | Real OAI longitudinal data not present (slide notes that simulation layer is acceptable if labeled SIMULATED) |
| **Module 5: Grounded Radiology Reporting** (Slide 3, 10) | Convert structured findings into clinical radiology report; grounding source (RSNA 2026 clinical tone); strictly zero hallucination outside structured findings; local open-weight LLM preference (Llama-3, Mistral) | **PARTIAL** | Strict deterministic template reporting (`render_template`), hallucination verification check (`verify_grounding`), findings validation | Code currently only supports Cloud API (Google Gemini REST / OpenAI) and deterministic template fallback; no local open-weight backend (e.g. Ollama / llama.cpp / HF) | Local open-weight LLM runner | None (Template works offline; local LLM can use Ollama or HuggingFace fallback) |
| **System Integration & PACS Workstation** (Slide 11) | Upload MRI $\to$ CNN diagnosis $\to$ Temporal Risk $\to$ Grounded LLM $\to$ Clinical Report. Web PACS interface with interactive 3D viewer & Grad-CAM | **PARTIAL** | Full FastAPI server, static PACS viewer, multi-model consensus prediction, live Grad-CAM overlay, report generation endpoint | Modules 1 and 4 outputs are hardcoded to `None` in `build_structured_findings()`; `metadata.csv` missing from `data/KneeMRI/` | Automated end-to-end integration test | None (can be wired with existing deploy models and simulated Module 4 model) |

---

## B. Per-Module Detail & Audit Evidence

### Module 1: Baseline ACL Classification
- **Deck Requirement (Slide 3, 7):** Tabular baseline establishing performance floor via Logistic Regression before MLP. Inputs listed: Demographics (Age, Sex, BMI), Clinical History, Symptoms, Scores (KL grade, WOMAC), Activity, CNN context. Output: 3 ACL classes. Metrics: Precision, Recall, F1, ROC-AUC, Confusion Matrix.
- **Status:** **PARTIAL / BLOCKED**
- **Evidence:**
  - `module1_baseline/run_baseline.py:23-27`: Handcrafted features extracted are 16 MRI image statistics: `["roi_mean", "roi_std", "roi_min", "roi_max", "roi_median", "roi_p10", "roi_p90", "slice_mean", "slice_mean_std", "roi_width", "roi_height", "roi_depth", "roi_x", "roi_y", "roi_z", "knee_side"]`. The dataset does not contain demographic or clinical variables (Age/Sex/BMI/KL/WOMAC).
  - `module1_baseline/run_baseline.py:94-136`: Both Logistic Regression and MLP pipelines (`StandardScaler` + `MLPClassifier(64, 32)`) are implemented.
  - `module1_baseline/artifacts/metrics.json:1-120`: Pre-computed metrics exist: MLP test accuracy = 66.2%, balanced accuracy = 43.1%, macro-F1 = 45.1%, ROC-AUC = 0.589.
  - `module1_baseline/artifacts/`: No `.joblib` model weight file exists in the repository (`baseline_mlp.joblib` was not committed).
  - `api/app.py:269`: `"tabular": None` (Module 1 is not wired into the API findings object).
- **Gap & Blockers:**
  - Gap 1: Deck mentions clinical tabular features (Age, BMI, WOMAC, KL) whereas dataset and code use 16 MRI ROI intensity statistics.
  - Gap 2: `baseline_mlp.joblib` is missing; `run_baseline.py` cannot retrain because it iterates over all 917 `.pck` files (only 6 exist). Retraining is **BLOCKED**.

### Module 2: MRI Preprocessing
- **Deck Requirement (Slide 3, 8):** Slice extraction across Sagittal, Coronal, and Axial planes. Targeted ROI localization/cropping. Normalization (intensity scaling to uniform tensor dimensions). Training controls: Dropout, Weight Decay, Batch Norm, Early Stopping.
- **Status:** **PARTIAL / BLOCKED**
- **Evidence:**
  - `module2_preprocessing/preprocessing.py:10-33`: `crop_roi()` crops bounding box `[z:z+d, y:y+h, x:x+w]`; `normalize_resize_roi()` computes 1st and 99th percentiles and resizes to 160x160.
  - `module2_preprocessing/preprocessing.py:36-45`: `augment_intensity()` applies random brightness/contrast/jitter.
  - `acl_mri_classification/model.py:121-127`: BatchNorm2d, Dropout (0.35), and AdamW / weight-decay (1e-4) with early stopping implemented in `train.py:140-160`.
  - Planes: The KneeMRI dataset contains only sagittal planes; coronal and axial planes are not present in KneeMRI.
  - `data/KneeMRI/metadata.csv`: Missing (scripts default to `data/KneeMRI/metadata.csv` which does not exist, though `splits.csv` has the required metadata).
- **Gap & Blockers:**
  - Gap 1: Deck states "Sagittal, Coronal, and Axial planes", but dataset only has Sagittal sequences.
  - Gap 2: Full ROI cache creation (`run_preprocessing.py`) requires all 917 volumes. Blocked from full re-run until full dataset is provided.

### Module 3: CNN / Transfer Learning & Explainability
- **Deck Requirement (Slide 3, 8):** Detect ACL abnormalities (3 classes) using transfer learning. Planned models: ResNet-50, DenseNet-121, EfficientNet-B0. Slice aggregation for volume prediction. Grad-CAM saliency mapping.
- **Status:** **DONE (with ResNet-18 mismatch)**
- **Evidence:**
  - `acl_mri_classification/artifacts/deploy/`: Three trained checkpoints exist: `resnet18.pt` (44.8 MB), `densenet121.pt` (28.4 MB), `efficientnet_b0.pt` (16.4 MB).
  - `acl_mri_classification/model.py:148-173`: `build_model()` supports `small_cnn`, `resnet18`, `densenet121`, `efficientnet_b0`. ResNet-50 is not in the code.
  - `acl_mri_classification/model.py:176-194`: Slice-level logits aggregated via mean pooling for volume-level prediction.
  - `acl_mri_classification/explain.py:34-48`: `gradcam()` computes gradient activations from the final convolutional layer (`layer4[-1]`, `features[-1]`, `denseblock4`).
  - `api/app.py:646-670`: Real multi-model ensemble consensus inference works end-to-end on deployed models.
- **Gap & Blockers:**
  - Gap: Deck slide 8 lists `RESNET-50`, but trained checkpoint and code use `RESNET-18`.

### Module 4: Longitudinal Risk and Recovery Modeling
- **Deck Requirement (Slide 3, 9):** Model longitudinal risk and recovery trajectories over clinical history ($t_0 \dots t_n$). Sequence modeling via LSTM or GRU. Explicit transparency for simulated components.
- **Status:** **PARTIAL**
- **Evidence:**
  - `module4_recovery/generate_simulated_data.py:1-41`: Generates synthetic recovery curves across months 0, 3, 6, 12, explicitly tagged `"record_type": "SIMULATED"`.
  - `module4_recovery/train_lstm.py:50-59`: `RecoveryLSTM` with `nn.LSTM(input_size=7, hidden_size=16)` trained to predict month 12 recovery score from months 0, 3, 6, compared against Ridge baseline.
  - `module4_recovery/artifacts/`: Directory is missing; neither `simulated_recovery.csv` nor `metrics.json` nor `simulated_lstm.pt` were committed.
  - `module4_recovery/train_lstm.py`: GRU architecture is NOT implemented (only LSTM).
  - `api/app.py:270`: `"risk": None` (Module 4 is not integrated into findings or reporting).
- **Gap & Blockers:**
  - Gap 1: No GRU model implemented (deck says "LSTM or GRU" in Slide 9, "LSTM / GRU" in Slide 3).
  - Gap 2: Script has not been executed; artifacts are absent.
  - Gap 3: Module 4 output not wired into the API findings.
  - Note: Not blocked by dataset because deck explicitly specifies simulated data for recovery trajectories.

### Module 5: Grounded Radiology Report Generation
- **Deck Requirement (Slide 3, 10):** Convert structured findings into clinical radiology-style reports. RSNA 2026 grounding style. Safety principle: strictly prohibited from hallucinating findings not in upstream outputs. Preference for local/open-weight models (Llama-3, Mistral).
- **Status:** **PARTIAL**
- **Evidence:**
  - `module5_reporting/report.py:92-143`: Strict deterministic template rendering (`render_template`) completely grounded in findings.
  - `api/app.py:334-370`: `verify_grounding()` verifies predicted class and probability figures in report text.
  - `api/app.py:275-331`: Cloud API (`call_gemini_report`) requires `GEMINI_API_KEY` (which is empty by default).
  - No local open-weight model integration (e.g., Ollama or local HuggingFace/transformers pipeline) is present.
- **Gap & Blockers:**
  - Gap 1: Deck Slide 10 specifically states preference for local/open-weight models (Llama-3, Mistral) for privacy and zero API dependency.
  - Gap 2: Module 5 template currently reports Module 1 and Module 4 as "Not run".

### Integration & Radiologist Workstation
- **Deck Requirement (Slide 11):** End-to-end integration: MRI Upload $\to$ Preprocessing/CNN $\to$ Temporal Risk $\to$ Grounded LLM $\to$ Clinical Report.
- **Status:** **PARTIAL**
- **Evidence:**
  - `api/app.py`: Real API endpoints `/health`, `/api/samples`, `/predict`, `/gradcam`, `/report`, `/slice/{id}/{idx}` are implemented.
  - Static frontend at `api/static/index.html` provides 3D slice browsing, cine loop, and Grad-CAM overlay.
  - Missing link: `metadata.csv` missing from `data/KneeMRI/` causing scripts to crash unless patched; Module 1 and Module 4 are not called during `/predict`.

---

## C. Smoke Test Results

All smoke tests were executed against the actual repo files and runtime environment:

| Test | Target | Command / Check | Result | Details / Output |
|---|---|---|:---:|---|
| **1. Package & Module Imports** | Core libraries & project modules | Python import test | **PASS** | `torch 2.14.0`, `torchvision 0.29.2`, `PIL`, `sklearn`, `fastapi`, `uvicorn`, `python-multipart` all import cleanly. |
| **2. Volume Loading (.pck)** | `data/KneeMRI/volumetric_data/425713-8.pck` | Restricted unpickler / NumPy check | **PASS** | Loaded 3D volume: shape `(27, 320, 320)`, `uint16`, values `[0, 2224]`. |
| **3. Model Weights Loading** | 3 deployed `.pt` checkpoints | `build_model(arch)` + `load_state_dict()` | **PASS** | `resnet18.pt` (11.18M params), `densenet121.pt` (6.96M params), `efficientnet_b0.pt` (4.01M params) all loaded. |
| **4. Forward Pass Inference** | 3 backbones on test volume `425713-8.pck` | Model evaluation forward pass | **PASS** | ResNet-18: Healthy ($99.88\%$), DenseNet-121: Healthy ($99.96\%$), EfficientNet-B0: Healthy ($99.90\%$). Consensus: Healthy. |
| **5. Grad-CAM Computation** | ResNet-18 final convolution layer | `gradcam()` on slice | **PASS** | Heatmap generated: shape `(160, 160)`, normalized range `[0.0, 1.0]`. |
| **6. Report Generation** | Module 5 template renderer | `render_template(findings)` | **PASS** | Structured report formatted according to clinical decision support template. |
| **7. API Server Endpoints** | FastAPI application | Direct service test (`health`, `samples`, `predict`) | **PASS** | `/health` $\to$ status: `ready`, 3 models loaded; `/api/samples` $\to$ 6 samples found; `/predict` $\to$ consensus: Healthy ($99.88\%$). |

---

## D. Deck vs. Repo Mismatches & Proposed Decisions

| # | Item | Deck Requirement | Current Repo State | Recommendation | Rationale |
|:---:|---|---|---|:---:|---|
| **1** | **ResNet Architecture** | Slide 8: `RESNET-50` | Repo trained and deployed `resnet18.pt` | **Change slide wording** | ResNet-18 is fully trained, evaluated on test set, and deployed. Training ResNet-50 from scratch requires the missing 911 volumes and substantial compute. |
| **2** | **MRI Imaging Planes** | Slide 8: "Sagittal, Coronal, and Axial planes" | KneeMRI dataset only provides Sagittal planes | **Change slide wording** | KneeMRI (Zenodo 14789903) contains only sagittal knee volumes. Coronal/axial planes exist in MRNet, but KneeMRI has only sagittal. |
| **3** | **Module 1 Features** | Slide 7: Demographics (Age, Sex, BMI), Symptoms, KL grade, WOMAC | Code extracts 16 MRI ROI intensity statistics | **Change slide wording** | KneeMRI metadata contains only ROI coordinates and knee laterality; clinical demographics were never part of this dataset. |
| **4** | **Module 4 GRU Variant** | Slide 3 & 9: "LSTM / GRU" | Code has only `RecoveryLSTM` | **Change code** | Adding `RecoveryGRU` to `train_lstm.py` is lightweight, can be trained in seconds on simulated data, and directly satisfies the deck. |
| **5** | **Module 4 Artifacts** | Slide 9: Longitudinal trajectory modeling | `module4_recovery/artifacts/` is missing | **Change code** | Run `generate_simulated_data.py` and `train_lstm.py` (with GRU option) and commit `metrics.json`. |
| **6** | **Module 5 Local LLM Backend** | Slide 10: "Preference for local/open-weight models (Llama-3, Mistral)" | Only Gemini cloud API and template fallback | **Change code** | Add an optional local open-weight provider (Ollama / HuggingFace pipeline) with graceful template fallback, maintaining strict findings grounding. |
| **7** | **Pipeline Wiring (Mod 1 & 4)** | Slide 11: End-to-end integration MRI $\to$ CNN $\to$ Temporal Risk $\to$ LLM Report | `api/app.py:269-270` sets `tabular=None, risk=None` | **Change code** | Wire Module 1 baseline predictions (from metadata ROI stats) and Module 4 risk score into `build_structured_findings()` and report. |
| **8** | **Missing metadata.csv** | Expected by `run_pipeline`, `check_data`, `run_preprocessing` | `splits.csv` exists, `metadata.csv` missing | **Change code** | Generate `data/KneeMRI/metadata.csv` from `splits.csv` (stripping the "split" column) or add fallback to `splits.csv`. |

---

## E. Implementation Plan (Phase 2 Tasks)

Tasks are ordered from smallest change first:

### Task 1: Provide `metadata.csv` Fallback
- **What to do:** Create `data/KneeMRI/metadata.csv` by exporting `splits.csv` without the `split` column, or make scripts fall back to `splits.csv` if `metadata.csv` is missing.
- **Files to change / create:** `data/KneeMRI/metadata.csv`
- **Verification Command:** `python -c "import csv; from pathlib import Path; p = Path('data/KneeMRI/metadata.csv'); assert p.is_file(); print('Rows:', len(list(csv.DictReader(p.open()))))"`
- **Effort:** **S** (Small, 5 mins)
- **Deck Requirement:** Unblocks standard CLI scripts (`preview_rois.py`, `check_data.py`).

### Task 2: Implement GRU and Run Module 4 (Simulated Recovery)
- **What to do:** Add `RecoveryGRU` class and `--model {lstm,gru,both}` option in `module4_recovery/train_lstm.py`. Run `generate_simulated_data.py` and `train_lstm.py` to produce and commit `simulated_lstm.pt`, `simulated_gru.pt`, and `metrics.json` clearly marked `SIMULATED`.
- **Files to change:** `module4_recovery/train_lstm.py`
- **Verification Command:** `python -m module4_recovery.train_lstm --model both`
- **Effort:** **S** (Small, 15 mins)
- **Deck Requirement:** Satisfies Slide 3 & Slide 9 ("LSTM / GRU Sequence Modelling").

### Task 3: Add Local Open-Weight LLM Option & Hallucination Guard to Module 5
- **What to do:** Add support for local open-weight LLM inference (via Ollama REST API or local Transformers) alongside existing Gemini and template fallback. Ensure strict grounding check ensures no unverified findings appear in report text.
- **Files to change:** `module5_reporting/report.py`, `api/app.py`
- **Verification Command:** `python -c "from module5_reporting.report import render_template, validate; print(render_template(validate({'predicted_label': 0, 'confidence': 0.99})))"`
- **Effort:** **M** (Medium, 25 mins)
- **Deck Requirement:** Satisfies Slide 10 ("Preference for local/open-weight models", "Safety & Grounding Principle").

### Task 4: Wire Module 1 and Module 4 Outputs into Integration Findings & Report
- **What to do:** In `api/app.py` (`build_structured_findings`), if volume metadata exists, compute Module 1 ROI features and evaluate tabular prediction (or note status honestly if `.joblib` is absent). Attach Module 4 risk trajectory score. Update `module5_reporting/report.py` to reflect actual module output instead of "Not run".
- **Files to change:** `api/app.py`, `module5_reporting/report.py`
- **Verification Command:** Run `python run_project.py --check` or call `/predict` and inspect `res['findings']['tabular']` and `res['findings']['risk']`.
- **Effort:** **M** (Medium, 30 mins)
- **Deck Requirement:** Satisfies Slide 11 ("Multimodal Integration Pipeline").

### Task 5: Update Documentation, Benchmark Tables, and Slide Notes
- **What to do:** Update `README.md` and slide notes so every reported metric matches actual committed files (replace placeholder *Comparative* tags with the real ResNet-18, DenseNet-121, EfficientNet-B0 metrics, and update Module 4 MAE to match newly generated `module4_recovery/artifacts/metrics.json`).
- **Files to change:** `README.md`, `project_v2_pptx.pptx` (notes or wording where approved)
- **Verification Command:** `python -c "import json; m = json.load(open('module1_baseline/artifacts/metrics.json')); print('Backing verified:', m['accuracy'])"`
- **Effort:** **S** (Small, 15 mins)
- **Deck Requirement:** Integrity & truthfulness across deliverables.

---

## F. Items Proposed to Skip (Not Required by Deck)

1. **Full CNN Retraining from Scratch:** Deployed models (`resnet18.pt`, `densenet121.pt`, `efficientnet_b0.pt`) are already trained, evaluated, and verified.
2. **Multi-Plane (Coronal / Axial) MRI Training:** The KneeMRI dataset only has sagittal sequences; training on coronal/axial would require external datasets (MRNet) which the deck acknowledges are separate.
3. **UI Polish / Redesign:** The existing PACS workstation web interface (`index.html`, `app.js`, `app.css`) is working cleanly and supports 3D cine loop, slice inspection, and Grad-CAM overlays.
4. **Clinical Demographic Data Synthesis:** The deck Slide 9 and 11 explicitly note dataset limitations; inventing fake demographic columns for KneeMRI exams would violate data integrity rules.
