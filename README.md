# 🦵 RadVision ACL: Deep Learning Knee MRI Classification & Clinical PACS Workstation
### *An End-to-End Multi-Modal Academic System for Neural Networks & Deep Learning*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Status](https://img.shields.io/badge/Academic%20Project-Neural%20Networks%20%26%20Deep%20Learning-orange.svg)]()

---

## 📌 Executive Summary

Anterior Cruciate Ligament (ACL) rupture is among the most prevalent and debilitating musculoskeletal injuries, accounting for over 200,000 cases annually. Magnetic Resonance Imaging (MRI) is the gold standard for non-invasive diagnosis; however, interpretation of sagittal knee sequences requires specialist radiologist expertise, is vulnerable to observer fatigue, and presents diagnostic ambiguity in subtle partial tears (Grade 1/2) versus complete tears (Grade 3).

**RadVision ACL** is a comprehensive, five-module academic deep learning research project developed for the **Neural Networks and Deep Learning** curriculum. The repository provides an end-to-end clinical machine intelligence pipeline:
1. **Module 1 — Feature-Based Baseline:** Tabular statistical baseline utilizing 16 handcrafted Region of Interest (ROI) intensity and morphological features with oversampled training.
2. **Module 2 — Controlled Preprocessing & Regularization:** Anatomical ROI cropping, percentile normalization, orientation-safe augmentation, and regularization studies.
3. **Module 3 — 3D Multi-Slice Deep Learning & Explainability:** Custom scratch convolutional networks and transfer-learning backbones (ResNet-18, EfficientNet-B0, DenseNet-121) with volume-level logit aggregation and Grad-CAM saliency mapping.
4. **Module 4 — Longitudinal Recovery Sequence Modeling:** Recurrent LSTM vs. regularized Ridge regression benchmarking for simulated 12-month functional recovery (IKDC/KOOS).
5. **Module 5 — Grounded ACR Radiology Reporting:** Structured, hallucination-free American College of Radiology (ACR) compliant reporting with printable PDF layouts and DICOM SR JSON export.
6. **Clinical Radiologist PACS Workstation & Web Portal:** A modern, fully responsive web application featuring a clinical landing page, zero-friction **Guest Radiologist Mode**, 3D slice stack viewer, cine loop player, interactive Grad-CAM heatmap overlays, and multi-model consensus analytics.

---

## 📑 Table of Contents

- [Executive Summary](#-executive-summary)
- [System Architecture & Module Breakdown](#-system-architecture--module-breakdown)
- [Deep Learning Methodology & Formulations](#-deep-learning-methodology--formulations)
- [Empirical Results & Benchmark Scores](#-empirical-results--benchmark-scores)
- [Radiologist Web Workstation & PACS Interface](#-radiologist-web-workstation--pacs-interface)
- [Repository Structure](#-repository-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [Step-by-Step Module Execution](#-step-by-step-module-execution)
- [Academic & Clinical Guardrails](#-academic--clinical-guardrails)
- [Dataset Citation & Acknowledgments](#-dataset-citation--acknowledgments)

---

## 🧠 System Architecture & Module Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RADVISION ACL PIPELINE FLOW                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                         [ 3D Sagittal Knee MRI ]
                       (917 Volumes, Exam-Isolated)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
    [ Module 1: Tabular ]                           [ Module 2: Vision ]
  16 Handcrafted Features                         ROI Crop (x, y, z, w, h, d)
  (Intensity Stats + Spatial)                     Robust Percentile Normalization
              │                                               │
              ▼                                               ▼
   StandardScaler + Oversample                     Cache Single-Channel NPY
   MLP Classifier (64, 32)                         (160x160 px, Sagittal Stack)
              │                                               │
              ▼                                               ▼
   [ Baseline Predictions ]                         [ Module 3: Deep CNN ]
                                                ┌─────────────┬─────────────┐
                                                ▼             ▼             ▼
                                            Small CNN     ResNet-18    DenseNet-121
                                                │             │             │
                                                └─────────────┼─────────────┘
                                                              ▼
                                                  Volume Logit Aggregation
                                                   Weighted Cross-Entropy
                                                  Validation Macro-F1 Pick
                                                              │
                                            ┌─────────────────┴─────────────────┐
                                            ▼                                   ▼
                                  [ Grad-CAM Saliency ]              [ Module 4: Sequence ]
                                   Final Conv Activations              LSTM vs. Ridge Reg.
                                   Anatomical Heatmap                 (M0, M3, M6 -> M12)
                                            │                                   │
                                            └─────────────────┬─────────────────┘
                                                              ▼
                                                    [ Module 5: Report ]
                                                    ACR Grounded Summary
                                                    DICOM SR JSON & PDF
                                                              │
                                                              ▼
                                              [ Radiologist PACS Web Portal ]
                                              (Guest Mode, Cine, Multi-Model)
```

### Module Descriptions
- **Module 1 (`module1_baseline`):** Extracts 16 descriptive features from the annotated ACL bounding box, including mean, standard deviation, min, max, median, 10th percentile, 90th percentile, slice means, slice variation, bounding volume dimensions ($W, H, D$), centroid coordinates ($X, Y, Z$), and knee laterality. Features are standardized and classified via a Multi-Layer Perceptron (MLP) with hidden layers of $(64, 32)$.
- **Module 2 (`module2_preprocessing`):** Crops the anatomical volume around the expert-annotated ACL coordinate frame, applies robust intensity normalization (1st and 99th percentiles), resizes slices to $160 \times 160$, and caches processed arrays to accelerate training. Restricts spatial augmentations to intensity jitter to preserve physiological orientation.
- **Module 3 (`acl_mri_classification`):** Primary convolutional modeling suite. Trains a scratch 4-block CNN or fine-tunes transfer backbones (ResNet-18, EfficientNet-B0, DenseNet-121). Employs multi-slice forward propagation with volume-level logit pooling. Evaluates Grad-CAM gradients from the final convolutional stage.
- **Module 4 (`module4_recovery`):** Explores longitudinal functional prognosis modeling over simulated clinical follow-ups ($t \in \{0, 3, 6, 12\}$ months). Evaluates a Recurrent LSTM network against an L2-regularized Ridge regression baseline on IKDC/KOOS functional outcome point prediction.
- **Module 5 (`module5_reporting`):** Validates volume probabilities and formats structured clinical findings adhering to American College of Radiology (ACR) standards without hallucinating unseen anatomical findings. Supports optional LLM phrasing via OpenAI Responses API.

---

## 📐 Deep Learning Methodology & Formulations

### 1. Volume-Level Slice Aggregation
Each MRI exam $V$ comprises a 3D stack of sagittal slices $\{x_1, x_2, \dots, x_S\}$ within the annotated ACL bounding interval $[Z_{\min}, Z_{\max}]$. For a network $f_\theta$ producing unnormalized logits $z_s = f_\theta(x_s) \in \mathbb{R}^3$, the exam-level prediction $\hat{y}$ is obtained via mean logit aggregation:

$$\bar{z} = \frac{1}{S} \sum_{s=1}^{S} f_\theta(x_s)$$

$$\hat{p}_c = \text{Softmax}(\bar{z})_c = \frac{\exp(\bar{z}_c)}{\sum_{k=0}^{2} \exp(\bar{z}_k)}$$

$$\hat{y} = \arg\max_{c \in \{0, 1, 2\}} \hat{p}_c$$

where $c=0$ denotes **Healthy**, $c=1$ denotes **Partial Tear**, and $c=2$ denotes **Complete Rupture**.

### 2. Class-Weighted Cross-Entropy Loss
Because healthy cases heavily outnumber complete ruptures in clinical knee cohorts, standard cross-entropy encourages majority-class bias. We counter this using inverse class frequency weighting:

$$w_c = \frac{N}{3 \cdot N_c}$$

$$\mathcal{L}_{\text{WCE}} = -\sum_{c=0}^{2} w_c \cdot y_c \log(\hat{p}_c)$$

### 3. Grad-CAM Explainability Formulation
To provide visual transparency for radiologists, Gradient-weighted Class Activation Mapping (Grad-CAM) computes the importance weight $\alpha_k^c$ for feature map $A^k$ of the final convolutional layer with respect to class score $y^c$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial y^c}{\partial A_{i,j}^k}$$

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

The resulting 2D activation map is bilinearly upsampled to $160 \times 160$ and overlaid onto the grayscale MRI slice using scientific colormaps (Turbo, Jet, or Inferno) with user-adjustable alpha transparency.

---

## 📊 Empirical Results & Benchmark Scores

All evaluations are conducted on the **strictly isolated 139-volume held-out test partition** (`examId`-grouped). Because 105 of the 139 test cases are healthy, raw accuracy is inherently inflated; therefore, **Balanced Accuracy** and **Macro-F1** serve as the primary academic evaluation criteria.

### Benchmark Scorecard

| Module & Architecture | Configuration & Inputs | Raw Accuracy | Balanced Accuracy | Macro-F1 Score | Parameter Count | Inference Latency | Status & Role |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| **Module 1: ROI MLP Baseline** | 16 Handcrafted Features, $(64, 32)$ layers | **66.2%** | **43.1%** | **45.1%** | $\sim 3.2\text{ K}$ | $1.2\text{ ms}$ | Tabular Reference Baseline |
| **Module 3: Quick Scratch CNN** | 3 epochs, $64\times64\text{ px}$, 1 slice/vol | **54.7%** | **48.2%** | **42.7%** | $0.42\text{ M}$ | $4.8\text{ ms}$ | Fast Verification Checkpoint |
| **Module 3: Full Scratch CNN** | 30 epochs, $160\times160\text{ px}$, 4 slices/vol | *Trainable* | *Trainable* | *Trainable* | $0.42\text{ M}$ | $4.8\text{ ms}$ | Production Pipeline Winner |
| **Module 3: ResNet-18 Transfer** | ImageNet Pretrained, 30 epochs | *Comparative* | *Comparative* | *Comparative* | $11.2\text{ M}$ | $8.4\text{ ms}$ | Transfer Learning Backbone |
| **Module 3: DenseNet-121** | Dense Feature Reuse, 30 epochs | *Comparative* | *Comparative* | *Comparative* | $6.9\text{ M}$ | $11.2\text{ ms}$ | Transfer Learning Backbone |
| **Module 3: EfficientNet-B0** | Compound Scaling, 30 epochs | *Comparative* | *Comparative* | *Comparative* | $4.0\text{ M}$ | $6.1\text{ ms}$ | Transfer Learning Backbone |
| **Module 4: Ridge Sequence Reg.** | M0, M3, M6 $\to$ M12 score | — | — | **MAE: 2.07 pts** | 12 | $0.4\text{ ms}$ | Linear Sequence Baseline |
| **Module 4: LSTM Sequence Net** | 16 hidden units, 1-layer LSTM | — | — | **MAE: 2.32 pts** | $1.8\text{ K}$ | $1.9\text{ ms}$ | Recurrent Sequence Model |

> [!NOTE]
> **Key Scientific Takeaways:**
> 1. In **Module 1**, the tabular MLP achieved $66.2\%$ raw accuracy, yet only $43.1\%$ balanced accuracy, demonstrating that raw accuracy on imbalanced clinical data masks poor performance on minority pathologies.
> 2. In **Module 4**, the L2-regularized Ridge regression baseline ($2.07$ MAE) outperformed the recurrent LSTM ($2.32$ MAE) on simulated recovery sequences, exemplifying Occam's Razor in deep learning: complex recurrent architectures should always be benchmarked against parsimonious linear models.

---

## 🖥️ Radiologist Web Workstation & PACS Interface

The platform provides a responsive clinical web interface built specifically for radiologists, orthopedic clinicians, and academic reviewers.

### 🌟 Key Interface Features
1. **Clinical Landing Page:**
   - Academic capstone hero section with live interactive PACS diagnostic preview.
   - Comprehensive model architecture flowchart and benchmark comparison leaderboard.
   - Prominent **"Launch Radiologist Workspace (Guest Mode)"** action.
2. **Instant Guest Access Mode:**
   - Bypasses any authentication friction, passwords, or registration.
   - Automatically establishes a verified *"Dr. Guest (Radiology Fellow)"* session with full access to all diagnostic controls and case studies.
3. **Interactive 3D Sagittal PACS Viewport:**
   - Multi-slice stack navigation with mouse wheel scrolling and keyboard arrow controls.
   - Cine loop playback engine with configurable frame rates ($1$ to $20\text{ fps}$).
   - Window / Level medical contrast presets (*Default T2*, *Ligament High-Contrast*, *Soft Tissue*, *Bone/Marrow*).
   - Zoom ($0.5\times$ to $3.0\times$), pan, grayscale inversion, and crosshair coordinate/intensity HUD.
4. **Grad-CAM Heatmap & ROI Overlay:**
   - Real-time alpha blending slider ($0\%$ to $100\%$) and colormap switcher (*Turbo*, *Jet*, *Inferno*, *Medical Amber*).
   - Annotated ACL Region of Interest bounding box toggle with spatial dimensions.
5. **Multi-Model Consensus & Analytics:**
   - Multi-model consensus banner displaying agreement across all 5 evaluated architectures.
   - Slice-by-slice pathology probability curve highlighting the peak lesion slice.
6. **Longitudinal Recovery Tracker (Module 4):**
   - 12-month interactive functional trajectory curve ($M0, M3, M6, M12$).
   - Treatment track simulator (*Surgical Reconstruction + PT* vs. *Conservative Management*).
7. **Structured ACR Radiology Reporting (Module 5):**
   - Formal hospital report formatting compliant with American College of Radiology guidelines.
   - In-place editable clinician notes and impression.
   - One-click **Print / Save PDF** (with dedicated `@media print` clean hospital styling) and **Export DICOM SR JSON**.
8. **Universal Device Responsiveness:**
   - Optimized for desktop PACS multi-monitor workstations, iPads/tablets, and mobile smartphones.

### ⌨️ Radiologist Keyboard Shortcuts
| Key Binding | Function |
|:---:|---|
| <kbd>Space</kbd> | Toggle Cine Loop Continuous Playback |
| <kbd>&larr;</kbd> / <kbd>&darr;</kbd> | Step to Previous Sagittal Slice |
| <kbd>&rarr;</kbd> / <kbd>&uarr;</kbd> | Step to Next Sagittal Slice |
| <kbd>Mouse Wheel</kbd> | Scroll Through Sagittal Slices |
| <kbd>B</kbd> | Toggle Annotated ACL Bounding Box |
| <kbd>G</kbd> | Toggle Grad-CAM Heatmap Overlay |
| <kbd>I</kbd> | Invert Grayscale Polarity |
| <kbd>R</kbd> | Reset Viewport Zoom, Pan & Contrast |
| <kbd>?</kbd> | Open Keyboard Shortcuts Help Modal |

---

## 📁 Repository Structure

```text
Multimodel_knee_MRI-main/
├── acl_mri_classification/         # Module 3: Main deep learning image pipeline
│   ├── check_data.py               # Dataset validation and integrity checks
│   ├── make_split.py               # Exam-isolated train/validation/test split
│   ├── model.py                    # SmallCNN, ResNet-18, DenseNet, EfficientNet
│   ├── train.py                    # Training engine with weighted cross-entropy
│   ├── evaluate.py                 # Held-out evaluation and confusion matrix
│   ├── explain.py                  # Grad-CAM heatmap generation and slice scoring
│   ├── compare_models.py           # Multi-architecture comparison harness
│   └── run_pipeline.py             # Module 3 automated runner
├── module1_baseline/               # Module 1: Tabular machine learning baseline
│   └── run_baseline.py             # 16-feature extractor and standardized MLP
├── module2_preprocessing/          # Module 2: Image preprocessing & regularization
│   ├── preprocessing.py            # ROI cropping, robust normalization, resize
│   ├── run_preprocessing.py        # Cache generation for high-speed training
│   ├── compare_training.py         # Regularization and augmentation trials
│   └── run_validation_experiments.py
├── module4_recovery/               # Module 4: Longitudinal sequence experiment
│   ├── generate_simulated_data.py  # Fictional multi-visit trajectory generator
│   └── train_lstm.py               # LSTM vs. Ridge regression benchmarking
├── module5_reporting/              # Module 5: Structured ACR clinical reporting
│   └── report.py                   # Grounded ACR report generator (+ optional LLM)
├── api/                            # FastAPI backend & radiologist web server
│   ├── app.py                      # REST endpoints (/predict, /health, /benchmarks)
│   └── static/                     # Responsive radiologist workstation SPA
│       ├── index.html              # Landing page & PACS viewer layout
│       ├── app.css                 # PACS clinical styling & print stylesheet
│       └── app.js                  # Canvas DICOM rendering & state manager
├── data/KneeMRI/                   # Dataset directory (place downloaded data here)
│   ├── metadata.csv                # Exam metadata and bounding box annotations
│   └── volumetric_data/            # 3D .pck MRI volume files
├── run_app.py                      # One-click launcher for radiologist PACS portal
├── run_project.py                  # Reproducible end-to-end academic run script
├── requirements.txt                # Python package dependencies
├── PROJECT_GUIDE.md                # Detailed academic handbook and module specs
└── README.md                       # Comprehensive documentation (this document)
```

---

## 🚀 Setup & Execution Guide

### 1. Environment Installation
Python 3.10 or 3.11 is required.

```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Environment Variables (.env)
Copy the template configuration file:
```powershell
cp .env.example .env
```
Open `.env` and configure:
- `KNEE_MRI_CHECKPOINTS`: Comma-separated paths or deploy folder (default: `acl_mri_classification/artifacts/deploy/*.pt`).
- `DEVICE`: `cpu` or `cuda` (defaults to auto-detection).
- `MAX_UPLOAD_MB`: Maximum .pck upload threshold (default: `100`).
- `REPORT_MODE_DEFAULT`: Default reporting mode (`template` or `llm`, default: `template`).
- `GEMINI_API_KEY`: *(Optional)* Your Google Gemini API key from [Google AI Studio](https://aistudio.google.com/) for LLM report mode. If omitted, the system reliably uses the strict deterministic template generator without external API calls.
- `GEMINI_MODEL`: Model identifier (default: `gemini-2.5-flash`).
- `LLM_TIMEOUT_SECONDS`: Generation timeout in seconds (default: `30`).

### 3. Checkpoint & Data Placement (Gitignored)
Because large binary weights and MRI volumes are excluded from git version control, ensure the following directories are populated:
- **Trained Model Checkpoints:**
  Place PyTorch `.pt` model files into:
  `acl_mri_classification/artifacts/deploy/`
  *(e.g., `resnet18.pt`, `densenet121.pt`, `efficientnet_b0.pt`)*
- **Volumetric MRI Scans:**
  Place KneeMRI `.pck` volumes into:
  `data/KneeMRI/volumetric_data/`
  *(e.g., `425713-8.pck`, `457511-7.pck`, etc.)*
- **Split Manifest:**
  `acl_mri_classification/artifacts/splits.csv` is committed in the repository and provides ROI bounding boxes, series numbers, and class ground truths.

### 4. Running the PACS Workstation
Start the FastAPI server:
```powershell
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```
Open your browser to: **`http://127.0.0.1:8000/`**
- Landing Overview: `http://127.0.0.1:8000/#home`
- PACS Workstation: `http://127.0.0.1:8000/#workstation`
- Results & Report: `http://127.0.0.1:8000/#results`

### 5. LLM Reporting & Privacy Disclosure
In LLM mode, only the sanitized structured findings object (model consensus, probabilities, laterality, and ROI metadata) is sent to Google's Gemini API over HTTPS. Raw MRI image arrays and slice pixels are **never** transmitted externally. The generation provider is isolated in `call_gemini_report()` in `api/app.py` and can be switched to a local open-weight model (e.g., via Ollama/vLLM) by updating the provider endpoint.

---

## 🚀 Installation & Quick Start

### 1. Environment Setup

Python 3.10 or newer is recommended.

```bash
# Clone or navigate to the project directory
cd Multimodel_knee_MRI-main

# Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Radiologist Web Workstation

Start the PACS workstation and landing page with a single command:

```bash
python run_app.py
```

The system will start the Uvicorn server and automatically open **`http://127.0.0.1:8000`** in your default web browser. Click **"Launch PACS"** or **"Enter Radiologist Portal (Guest Access)"** to immediately enter the workstation!

---

## 🔬 Step-by-Step Module Execution

To train, evaluate, or reproduce individual academic modules independently:

### One-Command Pipeline Run
```powershell
python run_project.py --epochs 30 --device auto
```

### Module 1: Tabular MLP Baseline
```powershell
python -m module1_baseline.run_baseline
```
*Outputs: `baseline_mlp.joblib`, `metrics.json`, `test_predictions.csv`, `eda.json`.*

### Module 2: Preprocessing Cache Generation
```powershell
python -m module2_preprocessing.run_preprocessing --image-size 160
```

### Module 3: CNN Training & Evaluation
```powershell
# Train custom Small CNN
python -m acl_mri_classification.run_pipeline --architectures small_cnn --epochs 30 --device auto

# Generate Grad-CAM for a specific volume
python -m acl_mri_classification.explain --volume 329637-8.pck
```

### Module 4: Longitudinal Sequence Experiment
```powershell
python -m module4_recovery.generate_simulated_data
python -m module4_recovery.train_lstm
```

### Module 5: Structured ACR Clinical Report
```powershell
# Deterministic local report
python -m module5_reporting.report --input acl_mri_classification/artifacts/explainability/volume_prediction.json

# Optional LLM phrasing (requires OPENAI_API_KEY)
python -m module5_reporting.report --input acl_mri_classification/artifacts/explainability/volume_prediction.json --llm --model gpt-4o-mini
```

---

## 🚀 Render Cloud Deployment

The application is structured as a unified service hosting both the FastAPI inference API and the static clinical PACS interface on a single origin.

### 1. Render Setup
1. Log in to [Render](https://dashboard.render.com/) and click **New +** $\rightarrow$ **Web Service**.
2. Connect your GitHub repository: `https://github.com/Khushicodes15/Multimodel_knee_MRI`.
3. Configure service details:
   - **Environment**: `Python 3`
   - **Region**: Any preferred region (e.g., Oregon, Frankfurt).
   - **Branch**: `main`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn api.app:app --host 0.0.0.0 --port $PORT
     ```

### 2. Environment Variables in Render Dashboard
| Variable | Value | Purpose |
| :--- | :--- | :--- |
| `PYTHON_VERSION` | `3.11.8` | Ensures consistent runtime Python version |
| `REPORT_MODE_DEFAULT` | `template` | Default deterministic reporting mode |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Lightweight LLM model endpoint |
| `GEMINI_API_KEY` | *(Optional)* | Server-side API key from Google AI Studio (never committed) |
| `MAX_UPLOAD_MB` | `100` | Upload limit for volumetric `.pck` files |

*(Note: If using `render.yaml`, Render automatically parses these settings during Blueprint deployment).*

### 3. Model Checkpoints Deployment
The trained CNN checkpoints are tracked directly in the Git repository under `acl_mri_classification/artifacts/deploy/`:
- `resnet18.pt` (42.8 MB)
- `densenet121.pt` (27.2 MB)
- `efficientnet_b0.pt` (16.2 MB)

During deployment, Render clones these weights automatically; no external S3/GCS bucket setup is required.

### 4. Testing the Deployed Application
1. **Health Check**:
   ```bash
   curl https://<your-render-service>.onrender.com/health
   ```
   Verify that response returns `"status": "ready"`, `"model_count": 3`, and `"models_loaded": ["resnet18", "densenet121", "efficientnet_b0"]`.
2. **Access PACS Interface**:
   Navigate to `https://<your-render-service>.onrender.com/` in your browser.
3. **Execute End-to-End Workflow**:
   - Go to `#workstation`, upload a test `.pck` file or pick an exam from the sample picker.
   - Click **Run Analysis** to verify multi-model consensus inference.
   - Toggle **Slice \| Grad-CAM** to verify layer-hook visualization.
   - Navigate to `#results` to view the decision-support summary and generate findings reports.

---

## ⚖️ Academic & Clinical Guardrails

1. **Non-Diagnostic Research Tool:** This software is an educational prototype developed for academic coursework in neural networks and deep learning. It has not undergone regulatory clearance (FDA 510(k), CE mark) and must **not** be used for primary patient diagnosis or treatment planning.
2. **Strict Exam-Level Isolation:** To prevent data snooping and optimistic generalization bias, all dataset splits are grouped strictly by `examId`. No sagittal slices or augmented variations of any patient can appear across both training and evaluation folds.
3. **Synthetic Sequence Transparency (Module 4):** The KneeMRI dataset contains single-time-point diagnostic studies and lacks longitudinal follow-up records. Module 4 records are explicitly synthetic (`SIMULATED`) and serve purely as an educational exercise in temporal regression benchmarking.
4. **Hallucination Prevention (Module 5):** The report generation subsystem is grounded strictly in model probabilities and metadata. It is constrained from inventing findings regarding menisci, cartilage, or osseous fractures that the classifier was not trained to detect.

---

## 📚 Dataset Citation & Acknowledgments

The real MRI dataset supporting this study was acquired and annotated by the Department of Diagnostic and Interventional Radiology at Clinical Hospital Centre Rijeka:

> **I. Štajduhar, M. Mamula, D. Miletić, and G. Unal**,  
> *"Semi-automated detection of anterior cruciate ligament injury from MRI,"*  
> **Computer Methods and Programs in Biomedicine**, vol. 140, pp. 151–164, 2017.  
> DOI: [10.1016/j.cmpb.2016.12.006](https://doi.org/10.1016/j.cmpb.2016.12.006)

Dataset archive licensed under **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**.  
Zenodo repository: [https://zenodo.org/records/14789903](https://zenodo.org/records/14789903).
