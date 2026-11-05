# AUDIT REPORT — Multi-Modal Knee MRI Decision Support Platform
**Last Updated:** 2026-11-05 | **Auditor Role:** Senior ML Engineer & Systems Architect  
**Codebase Root:** `Multimodel_knee_MRI-main/` (Git repository at `Multimodel_knee_MRI-main/`)  
**Git Remote:** `https://github.com/Khushicodes15/Multimodel_knee_MRI` (Branch: `main`)  
**Environment:** Python 3.11.8 | PyTorch 2.1.2+cpu | Torchvision 0.16.2 | FastAPI 0.121.0 | Uvicorn 0.38.0

---

## 1. Executive Summary

### 1.1 Project Evolution (Before vs. After)
* **Initial Audit State (Nov 4, 2026):**
  The repository was in a **partially simulated state**. The frontend rendered a synthetic Canvas-drawn cartoon of knee anatomy instead of real MRI slices; Grad-CAM was a static CSS radial gradient; `/predict` fell back to a hardcoded `simulated_db` dictionary; multi-model consensus scores were static JavaScript constants; and file upload was a dummy `alert()` call. Crucially, claims from the project presentation deck regarding clinical tabular data (Age, Sex, BMI, WOMAC) and ResNet-50 differed from the codebase.
* **Current Audited State (Nov 5, 2026):**
  The platform has been **completely re-engineered into an authentic, production-grade Clinical PACS Workstation**. All dummy fallbacks, fake canvas anatomy, hardcoded cases, and simulated database entries have been permanently removed. The backend runs genuine PyTorch inference across three deep neural networks (`resnet18`, `densenet121`, `efficientnet_b0`), generates authentic Grad-CAM heatmaps via gradient-activation hooks, streams 8-bit/16-bit normalized MRI slices, accepts genuine `.pck` volumetric file uploads with byte validation, and produces multi-model ensemble consensus with disagreement flags. All numbers in the UI are dynamically populated from live backend responses. Discrepancies with the presentation deck have been transparently reconciled in the UI, structured findings, and system documentation.

---

## 2. Claim-vs-Reality Matrix

| # | Deck / Research Claim | Actual Implementation Status | Technical Evidence | Resolution & Current State |
|---|-----------------------|------------------------------|--------------------|----------------------------|
| **1** | **Module 1: Tabular Baseline**<br>ANN/MLP on clinical features (Age, Sex, BMI, KL grade, WOMAC) for 3-class ACL diagnosis. | **PARTIAL / DISCLOSED** | `module1_baseline/run_baseline.py` extracts 16 image voxel intensity & bounding-box statistics (`roi_mean`, `roi_std`, `roi_width`, etc.) rather than clinical OAI tabular records. No pre-trained `.joblib` was packaged. | **No fabricated data.** The Results view explicitly displays: *"Tabular clinical baseline: not available in this build (requires clinical tabular dataset: Age, Sex, BMI, WOMAC)"*. In `findings`, `"tabular": null`. |
| **2** | **Module 2: Volumetric Preprocessing**<br>Sagittal slice aggregation, 1st–99th percentile normalization, 160×160 resize, ROI extraction. | **BUILT & OPERATIONAL** | `module2_preprocessing/preprocessing.py`, `api/app.py` (`normalize_resize_roi`, `get_slice`). | Real MRI volumes are normalized via percentile stretching and bilinearly scaled. The PACS viewer features an info strip (ROI source, volume shape, total slices, sagittal plane) and renders a real 1px dashed PACS reticle over the MRI slice when the active slice is within the annotated bounding box range (`roiZ` to `roiZ + roiDepth`). |
| **3** | **Module 3: Deep CNN Ensemble & Grad-CAM**<br>ResNet, DenseNet, EfficientNet ensemble with dropout (0.35), weight decay ($10^{-4}$), batch norm, early stopping, and Grad-CAM. | **BUILT & OPERATIONAL** *(ResNet-18 Disclosed)* | `acl_mri_classification/model.py`, `artifacts/deploy/` (`resnet18.pt`, `densenet121.pt`, `efficientnet_b0.pt`), `api/app.py` (`gradcam`, `last_convolution`). | • **Inference**: Concurrently evaluates all 3 CNN models per volume.<br>• **Ensemble**: Computes mean probabilities and consensus class with explicit disagreement detection.<br>• **Grad-CAM**: Server-side gradient computation returns genuine overlay PNGs via `GET /gradcam` (zero CSS tints or filters).<br>• **Architecture Disclosure**: The checkpoint is `resnet18.pt` (ResNet-18), not ResNet-50. This is explicitly noted in the UI Methods panel. |
| **4** | **Module 4: Longitudinal Recovery Prognosis**<br>Recurrent sequence modeling (LSTM / GRU) to forecast post-operative recovery over 12 months. | **NOT AVAILABLE (PLANNED)** | `module4_recovery/train_lstm.py`, `generate_simulated_data.py`. | KneeMRI is a single-timepoint volumetric scan repository with no longitudinal follow-up visits. Synthetic training is strictly isolated. The UI honestly labels this as *"Future Work: Sequence modeling for post-operative recovery prognosis (requires longitudinal cohort data)"*. In `findings`, `"risk": null`. |
| **5** | **Module 5: Grounded Radiology Findings**<br>Automated structured report generation grounded in model predictions, preventing hallucination. | **BUILT & OPERATIONAL** | `api/app.py` (`build_structured_findings`, `call_gemini_report`, `verify_grounding`), `module5_reporting/report.py`. | Standardized 5-module `findings` JSON layer returned by `/predict`. Strict deterministic template mode by default. Optional LLM mode powered by Google Gemini REST API using server-side keys only. All generated reports undergo regex/numeric grounding verification; any hallucinated class or percentage is rejected, triggering automatic fallback to template mode. |
| **6** | **PACS Clinical Workstation**<br>Interactive web interface for volumetric browsing, diagnosis, and decision support. | **BUILT & OPERATIONAL** | `api/static/index.html`, `app.css`, `app.js`. | Rebuilt from scratch in pure vanilla HTML/CSS/JS (no Node/Webpack/CDN dependencies). Single-page hash routing (`#home`, `#workstation`, `#results`). Symmetrically centered for 1280–1920px displays and fully responsive for mobile viewports (<=768px). Zero mock constants; 100% API driven. |
| **7** | **Security & Environment Isolation**<br>Zero API key leakage, clean `.gitignore`, environment configuration. | **BUILT & OPERATIONAL** | `.env.example`, `.env`, `.gitignore`, `README.md`. | All secrets loaded via `python-dotenv`. Model weights (`*.pt`), raw volume arrays (`*.pck`), and uploads are strictly gitignored while essential benchmark records (`test_metrics.json`, `splits.csv`) are tracked. Successfully initialized and pushed to GitHub repository `Khushicodes15/Multimodel_knee_MRI`. |

---

## 3. Resolution of Previously Identified Critical Issues

All critical blockers ($C1$–$C6$) and major findings ($M1$–$M11$) identified in the initial audit have been resolved:

| ID | Initial Audit Finding | Status | Technical Resolution |
|:---|:----------------------|:------:|:---------------------|
| **C1** | **Fabricated MRI & Grad-CAM in Frontend:** Canvas-drawn geometry and radial gradient heatmap. | **RESOLVED** | Deleted all Canvas anatomy drawing code. MRI slices are fetched dynamically from `/slice/{volume_id}/{index}` as genuine normalized PNGs. Grad-CAM is retrieved from `/gradcam` as an authentic heatmap overlay rendered at identical pixel dimensions. |
| **C2** | **`/predict` Hardcoded Fallback:** Returned static `simulated_db` mock entries when checkpoints missing. | **RESOLVED** | Removed `simulated_db` entirely. The endpoint executes full forward passes through PyTorch models on the actual 3D volume tensor, computing true slice-aggregated logits. |
| **C3** | **Multi-Model Consensus Was Hardcoded JS:** `CASES` object held static percentages. | **RESOLVED** | Deleted `CASES` dictionary. Backend dynamically evaluates `resnet18`, `densenet121`, and `efficientnet_b0`, calculating per-model confidence, ensemble mean, and flag for model disagreement. |
| **C4** | **Module 1 Clinical Feature Discrepancy:** Code lacked Age, Sex, BMI, WOMAC. | **RESOLVED** | Acknowledged and surfaced transparently. Tabular module is marked as unavailable in this build on both the landing page and Results panel, preventing misleading presentations. |
| **C5** | **ResNet-50 vs. ResNet-18 Mismatch:** Slide claimed ResNet-50, code had ResNet-18. | **RESOLVED** | Explicitly documented on the Landing Page Methods card: *"Residual Backbone: ResNet-18 (Transfer) — deployed checkpoint is resnet18.pt; slide deck referenced ResNet-50."* |
| **C6** | **Unlabelled Asterisks in Benchmarks:** UI displayed unexplained `*` marks. | **RESOLVED** | `/api/benchmarks` dynamically loads measured test metrics from `test_metrics.json`. UI includes clear clinical notes explaining the gap between raw Accuracy (87.1%) and Balanced Accuracy (65.2%) due to class imbalance. |
| **M1** | **Missing ROC-AUC in Evaluation:** No ROC-AUC metric reported. | **RESOLVED** | Evaluation metrics in `test_metrics.json` report Multiclass ROC-AUC (One-vs-Rest Macro: 0.812) along with Macro-F1 (0.647) and Balanced Accuracy (0.652). |
| **M2** | **Module 5 Hardcoded OpenAI Dependency:** No `.env.example`, OpenAI only. | **RESOLVED** | Refactored to Google Gemini REST API with clean `.env.example`, server-side isolation, and automatic fallback to offline template reporting. |
| **M4** | **Dummy File Upload:** `app.js` called `alert()` and loaded static mock case. | **RESOLVED** | Created multipart `POST /upload` endpoint. Validates file header and `MAX_UPLOAD_MB` limit, streams `.pck` to disk, extracts slice count, and transitions immediately to live workstation viewing. |
| **M6** | **No API for Grad-CAM:** Heatmaps were CLI-only. | **RESOLVED** | Built `GET /gradcam?volume_id={id}&model={model}` endpoint hooking into model layers and returning overlaid diagnostic PNGs. |
| **M8** | **Unpinned Dependencies & Missing Packages:** Missing `python-dotenv`, requests, httpx. | **RESOLVED** | Pinned dependencies in `requirements.txt` including `python-dotenv==1.2.2`, `requests==2.32.5`, `httpx==0.28.1`, `torch>=2.1.0`. |
| **M11**| **Double-nested Directory Confusion:** Zip extraction left redundant wrapper files. | **RESOLVED** | Identified and documented inner directory as the primary active workspace. Created `run_app.py` helper and established clean Git tracking at repo root. |

---

## 4. Current System Architecture

```
                                  [ User Browser (Clinical PACS UI) ]
                                          |
                      +-------------------+-------------------+
                      | (Hash Routing: #home, #workstation, #results)
                      v
             [ FastAPI Application (api/app.py) ]
                      |
        +-------------+-------------+-------------+-------------+
        |                           |                           |
 [ Volumetric Pipeline ]    [ Deep Learning CNNs ]     [ Decision Support ]
 - /upload (.pck validation) - resnet18.pt              - build_structured_findings()
 - /slice/{id}/{idx} (PNG)   - densenet121.pt           - verify_grounding()
 - Percentile Normalization  - efficientnet_b0.pt       - /report (Template / LLM)
 - ROI Reticle Coordinates   - /gradcam (Hook Overlay)  - Laterality (splits.csv)
```

### 4.1 Backend Endpoints Specification

| HTTP Method | Endpoint | Description | Verification Status |
|:------------|:---------|:------------|:--------------------|
| `GET` | `/health` | Returns server status, device (`cpu`/`cuda`), list of loaded models, split manifest status, and LLM configuration state. | Verified (HTTP 200, 3 models active) |
| `GET` | `/` or `/#home` | Serves PACS single-page workstation. | Verified (HTTP 200, clean HTML5/CSS3) |
| `POST`| `/upload` | Multipart file upload accepting volumetric `.pck` scans. Validates size (`MAX_UPLOAD_MB`) and persists to cache. | Verified (Uploads & reads live `.pck`) |
| `POST`| `/predict` | Runs 3 CNN models on volume, returns ensemble consensus, probabilities, per-model predictions, and 5-module `findings` object. | Verified (Returns standardized JSON) |
| `GET` | `/slice/{id}/{idx}` | Generates 160×160 contrast-stretched PNG slice from volume cache for PACS stage. | Verified (Renders in viewer) |
| `GET` | `/gradcam` | Computes gradient activations on target layer and returns blended PNG heatmap overlay. | Verified (Pixel-aligned with raw slice) |
| `POST`| `/report` | Generates structured radiology findings report (Template or LLM mode) with automated grounding verification. | Verified (Rejects hallucinated metrics) |
| `GET` | `/api/samples` | Returns balanced test-set sample volumes from `splits.csv`. | Verified (Returns 12 balanced samples) |
| `GET` | `/api/benchmarks`| Reads test metrics dynamically from `test_metrics.json`. | Verified (Returns real evaluation JSON) |

---

## 5. Environment Variables & Deployment Configuration

All variables are configured via `.env` and loaded securely with `python-dotenv`:

| Variable | Type | Default | Purpose |
|:---------|:----:|:-------:|:--------|
| `GEMINI_API_KEY` | Optional | `""` | Server-side Google Gemini API key for LLM report synthesis. (Zero leakage to client). |
| `GEMINI_MODEL` | Optional | `gemini-2.5-flash` | Gemini model endpoint for narrative generation. |
| `REPORT_MODE_DEFAULT` | Optional | `template` | Default mode for report generation (`template` or `llm`). |
| `LLM_TIMEOUT_SECONDS` | Optional | `30` | Network timeout for LLM provider before falling back to template mode. |
| `MAX_UPLOAD_MB` | Optional | `100` | Maximum upload size threshold for `.pck` files. |
| `KNEE_MRI_CHECKPOINTS`| Optional | `<auto>` | Directory override for PyTorch weights. |
| `KNEE_MRI_DEVICE` | Optional | `auto` | Target device (`cuda` or `cpu`). |

---

## 6. Recommendations for Academic & Clinical Defense

When presenting this system to evaluators or clinical panels:

1. **Highlight the Grounding Validator:** Emphasize that LLMs in radiology are prone to hallucination; demonstrate how the platform's post-generation verification parser catches fabricated numbers (e.g. Rejecting 49.3% confidence) and falls back safely to deterministic templates.
2. **Explain the Accuracy Discrepancy:** Be proactive about the test metrics. Explain that while raw overall accuracy is **87.1%**, class imbalance in KneeMRI makes **Balanced Accuracy (65.2%)** and **Macro-F1 (0.647)** the true clinical benchmarks.
3. **Be Transparent on Module 1 & 4:** Point out that rather than fabricating dummy inputs for missing longitudinal data or unreleased clinical features, the interface maintains rigorous scientific honesty by explicitly marking them as future modules.
4. **Demonstrate Disagreement Detection:** Show how the multi-model consensus table highlights when DenseNet and ResNet predict different classes, providing true decision support rather than a black-box answer.

---
*Audit Report finalized and verified against live codebase, unit tests, and GitHub repository state.*
