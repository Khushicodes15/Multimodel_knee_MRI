# Knee MRI ACL Classification Project

This is an academic project for exploring ACL injury classification from MRI
volume files. It connects five modules: a tabular baseline, image preprocessing,
CNN and transfer learning, a simulated recovery-sequence exercise, and a
structured report demo. The real KneeMRI dataset supports Modules 1–3. It does
not include longitudinal recovery records or radiologist report text, so Module
4 is explicitly simulated and Module 5 only restates model outputs.

## Project layout

```text
kneeMri/
├── data/KneeMRI/                   Original metadata.csv and volumetric_data/
├── acl_mri_classification/         Module 3 CNN, split, evaluation, Grad-CAM
├── module1_baseline/               ROI-statistics MLP baseline
├── module2_preprocessing/           ROI preprocessing and controlled variants
├── module4_recovery/                Simulated sequence generation and LSTM
├── module5_reporting/               Structured, grounded report generation
├── api/                             Optional local FastAPI demo
├── run_project.py                   End-to-end academic run
├── requirements.txt                 Python dependencies
└── PROJECT_GUIDE.md                 This documentation
```

Generated artifacts live inside each module. Do not put generated models or
caches into the source folders. The original dataset remains unchanged.

## Setup (PowerShell, from `D:\kneeMri`)

Use Python 3.10 or newer. Install a matching PyTorch/Torchvision pair using the
[official PyTorch selector](https://pytorch.org/get-started/locally/) for CPU
or the CUDA version available on the computer. Then install the remaining
dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For a new environment, create it first with `py -m venv .venv`. If `torchvision`
is not installed, the scratch CNN and non-transfer modules work, while ResNet,
EfficientNet, DenseNet and Grad-CAM for those backbones require Torchvision.
The current environment was observed to have CPU-only PyTorch and no
Torchvision; therefore the end-to-end default below uses the scratch CNN.

## One-command run

```powershell
.\.venv\Scripts\python.exe run_project.py --epochs 30 --device auto
```

The pipeline checks the data, creates an exam-grouped train/validation/test
split, trains the normal CNN, selects its best checkpoint using validation
macro-F1, evaluates once on the held-out test set, fits the MLP feature
baseline, creates/reuses preprocessed ROI cache files, then generates and
evaluates the clearly simulated recovery sequence example.

To run four extra Module 2 CNN trials for augmentation, dropout and weight
decay (takes longer):

```powershell
.\.venv\Scripts\python.exe run_project.py --epochs 10 --module2-experiments --device auto
```

The pipeline only keeps the winning CNN checkpoint at
`acl_mri_classification/artifacts/model/best_model.pt`; temporary comparison
checkpoints are deleted after use. Baseline and synthetic Module 4 model files
are saved in their own module artifact folders because they are separate
experiments, not competing MRI classifiers.

## Module guide

### Module 1 — simple machine-learning baseline

`module1_baseline.run_baseline` calculates basic ROI intensity statistics,
ROI dimensions/position and knee side, then trains a standardized MLP. It
oversamples only the training partition to reduce class imbalance. It uses the
shared group split, so its validation and test partitions match Module 3.

Outputs: `module1_baseline/artifacts/baseline_mlp.joblib`,
`metrics.json`, `test_predictions.csv`, `handcrafted_features.csv`, and
`eda.json`. The test report includes accuracy, balanced accuracy, macro-F1,
confusion matrix and per-class performance.

### Module 2 — preprocessing and regularization

`module2_preprocessing.preprocessing` crops the annotated ACL region,
normalizes each slice robustly, resizes it, and offers mild intensity/noise
augmentation for training. It does not flip or rotate scans, which could
change anatomical orientation. `run_preprocessing` writes reusable NPY files
and a report. `compare_training` optionally compares the unaugmented baseline
with augmentation, more dropout, and more weight decay using validation
macro-F1.

### Module 3 — MRI image classification

`acl_mri_classification` is the main modeling module. It loads the published
metadata and 3D `.pck` volumes, uses their ACL ROI annotations, and trains a
small CNN from scratch. Optional comparison architectures are ResNet-18,
EfficientNet-B0 and DenseNet-121. They replace the final classifier with three
outputs and can start from ImageNet weights. Training uses class-weighted
cross entropy; validation macro-F1 selects the model. Only after selection is
the test set evaluated.

The split is grouped by `examId`; slices and rows from one exam cannot leak
across train, validation and test. The target is the exam-level label: 0
Healthy, 1 Partial tear, 2 Complete rupture. Slice logits are averaged into a
volume prediction. Outputs include training history, comparison/evaluation
metrics, predictions, and one best checkpoint. `explain.py` also saves per-ROI
slice scores, a volume summary, ROI box preview, and a Grad-CAM overlay.

Example after training:

```powershell
.\.venv\Scripts\python.exe -m acl_mri_classification.explain --volume 329637-8.pck
```

Replace the example with a filename present in `metadata.csv`. The default
checkpoint path is the main model folder. Grad-CAM shows model sensitivity; it
does not prove anatomical localization.

### Module 4 — recovery/risk sequence experiment

The provided KneeMRI data has no patient follow-up, treatment outcomes, or
recovery scores. The generator therefore creates fictional records marked
`SIMULATED`, with month 0, 3, 6 and 12 visits. A Ridge regression baseline and
an LSTM predict a simulated month-12 score. These scores demonstrate sequence
modeling only; they cannot be presented as clinical evidence or linked to the
MRI exams.

### Module 5 — structured finding/report demo

The report program accepts `volume_prediction.json` from Module 3, validates
the class and probabilities, and writes structured JSON plus a concise text
summary. By default it uses a deterministic local template. Optional LLM
phrasing uses OpenAI's Responses API and requires `OPENAI_API_KEY` plus an
`OPENAI_MODEL` name; the prompt confines the output to the supplied fields.
The API usage follows the [OpenAI text-generation guide](https://developers.openai.com/api/docs/guides/text?lang=python).
No generated report can add findings that the model did not output, and every
summary states that it is research-only.

```powershell
.\.venv\Scripts\python.exe -m module5_reporting.report --input acl_mri_classification/artifacts/explainability/volume_prediction.json
```

Optional LLM mode:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:OPENAI_MODEL = "your-enabled-model"
.\.venv\Scripts\python.exe -m module5_reporting.report --input acl_mri_classification/artifacts/explainability/volume_prediction.json --llm
```

### Optional local API

Install FastAPI and Uvicorn from `requirements.txt`, train a model, then run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app:app --reload
```

Open `/docs`; `POST /predict` accepts `{"volume_filename":"...pck"}` for a
volume already listed in the local dataset manifest. This demo does not accept
arbitrary uploads and is not intended for deployment or clinical use.

## Suggested 10-member division and hand-off

Each module has two owners. Agree on one dataset version, label mapping, split
file, input/output schema, and artifact naming before parallel work starts.

| Members | Responsibility | Shared hand-off |
|---|---|---|
| 1–2 | Module 1 baseline and label/feature analysis | shared split and baseline metrics |
| 3–4 | Module 2 ROI preprocessing and controlled augmentation | cached input contract and preprocessing report |
| 5–6 | Module 3 CNN, transfer experiments and explainability | best checkpoint, evaluation and Grad-CAM artifacts |
| 7–8 | Module 4 sequence-model design and analysis | simulated dataset and clearly labeled metrics |
| 9–10 | Module 5 structured summary and API/integration/docs | prediction JSON schema, demo and final guide |

For an academically sound presentation, report the exam-level split seed,
class counts, per-class metrics, confusion matrix, selected checkpoint rule,
and the synthetic status of Module 4. Do not describe model predictions as
diagnoses.

### Current generated run snapshot

The Module 1 MLP completed on the shared 139-volume test partition with 66.2%
accuracy, 43.1% balanced accuracy and 45.1% macro-F1. The higher raw accuracy
is partly driven by the 105 healthy cases; it should not be presented alone.
The previously run quick CNN baseline (3 epochs, 64-pixel input, one sampled
ROI slice) reached 54.7% accuracy, 48.2% balanced accuracy and 42.7% macro-F1.
This quick model is saved under `acl_mri_classification/artifacts/quick_baseline/`
and is separate from the full 30-epoch pipeline checkpoint. On the synthetic
Module 4 sequence task, Ridge had 2.07-point MAE and the LSTM had 2.32-point
MAE; this demonstrates that a more complex model did not beat the baseline on
this generated data. These are reproducibility notes for this local run, not
clinical performance claims.

## Interpretation and limitations

The dataset is imbalanced, particularly for complete ACL ruptures. Accuracy
alone can hide poor performance on the smaller classes, so use balanced
accuracy, macro-F1, class recall and the confusion matrix. A single held-out
split has uncertainty; it does not establish performance at another hospital.
The ROI is provided by dataset annotations, so these experiments evaluate
classification within a known ROI rather than automatic ACL localization.
Only the winner from validation comparison should be described as the selected
model; do not choose based on test results.

The source dataset contains 917 annotated knee MRI volumes and is licensed
CC BY-NC-ND 4.0. Cite Clinical Hospital Centre Rijeka and the original paper:
I. Štajduhar, M. Mamula, D. Miletić, and G. Unal, “Semi-automated detection
of anterior cruciate ligament injury from MRI,” *Computer Methods and
Programs in Biomedicine*, 140 (2017), 151–164,
[doi:10.1016/j.cmpb.2016.12.006](https://doi.org/10.1016/j.cmpb.2016.12.006).
See the [dataset record](https://zenodo.org/records/14789903) for the license
and required acknowledgement. This software is an educational research
prototype and must not guide patient care.
