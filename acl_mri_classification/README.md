# Module 3 — ACL MRI image classification

This package contains the MRI data checks, exam-level split, ROI preprocessing,
CNN training, model comparison, held-out evaluation, and Grad-CAM explainability
code. The full project guide at [../PROJECT_GUIDE.md](../PROJECT_GUIDE.md)
documents setup, module hand-offs, interpretation, commands, limitations, and
all generated artifacts.

Quick scratch-CNN run from the repository root:

```powershell
python -m acl_mri_classification.run_pipeline --architectures small_cnn --epochs 30 --device auto
```

Transfer learning requires a compatible Torchvision installation. Available
architectures are `resnet18`, `efficientnet_b0`, and `densenet121`. The selected
model is saved as `artifacts/model/best_model.pt`; the test set is evaluated
after validation-based model selection.
