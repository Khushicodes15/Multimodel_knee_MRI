"""Adapter for Module 1 — Image-derived ROI/Intensity Feature Baselines.

This adapter interfaces with Module 1 without rewriting the underlying model code.
Module 1 uses 16 image-derived features (ROI intensity statistics + geometric bounds)
and evaluates both MLP and Logistic Regression baselines.
It is explicitly an image-derived baseline, NOT a clinical tabular model using
Age, Sex, BMI, or WOMAC.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from module1_baseline.run_baseline import CLASSES, FEATURE_NAMES

logger = logging.getLogger("adapters.module1")


class Module1Adapter:
    """Integration adapter for Module 1 MLP and Logistic Regression baseline models."""

    def __init__(self, artifacts_dir: Path | str | None = None) -> None:
        if artifacts_dir is None:
            # Default to repo root / module1_baseline / artifacts
            self.artifacts_dir = Path(__file__).resolve().parents[1] / "module1_baseline" / "artifacts"
        else:
            self.artifacts_dir = Path(artifacts_dir)

        self.mlp_path = self.artifacts_dir / "baseline_mlp.joblib"
        self.logreg_path = self.artifacts_dir / "baseline_logreg.joblib"
        self._mlp_model: Any = None
        self._logreg_model: Any = None
        self._loaded = False

    def load_artifacts(self) -> bool:
        """Attempt to load trained MLP and Logistic Regression models if present."""
        if self._loaded:
            return self._mlp_model is not None and self._logreg_model is not None

        if self.mlp_path.is_file() and self.logreg_path.is_file():
            try:
                self._mlp_model = joblib.load(self.mlp_path)
                self._logreg_model = joblib.load(self.logreg_path)
                self._loaded = True
                return True
            except Exception as exc:
                logger.warning("Failed loading Module 1 model artifacts from %s: %s", self.artifacts_dir, exc)
                self._mlp_model = None
                self._logreg_model = None
                return False
        return False

    @staticmethod
    def extract_features_from_volume(row: dict[str, Any], volume: np.ndarray) -> list[float]:
        """Extract the exact 16 features established by the model owner from an in-memory volume.
        
        Preserves the exact schema, ordering, and formulas from module1_baseline.run_baseline.
        """
        if "kneeLR" not in row or row["kneeLR"] is None or str(row["kneeLR"]).strip() == "":
            raise KeyError("Missing required 'kneeLR' laterality in metadata row")
        try:
            knee_lr_val = float(int(str(row["kneeLR"]).strip()))
        except ValueError as exc:
            raise ValueError(f"Invalid 'kneeLR' value {row.get('kneeLR')!r}") from exc

        x, y, z = (int(row[k]) for k in ("roiX", "roiY", "roiZ"))
        w, h, d = (int(row[k]) for k in ("roiWidth", "roiHeight", "roiDepth"))
        roi = volume[z : z + d, y : y + h, x : x + w]
        if roi.size == 0:
            raise ValueError(f"Empty ROI extracted with bounds [z={z}:{z+d}, y={y}:{y+h}, x={x}:{x+w}]")

        values = roi.reshape(-1).astype(np.float32)
        slice_means = roi.mean(axis=(1, 2))

        # Nine intensity measures + 6 ROI location/dimensions + 1 knee side:
        return [
            float(values.mean()),
            float(values.std()),
            float(values.min()),
            float(values.max()),
            float(np.median(values)),
            float(np.percentile(values, 10)),
            float(np.percentile(values, 90)),
            float(slice_means.mean()),
            float(slice_means.std()),
            float(w),
            float(h),
            float(d),
            float(x),
            float(y),
            float(z),
            knee_lr_val,
        ]

    def predict(
        self,
        row: dict[str, Any] | None,
        volume: np.ndarray | None = None,
        volumes_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Run Module 1 baseline prediction.
        
        Returns a structured dictionary conforming to the agreed Module 1 contract.
        If required artifacts or inputs are missing, returns status 'unavailable'.
        """
        if row is None:
            return {
                "status": "unavailable",
                "model_identity": "Module 1 Image-Derived ROI Feature Baselines",
                "reason": "Volume metadata not provided; ROI feature extraction requires bounding box annotations.",
                "is_clinical_tabular": False,
            }

        # Check required keys for bounding box
        required_keys = ("roiX", "roiY", "roiZ", "roiWidth", "roiHeight", "roiDepth")
        if not all(k in row for k in required_keys):
            return {
                "status": "unavailable",
                "model_identity": "Module 1 Image-Derived ROI Feature Baselines",
                "reason": f"Metadata missing required ROI bounding box coordinates: {required_keys}",
                "is_clinical_tabular": False,
            }

        # Attempt feature extraction
        feature_values: list[float] | None = None
        try:
            if volume is not None:
                feature_values = self.extract_features_from_volume(row, volume)
            elif volumes_dir is not None and "volumeFilename" in row:
                from module1_baseline.run_baseline import extract_features as owner_extract
                feature_values = owner_extract(row, volumes_dir)
        except Exception as exc:
            logger.debug("Feature extraction failed: %s", exc)
            return {
                "status": "unavailable",
                "model_identity": "Module 1 Image-Derived ROI Feature Baselines",
                "reason": f"Feature extraction failed: {exc}",
                "is_clinical_tabular": False,
            }

        feature_dict = None
        if feature_values is not None:
            feature_dict = {name: round(val, 4) for name, val in zip(FEATURE_NAMES, feature_values)}

        # Check model artifacts
        has_models = self.load_artifacts()
        if not has_models or feature_values is None:
            return {
                "status": "unavailable",
                "model_identity": "Module 1 Image-Derived ROI Feature Baselines",
                "reason": "Model weight artifacts (baseline_mlp.joblib, baseline_logreg.joblib) not present in build.",
                "is_clinical_tabular": False,
                "features_extracted": feature_dict is not None,
                "features": feature_dict,
                "feature_names": FEATURE_NAMES,
                "benchmark_reference": {
                    "mlp_test_accuracy": 0.662,
                    "mlp_test_balanced_accuracy": 0.431,
                    "mlp_test_macro_f1": 0.451,
                    "logreg_test_accuracy": 0.612,
                    "notes": "Pre-computed test partition scores from module1_baseline/artifacts/metrics.json",
                },
            }

        # Run inference using the model owner's loaded pipelines
        try:
            feats_array = np.asarray([feature_values], dtype=np.float32)

            # MLP inference
            mlp_probs = self._mlp_model.predict_proba(feats_array)[0]
            mlp_pred = int(np.argmax(mlp_probs))

            # Logistic Regression inference
            lr_probs = self._logreg_model.predict_proba(feats_array)[0]
            lr_pred = int(np.argmax(lr_probs))

            return {
                "status": "available",
                "model_identity": "Module 1 Image-Derived ROI Feature Baselines",
                "is_clinical_tabular": False,
                "mlp": {
                    "model_name": "MLP (64, 32)",
                    "predicted_label": mlp_pred,
                    "predicted_class": CLASSES[mlp_pred],
                    "confidence": float(round(mlp_probs[mlp_pred], 4)),
                    "probabilities": {cls: float(round(p, 4)) for cls, p in zip(CLASSES, mlp_probs)},
                },
                "logistic_regression": {
                    "model_name": "Logistic Regression Baseline",
                    "predicted_label": lr_pred,
                    "predicted_class": CLASSES[lr_pred],
                    "confidence": float(round(lr_probs[lr_pred], 4)),
                    "probabilities": {cls: float(round(p, 4)) for cls, p in zip(CLASSES, lr_probs)},
                },
                "features": feature_dict,
                "feature_names": FEATURE_NAMES,
                "feature_values": [round(v, 4) for v in feature_values],
            }
        except Exception as exc:
            logger.error("Inference execution failed on loaded Module 1 models: %s", exc)
            return {
                "status": "unavailable",
                "model_identity": "Module 1 Image-Derived ROI Feature Baselines",
                "reason": f"Inference execution failed: {exc}",
                "is_clinical_tabular": False,
                "features": feature_dict,
            }
