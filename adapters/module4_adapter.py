"""Adapter for Module 4 — GRU sequence classification over MRI slices.

The GRU predicts a 3-class ACL diagnosis from ordered sagittal slices belonging
to ONE MRI volume. It is not a longitudinal recovery prediction model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from acl_mri_classification.model import CLASS_NAMES, build_model
from module2_preprocessing.preprocessing import normalize_resize_roi

logger = logging.getLogger("adapters.module4")


class GRUSliceSequenceClassifier(nn.Module):
    """GRU architecture used by the Module 4 training script."""

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 64,
        num_classes: int = 3,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For one unpadded sequence, the final hidden state is the final timestep.
        _, hidden = self.gru(x)
        return self.classifier(hidden[-1])


class Module4Adapter:
    """Integration adapter for Module 4 GRU slice-sequence classification."""

    def __init__(
        self,
        artifacts_dir: Path | str | None = None,
        resnet_checkpoint_path: Path | str | None = None,
        device: torch.device | None = None,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        if artifacts_dir:
            self.artifacts_dir = Path(artifacts_dir)
        else:
            seq_dir = root / "module4_sequence" / "artifacts"
            rec_dir = root / "module4_recovery" / "artifacts"
            self.artifacts_dir = seq_dir if (seq_dir / "gru_checkpoint.pt").is_file() else rec_dir

        self.gru_checkpoint_path = (
            self.artifacts_dir / "gru_checkpoint.pt"
        )

        self.resnet_checkpoint_path = (
            Path(resnet_checkpoint_path)
            if resnet_checkpoint_path
            else root
            / "acl_mri_classification"
            / "artifacts"
            / "deploy"
            / "resnet18.pt"
        )

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._gru_model: nn.Module | None = None
        self._feature_extractor: nn.Module | None = None
        self._loaded = False

    def load_artifacts(self) -> bool:
        """Load the trained GRU checkpoint if it exists and is compatible."""
        if self._loaded:
            return self._gru_model is not None

        if not self.gru_checkpoint_path.is_file():
            return False

        try:
            checkpoint = torch.load(
                self.gru_checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            config = checkpoint.get("config", {})
            model = GRUSliceSequenceClassifier(
                input_size=int(config.get("input_size", 512)),
                hidden_size=int(config.get("hidden_size", 64)),
                num_classes=int(config.get("num_classes", 3)),
                num_layers=int(config.get("num_layers", 1)),
            )
            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
            model.to(self.device).eval()
            self._gru_model = model
            self._loaded = True
            return True
        except Exception as exc:
            logger.warning(
                "Failed loading GRU checkpoint from %s: %s",
                self.gru_checkpoint_path,
                exc,
            )
            self._gru_model = None
            self._loaded = False
            return False

    def _get_feature_extractor(
        self,
        existing_resnet_model: nn.Module | None = None,
    ) -> nn.Module | None:
        """Load the same trained ResNet-18 used to generate training features."""
        if self._feature_extractor is not None:
            return self._feature_extractor

        if existing_resnet_model is not None:
            self._feature_extractor = existing_resnet_model
            return self._feature_extractor

        if not self.resnet_checkpoint_path.is_file():
            logger.warning(
                "ResNet-18 checkpoint not found: %s",
                self.resnet_checkpoint_path,
            )
            return None

        try:
            checkpoint = torch.load(
                self.resnet_checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            model = build_model(
                "resnet18",
                pretrained=False,
            ).to(self.device)
            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
            model.eval()
            self._feature_extractor = model
            return self._feature_extractor
        except Exception as exc:
            logger.warning(
                "Could not load ResNet-18 feature extractor: %s",
                exc,
            )
            return None

    def predict(
        self,
        volume: np.ndarray | None,
        row: dict[str, Any] | None,
        image_size: int = 160,
        existing_resnet_model: nn.Module | None = None,
        precomputed_slice_features: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Predict ACL diagnosis from one volume or supplied slice embeddings."""
        has_gru = self.load_artifacts()
        if not has_gru:
            return {
                "status": "unavailable",
                "model_identity": "Module 4 GRU Slice Sequence Classifier",
                "architecture": "ResNet-18 Slice Embeddings + GRU(hidden=64, layers=1)",
                "reason": (
                    "GRU sequence classifier weights (gru_checkpoint.pt) "
                    "not found or failed to load in module4_sequence/artifacts/."
                ),
                "input_scope": "Ordered sagittal slices from a single MRI volume",
                "is_longitudinal_recovery": False,
                "slices_available": int(volume.shape[0]) if volume is not None else 0,
            }

        if volume is None and precomputed_slice_features is None:
            return {
                "status": "unavailable",
                "model_identity": "Module 4 GRU Slice Sequence Classifier",
                "reason": (
                    "No volume or precomputed slice features provided "
                    "for sequence evaluation."
                ),
                "is_longitudinal_recovery": False,
            }

        try:
            if precomputed_slice_features is not None:
                if isinstance(precomputed_slice_features, np.ndarray):
                    features = torch.from_numpy(
                        precomputed_slice_features
                    ).float().to(self.device)
                else:
                    features = precomputed_slice_features.to(
                        self.device
                    ).float()

                if features.ndim == 2:
                    features = features.unsqueeze(0)
                if features.ndim != 3 or features.shape[-1] != 512:
                    raise ValueError(
                        "Expected precomputed slice features with shape "
                        "[num_slices, 512] or [1, num_slices, 512]."
                    )
                feature_sequence = features
            else:
                assert volume is not None
                feature_extractor = self._get_feature_extractor(
                    existing_resnet_model
                )
                if feature_extractor is None:
                    return {
                        "status": "unavailable",
                        "model_identity": "Module 4 GRU Slice Sequence Classifier",
                        "reason": (
                            "Trained ResNet-18 feature extractor unavailable "
                            "for slice embeddings."
                        ),
                        "is_longitudinal_recovery": False,
                    }

                normalized_slices = normalize_resize_roi(
                    volume,
                    row or {},
                    image_size=image_size,
                )
                batch = (
                    torch.from_numpy(normalized_slices.copy())
                    .float()
                    .unsqueeze(1)
                    .to(self.device)
                    / 255.0
                )

                backbone = getattr(
                    feature_extractor,
                    "backbone",
                    feature_extractor,
                )
                mean = getattr(
                    feature_extractor,
                    "mean",
                    torch.tensor(
                        [0.485, 0.456, 0.406]
                    ).view(1, 3, 1, 1).to(self.device),
                )
                std = getattr(
                    feature_extractor,
                    "std",
                    torch.tensor(
                        [0.229, 0.224, 0.225]
                    ).view(1, 3, 1, 1).to(self.device),
                )

                with torch.no_grad():
                    rgb = batch.repeat(1, 3, 1, 1)
                    rgb = (rgb - mean) / std

                    x = backbone.conv1(rgb)
                    x = backbone.bn1(x)
                    x = backbone.relu(x)
                    x = backbone.maxpool(x)
                    x = backbone.layer1(x)
                    x = backbone.layer2(x)
                    x = backbone.layer3(x)
                    x = backbone.layer4(x)
                    x = backbone.avgpool(x)
                    slice_embeddings = torch.flatten(x, 1)
                    feature_sequence = slice_embeddings.unsqueeze(0)

            with torch.no_grad():
                logits = self._gru_model(feature_sequence)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
                predicted_label = int(np.argmax(probabilities))

            return {
                "status": "available",
                "model_identity": "Module 4 GRU Slice Sequence Classifier",
                "architecture": "ResNet-18 Slice Embeddings + GRU(hidden=64, layers=1)",
                "predicted_label": predicted_label,
                "predicted_class": CLASS_NAMES[predicted_label],
                "confidence": float(round(probabilities[predicted_label], 4)),
                "probabilities": {
                    class_name: float(round(probability, 4))
                    for class_name, probability in zip(
                        CLASS_NAMES,
                        probabilities,
                    )
                },
                "slices_evaluated": int(feature_sequence.shape[1]),
                "input_scope": "Ordered sagittal slices from a single MRI volume",
                "is_longitudinal_recovery": False,
            }

        except Exception as exc:
            logger.exception("GRU sequence classification failed")
            return {
                "status": "unavailable",
                "model_identity": "Module 4 GRU Slice Sequence Classifier",
                "reason": f"Inference execution failed: {exc}",
                "is_longitudinal_recovery": False,
            }

    @staticmethod
    def get_simulated_recovery_trajectory(
        acl_severity: int = 0,
        treatment: str = "rehab",
    ) -> dict[str, Any]:
        """Return clearly labeled illustrative data, never a trained prediction."""
        severity_adjustment = {
            0: 10.0,
            1: 0.0,
            2: -15.0,
        }.get(acl_severity, 0.0)
        treatment_adjustment = (
            4.0 if treatment == "surgery" and acl_severity > 0 else 0.0
        )

        month_0 = round(
            max(20.0, min(95.0, 60.0 + severity_adjustment)),
            1,
        )
        month_3 = round(
            min(100.0, month_0 + 15.0 + treatment_adjustment * 0.5),
            1,
        )
        month_6 = round(
            min(100.0, month_3 + 8.0 + treatment_adjustment * 0.3),
            1,
        )
        month_12 = round(
            min(100.0, month_6 + 5.0 + treatment_adjustment * 0.2),
            1,
        )

        degradation_risk = (
            "low"
            if acl_severity == 0
            else "moderate"
            if acl_severity == 1
            else "elevated"
        )

        return {
            "status": "simulated",
            "record_type": "SIMULATED",
            "label": "SIMULATED",
            "time_points_months": [0, 3, 6, 12],
            "trajectory": [
                {"month": 0, "score": month_0},
                {"month": 3, "score": month_3},
                {"month": 6, "score": month_6},
                {"month": 12, "score": month_12},
            ],
            "projected_12m_score": month_12,
            "score_system": "IKDC / KOOS Functional Outcome Points (0-100 scale)",
            "degradation_risk": degradation_risk,
            "treatment_context": treatment,
            "disclaimer": (
                "SIMULATED TEACHING DATA ONLY. Illustrative trajectory, "
                "NOT a trained model prediction or clinical prognosis."
            ),
            "is_trained_prediction": False,
        }
