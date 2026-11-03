"""Optional FastAPI wrapper around a trained model's local-volume prediction."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from acl_mri_classification.model import CLASS_NAMES, build_model, read_split
from module2_preprocessing.preprocessing import normalize_resize_roi


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "acl_mri_classification/artifacts/splits.csv"
VOLUMES = ROOT / "data/KneeMRI/volumetric_data"
CHECKPOINT = Path(os.environ.get("KNEE_MRI_CHECKPOINT", ROOT / "acl_mri_classification/artifacts/model/best_model.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Knee MRI Research Classifier", version="1.0")


class PredictRequest(BaseModel):
    volume_filename: str


def load_model():
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"Checkpoint does not exist: {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config.get("architecture", "small_cnn"), pretrained=False, dropout=float(config.get("dropout", 0.35))).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


@app.get("/health")
def health():
    return {"status": "ready", "device": str(DEVICE), "checkpoint_exists": CHECKPOINT.is_file()}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        model, config = load_model()
        rows = read_split(SPLITS, VOLUMES)
        matches = [row for row in rows if row["volumeFilename"] == request.volume_filename]
        if not matches:
            raise HTTPException(status_code=404, detail="Volume is not present in the dataset split manifest")
        row = matches[0]
        with row["path"].open("rb") as stream:
            volume = np.asarray(pickle.load(stream))
        images = normalize_resize_roi(volume, row, int(config["image_size"]))
        batch = torch.from_numpy(images.copy()).float().unsqueeze(1).to(DEVICE) / 255.0
        with torch.no_grad():
            logits = model(batch).mean(dim=0)
            probabilities = torch.softmax(logits, dim=0).cpu().tolist()
        pred = int(np.argmax(probabilities))
        return {
            "volume": request.volume_filename, "predicted_label": pred,
            "predicted_class": CLASS_NAMES[pred], "confidence": probabilities[pred],
            "probabilities": dict(zip(CLASS_NAMES, probabilities)), "roi_slices": len(images),
            "split": row["split"], "research_only": True,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
