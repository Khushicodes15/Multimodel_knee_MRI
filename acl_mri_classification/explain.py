"""Make slice predictions and a Grad-CAM image for one MRI volume."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from module2_preprocessing.preprocessing import crop_roi, normalize_resize_roi
from .model import CLASS_NAMES, RGBTransferModel, SmallCNN, build_model, read_split


def last_convolution(model: torch.nn.Module, architecture: str) -> torch.nn.Module:
    """Choose the last convolution block for the selected model."""
    if isinstance(model, SmallCNN):
        return model.features[12]
    backbone = model.backbone if isinstance(model, RGBTransferModel) else model
    if architecture == "resnet18":
        return backbone.layer4[-1]
    if architecture == "efficientnet_b0":
        return backbone.features[-1]
    if architecture == "densenet121":
        return backbone.features.denseblock4
    raise ValueError(f"No Grad-CAM layer for {architecture}")


def gradcam(model: torch.nn.Module, layer: torch.nn.Module, image: torch.Tensor, class_id: int) -> np.ndarray:
    """Return a normalized Grad-CAM heatmap for one single-channel MRI slice."""
    activations, gradients = [], []
    forward_hook = layer.register_forward_hook(lambda _m, _i, output: activations.append(output))
    backward_hook = layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))
    model.zero_grad(set_to_none=True)
    logits = model(image)
    logits[0, class_id].backward()
    forward_hook.remove()
    backward_hook.remove()
    weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
    heat = torch.relu((weights * activations[-1]).sum(dim=1, keepdim=True))
    heat = F.interpolate(heat, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    heat = heat.detach().cpu().numpy()
    return heat / max(float(heat.max()), 1e-8)


def save_overlay(gray: np.ndarray, heat: np.ndarray, output: Path) -> None:
    base = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2).astype(np.float32)
    color = np.zeros_like(base)
    color[..., 0] = heat * 255
    color[..., 1] = np.maximum(0, heat - 0.35) * 100
    overlay = np.clip(base * 0.60 + color * 0.40, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", required=True, help="Filename listed in the metadata, such as 329637-8.pck")
    parser.add_argument("--splits", type=Path, default=Path("acl_mri_classification/artifacts/splits.csv"))
    parser.add_argument("--volumes-dir", type=Path, default=Path("data/KneeMRI/volumetric_data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("acl_mri_classification/artifacts/quick_baseline/model/best_model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("acl_mri_classification/artifacts/explainability"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"Model checkpoint not found: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    architecture = config.get("architecture", "small_cnn")
    model = build_model(architecture, pretrained=False, dropout=float(config.get("dropout", 0.35))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = read_split(args.splits, args.volumes_dir)
    matches = [row for row in rows if row["volumeFilename"] == args.volume]
    if not matches:
        raise SystemExit(f"Volume not found in split file: {args.volume}")
    row = matches[0]
    with row["path"].open("rb") as f:
        volume = np.asarray(pickle.load(f))
    image_size = int(config["image_size"])
    prepared = normalize_resize_roi(volume, row, image_size)
    slice_logits = []
    slice_rows = []
    for roi_index, pixels in enumerate(prepared):
        image = torch.from_numpy(pixels.copy()).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        slice_logits.append(logits[0].detach().cpu())
        slice_rows.append((roi_index, int(row["roiZ"]) + roi_index, probs))
    volume_logits = torch.stack(slice_logits).mean(dim=0)
    volume_probs = torch.softmax(volume_logits, dim=0).numpy()
    predicted_class = int(volume_probs.argmax())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "slice_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["roi_slice", "volume_z", "p_healthy", "p_partial", "p_complete"])
        for roi_index, volume_z, probs in slice_rows:
            writer.writerow([roi_index, volume_z, *[f"{p:.6f}" for p in probs]])
    summary = {
        "volume": args.volume,
        "predicted_label": predicted_class,
        "predicted_class": CLASS_NAMES[predicted_class],
        "confidence": float(volume_probs[predicted_class]),
        "probabilities": {CLASS_NAMES[i]: float(volume_probs[i]) for i in range(3)},
        "roi_slices": len(prepared),
        "note": "Grad-CAM is a visualization of model sensitivity, not proof of anatomical localization.",
    }
    (args.output_dir / "volume_prediction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    selected = len(prepared) // 2
    image = torch.from_numpy(prepared[selected].copy()).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    heat = gradcam(model, last_convolution(model, architecture), image, predicted_class)
    save_overlay(prepared[selected].astype(np.float32) / 255.0, heat, args.output_dir / "gradcam_overlay.png")

    roi = crop_roi(volume, row)
    z = min(int(row["roiDepth"]) // 2, roi.shape[0] - 1)
    full_slice = volume[int(row["roiZ"]) + z].astype(np.float32)
    low, high = np.percentile(full_slice, (1, 99))
    full_slice = np.clip((full_slice - low) / (high - low), 0, 1) if high > low else np.zeros_like(full_slice)
    full_image = Image.fromarray(np.rint(full_slice * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(full_image)
    draw.rectangle((row["roiX"], row["roiY"], row["roiX"] + row["roiWidth"], row["roiY"] + row["roiHeight"]), outline="red", width=3)
    full_image.save(args.output_dir / "annotated_roi_box.png")
    print(f"Prediction: {CLASS_NAMES[predicted_class]} ({volume_probs[predicted_class]:.3f})")
    print(f"Slice predictions and Grad-CAM saved to {args.output_dir}")


if __name__ == "__main__":
    main()
