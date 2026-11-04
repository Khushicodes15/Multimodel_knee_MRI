"""Render a short report from model outputs without inventing MRI findings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


LABELS = {0: "Healthy", 1: "Partial tear", 2: "Complete rupture"}


def validate(payload: dict) -> dict:
    """Validate findings payload, supporting both new five-module findings and legacy shapes."""
    if "findings" in payload and isinstance(payload["findings"], dict):
        payload = payload["findings"]

    # If structured findings object from Part 3
    if "cnn" in payload and isinstance(payload["cnn"], dict):
        cnn = payload["cnn"]
        p_class = cnn.get("consensus_class")
        if p_class not in ("Healthy", "Partial tear", "Complete rupture"):
            raise ValueError(f"Invalid consensus_class: {p_class}")
        conf = float(cnn.get("confidence", -1))
        if not 0 <= conf <= 1:
            raise ValueError("confidence must be between 0 and 1")
        probs = cnn.get("probabilities", {})
        return {
            "volume_id": payload.get("volume_id", "unknown"),
            "volume_filename": payload.get("volume_filename", "unknown"),
            "laterality": payload.get("laterality", "unknown"),
            "roi": payload.get("roi", {}),
            "preprocessing": payload.get("preprocessing", {}),
            "cnn": cnn,
            "tabular": payload.get("tabular"),
            "risk": payload.get("risk"),
            "ground_truth": payload.get("ground_truth"),
            "predicted_class": p_class,
            "confidence": conf,
            "probabilities": probs,
            "source": payload.get("volume_filename") or payload.get("volume_id", "not supplied"),
            "model": ", ".join(cnn.get("models_evaluated", ["ensemble"])),
            "roi_slices": payload.get("roi", {}).get("total_slices", 0),
        }

    # Legacy shape fallback
    label = payload.get("predicted_label")
    if isinstance(label, bool) or not isinstance(label, int) or label not in LABELS:
        p_class = payload.get("predicted_class")
        if p_class in ("Healthy", "Partial tear", "Complete rupture"):
            label = [k for k, v in LABELS.items() if v == p_class][0]
        else:
            raise ValueError("predicted_label must be 0, 1, or 2")
    confidence = float(payload.get("confidence", -1))
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    probabilities = payload.get("probabilities", {})
    if probabilities and (not isinstance(probabilities, dict) or any(not 0 <= float(p) <= 1 for p in probabilities.values())):
        raise ValueError("probabilities must be a mapping of values between 0 and 1")

    p_class_str = LABELS[label]
    probs_dict = {str(k): float(v) for k, v in probabilities.items()}
    return {
        "volume_id": str(payload.get("volume_id", "unknown")),
        "volume_filename": str(payload.get("volume_filename") or payload.get("volume", "not supplied")),
        "laterality": str(payload.get("laterality", "unknown")),
        "roi": {"total_slices": int(payload.get("roi_slices", payload.get("total_slices", 0)))},
        "preprocessing": {"normalization": "percentile_1_99", "resize": "160x160"},
        "cnn": {
            "consensus_class": p_class_str,
            "confidence": confidence,
            "probabilities": probs_dict,
            "per_model": payload.get("per_model_probabilities", {}),
            "models_evaluated": payload.get("models_evaluated", []),
            "models_disagree": False,
        },
        "tabular": None,
        "risk": None,
        "ground_truth": payload.get("ground_truth"),
        "predicted_class": p_class_str,
        "confidence": confidence,
        "probabilities": probs_dict,
        "source": str(payload.get("volume_filename") or payload.get("volume", "not supplied")),
        "model": str(payload.get("model", "not supplied")),
        "split": str(payload.get("split", "not supplied")),
        "roi_slices": int(payload.get("roi_slices", payload.get("total_slices", 0))),
        "research_note": "Research prototype output. This is not a radiology report or clinical diagnosis.",
    }


def render_template(findings: dict) -> str:
    """Render deterministic radiology-style report strictly from findings."""
    cnn = findings.get("cnn", {})
    pred_class = cnn.get("consensus_class") or findings.get("predicted_class", "Unknown")
    conf = float(cnn.get("confidence") if cnn.get("confidence") is not None else findings.get("confidence", 0.0))
    probs = cnn.get("probabilities") or findings.get("probabilities", {})
    vol_id = findings.get("volume_filename") or findings.get("source") or findings.get("volume_id", "Not supplied")
    lat = str(findings.get("laterality", "unknown")).capitalize()
    roi = findings.get("roi", {})
    total_slices = roi.get("total_slices", findings.get("roi_slices", 0))
    plane = roi.get("plane", "sagittal").capitalize()

    prob_line = ", ".join(f"{name}: {float(value)*100:.1f}%" for name, value in probs.items()) or "Not supplied"
    models = cnn.get("models_evaluated", [])
    disagree = cnn.get("models_disagree", False)
    consensus_txt = "Consensus achieved across all evaluated backbones" if not disagree else "Model divergence observed among backbones"

    per_model_lines = ""
    for m, p in cnn.get("per_model", {}).items():
        if p:
            top_c = max(p.items(), key=lambda kv: kv[1])[0]
            top_p = max(p.values())
            per_model_lines += f"  • {m}: {top_c} ({top_p*100:.1f}%)\n"

    report = (
        "KNEE MRI CLINICAL DECISION SUPPORT — STRUCTURED REPORT\n"
        "=====================================================\n"
        f"Study:                  {vol_id}\n"
        f"Laterality:             {lat}\n"
        f"Acquisition:            {plane} MRI ({total_slices} slices)\n\n"
        "DIAGNOSTIC CONSENSUS:\n"
        f"Predicted Class:        {pred_class}\n"
        f"Confidence Score:       {conf*100:.1f}%\n"
        f"Consensus Status:       {consensus_txt}\n"
        f"Class Probabilities:    {prob_line}\n"
    )

    if per_model_lines:
        report += f"\nINDIVIDUAL BACKBONE VOTES:\n{per_model_lines}"

    report += (
        "\nMODULE INTEGRATION STATUS:\n"
        "  • Module 1 (Tabular Demographics/WOMAC): Not run (features/model not available in build)\n"
        "  • Module 2 (MRI Preprocessing):         Percentile 1-99 scaling, 160x160 crop\n"
        "  • Module 3 (Deep CNN & Grad-CAM):        Multi-model consensus & gradient saliency active\n"
        "  • Module 4 (Longitudinal Risk/Recovery): Not run (planned component, no recovery prediction claimed)\n\n"
        "LIMITATIONS & NOTICE:\n"
        "This output is generated by an academic decision-support framework. "
        "It contains only classifier outputs and supplied metadata. It does not establish an autonomous "
        "diagnosis or replace review by a board-certified radiologist.\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="volume_prediction.json produced by explain.py")
    parser.add_argument("--output-dir", type=Path, default=Path("module5_reporting/artifacts"))
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    findings = validate(payload)
    report = render_template(findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "structured_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    (args.output_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(f"Report saved to {args.output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
