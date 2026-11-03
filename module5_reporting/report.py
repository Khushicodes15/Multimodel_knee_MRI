"""Render a short report from model outputs without inventing MRI findings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


LABELS = {0: "Healthy", 1: "Partial tear", 2: "Complete rupture"}


def validate(payload: dict) -> dict:
    label = payload.get("predicted_label")
    if isinstance(label, bool) or not isinstance(label, int) or label not in LABELS:
        raise ValueError("predicted_label must be 0, 1, or 2")
    confidence = float(payload.get("confidence", -1))
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    probabilities = payload.get("probabilities", {})
    if probabilities and (not isinstance(probabilities, dict) or any(not 0 <= float(p) <= 1 for p in probabilities.values())):
        raise ValueError("probabilities must be a mapping of values between 0 and 1")
    return {
        "predicted_class": LABELS[label], "confidence": confidence,
        "probabilities": {str(k): float(v) for k, v in probabilities.items()},
        "source": str(payload.get("volume", "not supplied")),
        "laterality": str(payload.get("laterality", "not supplied")),
        "model": str(payload.get("model", "not supplied")),
        "split": str(payload.get("split", "not supplied")),
        "roi_slices": int(payload.get("roi_slices", 0)),
        "research_note": "Research prototype output. This is not a radiology report or clinical diagnosis.",
    }


def render_template(findings: dict) -> str:
    probs = findings["probabilities"]
    prob_line = ", ".join(f"{name}: {value:.1%}" for name, value in probs.items()) or "not supplied"
    return (
        "KNEE MRI MODEL SUMMARY (RESEARCH USE ONLY)\n\n"
        f"Input volume: {findings['source']}\n"
        f"Model prediction: {findings['predicted_class']}\n"
        f"Model confidence: {findings['confidence']:.1%}\n"
        f"Class probabilities: {prob_line}\n"
        f"Laterality: {findings['laterality']}\n"
        f"Model: {findings['model']}\n"
        f"ROI slices analyzed: {findings['roi_slices']}\n\n"
        "Limitations: This summary contains only classifier output and supplied metadata. "
        "It does not describe anatomy, establish a diagnosis, or replace review by a qualified clinician.\n"
    )


def generate_llm(findings: dict, model_name: str) -> str:
    """Optionally phrase supplied fields with the OpenAI Responses API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Optional LLM reports require the openai package. Install it from requirements.txt.") from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY to enable LLM report phrasing.")
    response = OpenAI().responses.create(
        model=model_name,
        instructions=(
            "Write a concise academic research summary using only the supplied JSON fields. "
            "Do not infer anatomy, severity beyond the predicted class, symptoms, treatment, or any unseen scan feature. "
            "Call this a model prediction, preserve uncertainty, and include the research-only limitation."
        ),
        input=json.dumps(findings, sort_keys=True),
    )
    return response.output_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="volume_prediction.json produced by explain.py")
    parser.add_argument("--output-dir", type=Path, default=Path("module5_reporting/artifacts"))
    parser.add_argument("--llm", action="store_true", help="Use optional OpenAI Responses API to phrase the supplied fields")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"), help="API model name (or set OPENAI_MODEL)")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    findings = validate(payload)
    if args.llm and not args.model:
        parser.error("--llm requires --model or OPENAI_MODEL")
    report = generate_llm(findings, args.model) if args.llm else render_template(findings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "structured_findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    (args.output_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(f"Report saved to {args.output_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
