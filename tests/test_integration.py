"""Integration tests for Module 1 and Module 4 adapters and pipeline wiring.

Run with:
    python -m pytest tests/test_integration.py -v

These tests run without trained model artifacts (all adapters return 'unavailable').
They verify the adapter contracts, fixture schemas, and reporting pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from adapters.module1_adapter import Module1Adapter
from adapters.module4_adapter import Module4Adapter
from module5_reporting.report import render_template, validate

FIXTURES = ROOT / "tests" / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Module 1 Adapter
# ---------------------------------------------------------------------------

class TestModule1Adapter:
    def test_predict_none_volume_returns_unavailable(self):
        r = Module1Adapter().predict(None)
        assert r["status"] == "unavailable"
        assert r.get("is_clinical_tabular") is False

    def test_predict_with_numpy_volume_returns_16_features(self):
        m = Module1Adapter()
        # Provide a full metadata row including ROI bounding box keys required by extract_features_from_volume
        row = {
            "volumeFilename": "test.pck",
            "label": "0",
            "kneeLR": "0",
            "roiX": "5", "roiY": "5", "roiZ": "1",
            "roiWidth": "50", "roiHeight": "50", "roiDepth": "10",
        }
        vol = np.ones((28, 320, 320), dtype=np.float32) * 100.0
        r = m.predict(row, volume=vol)
        # status is 'available' when trained .joblib artifacts are present;
        # 'unavailable' otherwise — both are correct depending on build state.
        assert r["status"] in ("available", "unavailable")
        # features dict always populated when ROI metadata supplied
        assert isinstance(r.get("features"), dict)
        assert len(r["features"]) == 16, f"Expected 16 features, got {len(r['features'])}"
        assert r.get("is_clinical_tabular") is False

    def test_fixture_available_schema(self):
        f = load_fixture("module1_available.json")
        assert f["status"] == "available"
        # Adapter uses "logistic_regression" key (not "logreg")
        assert "mlp" in f and "logistic_regression" in f
        assert f.get("is_clinical_tabular") is False

    def test_fixture_unavailable_schema(self):
        f = load_fixture("module1_unavailable.json")
        assert f["status"] == "unavailable"
        assert f.get("is_clinical_tabular") is False

    def test_fixture_failure_schema(self):
        f = load_fixture("module1_failure.json")
        assert f["status"] == "error"
        assert f.get("is_clinical_tabular") is False


# ---------------------------------------------------------------------------
# Module 4 Adapter
# ---------------------------------------------------------------------------

class TestModule4Adapter:
    def test_predict_none_returns_unavailable(self):
        r = Module4Adapter().predict(None, None)
        assert r["status"] == "unavailable"
        assert r.get("is_longitudinal_recovery") is False

    def test_predict_with_volume_returns_unavailable_no_checkpoint(self):
        m = Module4Adapter()
        vol = np.zeros((28, 320, 320), dtype=np.float32)
        r = m.predict(vol, {})
        assert r["status"] == "unavailable"
        assert r.get("is_longitudinal_recovery") is False

    def test_simulated_trajectory_schema(self):
        # get_simulated_recovery_trajectory takes an int (0=Healthy, 1=Partial, 2=Rupture)
        t = Module4Adapter.get_simulated_recovery_trajectory(1)  # partial tear
        assert t["record_type"] == "SIMULATED"
        assert t.get("is_trained_prediction") is False
        assert len(t["trajectory"]) == 4
        # Trajectory dicts use "month" (int) key — matches adapter output
        months = [pt["month"] for pt in t["trajectory"]]
        assert 0 in months and 12 in months

    def test_simulated_trajectory_never_mixes_with_gru(self):
        """Simulated trajectory must be a separate object from GRU predict output."""
        m = Module4Adapter()
        gru_result = m.predict(None, None)
        sim = Module4Adapter.get_simulated_recovery_trajectory(0)  # 0 = Healthy
        # They are distinct objects; sim must NOT appear inside gru_result
        assert "trajectory" not in gru_result
        assert sim["record_type"] == "SIMULATED"

    def test_fixture_available_schema(self):
        f = load_fixture("module4_available.json")
        assert f["status"] == "available"
        assert "probabilities" in f
        assert len(f["probabilities"]) == 3
        assert f.get("is_longitudinal_recovery") is False

    def test_fixture_unavailable_schema(self):
        f = load_fixture("module4_unavailable.json")
        assert f["status"] == "unavailable"
        assert f.get("is_longitudinal_recovery") is False

    def test_fixture_recovery_simulation_schema(self):
        f = load_fixture("recovery_simulation.json")
        assert f["record_type"] == "SIMULATED"
        assert f.get("is_trained_prediction") is False
        assert len(f["trajectory"]) == 4


# ---------------------------------------------------------------------------
# Reporting pipeline
# ---------------------------------------------------------------------------

class TestReportingPipeline:
    def test_validate_passes_recovery_simulation(self):
        payload = {
            "findings": {
                "cnn": {
                    "consensus_class": "Healthy",
                    "confidence": 0.9,
                    "probabilities": {"Healthy": 0.9, "Partial tear": 0.07, "Complete rupture": 0.03},
                    "models_evaluated": ["resnet18"],
                },
                "volume_id": "test",
                "tabular": None,
                "risk": None,
                "recovery_simulation": {"record_type": "SIMULATED", "trajectory": []},
            }
        }
        v = validate(payload)
        assert "recovery_simulation" in v
        assert v["recovery_simulation"]["record_type"] == "SIMULATED"

    def test_render_template_unavailable_state(self):
        f = load_fixture("complete_findings.json")
        v = validate({"findings": f})
        report = render_template(v)
        assert "Module 1" in report
        assert "Module 4" in report
        assert "Module 2" in report
        assert "Module 3" in report

    def test_render_template_available_module1(self):
        f = load_fixture("complete_findings.json")
        f["tabular"] = load_fixture("module1_available.json")
        v = validate({"findings": f})
        report = render_template(v)
        assert "AVAILABLE" in report
        assert "MLP" in report or "mlp" in report.lower()

    def test_render_template_available_module4(self):
        f = load_fixture("complete_findings.json")
        f["risk"] = load_fixture("module4_available.json")
        v = validate({"findings": f})
        report = render_template(v)
        assert "AVAILABLE" in report
        assert "GRU" in report or "Slice" in report

    def test_render_template_simulated_trajectory_labeled(self):
        f = load_fixture("complete_findings.json")
        f["recovery_simulation"] = load_fixture("recovery_simulation.json")
        v = validate({"findings": f})
        report = render_template(v)
        assert "SIMULATED" in report

    def test_validate_legacy_payload_no_crash(self):
        """Legacy payload without tabular/risk/recovery_simulation must not crash validate()."""
        legacy = {
            "predicted_label": 0,
            "confidence": 0.85,
            "probabilities": {"Healthy": 0.85, "Partial tear": 0.1, "Complete rupture": 0.05},
        }
        v = validate(legacy)
        assert v["predicted_class"] == "Healthy"
        # Legacy path sets tabular/risk to None — no crash
        assert v.get("tabular") is None
        assert v.get("risk") is None

    def test_complete_findings_fixture_validates(self):
        f = load_fixture("complete_findings.json")
        v = validate({"findings": f})
        assert v["predicted_class"] == "Healthy"
        assert v["confidence"] == pytest.approx(0.871, abs=1e-3)
