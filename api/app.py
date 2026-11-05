"""FastAPI application for Knee MRI classification and radiologist workstation."""

from __future__ import annotations

import csv
import io
import json
import os
import pickle
import re
import tempfile
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load local environment configuration from .env if present
load_dotenv()

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from acl_mri_classification.explain import gradcam, last_convolution, save_overlay
from acl_mri_classification.model import (
    CLASS_NAMES,
    RGBTransferModel,
    SmallCNN,
    build_model,
    read_split,
    select_deterministic_slice_indices,
)
from module2_preprocessing.preprocessing import crop_roi, normalize_resize_roi
from module5_reporting.report import render_template, validate

# Integration adapters (Module 1 & Module 4) — loaded lazily; return unavailable if artifacts missing
try:
    from adapters.module1_adapter import Module1Adapter as _Module1Adapter
    from adapters.module4_adapter import Module4Adapter as _Module4Adapter
    _M1 = _Module1Adapter()
    _M4 = _Module4Adapter()
except Exception:  # pragma: no cover
    _M1 = None  # type: ignore[assignment]
    _M4 = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
SPLITS = ROOT / "acl_mri_classification/artifacts/splits.csv"
VOLUMES = ROOT / "data/KneeMRI/volumetric_data"

# Environment Settings
DEVICE_STR = os.environ.get("DEVICE", "").strip().lower()
if DEVICE_STR in ("cuda", "cpu"):
    DEVICE = torch.device(DEVICE_STR)
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
except ValueError:
    MAX_UPLOAD_MB = 100

try:
    LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", "30"))
except ValueError:
    LLM_TIMEOUT_SECONDS = 30

REPORT_MODE_DEFAULT = os.environ.get("REPORT_MODE_DEFAULT", "template").strip().lower()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

UPLOAD_CACHE_DIR = Path(tempfile.gettempdir()) / "radvision_uploads"
UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_MANIFEST: dict[str, dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    global LOADED_MODELS
    if not LOADED_MODELS:
        LOADED_MODELS = discover_and_load_models()
    yield


app = FastAPI(
    title="RadVision ACL — Knee MRI Research Workstation",
    description="Deep Learning Multi-Model Classification & PACS Viewer for Knee MRI",
    version="2.2",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_cache_control_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class RestrictedUnpickler(pickle.Unpickler):
    """Restricted unpickler that allows only safe NumPy reconstructed arrays and codecs."""

    SAFE_MODULES = {
        "numpy",
        "numpy.core.multiarray",
        "numpy._core.multiarray",
        "_codecs",
        "builtins",
    }
    SAFE_NAMES = {
        "ndarray",
        "_reconstruct",
        "scalar",
        "dtype",
        "encode",
        "getattr",
        "bytearray",
        "bytes",
    }

    def find_class(self, module: str, name: str) -> Any:
        if module in self.SAFE_MODULES and (name in self.SAFE_NAMES or module.startswith("numpy")):
            return getattr(__import__(module, fromlist=[name]), name)
        raise pickle.UnpicklingError(f"Forbidden unpickling of {module}.{name}")


def safe_load_volume(data: bytes) -> np.ndarray:
    try:
        obj = RestrictedUnpickler(io.BytesIO(data)).load()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to unpickle file using safe array deserializer: {exc}",
        ) from exc
    vol = np.asarray(obj)
    if vol.ndim != 3 or min(vol.shape) <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected a 3D volumetric array; got shape {vol.shape}",
        )
    return vol


def discover_and_load_models() -> dict[str, dict[str, Any]]:
    """Load models defined in KNEE_MRI_CHECKPOINTS or default deploy directory."""
    loaded: dict[str, dict[str, Any]] = {}
    env_str = os.environ.get("KNEE_MRI_CHECKPOINTS", "").strip()
    checkpoint_specs: list[tuple[str, Path]] = []

    if env_str:
        for item in env_str.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                name, path_str = item.split("=", 1)
                checkpoint_specs.append((name.strip(), Path(path_str.strip())))
            else:
                p = Path(item)
                checkpoint_specs.append((p.stem, p))
    else:
        deploy_dir = ROOT / "acl_mri_classification/artifacts/deploy"
        if deploy_dir.is_dir():
            for p in sorted(deploy_dir.glob("*.pt")):
                checkpoint_specs.append((p.stem, p))
        single_best = ROOT / "acl_mri_classification/artifacts/model/best_model.pt"
        if not checkpoint_specs and single_best.is_file():
            checkpoint_specs.append(("best_model", single_best))

    for name, path in checkpoint_specs:
        if not path.is_file():
            continue
        try:
            cp = torch.load(path, map_location=DEVICE, weights_only=False)
            config = cp.get("config", {})
            arch = config.get("architecture", "small_cnn")
            dropout = float(config.get("dropout", 0.35))
            m = build_model(arch, pretrained=False, dropout=dropout).to(DEVICE)
            m.load_state_dict(cp["model_state_dict"])
            m.eval()
            loaded[name] = {"model": m, "config": config, "path": str(path.resolve())}
        except Exception as exc:
            print(f"Warning: Failed to load checkpoint {name} from {path}: {exc}")

    return loaded


LOADED_MODELS: dict[str, dict[str, Any]] = discover_and_load_models()


@app.on_event("startup")
def startup_event():
    global LOADED_MODELS
    if not LOADED_MODELS:
        LOADED_MODELS = discover_and_load_models()


class PredictJSONRequest(BaseModel):
    volume_filename: str | None = None
    volume_id: str | None = None


def build_structured_findings(
    volume_id: str,
    volume_filename: str | None,
    volume_shape: list[int] | None,
    total_slices: int,
    roi_meta: dict[str, Any] | None,
    roi_source: str,
    consensus_class: str,
    confidence: float,
    probabilities: dict[str, float],
    per_model: dict[str, dict[str, float]],
    models_evaluated: list[str],
    laterality: str = "unknown",
    ground_truth: str | None = None,
    _row: dict[str, Any] | None = None,
    _vol: "np.ndarray | None" = None,
) -> dict[str, Any]:
    """Integration layer: build standardized five-module findings object."""
    box = None
    if roi_meta and all(k in roi_meta for k in ("roiX", "roiY", "roiZ", "roiWidth", "roiHeight", "roiDepth")):
        try:
            box = {
                "x": int(roi_meta["roiX"]),
                "y": int(roi_meta["roiY"]),
                "z": int(roi_meta["roiZ"]),
                "width": int(roi_meta["roiWidth"]),
                "height": int(roi_meta["roiHeight"]),
                "depth": int(roi_meta["roiDepth"]),
            }
        except Exception:
            box = None

    # Determine whether models disagree on top predicted class
    model_top_picks = []
    for m_probs in per_model.values():
        if m_probs:
            top_c = max(m_probs.items(), key=lambda kv: kv[1])[0]
            model_top_picks.append(top_c)
    models_disagree = len(set(model_top_picks)) > 1 if model_top_picks else False

    # --- Module 1 adapter ---
    _tabular = None
    if _M1 is not None:
        try:
            _tabular = _M1.predict(_row, volume=_vol)
        except Exception as _e:
            _tabular = {"status": "error", "error": str(_e), "is_clinical_tabular": False}

    # --- Module 4 adapter ---
    _risk = None
    _recovery_sim = None
    if _M4 is not None:
        try:
            _risk = _M4.predict(_vol, _row)
            _acl_sev = consensus_class.lower().replace(" ", "_")
            _recovery_sim = _Module4Adapter.get_simulated_recovery_trajectory(_acl_sev)
        except Exception as _e:
            _risk = {"status": "error", "error": str(_e), "is_longitudinal_recovery": False}

    return {
        "volume_id": volume_id,
        "volume_filename": volume_filename or volume_id,
        "laterality": laterality,
        "roi": {
            "source": roi_source,
            "total_slices": total_slices,
            "volume_shape": volume_shape or [total_slices, 320, 320],
            "plane": "sagittal",
            "box": box,
        },
        "preprocessing": {
            "normalization": "percentile_1_99_clip_0_1",
            "resize": "bilinear_160x160",
            "roi_cropped": roi_source == "metadata",
        },
        "cnn": {
            "consensus_class": consensus_class,
            "ensemble_prediction": consensus_class,
            "confidence": round(confidence, 4),
            "ensemble_confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "per_model": {
                m: {k: round(v, 4) for k, v in p.items()} for m, p in per_model.items()
            },
            "models_evaluated": models_evaluated,
            "models_disagree": models_disagree,
        },
        "tabular": _tabular,
        "risk": _risk,
        "recovery_simulation": _recovery_sim,
        "ground_truth": ground_truth,
    }


def call_gemini_report(findings: dict[str, Any], model_name: str, timeout_seconds: int = 30) -> str:
    """Call Google Gemini REST API using server-side key. Key never leaks to client."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured on the server")

    # Clean findings so ground_truth is strictly never seen by LLM
    clean_findings = dict(findings)
    clean_findings.pop("ground_truth", None)

    prompt = (
        "You are an assistant for a clinical knee MRI research decision support workstation. "
        "Write a concise radiology findings summary based STRICTLY and ONLY on the following structured model output. "
        "RULES:\n"
        "1. DO NOT invent anatomy, anatomical measurements, cartilage status, meniscus findings, bone marrow edema, or symptoms.\n"
        "2. State the consensus classification and probabilities exactly as provided in the input.\n"
        "3. Explicitly report whether backbones agree or show divergence.\n"
        "4. Include laterality only if provided; do not guess.\n"
        "5. Conclude with a clear statement that this is a research-only decision-support finding requiring clinical correlation.\n\n"
        f"INPUT STRUCTURED FINDINGS:\n{json.dumps(clean_findings, indent=2)}"
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 600,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if not candidates:
                raise RuntimeError("No candidate generated by Gemini API")
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise RuntimeError("Empty parts in Gemini response")
            return parts[0].get("text", "").strip()
    except urllib.error.HTTPError as err:
        # Avoid leaking key from URL in error message
        raise RuntimeError(f"Gemini API HTTP {err.code}: {err.reason}") from None
    except urllib.error.URLError as err:
        raise RuntimeError(f"Network error contacting Gemini API: {err.reason}") from None
    except Exception as exc:
        raise RuntimeError(f"Gemini generation error: {str(exc)}") from None


def verify_grounding(report_text: str, findings: dict[str, Any]) -> tuple[bool, str]:
    """Verify that every class name and percentage in the report matches the findings."""
    text_lower = report_text.lower()
    cnn = findings.get("cnn", {}) if "cnn" in findings else findings
    consensus_class = (cnn.get("ensemble_prediction") or cnn.get("consensus_class") or "").lower()

    if consensus_class and consensus_class not in text_lower:
        return False, f"Report omitted predicted consensus class '{consensus_class}'"

    # Extract all percentages like '99.9%', '81.1%', '0.1%'
    pct_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", report_text)

    # Collect all valid percentages from findings (with rounding tolerance)
    valid_percentages = set()
    conf = cnn.get("ensemble_confidence", cnn.get("confidence"))
    if conf is not None and isinstance(conf, (int, float)):
        valid_percentages.add(round(conf * 100, 1))
        valid_percentages.add(round(conf * 100))
    for v in cnn.get("probabilities", {}).values():
        if isinstance(v, (int, float)):
            valid_percentages.add(round(v * 100, 1))
            valid_percentages.add(round(v * 100))
    for m_data in cnn.get("per_model", {}).values():
        if isinstance(m_data, dict):
            m_conf = m_data.get("confidence")
            if isinstance(m_conf, (int, float)):
                valid_percentages.add(round(m_conf * 100, 1))
                valid_percentages.add(round(m_conf * 100))
        elif isinstance(m_data, (int, float)):
            valid_percentages.add(round(m_data * 100, 1))
            valid_percentages.add(round(m_data * 100))

    for p_str in pct_matches:
        try:
            val = float(p_str)
            if not any(abs(val - vp) <= 1.0 for vp in valid_percentages):
                return False, f"Unverified percentage '{p_str}%' in report contradicts structured findings"
        except ValueError:
            continue

    return True, "passed"


@app.get("/")
@app.get("/index.html")
@app.get("/home")
@app.get("/workstation")
@app.get("/results")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return HTMLResponse("<h1>RadVision ACL Workstation</h1><p>Static files not found.</p>")


@app.get("/health")
def health():
    gemini_key_set = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    return {
        "status": "ready" if len(LOADED_MODELS) > 0 else "degraded_no_models",
        "device": str(DEVICE),
        "models_loaded": list(LOADED_MODELS.keys()),
        "model_count": len(LOADED_MODELS),
        "splits_exists": SPLITS.is_file(),
        "volumes_dir_exists": VOLUMES.is_dir(),
        "llm_configured": gemini_key_set,
        "default_report_mode": REPORT_MODE_DEFAULT,
        "gemini_model": GEMINI_MODEL,
        "max_upload_mb": MAX_UPLOAD_MB,
        "tabular_available": _M1.load_artifacts() if _M1 is not None else False,
        "risk_model_available": _M4.load_artifacts() if _M4 is not None else False,
    }


@app.get("/api/benchmarks")
def benchmarks():
    """Read metrics dynamically from artifact files. Return 404 if missing."""
    eval_metrics_file = ROOT / "acl_mri_classification/artifacts/evaluation/test_metrics.json"
    baseline_metrics_file = ROOT / "module1_baseline/artifacts/metrics.json"
    recovery_metrics_file = ROOT / "module4_recovery/artifacts/metrics.json"

    res: dict[str, Any] = {}
    if eval_metrics_file.is_file():
        try:
            res["module3_evaluation"] = json.loads(eval_metrics_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    if baseline_metrics_file.is_file():
        try:
            res["module1_baseline"] = json.loads(baseline_metrics_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    seq_metrics_file = ROOT / "module4_sequence/artifacts/metrics.json"
    if seq_metrics_file.is_file():
        try:
            m4_data = json.loads(seq_metrics_file.read_text(encoding="utf-8"))
            res["module4_sequence"] = m4_data
            res["module4_recovery"] = m4_data
        except Exception:
            pass
    elif recovery_metrics_file.is_file():
        try:
            m4_data = json.loads(recovery_metrics_file.read_text(encoding="utf-8"))
            res["module4_sequence"] = m4_data
            res["module4_recovery"] = m4_data
        except Exception:
            pass

    if not res:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No benchmark metric artifact files found. Run training and evaluation pipelines to generate metrics.",
        )
    return res


def get_manifest_rows(path: Path, volumes_dir: Path, existing_only: bool = True) -> list[dict[str, Any]]:
    """Read split manifest and return rows whose volume file exists if existing_only=True."""
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows: list[dict[str, Any]] = []
        for source in reader:
            vol_name = source.get("volumeFilename", "").strip()
            if not vol_name:
                continue
            vol_path = volumes_dir / vol_name
            if existing_only and not vol_path.is_file():
                continue
            row = dict(source)
            try:
                row["label"] = int(source["aclDiagnosis"])
                row["split"] = source.get("split", "").strip()
                row["path"] = vol_path
                for key in ("roiX", "roiY", "roiZ", "roiHeight", "roiWidth", "roiDepth"):
                    row[key] = int(source[key])
            except Exception:
                continue
            rows.append(row)
    return rows


@app.get("/api/samples")
def get_samples():
    """Return available test set samples with labels and filenames (max 12, balanced mix)."""
    if not SPLITS.is_file() or not VOLUMES.is_dir():
        return []

    test_rows = [r for r in get_manifest_rows(SPLITS, VOLUMES, existing_only=True) if r.get("split") == "test"]
    by_class: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for r in test_rows:
        lbl = int(r.get("label", -1))
        if lbl in by_class:
            by_class[lbl].append(r)

    samples: list[dict[str, Any]] = []
    max_per_class = 4
    for i in range(max_per_class):
        for lbl in (0, 1, 2):
            if i < len(by_class[lbl]):
                r = by_class[lbl][i]
                samples.append({
                    "volume_filename": r["volumeFilename"],
                    "label": int(r["label"]),
                    "label_name": CLASS_NAMES[int(r["label"])],
                })
                if len(samples) >= 12:
                    break
        if len(samples) >= 12:
            break

    return samples


def get_volume_and_metadata(
    volume_filename: str | None, volume_id: str | None
) -> tuple[np.ndarray, dict[str, Any], str, str]:
    """Retrieve volume array, row/metadata dict, roi_source tag, and identifier."""
    target_id = volume_id or volume_filename
    if target_id and target_id in UPLOAD_MANIFEST:
        entry = UPLOAD_MANIFEST[target_id]
        pck_path = Path(entry["path"])
        if not pck_path.is_file():
            raise HTTPException(status_code=404, detail=f"Cached upload file {target_id} not found on disk")
        with pck_path.open("rb") as f:
            vol = safe_load_volume(f.read())
        return vol, entry["row"], entry["roi_source"], target_id

    fn = volume_filename or volume_id
    if fn:
        if not SPLITS.is_file():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Missing split manifest at {SPLITS}. Cannot resolve volume metadata.",
            )
        if not VOLUMES.is_dir():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Missing volume data directory at {VOLUMES}.",
            )
        rows = get_manifest_rows(SPLITS, VOLUMES, existing_only=True)
        matches = [r for r in rows if r["volumeFilename"] == fn]
        if not matches:
            raise HTTPException(status_code=404, detail=f"Volume {fn} not found or volume file missing on disk")
        row = matches[0]
        with row["path"].open("rb") as f:
            vol = safe_load_volume(f.read())
        return vol, row, "metadata", fn

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Must provide volume_filename or volume_id",
    )


@app.post("/upload")
async def upload_volume(file: UploadFile = File(...)):
    """Accept and cache a single Knee MRI .pck volume (enforces MAX_UPLOAD_MB)."""
    if not file.filename.lower().endswith(".pck"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file extension. Only .pck files are accepted.",
        )
    content = await file.read()
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed threshold of {MAX_UPLOAD_MB}MB.",
        )

    vol = safe_load_volume(content)
    volume_id = f"up_{uuid.uuid4().hex[:12]}"
    dest_path = UPLOAD_CACHE_DIR / f"{volume_id}.pck"
    dest_path.write_bytes(content)

    row: dict[str, Any] | None = None
    roi_source = "center-crop-fallback"
    if SPLITS.is_file() and VOLUMES.is_dir():
        try:
            split_rows = read_split(SPLITS, VOLUMES)
            matched = [r for r in split_rows if r["volumeFilename"] == file.filename]
            if matched:
                row = dict(matched[0])
                roi_source = "metadata"
        except Exception:
            row = None

    if row is None:
        d, h, w = vol.shape
        rw, rh = int(w * 0.4), int(h * 0.4)
        rx, ry = (w - rw) // 2, (h - rh) // 2
        rd = max(1, int(d * 0.35))
        rz = (d - rd) // 2
        row = {
            "volumeFilename": file.filename,
            "roiX": rx,
            "roiY": ry,
            "roiZ": rz,
            "roiWidth": rw,
            "roiHeight": rh,
            "roiDepth": rd,
            "kneeLR": "0",
        }

    UPLOAD_MANIFEST[volume_id] = {
        "path": str(dest_path.resolve()),
        "filename": file.filename,
        "row": row,
        "roi_source": roi_source,
        "shape": list(vol.shape),
    }

    return {
        "volume_id": volume_id,
        "filename": file.filename,
        "shape": list(vol.shape),
        "roi_source": roi_source,
    }


@app.post("/predict")
async def predict(request: Request):
    """Run real inference on an uploaded or manifest volume and return structured findings."""
    if not LOADED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained model checkpoints are loaded on the server. Please provide checkpoints via KNEE_MRI_CHECKPOINTS.",
        )

    target_vol_id: str | None = None
    target_vol_fn: str | None = None

    ct = request.headers.get("content-type", "")
    if "application/json" in ct:
        body = await request.json()
        target_vol_id = body.get("volume_id")
        target_vol_fn = body.get("volume_filename")
    elif "multipart/form-data" in ct:
        form = await request.form()
        target_vol_id = form.get("volume_id")
        target_vol_fn = form.get("volume_filename")
        upload_file_obj = form.get("file")
        if upload_file_obj and hasattr(upload_file_obj, "filename") and upload_file_obj.filename:
            up_res = await upload_volume(upload_file_obj)
            target_vol_id = up_res["volume_id"]
    else:
        try:
            body = await request.json()
            target_vol_id = body.get("volume_id")
            target_vol_fn = body.get("volume_filename")
        except Exception:
            pass

    vol, row, roi_source, active_id = get_volume_and_metadata(target_vol_fn, target_vol_id)

    model_probs: dict[str, dict[str, float]] = {}
    model_volume_logits: list[torch.Tensor] = []

    for name, item in LOADED_MODELS.items():
        m = item["model"]
        cfg = item["config"]
        img_size = int(cfg.get("image_size", 160))
        slices_per_vol = int(cfg.get("slices_per_volume", 4))

        norm_slices = normalize_resize_roi(vol, row, img_size)
        indices = select_deterministic_slice_indices(len(norm_slices), slices_per_vol)
        sampled = norm_slices[indices]
        batch = torch.from_numpy(sampled.copy()).float().unsqueeze(1).to(DEVICE) / 255.0

        with torch.no_grad():
            slice_logits = m(batch)
            vol_logit = slice_logits.mean(dim=0)
            probs = torch.softmax(vol_logit, dim=0).cpu().tolist()

        model_volume_logits.append(vol_logit)
        model_probs[name] = {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}

    ensemble_logit = torch.stack(model_volume_logits, dim=0).mean(dim=0)
    ensemble_probs = torch.softmax(ensemble_logit, dim=0).cpu().tolist()
    pred_idx = int(np.argmax(ensemble_probs))
    consensus_class = CLASS_NAMES[pred_idx]
    confidence = float(ensemble_probs[pred_idx])
    probabilities = {cls: float(p) for cls, p in zip(CLASS_NAMES, ensemble_probs)}

    # Laterality from metadata (0: left, 1: right)
    knee_lr_raw = str(row.get("kneeLR", "")).strip() if row else ""
    if knee_lr_raw in ("0", "left", "l"):
        laterality = "left"
    elif knee_lr_raw in ("1", "right", "r"):
        laterality = "right"
    else:
        laterality = "unknown"

    ground_truth = None
    if row and "label" in row:
        ground_truth = CLASS_NAMES[int(row["label"])]

    # Build standardized structured findings object (Part 3)
    findings = build_structured_findings(
        volume_id=active_id,
        volume_filename=row.get("volumeFilename") if row else active_id,
        volume_shape=list(vol.shape),
        total_slices=int(vol.shape[0]),
        roi_meta=row,
        roi_source=roi_source,
        consensus_class=consensus_class,
        confidence=confidence,
        probabilities=probabilities,
        per_model=model_probs,
        models_evaluated=list(LOADED_MODELS.keys()),
        laterality=laterality,
        ground_truth=ground_truth,
        _row=row,
        _vol=vol,
    )

    return {
        "volume_id": active_id,
        "volume_filename": row.get("volumeFilename", "unknown") if row else active_id,
        "predicted_label": pred_idx,
        "predicted_class": consensus_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "per_model_probabilities": model_probs,
        "models_evaluated": list(LOADED_MODELS.keys()),
        "roi_source": roi_source,
        "roi_slices": int(row.get("roiDepth", 0)) if row else 0,
        "total_slices": int(vol.shape[0]),
        "laterality": laterality,
        "ground_truth": ground_truth,
        "research_only": True,
        "findings": findings,
    }


@app.get("/gradcam")
def get_gradcam(
    volume_id: str = Query(...),
    model: str | None = Query(None),
):
    """Compute and stream a real Grad-CAM overlay PNG for the most informative slice."""
    if not LOADED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No models loaded to compute Grad-CAM.",
        )

    model_key = model if (model and model in LOADED_MODELS) else next(iter(LOADED_MODELS.keys()))
    m_info = LOADED_MODELS[model_key]
    model_obj = m_info["model"]
    cfg = m_info["config"]
    arch = cfg.get("architecture", "small_cnn")
    img_size = int(cfg.get("image_size", 160))

    vol, row, _, _ = get_volume_and_metadata(None, volume_id)
    norm_slices = normalize_resize_roi(vol, row, img_size)
    if len(norm_slices) == 0:
        raise HTTPException(status_code=400, detail="Volume contains no valid ROI slices.")

    batch = torch.from_numpy(norm_slices.copy()).float().unsqueeze(1).to(DEVICE) / 255.0
    with torch.no_grad():
        all_logits = model_obj(batch)
        mean_logits = all_logits.mean(dim=0)
        target_class = int(mean_logits.argmax().item())
        slice_class_logits = all_logits[:, target_class].cpu().numpy()

    best_slice_idx = int(slice_class_logits.argmax())
    best_image_tensor = batch[best_slice_idx : best_slice_idx + 1]

    target_layer = last_convolution(model_obj, arch)
    heat = gradcam(model_obj, target_layer, best_image_tensor, target_class)

    gray = norm_slices[best_slice_idx].astype(np.float32) / 255.0
    base = np.repeat((gray * 255).astype(np.uint8)[..., None], 3, axis=2).astype(np.float32)
    color = np.zeros_like(base)
    color[..., 0] = heat * 255
    color[..., 1] = np.maximum(0, heat - 0.35) * 100
    overlay = np.clip(base * 0.60 + color * 0.40, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Gradcam-Model": model_key,
            "X-Gradcam-Class": CLASS_NAMES[target_class],
            "X-Gradcam-Slice-Index": str(best_slice_idx),
        },
    )


@app.get("/slice/{volume_id}/{index}")
def get_slice(volume_id: str, index: int):
    """Return a normalized PNG image of a specific volume slice."""
    vol, row, _, _ = get_volume_and_metadata(None, volume_id)
    total_depth = vol.shape[0]
    if not (0 <= index < total_depth):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Slice index {index} out of range (0 to {total_depth - 1}).",
        )
    slice_2d = vol[index].astype(np.float32)
    low, high = np.percentile(slice_2d, (1, 99))
    if high > low:
        norm = np.clip((slice_2d - low) / (high - low), 0, 1)
    else:
        norm = np.zeros_like(slice_2d)
    img_arr = np.rint(norm * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_arr, mode="L").save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/report")
def generate_report(payload: dict[str, Any]):
    """Generate grounded clinical findings report with hallucination verification."""
    # Handle either findings object directly or container {"findings": ...}
    findings_raw = payload.get("findings", payload)
    try:
        findings = validate(findings_raw)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid findings payload: {exc}",
        ) from exc

    req_mode = str(payload.get("mode", REPORT_MODE_DEFAULT)).strip().lower()

    # Template Mode (Default, no network, reliable)
    if req_mode != "llm":
        return {
            "mode": "template",
            "report": render_template(findings),
            "grounding": "passed",
            "findings": findings,
        }

    # LLM Mode via Google Gemini
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {
            "mode": "template_fallback_no_key",
            "report": render_template(findings),
            "grounding": "skipped_no_key",
            "status": "gemini_key_not_configured",
            "note": "GEMINI_API_KEY is not set on the server. Defaulting to strict template generation.",
            "findings": findings,
        }

    try:
        llm_text = call_gemini_report(findings, GEMINI_MODEL, timeout_seconds=LLM_TIMEOUT_SECONDS)
        is_grounded, reason = verify_grounding(llm_text, findings)

        if not is_grounded:
            return {
                "mode": "template_fallback_grounding_failed",
                "grounding": "failed_fallback_template",
                "grounding_reason": reason,
                "report": render_template(findings),
                "unverified_llm_excerpt": llm_text[:200] + "...",
                "findings": findings,
            }

        return {
            "mode": "llm",
            "model": GEMINI_MODEL,
            "grounding": "passed",
            "report": llm_text,
            "findings": findings,
        }
    except Exception as exc:
        # Fallback cleanly on timeout, API error, or rate limit without leaking secret
        return {
            "mode": "template_fallback_error",
            "grounding": "failed_service_error",
            "error": str(exc),
            "report": render_template(findings),
            "findings": findings,
        }
