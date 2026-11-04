"""Launch the RadVision ACL Radiologist PACS Workstation."""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:8000")


def main():
    print("=" * 70)
    print("  RadVision ACL — Radiologist Workstation & Diagnostic AI Engine")
    print("  Academic Capstone for Neural Networks and Deep Learning")
    print("=" * 70)
    print("\nStarting local server at http://127.0.0.1:8000 ...")
    print("• Landing Page & Guest Mode enabled")
    print("• Interactive 3D PACS Viewport & Grad-CAM Heatmap ready")
    print("• Multi-Model Consensus & Longitudinal Recovery Tracker ready")
    print("• Press Ctrl+C to terminate the server.\n")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    main()
