"""Filesystem layout: the only place in the codebase that knows where
pipeline artifacts live on disk.

Pure constants — no logic, no I/O, no imports from rallylens.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
RALLIES_DIR: Path = DATA_DIR / "rallies"
OVERLAYS_DIR: Path = DATA_DIR / "overlays"
CALIBRATION_DIR: Path = DATA_DIR / "calibration"
LABEL_FRAMES_DIR: Path = DATA_DIR / "label_frames"
DETECTIONS_DIR: Path = DATA_DIR / "detections"
TRACKS_DIR: Path = DATA_DIR / "tracks"
EVENTS_DIR: Path = DATA_DIR / "events"
REPORTS_DIR: Path = DATA_DIR / "reports"
HEATMAPS_DIR: Path = DATA_DIR / "heatmaps"
LABELQA_DIR: Path = DATA_DIR / "label_qa"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DEMO_DIR: Path = PROJECT_ROOT / "outputs" / "demo"

TARGET_HEIGHT: int = 720
TARGET_FPS: int = 30
