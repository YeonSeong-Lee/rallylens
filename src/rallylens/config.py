"""Filesystem layout: the only place in the codebase that knows where
pipeline artifacts live on disk.

Pure constants — no logic, no I/O, no imports from rallylens.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
DETECTIONS_DIR: Path = DATA_DIR / "detections"
TRACKS_DIR: Path = DATA_DIR / "tracks"
CALIBRATION_DIR: Path = DATA_DIR / "calibration"
MODELS_DIR: Path = PROJECT_ROOT / "models"
VIZ_DIR: Path = DATA_DIR / "viz"

TARGET_HEIGHT: int = 720
TARGET_FPS: int = 30
