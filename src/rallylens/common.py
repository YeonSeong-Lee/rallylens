"""Shared primitives used across ingest / preprocess / vision / viz."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import numpy as np

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

_YOUTUBE_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[?&]v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{11})"),
)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    level_name = os.environ.get("RALLYLENS_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    return logger


def load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def video_id_from_url(url: str) -> str:
    for pattern in _YOUTUBE_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:11]


def is_likely_youtube_url(url: str) -> bool:
    """Return True if `url` looks like a YouTube URL we know how to handle."""
    return any(pattern.search(url) for pattern in _YOUTUBE_ID_PATTERNS)


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it first: `brew install ffmpeg` (macOS)."
        )


class VideoMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    title: str
    upload_date: str | None
    duration_s: float
    url: str
    source_path: Path


class VideoProperties(BaseModel):
    model_config = ConfigDict(frozen=True)

    fps: float
    width: int
    height: int
    frame_count: int


def read_video_properties(path: Path) -> VideoProperties:
    """Open a video, read its metadata, and release the capture cleanly."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        return VideoProperties(
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_frame_at(path: Path, frame_idx: int) -> np.ndarray:
    """Seek to a specific frame and return it as a numpy array.

    Raises:
        FileNotFoundError: the video file is missing.
        RuntimeError: the frame could not be decoded (corrupt file or OOB index).
    """
    import cv2

    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {frame_idx} from {path}")
    return frame
