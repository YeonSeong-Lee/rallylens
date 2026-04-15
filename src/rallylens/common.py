"""Cross-cutting utilities: logging, environment, filesystem helpers, and video I/O.

Intentionally small — each concern lives in its own module:
  - Path constants  →  rallylens.config
  - Domain models   →  rallylens.domain.video
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from rallylens.config import PROJECT_ROOT
from rallylens.domain.video import VideoProperties

if TYPE_CHECKING:
    import numpy as np


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


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it first: `brew install ffmpeg` (macOS)."
        )


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
