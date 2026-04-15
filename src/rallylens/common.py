"""Shared primitives used across ingest / preprocess / vision / viz."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
RALLIES_DIR: Path = DATA_DIR / "rallies"
OVERLAYS_DIR: Path = DATA_DIR / "overlays"
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


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it first: `brew install ffmpeg` (macOS)."
        )


@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    title: str
    upload_date: str | None
    duration_s: float
    url: str
    source_path: Path

    def to_json_dict(self) -> dict[str, object]:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "upload_date": self.upload_date,
            "duration_s": self.duration_s,
            "url": self.url,
            "source_path": str(self.source_path),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, object]) -> VideoMeta:
        return cls(
            video_id=str(data["video_id"]),
            title=str(data["title"]),
            upload_date=(
                str(data["upload_date"]) if data.get("upload_date") is not None else None
            ),
            duration_s=float(data["duration_s"]),  # type: ignore[arg-type]
            url=str(data["url"]),
            source_path=Path(str(data["source_path"])),
        )
