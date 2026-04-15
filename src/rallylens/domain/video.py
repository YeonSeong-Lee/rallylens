"""Domain models and logic for match video identity.

Pure Python / Pydantic only — no I/O, no infrastructure imports.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_YOUTUBE_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[?&]v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{11})"),
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


def video_id_from_url(url: str) -> str:
    """Extract the 11-character YouTube video ID from any supported URL form.

    Falls back to an SHA-1 hash prefix when the URL does not match any
    recognised YouTube pattern (e.g. direct-file URLs).
    """
    for pattern in _YOUTUBE_ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:11]


def is_likely_youtube_url(url: str) -> bool:
    """Return True if *url* looks like a YouTube URL we know how to handle."""
    return any(pattern.search(url) for pattern in _YOUTUBE_ID_PATTERNS)
