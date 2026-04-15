"""Disk I/O for pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

from rallylens.config import DETECTIONS_DIR
from rallylens.serialization import load_jsonl, save_jsonl
from rallylens.vision.detect_track import Detection


def detections_path(video_id: str, video_stem: str) -> Path:
    return DETECTIONS_DIR / video_id / f"{video_stem}_players.jsonl"


def save_player_detections(
    detections: list[Detection], video_id: str, video_stem: str
) -> Path:
    path = detections_path(video_id, video_stem)
    save_jsonl(detections, path)
    return path


def load_player_detections(video_id: str, video_stem: str) -> list[Detection]:
    return load_jsonl(detections_path(video_id, video_stem), Detection)
