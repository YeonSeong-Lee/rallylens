"""Disk I/O for pipeline artifacts: video meta, shuttle tracks, rally stats."""

from __future__ import annotations

from pathlib import Path

from rallylens.analysis.events import RallyStats
from rallylens.common import DETECTIONS_DIR, EVENTS_DIR, TRACKS_DIR, VideoMeta
from rallylens.serialization import load_json, load_jsonl, save_json, save_jsonl
from rallylens.vision.detect_track import Detection
from rallylens.vision.kalman_tracker import ShuttleTrackPoint


def video_meta_path(video_id: str) -> Path:
    return EVENTS_DIR / video_id / "video_meta.json"


def save_video_meta(meta: VideoMeta) -> Path:
    path = video_meta_path(meta.video_id)
    save_json(meta, path)
    return path


def load_video_meta(video_id: str) -> VideoMeta:
    return load_json(video_meta_path(video_id), VideoMeta)


def shuttle_track_path(video_id: str, rally_stem: str) -> Path:
    return TRACKS_DIR / video_id / f"{rally_stem}_shuttle.jsonl"


def load_shuttle_track(path: Path) -> list[ShuttleTrackPoint]:
    return load_jsonl(path, ShuttleTrackPoint)


def detections_path(video_id: str, rally_stem: str) -> Path:
    return DETECTIONS_DIR / video_id / f"{rally_stem}_players.jsonl"


def save_player_detections(
    detections: list[Detection], video_id: str, rally_stem: str
) -> Path:
    path = detections_path(video_id, rally_stem)
    save_jsonl(detections, path)
    return path


def load_player_detections(video_id: str, rally_stem: str) -> list[Detection]:
    return load_jsonl(detections_path(video_id, rally_stem), Detection)


def load_all_stats(video_id: str) -> list[RallyStats]:
    events_dir = EVENTS_DIR / video_id
    if not events_dir.exists():
        return []
    return [
        load_json(stats_path, RallyStats)
        for stats_path in sorted(events_dir.glob("rally_*_stats.json"))
    ]
