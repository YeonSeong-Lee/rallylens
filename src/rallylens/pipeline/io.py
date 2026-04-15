"""Disk I/O for pipeline artifacts: video meta, shuttle tracks, rally stats."""

from __future__ import annotations

from pathlib import Path

from rallylens.analysis.events import HitEvent, RallyStats
from rallylens.config import CALIBRATION_DIR, DETECTIONS_DIR, EVENTS_DIR, TRACKS_DIR
from rallylens.domain.video import VideoMeta
from rallylens.serialization import load_json, load_jsonl, save_json, save_jsonl
from rallylens.vision.court_homography import CourtHomography
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


# ---------------------------------------------------------------------------
# Shuttle track persistence (was in vision/kalman_tracker.py)
# ---------------------------------------------------------------------------


def save_track_jsonl(track: list[ShuttleTrackPoint], path: Path) -> None:
    save_jsonl(track, path)


# ---------------------------------------------------------------------------
# Event / stats persistence (was in analysis/events.py)
# ---------------------------------------------------------------------------


def save_events_jsonl(events: list[HitEvent], path: Path) -> None:
    save_jsonl(events, path)


def load_events_jsonl(path: Path) -> list[HitEvent]:
    return load_jsonl(path, HitEvent)


def save_rally_stats(stats: RallyStats, path: Path) -> None:
    save_json(stats, path)


# ---------------------------------------------------------------------------
# Court homography persistence (was in vision/court_homography.py)
# ---------------------------------------------------------------------------


def homography_path(video_id: str) -> Path:
    return CALIBRATION_DIR / video_id / "homography.json"


def save_homography(h: CourtHomography, path: Path) -> None:
    save_json(h, path)


def load_homography(path: Path) -> CourtHomography:
    return load_json(path, CourtHomography)
