"""Disk I/O for pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

from rallylens.analysis.metrics import MatchMetrics
from rallylens.config import (
    CALIBRATION_DIR,
    DETECTIONS_DIR,
    REPORTS_DIR,
    TRACKS_DIR,
    VIZ_DIR,
)
from rallylens.llm.report_schema import ReportOutput
from rallylens.serialization import load_json, load_jsonl, save_json, save_jsonl
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import ShuttlePoint

# ---------------------------------------------------------------------------
# Player detections
# ---------------------------------------------------------------------------


def detections_path(video_id: str) -> Path:
    return DETECTIONS_DIR / video_id / f"{video_id}_players.jsonl"


def save_player_detections(detections: list[Detection], video_id: str) -> Path:
    path = detections_path(video_id)
    save_jsonl(detections, path)
    return path


def load_player_detections(video_id: str) -> list[Detection]:
    return load_jsonl(detections_path(video_id), Detection)


# ---------------------------------------------------------------------------
# Shuttle tracks
# ---------------------------------------------------------------------------


def shuttle_track_path(video_id: str) -> Path:
    return TRACKS_DIR / video_id / f"{video_id}_shuttle.jsonl"


def save_shuttle_track(track: list[ShuttlePoint], video_id: str) -> Path:
    path = shuttle_track_path(video_id)
    save_jsonl(track, path)
    return path


def load_shuttle_track(video_id: str) -> list[ShuttlePoint]:
    return load_jsonl(shuttle_track_path(video_id), ShuttlePoint)


# ---------------------------------------------------------------------------
# Court calibration
# ---------------------------------------------------------------------------


def court_corners_path(video_id: str) -> Path:
    return CALIBRATION_DIR / video_id / "corners.json"


def save_court_corners(corners: CourtCorners, video_id: str) -> Path:
    path = court_corners_path(video_id)
    save_json(corners, path)
    return path


def load_court_corners(video_id: str) -> CourtCorners:
    return load_json(court_corners_path(video_id), CourtCorners)


# ---------------------------------------------------------------------------
# Visualization outputs
# ---------------------------------------------------------------------------


def viz_overlay_path(video_id: str) -> Path:
    return VIZ_DIR / video_id / f"{video_id}_overlay.mp4"


def viz_heatmap_path(video_id: str) -> Path:
    return VIZ_DIR / video_id / "heatmap.png"


def viz_court_diagram_path(video_id: str) -> Path:
    return VIZ_DIR / video_id / "court_diagram.gif"


# ---------------------------------------------------------------------------
# Reports (deterministic metrics + LLM-generated analysis)
# ---------------------------------------------------------------------------


def metrics_path(video_id: str) -> Path:
    return REPORTS_DIR / video_id / "metrics.json"


def save_match_metrics(metrics: MatchMetrics, video_id: str) -> Path:
    path = metrics_path(video_id)
    save_json(metrics, path)
    return path


def load_match_metrics(video_id: str) -> MatchMetrics:
    return load_json(metrics_path(video_id), MatchMetrics)


def report_json_path(video_id: str) -> Path:
    return REPORTS_DIR / video_id / "report.json"


def report_markdown_path(video_id: str) -> Path:
    return REPORTS_DIR / video_id / "report.md"


def save_report(report: ReportOutput, video_id: str) -> Path:
    path = report_json_path(video_id)
    save_json(report, path)
    return path


def load_report(video_id: str) -> ReportOutput:
    return load_json(report_json_path(video_id), ReportOutput)
