"""End-to-end orchestration: (download →) detect → shuttle → calibrate → viz."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rallylens.common import get_logger, read_video_properties
from rallylens.ingest.downloader import download_video
from rallylens.pipeline.court import run_court_detection
from rallylens.pipeline.io import save_player_detections, viz_court_path, viz_overlay_path
from rallylens.pipeline.shuttle import run_shuttle_pipeline
from rallylens.vision.detect_track import TrackerName, detect_and_track_players
from rallylens.viz import render_overlay_video, render_viz_court

_log = get_logger(__name__)


@dataclass
class PipelineResult:
    video_id: str
    detections_path: Path
    overlay_path: Path | None = None
    court_path: Path | None = None


def run_full_pipeline(
    url_or_path: str,
    *,
    tracker: TrackerName = "bytetrack",
    singles: bool = True,
    imgsz: int = 1280,
) -> PipelineResult:
    """Run the full pipeline for a URL or local video path.

    Stages: ingest → player detect → shuttle detect → court calibrate → viz.
    If `url_or_path` points to an existing local file, it is used directly.
    Otherwise it is treated as a YouTube URL and downloaded first.
    """
    video_path = Path(url_or_path)
    if not video_path.exists():
        meta = download_video(url_or_path)
        video_path = meta.source_path
    video_id = video_path.stem

    # 1. Player detection
    player_detections = detect_and_track_players(
        video_path, tracker=tracker, singles=singles, imgsz=imgsz
    )
    det_path = save_player_detections(player_detections, video_id)

    # 2. Shuttle detection
    shuttle_track = run_shuttle_pipeline(video_path, video_id)

    # 3. Court calibration
    corners = run_court_detection(video_path, video_id)
    if corners is None:
        _log.warning("court detection failed — skipping visualization")
        return PipelineResult(video_id=video_id, detections_path=det_path)

    # 4. Visualization
    overlay_out = render_overlay_video(
        video_path,
        player_detections,
        shuttle_track,
        viz_overlay_path(video_id),
        corners=corners,
    )

    props = read_video_properties(video_path)
    court_out = render_viz_court(
        player_detections,
        shuttle_track,
        corners,
        viz_court_path(video_id),
        fps=props.fps,
    )

    return PipelineResult(
        video_id=video_id,
        detections_path=det_path,
        overlay_path=overlay_out,
        court_path=court_out,
    )
