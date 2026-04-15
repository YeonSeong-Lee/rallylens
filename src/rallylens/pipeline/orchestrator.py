"""End-to-end orchestration: (download →) detect players → save detections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rallylens.common import get_logger
from rallylens.ingest.downloader import download_video
from rallylens.pipeline.io import save_player_detections
from rallylens.vision.detect_track import TrackerName, detect_and_track_players

_log = get_logger(__name__)


@dataclass
class PipelineResult:
    video_id: str
    detections_path: Path


def run_full_pipeline(
    url_or_path: str,
    *,
    tracker: TrackerName = "bytetrack",
    singles: bool = True,
    imgsz: int = 1280,
) -> PipelineResult:
    """Run the player tracking pipeline for a URL or local video path.

    If `url_or_path` points to an existing local file, it is used directly.
    Otherwise it is treated as a YouTube URL and downloaded first.
    """
    video_path = Path(url_or_path)
    if video_path.exists():
        video_id = video_path.stem
    else:
        meta = download_video(url_or_path)
        video_path = meta.source_path
        video_id = meta.video_id

    player_detections = detect_and_track_players(
        video_path, tracker=tracker, singles=singles, imgsz=imgsz
    )
    out_path = save_player_detections(player_detections, video_id, video_path.stem)
    return PipelineResult(video_id=video_id, detections_path=out_path)
