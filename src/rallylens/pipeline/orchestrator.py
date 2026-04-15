"""End-to-end orchestration: (download →) detect players → save detections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rallylens.common import ensure_dir, get_logger
from rallylens.config import EVENTS_DIR
from rallylens.ingest.downloader import download_video
from rallylens.pipeline.io import save_player_detections, save_video_meta
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
        save_video_meta(meta)
        ensure_dir(EVENTS_DIR / meta.video_id)
        video_path = meta.source_path
        video_id = meta.video_id

    player_detections = detect_and_track_players(video_path, tracker=tracker)
    out_path = save_player_detections(player_detections, video_id, video_path.stem)
    return PipelineResult(video_id=video_id, detections_path=out_path)
