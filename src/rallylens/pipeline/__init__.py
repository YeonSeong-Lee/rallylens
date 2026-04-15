"""Pipeline orchestration for RallyLens.

Each stage of the CLI pipeline (download, segment, shuttle track, events,
heatmaps, report) lives in its own module here. `cli.py` stays focused on
click command definitions; the real work lives in these modules so it can be
imported and tested without touching Click or the filesystem layout.
"""

from rallylens.pipeline.events import run_events_pipeline
from rallylens.pipeline.heatmaps import render_match_heatmaps
from rallylens.pipeline.io import (
    load_all_stats,
    load_shuttle_track,
    load_video_meta,
    save_video_meta,
)
from rallylens.pipeline.orchestrator import run_full_pipeline

__all__ = [
    "load_all_stats",
    "load_shuttle_track",
    "load_video_meta",
    "render_match_heatmaps",
    "run_events_pipeline",
    "run_full_pipeline",
    "save_video_meta",
]
