"""Hit event detection + rally stats aggregation pipeline stage."""

from __future__ import annotations

from rallylens.analysis.events import (
    RallyStats,
    aggregate_rally_stats,
    detect_hit_events,
)
from rallylens.common import ensure_dir, get_logger, read_video_properties
from rallylens.config import EVENTS_DIR
from rallylens.pipeline.io import (
    load_shuttle_track,
    save_events_jsonl,
    save_rally_stats,
    shuttle_track_path,
)
from rallylens.preprocess.rally_segmenter import RallyClip
from rallylens.vision.kalman_tracker import ShuttleTrackPoint

_log = get_logger(__name__)


def run_events_pipeline(
    video_id: str,
    rallies: list[RallyClip],
    shuttle_tracks_in_memory: dict[int, list[ShuttleTrackPoint]] | None = None,
) -> list[RallyStats]:
    """Detect hits + write rally_*_events.jsonl and rally_*_stats.json per rally.

    If `shuttle_tracks_in_memory` is provided (e.g. from a fresh `run` invocation),
    tracks are read from memory. Otherwise they're loaded from disk.
    """
    events_out_dir = ensure_dir(EVENTS_DIR / video_id)
    stats_list: list[RallyStats] = []

    for rally in rallies:
        if shuttle_tracks_in_memory is not None:
            track = shuttle_tracks_in_memory.get(rally.index, [])
        else:
            track = load_shuttle_track(shuttle_track_path(video_id, rally.path.stem))

        props = read_video_properties(rally.path)
        events = detect_hit_events(track, fps=props.fps, frame_height=props.height)
        stats = aggregate_rally_stats(
            video_id=video_id,
            rally_index=rally.index,
            events=events,
            duration_s=rally.end_s - rally.start_s,
            total_frames=props.frame_count,
        )
        save_events_jsonl(
            events, events_out_dir / f"rally_{rally.index:03d}_events.jsonl"
        )
        save_rally_stats(
            stats, events_out_dir / f"rally_{rally.index:03d}_stats.json"
        )
        stats_list.append(stats)
    return stats_list
