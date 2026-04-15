"""Shared pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest

from rallylens.analysis.events import HitEvent, RallyStats
from rallylens.vision.kalman_tracker import ShuttleObservation, ShuttleTrackPoint

os.environ.setdefault("RALLYLENS_LOG_LEVEL", "CRITICAL")


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_track_point(
    frame_idx: int,
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
    interpolated: bool = False,
    residual: float = 1.0,
) -> ShuttleTrackPoint:
    return ShuttleTrackPoint(
        frame_idx=frame_idx,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        interpolated=interpolated,
        residual=residual,
    )


def make_observation(
    frame_idx: int, x: float, y: float, confidence: float = 0.9
) -> ShuttleObservation:
    return ShuttleObservation(
        frame_idx=frame_idx, x=x, y=y, confidence=confidence
    )


def make_hit_event(
    frame_idx: int = 0,
    time_s: float = 0.0,
    position_xy: tuple[float, float] = (100.0, 100.0),
    velocity_xy: tuple[float, float] = (5.0, 0.0),
    signals: tuple[str, ...] = ("velocity_reversal",),
    player_side: str | None = "top",
) -> HitEvent:
    return HitEvent(
        frame_idx=frame_idx,
        time_s=time_s,
        kind="hit",
        position_xy=position_xy,
        velocity_xy=velocity_xy,
        signals=signals,
        player_side=player_side,
    )


def make_rally_stats(
    video_id: str = "v",
    rally_index: int = 1,
    events: list[HitEvent] | None = None,
) -> RallyStats:
    events = events or []
    return RallyStats(
        video_id=video_id,
        rally_index=rally_index,
        duration_s=5.0,
        total_frames=150,
        shot_count=len(events),
        first_shot_frame=events[0].frame_idx if events else None,
        last_shot_frame=events[-1].frame_idx if events else None,
        avg_inter_shot_gap_s=None,
        top_side_shots=sum(1 for e in events if e.player_side == "top"),
        bottom_side_shots=sum(1 for e in events if e.player_side == "bottom"),
        events=events,
    )


@pytest.fixture
def track_point_factory():
    return make_track_point


@pytest.fixture
def hit_event_factory():
    return make_hit_event


@pytest.fixture
def rally_stats_factory():
    return make_rally_stats


_DIR_NAMES = (
    "DATA_DIR",
    "RAW_DIR",
    "RALLIES_DIR",
    "OVERLAYS_DIR",
    "CALIBRATION_DIR",
    "LABEL_FRAMES_DIR",
    "DETECTIONS_DIR",
    "TRACKS_DIR",
    "EVENTS_DIR",
    "REPORTS_DIR",
    "HEATMAPS_DIR",
    "LABELQA_DIR",
)


@pytest.fixture
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every module's DATA_DIR-derived constants into tmp_path.

    Needed for CLI + pipeline tests because the path constants are imported
    at module load time into every consumer (cli, pipeline.*, downloader,
    rally_segmenter), so patching `rallylens.common` alone is not enough.
    """
    import rallylens.cli
    import rallylens.common
    import rallylens.ingest.downloader
    import rallylens.pipeline.events as pipeline_events
    import rallylens.pipeline.heatmaps as pipeline_heatmaps
    import rallylens.pipeline.io as pipeline_io
    import rallylens.pipeline.orchestrator as pipeline_orchestrator
    import rallylens.pipeline.shuttle as pipeline_shuttle
    import rallylens.preprocess.rally_segmenter as rally_segmenter

    data = tmp_path / "data"
    data.mkdir()
    paths = {
        "DATA_DIR": data,
        "RAW_DIR": data / "raw",
        "RALLIES_DIR": data / "rallies",
        "OVERLAYS_DIR": data / "overlays",
        "CALIBRATION_DIR": data / "calibration",
        "LABEL_FRAMES_DIR": data / "label_frames",
        "DETECTIONS_DIR": data / "detections",
        "TRACKS_DIR": data / "tracks",
        "EVENTS_DIR": data / "events",
        "REPORTS_DIR": data / "reports",
        "HEATMAPS_DIR": data / "heatmaps",
        "LABELQA_DIR": data / "label_qa",
    }

    targets = (
        rallylens.common,
        rallylens.cli,
        rallylens.ingest.downloader,
        pipeline_io,
        pipeline_shuttle,
        pipeline_events,
        pipeline_heatmaps,
        pipeline_orchestrator,
        rally_segmenter,
    )
    for name in _DIR_NAMES:
        new = paths[name]
        for mod in targets:
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, new)

    return data
