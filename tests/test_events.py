import math
from pathlib import Path

from rallylens.analysis.events import (
    aggregate_rally_stats,
    detect_hit_events,
)
from rallylens.pipeline.io import load_events_jsonl, save_events_jsonl
from rallylens.vision.kalman_tracker import ShuttleTrackPoint


def _track(points: list[tuple[int, float, float, float, float, bool, float]]) -> list[ShuttleTrackPoint]:
    return [
        ShuttleTrackPoint(
            frame_idx=fi,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            interpolated=interp,
            residual=res,
        )
        for fi, x, y, vx, vy, interp, res in points
    ]


def test_detect_hit_on_velocity_reversal():
    # 10 frames moving right, then 10 frames moving left. Direction reverses at frame 10.
    pts = []
    for i in range(10):
        pts.append((i, 100.0 + i * 5, 50.0, 5.0, 0.0, False, 1.0))
    for i in range(10, 20):
        pts.append((i, 150.0 - (i - 10) * 5, 50.0, -5.0, 0.0, False, 1.0))
    track = _track(pts)
    events = detect_hit_events(track, fps=30.0, frame_height=480, z_threshold=10.0)
    assert len(events) >= 1
    assert any("velocity_reversal" in e.signals for e in events)


def test_detect_no_events_on_constant_velocity():
    pts = [(i, float(i * 5), 50.0, 5.0, 0.0, False, 1.0) for i in range(20)]
    events = detect_hit_events(_track(pts), fps=30.0, z_threshold=10.0)
    assert events == []


def test_refractory_collapses_adjacent_triggers():
    # Three rapid reversals in frames 5, 6, 7 → only the first should survive min_gap=5
    pts = []
    for i in range(10):
        sign = 1 if (i // 2) % 2 == 0 else -1
        pts.append((i, 100.0, 50.0, 5.0 * sign, 0.0, False, 1.0))
    events = detect_hit_events(_track(pts), fps=30.0, min_gap_frames=5, z_threshold=10.0)
    assert len(events) <= 2  # at most 2 triggers across 10 frames with refractory=5


def test_aggregate_stats_computes_gaps():
    pts = [(i, 100.0 + i * 5, 50.0, 5.0, 0.0, False, 1.0) for i in range(10)]
    pts += [(i, 150.0 - (i - 10) * 5, 50.0, -5.0, 0.0, False, 1.0) for i in range(10, 20)]
    events = detect_hit_events(_track(pts), fps=30.0, frame_height=480, z_threshold=10.0)
    stats = aggregate_rally_stats(
        video_id="test", rally_index=1, events=events, duration_s=0.67, total_frames=20
    )
    assert stats.shot_count == len(events)
    if len(events) >= 2:
        assert stats.avg_inter_shot_gap_s is not None
        assert stats.avg_inter_shot_gap_s > 0


def test_events_jsonl_round_trip(tmp_path: Path):
    pts = [(i, 100.0 + i * 5, 50.0, 5.0, 0.0, False, 1.0) for i in range(10)]
    pts += [(i, 150.0 - (i - 10) * 5, 50.0, -5.0, 0.0, False, 1.0) for i in range(10, 20)]
    events = detect_hit_events(_track(pts), fps=30.0, frame_height=480, z_threshold=10.0)

    path = tmp_path / "events.jsonl"
    save_events_jsonl(events, path)
    loaded = load_events_jsonl(path)
    assert len(loaded) == len(events)
    for a, b in zip(events, loaded, strict=True):
        assert a.frame_idx == b.frame_idx
        assert math.isclose(a.position_xy[0], b.position_xy[0])
        assert a.signals == b.signals


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_detect_hit_events_empty_track():
    assert detect_hit_events([], fps=30.0, frame_height=480) == []


def test_detect_hit_events_too_short_track():
    """Tracks shorter than 3 points cannot form a reversal trigger."""
    pts = [(0, 100.0, 50.0, 5.0, 0.0, False, 1.0), (1, 105.0, 50.0, 5.0, 0.0, False, 1.0)]
    assert detect_hit_events(_track(pts), fps=30.0, frame_height=480) == []


def test_aggregate_stats_empty_events():
    stats = aggregate_rally_stats(
        video_id="v",
        rally_index=1,
        events=[],
        duration_s=3.0,
        total_frames=90,
    )
    assert stats.shot_count == 0
    assert stats.first_shot_frame is None
    assert stats.last_shot_frame is None
    assert stats.avg_inter_shot_gap_s is None
    assert stats.top_side_shots == 0
    assert stats.bottom_side_shots == 0


def test_aggregate_stats_single_event():
    """A single event has no gaps to average, but first==last==its frame."""
    from rallylens.analysis.events import HitEvent

    event = HitEvent(
        frame_idx=42,
        time_s=1.4,
        kind="hit",
        position_xy=(100.0, 100.0),
        velocity_xy=(5.0, 0.0),
        signals=("velocity_reversal",),
        player_side="top",
    )
    stats = aggregate_rally_stats(
        video_id="v",
        rally_index=1,
        events=[event],
        duration_s=3.0,
        total_frames=90,
    )
    assert stats.shot_count == 1
    assert stats.first_shot_frame == 42
    assert stats.last_shot_frame == 42
    assert stats.avg_inter_shot_gap_s is None  # no gaps to average
    assert stats.top_side_shots == 1


def test_detect_no_false_positives_on_slow_motion():
    """Very low speed points (below min_speed) should not trigger reversals."""
    pts = [(i, 100.0 + i * 0.1, 50.0, 0.1, 0.0, False, 1.0) for i in range(20)]
    events = detect_hit_events(_track(pts), fps=30.0, frame_height=480, z_threshold=10.0)
    assert events == []
