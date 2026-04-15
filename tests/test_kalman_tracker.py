import math

import pytest

from rallylens.vision.kalman_tracker import (
    ShuttleKalmanTracker,
    ShuttleObservation,
    observations_from_detections,
    track_shuttle,
)


def _linear_shuttle_traj(total: int = 30, vx: float = 5.0, vy: float = 3.0) -> dict:
    obs = {}
    for i in range(total):
        x = 100.0 + vx * i
        y = 50.0 + vy * i
        obs[i] = [ShuttleObservation(frame_idx=i, x=x, y=y, confidence=0.9)]
    return obs


def test_tracker_locks_onto_linear_trajectory():
    obs = _linear_shuttle_traj(total=20, vx=5.0, vy=3.0)
    track = track_shuttle(obs, total_frames=20)
    assert len(track) == 20
    # after a few frames the velocity estimate should converge
    final = track[-1]
    assert final.vx == pytest.approx(5.0, abs=0.3)
    assert final.vy == pytest.approx(3.0, abs=0.3)
    assert not any(p.interpolated for p in track)


def test_tracker_interpolates_missed_frames():
    obs = _linear_shuttle_traj(total=15, vx=4.0, vy=2.0)
    # drop frames 5, 6, 7
    for i in (5, 6, 7):
        obs[i] = []
    track = track_shuttle(obs, total_frames=15)
    missed = [p for p in track if p.interpolated]
    assert len(missed) == 3
    # interpolated positions should still be close to linear expectation
    for p in missed:
        expected_x = 100.0 + 4.0 * p.frame_idx
        expected_y = 50.0 + 2.0 * p.frame_idx
        assert math.isclose(p.x, expected_x, abs_tol=20.0)
        assert math.isclose(p.y, expected_y, abs_tol=20.0)


def test_tracker_rejects_far_outlier():
    tracker = ShuttleKalmanTracker(association_max_distance=30.0)
    tracker.step(0, [ShuttleObservation(frame_idx=0, x=100.0, y=100.0, confidence=0.9)])
    tracker.step(1, [ShuttleObservation(frame_idx=1, x=105.0, y=102.0, confidence=0.9)])
    # outlier very far from predicted position
    point = tracker.step(2, [ShuttleObservation(frame_idx=2, x=500.0, y=500.0, confidence=0.9)])
    assert point is not None
    assert point.interpolated


def test_observations_from_detections_groups_per_frame():
    dets = [
        (0, (0.0, 0.0, 10.0, 10.0), 0.8),
        (0, (20.0, 20.0, 30.0, 30.0), 0.6),
        (5, (100.0, 100.0, 110.0, 110.0), 0.9),
    ]
    obs = observations_from_detections(dets)
    assert len(obs[0]) == 2
    assert len(obs[5]) == 1
    assert obs[0][0].x == pytest.approx(5.0)
    assert obs[0][0].y == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_tracker_resets_after_sustained_occlusion():
    """When missed frames exceed max_missed_frames, the tracker drops the track."""
    obs = _linear_shuttle_traj(total=30, vx=5.0, vy=3.0)
    for i in range(5, 25):  # 20 missed frames > max_missed_frames=10
        obs[i] = []
    track = track_shuttle(obs, total_frames=30)
    # only the early pre-gap points survive + the re-initialized post-gap points
    frame_indices = {p.frame_idx for p in track}
    assert 0 in frame_indices
    assert 4 in frame_indices  # last pre-gap frame
    # frames deep in the gap (>10 missed) should not exist
    assert 20 not in frame_indices


def test_tracker_handles_empty_observations():
    """Zero detections across all frames yields an empty track."""
    obs: dict[int, list[ShuttleObservation]] = {i: [] for i in range(10)}
    track = track_shuttle(obs, total_frames=10)
    assert track == []


def test_tracker_prefers_highest_confidence_on_init():
    obs = {
        0: [
            ShuttleObservation(frame_idx=0, x=100.0, y=100.0, confidence=0.3),
            ShuttleObservation(frame_idx=0, x=200.0, y=200.0, confidence=0.9),
            ShuttleObservation(frame_idx=0, x=300.0, y=300.0, confidence=0.5),
        ]
    }
    track = track_shuttle(obs, total_frames=1)
    assert len(track) == 1
    assert track[0].x == pytest.approx(200.0)
    assert track[0].y == pytest.approx(200.0)


def test_tracker_sudden_direction_change_continues_tracking():
    """Simulating a racket hit: shuttle abruptly reverses direction."""
    obs: dict[int, list[ShuttleObservation]] = {}
    # frames 0-9: moving right
    for i in range(10):
        obs[i] = [ShuttleObservation(frame_idx=i, x=100.0 + i * 5, y=50.0, confidence=0.9)]
    # frames 10-19: moving left
    for i in range(10, 20):
        obs[i] = [
            ShuttleObservation(frame_idx=i, x=150.0 - (i - 10) * 5, y=50.0, confidence=0.9)
        ]
    track = track_shuttle(obs, total_frames=20)
    assert len(track) == 20  # tracker never drops the shuttle
    # velocity should flip sign somewhere around frame 10-12
    vx_signs = [1 if p.vx > 0 else -1 for p in track if abs(p.vx) > 0.5]
    assert 1 in vx_signs and -1 in vx_signs
