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
    tracker.step(0, [ShuttleObservation(0, 100.0, 100.0, 0.9)])
    tracker.step(1, [ShuttleObservation(1, 105.0, 102.0, 0.9)])
    # outlier very far from predicted position
    point = tracker.step(2, [ShuttleObservation(2, 500.0, 500.0, 0.9)])
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
