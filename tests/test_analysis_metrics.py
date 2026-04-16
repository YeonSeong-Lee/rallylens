"""Unit tests for rallylens.analysis.metrics.

Uses synthetic pydantic artifacts (no real video, no cv2 inference) to
exercise the deterministic metric computation end to end. Court corners
are placed so that the homography becomes the identity mapping, which
lets tests assert known pixel-level positions directly in court-diagram
space.
"""

from __future__ import annotations

import pytest

from rallylens.analysis.metrics import (
    MatchMetrics,
    PlayerMetrics,
    ShuttleMetrics,
    compute_match_metrics,
)
from rallylens.domain.video import VideoProperties
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import ShuttlePoint

# Identity homography: src corners = court-diagram image corners, so
# project_point(H, x, y) ≈ (x, y). All downstream keypoint coordinates
# are therefore in court-diagram space directly.
_IDENTITY_CORNERS = CourtCorners(
    top_left=(60.0, 60.0),
    top_right=(670.0, 60.0),
    bottom_left=(60.0, 1400.0),
    bottom_right=(670.0, 1400.0),
)

_VIDEO_PROPS = VideoProperties(fps=30.0, width=1000, height=800, frame_count=300)
_VIDEO_ID = "test_clip"


def _make_detection(
    frame_idx: int,
    *,
    track_id: int,
    ankle_xy: tuple[float, float],
    nose_xy: tuple[float, float] | None = None,
    l_wrist_xy: tuple[float, float] | None = None,
) -> Detection:
    """Build a synthetic Detection with the keypoints the analysis layer reads.

    All other COCO-17 keypoints are set to (0, 0) with confidence 0 so they
    are ignored. The ankles are duplicated into both L and R slots so the
    foot midpoint equals `ankle_xy`.
    """
    keypoints_xy: list[tuple[float, float]] = [(0.0, 0.0)] * 17
    keypoints_conf: list[float] = [0.0] * 17
    keypoints_xy[15] = ankle_xy  # L ankle
    keypoints_xy[16] = ankle_xy  # R ankle
    keypoints_conf[15] = 0.9
    keypoints_conf[16] = 0.9
    if nose_xy is not None:
        keypoints_xy[0] = nose_xy
        keypoints_conf[0] = 0.9
    if l_wrist_xy is not None:
        keypoints_xy[9] = l_wrist_xy
        keypoints_conf[9] = 0.9
    return Detection(
        frame_idx=frame_idx,
        bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        confidence=0.95,
        keypoints_xy=keypoints_xy,
        keypoints_conf=keypoints_conf,
        track_id=track_id,
    )


def test_empty_detections_returns_empty_bundle() -> None:
    result = compute_match_metrics(
        detections=[],
        shuttle_track=[],
        corners=_IDENTITY_CORNERS,
        video_props=_VIDEO_PROPS,
        video_id=_VIDEO_ID,
    )
    assert isinstance(result, MatchMetrics)
    assert result.video_id == _VIDEO_ID
    assert result.players == []
    assert result.shuttle.total_hit_events == 0
    assert result.duration_seconds == pytest.approx(10.0)  # 300 / 30


def test_static_player_has_zero_distance_and_concentrated_zones() -> None:
    detections = [
        _make_detection(
            i,
            track_id=1,
            ankle_xy=(150.0, 800.0),
            nose_xy=(150.0, 700.0),
        )
        for i in range(30)
    ]
    result = compute_match_metrics(
        detections=detections,
        shuttle_track=[],
        corners=_IDENTITY_CORNERS,
        video_props=_VIDEO_PROPS,
        video_id=_VIDEO_ID,
    )
    assert len(result.players) == 1
    p = result.players[0]
    assert isinstance(p, PlayerMetrics)
    assert p.track_id == 1
    assert p.detection_frame_count == 30
    assert p.total_distance_m == pytest.approx(0.0)
    assert p.avg_speed_mps == pytest.approx(0.0)
    assert p.max_speed_mps == pytest.approx(0.0)
    assert p.convex_hull_area_m2 == pytest.approx(0.0)

    # Player is in near half (y=800 > net_y=730), front third (y ∈ [730, 953]),
    # left third (x=150 < 263). Zone sums for depth + lateral = 2.0 total.
    assert p.front_third_pct == pytest.approx(1.0)
    assert p.mid_third_pct == pytest.approx(0.0)
    assert p.back_third_pct == pytest.approx(0.0)
    assert p.left_third_pct == pytest.approx(1.0)
    assert p.center_third_pct == pytest.approx(0.0)
    assert p.right_third_pct == pytest.approx(0.0)
    depth_sum = p.front_third_pct + p.mid_third_pct + p.back_third_pct
    lateral_sum = p.left_third_pct + p.center_third_pct + p.right_third_pct
    assert depth_sum == pytest.approx(1.0)
    assert lateral_sum == pytest.approx(1.0)


def test_linear_motion_produces_nonzero_distance_and_area() -> None:
    # Move from (100, 800) to (190, 800) over 10 frames = 90 px = 0.9 m along
    # a horizontal line. Hull of colinear points is degenerate so area = 0.
    positions = [(100.0 + 10 * i, 800.0) for i in range(10)]
    detections = [
        _make_detection(
            i,
            track_id=1,
            ankle_xy=positions[i],
            nose_xy=(positions[i][0], 700.0),
        )
        for i in range(10)
    ]
    result = compute_match_metrics(
        detections=detections,
        shuttle_track=[],
        corners=_IDENTITY_CORNERS,
        video_props=_VIDEO_PROPS,
        video_id=_VIDEO_ID,
    )
    p = result.players[0]
    assert p.total_distance_m == pytest.approx(0.9, abs=1e-6)
    # 10 px per frame @ 30 fps = 10 * 30 cm/s = 300 cm/s = 3.0 m/s constant
    assert p.avg_speed_mps == pytest.approx(3.0, abs=1e-6)
    assert p.max_speed_mps == pytest.approx(3.0, abs=1e-6)
    assert p.convex_hull_area_m2 == pytest.approx(0.0)


def test_triangle_motion_gives_positive_convex_hull_area() -> None:
    # 3 distinct positions forming a triangle with a 100x100 px right angle
    # → triangle area = 5000 px² = 0.5 m². Stayed at each vertex for 3 frames.
    vertices: list[tuple[float, float]] = [
        (300.0, 800.0),
        (400.0, 800.0),
        (300.0, 900.0),
    ]
    detections: list[Detection] = []
    frame = 0
    for vx, vy in vertices:
        for _ in range(3):
            detections.append(
                _make_detection(
                    frame,
                    track_id=1,
                    ankle_xy=(vx, vy),
                    nose_xy=(vx, vy - 100.0),
                )
            )
            frame += 1

    result = compute_match_metrics(
        detections=detections,
        shuttle_track=[],
        corners=_IDENTITY_CORNERS,
        video_props=_VIDEO_PROPS,
        video_id=_VIDEO_ID,
    )
    p = result.players[0]
    # 100*100/2 = 5000 px² / 10000 px²/m² = 0.5 m²
    assert p.convex_hull_area_m2 == pytest.approx(0.5, abs=1e-6)


def test_hit_events_produce_shot_counts() -> None:
    # Player 1 (track 1) stays in near half front-left, wrist at (170, 750).
    # Player 2 (track 2) stays in far half back-right, wrist at (570, 100).
    # Shuttle appears near each wrist in turn, alternating 4 times.
    detections: list[Detection] = []
    shuttle: list[ShuttlePoint] = []

    hit_frames = [0, 6, 12, 18]
    for frame in range(20):
        detections.append(
            _make_detection(
                frame,
                track_id=1,
                ankle_xy=(150.0, 800.0),
                nose_xy=(150.0, 700.0),
                l_wrist_xy=(170.0, 750.0),
            )
        )
        detections.append(
            _make_detection(
                frame,
                track_id=2,
                ankle_xy=(550.0, 150.0),
                nose_xy=(550.0, 50.0),
                l_wrist_xy=(570.0, 100.0),
            )
        )

    for i, frame in enumerate(hit_frames):
        if i % 2 == 0:
            shuttle.append(ShuttlePoint(frame_idx=frame, x=170, y=750))
        else:
            shuttle.append(ShuttlePoint(frame_idx=frame, x=570, y=100))

    result = compute_match_metrics(
        detections=detections,
        shuttle_track=shuttle,
        corners=_IDENTITY_CORNERS,
        video_props=_VIDEO_PROPS,
        video_id=_VIDEO_ID,
    )
    assert result.shuttle.total_hit_events == 4

    players_by_id = {p.track_id: p for p in result.players}
    assert players_by_id[1].shot_count == 2
    assert players_by_id[2].shot_count == 2


def test_metrics_schema_is_stable() -> None:
    """Smoke test: MatchMetrics JSON round-trips through pydantic."""
    empty = compute_match_metrics(
        detections=[],
        shuttle_track=[],
        corners=_IDENTITY_CORNERS,
        video_props=_VIDEO_PROPS,
        video_id=_VIDEO_ID,
    )
    serialized = empty.model_dump_json()
    restored = MatchMetrics.model_validate_json(serialized)
    assert restored == empty
    assert isinstance(restored.shuttle, ShuttleMetrics)
