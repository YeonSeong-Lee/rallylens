"""Deterministic match metrics computed from pipeline artifacts.

Produces a compact LLM-friendly numerical summary from player detections,
shuttle tracks, and court calibration. The output is independent of any
specific LLM — it is consumed by `rallylens.llm` but could equally feed a
dashboard, notebook, or A/B evaluation harness.

All distances are reported in meters and all areas in square meters. The
court-diagram coordinate space is 1 px = 1 cm, so the conversions are
straightforward divisions.
"""

from __future__ import annotations

import math
import statistics
from typing import Final, Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from rallylens.domain.video import VideoProperties
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import HitEvent, ShuttlePoint
from rallylens.viz._utils import (
    COURT_H,
    COURT_W,
    MARGIN,
    compute_homography,
    extract_hit_events,
    foot_pixel_from_detection,
    project_point,
)

# Court image coordinates. Duplicated from viz._utils._NET_Y pending the
# court-geometry promotion called out in viz/_utils.py:8.
_NET_Y_OFFSET: Final[int] = 670  # from top of court (cm, 1 px = 1 cm)
_NET_Y_IMG: Final[int] = MARGIN + _NET_Y_OFFSET
_COURT_LEFT_X: Final[int] = MARGIN
_COURT_RIGHT_X: Final[int] = MARGIN + COURT_W
_COURT_TOP_Y: Final[int] = MARGIN
_COURT_BOTTOM_Y: Final[int] = MARGIN + COURT_H

# Court-diagram space: 1 px = 1 cm
_PX_PER_M: Final[float] = 100.0
_PX2_PER_M2: Final[float] = 10_000.0

# Suppresses single-frame outliers when computing max speed
_SPEED_SMOOTH_WINDOW: Final[int] = 3

_KP_CONF_THRESH: Final[float] = 0.3


class PlayerMetrics(BaseModel):
    """Per-track deterministic metrics in real-world units."""

    model_config = ConfigDict(frozen=True)

    track_id: int
    detection_frame_count: int
    total_distance_m: float
    avg_speed_mps: float
    max_speed_mps: float
    convex_hull_area_m2: float
    shot_count: int
    front_third_pct: float
    mid_third_pct: float
    back_third_pct: float
    left_third_pct: float
    center_third_pct: float
    right_third_pct: float


class ShuttleMetrics(BaseModel):
    """Shuttle-derived match metrics (hits, rallies, speed)."""

    model_config = ConfigDict(frozen=True)

    total_hit_events: int
    avg_inter_hit_seconds: float
    avg_shuttle_speed_mps: float
    max_shuttle_speed_mps: float


class MatchMetrics(BaseModel):
    """Top-level metrics bundle for a single video clip."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1"] = "1"
    video_id: str
    fps: float
    duration_seconds: float
    frame_count: int
    players: list[PlayerMetrics]
    shuttle: ShuttleMetrics


def compute_match_metrics(
    detections: list[Detection],
    shuttle_track: list[ShuttlePoint],
    corners: CourtCorners,
    video_props: VideoProperties,
    video_id: str,
) -> MatchMetrics:
    """Compute deterministic match metrics from pipeline artifacts."""
    fps = float(video_props.fps)
    frame_count = int(video_props.frame_count)
    duration_seconds = frame_count / fps if fps > 0 else 0.0

    if not detections:
        return MatchMetrics(
            video_id=video_id,
            fps=fps,
            duration_seconds=duration_seconds,
            frame_count=frame_count,
            players=[],
            shuttle=_empty_shuttle_metrics(),
        )

    H = compute_homography(corners)
    hits = extract_hit_events(
        detections, shuttle_track, H, kp_conf_thresh=_KP_CONF_THRESH
    )

    players = _compute_player_metrics(detections, H, fps, hits)
    shuttle = _compute_shuttle_metrics(hits, fps)

    return MatchMetrics(
        video_id=video_id,
        fps=fps,
        duration_seconds=duration_seconds,
        frame_count=frame_count,
        players=players,
        shuttle=shuttle,
    )


def _empty_shuttle_metrics() -> ShuttleMetrics:
    return ShuttleMetrics(
        total_hit_events=0,
        avg_inter_hit_seconds=0.0,
        avg_shuttle_speed_mps=0.0,
        max_shuttle_speed_mps=0.0,
    )


def _compute_player_metrics(
    detections: list[Detection],
    H: np.ndarray,
    fps: float,
    hits: list[HitEvent],
) -> list[PlayerMetrics]:
    by_track: dict[int, list[Detection]] = {}
    for det in detections:
        if det.track_id is None:
            continue
        by_track.setdefault(det.track_id, []).append(det)

    shots_by_track: dict[int, int] = {}
    for hit in hits:
        if hit.striker_track_id is not None:
            shots_by_track[hit.striker_track_id] = (
                shots_by_track.get(hit.striker_track_id, 0) + 1
            )

    results: list[PlayerMetrics] = []
    for track_id, dets in sorted(by_track.items()):
        dets_sorted = sorted(dets, key=lambda d: d.frame_idx)
        path = _project_foot_path(dets_sorted, H)
        points = [pt for _, pt in path]

        total_distance_m, avg_speed_mps, max_speed_mps = _compute_distance_speed(
            path, fps
        )
        zones = _compute_zone_distribution(points)

        results.append(
            PlayerMetrics(
                track_id=track_id,
                detection_frame_count=len(dets_sorted),
                total_distance_m=total_distance_m,
                avg_speed_mps=avg_speed_mps,
                max_speed_mps=max_speed_mps,
                convex_hull_area_m2=_convex_hull_area_m2(points),
                shot_count=shots_by_track.get(track_id, 0),
                front_third_pct=zones["front"],
                mid_third_pct=zones["mid"],
                back_third_pct=zones["back"],
                left_third_pct=zones["left"],
                center_third_pct=zones["center"],
                right_third_pct=zones["right"],
            )
        )

    return results


def _project_foot_path(
    dets_sorted: list[Detection],
    H: np.ndarray,
) -> list[tuple[int, tuple[int, int]]]:
    """Project each detection's foot midpoint into court-diagram space."""
    path: list[tuple[int, tuple[int, int]]] = []
    for det in dets_sorted:
        foot = foot_pixel_from_detection(det, _KP_CONF_THRESH)
        if foot is None:
            continue
        path.append((det.frame_idx, project_point(H, foot[0], foot[1])))
    return path


def _compute_distance_speed(
    path: list[tuple[int, tuple[int, int]]],
    fps: float,
) -> tuple[float, float, float]:
    """Return (total_distance_m, avg_speed_mps, max_speed_mps)."""
    if len(path) < 2 or fps <= 0:
        return 0.0, 0.0, 0.0

    total_distance_m = 0.0
    speeds_mps: list[float] = []
    for (f0, p0), (f1, p1) in zip(path, path[1:], strict=False):
        d_m = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / _PX_PER_M
        total_distance_m += d_m
        df = f1 - f0
        if df > 0:
            speeds_mps.append(d_m / (df / fps))

    avg_speed_mps, max_speed_mps = _speed_avg_max(speeds_mps)
    return total_distance_m, avg_speed_mps, max_speed_mps


def _speed_avg_max(speeds_mps: list[float]) -> tuple[float, float]:
    """Return (avg, max) after median-smoothing a raw speed series."""
    if not speeds_mps:
        return 0.0, 0.0
    smoothed = _median_smooth(speeds_mps, window=_SPEED_SMOOTH_WINDOW)
    return sum(smoothed) / len(smoothed), max(smoothed)


def _median_smooth(values: list[float], *, window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return list(values)
    half = window // 2
    result: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        result.append(statistics.median(values[lo:hi]))
    return result


def _convex_hull_area_m2(points: list[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(pts)
    area_px2 = float(cv2.contourArea(hull))
    return area_px2 / _PX2_PER_M2


_ZONE_KEYS: Final[tuple[str, ...]] = (
    "front", "mid", "back", "left", "center", "right",
)


def _compute_zone_distribution(
    points: list[tuple[int, int]],
) -> dict[str, float]:
    """Fraction of frames the player spent in each of 6 court zones.

    Front/mid/back are relative to the player's own half (front = nearest
    net). Left/center/right span the full court width. The player's half is
    chosen by median y relative to the net.
    """
    if not points:
        return dict.fromkeys(_ZONE_KEYS, 0.0)

    median_y = statistics.median(y for _, y in points)
    on_far_half = median_y < _NET_Y_IMG
    half_length = _NET_Y_IMG - _COURT_TOP_Y if on_far_half else _COURT_BOTTOM_Y - _NET_Y_IMG
    third_depth = half_length / 3

    court_width = _COURT_RIGHT_X - _COURT_LEFT_X
    third_width = court_width / 3

    counts = dict.fromkeys(_ZONE_KEYS, 0)
    for x, y in points:
        dist_from_net = _NET_Y_IMG - y if on_far_half else y - _NET_Y_IMG
        if dist_from_net < third_depth:
            counts["front"] += 1
        elif dist_from_net < 2 * third_depth:
            counts["mid"] += 1
        else:
            counts["back"] += 1

        offset_from_left = x - _COURT_LEFT_X
        if offset_from_left < third_width:
            counts["left"] += 1
        elif offset_from_left < 2 * third_width:
            counts["center"] += 1
        else:
            counts["right"] += 1

    n = len(points)
    return {k: counts[k] / n for k in _ZONE_KEYS}


def _compute_shuttle_metrics(
    hits: list[HitEvent],
    fps: float,
) -> ShuttleMetrics:
    total = len(hits)
    if total == 0:
        return _empty_shuttle_metrics()

    hits_sorted = sorted(hits, key=lambda h: h.frame_idx)

    intervals_s: list[float] = []
    speeds_mps: list[float] = []
    for h0, h1 in zip(hits_sorted, hits_sorted[1:], strict=False):
        df = h1.frame_idx - h0.frame_idx
        if df <= 0 or fps <= 0:
            continue
        dt = df / fps
        intervals_s.append(dt)
        p0, p1 = h0.event_court_xy, h1.event_court_xy
        d_m = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / _PX_PER_M
        speeds_mps.append(d_m / dt)

    avg_shuttle_speed_mps, max_shuttle_speed_mps = _speed_avg_max(speeds_mps)

    return ShuttleMetrics(
        total_hit_events=total,
        avg_inter_hit_seconds=sum(intervals_s) / len(intervals_s) if intervals_s else 0.0,
        avg_shuttle_speed_mps=avg_shuttle_speed_mps,
        max_shuttle_speed_mps=max_shuttle_speed_mps,
    )


