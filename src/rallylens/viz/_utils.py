"""Internal shared helpers for the viz package.

Court diagram constants, homography computation, and foot-position extraction
are centralised here so overlay.py and viz_court.py stay thin.
"""

# TODO: promote court geometry (COURT_W/H, MARGIN, _NET_Y, compute_homography,
# project_point, foot_pixel_from_detection) to a neutral module such as
# rallylens.vision.court_geometry. Today rallylens.analysis.metrics imports
# them from here, which inverts the viz → analysis layering. See plan
# dreamy-watching-tide for the reasoning.

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Final

import cv2
import numpy as np

from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import HitEvent, ShuttlePoint

# Consecutive candidate hit frames within this gap are collapsed into a single event
_HIT_GROUPING_MAX_GAP_FRAMES: Final[int] = 2

# Gaussian blur kernel width: (sigma * multiplier) + 1
_BLUR_KERNEL_MULTIPLIER: Final[int] = 4

# Fading-trail radius scaling
_TRAIL_RADIUS_MIN_FACTOR: Final[float] = 0.4
_TRAIL_RADIUS_MAX_FACTOR: Final[float] = 0.6
_TRAIL_RADIUS_MIN_PX: Final[int] = 2

# ---------------------------------------------------------------------------
# Court diagram constants (1 px = 1 cm)
# ---------------------------------------------------------------------------

COURT_W: int = 610   # 6.10 m — doubles width
COURT_H: int = 1340  # 13.40 m — court length
MARGIN: int = 60     # blank border around the court lines

IMG_W: int = COURT_W + 2 * MARGIN   # 730
IMG_H: int = COURT_H + 2 * MARGIN   # 1460

# Y-offsets from the top of the court (cm / px)
_NET_Y: int = 670
_SHORT_SVC_FAR: int = _NET_Y - 198   # 472
_SHORT_SVC_NEAR: int = _NET_Y + 198  # 868
_DBL_LONG_FAR: int = 76
_DBL_LONG_NEAR: int = COURT_H - 76   # 1264

# X-offsets
_SINGLES_LEFT: int = 46
_SINGLES_RIGHT: int = COURT_W - 46   # 564
_CENTER_X: int = COURT_W // 2        # 305

# ---------------------------------------------------------------------------
# Track-color palette (BGR)
# ---------------------------------------------------------------------------

TRACK_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),      # green
    (0, 128, 255),    # sky-blue / orange
    (255, 128, 0),    # orange-blue
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 255, 0),    # yellow
]

# Shared shuttle highlight color (BGR: yellow)
SHUTTLE_COLOR: tuple[int, int, int] = (0, 255, 255)


def track_color(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (200, 200, 200)
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def group_detections_by_frame(
    detections: Iterable[Detection],
) -> dict[int, list[Detection]]:
    """Group detections by their `frame_idx` for O(1) per-frame lookup."""
    grouped: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        grouped[det.frame_idx].append(det)
    return grouped


# ---------------------------------------------------------------------------
# Court diagram drawing
# ---------------------------------------------------------------------------


def draw_court_background() -> np.ndarray:
    """Return a (IMG_H, IMG_W, 3) BGR image of a standard badminton court."""
    img: np.ndarray = np.full((IMG_H, IMG_W, 3), (34, 85, 34), dtype=np.uint8)
    m = MARGIN
    white = (255, 255, 255)

    def hline(y: int) -> None:
        cv2.line(img, (m, m + y), (m + COURT_W, m + y), white, 2)

    def vline(x: int) -> None:
        cv2.line(img, (m + x, m), (m + x, m + COURT_H), white, 2)

    # Outer doubles boundary
    cv2.rectangle(img, (m, m), (m + COURT_W, m + COURT_H), white, 2)
    # Singles sidelines
    vline(_SINGLES_LEFT)
    vline(_SINGLES_RIGHT)
    # Net
    hline(_NET_Y)
    # Short service lines
    hline(_SHORT_SVC_FAR)
    hline(_SHORT_SVC_NEAR)
    # Doubles long service lines
    hline(_DBL_LONG_FAR)
    hline(_DBL_LONG_NEAR)
    # Center service line (full court length for clarity)
    vline(_CENTER_X)

    return img


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------


def compute_homography(corners: CourtCorners) -> np.ndarray:
    """Compute homography mapping video pixel corners to the court diagram."""
    src = np.array(
        [
            [corners.top_left[0], corners.top_left[1]],
            [corners.top_right[0], corners.top_right[1]],
            [corners.bottom_left[0], corners.bottom_left[1]],
            [corners.bottom_right[0], corners.bottom_right[1]],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [MARGIN, MARGIN],
            [MARGIN + COURT_W, MARGIN],
            [MARGIN, MARGIN + COURT_H],
            [MARGIN + COURT_W, MARGIN + COURT_H],
        ],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(src, dst)
    if H is None:
        raise RuntimeError(
            "could not compute homography from court corners — corners may be collinear"
        )
    return H


def project_point(H: np.ndarray, px: float, py: float) -> tuple[int, int]:
    """Project a single pixel coordinate into court diagram space.

    Only valid for pixels imaging points on the court floor plane. Applying
    this to aerial points (e.g. a shuttlecock mid-flight) yields the point
    where the camera ray hits the ground, not the shuttle's ground shadow.
    """
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return (int(result[0, 0, 0]), int(result[0, 0, 1]))


# ---------------------------------------------------------------------------
# Foot-position extraction
# ---------------------------------------------------------------------------


def foot_point_from_detection(
    det: Detection,
    H: np.ndarray,
    kp_conf_thresh: float = 0.3,
) -> tuple[int, int] | None:
    """Project a single detection's foot midpoint into court-diagram space.

    Returns None if neither ankle keypoint meets the confidence threshold
    or the keypoint list is incomplete.
    """
    foot = foot_pixel_from_detection(det, kp_conf_thresh)
    if foot is None:
        return None
    return project_point(H, foot[0], foot[1])


def extract_foot_positions(
    detections: list[Detection],
    H: np.ndarray,
    kp_conf_thresh: float = 0.3,
) -> list[tuple[int, int]]:
    """Project every detection with valid feet into court-diagram space."""
    positions: list[tuple[int, int]] = []
    for det in detections:
        pt = foot_point_from_detection(det, H, kp_conf_thresh)
        if pt is not None:
            positions.append(pt)
    return positions


# ---------------------------------------------------------------------------
# Shuttle court-space projection (TRACE-style)
# ---------------------------------------------------------------------------
#
# A flat ground-plane homography cannot correctly project aerial shuttle
# pixels: the camera ray through a shuttle pixel intersects z=0 somewhere
# past the shuttle's actual ground shadow. Following hgupt3/TRACE's
# CourtDetection.py approach, we sidestep this by:
#   1) Detecting "hit" events where the shuttle pixel is near a player's
#      wrist keypoint (racket hand).
#   2) For each hit event, composing the top-down coordinate as
#        (project(ball).x, project(foot_midpoint).y)
#      so the x axis tracks the shuttle's lateral position while the y
#      axis is snapped to the known ground y of the striking player.
#   3) Linearly interpolating between consecutive events to fill the
#      intermediate frames — a reasonable approximation of a badminton
#      shuttle's top-down trajectory, which is nearly straight between
#      hits (the curved 3D arc projects close to a line on the ground).
#
# COCO-17 keypoint indices used here
_KP_NOSE = 0
_KP_L_WRIST = 9
_KP_R_WRIST = 10
_KP_L_ANKLE = 15
_KP_R_ANKLE = 16


def foot_pixel_from_detection(
    det: Detection, kp_conf_thresh: float = 0.3
) -> tuple[float, float] | None:
    """Return the detection's foot midpoint in source-video pixel space.

    Mirrors `foot_point_from_detection` but skips the homography so the
    caller can combine this with raw shuttle pixels before projecting.
    Returns None if neither ankle keypoint meets the confidence threshold.
    """
    if len(det.keypoints_xy) < 17:
        return None
    l_conf = det.keypoints_conf[_KP_L_ANKLE] if len(det.keypoints_conf) > _KP_L_ANKLE else 0.0
    r_conf = det.keypoints_conf[_KP_R_ANKLE] if len(det.keypoints_conf) > _KP_R_ANKLE else 0.0
    if l_conf > kp_conf_thresh and r_conf > kp_conf_thresh:
        lx, ly = det.keypoints_xy[_KP_L_ANKLE]
        rx, ry = det.keypoints_xy[_KP_R_ANKLE]
        return ((lx + rx) / 2, (ly + ry) / 2)
    if l_conf > kp_conf_thresh:
        return det.keypoints_xy[_KP_L_ANKLE]
    if r_conf > kp_conf_thresh:
        return det.keypoints_xy[_KP_R_ANKLE]
    return None


def _best_wrist_pixel(
    det: Detection, ball_xy: tuple[float, float], kp_conf_thresh: float
) -> tuple[tuple[float, float], float] | None:
    """Return the wrist keypoint closest to the ball and its distance in pixels."""
    if len(det.keypoints_xy) <= _KP_R_WRIST:
        return None
    candidates: list[tuple[tuple[float, float], float]] = []
    for idx in (_KP_L_WRIST, _KP_R_WRIST):
        if len(det.keypoints_conf) <= idx:
            continue
        if det.keypoints_conf[idx] <= kp_conf_thresh:
            continue
        wx, wy = det.keypoints_xy[idx]
        d = math.hypot(ball_xy[0] - wx, ball_xy[1] - wy)
        candidates.append(((wx, wy), d))
    if not candidates:
        return None
    return min(candidates, key=lambda c: c[1])


def _nose_pixel(det: Detection, kp_conf_thresh: float) -> tuple[float, float] | None:
    if len(det.keypoints_xy) <= _KP_NOSE or len(det.keypoints_conf) <= _KP_NOSE:
        return None
    if det.keypoints_conf[_KP_NOSE] <= kp_conf_thresh:
        return None
    return det.keypoints_xy[_KP_NOSE]


def extract_hit_events(
    detections: list[Detection],
    shuttle_track: list[ShuttlePoint],
    H: np.ndarray,
    *,
    kp_conf_thresh: float = 0.3,
    hit_radius_factor: float = 0.9,
) -> list[HitEvent]:
    """Detect shuttle-to-racket contacts via TRACE's wrist-proximity heuristic.

    Implements steps 1–3 of the original `compute_shuttle_court_positions`
    algorithm without the interpolation step, so callers who need the raw
    event list (metrics, analytics) can consume it directly.

    Algorithm:
      1. For each shuttle frame, find the player whose nearest wrist is
         within `hit_radius_factor * body_length` of the ball. `body_length`
         is the nose-to-foot-midpoint distance of that player, serving as a
         camera-invariant proxy for racket reach.
      2. Frames passing the check become "hit candidates" with candidate
         position `(project(ball).x, project(foot_midpoint).y)`.
      3. Consecutive candidate frames (gap <= 2) are grouped; within each
         group the frame with the smallest wrist-to-ball distance is kept
         as the canonical hit event. This collapses a multi-frame near-
         contact into a single event.

    Returns a chronologically sorted list of `HitEvent`. Empty list if no
    contacts are detected, the shuttle track is empty, or detections are
    missing keypoints.
    """
    if not shuttle_track or not detections:
        return []

    dets_by_frame = group_detections_by_frame(detections)
    shuttle_by_frame: dict[int, ShuttlePoint] = {sp.frame_idx: sp for sp in shuttle_track}

    # (frame_idx, wrist_ball_distance, event_point, striker_track_id)
    candidates: list[tuple[int, float, tuple[int, int], int | None]] = []
    for fi in sorted(shuttle_by_frame.keys()):
        sp = shuttle_by_frame[fi]
        ball_xy = (float(sp.x), float(sp.y))
        best: tuple[float, tuple[int, int], int | None] | None = None
        for det in dets_by_frame.get(fi, []):
            foot = foot_pixel_from_detection(det, kp_conf_thresh)
            if foot is None:
                continue
            nose = _nose_pixel(det, kp_conf_thresh)
            if nose is None:
                continue
            wrist_hit = _best_wrist_pixel(det, ball_xy, kp_conf_thresh)
            if wrist_hit is None:
                continue
            _, wrist_dist = wrist_hit
            body_len = math.hypot(foot[0] - nose[0], foot[1] - nose[1])
            if body_len <= 0:
                continue
            radius = hit_radius_factor * body_len
            if wrist_dist >= radius:
                continue
            foot_proj = project_point(H, foot[0], foot[1])
            event_pt = foot_proj
            if best is None or wrist_dist < best[0]:
                best = (wrist_dist, event_pt, det.track_id)
        if best is not None:
            candidates.append((fi, best[0], best[1], best[2]))

    if not candidates:
        return []

    # Collapse consecutive near-contact frames into a single canonical hit
    hits: list[HitEvent] = []
    group: list[tuple[int, float, tuple[int, int], int | None]] = [candidates[0]]
    for c in candidates[1:]:
        if c[0] - group[-1][0] <= _HIT_GROUPING_MAX_GAP_FRAMES:
            group.append(c)
        else:
            best_in_group = min(group, key=lambda x: x[1])
            hits.append(
                HitEvent(
                    frame_idx=best_in_group[0],
                    striker_track_id=best_in_group[3],
                    wrist_distance_px=best_in_group[1],
                    event_court_xy=best_in_group[2],
                )
            )
            group = [c]
    best_in_group = min(group, key=lambda x: x[1])
    hits.append(
        HitEvent(
            frame_idx=best_in_group[0],
            striker_track_id=best_in_group[3],
            wrist_distance_px=best_in_group[1],
            event_court_xy=best_in_group[2],
        )
    )

    return hits


def compute_shuttle_court_positions(
    detections: list[Detection],
    shuttle_track: list[ShuttlePoint],
    H: np.ndarray,
    *,
    kp_conf_thresh: float = 0.3,
    hit_radius_factor: float = 0.9,
) -> dict[int, tuple[int, int]]:
    """Compute per-frame top-down shuttle positions via TRACE's hit+interp trick.

    Thin composition: calls `extract_hit_events` for the event list and then
    linearly interpolates between consecutive events so the trail draws
    cleanly. See `extract_hit_events` for the hit-detection details.

    Returns a dict mapping frame_idx -> (x, y) in court-diagram pixel space.
    Empty dict if no hit events are detected.
    """
    hits = extract_hit_events(
        detections,
        shuttle_track,
        H,
        kp_conf_thresh=kp_conf_thresh,
        hit_radius_factor=hit_radius_factor,
    )
    if not hits:
        return {}

    result: dict[int, tuple[int, int]] = {}
    for i in range(len(hits) - 1):
        t0, p0 = hits[i].frame_idx, hits[i].event_court_xy
        t1, p1 = hits[i + 1].frame_idx, hits[i + 1].event_court_xy
        dt = t1 - t0
        if dt <= 0:
            result[t0] = p0
            continue
        for j in range(dt):
            alpha = j / dt
            x = int(round(p0[0] + alpha * (p1[0] - p0[0])))
            y = int(round(p0[1] + alpha * (p1[1] - p0[1])))
            result[t0 + j] = (x, y)
    result[hits[-1].frame_idx] = hits[-1].event_court_xy

    return result


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------


def build_heatmap_over_court(
    court: np.ndarray,
    positions: list[tuple[int, int]],
    *,
    blur_sigma: int = 12,
    alpha_max: float = 0.75,
) -> np.ndarray:
    """Return a copy of `court` with a Gaussian-smoothed JET heatmap of `positions` blended on.

    Positions outside the image bounds are silently skipped. If positions is
    empty (or all out of bounds so the accumulated grid is zero) a plain copy
    of `court` is returned.
    """
    if not positions:
        return court.copy()

    grid: np.ndarray = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    for x, y in positions:
        if 0 <= x < IMG_W and 0 <= y < IMG_H:
            grid[y, x] += 1.0

    kernel_size = blur_sigma * _BLUR_KERNEL_MULTIPLIER + 1
    grid = cv2.GaussianBlur(grid, (kernel_size, kernel_size), blur_sigma)

    max_val = float(grid.max())
    if max_val == 0:
        return court.copy()

    normalized = (grid / max_val * 255).astype(np.uint8)
    heatmap_rgb = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    alpha_map = (grid / max_val * alpha_max).astype(np.float32)[:, :, np.newaxis]
    blended = (
        court.astype(np.float32) * (1 - alpha_map)
        + heatmap_rgb.astype(np.float32) * alpha_map
    ).astype(np.uint8)
    return blended


# ---------------------------------------------------------------------------
# Fading-trail rendering
# ---------------------------------------------------------------------------


def draw_fading_trail(
    frame: np.ndarray,
    points: Iterable[tuple[int, int]],
    *,
    color: tuple[int, int, int],
    head_radius: int,
) -> None:
    """Draw a chronological point trail: newest point is largest + brightest.

    Each point is rendered as a filled circle whose BGR channels and radius
    scale linearly with its position in the trail, so older points fade toward
    black and shrink. Works on any backdrop (plain court, heatmap, or raw
    video frame) because it fades toward black instead of toward a fixed
    background color.
    """
    pts = list(points)
    n = len(pts)
    if n == 0:
        return
    for i, pt in enumerate(pts):
        alpha = (i + 1) / n
        radius = max(
            _TRAIL_RADIUS_MIN_PX,
            int(head_radius * (_TRAIL_RADIUS_MIN_FACTOR + _TRAIL_RADIUS_MAX_FACTOR * alpha)),
        )
        faded = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
        cv2.circle(frame, pt, radius, faded, -1, cv2.LINE_AA)


def render_pip_court_frame(
    court_bg: np.ndarray,
    player_trails: Mapping[int, Iterable[tuple[int, int]]],
    shuttle_trail: Iterable[tuple[int, int]],
    *,
    player_radius: int = 6,
    shuttle_radius: int = 4,
) -> np.ndarray:
    """Render a single court-diagram frame with player and shuttle trails."""
    frame = court_bg.copy()

    for tid, trail in player_trails.items():
        color = track_color(tid)
        pts = list(trail)
        draw_fading_trail(frame, pts, color=color, head_radius=player_radius)
        if pts:
            cv2.circle(frame, pts[-1], player_radius, color, -1, cv2.LINE_AA)

    s_pts = list(shuttle_trail)
    draw_fading_trail(frame, s_pts, color=SHUTTLE_COLOR, head_radius=shuttle_radius)
    if s_pts:
        cv2.circle(frame, s_pts[-1], shuttle_radius, SHUTTLE_COLOR, -1, cv2.LINE_AA)

    return frame
