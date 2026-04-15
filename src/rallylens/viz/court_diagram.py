"""Court coordinate trajectory visualization.

Projects player foot positions and shuttlecock positions into standard court
diagram space via homography and saves the result as an animated GIF that
plays back chronologically.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Final

import cv2
import imageio.v2 as iio
import numpy as np

from rallylens.common import ensure_dir
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import ShuttlePoint
from rallylens.viz._utils import (
    IMG_H,
    IMG_W,
    SHUTTLE_COLOR,
    build_heatmap_over_court,
    compute_homography,
    compute_shuttle_court_positions,
    draw_court_background,
    draw_fading_trail,
    extract_foot_positions,
    foot_point_from_detection,
    group_detections_by_frame,
    track_color,
)

_MAX_DISPLAY_PLAYERS: Final[int] = 2
_LABEL_OFFSET_PX: Final[int] = 4
_PLAYER_LABEL_FONT_SCALE: Final[float] = 0.6

__all__ = ["render_court_diagram"]


def render_court_diagram(
    detections: list[Detection],
    shuttle_track: list[ShuttlePoint],
    corners: CourtCorners,
    out_path: Path,
    *,
    fps: float,
    stride: int = 5,
    scale: float = 0.5,
    trail_len: int = 30,
    blur_sigma: int = 12,
    kp_conf_thresh: float = 0.3,
    player_radius: int = 6,
    shuttle_radius: int = 4,
) -> Path:
    """Render an animated GIF of player1/player2/shuttle movement over the court.

    The background of every frame is a static player-footprint heatmap (same
    style as `render_heatmap`), and the animated player and shuttle trails are
    drawn on top in chronological order. imageio expects RGB frames while
    OpenCV draws in BGR, so each emitted frame is color-converted before being
    appended.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    H = compute_homography(corners)
    foot_positions = extract_foot_positions(detections, H, kp_conf_thresh)
    bg = build_heatmap_over_court(
        draw_court_background(),
        foot_positions,
        blur_sigma=blur_sigma,
    )

    dets_by_frame = group_detections_by_frame(detections)

    shuttle_court_positions = compute_shuttle_court_positions(
        detections, shuttle_track, H, kp_conf_thresh=kp_conf_thresh
    )

    max_det_frame = max((det.frame_idx for det in detections), default=-1)
    max_shuttle_frame = max((sp.frame_idx for sp in shuttle_track), default=-1)
    max_frame = max(max_det_frame, max_shuttle_frame, 0)

    observed_ids = sorted({det.track_id for det in detections if det.track_id is not None})
    id_to_label: dict[int, str] = {}
    for rank, tid in enumerate(observed_ids[:_MAX_DISPLAY_PLAYERS]):
        id_to_label[tid] = f"P{rank + 1}"

    player_trails: dict[int, deque[tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=trail_len)
    )
    shuttle_trail: deque[tuple[int, int]] = deque(maxlen=trail_len)

    out_w = max(1, int(IMG_W * scale))
    out_h = max(1, int(IMG_H * scale))

    ensure_dir(out_path.parent)

    writer = iio.get_writer(
        str(out_path),
        mode="I",
        duration=int(round(1000 * stride / fps)),
        loop=0,
    )

    def emit(frame_bgr: np.ndarray) -> None:
        if scale != 1.0:
            frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)

    try:
        emitted = False
        for fi in range(max_frame + 1):
            for det in dets_by_frame.get(fi, []):
                pt = foot_point_from_detection(det, H, kp_conf_thresh)
                if pt is None or det.track_id is None:
                    continue
                player_trails[det.track_id].append(pt)

            pos = shuttle_court_positions.get(fi)
            if pos is not None and 0 <= pos[0] < IMG_W and 0 <= pos[1] < IMG_H:
                shuttle_trail.append(pos)

            if fi % stride != 0:
                continue

            frame = bg.copy()

            for tid, trail in player_trails.items():
                color = track_color(tid)
                draw_fading_trail(frame, trail, color=color, head_radius=player_radius)
                if trail:
                    cx, cy = trail[-1]
                    cv2.circle(frame, (cx, cy), player_radius, color, -1, cv2.LINE_AA)
                    label = id_to_label.get(tid, "")
                    if label:
                        cv2.putText(
                            frame,
                            label,
                            (cx + player_radius + _LABEL_OFFSET_PX, cy + _LABEL_OFFSET_PX),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            _PLAYER_LABEL_FONT_SCALE,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

            draw_fading_trail(
                frame, shuttle_trail, color=SHUTTLE_COLOR, head_radius=shuttle_radius
            )
            if shuttle_trail:
                sx, sy = shuttle_trail[-1]
                cv2.circle(frame, (sx, sy), shuttle_radius, SHUTTLE_COLOR, -1, cv2.LINE_AA)

            emit(frame)
            emitted = True

        if not emitted:
            emit(bg.copy())
    finally:
        writer.close()

    return out_path
