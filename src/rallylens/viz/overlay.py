"""Video overlay renderer.

Draws player bounding boxes, COCO-17 skeletons, and a fading shuttle trail
onto each frame of the source video and writes the result as an MP4.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from rallylens.common import open_video, open_video_writer, read_video_properties
from rallylens.vision.court_detector import CourtCorners
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import ShuttlePoint
from rallylens.viz._utils import (
    IMG_H,
    IMG_W,
    SHUTTLE_COLOR,
    compute_homography,
    compute_shuttle_court_positions,
    draw_court_background,
    draw_fading_trail,
    foot_point_from_detection,
    group_detections_by_frame,
    render_pip_court_frame,
    track_color,
)

_KEYPOINT_RADIUS: Final[int] = 3
_LABEL_FONT_SCALE: Final[float] = 0.5
_LABEL_TEXT_THICKNESS: Final[int] = 1

# COCO-17 skeleton connections (0-indexed keypoint pairs)
_COCO_SKELETON: list[tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # face
    (5, 7), (7, 9),                          # left arm
    (6, 8), (8, 10),                         # right arm
    (5, 6),                                  # shoulders
    (5, 11), (6, 12), (11, 12),             # torso
    (11, 13), (13, 15),                      # left leg
    (12, 14), (14, 16),                      # right leg
]

__all__ = ["render_overlay_video"]


def _draw_bbox(frame: np.ndarray, det: Detection, thickness: int) -> None:
    x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
    color = track_color(det.track_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"id={det.track_id}" if det.track_id is not None else f"{det.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, _LABEL_FONT_SCALE, _LABEL_TEXT_THICKNESS
    )
    lbl_y = max(y1, th + 4)
    cv2.rectangle(frame, (x1, lbl_y - th - 4), (x1 + tw + 4, lbl_y), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, lbl_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, _LABEL_FONT_SCALE, (0, 0, 0),
        _LABEL_TEXT_THICKNESS, cv2.LINE_AA,
    )


def _draw_skeleton(
    frame: np.ndarray,
    det: Detection,
    kp_conf_thresh: float,
    thickness: int,
) -> None:
    kps = det.keypoints_xy
    confs = det.keypoints_conf
    color = track_color(det.track_id)
    for i, (kx, ky) in enumerate(kps):
        if i < len(confs) and confs[i] > kp_conf_thresh:
            cv2.circle(frame, (int(kx), int(ky)), _KEYPOINT_RADIUS, color, -1, cv2.LINE_AA)
    for a, b in _COCO_SKELETON:
        if a >= len(kps) or b >= len(kps):
            continue
        if a >= len(confs) or b >= len(confs):
            continue
        if confs[a] <= kp_conf_thresh or confs[b] <= kp_conf_thresh:
            continue
        ax, ay = int(kps[a][0]), int(kps[a][1])
        bx, by = int(kps[b][0]), int(kps[b][1])
        cv2.line(frame, (ax, ay), (bx, by), color, thickness, cv2.LINE_AA)


def _draw_shuttle_trail(
    frame: np.ndarray, trail: collections.deque  # type: ignore[type-arg]
) -> None:
    draw_fading_trail(
        frame,
        [(pt.x, pt.y) for pt in trail],
        color=SHUTTLE_COLOR,
        head_radius=8,
    )


def render_overlay_video(
    video_path: Path,
    detections: list[Detection],
    shuttle_track: list[ShuttlePoint],
    out_path: Path,
    *,
    corners: CourtCorners | None = None,
    trail_len: int = 30,
    kp_conf_thresh: float = 0.3,
    bbox_thickness: int = 2,
    skeleton_thickness: int = 2,
    pip_scale: float = 0.5,
    pip_margin: int = 10,
    fourcc: str = "mp4v",
) -> Path:
    """Read source video frame-by-frame, draw overlays, and write MP4.

    Returns out_path on success.
    """
    props = read_video_properties(video_path)

    detections_by_frame = group_detections_by_frame(detections)
    shuttle_by_frame: dict[int, ShuttlePoint] = {pt.frame_idx: pt for pt in shuttle_track}

    trail: collections.deque[ShuttlePoint] = collections.deque(maxlen=trail_len)

    pip_enabled = corners is not None
    H: np.ndarray | None = None
    court_bg: np.ndarray | None = None
    shuttle_court: dict[int, tuple[int, int]] = {}
    pip_h = pip_w = pip_x = pip_y = 0
    pip_player_trails: dict[int, collections.deque[tuple[int, int]]] = {}
    pip_shuttle_trail: collections.deque[tuple[int, int]] = collections.deque(maxlen=trail_len)

    if pip_enabled:
        assert corners is not None
        H = compute_homography(corners)
        court_bg = draw_court_background()
        shuttle_court = compute_shuttle_court_positions(
            detections, shuttle_track, H, kp_conf_thresh=kp_conf_thresh
        )
        pip_h = max(1, int(props.height * pip_scale))
        pip_w = max(1, int(pip_h * IMG_W / IMG_H))
        pip_x = pip_margin
        pip_y = props.height - pip_h - pip_margin

    frame_idx = 0

    with (
        open_video(video_path) as cap,
        open_video_writer(out_path, fourcc, props.fps, (props.width, props.height)) as writer,
    ):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in shuttle_by_frame:
                trail.append(shuttle_by_frame[frame_idx])

            _draw_shuttle_trail(frame, trail)

            for det in detections_by_frame.get(frame_idx, []):
                _draw_bbox(frame, det, bbox_thickness)
                _draw_skeleton(frame, det, kp_conf_thresh, skeleton_thickness)

            if pip_enabled and H is not None and court_bg is not None:
                for det in detections_by_frame.get(frame_idx, []):
                    if det.track_id is None:
                        continue
                    pt = foot_point_from_detection(det, H, kp_conf_thresh)
                    if pt is None:
                        continue
                    if det.track_id not in pip_player_trails:
                        pip_player_trails[det.track_id] = collections.deque(
                            maxlen=trail_len
                        )
                    pip_player_trails[det.track_id].append(pt)

                spos = shuttle_court.get(frame_idx)
                if spos is not None:
                    pip_shuttle_trail.append(spos)

                pip_frame = render_pip_court_frame(
                    court_bg, pip_player_trails, pip_shuttle_trail
                )
                pip_resized = cv2.resize(
                    pip_frame, (pip_w, pip_h), interpolation=cv2.INTER_AREA
                )

                cv2.rectangle(
                    frame,
                    (pip_x - 1, pip_y - 1),
                    (pip_x + pip_w, pip_y + pip_h),
                    (255, 255, 255),
                    1,
                )
                frame[pip_y : pip_y + pip_h, pip_x : pip_x + pip_w] = pip_resized

            writer.write(frame)
            frame_idx += 1

    return out_path
