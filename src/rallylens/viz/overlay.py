"""Video overlay renderer.

Draws player bounding boxes, COCO-17 skeletons, and a fading shuttle trail
onto each frame of the source video and writes the result as an MP4.
"""

from __future__ import annotations

import collections
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from rallylens.common import ensure_dir, read_video_properties
from rallylens.vision.detect_track import Detection
from rallylens.vision.shuttle_tracker import ShuttlePoint
from rallylens.viz._utils import track_color

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


def _draw_bbox(frame: np.ndarray, det: Detection, thickness: int) -> None:  # type: ignore[type-arg]
    x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
    color = track_color(det.track_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"id={det.track_id}" if det.track_id is not None else f"{det.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    lbl_y = max(y1, th + 4)
    cv2.rectangle(frame, (x1, lbl_y - th - 4), (x1 + tw + 4, lbl_y), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, lbl_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )


def _draw_skeleton(
    frame: np.ndarray,  # type: ignore[type-arg]
    det: Detection,
    kp_conf_thresh: float,
    thickness: int,
) -> None:
    kps = det.keypoints_xy
    confs = det.keypoints_conf
    color = track_color(det.track_id)
    for i, (kx, ky) in enumerate(kps):
        if i < len(confs) and confs[i] > kp_conf_thresh:
            cv2.circle(frame, (int(kx), int(ky)), 3, color, -1, cv2.LINE_AA)
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
    n = len(trail)
    if n == 0:
        return
    for i, pt in enumerate(trail):
        alpha = (i + 1) / n
        radius = max(2, int(8 * alpha))
        intensity = int(255 * alpha)
        color = (0, intensity, intensity)  # fades dark → yellow
        cv2.circle(frame, (pt.x, pt.y), radius, color, -1, cv2.LINE_AA)


def render_overlay_video(
    video_path: Path,
    detections: list[Detection],
    shuttle_track: list[ShuttlePoint],
    out_path: Path,
    *,
    trail_len: int = 30,
    kp_conf_thresh: float = 0.3,
    bbox_thickness: int = 2,
    skeleton_thickness: int = 2,
    fourcc: str = "mp4v",
) -> Path:
    """Read source video frame-by-frame, draw overlays, and write MP4.

    Returns out_path on success.
    """
    props = read_video_properties(video_path)
    ensure_dir(out_path.parent)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*fourcc),
        props.fps,
        (props.width, props.height),
    )

    detections_by_frame: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        detections_by_frame[det.frame_idx].append(det)

    shuttle_by_frame: dict[int, ShuttlePoint] = {pt.frame_idx: pt for pt in shuttle_track}

    cap = cv2.VideoCapture(str(video_path))
    trail: collections.deque[ShuttlePoint] = collections.deque(maxlen=trail_len)
    frame_idx = 0

    try:
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

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    return out_path
