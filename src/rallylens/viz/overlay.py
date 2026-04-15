"""OpenCV overlay renderer: draw player bbox + COCO-17 pose keypoints per frame.

Week 2 adds optional shuttlecock trajectory rendering (fading polyline tail +
per-frame dot, dashed when interpolated).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2

from rallylens.common import ensure_dir, get_logger
from rallylens.vision.detect_track import Detection
from rallylens.vision.kalman_tracker import ShuttleTrackPoint

_log = get_logger(__name__)

# COCO-17 skeleton edges (pairs of keypoint indices).
_COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (5, 6),   # shoulders
    (5, 7), (7, 9),    # left arm
    (6, 8), (8, 10),   # right arm
    (5, 11), (6, 12),  # torso sides
    (11, 12),          # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
    (0, 1), (0, 2),    # nose-eyes
    (1, 3), (2, 4),    # eyes-ears
)


def render_overlay(
    clip_path: Path,
    detections: list[Detection],
    out_path: Path,
    shuttle_track: list[ShuttleTrackPoint] | None = None,
    box_color: tuple[int, int, int] = (0, 255, 0),
    kp_color: tuple[int, int, int] = (0, 165, 255),
    edge_color: tuple[int, int, int] = (255, 255, 0),
    shuttle_color: tuple[int, int, int] = (0, 0, 255),
    shuttle_interp_color: tuple[int, int, int] = (80, 80, 255),
    shuttle_tail_len: int = 15,
    kp_min_conf: float = 0.3,
) -> Path:
    if not clip_path.exists():
        raise FileNotFoundError(clip_path)
    ensure_dir(out_path.parent)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open clip: {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"could not open VideoWriter for {out_path}")

    dets_by_frame: dict[int, list[Detection]] = defaultdict(list)
    for d in detections:
        dets_by_frame[d.frame_idx].append(d)

    shuttle_by_frame: dict[int, ShuttleTrackPoint] = {}
    sorted_shuttle: list[ShuttleTrackPoint] = []
    if shuttle_track:
        sorted_shuttle = sorted(shuttle_track, key=lambda p: p.frame_idx)
        shuttle_by_frame = {p.frame_idx: p for p in sorted_shuttle}

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            for det in dets_by_frame.get(frame_idx, []):
                x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label = (
                    f"P{det.track_id} {det.confidence:.2f}"
                    if det.track_id is not None
                    else f"{det.confidence:.2f}"
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )

                points = det.keypoints_xy
                scores = det.keypoints_conf
                for (px, py), score in zip(points, scores, strict=False):
                    if score < kp_min_conf:
                        continue
                    cv2.circle(frame, (int(px), int(py)), 3, kp_color, -1)

                for a, b in _COCO_SKELETON:
                    if a >= len(points) or b >= len(points):
                        continue
                    if scores[a] < kp_min_conf or scores[b] < kp_min_conf:
                        continue
                    cv2.line(
                        frame,
                        (int(points[a][0]), int(points[a][1])),
                        (int(points[b][0]), int(points[b][1])),
                        edge_color,
                        1,
                        cv2.LINE_AA,
                    )

            if sorted_shuttle:
                tail = [
                    p
                    for p in sorted_shuttle
                    if frame_idx - shuttle_tail_len <= p.frame_idx <= frame_idx
                ]
                for prev, curr in zip(tail, tail[1:], strict=False):
                    color = (
                        shuttle_interp_color
                        if prev.interpolated or curr.interpolated
                        else shuttle_color
                    )
                    cv2.line(
                        frame,
                        (int(prev.x), int(prev.y)),
                        (int(curr.x), int(curr.y)),
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                current = shuttle_by_frame.get(frame_idx)
                if current is not None:
                    cv2.circle(
                        frame,
                        (int(current.x), int(current.y)),
                        5,
                        shuttle_color if not current.interpolated else shuttle_interp_color,
                        -1,
                    )

            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    _log.info("wrote overlay %s (%d frames)", out_path.name, frame_idx)
    return out_path
