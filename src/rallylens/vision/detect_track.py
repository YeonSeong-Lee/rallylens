"""YOLO11-pose player detection with optional ByteTrack tracker."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from rallylens.common import MODELS_DIR, ensure_dir, get_logger

_log = get_logger(__name__)

TrackerName = Literal["bytetrack", "botsort", None]


def coerce_tracker_name(name: str | None) -> TrackerName:
    """Validate a CLI-supplied tracker name into a `TrackerName` literal."""
    if name is None or name in ("bytetrack", "botsort"):
        return name  # type: ignore[return-value]
    raise ValueError(f"unknown tracker: {name!r}")


class Detection(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_idx: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    keypoints_xy: list[tuple[float, float]]
    keypoints_conf: list[float]
    track_id: int | None = None


def _tracker_config_name(tracker: TrackerName) -> str | None:
    if tracker is None:
        return None
    if tracker == "bytetrack":
        return "bytetrack.yaml"
    if tracker == "botsort":
        return "botsort.yaml"
    raise ValueError(f"unknown tracker: {tracker}")


def detect_and_track_players(
    clip_path: Path,
    weights: str = "yolo11n-pose.pt",
    conf: float = 0.25,
    max_det: int = 4,
    tracker: TrackerName = None,
) -> list[Detection]:
    """Run YOLO11-pose on every frame; return flat list of per-frame detections.

    When `tracker` is None (Week 1 behavior): uses `model.predict()`, no track IDs.
    When `tracker="bytetrack"` (Week 2): uses `model.track(tracker="bytetrack.yaml")`,
    which assigns stable `track_id`s to each detection via ultralytics' built-in
    ByteTrack implementation. Used to keep singles players' IDs consistent across
    missed detections.

    First call downloads the weight file into MODELS_DIR, not the project root.
    """
    if not clip_path.exists():
        raise FileNotFoundError(clip_path)

    ensure_dir(MODELS_DIR)
    from ultralytics import YOLO

    weight_path = MODELS_DIR / weights
    with contextlib.chdir(MODELS_DIR):
        model = YOLO(str(weight_path) if weight_path.exists() else weights)

    tracker_cfg = _tracker_config_name(tracker)
    _log.info(
        "running %s on %s (tracker=%s)",
        weights,
        clip_path.name,
        tracker_cfg or "none",
    )

    common_kwargs = dict(
        source=str(clip_path),
        stream=True,
        conf=conf,
        max_det=max_det,
        classes=[0],
        verbose=False,
    )
    if tracker_cfg is None:
        results = model.predict(**common_kwargs)
    else:
        results = model.track(tracker=tracker_cfg, persist=True, **common_kwargs)

    detections: list[Detection] = []
    frames_seen = 0
    for frame_idx, result in enumerate(results):
        frames_seen = frame_idx + 1
        boxes = result.boxes
        kps = result.keypoints
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = (
            boxes.id.cpu().numpy().astype(int).tolist()
            if getattr(boxes, "id", None) is not None
            else None
        )

        if kps is not None and kps.xy is not None:
            kp_xy = kps.xy.cpu().numpy()
            kp_conf = kps.conf.cpu().numpy() if kps.conf is not None else None
        else:
            kp_xy = None
            kp_conf = None

        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            if kp_xy is not None:
                points = [(float(p[0]), float(p[1])) for p in kp_xy[i]]
                scores = (
                    [float(s) for s in kp_conf[i]]
                    if kp_conf is not None
                    else [1.0] * len(points)
                )
            else:
                points = []
                scores = []
            detections.append(
                Detection(
                    frame_idx=frame_idx,
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(confs[i]),
                    keypoints_xy=points,
                    keypoints_conf=scores,
                    track_id=int(ids[i]) if ids is not None and i < len(ids) else None,
                )
            )

    _log.info(
        "collected %d detections over %d frames (with_track_ids=%s)",
        len(detections),
        frames_seen,
        tracker_cfg is not None,
    )
    return detections
