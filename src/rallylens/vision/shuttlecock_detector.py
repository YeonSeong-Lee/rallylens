"""Fine-tuned YOLO11 shuttlecock detector wrapper.

This module is intended to be used with a fine-tuned weight file at
`models/shuttle_best.pt`. If that file is missing (e.g. before Week 2
fine-tuning has been run on Colab), the detector falls back to the pretrained
`yolo11n.pt` and emits a warning — the downstream tracker still runs, but
precision is garbage until fine-tuning completes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rallylens.common import MODELS_DIR, ensure_dir, get_logger

_log = get_logger(__name__)

FINE_TUNED_WEIGHTS = "shuttle_best.pt"
PRETRAINED_FALLBACK = "yolo11n.pt"
SHUTTLE_CLASS_ID = 0  # single-class fine-tuning convention


@dataclass(frozen=True)
class ShuttleDetection:
    frame_idx: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float


def resolve_weights(weights: str | None = None) -> tuple[Path, bool]:
    """Return (weights_path, is_fine_tuned).

    Caller order of precedence:
        1. explicit `weights` argument
        2. `RALLYLENS_SHUTTLE_WEIGHTS` environment variable
        3. `models/shuttle_best.pt` on disk
        4. pretrained yolo11n.pt fallback (emits warning, not fine-tuned)
    """
    ensure_dir(MODELS_DIR)
    if weights is not None:
        return Path(weights), True
    env = os.environ.get("RALLYLENS_SHUTTLE_WEIGHTS")
    if env:
        return Path(env), True
    candidate = MODELS_DIR / FINE_TUNED_WEIGHTS
    if candidate.exists():
        return candidate, True
    _log.warning(
        "no fine-tuned shuttle weights at %s — falling back to pretrained %s "
        "(shuttle precision will be very poor until Week 2 fine-tuning is run)",
        candidate,
        PRETRAINED_FALLBACK,
    )
    return Path(PRETRAINED_FALLBACK), False


def detect_shuttlecocks(
    clip_path: Path,
    weights: str | None = None,
    conf: float = 0.15,  # lower than player detector — shuttles are small/dim
    imgsz: int = 1280,   # higher resolution helps small-object recall
) -> list[ShuttleDetection]:
    if not clip_path.exists():
        raise FileNotFoundError(clip_path)

    from ultralytics import YOLO  # local import keeps module cheap to import

    weight_path, is_fine_tuned = resolve_weights(weights)

    prev_cwd = Path.cwd()
    os.chdir(MODELS_DIR)
    try:
        model = YOLO(str(weight_path))
    finally:
        os.chdir(prev_cwd)

    target_class = SHUTTLE_CLASS_ID if is_fine_tuned else None
    _log.info(
        "running %s (fine_tuned=%s) on %s at imgsz=%d",
        weight_path.name,
        is_fine_tuned,
        clip_path.name,
        imgsz,
    )
    results = model.predict(
        source=str(clip_path),
        stream=True,
        conf=conf,
        imgsz=imgsz,
        classes=[target_class] if target_class is not None else None,
        verbose=False,
    )

    detections: list[ShuttleDetection] = []
    for frame_idx, result in enumerate(results):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            detections.append(
                ShuttleDetection(
                    frame_idx=frame_idx,
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(confs[i]),
                )
            )
    _log.info("collected %d shuttle candidates", len(detections))
    return detections
