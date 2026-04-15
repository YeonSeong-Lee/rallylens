"""Frame sampler: extract N uniformly-spaced frames per rally clip for labeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rallylens.common import ensure_dir, get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class SampledFrame:
    clip_path: Path
    frame_idx: int
    image_path: Path


def sample_frames_from_clip(
    clip_path: Path,
    out_dir: Path,
    num_frames: int = 20,
    jpeg_quality: int = 92,
) -> list[SampledFrame]:
    """Extract `num_frames` uniformly-spaced frames and write them as jpeg."""
    if not clip_path.exists():
        raise FileNotFoundError(clip_path)
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open clip: {clip_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    pick = max(1, min(num_frames, total))
    indices = np.linspace(0, total - 1, pick, dtype=int)

    stem = clip_path.stem
    results: list[SampledFrame] = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        image_path = out_dir / f"{stem}_f{int(fi):06d}.jpg"
        cv2.imwrite(
            str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        results.append(SampledFrame(clip_path=clip_path, frame_idx=int(fi), image_path=image_path))
    cap.release()
    _log.info("sampled %d frames from %s -> %s", len(results), clip_path.name, out_dir)
    return results


def sample_frames_from_rallies(
    rally_dir: Path,
    out_dir: Path,
    per_clip: int = 20,
    total_budget: int | None = 400,
) -> list[SampledFrame]:
    """Sample across all rally_*.mp4 under `rally_dir`, optionally capping total count."""
    clips = sorted(rally_dir.glob("rally_*.mp4"))
    if not clips:
        return []

    if total_budget is not None:
        per_clip = max(1, min(per_clip, total_budget // max(1, len(clips))))

    ensure_dir(out_dir)
    all_frames: list[SampledFrame] = []
    for clip in clips:
        if total_budget is not None and len(all_frames) >= total_budget:
            break
        remaining = (
            total_budget - len(all_frames)
            if total_budget is not None
            else per_clip
        )
        take = min(per_clip, remaining) if total_budget is not None else per_clip
        if take <= 0:
            break
        all_frames.extend(sample_frames_from_clip(clip, out_dir, num_frames=take))

    _log.info("sampled %d frames total across %d clips", len(all_frames), len(clips))
    return all_frames
