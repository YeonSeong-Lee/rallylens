from pathlib import Path

import cv2
import numpy as np

from rallylens.preprocess.frame_sampler import (
    sample_frames_from_clip,
    sample_frames_from_rallies,
)


def _make_synthetic_clip(path: Path, n_frames: int = 30) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (64, 48))
    for i in range(n_frames):
        frame = np.full((48, 64, 3), i * 8 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_sample_frames_from_clip_writes_jpegs(tmp_path: Path):
    clip = tmp_path / "rally_001.mp4"
    _make_synthetic_clip(clip, n_frames=30)
    out_dir = tmp_path / "sampled"
    frames = sample_frames_from_clip(clip, out_dir, num_frames=6)
    assert len(frames) == 6
    for f in frames:
        assert f.image_path.exists()
        assert f.image_path.suffix == ".jpg"


def test_sample_frames_from_rallies_respects_total_budget(tmp_path: Path):
    rally_dir = tmp_path / "rallies"
    rally_dir.mkdir()
    for i in range(1, 4):
        _make_synthetic_clip(rally_dir / f"rally_{i:03d}.mp4", n_frames=20)
    out_dir = tmp_path / "out"
    frames = sample_frames_from_rallies(
        rally_dir, out_dir, per_clip=10, total_budget=12
    )
    assert len(frames) <= 12
    assert len(frames) >= 9  # 3 clips * ~3 frames each (budget / 3 = 4 per clip, capped by budget)
