"""Rally segmentation via PySceneDetect ContentDetector + motion-energy filter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_splitter import split_video_ffmpeg

from rallylens.common import ensure_dir, get_logger, require_ffmpeg

_log = get_logger(__name__)


@dataclass(frozen=True)
class RallyClip:
    index: int
    start_s: float
    end_s: float
    path: Path


def _mean_motion_energy(video_path: Path, start_s: float, end_s: float, samples: int = 8) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = int(start_s * fps)
        end_frame = max(start_frame + samples + 1, int(end_s * fps))
        frame_idxs = np.linspace(start_frame, end_frame - 1, samples, dtype=int)
        prev_gray: np.ndarray | None = None
        diffs: list[float] = []
        for fi in frame_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diffs.append(float(np.mean(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)))))
            prev_gray = gray
        return float(np.mean(diffs)) if diffs else 0.0
    finally:
        cap.release()


def segment_rallies(
    video_path: Path,
    out_dir: Path,
    threshold: float = 27.0,
    min_duration_s: float = 3.0,
    motion_energy_threshold: float = 2.0,
) -> list[RallyClip]:
    require_ffmpeg()
    ensure_dir(out_dir)

    if not video_path.exists():
        raise FileNotFoundError(video_path)

    _log.info("scanning %s for scene cuts (threshold=%.1f)", video_path.name, threshold)
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    raw_scenes: list[tuple[FrameTimecode, FrameTimecode]] = scene_manager.get_scene_list()
    _log.info("detected %d raw scenes", len(raw_scenes))

    kept: list[tuple[FrameTimecode, FrameTimecode]] = []
    for start, end in raw_scenes:
        start_s = start.get_seconds()
        end_s = end.get_seconds()
        if (end_s - start_s) < min_duration_s:
            continue
        energy = _mean_motion_energy(video_path, start_s, end_s)
        if energy < motion_energy_threshold:
            _log.info(
                "drop scene %.1f-%.1fs (motion energy %.2f below %.2f)",
                start_s,
                end_s,
                energy,
                motion_energy_threshold,
            )
            continue
        kept.append((start, end))
    _log.info("kept %d rallies after filtering", len(kept))

    if not kept:
        _write_manifest(out_dir, [])
        return []

    output_template = str(out_dir / "rally_$SCENE_NUMBER.mp4")
    split_video_ffmpeg(
        str(video_path),
        kept,
        output_file_template=output_template,
        show_progress=False,
        show_output=False,
    )

    clips: list[RallyClip] = []
    for idx, (start, end) in enumerate(kept, start=1):
        clip_path = out_dir / f"rally_{idx:03d}.mp4"
        if not clip_path.exists():
            alt = out_dir / f"rally_{idx:04d}.mp4"
            if alt.exists():
                clip_path = alt
        clips.append(
            RallyClip(
                index=idx,
                start_s=float(start.get_seconds()),
                end_s=float(end.get_seconds()),
                path=clip_path,
            )
        )

    _write_manifest(out_dir, clips)
    return clips


def _write_manifest(out_dir: Path, clips: list[RallyClip]) -> None:
    manifest_path = out_dir / "rallies.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "index": c.index,
                    "start_s": c.start_s,
                    "end_s": c.end_s,
                    "path": str(c.path),
                }
                for c in clips
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def load_manifest(out_dir: Path) -> list[RallyClip]:
    manifest_path = out_dir / "rallies.json"
    if not manifest_path.exists():
        return []
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        RallyClip(
            index=int(row["index"]),
            start_s=float(row["start_s"]),
            end_s=float(row["end_s"]),
            path=Path(row["path"]),
        )
        for row in data
    ]
