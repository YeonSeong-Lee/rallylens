"""Round-trip tests for the pydantic-based serialization helpers."""

from __future__ import annotations

from pathlib import Path

from rallylens.domain.video import VideoMeta
from rallylens.serialization import load_json, load_jsonl, save_json, save_jsonl
from rallylens.vision.detect_track import Detection


def test_save_load_json_video_meta(tmp_path: Path):
    meta = VideoMeta(
        video_id="abc",
        title="t",
        upload_date=None,
        duration_s=1.5,
        url="https://youtu.be/abc",
        source_path=tmp_path / "abc.mp4",
    )
    path = tmp_path / "meta.json"
    save_json(meta, path)
    assert path.exists()
    loaded = load_json(path, VideoMeta)
    assert loaded == meta


def test_load_jsonl_missing_returns_empty(tmp_path: Path):
    assert load_jsonl(tmp_path / "nope.jsonl", Detection) == []


def test_save_jsonl_empty_list(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    save_jsonl([], path)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""
    assert load_jsonl(path, Detection) == []


def test_save_json_creates_missing_parent(tmp_path: Path):
    nested = tmp_path / "deep" / "nest" / "meta.json"
    meta = VideoMeta(
        video_id="a",
        title="t",
        upload_date=None,
        duration_s=1.0,
        url="u",
        source_path=tmp_path / "a.mp4",
    )
    save_json(meta, nested)
    assert nested.exists()
    assert load_json(nested, VideoMeta) == meta


def test_detection_save_load_jsonl_roundtrip(tmp_path: Path):
    dets = [
        Detection(
            frame_idx=0,
            bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
            confidence=0.9,
            keypoints_xy=[(15.0, 25.0), (18.0, 28.0)],
            keypoints_conf=[0.8, 0.7],
            track_id=1,
        )
    ]
    path = tmp_path / "dets.jsonl"
    save_jsonl(dets, path)
    loaded = load_jsonl(path, Detection)
    assert loaded == dets
