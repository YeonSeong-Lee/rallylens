"""Round-trip tests for the pydantic-based serialization helpers.

These assert that every pipeline artifact can survive a save -> load cycle
without losing data, and that JSONL files are ignored cleanly when absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from rallylens.analysis.events import HitEvent, RallyStats
from rallylens.domain.video import VideoMeta
from rallylens.preprocess.rally_segmenter import RallyClip
from rallylens.serialization import load_json, load_jsonl, save_json, save_jsonl
from rallylens.vision.court_homography import CourtHomography
from rallylens.vision.detect_track import Detection
from rallylens.vision.kalman_tracker import ShuttleTrackPoint


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


def test_save_load_jsonl_track_points_preserves_nan(tmp_path: Path):
    track = [
        ShuttleTrackPoint(
            frame_idx=0, x=1.0, y=2.0, vx=3.0, vy=4.0, interpolated=False, residual=0.5
        ),
        ShuttleTrackPoint(
            frame_idx=1,
            x=2.0,
            y=3.0,
            vx=3.0,
            vy=4.0,
            interpolated=True,
            residual=float("nan"),
        ),
    ]
    path = tmp_path / "track.jsonl"
    save_jsonl(track, path)
    loaded = load_jsonl(path, ShuttleTrackPoint)
    assert len(loaded) == 2
    assert loaded[0] == track[0]
    assert loaded[1].interpolated is True
    assert math.isnan(loaded[1].residual)


def test_load_jsonl_missing_returns_empty(tmp_path: Path):
    assert load_jsonl(tmp_path / "nope.jsonl", HitEvent) == []


def test_save_load_json_rally_stats_with_events(tmp_path: Path):
    events = [
        HitEvent(
            frame_idx=10,
            time_s=0.33,
            kind="hit",
            position_xy=(100.0, 200.0),
            velocity_xy=(5.0, -3.0),
            signals=("velocity_reversal",),
            player_side="top",
        )
    ]
    stats = RallyStats(
        video_id="v",
        rally_index=1,
        duration_s=4.5,
        total_frames=135,
        shot_count=1,
        first_shot_frame=10,
        last_shot_frame=10,
        avg_inter_shot_gap_s=None,
        top_side_shots=1,
        bottom_side_shots=0,
        events=events,
    )
    path = tmp_path / "stats.json"
    save_json(stats, path)
    loaded = load_json(path, RallyStats)
    assert loaded == stats
    assert loaded.events[0].signals == ("velocity_reversal",)


def test_save_load_json_court_homography_preserves_numpy(tmp_path: Path):
    image_points = [(100.0, 50.0), (500.0, 50.0), (500.0, 350.0), (100.0, 350.0)]
    h = CourtHomography(
        image_points=image_points,
        court_points_m=[(0.0, 0.0), (5.18, 0.0), (5.18, 13.4), (0.0, 13.4)],
        H=np.eye(3),
    )
    path = tmp_path / "h.json"
    save_json(h, path)
    loaded = load_json(path, CourtHomography)
    assert np.allclose(loaded.H, h.H)
    assert loaded.image_points == h.image_points


def test_save_load_json_rally_clip_with_path(tmp_path: Path):
    clip = RallyClip(index=1, start_s=0.0, end_s=5.0, path=tmp_path / "rally_001.mp4")
    path = tmp_path / "clip.json"
    save_json(clip, path)
    loaded = load_json(path, RallyClip)
    assert loaded == clip


def test_save_jsonl_empty_list(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    save_jsonl([], path)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""
    assert load_jsonl(path, HitEvent) == []


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
