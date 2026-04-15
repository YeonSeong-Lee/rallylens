"""CLI integration tests using Click's CliRunner.

These tests cover each subcommand's error paths and (where possible) happy
paths by stubbing out the expensive pipeline stages. YOLO, cv2 video I/O,
and Claude are never invoked — we verify Click wiring, argument handling,
and pipeline orchestration, not the underlying ML.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from rallylens.analysis.events import HitEvent, RallyStats
from rallylens.cli import cli
from rallylens.domain.video import VideoMeta, VideoProperties
from rallylens.preprocess.rally_segmenter import RallyClip
from rallylens.serialization import save_json, save_jsonl
from rallylens.vision.kalman_tracker import ShuttleTrackPoint

pytestmark = pytest.mark.usefixtures("isolated_data")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def video_id() -> str:
    return "testvid_0001"


def _write_manifest(rallies_dir: Path, video_id: str, clips: list[RallyClip]) -> Path:
    video_dir = rallies_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    from rallylens.preprocess.rally_segmenter import _RALLY_CLIP_LIST

    manifest_path = video_dir / "rallies.json"
    manifest_path.write_bytes(_RALLY_CLIP_LIST.dump_json(clips, indent=2))
    return manifest_path


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------


def test_segment_cmd_missing_video(runner: CliRunner, video_id: str):
    result = runner.invoke(cli, ["segment", video_id])
    assert result.exit_code != 0
    assert "no cached video" in result.output


# ---------------------------------------------------------------------------
# sample-frames
# ---------------------------------------------------------------------------


def test_sample_frames_cmd_missing_rally_dir(runner: CliRunner, video_id: str):
    result = runner.invoke(cli, ["sample-frames", video_id])
    assert result.exit_code != 0
    assert "no rally dir" in result.output


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def test_calibrate_cmd_missing_manifest(runner: CliRunner, video_id: str):
    result = runner.invoke(cli, ["calibrate", video_id])
    assert result.exit_code != 0
    assert "rallies.json" in result.output or "run `segment`" in result.output


def test_calibrate_cmd_missing_rally_index(
    runner: CliRunner, video_id: str, isolated_data: Path
):
    rally_path = isolated_data / "rallies" / video_id / "rally_001.mp4"
    rally_path.parent.mkdir(parents=True, exist_ok=True)
    rally_path.touch()
    _write_manifest(
        isolated_data / "rallies",
        video_id,
        [RallyClip(index=1, start_s=0.0, end_s=5.0, path=rally_path)],
    )
    result = runner.invoke(cli, ["calibrate", video_id, "--rally-index", "99"])
    assert result.exit_code != 0
    assert "not found" in result.output


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


def test_detect_cmd_missing_manifest(runner: CliRunner, video_id: str):
    result = runner.invoke(cli, ["detect", video_id])
    assert result.exit_code != 0
    assert "rallies.json" in result.output or "run `segment`" in result.output


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------


def test_events_cmd_missing_tracks(runner: CliRunner, video_id: str):
    result = runner.invoke(cli, ["events", video_id])
    assert result.exit_code != 0
    assert "no shuttle tracks" in result.output


def test_events_cmd_happy_path_with_stub_tracks(
    runner: CliRunner, video_id: str, isolated_data: Path, monkeypatch: pytest.MonkeyPatch
):
    """Events pipeline runs to completion when tracks exist on disk.

    cv2.VideoCapture is stubbed so we don't need a real MP4 file.
    """
    rally_path = isolated_data / "rallies" / video_id / "rally_001.mp4"
    rally_path.parent.mkdir(parents=True, exist_ok=True)
    rally_path.touch()
    _write_manifest(
        isolated_data / "rallies",
        video_id,
        [RallyClip(index=1, start_s=0.0, end_s=2.0, path=rally_path)],
    )

    track_path = isolated_data / "tracks" / video_id / "rally_001_shuttle.jsonl"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(
        [
            ShuttleTrackPoint(
                frame_idx=i,
                x=float(100 + i * 5),
                y=200.0,
                vx=5.0,
                vy=0.0,
                interpolated=False,
                residual=1.0,
            )
            for i in range(10)
        ],
        track_path,
    )

    monkeypatch.setattr(
        "rallylens.pipeline.events.read_video_properties",
        lambda _path: VideoProperties(fps=30.0, width=640, height=480, frame_count=60),
    )

    result = runner.invoke(cli, ["events", video_id])
    assert result.exit_code == 0, result.output
    assert "wrote events for 1 rallies" in result.output

    stats_path = isolated_data / "events" / video_id / "rally_001_stats.json"
    assert stats_path.exists()


# ---------------------------------------------------------------------------
# heatmaps
# ---------------------------------------------------------------------------


def test_heatmaps_cmd_nothing_to_render(runner: CliRunner, video_id: str):
    result = runner.invoke(cli, ["heatmaps", video_id])
    assert result.exit_code != 0
    assert "nothing to render" in result.output


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def test_report_cmd_missing_events_dir(runner: CliRunner, video_id: str):
    result = runner.invoke(cli, ["report", video_id])
    assert result.exit_code != 0
    assert "no events" in result.output


def test_report_cmd_missing_stats_files(
    runner: CliRunner, video_id: str, isolated_data: Path
):
    (isolated_data / "events" / video_id).mkdir(parents=True)
    result = runner.invoke(cli, ["report", video_id])
    assert result.exit_code != 0
    assert "no rally stats" in result.output


def test_report_cmd_missing_video_meta(
    runner: CliRunner, video_id: str, isolated_data: Path
):
    events_dir = isolated_data / "events" / video_id
    events_dir.mkdir(parents=True)
    stats = RallyStats(
        video_id=video_id,
        rally_index=1,
        duration_s=5.0,
        total_frames=150,
        shot_count=0,
        first_shot_frame=None,
        last_shot_frame=None,
        avg_inter_shot_gap_s=None,
        top_side_shots=0,
        bottom_side_shots=0,
        events=[],
    )
    save_json(stats, events_dir / "rally_001_stats.json")
    result = runner.invoke(cli, ["report", video_id])
    assert result.exit_code != 0
    assert "video_meta.json" in result.output


def test_report_cmd_happy_path_with_stubbed_claude(
    runner: CliRunner,
    video_id: str,
    isolated_data: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """report command runs to completion with a stubbed Claude client."""
    events_dir = isolated_data / "events" / video_id
    events_dir.mkdir(parents=True)

    meta = VideoMeta(
        video_id=video_id,
        title="Test Match",
        upload_date="20260101",
        duration_s=60.0,
        url=f"https://youtu.be/{video_id}",
        source_path=isolated_data / "raw" / f"{video_id}.mp4",
    )
    save_json(meta, events_dir / "video_meta.json")

    stats = RallyStats(
        video_id=video_id,
        rally_index=1,
        duration_s=5.0,
        total_frames=150,
        shot_count=1,
        first_shot_frame=10,
        last_shot_frame=10,
        avg_inter_shot_gap_s=None,
        top_side_shots=1,
        bottom_side_shots=0,
        events=[
            HitEvent(
                frame_idx=10,
                time_s=0.33,
                kind="hit",
                position_xy=(100.0, 200.0),
                velocity_xy=(5.0, 0.0),
                signals=("velocity_reversal",),
                player_side="top",
            )
        ],
    )
    save_json(stats, events_dir / "rally_001_stats.json")

    fake_response = MagicMock()
    fake_response.content = [MagicMock(type="text", text="# Test Report\n\n본문")]
    fake_response.stop_reason = "end_turn"
    fake_response.usage = MagicMock(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response
    monkeypatch.setattr("anthropic.Anthropic", lambda: fake_client)

    result = runner.invoke(cli, ["report", video_id])
    assert result.exit_code == 0, result.output
    report_path = isolated_data / "reports" / video_id / "match_report.md"
    assert report_path.exists()
    assert "# Test Report" in report_path.read_text()


# ---------------------------------------------------------------------------
# label-qa
# ---------------------------------------------------------------------------


def test_label_qa_cmd_missing_rally_index(runner: CliRunner, video_id: str):
    result = runner.invoke(cli, ["label-qa", video_id])
    assert result.exit_code != 0
    assert "rallies.json" in result.output


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_cmd_zero_rallies(
    runner: CliRunner, isolated_data: Path, monkeypatch: pytest.MonkeyPatch, video_id: str
):
    """End-to-end run command bails out cleanly when segmentation produces nothing."""
    meta = VideoMeta(
        video_id=video_id,
        title="Empty Match",
        upload_date=None,
        duration_s=1.0,
        url=f"https://youtu.be/{video_id}",
        source_path=isolated_data / "raw" / f"{video_id}.mp4",
    )
    monkeypatch.setattr(
        "rallylens.pipeline.orchestrator.download_video",
        lambda url, **kwargs: meta,
    )
    monkeypatch.setattr(
        "rallylens.pipeline.orchestrator.segment_rallies",
        lambda *_args, **_kwargs: [],
    )

    result = runner.invoke(cli, ["run", f"https://youtu.be/{video_id}"])
    assert result.exit_code != 0
    assert "no rallies detected" in result.output
