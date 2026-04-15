"""CLI integration tests using Click's CliRunner."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from rallylens.cli import cli

pytestmark = pytest.mark.usefixtures("isolated_data")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def video_id() -> str:
    return "testvid_0001"


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


def test_detect_cmd_missing_file(runner: CliRunner):
    result = runner.invoke(cli, ["detect", "/nonexistent/path/to/video.mp4"])
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_detect_cmd_happy_path(
    runner: CliRunner, isolated_data: Path, monkeypatch: pytest.MonkeyPatch
):
    video_path = isolated_data / "raw" / "match.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()

    monkeypatch.setattr(
        "rallylens.cli.detect_and_track_players",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "rallylens.cli.save_player_detections",
        lambda *_args, **_kwargs: isolated_data / "detections" / "match" / "match_players.jsonl",
    )

    result = runner.invoke(cli, ["detect", str(video_path)])
    assert result.exit_code == 0, result.output
    assert "detections:" in result.output


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def test_run_cmd_local_file_happy_path(
    runner: CliRunner, isolated_data: Path, monkeypatch: pytest.MonkeyPatch, video_id: str
):
    """run command works directly on a local video file without downloading or splitting."""
    video_path = isolated_data / "raw" / f"{video_id}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()

    monkeypatch.setattr(
        "rallylens.pipeline.orchestrator.detect_and_track_players",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "rallylens.pipeline.orchestrator.save_player_detections",
        lambda *_args, **_kwargs: isolated_data / "detections" / video_id / f"{video_id}_players.jsonl",
    )

    result = runner.invoke(cli, ["run", str(video_path)])
    assert result.exit_code == 0, result.output
    assert "detections saved" in result.output
