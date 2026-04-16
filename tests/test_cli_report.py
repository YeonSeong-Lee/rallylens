"""CLI tests for the `rallylens report` subcommand.

The real `run_report_pipeline` is monkeypatched with a fake so these tests
exercise the click wiring, argument parsing, error handling, and output
formatting — not the actual Vertex AI call or CV work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from rallylens.cli import cli

pytestmark = pytest.mark.usefixtures("isolated_data")


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@dataclass
class _FakeResult:
    video_id: str
    metrics_path: Path
    report_json_path: Path | None
    report_md_path: Path | None
    court_diagram_path: Path | None
    heatmap_path: Path | None


def _make_video(isolated_data: Path, stem: str = "match") -> Path:
    video_path = isolated_data / "raw" / f"{stem}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()
    return video_path


def test_report_cmd_missing_video(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["report", "/nonexistent/video.mp4"])
    assert result.exit_code != 0
    assert "does not exist" in result.output


def test_report_cmd_metrics_only_happy_path(
    runner: CliRunner,
    isolated_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _make_video(isolated_data)
    captured: dict[str, Any] = {}

    def fake_run(video_path_arg: Path, **kwargs: Any) -> _FakeResult:
        captured["video_path"] = video_path_arg
        captured["kwargs"] = kwargs
        return _FakeResult(
            video_id=video_path_arg.stem,
            metrics_path=isolated_data / "reports" / video_path_arg.stem / "metrics.json",
            report_json_path=None,
            report_md_path=None,
            court_diagram_path=isolated_data
            / "viz"
            / video_path_arg.stem
            / "court_diagram.gif",
            heatmap_path=isolated_data / "viz" / video_path_arg.stem / "heatmap.png",
        )

    monkeypatch.setattr("rallylens.pipeline.run_report_pipeline", fake_run)

    result = runner.invoke(cli, ["report", str(video_path), "--metrics-only"])
    assert result.exit_code == 0, result.output
    assert "metrics:" in result.output
    assert "gif:" in result.output
    assert "heatmap:" in result.output
    assert "report:" not in result.output
    assert captured["kwargs"]["metrics_only"] is True
    assert captured["kwargs"]["skip_viz"] is False


def test_report_cmd_skip_viz(
    runner: CliRunner,
    isolated_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _make_video(isolated_data)
    captured: dict[str, Any] = {}

    def fake_run(video_path_arg: Path, **kwargs: Any) -> _FakeResult:
        captured["kwargs"] = kwargs
        return _FakeResult(
            video_id=video_path_arg.stem,
            metrics_path=isolated_data / "reports" / video_path_arg.stem / "metrics.json",
            report_json_path=None,
            report_md_path=None,
            court_diagram_path=None,
            heatmap_path=None,
        )

    monkeypatch.setattr("rallylens.pipeline.run_report_pipeline", fake_run)

    result = runner.invoke(
        cli, ["report", str(video_path), "--metrics-only", "--skip-viz"]
    )
    assert result.exit_code == 0, result.output
    assert captured["kwargs"]["skip_viz"] is True
    assert "gif:" not in result.output
    assert "heatmap:" not in result.output


def test_report_cmd_full_report_happy_path(
    runner: CliRunner,
    isolated_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _make_video(isolated_data, stem="clip_001")

    def fake_run(video_path_arg: Path, **_kwargs: Any) -> _FakeResult:
        video_id = video_path_arg.stem
        return _FakeResult(
            video_id=video_id,
            metrics_path=isolated_data / "reports" / video_id / "metrics.json",
            report_json_path=isolated_data / "reports" / video_id / "report.json",
            report_md_path=isolated_data / "reports" / video_id / "report.md",
            court_diagram_path=isolated_data / "viz" / video_id / "court_diagram.gif",
            heatmap_path=isolated_data / "viz" / video_id / "heatmap.png",
        )

    monkeypatch.setattr("rallylens.pipeline.run_report_pipeline", fake_run)

    result = runner.invoke(cli, ["report", str(video_path)])
    assert result.exit_code == 0, result.output
    assert "metrics:" in result.output
    assert "gif:" in result.output
    assert "heatmap:" in result.output
    assert "report:" in result.output


def test_report_cmd_missing_credentials_shows_helpful_message(
    runner: CliRunner,
    isolated_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _make_video(isolated_data)

    def fake_run(*_args: Any, **_kwargs: Any) -> _FakeResult:
        raise RuntimeError(
            "GOOGLE_CLOUD_PROJECT is not set. Add it to .env (and set "
            "GOOGLE_CLOUD_LOCATION), then ensure Application Default "
            "Credentials are configured."
        )

    monkeypatch.setattr("rallylens.pipeline.run_report_pipeline", fake_run)

    result = runner.invoke(cli, ["report", str(video_path)])
    assert result.exit_code != 0
    assert "GOOGLE_CLOUD_PROJECT" in result.output


def test_report_cmd_missing_artifacts_shows_helpful_message(
    runner: CliRunner,
    isolated_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _make_video(isolated_data)

    def fake_run(*_args: Any, **_kwargs: Any) -> _FakeResult:
        raise FileNotFoundError(
            "player detections missing for 'match' — run `rallylens detect ...` first"
        )

    monkeypatch.setattr("rallylens.pipeline.run_report_pipeline", fake_run)

    result = runner.invoke(cli, ["report", str(video_path)])
    assert result.exit_code != 0
    assert "rallylens detect" in result.output


def test_report_cmd_missing_google_genai_shows_install_hint(
    runner: CliRunner,
    isolated_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When google-genai is not installed the user sees the install command,
    not a raw ModuleNotFoundError traceback."""
    from rallylens.pipeline.report import _GENAI_MISSING_HINT

    video_path = _make_video(isolated_data)

    def fake_run(*_args: Any, **_kwargs: Any) -> _FakeResult:
        raise RuntimeError(_GENAI_MISSING_HINT)

    monkeypatch.setattr("rallylens.pipeline.run_report_pipeline", fake_run)

    result = runner.invoke(cli, ["report", str(video_path)])
    assert result.exit_code != 0
    assert "uv sync --extra report" in result.output
    assert "--metrics-only" in result.output
