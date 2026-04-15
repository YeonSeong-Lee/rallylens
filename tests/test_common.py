from pathlib import Path

import pytest

from rallylens import common


def test_video_id_from_url_watch():
    assert (
        common.video_id_from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s")
        == "dQw4w9WgXcQ"
    )


def test_video_id_from_url_short():
    assert common.video_id_from_url("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_video_id_from_url_shorts():
    assert (
        common.video_id_from_url("https://www.youtube.com/shorts/dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )


def test_video_id_from_url_fallback_is_deterministic():
    url = "https://example.com/not-a-youtube-url"
    got = common.video_id_from_url(url)
    assert len(got) == 11
    assert got == common.video_id_from_url(url)
    assert got != common.video_id_from_url("https://example.com/other")


def test_ensure_dir_idempotent(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c"
    assert not target.exists()
    assert common.ensure_dir(target) == target
    assert target.is_dir()
    assert common.ensure_dir(target) == target


def test_require_ffmpeg_missing_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(common.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        common.require_ffmpeg()


def test_require_ffmpeg_present(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(common.shutil, "which", lambda _name: "/usr/local/bin/ffmpeg")
    common.require_ffmpeg()
