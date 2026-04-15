from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rallylens.domain.video import VideoMeta
from rallylens.ingest import downloader


def _fake_info(video_id: str = "dQw4w9WgXcQ") -> dict:
    return {
        "id": video_id,
        "title": "Sample Match Highlights",
        "upload_date": "20240115",
        "duration": 123.5,
    }


def test_download_video_calls_extract_info(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    fake_ydl = MagicMock()
    fake_ydl.__enter__.return_value = fake_ydl
    fake_ydl.__exit__.return_value = False
    info = _fake_info()
    fake_ydl.extract_info.return_value = info

    captured_opts: dict = {}

    def fake_constructor(opts):
        captured_opts.update(opts)
        return fake_ydl

    monkeypatch.setattr(downloader, "YoutubeDL", fake_constructor)

    video_path = tmp_path / f"{info['id']}.mp4"

    def fake_extract(url, download):
        assert download is True
        video_path.write_bytes(b"fake mp4 bytes")
        return info

    fake_ydl.extract_info.side_effect = fake_extract

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    meta = downloader.download_video(url, out_dir=tmp_path)

    assert isinstance(meta, VideoMeta)
    assert meta.video_id == "dQw4w9WgXcQ"
    assert meta.title == "Sample Match Highlights"
    assert meta.upload_date == "20240115"
    assert meta.duration_s == pytest.approx(123.5)
    assert meta.source_path == video_path
    assert meta.url == url

    assert "height<=720" in captured_opts["format"]
    assert captured_opts["merge_output_format"] == "mp4"
    assert captured_opts["quiet"] is True

    sidecar = tmp_path / f"{info['id']}.meta.json"
    assert sidecar.exists()


def test_download_video_cache_hit_skips_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    video_id = "dQw4w9WgXcQ"
    (tmp_path / f"{video_id}.mp4").write_bytes(b"fake mp4")
    meta = VideoMeta(
        video_id=video_id,
        title="Cached Title",
        upload_date="20240101",
        duration_s=60.0,
        url=f"https://www.youtube.com/watch?v={video_id}",
        source_path=tmp_path / f"{video_id}.mp4",
    )
    (tmp_path / f"{video_id}.meta.json").write_text(
        meta.model_dump_json(), encoding="utf-8"
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("YoutubeDL must not be constructed on cache hit")

    monkeypatch.setattr(downloader, "YoutubeDL", boom)

    result = downloader.download_video(meta.url, out_dir=tmp_path)
    assert result.video_id == video_id
    assert result.title == "Cached Title"
    assert result.duration_s == 60.0
