"""yt-dlp wrapper: downloads YouTube badminton videos at <=720p/30fps mp4."""

from __future__ import annotations

import json
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from rallylens.common import (
    RAW_DIR,
    VideoMeta,
    ensure_dir,
    get_logger,
    video_id_from_url,
)

_log = get_logger(__name__)

_FORMAT_SELECTOR = (
    "bv*[height<=720][fps<=30][ext=mp4]+ba[ext=m4a]/"
    "b[height<=720][ext=mp4]/"
    "b[height<=720]"
)


def _sidecar_path(out_dir: Path, video_id: str) -> Path:
    return out_dir / f"{video_id}.meta.json"


def _video_path(out_dir: Path, video_id: str) -> Path:
    return out_dir / f"{video_id}.mp4"


def _read_cached_meta(out_dir: Path, video_id: str) -> VideoMeta | None:
    sidecar = _sidecar_path(out_dir, video_id)
    video = _video_path(out_dir, video_id)
    if not (sidecar.exists() and video.exists()):
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return VideoMeta.from_json_dict(data)


def _write_sidecar(meta: VideoMeta) -> None:
    path = _sidecar_path(meta.source_path.parent, meta.video_id)
    path.write_text(
        json.dumps(meta.to_json_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def download_video(
    url: str,
    out_dir: Path = RAW_DIR,
    force: bool = False,
) -> VideoMeta:
    """Download a YouTube video at <=720p/30fps mp4 and return metadata.

    Cache: if `{out_dir}/{video_id}.mp4` and `{video_id}.meta.json` both exist
    and force=False, the sidecar is read and no network call is made.
    """
    ensure_dir(out_dir)
    video_id = video_id_from_url(url)

    if not force:
        cached = _read_cached_meta(out_dir, video_id)
        if cached is not None:
            _log.info("cache hit for %s (%s)", video_id, cached.title)
            return cached

    ydl_opts = {
        "format": _FORMAT_SELECTOR,
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": force,
    }

    _log.info("downloading %s with format=%s", url, _FORMAT_SELECTOR)
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except DownloadError:
            raise
    if info is None:
        raise RuntimeError(f"yt-dlp returned no info for {url}")

    resolved_id = str(info.get("id") or video_id)
    video_path = _video_path(out_dir, resolved_id)
    if not video_path.exists():
        raise RuntimeError(
            f"expected {video_path} to exist after download — no compatible format?"
        )

    meta = VideoMeta(
        video_id=resolved_id,
        title=str(info.get("title") or ""),
        upload_date=(str(info["upload_date"]) if info.get("upload_date") else None),
        duration_s=float(info.get("duration") or 0.0),
        url=url,
        source_path=video_path,
    )
    _write_sidecar(meta)
    _log.info("saved %s (%.1fs)", video_path.name, meta.duration_s)
    return meta
