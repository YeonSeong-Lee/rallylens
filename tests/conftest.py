"""Shared pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("RALLYLENS_LOG_LEVEL", "CRITICAL")

_DIR_NAMES = (
    "DATA_DIR",
    "RAW_DIR",
    "DETECTIONS_DIR",
    "TRACKS_DIR",
    "CALIBRATION_DIR",
    "REPORTS_DIR",
)


@pytest.fixture
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every module's DATA_DIR-derived constants into tmp_path."""
    import rallylens.cli
    import rallylens.common
    import rallylens.ingest.downloader
    import rallylens.pipeline.io as pipeline_io
    import rallylens.pipeline.orchestrator as pipeline_orchestrator

    data = tmp_path / "data"
    data.mkdir()
    paths = {
        "DATA_DIR": data,
        "RAW_DIR": data / "raw",
        "DETECTIONS_DIR": data / "detections",
        "TRACKS_DIR": data / "tracks",
        "CALIBRATION_DIR": data / "calibration",
        "REPORTS_DIR": data / "reports",
    }

    targets = (
        rallylens.common,
        rallylens.cli,
        rallylens.ingest.downloader,
        pipeline_io,
        pipeline_orchestrator,
    )
    for name in _DIR_NAMES:
        new = paths[name]
        for mod in targets:
            if hasattr(mod, name):
                monkeypatch.setattr(mod, name, new)

    return data
