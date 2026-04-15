"""Generic pydantic JSON / JSONL I/O helpers.

All pipeline artifacts on disk are pydantic `BaseModel` instances, so
reading/writing them is a one-liner via this module. Consolidates
`ensure_dir`, UTF-8 encoding, and missing-file handling.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from rallylens.common import ensure_dir

T = TypeVar("T", bound=BaseModel)


def save_json(item: BaseModel, path: Path, *, indent: int = 2) -> None:
    """Write a single pydantic model to `path` as pretty-printed JSON."""
    ensure_dir(path.parent)
    path.write_text(item.model_dump_json(indent=indent), encoding="utf-8")


def load_json(path: Path, model: type[T]) -> T:
    """Load a single pydantic model from `path`. Raises if the file is missing."""
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def save_jsonl(items: Iterable[BaseModel], path: Path) -> None:
    """Write an iterable of pydantic models to `path` as one object per line."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def load_jsonl(path: Path, model: type[T]) -> list[T]:
    """Load a list of pydantic models from a JSONL file. Returns [] if missing."""
    if not path.exists():
        return []
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
