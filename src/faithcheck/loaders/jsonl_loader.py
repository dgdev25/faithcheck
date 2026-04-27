"""JSONL task suite loader with streaming support."""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from faithcheck.models import TaskItem

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class JsonlLoader:
    """Load an entire JSONL task suite into memory."""

    @staticmethod
    def load(path: Path) -> list[TaskItem]:
        if not path.exists():
            raise FileNotFoundError(f"Task suite not found: {path}")
        items: list[TaskItem] = []
        with open(path) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e
                try:
                    items.append(TaskItem.model_validate(data))
                except Exception as e:
                    raise ValueError(f"Validation error at line {line_num}: {e}") from e
        return items

    @staticmethod
    def compute_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


class StreamingJsonlLoader:
    """Stream JSONL items one at a time without loading all into memory."""

    @staticmethod
    def stream(path: Path) -> Iterator[TaskItem]:
        if not path.exists():
            raise FileNotFoundError(f"Task suite not found: {path}")
        with open(path) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e
                yield TaskItem.model_validate(data)
