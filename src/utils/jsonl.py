"""JSONL input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""

    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield records from a JSONL file one row at a time."""

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)
