"""Writes eval/offline.json, eval/model.json and the readable eval/RESULTS.md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..paths import EVAL


def save_json(name: str, data: dict[str, Any]) -> Path:
    EVAL.mkdir(parents=True, exist_ok=True)
    path = EVAL / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


def load_json(name: str) -> dict[str, Any] | None:
    path = EVAL / name
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
