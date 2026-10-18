"""The benchmark: 80 long agent conversations, 10 in each of 8 families, rebuilt deterministically from seeds.

All people, companies, hosts and figures are invented. Nothing is downloaded, so there is no third-party licence
to record; the generated data is covered by the repository's MIT licence.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .families import FAMILIES

PER_FAMILY = 10


def conversation_json(conv: dict[str, Any]) -> str:
    return json.dumps(conv, ensure_ascii=False, indent=1) + "\n"


def build() -> list[dict[str, Any]]:
    out = []
    for f_index, family in enumerate(FAMILIES):
        for i in range(PER_FAMILY):
            rng = random.Random(f"ctxb-bench-v1/{f_index}/{i}")
            out.append(family(i, rng))
    return out


def digest(convs: list[dict[str, Any]]) -> dict[str, str]:
    return {c["id"]: hashlib.sha256(conversation_json(c).encode()).hexdigest()[:16] for c in convs}


def write(convs: list[dict[str, Any]], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for c in convs:
        (folder / f"{c['id']}.json").write_text(conversation_json(c), encoding="utf-8", newline="\n")


_CACHE: list[dict[str, Any]] = []


def load() -> list[dict[str, Any]]:
    """All benchmark conversations, built in memory (about a second)."""
    if not _CACHE:
        _CACHE.extend(build())
    return _CACHE
