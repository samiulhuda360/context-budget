"""Strategy 3: remove repeated documents and messages.

Agents often paste the same file, page or tool result more than once. The latest copy is kept (it is the one
least likely to be cut later) and earlier copies become a one-line pointer. This works at two levels: whole
messages, and blocks (paragraphs separated by blank lines) of at least `min_chars` characters.
"""

from __future__ import annotations

import hashlib
import re

SPACE = re.compile(r"\s+")


def norm_hash(text: str) -> str:
    return hashlib.sha1(SPACE.sub(" ", text).strip().lower().encode()).hexdigest()  # noqa: S324 - not security


def blocks(text: str) -> list[str]:
    return re.split(r"\n\s*\n", text)


def pointer(text: str, where: int) -> str:
    gist = SPACE.sub(" ", text).strip()[:70]
    return f'[duplicate removed: the same text appears again in message {where}: "{gist}..."]'
