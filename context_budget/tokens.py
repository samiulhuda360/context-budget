"""Token counting.

Counts use the `o200k_base` encoding from tiktoken (the tokenizer of current OpenAI models). Other providers
tokenize differently, so the counts are a consistent yardstick rather than an exact bill; the evaluation also
records the prompt tokens that Gemini itself reports, which agree with these counts to within a few percent.
"""

from __future__ import annotations

import functools
import json
from typing import Any

import tiktoken

# Per-message framing overhead used by OpenAI chat models (role markers and separators).
MESSAGE_OVERHEAD = 4


@functools.lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


@functools.lru_cache(maxsize=65536)
def count_text(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text, disallowed_special=()))


def count_message(message: dict[str, Any]) -> int:
    from .messages import text_of

    total = MESSAGE_OVERHEAD + count_text(text_of(message))
    calls = message.get("tool_calls")
    if calls:
        total += count_text(json.dumps(calls, ensure_ascii=False, sort_keys=True))
    return total


def count_messages(messages: list[dict[str, Any]]) -> int:
    return sum(count_message(m) for m in messages) + 3
