"""The naive comparison: a sliding window that keeps the newest messages that fit the budget.

This is what many agent frameworks do by default. It keeps the system prompt, then walks back from the newest
message; the first message that does not fit is cut down to its last part, and everything older is dropped. A
kept tool result always keeps the assistant message that called it, so the request stays valid.
"""

from __future__ import annotations

import copy
from typing import Any

from ..messages import text_of, with_text
from ..tokens import _encoding, count_message, count_messages


def tail_tokens(text: str, n: int) -> str:
    ids = _encoding().encode(text, disallowed_special=())
    return _encoding().decode(ids[-n:]) if n > 0 else ""


def sliding_window(messages: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    msgs = [copy.deepcopy(m) for m in messages]
    system = [i for i, m in enumerate(msgs) if m.get("role") == "system"]
    used = sum(count_message(msgs[i]) for i in system) + 3
    keep: set[int] = set(system)
    for i in range(len(msgs) - 1, -1, -1):
        if i in keep:
            continue
        cost = count_message(msgs[i])
        if used + cost <= budget:
            keep.add(i)
            used += cost
            continue
        room = budget - used - 8
        if room > 50 and msgs[i].get("role") != "assistant":
            msgs[i] = with_text(msgs[i], "[...]" + tail_tokens(text_of(msgs[i]), room))
            keep.add(i)
        break
    # keep the assistant message that made each kept tool call
    for i in sorted(keep):
        if msgs[i].get("role") == "tool":
            for j in range(i - 1, -1, -1):
                if any(c.get("id") == msgs[i].get("tool_call_id") for c in msgs[j].get("tool_calls") or []):
                    keep.add(j)
                    break
    # the window starts at a user message, as chat APIs expect after the system prompt
    first = min((i for i in keep if i not in system), default=None)
    if first is not None and msgs[first].get("role") != "user":
        for j in range(first - 1, -1, -1):
            if msgs[j].get("role") == "user":
                keep.add(j)
                break
    return [m for i, m in enumerate(msgs) if i in keep]


def window_tokens(messages: list[dict[str, Any]], budget: int) -> int:
    return count_messages(sliding_window(messages, budget))
