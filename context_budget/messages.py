"""Helpers for OpenAI-style chat messages.

A message's `content` may be a string, a list of parts (`{"type": "text", "text": ...}` and others), or null
(an assistant message that only makes tool calls). The strategies only ever rewrite text, so non-text parts such
as images pass through untouched.
"""

from __future__ import annotations

import copy
from typing import Any

PIN_FIELD = "ctxb_pin"  # an app may set "ctxb_pin": true on a message; it is removed before forwarding
PIN_MARKER = "[ctxb:pin]"  # or put this marker anywhere in the message text


def text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)


def with_text(message: dict[str, Any], text: str) -> dict[str, Any]:
    """Returns a copy of the message whose text is replaced; non-text parts are kept."""
    out = copy.deepcopy(message)
    content = message.get("content")
    if isinstance(content, list):
        others = [p for p in content if not (isinstance(p, dict) and p.get("type") == "text")]
        out["content"] = [{"type": "text", "text": text}, *others]
    else:
        out["content"] = text
    return out


def is_pinned_by_app(message: dict[str, Any]) -> bool:
    return bool(message.get(PIN_FIELD)) or PIN_MARKER in text_of(message)


def strip_private_fields(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Removes the proxy's own fields so the upstream model sees a standard request."""
    out = []
    for m in messages:
        if PIN_FIELD in m:
            m = {k: v for k, v in m.items() if k != PIN_FIELD}
        out.append(m)
    return out


def last_user_index(messages: list[dict[str, Any]]) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return i
    return -1


def tool_name_for(messages: list[dict[str, Any]], index: int) -> str:
    """The name of the tool whose call produced the tool message at `index`."""
    call_id = messages[index].get("tool_call_id")
    for m in reversed(messages[:index]):
        for call in m.get("tool_calls") or []:
            if call.get("id") == call_id:
                return str(call.get("function", {}).get("name", "tool"))
    return str(messages[index].get("name", "tool"))


def tool_args_for(messages: list[dict[str, Any]], index: int) -> str:
    call_id = messages[index].get("tool_call_id")
    for m in reversed(messages[:index]):
        for call in m.get("tool_calls") or []:
            if call.get("id") == call_id:
                return str(call.get("function", {}).get("arguments", ""))
    return ""


def turns(messages: list[dict[str, Any]]) -> list[list[int]]:
    """Splits message indices into turns. A turn starts at a user message and runs until the next one.

    Messages before the first user message (normally the system prompt) form turn 0. Dropping whole turns keeps
    every assistant tool call next to its tool results, which the chat APIs require.
    """
    groups: list[list[int]] = [[]]
    for i, m in enumerate(messages):
        if m.get("role") == "user" and groups[-1]:
            groups.append([])
        groups[-1].append(i)
    return [g for g in groups if g]
