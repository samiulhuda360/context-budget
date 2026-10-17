"""Strategy 2: replace stale tool results with a one-line summary.

A tool result is stale when the conversation has moved on: at least `after_user_turns` newer user messages follow
it. Unless the latest question still refers to it (by an id, a name, or the tool's own name), its body is
replaced with one line saying which tool ran, with what arguments, how big the result was, and its gist.

The gist comes from rules by default (the result's shape and first meaningful line). With
`summariser: model`, a small model writes it instead; replies are cached like every other model call.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..terms import Query, stems
from .trim import html_to_text, kind_of

Summariser = Callable[[str, str], str]


def describe_shape(text: str) -> str:
    kind = kind_of(text)
    if kind == "json":
        value = json.loads(text)
        if isinstance(value, list):
            keys = sorted({k for x in value if isinstance(x, dict) for k in x})[:8]
            return f"JSON list of {len(value)} items" + (f" with fields {', '.join(keys)}" if keys else "")
        if isinstance(value, dict):
            scalars = ", ".join(f"{k}={v}" for k, v in value.items() if isinstance(v, str | int | float | bool))[:240]
            bulky = [f"{k} ({len(v)} items)" for k, v in value.items() if isinstance(v, list | dict)]
            return f"JSON object: {scalars}" + (f"; also {', '.join(bulky[:6])}" if bulky else "")
        return "JSON value"
    if kind == "html":
        text = html_to_text(text)
        first = next((ln.strip() for ln in text.split("\n") if ln.strip()), "")
        return f"web page, starts: {first[:120]}"
    lines = [ln for ln in text.split("\n") if ln.strip()]
    first = lines[0].strip()[:120] if lines else ""
    return f"{len(lines)} lines, starts: {first}"


def rules_summary(tool: str, args: str, text: str, tokens: int) -> str:
    call = f"{tool}({args[:100]})" if args else tool
    return f"[stale result of {call}, {tokens:,} tokens, summarised: {describe_shape(text)}]"


def is_referenced(query: Query, tool: str, args: str, text: str) -> bool:
    """True when the latest question still points at this result."""
    if query.strong_hit(text) or query.strong_hit(args):
        return True
    tool_words = stems(tool.replace("_", " ") + " " + args)
    return bool({w for w in query.stems if len(w) >= 4} & tool_words)


def make_model_summariser(chat: Callable[[list[dict[str, str]]], str]) -> Summariser:
    def summarise(tool: str, text: str) -> str:
        prompt = (
            "Summarise this tool output in one line of at most 40 words. Keep identifiers, names, amounts and "
            f"dates exactly as written.\n\nTool: {tool}\nOutput:\n{text[:12000]}"
        )
        return chat([{"role": "user", "content": prompt}]).strip().replace("\n", " ")

    return summarise


def summary_line(tool: str, args: str, text: str, tokens: int, summariser: Summariser | None) -> str:
    if summariser is None:
        return rules_summary(tool, args, text, tokens)
    call = f"{tool}({args[:100]})" if args else tool
    return f"[stale result of {call}, {tokens:,} tokens, summary: {summariser(tool, text)}]"


def stale_indices(messages: list[dict[str, Any]], after_user_turns: int) -> list[int]:
    users_after = 0
    out: list[int] = []
    for i in range(len(messages) - 1, -1, -1):
        role = messages[i].get("role")
        if role == "user":
            users_after += 1
        elif role == "tool" and users_after >= after_user_turns:
            out.append(i)
    return sorted(out)
