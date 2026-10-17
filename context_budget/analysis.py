"""What was cut and why, in a form the CLI table and the web page can both show."""

from __future__ import annotations

import difflib
from dataclasses import asdict
from typing import Any

from .messages import text_of
from .pipeline import Result

FOLD_AT = 6  # runs of cut or unchanged lines longer than this are folded in the web page


def segments(before: str, after: str) -> list[dict[str, Any]]:
    """Line-level diff of one message: runs of kept, cut and added lines."""
    a, b = before.split("\n"), after.split("\n")
    out: list[dict[str, Any]] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append({"op": "keep", "lines": a[i1:i2]})
        else:
            if i2 > i1:
                out.append({"op": "cut", "lines": a[i1:i2]})
            if j2 > j1:
                out.append({"op": "add", "lines": b[j1:j2]})
    return out


def explain(result: Result) -> dict[str, Any]:
    """The pipeline result plus a per-message diff."""
    after_by_index: dict[int, dict[str, Any]] = {}
    k = 0
    for rep in result.reports:
        if not rep.dropped:
            after_by_index[rep.index] = result.messages[k]
            k += 1
    rows = []
    for rep in result.reports:
        original = result.original[rep.index]
        before_text = text_of(original)
        calls = [c.get("function", {}) for c in original.get("tool_calls") or []]
        row = asdict(rep)
        row["calls"] = calls
        if rep.dropped:
            row["segments"] = [{"op": "cut", "lines": before_text.split("\n")}] if before_text else []
        elif rep.actions:
            row["segments"] = segments(before_text, text_of(after_by_index[rep.index]))
        else:
            row["segments"] = [{"op": "keep", "lines": before_text.split("\n")}] if before_text else []
        rows.append(row)
    return {**result.summary(), "messages": rows}


def table(result: Result) -> list[list[str]]:
    """Rows for the terminal: one per message, with what happened and why."""
    out = []
    for rep in result.reports:
        if rep.pinned:
            what, why = "pinned", rep.pinned
        elif rep.dropped:
            what, why = "dropped", rep.actions[-1].note
        elif rep.actions:
            what = "+".join(dict.fromkeys(a.strategy for a in rep.actions))
            why = "; ".join(dict.fromkeys(a.note for a in rep.actions))
        else:
            what, why = "kept", rep.kept_note or ""
        label = rep.role + (f":{rep.tool}" if rep.tool else "")
        out.append([str(rep.index), label, f"{rep.tokens_before:,}", f"{rep.tokens_after:,}", what, why])
    return out
