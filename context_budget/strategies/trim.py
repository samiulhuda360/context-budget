"""Strategy 1: trim long tool outputs.

Three kinds of output are recognised:
- HTML is converted to readable text first (scripts, styles, navigation and footers removed).
- JSON keeps its first and last few list items, every item that mentions the question, and a count of each
  field's values, so "how many orders failed?" can still be answered after most rows are cut.
- Everything else is treated as lines (logs, listings, command output): runs of near-identical lines are
  collapsed, then the head, the tail, lines matching the question and error lines are kept.
Cut regions are replaced by a short marker that says how much was removed.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from html.parser import HTMLParser
from typing import Any

from ..policy import TrimSettings
from ..terms import Query
from ..tokens import count_text

SIGNAL = re.compile(r"\b(ERROR|FATAL|CRITICAL|PANIC|Exception|Traceback|Caused by|panic:|exit code [1-9]|FAILED)\b")
VOLATILE = re.compile(r"0x[0-9a-fA-F]+|[0-9a-f]{8,}|\d+")


def kind_of(text: str) -> str:
    head = text.lstrip()[:2000].lower()
    if head.startswith(("{", "[")):
        try:
            json.loads(text)
            return "json"
        except ValueError:
            pass
    if "<html" in head or "<!doctype html" in head or head.count("<div") + head.count("<p") + head.count("<td") >= 5:
        return "html"
    return "text"


# --- HTML -------------------------------------------------------------------------------------------------

DROP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "button", "iframe", "template"}
BLOCK_TAGS = {"p", "div", "section", "article", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br", "table", "ul", "ol", "dd", "dt", "pre", "main"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in DROP_TAGS:
            self.skip += 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag: str) -> None:
        if tag in DROP_TAGS and self.skip:
            self.skip -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _TextExtractor()
    p.feed(html)
    p.close()
    lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip(" |") for line in "".join(p.parts).split("\n")]
    return "\n".join(line for line in lines if line)


# --- lines ------------------------------------------------------------------------------------------------


def _shape(line: str) -> str:
    return VOLATILE.sub("#", line.strip())


def collapse_repeats(lines: list[str], query: Query) -> list[str]:
    """Collapses runs of three or more lines that differ only in numbers, ids or timestamps.

    Lines that match the question are never folded into a run.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        j = i + 1
        shape = _shape(lines[i])
        if shape and not query.matches(lines[i]):
            while j < len(lines) and _shape(lines[j]) == shape and not query.matches(lines[j]):
                j += 1
        run = j - i
        if run >= 3:
            out.append(lines[i])
            out.append(f"[... {run - 2} similar lines collapsed ...]")
            out.append(lines[j - 1])
        else:
            out.extend(lines[i:j])
        i = j
    return out


def select_lines(lines: list[str], query: Query, s: TrimSettings) -> list[str]:
    n = len(lines)
    query = query.narrow(lines)
    keep: set[int] = set(range(min(s.head_lines, n))) | set(range(max(0, n - s.tail_lines), n))
    strong: list[int] = []
    weak: list[int] = []
    for i, line in enumerate(lines):
        if query.strong_hit(line):
            strong.append(i)
        elif SIGNAL.search(line) or query.matches(line):
            weak.append(i)
    chosen = (strong + weak)[: s.max_match_lines]
    for i in chosen:
        keep.update(range(max(0, i - s.match_context), min(n, i + s.match_context + 1)))
    out: list[str] = []
    gap = 0
    for i in range(n):
        if i in keep:
            if gap:
                out.append(f"[... {gap} lines cut ...]")
                gap = 0
            out.append(lines[i])
        else:
            gap += 1
    if gap:
        out.append(f"[... {gap} lines cut ...]")
    return out


def trim_text(text: str, query: Query, s: TrimSettings) -> str:
    raw = text.split("\n")
    lines = collapse_repeats(raw, query.narrow(raw))
    collapsed = "\n".join(lines)
    if count_text(collapsed) <= s.min_tokens:
        return collapsed
    return "\n".join(select_lines(lines, query, s))


# --- JSON -------------------------------------------------------------------------------------------------


def _field_summary(items: list[Any]) -> dict[str, dict[str, dict[str, float]]]:
    """Per value of each low-cardinality field (status, region, type ...): the item count and numeric sums.

    Computed over every item before any are cut, so "how many orders were refunded?" and "what did the
    refunds add up to?" can still be answered from a trimmed result.
    """
    if not items or not all(isinstance(x, dict) for x in items):
        return {}
    keys = sorted({k for x in items for k in x})
    numeric = [k for k in keys if all(isinstance(x.get(k), int | float) and not isinstance(x.get(k), bool) for x in items)]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for k in keys:
        values = [x.get(k) for x in items if isinstance(x.get(k), str | bool)]
        distinct = Counter(str(v) for v in values)
        if not (1 < len(distinct) <= 8 and len(values) >= len(items) * 0.8):
            continue
        groups: dict[str, dict[str, float]] = {}
        for value, n in distinct.most_common():
            members = [x for x in items if str(x.get(k)) == value]
            entry: dict[str, float] = {"count": n}
            for f in numeric:
                entry[f"{f}_sum"] = round(sum(float(x[f]) for x in members), 2)
            groups[value] = entry
        out[k] = groups
    return out


def _trim_json(value: Any, query: Query, s: TrimSettings) -> Any:
    keep_n = s.json_head_items + s.json_tail_items
    if isinstance(value, list):
        if len(value) <= keep_n + 2:
            return [_trim_json(v, query, s) for v in value]
        n = len(value)
        query = query.narrow([json.dumps(v, ensure_ascii=False) for v in value], max_share=0.3)
        head = set(range(s.json_head_items)) | set(range(n - s.json_tail_items, n))
        matches = [i for i, v in enumerate(value) if i not in head and query.strong_hit(json.dumps(v, ensure_ascii=False))]
        if len(matches) < s.json_max_matches:
            matches += [i for i, v in enumerate(value) if i not in head and i not in matches and query.matches(json.dumps(v, ensure_ascii=False))][
                : s.json_max_matches - len(matches)
            ]
        keep = sorted(head | set(matches[: s.json_max_matches]))
        out: list[Any] = [value[i] for i in keep]
        note: dict[str, Any] = {
            "_ctxb": f"{n - len(keep)} of {n} items cut; kept the first {s.json_head_items}, the last {s.json_tail_items} and those matching the question"
        }
        summary = _field_summary(value)
        if summary:
            note["_summary_over_all_items"] = summary
        return [note, *out]
    if isinstance(value, dict):
        return {k: _trim_json(v, query, s) for k, v in value.items()}
    return value


def trim_json(text: str, query: Query, s: TrimSettings) -> str:
    return json.dumps(_trim_json(json.loads(text), query, s), ensure_ascii=False, separators=(",", ":"))


# --- entry point ------------------------------------------------------------------------------------------


def trim(text: str, query: Query, s: TrimSettings) -> tuple[str, str]:
    """Returns (new text, a short note on what was done). The text is unchanged when it is short."""
    if count_text(text) < s.min_tokens:
        return text, ""
    kind = kind_of(text)
    if kind == "json":
        return trim_json(text, query, s), "JSON list items cut, per-field counts and sums kept"
    if kind == "html":
        plain = html_to_text(text)
        if count_text(plain) < s.min_tokens:
            return plain, "HTML converted to text"
        return trim_text(plain, query, s), "HTML converted to text, then lines cut"
    return trim_text(text, query, s), "head, tail and matching lines kept"
