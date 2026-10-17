"""Deterministic checks: is the answer right, and did the facts it needs survive compression?

`answer_correct` needs every answer group to be present in the model's reply (each group lists accepted
spellings). Numbers must match as whole numbers, so "7" does not match "17", and thousands separators are ignored.

`evidence_kept` checks the compressed request itself, without any model: every evidence item must still be
there. That makes it free to run on every conversation and every strategy combination; the model runs then show
how well it predicts answer quality.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..messages import text_of

NUMBER_WORDS = {
    str(k): w
    for k, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}


def normalise(text: str) -> str:
    text = text.lower().replace("‑", "-").replace("–", "-")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 1,850 -> 1850
    return re.sub(r"\s+", " ", text)


def _alias_found(alias: str, answer: str) -> bool:
    a = normalise(alias)
    if re.fullmatch(r"\d+(\.\d+)?", a):
        if re.search(rf"(?<![\d.]){re.escape(a)}(?![\d]|\.\d)", answer):
            return True
        word = NUMBER_WORDS.get(a)
        return bool(word and re.search(rf"\b{word}\b", answer))
    return a in answer


def answer_correct(answer: str, groups: list[list[str]]) -> bool:
    ans = normalise(answer)
    return all(any(_alias_found(alias, ans) for alias in group) for group in groups)


def groups_found(answer: str, groups: list[list[str]]) -> int:
    ans = normalise(answer)
    return sum(1 for group in groups if any(_alias_found(alias, ans) for alias in group))


def context_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        parts.append(text_of(m))
        for call in m.get("tool_calls") or []:
            parts.append(json.dumps(call.get("function", {}), ensure_ascii=False))
    return "\n".join(parts)


def _item_kept(item: Any, text: str) -> bool:
    if isinstance(item, str):
        return item in text
    if any(s in text for s in item.get("or_any_of", [])):
        return True  # an alternative form of the same fact, such as a computed summary
    if "any_of" in item:
        return any(s in text for s in item["any_of"])
    if "all_of" in item:
        return all(s in text for s in item["all_of"])
    if "count_of" in item:
        return sum(text.count(s) for s in item["count_of"]) == item["equals"]
    raise ValueError(f"unknown evidence item {item!r}")


def evidence_kept(messages: list[dict[str, Any]], evidence: list[Any]) -> bool:
    text = context_text(messages)
    return all(_item_kept(e, text) for e in evidence)
