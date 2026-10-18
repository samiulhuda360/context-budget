"""Relevance by Jev: ask TypeSafe's decision model which earlier messages the latest question needs.

This is an alternative to BM25 for the retrieve step. The latest user question and the candidate earlier messages,
each with an id such as `m7`, go into one Jev request as the state, with one Noul question per message: "is message
m7 needed to answer the latest question?". Jev returns a probability for each. A turn's score is the highest
probability among its messages, and the retrieve step drops the lowest-scoring turns first, exactly as it does
with BM25 scores, so tool calls always stay with their results.

Jev reads at most 64k tokens per request (32k per question), so long conversations are split into chunks that each
stay well under that, and a single very long tool output is shown to Jev as its head and tail only. All questions
in a chunk go in one call. Any error is raised to the pipeline, which falls back to BM25 and records why.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..jev import Jev, JevReply
from ..messages import text_of
from ..tokens import _encoding, count_text
from .trim import html_to_text, kind_of

QUESTION = (
    "Is the earlier message with id `{id}` in `earlier_messages` needed to answer `latest_question`? "
    "Answer yes if it contains a fact, value, record, constraint or instruction that the answer depends on, or that "
    "the question refers back to. Answer no if it is unrelated to the question or only routine."
)


@dataclass
class JevStats:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    latency_ms: float = 0.0  # summed over the request's calls: what Jev adds to the request
    model: str = ""
    fallback: str = ""  # why BM25 was used instead, or empty
    scores: dict[str, float] = field(default_factory=dict)

    def add(self, reply: JevReply) -> None:
        self.calls += 1
        self.cached += int(reply.cached)
        self.input_tokens += reply.input_tokens
        self.latency_ms += reply.latency_s * 1000
        self.model = reply.model

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cached": self.cached,
            "input_tokens": self.input_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "model": self.model,
            "fallback": self.fallback,
        }


def head_tail(text: str, max_tokens: int) -> str:
    """The text if it fits, else its first and last halves of `max_tokens` tokens with a note between them."""
    enc = _encoding()
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return text
    half = max_tokens // 2
    return enc.decode(ids[:half]) + f"\n[... {len(ids) - 2 * half:,} tokens not shown ...]\n" + enc.decode(ids[-half:])


def describe(msgs: list[dict[str, Any]], i: int, tool_names: dict[str, str], max_tokens: int) -> dict[str, Any]:
    m = msgs[i]
    role = str(m.get("role", ""))
    item: dict[str, Any] = {"id": f"m{i}", "from": role}
    if role == "tool":
        item["from"] = f"tool result of {tool_names.get(str(m.get('tool_call_id')), 'a tool')}"
    text = text_of(m)
    if text and kind_of(text) == "html":
        text = html_to_text(text)  # markup and styles carry no meaning and would push the content out of view
    if text:
        item["text"] = head_tail(text, max_tokens)
    calls = m.get("tool_calls") or []
    if calls:
        item["calls"] = [f"{c.get('function', {}).get('name', 'tool')}({c.get('function', {}).get('arguments', '')})" for c in calls]
    return item


def chunks(items: list[list[dict[str, Any]]], chunk_tokens: int) -> list[list[dict[str, Any]]]:
    """Packs whole turns into chunks of at most `chunk_tokens` (a turn bigger than that gets a chunk of its own)."""
    out: list[list[dict[str, Any]]] = []
    size = 0
    for turn in items:
        t = sum(count_text(json.dumps(it, ensure_ascii=False)) for it in turn)
        if out and size + t <= chunk_tokens:
            out[-1].extend(turn)
            size += t
        else:
            out.append(list(turn))
            size = t
    return out


def turn_scores(
    question: str,
    msgs: list[dict[str, Any]],
    cands: list[list[int]],
    jev: Jev,
    chunk_tokens: int = 20000,
    message_tokens: int = 6000,
) -> tuple[list[float], JevStats]:
    """One probability per candidate turn (the highest of its messages'), and what the calls cost."""
    tool_names = {str(c.get("id")): str(c.get("function", {}).get("name", "tool")) for m in msgs for c in m.get("tool_calls") or []}
    described = [[describe(msgs, i, tool_names, message_tokens) for i in g] for g in cands]
    stats = JevStats()
    for chunk in chunks(described, chunk_tokens):
        state = {"latest_question": question, "earlier_messages": chunk}
        questions = {it["id"]: {"type": "noul", "instructions": QUESTION.format(id=it["id"])} for it in chunk}
        reply = jev.ask(state, questions)
        stats.add(reply)
        for it in chunk:
            stats.scores[it["id"]] = reply.noul(it["id"])
    return [max(stats.scores[f"m{i}"] for i in g) for g in cands], stats
