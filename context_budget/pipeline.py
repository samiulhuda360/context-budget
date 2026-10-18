"""The compression pipeline: pinned section first, then dedup, trim, stale and retrieve, within a token budget.

Order matters and is fixed:
1. **Pin** the system prompt, the latest user message and anything the app marked must-keep. Pinned messages
   are never changed or dropped.
2. **Dedup** first, so a document pasted twice is trimmed once.
3. **Trim** long tool outputs (logs, JSON, HTML), keeping what matches the question.
4. **Stale**: old tool results the conversation has moved past become one line.
5. **Retrieve**: only if the request is still over budget, drop the earlier turns least relevant to the question.
   Relevance is BM25 (or embeddings) by default; with `relevance: jev` TypeSafe's Jev model scores each earlier
   message instead, and any Jev error or timeout falls back to BM25 for that request.

Every change is recorded per message with the tokens it saved, so the proxy log and `ctxb analyse` can say
exactly what was cut and why.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .jev import Jev, JevError
from .messages import is_pinned_by_app, last_user_index, strip_private_fields, text_of, tool_args_for, tool_name_for, turns, with_text
from .policy import STRATEGIES, Route, TrimSettings
from .strategies import dedup as dedup_s
from .strategies import jev_relevance as jev_s
from .strategies import retrieve as retrieve_s
from .strategies import stale as stale_s
from .strategies import trim as trim_s
from .terms import Query
from .tokens import count_message, count_messages

log = logging.getLogger("context_budget")


@dataclass
class Action:
    strategy: str
    saved: int
    note: str


@dataclass
class MessageReport:
    index: int
    role: str
    tool: str
    tokens_before: int
    tokens_after: int
    pinned: str = ""  # why it is pinned, or empty
    dropped: bool = False
    actions: list[Action] = field(default_factory=list)
    kept_note: str = ""  # why a strategy left it alone, when that is worth saying


@dataclass
class Result:
    route: str
    budget: int
    enabled: list[str]
    messages: list[dict[str, Any]]
    original: list[dict[str, Any]]
    reports: list[MessageReport]
    tokens_before: int
    tokens_after: int
    elapsed_ms: float
    jev: dict[str, Any] | None = None  # Jev calls, tokens, latency and any fallback, when the route uses relevance: jev

    @property
    def saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def saved_pct(self) -> float:
        return 100.0 * self.saved / self.tokens_before if self.tokens_before else 0.0

    @property
    def over_budget(self) -> int:
        return max(0, self.tokens_after - self.budget)

    def saved_by(self) -> dict[str, int]:
        out = dict.fromkeys(STRATEGIES, 0)
        for r in self.reports:
            for a in r.actions:
                out[a.strategy] += a.saved
        return out

    def summary(self) -> dict[str, Any]:
        out = {
            "route": self.route,
            "budget": self.budget,
            "enabled": self.enabled,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "saved_pct": round(self.saved_pct, 1),
            "saved_by": self.saved_by(),
            "over_budget": self.over_budget,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }
        if self.jev is not None:
            out["jev"] = self.jev
        return out

    def to_dict(self) -> dict[str, Any]:
        return {**self.summary(), "reports": [asdict(r) for r in self.reports], "messages": self.messages, "original": self.original}


def _turn_text(msgs: list[dict[str, Any]], idx: list[int]) -> str:
    parts = []
    for i in idx:
        parts.append(text_of(msgs[i]))
        for call in msgs[i].get("tool_calls") or []:
            parts.append(json.dumps(call.get("function", {}), ensure_ascii=False))
    return "\n".join(parts)


def tighter(s: TrimSettings) -> TrimSettings:
    """Halves what trimming keeps; used only when a request is still over budget after every other step."""
    return replace(
        s,
        min_tokens=max(100, s.min_tokens // 2),
        head_lines=max(2, s.head_lines // 2),
        tail_lines=max(2, s.tail_lines // 2),
        match_context=0,
        max_match_lines=max(5, s.max_match_lines // 2),
        json_head_items=max(1, s.json_head_items // 2),
        json_tail_items=max(1, s.json_tail_items // 2),
        json_max_matches=max(5, s.json_max_matches // 2),
    )


def compress(
    messages: list[dict[str, Any]],
    route: Route,
    summariser: stale_s.Summariser | None = None,
    embedder: retrieve_s.Embedder | None = None,
    jev: Jev | None = None,
) -> Result:
    started = time.perf_counter()
    original = strip_private_fields(copy.deepcopy(messages))
    msgs = [copy.deepcopy(m) for m in original]
    last_user = last_user_index(messages)
    reports = [
        MessageReport(i, str(m.get("role", "")), tool_name_for(messages, i) if m.get("role") == "tool" else "", count_message(m), 0)
        for i, m in enumerate(original)
    ]
    for i, m in enumerate(messages):
        if m.get("role") == "system":
            reports[i].pinned = "system prompt"
        elif i == last_user:
            reports[i].pinned = "latest user message"
        elif is_pinned_by_app(m):
            reports[i].pinned = "marked must-keep by the app"
    question = text_of(messages[last_user]) if last_user >= 0 else ""
    query = Query.from_text(question)
    dropped: set[int] = set()
    jev_stats: jev_s.JevStats | None = None

    def change(i: int, strategy: str, text: str, note: str) -> None:
        previous = msgs[i]
        before = count_message(previous)
        msgs[i] = with_text(previous, text)
        saved = before - count_message(msgs[i])
        if saved > 0:
            reports[i].actions.append(Action(strategy, saved, note))
        else:  # a strategy never makes a message longer
            msgs[i] = previous

    # 3. dedup: newest copy wins, earlier copies become pointers
    if route.on("dedup"):
        seen_msg: dict[str, int] = {}
        seen_block: dict[str, int] = {}
        for i in range(len(msgs) - 1, -1, -1):
            text = text_of(msgs[i])
            if len(text) < route.dedup.min_chars:
                continue
            h = dedup_s.norm_hash(text)
            if not reports[i].pinned and msgs[i].get("role") != "system":
                if h in seen_msg:
                    change(i, "dedup", dedup_s.pointer(text, seen_msg[h]), f"same as message {seen_msg[h]}")
                    continue
                parts = dedup_s.blocks(text)
                new_parts = []
                for part in parts:
                    bh = dedup_s.norm_hash(part)
                    if len(part.strip()) >= route.dedup.min_chars and bh in seen_block:
                        new_parts.append(dedup_s.pointer(part, seen_block[bh]))
                    else:
                        new_parts.append(part)
                if new_parts != parts:
                    change(i, "dedup", "\n\n".join(new_parts), "repeated block(s) replaced by pointers")
            seen_msg.setdefault(h, i)
            for part in dedup_s.blocks(text):
                if len(part.strip()) >= route.dedup.min_chars:
                    seen_block.setdefault(dedup_s.norm_hash(part), i)

    # 1. trim long tool outputs
    if route.on("trim"):
        for i, m in enumerate(msgs):
            if m.get("role") != "tool" or reports[i].pinned or any(a.strategy == "dedup" for a in reports[i].actions):
                continue
            q = query.merge(Query.from_text(tool_args_for(messages, i)))
            text, note = trim_s.trim(text_of(m), q, route.trim)
            if note:
                change(i, "trim", text, note)

    # 2. stale tool results
    if route.on("stale"):
        for i in stale_s.stale_indices(original, route.stale.after_user_turns):
            if reports[i].pinned or any(a.strategy == "dedup" for a in reports[i].actions):
                continue
            body = text_of(original[i])
            if reports[i].tokens_before < route.stale.min_tokens:
                continue
            tool, args = tool_name_for(messages, i), tool_args_for(messages, i)
            if route.stale.keep_if_referenced and stale_s.is_referenced(query, tool, args, body):
                reports[i].kept_note = "old result kept: the latest question refers to it"
                continue
            use = summariser if route.stale.summariser == "model" else None
            change(i, "stale", stale_s.summary_line(tool, args, body, reports[i].tokens_before, use), "old tool result summarised")

    # 4. retrieve: drop the least relevant earlier turns while over budget
    total = count_messages(msgs)
    if route.on("retrieve") and (total > route.budget_tokens or not route.retrieve.only_over_budget):
        groups = turns(msgs)
        protected_tail = set(range(max(0, len(groups) - route.retrieve.keep_recent_turns), len(groups)))
        cands = [g for k, g in enumerate(groups) if k not in protected_tail and not any(reports[i].pinned for i in g)]
        if cands:
            texts = [_turn_text(msgs, g) for g in cands]
            scores: list[float] = []
            threshold = 0.0  # a turn is among the protected top_k only if it scores above this
            if route.relevance == "jev":
                client = jev or Jev(model=route.jev.model, timeout_s=route.jev.timeout_s, retries=1)
                try:
                    scores, jev_stats = jev_s.turn_scores(question, msgs, cands, client, route.jev.chunk_tokens, route.jev.message_tokens)
                    threshold = route.jev.keep_threshold - 1e-9
                    label = "Jev"
                except (JevError, KeyError, ValueError, TypeError) as e:
                    jev_stats = jev_s.JevStats(fallback=str(e)[:200])
                    log.warning("Jev relevance failed (%s); falling back to BM25 for this request", jev_stats.fallback)
                    scores = []
            if not scores:
                label = "relevance"
                if route.retrieve.method == "embed" and embedder is not None:
                    scores = retrieve_s.embed_scores(question, texts, embedder)
                else:
                    scores = retrieve_s.bm25_scores(question, texts)
            ranked = sorted(range(len(cands)), key=lambda k: -scores[k])
            top = {k for k in ranked[: route.retrieve.top_k] if scores[k] > threshold}
            order = sorted((k for k in range(len(cands)) if k not in top), key=lambda k: (scores[k], k))
            for k in order:
                if route.retrieve.only_over_budget and total <= route.budget_tokens:
                    break
                for i in cands[k]:
                    t = count_message(msgs[i])
                    reports[i].actions.append(Action("retrieve", t, f"earlier turn dropped ({label} {scores[k]:.2f})"))
                    reports[i].dropped = True
                    dropped.add(i)
                    total -= t

    # 5. still over budget: trim the largest tool outputs again with tighter settings, largest first
    if route.on("trim") and total > route.budget_tokens:
        settings = route.trim
        for level in range(1, 4):
            settings = tighter(settings)
            for i in sorted(
                (i for i in range(len(msgs)) if i not in dropped and msgs[i].get("role") == "tool" and not reports[i].pinned),
                key=lambda i: -count_message(msgs[i]),
            ):
                if total <= route.budget_tokens:
                    break
                if any(a.strategy in ("dedup", "stale") for a in reports[i].actions):
                    continue
                q = query.merge(Query.from_text(tool_args_for(messages, i)))
                text, note = trim_s.trim(text_of(original[i]), q, settings)
                before = count_message(msgs[i])
                if note:
                    change(i, "trim", text, f"{note} (tightened x{level} to fit the budget)")
                total -= before - count_message(msgs[i])
            if total <= route.budget_tokens:
                break

    out = [m for i, m in enumerate(msgs) if i not in dropped]
    for i, m in enumerate(msgs):
        reports[i].tokens_after = 0 if i in dropped else count_message(m)
    return Result(
        route=route.name,
        budget=route.budget_tokens,
        enabled=list(route.enabled),
        messages=strip_private_fields(out),
        original=original,
        reports=reports,
        tokens_before=count_messages(original),
        tokens_after=count_messages(out),
        elapsed_ms=(time.perf_counter() - started) * 1000,
        jev=jev_stats.summary() if jev_stats is not None else None,
    )
