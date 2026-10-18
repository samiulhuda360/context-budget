"""The offline evaluation: every strategy combination and budget over all 80 conversations, with no answering model.

For each configuration it measures tokens saved, the cost per 1,000 requests, the time the proxy adds, and
whether the facts each answer needs are still in the request ("evidence kept").

The relevance comparison runs the retrieve step with BM25 and with Jev at four budgets, alone and inside the full
pipeline. Jev's replies come from cache/jev (recorded once with TYPESAFE_API_KEY), so this also runs offline.
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass
from typing import Any

from .. import bench
from ..jev import Jev, cost_usd
from ..pipeline import compress
from ..policy import STRATEGIES, Pricing, Route
from ..tokens import count_messages
from .checks import evidence_kept
from .naive import sliding_window

DEFAULT_BUDGET = 8000
BUDGETS = (32000, 16000, 8000, 4000, 2000, 1000)
OUTPUT_TOKENS = 100  # assumed reply length for the cost figures
PRICES = {
    "flash_lite": Pricing(0.10, 0.40, "Gemini Flash-Lite list price ($0.10 in / $0.40 out per million tokens)"),
    "frontier": Pricing(3.00, 15.00, "frontier-model price ($3 in / $15 out per million tokens)"),
}


@dataclass(frozen=True)
class Config:
    name: str
    strategies: tuple[str, ...]
    budget: int
    naive: bool = False
    relevance: str = "bm25"

    def route(self) -> Route:
        return Route(name=self.name, budget_tokens=self.budget, enabled=self.strategies, relevance=self.relevance)


def combo_name(strategies: tuple[str, ...]) -> str:
    return "+".join(strategies) if strategies else "none"


def configs() -> list[Config]:
    out = [Config(combo_name(c), c, DEFAULT_BUDGET) for k in range(len(STRATEGIES) + 1) for c in itertools.combinations(STRATEGIES, k)]
    out += [Config(f"all@{b}", STRATEGIES, b) for b in BUDGETS if b != DEFAULT_BUDGET]
    out += [Config(f"window@{b}", (), b, naive=True) for b in BUDGETS]
    return out


def cost_per_1000(input_tokens: float, price: Pricing, output_tokens: float = OUTPUT_TOKENS) -> float:
    return 1000 * (input_tokens * price.input_per_million + output_tokens * price.output_per_million) / 1_000_000


class JevFallback(RuntimeError):
    """The evaluation never silently uses BM25 in place of Jev: a missing or failed Jev reply stops the run."""


def compress_cfg(cfg: Config, conv: dict[str, Any], jev: Jev | None = None) -> Any:
    r = compress(conv["messages"], cfg.route(), jev=jev)
    if r.jev and r.jev["fallback"]:
        raise JevFallback(f"{conv['id']} ({cfg.name}): {r.jev['fallback']}")
    return r


def apply(cfg: Config, conv: dict[str, Any], jev: Jev | None = None) -> tuple[list[dict[str, Any]], float]:
    """Returns the compressed messages and the milliseconds it took (Jev's time is reported separately)."""
    import time

    if cfg.naive:
        t = time.perf_counter()
        msgs = sliding_window(conv["messages"], cfg.budget)
        return msgs, (time.perf_counter() - t) * 1000
    r = compress_cfg(cfg, conv, jev)
    return r.messages, r.elapsed_ms - (r.jev["latency_ms"] if r.jev and not r.jev["cached"] else 0.0)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[k]


RELEVANCE_BUDGETS = (8000, 4000, 2000, 1000)
RELEVANCE_SETS = {"retrieve": ("retrieve",), "all": STRATEGIES}


def relevance_configs() -> list[Config]:
    return [
        Config(f"{label}+{rel}@{b}", strategies, b, relevance=rel)
        for b in RELEVANCE_BUDGETS
        for label, strategies in RELEVANCE_SETS.items()
        for rel in ("bm25", "jev")
    ]


def relevance_comparison(convs: list[dict[str, Any]], jev: Jev) -> list[dict[str, Any]]:
    """BM25 against Jev for the retrieve step: facts kept and tokens, plus Jev's calls, tokens, cost and latency."""
    out = []
    kept_by: dict[str, dict[str, bool]] = {}
    for cfg in relevance_configs():
        rows = []
        for c in convs:
            r = compress_cfg(cfg, c, jev)
            rows.append(
                {
                    "id": c["id"],
                    "family": c["family"],
                    "before": r.tokens_before,
                    "after": r.tokens_after,
                    "kept": evidence_kept(r.messages, c["evidence"]),
                    "jev": r.jev,
                }
            )
        kept_by[cfg.name] = {r["id"]: r["kept"] for r in rows}
        twin = cfg.name.replace("+jev@", "+bm25@")
        used = [r["jev"] for r in rows if r["jev"] and r["jev"]["calls"]]
        tb = sum(r["before"] for r in rows)
        ta = sum(r["after"] for r in rows)
        fams: dict[str, list[bool]] = {}
        for r in rows:
            fams.setdefault(r["family"], []).append(r["kept"])
        tokens = sum(u["input_tokens"] for u in used)
        row: dict[str, Any] = {
            "config": cfg.name,
            "strategies": list(cfg.strategies),
            "relevance": cfg.relevance,
            "budget": cfg.budget,
            "saved_pct": round(100 * (tb - ta) / tb, 1),
            "evidence_kept_pct": round(100 * sum(r["kept"] for r in rows) / len(rows), 1),
            "evidence_kept_by_family": {f: round(100 * sum(v) / len(v)) for f, v in fams.items()},
        }
        if cfg.relevance == "jev":
            twin_kept = kept_by[twin]
            row |= {
                "kept_where_bm25_lost": [r["id"] for r in rows if r["kept"] and not twin_kept[r["id"]]],
                "lost_where_bm25_kept": [r["id"] for r in rows if not r["kept"] and twin_kept[r["id"]]],
                "requests_calling_jev": len(used),
                "jev_calls": sum(u["calls"] for u in used),
                "jev_input_tokens": tokens,
                "jev_cost_usd": round(cost_usd(tokens), 5),
                "jev_cost_per_1000_requests_usd": round(1000 * cost_usd(tokens) / len(rows), 4),
                "jev_added_ms_p50": round(percentile([u["latency_ms"] for u in used], 50), 1),
                "jev_added_ms_p95": round(percentile([u["latency_ms"] for u in used], 95), 1),
                "jev_model": sorted({u["model"] for u in used}),
            }
        out.append(row)
    return out


def run(convs: list[dict[str, Any]] | None = None, only: list[str] | None = None, jev: Jev | None = None) -> dict[str, Any]:
    convs = convs or bench.load()
    before = {c["id"]: count_messages(c["messages"]) for c in convs}
    baseline_kept = {c["id"]: evidence_kept(c["messages"], c["evidence"]) for c in convs}
    results = []
    for cfg in configs():
        if only and cfg.name not in only:
            continue
        rows = []
        for c in convs:
            msgs, ms = apply(cfg, c)
            rows.append(
                {
                    "id": c["id"],
                    "family": c["family"],
                    "before": before[c["id"]],
                    "after": count_messages(msgs),
                    "kept": evidence_kept(msgs, c["evidence"]),
                    "ms": ms,
                }
            )
        tb = sum(r["before"] for r in rows)
        ta = sum(r["after"] for r in rows)
        fams: dict[str, list[bool]] = {}
        for r in rows:
            fams.setdefault(r["family"], []).append(r["kept"])
        avg_after = ta / len(rows)
        results.append(
            {
                "config": cfg.name,
                "strategies": list(cfg.strategies),
                "budget": cfg.budget,
                "naive": cfg.naive,
                "tokens_before": tb,
                "tokens_after": ta,
                "saved_pct": round(100 * (tb - ta) / tb, 1),
                "median_saved_pct": round(statistics.median(100 * (r["before"] - r["after"]) / r["before"] for r in rows), 1),
                "evidence_kept_pct": round(100 * sum(r["kept"] for r in rows) / len(rows), 1),
                "evidence_kept_by_family": {f: round(100 * sum(v) / len(v)) for f, v in fams.items()},
                "lost": [r["id"] for r in rows if not r["kept"] and baseline_kept[r["id"]]],
                "latency_ms_p50": round(percentile([r["ms"] for r in rows], 50), 2),
                "latency_ms_p95": round(percentile([r["ms"] for r in rows], 95), 2),
                "cost_per_1000": {k: round(cost_per_1000(avg_after, p), 4) for k, p in PRICES.items()},
            }
        )
    return {
        "conversations": len(convs),
        "baseline_evidence_kept_pct": round(100 * sum(baseline_kept.values()) / len(convs), 1),
        "avg_tokens_before": round(sum(before.values()) / len(convs)),
        "output_tokens_assumed": OUTPUT_TOKENS,
        "prices": {k: p.label for k, p in PRICES.items()},
        "configs": results,
        "relevance_comparison": relevance_comparison(convs, jev or Jev(offline=True)) if not only else [],
    }
