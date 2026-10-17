"""The offline evaluation: every strategy combination and budget over all 80 conversations, with no model calls.

For each configuration it measures tokens saved, the cost per 1,000 requests, the time the proxy adds, and
whether the facts each answer needs are still in the request ("evidence kept").
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass
from typing import Any

from .. import bench
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

    def route(self) -> Route:
        return Route(name=self.name, budget_tokens=self.budget, enabled=self.strategies)


def combo_name(strategies: tuple[str, ...]) -> str:
    return "+".join(strategies) if strategies else "none"


def configs() -> list[Config]:
    out = [Config(combo_name(c), c, DEFAULT_BUDGET) for k in range(len(STRATEGIES) + 1) for c in itertools.combinations(STRATEGIES, k)]
    out += [Config(f"all@{b}", STRATEGIES, b) for b in BUDGETS if b != DEFAULT_BUDGET]
    out += [Config(f"window@{b}", (), b, naive=True) for b in BUDGETS]
    return out


def cost_per_1000(input_tokens: float, price: Pricing, output_tokens: float = OUTPUT_TOKENS) -> float:
    return 1000 * (input_tokens * price.input_per_million + output_tokens * price.output_per_million) / 1_000_000


def apply(cfg: Config, conv: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    """Returns the compressed messages and the milliseconds it took."""
    import time

    if cfg.naive:
        t = time.perf_counter()
        msgs = sliding_window(conv["messages"], cfg.budget)
        return msgs, (time.perf_counter() - t) * 1000
    r = compress(conv["messages"], cfg.route())
    return r.messages, r.elapsed_ms


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[k]


def run(convs: list[dict[str, Any]] | None = None, only: list[str] | None = None) -> dict[str, Any]:
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
    }
