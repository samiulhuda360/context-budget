"""The YAML policy: which strategies run, with what settings, for which route.

A route is chosen by the `X-Ctxb-Route` request header, else by the first route whose `models` patterns match
the request's model name, else `default`. Settings a route leaves out fall back to the `defaults` section.
"""

from __future__ import annotations

import fnmatch
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

STRATEGIES = ("trim", "stale", "dedup", "retrieve")


@dataclass(frozen=True)
class TrimSettings:
    min_tokens: int = 600  # tool outputs shorter than this are left alone
    head_lines: int = 15
    tail_lines: int = 15
    match_context: int = 1  # lines kept either side of a line that matches the question
    max_match_lines: int = 80
    json_head_items: int = 3
    json_tail_items: int = 2
    json_max_matches: int = 40


@dataclass(frozen=True)
class StaleSettings:
    after_user_turns: int = 2  # a tool result is stale once this many newer user messages follow it
    min_tokens: int = 200
    keep_if_referenced: bool = True  # keep an old result the latest question still refers to
    summariser: str = "rules"  # "rules" or "model" (a small model writes the one-line summary)


@dataclass(frozen=True)
class DedupSettings:
    min_chars: int = 200  # blocks shorter than this are never treated as duplicates


@dataclass(frozen=True)
class RetrieveSettings:
    method: str = "bm25"  # "bm25" or "embed" (an OpenAI-compatible /embeddings endpoint)
    top_k: int = 4  # earlier turns kept by relevance
    keep_recent_turns: int = 2  # the most recent turns are always kept
    only_over_budget: bool = True  # leave earlier turns alone while the request already fits the budget


@dataclass(frozen=True)
class Route:
    name: str = "default"
    models: tuple[str, ...] = ()
    budget_tokens: int = 8000
    enabled: tuple[str, ...] = STRATEGIES
    trim: TrimSettings = field(default_factory=TrimSettings)
    stale: StaleSettings = field(default_factory=StaleSettings)
    dedup: DedupSettings = field(default_factory=DedupSettings)
    retrieve: RetrieveSettings = field(default_factory=RetrieveSettings)

    def on(self, strategy: str) -> bool:
        return strategy in self.enabled

    def with_strategies(self, enabled: tuple[str, ...] | list[str], budget: int | None = None) -> Route:
        return replace(self, enabled=tuple(s for s in STRATEGIES if s in enabled), budget_tokens=budget or self.budget_tokens)

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Pricing:
    input_per_million: float = 0.10
    output_per_million: float = 0.40
    label: str = "Gemini Flash-Lite list price"


@dataclass(frozen=True)
class Policy:
    routes: tuple[Route, ...] = (Route(),)
    upstream: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    log_path: str = "ctxb-requests.jsonl"
    pricing: Pricing = field(default_factory=Pricing)

    def route(self, name: str | None = None, model: str | None = None) -> Route:
        if name:
            for r in self.routes:
                if r.name == name:
                    return r
            raise KeyError(f"no route named {name!r}")
        if model:
            for r in self.routes:
                if any(fnmatch.fnmatch(model, p) for p in r.models):
                    return r
        for r in self.routes:
            if r.name == "default":
                return r
        return self.routes[0]


class PolicyError(ValueError):
    pass


def _build(cls: type, data: dict[str, Any] | None, where: str) -> Any:
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise PolicyError(f"{where}: unknown setting(s) {', '.join(sorted(unknown))}")
    return cls(**data)


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _route(name: str, raw: dict[str, Any]) -> Route:
    where = f"route {name!r}"
    allowed = {"models", "budget_tokens", "strategies", *STRATEGIES}
    unknown = set(raw) - allowed
    if unknown:
        raise PolicyError(f"{where}: unknown setting(s) {', '.join(sorted(unknown))}")
    strategies = raw.get("strategies", {})
    if isinstance(strategies, list):
        enabled = tuple(s for s in STRATEGIES if s in strategies)
        bad = set(strategies) - set(STRATEGIES)
    else:
        enabled = tuple(s for s in STRATEGIES if strategies.get(s, True))
        bad = set(strategies) - set(STRATEGIES)
    if bad:
        raise PolicyError(f"{where}: unknown strategy {', '.join(sorted(bad))}; known: {', '.join(STRATEGIES)}")
    budget = int(raw.get("budget_tokens", 8000))
    if budget < 256:
        raise PolicyError(f"{where}: budget_tokens must be at least 256")
    route = Route(
        name=name,
        models=tuple(raw.get("models", ())),
        budget_tokens=budget,
        enabled=enabled,
        trim=_build(TrimSettings, raw.get("trim"), f"{where}.trim"),
        stale=_build(StaleSettings, raw.get("stale"), f"{where}.stale"),
        dedup=_build(DedupSettings, raw.get("dedup"), f"{where}.dedup"),
        retrieve=_build(RetrieveSettings, raw.get("retrieve"), f"{where}.retrieve"),
    )
    if route.stale.summariser not in ("rules", "model"):
        raise PolicyError(f"{where}.stale.summariser must be 'rules' or 'model'")
    if route.retrieve.method not in ("bm25", "embed"):
        raise PolicyError(f"{where}.retrieve.method must be 'bm25' or 'embed'")
    return route


def parse(data: dict[str, Any]) -> Policy:
    if not isinstance(data, dict):
        raise PolicyError("the policy must be a YAML mapping")
    unknown = set(data) - {"upstream", "log_path", "pricing", "defaults", "routes"}
    if unknown:
        raise PolicyError(f"unknown top-level key(s) {', '.join(sorted(unknown))}")
    defaults = data.get("defaults") or {}
    raw_routes = data.get("routes") or {"default": {}}
    if "default" not in raw_routes:
        raw_routes = {**raw_routes, "default": {}}
    routes = tuple(_route(name, _merge(defaults, raw or {})) for name, raw in raw_routes.items())
    return Policy(
        routes=routes,
        upstream=str(data.get("upstream", Policy.upstream)),
        log_path=str(data.get("log_path", Policy.log_path)),
        pricing=_build(Pricing, data.get("pricing"), "pricing"),
    )


def load(path: str | Path | None) -> Policy:
    if path is None:
        return Policy()
    return parse(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
