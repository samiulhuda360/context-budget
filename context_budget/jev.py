"""A small client for Jev, TypeSafe AI's typed decision model, with a disk cache.

Jev answers typed questions about a piece of text (the "state"): a Noul question returns the probability that a
yes/no statement is true. context-budget uses it to decide which earlier messages the latest question needs.

Configuration (environment variables; nothing is stored in files):
    TYPESAFE_API_KEY   the key; without it, calls replay from the cache or fail with JevError
    JEV_BASE_URL       default: https://api.typesafe.ai/v1

Every reply is cached under cache/jev by a hash of (model, state, questions). Text is normalised before hashing
(Unicode NFC, Windows and old-Mac line endings turned into "\\n"), and the JSON is serialised with sorted keys and
ASCII escapes, so the same request has the same key on every OS and Python version. The evaluation and CI replay
from this cache and never call the API.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .paths import CACHE

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"
DEFAULT_MODEL = "jev-latest"
PRICE_PER_MILLION_INPUT = 0.042  # US$; output tokens are free


class JevError(RuntimeError):
    """Any failure to get an answer: no key and no cached reply, a timeout, an HTTP error or a malformed reply."""


def normalise(value: Any) -> Any:
    """The form that is hashed and sent: NFC text with "\\n" line endings, recursively."""
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    if isinstance(value, dict):
        return {normalise(k): normalise(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [normalise(v) for v in value]
    return value


def cache_key(model: str, state: Any, questions: dict[str, Any]) -> str:
    blob = json.dumps(normalise({"model": model, "state": state, "questions": questions}), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("ascii")).hexdigest()[:32]


@dataclass
class JevReply:
    answers: dict[str, Any]
    model: str
    input_tokens: int
    latency_s: float
    cached: bool

    def noul(self, name: str) -> float:
        answer = self.answers.get(name)
        if not isinstance(answer, dict) or not isinstance(answer.get("noul"), int | float):
            raise JevError(f"reply has no Noul answer for {name!r}")
        return float(answer["noul"])


@dataclass
class Jev:
    model: str = DEFAULT_MODEL
    base_url: str = field(default_factory=lambda: os.getenv("JEV_BASE_URL", DEFAULT_BASE_URL))
    cache_dir: Path = field(default_factory=lambda: CACHE / "jev")
    offline: bool = False  # replay from the cache only
    timeout_s: float = 20.0
    retries: int = 3  # for 429, 529 and 5xx replies, with backoff
    live_calls: int = 0
    cached_calls: int = 0

    @property
    def live(self) -> bool:
        return not self.offline and bool(os.getenv("TYPESAFE_API_KEY"))

    def ask(self, state: Any, questions: dict[str, Any]) -> JevReply:
        state, questions = normalise(state), normalise(questions)
        path = self.cache_dir / f"{cache_key(self.model, state, questions)}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.cached_calls += 1
            return JevReply(data["answers"], data["model"], int(data["input_tokens"]), float(data["latency_s"]), True)
        if not self.live:
            raise JevError("no TYPESAFE_API_KEY and no cached Jev reply")
        reply = self._call(state, questions)
        missing = set(questions) - set(reply.answers)
        if missing:
            raise JevError(f"Jev reply is missing {len(missing)} answer(s)")
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"model": reply.model, "input_tokens": reply.input_tokens, "latency_s": reply.latency_s, "answers": reply.answers}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return reply

    def _call(self, state: Any, questions: dict[str, Any]) -> JevReply:
        body = {"model": self.model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}", "Content-Type": "application/json"}
        url = self.base_url.rstrip("/") + "/systemone"
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                r = httpx.post(url, json=body, headers=headers, timeout=self.timeout_s)
            except httpx.HTTPError as e:
                if attempt < self.retries:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise JevError(f"Jev request failed: {type(e).__name__}") from e
            latency = round(time.perf_counter() - started, 3)
            if (r.status_code in (429, 529) or r.status_code >= 500) and attempt < self.retries:
                wait = r.headers.get("retry-after", "")
                time.sleep(min(30.0, float(wait)) if wait.replace(".", "", 1).isdigit() else 2.0 * 2**attempt)
                continue
            if r.status_code != 200:
                raise JevError(f"Jev returned HTTP {r.status_code}: {r.text[:300]}")
            try:
                data = r.json()
                answers = dict(data["answers"])
                tokens = int((data.get("usage") or {}).get("input_tokens", 0))
            except (ValueError, KeyError, TypeError) as e:
                raise JevError("Jev reply is not the expected JSON") from e
            self.live_calls += 1
            return JevReply(answers, str(data.get("model", self.model)), tokens, latency, False)
        raise JevError("Jev retries exhausted")


def cost_usd(input_tokens: int) -> float:
    return input_tokens * PRICE_PER_MILLION_INPUT / 1_000_000
