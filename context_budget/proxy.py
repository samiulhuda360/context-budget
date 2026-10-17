"""The OpenAI-compatible proxy.

Apps point their base URL at `http://host:8787/v1`. For each chat completion the proxy picks a route, compresses
the messages, forwards the request to the upstream provider with the caller's own Authorization header, and
streams the reply straight back. Every request is logged as one JSON line: tokens before and after, and what each
strategy saved. Other `/v1/*` calls (models, embeddings) pass through unchanged.

Extra endpoints for people:
- `GET /` the before/after page;
- `POST /ctxb/analyse` compress without forwarding (what the page and `ctxb analyse` show);
- `GET /ctxb/recent` the latest proxied requests; `GET /ctxb/samples` the benchmark conversations.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from importlib import resources
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route as HttpRoute

from . import bench
from .analysis import explain
from .pipeline import Result, compress
from .policy import STRATEGIES, Policy, Route
from .strategies.stale import make_model_summariser
from .tokens import count_text

HOP_HEADERS = {"host", "content-length", "connection", "accept-encoding", "transfer-encoding"}
MOCK = "mock"


def page() -> str:
    return resources.files("context_budget").joinpath("web/index.html").read_text(encoding="utf-8")


def mock_completion(body: dict[str, Any], result: Result) -> dict[str, Any]:
    """A stand-in upstream for demos and tests: it reports what it received."""
    text = f"[mock upstream] received {len(result.messages)} messages, {result.tokens_after:,} tokens ({result.saved_pct:.0f}% smaller than sent)."
    return {
        "id": "ctxb-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": result.tokens_after, "completion_tokens": 20, "total_tokens": result.tokens_after + 20},
    }


class Proxy:
    def __init__(self, policy: Policy, upstream: str | None = None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.policy = policy
        self.upstream = (upstream or os.getenv("CTXB_UPSTREAM") or policy.upstream).rstrip("/")
        self.recent: deque[dict[str, Any]] = deque(maxlen=50)
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0), transport=transport)
        self.log_path = Path(os.getenv("CTXB_LOG", policy.log_path))
        count_text("warm up")  # load the tokenizer once at start-up, not on the first request

    def forward_headers(self, request: Request) -> dict[str, str]:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS and not k.lower().startswith("x-ctxb")}
        key = os.getenv("CTXB_UPSTREAM_KEY")
        if key and "authorization" not in {k.lower() for k in headers}:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def summariser(self, request: Request, route: Route) -> Any:
        if route.stale.summariser != "model" or self.upstream == MOCK:
            return None
        headers = self.forward_headers(request)

        def chat(messages: list[dict[str, str]]) -> str:
            r = httpx.post(
                f"{self.upstream}/chat/completions",
                headers=headers,
                json={"model": route.stale.summary_model, "messages": messages, "max_tokens": 80},
                timeout=60,
            )
            r.raise_for_status()
            return str(r.json()["choices"][0]["message"]["content"])

        return make_model_summariser(chat)

    def embedder(self, request: Request, route: Route) -> Any:
        if route.retrieve.method != "embed" or self.upstream == MOCK:
            return None
        headers = self.forward_headers(request)

        def embed(texts: list[str]) -> list[list[float]]:
            r = httpx.post(f"{self.upstream}/embeddings", headers=headers, json={"model": route.retrieve.embed_model, "input": texts}, timeout=60)
            r.raise_for_status()
            return [d["embedding"] for d in r.json()["data"]]

        return embed

    def log(self, entry: dict[str, Any]) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # logging must never break a request

    async def chat_completions(self, request: Request) -> Response:
        body = await request.json()
        try:
            route = self.policy.route(request.headers.get("x-ctxb-route"), body.get("model"))
        except KeyError as e:
            return JSONResponse({"error": {"message": str(e), "type": "ctxb_route"}}, status_code=400)
        result = await run_in_threadpool(compress, body.get("messages", []), route, self.summariser(request, route), self.embedder(request, route))
        body = {**body, "messages": result.messages}
        stats = {
            "x-ctxb-route": route.name,
            "x-ctxb-tokens-before": str(result.tokens_before),
            "x-ctxb-tokens-after": str(result.tokens_after),
            "x-ctxb-saved-pct": f"{result.saved_pct:.1f}",
        }
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": body.get("model"), **result.summary()}
        self.recent.appendleft({**entry, "explain": explain(result)})
        started = time.perf_counter()
        if self.upstream == MOCK:
            self.log({**entry, "upstream_status": 200, "upstream_ms": 0})
            return JSONResponse(mock_completion(body, result), headers=stats)
        req = self.client.build_request("POST", f"{self.upstream}/chat/completions", headers=self.forward_headers(request), json=body)
        upstream = await self.client.send(req, stream=True)
        headers = {**{k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS | {"content-encoding"}}, **stats}

        async def relay() -> Any:
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                self.log({**entry, "upstream_status": upstream.status_code, "upstream_ms": round((time.perf_counter() - started) * 1000)})

        return StreamingResponse(relay(), status_code=upstream.status_code, headers=headers)

    async def passthrough(self, request: Request) -> Response:
        if self.upstream == MOCK:
            return JSONResponse({"object": "list", "data": [{"id": "mock", "object": "model"}]})
        path = request.path_params["path"]
        req = self.client.build_request(
            request.method, f"{self.upstream}/{path}", headers=self.forward_headers(request), content=await request.body(), params=request.query_params
        )
        r = await self.client.send(req)
        headers = {k: v for k, v in r.headers.items() if k.lower() not in HOP_HEADERS | {"content-encoding"}}
        return Response(r.content, status_code=r.status_code, headers=headers)

    async def analyse(self, request: Request) -> Response:
        body = await request.json()
        messages = body.get("messages")
        if not isinstance(messages, list):
            return JSONResponse({"error": 'send {"messages": [...]}'}, status_code=400)
        try:
            route = self.policy.route(body.get("route"), body.get("model"))
        except KeyError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        if "strategies" in body or "budget" in body:
            route = route.with_strategies(body.get("strategies", list(route.enabled)), int(body["budget"]) if body.get("budget") else None)
        result = await run_in_threadpool(compress, messages, route)
        return JSONResponse(explain(result))

    async def recent_requests(self, request: Request) -> Response:
        return JSONResponse([{k: v for k, v in r.items() if k != "explain"} | {"n": i} for i, r in enumerate(self.recent)])

    async def recent_one(self, request: Request) -> Response:
        n = int(request.path_params["n"])
        if n >= len(self.recent):
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(self.recent[n]["explain"])

    async def samples(self, request: Request) -> Response:
        return JSONResponse([{"id": c["id"], "family": c["family"], "question": c["question"]} for c in bench.load()])

    async def sample(self, request: Request) -> Response:
        for c in bench.load():
            if c["id"] == request.path_params["id"]:
                return JSONResponse(c)
        return JSONResponse({"error": "not found"}, status_code=404)

    async def index(self, request: Request) -> Response:
        return HTMLResponse(page())

    async def config(self, request: Request) -> Response:
        return JSONResponse({"strategies": list(STRATEGIES), "routes": [r.describe() for r in self.policy.routes], "upstream": self.upstream})

    async def health(self, request: Request) -> Response:
        return JSONResponse({"ok": True})

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                HttpRoute("/", self.index),
                HttpRoute("/healthz", self.health),
                HttpRoute("/ctxb/config", self.config),
                HttpRoute("/ctxb/analyse", self.analyse, methods=["POST"]),
                HttpRoute("/ctxb/recent", self.recent_requests),
                HttpRoute("/ctxb/recent/{n:int}", self.recent_one),
                HttpRoute("/ctxb/samples", self.samples),
                HttpRoute("/ctxb/samples/{id}", self.sample),
                HttpRoute("/v1/chat/completions", self.chat_completions, methods=["POST"]),
                HttpRoute("/v1/{path:path}", self.passthrough, methods=["GET", "POST"]),
            ]
        )


def create_app(policy: Policy | None = None, upstream: str | None = None, transport: httpx.AsyncBaseTransport | None = None) -> Starlette:
    return Proxy(policy or Policy(), upstream, transport).app()
