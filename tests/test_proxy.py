import json

import httpx
from starlette.testclient import TestClient

from context_budget.policy import parse
from context_budget.proxy import create_app

from .test_pipeline import conversation


def upstream_recorder() -> tuple[httpx.MockTransport, list[dict]]:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        seen.append({"url": str(request.url), "auth": request.headers.get("authorization"), "body": body})
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {"prompt_tokens": 1}})
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    return httpx.MockTransport(handler), seen


def test_chat_completion_is_compressed_and_forwarded(tmp_path, monkeypatch):
    monkeypatch.setenv("CTXB_LOG", str(tmp_path / "log.jsonl"))
    transport, seen = upstream_recorder()
    client = TestClient(create_app(upstream="https://upstream.example/v1", transport=transport))
    msgs = conversation()
    r = client.post("/v1/chat/completions", json={"model": "gemini-flash-lite-latest", "messages": msgs}, headers={"Authorization": "Bearer caller-key"})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "ok"
    sent = seen[0]
    assert sent["url"] == "https://upstream.example/v1/chat/completions"
    assert sent["auth"] == "Bearer caller-key"
    assert len(json.dumps(sent["body"]["messages"])) < len(json.dumps(msgs)) / 3
    assert all("ctxb_pin" not in m for m in sent["body"]["messages"])
    assert int(r.headers["x-ctxb-tokens-after"]) < int(r.headers["x-ctxb-tokens-before"])
    entry = json.loads((tmp_path / "log.jsonl").read_text().splitlines()[0])
    assert entry["saved_by"]["trim"] > 0 and entry["upstream_status"] == 200


def test_route_header_selects_a_route(tmp_path, monkeypatch):
    monkeypatch.setenv("CTXB_LOG", str(tmp_path / "log.jsonl"))
    policy = parse({"routes": {"raw": {"strategies": []}, "default": {}}})
    transport, _seen = upstream_recorder()
    client = TestClient(create_app(policy, "https://u.example/v1", transport))
    msgs = conversation()
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": msgs}, headers={"X-Ctxb-Route": "raw"})
    assert r.headers["x-ctxb-route"] == "raw"
    assert r.headers["x-ctxb-tokens-after"] == r.headers["x-ctxb-tokens-before"]
    assert client.post("/v1/chat/completions", json={"model": "m", "messages": msgs}, headers={"X-Ctxb-Route": "nope"}).status_code == 400


def test_mock_upstream_and_recent_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("CTXB_LOG", str(tmp_path / "log.jsonl"))
    client = TestClient(create_app(upstream="mock"))
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": conversation()})
    assert "[mock upstream]" in r.json()["choices"][0]["message"]["content"]
    recent = client.get("/ctxb/recent").json()
    assert recent[0]["tokens_after"] < recent[0]["tokens_before"]
    detail = client.get("/ctxb/recent/0").json()
    assert any(s["op"] == "cut" for m in detail["messages"] for s in m["segments"])
    assert client.get("/ctxb/recent/9").status_code == 404


def test_analyse_endpoint_and_page():
    client = TestClient(create_app(upstream="mock"))
    r = client.post("/ctxb/analyse", json={"messages": conversation(), "strategies": ["trim"], "budget": 4000})
    data = r.json()
    assert data["enabled"] == ["trim"] and data["budget"] == 4000
    assert data["saved_by"]["trim"] > 0 and data["saved_by"]["stale"] == 0
    assert client.post("/ctxb/analyse", json={"nope": 1}).status_code == 400
    assert "context-budget" in client.get("/").text
    assert client.get("/healthz").json() == {"ok": True}
    samples = client.get("/ctxb/samples").json()
    assert len(samples) == 80
    assert client.get(f"/ctxb/samples/{samples[0]['id']}").json()["messages"]
    assert client.get("/ctxb/samples/none").status_code == 404


def test_other_v1_calls_pass_through():
    transport, seen = upstream_recorder()
    client = TestClient(create_app(upstream="https://u.example/v1", transport=transport))
    r = client.get("/v1/models", headers={"Authorization": "Bearer k"})
    assert r.json()["data"][0]["id"] == "m"
    assert seen[0]["url"] == "https://u.example/v1/models"
