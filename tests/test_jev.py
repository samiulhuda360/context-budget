import itertools
import json
import unicodedata

import httpx
import pytest
from starlette.testclient import TestClient

from context_budget import bench
from context_budget.evaluation import model, offline
from context_budget.jev import Jev, JevError, JevReply, cache_key
from context_budget.messages import text_of
from context_budget.pipeline import compress
from context_budget.policy import JevSettings, PolicyError, Route, parse
from context_budget.proxy import create_app
from context_budget.strategies import jev_relevance

from .test_pipeline import valid_pairing


class FakeJev(Jev):
    """Answers every Noul from a table of message ids (default 0.1) and records the requests."""

    def __init__(self, probs: dict[str, float] | None = None, fail: bool = False) -> None:
        super().__init__(offline=True)
        self.probs = probs or {}
        self.fail = fail
        self.requests: list[tuple[dict, dict]] = []

    def ask(self, state, questions):
        self.requests.append((state, questions))
        if self.fail:
            raise JevError("simulated timeout")
        return JevReply({q: {"type": "noul", "noul": self.probs.get(q, 0.1)} for q in questions}, "jev-test", 1000, 0.25, False)


def agent_conversation() -> list[dict]:
    """A budget stated in turn 1, then three unrelated lookups, then a question about the budget."""
    msgs = [{"role": "system", "content": "You are a purchasing assistant."}, {"role": "user", "content": "We can spend NZ$1,850 at most."}]
    msgs.append({"role": "assistant", "content": "Noted."})
    for k in range(3):
        msgs.append({"role": "user", "content": f"Search the catalogue for item {k}."})
        msgs.append(
            {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{k}", "type": "function", "function": {"name": "search", "arguments": "{}"}}]}
        )
        msgs.append({"role": "tool", "tool_call_id": f"c{k}", "content": "\n".join(f"result {k}.{i}: tent, 2 person, {i} kg" for i in range(150))})
        msgs.append({"role": "assistant", "content": "Here are the results."})
    msgs.append({"role": "user", "content": "Thanks. Now check the weather."})
    msgs.append({"role": "assistant", "content": "Sunny."})
    msgs.append({"role": "user", "content": "What did I say about money?"})
    return msgs


def route(**kw) -> Route:
    return Route(budget_tokens=kw.pop("budget", 400), enabled=("retrieve",), relevance="jev", **kw)


def test_cache_key_is_the_same_for_any_line_ending_and_unicode_form():
    text = "café\nline two"
    variants = [text, text.replace("\n", "\r\n"), text.replace("\n", "\r"), unicodedata.normalize("NFD", text)]
    keys = {cache_key("jev-latest", {"q": v}, {"m1": {"type": "noul", "instructions": v}}) for v in variants}
    assert len(keys) == 1
    assert cache_key("jev-latest", "a", {}) != cache_key("jev-latest", "b", {})


def test_offline_client_without_a_recording_raises(tmp_path):
    with pytest.raises(JevError):
        Jev(offline=True, cache_dir=tmp_path).ask("state", {"x": {"type": "noul", "instructions": "?"}})


def test_jev_keeps_what_it_scores_high_and_drops_the_rest():
    msgs = agent_conversation()
    jev = FakeJev({"m1": 0.97})
    r = compress(msgs, route(), jev=jev)
    texts = [text_of(m) for m in r.messages]
    assert "We can spend NZ$1,850 at most." in texts
    assert not any(t.startswith("result 0.0") for t in texts)
    assert valid_pairing(r.messages)
    assert r.jev and r.jev["calls"] == 1 and r.jev["input_tokens"] == 1000 and not r.jev["fallback"]
    assert r.summary()["jev"]["latency_ms"] == 250.0
    assert any("Jev 0.10" in a.note for rep in r.reports for a in rep.actions)
    state, questions = jev.requests[0]
    assert state["latest_question"] == "What did I say about money?"
    assert {it["id"] for it in state["earlier_messages"]} == set(questions)
    assert all(q["type"] == "noul" for q in questions.values())


def test_jev_errors_fall_back_to_bm25():
    msgs = agent_conversation()
    r = compress(msgs, route(), jev=FakeJev(fail=True))
    bm25 = compress(msgs, Route(budget_tokens=400, enabled=("retrieve",)))
    assert r.messages == bm25.messages
    assert r.jev and r.jev["fallback"] == "simulated timeout" and r.jev["calls"] == 0


def test_no_jev_call_while_the_request_fits():
    jev = FakeJev()
    r = compress(agent_conversation(), route(budget=100000), jev=jev)
    assert not jev.requests and r.jev is None


def test_long_conversations_are_split_into_chunks_and_long_messages_cut():
    msgs = agent_conversation()
    jev = FakeJev({"m1": 0.9})
    compress(msgs, Route(budget_tokens=400, enabled=("retrieve",), relevance="jev"), jev=jev)
    assert len(jev.requests) == 1
    jev = FakeJev({"m1": 0.9})
    r = compress(msgs, route(jev=JevSettings(chunk_tokens=1200, message_tokens=300)), jev=jev)
    assert len(jev.requests) > 1 and r.jev and r.jev["calls"] == len(jev.requests)
    asked = [q for _, qs in jev.requests for q in qs]
    assert len(asked) == len(set(asked))
    assert "tokens not shown" in jev_relevance.head_tail("word " * 2000, 100)
    assert jev_relevance.head_tail("short", 100) == "short"


def test_html_is_shown_to_jev_as_text():
    page = "<!doctype html><html><head><style>.a{color:red}</style></head><body><p>Weight 1.6 kg</p></body></html>"
    msgs = [{"role": "tool", "tool_call_id": "c1", "content": page}]
    item = jev_relevance.describe(msgs, 0, {"c1": "fetch_url"}, 1000)
    assert "Weight 1.6 kg" in item["text"] and "color" not in item["text"]
    assert item["from"] == "tool result of fetch_url"


def test_live_call_retries_a_rate_limit_and_records_the_reply(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr("time.sleep", lambda s: None)
    replies = [
        httpx.Response(429, headers={"retry-after": "1"}),
        httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"m1": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 42}}),
    ]
    seen = []

    def post(url, json, headers, timeout):
        seen.append((url, json, headers))
        return replies.pop(0)

    monkeypatch.setattr(httpx, "post", post)
    jev = Jev(cache_dir=tmp_path)
    q = {"m1": {"type": "noul", "instructions": "needed?"}}
    reply = jev.ask("state", q)
    assert reply.noul("m1") == 0.8 and reply.input_tokens == 42 and not reply.cached
    assert seen[0][0] == "https://api.typesafe.ai/v1/systemone" and seen[0][1]["model"] == "jev-latest"
    assert seen[0][2]["Authorization"] == "Bearer test-key"
    again = Jev(cache_dir=tmp_path, offline=True).ask("state", q)
    assert again.cached and again.noul("m1") == 0.8
    record = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert "test-key" not in json.dumps(record)


def test_http_errors_become_jev_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(400, json={"error": "max_tokens_exceeded"}))
    with pytest.raises(JevError, match="HTTP 400"):
        Jev(cache_dir=tmp_path).ask("state", {"m1": {"type": "noul", "instructions": "?"}})


def test_policy_relevance_switch():
    p = parse({"routes": {"agents": {"relevance": "jev", "jev": {"timeout_s": 5, "keep_threshold": 0.6}}, "default": {}}})
    assert p.route("agents").relevance == "jev" and p.route("agents").jev.timeout_s == 5
    assert p.route("default").relevance == "bm25"
    for bad in ({"relevance": "magic"}, {"jev": {"nope": 1}}, {"jev": {"keep_threshold": 2}}, {"jev": {"chunk_tokens": 90000}}):
        with pytest.raises(PolicyError):
            parse({"routes": {"r": bad}})


def test_proxy_logs_jev_usage_and_survives_a_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("CTXB_LOG", str(tmp_path / "log.jsonl"))
    policy = parse({"routes": {"default": {"relevance": "jev", "budget_tokens": 400, "strategies": ["retrieve"]}}})
    client = TestClient(create_app(policy, upstream="mock"))
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": agent_conversation()})
    assert r.status_code == 200
    entry = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["jev"]["fallback"].startswith("no TYPESAFE_API_KEY")


def test_evaluation_rows_replay_from_the_committed_jev_cache():
    jev = Jev(offline=True)
    for cfg in [*offline.relevance_configs()[1::2], *model.JEV_CONFIGS]:
        for conv in bench.load()[::9]:
            msgs, _ = offline.apply(cfg, conv, jev)
            rest = [m for m in msgs if m["role"] != "system"]
            assert rest[0]["role"] == "user", (cfg.name, conv["id"])
            assert valid_pairing(msgs)
            for prev, m in itertools.pairwise(msgs):
                if m.get("tool_calls"):
                    assert prev["role"] in ("user", "tool"), (cfg.name, conv["id"])
    assert jev.live_calls == 0
