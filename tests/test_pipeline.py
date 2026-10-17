import json

import pytest

from context_budget import bench
from context_budget.messages import PIN_FIELD, text_of, turns
from context_budget.pipeline import compress
from context_budget.policy import Route
from context_budget.tokens import count_messages


def big_log(n: int = 1500) -> str:
    return "\n".join(f"2026-09-14T02:{i // 60 % 60:02d}:{i % 60:02d}Z INFO [api] GET /x/{i} 200 {i % 90}ms" for i in range(n))


def conversation() -> list[dict]:
    return [
        {"role": "system", "content": "You are an ops assistant."},
        {"role": "user", "content": "Budget is NZ$1,850. Check the logs."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "fetch_logs", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": big_log()},
        {"role": "user", "content": "Now check disk."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "run_command", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": big_log(400)},
        {"role": "user", "content": "Keep this note: the release is frozen.", PIN_FIELD: True},
        {"role": "user", "content": "What budget did I give you?"},
    ]


def valid_pairing(msgs: list[dict]) -> bool:
    open_calls: set[str] = set()
    for m in msgs:
        for c in m.get("tool_calls") or []:
            open_calls.add(c["id"])
        if m.get("role") == "tool" and m["tool_call_id"] not in open_calls:
            return False
    return True


def test_pinned_messages_are_never_changed():
    msgs = conversation()
    r = compress(msgs, Route(budget_tokens=300))
    assert r.messages[0] == msgs[0]
    assert r.messages[-1] == msgs[-1]
    assert any(text_of(m) == "Keep this note: the release is frozen." for m in r.messages)
    assert all(PIN_FIELD not in m for m in r.messages)
    assert {rep.pinned for rep in r.reports if rep.pinned} == {"system prompt", "latest user message", "marked must-keep by the app"}


def test_saves_tokens_and_reports_by_strategy():
    r = compress(conversation(), Route())
    assert r.tokens_after < r.tokens_before / 4
    assert r.saved_by()["trim"] > 0
    assert r.summary()["saved_pct"] == round(r.saved_pct, 1)
    assert r.tokens_after == count_messages(r.messages)


def test_turns_and_pairing_survive_a_tight_budget():
    r = compress(conversation(), Route(budget_tokens=300))
    assert valid_pairing(r.messages)
    assert r.saved_by()["retrieve"] > 0 or r.saved_by()["stale"] > 0


def test_nothing_enabled_changes_nothing():
    msgs = conversation()
    r = compress(msgs, Route(enabled=()))
    assert r.tokens_before == r.tokens_after
    assert [text_of(m) for m in r.messages] == [text_of(m) for m in msgs]


def test_input_is_not_mutated():
    msgs = conversation()
    before = json.dumps(msgs, sort_keys=True)
    compress(msgs, Route(budget_tokens=300))
    assert json.dumps(msgs, sort_keys=True) == before


def test_dedup_keeps_the_latest_copy():
    doc = "Returns policy. Opened items can be returned within 30 days. " * 10
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "search_docs", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": doc},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "b", "type": "function", "function": {"name": "search_docs", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "b", "content": doc},
        {"role": "user", "content": "How many days?"},
    ]
    r = compress(msgs, Route(enabled=("dedup",)))
    assert "duplicate removed" in text_of(r.messages[3])
    assert text_of(r.messages[5]) == doc


def test_content_parts_keep_non_text_parts():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "a", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": big_log()}, image]},
        {"role": "user", "content": "what happened?"},
    ]
    r = compress(msgs, Route(enabled=("trim",)))
    assert image in r.messages[3]["content"]


def test_turns_split_on_user_messages():
    msgs = [{"role": "system"}, {"role": "user"}, {"role": "assistant"}, {"role": "user"}, {"role": "tool"}]
    assert turns(msgs) == [[0], [1, 2], [3, 4]]


@pytest.mark.parametrize("conv", bench.load()[::9], ids=lambda c: c["id"])
def test_benchmark_conversations_stay_valid(conv):
    r = compress(conv["messages"], Route(budget_tokens=2000))
    assert valid_pairing(r.messages)
    assert r.messages[-1] == conv["messages"][-1]
    for rep in r.reports:
        assert rep.tokens_after <= rep.tokens_before
