import itertools
import json
from pathlib import Path

import pytest

from context_budget import bench
from context_budget.cli import main
from context_budget.evaluation import checks, model, naive, offline
from context_budget.policy import PolicyError, load, parse

from .test_pipeline import conversation, valid_pairing

ROOT = Path(__file__).resolve().parent.parent


def test_example_policy_loads_and_routes_by_model():
    p = load(ROOT / "examples" / "policy.yaml")
    assert p.route(model="gemini-flash-lite-latest").name == "agents"
    assert p.route(model="some-other-model").name == "default"
    assert p.route("raw").enabled == ()


@pytest.mark.parametrize(
    "bad",
    [
        {"routes": {"x": {"strategies": ["compress"]}}},
        {"routes": {"x": {"budget_tokens": 10}}},
        {"routes": {"x": {"trim": {"head": 3}}}},
        {"routes": {"x": {"stale": {"summariser": "magic"}}}},
        {"nonsense": 1},
    ],
)
def test_bad_policies_are_rejected(bad):
    with pytest.raises(PolicyError):
        parse(bad)


def test_defaults_merge_into_routes():
    p = parse({"defaults": {"budget_tokens": 5000, "trim": {"head_lines": 4}}, "routes": {"a": {"trim": {"tail_lines": 3}}}})
    a = p.route("a")
    assert a.budget_tokens == 5000 and a.trim.head_lines == 4 and a.trim.tail_lines == 3
    assert p.route("default").trim.head_lines == 4


def test_answer_checker_matches_whole_numbers_and_aliases():
    assert checks.answer_correct("It was refunded 7 times.", [["7"]])
    assert not checks.answer_correct("It was 17.", [["7"]])
    assert checks.answer_correct("The budget was $1,850.", [["1850"]])
    assert checks.answer_correct("Seven orders.", [["7"]])
    assert checks.answer_correct("pool exhausted on orders-db-3", [["exhaust"], ["orders-db-3"]])
    assert not checks.answer_correct("pool exhausted", [["exhaust"], ["orders-db-3"]])
    assert checks.groups_found("pool exhausted", [["exhaust"], ["orders-db-3"]]) == 1


def test_evidence_items():
    msgs = [{"role": "tool", "content": '"status": "refunded" "status": "refunded" ORD-1'}]
    assert checks.evidence_kept(msgs, ["ORD-1", {"count_of": ['"status": "refunded"'], "equals": 2}])
    assert not checks.evidence_kept(msgs, [{"count_of": ['"status": "refunded"'], "equals": 3}])
    assert checks.evidence_kept(msgs, [{"all_of": ["ORD-1", "ORD-2"], "or_any_of": ["ORD-1"]}])
    assert checks.evidence_kept(msgs, [{"any_of": ["nope", "ORD-1"]}])


def test_sliding_window_stays_valid_and_within_budget():
    msgs = conversation()
    out = naive.sliding_window(msgs, 600)
    assert out[0]["role"] == "system" and out[-1] == msgs[-1]
    assert valid_pairing(out)
    assert next(m for m in out if m["role"] != "system")["role"] == "user"


def test_requests_follow_chat_api_turn_order():
    """A tool call must follow a user message or a tool result, and the first turn must be the user's."""
    for cfg in [*offline.configs()[::4], *model.MODEL_CONFIGS]:
        for conv in bench.load()[::7]:
            msgs, _ = offline.apply(cfg, conv)
            rest = [m for m in msgs if m["role"] != "system"]
            assert rest[0]["role"] == "user", (cfg.name, conv["id"])
            for prev, m in itertools.pairwise(msgs):
                if m.get("tool_calls"):
                    assert prev["role"] in ("user", "tool"), (cfg.name, conv["id"])


def test_benchmark_matches_manifest():
    expected = json.loads((ROOT / "eval" / "bench-manifest.json").read_text(encoding="utf-8"))
    assert bench.digest(bench.build()) == expected


def test_benchmark_labels_are_consistent():
    convs = bench.load()
    assert len(convs) == 80 and len({c["id"] for c in convs}) == 80
    for c in convs:
        assert c["messages"][-1]["content"].startswith(c["question"])
        assert checks.evidence_kept(c["messages"], c["evidence"]), c["id"]
        assert checks.answer_correct(c["reference_answer"], c["answers"]), c["id"]


def test_cli_analyse_prints_table_and_writes_html(tmp_path, capsys):
    conv = tmp_path / "conv.json"
    conv.write_text(json.dumps({"messages": conversation()}), encoding="utf-8")
    html = tmp_path / "out.html"
    assert main(["analyse", str(conv), "--budget", "2000", "--html", str(html)]) == 0
    out = capsys.readouterr().out
    assert "tool:fetch_logs" in out and "saved" in out and "pinned" in out
    assert "window.CTXB_DATA" in html.read_text(encoding="utf-8")
    assert main(["analyse", str(conv), "--json", "--strategies", ""]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["tokens_before"] == data["tokens_after"]


def test_cli_check_policy(capsys):
    assert main(["check-policy", str(ROOT / "examples" / "policy.yaml")]) == 0
    assert "policy is valid" in capsys.readouterr().out


def test_judge_batches_are_stable_and_unique():
    items = [{"id": f"c{i % 5}", "config": f"k{i}", "answer": f"a{i % 7}"} for i in range(40)]
    batches = model.judge_batches(items)
    flat = [(x["id"], x["answer"]) for b in batches for x in b]
    assert len(flat) == len(set(flat))
    assert batches == model.judge_batches(list(reversed(items)))
    assert model.parse_scores('noise [{"id": "a", "score": 2}] trailing') == {"a": 2}


def test_kappa_and_bootstrap():
    assert model.cohen_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == 1.0
    lo, hi = model.bootstrap_diff([1, 1, 0, 1], [1, 1, 0, 1])
    assert lo == hi == 0.0
