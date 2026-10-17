import json

from context_budget.policy import TrimSettings
from context_budget.strategies import dedup, retrieve, stale, trim
from context_budget.terms import Query, stem


def test_stem_groups_word_forms():
    assert stem("failed") == stem("failure") == stem("fails") == "fail"
    assert stem("files") == "file"
    assert stem("boxes") == "box"
    assert stem("access") == "access"


def test_query_keeps_identifiers_and_names():
    q = Query.from_text("Who placed ORD-58213, and did Emma get a refund?")
    assert "ord-58213" in q.identifiers
    assert "Emma" in q.names
    assert q.matches("2026-09-14 refund issued for ORD-58213")


def test_narrow_drops_terms_on_every_line():
    lines = [f"INFO [payments-worker] request {i}" for i in range(100)] + ["WARN acquirer timeout"]
    q = Query.from_text("why were payments stuck at the acquirer?").narrow(lines)
    assert "acquirer" in q.stems
    assert stem("payments") not in q.stems


def test_trim_text_keeps_head_tail_and_the_needle():
    lines = [f"2026-09-14T02:00:{i % 60:02d}Z INFO [api] GET /x/{i} 200 {i}ms" for i in range(800)]
    lines[400] = "2026-09-14T02:06:40Z ERROR [api] pool exhausted on orders-db-3"
    out, note = trim.trim("\n".join(lines), Query.from_text("what caused the errors?"), TrimSettings())
    assert note
    assert "pool exhausted on orders-db-3" in out
    assert lines[0] in out and lines[-1] in out
    assert "lines collapsed" in out or "lines cut" in out
    assert len(out) < len("\n".join(lines)) / 5


def test_collapse_repeats_folds_runs_but_not_matches():
    lines = [f"health check ok uptime={i}s" for i in range(50)] + ["refund failed for INV-1"]
    out = trim.collapse_repeats(lines, Query.from_text("which refund failed?"))
    assert len(out) == 4
    assert "48 similar lines collapsed" in out[1]
    assert out[-1] == "refund failed for INV-1"


def test_short_output_is_untouched():
    text = "a short tool result"
    assert trim.trim(text, Query.from_text("anything"), TrimSettings()) == (text, "")


def test_json_trim_keeps_matches_and_summary():
    rows = [{"order_id": f"ORD-{i}", "status": "refunded" if i % 10 == 0 else "delivered", "total": 10.0} for i in range(300)]
    out, _ = trim.trim(json.dumps(rows, indent=2), Query.from_text("which orders were refunded?"), TrimSettings())
    data = json.loads(out)
    assert data[0]["_summary_over_all_items"]["status"]["refunded"] == {"count": 30, "total_sum": 300.0}
    kept_ids = {r["order_id"] for r in data[1:]}
    assert {f"ORD-{i}" for i in range(0, 300, 10)} <= kept_ids
    assert len(data) < 50


def test_html_becomes_text_without_scripts_or_nav():
    html = "<html><head><script>var x=1;</script><style>.a{}</style></head><body><nav><a>Home</a></nav><h1>Tent</h1><table><tr><th>Weight</th><td>2.1 kg</td></tr></table><footer>Copyright</footer></body></html>"
    text = trim.html_to_text(html)
    assert "Weight | 2.1 kg" in text
    assert "var x" not in text and "Home" not in text and "Copyright" not in text


def test_kind_detection():
    assert trim.kind_of('[{"a": 1}]') == "json"
    assert trim.kind_of("<!doctype html><html></html>") == "html"
    assert trim.kind_of("plain log line") == "text"
    assert trim.kind_of("[not json") == "text"


def test_stale_summary_keeps_record_fields():
    rec = json.dumps({"account_id": "ACB-1", "renewal_date": "2026-11-02", "devices": [1, 2, 3]})
    line = stale.rules_summary("get_customer", '{"name": "Emma"}', rec, 900)
    assert "ACB-1" in line and "2026-11-02" in line and "devices (3 items)" in line
    assert line.startswith("[stale result of get_customer(")


def test_stale_reference_detection():
    q = Query.from_text("What was Emma Clarke's renewal date?")
    assert stale.is_referenced(q, "get_customer", '{"name": "Emma Clarke"}', "{}")
    assert not stale.is_referenced(q, "run_command", '{"command": "df -h"}', "Filesystem Size")


def test_stale_indices_counts_user_turns_after():
    msgs = [{"role": "system"}, {"role": "user"}, {"role": "tool"}, {"role": "user"}, {"role": "tool"}, {"role": "user"}]
    assert stale.stale_indices(msgs, 2) == [2]


def test_dedup_hash_ignores_whitespace_and_case():
    assert dedup.norm_hash("Hello   World\n") == dedup.norm_hash("hello world")


def test_bm25_ranks_the_relevant_turn_first():
    docs = ["Show me tents", "My budget is NZ$1,850 and delivery by 14 March", "Check disk on web-3"]
    scores = retrieve.bm25_scores("what budget did I give you?", docs)
    assert scores.index(max(scores)) == 1
    assert retrieve.bm25_scores("x", []) == []


def test_embed_scores_use_cosine():
    def fake(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "budget" in t else [0.0, 1.0] for t in texts]

    assert retrieve.embed_scores("budget?", ["tents", "budget"], fake) == [0.0, 1.0]
