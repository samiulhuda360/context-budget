"""Writes eval/offline.json, eval/model.json and the readable eval/RESULTS.md."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..paths import EVAL

FAMILY_NAMES = {
    "log_incident": "Incident logs",
    "db_rows": "Database rows",
    "web_research": "Web pages",
    "file_listing": "File listings and config",
    "early_constraint": "Constraint stated early",
    "stale_reference": "Old result asked about later",
    "duplicate_docs": "Repeated documents",
    "ci_failure": "CI build logs",
}


TIMING_KEYS = {"elapsed_ms", "latency_ms_p50", "latency_ms_p95", "proxy_ms", "proxy_ms_p50", "proxy_ms_p95"}


def without_timing(value: Any) -> Any:
    """Drops the machine-dependent timings so a rerun can be compared with the committed results."""
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def matches_committed(name: str, data: dict[str, Any]) -> bool:
    committed = load_json(name)
    return committed is not None and without_timing(json.loads(json.dumps(data))) == without_timing(committed)


def save_json(name: str, data: dict[str, Any]) -> Path:
    EVAL.mkdir(parents=True, exist_ok=True)
    path = EVAL / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


def load_json(name: str) -> dict[str, Any] | None:
    path = EVAL / name
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _row(cells: list[Any]) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(headers: list[str], rows: list[list[Any]], align: str = "") -> list[str]:
    sep = [("---:" if (align[i : i + 1] or "l") == "r" else "---") for i in range(len(headers))]
    return [_row(headers), _row(sep), *(_row(r) for r in rows)]


def headline(offline: dict[str, Any], tolerance: float = 2.0) -> dict[str, Any]:
    """The largest saving whose evidence kept stays within `tolerance` points of the untrimmed request."""
    base = offline["baseline_evidence_kept_pct"]
    sweep = [c for c in offline["configs"] if not c["naive"] and set(c["strategies"]) == {"trim", "stale", "dedup", "retrieve"}]
    sweep.sort(key=lambda c: -c["budget"])
    ok = [c for c in sweep if c["evidence_kept_pct"] >= base - tolerance]
    broke = [c for c in sweep if c["evidence_kept_pct"] < base - tolerance]
    best = max(ok, key=lambda c: c["saved_pct"])
    first_break = max(broke, key=lambda c: c["budget"]) if broke else None
    return {"best": best, "break": first_break, "sweep": sweep, "base": base}


def write_results() -> Path:
    off = load_json("offline.json")
    mod = load_json("model.json")
    if off is None:
        raise SystemExit("run `ctxb eval offline` first")
    h = headline(off)
    lines = ["# Evaluation results", "", "All numbers below come from `ctxb eval offline` and `ctxb eval model` and are rebuilt by `ctxb eval report`.", ""]
    lines += ["## Headline", ""]
    if mod:
        cfg = {c["config"]: c for c in mod["configs"]}
        d, b = cfg["context-budget@8000"], cfg["baseline"]
        w = cfg["window@2000"]
        judge = f" (judge {d['judge_score_pct']}% vs {b['judge_score_pct']}%)" if d["judge_score_pct"] is not None else ""
        lines += [
            f"- With every strategy on and an 8,000-token budget, requests to {mod['model']} were **{d['prompt_tokens_saved_pct']}% smaller** "
            f"(prompt tokens as Gemini reported them), and answers passed the checker **{d['checker_accuracy_pct']}%** of the time against "
            f"**{b['checker_accuracy_pct']}%** for the untrimmed requests{judge}, on {mod['conversations']} conversations.",
            f"- At almost the same saving ({w['prompt_tokens_saved_pct']}%), a plain sliding window passed **{w['checker_accuracy_pct']}%**.",
        ]
    best, brk = h["best"], h["break"]
    lines.append(
        f"- Over all {off['conversations']} conversations, the facts each answer needs stayed in the request in {best['evidence_kept_pct']}% of cases "
        f"(untrimmed: {h['base']}%) with **{best['saved_pct']}% fewer tokens** (budget {best['budget']:,})."
    )
    if brk:
        lost = sorted({FAMILY_NAMES[f] for f, v in brk["evidence_kept_by_family"].items() if v < 100})
        worst = min(h["sweep"], key=lambda c: c["budget"])
        lines.append(
            f"- It starts to break at a {brk['budget']:,}-token budget ({brk['saved_pct']}% saved, facts kept {brk['evidence_kept_pct']}%, lost in: "
            f"{', '.join(lost).lower()}) and clearly breaks at {worst['budget']:,} ({worst['saved_pct']}% saved, facts kept {worst['evidence_kept_pct']}%)."
        )
    lines.append("")

    if mod:
        lines += ["## Answer quality with a real model", ""]
        scope = (
            f"{mod['conversations']} conversations"
            if mod.get("complete", True)
            else f"{mod['conversations']} of the {mod['planned_conversations']} planned conversations (those with recorded replies so far; "
            "`ctxb eval model` records the rest and resumes from the cache)"
        )
        lines += [
            f"{scope} were each sent {len(mod['configs'])} ways to `{mod['model']}`: untrimmed, compressed at three budgets with BM25 "
            "relevance, compressed at two budgets with Jev relevance (`+jev`), and cut by a sliding window. Identical requests share one "
            "recorded reply. "
            'The checker passes an answer when every labelled fact is in it. "Tool call" is how often the model asked to run another '
            "tool instead of answering, which in a live agent means another round trip. Differences are in percentage points against "
            "the untrimmed request, with a 95% paired bootstrap interval.",
            "",
        ]
        has_judge = mod["configs"][0]["judge_score_pct"] is not None
        rows = []
        for c in mod["configs"]:
            ci = c["checker_diff_95ci_pts"]
            rows.append(
                [
                    c["config"],
                    f"{c['avg_prompt_tokens_reported']:,}",
                    f"{c['prompt_tokens_saved_pct']}%",
                    f"{c['checker_accuracy_pct']}%",
                    "" if c["config"] == "baseline" else f"{c['checker_diff_vs_baseline_pts']:+} [{ci[0]:+}, {ci[1]:+}]",
                    *([f"{c['judge_score_pct']}%"] if has_judge else []),
                    f"{c['tool_call_instead_of_answer_pct']}%",
                    f"${c['cost_per_1000_usd']['flash_lite']:.3f}",
                    f"${c['cost_per_1000_usd']['frontier']:.2f}",
                    f"{c['model_latency_s_mean']} s",
                ]
            )
        lines += _table(
            [
                "Request",
                "Avg prompt tokens",
                "Saved",
                "Checker",
                "vs untrimmed (pts)",
                *(["Judge"] if has_judge else []),
                "Tool call",
                "Cost / 1,000 (Flash-Lite)",
                "Cost / 1,000 (frontier)",
                "Model latency",
            ],
            rows,
            "lrrrrrrrrr",
        )
        lines += ["", "Checker pass rate by family:", ""]
        fams = list(mod["configs"][0]["by_family_checker_pct"])
        lines += _table(
            ["Request", *[FAMILY_NAMES[f] for f in fams]],
            [[c["config"], *[f"{c['by_family_checker_pct'][f]}%" for f in fams]] for c in mod["configs"]],
            "l" + "r" * len(fams),
        )
        jev_rows = [c for c in mod["configs"] if c.get("relevance") == "jev"]
        if jev_rows:
            cfg_by = {c["config"]: c for c in mod["configs"]}
            lines += ["", "Jev against BM25 at the same budget (paired by conversation):", ""]
            rows = []
            for c in jev_rows:
                b = cfg_by[c["bm25_row"]]
                ci = c["checker_diff_vs_bm25_95ci_pts"]
                rows.append(
                    [
                        f"{c['config']} vs {b['config']}",
                        f"{c['same_request_as_bm25']} of {c['n']}",
                        f"{b['checker_accuracy_pct']}% / {c['checker_accuracy_pct']}%",
                        f"{c['checker_diff_vs_bm25_pts']:+} [{ci[0]:+}, {ci[1]:+}]",
                        *([f"{b['judge_score_pct']}% / {c['judge_score_pct']}%", f"{c['judge_diff_vs_bm25_pts']:+}"] if has_judge else []),
                        f"{b['avg_prompt_tokens_reported']:,} / {c['avg_prompt_tokens_reported']:,}",
                    ]
                )
            lines += _table(
                [
                    "Rows",
                    "Same request and answer",
                    "Checker BM25 / Jev",
                    "Jev vs BM25 (pts)",
                    *(["Judge BM25 / Jev", "Judge diff (pts)"] if has_judge else []),
                    "Prompt tokens BM25 / Jev",
                ],
                rows,
                "lrrrrrr",
            )
        cal = mod["calibration"]
        lines += ["", "### Hand labels and the judge", ""]
        if cal.get("labelled"):
            distinct = cal.get("distinct_answers", cal["labelled"])
            scope = (
                f"All {cal['labelled']} distinct answers were"
                if cal["labelled"] >= distinct
                else f"{cal['labelled']} of the {distinct} distinct answers (every answer in the first four rows) were"
            )
            lines.append(
                f"{scope} read and scored by hand (0 wrong or missing, 1 partly right, 2 right and complete). "
                f"The checker agrees with the hand labels on full marks or not for {cal['checker_agreement_pct']}% of them (Cohen's kappa {cal['checker_kappa']})."
            )
            if "judge_kappa" in cal:
                lines.append(
                    f"The model judge agrees exactly with the hand labels for {cal['judge_exact_agreement_pct']}% (kappa {cal['judge_kappa']}), "
                    f"and on full marks or not for {cal['judge_full_marks_agreement_pct']}%."
                )
            else:
                lines.append(
                    "The model judge (batches of 16 answers, same 0/1/2 rubric) runs after all answers are recorded, so it has not run on this partial set yet."
                )
        lines.append("")

    lines += ["## Every strategy combination (no model, all conversations)", ""]
    lines += [
        f"{off['conversations']} conversations, {off['avg_tokens_before']:,} tokens on average before compression. "
        '"Facts kept" is the share of conversations whose labelled evidence is still in the compressed request. '
        f"Cost assumes a {off['output_tokens_assumed']}-token reply at the stated list prices.",
        "",
    ]
    rows = []
    for c in off["configs"]:
        if c["naive"] or c["budget"] != 8000:
            continue
        rows.append(
            [
                c["config"],
                f"{c['saved_pct']}%",
                f"{c['evidence_kept_pct']}%",
                f"${c['cost_per_1000']['flash_lite']:.3f}",
                f"${c['cost_per_1000']['frontier']:.2f}",
                f"{c['latency_ms_p50']} / {c['latency_ms_p95']}",
            ]
        )
    lines += _table(
        ["Strategies (budget 8,000)", "Tokens saved", "Facts kept", "Cost / 1,000 (Flash-Lite)", "Cost / 1,000 (frontier)", "Time p50 / p95 (ms)"],
        rows,
        "lrrrrr",
    )
    lines += ["", "## Budget sweep: where it breaks", ""]
    rows = []
    windows = {c["budget"]: c for c in off["configs"] if c["naive"]}
    for c in h["sweep"]:
        w = windows.get(c["budget"])
        rows.append(
            [
                f"{c['budget']:,}",
                f"{c['saved_pct']}%",
                f"{c['evidence_kept_pct']}%",
                f"{w['saved_pct']}%" if w else "",
                f"{w['evidence_kept_pct']}%" if w else "",
            ]
        )
    lines += _table(
        ["Budget (tokens)", "context-budget saved", "context-budget facts kept", "Sliding window saved", "Sliding window facts kept"], rows, "rrrrr"
    )
    lines += ["", "Facts kept by family:", ""]
    pick = [c for c in h["sweep"] if c["budget"] in (8000, 2000, 1000)] + [windows[b] for b in (8000, 2000) if b in windows]
    fams = list(off["configs"][0]["evidence_kept_by_family"])
    lines += _table(
        ["Configuration", *[FAMILY_NAMES[f] for f in fams]],
        [[c["config"], *[f"{c['evidence_kept_by_family'][f]}%" for f in fams]] for c in pick],
        "l" + "r" * len(fams),
    )
    rel = off.get("relevance_comparison") or []
    if rel:
        by = {c["config"]: c for c in rel}
        lines += ["", "## Relevance: BM25 or Jev (no answering model, all conversations)", ""]
        lines += [
            "The retrieve step drops the earlier turns least relevant to the latest question. Here it ranks them with BM25 "
            'or asks Jev (TypeSafe\'s decision model) one yes/no question per earlier message. "Retrieve only" runs that step '
            'alone; "full pipeline" runs it after dedup, trim and stale, which is how the proxy is used. Jev cost is at '
            "US$0.042 per million input tokens; its time is the extra wait per request that called it.",
            "",
        ]
        rows = []
        for c in rel:
            if c["relevance"] != "jev":
                continue
            b = by[c["config"].replace("+jev@", "+bm25@")]
            rows.append(
                [
                    "retrieve only" if c["strategies"] == ["retrieve"] else "full pipeline",
                    f"{c['budget']:,}",
                    f"{b['evidence_kept_pct']}% / {c['evidence_kept_pct']}%",
                    f"{b['saved_pct']}% / {c['saved_pct']}%",
                    f"+{len(c['kept_where_bm25_lost'])} / -{len(c['lost_where_bm25_kept'])}",
                    f"{c['requests_calling_jev']} ({c['jev_calls']} call{'s' if c['jev_calls'] != 1 else ''})",
                    f"{c['jev_input_tokens']:,}",
                    f"${c['jev_cost_per_1000_requests_usd']:.3f}",
                    f"{c['jev_added_ms_p50']:.0f} / {c['jev_added_ms_p95']:.0f}",
                ]
            )
        lines += _table(
            [
                "Configuration",
                "Budget",
                "Facts kept BM25 / Jev",
                "Tokens saved BM25 / Jev",
                "Conversations won / lost by Jev",
                "Requests calling Jev",
                "Jev input tokens",
                "Jev cost / 1,000 requests",
                "Jev time p50 / p95 (ms)",
            ],
            rows,
            "lrrrrrrrr",
        )
        changed = sorted({i for c in rel if c["relevance"] == "jev" for i in c["kept_where_bm25_lost"] + c["lost_where_bm25_kept"]})
        if changed:
            lines += ["", f"Conversations where the two disagree on facts kept: {', '.join(changed)}."]
    lines += [
        "",
        "## How it is measured",
        "",
        "- **Benchmark:** 80 invented conversations in 8 families (`context_budget/bench/`), rebuilt from fixed seeds; `ctxb bench --check` proves they match `eval/bench-manifest.json`.",
        "- **Tokens:** counted with the `o200k_base` tokenizer; the model table also shows the prompt tokens Gemini reported for each request.",
        "- **Facts kept:** a deterministic check that every labelled evidence string (or a computed summary of it, such as a count) is still in the request.",
        "- **Checker:** every labelled answer fact must appear in the model's reply; numbers must match whole.",
        "- **Judge:** once every answer is recorded, the same model grades batches of 16 answers with a 0/1/2 rubric, and its agreement with the hand labels is reported.",
        "- **Hand labels:** `eval/hand_labels.json`, one 0/1/2 score per distinct answer, read against the reference answer.",
        "- **Proxy time:** wall-clock time of the compression step on one laptop core (Python 3.13); model latency is Gemini's response time.",
        "- **Jev:** `jev-latest` (answered by jev-1.13.0), one Noul question per earlier message, every reply recorded in `cache/jev`; "
        "token counts and times are the ones Jev reported and measured when recorded.",
        f"- **Prices:** {off['prices']['flash_lite']}; {off['prices']['frontier']}.",
        "",
    ]
    path = EVAL / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path
