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
        lines += [
            f"- With every strategy on and an 8,000-token budget, requests to {mod['model']} were **{d['prompt_tokens_saved_pct']}% smaller** "
            f"(tokens Gemini itself reported), and answers scored **{d['checker_accuracy_pct']}%** on the checker against "
            f"**{b['checker_accuracy_pct']}%** for the untrimmed requests ({d['checker_diff_vs_baseline_pts']:+} points; judge "
            f"{d['judge_score_pct']}% vs {b['judge_score_pct']}%).",
            f"- At almost the same saving ({w['prompt_tokens_saved_pct']}%), a plain sliding window scored **{w['checker_accuracy_pct']}%**.",
        ]
    best, brk = h["best"], h["break"]
    lines.append(
        f"- Over all {off['conversations']} conversations, the facts each answer needs stayed in the request in {best['evidence_kept_pct']}% of cases "
        f"(untrimmed: {h['base']}%) up to a **{best['saved_pct']}% token saving** (budget {best['budget']:,})."
    )
    if brk:
        lost = sorted({FAMILY_NAMES[f] for f, v in brk["evidence_kept_by_family"].items() if v < 100})
        lines.append(
            f"- It breaks at a {brk['budget']:,}-token budget: {brk['saved_pct']}% saved but only {brk['evidence_kept_pct']}% of answers still have their facts (lost in: {', '.join(lost).lower()})."
        )
    lines.append("")

    if mod:
        lines += ["## Answer quality with a real model", ""]
        lines += [
            f"{mod['conversations']} conversations (4 from each family) were each sent four ways to `{mod['model']}`. "
            "The checker passes an answer when every labelled fact is in it; the judge scores 0, 1 or 2 against the reference answer. "
            "Differences are in percentage points against the untrimmed request, with a 95% paired bootstrap interval.",
            "",
        ]
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
                    f"{c['judge_score_pct']}%",
                    f"${c['cost_per_1000_usd']['flash_lite']:.3f}",
                    f"${c['cost_per_1000_usd']['frontier']:.2f}",
                    f"{c['model_latency_s_mean']} s",
                    f"{c['proxy_ms_p50']} / {c['proxy_ms_p95']} ms",
                ]
            )
        lines += _table(
            [
                "Request",
                "Avg prompt tokens",
                "Saved",
                "Checker",
                "vs untrimmed (pts)",
                "Judge",
                "Cost / 1,000 (Flash-Lite)",
                "Cost / 1,000 (frontier)",
                "Model latency",
                "Proxy time p50 / p95",
            ],
            rows,
            "lrrrrrrrrr",
        )
        lines += ["", "Checker accuracy by family:", ""]
        fams = list(mod["configs"][0]["by_family_checker_pct"])
        lines += _table(
            ["Request", *[FAMILY_NAMES[f] for f in fams]],
            [[c["config"], *[f"{c['by_family_checker_pct'][f]}%" for f in fams]] for c in mod["configs"]],
            "l" + "r" * len(fams),
        )
        cal = mod["calibration"]
        lines += ["", "### Judge calibration", ""]
        if cal.get("labelled"):
            lines += [
                f"{cal['labelled']} answers were labelled by hand (0/1/2, same rubric) before looking at the judge's scores.",
                "",
                f"- Judge vs hand labels: {cal['judge_exact_agreement_pct']}% exact agreement, Cohen's kappa {cal['judge_kappa']}; "
                f"{cal['judge_full_marks_agreement_pct']}% agreement on full marks or not.",
                f"- Checker vs hand labels (full marks or not): {cal['checker_agreement_pct']}% agreement, kappa {cal['checker_kappa']}.",
            ]
        else:
            lines.append("No hand labels yet.")
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
    lines += [
        "",
        "## How it is measured",
        "",
        "- **Benchmark:** 80 invented conversations in 8 families (`context_budget/bench/`), rebuilt from fixed seeds; `ctxb bench --check` proves they match `eval/bench-manifest.json`.",
        "- **Tokens:** counted with the `o200k_base` tokenizer; the model table also shows the prompt tokens Gemini reported for each request.",
        "- **Facts kept:** a deterministic check that every labelled evidence string (or a computed summary of it, such as a count) is still in the request.",
        "- **Checker:** every labelled answer fact must appear in the model's reply; numbers must match whole.",
        "- **Judge:** the same model grades batches of 12 answers with a 0/1/2 rubric; the judge's agreement with hand labels is reported above.",
        "- **Proxy time:** wall-clock time of the compression step on one laptop core (Python 3.13); model latency is Gemini's response time.",
        f"- **Prices:** {off['prices']['flash_lite']}; {off['prices']['frontier']}.",
        "",
    ]
    path = EVAL / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path
