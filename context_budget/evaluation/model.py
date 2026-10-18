"""The model evaluation: does the answer stay as good when the request is compressed?

A stratified subset of the benchmark (4 conversations per family, 32 in all) is sent to the model four ways:
untrimmed, compressed at the default budget, compressed aggressively, and cut by the naive sliding window at about
the same saving as the default. Each answer is scored twice:
- by the deterministic checker (are the labelled facts in the answer?), and
- by a model judge with a 0/1/2 rubric, batched so one call grades several answers.
The judge is calibrated against hand labels on a subset of the answers, and its agreement is reported.

The Jev rows repeat the compressed runs with Jev choosing which earlier turns to keep (`relevance: jev`), next to
BM25 at the same budgets. Their answers are judged in batches of their own, so adding them leaves the original
judge prompts, and so the original rows, unchanged.

Every model reply is cached under cache/llm, and every Jev reply under cache/jev, so `ctxb eval model --offline`
and CI replay the run without a key.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
from collections.abc import Callable
from typing import Any

from .. import bench
from ..jev import Jev
from ..llm import LLM
from ..paths import EVAL
from ..policy import STRATEGIES
from ..tokens import count_messages
from .checks import answer_correct, groups_found
from .offline import PRICES, Config, apply, cost_per_1000, percentile

SUBSET_INDICES = (1, 2, 5, 8)
MODEL_CONFIGS = (
    Config("baseline", (), 10**9),
    Config("context-budget@8000", STRATEGIES, 8000),
    Config("context-budget@1000", STRATEGIES, 1000),
    Config("window@2000", (), 2000, naive=True),
)
JEV_CONFIGS = (
    Config("context-budget@2000", STRATEGIES, 2000),
    Config("context-budget+jev@8000", STRATEGIES, 8000, relevance="jev"),
    Config("context-budget+jev@2000", STRATEGIES, 2000, relevance="jev"),
)
ALL_CONFIGS = MODEL_CONFIGS + JEV_CONFIGS
BM25_TWIN = {"context-budget+jev@8000": "context-budget@8000", "context-budget+jev@2000": "context-budget@2000"}
JUDGE_BATCH = 16

JUDGE_PROMPT = """You are grading answers from an AI assistant. For each item you get the question, the reference
answer, and the assistant's answer. Score each one:
2 = correct and complete: every fact in the reference is present and right (wording may differ).
1 = partly correct: some but not all facts right, or right but hedged with a wrong alternative.
0 = wrong, missing, or the assistant says it does not have the information.
Reply with only a JSON list, one object per item, in the same order: [{"id": "...", "score": 0|1|2}].

Items:
"""


def subset() -> list[dict[str, Any]]:
    convs = bench.load()
    return [c for c in convs if int(c["id"].rsplit("-", 1)[1]) in SUBSET_INDICES]


def _order_key(item: dict[str, Any]) -> str:
    return hashlib.sha1(f"{item['id']}/{item['config']}".encode()).hexdigest()  # noqa: S324 - shuffling only


def judge_batches(items: list[dict[str, Any]], skip: set[tuple[str, str]] | None = None) -> list[list[dict[str, Any]]]:
    """Unique (conversation, answer) pairs in a fixed shuffled order, so a batch mixes questions."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for it in sorted(items, key=_order_key):
        if (it["id"], it["answer"]) not in (skip or set()):
            seen.setdefault((it["id"], it["answer"]), it)
    uniq = list(seen.values())
    return [uniq[k : k + JUDGE_BATCH] for k in range(0, len(uniq), JUDGE_BATCH)]


def judge_key(item: dict[str, Any]) -> str:
    return f"{item['id']}/{hashlib.sha1(item['answer'].encode()).hexdigest()[:10]}"  # noqa: S324


def judge_prompt(batch: list[dict[str, Any]], convs: dict[str, dict[str, Any]]) -> str:
    lines = []
    for it in batch:
        c = convs[it["id"]]
        lines.append(
            json.dumps({"id": judge_key(it), "question": c["question"], "reference": c["reference_answer"], "answer": it["answer"]}, ensure_ascii=False)
        )
    return JUDGE_PROMPT + "\n".join(lines)


def parse_scores(text: str) -> dict[str, int]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"judge reply has no JSON list: {text[:200]}")
    return {str(d["id"]): int(d["score"]) for d in json.loads(m.group(0))}


def bootstrap_diff(a: list[float], b: list[float], seed: int = 7, n: int = 2000) -> tuple[float, float]:
    """95% interval of mean(b - a) over paired items."""
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b, strict=True)]
    means = sorted(statistics.fmean(rng.choice(d) for _ in d) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def cohen_kappa(x: list[int], y: list[int]) -> float:
    labels = sorted(set(x) | set(y))
    n = len(x)
    po = sum(1 for a, b in zip(x, y, strict=True) if a == b) / n
    pe = sum((x.count(k) / n) * (y.count(k) / n) for k in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def run(llm: LLM, progress: Callable[[str], None] = print, partial: bool = False, jev: Jev | None = None) -> dict[str, Any]:
    """Answers every subset conversation seven ways, then judges the answers.

    With `partial`, conversations whose four answers are not all recorded are left out, and the judge runs only if
    every batch is recorded; the result says how many conversations it covers.
    """
    convs = subset()
    by_id = {c["id"]: c for c in bench.load()}
    items: list[dict[str, Any]] = []
    for c in convs:
        prepared = [(cfg, *apply(cfg, c, jev or Jev(offline=True))) for cfg in ALL_CONFIGS]
        if partial and not all(llm.cached(msgs, c["tools"]) for _, msgs, _ in prepared):
            continue
        for cfg, msgs, ms in prepared:
            reply = llm.chat(msgs, tools=c["tools"], max_tokens=300)
            items.append(
                {
                    "id": c["id"],
                    "family": c["family"],
                    "config": cfg.name,
                    "tokens": count_messages(msgs),
                    "prompt_tokens": reply.prompt_tokens,
                    "completion_tokens": reply.completion_tokens,
                    "latency_s": reply.latency_s,
                    "proxy_ms": round(ms, 2),
                    "answer": reply.text,
                    "correct": answer_correct(reply.text, c["answers"]),
                    "groups_found": groups_found(reply.text, c["answers"]),
                    "groups": len(c["answers"]),
                }
            )
        progress(f"{c['id']}: answered ({llm.live_calls} live calls, {llm.cached_calls} from cache)")
    first = [it for it in items if it["config"] in {c.name for c in MODEL_CONFIGS}]
    batches = judge_batches(first)
    batches += judge_batches([it for it in items if it not in first], skip={(it["id"], it["answer"]) for it in first})
    prompts = [[{"role": "user", "content": judge_prompt(batch, by_id)}] for batch in batches]
    judged = not partial or all(llm.cached(p) for p in prompts)
    scores: dict[str, int] = {}
    if judged:
        for k, (batch, prompt) in enumerate(zip(batches, prompts, strict=True)):
            reply = llm.chat(prompt, max_tokens=60 * len(batch))
            scores.update(parse_scores(reply.text))
            progress(f"judge batch {k + 1}: {len(batch)} answers graded")
    for it in items:
        it["judge"] = scores.get(judge_key(it))
    result = summarise(items)
    result["planned_conversations"] = len(convs)
    result["complete"] = len(items) == len(convs) * len(ALL_CONFIGS) and judged
    return result


def summarise(items: list[dict[str, Any]]) -> dict[str, Any]:
    hand = load_hand_labels()
    base = {it["id"]: it for it in items if it["config"] == "baseline"}
    out_cfgs = []
    by_cfg = {cfg.name: {it["id"]: it for it in items if it["config"] == cfg.name} for cfg in ALL_CONFIGS}
    for cfg in ALL_CONFIGS:
        rows = [it for it in items if it["config"] == cfg.name]
        ids = [r["id"] for r in rows]
        acc = [1.0 if r["correct"] else 0.0 for r in rows]
        has_judge = all(r["judge"] is not None for r in items)
        jud = [r["judge"] / 2 for r in rows] if has_judge else [0.0]
        base_acc = [1.0 if base[i]["correct"] else 0.0 for i in ids]
        base_jud = [base[i]["judge"] / 2 for i in ids] if has_judge else [0.0]
        avg_prompt = statistics.fmean(r["prompt_tokens"] for r in rows)
        avg_out = statistics.fmean(r["completion_tokens"] for r in rows)
        base_prompt = statistics.fmean(base[i]["prompt_tokens"] for i in ids)
        lo, hi = bootstrap_diff(base_acc, acc)
        out_cfgs.append(
            {
                "config": cfg.name,
                "n": len(rows),
                "checker_accuracy_pct": round(100 * statistics.fmean(acc), 1),
                "judge_score_pct": round(100 * statistics.fmean(jud), 1) if has_judge else None,
                "judge_full_marks_pct": round(100 * sum(1 for r in rows if r["judge"] == 2) / len(rows), 1) if has_judge else None,
                "checker_diff_vs_baseline_pts": round(100 * (statistics.fmean(acc) - statistics.fmean(base_acc)), 1),
                "checker_diff_95ci_pts": [round(100 * lo, 1), round(100 * hi, 1)],
                "judge_diff_vs_baseline_pts": round(100 * (statistics.fmean(jud) - statistics.fmean(base_jud)), 1) if has_judge else None,
                "tool_call_instead_of_answer_pct": round(100 * sum(1 for r in rows if r["answer"].startswith("[tool call]")) / len(rows), 1),
                "avg_tokens_counted": round(statistics.fmean(r["tokens"] for r in rows)),
                "tokens_counted_saved_pct": round(
                    100 * (1 - statistics.fmean(r["tokens"] for r in rows) / statistics.fmean(base[i]["tokens"] for i in ids)), 1
                ),
                "avg_prompt_tokens_reported": round(avg_prompt),
                "prompt_tokens_saved_pct": round(100 * (1 - avg_prompt / base_prompt), 1),
                "cost_per_1000_usd": {k: round(cost_per_1000(avg_prompt, p, avg_out), 4) for k, p in PRICES.items()},
                "model_latency_s_mean": round(statistics.fmean(r["latency_s"] for r in rows), 2),
                "model_latency_s_p50": round(percentile([r["latency_s"] for r in rows], 50), 2),
                "proxy_ms_p50": round(percentile([r["proxy_ms"] for r in rows], 50), 2),
                "proxy_ms_p95": round(percentile([r["proxy_ms"] for r in rows], 95), 2),
                **(vs_bm25(rows, by_cfg[BM25_TWIN[cfg.name]], has_judge) if cfg.name in BM25_TWIN else {}),
                "by_family_checker_pct": {
                    f: round(100 * sum(1 for r in rows if r["family"] == f and r["correct"]) / sum(1 for r in rows if r["family"] == f))
                    for f in dict.fromkeys(r["family"] for r in rows)
                },
            }
        )
    calibration = calibrate(items, hand)
    return {"model": LLM().model, "conversations": len(base), "configs": out_cfgs, "calibration": calibration, "items": items}


def vs_bm25(rows: list[dict[str, Any]], twin: dict[str, dict[str, Any]], has_judge: bool) -> dict[str, Any]:
    """The Jev row against BM25 at the same budget, paired by conversation."""
    acc = [1.0 if r["correct"] else 0.0 for r in rows]
    twin_acc = [1.0 if twin[r["id"]]["correct"] else 0.0 for r in rows]
    lo, hi = bootstrap_diff(twin_acc, acc)
    out: dict[str, Any] = {
        "relevance": "jev",
        "bm25_row": next(iter(twin.values()))["config"],
        "checker_diff_vs_bm25_pts": round(100 * (statistics.fmean(acc) - statistics.fmean(twin_acc)), 1),
        "checker_diff_vs_bm25_95ci_pts": [round(100 * lo, 1), round(100 * hi, 1)],
        "same_request_as_bm25": sum(1 for r in rows if r["tokens"] == twin[r["id"]]["tokens"] and r["answer"] == twin[r["id"]]["answer"]),
    }
    if has_judge:
        jud = statistics.fmean(r["judge"] / 2 for r in rows)
        out["judge_diff_vs_bm25_pts"] = round(100 * (jud - statistics.fmean(twin[r["id"]]["judge"] / 2 for r in rows)), 1)
    return out


def load_hand_labels() -> dict[str, dict[str, Any]]:
    path = EVAL / "hand_labels.json"
    if not path.exists():
        return {}
    data: dict[str, dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def calibrate(items: list[dict[str, Any]], hand: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Agreement of the judge and of the checker with the hand labels."""
    pairs = list({judge_key(it): (hand[judge_key(it)]["score"], it["judge"], it["correct"]) for it in items if judge_key(it) in hand}.values())
    if not pairs:
        return {"labelled": 0}
    h = [p[0] for p in pairs]
    c = [2 if p[2] else 0 for p in pairs]
    h_bin = [1 if x == 2 else 0 for x in h]
    out: dict[str, Any] = {
        "labelled": len(pairs),
        "distinct_answers": len({judge_key(it) for it in items}),
        "checker_agreement_pct": round(100 * sum(1 for a, b in zip(h_bin, c, strict=True) if a == (1 if b == 2 else 0)) / len(pairs), 1),
        "checker_kappa": round(cohen_kappa(h_bin, [1 if x == 2 else 0 for x in c]), 2),
    }
    if all(p[1] is not None for p in pairs):
        j = [int(p[1]) for p in pairs]
        out["judge_exact_agreement_pct"] = round(100 * sum(1 for a, b in zip(h, j, strict=True) if a == b) / len(pairs), 1)
        out["judge_kappa"] = round(cohen_kappa(h, j), 2)
        out["judge_full_marks_agreement_pct"] = round(100 * sum(1 for a, b in zip(h_bin, j, strict=True) if a == (1 if b == 2 else 0)) / len(pairs), 1)
    return out
