"""The model evaluation: does the answer stay as good when the request is compressed?

A stratified subset of the benchmark (4 conversations per family, 32 in all) is sent to the model four ways:
untrimmed, compressed at the default budget, compressed aggressively, and cut by the naive sliding window at about
the same saving as the default. Each answer is scored twice:
- by the deterministic checker (are the labelled facts in the answer?), and
- by a model judge with a 0/1/2 rubric, batched so one call grades several answers.
The judge is calibrated against hand labels on a subset of the answers, and its agreement is reported.

Every model reply is cached under cache/llm, so `ctxb eval model --offline` and CI replay the run without a key.
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
JUDGE_BATCH = 12

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


def judge_batches(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Unique (conversation, answer) pairs in a fixed shuffled order, so a batch mixes questions."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for it in sorted(items, key=_order_key):
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


def run(llm: LLM, progress: Callable[[str], None] = print) -> dict[str, Any]:
    convs = subset()
    by_id = {c["id"]: c for c in bench.load()}
    items: list[dict[str, Any]] = []
    for c in convs:
        for cfg in MODEL_CONFIGS:
            msgs, ms = apply(cfg, c)
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
    scores: dict[str, int] = {}
    for k, batch in enumerate(judge_batches(items)):
        reply = llm.chat([{"role": "user", "content": judge_prompt(batch, by_id)}], max_tokens=60 * len(batch))
        scores.update(parse_scores(reply.text))
        progress(f"judge batch {k + 1}: {len(batch)} answers graded")
    for it in items:
        it["judge"] = scores[judge_key(it)]
    return summarise(items)


def summarise(items: list[dict[str, Any]]) -> dict[str, Any]:
    hand = load_hand_labels()
    base = {it["id"]: it for it in items if it["config"] == "baseline"}
    out_cfgs = []
    for cfg in MODEL_CONFIGS:
        rows = [it for it in items if it["config"] == cfg.name]
        ids = [r["id"] for r in rows]
        acc = [1.0 if r["correct"] else 0.0 for r in rows]
        jud = [r["judge"] / 2 for r in rows]
        base_acc = [1.0 if base[i]["correct"] else 0.0 for i in ids]
        base_jud = [base[i]["judge"] / 2 for i in ids]
        avg_prompt = statistics.fmean(r["prompt_tokens"] for r in rows)
        avg_out = statistics.fmean(r["completion_tokens"] for r in rows)
        base_prompt = statistics.fmean(base[i]["prompt_tokens"] for i in ids)
        lo, hi = bootstrap_diff(base_acc, acc)
        out_cfgs.append(
            {
                "config": cfg.name,
                "n": len(rows),
                "checker_accuracy_pct": round(100 * statistics.fmean(acc), 1),
                "judge_score_pct": round(100 * statistics.fmean(jud), 1),
                "judge_full_marks_pct": round(100 * sum(1 for r in rows if r["judge"] == 2) / len(rows), 1),
                "checker_diff_vs_baseline_pts": round(100 * (statistics.fmean(acc) - statistics.fmean(base_acc)), 1),
                "checker_diff_95ci_pts": [round(100 * lo, 1), round(100 * hi, 1)],
                "judge_diff_vs_baseline_pts": round(100 * (statistics.fmean(jud) - statistics.fmean(base_jud)), 1),
                "avg_tokens_counted": round(statistics.fmean(r["tokens"] for r in rows)),
                "avg_prompt_tokens_reported": round(avg_prompt),
                "prompt_tokens_saved_pct": round(100 * (1 - avg_prompt / base_prompt), 1),
                "cost_per_1000_usd": {k: round(cost_per_1000(avg_prompt, p, avg_out), 4) for k, p in PRICES.items()},
                "model_latency_s_mean": round(statistics.fmean(r["latency_s"] for r in rows), 2),
                "model_latency_s_p50": round(percentile([r["latency_s"] for r in rows], 50), 2),
                "proxy_ms_p50": round(percentile([r["proxy_ms"] for r in rows], 50), 2),
                "proxy_ms_p95": round(percentile([r["proxy_ms"] for r in rows], 95), 2),
                "by_family_checker_pct": {
                    f: round(100 * sum(1 for r in rows if r["family"] == f and r["correct"]) / sum(1 for r in rows if r["family"] == f))
                    for f in dict.fromkeys(r["family"] for r in rows)
                },
            }
        )
    calibration = calibrate(items, hand)
    return {"model": LLM().model, "conversations": len(base), "configs": out_cfgs, "calibration": calibration, "items": items}


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
    j = [p[1] for p in pairs]
    c = [2 if p[2] else 0 for p in pairs]
    h_bin = [1 if x == 2 else 0 for x in h]
    return {
        "labelled": len(pairs),
        "judge_exact_agreement_pct": round(100 * sum(1 for a, b in zip(h, j, strict=True) if a == b) / len(pairs), 1),
        "judge_kappa": round(cohen_kappa(h, j), 2),
        "judge_full_marks_agreement_pct": round(100 * sum(1 for a, b in zip(h_bin, j, strict=True) if a == (1 if b == 2 else 0)) / len(pairs), 1),
        "checker_agreement_pct": round(100 * sum(1 for a, b in zip(h_bin, c, strict=True) if a == (1 if b == 2 else 0)) / len(pairs), 1),
        "checker_kappa": round(cohen_kappa(h_bin, [1 if x == 2 else 0 for x in c]), 2),
    }
