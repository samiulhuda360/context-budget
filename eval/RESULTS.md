# Evaluation results

All numbers below come from `ctxb eval offline` and `ctxb eval model` and are rebuilt by `ctxb eval report`.

## Headline

- With every strategy on and an 8,000-token budget, requests to gemini-flash-lite-latest were **90.1% smaller** (prompt tokens as Gemini reported them), and answers passed the checker **88.2%** of the time against **82.4%** for the untrimmed requests, on 17 conversations.
- At almost the same saving (89.8%), a plain sliding window passed **17.6%**.
- Over all 80 conversations, the facts each answer needs stayed in the request in 100.0% of cases (untrimmed: 100.0%) with **88.1% fewer tokens** (budget 8,000).
- It starts to break at a 2,000-token budget (90.5% saved, facts kept 97.5%, lost in: incident logs) and clearly breaks at 1,000 (93.0% saved, facts kept 80.0%).

## Answer quality with a real model

17 of the 32 planned conversations (those with recorded replies so far; `ctxb eval model` records the rest and resumes from the cache) were each sent four ways to `gemini-flash-lite-latest`. The checker passes an answer when every labelled fact is in it. "Tool call" is how often the model asked to run another tool instead of answering, which in a live agent means another round trip. Differences are in percentage points against the untrimmed request, with a 95% paired bootstrap interval.

| Request | Avg prompt tokens | Saved | Checker | vs untrimmed (pts) | Tool call | Cost / 1,000 (Flash-Lite) | Cost / 1,000 (frontier) | Model latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 26,049 | 0.0% | 82.4% |  | 17.6% | $2.618 | $78.63 | 2.64 s |
| context-budget@8000 | 2,574 | 90.1% | 88.2% | +5.9 [-11.8, +29.4] | 11.8% | $0.268 | $8.13 | 1.86 s |
| context-budget@1000 | 1,125 | 95.7% | 41.2% | -41.2 [-70.6, -5.9] | 41.2% | $0.124 | $3.82 | 1.34 s |
| window@2000 | 2,649 | 89.8% | 17.6% | -64.7 [-88.2, -41.2] | 58.8% | $0.280 | $8.50 | 1.69 s |

Checker pass rate by family:

| Request | Incident logs | Database rows | Web pages | File listings and config | Constraint stated early |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 100% | 50% | 100% | 75% | 100% |
| context-budget@8000 | 100% | 75% | 100% | 75% | 100% |
| context-budget@1000 | 75% | 50% | 25% | 0% | 100% |
| window@2000 | 0% | 0% | 25% | 50% | 0% |

### Hand labels and the judge

All 60 distinct answers were read and scored by hand (0 wrong or missing, 1 partly right, 2 right and complete). The checker agrees with the hand labels on full marks or not for 100.0% of them (Cohen's kappa 1.0).
The model judge (batches of 16 answers, same 0/1/2 rubric) runs after all answers are recorded, so it has not run on this partial set yet.

## Every strategy combination (no model, all conversations)

80 conversations, 16,459 tokens on average before compression. "Facts kept" is the share of conversations whose labelled evidence is still in the compressed request. Cost assumes a 100-token reply at the stated list prices.

| Strategies (budget 8,000) | Tokens saved | Facts kept | Cost / 1,000 (Flash-Lite) | Cost / 1,000 (frontier) | Time p50 / p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.0% | 100.0% | $1.686 | $50.88 | 0.29 / 0.61 |
| trim | 81.7% | 100.0% | $0.341 | $10.54 | 8.8 / 31.41 |
| stale | 29.1% | 100.0% | $1.208 | $36.53 | 0.45 / 3.26 |
| dedup | 1.1% | 100.0% | $1.667 | $50.32 | 3.48 / 6.6 |
| retrieve | 6.7% | 100.0% | $1.575 | $47.55 | 1.77 / 3.6 |
| trim+stale | 87.3% | 100.0% | $0.249 | $7.78 | 7.18 / 16.4 |
| trim+dedup | 82.8% | 100.0% | $0.323 | $9.99 | 8.98 / 23.92 |
| trim+retrieve | 81.7% | 100.0% | $0.341 | $10.54 | 5.64 / 16.42 |
| stale+dedup | 29.8% | 100.0% | $1.195 | $36.15 | 3.94 / 6.61 |
| stale+retrieve | 29.2% | 100.0% | $1.205 | $36.44 | 0.87 / 3.36 |
| dedup+retrieve | 7.9% | 100.0% | $1.556 | $46.99 | 5.02 / 9.63 |
| trim+stale+dedup | 88.1% | 100.0% | $0.236 | $7.39 | 9.47 / 23.96 |
| trim+stale+retrieve | 87.3% | 100.0% | $0.249 | $7.78 | 7.02 / 17.03 |
| trim+dedup+retrieve | 82.8% | 100.0% | $0.323 | $9.99 | 8.38 / 23.86 |
| stale+dedup+retrieve | 30.0% | 100.0% | $1.192 | $36.06 | 4.51 / 7.54 |
| trim+stale+dedup+retrieve | 88.1% | 100.0% | $0.236 | $7.39 | 9.74 / 24.44 |

## Budget sweep: where it breaks

| Budget (tokens) | context-budget saved | context-budget facts kept | Sliding window saved | Sliding window facts kept |
| ---: | ---: | ---: | ---: | ---: |
| 32,000 | 87.2% | 100.0% | 2.0% | 100.0% |
| 16,000 | 87.2% | 100.0% | 23.4% | 83.8% |
| 8,000 | 88.1% | 100.0% | 53.6% | 42.5% |
| 4,000 | 88.1% | 100.0% | 75.4% | 32.5% |
| 2,000 | 90.5% | 97.5% | 87.4% | 26.2% |
| 1,000 | 93.0% | 80.0% | 93.5% | 23.8% |

Facts kept by family:

| Configuration | Incident logs | Database rows | Web pages | File listings and config | Constraint stated early | Old result asked about later | Repeated documents | CI build logs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trim+stale+dedup+retrieve | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| all@2000 | 80% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| all@1000 | 80% | 80% | 70% | 20% | 100% | 100% | 100% | 90% |
| window@8000 | 0% | 20% | 30% | 80% | 0% | 10% | 100% | 100% |
| window@2000 | 0% | 0% | 10% | 50% | 0% | 0% | 50% | 100% |

## How it is measured

- **Benchmark:** 80 invented conversations in 8 families (`context_budget/bench/`), rebuilt from fixed seeds; `ctxb bench --check` proves they match `eval/bench-manifest.json`.
- **Tokens:** counted with the `o200k_base` tokenizer; the model table also shows the prompt tokens Gemini reported for each request.
- **Facts kept:** a deterministic check that every labelled evidence string (or a computed summary of it, such as a count) is still in the request.
- **Checker:** every labelled answer fact must appear in the model's reply; numbers must match whole.
- **Judge:** once every answer is recorded, the same model grades batches of 16 answers with a 0/1/2 rubric, and its agreement with the hand labels is reported.
- **Hand labels:** `eval/hand_labels.json`, one 0/1/2 score per distinct answer, read against the reference answer.
- **Proxy time:** wall-clock time of the compression step on one laptop core (Python 3.13); model latency is Gemini's response time.
- **Prices:** Gemini Flash-Lite list price ($0.10 in / $0.40 out per million tokens); frontier-model price ($3 in / $15 out per million tokens).
