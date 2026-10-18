# Evaluation results

All numbers below come from `ctxb eval offline` and `ctxb eval model` and are rebuilt by `ctxb eval report`.

## Headline

- With every strategy on and an 8,000-token budget, requests to gemini-flash-lite-latest were **89.4% smaller** (prompt tokens as Gemini reported them), and answers passed the checker **93.8%** of the time against **90.6%** for the untrimmed requests (judge 92.2% vs 90.6%), on 32 conversations.
- At almost the same saving (88.2%), a plain sliding window passed **31.2%**.
- Over all 80 conversations, the facts each answer needs stayed in the request in 100.0% of cases (untrimmed: 100.0%) with **88.1% fewer tokens** (budget 8,000).
- It starts to break at a 2,000-token budget (90.5% saved, facts kept 97.5%, lost in: incident logs) and clearly breaks at 1,000 (93.0% saved, facts kept 80.0%).

## Answer quality with a real model

32 conversations were each sent four ways to `gemini-flash-lite-latest`. The checker passes an answer when every labelled fact is in it. "Tool call" is how often the model asked to run another tool instead of answering, which in a live agent means another round trip. Differences are in percentage points against the untrimmed request, with a 95% paired bootstrap interval.

| Request | Avg prompt tokens | Saved | Checker | vs untrimmed (pts) | Judge | Tool call | Cost / 1,000 (Flash-Lite) | Cost / 1,000 (frontier) | Model latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 22,200 | 0.0% | 90.6% |  | 90.6% | 9.4% | $2.233 | $67.10 | 2.18 s |
| context-budget@8000 | 2,355 | 89.4% | 93.8% | +3.1 [-6.2, +15.6] | 92.2% | 6.2% | $0.248 | $7.53 | 1.3 s |
| context-budget@1000 | 1,305 | 94.1% | 65.6% | -25.0 [-43.8, -6.2] | 68.8% | 21.9% | $0.143 | $4.39 | 1.23 s |
| window@2000 | 2,620 | 88.2% | 31.2% | -59.4 [-75.0, -40.6] | 31.2% | 40.6% | $0.276 | $8.39 | 1.48 s |

Checker pass rate by family:

| Request | Incident logs | Database rows | Web pages | File listings and config | Constraint stated early | Old result asked about later | Repeated documents | CI build logs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 100% | 50% | 100% | 75% | 100% | 100% | 100% | 100% |
| context-budget@8000 | 100% | 75% | 100% | 75% | 100% | 100% | 100% | 100% |
| context-budget@1000 | 75% | 50% | 25% | 0% | 100% | 100% | 100% | 75% |
| window@2000 | 0% | 0% | 25% | 50% | 0% | 0% | 75% | 100% |

### Hand labels and the judge

All 114 distinct answers were read and scored by hand (0 wrong or missing, 1 partly right, 2 right and complete). The checker agrees with the hand labels on full marks or not for 100.0% of them (Cohen's kappa 1.0).
The model judge agrees exactly with the hand labels for 99.1% (kappa 0.98), and on full marks or not for 99.1%.

## Every strategy combination (no model, all conversations)

80 conversations, 16,459 tokens on average before compression. "Facts kept" is the share of conversations whose labelled evidence is still in the compressed request. Cost assumes a 100-token reply at the stated list prices.

| Strategies (budget 8,000) | Tokens saved | Facts kept | Cost / 1,000 (Flash-Lite) | Cost / 1,000 (frontier) | Time p50 / p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.0% | 100.0% | $1.686 | $50.88 | 0.29 / 0.61 |
| trim | 81.8% | 100.0% | $0.339 | $10.48 | 9.04 / 32.63 |
| stale | 29.1% | 100.0% | $1.208 | $36.53 | 0.47 / 3.33 |
| dedup | 1.1% | 100.0% | $1.667 | $50.32 | 3.58 / 6.82 |
| retrieve | 6.7% | 100.0% | $1.575 | $47.55 | 1.98 / 3.75 |
| trim+stale | 87.4% | 100.0% | $0.248 | $7.74 | 9.25 / 20.67 |
| trim+dedup | 82.9% | 100.0% | $0.321 | $9.92 | 11.94 / 32.28 |
| trim+retrieve | 81.8% | 100.0% | $0.339 | $10.48 | 5.72 / 17.48 |
| stale+dedup | 29.8% | 100.0% | $1.195 | $36.15 | 4.68 / 7.04 |
| stale+retrieve | 29.2% | 100.0% | $1.205 | $36.44 | 0.9 / 3.49 |
| dedup+retrieve | 7.9% | 100.0% | $1.556 | $46.99 | 5.83 / 11.48 |
| trim+stale+dedup | 88.1% | 100.0% | $0.235 | $7.36 | 12.3 / 34.36 |
| trim+stale+retrieve | 87.4% | 100.0% | $0.248 | $7.74 | 7.45 / 22.43 |
| trim+dedup+retrieve | 82.9% | 100.0% | $0.321 | $9.92 | 9.41 / 25.05 |
| stale+dedup+retrieve | 30.0% | 100.0% | $1.192 | $36.06 | 4.59 / 8.29 |
| trim+stale+dedup+retrieve | 88.1% | 100.0% | $0.235 | $7.36 | 10.25 / 25.34 |

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
