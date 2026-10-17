"""Strategy 4: keep only the earlier turns that matter to the current question.

When a request is still over budget after trimming, earlier turns are ranked by relevance to the latest user
message, and the least relevant are dropped (whole turns, so tool calls stay paired with their results). The most
recent turns and the `top_k` best matches are always kept.

Relevance is BM25 by default: a classic keyword-ranking formula that needs no model and runs in microseconds.
With `method: embed`, turns are ranked by cosine similarity of embeddings from an OpenAI-compatible
/embeddings endpoint instead.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

from ..terms import stem, words

Embedder = Callable[[list[str]], list[list[float]]]


def _terms(text: str) -> list[str]:
    return [stem(w) for w in words(text) if len(w) > 2]


def bm25_scores(query: str, docs: Sequence[str], k1: float = 1.4, b: float = 0.75) -> list[float]:
    tokenised = [_terms(d) for d in docs]
    n = len(docs)
    if n == 0:
        return []
    avg = sum(len(t) for t in tokenised) / n or 1.0
    df: Counter[str] = Counter()
    for t in tokenised:
        df.update(set(t))
    q = set(_terms(query))
    scores = []
    for t in tokenised:
        tf = Counter(t)
        s = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            f = tf[term]
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(t) / avg))
        scores.append(s)
    return scores


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def embed_scores(query: str, docs: Sequence[str], embed: Embedder) -> list[float]:
    vectors = embed([query, *[d[:8000] for d in docs]])
    return [cosine(vectors[0], v) for v in vectors[1:]]
