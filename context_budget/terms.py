"""Turning the current question into search terms.

The trimmers keep lines that mention what the user is asking about, so they need the question's words in a form
that matches log lines and JSON fields: lower case, light suffix stripping ("failed", "failure" and "fails" all
become "fail"), and identifiers kept whole ("INV-20931", "pod/api-7d9f", "order_id").
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    me my no not of on or our please so than that the their them then there these they this to was we were what
    when where which who why will with would you your yes ok okay just also any all about again earlier above
    tell show give find look looked check checked use used get got need want know let one two out up down over
    still now more most some such only very much many make made like said say see seen time last first next
    should shall may might must am being each other same than too via per vs""".split()
)

WORD = re.compile(r"[A-Za-z][A-Za-z0-9']*")
IDENT = re.compile(r"\b(?=[A-Za-z0-9_./:-]*\d)[A-Za-z0-9][A-Za-z0-9_./:-]{2,}\b|\b[a-z]+_[a-z_]+\b")
QUOTED = re.compile(r"[\"'`]([^\"'`]{3,60})[\"'`]")
CAPITALISED = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")

SUFFIXES = ("ations", "ation", "ings", "ing", "ures", "ure", "ies", "ed", "es", "s")


@functools.lru_cache(maxsize=100_000)
def stem(word: str) -> str:
    w = word.lower().strip("'")
    for suf in SUFFIXES:
        if len(w) - len(suf) >= 3 and w.endswith(suf):
            if suf == "es" and not w.endswith(("ses", "xes", "zes", "ches", "shes")):
                continue  # "files" -> "file" (via "s"), but "boxes" -> "box"
            if suf == "s" and w.endswith("ss"):
                return w  # "access", "class"
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def words(text: str) -> list[str]:
    return [m.group(0) for m in WORD.finditer(text)]


@functools.lru_cache(maxsize=200_000)
def stems(text: str) -> frozenset[str]:
    return frozenset(stem(w) for w in words(text) if w.lower() not in STOPWORDS and len(w) > 2)


@dataclass(frozen=True)
class Query:
    """Search terms taken from the latest user message (and the tool call being answered)."""

    stems: frozenset[str]
    identifiers: frozenset[str]  # matched as case-insensitive substrings
    names: frozenset[str]  # capitalised words such as people and products

    @classmethod
    def from_text(cls, text: str) -> Query:
        idents = {m.group(0).lower().strip(".:/-") for m in IDENT.finditer(text)}
        idents |= {m.group(1).lower() for m in QUOTED.finditer(text)}
        idents = {i for i in idents if len(i) >= 3}
        names = {m.group(0) for m in CAPITALISED.finditer(text) if m.group(0).lower() not in STOPWORDS}
        return cls(frozenset(stems(text)), frozenset(idents), frozenset(names))

    def merge(self, other: Query) -> Query:
        return Query(self.stems | other.stems, self.identifiers | other.identifiers, self.names | other.names)

    def narrow(self, lines: list[str], max_share: float = 0.2) -> Query:
        """Drops terms that appear in more than `max_share` of the lines.

        In a log where every line says "[payments-worker]", the word "payment" says nothing about which lines
        matter, so it would only fill the kept lines with noise (the same idea as inverse document frequency).
        """
        if len(lines) < 20:
            return self
        limit = max(3, int(len(lines) * max_share))
        line_stems = [stems(ln) for ln in lines]
        lows = [ln.lower() for ln in lines]
        keep_stems = frozenset(s for s in self.stems if sum(1 for ls in line_stems if s in ls) <= limit)
        keep_ids = frozenset(i for i in self.identifiers if sum(1 for ln in lows if i in ln) <= limit)
        keep_names = frozenset(n for n in self.names if sum(1 for ln in lows if n.lower() in ln) <= limit)
        return Query(keep_stems, keep_ids, keep_names)

    def matches(self, line: str) -> bool:
        low = line.lower()
        if any(i in low for i in self.identifiers):
            return True
        return bool(self.stems & stems(line))

    def strong_hit(self, text: str) -> bool:
        """True when the text mentions an identifier or a name from the question (not just a common word)."""
        low = text.lower()
        return any(i in low for i in self.identifiers) or any(n.lower() in low for n in self.names)
