"""Turn a performance description into a target node (or a miss).

Near-ties within `spread` of the best are sampled with a softmax so the same request can
land on either of two equally good nodes; spread = 0 gives deterministic top-1.
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass, field
from .index import ClipIndex


@dataclass
class Match:
    status: str                 # hit | miss | empty
    node: str | None
    similarity: float
    clip: str | None
    hits: list[dict] = field(default_factory=list)


class Matcher:
    def __init__(self, index, tau: float = 0.45, spread: float = 0.02, sample_temperature: float = 0.01, rng=None):
        self.index, self.tau, self.spread, self.temp = index, tau, spread, sample_temperature
        self.rng = rng or random.Random()

    def match(self, description: str, k: int = 5) -> Match:
        if not description.strip():
            return Match("empty", None, 0.0, None)
        hits = self.index.query_nodes(description, k=k)
        if not hits:
            return Match("empty", None, 0.0, None)
        top = hits[0]["similarity"]
        if top < self.tau:
            return Match("miss", hits[0]["node"], top, hits[0]["clip"], hits)
        pool = [h for h in hits if top - h["similarity"] <= self.spread] if self.spread > 0 else hits[:1]
        if len(pool) > 1 and self.temp > 0:
            w = [math.exp((h["similarity"] - top) / self.temp) for h in pool]
            chosen = self.rng.choices(pool, weights=w, k=1)[0]
        else:
            chosen = pool[0]
        return Match("hit", chosen["node"], chosen["similarity"], chosen["clip"], hits)
