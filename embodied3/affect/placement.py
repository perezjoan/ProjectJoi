"""Fallback placement: when the lexicon finds too few emotional words, place the text by embedding similarity to
the seed samples' emotional texts (a tiny index of nine strings, NOT the clip RAG). Returns a similarity-weighted
mean of the seed coordinates. Lazy: the embedder is only loaded on first use.
"""
from __future__ import annotations
import math
from .space import Space
from .lexicon import Vec


class Placer:
    def __init__(self, space: Space, embedder=None, embedder_name: str = "BAAI/bge-small-en-v1.5", temperature: float = 0.05):
        self.space, self._emb, self.name, self.temp = space, embedder, embedder_name, temperature
        self._seed_vecs = None

    def _embedder(self):
        if self._emb is None:
            from sentence_transformers import SentenceTransformer
            self._emb = SentenceTransformer(self.name, device="cpu")
        return self._emb

    def _seed(self):
        if self._seed_vecs is None:
            samples = [s for s in self.space.samples if s.coords is not None]
            vecs = self._embedder().encode([s.text for s in samples], normalize_embeddings=True)
            self._seed_vecs = (samples, vecs)
        return self._seed_vecs

    def place(self, text: str) -> tuple[Vec, list[dict]]:
        samples, vecs = self._seed()
        q = self._embedder().encode([text], normalize_embeddings=True)[0]
        sims = [float(sum(a * b for a, b in zip(q, v))) for v in vecs]
        top = max(sims)
        w = [math.exp((s - top) / self.temp) for s in sims]
        W = sum(w)
        p = tuple(sum(wi * s.coords[k] for wi, s in zip(w, samples)) / W for k in range(3))
        ranked = sorted(({"node": s.node, "label": s.label, "similarity": si} for s, si in zip(samples, sims)),
                        key=lambda h: -h["similarity"])
        return p, ranked[:3]
