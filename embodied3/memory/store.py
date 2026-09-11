"""The memory RAG: one record per interaction, embedded for retrieval, tagged with when it happened and how she felt.

A separate store from the clip RAG on purpose (different filters, writers and pruning). Records link to the episode
log by time only. Tags per the report: the mood point at encoding, the reading of her emotion line, the body node,
a surprise placeholder (no perception yet), and `refs`, the number of times the memory was retrieved later, which is
the ground truth the archival reward will train on.

Retrieval is adaptive rather than top-k: candidates above an absolute similarity floor AND within a margin of the
best candidate, at most `max_items`. A question about the past pulls several; small talk pulls none.
"""
from __future__ import annotations
import time
from datetime import datetime
from pathlib import Path

Vec = tuple[float, float, float]


def when_str(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%a %d %b %Y %H:%M")


def ago_str(t: float, now: float | None = None) -> str:
    d = max(0.0, (time.time() if now is None else now) - t)
    if d < 60:
        return "just now"
    if d < 3600:
        return f"{int(d // 60)} min ago"
    if d < 86400:
        h = d / 3600
        return f"{h:.0f} h ago" if h >= 2 else "an hour ago"
    days = d / 86400
    return f"{days:.0f} days ago" if days >= 2 else "yesterday"


class MemoryStore:
    COLLECTION = "memories"

    def __init__(self, memory_dir: str | Path, embedder, floor: float = 0.62, margin: float = 0.10, max_items: int = 5, candidates: int = 8):
        import chromadb
        self.dir = Path(memory_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db = chromadb.PersistentClient(path=str(self.dir))
        self.coll = self.db.get_or_create_collection(self.COLLECTION, metadata={"hnsw:space": "cosine"})
        self.embedder = embedder
        self.floor, self.margin, self.max_items, self.candidates = floor, margin, max_items, candidates

    def _embed(self, texts):
        return self.embedder.encode(list(texts), normalize_embeddings=True).tolist()

    @staticmethod
    def document(user: str, reply: str, emotion: str) -> str:
        return f"User said: {user}\nShe replied: {reply}\nShe felt: {emotion}"

    def count(self) -> int:
        return self.coll.count()

    def write(self, user: str, reply: str, emotion: str, mood: Vec, reading: Vec | None, body: str, source: str = "user",
              t: float | None = None, session: str = "") -> str:
        t = time.time() if t is None else t
        mid = f"m{int(t * 1000)}"
        meta = {"t": float(t), "when": when_str(t), "user": user[:500], "reply": reply[:500], "emotion": emotion[:300],
                "v": float(mood[0]), "a": float(mood[1]), "d": float(mood[2]),
                "rv": float(reading[0]) if reading else 0.0, "ra": float(reading[1]) if reading else 0.0, "rd": float(reading[2]) if reading else 0.0,
                "has_reading": bool(reading), "body": body or "", "source": source, "surprise": 0.0, "refs": 0, "session": session}
        doc = self.document(user, reply, emotion)
        self.coll.add(documents=[doc], embeddings=self._embed([doc]), ids=[mid], metadatas=[meta])
        return mid

    NEGATIVE_V = -0.4       # a memory written in a clearly unpleasant mood
    FINE_V = 0.3            # ...is held to a tighter margin when the current mood is clearly pleasant

    def retrieve(self, query: str, exclude: set[str] | None = None, now: float | None = None, mood_valence: float | None = None) -> list[dict]:
        """Adaptive retrieval. Returns records (chronological) with similarity and the relative time.
        `mood_valence` enables the congruence gate: when she is fine, an old hurt has to match twice as closely to
        come back, so recall does not re-inject feelings that were not asked about (the mirror-loop trap)."""
        if not query.strip() or self.coll.count() == 0:
            return []
        n = min(self.candidates + len(exclude or ()), self.coll.count())
        res = self.coll.query(query_embeddings=self._embed([query]), n_results=n, include=["metadatas", "distances", "documents"])
        cands = []
        for mid, meta, dist, doc in zip(res["ids"][0], res["metadatas"][0], res["distances"][0], res["documents"][0]):
            if exclude and mid in exclude:
                continue
            cands.append({"id": mid, "similarity": 1.0 - dist, "doc": doc, **meta})
        if not cands:
            return []
        top = max(c["similarity"] for c in cands)
        keep = []
        for c in cands:
            margin = self.margin
            if mood_valence is not None and mood_valence > self.FINE_V and c.get("v", 0.0) < self.NEGATIVE_V:
                margin = self.margin / 2
                c["gated"] = True
            if c["similarity"] >= self.floor and c["similarity"] >= top - margin:
                keep.append(c)
        keep = sorted(keep, key=lambda c: -c["similarity"])[: self.max_items]
        for c in keep:
            c["ago"] = ago_str(c["t"], now)
        return sorted(keep, key=lambda c: c["t"])

    def bump_refs(self, ids: list[str]) -> None:
        if not ids:
            return
        got = self.coll.get(ids=ids, include=["metadatas"])
        metas = []
        for m in got["metadatas"]:
            m = dict(m)
            m["refs"] = int(m.get("refs", 0)) + 1
            m["last_ref_t"] = time.time()
            metas.append(m)
        self.coll.update(ids=got["ids"], metadatas=metas)

    def recent(self, n: int = 20) -> list[dict]:
        got = self.coll.get(include=["metadatas"])
        recs = [{"id": i, **m} for i, m in zip(got["ids"], got["metadatas"])]
        return sorted(recs, key=lambda r: -r["t"])[:n]
