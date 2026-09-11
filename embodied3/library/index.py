"""The clip RAG: descriptions embedded into a Chroma collection, queried by node.

Separate from any memory RAG on purpose (different filters, writers and pruning).
"""
from __future__ import annotations
from pathlib import Path
from .records import Library


class ClipIndex:
    COLLECTION = "clips"

    def __init__(self, index_dir: str | Path, embedder_name: str = "BAAI/bge-small-en-v1.5", embedder=None, device: str = "cpu"):
        import chromadb
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.db = chromadb.PersistentClient(path=str(self.index_dir))
        if embedder is None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(embedder_name, device=device)   # small model; keep VRAM for the brain
        self.embedder = embedder

    def _embed(self, texts):
        return self.embedder.encode(list(texts), normalize_embeddings=True).tolist()

    def build(self, lib: Library) -> int:
        try:
            self.db.delete_collection(self.COLLECTION)
        except Exception:
            pass
        coll = self.db.create_collection(self.COLLECTION, metadata={"hnsw:space": "cosine"})
        docs, ids, metas = [], [], []
        for c in lib.clips.values():
            for j, doc in enumerate(c.index_docs):
                docs.append(doc); ids.append(f"{c.id}#{j}")
                metas.append({"clip": c.id, "kind": c.kind, "start": c.start, "end": c.end})
        coll.add(documents=docs, embeddings=self._embed(docs), ids=ids, metadatas=metas)
        return len(docs)

    def add_clip(self, clip) -> None:
        coll = self.db.get_collection(self.COLLECTION)
        docs = clip.index_docs
        coll.add(documents=docs, embeddings=self._embed(docs), ids=[f"{clip.id}#{j}" for j in range(len(docs))],
                 metadatas=[{"clip": clip.id, "kind": clip.kind, "start": clip.start, "end": clip.end}] * len(docs))

    def query_nodes(self, description: str, k: int = 5, n_docs: int = 30) -> list[dict]:
        """Rank NODES by the best-matching document among clips that end on them."""
        coll = self.db.get_collection(self.COLLECTION)
        res = coll.query(query_embeddings=self._embed([description]), n_results=n_docs,
                         include=["metadatas", "distances", "documents"])
        best: dict[str, dict] = {}
        for meta, dist, doc in zip(res["metadatas"][0], res["distances"][0], res["documents"][0]):
            sim, node = 1.0 - dist, meta["end"]
            if node not in best or sim > best[node]["similarity"]:
                best[node] = {"node": node, "similarity": float(sim), "clip": meta["clip"], "doc": doc}
        return sorted(best.values(), key=lambda h: -h["similarity"])[:k]
