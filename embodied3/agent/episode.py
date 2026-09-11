"""Append-only episode log: one JSON line per record. Everything that learns later reads this."""
from __future__ import annotations
import json, time
from pathlib import Path


class EpisodeLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields) -> None:
        t = time.time()
        rec = {"t": t, "when": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)), "kind": kind, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
