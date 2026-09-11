"""The VAD space and its painted samples (the seed keyframes).

Coordinates are computed, not guessed: each seed keyframe is scored by the lexicon from its emotional text
(label + aliases + the expression clause of its description), and the space is standardised against the seed so
that the seed spans about [-1, 1] on every axis. Every later reading goes through `standardise` into the same frame.
The hand-set valence/arousal in clip_descriptions.json are kept only as a sanity check (see `hand_correlation`).
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from pathlib import Path
from .lexicon import Lexicon, Reading, Vec

AXES = ("valence", "arousal", "dominance")


@dataclass
class Sample:
    node: str
    label: str
    file: str
    text: str                       # the emotional text the lexicon scored
    raw: Vec | None
    coords: Vec | None = None       # standardised
    hits: int = 0
    hand: dict = field(default_factory=dict)   # {"v":..,"a":..} from the JSON, for comparison only


def seed_text(entry: dict) -> str:
    """The emotional text of a keyframe. If the entry has a `feeling` field (emotional language written for the
    lexicon), that is used alone. Otherwise: label, aliases and the expression clause; symbols and hands are physical."""
    if entry.get("feeling"):
        return str(entry["feeling"])
    m = re.search(r"expression:\s*(.*?)(?:\.\s*symbols|\.\s*hands|$)", entry.get("description", ""), re.S)
    expr = m.group(1).strip() if m else ""
    return f"{entry.get('label', entry['node'])}, {entry.get('aliases', '')}. {expr}"


class Space:
    def __init__(self, lexicon: Lexicon, samples: list[Sample], clip: float = 1.5, baseline_node: str | None = None):
        self.lex = lexicon
        self.samples = samples
        self.clip = clip
        scored = [s.raw for s in samples if s.raw is not None]
        if len(scored) < 2:
            self.center, self.scale = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        else:
            lo = tuple(min(r[k] for r in scored) for k in range(3))
            hi = tuple(max(r[k] for r in scored) for k in range(3))
            self.center = tuple((lo[k] + hi[k]) / 2 for k in range(3))
            self.scale = tuple(max((hi[k] - lo[k]) / 2, 1e-3) for k in range(3))
        for s in samples:
            s.coords = self.standardise(s.raw) if s.raw is not None else None
        self.by_node = {s.node: s for s in samples}
        base = self.by_node.get(baseline_node) if baseline_node else None
        self.baseline: Vec = base.coords if base and base.coords else (0.0, 0.0, 0.0)

    @classmethod
    def from_descriptions(cls, lexicon: Lexicon, descriptions_file: str | Path, clip: float = 1.5, baseline_node: str | None = None) -> "Space":
        with open(descriptions_file, encoding="utf-8") as f:
            d = json.load(f)
        samples = []
        for k in d["keyframes"]:
            txt = seed_text(k)
            r = lexicon.score(txt)
            samples.append(Sample(node=k["node"], label=k.get("label", k["node"]), file=k.get("file", k["node"] + ".png"),
                                  text=txt, raw=r.raw, hits=r.hits, hand={"v": k.get("valence"), "a": k.get("arousal")}))
        return cls(lexicon, samples, clip, baseline_node)

    def standardise(self, raw: Vec) -> Vec:
        return tuple(max(-self.clip, min(self.clip, (raw[k] - self.center[k]) / self.scale[k])) for k in range(3))

    def read(self, text: str) -> tuple[Reading, Vec | None]:
        """Score a text and return (reading, standardised coords or None if the reading failed)."""
        r = self.lex.score(text)
        return r, (self.standardise(r.raw) if r.ok and r.raw is not None else None)

    @staticmethod
    def distance(a: Vec, b: Vec) -> float:
        return sum((a[k] - b[k]) ** 2 for k in range(3)) ** 0.5

    def nearest(self, p: Vec) -> tuple[Sample | None, float]:
        best, bd = None, float("inf")
        for s in self.samples:
            if s.coords is None:
                continue
            d = self.distance(p, s.coords)
            if d < bd:
                best, bd = s, d
        return best, bd

    def hand_correlation(self) -> dict:
        """Pearson r between the lexicon coordinate and the hand value, per axis that has hand values (v, a)."""
        out = {}
        for k, key in ((0, "v"), (1, "a")):
            xs = [(s.coords[k], s.hand.get(key)) for s in self.samples if s.coords and s.hand.get(key) is not None]
            if len(xs) < 3:
                continue
            a = [x for x, _ in xs]
            b = [y for _, y in xs]
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
            den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
            out[AXES[k]] = num / den if den else 0.0
        return out

    def table(self) -> list[dict]:
        return [{"node": s.node, "label": s.label, "file": s.file, "coords": s.coords, "hand": s.hand, "hits": s.hits, "text": s.text}
                for s in self.samples]
