"""Clip and keyframe records, loaded from clip_descriptions.json next to the videos.

A clip is an edge: first frame = `start` still, last frame = `end` still.
`index_docs` are the only texts embedded (motion + end expression, end expression, aliases).
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field
from pathlib import Path

FNAME_RE = re.compile(r"^(loop|edge|act)__([a-z0-9_]+?)(?:__([a-z0-9_]+?))?(?:__v(\d+))?\.mp4$")
KIND_ALIASES = {"transition": "edge", "return": "edge", "transition_slow": "edge", "idle": "loop"}


def parse_filename(name: str) -> dict | None:
    """loop__angry__v1.mp4 -> {kind: loop, start: angry, end: angry, variant: 1}
    edge__neutral__angry__v1.mp4 -> {kind: edge, start: neutral, end: angry, variant: 1}
    act__neutral__wave__v1.mp4 -> {kind: act, start: neutral, end: neutral, act: wave, variant: 1}"""
    m = FNAME_RE.match(name)
    if not m:
        return None
    kind, a, b, v = m.groups()
    if kind == "loop":
        return {"kind": kind, "start": a, "end": a, "variant": int(v or 1)}
    if kind == "edge":
        if b is None:
            return None
        return {"kind": kind, "start": a, "end": b, "variant": int(v or 1)}
    return {"kind": kind, "start": a, "end": a, "act": b or "", "variant": int(v or 1)}


@dataclass
class Keyframe:
    node: str
    file: str
    valence: float
    arousal: float
    description: str
    aliases: str = ""
    label: str = ""             # human label; `node` is the id


@dataclass
class Clip:
    id: str
    file: str
    kind: str                   # loop | edge | act
    start: str
    end: str
    duration_s: float
    description: str
    index_docs: list[str] = field(default_factory=list)
    aliases: str = ""
    origin: str = "recorded"    # recorded | generated | composite | reversed
    grade: float | None = None
    fields: dict = field(default_factory=dict)


@dataclass
class Library:
    keyframes: dict[str, Keyframe]
    clips: dict[str, Clip]
    videos_dir: Path
    keyframes_dir: Path

    def clip_path(self, clip_id: str) -> Path:
        return self.videos_dir / self.clips[clip_id].file

    def missing_files(self) -> list[str]:
        return [c.file for c in self.clips.values() if not (self.videos_dir / c.file).exists()]

    def by_kind(self, kind: str) -> list[Clip]:
        return [c for c in self.clips.values() if c.kind == kind]


def load_library(videos_dir: str | Path, keyframes_dir: str | Path, descriptions_file: str = "clip_descriptions.json") -> Library:
    """`descriptions_file` is a name inside videos_dir, or a full path. Hand-set valence/arousal are optional in v3:
    the space gives every node its coordinates from the lexicon."""
    videos_dir, keyframes_dir = Path(videos_dir), Path(keyframes_dir)
    desc = Path(descriptions_file)
    with open(desc if desc.is_absolute() or desc.exists() else videos_dir / descriptions_file, encoding="utf-8") as f:
        d = json.load(f)
    kfs = {k["node"]: Keyframe(node=k["node"], file=k["file"], valence=float(k.get("valence", 0.0)), arousal=float(k.get("arousal", 0.0)),
                                description=k["description"], aliases=k.get("aliases", ""), label=k.get("label", k["node"]))
           for k in d["keyframes"]}
    clips = {}
    for c in d["clips"]:
        kind = KIND_ALIASES.get(c["kind"], c["kind"])
        clips[c["id"]] = Clip(id=c["id"], file=c["file"], kind=kind, start=c["start"], end=c["end"],
                              duration_s=float(c["duration_s"]), description=c["description"],
                              index_docs=list(c.get("index_docs") or [c["description"]]),
                              aliases=c.get("aliases", ""), origin=c.get("origin", "recorded"),
                              grade=c.get("grade"), fields=c.get("fields", {}))
    return Library(keyframes=kfs, clips=clips, videos_dir=videos_dir, keyframes_dir=keyframes_dir)
