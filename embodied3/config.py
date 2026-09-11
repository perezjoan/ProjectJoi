"""Configuration for v3: one dataclass, loaded from a JSON file, overridable from the environment (EMBODIED3_*)."""
from __future__ import annotations
import json, os
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Config:
    # data
    lexicon_file: str = "NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"
    videos_dir: str = "graph/videos"                                  # clips
    descriptions_file: str = "graph/videos/clip_descriptions.json"   # keyframes (seed of the space) + clips (the library)
    keyframes_dir: str = "graph/keyframes"
    index_dir: str = "graph/kb"                                       # the clip RAG (Chroma); shared with v1, same docs
    persona_file: str = "graph/persona.md"
    instructions_file: str = "graph/instructions_v3.md"
    episode_log: str = "graph/kb/episodes_v3.jsonl"
    # brain
    backend: str = "qwen"                   # "qwen" (HF, 4-bit) or "echo" (no model)
    model_id: str = "Qwen/Qwen3-8B"
    temperature: float = 0.7
    max_new_tokens: int = 400
    history_turns: int = 8
    seed: int = 0                           # 0 = a fresh random seed per call; set for reproducible runs
    # affect
    min_hits: int = 2                       # two real feeling words beat a seed-snapped placement (owner's call, 2026-09-10)
    placement_fallback: bool = True
    embedder: str = "BAAI/bge-small-en-v1.5"   # one embedder, on CPU, shared by the clip RAG and the placement fallback
    mood_baseline: str = "n001"             # the rest sample (neutral, with its `feeling` field), or "center" of the box
    mood_half_life_s: float = 120.0
    mood_nudge: float = 0.5
    mood_nudge_by_intensity: bool = True
    mood_clip: float = 1.5
    temperament_half_life_s: float = 259200.0   # 3 days: the baseline is an average of where the mood has been
    temperament_max_shift: float = 0.6          # ...clamped to this radius around the rest sample (the persona's clamp)
    recall_congruence: bool = True          # strongly negative memories need a closer match when the mood is fine
    context_budget_tokens: int = 8192       # for the KV gauge only
    # memory (the memory RAG, separate from the clip RAG) and persistence
    memory_dir: str = "graph/memory"        # Chroma store of one record per interaction
    memory_floor: float = 0.66              # below this nothing is recalled: bge-small scores unrelated short texts 0.62-0.66
    memory_margin: float = 0.12             # ...and only within this of the best candidate
    memory_max: int = 5                     # at most this many memories in the prompt
    memory_candidates: int = 8
    state_file: str = "graph/kb/state_v3.json"   # her self between runs: history, mood + clock, body node, timestamps
    wake_min_gap_s: float = 60.0            # on restart, tell her she was away if the gap is at least this long
    # body
    tau: float = 0.70                       # clip RAG similarity below this = miss (bge-small; re-pick with check_retrieval.py)
    spread: float = 0.02                    # near-tie margin for sampling among top nodes
    sample_temperature: float = 0.01
    hub_node: str = "n001"
    move_budget: float = 2.0                # a move costs distance(mood, target) in the standardised space; over budget = drift
    drift_check_seconds: float = 60.0       # during silence, walk to the node nearest the mood if it changed (0 = off)
    drift_hysteresis: float = 0.2           # ...only if that node is this much nearer than where the body stands
    # agent / server
    tick_seconds: float = 1.0
    idle_prompt_seconds: float = 180
    idle_prompt_min_gap: float = 300
    host: str = "127.0.0.1"
    port: int = 8767

    PATH_FIELDS = ("lexicon_file", "videos_dir", "descriptions_file", "keyframes_dir", "index_dir", "persona_file",
                   "instructions_file", "episode_log", "memory_dir", "state_file")

    @classmethod
    def load(cls, path: str | os.PathLike | None) -> "Config":
        """Relative paths in the config file are resolved against the config file's folder, so a whole agent folder
        (code + data) can be copied or renamed and still find its own lexicon, clips, persona and memory."""
        cfg = cls()
        base = Path(path).resolve().parent if path else Path.cwd()
        if path:
            if not Path(path).exists():
                raise FileNotFoundError(f"config file not found: {path}")
            with open(path, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
        for k in asdict(cfg):
            env = os.environ.get("EMBODIED3_" + k.upper())
            if env is not None:
                cur = getattr(cfg, k)
                setattr(cfg, k, env.lower() in ("1", "true") if isinstance(cur, bool) else type(cur)(env))
        for k in cls.PATH_FIELDS:
            v = getattr(cfg, k)
            if v and not Path(v).is_absolute():
                setattr(cfg, k, str((base / v).resolve()))
        return cfg
