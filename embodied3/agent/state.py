"""Persistence of the agent's self between runs: what makes her the same agent tomorrow.

Not the KV cache: the prompt is rebuilt every turn (mood line, body line, memories), so a saved cache would be
invalid on the next turn anyway. What persists is the state the model reads: the recent history, the mood with its
clock, the body's node, and the timestamps that let her notice how long she was away. The memory store persists on
its own (Chroma on disk).
"""
from __future__ import annotations
import json, time
from pathlib import Path


def save_state(path: str | Path, agent) -> dict:
    st = {"saved_at": time.time(), "started": agent.started, "last_interaction": agent.last_interaction,
          "history": agent.brain.history, "history_memory_ids": sorted(agent.history_memory_ids),
          "mood": {"point": agent.mood.point, "baseline": agent.mood.baseline, "last_t": agent.mood.last_t,
                   "since_change": agent.mood.since_change, "trail": agent.mood.trail[-50:], "words": list(agent.mood.words())},
          "pos": agent.graph.pos, "arrived_at": agent.arrived_at, "session": agent.session}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(p)
    return st


def load_state(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def restore_state(agent, st: dict) -> float:
    """Put the saved self back; returns the gap in seconds since it was saved. The mood's clock is set to the save
    time, so the next tick applies the decay for the whole gap (a day away settles her to the baseline)."""
    agent.brain.history = list(st.get("history", []))
    agent.history_memory_ids = set(st.get("history_memory_ids", []))
    m = st.get("mood", {})
    if m.get("point"):
        agent.mood.point = tuple(m["point"])
        if m.get("baseline"):
            agent.mood.baseline = tuple(m["baseline"])          # the temperament survives the restart
        agent.mood.last_t = float(m.get("last_t", st["saved_at"]))
        agent.mood.since_change = float(m.get("since_change", st["saved_at"]))
        agent.mood.trail = [{"t": x["t"], "p": tuple(x["p"])} for x in m.get("trail", [])]
        agent.mood._words = tuple(m.get("words") or agent.mood.words())
    if st.get("pos") in agent.lib.keyframes:
        agent.graph.pos = st["pos"]
    agent.arrived_at = float(st.get("arrived_at", time.time()))
    agent.last_interaction = float(st.get("last_interaction", st["saved_at"]))
    agent.previous_session = st.get("session", "")
    return max(0.0, time.time() - float(st["saved_at"]))
