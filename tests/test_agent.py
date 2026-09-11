import asyncio
import pytest
from embodied3.config import Config
from embodied3.agent import Agent
from conftest import HashEmbedder


def make_agent(workspace, **over):
    pytest.importorskip("chromadb")
    kw = dict(lexicon_file=str(workspace / "lex.txt"), videos_dir=str(workspace / "videos"),
              descriptions_file=str(workspace / "videos" / "clip_descriptions.json"), keyframes_dir=str(workspace / "keyframes"),
              index_dir=str(workspace / "kb"), episode_log=str(workspace / "kb" / "episodes.jsonl"),
              memory_dir=str(workspace / "memory"), state_file=str(workspace / "kb" / "state.json"),
              persona_file=None, instructions_file=None, backend="echo", placement_fallback=False, mood_half_life_s=0,
              tau=0.15, spread=0.0, drift_check_seconds=0, mood_baseline="center")   # tests reason from the origin
    kw.update(over)
    cfg = Config(**kw)
    out = []

    async def sink(m):
        out.append(m)
    a = Agent(cfg, sink, embedder=HashEmbedder())
    a.ensure_index()
    return a, out


def turn(agent, out, text):
    asyncio.run(agent.on_user(text))
    reply = [m for m in out if m["type"] == "reply"][-1]
    plays = [m for m in out if m["type"] == "play"]
    return reply, plays


def test_hit_moves_the_body_and_mood(workspace):
    agent, out = make_agent(workspace)
    reply, plays = turn(agent, out, "thank you, well done")            # echo move: "smile broadly with bright eyes"
    assert reply["body"]["reason"] == "hit" and reply["body"]["target"] == "n004"
    assert plays[-1]["plan"] == ["edge__n001__n004__v1"] and plays[-1]["idle"] == "loop__n004__v1"
    assert plays[-1]["path"] == ["n001", "n004"] and agent.graph.pos == "n004"
    assert reply["coords"] is not None and agent.mood.point[0] > 0
    assert reply["mood"]["mask"] is not None


def test_miss_is_logged_and_the_body_drifts(workspace):
    agent, out = make_agent(workspace, tau=0.5)                          # bag-of-words overlap ("hands", "up") stays below this
    reply, plays = turn(agent, out, "hold the cat")                      # echo move: hold up a cartoon cat...; no such clip
    assert reply["body"]["reason"].startswith("miss")
    assert reply["body"]["requested"] is not None                       # the nearest thing it found
    assert reply["body"]["target"] == agent.graph.nearest(agent.mood.point)[0]
    log = (workspace / "kb" / "episodes.jsonl").read_text(encoding="utf-8")
    assert '"reason": "miss' in log


def test_no_move_drifts_to_the_node_nearest_the_mood(workspace):
    agent, out = make_agent(workspace)
    reply, plays = turn(agent, out, "let's be calm and quiet")          # echo: settled/content, move ""
    assert reply["move"]["description"] == "" and reply["body"]["reason"].startswith("drift")
    near = agent.graph.nearest(agent.mood.point)[0]
    assert agent.graph.pos == near


def test_over_budget_move_drifts_and_is_logged(workspace):
    agent, out = make_agent(workspace, move_budget=0.01)
    reply, plays = turn(agent, out, "you are ugly and stupid")           # angry move, mood is hurt: cost > 0.01
    assert reply["body"]["reason"].startswith("out_of_reach") and reply["body"]["requested"] == "n002"
    assert reply["body"]["cost"] > 0.01


def test_goto_and_position_feedback(workspace):
    agent, out = make_agent(workspace)
    asyncio.run(agent.on_command({"cmd": "goto", "node": "n005"}))
    assert agent.graph.pos == "n005" and [m for m in out if m["type"] == "play"][-1]["end_node"] == "n005"
    agent.on_window({"event": "position", "node": "n001"})
    assert agent.graph.pos == "n001"


def test_drift_during_silence_follows_the_mood(workspace):
    agent, out = make_agent(workspace)
    agent.mood.apply(agent.graph.coords["n004"])                       # the mood sits on happy; the body is on the hub
    asyncio.run(agent.drift_if_needed())
    assert agent.graph.pos == "n004" and [m for m in out if m["type"] == "play"][-1]["reason"] == "drift"
    asyncio.run(agent.drift_if_needed())                               # already there: nothing new
    assert len([m for m in out if m["type"] == "play"]) == 1


def test_gpu_gauge_from_echo_stats(workspace):
    agent, out = make_agent(workspace)
    assert agent.gpu_msg() is None
    turn(agent, out, "hello there")
    g = next(m for m in out if m["type"] == "gpu")
    assert g["context_tokens"] == g["prompt_tokens"] + g["new_tokens"] > 0 and "allocated_mb" not in g
