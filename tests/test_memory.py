import asyncio, json, time
import pytest
from conftest import HashEmbedder
from embodied3.memory import MemoryStore, memory_block, ago_str, wake_line
from test_agent import make_agent, turn


def test_store_writes_retrieves_and_counts_refs(tmp_path):
    pytest.importorskip("chromadb")
    st = MemoryStore(tmp_path / "mem", HashEmbedder(), floor=0.2, margin=0.15, max_items=5)   # bag-of-words: the template words overlap
    t0 = time.time() - 3 * 86400
    a = st.write("do you like cats", "I adore cats", "I feel delighted and warm", (0.5, 0.2, 0.3), (0.8, 0.3, 0.4), "n004", t=t0)
    b = st.write("what is the weather", "no idea, I have no window", "I feel puzzled", (0.0, 0.0, 0.0), None, "n001", t=t0 + 60)
    assert st.count() == 2
    got = st.retrieve("cats, do you like them")
    assert [m["id"] for m in got] == [a] and got[0]["ago"] == "3 days ago" and "delighted" in got[0]["emotion"]
    st.bump_refs([a])
    assert st.recent(1)[0]["id"] == b and next(m for m in st.recent(5) if m["id"] == a)["refs"] == 1
    assert st.retrieve("cats", exclude={a}) == [] or all(m["id"] != a for m in st.retrieve("cats", exclude={a}))
    assert st.retrieve("") == []


def test_retrieval_is_adaptive(tmp_path):
    pytest.importorskip("chromadb")
    st = MemoryStore(tmp_path / "mem", HashEmbedder(), floor=0.3, margin=0.15, max_items=5)
    for i in range(6):
        st.write(f"tell me about cats number {i}", "cats are great", "I feel glad", (0, 0, 0), None, "n001", t=time.time() - i)
    st.write("the weather is grey", "yes", "I feel flat", (0, 0, 0), None, "n001")
    got = st.retrieve("tell me about cats")
    assert 1 <= len(got) <= 5 and all("cats" in m["user"] for m in got)
    assert st.retrieve("something entirely unrelated zzz") == []          # below the floor: nothing recalled


def test_memory_block_and_time_words():
    now = time.time()
    mems = [{"id": "m1", "t": now - 90000, "when": "Mon 08 Sep 2025 16:31", "ago": ago_str(now - 90000, now), "user": "hi", "reply": "hello",
             "emotion": "I felt lonely", "v": -0.5, "a": -0.3, "d": -0.4}]
    b = memory_block(mems, now)
    assert "yesterday" in b and "I felt lonely" in b and "Your mood then" in b and "not the current conversation" in b
    long_reply = "Hello again, it's been a while and I was just dozing off in the corner of my little world. What's on your mind?"
    b2 = memory_block([{**mems[0], "reply": long_reply}], now)
    assert "What's on your mind" not in b2 and "along the lines of: Hello again, it's been a while" in b2   # gist, not verbatim
    assert memory_block([]) == ""
    assert ago_str(now - 30, now) == "just now" and ago_str(now - 5 * 3600, now) == "5 h ago" and ago_str(now - 3 * 86400, now) == "3 days ago"
    assert "switched off" in wake_line(now - 86400 * 2, now) and "2 days ago" in wake_line(now - 86400 * 2, now)


def test_turns_are_remembered_and_recalled_later(workspace):
    agent, out = make_agent(workspace, memory_floor=0.2, memory_margin=0.6)
    turn(agent, out, "do you like cats")
    assert agent.memory.count() == 1 and len(agent.history_memory_ids) == 1
    agent.brain.reset(); agent.history_memory_ids = set()                # a new conversation: the memory is fair game
    reply, _ = turn(agent, out, "cats: do you like them")
    assert len(reply["memories"]) == 1 and reply["memories"][0]["user"] == "do you like cats"
    block = memory_block(agent.memory.retrieve("cats: do you like them", set()))
    assert "do you like cats" in block and "do you like cats" in agent.brain.system_prompt("", "", block, "")
    assert next(m for m in agent.memory.recent(5) if m["user"] == "do you like cats")["refs"] == 1
    log = (workspace / "kb" / "episodes.jsonl").read_text(encoding="utf-8")
    assert '"when": "' in log and '"recalled": [{' in log


def test_state_survives_a_restart_and_she_notices_the_gap(workspace):
    agent, out = make_agent(workspace, mood_half_life_s=3600)
    turn(agent, out, "thank you, well done")
    pos, point = agent.graph.pos, agent.mood.point
    assert (workspace / "kb" / "state.json").exists()
    # pretend it was saved a day ago
    st = json.loads(open(agent.cfg.state_file, encoding="utf-8").read())
    st["saved_at"] -= 86400
    st["mood"]["last_t"] -= 86400
    open(agent.cfg.state_file, "w", encoding="utf-8").write(json.dumps(st))
    agent2, out2 = make_agent(workspace, mood_half_life_s=3600)
    assert agent2.graph.pos == pos and len(agent2.brain.history) == 2 and agent2.away_gap > 86000
    assert agent2.mood.point == point                                     # not yet ticked
    agent2.mood.tick()
    assert abs(agent2.mood.point[0]) < abs(point[0]) * 0.01               # a day at a 1 h half-life: back on the baseline
    asyncio.run(agent2.wake())
    wake = [m for m in out2 if m["type"] == "reply"][-1]
    assert wake["internal"] and len(agent2.brain.history) == 2            # the wake-up is not a history turn
    assert agent2.memory.count() == 2                                     # ...but it is remembered, with its time
