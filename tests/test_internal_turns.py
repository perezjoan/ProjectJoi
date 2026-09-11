import asyncio, json, time
from embodied3.memory import time_line
from test_agent import make_agent, turn


def test_internal_turns_do_not_enter_history_and_do_not_recall(workspace):
    agent, out = make_agent(workspace, memory_floor=0.2, memory_margin=0.6)
    turn(agent, out, "do you like cats")
    assert len(agent.brain.history) == 2
    asyncio.run(agent.think("[internal event, not from the user] Nothing has happened for 3 min. cats cats cats", "idle"))
    reply = [m for m in out if m["type"] == "reply"][-1]
    assert reply["internal"] and reply["memories"] == []            # no recall on an idle prompt, even a matching one
    assert len(agent.brain.history) == 2                            # and nothing appended
    assert agent.memory.count() == 1                                # idle turns are not remembered either
    assert all(m.get("refs", 0) == 0 for m in agent.memory.recent(5))


def test_wake_is_remembered_but_kept_out_of_history(workspace):
    agent, out = make_agent(workspace, mood_half_life_s=3600)
    turn(agent, out, "thank you, well done")
    st = json.loads(open(agent.cfg.state_file, encoding="utf-8").read())
    st["saved_at"] -= 3600; st["mood"]["last_t"] -= 3600
    open(agent.cfg.state_file, "w", encoding="utf-8").write(json.dumps(st))
    agent2, out2 = make_agent(workspace, mood_half_life_s=3600)
    asyncio.run(agent2.wake())
    assert len(agent2.brain.history) == 2 and "switched" not in agent2.brain.history[-2]["content"]
    assert agent2.memory.count() == 2                                # the wake-up itself is a memory with its time
    assert "switched back on just now, after being off for 1 h" in agent2.time_line_now()
    assert not agent2.history_memory_ids - {m["id"] for m in agent2.memory.recent(5)}


def test_time_line_note_expires():
    now = time.time()
    assert "switched back on" in time_line(now, off_at=now - 7200, on_at=now - 60)
    assert "after being off for 2 h" in time_line(now, off_at=now - 7260, on_at=now - 60)
    assert "switched back on" not in time_line(now, off_at=now - 7200, on_at=now - 2000)
    assert time_line(now) == time_line(now, off_at=None, on_at=now)
