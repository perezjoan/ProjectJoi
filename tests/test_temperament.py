import asyncio, json
from embodied3.affect import Mood
from embodied3.memory import MemoryStore, memory_block
from conftest import HashEmbedder
from test_agent import make_agent, turn


def test_trajectory_sentence_names_the_recent_excursion():
    m = Mood(baseline=(0.0, 0.0, 0.0), half_life_s=0, nudge=1.0, last_t=0.0, since_change=0.0)
    m.apply((-1.0, 0.8, -0.8), now=0.0)                 # a shock
    m.apply((0.0, 0.0, 0.0), now=300.0)                 # calmed down five minutes later
    s = m.sentence(now=310.0)
    assert "Not long ago (about 5 min ago) your spirits were low, you were buzzing with energy, and you felt small" in s and "drifted from there since" in s
    assert m.excursion(now=310.0)["distance"] > 0.5
    assert m.excursion(now=5000.0) is None              # out of the window
    assert m.excursion(now=30.0) is None                # too recent: that turn is still in the verbatim history
    m2 = Mood(baseline=(0.0, 0.0, 0.0), half_life_s=0, nudge=1.0, last_t=0.0)
    m2.apply((0.1, 0.1, 0.0), now=0.0)
    assert "Not long ago" not in m2.sentence(now=10.0)   # too small to mention


def test_temperament_follows_the_mood_slowly_and_is_clamped():
    m = Mood(baseline=(0.0, 0.0, 0.0), rest=(0.0, 0.0, 0.0), half_life_s=0, nudge=1.0, temperament_half_life_s=3600,
             temperament_max_shift=0.5, temperament_dt_cap_s=600, last_t=0.0)
    m.apply((1.0, 0.0, 0.0), now=0.0)
    m.tick(now=600.0)                                    # ten minutes of feeling good
    assert 0.05 < m.baseline[0] < 0.2 and m.baseline[1] == 0.0
    for t in range(1, 40):
        m.point = (1.0, 0.0, 0.0)
        m.tick(now=600.0 + t * 600.0)                    # hours of it: the baseline rises but never past the clamp
    assert abs(m.baseline[0] - 0.5) < 1e-6
    assert "sunny side" in m.temperament() and "sunny side" in m.sentence()
    m.tick(now=600.0 + 40 * 600.0 + 86400 * 3)           # a long gap counts as ten minutes, not three days
    assert m.baseline[0] > 0.4
    m.reset()
    assert m.point == m.baseline and m.baseline[0] > 0.4  # reset keeps the temperament


def test_temperament_survives_a_restart(workspace):
    agent, out = make_agent(workspace, mood_half_life_s=3600)
    agent.mood.baseline = (0.3, 0.0, 0.1)
    turn(agent, out, "thank you")                        # saves state
    agent2, _ = make_agent(workspace, mood_half_life_s=3600)
    assert tuple(round(x, 6) for x in agent2.mood.baseline) == (0.3, 0.0, 0.1)


def test_recall_is_history_and_gated_by_congruence(tmp_path):
    st = MemoryStore(tmp_path / "mem", HashEmbedder(), floor=0.2, margin=0.3, max_items=5)
    hurt = st.write("you are useless", "that hurt", "I feel hurt and small", (-0.9, 0.3, -0.8), None, "n005")
    nice = st.write("you are useful", "thank you", "I feel glad", (0.8, 0.3, 0.5), None, "n004")
    plain = st.retrieve("you are useless or useful")
    assert {m["id"] for m in plain} == {hurt, nice}
    fine = st.retrieve("you are useless or useful", mood_valence=0.8)
    assert all(m.get("gated", False) == (m["id"] == hurt) for m in fine)   # the hurt memory is held to the tighter margin
    exact = st.retrieve("you are useless", mood_valence=0.8)
    assert hurt in {m["id"] for m in exact}                                 # ...but a direct question still brings it back
    block = memory_block(st.retrieve("you are useless"))
    assert "At the time you felt" in block and "time has passed since" in block and "does not decide how you feel now" in block


def test_sincerity_gap_is_logged(workspace):
    agent, out = make_agent(workspace)
    reply, _ = turn(agent, out, "thank you, well done")
    assert reply["sincerity_gap"] is not None and reply["sincerity_gap"] > 0
    log = (workspace / "kb" / "episodes.jsonl").read_text(encoding="utf-8")
    assert '"sincerity_gap": ' in log and '"baseline": [' in log
