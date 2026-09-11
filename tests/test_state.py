from embodied3.affect import Mood


def test_decay_halves_the_gap_per_half_life():
    m = Mood(baseline=(0.0, 0.0, 0.0), half_life_s=10, nudge=1.0, last_t=0.0, since_change=0.0, temperament_half_life_s=0)
    m.apply((1.0, -1.0, 0.5), now=0.0)
    assert m.point == (1.0, -1.0, 0.5)
    p = m.tick(now=10.0)
    assert all(abs(p[i] - (1.0, -1.0, 0.5)[i] / 2) < 1e-9 for i in range(3))
    m.tick(now=1000.0)
    assert all(abs(x) < 1e-6 for x in m.point)


def test_nudge_takes_a_share_of_the_gap_and_clips():
    m = Mood(baseline=(0.0, 0.0, 0.0), half_life_s=0, nudge=0.5, clip=1.5, temperament_half_life_s=0)
    m.apply((1.0, 1.0, 1.0))
    assert m.point == (0.5, 0.5, 0.5) and m.last_nudge == 0.5
    for _ in range(20):
        m.apply((5.0, 5.0, 5.0))
    assert m.point == (1.5, 1.5, 1.5)
    assert len(m.trail) == 21


def test_nudge_by_intensity():
    m = Mood(baseline=(0.0, 0.0, 0.0), half_life_s=0, nudge=0.5, by_intensity=True, intensity_scale=1.0)
    assert abs(m.effective_nudge((0.0, 0.0, 0.0)) - 0.25) < 1e-9          # a neutral reading pulls at half strength
    assert abs(m.effective_nudge((1.0, 0.0, 0.0)) - 0.5) < 1e-9           # a full-strength reading at the nominal nudge
    assert abs(m.effective_nudge((3.0, 3.0, 3.0)) - 0.5) < 1e-9           # never more than the nominal nudge
    m.apply((0.0, 0.0, 0.0))
    assert m.last_nudge == 0.25


def test_sentence_is_prose_and_reset():
    m = Mood(baseline=(0.0, 0.0, 0.0), nudge=1.0)
    m.apply((1.0, -1.0, 1.0))
    s = m.sentence()
    assert "bright" in s and "drowsy" in s and "in charge" in s and "just changed" in s and "not something to repeat" in s
    m.reset()
    assert m.point == (0.0, 0.0, 0.0) and m.trail == [] and m.last_nudge == 0.0


def test_since_change_tracks_the_worded_mood_not_the_nudge():
    m = Mood(baseline=(0.0, 0.0, 0.0), half_life_s=0, nudge=1.0, last_t=0.0, since_change=0.0)
    m._words = m.words()
    m.apply((1.0, 0.0, 0.0), now=0.0)                 # "in good spirits": worded mood changes at t=0
    m.apply((1.05, 0.0, 0.0), now=200.0)              # same words: the clock keeps running
    assert m.since_change == 0.0 and "about 3 min" in m.sentence(now=200.0)
    m.apply((-1.0, 0.0, 0.0), now=300.0)              # new words: clock resets
    assert m.since_change == 300.0 and "just changed" in m.sentence(now=310.0)
