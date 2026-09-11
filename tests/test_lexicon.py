from embodied3.affect import Lexicon, Space, seed_text
from conftest import SEED


def test_basic_and_min_hits(fake_lexicon):
    r = fake_lexicon.score("happy")
    assert r.hits == 1 and not r.ok and r.raw[0] > 0.8
    r = fake_lexicon.score("happy, glad and proud")
    assert r.ok and r.hits == 3 and r.raw[0] > 0.7


def test_negation_flips_valence_within_window_only(fake_lexicon):
    r = fake_lexicon.score("not happy, then curious and later proud")
    terms = {t["term"]: t for t in r.terms}
    assert terms["happy"]["vec"][0] < 0 and "negated" in terms["happy"]["flags"]
    assert terms["proud"]["vec"][0] > 0 and "negated" not in terms["proud"]["flags"]
    assert fake_lexicon.score("I'm not sad").terms[0]["vec"][0] > 0


def test_intensifier_and_diminisher(fake_lexicon):
    plain = fake_lexicon.score("angry").terms[0]["vec"]
    strong = fake_lexicon.score("very angry").terms[0]["vec"]
    weak = fake_lexicon.score("slightly angry").terms[0]["vec"]
    assert strong[1] > plain[1] and abs(weak[0]) < abs(plain[0])
    assert fake_lexicon.score("a little sad").terms[0]["flags"] == ["diminished"]


def test_stoplist_and_neutral_weighting(fake_lexicon):
    r = fake_lexicon.score("still happy and glad and proud")
    assert "still" not in {t["term"] for t in r.terms}
    assert "and" not in {t["term"] for t in r.terms}
    r = fake_lexicon.score("happy glad proud fine")
    assert not next(t for t in r.terms if t["term"] == "fine")["hit"]
    assert r.raw[1] > 0.25    # near-neutral words carry little weight


def test_multiword_and_lemma(fake_lexicon):
    r = fake_lexicon.score("feeling in control, happily")
    terms = {t["term"] for t in r.terms}
    assert "in control" in terms and "happily" in terms


def test_load(workspace):
    lex = Lexicon.load(workspace / "lex.txt")
    assert lex.lookup("in control") and lex.max_n == 2 and lex.lookup("term") is None


def test_space_standardises_seed(fake_lexicon):
    samples = []
    from embodied3.affect.space import Sample
    for k in SEED["keyframes"]:
        r = fake_lexicon.score(seed_text(k))
        samples.append(Sample(k["node"], k["label"], k["file"], seed_text(k), r.raw, hits=r.hits, hand={"v": k["valence"], "a": k["arousal"]}))
    sp = Space(fake_lexicon, samples, clip=1.5, baseline_node="n001")
    for s in sp.samples:
        assert all(-1.0001 <= c <= 1.0001 for c in s.coords)
    assert sp.by_node["n004"].coords[0] > sp.by_node["n005"].coords[0]        # happy right of crying
    assert sp.hand_correlation()["valence"] > 0.8
    reading, c = sp.read("happy, glad, proud")
    assert reading.ok and sp.nearest(c)[0].node == "n004"
    assert sp.read("fine")[1] is None                                           # too few hits


def test_seed_text_strips_physical_parts():
    t = seed_text(SEED["keyframes"][1])
    assert "furrowed brows" in t and "steam" not in t and "fists" not in t


def test_hedged_terms_are_discounted_and_not_hits(fake_lexicon):
    r = fake_lexicon.score("afraid, and trying to stay calm")
    terms = {t["term"]: t for t in r.terms}
    assert "hedged" in terms["calm"]["flags"] and not terms["calm"]["hit"] and terms["calm"]["weight"] < 0.2
    assert terms["afraid"]["hit"] and r.raw[0] < 0                       # calm no longer cancels afraid
    assert r.hits == 1
    r = fake_lexicon.score("shocked but calm")
    assert next(t for t in r.terms if t["term"] == "calm")["hit"]          # no hedge, calm counts


def test_later_clauses_count_less(fake_lexicon):
    one = fake_lexicon.score("sad, tired, afraid. happy, glad, proud.")
    flags = {t["term"]: t["flags"] for t in one.terms}
    assert flags["sad"] == [] and flags["happy"] == ["clause 2"]
    assert one.raw[0] < 0                                            # the first clause wins
    flipped = fake_lexicon.score("happy, glad, proud. sad, tired, afraid.")
    assert flipped.raw[0] > 0 and one.hits == flipped.hits == 6      # hits are unaffected by the weighting
    conj = fake_lexicon.score("sad and tired because I was happy and glad")
    flags = {t["term"]: t["flags"] for t in conj.terms}
    assert flags["tired"] == [] and flags["happy"] == ["clause 2"] and conj.raw[0] < 0
    both = fake_lexicon.score("sad but also happy")
    assert next(t for t in both.terms if t["term"] == "happy")["flags"] == ["clause 2"] and both.raw[0] < 0


def test_diminisher_scales_the_value_not_the_weight(fake_lexicon):
    plain = next(t for t in fake_lexicon.score("hurt").terms)
    dim = next(t for t in fake_lexicon.score("a bit hurt").terms)
    assert dim["weight"] == plain["weight"] and abs(dim["vec"][0]) < abs(plain["vec"][0]) and dim["hit"]
    # a diminished feeling in the first clause still outweighs a pleasant word in the why-clause
    r = fake_lexicon.score("a bit hurt and curious, because I was starting to think I was improving")
    w = {t["term"]: t["weight"] for t in r.terms}
    assert w["hurt"] > w["curious"] and w["hurt"] > w.get("improving", 0) and r.raw[0] < 0
