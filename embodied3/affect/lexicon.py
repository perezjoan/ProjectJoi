"""The lexicon scorer: emotional text -> a raw (valence, arousal, dominance) point.

One function for every emotional text in the system (the brain's emotion line, later the VLM readings and the
keyframe descriptions). Rules, in order, for each term of the text:
  - longest match first against the NRC VAD lexicon (unigrams and multiword terms), with a light lemma fallback
  - a short stoplist of function/temporal words the lexicon rates strongly ("still" has arousal -1.0)
  - negation ("not", "never", "no", "n't", ...) flips valence for the next NEG_WINDOW scored terms
  - intensifiers scale the next term's deviation up (arousal most), diminishers scale it down
  - each term is weighted by its distance from neutral, so "and" or "very" cannot dilute "furious"
Fewer than `min_hits` non-neutral matches is a failed reading: the caller falls back to embedding placement.
The lexicon never sees physical descriptions; that is the embedder's job.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

Vec = tuple[float, float, float]

NEGATORS = {"not", "no", "never", "neither", "nor", "hardly", "without", "cannot", "nothing", "none"}
INTENSIFIERS = {"very", "really", "extremely", "so", "incredibly", "deeply", "truly", "utterly", "terribly",
                "awfully", "totally", "completely", "quite", "genuinely", "intensely", "thoroughly"}
DIMINISHERS = {"slightly", "somewhat", "a little", "a bit", "mildly", "faintly", "barely", "a touch", "kind of",
               "sort of", "a tad", "little"}
# in the lexicon, but function words or time words in an emotion sentence, not feelings
STOPLIST = {"still", "and", "or", "but", "after", "before", "earlier", "later", "now", "then", "again", "very",
            "really", "so", "quite", "about", "with", "from", "into", "over", "under", "just", "only", "also",
            "even", "yet", "while", "during", "since", "though", "although", "because", "if", "than", "as",
            "this", "that", "these", "those", "here", "there", "what", "which", "who", "how", "when", "where",
            "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "feel", "feels", "feeling", "felt", "seem", "seems", "get", "got", "bit", "little", "more", "less",
            "much", "many", "some", "same", "one", "thing", "things", "time", "moment", "exchange", "user", "person",
            "conversation", "question", "answer", "reply", "correction", "last", "first", "next",
            "it", "its", "i", "me", "my", "you", "your", "we", "they", "them", "whether", "actually", "rather",
            "back", "all", "at all", "in", "on", "of", "to", "for", "by", "at", "up", "down", "out", "off",
            "will", "would", "could", "should", "can", "may", "might", "must", "way", "well", "going", "come", "came",
            "try", "trying", "tried", "stay", "keep", "remain", "seem", "look", "sound", "act", "myself", "yourself",
            "know", "knew", "known", "think", "thought", "expect", "expected", "guess", "mean", "meant", "suppose",
            "right", "right now", "more than", "less than", "a lot", "at least", "of course", "at the moment", "even though",
            "even if", "as if", "sort", "kind", "made", "make", "makes", "making", "find", "found", "someone",
            "something", "anything", "talk", "talk to", "share", "tell", "say", "said", "ask", "asked", "start", "starting",
            "started", "begin", "beginning"}
# an effort or wish to feel something is not the feeling: "trying to stay calm" is not calm
HEDGES = {"trying to", "try to", "tries to", "tried to", "attempting to", "attempt to", "pretending to", "pretend to",
          "hoping to", "hope to", "hoped to", "wanting to", "want to", "wanted to", "wish to", "wishing to", "need to",
          "needing to", "managing to", "manage to", "struggling to", "struggle to", "supposed to", "meant to",
          "forcing myself to", "willing myself to", "telling myself to", "doing my best to"}
NEG_WINDOW = 3                     # tokens after a negator whose valence is flipped ("not happy", "not at all sad")
HEDGE_WINDOW = 4                   # tokens after a hedge whose terms are discounted ("trying to [stay] [calm]")
HEDGE_FACTOR = 0.15                # weight multiplier for hedged terms; they never count as hits
INTENSIFY = (1.25, 1.5, 1.25)      # (v, a, d) scale after an intensifier
DIMINISH = (0.7, 0.7, 0.7)         # "a bit hurt" is still hurt: scales the value, never the term's weight in the mean
NEUTRAL_RADIUS = 0.15              # matches closer than this to the origin do not count as hits
# the feeling comes first: the first clause carries full weight, every later clause (the why, the consolation)
# counts LATER_CLAUSE_FACTOR. Boundaries are sentence punctuation and these conjunctions.
CLAUSE_MARKERS = {"because", "since", "but", "although", "though", "while", "whereas", "yet", "even if", "even though",
                  "so that", "as if"}
LATER_CLAUSE_FACTOR = 0.4
IDENTITY = (1.0, 1.0, 1.0)


@dataclass
class Reading:
    ok: bool                       # at least min_hits non-neutral matches
    raw: Vec | None                # weighted mean in lexicon units (-1..1 each), None if nothing matched
    hits: int
    terms: list[dict] = field(default_factory=list)   # per matched term: term, vec (after modifiers), weight, flags
    method: str = "lexicon"        # lexicon | placement | none
    text: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "raw": self.raw, "hits": self.hits, "method": self.method,
                "terms": [{"term": t["term"], "vec": [round(x, 3) for x in t["vec"]], "weight": round(t["weight"], 3),
                           "hit": t["hit"], "flags": t["flags"]} for t in self.terms]}


def _tokens(text: str) -> tuple[list[str], list[int]]:
    """Lower-cased word tokens and, per token, the index of the clause it belongs to. A clause starts at sentence
    punctuation or at a CLAUSE_MARKER (the marker itself opens the new clause)."""
    toks, clause = [], []
    c = -1
    for chunk in re.split(r"[.;!?]+", text.lower().replace("n't", " not").replace("’", "'")):
        chunk = re.sub(r"[^a-z' ]+", " ", chunk)
        words = [w.strip("'") for w in chunk.split() if w.strip("'")]
        if not words:
            continue
        c += 1
        j = 0
        while j < len(words):
            if j > 0 and (words[j] in CLAUSE_MARKERS or " ".join(words[j:j + 2]) in CLAUSE_MARKERS):
                c += 1
            toks.append(words[j])
            clause.append(c)
            j += 1
    return toks, clause


def _lemmas(w: str):
    yield w
    for suf, rep in (("ies", "y"), ("ied", "y"), ("ily", "y"), ("es", ""), ("s", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"), ("ly", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            yield w[: -len(suf)] + rep


class Lexicon:
    def __init__(self, entries: dict[str, Vec], min_hits: int = 3):
        self.entries = entries
        self.min_hits = min_hits
        self.max_n = max((len(k.split()) for k in entries), default=1)

    @classmethod
    def load(cls, path: str | Path, min_hits: int = 3) -> "Lexicon":
        entries: dict[str, Vec] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) != 4 or p[0] == "term":
                    continue
                try:
                    entries[p[0].strip().lower()] = (float(p[1]), float(p[2]), float(p[3]))
                except ValueError:
                    continue
        if not entries:
            raise ValueError(f"no entries read from {path}")
        return cls(entries, min_hits)

    def lookup(self, term: str) -> Vec | None:
        v = self.entries.get(term)
        if v is not None or " " in term:
            return v
        for lem in _lemmas(term):
            if lem in self.entries:
                return self.entries[lem]
        return None

    def score(self, text: str) -> Reading:
        toks, clause = _tokens(text)
        terms: list[dict] = []
        neg_until, hedge_until, scale = -1, -1, IDENTITY
        i = 0
        while i < len(toks):
            start = i
            matched = None
            for n in range(min(max(self.max_n, 3), len(toks) - i), 0, -1):
                cand = " ".join(toks[i:i + n])
                if cand in DIMINISHERS or cand in INTENSIFIERS or cand in NEGATORS or cand in HEDGES:
                    matched = (cand, n, None)
                    break
                if n > self.max_n:
                    continue
                stop = cand in STOPLIST or (n == 1 and any(l in STOPLIST for l in _lemmas(cand)))
                vec = None if stop else self.lookup(cand)
                if vec is not None:
                    matched = (cand, n, vec)
                    break
            if matched is None:
                i += 1
                continue
            term, n, vec = matched
            i += n
            if term in HEDGES:
                hedge_until = i + HEDGE_WINDOW
                continue
            if term in NEGATORS:
                neg_until = i + NEG_WINDOW
                continue
            if term in INTENSIFIERS:
                scale = INTENSIFY
                continue
            if term in DIMINISHERS:
                scale = DIMINISH
                continue
            v, a, d = vec
            flags = []
            norm = (v * v + a * a + d * d) ** 0.5          # the term's say in the mean: how emotional the word is
            weight, hit = max(norm, 0.05), norm >= NEUTRAL_RADIUS
            if scale != IDENTITY:
                v, a, d = v * scale[0], a * scale[1], d * scale[2]
                flags.append("intensified" if scale == INTENSIFY else "diminished")
                scale = IDENTITY
            if start < neg_until:
                v = -v
                flags.append("negated")
            v, a, d = (max(-1.0, min(1.0, x)) for x in (v, a, d))
            if start < hedge_until:
                weight, hit = weight * HEDGE_FACTOR, False
                flags.append("hedged")
            if clause[start] > 0:
                weight *= LATER_CLAUSE_FACTOR
                flags.append(f"clause {clause[start] + 1}")
            terms.append({"term": term, "vec": (v, a, d), "weight": weight, "hit": hit, "flags": flags})
        hits = sum(1 for t in terms if t["hit"])
        if not terms:
            return Reading(False, None, 0, [], "none", text)
        W = sum(t["weight"] for t in terms)
        raw = tuple(sum(t["vec"][k] * t["weight"] for t in terms) / W for k in range(3))
        return Reading(hits >= self.min_hits, raw, hits, terms, "lexicon", text)
