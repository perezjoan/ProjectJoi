"""The mood: a point in the standardised VAD space that decays toward a baseline and is nudged by readings.

    tick:   mood = baseline + (mood - baseline) * 0.5 ** (dt / half_life)
    nudge:  mood += w * (reading - mood)        w = nudge, or nudge * (0.5 + 0.5 * intensity) when by_intensity

Temperament: the baseline itself is slow state. It is an exponential average of where the mood has actually been
(half-life of days), clamped to a radius around the fixed rest point (the neutral sample), so weeks of praise or
hurt shift where she settles without touching the persona file, and nothing can run away.

The mood is written into the brain's prompt as prose, in vocabulary deliberately unlike the emotion lines we want
back, with the trajectory (where it came from recently) and the temperament (where it tends to rest).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from .lexicon import Vec

WORDS = {
    0: [(-0.7, "your spirits are low"), (-0.25, "you are a bit flat"), (0.25, "you are on an even keel"),
        (0.7, "you are in good spirits"), (9, "you are bright and light")],
    1: [(-0.7, "very settled, almost drowsy"), (-0.25, "unhurried"), (0.25, "neither restless nor sleepy"),
        (0.7, "keyed up"), (9, "buzzing with energy")],
    2: [(-0.7, "you feel small"), (-0.25, "you feel a little hesitant"), (0.25, "you feel on an even footing"),
        (0.7, "you feel sure of yourself"), (9, "you feel fully in charge")],
}
TEMPERAMENT_WORDS = {
    0: [(-0.35, "on the low side"), (0.35, None), (9, "on the sunny side")],
    1: [(-0.35, "quiet"), (0.35, None), (9, "lively")],
    2: [(-0.35, "unsure of yourself"), (0.35, None), (9, "self-assured")],
}


def axis_word(k: int, x: float) -> str:
    for thr, w in WORDS[k]:
        if x < thr:
            return w
    return WORDS[k][-1][1]


def _dist(a: Vec, b: Vec) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _ago(seconds: float) -> str:
    if seconds < 90:
        return "a moment ago"
    if seconds < 3600:
        return f"about {int(round(seconds / 60))} min ago"
    return f"about {seconds / 3600:.0f} h ago"


@dataclass
class Mood:
    baseline: Vec                    # where the mood settles; slow state (temperament), starts at `rest`
    half_life_s: float = 120.0
    nudge: float = 0.5
    by_intensity: bool = False
    intensity_scale: float = 1.0
    clip: float = 1.5
    rest: Vec | None = None          # the fixed rest point (the neutral sample); the baseline may drift around it
    temperament_half_life_s: float = 3 * 86400.0
    temperament_max_shift: float = 0.6
    temperament_dt_cap_s: float = 600.0   # a long gap is not a long feeling: cap the averaging step
    point: Vec = None
    last_t: float = field(default_factory=time.time)
    since_change: float = field(default_factory=time.time)
    last_reading: Vec | None = None
    last_nudge: float = 0.0
    trail: list[dict] = field(default_factory=list)   # [{t, p}] points after each nudge, for the window and the trajectory

    def __post_init__(self):
        self.baseline = tuple(self.baseline)
        if self.rest is None:
            self.rest = self.baseline
        self.rest = tuple(self.rest)
        if self.point is None:
            self.point = tuple(self.baseline)
        self._words = self.words()

    # ---- words ----
    def words(self, p: Vec | None = None) -> tuple[str, str, str]:
        p = self.point if p is None else p
        return tuple(axis_word(k, p[k]) for k in range(3))

    def _track_words(self, now: float) -> None:
        w = self.words()
        if w != self._words:
            self._words = w
            self.since_change = now

    # ---- dynamics ----
    def tick(self, now: float | None = None) -> Vec:
        now = time.time() if now is None else now
        dt = max(0.0, now - self.last_t)
        self.last_t = now
        if dt > 0:
            if self.temperament_half_life_s > 0:
                k2 = 0.5 ** (min(dt, self.temperament_dt_cap_s) / self.temperament_half_life_s)
                b = tuple(self.baseline[i] * k2 + self.point[i] * (1 - k2) for i in range(3))
                d = _dist(b, self.rest)
                if d > self.temperament_max_shift:                       # the persona's clamp
                    b = tuple(self.rest[i] + (b[i] - self.rest[i]) * self.temperament_max_shift / d for i in range(3))
                self.baseline = b
            if self.half_life_s > 0:
                k = 0.5 ** (dt / self.half_life_s)
                self.point = tuple(self.baseline[i] + (self.point[i] - self.baseline[i]) * k for i in range(3))
        self._track_words(now)
        return self.point

    def effective_nudge(self, reading: Vec) -> float:
        if not self.by_intensity:
            return self.nudge
        norm = sum(r * r for r in reading) ** 0.5
        return self.nudge * (0.5 + 0.5 * min(1.0, norm / max(self.intensity_scale, 1e-6)))

    def apply(self, reading: Vec, now: float | None = None) -> Vec:
        now = time.time() if now is None else now
        self.tick(now)
        w = self.effective_nudge(reading)
        self.point = tuple(max(-self.clip, min(self.clip, self.point[i] + w * (reading[i] - self.point[i]))) for i in range(3))
        self.last_reading = tuple(reading)
        self.last_nudge = w
        self._track_words(now)
        self.trail.append({"t": now, "p": self.point})
        self.trail = self.trail[-200:]
        return self.point

    def reset(self) -> None:
        """Back to the baseline. The temperament is kept: it is what experience left, not conversation state."""
        self.point = tuple(self.baseline)
        self.last_reading = None
        self.last_nudge = 0.0
        self.trail = []
        self.since_change = time.time()
        self._words = self.words()

    # ---- the sentence ----
    def excursion(self, now: float | None = None, window_s: float = 900.0, min_dist: float = 0.5, min_age_s: float = 60.0) -> dict | None:
        """The recent trail point farthest from the current mood, if it is far enough to be worth a sentence and
        old enough not to be the turn she just had (that one is in her history verbatim)."""
        now = time.time() if now is None else now
        best, bd = None, 0.0
        for e in self.trail:
            if now - e["t"] > window_s or now - e["t"] < min_age_s:
                continue
            d = _dist(e["p"], self.point)
            if d > bd:
                best, bd = e, d
        if best is None or bd < min_dist:
            return None
        return {"t": best["t"], "p": tuple(best["p"]), "distance": bd, "ago": _ago(now - best["t"])}

    def trajectory(self, now: float | None = None) -> str:
        e = self.excursion(now)
        if not e:
            return ""
        past = [x.replace("your spirits are", "your spirits were").replace("you are", "you were").replace("you feel", "you felt")
                for x in self.words(e["p"])]
        return f"Not long ago ({e['ago']}) {past[0]}, you were {past[1]}, and {past[2]}; you have drifted from there since."

    def temperament(self) -> str:
        parts = []
        for k in range(3):
            x = self.baseline[k] - self.rest[k]
            for thr, w in TEMPERAMENT_WORDS[k]:
                if x < thr:
                    if w:
                        parts.append(w)
                    break
        return f"Lately you have tended to be {', '.join(parts)}." if parts else ""

    def sentence(self, now: float | None = None) -> str:
        now = time.time() if now is None else now
        w0, w1, w2 = self.words()
        mins = int((now - self.since_change) // 60)
        ago = "this just changed" if mins < 1 else f"it has been like this for about {mins} min"
        s = f"Where you are starting from this turn: {w0}, {w1}, and {w2}; {ago}."
        traj, temp = self.trajectory(now), self.temperament()
        if traj:
            s += " " + traj
        if temp:
            s += " " + temp
        return s + " This is background, not something to repeat."

    def to_dict(self) -> dict:
        return {"point": self.point, "baseline": self.baseline, "rest": self.rest, "half_life_s": self.half_life_s,
                "nudge": self.nudge, "by_intensity": self.by_intensity, "last_reading": self.last_reading, "last_nudge": self.last_nudge}
