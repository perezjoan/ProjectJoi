"""How memories and time are written into the brain's prompt."""
from __future__ import annotations
import re, time
from ..affect.state import axis_word
from .store import when_str, ago_str


def time_line(now: float | None = None, off_at: float | None = None, on_at: float | None = None, note_for_s: float = 900.0) -> str:
    """The clock, plus, for a while after a restart, the fact of the gap: this replaces keeping the wake-up exchange
    in the history, which made the model relive it every turn."""
    now = time.time() if now is None else now
    s = f"Now: {when_str(now)}."
    if off_at is not None and on_at is not None and 0 <= now - on_at < note_for_s:
        s += f" You were switched back on {ago_str(on_at, now)}, after being off for {_span_str(on_at - off_at)}."
    return s


def _span_str(seconds: float) -> str:
    if seconds < 3600:
        return f"{max(1, int(round(seconds / 60)))} min"
    if seconds < 86400:
        return f"{seconds / 3600:.0f} h"
    return f"{seconds / 86400:.0f} days"


def mood_then(v: float, a: float, d: float) -> str:
    return f"{axis_word(0, v)}, {axis_word(1, a)}, {axis_word(2, d)}"


def gist(text: str, max_chars: int = 90) -> str:
    """The first clause of a reply, shortened: enough to know what was said, not enough to say it again."""
    t = " ".join((text or "").split())
    if not t:
        return ""
    first = re.split(r"(?<=[.!?])\s", t, maxsplit=1)[0]
    if len(first) > max_chars:
        first = first[:max_chars].rsplit(" ", 1)[0] + "..."
    return first


def memory_block(memories: list[dict], now: float | None = None) -> str:
    """The retrieved memories, oldest first, each with when it happened and how she felt then. Her own past reply
    is given as a gist, never verbatim: a quoted reply is an invitation to say it again."""
    if not memories:
        return ""
    lines = ["Things you remember from earlier conversations (when, and how you felt at the time):"]
    for m in memories:
        when = f"{m.get('ago') or ago_str(m['t'], now)} ({m['when']})"
        felt = f' At the time you felt: "{m["emotion"]}".' if m.get("emotion") else ""
        mood = f" Your mood then was {mood_then(m['v'], m['a'], m['d'])}; time has passed since."
        g = gist(m.get("reply", ""))
        said = f' The user said: "{m["user"]}". You answered along the lines of: {g}' if g else f' The user said: "{m["user"]}".'
        lines.append(f"- {when}:{said}{felt}{mood}")
    lines.append("These are memories, not the current conversation. What you felt then is history: it can inform you, "
                 "it does not decide how you feel now. Do not repeat the words of an earlier answer; this moment is new.")
    return "\n".join(lines)


def wake_line(off_at: float, now: float | None = None) -> str:
    now = time.time() if now is None else now
    return (f"[internal event, not from the user] You were switched off on {when_str(off_at)} and have just been switched "
            f"back on; that was {ago_str(off_at, now)}. It is now {when_str(now)}. What you remember of before is in your "
            "history and your memories. React as yourself: say how you feel about the time that passed; keep the reply to "
            "one short line or \"\" if you would rather stay quiet.")
