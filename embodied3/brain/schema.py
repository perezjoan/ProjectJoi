"""The brain's v3 output schema and its lenient parser: three channels, emotion first, then move, then reply.

    {"emotion": "<how you feel, one or two sentences of emotional language>",
     "move":    {"description": "<what your face and hands do, physical language; "" = leave the face as it is>",
                 "hold": "turn"},
     "reply":   "<what you say>"}

Failure is soft: a missing emotion leaves the mood to decay; a missing or empty move means the body drifts to the
node nearest the mood; output that is not JSON at all is a plain reply.
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass, field, asdict

SCHEMA = (
    'Reply ONLY with one JSON object, no prose before or after, exactly this shape and in this order:\n'
    '{"emotion": "<how you feel about this exchange given what came before: one or two full sentences in the first '
    'person, feelings not facts, not a list of labels, not what your face does, not an effort to stay composed>",\n'
    ' "move": {"description": "<what your face and hands do while you speak, in physical language, one concrete '
    'sentence (eyes, brows, mouth, head, hands, props); or \\"\\" to leave your face as it is>", "hold": "turn"},\n'
    ' "reply": "<what you say to the user: two to four sentences, more if telling a story>"}'
)
HOLDS = ("turn", "sustain", "brief")


@dataclass
class Move:
    description: str = ""
    elements: list[str] = field(default_factory=list)
    when: str = "start"
    hold: str = "turn"

    def __bool__(self) -> bool:
        return bool(self.description.strip())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Turn:
    reply: str
    emotion: str = ""
    move: Move = field(default_factory=Move)
    raw: str = ""
    parsed: bool = True            # False when the output was not JSON (reply = the raw text)

    def to_dict(self) -> dict:
        return {"reply": self.reply, "emotion": self.emotion, "move": self.move.to_dict(), "parsed": self.parsed}


def lenient_json(txt: str) -> dict:
    txt = re.sub(r"```(?:json)?", "", txt)
    txt = re.sub(r",\s*([}\]])", r"\1", txt)          # trailing commas
    return json.loads(txt)


def _first_object(raw: str) -> dict | None:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        obj = lenient_json(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _str(x) -> str:
    return "" if x is None else str(x).strip()


def parse_move(obj) -> Move:
    if obj is None:
        return Move()
    if isinstance(obj, str):
        return Move(description=obj.strip())
    if not isinstance(obj, dict):
        return Move()
    elems = obj.get("elements") or []
    if isinstance(elems, str):
        elems = [e.strip() for e in elems.split(",") if e.strip()]
    hold = _str(obj.get("hold")) or "turn"
    return Move(description=_str(obj.get("description")), elements=[str(e) for e in elems],
                when=_str(obj.get("when")) or "start", hold=hold if hold in HOLDS else "turn")


def parse_turn(raw: str) -> Turn:
    obj = _first_object(raw)
    if obj is None:
        return Turn(reply=raw.strip(), emotion="", raw=raw, parsed=False)
    move = parse_move(obj.get("move"))
    if not move and obj.get("perform"):          # tolerate the v1 shape
        move = parse_move(obj.get("perform"))
    return Turn(reply=_str(obj.get("reply")), emotion=_str(obj.get("emotion")), move=move, raw=raw, parsed=True)


def as_schema_json(turn: Turn, reply_only: bool = False) -> str:
    """What goes into the history: the shape the model was asked to produce; or the reply alone for older turns."""
    if reply_only:
        return json.dumps({"reply": turn.reply}, ensure_ascii=False)
    return json.dumps({"emotion": turn.emotion, "move": {"description": turn.move.description, "hold": turn.move.hold},
                       "reply": turn.reply}, ensure_ascii=False)
