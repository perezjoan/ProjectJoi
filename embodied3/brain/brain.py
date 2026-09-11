"""The brain: persona + instructions + mood line + schema + history -> a Turn (emotion + move + reply).

The mood line is supplied by the caller each turn (the loop owns the mood; the brain only reports against it).
"""
from __future__ import annotations
import random
from pathlib import Path
from .schema import SCHEMA, Turn, Move, parse_turn, as_schema_json
from .backends import Backend

DEFAULT_PERSONA = "You are a small, warm, curious cartoon girl with an animated face and hands."
DEFAULT_INSTRUCTIONS = (
    "Each turn, first say how you feel about the exchange in context, then what your face and hands do, then answer. "
    "Your feelings follow from what was said and from your current mood, which is given to you each turn."
)


class Brain:
    def __init__(self, backend: Backend, persona_file: str | None = None, instructions_file: str | None = None,
                 temperature: float = 0.7, max_new_tokens: int = 400, history_turns: int = 8, seed: int = 0):
        self.backend = backend
        self.persona_file = persona_file
        self.instructions_file = instructions_file
        self.temperature, self.max_new_tokens, self.history_turns, self.seed = temperature, max_new_tokens, history_turns, seed
        self.history: list[dict] = []

    def _read(self, path: str | None, default: str) -> str:
        p = Path(path) if path else None
        return p.read_text(encoding="utf-8").strip() if p and p.exists() else default

    def persona(self) -> str:
        return self._read(self.persona_file, DEFAULT_PERSONA)

    def instructions(self) -> str:
        return self._read(self.instructions_file, DEFAULT_INSTRUCTIONS)

    def system_prompt(self, mood_line: str = "", body_line: str = "", memory_block: str = "", time_line: str = "") -> str:
        """persona.md + instructions.md + time + mood line + body line + retrieved memories + the JSON schema."""
        state = "".join(f"{s}\n\n" for s in (time_line, mood_line, body_line, memory_block) if s)
        return f"{self.persona()}\n\n{self.instructions()}\n\n{state}{SCHEMA}"

    def context_history(self) -> list[dict]:
        """Only the most recent assistant turn keeps its emotion and move: the brain should see what it last
        reported, not eight of its own lines to imitate (that bred a template)."""
        recent = self.history[-2 * self.history_turns:]
        last_assistant = max((i for i, m in enumerate(recent) if m["role"] == "assistant"), default=-1)
        out = []
        for i, m in enumerate(recent):
            if m["role"] != "assistant":
                out.append(m)
            elif i == last_assistant:
                out.append({"role": "assistant", "content": as_schema_json(Turn(reply=m["reply"], emotion=m["emotion"], move=Move(**m["move"])))})
            else:
                out.append({"role": "assistant", "content": as_schema_json(Turn(reply=m["reply"]), reply_only=True)})
        return out

    def turn(self, user_text: str, mood_line: str = "", body_line: str = "", memory_block: str = "", time_line: str = "",
             keep_in_history: bool = True) -> Turn:
        """One brain call. Internal events (wake-up, idle prompts) are answered with the same history in view but are
        not appended to it: they are state, not conversation, and a model that sees its own wake-up line as the last
        turn repeats it for the next ten minutes."""
        messages = [{"role": "system", "content": self.system_prompt(mood_line, body_line, memory_block, time_line)}] \
                   + self.context_history() + [{"role": "user", "content": user_text}]
        seed = self.seed or random.randrange(1, 2**31)
        raw = self.backend.generate(messages, self.temperature, self.max_new_tokens, seed)
        turn = parse_turn(raw)
        if keep_in_history:
            self.history += [{"role": "user", "content": user_text},
                             {"role": "assistant", "reply": turn.reply, "emotion": turn.emotion, "move": turn.move.to_dict()}]
        return turn

    def reset(self) -> None:
        self.history = []
