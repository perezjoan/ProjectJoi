from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    USER_MESSAGE = "user_message"
    TIMER = "timer"
    WINDOW = "window"            # renderer feedback: position, visibility
    COMMAND = "command"          # UI commands: reset, set, probe, goto, rebuild_index
    PERCEPT = "percept"          # reserved
    TOOL_RESULT = "tool_result"  # reserved


@dataclass
class Event:
    type: EventType
    payload: dict = field(default_factory=dict)
    t: float = field(default_factory=time.time)
