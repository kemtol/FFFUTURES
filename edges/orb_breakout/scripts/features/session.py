"""Session boundary definitions and helpers for MGC trading sessions (UTC)."""

from datetime import time
from typing import Literal

Session = Literal["tokyo", "london", "us"]

SESSION_BOUNDARIES: dict[Session, tuple[time, time]] = {
    "tokyo":  (time(0, 0),  time(3, 0)),
    "london": (time(7, 0),  time(10, 0)),
    "us":     (time(13, 30), time(16, 30)),
}

SESSION_INDEX: dict[Session, int] = {"tokyo": 0, "london": 1, "us": 2}


def get_session(t: time) -> Session | None:
    """Return session name for a given UTC time, or None if outside all sessions."""
    for name, (start, end) in SESSION_BOUNDARIES.items():
        if start <= t < end:
            return name
    return None


def minutes_into_session(t: time, session: Session) -> int:
    """Minutes elapsed since session open."""
    start = SESSION_BOUNDARIES[session][0]
    return (t.hour * 60 + t.minute) - (start.hour * 60 + start.minute)
