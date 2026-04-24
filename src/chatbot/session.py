"""Per-Chainlit-session state.

Wraps cl.user_session (per-WebSocket, in-memory, per-reload) with:
  - idempotent init (no clobber on reconnect)
  - per-session asyncio.Lock to serialize concurrent messages on the
    same socket (Stress Tester #2 — double-click corrupts history)
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

from chatbot.foundry_client import ChatMessage

if TYPE_CHECKING:
    pass


HISTORY_KEY = "history"
SESSION_ID_KEY = "session_id"
LOCK_KEY = "turn_lock"


def _cl_session():  # noqa: ANN202 — Chainlit lazy import keeps tests unit-testable
    import chainlit as cl

    return cl.user_session


def init_session() -> None:
    s = _cl_session()
    if s.get(HISTORY_KEY) is None:
        s.set(HISTORY_KEY, [])
    if s.get(SESSION_ID_KEY) is None:
        s.set(SESSION_ID_KEY, str(uuid.uuid4()))
    if s.get(LOCK_KEY) is None:
        s.set(LOCK_KEY, asyncio.Lock())


def get_history() -> list[ChatMessage]:
    return _cl_session().get(HISTORY_KEY, []) or []


def append_turn(user_msg: str, assistant_msg: str, max_turns: int) -> None:
    s = _cl_session()
    history = list(s.get(HISTORY_KEY, []) or [])
    history.append(ChatMessage(role="user", content=user_msg))
    history.append(ChatMessage(role="assistant", content=assistant_msg))
    if max_turns > 0:
        history = history[-(max_turns * 2):]
    s.set(HISTORY_KEY, history)


def get_session_id() -> str:
    return _cl_session().get(SESSION_ID_KEY) or "unknown"


def get_turn_lock() -> asyncio.Lock:
    s = _cl_session()
    lock = s.get(LOCK_KEY)
    if lock is None:
        lock = asyncio.Lock()
        s.set(LOCK_KEY, lock)
    return lock
