"""History-aware query reformulation.

Two optimizations over the naive pattern:
  1. Skip reformulation for messages that have no reference tokens
     ("canebrake restoration" is already standalone; "what about Bethel?"
     needs rewrite). Saves ~60% of reformulation calls (QC finding #5).
  2. Sanity-check the result: if the reformulated query is >2× the raw
     length, or contains telltale "the user" phrasings, fall back to raw.
"""

from __future__ import annotations

import asyncio
import logging
import re

from chatbot import foundry_client
from chatbot.config import Settings, get_settings
from chatbot.foundry_client import ChatMessage
from chatbot.prompt_builder import build_reformulation_messages

logger = logging.getLogger(__name__)


# Cheap heuristic: messages without any of these tokens are likely
# already standalone and don't need reformulation.
_REFERENCE_TOKEN_RE = re.compile(
    r"\b(it|that|this|those|these|they|them|he|she|his|her|"
    r"there|then|same|similar|"
    r"what about|how about|and|also|which one|where)\b",
    re.IGNORECASE,
)

# Telltale signs that the model returned a summary instead of a query.
_BAD_SIGNALS = (
    "the user",
    "the conversation",
    "based on",
    "according to the",
)


def needs_reformulation(query: str, history_len: int) -> bool:
    """No history → no need. No reference tokens → almost certainly standalone."""
    if history_len == 0:
        return False
    return bool(_REFERENCE_TOKEN_RE.search(query))


def _is_bad_reformulation(original: str, reformulated: str) -> bool:
    if not reformulated or not reformulated.strip():
        return True
    if len(reformulated) > 2 * max(len(original), 40):
        return True
    lowered = reformulated.lower()
    return any(s in lowered for s in _BAD_SIGNALS)


async def reformulate(
    query: str,
    history: list[ChatMessage],
    settings: Settings | None = None,
) -> str:
    """Return a standalone search query. Falls through to raw on any failure."""
    s = settings or get_settings()
    if not needs_reformulation(query, len(history)):
        return query

    messages = build_reformulation_messages(query, history)
    try:
        tokens: list[str] = []
        reformulation_coro = _collect_stream(
            messages, s.chatbot_model_id, s.chatbot_reformulation_max_tokens, s
        )
        tokens = await asyncio.wait_for(
            reformulation_coro, timeout=s.reformulation_timeout_s
        )
        candidate = "".join(tokens).strip().strip('"').strip("'")
    except asyncio.TimeoutError:
        logger.warning("reformulation timeout — falling back to raw query")
        return query
    except Exception as e:
        logger.warning("reformulation failed (%s) — falling back to raw query", e)
        return query

    if _is_bad_reformulation(query, candidate):
        logger.info(
            "reformulation rejected as malformed: %r → %r; using raw",
            query,
            candidate,
        )
        return query

    logger.info("reformulated: %r → %r", query, candidate)
    return candidate


async def _collect_stream(
    messages: list[ChatMessage],
    model_id: str,
    max_tokens: int,
    settings: Settings,
) -> list[str]:
    tokens: list[str] = []
    async for chunk in foundry_client.stream_chat(
        messages=messages,
        model_id=model_id,
        temperature=0.0,
        max_tokens=max_tokens,
        settings=settings,
    ):
        if chunk.content:
            tokens.append(chunk.content)
    return tokens
