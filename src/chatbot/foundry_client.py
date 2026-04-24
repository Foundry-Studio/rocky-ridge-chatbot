"""HTTP client for Foundry-Agent-System.

Single long-lived ``httpx.AsyncClient`` (Stress Tester finding #1) —
connection pool reused across all retrieval + SSE calls.

Required headers on every call:
    X-Internal-Token  (CHATBOT_INTERNAL_TOKEN — D-128/D-136)
    X-Actor-Type      = "ai_assistant" (D-129)
    X-Actor-ID        = CHATBOT_ACTOR_ID (D-129 + D-137 authz key)
    X-Request-ID      (correlation across chatbot ↔ Foundry logs)
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
from httpx_sse import EventSource
from pydantic import BaseModel, ConfigDict, Field

from chatbot.config import Settings, get_settings
from chatbot.exceptions import (
    FoundryAuthError,
    FoundryMalformedResponseError,
    FoundryTransientError,
)

logger = logging.getLogger(__name__)


# ── Response models (mirror Foundry shapes; extra="ignore" for fwd compat) ──


class KnowledgeChunk(BaseModel):
    """One chunk from /api/v1/knowledge/search results."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    content: str
    relevance_score: float | None = None
    source_file_id: str | None = None
    source_name: str | None = None
    page_numbers: list[int] | None = None
    section_title: str | None = None
    authority_level: str | None = None
    source_id: str | None = None


class KnowledgeSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[KnowledgeChunk]
    total: int
    query: str
    tenant_id: str


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class StreamChunk:
    """One SSE delta from /v1/chat/completions.

    Normal content lives in ``content``. When the upstream emits a
    finish_reason (stop/length/error), that lands here so callers can
    footer-annotate length-truncated or errored streams (Gap Analyst #3).
    """

    content: str = ""
    finish_reason: str | None = None


# ── Singleton client (lazy-init, closed on Chainlit shutdown) ──────────


_client: httpx.AsyncClient | None = None


def get_client(settings: Settings | None = None) -> httpx.AsyncClient:
    """Return the module-level async client, creating it on first call."""
    global _client
    if _client is None:
        s = settings or get_settings()
        _client = httpx.AsyncClient(
            base_url=s.foundry_api_base_url,
            timeout=httpx.Timeout(
                connect=5.0,
                read=s.llm_timeout_s,
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
            headers={
                "X-Internal-Token": s.foundry_internal_token,
                "X-Actor-Type": "ai_assistant",
                "X-Actor-ID": s.chatbot_actor_id,
                "Content-Type": "application/json",
            },
        )
    return _client


async def close_client() -> None:
    """Close the pool — call from Chainlit ``@cl.on_stop``."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def set_test_client(client: httpx.AsyncClient | None) -> None:
    """Inject a pre-configured client for tests. Pass None to reset."""
    global _client
    _client = client


# ── API surface ────────────────────────────────────────────────────────


async def search_knowledge(
    tenant_id: str,
    query: str,
    max_results: int,
    source_id: str | None = None,
    request_id: str | None = None,
    settings: Settings | None = None,
) -> KnowledgeSearchResponse:
    """Call POST /api/v1/knowledge/search.

    Raises FoundryAuthError on 401/403, FoundryTransientError on 5xx/timeout,
    FoundryMalformedResponseError on unparseable body. Never retries — let
    the caller decide.
    """
    s = settings or get_settings()
    client = get_client(s)
    req_id = request_id or str(uuid.uuid4())
    body: dict[str, Any] = {
        "tenant_id": tenant_id,
        "query": query,
        "max_results": max_results,
        "tenant_scope": "venture",
    }
    if source_id:
        body["source_id"] = source_id

    try:
        resp = await client.post(
            "/api/v1/knowledge/search",
            json=body,
            headers={"X-Request-ID": req_id},
            timeout=s.retrieval_timeout_s,
        )
    except httpx.TimeoutException as e:
        raise FoundryTransientError(f"knowledge/search timeout: {e}") from e
    except httpx.TransportError as e:
        raise FoundryTransientError(f"knowledge/search transport: {e}") from e

    if resp.status_code in (401, 403):
        logger.error(
            "*** FoundryAuthError on /api/v1/knowledge/search: %d — "
            "CHECK CHATBOT_INTERNAL_TOKEN + actor_tenant_authorizations.yaml ***",
            resp.status_code,
        )
        raise FoundryAuthError(f"auth failed: HTTP {resp.status_code}: {resp.text}")
    if resp.status_code >= 500:
        raise FoundryTransientError(f"5xx: {resp.status_code} {resp.text[:200]}")
    if resp.status_code >= 400:
        raise FoundryMalformedResponseError(
            f"4xx: {resp.status_code} {resp.text[:200]}"
        )

    try:
        return KnowledgeSearchResponse.model_validate(resp.json())
    except Exception as e:
        raise FoundryMalformedResponseError(f"parse failed: {e}") from e


async def stream_chat(
    messages: list[ChatMessage],
    model_id: str,
    temperature: float,
    max_tokens: int,
    request_id: str | None = None,
    settings: Settings | None = None,
) -> AsyncIterator[StreamChunk]:
    """Stream tokens from POST /api/v1/roster/v1/chat/completions.

    Yields StreamChunk(content, finish_reason). Caller is responsible for
    consuming the iterator to completion (or cancelling cleanly — the
    underlying stream is closed automatically when iteration ends).
    """
    s = settings or get_settings()
    client = get_client(s)
    req_id = request_id or str(uuid.uuid4())
    body = {
        "model": model_id,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    headers = {
        "X-Request-ID": req_id,
        "Accept": "text/event-stream",
    }

    try:
        async with client.stream(
            "POST",
            "/api/v1/roster/v1/chat/completions",
            json=body,
            headers=headers,
            timeout=s.llm_timeout_s,
        ) as resp:
            if resp.status_code in (401, 403):
                await resp.aread()
                logger.error(
                    "*** FoundryAuthError on chat/completions: %d — "
                    "CHECK CHATBOT_INTERNAL_TOKEN ***",
                    resp.status_code,
                )
                raise FoundryAuthError(f"auth failed: HTTP {resp.status_code}")
            if resp.status_code >= 500:
                await resp.aread()
                raise FoundryTransientError(
                    f"chat 5xx: {resp.status_code} {resp.text[:200]}"
                )
            if resp.status_code >= 400:
                await resp.aread()
                raise FoundryMalformedResponseError(
                    f"chat 4xx: {resp.status_code} {resp.text[:200]}"
                )

            event_source = EventSource(resp)
            async for event in event_source.aiter_sse():
                data = event.data
                if not data or data == "[DONE]":
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed SSE event: %r", data[:200])
                    continue
                choices = payload.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                finish = choice.get("finish_reason")
                # Yield content deltas OR finish events (both matter — finish
                # may arrive in its own empty-delta chunk per OpenAI spec).
                if content or finish:
                    yield StreamChunk(content=content, finish_reason=finish)
    except httpx.TimeoutException as e:
        raise FoundryTransientError(f"chat timeout: {e}") from e
    except httpx.TransportError as e:
        raise FoundryTransientError(f"chat transport: {e}") from e
