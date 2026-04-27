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
from dataclasses import dataclass
from typing import Any

import httpx
from httpx_sse import EventSource
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

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
    chunk_type: str | None = None  # "text" / "figure_caption" — surfacing as a credibility cue
    retrieval_method: str | None = None  # "bm25_fulltext" / "pinecone_cosine" / "both"


class KnowledgeSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[KnowledgeChunk]
    total: int
    query: str
    tenant_id: str


class FileMetadata(BaseModel):
    """File-level metadata returned by /api/v1/knowledge/file-chunks."""

    model_config = ConfigDict(extra="ignore")

    file_id: str
    original_filename: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    processing_status: str | None = None
    source_id: str | None = None
    source_name: str | None = None
    created_at: str | None = None  # ISO 8601


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


@dataclass(frozen=True)
class ToolCall:
    """Single tool call in an assistant response (OpenAI shape)."""

    id: str
    name: str
    arguments: str  # JSON string as emitted by the model


@dataclass(frozen=True)
class AssistantResponse:
    """Non-streamed completion response with tool support.

    Either ``content`` (plain text final answer) or ``tool_calls`` (one
    or more tool requests) will be populated. Both can be present —
    Anthropic supports text + tool_use in the same response.
    """

    content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None  # "stop" | "tool_calls" | "length" | "content_filter"
    input_tokens: int
    output_tokens: int

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ── Singleton client (lazy-init, closed on Chainlit shutdown) ──────────


_client: httpx.AsyncClient | None = None
_openai_client: AsyncOpenAI | None = None


def get_openai_client(settings: Settings | None = None) -> AsyncOpenAI:
    """Lazy-init AsyncOpenAI pointed at our Foundry roster endpoint.

    The OpenAI SDK is used as a typed transport — it talks to
    foundry-agent-system's OpenAI-compat router (NOT to openai.com).
    Foundry then routes to Anthropic via the LLM Roster (D-018 compliant
    — caller never imports anthropic SDK).

    Headers required by Foundry (X-Internal-Token, X-Actor-Type,
    X-Actor-ID) are passed via ``default_headers``. Per-request
    X-Request-ID gets attached via ``extra_headers`` at call time.
    """
    global _openai_client
    if _openai_client is None:
        s = settings or get_settings()
        _openai_client = AsyncOpenAI(
            base_url=s.foundry_api_base_url.rstrip("/") + "/api/v1/roster/v1",
            api_key=s.foundry_internal_token,  # required by SDK; not used by Foundry
            default_headers={
                "X-Internal-Token": s.foundry_internal_token,
                "X-Actor-Type": "ai_assistant",
                "X-Actor-ID": s.chatbot_actor_id,
            },
            timeout=s.llm_timeout_s,
            max_retries=0,  # we own retry policy upstream
        )
    return _openai_client


async def close_openai_client() -> None:
    global _openai_client
    if _openai_client is not None:
        await _openai_client.close()
        _openai_client = None


def set_test_openai_client(client: AsyncOpenAI | None) -> None:
    """Inject a pre-configured OpenAI client for tests."""
    global _openai_client
    _openai_client = client


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


async def get_file_metadata(
    tenant_id: str,
    source_file_id: str,
    request_id: str | None = None,
    settings: Settings | None = None,
) -> FileMetadata | None:
    """Fetch file-level metadata via /api/v1/knowledge/file-chunks.

    Returns None on:
      - 4xx (file not found / wrong tenant — Foundry returns file=null silently)
      - transport / timeout errors (best-effort enrichment, never blocks the
        retrieval flow)

    Calls with chunk_limit=1 because we only need the file metadata; the
    one chunk we get back is discarded.
    """
    s = settings or get_settings()
    client = get_client(s)
    req_id = request_id or str(uuid.uuid4())
    body = {
        "tenant_id": tenant_id,
        "source_file_id": source_file_id,
        "chunk_start": 0,
        "chunk_limit": 1,
    }
    try:
        resp = await client.post(
            "/api/v1/knowledge/file-chunks",
            json=body,
            headers={"X-Request-ID": req_id},
            timeout=s.retrieval_timeout_s,
        )
    except (httpx.TimeoutException, httpx.TransportError) as e:
        logger.warning("file-chunks fetch failed for %s: %s", source_file_id, e)
        return None

    if resp.status_code != 200:
        if resp.status_code in (401, 403):
            # Surface auth issues — don't silently degrade
            raise FoundryAuthError(
                f"file-chunks auth failed: HTTP {resp.status_code}"
            )
        logger.warning(
            "file-chunks non-200 for %s: %d %s",
            source_file_id, resp.status_code, resp.text[:200],
        )
        return None

    try:
        data = resp.json()
    except Exception as e:
        logger.warning("file-chunks JSON parse failed: %s", e)
        return None
    file_obj = data.get("file")
    if not file_obj:
        return None
    try:
        return FileMetadata.model_validate(file_obj)
    except Exception as e:
        logger.warning("file-chunks metadata validation failed: %s", e)
        return None


async def complete_chat_with_tools(
    openai_messages: list[dict[str, Any]],
    model_id: str,
    temperature: float,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    request_id: str | None = None,
    settings: Settings | None = None,
) -> AssistantResponse:
    """Non-streamed chat completion with tool support, using AsyncOpenAI SDK.

    The SDK is pointed at our Foundry roster endpoint via base_url. We talk
    OpenAI-compat protocol; Foundry routes to Anthropic. D-018 compliant —
    no anthropic SDK import.

    ``openai_messages`` is the FULL multi-turn message array including
    prior assistant turns (with tool_calls) and tool result messages.
    The Foundry roster (post-VKC-Phase-2 patch, commit 6bbb3e23 on
    foundry-agent-system) preserves these via _build_foundry_messages_multi_turn.

    Returns AssistantResponse with text content, tool_calls (if any),
    finish_reason, and token usage. Raises FoundryAuthError on 401/403,
    FoundryTransientError on 5xx/timeout, FoundryMalformedResponseError
    on parse failures.
    """
    import openai as _openai

    s = settings or get_settings()
    client = get_openai_client(s)
    req_id = request_id or str(uuid.uuid4())

    try:
        resp = await client.chat.completions.create(
            model=model_id,
            messages=openai_messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,  # type: ignore[arg-type]
            stream=False,
            extra_headers={"X-Request-ID": req_id},
        )
    except _openai.AuthenticationError as e:
        logger.error(
            "*** FoundryAuthError on chat completion (auth): %s — "
            "CHECK CHATBOT_INTERNAL_TOKEN ***",
            e,
        )
        raise FoundryAuthError(f"openai auth: {e}") from e
    except _openai.PermissionDeniedError as e:
        logger.error(
            "*** FoundryAuthError on chat completion (permission): %s ***", e
        )
        raise FoundryAuthError(f"openai permission denied: {e}") from e
    except _openai.APIStatusError as e:
        if e.status_code in (401, 403):
            raise FoundryAuthError(
                f"openai status {e.status_code}: {e}"
            ) from e
        if e.status_code >= 500:
            raise FoundryTransientError(
                f"openai 5xx: {e.status_code}: {e}"
            ) from e
        raise FoundryMalformedResponseError(
            f"openai {e.status_code}: {e}"
        ) from e
    except (_openai.APITimeoutError, _openai.APIConnectionError) as e:
        raise FoundryTransientError(f"openai transport: {e}") from e
    except Exception as e:
        raise FoundryMalformedResponseError(
            f"openai unexpected: {type(e).__name__}: {e}"
        ) from e

    if not resp.choices:
        raise FoundryMalformedResponseError("response has no choices")
    choice = resp.choices[0]
    msg = choice.message
    content = msg.content or ""
    raw_tool_calls = msg.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tc.id,
            name=tc.function.name,
            arguments=tc.function.arguments or "{}",
        )
        for tc in raw_tool_calls
    ]
    usage = resp.usage
    return AssistantResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )


async def stream_final_answer(
    openai_messages: list[dict[str, Any]],
    model_id: str,
    temperature: float,
    max_tokens: int,
    request_id: str | None = None,
    settings: Settings | None = None,
) -> AsyncIterator[StreamChunk]:
    """Stream the FINAL answer text via AsyncOpenAI SDK (no tools — caller
    has decided no more tool rounds).

    Use this for the final iteration after the agent loop has finished
    gathering. Yields StreamChunk(content, finish_reason). Tool calls
    are NOT supported in this path — call complete_chat_with_tools for
    iterations that may need tool dispatch.
    """
    import openai as _openai

    s = settings or get_settings()
    client = get_openai_client(s)
    req_id = request_id or str(uuid.uuid4())

    try:
        stream = await client.chat.completions.create(
            model=model_id,
            messages=openai_messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_headers={"X-Request-ID": req_id},
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            content = (delta.content or "") if delta else ""
            finish = choice.finish_reason
            if content or finish:
                yield StreamChunk(content=content, finish_reason=finish)
    except _openai.AuthenticationError as e:
        raise FoundryAuthError(f"stream auth: {e}") from e
    except _openai.APIStatusError as e:
        if e.status_code in (401, 403):
            raise FoundryAuthError(f"stream status {e.status_code}") from e
        if e.status_code >= 500:
            raise FoundryTransientError(
                f"stream 5xx: {e.status_code}"
            ) from e
        raise FoundryMalformedResponseError(
            f"stream {e.status_code}"
        ) from e
    except (_openai.APITimeoutError, _openai.APIConnectionError) as e:
        raise FoundryTransientError(f"stream transport: {e}") from e


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
