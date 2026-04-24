"""FoundryClient HTTP layer tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from chatbot import foundry_client
from chatbot.exceptions import (
    FoundryAuthError,
    FoundryMalformedResponseError,
    FoundryTransientError,
)
from chatbot.foundry_client import ChatMessage


def _mock_client():
    c = httpx.AsyncClient(
        base_url="https://foundry-test.invalid",
        headers={
            "X-Internal-Token": "test",
            "X-Actor-Type": "ai_assistant",
            "X-Actor-ID": "rocky-ridge-chatbot",
            "Content-Type": "application/json",
        },
    )
    foundry_client._client = c
    return c


@pytest.mark.asyncio
async def test_search_knowledge_happy_path(settings):
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        route = mock.post("/api/v1/knowledge/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "chunk_id": "abc",
                            "content": "stuff",
                            "relevance_score": 0.01,
                        }
                    ],
                    "total": 1,
                    "query": "x",
                    "tenant_id": settings.chatbot_tenant_id,
                },
            )
        )
        _mock_client()
        resp = await foundry_client.search_knowledge(
            tenant_id=settings.chatbot_tenant_id,
            query="x",
            max_results=5,
            settings=settings,
        )
    assert resp.total == 1
    assert resp.results[0].chunk_id == "abc"
    # Required headers are sent
    req = route.calls.last.request
    assert req.headers["x-internal-token"] == "test"
    assert req.headers["x-actor-id"] == "rocky-ridge-chatbot"


@pytest.mark.asyncio
async def test_search_401_raises_auth_error(settings):
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/knowledge/search").mock(
            return_value=httpx.Response(401, json={"error": "unauthorized"})
        )
        _mock_client()
        with pytest.raises(FoundryAuthError):
            await foundry_client.search_knowledge(
                tenant_id=settings.chatbot_tenant_id,
                query="x",
                max_results=5,
                settings=settings,
            )


@pytest.mark.asyncio
async def test_search_403_actor_tenant_mismatch_raises_auth(settings):
    """D-137 — actor_tenant_mismatch returns 403 from Foundry."""
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/knowledge/search").mock(
            return_value=httpx.Response(
                403,
                json={
                    "detail": {
                        "error": {
                            "code": "actor_tenant_mismatch",
                            "message": "not authorized",
                        }
                    }
                },
            )
        )
        _mock_client()
        with pytest.raises(FoundryAuthError):
            await foundry_client.search_knowledge(
                tenant_id=settings.chatbot_tenant_id,
                query="x",
                max_results=5,
                settings=settings,
            )


@pytest.mark.asyncio
async def test_search_5xx_raises_transient(settings):
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/knowledge/search").mock(
            return_value=httpx.Response(502, text="upstream")
        )
        _mock_client()
        with pytest.raises(FoundryTransientError):
            await foundry_client.search_knowledge(
                tenant_id=settings.chatbot_tenant_id,
                query="x",
                max_results=5,
                settings=settings,
            )


@pytest.mark.asyncio
async def test_search_timeout_raises_transient(settings):
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/knowledge/search").mock(
            side_effect=httpx.ReadTimeout("slow")
        )
        _mock_client()
        with pytest.raises(FoundryTransientError):
            await foundry_client.search_knowledge(
                tenant_id=settings.chatbot_tenant_id,
                query="x",
                max_results=5,
                settings=settings,
            )


@pytest.mark.asyncio
async def test_search_malformed_body_raises_malformed(settings):
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/knowledge/search").mock(
            return_value=httpx.Response(200, text="not json at all")
        )
        _mock_client()
        with pytest.raises(FoundryMalformedResponseError):
            await foundry_client.search_knowledge(
                tenant_id=settings.chatbot_tenant_id,
                query="x",
                max_results=5,
                settings=settings,
            )


# ── stream_chat ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_chat_yields_content_deltas(settings):
    sse_body = (
        'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"content":" world"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/roster/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                text=sse_body,
                headers={"content-type": "text/event-stream"},
            )
        )
        _mock_client()
        chunks = []
        async for c in foundry_client.stream_chat(
            messages=[ChatMessage(role="user", content="hi")],
            model_id="anthropic/claude-sonnet-4-5",
            temperature=0.2,
            max_tokens=100,
            settings=settings,
        ):
            chunks.append(c)
    # Two content chunks + one finish chunk
    content_only = [c.content for c in chunks if c.content]
    assert "".join(content_only) == "Hello world"
    finish_chunks = [c for c in chunks if c.finish_reason]
    assert finish_chunks and finish_chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_stream_chat_finish_reason_length(settings):
    """finish_reason=length must surface so the app footers the answer."""
    sse_body = (
        'data: {"choices":[{"index":0,"delta":{"content":"Cut off mid"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
        "data: [DONE]\n\n"
    )
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/roster/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                text=sse_body,
                headers={"content-type": "text/event-stream"},
            )
        )
        _mock_client()
        chunks = [
            c
            async for c in foundry_client.stream_chat(
                messages=[ChatMessage(role="user", content="long q")],
                model_id="anthropic/claude-sonnet-4-5",
                temperature=0.2,
                max_tokens=100,
                settings=settings,
            )
        ]
    finishes = [c.finish_reason for c in chunks if c.finish_reason]
    assert finishes == ["length"]


@pytest.mark.asyncio
async def test_stream_chat_finish_reason_error(settings):
    """Gap Analyst #3 — finish_reason=error must surface."""
    sse_body = (
        'data: {"choices":[{"index":0,"delta":{"content":"Partial"}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"error"}]}\n\n'
        "data: [DONE]\n\n"
    )
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/roster/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                text=sse_body,
                headers={"content-type": "text/event-stream"},
            )
        )
        _mock_client()
        chunks = [
            c
            async for c in foundry_client.stream_chat(
                messages=[ChatMessage(role="user", content="q")],
                model_id="anthropic/claude-sonnet-4-5",
                temperature=0.2,
                max_tokens=100,
                settings=settings,
            )
        ]
    finishes = [c.finish_reason for c in chunks if c.finish_reason]
    assert "error" in finishes


@pytest.mark.asyncio
async def test_stream_chat_401_raises_auth(settings):
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/roster/v1/chat/completions").mock(
            return_value=httpx.Response(401, text="unauth")
        )
        _mock_client()
        with pytest.raises(FoundryAuthError):
            async for _ in foundry_client.stream_chat(
                messages=[ChatMessage(role="user", content="q")],
                model_id="anthropic/claude-sonnet-4-5",
                temperature=0.2,
                max_tokens=100,
                settings=settings,
            ):
                pass
