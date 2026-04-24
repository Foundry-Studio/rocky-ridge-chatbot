"""Query reformulator tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from chatbot import foundry_client
from chatbot.foundry_client import ChatMessage
from chatbot.query_reformulator import (
    _is_bad_reformulation,
    needs_reformulation,
    reformulate,
)


# ── needs_reformulation heuristic ───────────────────────────────


def test_empty_history_skips():
    assert needs_reformulation("what is canebrake", 0) is False


def test_no_reference_tokens_skips():
    assert needs_reformulation("what is canebrake restoration", 3) is False


def test_pronoun_needs_reformulation():
    assert needs_reformulation("what about it?", 2) is True


def test_reference_phrase_needs_reformulation():
    assert needs_reformulation("how about Bethel?", 2) is True


def test_they_needs_reformulation():
    assert needs_reformulation("when did they start?", 4) is True


# ── sanity checks ───────────────────────────────────────────────


def test_empty_is_bad():
    assert _is_bad_reformulation("original", "") is True
    assert _is_bad_reformulation("original", "   ") is True


def test_long_expansion_is_bad():
    original = "what about Bethel?"
    too_long = "The user asked about Bethel which is one of the properties under the NFWF grant and they want to know more. " * 3
    assert _is_bad_reformulation(original, too_long) is True


def test_clean_reformulation_is_good():
    original = "what about Bethel?"
    clean = "What restoration work is planned for Bethel Springs?"
    assert _is_bad_reformulation(original, clean) is False


def test_the_user_pattern_is_bad():
    original = "what about it?"
    bad = "The user is asking about the topic discussed."
    assert _is_bad_reformulation(original, bad) is True


# ── reformulate() integration (mocked Foundry) ──────────────────


@pytest.mark.asyncio
async def test_reformulate_empty_history_returns_raw(settings):
    # needs_reformulation returns False → no network call
    out = await reformulate("what is X", [], settings=settings)
    assert out == "what is X"


@pytest.mark.asyncio
async def test_reformulate_calls_foundry_with_pronoun(settings):
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/roster/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                text=(
                    'data: {"choices":[{"index":0,"delta":{"content":"What restoration work is planned for Bethel Springs?"}}]}\n\n'
                    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                    "data: [DONE]\n\n"
                ),
                headers={"content-type": "text/event-stream"},
            )
        )
        foundry_client._client = httpx.AsyncClient(
            base_url="https://foundry-test.invalid",
            headers={
                "X-Internal-Token": "test",
                "X-Actor-Type": "ai_assistant",
                "X-Actor-ID": "rocky-ridge-chatbot",
            },
        )
        history = [
            ChatMessage(role="user", content="tell me about the properties"),
            ChatMessage(
                role="assistant", content="There are 12 properties..."
            ),
        ]
        result = await reformulate("what about Bethel?", history, settings=settings)
    assert "Bethel" in result


@pytest.mark.asyncio
async def test_reformulate_falls_back_on_network_error(settings):
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/roster/v1/chat/completions").mock(
            side_effect=httpx.ConnectError("boom")
        )
        foundry_client._client = httpx.AsyncClient(
            base_url="https://foundry-test.invalid",
            headers={
                "X-Internal-Token": "test",
                "X-Actor-Type": "ai_assistant",
                "X-Actor-ID": "rocky-ridge-chatbot",
            },
        )
        history = [ChatMessage(role="user", content="context")]
        out = await reformulate("what about it?", history, settings=settings)
    # Falls through to raw query
    assert out == "what about it?"
