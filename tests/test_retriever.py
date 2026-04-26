"""Retriever tests — RRF normalization is the load-bearing logic."""

from __future__ import annotations

import httpx
import pytest
import respx
from chatbot import foundry_client
from chatbot.retriever import (
    _RRF_SCORE_MAX,
    normalize_rrf_score,
    retrieve,
)


def test_normalize_zero_is_zero():
    assert normalize_rrf_score(0.0) == 0.0


def test_normalize_max_raw_is_one():
    assert normalize_rrf_score(_RRF_SCORE_MAX) == pytest.approx(1.0)


def test_normalize_half_max_is_half():
    assert normalize_rrf_score(_RRF_SCORE_MAX / 2) == pytest.approx(0.5)


def test_normalize_clamps_above_max():
    assert normalize_rrf_score(_RRF_SCORE_MAX * 5) == 1.0


def _make_client_against(mock_transport_url: str):
    client = httpx.AsyncClient(
        base_url="https://foundry-test.invalid",
        headers={
            "X-Internal-Token": "test",
            "X-Actor-Type": "ai_assistant",
            "X-Actor-ID": "rocky-ridge-chatbot",
            "Content-Type": "application/json",
        },
    )
    foundry_client.set_test_client(client)
    return client


@pytest.mark.asyncio
async def test_retrieve_happy_path_with_real_rrf_scores(settings):
    """Top chunk raw=0.0167 (rank-1 single-list) → normalized ≈ 0.51.
    With threshold 0.3, must be sufficient."""
    async with respx.mock(
        base_url="https://foundry-test.invalid", assert_all_called=False
    ) as mock:
        mock.post("/api/v1/knowledge/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "chunk_id": "11111111-2222-3333-4444-555555555555",
                            "content": "Cane grows along riparian areas.",
                            "relevance_score": 0.01666666,  # real RRF rank-1 single-list
                            "source_file_id": "file1",
                            "source_name": "Guide",
                            "page_numbers": [3, 4],
                            "section_title": "Canebrake",
                            "authority_level": "validated",
                            "source_id": "src1",
                        },
                    ],
                    "total": 1,
                    "query": "canebrake",
                    "tenant_id": settings.chatbot_tenant_id,
                },
            )
        )
        _make_client_against("https://foundry-test.invalid")
        # Point foundry_client at mock
        import chatbot.foundry_client as fc

        fc._client = httpx.AsyncClient(
            base_url="https://foundry-test.invalid",
            headers={
                "X-Internal-Token": "test",
                "X-Actor-Type": "ai_assistant",
                "X-Actor-ID": "rocky-ridge-chatbot",
            },
        )
        result = await retrieve("canebrake", settings=settings)
    assert result.total_returned == 1
    assert result.max_score_raw == pytest.approx(0.01666666)
    assert result.max_score_normalized > 0.5
    assert result.is_sufficient is True
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "11111111-2222-3333-4444-555555555555"


@pytest.mark.asyncio
async def test_retrieve_empty_result_is_insufficient(settings):
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/knowledge/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [],
                    "total": 0,
                    "query": "bogus",
                    "tenant_id": settings.chatbot_tenant_id,
                },
            )
        )
        import chatbot.foundry_client as fc

        fc._client = httpx.AsyncClient(
            base_url="https://foundry-test.invalid",
            headers={
                "X-Internal-Token": "test",
                "X-Actor-Type": "ai_assistant",
                "X-Actor-ID": "rocky-ridge-chatbot",
            },
        )
        result = await retrieve("bogus", settings=settings)
    assert result.total_returned == 0
    assert result.max_score_normalized == 0.0
    assert result.is_sufficient is False


@pytest.mark.asyncio
async def test_retrieve_below_threshold_is_insufficient(settings):
    """Raw 0.0001 → normalized ~0.003 → below 0.3 threshold → refuse."""
    async with respx.mock(base_url="https://foundry-test.invalid") as mock:
        mock.post("/api/v1/knowledge/search").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "chunk_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                            "content": "barely relevant",
                            "relevance_score": 0.0001,
                        }
                    ],
                    "total": 1,
                    "query": "weak match",
                    "tenant_id": settings.chatbot_tenant_id,
                },
            )
        )
        import chatbot.foundry_client as fc

        fc._client = httpx.AsyncClient(
            base_url="https://foundry-test.invalid",
            headers={
                "X-Internal-Token": "test",
                "X-Actor-Type": "ai_assistant",
                "X-Actor-ID": "rocky-ridge-chatbot",
            },
        )
        result = await retrieve("weak match", settings=settings)
    assert result.is_sufficient is False
