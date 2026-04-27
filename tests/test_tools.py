"""Tool dispatch tests — covers schema shape + the three tool executors.

Search-path tests mock ``foundry_client.search_knowledge`` directly.
File-chunks-path tests mock the underlying httpx client via respx.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from chatbot import foundry_client, tools
from chatbot.exceptions import FoundryAuthError
from chatbot.foundry_client import KnowledgeChunk, KnowledgeSearchResponse

# ── Schema sanity ─────────────────────────────────────────────────────


def test_tool_schemas_shape():
    names = [t["function"]["name"] for t in tools.TOOL_SCHEMAS]
    assert names == [
        "search_knowledge",
        "get_chunk_neighbors",
        "read_document_section",
    ]
    for t in tools.TOOL_SCHEMAS:
        assert t["type"] == "function"
        assert "description" in t["function"]
        assert "parameters" in t["function"]
        assert t["function"]["parameters"]["type"] == "object"


def test_tool_dispatch_keys_match_schemas():
    schema_names = {t["function"]["name"] for t in tools.TOOL_SCHEMAS}
    assert schema_names == set(tools.TOOL_DISPATCH.keys())


def _chunk(chunk_id: str, content: str = "stuff", **kw) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        content=content,
        relevance_score=kw.get("relevance_score", 0.5),
        source_file_id=kw.get("source_file_id", "doc-1"),
        source_name=kw.get("source_name", "Rocky Ridge KB"),
        section_title=kw.get("section_title", "Section A"),
    )


# ── search_knowledge ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_knowledge_assigns_global_n(settings, monkeypatch):
    chunks_seen: dict = {}
    chunk_id_to_n: dict = {}

    async def fake_search(**kw):
        return KnowledgeSearchResponse(
            results=[_chunk("c1"), _chunk("c2")],
            total=2,
            query=kw["query"],
            tenant_id=kw["tenant_id"],
        )

    monkeypatch.setattr(foundry_client, "search_knowledge", fake_search)

    out = await tools.execute_search_knowledge(
        {"query": "fire", "top_k": 2},
        chunks_seen,
        chunk_id_to_n,
        settings,
    )
    assert "chunks" in out
    ns = [c["n"] for c in out["chunks"]]
    assert ns == [1, 2]
    assert chunk_id_to_n == {"c1": 1, "c2": 2}
    assert set(chunks_seen.keys()) == {"c1", "c2"}


@pytest.mark.asyncio
async def test_search_knowledge_n_is_stable_across_calls(settings, monkeypatch):
    chunks_seen: dict = {}
    chunk_id_to_n: dict = {}

    async def fake_search(**kw):
        # Both calls return overlapping chunk c1 + a new one
        if "fire" in kw["query"]:
            return KnowledgeSearchResponse(
                results=[_chunk("c1"), _chunk("c2")],
                total=2,
                query=kw["query"],
                tenant_id=kw["tenant_id"],
            )
        return KnowledgeSearchResponse(
            results=[_chunk("c1"), _chunk("c3")],
            total=2,
            query=kw["query"],
            tenant_id=kw["tenant_id"],
        )

    monkeypatch.setattr(foundry_client, "search_knowledge", fake_search)
    await tools.execute_search_knowledge(
        {"query": "fire"}, chunks_seen, chunk_id_to_n, settings
    )
    out2 = await tools.execute_search_knowledge(
        {"query": "soil"}, chunks_seen, chunk_id_to_n, settings
    )
    # c1 keeps its original N=1
    n_by_id = {c["chunk_id"]: c["n"] for c in out2["chunks"]}
    assert n_by_id["c1"] == 1
    assert n_by_id["c3"] == 3


@pytest.mark.asyncio
async def test_search_knowledge_empty_query_errors(settings):
    chunks_seen: dict = {}
    chunk_id_to_n: dict = {}
    out = await tools.execute_search_knowledge(
        {"query": "  "}, chunks_seen, chunk_id_to_n, settings
    )
    assert out["is_error"] is True
    assert "query" in out["error"]


@pytest.mark.asyncio
async def test_search_knowledge_top_k_clamped(settings, monkeypatch):
    captured: dict = {}

    async def fake_search(**kw):
        captured.update(kw)
        return KnowledgeSearchResponse(
            results=[], total=0, query=kw["query"], tenant_id=kw["tenant_id"]
        )

    monkeypatch.setattr(foundry_client, "search_knowledge", fake_search)
    await tools.execute_search_knowledge(
        {"query": "x", "top_k": 999}, {}, {}, settings
    )
    assert captured["max_results"] == 15
    await tools.execute_search_knowledge(
        {"query": "x", "top_k": -3}, {}, {}, settings
    )
    assert captured["max_results"] == 1


@pytest.mark.asyncio
async def test_search_knowledge_auth_bubbles(settings, monkeypatch):
    async def fake_search(**kw):
        raise FoundryAuthError("nope")

    monkeypatch.setattr(foundry_client, "search_knowledge", fake_search)
    with pytest.raises(FoundryAuthError):
        await tools.execute_search_knowledge(
            {"query": "x"}, {}, {}, settings
        )


# ── get_chunk_neighbors ───────────────────────────────────────────────


def _seed_chunks_seen() -> tuple[dict, dict]:
    seed = _chunk("c1", source_file_id="doc-1", source_name="lib")
    return {"c1": seed}, {"c1": 1}


def _mock_httpx_client():
    c = httpx.AsyncClient(
        base_url="https://foundry-agent-system-test.invalid",
        headers={
            "X-Internal-Token": "test",
            "X-Actor-Type": "ai_assistant",
            "X-Actor-ID": "rocky-ridge-chatbot",
            "Content-Type": "application/json",
        },
    )
    foundry_client.set_test_client(c)
    return c


@pytest.mark.asyncio
async def test_get_chunk_neighbors_unknown_chunk_id(settings):
    out = await tools.execute_get_chunk_neighbors(
        {"chunk_id": "ghost"}, {}, {}, settings
    )
    assert out["is_error"] is True
    assert "not in current research session" in out["error"]


@pytest.mark.asyncio
async def test_get_chunk_neighbors_happy_path(settings):
    chunks_seen, chunk_id_to_n = _seed_chunks_seen()
    async with respx.mock(
        base_url="https://foundry-agent-system-test.invalid"
    ) as mock:
        # First call: scan to find seed_index
        mock.post("/api/v1/knowledge/file-chunks").mock(
            side_effect=[
                # scan response
                httpx.Response(
                    200,
                    json={
                        "total_chunks": 5,
                        "has_more": False,
                        "chunks": [
                            {"chunk_id": "c0", "chunk_index": 0, "content": "a"},
                            {"chunk_id": "c1", "chunk_index": 1, "content": "b"},
                            {"chunk_id": "c2", "chunk_index": 2, "content": "c"},
                        ],
                    },
                ),
                # window response
                httpx.Response(
                    200,
                    json={
                        "total_chunks": 5,
                        "has_more": False,
                        "chunks": [
                            {"chunk_id": "c0", "chunk_index": 0, "content": "a"},
                            {"chunk_id": "c1", "chunk_index": 1, "content": "b"},
                            {"chunk_id": "c2", "chunk_index": 2, "content": "c"},
                        ],
                    },
                ),
            ]
        )
        _mock_httpx_client()
        out = await tools.execute_get_chunk_neighbors(
            {"chunk_id": "c1", "before": 1, "after": 1},
            chunks_seen,
            chunk_id_to_n,
            settings,
        )
    assert out.get("is_error") is None or out.get("is_error") is False
    assert out["seed_chunk_index"] == 1
    assert out["window"]["start_chunk"] == 0
    assert out["window"]["chunk_count"] == 3
    # New chunks c0, c2 picked up global N
    assert "c0" in chunk_id_to_n
    assert "c2" in chunk_id_to_n
    assert chunk_id_to_n["c1"] == 1  # seed N preserved


# ── read_document_section ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_document_section_happy_path(settings):
    chunks_seen: dict = {}
    chunk_id_to_n: dict = {}
    async with respx.mock(
        base_url="https://foundry-agent-system-test.invalid"
    ) as mock:
        mock.post("/api/v1/knowledge/file-chunks").mock(
            return_value=httpx.Response(
                200,
                json={
                    "file": {
                        "file_id": "doc-1",
                        "original_filename": "report.pdf",
                        "source_name": "Rocky Ridge KB",
                        "size_bytes": 12345,
                    },
                    "total_chunks": 30,
                    "has_more": True,
                    "chunks": [
                        {"chunk_id": "c10", "chunk_index": 10, "content": "x"},
                        {"chunk_id": "c11", "chunk_index": 11, "content": "y"},
                    ],
                },
            )
        )
        _mock_httpx_client()
        out = await tools.execute_read_document_section(
            {"document_id": "doc-1", "start_chunk": 10, "chunk_count": 2},
            chunks_seen,
            chunk_id_to_n,
            settings,
        )
    assert out["document"]["filename"] == "report.pdf"
    assert out["window"]["has_more"] is True
    assert chunk_id_to_n == {"c10": 1, "c11": 2}


@pytest.mark.asyncio
async def test_read_document_section_auth_raises(settings):
    async with respx.mock(
        base_url="https://foundry-agent-system-test.invalid"
    ) as mock:
        mock.post("/api/v1/knowledge/file-chunks").mock(
            return_value=httpx.Response(401)
        )
        _mock_httpx_client()
        with pytest.raises(FoundryAuthError):
            await tools.execute_read_document_section(
                {"document_id": "doc-1"}, {}, {}, settings
            )


@pytest.mark.asyncio
async def test_read_document_section_missing_id(settings):
    out = await tools.execute_read_document_section(
        {"document_id": ""}, {}, {}, settings
    )
    assert out["is_error"] is True


@pytest.mark.asyncio
async def test_read_document_section_chunk_count_clamped(settings):
    import json as _json

    async with respx.mock(
        base_url="https://foundry-agent-system-test.invalid"
    ) as mock:
        route = mock.post("/api/v1/knowledge/file-chunks").mock(
            return_value=httpx.Response(
                200,
                json={
                    "file": {"file_id": "doc-1"},
                    "total_chunks": 0,
                    "has_more": False,
                    "chunks": [],
                },
            )
        )
        _mock_httpx_client()
        await tools.execute_read_document_section(
            {"document_id": "doc-1", "chunk_count": 999}, {}, {}, settings
        )
        body = _json.loads(route.calls.last.request.content.decode())
    assert body["chunk_limit"] == 25
