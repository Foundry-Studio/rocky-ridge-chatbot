"""Tool definitions for agentic RAG.

Three tools, all backed by existing Foundry endpoints:

    1. search_knowledge      — POST /api/v1/knowledge/search
    2. get_chunk_neighbors   — POST /api/v1/knowledge/file-chunks (paginated)
    3. read_document_section — POST /api/v1/knowledge/file-chunks (paginated)

Each tool returns a JSON-serializable dict that gets fed back to the LLM
as a `tool` message. On error, the result is still returned (with
``error`` field populated) so the LLM can see what went wrong and retry
or proceed gracefully — tools NEVER raise from the perspective of the
agent loop.

Citation note: every chunk-bearing result includes ``chunk_id`` (stable
UUID) so the agent loop's chunks_seen map can resolve [N] back to a
specific source across multiple tool calls.
"""

from __future__ import annotations

import logging
from typing import Any

from chatbot import foundry_client
from chatbot.config import Settings
from chatbot.exceptions import (
    FoundryAuthError,
    FoundryMalformedResponseError,
    FoundryTransientError,
)
from chatbot.foundry_client import KnowledgeChunk

logger = logging.getLogger(__name__)


# ── Tool schemas (OpenAI tools-array format) ──────────────────────────


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Hybrid search (BM25 + vector + RRF) over the knowledge base. "
                "Use this as your primary information-gathering tool. "
                "You can call it multiple times with different queries to gather "
                "different angles. If results are weak (low scores, off-topic), "
                "reformulate and search again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language search query. Be specific. "
                            "Multi-word phrases work better than single terms. "
                            "Example: 'canebrake restoration prescribed fire'."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (1-15). Default 8.",
                        "minimum": 1,
                        "maximum": 15,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunk_neighbors",
            "description": (
                "Pull the chunks immediately before and after a given chunk "
                "from the same document. Use when a chunk is relevant but "
                "feels mid-thought, or when you want to read a methods/results "
                "passage in flow. Pass the chunk_id from a prior search result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "UUID of a chunk previously returned by search_knowledge.",
                    },
                    "before": {
                        "type": "integer",
                        "description": "Number of chunks before to include (0-5). Default 2.",
                        "minimum": 0,
                        "maximum": 5,
                    },
                    "after": {
                        "type": "integer",
                        "description": "Number of chunks after to include (0-5). Default 2.",
                        "minimum": 0,
                        "maximum": 5,
                    },
                },
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document_section",
            "description": (
                "Read a window of consecutive chunks from a single document, "
                "in document order. Use when you've identified a highly relevant "
                "document via search_knowledge and want to read a section more "
                "thoroughly than one chunk allows (methods, results, "
                "conclusions, etc.). For long docs, paginate by calling again "
                "with a higher start_chunk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "source_file_id from a prior search result.",
                    },
                    "start_chunk": {
                        "type": "integer",
                        "description": "0-based index of the first chunk. Default 0.",
                        "minimum": 0,
                    },
                    "chunk_count": {
                        "type": "integer",
                        "description": "Number of chunks to fetch (1-25). Default 10.",
                        "minimum": 1,
                        "maximum": 25,
                    },
                },
                "required": ["document_id"],
            },
        },
    },
]


# ── Tool dispatch + execution ──────────────────────────────────────────


def _serialize_chunk(chunk: KnowledgeChunk, n: int) -> dict[str, Any]:
    """Compact LLM-facing chunk view. Includes the global [N] number
    so the LLM can cite by it."""
    return {
        "n": n,
        "chunk_id": chunk.chunk_id,
        "document": chunk.section_title or "—",
        "source_file_id": chunk.source_file_id,
        "library": chunk.source_name,
        "pages": chunk.page_numbers or [],
        "score": round(chunk.relevance_score or 0.0, 4),
        "retrieval": chunk.retrieval_method or "—",
        "content": (chunk.content or "")[:1500],
    }


async def _err(name: str, message: str, retryable: bool = False) -> dict[str, Any]:
    logger.warning("tool %s error: %s", name, message)
    return {"error": message, "is_error": True, "retryable": retryable}


async def execute_search_knowledge(
    args: dict[str, Any],
    chunks_seen: dict[str, KnowledgeChunk],
    chunk_id_to_n: dict[str, int],
    settings: Settings,
) -> dict[str, Any]:
    """Run search_knowledge against Foundry and update chunks_seen.

    Each chunk gets a globally-monotonic [N] assigned at first sighting.
    Returns the LLM-facing JSON: {chunks: [...]} where each chunk has
    its global N.
    """
    query = (args.get("query") or "").strip()
    if not query:
        return await _err(
            "search_knowledge", "missing or empty 'query' arg"
        )
    top_k = int(args.get("top_k") or 8)
    top_k = max(1, min(15, top_k))

    try:
        resp = await foundry_client.search_knowledge(
            tenant_id=settings.chatbot_tenant_id,
            query=query,
            max_results=top_k,
            source_id=settings.chatbot_knowledge_source_id,
            settings=settings,
        )
    except FoundryAuthError:
        raise  # surface to agent loop — auth issues should NOT be silently swallowed
    except FoundryTransientError as e:
        return await _err("search_knowledge", f"transient: {e}", retryable=True)
    except FoundryMalformedResponseError as e:
        return await _err("search_knowledge", f"malformed: {e}")

    out_chunks: list[dict[str, Any]] = []
    for c in resp.results:
        if c.chunk_id not in chunk_id_to_n:
            n = len(chunk_id_to_n) + 1
            chunk_id_to_n[c.chunk_id] = n
            chunks_seen[c.chunk_id] = c
        n = chunk_id_to_n[c.chunk_id]
        out_chunks.append(_serialize_chunk(c, n))

    return {
        "query": query,
        "total": resp.total,
        "chunks": out_chunks,
        "note": (
            "Each chunk has a global [N] number — cite findings inline as "
            "[N] (e.g., [3]). For multiple sources: [3][7]. Numbers persist "
            "across all your tool calls this turn."
        )
        if out_chunks
        else "No matches. Reformulate the query and try again.",
    }


async def execute_get_chunk_neighbors(
    args: dict[str, Any],
    chunks_seen: dict[str, KnowledgeChunk],
    chunk_id_to_n: dict[str, int],
    settings: Settings,
) -> dict[str, Any]:
    """Pull chunks before/after a known chunk via file-chunks API."""
    chunk_id = (args.get("chunk_id") or "").strip()
    if not chunk_id:
        return await _err(
            "get_chunk_neighbors", "missing or empty 'chunk_id' arg"
        )
    if chunk_id not in chunks_seen:
        return await _err(
            "get_chunk_neighbors",
            f"chunk_id {chunk_id} not in current research session — "
            "use search_knowledge first to discover chunks, then reference their chunk_id.",
        )
    before = max(0, min(5, int(args.get("before") or 2)))
    after = max(0, min(5, int(args.get("after") or 2)))

    seed = chunks_seen[chunk_id]
    if not seed.source_file_id:
        return await _err(
            "get_chunk_neighbors",
            f"chunk {chunk_id} has no source_file_id; cannot fetch neighbors.",
        )

    # We need the seed chunk's index in the document. file-chunks pagination
    # doesn't expose a direct lookup-by-chunk_id; we page the document and
    # find the index.
    try:
        # Fetch first page to get total_chunks; if doc is larger than a few
        # pages, we widen iteratively. For most docs ≤25 chunks one call is
        # enough. For larger docs, we fall back to getting just a window
        # around the seed by binary search through pages.
        from chatbot import foundry_client as fc
        _client = fc.get_client(settings)
        page_size = 25
        page_start = 0
        seed_index: int | None = None
        total_chunks: int = 0
        scanned: list[dict[str, Any]] = []
        for _ in range(8):  # cap pages we scan to avoid runaway
            r = await _client.post(
                "/api/v1/knowledge/file-chunks",
                json={
                    "tenant_id": settings.chatbot_tenant_id,
                    "source_file_id": seed.source_file_id,
                    "chunk_start": page_start,
                    "chunk_limit": page_size,
                },
                timeout=settings.retrieval_timeout_s,
            )
            if r.status_code != 200:
                return await _err(
                    "get_chunk_neighbors",
                    f"file-chunks page fetch returned {r.status_code}",
                )
            data = r.json()
            total_chunks = data.get("total_chunks", 0)
            page_chunks = data.get("chunks") or []
            scanned.extend(page_chunks)
            for c in page_chunks:
                if c.get("chunk_id") == chunk_id:
                    seed_index = c.get("chunk_index")
                    break
            if seed_index is not None:
                break
            if not data.get("has_more"):
                break
            page_start += page_size

        if seed_index is None:
            return await _err(
                "get_chunk_neighbors",
                f"could not locate chunk {chunk_id} in document {seed.source_file_id}",
            )
    except Exception as e:
        return await _err(
            "get_chunk_neighbors",
            f"file-chunks scan: {type(e).__name__}: {e}",
            retryable=True,
        )

    start_chunk = max(0, seed_index - before)
    chunk_count = before + 1 + after

    try:
        r = await _client.post(
            "/api/v1/knowledge/file-chunks",
            json={
                "tenant_id": settings.chatbot_tenant_id,
                "source_file_id": seed.source_file_id,
                "chunk_start": start_chunk,
                "chunk_limit": chunk_count,
            },
            timeout=settings.retrieval_timeout_s,
        )
        if r.status_code != 200:
            return await _err(
                "get_chunk_neighbors",
                f"window fetch returned {r.status_code}",
            )
        data = r.json()
    except Exception as e:
        return await _err(
            "get_chunk_neighbors",
            f"window fetch: {type(e).__name__}: {e}",
            retryable=True,
        )

    out_chunks: list[dict[str, Any]] = []
    for raw in data.get("chunks", []):
        # Build a KnowledgeChunk view from the file-chunks shape so we can
        # share the same serializer + chunks_seen registration.
        try:
            chunk = KnowledgeChunk.model_validate(
                {
                    **raw,
                    "source_name": seed.source_name,
                    "source_file_id": seed.source_file_id,
                }
            )
        except Exception as e:
            logger.warning("neighbor chunk validate failed: %s", e)
            continue
        if chunk.chunk_id not in chunk_id_to_n:
            chunk_id_to_n[chunk.chunk_id] = len(chunk_id_to_n) + 1
            chunks_seen[chunk.chunk_id] = chunk
        n = chunk_id_to_n[chunk.chunk_id]
        out_chunks.append(_serialize_chunk(chunk, n))

    return {
        "seed_chunk_id": chunk_id,
        "seed_chunk_index": seed_index,
        "document_total_chunks": total_chunks,
        "window": {
            "start_chunk": start_chunk,
            "chunk_count": chunk_count,
        },
        "chunks": out_chunks,
    }


async def execute_read_document_section(
    args: dict[str, Any],
    chunks_seen: dict[str, KnowledgeChunk],
    chunk_id_to_n: dict[str, int],
    settings: Settings,
) -> dict[str, Any]:
    """Paginate consecutive chunks from a single document."""
    document_id = (args.get("document_id") or "").strip()
    if not document_id:
        return await _err(
            "read_document_section", "missing or empty 'document_id'"
        )
    start_chunk = max(0, int(args.get("start_chunk") or 0))
    chunk_count = max(1, min(25, int(args.get("chunk_count") or 10)))

    try:
        from chatbot import foundry_client as fc
        _client = fc.get_client(settings)
        r = await _client.post(
            "/api/v1/knowledge/file-chunks",
            json={
                "tenant_id": settings.chatbot_tenant_id,
                "source_file_id": document_id,
                "chunk_start": start_chunk,
                "chunk_limit": chunk_count,
            },
            timeout=settings.retrieval_timeout_s,
        )
        if r.status_code in (401, 403):
            raise FoundryAuthError(f"file-chunks auth: HTTP {r.status_code}")
        if r.status_code != 200:
            return await _err(
                "read_document_section",
                f"file-chunks returned {r.status_code}",
                retryable=r.status_code >= 500,
            )
        data = r.json()
    except FoundryAuthError:
        raise
    except Exception as e:
        return await _err(
            "read_document_section",
            f"transport: {type(e).__name__}: {e}",
            retryable=True,
        )

    file_obj = data.get("file") or {}
    if not file_obj:
        return await _err(
            "read_document_section",
            f"no file metadata for document_id {document_id} (wrong tenant or not found)",
        )

    out_chunks: list[dict[str, Any]] = []
    for raw in data.get("chunks") or []:
        try:
            chunk = KnowledgeChunk.model_validate(
                {
                    **raw,
                    "source_name": file_obj.get("source_name"),
                    "source_file_id": document_id,
                }
            )
        except Exception as e:
            logger.warning("read_document chunk validate failed: %s", e)
            continue
        if chunk.chunk_id not in chunk_id_to_n:
            chunk_id_to_n[chunk.chunk_id] = len(chunk_id_to_n) + 1
            chunks_seen[chunk.chunk_id] = chunk
        n = chunk_id_to_n[chunk.chunk_id]
        out_chunks.append(_serialize_chunk(chunk, n))

    return {
        "document": {
            "filename": file_obj.get("original_filename"),
            "total_chunks": data.get("total_chunks"),
            "size_bytes": file_obj.get("size_bytes"),
        },
        "window": {
            "start_chunk": start_chunk,
            "chunk_count": chunk_count,
            "has_more": data.get("has_more", False),
        },
        "chunks": out_chunks,
    }


# ── Dispatch table ─────────────────────────────────────────────────────


TOOL_DISPATCH = {
    "search_knowledge": execute_search_knowledge,
    "get_chunk_neighbors": execute_get_chunk_neighbors,
    "read_document_section": execute_read_document_section,
}
