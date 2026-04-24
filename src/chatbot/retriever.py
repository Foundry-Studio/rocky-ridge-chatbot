"""Thin wrapper over foundry_client.search_knowledge.

Handles RRF score normalization (Day 0 pre-flight finding — raw RRF
scores with k=60 max at ~0.033, far below a naive 0.3 threshold).
Normalizes to 0-1 so Tim's locked 0.3 threshold has the intended
semantics (~30% of theoretical max retrieval quality).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from chatbot import foundry_client
from chatbot.config import Settings, get_settings
from chatbot.foundry_client import KnowledgeChunk

logger = logging.getLogger(__name__)


# Foundry's retriever uses RRF with k=60. A chunk at rank 1 in BOTH the
# BM25 list and the vector list scores 2/(60+1) ≈ 0.0328. That's the
# ceiling for a single-chunk score. We normalize against it so the
# refusal threshold lives in intuitive 0-1 space.
_RRF_K = 60
_RRF_SCORE_MAX = 2.0 / (_RRF_K + 1)  # ≈ 0.03278


def normalize_rrf_score(raw: float) -> float:
    """Map raw RRF score → 0-1 scale. Clamped at 1.0 (rare, but possible if
    both lists agree at rank 1)."""
    if raw <= 0:
        return 0.0
    return min(raw / _RRF_SCORE_MAX, 1.0)


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[KnowledgeChunk]
    max_score_normalized: float  # 0-1
    max_score_raw: float  # original RRF
    total_returned: int
    latency_ms: int
    is_sufficient: bool  # max_score_normalized >= refusal_threshold
    short_id_map: dict[str, str] = field(default_factory=dict)
    # short_id (e.g. "c_3f2a18b7") → full chunk_id UUID


def _short_id(chunk_id: str) -> str:
    """Derive a short, Sonnet-friendly citation ID from the full UUID.
    First 8 hex chars of the stripped UUID, prefixed c_."""
    hex_only = chunk_id.replace("-", "")
    return f"c_{hex_only[:8]}"


async def retrieve(
    query: str, settings: Settings | None = None
) -> RetrievalResult:
    s = settings or get_settings()
    started = time.perf_counter()
    resp = await foundry_client.search_knowledge(
        tenant_id=s.chatbot_tenant_id,
        query=query,
        max_results=s.chatbot_max_chunks,
        source_id=s.chatbot_knowledge_source_id,
        settings=s,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    chunks = resp.results
    raw_scores = [c.relevance_score or 0.0 for c in chunks]
    max_raw = max(raw_scores, default=0.0)
    max_norm = normalize_rrf_score(max_raw)

    short_map: dict[str, str] = {}
    for c in chunks:
        sid = _short_id(c.chunk_id)
        # Collision guard — vanishingly rare but defend anyway.
        if sid in short_map and short_map[sid] != c.chunk_id:
            sid = f"c_{c.chunk_id.replace('-', '')[:12]}"
        short_map[sid] = c.chunk_id

    is_sufficient = max_norm >= s.chatbot_refusal_threshold
    logger.info(
        "retrieve: returned=%d max_raw=%.4f max_norm=%.2f "
        "sufficient=%s latency_ms=%d",
        resp.total,
        max_raw,
        max_norm,
        is_sufficient,
        latency_ms,
    )

    return RetrievalResult(
        chunks=chunks,
        max_score_normalized=max_norm,
        max_score_raw=max_raw,
        total_returned=resp.total,
        latency_ms=latency_ms,
        is_sufficient=is_sufficient,
        short_id_map=short_map,
    )
