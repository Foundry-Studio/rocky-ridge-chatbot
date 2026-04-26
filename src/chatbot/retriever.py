"""Thin wrapper over foundry_client.search_knowledge.

Handles RRF score normalization (Day 0 pre-flight finding — raw RRF
scores with k=60 max at ~0.033, far below a naive 0.3 threshold).
Normalizes to 0-1 so Tim's locked 0.3 threshold has the intended
semantics (~30% of theoretical max retrieval quality).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from chatbot import foundry_client
from chatbot.config import Settings, get_settings
from chatbot.foundry_client import FileMetadata, KnowledgeChunk

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
    # source_file_id → FileMetadata (filename, ingestion date, etc.).
    # Best-effort enrichment for citation credibility; missing entries
    # mean the file-chunks lookup failed for that source_file_id.
    file_metadata_by_id: dict[str, FileMetadata] = field(default_factory=dict)


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
    chunks = resp.results
    raw_scores = [c.relevance_score or 0.0 for c in chunks]
    max_raw = max(raw_scores, default=0.0)
    max_norm = normalize_rrf_score(max_raw)
    is_sufficient = max_norm >= s.chatbot_refusal_threshold

    # Best-effort enrichment: fetch file metadata for unique source_file_ids
    # in parallel. Failures don't block the answer flow — we just lose the
    # filename + ingestion-date credibility info for that source.
    unique_file_ids = list(
        {c.source_file_id for c in chunks if c.source_file_id}
    )
    file_metadata_by_id: dict[str, FileMetadata] = {}
    if unique_file_ids:
        results = await asyncio.gather(
            *[
                foundry_client.get_file_metadata(
                    tenant_id=s.chatbot_tenant_id,
                    source_file_id=fid,
                    settings=s,
                )
                for fid in unique_file_ids
            ],
            return_exceptions=True,
        )
        for fid, r in zip(unique_file_ids, results, strict=False):
            if isinstance(r, FileMetadata):
                file_metadata_by_id[fid] = r
            elif isinstance(r, Exception):
                logger.warning(
                    "file metadata fetch raised for %s: %r", fid, r
                )

    latency_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "retrieve: returned=%d max_raw=%.4f max_norm=%.2f "
        "sufficient=%s latency_ms=%d files_enriched=%d/%d",
        resp.total,
        max_raw,
        max_norm,
        is_sufficient,
        latency_ms,
        len(file_metadata_by_id),
        len(unique_file_ids),
    )

    return RetrievalResult(
        chunks=chunks,
        max_score_normalized=max_norm,
        max_score_raw=max_raw,
        total_returned=resp.total,
        latency_ms=latency_ms,
        is_sufficient=is_sufficient,
        file_metadata_by_id=file_metadata_by_id,
    )
