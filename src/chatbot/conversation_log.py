"""Append-only JSONL conversation log with size-based rotation.

Stress Tester #3 fix: uses aiofiles, no fsync. Best-effort durability is
acceptable for a demo log. Rotation at CHATBOT_LOG_ROTATE_MB bytes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import aiofiles

from chatbot.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    ts: str
    session_id: str
    turn_n: int
    request_id: str
    user_msg: str
    reformulated_query: str
    reformulation_skipped: bool
    refused: bool
    refusal_reason: str | None
    top_chunk_scores_raw: list[float]
    top_chunk_scores_normalized: list[float]
    cited_chunk_ids: list[str]
    unmatched_ref_ids: list[str]
    post_llm_refusal: bool
    answer: str
    latency_ms_total: int
    latency_ms_retrieval: int
    error: str | None
    finish_reason: str | None


class ConversationLog:
    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self._path = Path(self._s.chatbot_log_path)
        self._lock = asyncio.Lock()

    async def append(self, entry: LogEntry) -> None:
        line = json.dumps(asdict(entry), ensure_ascii=False)
        try:
            async with self._lock:
                await self._maybe_rotate()
                self._path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(
                    self._path, mode="a", encoding="utf-8"
                ) as f:
                    await f.write(line + "\n")
        except OSError as e:
            # Never propagate — Stress Tester #3: log failure must not kill chat.
            logger.error("log_write_failed: %s entry=%s", e, line[:200])

    async def _maybe_rotate(self) -> None:
        try:
            if not self._path.exists():
                return
            size_mb = self._path.stat().st_size / (1024 * 1024)
            if size_mb < self._s.chatbot_log_rotate_mb:
                return
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            # Name pattern: conversations-YYYYMMDDTHHMMSSZ.jsonl — keeps the
            # archive glob ('conversations-*.jsonl') disjoint from the live file.
            stem = self._path.stem  # 'conversations'
            rotated = self._path.with_name(f"{stem}-{stamp}.jsonl")
            os.rename(self._path, rotated)
            logger.info("rotated conversation log → %s", rotated.name)
            # Fire-and-forget cleanup of stale archives.
            try:
                self._purge_old_archives()
            except Exception as e:
                logger.warning("archive purge failed: %s", e)
        except OSError as e:
            logger.warning("rotate check failed: %s", e)

    def _purge_old_archives(self) -> None:
        cutoff = time.time() - self._s.chatbot_log_retention_days * 86400
        parent = self._path.parent
        stem = self._path.stem
        for p in parent.glob(f"{stem}-*.jsonl"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    logger.info("purged old log archive: %s", p.name)
            except OSError:
                pass


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()
