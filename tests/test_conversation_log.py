"""Conversation log tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chatbot.conversation_log import ConversationLog, LogEntry, utcnow_iso


def _mk_entry(**overrides) -> LogEntry:
    base = dict(
        ts=utcnow_iso(),
        session_id="s1",
        turn_n=1,
        request_id="r1",
        user_msg="hi",
        reformulated_query="hi",
        reformulation_skipped=True,
        refused=False,
        refusal_reason=None,
        top_chunk_scores_raw=[0.015],
        top_chunk_scores_normalized=[0.46],
        cited_chunk_ids=["c_abc"],
        unmatched_ref_ids=[],
        post_llm_refusal=False,
        answer="Some answer [ref:c_abc].",
        latency_ms_total=1234,
        latency_ms_retrieval=234,
        error=None,
        finish_reason="stop",
    )
    base.update(overrides)
    return LogEntry(**base)


@pytest.mark.asyncio
async def test_append_writes_one_line(tmp_log_settings):
    path, s = tmp_log_settings
    log = ConversationLog(s)
    await log.append(_mk_entry())
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["session_id"] == "s1"
    assert data["cited_chunk_ids"] == ["c_abc"]


@pytest.mark.asyncio
async def test_append_does_not_raise_on_readonly_volume(
    tmp_log_settings, monkeypatch
):
    """Stress #3 — write failure must not kill the chat."""
    _, s = tmp_log_settings
    log = ConversationLog(s)

    class BadCtx:
        async def __aenter__(self):
            raise OSError("volume unmounted")

        async def __aexit__(self, *a, **k):
            return False

    def bad_open(*args, **kwargs):
        return BadCtx()

    monkeypatch.setattr("aiofiles.open", bad_open)
    # Must not raise
    await log.append(_mk_entry())


@pytest.mark.asyncio
async def test_rotation_triggers_at_threshold(tmp_log_settings, monkeypatch):
    # Pre-write a file larger than rotate threshold (1MB here via monkeypatch)
    path, _ = tmp_log_settings
    monkeypatch.setenv("CHATBOT_LOG_ROTATE_MB", "1")
    from chatbot.config import Settings

    s = Settings()  # type: ignore[call-arg]
    Path(s.chatbot_log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(s.chatbot_log_path).write_bytes(b"x" * (2 * 1024 * 1024))

    log = ConversationLog(s)
    await log.append(_mk_entry())

    # Original file should now be small (fresh write), rotated file exists
    base = Path(s.chatbot_log_path)
    # Rotated filename pattern: conversations-YYYYMMDDTHHMMSSZ.jsonl
    archives = list(base.parent.glob("conversations-*.jsonl"))
    assert len(archives) >= 1, f"No archives found in {base.parent}: {list(base.parent.iterdir())}"
    assert base.stat().st_size < 2 * 1024 * 1024


@pytest.mark.asyncio
async def test_refused_entry_serializes(tmp_log_settings):
    path, s = tmp_log_settings
    log = ConversationLog(s)
    await log.append(
        _mk_entry(
            refused=True,
            refusal_reason="max_normalized_score_0.100_below_threshold",
            cited_chunk_ids=[],
            answer="",
        )
    )
    data = json.loads(path.read_text().splitlines()[0])
    assert data["refused"] is True
    assert "below_threshold" in data["refusal_reason"]
