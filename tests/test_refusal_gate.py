"""Refusal-gate tests."""

from __future__ import annotations

from chatbot.foundry_client import KnowledgeChunk
from chatbot.refusal_gate import contains_model_refusal, should_refuse
from chatbot.retriever import RetrievalResult


def _make_result(chunks, max_norm, max_raw, total, is_suff) -> RetrievalResult:
    return RetrievalResult(
        chunks=chunks,
        max_score_normalized=max_norm,
        max_score_raw=max_raw,
        total_returned=total,
        latency_ms=10,
        is_sufficient=is_suff,
    )


def test_refuse_when_no_chunks():
    r = _make_result([], 0.0, 0.0, 0, False)
    refuse, reason = should_refuse(r)
    assert refuse is True
    assert reason == "no_chunks_returned"


def test_refuse_when_below_threshold():
    c = KnowledgeChunk(chunk_id="x", content="weak")
    r = _make_result([c], 0.1, 0.003, 1, False)
    refuse, reason = should_refuse(r)
    assert refuse is True
    assert "below_threshold" in reason


def test_pass_when_sufficient():
    c = KnowledgeChunk(chunk_id="x", content="strong")
    r = _make_result([c], 0.8, 0.025, 1, True)
    refuse, reason = should_refuse(r)
    assert refuse is False
    assert reason == ""


def test_model_refusal_detector_catches_canned_phrase():
    text = (
        "I don't have enough information in the knowledge base to answer "
        "that confidently."
    )
    assert contains_model_refusal(text) is True


def test_model_refusal_detector_ignores_normal_answer():
    text = "Canebrake restoration involves prescribed burning and cane propagation."
    assert contains_model_refusal(text) is False


def test_model_refusal_detector_case_insensitive():
    text = "i don't HAVE enough information"
    assert contains_model_refusal(text) is True
