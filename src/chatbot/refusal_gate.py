"""Pre-LLM refusal gate + post-LLM double-refusal detector."""

from __future__ import annotations

from chatbot.retriever import RetrievalResult


def should_refuse(result: RetrievalResult) -> tuple[bool, str]:
    """True if retrieval is too weak to ground an answer.
    Returns (should_refuse, reason_string) for logging."""
    if result.total_returned == 0:
        return True, "no_chunks_returned"
    if not result.is_sufficient:
        return True, (
            f"max_normalized_score_{result.max_score_normalized:.3f}_below_threshold"
        )
    return False, ""


# Substrings present in our canned refusal message. If any appears in the
# model's output despite the pre-gate passing, log it — calibration signal.
_REFUSAL_MARKERS = (
    "I don't have enough information",
    "don't have grounded information",
)


def contains_model_refusal(text: str) -> bool:
    """Detect whether the model itself refused even though pre-gate passed.
    Not user-visible; purely for telemetry (Stress Tester #6 —
    'double-refusal detector')."""
    lowered = text.lower()
    return any(m.lower() in lowered for m in _REFUSAL_MARKERS)
