"""Best-effort token-count-based cost estimation.

We don't get token counts from the SSE stream (Foundry's OpenAI-compat
endpoint doesn't emit a final usage object in our current setup), so we
approximate using char length. Imperfect but sufficient for the $50/day
circuit breaker — we err on the side of over-counting cost.
"""

from __future__ import annotations

# Claude Sonnet 4.5 pricing (USD per 1K tokens)
PRICE_PER_1K_IN = 0.003
PRICE_PER_1K_OUT = 0.015


def estimate_tokens_from_chars(n_chars: int) -> int:
    """Rough 4-char-per-token heuristic for English prose.
    Over-counts for code/data; fine for conservative budget tracking."""
    return max(1, n_chars // 4)


def estimate_cost_usd(input_chars: int, output_chars: int) -> float:
    in_tok = estimate_tokens_from_chars(input_chars)
    out_tok = estimate_tokens_from_chars(output_chars)
    return (
        in_tok * PRICE_PER_1K_IN / 1000.0
        + out_tok * PRICE_PER_1K_OUT / 1000.0
    )
