"""Self-test dry-run — ~10 wildlife-biologist-style questions against the
live Rocky Ridge corpus + self-evaluation of answer quality.

Runs directly against the chatbot's own retrieval + LLM pipeline (not
through the Chainlit UI). Outputs a Markdown scorecard to stdout.

Usage:
    # With chatbot repo .env loaded:
    python eval/run_dry_run.py
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import time
from dataclasses import dataclass

# Force UTF-8 stdout (Windows cp1252 can't print PDF private-use-area chars).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

from chatbot import foundry_client  # noqa: E402
from chatbot.citation_parser import extract_cited_ids, strip_unmatched  # noqa: E402
from chatbot.config import get_settings  # noqa: E402
from chatbot.prompt_builder import build_answer_messages  # noqa: E402
from chatbot.refusal_gate import should_refuse  # noqa: E402
from chatbot.retriever import retrieve  # noqa: E402


# In-corpus answerable (should be grounded, no refusal).
IN_CORPUS = [
    "What is canebrake restoration and why is it important?",
    "What is a prescribed burn and how does it support wildlife habitat?",
    "What is Timber Stand Improvement (TSI)?",
    "Which bird species benefit from early-successional habitat?",
    "What is the role of giant cane (Arundinaria gigantea) in riparian systems?",
    "What are the main threats to canebrake ecosystems?",
    "How does fire disturbance affect cane regeneration?",
    "What is the NFWF grant supporting in North Alabama?",
]

# Out-of-corpus — must refuse.
OUT_OF_CORPUS = [
    "What is the current market price of corn futures?",
    "Who won the 2024 World Series?",
]


@dataclass
class TurnResult:
    question: str
    retrieved_n: int
    max_norm_score: float
    refused_pre_gate: bool
    refused_model: bool
    cited_ids: list[str]
    unmatched_cites: list[str]
    answer: str
    latency_ms: int
    in_corpus_expected: bool


async def run_one(question: str, expected_in_corpus: bool) -> TurnResult:
    s = get_settings()
    start = time.perf_counter()
    result = await retrieve(question, settings=s)
    refuse_pre, _ = should_refuse(result)
    if refuse_pre:
        return TurnResult(
            question=question,
            retrieved_n=result.total_returned,
            max_norm_score=result.max_score_normalized,
            refused_pre_gate=True,
            refused_model=False,
            cited_ids=[],
            unmatched_cites=[],
            answer="(pre-gate refusal)",
            latency_ms=int((time.perf_counter() - start) * 1000),
            in_corpus_expected=expected_in_corpus,
        )

    messages = build_answer_messages(
        user_query=question,
        chunks=result.chunks,
        history=[],
        tenant_display_name=s.chatbot_tenant_display_name,
        short_id_map=result.short_id_map,
        max_history_turns=s.chatbot_max_history_turns,
    )
    parts: list[str] = []
    async for chunk in foundry_client.stream_chat(
        messages=messages,
        model_id=s.chatbot_model_id,
        temperature=s.chatbot_temperature,
        max_tokens=s.chatbot_max_tokens,
        settings=s,
    ):
        if chunk.content:
            parts.append(chunk.content)

    raw_answer = "".join(parts)
    cleaned, matched, unmatched = strip_unmatched(raw_answer, result.short_id_map)
    refused_model = (
        "don't have enough information" in cleaned.lower()
        or "don't have grounded" in cleaned.lower()
    )
    return TurnResult(
        question=question,
        retrieved_n=result.total_returned,
        max_norm_score=result.max_score_normalized,
        refused_pre_gate=False,
        refused_model=refused_model,
        cited_ids=matched,
        unmatched_cites=unmatched,
        answer=cleaned,
        latency_ms=int((time.perf_counter() - start) * 1000),
        in_corpus_expected=expected_in_corpus,
    )


def grade(r: TurnResult) -> tuple[str, str]:
    """Return (grade, rationale). Grades: PASS / WEAK / FAIL."""
    refused = r.refused_pre_gate or r.refused_model
    if r.in_corpus_expected:
        if refused:
            return "FAIL", "expected grounded answer but chatbot refused"
        if not r.cited_ids:
            return "WEAK", "answered without any citations (ungrounded)"
        if r.unmatched_cites:
            return (
                "WEAK",
                f"emitted {len(r.unmatched_cites)} fabricated citation(s), stripped",
            )
        return "PASS", f"grounded with {len(r.cited_ids)} citation(s)"
    else:
        if refused:
            return "PASS", "correctly refused out-of-corpus question"
        return "FAIL", "answered an out-of-corpus question (should have refused)"


async def main() -> int:
    s = get_settings()
    print(f"# Dry-Run Scorecard — {s.chatbot_tenant_display_name}")
    print(f"**Model:** `{s.chatbot_model_id}`  •  **Threshold:** {s.chatbot_refusal_threshold}")
    print()

    results: list[tuple[TurnResult, str, str]] = []

    for q in IN_CORPUS:
        r = await run_one(q, expected_in_corpus=True)
        g, why = grade(r)
        results.append((r, g, why))
        print(f"- **{g}** — {q}  _(retrieved={r.retrieved_n}, norm={r.max_norm_score:.2f}, cites={len(r.cited_ids)}, latency={r.latency_ms}ms)_")

    for q in OUT_OF_CORPUS:
        r = await run_one(q, expected_in_corpus=False)
        g, why = grade(r)
        results.append((r, g, why))
        print(f"- **{g}** — {q}  _(retrieved={r.retrieved_n}, norm={r.max_norm_score:.2f}, refused={r.refused_pre_gate or r.refused_model})_")

    # Summary
    passes = sum(1 for _, g, _ in results if g == "PASS")
    weaks = sum(1 for _, g, _ in results if g == "WEAK")
    fails = sum(1 for _, g, _ in results if g == "FAIL")
    total = len(results)
    print()
    print(f"## Summary")
    print(f"- PASS: {passes}/{total}")
    print(f"- WEAK: {weaks}/{total}")
    print(f"- FAIL: {fails}/{total}")
    print()
    print(f"## Pass criterion: ≥85% PASS across answerable + all out-of-corpus refuse.")
    total_answerable = len(IN_CORPUS)
    answerable_passes = sum(
        1 for r, g, _ in results if r.in_corpus_expected and g == "PASS"
    )
    print(f"- Answerable PASS rate: {answerable_passes}/{total_answerable} "
          f"({100 * answerable_passes / total_answerable:.0f}%)")

    # Detail for non-PASS
    non_pass = [(r, g, w) for r, g, w in results if g != "PASS"]
    if non_pass:
        print()
        print("## Non-PASS detail")
        for r, g, w in non_pass:
            print(f"\n### {g}: {r.question}")
            print(f"**Why:** {w}")
            print(f"**Answer:** {r.answer[:500]}")
            if r.unmatched_cites:
                print(f"**Fabricated cites stripped:** {r.unmatched_cites}")

    # Also dump a machine-readable JSON summary sibling to the markdown.
    sys.stderr.write(
        json.dumps(
            {
                "total": total,
                "pass": passes,
                "weak": weaks,
                "fail": fails,
                "answerable_pass_rate": answerable_passes / total_answerable,
            }
        )
        + "\n"
    )

    await foundry_client.close_client()
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
