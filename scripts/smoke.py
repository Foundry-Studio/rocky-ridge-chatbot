"""One-off end-to-end smoke test against live Foundry.

Usage:
    python scripts/smoke.py "what is canebrake"

Env required: FOUNDRY_API_BASE_URL, FOUNDRY_INTERNAL_TOKEN,
CHATBOT_TENANT_ID, CHATBOT_TENANT_DISPLAY_NAME, CHATBOT_MODEL_ID.
"""

from __future__ import annotations

import asyncio
import io
import sys

# Force UTF-8 stdout so PDF-extracted chunks (which often contain Private-Use-
# Area unicode from bad OCR) don't crash cp1252-defaulted Windows consoles.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

from chatbot import foundry_client
from chatbot.citation_parser import strip_unmatched
from chatbot.config import get_settings
from chatbot.prompt_builder import build_answer_messages
from chatbot.refusal_gate import should_refuse
from chatbot.retriever import retrieve


async def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/smoke.py \"your question\"")
        return 2
    question = " ".join(sys.argv[1:])
    s = get_settings()

    print(f"[smoke] tenant={s.chatbot_tenant_display_name} model={s.chatbot_model_id}")
    print(f"[smoke] question: {question}")

    # ── Retrieve ────────────────────────────────────────────────
    result = await retrieve(question, settings=s)
    print(f"[smoke] retrieved {result.total_returned} chunks "
          f"(raw max {result.max_score_raw:.4f}, "
          f"normalized max {result.max_score_normalized:.2f})")

    refuse, reason = should_refuse(result)
    if refuse:
        print(f"[smoke] REFUSED: {reason}")
        await foundry_client.close_client()
        return 0

    print("[smoke] top chunks:")
    for i, c in enumerate(result.chunks[:3], start=1):
        print(f"  {i}. {c.source_name or '?'} "
              f"(score={c.relevance_score or 0:.4f}) "
              f"{(c.content or '')[:100]}...")

    # ── Stream answer ───────────────────────────────────────────
    messages = build_answer_messages(
        user_query=question,
        chunks=result.chunks,
        history=[],
        tenant_display_name=s.chatbot_tenant_display_name,
        short_id_map=result.short_id_map,
        max_history_turns=s.chatbot_max_history_turns,
    )

    print("\n[smoke] --- ANSWER ---")
    parts: list[str] = []
    finish: str | None = None
    async for chunk in foundry_client.stream_chat(
        messages=messages,
        model_id=s.chatbot_model_id,
        temperature=s.chatbot_temperature,
        max_tokens=s.chatbot_max_tokens,
        settings=s,
    ):
        if chunk.content:
            parts.append(chunk.content)
            print(chunk.content, end="", flush=True)
        if chunk.finish_reason:
            finish = chunk.finish_reason
    print(f"\n\n[smoke] finish={finish}")

    raw = "".join(parts)
    cleaned, matched, unmatched = strip_unmatched(raw, result.short_id_map)
    print(f"[smoke] citations matched={matched} unmatched={unmatched}")

    await foundry_client.close_client()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
