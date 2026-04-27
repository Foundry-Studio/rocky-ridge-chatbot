"""Chainlit entry point — VKC v0 chatbot.

Single-worker, Socket.IO-backed chat UI that streams Sonnet answers
grounded in Foundry's retrieval. See README.md for architecture.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import asdict

import chainlit as cl

from chatbot import foundry_client, healthcheck, injection_filter, session
from chatbot.agent import AgentTurnResult, run_agent_turn
from chatbot.citation_parser import (
    UNMATCHED_WARNING,
    render_sources_section_global,
    strip_unmatched,
    stylize_inline_citations,
)
from chatbot.config import get_settings
from chatbot.conversation_log import ConversationLog, LogEntry, utcnow_iso
from chatbot.prompt_builder import build_agentic_system_prompt, build_packed_history
from chatbot.rate_limiter import RateLimiter
from chatbot.refusal_gate import contains_model_refusal
from chatbot.research_trace import render_research_trace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("chatbot.app")

settings = get_settings()
conv_log = ConversationLog(settings)
rate_limiter = RateLimiter(settings)

REFUSAL_TEXT_RETRIEVAL = (
    "I can't reach my knowledge base right now. Please try again in a moment."
)
REFUSAL_TEXT_NO_MATCH = (
    f"I don't have enough information in {settings.chatbot_tenant_display_name}'s "
    "knowledge base to answer that confidently. Try rephrasing, or ask about "
    "something I might have sources for."
)
REJECT_TEXT_TOO_LONG = (
    f"Please keep questions under {settings.message_len_max} characters."
)
REJECT_TEXT_INJECTION = (
    f"Please ask a question about {settings.chatbot_tenant_display_name}'s "
    "knowledge base."
)
REJECT_TEXT_RATE_LIMIT = (
    "You're sending messages faster than my rate limit allows. "
    "Please wait a minute and try again."
)
REJECT_TEXT_BUDGET = (
    "This demo has hit its daily budget. Please try again tomorrow."
)


@cl.on_chat_start
async def on_start() -> None:
    session.init_session()
    await rate_limiter.load_persisted_spend()
    # Kick off a Foundry probe in the background — surfaces misconfig early.
    _, _ = await healthcheck.probe(settings)
    welcome = (
        f"Welcome — I'm **{settings.chatbot_tenant_display_name}**'s research assistant.\n\n"
        "I answer questions grounded in **57 academic and research documents** "
        "in our library — peer-reviewed papers, NRCS publications, and field "
        "literature on canebrake restoration, wildlife habitat, prescribed "
        "fire, and southeastern land management — drawing on research curated "
        "and approved by **Patience Knight and the team at Alabama A&M "
        "University**.\n\n"
        "Ask me anything about Rocky Ridge's work or the science behind it. "
        "I'll cite every claim — click a **[1]** in my answer to see the "
        "source passage, or expand **🔬 Research** under each reply to see "
        "exactly what I read to get there.\n\n"
        "If I can't find grounded information, I'll say so rather than guess.\n\n"
        "_Conversations are logged for demo review and deleted after 30 days._"
    )
    await cl.Message(
        content=welcome, author=settings.chatbot_tenant_display_name
    ).send()


@cl.on_stop
async def on_stop() -> None:
    await foundry_client.close_client()


@cl.on_message
async def on_message(msg_in: cl.Message) -> None:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    sid = session.get_session_id()
    user_msg = (msg_in.content or "").strip()

    # ── Input validation (no LLM, no retrieval) ─────────────────
    if not user_msg:
        return
    if len(user_msg) > settings.message_len_max:
        await cl.Message(content=REJECT_TEXT_TOO_LONG).send()
        return
    if injection_filter.is_injection(user_msg):
        logger.warning("injection attempt: %r", user_msg[:200])
        await cl.Message(content=REJECT_TEXT_INJECTION).send()
        await _log_reject(
            start,
            sid,
            request_id,
            user_msg,
            "injection_pattern",
            error=None,
        )
        return

    # ── IP rate limit (Chainlit exposes client IP on request.state) ──
    ip = _client_ip_hash()
    if not await rate_limiter.check_ip(ip):
        await cl.Message(content=REJECT_TEXT_RATE_LIMIT).send()
        await _log_reject(
            start, sid, request_id, user_msg, "rate_limit", error=None
        )
        return

    # ── Daily budget circuit breaker ─────────────────────────────
    if not await rate_limiter.check_budget():
        await cl.Message(content=REJECT_TEXT_BUDGET).send()
        await _log_reject(
            start, sid, request_id, user_msg, "daily_budget_hit", error=None
        )
        return

    # ── Serialize concurrent turns on the same socket ───────────
    lock = session.get_turn_lock()
    async with lock:
        await _handle_message(user_msg, request_id, sid, start)


async def _handle_message(
    user_msg: str, request_id: str, sid: str, start: float
) -> None:
    """Agentic-RAG message handler (Phase 2).

    Runs the agent loop (Sonnet drives retrieval via tool calls), streams
    the final answer to a Chainlit message, then renders citations +
    Sources block + research-trace block.

    Status updates fire to a parent Chainlit Step ('🔬 Research') while
    tool calls execute. The final answer message is created OUTSIDE
    that Step (per Chainlit best practice — see issue #1372 / #2365: a
    Message inside a Step renders inside the disclosure). Step name is
    a noun so Chainlit's auto-generated past-tense label reads cleanly
    after completion ("Used 🔬 Research").
    """
    history = session.get_history()
    turn_n = len(history) // 2 + 1

    # Open the parent "Research" step BEFORE the answer message —
    # ordering matters for Chainlit UI (#2365 workaround).
    research_step: cl.Step | None = None
    try:
        research_step = cl.Step(
            name="🔬 Research", type="run", default_open=False
        )
        await research_step.send()
    except Exception as e:
        logger.warning("could not open research step: %s", e)
        research_step = None

    # Status callback the agent loop uses to emit per-tool-call status.
    async def on_status(message: str) -> None:
        if research_step is None:
            return
        try:
            child = cl.Step(
                name=message,
                type="tool",
                parent_id=research_step.id,
                default_open=False,
            )
            await child.send()
            child.end = None  # let elapsed auto-set on update
            await child.update()
        except Exception as e:
            logger.warning("on_status emit failed: %s", e)

    # Final-answer Chainlit message — created AFTER the research step,
    # OUTSIDE its lexical scope (key Chainlit pattern).
    out = cl.Message(content="")
    await out.send()

    # Stream callback: each agentic text delta goes straight to the bubble.
    async def on_text_token(text: str) -> None:
        try:
            await out.stream_token(text)
        except Exception as e:
            logger.warning("stream_token failed: %s", e)

    # Build agentic system prompt + packed history block
    system_prompt = build_agentic_system_prompt(settings.chatbot_tenant_display_name)
    packed_history = build_packed_history(
        history, settings.chatbot_max_history_turns
    )

    # Run the agent loop
    result = await run_agent_turn(
        user_query=user_msg,
        system_prompt=system_prompt,
        packed_history=packed_history,
        settings=settings,
        on_status=on_status,
        on_text_token=on_text_token,
    )

    # Close the research step
    if research_step is not None:
        try:
            await research_step.update()
        except Exception:
            pass

    # ── Handle hard-stop conditions ─────────────────────────────
    if result.finish_reason == "auth_error":
        out.content = REFUSAL_TEXT_RETRIEVAL
        out.elements = []
        await out.update()
        await _log_agentic(
            start, sid, request_id, turn_n, user_msg,
            reformulated_query=user_msg,
            reformulation_skipped=True,
            refused=True,
            refusal_reason="foundry_auth_error",
            answer=REFUSAL_TEXT_RETRIEVAL,
            result=result,
            cited_chunk_ids=[],
            unmatched_ref_ids=[],
            post_llm_refused=False,
            error=result.error,
        )
        return
    if result.finish_reason == "transient":
        out.content = REFUSAL_TEXT_RETRIEVAL
        out.elements = []
        await out.update()
        await _log_agentic(
            start, sid, request_id, turn_n, user_msg,
            reformulated_query=user_msg,
            reformulation_skipped=True,
            refused=True,
            refusal_reason="foundry_transient",
            answer=REFUSAL_TEXT_RETRIEVAL,
            result=result,
            cited_chunk_ids=[],
            unmatched_ref_ids=[],
            post_llm_refused=False,
            error=result.error,
        )
        return

    raw_answer = result.final_answer

    # ── Citation pass (strip fakes from final text) ─────────────
    max_n = len(result.chunk_id_to_n)
    cleaned_text, matched_indices, unmatched_indices = strip_unmatched(
        raw_answer, max_index=max_n
    )

    # Post-LLM checks
    post_llm_refused = contains_model_refusal(cleaned_text)
    if injection_filter.leaks_system_prompt(cleaned_text):
        logger.error("system prompt leak detected in answer — redacting")
        cleaned_text = (
            "*[Answer redacted: model echoed protected content. "
            "Please rephrase your question.]*"
        )
        matched_indices = []

    # Build chunks_by_n for the Sources renderer
    n_to_chunk = {}
    for chunk_id, n in result.chunk_id_to_n.items():
        chunk = result.chunks_seen.get(chunk_id)
        if chunk is not None:
            n_to_chunk[n] = chunk

    # File-metadata enrichment (concurrent batch over unique source_file_ids)
    file_metadata_by_id: dict = {}
    unique_file_ids = list(
        {c.source_file_id for c in result.chunks_seen.values() if c.source_file_id}
    )
    if unique_file_ids and matched_indices:
        try:
            md_results = await asyncio.gather(
                *(
                    foundry_client.get_file_metadata(
                        tenant_id=settings.chatbot_tenant_id,
                        source_file_id=fid,
                        settings=settings,
                    )
                    for fid in unique_file_ids
                ),
                return_exceptions=True,
            )
            for fid, m in zip(unique_file_ids, md_results, strict=False):
                if m is not None and not isinstance(m, Exception):
                    file_metadata_by_id[fid] = m
        except Exception as e:
            logger.warning("file metadata enrichment failed: %s", e)

    # Stylize inline [N] markers, render Sources + research trace
    styled_text = stylize_inline_citations(cleaned_text)
    warning = UNMATCHED_WARNING if unmatched_indices else ""
    sources_md = render_sources_section_global(
        n_to_chunk, matched_indices, file_metadata_by_id=file_metadata_by_id
    )
    trace_md = render_research_trace(
        result.tool_calls,
        iterations=result.iterations,
        total_latency_ms=result.latency_ms,
        error=result.error,
    )
    final_visible_text = warning + styled_text + sources_md + trace_md

    out.content = final_visible_text
    out.elements = []
    await out.update()

    # ── Record spend, history, log ──────────────────────────────
    await rate_limiter.record_spend(result.estimated_cost_usd)

    cited_chunk_ids = [
        result.chunks_seen[cid].chunk_id
        for cid, n in result.chunk_id_to_n.items()
        if n in matched_indices
    ]
    unmatched_ref_ids = [str(n) for n in unmatched_indices]

    session.append_turn(
        user_msg, cleaned_text, settings.chatbot_max_history_turns
    )
    await _log_agentic(
        start, sid, request_id, turn_n, user_msg,
        reformulated_query=user_msg,  # no reformulator in agentic path
        reformulation_skipped=True,
        refused=False,
        refusal_reason=None,
        answer=cleaned_text,
        result=result,
        cited_chunk_ids=cited_chunk_ids,
        unmatched_ref_ids=unmatched_ref_ids,
        post_llm_refused=post_llm_refused,
        error=None,
    )


# ── Helpers ──────────────────────────────────────────────────────


def _client_ip_hash() -> str:
    """Extract client IP from the active Chainlit context and HMAC-hash it.
    Falls back to 'unknown' when context is absent (tests, local dev)."""
    try:
        ctx = cl.context.get_context()
        # Chainlit stores HTTP headers on the session; X-Forwarded-For is
        # Railway's forwarded client IP header.
        headers = getattr(ctx.session, "http_headers", None) or {}
        ip = (
            headers.get("x-forwarded-for")
            or headers.get("x-real-ip")
            or "unknown"
        ).split(",")[0].strip()
    except Exception:
        ip = "unknown"
    # We don't want raw IPs in logs — hash them with a per-deploy salt.
    import hashlib

    return hashlib.sha256(
        (ip + settings.foundry_internal_token[:8]).encode("utf-8")
    ).hexdigest()[:16]


# ── Log helpers ──────────────────────────────────────────────────


async def _log_reject(
    start: float,
    sid: str,
    request_id: str,
    user_msg: str,
    reason: str,
    error: str | None,
) -> None:
    await conv_log.append(
        LogEntry(
            ts=utcnow_iso(),
            session_id=sid,
            turn_n=0,
            request_id=request_id,
            user_msg=user_msg[:500],
            reformulated_query="",
            reformulation_skipped=True,
            refused=True,
            refusal_reason=reason,
            top_chunk_scores_raw=[],
            top_chunk_scores_normalized=[],
            cited_chunk_ids=[],
            unmatched_ref_ids=[],
            post_llm_refusal=False,
            answer="",
            latency_ms_total=int((time.perf_counter() - start) * 1000),
            latency_ms_retrieval=0,
            error=error,
            finish_reason=None,
        )
    )


async def _log_agentic(
    start: float,
    sid: str,
    request_id: str,
    turn_n: int,
    user_msg: str,
    *,
    reformulated_query: str,
    reformulation_skipped: bool,
    refused: bool,
    refusal_reason: str | None,
    answer: str,
    result: AgentTurnResult,
    cited_chunk_ids: list[str],
    unmatched_ref_ids: list[str],
    post_llm_refused: bool,
    error: str | None,
) -> None:
    """Persist a JSONL log entry for one agentic turn.

    Populates both legacy LogEntry fields (kept for log-format
    continuity) and the agentic fields added in Phase 2.
    """
    tool_calls_serialized = [asdict(t) for t in result.tool_calls]
    await conv_log.append(
        LogEntry(
            ts=utcnow_iso(),
            session_id=sid,
            turn_n=turn_n,
            request_id=request_id,
            user_msg=user_msg[:500],
            reformulated_query=reformulated_query[:500],
            reformulation_skipped=reformulation_skipped,
            refused=refused,
            refusal_reason=refusal_reason,
            top_chunk_scores_raw=[],  # n/a in agentic path
            top_chunk_scores_normalized=[],
            cited_chunk_ids=cited_chunk_ids,
            unmatched_ref_ids=unmatched_ref_ids,
            post_llm_refusal=post_llm_refused,
            answer=answer[:4000],
            latency_ms_total=int((time.perf_counter() - start) * 1000),
            latency_ms_retrieval=0,  # multi-call; per-tool latency lives in agent_tool_calls
            error=error,
            finish_reason=result.finish_reason,
            agent_iterations=result.iterations,
            agent_tool_calls=tool_calls_serialized,
            agent_chunks_seen_count=len(result.chunks_seen),
            agent_input_tokens=result.input_tokens,
            agent_output_tokens=result.output_tokens,
            agent_estimated_cost_usd=round(result.estimated_cost_usd, 6),
            agent_finish_reason=result.finish_reason,
        )
    )


# ── Register /healthz on Chainlit's FastAPI app at import time ──
try:
    from chainlit.server import app as _cl_app

    healthcheck.register_healthz(_cl_app)
except Exception as e:  # pragma: no cover
    logger.warning("could not register /healthz (non-fatal): %s", e)
