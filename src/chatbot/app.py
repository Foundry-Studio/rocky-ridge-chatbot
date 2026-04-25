"""Chainlit entry point — VKC v0 chatbot.

Single-worker, Socket.IO-backed chat UI that streams Sonnet answers
grounded in Foundry's retrieval. See README.md for architecture.
"""

from __future__ import annotations

import logging
import time
import uuid

import chainlit as cl

from chatbot import foundry_client, healthcheck, injection_filter, session
from chatbot.citation_parser import (
    UNMATCHED_WARNING,
    strip_unmatched,
)
from chatbot.config import get_settings
from chatbot.conversation_log import ConversationLog, LogEntry, utcnow_iso
from chatbot.cost import estimate_cost_usd
from chatbot.exceptions import (
    FoundryAuthError,
    FoundryMalformedResponseError,
    FoundryTransientError,
)
from chatbot.foundry_client import KnowledgeChunk
from chatbot.prompt_builder import build_answer_messages
from chatbot.query_reformulator import (
    needs_reformulation,
    reformulate,
)
from chatbot.rate_limiter import RateLimiter
from chatbot.refusal_gate import contains_model_refusal, should_refuse
from chatbot.retriever import retrieve

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
        f"Hello — I answer questions about **{settings.chatbot_tenant_display_name}** "
        "using only its own knowledge base.\n\n"
        "Every claim I make will be cited — click the `ref:...` badges to see the "
        "source text. If I can't find grounded information, I'll tell you directly "
        "rather than guess.\n\n"
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
    history = session.get_history()
    turn_n = len(history) // 2 + 1

    out = cl.Message(content="")
    await out.send()

    # ── Reformulate (skipped if already standalone) ─────────────
    reformulation_skipped = not needs_reformulation(user_msg, len(history))
    reformulated = user_msg
    if not reformulation_skipped:
        try:
            reformulated = await reformulate(
                user_msg, history, settings=settings
            )
        except Exception as e:
            logger.warning("reformulation layer raised: %s", e)
            reformulated = user_msg

    # ── Retrieve ────────────────────────────────────────────────
    try:
        result = await retrieve(reformulated, settings=settings)
    except FoundryAuthError as e:
        logger.error("retrieval auth error: %s", e)
        await _stream_and_update(out, REFUSAL_TEXT_RETRIEVAL)
        await _log_refusal(
            start,
            sid,
            request_id,
            turn_n,
            user_msg,
            reformulated,
            reformulation_skipped,
            reason="foundry_auth_error",
            answer=REFUSAL_TEXT_RETRIEVAL,
            error=str(e),
        )
        return
    except (FoundryTransientError, FoundryMalformedResponseError) as e:
        logger.warning("retrieval error: %s", e)
        await _stream_and_update(out, REFUSAL_TEXT_RETRIEVAL)
        await _log_refusal(
            start,
            sid,
            request_id,
            turn_n,
            user_msg,
            reformulated,
            reformulation_skipped,
            reason=f"retrieval_error_{type(e).__name__}",
            answer=REFUSAL_TEXT_RETRIEVAL,
            error=str(e),
        )
        return

    # ── Pre-LLM refusal gate ────────────────────────────────────
    refuse, reason = should_refuse(result)
    if refuse:
        await _stream_and_update(out, REFUSAL_TEXT_NO_MATCH)
        session.append_turn(
            user_msg, REFUSAL_TEXT_NO_MATCH, settings.chatbot_max_history_turns
        )
        await _log_refusal(
            start,
            sid,
            request_id,
            turn_n,
            user_msg,
            reformulated,
            reformulation_skipped,
            reason=reason,
            answer=REFUSAL_TEXT_NO_MATCH,
            error=None,
            result=result,
        )
        return

    # ── Build prompt + stream answer ────────────────────────────
    messages = build_answer_messages(
        user_query=user_msg,
        chunks=result.chunks,
        history=history,
        tenant_display_name=settings.chatbot_tenant_display_name,
        short_id_map=result.short_id_map,
        max_history_turns=settings.chatbot_max_history_turns,
    )

    full_answer_parts: list[str] = []
    final_finish: str | None = None
    try:
        async for chunk in foundry_client.stream_chat(
            messages=messages,
            model_id=settings.chatbot_model_id,
            temperature=settings.chatbot_temperature,
            max_tokens=settings.chatbot_max_tokens,
            request_id=request_id,
            settings=settings,
        ):
            if chunk.content:
                full_answer_parts.append(chunk.content)
                await out.stream_token(chunk.content)
            if chunk.finish_reason:
                final_finish = chunk.finish_reason
    except FoundryAuthError as e:
        logger.error("chat auth error mid-stream: %s", e)
        interrupted = "\n\n*[Stream interrupted.]*"
        full_answer_parts.append(interrupted)
        await out.stream_token(interrupted)
        final_finish = "error"
    except (FoundryTransientError, FoundryMalformedResponseError) as e:
        logger.warning("chat error mid-stream: %s", e)
        interrupted = "\n\n*[Stream interrupted.]*"
        full_answer_parts.append(interrupted)
        await out.stream_token(interrupted)
        final_finish = "error"

    raw_answer = "".join(full_answer_parts)

    # ── Append finish-reason footer when relevant ───────────────
    if final_finish == "length":
        footer = (
            "\n\n*[Answer truncated — ask a more focused follow-up for detail.]*"
        )
        raw_answer += footer
        await out.stream_token(footer)
    elif final_finish == "error":
        # Only append if the stream ended cleanly with error but we didn't
        # already show an interruption marker above.
        if not raw_answer.endswith("*[Stream interrupted.]*"):
            footer = "\n\n*[Stream interrupted upstream.]*"
            raw_answer += footer
            await out.stream_token(footer)

    # ── Citation pass (strip fakes + build side-panel elements) ─
    cleaned_text, matched_ids, unmatched_ids = strip_unmatched(
        raw_answer, result.short_id_map
    )

    # Post-LLM checks
    post_llm_refused = contains_model_refusal(cleaned_text)
    if injection_filter.leaks_system_prompt(cleaned_text):
        logger.error(
            "system prompt leak detected in answer — redacting",
        )
        cleaned_text = (
            "*[Answer redacted: model echoed protected content. "
            "Please rephrase your question.]*"
        )
        matched_ids = []

    # Prepend warning banner if fake citations were dropped
    final_visible_text = cleaned_text
    if unmatched_ids:
        final_visible_text = UNMATCHED_WARNING + cleaned_text

    # Update message: swap content to cleaned version + attach elements
    out.content = final_visible_text
    out.elements = _build_side_elements(matched_ids, result.chunks, result.short_id_map)
    await out.update()

    # ── Record spend, history, log ──────────────────────────────
    input_chars = sum(len(m.content) for m in messages)
    output_chars = len(cleaned_text)
    cost = estimate_cost_usd(input_chars, output_chars)
    await rate_limiter.record_spend(cost)

    session.append_turn(
        user_msg, cleaned_text, settings.chatbot_max_history_turns
    )
    await _log_success(
        start,
        sid,
        request_id,
        turn_n,
        user_msg,
        reformulated,
        reformulation_skipped,
        result,
        cleaned_text,
        matched_ids,
        unmatched_ids,
        post_llm_refused,
        final_finish,
    )


# ── Helpers ──────────────────────────────────────────────────────


def _build_side_elements(
    cited_short_ids: list[str],
    chunks: list[KnowledgeChunk],
    short_id_map: dict[str, str],
) -> list[cl.Text]:
    by_full_id = {c.chunk_id: c for c in chunks}
    elements: list[cl.Text] = []
    for sid in cited_short_ids:
        full_id = short_id_map.get(sid)
        if not full_id:
            continue
        chunk = by_full_id.get(full_id)
        if chunk is None:
            continue
        pages = (
            ", ".join(str(p) for p in chunk.page_numbers)
            if chunk.page_numbers
            else "(n/a)"
        )
        body = (
            f"**Source:** {chunk.source_name or '(unknown)'}\n"
            f"**Section:** {chunk.section_title or '(n/a)'}\n"
            f"**Page(s):** {pages}\n"
            f"**Relevance:** {chunk.relevance_score or 0.0:.4f}\n\n"
            f"---\n\n"
            f"{chunk.content}"
        )
        # Name MUST equal the literal ref token in the message text for
        # Chainlit to auto-link it as a clickable badge.
        elements.append(
            cl.Text(name=f"ref:{sid}", content=body, display="side")
        )
    return elements


async def _stream_and_update(out: cl.Message, text: str) -> None:
    await out.stream_token(text)
    await out.update()


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


async def _log_refusal(
    start: float,
    sid: str,
    request_id: str,
    turn_n: int,
    user_msg: str,
    reformulated: str,
    reformulation_skipped: bool,
    reason: str,
    answer: str,
    error: str | None,
    result=None,  # RetrievalResult | None
) -> None:
    raw = sorted((c.relevance_score or 0.0 for c in result.chunks), reverse=True) if result else []
    norm = (
        sorted(
            (_normalized_for_log(c.relevance_score or 0.0) for c in result.chunks),
            reverse=True,
        )
        if result
        else []
    )
    await conv_log.append(
        LogEntry(
            ts=utcnow_iso(),
            session_id=sid,
            turn_n=turn_n,
            request_id=request_id,
            user_msg=user_msg[:500],
            reformulated_query=reformulated[:500],
            reformulation_skipped=reformulation_skipped,
            refused=True,
            refusal_reason=reason,
            top_chunk_scores_raw=raw,
            top_chunk_scores_normalized=norm,
            cited_chunk_ids=[],
            unmatched_ref_ids=[],
            post_llm_refusal=False,
            answer=answer[:2000],
            latency_ms_total=int((time.perf_counter() - start) * 1000),
            latency_ms_retrieval=result.latency_ms if result else 0,
            error=error,
            finish_reason=None,
        )
    )


async def _log_success(
    start: float,
    sid: str,
    request_id: str,
    turn_n: int,
    user_msg: str,
    reformulated: str,
    reformulation_skipped: bool,
    result,
    answer: str,
    cited_ids: list[str],
    unmatched_ids: list[str],
    post_llm_refused: bool,
    finish_reason: str | None,
) -> None:
    raw = sorted((c.relevance_score or 0.0 for c in result.chunks), reverse=True)
    norm = sorted(
        (_normalized_for_log(c.relevance_score or 0.0) for c in result.chunks),
        reverse=True,
    )
    await conv_log.append(
        LogEntry(
            ts=utcnow_iso(),
            session_id=sid,
            turn_n=turn_n,
            request_id=request_id,
            user_msg=user_msg[:500],
            reformulated_query=reformulated[:500],
            reformulation_skipped=reformulation_skipped,
            refused=False,
            refusal_reason=None,
            top_chunk_scores_raw=raw,
            top_chunk_scores_normalized=norm,
            cited_chunk_ids=cited_ids,
            unmatched_ref_ids=unmatched_ids,
            post_llm_refusal=post_llm_refused,
            answer=answer[:4000],
            latency_ms_total=int((time.perf_counter() - start) * 1000),
            latency_ms_retrieval=result.latency_ms,
            error=None,
            finish_reason=finish_reason,
        )
    )


def _normalized_for_log(raw: float) -> float:
    from chatbot.retriever import normalize_rrf_score

    return round(normalize_rrf_score(raw), 4)


# ── Register /healthz on Chainlit's FastAPI app at import time ──
try:
    from chainlit.server import app as _cl_app

    healthcheck.register_healthz(_cl_app)
except Exception as e:  # pragma: no cover
    logger.warning("could not register /healthz (non-fatal): %s", e)
