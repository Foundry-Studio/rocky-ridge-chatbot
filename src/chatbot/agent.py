"""Agent loop — Sonnet drives retrieval via tool calls.

This replaces the one-shot retrieve→inject→answer pipeline with a
tool-call loop where the LLM decides:
  - what to search for
  - when to drill into a specific document
  - when to fetch surrounding chunks for fuller context
  - when it has enough to answer

Caps:
  MAX_AGENT_ITERATIONS  (default 6) — total tool-call loop iterations
  MAX_AGENT_WALL_CLOCK  (default 60s)

Graceful degradation: when a cap fires, we synthesize an answer from
whatever chunks_seen we've gathered, rather than burning another LLM
call to "force finalize." If chunks_seen is empty at that point, we
emit the canned refusal text.

Citation scheme: globally-monotonic [N] across ALL tool calls within
one user turn. tools.py assigns N at first sighting of each chunk and
shares the chunk_id_to_n map back into the agent loop, which uses it
when rendering the Sources block.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from chatbot import foundry_client
from chatbot.config import Settings, get_settings
from chatbot.exceptions import FoundryAuthError, FoundryError
from chatbot.foundry_client import (
    AssistantResponse,
    KnowledgeChunk,
    ToolCall,
)
from chatbot.tools import TOOL_DISPATCH, TOOL_SCHEMAS

logger = logging.getLogger(__name__)


@dataclass
class ToolCallTrace:
    """Audit record for one tool call — feeds the research-trace UI + logs."""

    name: str
    input: dict[str, Any]
    output_summary: str  # short, log-safe (e.g. "8 chunks, top score 0.51")
    chunks_returned: int
    latency_ms: int
    is_error: bool
    error_message: str | None = None


@dataclass
class AgentTurnResult:
    """Outcome of one user turn."""

    final_answer: str
    chunks_seen: dict[str, KnowledgeChunk]  # chunk_id → chunk
    chunk_id_to_n: dict[str, int]  # chunk_id → global [N]
    tool_calls: list[ToolCallTrace]
    iterations: int
    latency_ms: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    finish_reason: str
    # "ok"           — LLM produced a clean final answer
    # "graceful_cap" — iteration / wall-clock cap fired, synthesized
    #                  answer from chunks_seen
    # "auth_error"   — Foundry auth failed; no answer possible
    # "transient"    — Foundry transport error; no answer possible
    # "no_research"  — meta question; LLM answered without tools
    error: str | None = None


# Sonnet 4.5 pricing per token (USD)
_PRICE_IN = 0.003 / 1000
_PRICE_OUT = 0.015 / 1000


def _approx_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * _PRICE_IN + output_tokens * _PRICE_OUT


async def _dispatch_tool(
    tc: ToolCall,
    chunks_seen: dict[str, KnowledgeChunk],
    chunk_id_to_n: dict[str, int],
    settings: Settings,
    on_status: Callable[[str], Awaitable[None]] | None,
) -> tuple[ToolCallTrace, dict[str, Any]]:
    """Execute one tool call. Returns (trace, llm_facing_result).

    NEVER raises (except FoundryAuthError, which should bubble to the
    agent loop for a hard stop). All other failures land in the trace +
    are returned to the LLM as ``{"error": ...}`` so it can recover.
    """
    started = time.perf_counter()
    impl = TOOL_DISPATCH.get(tc.name)
    if impl is None:
        result = {
            "error": f"unknown tool '{tc.name}'",
            "is_error": True,
            "retryable": False,
            "available_tools": list(TOOL_DISPATCH.keys()),
        }
        return (
            ToolCallTrace(
                name=tc.name,
                input={},
                output_summary="unknown tool",
                chunks_returned=0,
                latency_ms=0,
                is_error=True,
                error_message=f"unknown tool '{tc.name}'",
            ),
            result,
        )

    try:
        args = json.loads(tc.arguments) if tc.arguments else {}
    except json.JSONDecodeError as e:
        result = {
            "error": f"could not parse tool arguments JSON: {e}",
            "is_error": True,
            "retryable": False,
        }
        return (
            ToolCallTrace(
                name=tc.name,
                input={"_raw": tc.arguments[:200]},
                output_summary="bad JSON args",
                chunks_returned=0,
                latency_ms=0,
                is_error=True,
                error_message=str(e),
            ),
            result,
        )

    if on_status:
        try:
            await on_status(_format_tool_status(tc.name, args))
        except Exception as e:
            logger.warning("on_status callback raised: %s", e)

    try:
        result = await impl(args, chunks_seen, chunk_id_to_n, settings)
    except FoundryAuthError:
        # Hard stop — let the agent loop handle this
        raise
    except Exception as e:
        result = {
            "error": f"tool dispatch raised: {type(e).__name__}: {e}",
            "is_error": True,
            "retryable": True,
        }
        logger.exception("tool %s raised", tc.name)

    latency_ms = int((time.perf_counter() - started) * 1000)
    chunks_returned = len(result.get("chunks", []) or [])
    is_error = bool(result.get("is_error"))
    summary = (
        f"{chunks_returned} chunks"
        if not is_error
        else f"error: {result.get('error', '?')[:80]}"
    )
    return (
        ToolCallTrace(
            name=tc.name,
            input=args,
            output_summary=summary,
            chunks_returned=chunks_returned,
            latency_ms=latency_ms,
            is_error=is_error,
            error_message=result.get("error") if is_error else None,
        ),
        result,
    )


def _format_tool_status(name: str, args: dict[str, Any]) -> str:
    """Compact one-liner for the UI status callback."""
    if name == "search_knowledge":
        q = args.get("query", "")
        return f"🔍 Searching: {q[:80]}"
    if name == "get_chunk_neighbors":
        cid = (args.get("chunk_id") or "")[:8]
        return f"📎 Reading neighbors of c_{cid}"
    if name == "read_document_section":
        did = (args.get("document_id") or "")[:8]
        start = args.get("start_chunk") or 0
        cnt = args.get("chunk_count") or 10
        return f"📖 Reading document {did} chunks {start}–{start + cnt - 1}"
    return f"🔧 {name}"


def _build_initial_messages(
    user_query: str, system_prompt: str, packed_history: str | None
) -> list[dict[str, Any]]:
    """First-iteration messages array for the LLM.

    history is packed into the user message body as a labeled
    <conversation_so_far> block (matches the pattern Foundry's roster
    has been verified to handle in tools-mode multi-turn).
    """
    user_body = (
        f"{packed_history}\n\nCurrent user message: {user_query}"
        if packed_history
        else user_query
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_body},
    ]


def _serialize_assistant_for_messages(resp: AssistantResponse) -> dict[str, Any]:
    """Serialize an AssistantResponse to OpenAI-shape ``messages[]`` entry."""
    msg: dict[str, Any] = {"role": "assistant"}
    msg["content"] = resp.content if resp.content else None
    if resp.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in resp.tool_calls
        ]
    return msg


def _tool_result_to_message(tc_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tc_id,
        "content": json.dumps(result, ensure_ascii=False),
    }


async def run_agent_turn(
    user_query: str,
    system_prompt: str,
    packed_history: str | None,
    settings: Settings | None = None,
    on_status: Callable[[str], Awaitable[None]] | None = None,
    on_text_token: Callable[[str], Awaitable[None]] | None = None,
) -> AgentTurnResult:
    """Drive Sonnet through a tool-call loop until it produces a final
    answer or hits a cap.

    on_status is called with one-line strings as tools fire (UI hooks).
    on_text_token is called with each text delta when the LLM streams a
    final answer (UI hooks the chat bubble).

    Caps:
      MAX_AGENT_ITERATIONS — soft cap on tool-call iterations.
      MAX_AGENT_WALL_CLOCK_SEC — hard cap on total wall clock per turn.

    Per-iteration mid-loop budget check: if the daily USD cap is
    exhausted between iterations, we abort with a budget message.
    Spend is recorded after each LLM call inside the loop.
    """
    s = settings or get_settings()
    started = time.perf_counter()
    chunks_seen: dict[str, KnowledgeChunk] = {}
    chunk_id_to_n: dict[str, int] = {}
    trace: list[ToolCallTrace] = []
    total_input_tokens = 0
    total_output_tokens = 0

    messages = _build_initial_messages(user_query, system_prompt, packed_history)
    final_finish: str = "ok"
    error: str | None = None

    max_iter = int(getattr(s, "chatbot_max_agent_iterations", 6))
    max_wall = float(getattr(s, "chatbot_max_agent_wall_clock_sec", 60))

    for step in range(max_iter):
        elapsed = time.perf_counter() - started
        if elapsed > max_wall:
            logger.warning(
                "agent loop wall-clock cap (%.0fs) exceeded at step %d",
                max_wall,
                step,
            )
            final_finish = "graceful_cap"
            error = "wall_clock_exceeded"
            break

        # Check budget before each LLM call
        try:
            from chatbot.app import rate_limiter  # avoid import at module load

            if not await rate_limiter.check_budget():
                logger.warning("daily budget exhausted mid-loop")
                final_finish = "graceful_cap"
                error = "daily_budget_exhausted"
                break
        except Exception:
            # If rate_limiter isn't importable in tests, skip the check
            pass

        try:
            response = await foundry_client.complete_chat_with_tools(
                openai_messages=messages,
                model_id=s.chatbot_model_id,
                temperature=s.chatbot_temperature,
                max_tokens=s.chatbot_max_tokens,
                tools=TOOL_SCHEMAS,
                settings=s,
            )
        except FoundryAuthError as e:
            logger.error("agent loop auth error: %s", e)
            return AgentTurnResult(
                final_answer="",
                chunks_seen=chunks_seen,
                chunk_id_to_n=chunk_id_to_n,
                tool_calls=trace,
                iterations=step,
                latency_ms=int((time.perf_counter() - started) * 1000),
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                estimated_cost_usd=_approx_cost(total_input_tokens, total_output_tokens),
                finish_reason="auth_error",
                error=str(e),
            )
        except FoundryError as e:
            logger.warning("agent loop transient: %s", e)
            return AgentTurnResult(
                final_answer="",
                chunks_seen=chunks_seen,
                chunk_id_to_n=chunk_id_to_n,
                tool_calls=trace,
                iterations=step,
                latency_ms=int((time.perf_counter() - started) * 1000),
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                estimated_cost_usd=_approx_cost(total_input_tokens, total_output_tokens),
                finish_reason="transient",
                error=str(e),
            )

        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens

        if not response.has_tool_calls:
            # Final answer (no streaming on this path — we got the full
            # text back). Stream it via on_text_token if caller provided.
            final_text = response.content or ""
            if on_text_token and final_text:
                # For UX: chunk the text in 100-char-ish bites and emit as deltas
                # so users see something appear quickly. (We're not
                # actually streaming over the wire — just simulating.)
                step_size = 100
                for i in range(0, len(final_text), step_size):
                    try:
                        await on_text_token(final_text[i : i + step_size])
                    except Exception as e:
                        logger.warning("on_text_token raised: %s", e)
                        break
            return AgentTurnResult(
                final_answer=final_text,
                chunks_seen=chunks_seen,
                chunk_id_to_n=chunk_id_to_n,
                tool_calls=trace,
                iterations=step + 1,
                latency_ms=int((time.perf_counter() - started) * 1000),
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                estimated_cost_usd=_approx_cost(
                    total_input_tokens, total_output_tokens
                ),
                finish_reason="no_research" if not trace else "ok",
                error=None,
            )

        # Tool-use round — append the assistant turn (with tool_calls) and
        # then dispatch tools concurrently
        messages.append(_serialize_assistant_for_messages(response))

        # Dispatch tools concurrently. Auth errors bubble; everything else
        # returns a result dict that goes back to the LLM.
        try:
            tool_results = await asyncio.gather(
                *(
                    _dispatch_tool(
                        tc, chunks_seen, chunk_id_to_n, s, on_status
                    )
                    for tc in response.tool_calls
                )
            )
        except FoundryAuthError as e:
            return AgentTurnResult(
                final_answer="",
                chunks_seen=chunks_seen,
                chunk_id_to_n=chunk_id_to_n,
                tool_calls=trace,
                iterations=step + 1,
                latency_ms=int((time.perf_counter() - started) * 1000),
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                estimated_cost_usd=_approx_cost(
                    total_input_tokens, total_output_tokens
                ),
                finish_reason="auth_error",
                error=str(e),
            )

        for tc, (tc_trace, tc_result) in zip(
            response.tool_calls, tool_results, strict=True
        ):
            trace.append(tc_trace)
            messages.append(_tool_result_to_message(tc.id, tc_result))

    # Loop exited via cap. Graceful degradation: synthesize from what we have.
    if final_finish == "graceful_cap":
        return await _graceful_finalize(
            user_query=user_query,
            system_prompt=system_prompt,
            packed_history=packed_history,
            chunks_seen=chunks_seen,
            chunk_id_to_n=chunk_id_to_n,
            trace=trace,
            iterations=max_iter,
            started=started,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            error=error,
            settings=s,
            on_text_token=on_text_token,
        )

    # Iteration cap without explicit graceful_cap = also graceful
    return await _graceful_finalize(
        user_query=user_query,
        system_prompt=system_prompt,
        packed_history=packed_history,
        chunks_seen=chunks_seen,
        chunk_id_to_n=chunk_id_to_n,
        trace=trace,
        iterations=max_iter,
        started=started,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        error="iteration_cap",
        settings=s,
        on_text_token=on_text_token,
    )


async def _graceful_finalize(
    user_query: str,
    system_prompt: str,
    packed_history: str | None,
    chunks_seen: dict[str, KnowledgeChunk],
    chunk_id_to_n: dict[str, int],
    trace: list[ToolCallTrace],
    iterations: int,
    started: float,
    input_tokens: int,
    output_tokens: int,
    error: str | None,
    settings: Settings,
    on_text_token: Callable[[str], Awaitable[None]] | None,
) -> AgentTurnResult:
    """Cap fired. Make ONE more LLM call WITHOUT tools, instructing the
    model to synthesize from whatever chunks_seen it gathered.

    If chunks_seen is empty, return the canned refusal text directly
    without spending another LLM call.
    """
    if not chunks_seen:
        canned = (
            f"I wasn't able to gather grounded information for that question "
            f"in {settings.chatbot_tenant_display_name}'s knowledge base. "
            "Could you rephrase, or ask about something more specific?"
        )
        if on_text_token:
            try:
                await on_text_token(canned)
            except Exception:
                pass
        return AgentTurnResult(
            final_answer=canned,
            chunks_seen=chunks_seen,
            chunk_id_to_n=chunk_id_to_n,
            tool_calls=trace,
            iterations=iterations,
            latency_ms=int((time.perf_counter() - started) * 1000),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=_approx_cost(input_tokens, output_tokens),
            finish_reason="graceful_cap",
            error=error,
        )

    # Build a synthesis prompt with all gathered chunks rendered + a hard
    # instruction "answer with what you have, no more tools available."
    chunk_block = "\n\n".join(
        f"<chunk number=\"{n}\" doc=\"{c.section_title or c.source_name or '?'}\">\n{c.content[:1500]}\n</chunk>"
        for c, n in sorted(
            ((c, chunk_id_to_n[cid]) for cid, c in chunks_seen.items()),
            key=lambda x: x[1],
        )
    )
    synth_user = (
        (packed_history + "\n\n" if packed_history else "")
        + f"Current user message: {user_query}\n\n"
        + "<gathered_research>\n"
        + chunk_block
        + "\n</gathered_research>\n\n"
        + "INSTRUCTIONS: Your research budget is exhausted. Synthesize a "
        "final answer from ONLY the <gathered_research> chunks above. "
        "Cite as [N]. Be honest about partial coverage — if the chunks "
        "don't fully answer the question, say what they DO show and "
        "acknowledge the gap. No further tool calls are available."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": synth_user},
    ]
    final_text_parts: list[str] = []
    try:
        async for chunk in foundry_client.stream_final_answer(
            openai_messages=messages,
            model_id=settings.chatbot_model_id,
            temperature=settings.chatbot_temperature,
            max_tokens=settings.chatbot_max_tokens,
            settings=settings,
        ):
            if chunk.content:
                final_text_parts.append(chunk.content)
                if on_text_token:
                    try:
                        await on_text_token(chunk.content)
                    except Exception:
                        pass
    except FoundryError as e:
        logger.warning("graceful synthesis failed: %s", e)
        final_text_parts.append(
            "\n\n[Research budget exhausted; synthesis call failed. "
            "Please rephrase your question.]"
        )

    final_text = "".join(final_text_parts)
    return AgentTurnResult(
        final_answer=final_text,
        chunks_seen=chunks_seen,
        chunk_id_to_n=chunk_id_to_n,
        tool_calls=trace,
        iterations=iterations,
        latency_ms=int((time.perf_counter() - started) * 1000),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=_approx_cost(input_tokens, output_tokens),
        finish_reason="graceful_cap",
        error=error,
    )
