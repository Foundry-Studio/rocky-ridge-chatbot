"""Agent loop tests — exercise run_agent_turn against mocked Foundry.

We mock ``foundry_client.complete_chat_with_tools`` directly so the loop
runs deterministically without touching the network. The tools are
mocked in their own dispatch entries via monkeypatching TOOL_DISPATCH.
"""

from __future__ import annotations

import pytest
from chatbot import agent, foundry_client, tools
from chatbot.exceptions import FoundryAuthError, FoundryTransientError
from chatbot.foundry_client import AssistantResponse, KnowledgeChunk, ToolCall


def _ar(
    content: str = "",
    tool_calls: list[ToolCall] | None = None,
    finish: str = "stop",
    in_tok: int = 100,
    out_tok: int = 50,
) -> AssistantResponse:
    return AssistantResponse(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


def _kc(chunk_id: str, content: str = "blah") -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        content=content,
        source_file_id="doc-1",
        section_title="Section",
    )


# ── No-research / meta-question path ──────────────────────────────────


@pytest.mark.asyncio
async def test_no_research_path_returns_no_tools_finish(settings, monkeypatch):
    """LLM answers immediately without calling tools (e.g. meta question)."""

    async def fake_complete(**kw):
        return _ar(content="I'm an AI assistant for Rocky Ridge.", finish="stop")

    monkeypatch.setattr(
        foundry_client, "complete_chat_with_tools", fake_complete
    )
    result = await agent.run_agent_turn(
        user_query="what are you?",
        system_prompt="sys",
        packed_history=None,
        settings=settings,
    )
    assert result.finish_reason == "no_research"
    assert "AI assistant" in result.final_answer
    assert result.tool_calls == []
    assert result.iterations == 1


# ── Single-tool happy path ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_tool_call_then_final_answer(settings, monkeypatch):
    call_lens: list[int] = []

    async def fake_complete(**kw):
        # Capture length at call time — the agent mutates the same list
        call_lens.append(len(kw["openai_messages"]))
        if len(call_lens) == 1:
            return _ar(
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="search_knowledge",
                        arguments='{"query": "fire"}',
                    )
                ],
                finish="tool_calls",
            )
        return _ar(content="Fire is good [1].", finish="stop")

    monkeypatch.setattr(
        foundry_client, "complete_chat_with_tools", fake_complete
    )

    async def fake_search_tool(args, chunks_seen, chunk_id_to_n, s):
        chunks_seen["c1"] = _kc("c1", "fire content")
        chunk_id_to_n["c1"] = 1
        return {"chunks": [{"n": 1, "chunk_id": "c1", "content": "fire content"}]}

    monkeypatch.setitem(tools.TOOL_DISPATCH, "search_knowledge", fake_search_tool)

    result = await agent.run_agent_turn(
        user_query="tell me about fire",
        system_prompt="sys",
        packed_history=None,
        settings=settings,
    )
    assert result.finish_reason == "ok"
    assert result.final_answer == "Fire is good [1]."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_knowledge"
    assert result.tool_calls[0].chunks_returned == 1
    assert result.iterations == 2
    assert "c1" in result.chunks_seen
    assert result.chunk_id_to_n == {"c1": 1}
    # Multi-turn was preserved across iterations: iter 2 saw assistant + tool
    # messages appended on top of the iter-1 [system, user] base
    assert call_lens == [2, 4]


# ── Streaming callback fires for final answer ─────────────────────────


@pytest.mark.asyncio
async def test_on_text_token_invoked_for_final_answer(settings, monkeypatch):
    async def fake_complete(**kw):
        return _ar(content="abc" * 50, finish="stop")  # 150 chars

    monkeypatch.setattr(
        foundry_client, "complete_chat_with_tools", fake_complete
    )

    received: list[str] = []

    async def on_tok(t: str) -> None:
        received.append(t)

    result = await agent.run_agent_turn(
        user_query="x",
        system_prompt="sys",
        packed_history=None,
        settings=settings,
        on_text_token=on_tok,
    )
    assert result.finish_reason == "no_research"
    assert "".join(received) == "abc" * 50


# ── Auth error → hard stop ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auth_error_returns_auth_finish(settings, monkeypatch):
    async def fake_complete(**kw):
        raise FoundryAuthError("bad token")

    monkeypatch.setattr(
        foundry_client, "complete_chat_with_tools", fake_complete
    )
    result = await agent.run_agent_turn(
        user_query="x",
        system_prompt="sys",
        packed_history=None,
        settings=settings,
    )
    assert result.finish_reason == "auth_error"
    assert result.final_answer == ""
    assert "bad token" in (result.error or "")


# ── Transient error → hard stop ───────────────────────────────────────


@pytest.mark.asyncio
async def test_transient_error_returns_transient_finish(settings, monkeypatch):
    async def fake_complete(**kw):
        raise FoundryTransientError("502")

    monkeypatch.setattr(
        foundry_client, "complete_chat_with_tools", fake_complete
    )
    result = await agent.run_agent_turn(
        user_query="x",
        system_prompt="sys",
        packed_history=None,
        settings=settings,
    )
    assert result.finish_reason == "transient"


# ── Iteration cap → graceful finalize with chunks_seen ────────────────


@pytest.mark.asyncio
async def test_iteration_cap_triggers_graceful_finalize(
    settings, monkeypatch
):
    """LLM keeps requesting tools forever; loop hits iter cap and synthesizes."""

    async def fake_complete(**kw):
        # Always request another tool — never produce a final answer
        return _ar(
            tool_calls=[
                ToolCall(
                    id="tc",
                    name="search_knowledge",
                    arguments='{"query": "x"}',
                )
            ],
            finish="tool_calls",
        )

    async def fake_search_tool(args, chunks_seen, chunk_id_to_n, s):
        cid = f"c{len(chunks_seen) + 1}"
        chunks_seen[cid] = _kc(cid, "stuff")
        chunk_id_to_n[cid] = len(chunk_id_to_n) + 1
        return {"chunks": [{"n": chunk_id_to_n[cid], "chunk_id": cid}]}

    monkeypatch.setattr(
        foundry_client, "complete_chat_with_tools", fake_complete
    )
    monkeypatch.setitem(tools.TOOL_DISPATCH, "search_knowledge", fake_search_tool)

    # Stub stream_final_answer for the synthesis call
    async def fake_stream(**kw):
        from chatbot.foundry_client import StreamChunk

        yield StreamChunk(content="Synthesized answer.", finish_reason="stop")

    monkeypatch.setattr(foundry_client, "stream_final_answer", fake_stream)

    # Tighten the cap so the test runs quickly
    monkeypatch.setattr(settings, "chatbot_max_agent_iterations", 2)

    result = await agent.run_agent_turn(
        user_query="x",
        system_prompt="sys",
        packed_history=None,
        settings=settings,
    )
    assert result.finish_reason == "graceful_cap"
    assert result.final_answer == "Synthesized answer."
    assert len(result.chunks_seen) >= 1
    assert result.error == "iteration_cap"


# ── Iteration cap with NO chunks → canned refusal, no extra LLM call ──


@pytest.mark.asyncio
async def test_graceful_finalize_with_no_chunks_returns_canned_refusal(
    settings, monkeypatch
):
    async def fake_complete(**kw):
        # Request a tool that errors so chunks_seen stays empty
        return _ar(
            tool_calls=[
                ToolCall(
                    id="tc",
                    name="search_knowledge",
                    arguments='{"query": "x"}',
                )
            ],
            finish="tool_calls",
        )

    async def fake_search_tool(args, chunks_seen, chunk_id_to_n, s):
        return {"is_error": True, "error": "search died"}

    monkeypatch.setattr(
        foundry_client, "complete_chat_with_tools", fake_complete
    )
    monkeypatch.setitem(tools.TOOL_DISPATCH, "search_knowledge", fake_search_tool)
    monkeypatch.setattr(settings, "chatbot_max_agent_iterations", 2)

    # If stream_final_answer ever gets called, fail the test
    async def boom(**kw):
        raise AssertionError(
            "stream_final_answer should not run when chunks_seen is empty"
        )
        yield  # pragma: no cover  # pylint: disable=unreachable

    monkeypatch.setattr(foundry_client, "stream_final_answer", boom)

    result = await agent.run_agent_turn(
        user_query="x",
        system_prompt="sys",
        packed_history=None,
        settings=settings,
    )
    assert result.finish_reason == "graceful_cap"
    assert "wasn't able to gather grounded information" in result.final_answer
    assert result.chunks_seen == {}


# ── Parallel tool dispatch in one round ───────────────────────────────


@pytest.mark.asyncio
async def test_parallel_tool_dispatch(settings, monkeypatch):
    calls = []

    async def fake_complete(**kw):
        calls.append(1)
        if len(calls) == 1:
            return _ar(
                tool_calls=[
                    ToolCall(
                        id="a",
                        name="search_knowledge",
                        arguments='{"query": "fire"}',
                    ),
                    ToolCall(
                        id="b",
                        name="search_knowledge",
                        arguments='{"query": "soil"}',
                    ),
                ],
                finish="tool_calls",
            )
        return _ar(content="Done [1][2].", finish="stop")

    async def fake_search_tool(args, chunks_seen, chunk_id_to_n, s):
        cid = "c_" + args["query"][:1]
        chunks_seen[cid] = _kc(cid)
        chunk_id_to_n[cid] = len(chunk_id_to_n) + 1
        return {"chunks": [{"n": chunk_id_to_n[cid], "chunk_id": cid}]}

    monkeypatch.setattr(
        foundry_client, "complete_chat_with_tools", fake_complete
    )
    monkeypatch.setitem(tools.TOOL_DISPATCH, "search_knowledge", fake_search_tool)

    result = await agent.run_agent_turn(
        user_query="x",
        system_prompt="sys",
        packed_history=None,
        settings=settings,
    )
    assert result.finish_reason == "ok"
    assert len(result.tool_calls) == 2
    assert len(result.chunks_seen) == 2


# ── Tool dispatcher rejects unknown tool name ─────────────────────────


@pytest.mark.asyncio
async def test_unknown_tool_returns_error_to_llm(settings, monkeypatch):
    calls = []

    async def fake_complete(**kw):
        calls.append(kw["openai_messages"])
        if len(calls) == 1:
            return _ar(
                tool_calls=[
                    ToolCall(
                        id="x", name="nonexistent_tool", arguments="{}"
                    )
                ],
                finish="tool_calls",
            )
        return _ar(content="Fallback answer.", finish="stop")

    monkeypatch.setattr(
        foundry_client, "complete_chat_with_tools", fake_complete
    )
    result = await agent.run_agent_turn(
        user_query="x",
        system_prompt="sys",
        packed_history=None,
        settings=settings,
    )
    assert result.finish_reason == "ok"
    assert result.tool_calls[0].is_error is True
    assert "unknown tool" in (result.tool_calls[0].error_message or "")


# ── Cost tracking accumulates across iterations ───────────────────────


@pytest.mark.asyncio
async def test_cost_accumulates_across_iterations(settings, monkeypatch):
    async def fake_complete(**kw):
        return _ar(content="done", finish="stop", in_tok=200, out_tok=80)

    monkeypatch.setattr(
        foundry_client, "complete_chat_with_tools", fake_complete
    )
    result = await agent.run_agent_turn(
        user_query="x",
        system_prompt="sys",
        packed_history=None,
        settings=settings,
    )
    # Sonnet 4.5: $3/1M in, $15/1M out → 200*3e-6 + 80*15e-6 = 0.0006 + 0.0012
    assert result.input_tokens == 200
    assert result.output_tokens == 80
    expected = 200 * (0.003 / 1000) + 80 * (0.015 / 1000)
    assert abs(result.estimated_cost_usd - expected) < 1e-9
