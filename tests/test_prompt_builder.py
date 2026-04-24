"""Prompt builder tests."""

from __future__ import annotations

from chatbot.foundry_client import ChatMessage, KnowledgeChunk
from chatbot.prompt_builder import (
    build_answer_messages,
    build_reformulation_messages,
    build_system_prompt,
)


def _mk_chunk(cid, content, source="Guide", section=None, pages=None):
    return KnowledgeChunk(
        chunk_id=cid,
        content=content,
        source_name=source,
        section_title=section,
        page_numbers=pages,
        relevance_score=0.01,
    )


def test_system_prompt_includes_tenant_name_and_rules():
    chunks = [_mk_chunk("uuid-1", "fact")]
    prompt = build_system_prompt(
        tenant_display_name="Rocky Ridge Land Management",
        chunks=chunks,
        short_id_map={"c_uuid1": "uuid-1"},
    )
    assert "Rocky Ridge Land Management" in prompt
    assert "Answer ONLY from the" in prompt
    assert "[ref:<chunk_id>]" in prompt
    assert '<chunk id="c_uuid1"' in prompt
    assert "fact" in prompt


def test_system_prompt_chunk_rendering_escapes_quotes():
    chunks = [_mk_chunk("u1", "body", source='Name with "quotes"')]
    prompt = build_system_prompt("Tenant", chunks, {"c_u1": "u1"})
    # Double quotes in source name were replaced with single quotes to
    # not break the XML-ish tag.
    assert 'source="Name with' in prompt
    assert 'quotes\'"' in prompt


def test_answer_messages_trims_history_to_window():
    chunks = [_mk_chunk("u1", "body")]
    history = [
        ChatMessage(role="user", content=f"q{i}") if i % 2 == 0
        else ChatMessage(role="assistant", content=f"a{i}")
        for i in range(20)
    ]
    messages = build_answer_messages(
        user_query="new question",
        chunks=chunks,
        history=history,
        tenant_display_name="T",
        short_id_map={"c_u1": "u1"},
        max_history_turns=3,
    )
    # 1 system + 6 history (3 turns) + 1 user = 8
    assert len(messages) == 8
    assert messages[0].role == "system"
    assert messages[-1].role == "user"
    assert messages[-1].content == "new question"


def test_answer_messages_empty_history():
    chunks = [_mk_chunk("u1", "body")]
    messages = build_answer_messages(
        user_query="q",
        chunks=chunks,
        history=[],
        tenant_display_name="T",
        short_id_map={"c_u1": "u1"},
        max_history_turns=6,
    )
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"


def test_reformulation_messages_shape():
    history = [ChatMessage(role="user", content="prior")]
    msgs = build_reformulation_messages("what about it?", history)
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert "standalone" in msgs[0].content.lower()
    assert "prior" in msgs[1].content
    assert "what about it?" in msgs[1].content


def test_system_prompt_few_shot_examples_present():
    prompt = build_system_prompt("T", [_mk_chunk("u1", "body")], {"c_u1": "u1"})
    assert "<example>" in prompt
    assert "c_a1b2c3d4" in prompt
    assert "<assistant>" in prompt
