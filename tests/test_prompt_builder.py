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
    )
    assert "Rocky Ridge Land Management" in prompt
    assert "Answer ONLY from the" in prompt
    # Numeric citation rule + format
    assert "[N]" in prompt
    assert 'number="1"' in prompt
    assert "fact" in prompt


def test_system_prompt_chunk_rendering_escapes_quotes():
    chunks = [_mk_chunk("u1", "body", source='Name with "quotes"')]
    prompt = build_system_prompt("Tenant", chunks)
    # Double quotes in source name were replaced with single quotes to
    # not break the XML-ish tag.
    assert "Name with 'quotes'" in prompt


def test_system_prompt_chunks_numbered_one_indexed():
    chunks = [
        _mk_chunk("u1", "first"),
        _mk_chunk("u2", "second"),
        _mk_chunk("u3", "third"),
    ]
    prompt = build_system_prompt("Tenant", chunks)
    assert 'number="1"' in prompt
    assert 'number="2"' in prompt
    assert 'number="3"' in prompt
    # Order preserved — chunk content appears in sequence
    pos1 = prompt.index("first")
    pos2 = prompt.index("second")
    pos3 = prompt.index("third")
    assert pos1 < pos2 < pos3


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
    prompt = build_system_prompt("T", [_mk_chunk("u1", "body")])
    assert "<example>" in prompt
    # Few-shots use the new [N] format
    assert "[1]" in prompt
    assert "<assistant>" in prompt


def test_system_prompt_explains_multi_cite_form():
    """Prompt must instruct [1][2] (separate brackets), not [1, 2]."""
    prompt = build_system_prompt("T", [_mk_chunk("u1", "body")])
    assert "[1][2]" in prompt
