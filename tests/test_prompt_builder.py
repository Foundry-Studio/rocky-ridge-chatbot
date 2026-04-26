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
    # Conversational + grounded — model should be told it can see history
    assert "ongoing conversation" in prompt
    assert "multi-turn conversation" in prompt
    assert "Prior user questions and your prior assistant responses" in prompt
    # Citation rule still present, just for NEW claims
    assert "NEW factual claims must be supported" in prompt
    # Numeric citation format intact
    assert "[N]" in prompt
    assert 'number="1"' in prompt
    assert "fact" in prompt
    # Meta-question handling
    assert "META" in prompt or "meta" in prompt
    assert "do you remember" in prompt.lower()


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


def test_answer_messages_packs_history_into_user_prompt():
    """Foundry's OpenAI-compat router only forwards system + last user
    message. We pack history into the user message body as a labeled
    <conversation_so_far> block."""
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
    # Always 2 messages — system + packed user
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    body = messages[1].content
    # History is packed as a labeled block
    assert "<conversation_so_far>" in body
    assert "</conversation_so_far>" in body
    # Trimmed to last 3 turns (= 6 messages: q14 onward)
    assert "q0" not in body
    assert "q14" in body
    assert "q18" in body
    # Current question follows the block
    assert "Current user message: new question" in body


def test_answer_messages_no_history_just_passes_through():
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
    # Empty history → user message is just the raw query (no labeled block)
    assert messages[1].content == "q"


def test_answer_messages_strips_stale_citation_markers_from_history():
    """[N] markers in history reference prior turns' chunks — strip them."""
    chunks = [_mk_chunk("u1", "body")]
    history = [
        ChatMessage(role="user", content="What is post oak?"),
        ChatMessage(
            role="assistant",
            content="Post oak grows in upland landtypes [3][5]. It is also common on south aspects [7].",
        ),
    ]
    messages = build_answer_messages(
        user_query="tell me more",
        chunks=chunks,
        history=history,
        tenant_display_name="T",
        max_history_turns=6,
    )
    body = messages[1].content
    assert "Post oak grows in upland landtypes" in body
    assert "[3]" not in body
    assert "[5]" not in body
    assert "[7]" not in body


def test_answer_messages_strips_appended_details_block_from_history():
    """If a prior assistant message accidentally got the <details> Sources
    block stored, strip it before packing — prevents giant nested HTML
    in the user prompt."""
    chunks = [_mk_chunk("u1", "body")]
    history = [
        ChatMessage(role="user", content="q"),
        ChatMessage(
            role="assistant",
            content=(
                "Real answer text.\n\n<details>\n<summary>📚 Sources (3)</summary>\n"
                "**[1]** **doc.pdf**\n> snippet\n</details>"
            ),
        ),
    ]
    messages = build_answer_messages(
        user_query="follow up",
        chunks=chunks,
        history=history,
        tenant_display_name="T",
        max_history_turns=6,
    )
    body = messages[1].content
    assert "Real answer text." in body
    assert "<details>" not in body
    assert "📚 Sources" not in body
    assert "snippet" not in body


def test_answer_messages_strips_sup_chip_styling_from_history():
    chunks = [_mk_chunk("u1", "body")]
    history = [
        ChatMessage(role="user", content="q"),
        ChatMessage(
            role="assistant",
            content="Post oak<sup><b>[1]</b></sup> grows in uplands<sup><b>[2]</b></sup>.",
        ),
    ]
    messages = build_answer_messages(
        user_query="more",
        chunks=chunks,
        history=history,
        tenant_display_name="T",
        max_history_turns=6,
    )
    body = messages[1].content
    assert "<sup>" not in body
    assert "</b></sup>" not in body
    assert "[1]" not in body
    assert "[2]" not in body
    assert "Post oak grows in uplands." in body


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
