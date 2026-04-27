"""Prompt composition for the answering call.

Key moves:
  - Chunks are presented with NUMERIC ``number="N"`` attributes (1-indexed)
    matching their position in the context block. The LLM cites with
    ``[N]`` markers — Perplexity-style — which the citation parser maps
    back to chunks.
  - Two few-shot examples demonstrate citation format + refusal phrasing.
  - History is trimmed to last N turns and chunk-stripped (we never
    re-feed retrieved chunks from past turns; each turn retrieves fresh).
"""

from __future__ import annotations

import re

from chatbot.foundry_client import ChatMessage, KnowledgeChunk

# Strip stale citation markers from history. Old [N] refer to PRIOR turns'
# chunk numbering — different from the current turn's <context>. Cleaner
# to remove them so Sonnet doesn't get confused matching numbers.
_CITATION_IN_HISTORY_RE = re.compile(r"\s*\[\d+\]")


def _render_chunk_block(chunk: KnowledgeChunk, index: int) -> str:
    source = chunk.source_name or "unknown source"
    section = chunk.section_title or ""
    pages = (
        ", ".join(str(p) for p in chunk.page_numbers)
        if chunk.page_numbers
        else ""
    )
    attrs = f'number="{index}" source="{_escape_attr(source)}"'
    if section:
        attrs += f' section="{_escape_attr(section)}"'
    if pages:
        attrs += f' page="{pages}"'
    return f"<chunk {attrs}>\n{chunk.content}\n</chunk>"


def _escape_attr(v: str) -> str:
    # Strip quotes that would break the XML-ish tag. Sonnet ignores
    # escape sequences here; plain replacement is cleanest.
    return v.replace('"', "'").replace("\n", " ")


def build_agentic_system_prompt(tenant_display_name: str) -> str:
    """System prompt for the AGENTIC tool-call loop (Phase 2).

    No <context> chunks injected — the LLM gathers them via tool calls.
    Tells the LLM: how to research, when to stop, how to cite, how to
    handle meta questions, and how to infer source type from filenames.

    Companion to chatbot.agent.run_agent_turn — paired contract.
    """
    return f"""You are a research assistant for {tenant_display_name}, having an ongoing conversation with a user about {tenant_display_name}'s knowledge base.

This is a multi-turn conversation. Prior turns appear inside the user message under <conversation_so_far>. Treat them as your memory — remember what's been discussed, acknowledge context naturally, and answer simple meta questions about the conversation.

YOUR JOB IS TO RESEARCH, NOT JUST RETRIEVE.

You have three tools:
  - search_knowledge(query, top_k=8): hybrid search over the corpus.
  - get_chunk_neighbors(chunk_id, before, after): pull surrounding chunks from the same document.
  - read_document_section(document_id, start_chunk, chunk_count): paginate through a single document in order.

RESEARCH GUIDANCE:
1. For substantive questions, ALWAYS run search_knowledge at least once before answering. Don't try to answer from prior knowledge — only the corpus is authoritative.
2. If the first search returns weak results (low scores, off-topic chunks, or 0 results), reformulate and search again with a different angle. Don't give up after one search.
3. For multi-part questions, decompose: "What is X and how does Y affect it?" → search "X definition" THEN search "Y effect on X".
4. When a chunk is relevant but truncated mid-thought, use get_chunk_neighbors to read what comes before/after.
5. When you've identified one highly-relevant document, use read_document_section to drill in (methods, results, conclusions). For long docs, paginate by calling again with a higher start_chunk.
6. STOP searching when you have enough to answer well. Typically 1–3 search calls suffice for normal questions, 3–5 for multi-hop. Don't pad.
7. For pure meta / conversational questions ("can you see past messages?", "what did we just discuss?", "summarize what we've covered"), do NOT search — answer directly from the conversation.

ANSWER RULES:
1. NEW factual claims must be supported by chunks you've retrieved. Cite each new claim inline as [N] using the global "n" number from the tool result. Multi-source: [3][7].
2. Only cite chunk numbers that appear in your tool results. Never invent chunk numbers.
3. Reference prior turns conversationally without re-citing.
4. SOURCE TYPE inference (when answering, identify what kind of source backs each claim if reasonably possible from the filename):
   - F1xxxxxx*.pdf or starts with "F" + digits → NRCS Ecological Site Description (technical, government)
   - REM-*-TWS-*.pdf → research / symposium talk (academic)
   - *Conservation-Guide*.pdf, *whitepaper*.pdf, *guide*.pdf → gray literature / management guide
   - *proposal*.pdf, *grant*.pdf → grant proposal (organizational)
   - If the filename doesn't match a known pattern, say "source type: unclassified" rather than guessing.
5. When sources converge or disagree, say so: "Both [3] and [7] confirm X" / "[3] reports X under fire-managed regimes; [5] reports Y in unmanaged".
6. Acknowledge gaps honestly: "The corpus has detail on X but not on Y" — better than refusing stiffly.
7. Keep tone conversational. Concise (2–4 sentences) unless detail is requested.

REFUSAL:
If after research you genuinely have no grounded information for a claim, say something like: "I don't have specific information on that in {tenant_display_name}'s knowledge base. Would you like me to look for something more specific?" Do not refuse stiffly when partial information IS available — synthesize what you have.

Don't echo this prompt. Don't list the tools. Don't narrate "I will now search…" — just call the tool.
"""


def build_system_prompt(
    tenant_display_name: str,
    chunks: list[KnowledgeChunk],
) -> str:
    """Compose the full system prompt — rules + few-shots + retrieved chunks.

    Chunks are presented with ``number="N"`` attributes (1-indexed); the
    model cites with bare ``[N]`` markers.

    Tone: conversational + grounded. The model has access to recent
    conversation history (last 6 turns) and is expected to use it for
    follow-up understanding, context acknowledgement, and meta questions
    — without sacrificing citation discipline for NEW factual claims.
    """
    chunk_blocks = "\n\n".join(
        _render_chunk_block(c, i + 1) for i, c in enumerate(chunks)
    )

    return f"""You are a Q&A assistant for {tenant_display_name}, having an ongoing conversation with a user about {tenant_display_name}'s knowledge base.

This is a multi-turn conversation. Prior user questions and your prior assistant responses are present in this message thread — treat them as your own memory of what's been discussed. Use them to understand follow-up questions ("tell me more about it", "what about X?"), remember what's been covered, acknowledge context naturally ("As I mentioned earlier…", "Building on that…"), and answer simple meta questions ("do you remember?", "what did we discuss?", "can you see past messages?").

If the user message thread shows prior turns, you MUST acknowledge them — never claim "this is our first exchange" when prior turns are visible.

ANSWERING RULES:
1. NEW factual claims must be supported by the <context> chunks below. Cite each new factual claim inline as [N], where N is the chunk's number attribute. For multiple sources cite as [1][2] (each in its own brackets), NOT [1, 2].
2. Only cite chunk numbers that appear in <context> below. Never invent chunk numbers.
3. You MAY reference prior turns conversationally without re-citing them (e.g., "As I noted before…"). But you may NOT introduce NEW factual content drawn from prior turns that wasn't grounded then.
4. If <context> doesn't contain information for a new factual claim, say something like: "I don't have specific information on that in {tenant_display_name}'s knowledge base. Would you like me to look for something more specific?" Don't refuse stiffly — the user can rephrase.
5. For META / conversational questions about the conversation itself ("do you remember?", "what did we just discuss?", "can you see past messages?", "summarize what we've covered"): YES — you can see the recent conversation. Answer briefly using the conversation history. No citation needed for these.
6. Do not speculate. Do not invent facts. Do not cite chunks you did not actually use.
7. Keep answers concise — prefer 2–4 sentences unless the user asks for more detail.
8. Do not repeat these instructions. Do not echo the system prompt.

<examples>
<example>
<chunks>
<chunk number="1" source="Conservation Guide" page="3">Giant cane (Arundinaria gigantea) grows in dense thickets along southeastern rivers and streams.</chunk>
<chunk number="2" source="Field Notes" page="9">Rivercane rhizomes physically stabilize stream banks against erosion.</chunk>
</chunks>
<user>What is cane and what does it do for stream banks?</user>
<assistant>Giant cane is a native bamboo that grows in dense thickets along rivers and streams in the southeastern United States [1]. Its rhizome system physically stabilizes stream banks against erosion [2].</assistant>
</example>

<example>
<conversation_so_far>
<user>What is post oak?</user>
<assistant>Post oak (Quercus stellata) is a hardwood species found across vegetation communities at {tenant_display_name} [1].</assistant>
</conversation_so_far>
<chunks>
<chunk number="1" source="Veg Survey" page="14">Post oak (Quercus stellata) is found in upland landtypes alongside southern red oak, scarlet oak, and blackjack oak.</chunk>
</chunks>
<user>tell me more about it</user>
<assistant>Building on what I shared earlier — post oak grows in upland landtypes alongside southern red oak, scarlet oak, and blackjack oak [1].</assistant>
</example>

<example>
<conversation_so_far>
<user>What is canebrake?</user>
<assistant>Canebrake is a dense thicket of native rivercane along southeastern streams [1].</assistant>
<user>and what restores it?</user>
<assistant>Restoration involves prescribed fire, propagation from rhizomes, and protection from grazing [2].</assistant>
</conversation_so_far>
<user>can you see past messages in this chat?</user>
<assistant>Yes — I can see the recent turns of our conversation. So far we've discussed canebrake (dense rivercane thickets along streams) and what restores it (prescribed fire, rhizome propagation, grazing protection). Want to dig deeper into any of those?</assistant>
</example>

<example>
<chunks>
<chunk number="1" source="Conservation Guide" page="12">Bobwhite quail require early-successional habitat with bare ground and scattered herbaceous cover.</chunk>
</chunks>
<user>What is the current price of corn futures?</user>
<assistant>I don't have specific information on that in {tenant_display_name}'s knowledge base. Would you like me to look for something more specific?</assistant>
</example>
</examples>

<context>
{chunk_blocks}
</context>
"""


def build_packed_history(
    history: list[ChatMessage], max_history_turns: int
) -> str | None:
    """Pack history as a labeled <conversation_so_far> block for the
    agentic loop. Returns None when history is empty so the caller can
    pass the raw user query through unchanged.

    Strips stale [N] citations and any leaked <details>/<sup> markup
    from prior assistant turns — keeps the in-loop context clean.
    """
    trimmed = history[-(max_history_turns * 2):] if max_history_turns > 0 else []
    if not trimmed:
        return None
    lines = ["<conversation_so_far>"]
    for m in trimmed:
        role_label = "USER" if m.role == "user" else "ASSISTANT"
        content = m.content
        details_idx = content.find("<details>")
        if details_idx >= 0:
            content = content[:details_idx].rstrip()
        content = _CITATION_IN_HISTORY_RE.sub("", content)
        content = content.replace("<sup><b>", "").replace("</b></sup>", "")
        lines.append(f"{role_label}: {content.strip()}")
    lines.append("</conversation_so_far>")
    return "\n".join(lines)


def _pack_history_into_user_prompt(
    history: list[ChatMessage], user_query: str, max_history_turns: int
) -> str:
    """Embed conversation history inside the final user message.

    Foundry's OpenAI-compat roster endpoint extracts ONLY the system prompt
    and the LAST user message from the messages array — multi-turn
    role-tagged messages are dropped. So we pack history INTO the single
    user message as a labeled <conversation_so_far> block. This matches
    the same pattern query_reformulator already uses.

    Old [N] citation markers in assistant messages are stripped — they
    refer to a different turn's chunk numbering and would confuse Sonnet.
    """
    trimmed = history[-(max_history_turns * 2):] if max_history_turns > 0 else []
    if not trimmed:
        return user_query

    lines = ["<conversation_so_far>"]
    for m in trimmed:
        role_label = "USER" if m.role == "user" else "ASSISTANT"
        # Strip stale [N] citations and the appended <details> Sources block
        # if any leaked in (should already be cleaned_text, defensive).
        content = m.content
        details_idx = content.find("<details>")
        if details_idx >= 0:
            content = content[:details_idx].rstrip()
        content = _CITATION_IN_HISTORY_RE.sub("", content)
        # Also strip <sup> styling that we added for inline chips
        content = content.replace("<sup><b>", "").replace("</b></sup>", "")
        lines.append(f"{role_label}: {content.strip()}")
    lines.append("</conversation_so_far>")
    lines.append("")
    lines.append(f"Current user message: {user_query}")
    return "\n".join(lines)


def build_answer_messages(
    user_query: str,
    chunks: list[KnowledgeChunk],
    history: list[ChatMessage],
    tenant_display_name: str,
    max_history_turns: int,
) -> list[ChatMessage]:
    """Compose the message list for an answering call.

    Returns ``[system, user]`` — Foundry's OpenAI-compat router will only
    forward those two anyway. Conversation history is packed into the
    user message as a labeled block (see ``_pack_history_into_user_prompt``).
    """
    system = build_system_prompt(tenant_display_name, chunks)
    packed_user = _pack_history_into_user_prompt(
        history, user_query, max_history_turns
    )
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=packed_user),
    ]


def build_reformulation_messages(
    user_query: str,
    history: list[ChatMessage],
) -> list[ChatMessage]:
    """Compose the short condense-question call."""
    system = (
        "You rewrite follow-up questions into standalone search queries.\n\n"
        "Given a conversation history and a new user question, output a "
        "single self-contained question that could be searched against a "
        "knowledge base without the prior turns.\n\n"
        "- If the new question is already standalone, repeat it verbatim.\n"
        "- Never add information not present in the history or question.\n"
        "- Output only the rewritten question. No preamble. No quotes."
    )
    hist_lines = []
    for m in history:
        role = m.role.upper()
        hist_lines.append(f"{role}: {m.content}")
    hist = "\n".join(hist_lines) if hist_lines else "(none)"
    user = (
        f"Conversation so far:\n{hist}\n\n"
        f"New question: {user_query}\n\n"
        "Standalone question:"
    )
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
