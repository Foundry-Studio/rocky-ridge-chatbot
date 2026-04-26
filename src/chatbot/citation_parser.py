"""Numeric citation parsing + Sources section rendering (Perplexity-style).

The model emits ``[N]`` markers where N is the 1-based index of a chunk in
the retrieved-context block. This module:

  * extracts the indices the model actually cited (for logging),
  * strips any ``[N]`` whose N is outside the valid range (G6 integrity —
    silently invented citations would be the worst RAG failure mode), and
  * renders a Markdown "Sources" section that lists ONLY the chunks the
    model actually cited.

We deliberately do NOT use Chainlit ``cl.Text(display="side")`` elements
here — those have a bug cluster across multi-turn updates in 2.x
(see plan §UX-investigation). Plain Markdown in the message body is more
reliable, has no cross-turn state, and matches the citation pattern users
already know from Perplexity / You.com / Claude.ai.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chatbot.foundry_client import KnowledgeChunk


# Single-bracket form ``[1]``, ``[2]``, etc. Multi-cite groupings should be
# emitted as ``[1][2]`` per the system prompt; ``[1, 2]`` would only match
# the first integer (visible defect, not a silent bug).
CITATION_RE = re.compile(r"\[(\d+)\]")

SNIPPET_MAX_CHARS = 300

UNMATCHED_WARNING = (
    "*[Note: some citation markers in the answer below referenced "
    "non-existent source numbers and were removed. Treat any un-cited "
    "claim as unsupported by the retrieved documents.]*\n\n"
)


def extract_cited_indices(text: str) -> list[int]:
    """Return unique 1-based citation numbers in first-appearance order."""
    seen: set[int] = set()
    out: list[int] = []
    for match in CITATION_RE.finditer(text):
        try:
            n = int(match.group(1))
        except ValueError:
            continue
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def strip_unmatched(
    text: str, max_index: int
) -> tuple[str, list[int], list[int]]:
    """Remove ``[N]`` markers whose N is outside ``1..max_index``.

    Returns ``(cleaned_text, matched_indices, unmatched_indices)``. Matched
    and unmatched indices are deduplicated, in first-appearance order.

    Whitespace and punctuation around stripped markers are normalized so
    the cleaned text doesn't end up with double-spaces or " ." artifacts.
    """
    matched: list[int] = []
    unmatched: list[int] = []
    seen_matched: set[int] = set()
    seen_unmatched: set[int] = set()

    def _sub(m: re.Match[str]) -> str:
        try:
            n = int(m.group(1))
        except ValueError:
            return ""
        if 1 <= n <= max_index:
            if n not in seen_matched:
                seen_matched.add(n)
                matched.append(n)
            return m.group(0)  # keep the marker intact
        if n not in seen_unmatched:
            seen_unmatched.add(n)
            unmatched.append(n)
        return ""  # strip the fake marker

    cleaned = CITATION_RE.sub(_sub, text)
    # Collapse double-spaces and tighten orphaned punctuation.
    cleaned = re.sub(r"  +", " ", cleaned)
    cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
    return cleaned, matched, unmatched


def _truncate_snippet(content: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    if not content:
        return ""
    # Collapse all whitespace runs (including newlines) into single spaces
    # so the blockquote rendering stays compact.
    collapsed = re.sub(r"\s+", " ", content).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    # Cut at a word boundary close to the limit when possible.
    cut = collapsed[:max_chars].rsplit(" ", 1)[0] or collapsed[:max_chars]
    return cut + "…"


def render_sources_section(
    chunks: list[KnowledgeChunk], cited_indices: list[int]
) -> str:
    """Render a Markdown "Sources" section listing only cited chunks.

    Numbers are 1-based and refer to chunk position in ``chunks`` (the
    same order the chunks appear in the LLM's context block, so [1] in
    the answer maps to ``chunks[0]``).

    Returns an empty string when no chunks are cited.
    """
    if not cited_indices:
        return ""

    lines: list[str] = ["", "---", "", "**Sources**", ""]
    for n in cited_indices:
        idx = n - 1
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        title = chunk.source_name or "Unknown source"
        section = (
            f" — *{chunk.section_title}*"
            if getattr(chunk, "section_title", None)
            else ""
        )
        pages = (
            f" (p. {', '.join(str(p) for p in chunk.page_numbers)})"
            if getattr(chunk, "page_numbers", None)
            else ""
        )
        snippet = _truncate_snippet(chunk.content or "")
        lines.append(f"**[{n}]** *{title}*{section}{pages}")
        if snippet:
            lines.append(f"> {snippet}")
        lines.append("")
    return "\n".join(lines)
