"""Numeric citation parsing + Sources block rendering (Perplexity-style).

The model emits ``[N]`` markers where N is the 1-based index of a chunk
in the retrieved-context block. This module:

  * extracts the indices the model actually cited (for logging),
  * strips any ``[N]`` whose N is outside the valid range (G6 integrity —
    silently invented citations would be the worst RAG failure mode),
  * styles inline citations as small chip-like superscript using minimal
    HTML (requires ``unsafe_allow_html=true`` in Chainlit), and
  * renders a collapsible Markdown/HTML "Sources" block with rich
    credibility metadata (real document filename, library, section,
    pages, authority level, match score, retrieval method, ingestion
    date, audit reference ID, snippet excerpt).

Why HTML and not pure Markdown:
  Pure Markdown can't deliver "click to collapse" without ``<details>``,
  and can't deliver chip/superscript styling for citation badges. Both
  are baseline for a Perplexity-style citation UX. We HTML-escape any
  user/chunk-derived content embedded inside the block to keep the XSS
  surface narrow even with HTML enabled.
"""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chatbot.foundry_client import FileMetadata, KnowledgeChunk


# Single-bracket form ``[1]``, ``[2]``, etc. Multi-cite groupings should be
# emitted as ``[1][2]`` per the system prompt; ``[1, 2]`` would only match
# the first integer (visible defect, not a silent bug).
CITATION_RE = re.compile(r"\[(\d+)\]")

SNIPPET_MAX_CHARS = 320

UNMATCHED_WARNING = (
    "*[Note: some citation markers in the answer below referenced "
    "non-existent source numbers and were removed. Treat any un-cited "
    "claim as unsupported by the retrieved documents.]*\n\n"
)


# ── Citation extraction / stripping ────────────────────────────────────


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


def stylize_inline_citations(text: str) -> str:
    """Wrap remaining ``[N]`` markers in ``<sup><b>[N]</b></sup>`` for
    chip-like superscript appearance. Call AFTER ``strip_unmatched`` so
    only valid markers get styled. Requires ``unsafe_allow_html=true``."""
    return CITATION_RE.sub(lambda m: f"<sup><b>{m.group(0)}</b></sup>", text)


# ── Sources block ──────────────────────────────────────────────────────


def _truncate_snippet(content: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    if not content:
        return ""
    collapsed = re.sub(r"\s+", " ", content).strip()
    if len(collapsed) <= max_chars:
        return collapsed
    cut = collapsed[:max_chars].rsplit(" ", 1)[0] or collapsed[:max_chars]
    return cut + "…"


def _format_match_type(retrieval_method: str | None) -> str:
    """Map raw service field to a user-friendly label."""
    if retrieval_method == "bm25_fulltext":
        return "full-text"
    if retrieval_method == "pinecone_cosine":
        return "semantic"
    if retrieval_method == "both":
        return "full-text + semantic"
    return retrieval_method or "—"


def _format_authority(authority: str | None) -> str:
    if not authority:
        return "—"
    return authority.lower()


def _format_size_kb(size_bytes: int | None) -> str | None:
    if not size_bytes:
        return None
    kb = size_bytes / 1024
    if kb < 1024:
        return f"{kb:.0f} KB"
    return f"{kb / 1024:.1f} MB"


def _format_ingested(created_at: str | None) -> str | None:
    if not created_at or not isinstance(created_at, str):
        return None
    # Expect ISO 8601 like '2026-02-24T10:34:41.599204+00:00'; take YYYY-MM-DD.
    return created_at[:10] if len(created_at) >= 10 else None


def _normalize_score(raw: float) -> float:
    """Inline copy of retriever.normalize_rrf_score to avoid circular import.

    Foundry's RRF uses k=60, max raw = 2/(k+1). Match scaling.
    """
    rrf_max = 2.0 / 61.0
    if raw <= 0:
        return 0.0
    return min(raw / rrf_max, 1.0)


def render_sources_section(
    chunks: list[KnowledgeChunk],
    cited_indices: list[int],
    file_metadata_by_id: dict[str, FileMetadata] | None = None,
) -> str:
    """Render a collapsible Sources block (HTML <details> + Markdown body).

    Numbers are 1-based and refer to chunk position in ``chunks`` (the
    same order the chunks appear in the LLM's context block, so [1] in
    the answer maps to ``chunks[0]``).

    Returns an empty string when no chunks are cited.

    Output shape:
        <blank line>
        <blank line>
        <details>
        <summary>📚 Sources (N)</summary>

        **[1]** **document.pdf**
        📚 Library · 📑 Section · 📃 p. 4–5 · ✓ Authority: validated
          · 🎯 Match: 51% (full-text) · 📅 Ingested: 2026-02-24
          · 🔖 Ref: c_91c59123

        > Snippet excerpt…

        **[2]** ...

        </details>

    The leading two blank lines force a paragraph break before <details>
    so the answer's last line is never absorbed into setext-H2 syntax.
    """
    if not cited_indices:
        return ""

    fm_map = file_metadata_by_id or {}

    parts: list[str] = [
        "",
        "",
        "<details>",
        f"<summary>📚 Sources ({len(cited_indices)})</summary>",
        "",
    ]

    for n in cited_indices:
        idx = n - 1
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        meta = fm_map.get(chunk.source_file_id) if chunk.source_file_id else None

        # ── Document name ────────────────────────────────────────
        doc_name = (
            meta.original_filename
            if meta and meta.original_filename
            else (chunk.section_title or "Unknown document")
        )
        doc_name_safe = html.escape(doc_name)

        # ── Metadata pieces ──────────────────────────────────────
        library = chunk.source_name or (
            meta.source_name if meta else "Unknown library"
        )
        meta_bits: list[str] = [f"📚 {html.escape(library)}"]

        if chunk.section_title:
            # Avoid duplicating the section if it's the same as the filename
            if not doc_name.startswith(chunk.section_title):
                meta_bits.append(
                    f"📑 Section: *{html.escape(chunk.section_title)}*"
                )

        if chunk.page_numbers:
            pages_str = ", ".join(str(p) for p in chunk.page_numbers)
            meta_bits.append(f"📃 Page(s): {html.escape(pages_str)}")

        meta_bits.append(
            f"✓ Authority: {html.escape(_format_authority(chunk.authority_level))}"
        )

        match_pct = round(_normalize_score(chunk.relevance_score or 0.0) * 100)
        match_type = _format_match_type(chunk.retrieval_method)
        meta_bits.append(f"🎯 Match: {match_pct}% ({html.escape(match_type)})")

        if chunk.chunk_type and chunk.chunk_type != "text":
            meta_bits.append(f"🏷️ Type: {html.escape(chunk.chunk_type)}")

        if meta:
            ingested = _format_ingested(meta.created_at)
            if ingested:
                meta_bits.append(f"📅 Ingested: {ingested}")
            size = _format_size_kb(meta.size_bytes)
            if size:
                meta_bits.append(f"💾 {size}")

        # 8-char chunk ID prefix — full UUID truncated for the audit trail.
        ref_short = chunk.chunk_id[:8] if chunk.chunk_id else "—"
        meta_bits.append(f"🔖 Ref: c_{html.escape(ref_short)}")

        meta_line = " · ".join(meta_bits)
        snippet = _truncate_snippet(chunk.content or "")

        parts.append(f"**[{n}]** **{doc_name_safe}**  ")
        parts.append(meta_line)
        parts.append("")
        if snippet:
            parts.append(f"> {html.escape(snippet)}")
        parts.append("")

    parts.append("</details>")
    return "\n".join(parts)
