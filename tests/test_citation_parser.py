"""Citation parser tests — Perplexity-style numeric citations.

The G6 integrity layer: model emits ``[N]``; parser strips any [N] whose
N is outside ``1..len(chunks)``; render_sources_section emits a
collapsible HTML block with credibility metadata.
"""

from __future__ import annotations

import pytest
from chatbot.citation_parser import (
    CITATION_RE,
    SNIPPET_MAX_CHARS,
    extract_cited_indices,
    render_sources_section,
    strip_unmatched,
    stylize_inline_citations,
)

# ── Tiny stand-ins (avoid importing pydantic for parser-only tests) ──


class _Chunk:
    """Duck-typed stand-in for KnowledgeChunk."""

    def __init__(
        self,
        content: str,
        chunk_id: str = "00000000-0000-0000-0000-000000000000",
        source_file_id: str | None = None,
        source_name: str | None = None,
        section_title: str | None = None,
        page_numbers: list[int] | None = None,
        authority_level: str | None = "validated",
        relevance_score: float | None = 0.01666,
        chunk_type: str | None = "text",
        retrieval_method: str | None = "bm25_fulltext",
    ):
        self.content = content
        self.chunk_id = chunk_id
        self.source_file_id = source_file_id
        self.source_name = source_name
        self.section_title = section_title
        self.page_numbers = page_numbers
        self.authority_level = authority_level
        self.relevance_score = relevance_score
        self.chunk_type = chunk_type
        self.retrieval_method = retrieval_method


class _FileMeta:
    """Duck-typed stand-in for FileMetadata."""

    def __init__(
        self,
        original_filename: str | None = None,
        source_name: str | None = None,
        size_bytes: int | None = None,
        created_at: str | None = None,
    ):
        self.original_filename = original_filename
        self.source_name = source_name
        self.size_bytes = size_bytes
        self.created_at = created_at


# ── Regex / extraction ──────────────────────────────────────────────────


def test_extract_returns_unique_order():
    t = "A [1] B [2] A again [1] then [3]."
    assert extract_cited_indices(t) == [1, 2, 3]


def test_extract_ignores_non_numeric():
    t = "Start [ref:c_xxx] middle [abc] [12]."
    assert extract_cited_indices(t) == [12]


def test_extract_handles_multi_digit():
    t = "Cite [10] and [3] then [100]."
    assert extract_cited_indices(t) == [10, 3, 100]


def test_regex_only_matches_bracket_integer():
    matches = CITATION_RE.findall("[1] [2.0] [3a] [4]")
    assert matches == ["1", "4"]


# ── strip_unmatched ─────────────────────────────────────────────────────


def test_strip_keeps_in_range_strips_out_of_range():
    t = "Fact one [1]. Fact two [99]."
    cleaned, matched, unmatched = strip_unmatched(t, max_index=3)
    assert "[1]" in cleaned
    assert "[99]" not in cleaned
    assert matched == [1]
    assert unmatched == [99]


def test_strip_n_zero_is_unmatched():
    cleaned, matched, unmatched = strip_unmatched("Bad cite [0].", max_index=5)
    assert "[0]" not in cleaned
    assert matched == []
    assert unmatched == [0]


def test_strip_preserves_ordering_for_repeats():
    t = "[1] then [2] then [1] again [3] [2]."
    cleaned, matched, unmatched = strip_unmatched(t, max_index=5)
    assert matched == [1, 2, 3]
    assert unmatched == []
    assert cleaned.count("[1]") == 2
    assert cleaned.count("[2]") == 2
    assert cleaned.count("[3]") == 1


def test_strip_collapses_whitespace_after_removal():
    t = "Hello [99] world."
    cleaned, _, unmatched = strip_unmatched(t, max_index=3)
    assert unmatched == [99]
    assert "  " not in cleaned
    assert "[99]" not in cleaned


def test_strip_tightens_orphan_punctuation():
    t = "End of clause [99] ."
    cleaned, _, _ = strip_unmatched(t, max_index=3)
    assert "[99]" not in cleaned
    assert cleaned.endswith(" .") is False


def test_strip_empty_text_is_safe():
    cleaned, m, u = strip_unmatched("", max_index=0)
    assert cleaned == ""
    assert m == [] and u == []


def test_strip_max_index_zero_strips_everything():
    cleaned, m, u = strip_unmatched("Cite [1] [2].", max_index=0)
    assert m == []
    assert u == [1, 2]
    assert "[1]" not in cleaned
    assert "[2]" not in cleaned


@pytest.mark.parametrize(
    "raw,max_index,want_matched,want_unmatched",
    [
        ("[1]", 1, [1], []),
        ("[2]", 1, [], [2]),
        ("[1][2]", 2, [1, 2], []),
        ("[1][2][3]", 2, [1, 2], [3]),
        ("[0]", 5, [], [0]),
    ],
)
def test_strip_table(raw, max_index, want_matched, want_unmatched):
    _, m, u = strip_unmatched(raw, max_index=max_index)
    assert m == want_matched
    assert u == want_unmatched


# ── stylize_inline_citations ────────────────────────────────────────────


def test_stylize_wraps_each_citation_in_sup():
    t = "Claim one [1]. Claim two [2]."
    out = stylize_inline_citations(t)
    assert "<sup><b>[1]</b></sup>" in out
    assert "<sup><b>[2]</b></sup>" in out
    # Non-citation text untouched
    assert "Claim one " in out
    assert "Claim two " in out


def test_stylize_preserves_text_with_no_citations():
    t = "No citations here at all."
    assert stylize_inline_citations(t) == t


def test_stylize_handles_multi_cite():
    out = stylize_inline_citations("multi [1][2] cite")
    assert "<sup><b>[1]</b></sup><sup><b>[2]</b></sup>" in out


# ── render_sources_section — collapsible <details> + credibility ────────


def test_render_empty_when_no_citations():
    chunks = [_Chunk("body", source_name="X")]
    assert render_sources_section(chunks, []) == ""


def test_render_wraps_in_details_summary():
    chunks = [_Chunk("body", source_name="Lib")]
    out = render_sources_section(chunks, [1])
    assert "<details>" in out
    assert "<summary>📚 Sources (1)</summary>" in out
    assert "</details>" in out


def test_render_summary_count_matches_cited():
    chunks = [_Chunk(f"c{i}", source_name="Lib") for i in range(5)]
    out = render_sources_section(chunks, [1, 3, 5])
    assert "<summary>📚 Sources (3)</summary>" in out


def test_render_uses_file_metadata_filename_when_available():
    chunks = [
        _Chunk("body", source_file_id="fid-1", source_name="Lib"),
    ]
    fm = {"fid-1": _FileMeta(original_filename="Cumberland_Plateau_Guide.pdf")}
    out = render_sources_section(chunks, [1], file_metadata_by_id=fm)
    assert "Cumberland_Plateau_Guide.pdf" in out


def test_render_falls_back_to_section_title_when_no_metadata():
    chunks = [
        _Chunk(
            "body",
            source_file_id="fid-missing",
            section_title="REM-Canebrake-Ecology",
            source_name="Lib",
        )
    ]
    out = render_sources_section(chunks, [1], file_metadata_by_id={})
    assert "REM-Canebrake-Ecology" in out


def test_render_falls_back_to_unknown_document_when_nothing_available():
    chunks = [_Chunk("body", source_name="Lib")]
    out = render_sources_section(chunks, [1])
    assert "Unknown document" in out


def test_render_includes_library_authority_match_ref():
    chunks = [
        _Chunk(
            "body",
            chunk_id="91c59123-8971-4bf4-9e09-96ff6f7cb7a4",
            source_name="Rocky Ridge Research Library",
            authority_level="validated",
            relevance_score=0.0167,
            retrieval_method="bm25_fulltext",
        )
    ]
    out = render_sources_section(chunks, [1])
    assert "📚 Rocky Ridge Research Library" in out
    assert "Authority: validated" in out
    assert "Match: 51% (full-text)" in out
    assert "Ref: c_91c59123" in out


def test_render_includes_section_when_distinct_from_filename():
    chunks = [
        _Chunk(
            "body",
            source_file_id="fid-1",
            section_title="Canebrake Restoration",
            source_name="Lib",
        )
    ]
    fm = {"fid-1": _FileMeta(original_filename="Conservation_Guide.pdf")}
    out = render_sources_section(chunks, [1], file_metadata_by_id=fm)
    assert "Canebrake Restoration" in out


def test_render_skips_section_when_same_as_filename():
    chunks = [
        _Chunk(
            "body",
            source_file_id="fid-1",
            section_title="F122XY026TN",
            source_name="Lib",
        )
    ]
    fm = {"fid-1": _FileMeta(original_filename="F122XY026TN.pdf")}
    out = render_sources_section(chunks, [1], file_metadata_by_id=fm)
    # section_title is a prefix of filename — don't duplicate
    assert "📑 Section: *F122XY026TN*" not in out
    # But filename is still there
    assert "F122XY026TN.pdf" in out


def test_render_includes_pages_when_present():
    chunks = [_Chunk("body", source_name="Lib", page_numbers=[3, 4])]
    out = render_sources_section(chunks, [1])
    assert "Page(s): 3, 4" in out


def test_render_omits_pages_when_absent():
    chunks = [_Chunk("body", source_name="Lib", page_numbers=None)]
    out = render_sources_section(chunks, [1])
    assert "Page(s)" not in out


def test_render_includes_ingested_date_from_metadata():
    chunks = [_Chunk("body", source_file_id="fid-1", source_name="Lib")]
    fm = {
        "fid-1": _FileMeta(
            original_filename="x.pdf",
            created_at="2026-02-24T10:34:41.599204+00:00",
        )
    }
    out = render_sources_section(chunks, [1], file_metadata_by_id=fm)
    assert "Ingested: 2026-02-24" in out


def test_render_includes_size_when_metadata_present():
    chunks = [_Chunk("body", source_file_id="fid-1", source_name="Lib")]
    fm = {
        "fid-1": _FileMeta(original_filename="x.pdf", size_bytes=252330)
    }
    out = render_sources_section(chunks, [1], file_metadata_by_id=fm)
    assert "246 KB" in out


def test_render_match_type_labels():
    cases = [
        ("bm25_fulltext", "full-text"),
        ("pinecone_cosine", "semantic"),
        ("both", "full-text + semantic"),
    ]
    for raw, label in cases:
        chunks = [_Chunk("body", source_name="Lib", retrieval_method=raw)]
        out = render_sources_section(chunks, [1])
        assert label in out, f"label {label!r} missing for raw {raw!r}"


def test_render_includes_chunk_type_when_not_text():
    chunks = [
        _Chunk("body", source_name="Lib", chunk_type="figure_caption")
    ]
    out = render_sources_section(chunks, [1])
    assert "Type: figure_caption" in out


def test_render_omits_chunk_type_when_text():
    chunks = [_Chunk("body", source_name="Lib", chunk_type="text")]
    out = render_sources_section(chunks, [1])
    assert "Type: text" not in out


def test_render_truncates_long_snippet():
    long_content = "x" * (SNIPPET_MAX_CHARS + 200)
    chunks = [_Chunk(long_content, source_name="X")]
    out = render_sources_section(chunks, [1])
    assert "…" in out
    assert long_content not in out


def test_render_html_escapes_user_visible_content():
    chunks = [
        _Chunk(
            "<script>alert(1)</script>",
            source_file_id="fid-1",
            source_name="<b>Library</b>",
            section_title="<em>Section</em>",
        )
    ]
    fm = {"fid-1": _FileMeta(original_filename="<img src=x onerror=alert(1)>")}
    out = render_sources_section(chunks, [1], file_metadata_by_id=fm)
    # All raw HTML chars escaped — no live tags inside the Sources block
    assert "<script>" not in out
    assert "<img src=x" not in out
    assert "<b>Library</b>" not in out
    assert "&lt;script&gt;" in out
    assert "&lt;b&gt;Library&lt;/b&gt;" in out


def test_render_does_not_create_setext_h2_when_appended_to_answer():
    """Two leading blank lines guarantee no setext-H2 even if answer ends
    with text and no trailing newline."""
    chunks = [_Chunk("body", source_name="X")]
    answer = "The final sentence of the answer."
    sources = render_sources_section(chunks, [1])
    combined = answer + sources
    # Must have a paragraph break before the <details> block
    assert "answer.\n\n<details>" in combined


def test_render_preserves_citation_order():
    chunks = [
        _Chunk("alpha", source_name="A"),
        _Chunk("beta", source_name="B"),
        _Chunk("gamma", source_name="C"),
    ]
    out = render_sources_section(chunks, [3, 1, 2])
    pos_a = out.index("alpha")
    pos_b = out.index("beta")
    pos_c = out.index("gamma")
    assert pos_c < pos_a < pos_b


def test_render_skips_out_of_range_indices_silently():
    chunks = [_Chunk("a", source_name="X")]
    out = render_sources_section(chunks, [1, 99])
    assert "**[1]**" in out
    assert "**[99]**" not in out
