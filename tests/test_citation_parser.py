"""Citation parser tests — Perplexity-style numeric citations.

The G6 integrity layer: model emits ``[N]``; parser strips any [N] whose
N is outside ``1..len(chunks)``; render_sources_section emits a Markdown
section listing only cited chunks.
"""

from __future__ import annotations

import pytest
from chatbot.citation_parser import (
    CITATION_RE,
    SNIPPET_MAX_CHARS,
    extract_cited_indices,
    render_sources_section,
    strip_unmatched,
)

# ── Tiny chunk stand-in (avoid importing pydantic in the parser tests) ──


class _Chunk:
    """Duck-typed stand-in for KnowledgeChunk with the attrs we read."""

    def __init__(
        self,
        content: str,
        source_name: str | None = None,
        section_title: str | None = None,
        page_numbers: list[int] | None = None,
    ):
        self.content = content
        self.source_name = source_name
        self.section_title = section_title
        self.page_numbers = page_numbers


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
    # No floating space before the period
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


# ── render_sources_section ──────────────────────────────────────────────


def test_render_empty_when_no_citations():
    chunks = [_Chunk("body", source_name="X")]
    assert render_sources_section(chunks, []) == ""


def test_render_single_citation_full_metadata():
    chunks = [
        _Chunk(
            "Canebrake (Arundinaria gigantea) forms dense thickets.",
            source_name="Conservation Guide",
            section_title="Canebrake Restoration",
            page_numbers=[12, 13],
        )
    ]
    out = render_sources_section(chunks, [1])
    assert "**Sources**" in out
    assert "**[1]**" in out
    assert "*Conservation Guide*" in out
    assert "*Canebrake Restoration*" in out
    assert "p. 12, 13" in out
    assert "Canebrake (Arundinaria gigantea)" in out


def test_render_handles_missing_optional_fields():
    chunks = [_Chunk("Just text.", source_name=None)]
    out = render_sources_section(chunks, [1])
    assert "Unknown source" in out
    # No section, no pages
    assert " — *" not in out
    assert "(p." not in out


def test_render_truncates_long_snippet():
    long_content = "x" * (SNIPPET_MAX_CHARS + 200)
    chunks = [_Chunk(long_content, source_name="X")]
    out = render_sources_section(chunks, [1])
    # Snippet should be truncated with ellipsis
    assert "…" in out
    # And shouldn't include the full content
    assert long_content not in out


def test_render_collapses_whitespace_in_snippet():
    chunks = [_Chunk("line one\n\n\nline   two\nline three", source_name="X")]
    out = render_sources_section(chunks, [1])
    assert "line one line two line three" in out


def test_render_skips_out_of_range_indices_silently():
    chunks = [_Chunk("a", source_name="X")]
    out = render_sources_section(chunks, [1, 99])
    # [1] renders, [99] silently skipped (strip_unmatched should have caught
    # this earlier — defense in depth here)
    assert "**[1]**" in out
    assert "**[99]**" not in out


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


def test_render_separator_above_section():
    chunks = [_Chunk("body", source_name="X")]
    out = render_sources_section(chunks, [1])
    assert "\n---\n" in out


# ── Cross-module sanity: extract followed by render ─────────────────────


def test_full_pipeline_basic():
    chunks = [
        _Chunk("alpha", source_name="A"),
        _Chunk("beta", source_name="B"),
    ]
    answer = "Some claim [1]. Another claim [2]. Repeat [1]."
    cleaned, matched, unmatched = strip_unmatched(answer, max_index=len(chunks))
    section = render_sources_section(chunks, matched)
    assert matched == [1, 2]
    assert unmatched == []
    assert "[1]" in cleaned
    assert "[2]" in cleaned
    assert "**Sources**" in section
    assert "*A*" in section
    assert "*B*" in section


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
