"""Citation parser tests — the G6 integrity layer.

Fake [ref:X] markers must be STRIPPED from visible text (not silently
kept), and the caller must be informed so it can prepend a warning banner.
"""

from __future__ import annotations

from chatbot.citation_parser import (
    CITATION_RE,
    extract_cited_ids,
    strip_unmatched,
)


def test_extract_returns_unique_order():
    t = "A [ref:c_11111111] B [ref:c_22222222] A again [ref:c_11111111]."
    assert extract_cited_ids(t) == ["c_11111111", "c_22222222"]


def test_extract_ignores_malformed():
    t = "Start [ref:] middle [ref:ab] too-short [ref:c_99999999] end."
    # [ref:] doesn't match {4,64}; [ref:ab] is 2 chars < 4.
    assert extract_cited_ids(t) == ["c_99999999"]


def test_strip_keeps_matched_removes_unmatched():
    t = "Fact one [ref:c_real1234]. Fact two [ref:c_fake9999]."
    cleaned, matched, unmatched = strip_unmatched(
        t, short_id_map={"c_real1234": "uuid-1"}
    )
    assert "ref:c_real1234" in cleaned
    assert "ref:c_fake9999" not in cleaned
    assert matched == ["c_real1234"]
    assert unmatched == ["c_fake9999"]


def test_strip_preserves_ordering():
    t = "[ref:c_aaaa0001] then [ref:c_bbbb0002] then [ref:c_aaaa0001]."
    cleaned, matched, unmatched = strip_unmatched(
        t,
        short_id_map={"c_aaaa0001": "u1", "c_bbbb0002": "u2"},
    )
    assert matched == ["c_aaaa0001", "c_bbbb0002"]
    assert unmatched == []
    assert cleaned.count("ref:c_aaaa0001") == 2


def test_strip_collapses_whitespace():
    t = "Hello [ref:c_bogus123] world."
    cleaned, _, unmatched = strip_unmatched(t, short_id_map={})
    assert unmatched == ["c_bogus123"]
    # Double-space gap should collapse; no leading space before period.
    assert "  " not in cleaned


def test_strip_all_unmatched_returns_clean_prose():
    t = "All fabricated [ref:c_fake1234] including [ref:c_fake5678]."
    cleaned, matched, unmatched = strip_unmatched(t, short_id_map={})
    assert matched == []
    assert unmatched == ["c_fake1234", "c_fake5678"]
    assert "ref:" not in cleaned


def test_regex_accepts_uuid_style():
    # Chunks often have full UUID IDs; regex allows up to 64 chars.
    t = "[ref:ceea25d1-3d24-4d12-a134-9343adc8cb32]"
    ids = CITATION_RE.findall(t)
    assert ids == ["ceea25d1-3d24-4d12-a134-9343adc8cb32"]
