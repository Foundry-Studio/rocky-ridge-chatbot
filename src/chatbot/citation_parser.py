"""Citation extraction + fake-citation stripping.

This is the G6 integrity layer — if the model emits [ref:<id>] where <id>
doesn't match any retrieved chunk, we REMOVE the marker from the visible
text and prepend a warning banner. Silent drop (which was v1's behavior)
leaves the user looking at a plain-text 'ref:bogus' badge that clicks to
nothing — worst-case RAG failure (looks grounded, isn't).
"""

from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[ref:([a-zA-Z0-9_-]{4,64})\]")


def extract_cited_ids(text: str) -> list[str]:
    """Return unique citation IDs in first-appearance order."""
    seen: set[str] = set()
    out: list[str] = []
    for match in CITATION_RE.finditer(text):
        cid = match.group(1)
        if cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def strip_unmatched(
    text: str, short_id_map: dict[str, str]
) -> tuple[str, list[str], list[str]]:
    """Remove [ref:X] markers whose X is not in short_id_map.

    Returns (cleaned_text, matched_ids_in_order, unmatched_ids_in_order).
    """
    matched: list[str] = []
    unmatched: list[str] = []
    seen_matched: set[str] = set()
    seen_unmatched: set[str] = set()

    def _sub(m: re.Match[str]) -> str:
        cid = m.group(1)
        if cid in short_id_map:
            if cid not in seen_matched:
                seen_matched.add(cid)
                matched.append(cid)
            return m.group(0)  # keep the marker intact
        else:
            if cid not in seen_unmatched:
                seen_unmatched.add(cid)
                unmatched.append(cid)
            return ""  # strip the fake marker

    cleaned = CITATION_RE.sub(_sub, text)
    # Collapse double-spaces that resulted from mid-text marker removal.
    cleaned = re.sub(r"  +", " ", cleaned)
    cleaned = re.sub(r" +([.,;:!?])", r"\1", cleaned)
    return cleaned, matched, unmatched


UNMATCHED_WARNING = (
    "*[Note: some citation markers in the answer below referenced "
    "non-existent source chunks and were removed. Treat any un-cited "
    "claim as unsupported by the retrieved documents.]*\n\n"
)
