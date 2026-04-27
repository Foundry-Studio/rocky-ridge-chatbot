"""Research-trace rendering — collapsible '🔬 Research trace' block
appended to the message body, showing the tool calls the agent made
during the turn.

Pure Markdown + HTML (renders via Chainlit's unsafe_allow_html=true).
Sibling of citation_parser.render_sources_section_global.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chatbot.agent import ToolCallTrace


def _icon_for(tool_name: str) -> str:
    return {
        "search_knowledge": "🔍",
        "get_chunk_neighbors": "📎",
        "read_document_section": "📖",
    }.get(tool_name, "🔧")


def _short(s: str, n: int = 80) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def _summarize_input(name: str, args: dict) -> str:
    if name == "search_knowledge":
        q = args.get("query", "")
        k = args.get("top_k")
        suffix = f" (top {k})" if k else ""
        return f"\"{_short(str(q), 80)}\"{suffix}"
    if name == "get_chunk_neighbors":
        cid = (args.get("chunk_id") or "")[:8]
        b = args.get("before", 2)
        a = args.get("after", 2)
        return f"chunk c_{cid} (±{b}/{a})"
    if name == "read_document_section":
        did = (args.get("document_id") or "")[:8]
        start = args.get("start_chunk", 0)
        cnt = args.get("chunk_count", 10)
        return f"doc {did} chunks {start}–{start + cnt - 1}"
    # Fallback — JSON-ish
    return _short(str(args), 80)


def render_research_trace(
    trace: list[ToolCallTrace],
    iterations: int,
    total_latency_ms: int,
    error: str | None = None,
) -> str:
    """Render a collapsible '🔬 Research trace' block.

    Returns "" when no tool calls happened (meta-question path) and
    there's no error to surface.
    """
    if not trace and not error:
        return ""

    total_chunks = sum(t.chunks_returned for t in trace)
    summary = (
        f"🔬 Research trace ({len(trace)} step"
        f"{'s' if len(trace) != 1 else ''} · "
        f"{total_chunks} chunks · "
        f"{total_latency_ms / 1000:.1f}s · "
        f"{iterations} iter)"
    )

    parts: list[str] = ["", "", "<details>", f"<summary>{summary}</summary>", ""]
    if error:
        parts.append(f"⚠️ *Stopped: {html.escape(error)}*")
        parts.append("")

    for i, tc in enumerate(trace, start=1):
        icon = _icon_for(tc.name)
        input_summary = _summarize_input(tc.name, tc.input)
        line = (
            f"**{i}.** {icon} `{html.escape(tc.name)}` "
            f"— {html.escape(input_summary)}"
        )
        if tc.is_error:
            line += (
                f" — ❌ *{html.escape(tc.error_message or 'error')[:120]}*"
            )
        else:
            line += (
                f" — *{tc.chunks_returned} chunks · {tc.latency_ms}ms*"
            )
        parts.append(line)
    parts.append("")
    parts.append("</details>")
    return "\n".join(parts)
