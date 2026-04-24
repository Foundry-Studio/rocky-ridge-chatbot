"""Render chainlit.md + .chainlit/config.toml at container start.

Reads runtime env (CHATBOT_TENANT_DISPLAY_NAME) and substitutes into
templates so Railway redeploys pick up env changes without rebuilds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def render(template_path: Path, out_path: Path, replacements: dict[str, str]) -> None:
    if not template_path.exists():
        print(f"[render_config] template not found, skipping: {template_path}")
        return
    text = template_path.read_text(encoding="utf-8")
    for key, val in replacements.items():
        text = text.replace("{{" + key + "}}", val)
    out_path.write_text(text, encoding="utf-8")
    print(f"[render_config] wrote {out_path}")


def main() -> int:
    tenant = os.environ.get("CHATBOT_TENANT_DISPLAY_NAME", "Knowledge Base")
    replacements = {"TENANT_DISPLAY_NAME": tenant}

    render(
        ROOT / "chainlit.md.template",
        ROOT / "chainlit.md",
        replacements,
    )
    render(
        ROOT / ".chainlit" / "config.toml.template",
        ROOT / ".chainlit" / "config.toml",
        replacements,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
