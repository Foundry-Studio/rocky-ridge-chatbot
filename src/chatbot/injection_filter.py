"""Cheap pre-LLM filter for obvious prompt-injection shapes + post-LLM
leak detection.

NOT a silver bullet. A sophisticated attacker will still slip things past
this. Defense in depth against casual probes, not security boundary.
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"^\s*system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"</?context>", re.IGNORECASE),
    re.compile(r"repeat\s+(everything|all|the\s+text)\s+above", re.IGNORECASE),
    re.compile(r"\n{4,}"),  # 4+ newlines — smuggling attempt
]

# Distinctive phrases from our system prompt. If any appears in the model's
# output, it probably leaked the system prompt. Update when prompt is rewritten.
_SYSTEM_PROMPT_LEAK_MARKERS = (
    "having an ongoing conversation with a user about",
    "NEW factual claims must be supported by the <context>",
    "Cite each new factual claim inline as [N]",
    "Never invent chunk numbers",
    "<conversation_so_far>",
)


def is_injection(message: str) -> bool:
    return any(p.search(message) for p in _INJECTION_PATTERNS)


def leaks_system_prompt(text: str) -> bool:
    return any(m in text for m in _SYSTEM_PROMPT_LEAK_MARKERS)
