"""Typed settings loaded from environment at import time.

Required vars fail loudly at startup — never silently at first user turn.
Optional vars default to the demo-sized values locked in the VKC plan §5.
"""

from __future__ import annotations

import re

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Tenant display name is substituted into the system prompt at runtime.
# Restrict to a safe character set to prevent prompt-injection via env vars
# (Stress Tester finding #21).
_TENANT_NAME_RE = re.compile(r"^[A-Za-z0-9 \-'.&,()]{1,80}$")


class Settings(BaseSettings):
    """Chatbot runtime configuration.

    All tunables come from environment variables. No defaults for the 6
    required fields — the app will refuse to start without them so a
    misconfigured Railway deploy fails on boot, not on first user.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Required ─────────────────────────────────────────────────
    foundry_api_base_url: str
    foundry_internal_token: str
    chatbot_tenant_id: str
    chatbot_knowledge_source_id: str | None = None
    chatbot_tenant_display_name: str = Field(..., min_length=1, max_length=80)
    chatbot_model_id: str = "anthropic/claude-sonnet-4-5"

    # ── Identity ─────────────────────────────────────────────────
    chatbot_actor_id: str = "rocky-ridge-chatbot"

    # ── Retrieval ────────────────────────────────────────────────
    chatbot_max_chunks: int = Field(default=8, ge=1, le=20)
    chatbot_refusal_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # ── History ──────────────────────────────────────────────────
    chatbot_max_history_turns: int = Field(default=6, ge=0, le=20)

    # ── LLM ──────────────────────────────────────────────────────
    chatbot_max_tokens: int = Field(default=800, ge=50, le=4096)
    chatbot_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    chatbot_reformulation_max_tokens: int = Field(default=128, ge=32, le=512)

    # ── Timeouts (seconds) ──────────────────────────────────────
    retrieval_timeout_s: float = Field(default=10.0, ge=1.0, le=60.0)
    llm_timeout_s: float = Field(default=60.0, ge=5.0, le=300.0)
    reformulation_timeout_s: float = Field(default=5.0, ge=1.0, le=30.0)

    # ── Message length ──────────────────────────────────────────
    message_len_max: int = Field(default=2000, ge=100, le=10000)

    # ── Agentic loop ────────────────────────────────────────────
    chatbot_max_agent_iterations: int = Field(default=6, ge=1, le=12)
    chatbot_max_agent_wall_clock_sec: float = Field(default=60.0, ge=10.0, le=300.0)

    # ── Logging ─────────────────────────────────────────────────
    chatbot_log_path: str = "/data/conversations.jsonl"
    chatbot_log_rotate_mb: int = Field(default=100, ge=1, le=2048)
    chatbot_log_retention_days: int = Field(default=30, ge=1, le=3650)

    # ── Cost control ─────────────────────────────────────────────
    chatbot_daily_usd_cap: float = Field(default=50.0, ge=0.0, le=10000.0)
    chatbot_rate_limit_per_ip_per_min: int = Field(default=5, ge=1, le=1000)
    chatbot_daily_spend_path: str = "/data/daily_spend.json"

    # ── Alerting ─────────────────────────────────────────────────
    chatbot_alert_slack_webhook: str | None = None

    # ── Validation ───────────────────────────────────────────────
    @field_validator("chatbot_tenant_display_name")
    @classmethod
    def _validate_display_name(cls, v: str) -> str:
        if not _TENANT_NAME_RE.match(v):
            raise ValueError(
                "chatbot_tenant_display_name must match "
                f"{_TENANT_NAME_RE.pattern} — got {v!r}"
            )
        return v

    @field_validator(
        "foundry_api_base_url",
        "foundry_internal_token",
        "chatbot_tenant_id",
    )
    @classmethod
    def _reject_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("required field cannot be blank")
        return v


def get_settings() -> Settings:
    """Lazy-load settings so tests can inject mocked env vars before import."""
    return Settings()  # type: ignore[call-arg]
