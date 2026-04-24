"""Pytest fixtures for the chatbot test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Force required env vars BEFORE any chatbot module imports happen so
# Settings() validation passes. Tests that need specific values override
# via monkeypatch.
_TEST_DEFAULTS = {
    "FOUNDRY_API_BASE_URL": "https://foundry-agent-system-test.invalid",
    "FOUNDRY_INTERNAL_TOKEN": "test-chatbot-token-32-chars-or-more-xyz",
    "CHATBOT_TENANT_ID": "80252ad9-72d5-4c5a-b273-af804224872e",
    "CHATBOT_KNOWLEDGE_SOURCE_ID": "52ad54c0-acfd-49db-b9b4-dc9c6098d9f6",
    "CHATBOT_TENANT_DISPLAY_NAME": "Rocky Ridge Land Management",
    "CHATBOT_MODEL_ID": "anthropic/claude-sonnet-4-5",
    "CHATBOT_LOG_PATH": "./test-conversations.jsonl",
    "CHATBOT_DAILY_SPEND_PATH": "./test-daily-spend.json",
}
for k, v in _TEST_DEFAULTS.items():
    os.environ.setdefault(k, v)


@pytest.fixture
def settings():
    """Fresh Settings instance built from current env."""
    from chatbot.config import Settings

    return Settings()  # type: ignore[call-arg]


@pytest.fixture(autouse=True)
def _reset_foundry_client():
    """Each test gets a clean module-level httpx client slot.
    Tests that want a pre-configured client call foundry_client.set_test_client."""
    from chatbot import foundry_client

    foundry_client.set_test_client(None)
    yield
    foundry_client.set_test_client(None)


@pytest.fixture
def tmp_log_settings(tmp_path: Path, monkeypatch):
    """Per-test tmp paths + a Settings instance that reflects them.

    Use this INSTEAD of the ``settings`` fixture when a test writes to the
    conversation log or daily-spend file — otherwise the ``settings``
    fixture may resolve before env patching and hand you the conftest
    default paths.

    Returns a (path, settings) tuple: path is the log file, settings is a
    fresh Settings() instance built after env patching.
    """
    from chatbot.config import Settings

    log_path = tmp_path / "conversations.jsonl"
    spend_path = tmp_path / "spend.json"
    monkeypatch.setenv("CHATBOT_LOG_PATH", str(log_path))
    monkeypatch.setenv("CHATBOT_DAILY_SPEND_PATH", str(spend_path))
    fresh = Settings()  # type: ignore[call-arg]
    return log_path, fresh


@pytest.fixture
def tmp_log_path(tmp_log_settings):
    """Compatibility alias — just returns the path."""
    path, _ = tmp_log_settings
    return path
