"""Typed exceptions for distinguishing error handling paths."""

from __future__ import annotations


class FoundryError(Exception):
    """Base for all Foundry-API-adjacent errors."""


class FoundryAuthError(FoundryError):
    """401/403 from Foundry — misconfigured token or actor-tenant mismatch.

    Do NOT show raw to end users; surface the canned friendly message. But
    DO log at ERROR level so operators see "CHECK CHATBOT_INTERNAL_TOKEN"
    in Railway logs (Senior Reviewer finding).
    """


class FoundryTransientError(FoundryError):
    """5xx / timeout / connection reset — eligible for retry."""


class FoundryMalformedResponseError(FoundryError):
    """Response body wasn't the shape we expected. Likely contract drift."""
