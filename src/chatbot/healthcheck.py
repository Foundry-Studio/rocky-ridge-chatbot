"""/healthz endpoint that actually probes Foundry with the real token.

Gap Analyst #9 / Senior Reviewer top-5: GET / returns 200 even when
FOUNDRY_INTERNAL_TOKEN is wrong. /healthz hits search_knowledge on
startup + every N seconds so misconfig surfaces at deploy time.
"""

from __future__ import annotations

import logging
import time

from fastapi import Response

from chatbot import foundry_client
from chatbot.config import Settings, get_settings
from chatbot.exceptions import FoundryAuthError

logger = logging.getLogger(__name__)

HEALTH_CACHE_SECONDS = 60

_last_probe_ts: float = 0.0
_last_probe_ok: bool = False
_last_probe_error: str | None = None


async def probe(settings: Settings | None = None) -> tuple[bool, str | None]:
    """Issue a cheap canary search against the configured tenant.
    Returns (ok, error_string). Sets module-level cache."""
    global _last_probe_ts, _last_probe_ok, _last_probe_error
    s = settings or get_settings()
    try:
        await foundry_client.search_knowledge(
            tenant_id=s.chatbot_tenant_id,
            query="health_probe",
            max_results=1,
            source_id=s.chatbot_knowledge_source_id,
            settings=s,
        )
        _last_probe_ok = True
        _last_probe_error = None
    except FoundryAuthError as e:
        _last_probe_ok = False
        _last_probe_error = f"auth: {e}"
        logger.error("*** /healthz probe failed: %s ***", _last_probe_error)
    except Exception as e:
        _last_probe_ok = False
        _last_probe_error = f"{type(e).__name__}: {e}"
        logger.warning("/healthz probe failed: %s", _last_probe_error)
    _last_probe_ts = time.monotonic()
    return _last_probe_ok, _last_probe_error


async def healthz_handler() -> Response:
    now = time.monotonic()
    if now - _last_probe_ts > HEALTH_CACHE_SECONDS:
        await probe()
    if _last_probe_ok:
        return Response(
            content='{"status":"ok"}',
            media_type="application/json",
            status_code=200,
        )
    return Response(
        content=f'{{"status":"unhealthy","error":"{_last_probe_error}"}}',
        media_type="application/json",
        status_code=503,
    )


def register_healthz(cl_app) -> None:  # noqa: ANN001 — FastAPI app
    """Mount /healthz on Chainlit's underlying FastAPI app."""
    cl_app.add_api_route(
        "/healthz", healthz_handler, methods=["GET"], include_in_schema=False
    )
