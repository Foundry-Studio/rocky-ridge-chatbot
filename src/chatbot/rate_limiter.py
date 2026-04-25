"""Hand-rolled per-IP sliding-window rate limiter + daily USD circuit breaker.

Why not slowapi? Gap Analyst #10: Chainlit runs Socket.IO; slowapi's
FastAPI decorators don't fire on Socket.IO events. We rate-limit inside
on_message by IP extracted from the Chainlit request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from pathlib import Path

import aiofiles

from chatbot.config import Settings, get_settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding 60-second window per IP, bounded LRU of 10k IPs."""

    MAX_TRACKED_IPS = 10_000
    WINDOW_SECONDS = 60.0

    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self._hits: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()
        self._daily_spend: float = 0.0
        self._daily_reset_ts: float = self._next_midnight_utc()
        self._spend_path = Path(self._s.chatbot_daily_spend_path)
        self._spend_loaded = False

    # ── IP rate limit ────────────────────────────────────────────

    async def check_ip(self, ip: str) -> bool:
        """True if allowed. False if this IP is over quota."""
        async with self._lock:
            now = time.monotonic()
            if ip not in self._hits:
                if len(self._hits) >= self.MAX_TRACKED_IPS:
                    self._evict_oldest_ip(now)
                self._hits[ip] = deque()
            dq = self._hits[ip]
            # Drop stale entries outside window
            while dq and dq[0] < now - self.WINDOW_SECONDS:
                dq.popleft()
            if len(dq) >= self._s.chatbot_rate_limit_per_ip_per_min:
                return False
            dq.append(now)
            return True

    def _evict_oldest_ip(self, now: float) -> None:
        """When the LRU is full, drop IPs whose newest hit is oldest."""
        oldest_ip: str | None = None
        oldest_ts = float("inf")
        for ip, dq in self._hits.items():
            if not dq:
                continue
            if dq[-1] < oldest_ts:
                oldest_ts = dq[-1]
                oldest_ip = ip
        if oldest_ip is not None:
            del self._hits[oldest_ip]

    # ── Daily budget ─────────────────────────────────────────────

    async def load_persisted_spend(self) -> None:
        """Load daily_spend.json on startup. Called once from Chainlit on_start."""
        if self._spend_loaded:
            return
        try:
            if self._spend_path.exists():
                async with aiofiles.open(
                    self._spend_path, mode="r", encoding="utf-8"
                ) as f:
                    raw = await f.read()
                data = json.loads(raw)
                if (
                    isinstance(data, dict)
                    and isinstance(data.get("spent"), int | float)
                    and isinstance(data.get("reset_ts"), int | float)
                ):
                    # Only honor the persisted value if we're still before
                    # the reset timestamp. Otherwise start fresh.
                    if time.time() < data["reset_ts"]:
                        self._daily_spend = float(data["spent"])
                        self._daily_reset_ts = float(data["reset_ts"])
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "daily spend file unreadable, resetting to 0: %s", e
            )
        self._spend_loaded = True

    async def check_budget(self) -> bool:
        """True if under daily cap."""
        async with self._lock:
            self._maybe_reset_daily()
            return self._daily_spend < self._s.chatbot_daily_usd_cap

    async def record_spend(self, usd: float) -> None:
        async with self._lock:
            self._maybe_reset_daily()
            self._daily_spend += max(0.0, usd)
            await self._persist()

    def _maybe_reset_daily(self) -> None:
        now = time.time()
        if now >= self._daily_reset_ts:
            self._daily_spend = 0.0
            self._daily_reset_ts = self._next_midnight_utc()

    @staticmethod
    def _next_midnight_utc() -> float:
        import datetime as _dt

        today_utc = _dt.datetime.now(_dt.UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        tomorrow = today_utc + _dt.timedelta(days=1)
        return tomorrow.timestamp()

    async def _persist(self) -> None:
        try:
            self._spend_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {"spent": self._daily_spend, "reset_ts": self._daily_reset_ts}
            )
            async with aiofiles.open(
                self._spend_path, mode="w", encoding="utf-8"
            ) as f:
                await f.write(payload)
        except OSError as e:
            logger.warning("could not persist daily spend: %s", e)

    @property
    def daily_spend(self) -> float:
        return self._daily_spend
