"""Rate limiter + daily budget tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from chatbot.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_ip_within_limit_passes(settings, tmp_log_path: Path):
    limiter = RateLimiter(settings)
    for _ in range(settings.chatbot_rate_limit_per_ip_per_min):
        assert await limiter.check_ip("ip-a") is True


@pytest.mark.asyncio
async def test_ip_over_limit_blocks(settings, tmp_log_path: Path):
    limiter = RateLimiter(settings)
    for _ in range(settings.chatbot_rate_limit_per_ip_per_min):
        await limiter.check_ip("ip-b")
    assert await limiter.check_ip("ip-b") is False


@pytest.mark.asyncio
async def test_ips_isolated(settings, tmp_log_path: Path):
    limiter = RateLimiter(settings)
    for _ in range(settings.chatbot_rate_limit_per_ip_per_min):
        await limiter.check_ip("ip-c")
    # Different IP, fresh counter
    assert await limiter.check_ip("ip-d") is True


@pytest.mark.asyncio
async def test_budget_initially_under_cap(settings, tmp_log_path: Path):
    limiter = RateLimiter(settings)
    assert await limiter.check_budget() is True


@pytest.mark.asyncio
async def test_record_spend_accumulates(settings, tmp_log_path: Path):
    limiter = RateLimiter(settings)
    await limiter.record_spend(5.0)
    await limiter.record_spend(10.0)
    assert limiter.daily_spend == pytest.approx(15.0)


@pytest.mark.asyncio
async def test_budget_hits_cap(settings, tmp_log_path: Path):
    limiter = RateLimiter(settings)
    await limiter.record_spend(settings.chatbot_daily_usd_cap + 1.0)
    assert await limiter.check_budget() is False


@pytest.mark.asyncio
async def test_spend_persists_to_file(settings, tmp_log_path: Path):
    limiter = RateLimiter(settings)
    await limiter.record_spend(3.5)
    path = Path(settings.chatbot_daily_spend_path)
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["spent"] == pytest.approx(3.5)


@pytest.mark.asyncio
async def test_load_persisted_spend(settings, tmp_log_path: Path):
    limiter = RateLimiter(settings)
    await limiter.record_spend(7.0)

    # Fresh limiter should pick up persisted value.
    limiter2 = RateLimiter(settings)
    await limiter2.load_persisted_spend()
    assert limiter2.daily_spend == pytest.approx(7.0)


@pytest.mark.asyncio
async def test_load_corrupt_file_resets_to_zero(settings, tmp_log_path: Path):
    Path(settings.chatbot_daily_spend_path).write_text("not json", encoding="utf-8")
    limiter = RateLimiter(settings)
    await limiter.load_persisted_spend()
    assert limiter.daily_spend == 0.0
