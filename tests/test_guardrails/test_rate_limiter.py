"""Tests for faithcheck.guardrails.rate_limiter."""
import asyncio

import pytest

from faithcheck.guardrails.rate_limiter import RateLimiter


@pytest.mark.asyncio
class TestRateLimiter:
    async def test_acquire_within_limit(self):
        limiter = RateLimiter(max_requests_per_minute=60)
        await limiter.acquire()

    async def test_acquire_tracks_requests(self):
        limiter = RateLimiter(max_requests_per_minute=10)
        for _ in range(10):
            await limiter.acquire()
        assert limiter.requests_in_window == 10

    async def test_reset_on_window_expiry(self):
        limiter = RateLimiter(max_requests_per_minute=10, window_seconds=0.1)
        for _ in range(10):
            await limiter.acquire()
        await asyncio.sleep(0.15)
        assert limiter.requests_in_window == 0
