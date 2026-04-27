"""Token-bucket rate limiter for API calls."""
from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_requests_per_minute: int, window_seconds: float = 60.0) -> None:
        self._max = max_requests_per_minute
        self._window = window_seconds
        self._timestamps: list[float] = []

    @property
    def requests_in_window(self) -> int:
        self._evict_old()
        return len(self._timestamps)

    async def acquire(self) -> None:
        """Wait until a request slot is available, then record it."""
        while True:
            self._evict_old()
            if len(self._timestamps) < self._max:
                self._timestamps.append(time.monotonic())
                return
            await asyncio.sleep(0.1)

    def _evict_old(self) -> None:
        cutoff = time.monotonic() - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]
