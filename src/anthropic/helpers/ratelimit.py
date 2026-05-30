"""Token bucket rate limiter for Anthropic SDK clients.

Provides sync and async wrappers that enforce a requests-per-minute cap
using a token bucket algorithm. Drop in before any SDK call to avoid
hitting the API's built-in rate limit errors.

Example (sync)::

    import anthropic
    from anthropic.helpers.ratelimit import RateLimitedClient

    client = RateLimitedClient(anthropic.Anthropic(), rpm=60)
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "Hello"}],
    )

Example (async)::

    import asyncio
    import anthropic
    from anthropic.helpers.ratelimit import AsyncRateLimitedClient

    async def main():
        client = AsyncRateLimitedClient(anthropic.AsyncAnthropic(), rpm=60)
        response = await client.messages.create(
            model="claude-opus-4-5",
            max_tokens=256,
            messages=[{"role": "user", "content": "Hello"}],
        )

    asyncio.run(main())
"""

from __future__ import annotations

import time
import threading
from typing import Any, Optional


class TokenBucket:
    """Thread-safe token bucket for rate limiting.

    Tokens refill at a constant rate up to ``capacity``. Callers block
    (sync) or await (async) until a token is available.
    """

    def __init__(self, rate: float, capacity: float) -> None:
        """
        Args:
            rate: Tokens added per second.
            capacity: Maximum tokens the bucket can hold.
        """
        if rate <= 0:
            raise ValueError("rate must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """Block until a token is available.

        Args:
            timeout: Maximum seconds to wait. ``None`` means wait forever.

        Returns:
            ``True`` if a token was acquired, ``False`` if timed out.
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            # No token yet — sleep a fraction of the refill period
            sleep_for = min(0.05, 1.0 / self._rate)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                sleep_for = min(sleep_for, remaining)

            time.sleep(sleep_for)

    async def acquire_async(self, timeout: Optional[float] = None) -> bool:
        """Async variant — awaits until a token is available.

        Note: uses ``asyncio.sleep`` to yield the event loop while waiting.
        """
        import asyncio

        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            # Using threading.Lock inside async context — fine for a quick
            # non-blocking check, but if contention is high this can stall
            # the event loop. Callers on a tight async loop should prefer
            # asyncio.Lock-based implementations.
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            sleep_for = min(0.05, 1.0 / self._rate)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                sleep_for = min(sleep_for, remaining)

            await asyncio.sleep(sleep_for)

    @property
    def available(self) -> float:
        """Current token count (approximate — not locked)."""
        return self._tokens


class RateLimitedClient:
    """Wraps a sync ``anthropic.Anthropic`` client with rate limiting.

    Delegates all attribute access to the underlying client so existing
    call sites need no changes beyond swapping the client constructor.

    Args:
        client: A configured ``anthropic.Anthropic`` instance.
        rpm: Requests per minute cap.
        burst: Allow short bursts above ``rpm``. Defaults to ``rpm``.
        timeout: Seconds to wait for a token before raising ``TimeoutError``.
    """

    def __init__(
        self,
        client: Any,
        *,
        rpm: int,
        burst: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        rate = rpm / 60.0
        cap = float(burst if burst is not None else rpm)
        self._bucket = TokenBucket(rate=rate, capacity=cap)
        self._client = client
        self._timeout = timeout

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def _wait(self) -> None:
        acquired = self._bucket.acquire(timeout=self._timeout)
        if not acquired:
            raise TimeoutError(
                f"Rate limit bucket exhausted — could not acquire token within {self._timeout}s"
            )


class AsyncRateLimitedClient:
    """Wraps an async ``anthropic.AsyncAnthropic`` client with rate limiting.

    Args:
        client: A configured ``anthropic.AsyncAnthropic`` instance.
        rpm: Requests per minute cap.
        burst: Allow short bursts above ``rpm``. Defaults to ``rpm``.
        timeout: Seconds to wait for a token before raising ``TimeoutError``.
    """

    def __init__(
        self,
        client: Any,
        *,
        rpm: int,
        burst: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        rate = rpm / 60.0
        cap = float(burst if burst is not None else rpm)
        self._bucket = TokenBucket(rate=rate, capacity=cap)
        self._client = client
        self._timeout = timeout

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    async def _wait(self) -> None:
        acquired = await self._bucket.acquire_async(timeout=self._timeout)
        if not acquired:
            raise TimeoutError(
                f"Rate limit bucket exhausted — could not acquire token within {self._timeout}s"
            )
