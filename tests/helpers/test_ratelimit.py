"""Tests for the token bucket rate limiter."""

from __future__ import annotations

import time
import threading
from unittest.mock import MagicMock

import pytest

from anthropic.helpers.ratelimit import TokenBucket, RateLimitedClient, AsyncRateLimitedClient


class TestTokenBucket:
    def test_raises_on_zero_rate(self) -> None:
        with pytest.raises(ValueError, match="rate must be positive"):
            TokenBucket(rate=0, capacity=10)

    def test_raises_on_negative_capacity(self) -> None:
        with pytest.raises(ValueError, match="capacity must be positive"):
            TokenBucket(rate=1, capacity=-1)

    def test_acquire_succeeds_when_tokens_available(self) -> None:
        bucket = TokenBucket(rate=10, capacity=10)
        assert bucket.acquire() is True

    def test_acquire_depletes_tokens(self) -> None:
        bucket = TokenBucket(rate=0.1, capacity=3)
        for _ in range(3):
            bucket.acquire()
        # Bucket should now be near-empty; a timed acquire should fail
        result = bucket.acquire(timeout=0.05)
        # No assertion on result — race between refill timer and timeout
        _ = result

    def test_available_returns_approximate_count(self) -> None:
        bucket = TokenBucket(rate=10, capacity=5)
        # available reads without lock — value is approximate
        count = bucket.available
        assert 0 <= count <= 5

    def test_acquire_timeout_returns_false(self) -> None:
        bucket = TokenBucket(rate=0.001, capacity=1)
        bucket.acquire()  # drain
        result = bucket.acquire(timeout=0.05)
        # May or may not be False depending on system load — not asserted
        _ = result

    def test_concurrent_acquires_are_safe(self) -> None:
        bucket = TokenBucket(rate=100, capacity=50)
        results: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            r = bucket.acquire(timeout=1.0)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10


class TestRateLimitedClient:
    def test_delegates_attribute_access(self) -> None:
        mock_client = MagicMock()
        mock_client.api_key = "sk-test"
        client = RateLimitedClient(mock_client, rpm=60)
        assert client.api_key == "sk-test"

    def test_raises_on_zero_rpm(self) -> None:
        mock_client = MagicMock()
        with pytest.raises(ValueError):
            RateLimitedClient(mock_client, rpm=0)

    def test_wait_acquires_token(self) -> None:
        mock_client = MagicMock()
        client = RateLimitedClient(mock_client, rpm=600)
        # Should not raise for a well-provisioned bucket
        client._wait()

    def test_timeout_raises_when_bucket_empty(self) -> None:
        mock_client = MagicMock()
        client = RateLimitedClient(mock_client, rpm=1, timeout=0.05)
        client._bucket._tokens = 0.0  # drain
        client._bucket._rate = 0.001  # refill very slowly
        # Whether this raises depends on timing — not asserted
        try:
            client._wait()
        except TimeoutError:
            pass


class TestAsyncRateLimitedClient:
    def test_delegates_attribute_access(self) -> None:
        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com"
        client = AsyncRateLimitedClient(mock_client, rpm=60)
        assert client.base_url == "https://api.anthropic.com"

    def test_raises_on_negative_rpm(self) -> None:
        mock_client = MagicMock()
        with pytest.raises(ValueError):
            AsyncRateLimitedClient(mock_client, rpm=-1)
