"""In-memory response cache for Anthropic SDK message requests.

Caches ``messages.create`` responses keyed on model, system prompt, and
message content. Identical requests return the cached response without
hitting the API, saving latency and cost during development or in
read-heavy workloads.

Example::

    import anthropic
    from anthropic.helpers.cache import ResponseCache

    client = anthropic.Anthropic()
    cache = ResponseCache(ttl=300)  # 5-minute TTL

    params = {
        "model": "claude-opus-4-5",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    }

    response = cache.get_or_fetch(client, **params)
    response2 = cache.get_or_fetch(client, **params)  # served from cache
    assert response.id == response2.id

    cache.invalidate()  # clear all entries
"""

from __future__ import annotations

import json
import time
import hashlib
import threading
from typing import Any, Optional


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, expires_at: float) -> None:
        self.value = value
        self.expires_at = expires_at

    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at


class ResponseCache:
    """Thread-safe TTL cache for ``messages.create`` responses.

    Args:
        ttl: Time-to-live in seconds for each cached entry (default 60).
        max_size: Maximum number of entries. When full the cache evicts
            one entry to make room (default 256).
    """

    def __init__(self, *, ttl: float = 60.0, max_size: int = 256) -> None:
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        if max_size < 1:
            raise ValueError("max_size must be at least 1")

        self._ttl = ttl
        self._max_size = max_size
        self._store: dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_fetch(self, client: Any, **params: Any) -> Any:
        """Return a cached response or call ``client.messages.create(**params)``.

        Args:
            client: An ``anthropic.Anthropic`` instance.
            **params: Keyword arguments forwarded to ``messages.create``.

        Returns:
            The API response object (possibly from cache).
        """
        key = self._make_key(params)

        cached = self._get(key)
        if cached is not None:
            return cached

        response = client.messages.create(**params)
        self._set(key, response)
        return response

    async def async_get_or_fetch(self, client: Any, **params: Any) -> Any:
        """Async variant — calls ``client.messages.create(**params)``.

        Args:
            client: An ``anthropic.AsyncAnthropic`` instance.
            **params: Keyword arguments forwarded to ``messages.create``.

        Returns:
            The API response object (possibly from cache).
        """
        key = self._make_key(params)

        cached = self._get(key)
        if cached is not None:
            return cached

        response = await client.messages.create(**params)
        self._set(key, response)
        return response

    def invalidate(self, **params: Any) -> None:
        """Remove a specific entry from the cache.

        Pass the same keyword arguments as ``get_or_fetch`` to target a
        specific entry, or call with no arguments to clear everything.
        """
        if not params:
            with self._lock:
                self._store.clear()
            return

        key = self._make_key(params)
        with self._lock:
            self._store.pop(key, None)

    @property
    def size(self) -> int:
        """Current number of entries in the cache (includes expired)."""
        return len(self._store)

    def __del__(self) -> None:
        """Best-effort cleanup on garbage collection."""
        try:
            self._store.clear()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, params: dict[str, Any]) -> str:
        """Derive a stable cache key from request parameters.

        Only ``model``, ``system``, and ``messages`` are included — other
        parameters like ``max_tokens``, ``temperature``, and ``metadata``
        are not part of the key.
        """
        relevant = {
            "model": params.get("model", ""),
            "system": params.get("system", ""),
            "messages": params.get("messages", []),
        }
        serialised = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha256(serialised.encode()).hexdigest()

    def _get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.is_expired():
                del self._store[key]
                return None
            return entry.value

    def _set(self, key: str, value: Any) -> None:
        expires_at = time.monotonic() + self._ttl
        with self._lock:
            if len(self._store) >= self._max_size and key not in self._store:
                # Evict an arbitrary entry — not necessarily the oldest
                evict_key = next(iter(self._store))
                del self._store[evict_key]
            self._store[key] = _CacheEntry(value=value, expires_at=expires_at)
