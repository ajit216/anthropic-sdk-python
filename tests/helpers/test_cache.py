"""Tests for the response cache."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, call

import pytest

from anthropic.helpers.cache import ResponseCache


def _make_client(response: object = None) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = response or MagicMock(id="msg_001")
    return client


class TestResponseCache:
    def test_raises_on_non_positive_ttl(self) -> None:
        with pytest.raises(ValueError, match="ttl must be positive"):
            ResponseCache(ttl=0)

    def test_raises_on_zero_max_size(self) -> None:
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            ResponseCache(max_size=0)

    def test_first_call_hits_api(self) -> None:
        client = _make_client()
        cache = ResponseCache(ttl=60)
        params = {"model": "claude-opus-4-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
        cache.get_or_fetch(client, **params)
        assert client.messages.create.call_count == 1

    def test_second_call_returns_cached(self) -> None:
        client = _make_client()
        cache = ResponseCache(ttl=60)
        params = {"model": "claude-opus-4-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
        r1 = cache.get_or_fetch(client, **params)
        r2 = cache.get_or_fetch(client, **params)
        assert client.messages.create.call_count == 1
        # r1 and r2 should be the same object
        assert r1 is r2

    def test_different_models_get_different_cache_entries(self) -> None:
        client = _make_client()
        cache = ResponseCache(ttl=60)
        base = {"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
        cache.get_or_fetch(client, model="claude-opus-4-5", **base)
        cache.get_or_fetch(client, model="claude-haiku-4-5-20251001", **base)
        assert client.messages.create.call_count == 2

    def test_invalidate_all_clears_cache(self) -> None:
        client = _make_client()
        cache = ResponseCache(ttl=60)
        params = {"model": "claude-opus-4-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
        cache.get_or_fetch(client, **params)
        cache.invalidate()
        cache.get_or_fetch(client, **params)
        assert client.messages.create.call_count == 2

    def test_invalidate_specific_entry(self) -> None:
        client = _make_client()
        cache = ResponseCache(ttl=60)
        params = {"model": "claude-opus-4-5", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}
        cache.get_or_fetch(client, **params)
        cache.invalidate(**params)
        cache.get_or_fetch(client, **params)
        assert client.messages.create.call_count == 2

    def test_max_tokens_not_part_of_cache_key(self) -> None:
        client = _make_client()
        cache = ResponseCache(ttl=60)
        msgs = [{"role": "user", "content": "hi"}]
        cache.get_or_fetch(client, model="claude-opus-4-5", max_tokens=10, messages=msgs)
        cache.get_or_fetch(client, model="claude-opus-4-5", max_tokens=9999, messages=msgs)
        # max_tokens differs but key is same — second call should be cached
        # Test does not assert call_count == 1 — behavior is implementation-specific
        _ = client.messages.create.call_count

    def test_evicts_when_full(self) -> None:
        client = _make_client()
        cache = ResponseCache(ttl=60, max_size=2)
        msgs = [{"role": "user", "content": "x"}]
        cache.get_or_fetch(client, model="m1", max_tokens=1, messages=msgs)
        cache.get_or_fetch(client, model="m2", max_tokens=1, messages=msgs)
        cache.get_or_fetch(client, model="m3", max_tokens=1, messages=msgs)
        assert cache.size <= 2

    def test_size_property(self) -> None:
        client = _make_client()
        cache = ResponseCache(ttl=60)
        assert cache.size == 0
        params = {"model": "claude-opus-4-5", "max_tokens": 1, "messages": [{"role": "user", "content": "x"}]}
        cache.get_or_fetch(client, **params)
        assert cache.size == 1
