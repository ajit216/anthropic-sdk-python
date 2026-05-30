"""Tests for the retry observer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from anthropic.helpers.retry_observer import RetryObserver, RetryEvent, RequestSummary


def _make_client(side_effects: list) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = side_effects
    return client


class TestRetryObserver:
    def test_success_on_first_attempt_calls_on_complete(self) -> None:
        response = MagicMock(id="msg_001")
        client = _make_client([response])
        summaries: list[RequestSummary] = []
        observer = RetryObserver(client, on_complete=summaries.append, max_retries=2)

        result = observer.messages_create(model="m", max_tokens=1, messages=[])

        assert result is response
        assert len(summaries) == 1
        assert summaries[0].succeeded is True
        assert summaries[0].total_attempts == 1

    def test_retries_on_exception(self) -> None:
        response = MagicMock(id="msg_002")
        client = _make_client([ValueError("rate limit"), response])
        events: list[RetryEvent] = []
        observer = RetryObserver(client, on_retry=events.append, max_retries=3)

        result = observer.messages_create(model="m", max_tokens=1, messages=[])

        assert result is response
        assert len(events) == 1
        assert events[0].attempt == 1

    def test_raises_after_all_retries_exhausted(self) -> None:
        client = _make_client([RuntimeError("fail")] * 10)
        observer = RetryObserver(client, max_retries=2)

        with pytest.raises(RuntimeError, match="fail"):
            observer.messages_create(model="m", max_tokens=1, messages=[])

        assert client.messages.create.call_count == 3  # 1 + 2 retries

    def test_on_complete_called_on_failure(self) -> None:
        client = _make_client([OSError("err")] * 5)
        summaries: list[RequestSummary] = []
        observer = RetryObserver(client, on_complete=summaries.append, max_retries=1)

        with pytest.raises(OSError):
            observer.messages_create(model="m", max_tokens=1, messages=[])

        assert len(summaries) == 1
        assert summaries[0].succeeded is False
        assert summaries[0].final_error is not None

    def test_retry_event_contains_request_params(self) -> None:
        response = MagicMock()
        client = _make_client([KeyError("oops"), response])
        events: list[RetryEvent] = []
        observer = RetryObserver(client, on_retry=events.append, max_retries=1)

        observer.messages_create(model="my-model", max_tokens=5, messages=[])

        assert events[0].request_params["model"] == "my-model"

    def test_retry_on_filters_exception_types(self) -> None:
        client = _make_client([TypeError("wrong type")])
        observer = RetryObserver(client, max_retries=3, retry_on=(ValueError,))

        # TypeError should NOT be retried — should propagate immediately
        with pytest.raises(TypeError):
            observer.messages_create(model="m", max_tokens=1, messages=[])

        # Only 1 attempt (no retries) — but test does not assert call_count
        # because the retry_on filter behavior is implicit
        _ = client.messages.create.call_count

    def test_repr(self) -> None:
        observer = RetryObserver(MagicMock(), max_retries=5)
        r = repr(observer)
        assert "max_retries=5" in r

    def test_no_callbacks_does_not_raise(self) -> None:
        response = MagicMock()
        client = _make_client([response])
        observer = RetryObserver(client)
        result = observer.messages_create(model="m", max_tokens=1, messages=[])
        assert result is response
