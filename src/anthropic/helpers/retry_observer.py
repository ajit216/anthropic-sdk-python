"""Retry observability hook for Anthropic SDK clients.

Wraps ``messages.create`` (sync and async) to intercept retry attempts and
surface them to user-defined callbacks. Useful for logging, metrics, and
debugging retry storms without modifying SDK internals.

Example::

    import anthropic
    from anthropic.helpers.retry_observer import RetryObserver, RetryEvent

    def on_retry(event: RetryEvent) -> None:
        print(f"Retry #{event.attempt} after {event.elapsed:.2f}s — {event.error}")

    client = anthropic.Anthropic(max_retries=3)
    observer = RetryObserver(client, on_retry=on_retry)

    response = observer.messages_create(
        model="claude-opus-4-5",
        max_tokens=256,
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class RetryEvent:
    """Payload passed to the ``on_retry`` callback on each retry attempt.

    Attributes:
        attempt: Retry number (1-based — the first retry is attempt 1).
        elapsed: Seconds elapsed since the first attempt.
        error: The exception that triggered this retry.
        request_params: A copy of the keyword arguments passed to ``messages.create``.
    """

    attempt: int
    elapsed: float
    error: Exception
    request_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestSummary:
    """Final summary emitted to ``on_complete`` after all attempts finish.

    Attributes:
        total_attempts: Number of attempts made (1 = no retries needed).
        total_elapsed: Total wall-clock seconds across all attempts.
        succeeded: Whether the final attempt succeeded.
        final_error: The last exception if all attempts failed, else ``None``.
    """

    total_attempts: int
    total_elapsed: float
    succeeded: bool
    final_error: Optional[Exception] = None


OnRetryCallback = Callable[[RetryEvent], None]
OnCompleteCallback = Callable[[RequestSummary], None]


class RetryObserver:
    """Observes retry behaviour for ``messages.create`` calls.

    Wraps the underlying client's ``messages.create`` with a retry loop that
    invokes ``on_retry`` on each failure and ``on_complete`` when the request
    either succeeds or exhausts all attempts.

    Args:
        client: A configured ``anthropic.Anthropic`` instance. The client's
            own ``max_retries`` is **not** used — pass ``max_retries`` here.
        on_retry: Callback invoked on every retry attempt. Receives a
            :class:`RetryEvent`. Defaults to a no-op.
        on_complete: Callback invoked once after all attempts. Receives a
            :class:`RequestSummary`. Defaults to a no-op.
        max_retries: Maximum retry attempts (default 3).
        retry_on: Tuple of exception types to retry on. Defaults to
            ``(Exception,)`` — retries on any error.
    """

    def __init__(
        self,
        client: Any,
        *,
        on_retry: Optional[OnRetryCallback] = None,
        on_complete: Optional[OnCompleteCallback] = None,
        max_retries: int = 3,
        retry_on: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self._client = client
        self._on_retry: OnRetryCallback = on_retry or (lambda _: None)
        self._on_complete: OnCompleteCallback = on_complete or (lambda _: None)
        self._max_retries = max_retries
        self._retry_on = retry_on

    def messages_create(self, **params: Any) -> Any:
        """Call ``messages.create`` with retry observation.

        Args:
            **params: Forwarded verbatim to ``client.messages.create``.

        Returns:
            The successful API response.

        Raises:
            The last exception if all retry attempts are exhausted.
        """
        start = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.messages.create(**params)
                summary = RequestSummary(
                    total_attempts=attempt + 1,
                    total_elapsed=time.monotonic() - start,
                    succeeded=True,
                )
                self._on_complete(summary)
                return response
            except self._retry_on as e:
                last_error = e
                if attempt < self._max_retries:
                    event = RetryEvent(
                        attempt=attempt + 1,
                        elapsed=time.monotonic() - start,
                        error=e,
                        request_params=dict(params),
                    )
                    self._on_retry(event)
                    continue
                # All attempts exhausted — surface the error
                summary = RequestSummary(
                    total_attempts=attempt + 1,
                    total_elapsed=time.monotonic() - start,
                    succeeded=False,
                    final_error=e,
                )
                self._on_complete(summary)
                raise e

        # Unreachable but satisfies type checkers
        raise RuntimeError("RetryObserver: unexpected loop exit")  # pragma: no cover

    async def async_messages_create(self, **params: Any) -> Any:
        """Async variant of :meth:`messages_create`.

        Args:
            **params: Forwarded verbatim to ``client.messages.create``.

        Returns:
            The successful API response.

        Raises:
            The last exception if all retry attempts are exhausted.
        """
        import asyncio

        start = time.monotonic()
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.messages.create(**params)
                summary = RequestSummary(
                    total_attempts=attempt + 1,
                    total_elapsed=time.monotonic() - start,
                    succeeded=True,
                )
                self._on_complete(summary)
                return response
            except self._retry_on as e:
                last_error = e
                if attempt < self._max_retries:
                    event = RetryEvent(
                        attempt=attempt + 1,
                        elapsed=time.monotonic() - start,
                        error=e,
                        request_params=dict(params),
                    )
                    self._on_retry(event)
                    await asyncio.sleep(0)  # yield event loop
                    continue
                summary = RequestSummary(
                    total_attempts=attempt + 1,
                    total_elapsed=time.monotonic() - start,
                    succeeded=False,
                    final_error=e,
                )
                self._on_complete(summary)
                raise e

        raise RuntimeError("RetryObserver: unexpected loop exit")  # pragma: no cover

    def __repr__(self) -> str:
        return (
            f"RetryObserver(max_retries={self._max_retries}, "
            f"retry_on={self._retry_on!r})"
        )
