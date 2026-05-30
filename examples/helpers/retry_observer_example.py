"""Example: retry observer for logging and metrics."""

import anthropic
from anthropic.helpers.retry_observer import RetryObserver, RetryEvent, RequestSummary


def on_retry(event: RetryEvent) -> None:
    print(f"[retry] attempt={event.attempt} elapsed={event.elapsed:.2f}s error={event.error!r}")


def on_complete(summary: RequestSummary) -> None:
    status = "ok" if summary.succeeded else "failed"
    print(f"[done] attempts={summary.total_attempts} elapsed={summary.total_elapsed:.2f}s status={status}")


client = anthropic.Anthropic()
observer = RetryObserver(client, on_retry=on_retry, on_complete=on_complete, max_retries=3)

response = observer.messages_create(
    model="claude-opus-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello, Claude!"}],
)
print(response.content[0].text)
