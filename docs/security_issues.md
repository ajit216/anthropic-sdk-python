# Security Issues: anthropic-sdk-python

**Reviewed version:** 0.89.0  
**Review date:** 2025-05-12  
**Reviewer:** Senior Software Developer (automated review)

---

## Summary

This document covers security-related findings from the code review. The SDK handles API credentials (API key, bearer tokens) and communicates with the Anthropic API over HTTPS. No critical vulnerabilities were identified. The issues below are medium-to-low severity concerns around credential management, error information exposure, and resource cleanup.

---

## SEC-001 — API key stored as a plain `str` attribute on the client

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_client.py`, lines 55-56, 295-296 |
| **Category** | Security |
| **Severity** | Low |

**Description:**  
`api_key` and `auth_token` are stored as plain string attributes (`self.api_key`, `self.auth_token`) on the client instance. This means:

1. The key is directly accessible to any code that holds a reference to the client, including third-party libraries.
2. The key will appear in memory dumps and heap snapshots.
3. The key is included by default when the object is serialised (e.g., `vars(client)`, `client.__dict__`).

This is common practice for Python HTTP client libraries, but it is worth noting.

**Recommendation:**  
For environments where in-memory credential protection matters, consider storing credentials in a callable (e.g., `Callable[[], str]`) so they can be managed by a secret manager or fetched on-demand. Add a note to the documentation advising users not to log or serialise client objects.

---

## SEC-002 — Error response body returned verbatim in exception messages

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py`, lines 409-429 |
| **Category** | Security |
| **Severity** | Low |

**Description:**  
`_make_status_error_from_response()` includes the raw server error body in the exception message:

```python
err_msg = f"Error code: {response.status_code} - {body}"
```

If the server echoes back any sensitive data from the request (e.g., a partial API key in an error about a malformed header, internal server paths, or stack traces), that data is surfaced in the exception and potentially in application logs.

**Recommendation:**  
Consider sanitising or truncating the error body before including it in the exception message, particularly stripping any values that look like API keys or tokens. At a minimum, document that `body` may contain sensitive data and advise users to handle exceptions carefully in logging pipelines.

---

## SEC-003 — Authentication header sent regardless of endpoint origin (no hostname pinning)

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_client.py`, lines 155-172, 395-412 |
| **Category** | Security |
| **Severity** | Low |

**Description:**  
The `auth_headers` property unconditionally returns both the `X-Api-Key` and `Authorization: Bearer` headers on every request, regardless of the target hostname. If `base_url` is changed by a caller to a non-Anthropic endpoint (or via `ANTHROPIC_BASE_URL` environment variable), the API key is sent to that third-party server.

This is a standard risk for any configurable base-URL SDK, but it is worth documenting.

**Recommendation:**  
Document clearly that users should only override `base_url` / `ANTHROPIC_BASE_URL` when pointing to a trusted proxy or a corporate gateway, and that credentials will be forwarded to whatever endpoint is configured. Optionally add a warning log when `base_url` differs from `https://api.anthropic.com`.

---

## SEC-004 — `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` environment variable injection

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_client.py`, lines 93, 97, 101 |
| **Category** | Security |
| **Severity** | Low |

**Description:**  
The SDK reads three environment variables (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`) at client construction time with no validation or sanitisation:

```python
api_key = os.environ.get("ANTHROPIC_API_KEY")
auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
base_url = os.environ.get("ANTHROPIC_BASE_URL")
```

In containerised or multi-tenant environments, if an attacker can inject environment variables they can:
- Redirect API traffic to an attacker-controlled server (`ANTHROPIC_BASE_URL`).
- Substitute a compromised API key.

**Recommendation:**  
This is an environment-level concern rather than an SDK defect. Document in the security policy (`SECURITY.md`) that environment variable injection is a platform responsibility. Optionally validate `base_url` against an `https://` scheme requirement and log a warning for non-Anthropic domains.

---

## SEC-005 — Retry-After header value is trusted without strict upper-bound

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py`, lines 747-779, 791 |
| **Category** | Security |
| **Severity** | Low |

**Description:**  
`_parse_retry_after_header()` parses the server-provided `Retry-After` header and returns a float in seconds. The calling code in `_calculate_retry_timeout()` already applies an upper bound check (`if retry_after is not None and 0 < retry_after <= 60`), capping the server-directed sleep at 60 seconds. This is good.

However, the `retry-after-ms` path returns `float(retry_ms_header) / 1000` without any bounds check before the value is returned. The caller's `<= 60` check does apply downstream, so the risk is contained, but the symmetry is imperfect.

**Recommendation:**  
Apply the same `0 < value <= 60` guard inside `_parse_retry_after_header()` to make the function self-contained and easier to audit:

```python
ms_value = float(retry_ms_header) / 1000
if 0 < ms_value <= 60:
    return ms_value
```

---

## SEC-006 — SSE error events may trigger error paths with attacker-influenced body content

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_streaming.py`, lines 105-118, 225-238 |
| **Category** | Security |
| **Severity** | Low |

**Description:**  
When the SSE stream receives an `error` event, the body string is passed directly to `_make_status_error()`:

```python
if sse.event == "error":
    body = sse.data
    try:
        body = sse.json()
        err_msg = f"{body}"
    except Exception:
        err_msg = sse.data or f"Error code: {response.status_code}"

    raise self._client._make_status_error(err_msg, body=body, response=self.response)
```

If a man-in-the-middle (or a compromised intermediate proxy) injects a crafted SSE `error` event with a large body, this will be parsed and included verbatim in the exception. While this does not constitute remote code execution, it may be used for log injection or to produce misleading errors.

**Recommendation:**  
Truncate `err_msg` to a reasonable length (e.g., 2 KB) before including it in the exception:

```python
err_msg = err_msg[:2048] if len(err_msg) > 2048 else err_msg
```

---

## SEC-007 — `SyncHttpxClientWrapper.__del__` silently ignores close errors

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py`, lines 905-913 |
| **Category** | Security |
| **Severity** | Low |

**Description:**  
```python
class SyncHttpxClientWrapper(DefaultHttpxClient):
    def __del__(self) -> None:
        if self.is_closed:
            return
        try:
            self.close()
        except Exception:
            pass
```

If `close()` raises (e.g., due to a partial write in a request being retried), the exception is silently swallowed. While not a direct security issue, unclosed connections in some environments can lead to stale authenticated sessions remaining open longer than expected.

**Recommendation:**  
Log close failures at DEBUG level, and consider emitting a `ResourceWarning` for unclosed clients (consistent with stdlib behaviour for file and socket objects).

---

*For general code quality and bug findings, see [docs/code_review.md](./code_review.md).*
