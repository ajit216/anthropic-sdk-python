# Security Issues: anthropic-sdk-python

**Review Date:** 2026-05-02  
**Reviewer:** Automated Senior Python Developer Review  
**Codebase Version:** 0.89.0

---

## Summary

Three security-relevant findings were identified. None are critical vulnerabilities that would allow direct exploitation, but two involve credential handling concerns and one involves information disclosure risk via logging.

---

## Findings

### SEC-001 — AWS credentials cached indefinitely in LRU cache (credential rotation ignored)

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/lib/aws/_auth.py`, lines 13–30 |
| **Category** | Security |
| **Severity** | High |

**Description:**  
The `_get_session()` function uses `@lru_cache(maxsize=512)` with AWS credentials as cache keys:

```python
@lru_cache(maxsize=512)
def _get_session(
    *,
    aws_access_key: str | None,
    aws_secret_key: str | None,
    aws_session_token: str | None,
    region: str | None,
    profile: str | None,
) -> boto3.Session:
    ...
```

There are two problems:

1. **Expired session tokens are never refreshed.** If a caller uses temporary credentials (e.g., via STS `AssumeRole`), the `aws_session_token` will expire. A cached `boto3.Session` built with an expired token will continue to be returned, causing silent authentication failures instead of refreshing.

2. **Sensitive credentials are held as cache keys indefinitely.** The LRU cache keeps strong references to the key arguments (including plaintext `aws_secret_key` and `aws_session_token`) for the lifetime of the process. In environments where credentials are rotated, old credentials persist in memory even after they are no longer valid.

**Recommendation:**  
- Remove the `lru_cache` from `_get_session`, or add a TTL-based invalidation mechanism.
- For temporary credentials, always create a new session or implement explicit expiry checking by inspecting `credentials.get_frozen_credentials()` before returning a cached session.
- Consider using `boto3.Session().get_credentials().refresh_needed()` to detect stale credentials.

---

### SEC-002 — API key exposed in `debug`-level logs via `User-Agent` and `auth_headers`

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py`, lines 496–509 (request debug log), `src/anthropic/_utils/_logs.py` |
| **Category** | Security |
| **Severity** | Medium |

**Description:**  
When `ANTHROPIC_LOG=debug` is set, the SDK logs full request options and HTTP headers via `log.debug()`. The `default_headers` property includes `auth_headers`, which includes the `X-Api-Key` (API key) and `Authorization` (bearer token):

```python
@property
def default_headers(self) -> dict[str, str | Omit]:
    return {
        ...
        **self.auth_headers,   # Contains X-Api-Key or Authorization header
        ...
    }
```

The debug-level request log at line 498–509 dumps `model_dump(options, ...)`, and the HTTP response log at line 1112–1120 logs `response.headers`. Depending on what headers are echoed back, the API key could appear in logs.

**Recommendation:**  
- Redact or mask auth headers before logging. For example, replace the value of `X-Api-Key` and `Authorization` headers in the logged copy with `"[REDACTED]"`.
- Add a helper such as `_redact_sensitive_headers(headers)` applied before any debug log statements that include request headers.

---

### SEC-003 — Error messages may expose internal API response body content to end users

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py`, lines 420–428 (`_make_status_error_from_response`) |
| **Category** | Security |
| **Severity** | Low |

**Description:**  
When a non-2xx HTTP response is received, the full response body is decoded and included in the exception message:

```python
err_text = response.text.strip()
body = err_text

try:
    body = json.loads(err_text)
    err_msg = f"Error code: {response.status_code} - {body}"
except Exception:
    err_msg = err_text or f"Error code: {response.status_code}"

return self._make_status_error(err_msg, body=body, response=response)
```

In production applications that surface exception messages to end users (e.g., via exception handlers that render `str(e)`), this would expose the full API error body, which may contain internal implementation details from Anthropic's infrastructure.

**Recommendation:**  
This is acceptable for an SDK where callers are expected to be developers. However, the documentation should clearly note that `err.message` and `str(err)` may contain sensitive server error information, and applications should avoid surfacing raw exception messages to end users.

---
