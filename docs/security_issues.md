# Security Issues: anthropic-sdk-python

**Review Date:** 2026-05-02  
**Reviewer:** Automated Senior Python Developer Review  
**Scope:** Security-focused review of `src/anthropic/`

---

## Summary

This document collects all security-relevant findings from the full code review. Three issues were identified, ranging from high severity (information disclosure in debug logs) to medium severity (cached short-lived credentials).

---

## SEC-1: Debug Logging Exposes Full Response Headers and Error Bodies

- **Location:** `src/anthropic/_base_client.py`
  - `SyncAPIClient.request`, lines 1112–1120
  - `AsyncAPIClient.request`, lines 1752–1760
  - `BaseClient._make_status_error_from_response`, lines 409–429
- **Category:** Security
- **Severity:** High

### Description

When the `ANTHROPIC_LOG=debug` environment variable is set (or the `anthropic` logger is configured at `DEBUG` level), the SDK writes the complete HTTP response headers dict to the log:

```python
log.debug(
    'HTTP Response: %s %s "%i %s" %s',
    request.method,
    request.url,
    response.status_code,
    response.reason_phrase,
    response.headers,   # <— ALL headers, including Set-Cookie, auth tokens, etc.
)
```

HTTP response headers may contain:
- `Set-Cookie` headers with session tokens
- Custom authorisation tokens or bearer tokens returned by the API
- Internal routing metadata that could assist an attacker

Additionally, `_make_status_error_from_response` formats the **full response body** into the exception message:

```python
body = json.loads(err_text)
err_msg = f"Error code: {response.status_code} - {body}"
```

Error bodies from the Anthropic API may contain detailed account information, input echo, or PII from the request payload that a user inadvertently provided.

### Risk

- If logs are shipped to a centralised log aggregation system (e.g., Splunk, ELK, Datadog), any operator or developer with log read access can harvest authentication tokens or user data.
- Applications that propagate the exception message string to external systems (e.g., error tracking tools) may inadvertently expose sensitive data to third parties.

### Recommendation

1. **Filter headers before logging.** Only log a safe allowlist such as `content-type`, `x-request-id`, `x-ratelimit-*`, `x-stainless-*`:

   ```python
   SAFE_LOG_HEADERS = {"content-type", "x-request-id", "x-ratelimit-requests-limit",
                       "x-ratelimit-tokens-limit", "x-ratelimit-requests-remaining",
                       "x-ratelimit-tokens-remaining"}

   safe_headers = {k: v for k, v in response.headers.items()
                   if k.lower() in SAFE_LOG_HEADERS}
   log.debug('HTTP Response: %s %s "%i %s" %s', ..., safe_headers)
   ```

2. **Truncate or omit the error body** from `err_msg`. Keep it only in the `body` attribute of the raised exception (which callers can inspect at their discretion, but which won't be emitted to logs automatically).

---

## SEC-2: AWS Credentials Cached Indefinitely via `lru_cache`

- **Location:**
  - `src/anthropic/lib/bedrock/_auth.py`, lines 13–30
  - `src/anthropic/lib/aws/_auth.py`, lines 13–30
- **Category:** Security
- **Severity:** Medium

### Description

Both Bedrock and AWS auth modules decorate `_get_session` with `@lru_cache(maxsize=512)`:

```python
@lru_cache(maxsize=512)
def _get_session(
    *,
    aws_access_key: str | None,
    aws_secret_key: str | None,
    aws_session_token: str | None,   # <— STS tokens have short TTLs (default 1 hour)
    region: str | None,
    profile: str | None,
) -> boto3.Session:
    import boto3
    return boto3.Session(...)
```

`lru_cache` caches by the exact argument tuple. If a caller passes a short-lived STS `aws_session_token`, the resulting `Session` is cached and returned on subsequent calls even after the token has expired. This causes:

1. **Authentication failures** once the STS token expires, with the SDK silently continuing to use the stale session.
2. **Security confusion** — a developer rotating credentials may not realise that in-process caching is preventing the new credentials from taking effect without a process restart.

### Recommendation

- Replace `lru_cache` with a TTL-aware cache (e.g., `cachetools.TTLCache` with a 15-minute TTL):

  ```python
  from cachetools import TTLCache, cached

  _session_cache: TTLCache = TTLCache(maxsize=512, ttl=900)  # 15 minutes

  @cached(_session_cache)
  def _get_session(...) -> boto3.Session:
      ...
  ```

- Alternatively, do not cache sessions with non-`None` `aws_session_token` values since these are inherently time-limited.

---

## SEC-3: API Key Stored as a Plain String on the Client Object

- **Location:** `src/anthropic/_client.py`
  - `Anthropic.__init__`, line 94: `self.api_key = api_key`
  - `AsyncAnthropic.__init__`, line 334: `self.api_key = api_key`
- **Category:** Security
- **Severity:** Low

### Description

The API key is stored as a plain `str` attribute on the public `api_key` property of the client. This means:

1. **Heap dumps and memory inspections** (e.g., via `gc.get_objects()` or `tracemalloc` snapshots) can reveal the API key.
2. **Serialisation libraries** (e.g., `pickle`, `copy`, object introspection tools) will include the key in serialised representations.
3. **Logging of the client object** (e.g., `log.debug("client: %r", client)`) would print the key in clear text if a `__repr__` were ever added that included instance attributes.

The `auth_token` attribute (`self.auth_token`) is subject to the same concern.

### Recommendation

- Override `__repr__` and `__str__` on both `Anthropic` and `AsyncAnthropic` to mask the key:

  ```python
  def __repr__(self) -> str:
      masked = f"...{self.api_key[-4:]}" if self.api_key else "<not set>"
      return f"Anthropic(api_key={masked!r})"
  ```

- Consider wrapping the key in a simple container class that overrides `__repr__` and `__str__` to always return a masked representation, making accidental disclosure harder.

---

## Checklist of Security Controls

| Control | Status |
|---------|--------|
| API key not logged at DEBUG level | ✅ Outgoing request headers are not logged (only response headers are) |
| API key not included in `__repr__` | ⚠️ No `__repr__` defined — currently safe but fragile |
| STS token rotation respected | ❌ `lru_cache` prevents rotation without process restart |
| Response headers filtered before logging | ❌ Full header dict is logged at DEBUG |
| Error bodies filtered before logging | ❌ Full body included in `err_msg` |
| HTTPS enforced for all API calls | ✅ Default base URL uses HTTPS |
| Certificate verification enabled by default | ✅ httpx defaults to verifying certificates |

---

*Full details of non-security issues are in [`docs/code_review.md`](./code_review.md).*
