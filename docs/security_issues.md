# Security Issues — anthropic-sdk-python

**Reviewed by:** Senior Software Developer (automated review, 2026-05-12)  
**Related document:** [`docs/code_review.md`](./code_review.md)

---

## Summary

Three security-relevant issues have been identified. None are exploitable in isolation under normal usage, but each represents a risk in production deployments where logs may be aggregated, memory dumps may be captured, or credentials may be rotated on a schedule.

---

## Issue S1 — API Keys Logged at DEBUG Level via `httpx` Request Headers

- **Location:** `src/anthropic/_base_client.py`, lines 497–509 and line 1119
- **Category:** Security
- **Severity:** High

**Description:**  
When the `ANTHROPIC_LOG=debug` environment variable is set (see `src/anthropic/_utils/_logs.py`), both the `anthropic` logger and the `httpx` logger are set to `DEBUG`. The `httpx` logger at DEBUG level logs full request headers, which include the `X-Api-Key` or `Authorization: Bearer <token>` headers set in `auth_headers`.

Additionally, `_base_client.py` line 1119 explicitly logs all HTTP response headers at DEBUG:

```python
log.debug(
    'HTTP Response: %s %s "%i %s" %s',
    request.method,
    request.url,
    response.status_code,
    response.reason_phrase,
    response.headers,           # ← full header dict, can include Set-Cookie / auth echoes
)
```

In a production environment where `ANTHROPIC_LOG=debug` is set and logs are shipped to a centralised log management platform, API keys will be transmitted in plain text to the log aggregator.

**Recommendation:**  
1. Redact authentication headers before logging. For example:
   ```python
   safe_headers = {k: ("***" if k.lower() in ("x-api-key", "authorization") else v)
                   for k, v in response.headers.items()}
   log.debug('HTTP Response: %s %s "%i %s" %s', ..., safe_headers)
   ```
2. Consider adding a warning in `_logs.py` `setup_logging()` that debug mode logs sensitive headers.
3. Evaluate suppressing `httpx_logger` at DEBUG to `INFO` in `setup_logging` to avoid httpx's own request-header logging.

---

## Issue S2 — AWS Credentials Cached in `lru_cache` as Plain-Text Cache Keys

- **Location:** `src/anthropic/lib/aws/_auth.py`, lines 13–30
- **Category:** Security
- **Severity:** High

**Description:**  
`_get_session` is decorated with `@lru_cache(maxsize=512)`. The cache key is a tuple of `(aws_access_key, aws_secret_key, aws_session_token, region, profile)`. This means:

1. **Long-lived secrets in memory:** Credentials remain in the LRU cache for the lifetime of the process (or until evicted by reaching 512 entries). Temporary STS session tokens that have been rotated remain in the cache — new requests may use a stale/expired token rather than refreshing credentials.
2. **Secret key in cache key:** The raw `aws_secret_key` string is stored as a dictionary key in the LRU cache's internal structure, making it accessible via `_get_session.cache_info()` or memory inspection.

**Recommendation:**  
1. **Remove `lru_cache` from `_get_session`** or replace with a non-key-exposing pattern. If session reuse is desired for performance, cache by a non-secret identifier (e.g., profile name + region) and keep credentials out of the cache key.
2. **Alternatively**, accept a pre-constructed `boto3.Session` from callers so the SDK is not responsible for session lifecycle.
3. Consider calling `credentials.get_frozen_credentials()` inside `get_auth_headers` so that credential refreshes are honoured on each request.

---

## Issue S3 — `api_key` and `auth_token` Are Publicly Readable Attributes

- **Location:** `src/anthropic/_client.py`, lines 55–56 and lines 294–295
- **Category:** Security
- **Severity:** Medium

**Description:**  
Both `Anthropic` and `AsyncAnthropic` expose credentials as plain public attributes:

```python
class Anthropic(SyncAPIClient):
    api_key: str | None      # publicly readable
    auth_token: str | None   # publicly readable
```

Any code (including third-party libraries, monkey-patching code, or debugging tools) that has a reference to the client can trivially read `client.api_key`. While this is intentional for `copy()` / `with_options()` usability, it creates a risk in environments where the client object may be serialised, logged, or passed through untrusted code paths.

Additionally, neither `__repr__` nor `__str__` is overridden on `Anthropic`/`AsyncAnthropic`, so if the default pydantic/object repr is invoked the full class dict could be printed.

**Recommendation:**  
1. Override `__repr__` to redact credentials:
   ```python
   def __repr__(self) -> str:
       return (f"{self.__class__.__name__}("
               f"api_key={'[redacted]' if self.api_key else 'None'}, "
               f"base_url={self.base_url!r})")
   ```
2. Document that `api_key` is a sensitive attribute and advise users not to pass client objects through untrusted code.
3. (Optional) Consider a `SecretStr` wrapper (pydantic provides one) to prevent accidental printing.

---

## Issue S4 — No Validation That `base_url` Uses HTTPS in Production Contexts

- **Location:** `src/anthropic/_client.py`, lines 100–103 and `src/anthropic/lib/bedrock/_client.py`, `src/anthropic/lib/vertex/_client.py`
- **Category:** Security
- **Severity:** Low

**Description:**  
The `base_url` parameter is accepted without any scheme validation. A user who accidentally sets `ANTHROPIC_BASE_URL=http://api.anthropic.com` (HTTP instead of HTTPS) would have all API calls — including API keys in headers — transmitted in plain text with no warning.

**Recommendation:**  
Add a warning (or optionally an error) when `base_url` uses a non-HTTPS scheme:

```python
if URL(base_url).scheme not in ("https", "http+unix"):
    import warnings
    warnings.warn(
        f"base_url '{base_url}' does not use HTTPS. API keys will be transmitted in plain text.",
        UserWarning,
        stacklevel=2,
    )
```

Allowing `http://` for local testing/proxies is reasonable but should be a deliberate opt-in with a visible warning.

---

## Non-Issues (Investigated and Dismissed)

- **SSRF via user-supplied `base_url`:** The SDK is a client library, not a server. The caller already controls the outbound URL and has the credentials. No SSRF risk.
- **Deserialization attacks via Pydantic:** Models use `construct_type` / `validate_type` from trusted API responses. Pydantic validation provides adequate protection against unexpected shapes.
- **Injection via `tool_use.input`:** Tool inputs are passed directly to registered tool functions. The SDK correctly documents that callers are responsible for safe handling of tool input — no injection vector exists within the SDK itself.
