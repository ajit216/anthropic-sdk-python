# Security Review — anthropic-sdk-python

**Reviewed version:** 0.89.0  
**Review date:** 2026-05-04  
**Scope:** Full codebase under `src/anthropic/`  

> **Note:** No critical vulnerabilities were found. This document records lower-severity observations that are worth tracking for defence-in-depth.

---

## SEC-001 — API key can appear in DEBUG-level logs via request options dump

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_base_client.py`, lines 496–509 |
| **Severity** | Medium |
| **Category** | Security |

**Description**  
When `ANTHROPIC_LOG=debug` is set, `_build_request` serialises the entire `FinalRequestOptions` object and writes it to the log:

```python
if log.isEnabledFor(logging.DEBUG):
    log.debug(
        "Request options: %s",
        model_dump(
            options,
            exclude_unset=True,
            exclude={"content"} if PYDANTIC_V1 else {},
        ),
    )
```

`FinalRequestOptions` contains `headers`, which at this point already includes the merged `default_headers` (via `_build_headers`) **and any custom headers** — but crucially `_build_request` is called *before* `_build_headers` so at this stage `options.headers` still holds whatever the caller supplied, which may contain `X-Api-Key` or `Authorization` values set as custom headers.

More directly, `options.json_data` (the full request payload) is also serialised here. For endpoints that accept credentials or sensitive data in the body (e.g. third-party OAuth flows), this would log them in plain text.

**Risk**  
Log files written to disk, shipped to a SIEM, or printed to a CI/CD console may contain API keys or sensitive payloads.

**Recommendation**  
1. Explicitly exclude `headers` from the debug dump:

```python
model_dump(
    options,
    exclude_unset=True,
    exclude={"content", "headers"} if PYDANTIC_V1 else {"headers"},
)
```

2. Consider scrubbing known sensitive header names (`X-Api-Key`, `Authorization`) from any log output rather than a blanket exclusion of all headers.

---

## SEC-002 — Full HTTP response headers logged at DEBUG level

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_base_client.py`, lines 1112–1119 (sync) and the analogous async path around line 1760 |
| **Severity** | Low |
| **Category** | Security |

**Description**  
Both sync and async request loops log the complete `response.headers` object at `DEBUG` level:

```python
log.debug(
    'HTTP Response: %s %s "%i %s" %s',
    request.method,
    request.url,
    response.status_code,
    response.reason_phrase,
    response.headers,   # all headers
)
```

HTTP response headers can contain `Set-Cookie`, session tokens, or internal service tokens. In a debug log, these values would be recorded in plain text.

**Recommendation**  
Log only header names (not values) unless explicitly requested:

```python
log.debug(
    'HTTP Response: %s %s "%i %s" headers=[%s]',
    request.method,
    request.url,
    response.status_code,
    response.reason_phrase,
    ", ".join(response.headers.keys()),
)
```

---

## SEC-003 — AWS secret key handled as plain `str`; not zeroed from memory

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/lib/aws/_credentials.py`; `src/anthropic/lib/aws/_client.py` |
| **Severity** | Low |
| **Category** | Security |

**Description**  
AWS credentials (`aws_access_key`, `aws_secret_key`) are accepted as plain Python `str` objects and stored on the client instance. Python strings are immutable and cannot be zero-wiped. This is an inherent language limitation and not a bug specific to this SDK — however, it is worth documenting for security-conscious users operating in environments with memory-safety requirements (e.g. secrets scanning tools, core-dump analysis).

**Recommendation**  
Document this limitation explicitly in the security documentation and advise users to rely on the default AWS credential chain (IAM roles / instance profiles) rather than passing explicit key material when possible.

---

## SEC-004 — `copy()` cannot revoke credentials — `api_key or self.api_key` always inherits

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_client.py`, lines 238–239 and 478–479 |
| **Severity** | Low |
| **Category** | Security |

**Description**  
(Also noted as CQ-002 in the main code review.)

Because `copy()` uses Python's `or` operator to merge credentials, a caller cannot explicitly remove an `api_key` from a copied client by passing `api_key=None`. The old key is always silently inherited:

```python
new_client = client.copy(api_key=None, auth_token="my-token")
# new_client.api_key is still set to the original api_key — not None
```

This can lead to unintentional multi-factor authentication (both API key and bearer token sent simultaneously) or difficulty rotating credentials in long-lived client trees.

**Recommendation**  
Use a sentinel value to distinguish "not provided" from "explicitly cleared". See CQ-002 in `code_review.md` for a recommended fix.

---

## SEC-005 — `x-should-retry: true` header can force unlimited retries from server

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_base_client.py`, lines 810–812 |
| **Severity** | Low |
| **Category** | Security |

**Description**  
The `_should_retry` method unconditionally honours the server-sent `x-should-retry: true` header for any response:

```python
if should_retry_header == "true":
    return True
```

This is intentional for the Anthropic API itself but represents a Server-Side Request Forgery (SSRF) amplification risk if the SDK is ever pointed at an untrusted base URL (e.g., a misconfigured `ANTHROPIC_BASE_URL`). A malicious server could force the client into an extended retry loop, exhausting connections or causing a denial-of-service against the caller.

**Recommendation**  
This is an acceptable trade-off for a purpose-built SDK. Document the assumption that `base_url` must point to a trusted endpoint. Consider adding a warning in the README about the risks of custom `base_url` values in untrusted environments.

---

## SEC-006 — Vertex AI access token logged via `assert isinstance(self.credentials.token, str)`

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/lib/vertex/_client.py`, lines 175–176 and 320–321 |
| **Severity** | Low |
| **Category** | Security |

**Description**  
The Vertex client asserts that the refreshed credential token is a string:

```python
assert isinstance(self.credentials.token, str)
return self.credentials.token
```

The token itself is returned and then appended to the `Authorization: Bearer ...` header. While the token is not directly logged here, if `ANTHROPIC_LOG=debug` is enabled, SEC-001 above would capture it via the request options dump or the full headers dump at SEC-002.

This is a combined risk that amplifies SEC-001 and SEC-002 for Vertex AI users specifically.

**Recommendation**  
Apply the header-exclusion fix from SEC-001 and SEC-002, which would eliminate the exposure of this token as well.
