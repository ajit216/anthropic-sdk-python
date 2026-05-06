# Security Issues: anthropic-sdk-python

**Review Date:** 2026-05-06  
**SDK Version:** 0.89.0  

---

## Overview

The Anthropic Python SDK handles credentials (API keys, bearer tokens, AWS SigV4, Google OAuth2) and communicates over HTTPS with the Anthropic API. This document focuses on security-relevant findings from the code review.

No critical or high severity security vulnerabilities were identified. The SDK correctly uses HTTPS for all API communication, employs cryptographically secure UUID generation for idempotency keys, and delegates TLS certificate verification to `httpx` (which uses the system trust store by default).

---

## Findings

### SEC-001 — Full Response Headers Logged at DEBUG Level

- **Location:** `src/anthropic/_base_client.py`, line 1118  
- **Severity:** Low  
- **OWASP Category:** A09:2021 – Security Logging and Monitoring Failures  

**Description:**  
The full set of HTTP response headers is passed to `log.debug()`:

```python
log.debug(
    'HTTP Response: %s %s "%i %s" %s',
    request.method,
    request.url,
    response.status_code,
    response.reason_phrase,
    response.headers,   # ← complete header dict
)
```

Under normal Anthropic API operation, response headers do not contain credentials or secrets. However, if a corporate proxy or custom transport injects `Authorization` or `Set-Cookie` response headers, those values would appear in application logs when `ANTHROPIC_LOG=debug` is set.

**Risk Assessment:**  
Low risk in practice. The `ANTHROPIC_LOG=debug` environment variable must be explicitly set, and Anthropic's own response headers do not carry secrets.

**Recommendation:**  
Document in the README/logging section that debug logging should not be enabled in production. Optionally, filter sensitive header names before logging:

```python
SENSITIVE_HEADERS = frozenset({"authorization", "x-api-key", "set-cookie", "cookie"})
safe_headers = {k: v for k, v in response.headers.items() if k.lower() not in SENSITIVE_HEADERS}
log.debug('HTTP Response: %s %s "%i %s" %s', ..., safe_headers)
```

---

### SEC-002 — Non-Cryptographic `random` Module for Retry Jitter

- **Location:** `src/anthropic/_base_client.py`, lines 16 and 801  
- **Severity:** Low (Informational)  
- **OWASP Category:** A02:2021 – Cryptographic Failures  

**Description:**  
Retry delay jitter is computed using the standard `random` module (Mersenne Twister PRNG):

```python
from random import random
# ...
jitter = 1 - 0.25 * random()
timeout = sleep_seconds * jitter
```

Mersenne Twister is not a cryptographically secure PRNG (CSPRNG). However, for retry jitter purposes, cryptographic randomness is not required — the goal is simply to de-correlate retry attempts across clients, not to produce unpredictable values for security purposes.

**Risk Assessment:**  
Not a vulnerability. The use of `random` here is appropriate. Idempotency keys are correctly generated using `uuid.uuid4()` (line 841), which uses `os.urandom()` internally and is cryptographically secure.

**Recommendation:**  
No action required. This is noted purely for completeness.

---

### SEC-003 — API Key Stored as Plain Instance Attribute

- **Location:** `src/anthropic/_client.py`, lines 55, 94, 296, 334  
- **Severity:** Low (Design Observation)  
- **OWASP Category:** A02:2021 – Cryptographic Failures  

**Description:**  
The API key is stored as a plain Python string attribute on the client instance:

```python
class Anthropic(SyncAPIClient):
    api_key: str | None
    auth_token: str | None
```

This is standard practice for SDK clients. However, it means the key is accessible via `client.api_key` and will appear in memory dumps, serialized objects, and debug `repr()` output.

**Risk Assessment:**  
This is the standard design for Python SDK clients (AWS Boto3, Google Cloud SDK, OpenAI SDK all use the same pattern). The risk is inherent to the design, not a flaw.

**Recommendation:**  
Consider implementing a `__repr__` override on the `Anthropic` class that masks the API key, similar to how some SDKs show `api_key=***`. This prevents accidental credential exposure in logs when developers print the client object.

---

### SEC-004 — `_extra_kwargs` in `copy()` Allows Arbitrary Constructor Parameter Injection

- **Location:** `src/anthropic/_client.py`, lines 213 (Anthropic.copy) and 453 (AsyncAnthropic.copy)  
- **Severity:** Low (Design Observation)  

**Description:**  
The `copy()` method accepts an `_extra_kwargs: Mapping[str, Any] = {}` parameter that is passed directly to the class constructor with `**_extra_kwargs`:

```python
def copy(self, ..., _extra_kwargs: Mapping[str, Any] = {}) -> Self:
    return self.__class__(
        ...
        **_extra_kwargs,
    )
```

The leading underscore convention suggests this is internal/private, but it is part of the public method signature. A caller could use this to pass undocumented constructor arguments that bypass the public API.

**Risk Assessment:**  
Low risk. This is intentionally designed for internal SDK evolution. No security boundary is crossed since the caller already has full access to `__class__` directly.

**Recommendation:**  
Consider adding a `# private/internal` comment above the parameter to clarify intent, or use `**kwargs` with explicit handling to avoid the mutable default argument issue.

---

## Credential Handling Assessment

| Credential Type | Storage | Transport | Risk |
|----------------|---------|-----------|------|
| `api_key` (Anthropic) | Plain `str` on client instance | HTTPS header (`X-Api-Key`) | Low |
| `auth_token` (Bearer) | Plain `str` on client instance | HTTPS header (`Authorization: Bearer`) | Low |
| AWS SigV4 credentials | Via `boto3` session / env vars | Signed request headers | Low (delegates to boto3) |
| Google OAuth2 credentials | Via `google-auth` library | HTTPS header (`Authorization: Bearer`) | Low (delegates to google-auth) |
| Vertex AI access token | `access_token: str` on client | HTTPS header (`Authorization: Bearer`) | Low |

All credentials are transmitted exclusively over HTTPS. The SDK correctly sets `follow_redirects=True` on the httpx transport, but since all base URLs are hardcoded to HTTPS endpoints (`api.anthropic.com`, AWS, GCP), credential downgrade via redirect is not a practical concern.

---

## Recommendations Summary

| ID | Severity | Recommendation |
|----|----------|----------------|
| SEC-001 | Low | Filter sensitive headers before debug logging, or document that debug mode is not for production |
| SEC-002 | Info | No action required; `random` is appropriate for jitter |
| SEC-003 | Low | Add `__repr__` to Anthropic client that masks `api_key` |
| SEC-004 | Low | Add clarifying comment on `_extra_kwargs` parameter |
