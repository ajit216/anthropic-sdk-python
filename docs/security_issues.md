# Security Review — anthropic-sdk-python

**Reviewer:** Senior Python Developer (automated review)  
**Version reviewed:** 0.89.0  
**Date:** 2026-05-01  

---

## Summary

No critical or high-severity security vulnerabilities were identified. The SDK correctly handles authentication, does not log sensitive data, and does not hardcode credentials. Three low-severity observations are documented below for awareness.

---

## SEC-001 — Internal override header allows cast-type substitution via request headers

| Field | Detail |
|---|---|
| **Location** | `src/anthropic/_constants.py` line 6; `src/anthropic/_base_client.py` line 640; `src/anthropic/_response.py` lines 744, 769, 828, 851 |
| **Category** | Security |
| **Severity** | Low |

**Description:**  
An internal mechanism uses an HTTP request header (`____stainless_override_cast_to`) to communicate a desired response cast type from the request-building phase to the response-processing phase. This header is set by the SDK itself on internally constructed requests and is not sent to the API server.

```python
# _constants.py
OVERRIDE_CAST_TO_HEADER = "____stainless_override_cast_to"

# _response.py — set before sending
extra_headers[OVERRIDE_CAST_TO_HEADER] = response_cls

# _base_client.py — read during response processing
override_cast_to = headers.pop(OVERRIDE_CAST_TO_HEADER, not_given)
```

**Risk:**  
If a user passes `extra_headers={OVERRIDE_CAST_TO_HEADER: SomeMaliciousType}` to any API call, they can override the SDK's intended response parsing type. This is a deliberate user action and not exploitable remotely, but it is an undocumented internal interface that could be abused.

**Recommendation:**  
- Add a comment to `_constants.py` clearly marking this as an internal-only header.  
- In `_base_client._build_headers`, strip the `OVERRIDE_CAST_TO_HEADER` from any user-supplied `extra_headers` before merging, or document in the public API that this header must not be set externally.

---

## SEC-002 — Authentication validation does not emit a warning when no credentials are supplied to a forked client

| Field | Detail |
|---|---|
| **Location** | `src/anthropic/_client.py`, `Anthropic.copy()` lines 200–247 |
| **Category** | Security |
| **Severity** | Low |

**Description:**  
The `copy()` method (and its alias `with_options()`) constructs a new client instance. It uses `api_key or self.api_key` to inherit credentials:

```python
return self.__class__(
    api_key=api_key or self.api_key,
    auth_token=auth_token or self.auth_token,
    ...
)
```

If the original client was created without credentials (e.g., intending to set them later), and `copy()` is called, the new client will silently inherit `None` for both `api_key` and `auth_token`. The `_validate_headers` check only runs at request time, not at client construction time. A developer may not realize no credentials were inherited until a request actually fails.

**Recommendation:**  
Consider emitting a `warnings.warn(DeprecationWarning)` or `logging.warning` at client construction time (not request time) when both `api_key` and `auth_token` are `None` and the validation isn't bypassed by explicit `Omit` headers.

---

## SEC-003 — Request body may be logged at DEBUG level

| Field | Detail |
|---|---|
| **Location** | `src/anthropic/_base_client.py`, lines 496–509 (`_build_request`) |
| **Category** | Security |
| **Severity** | Low |

**Description:**  
At `logging.DEBUG` level, the full request options are serialised and logged:

```python
if log.isEnabledFor(logging.DEBUG):
    log.debug(
        "Request options: %s",
        model_dump(options, exclude_unset=True, exclude={"content"} if PYDANTIC_V1 else {}),
    )
```

The `content` field is excluded (binary data), but `json_data` is not. This means request bodies — which can include user messages, tool inputs, or other potentially sensitive content — are emitted to the debug log if the application enables `DEBUG` logging for the `anthropic` logger.

**Recommendation:**  
- Document this behaviour clearly in the logging section of the README.
- Consider also excluding `json_data` by default, or adding a redaction mechanism for known sensitive fields.
- Alternatively, log only metadata (method, URL, header names) at DEBUG, and require a separate `TRACE`-level opt-in for body logging.

---

## Positive Security Observations

- **No hardcoded credentials** anywhere in `src/` or `tests/`.
- **API keys read from environment variables** only, never persisted to disk by the SDK.
- **TLS**: httpx enforces TLS by default; the SDK does not disable certificate verification.
- **Retry idempotency keys** use `uuid.uuid4()` — cryptographically random, preventing predictability.
- **Error responses** do not re-echo user input back in exception messages.
- **Token logging**: The `auth_headers` property constructs the `Authorization: Bearer ...` header on-demand and is not logged.
- **Dependency pinning**: `pyproject.toml` specifies upper-bound version pins for all dependencies, reducing supply-chain exposure from unexpected major upgrades.
- **`sniffio` used** to detect async context, preventing silent cross-runtime issues.
