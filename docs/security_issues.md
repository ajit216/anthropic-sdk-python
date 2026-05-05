# Security Issues: anthropic-sdk-python

**Review Date:** 2025-05-05  
**Reviewer:** Senior Python Developer (automated review)  
**Codebase Version:** 0.89.0  

---

## Overview

This document covers security-relevant findings identified during the code review of `anthropic-sdk-python`. No critical vulnerabilities were found. The most significant concern is a medium-severity issue around internal header handling that could be exploited if the SDK is used in a context where user-controlled header names are forwarded directly to the client.

---

## Findings

### SEC-01 — Internal override header not sanitised against user input  
**Location:** `src/anthropic/_constants.py:6`, `src/anthropic/_response.py:744`, `src/anthropic/_base_client.py:640`  
**Category:** Security  
**Severity:** Medium  

**Description:**  
The SDK uses the internal header `____stainless_override_cast_to` to pass a type object between `with_raw_response` / `with_streaming_response` wrapper construction and the actual HTTP request pipeline. This header is popped from the outgoing request before it is sent, but there is no validation to prevent a user-supplied header with the same name from influencing the `cast_to` type used to deserialise the response.

```python
# _constants.py
OVERRIDE_CAST_TO_HEADER = "____stainless_override_cast_to"

# _base_client.py — happens before request is sent
override_cast_to = headers.pop(OVERRIDE_CAST_TO_HEADER, not_given)
if is_given(override_cast_to):
    options.headers = headers
    return cast(Type[ResponseT], override_cast_to)
```

**Attack scenario:** An attacker who can control `extra_headers` passed by the application to the SDK (e.g. via user-controlled configuration, middleware forwarding, or insecure proxy injection) could supply `____stainless_override_cast_to = <arbitrary object>` and cause the response-parsing code to attempt construction with an unexpected type, potentially leading to `AttributeError`, information leakage through exception messages, or unexpected code execution in extreme cases with custom `ModelBuilderProtocol` implementations.

**Recommendation:**  
1. Validate that `override_cast_to` is actually a type/class before using it.
2. Consider using a request-local mechanism (e.g. a thread-local or context-var) instead of a special header to pass the cast type, eliminating the attack surface entirely.
3. At minimum, document that `extra_headers` must not be user-controlled without sanitisation.

---

### SEC-02 — `copy()` cannot clear credentials via `None`  
**Location:** `src/anthropic/_client.py`, lines 238–239 and 478–479  
**Category:** Security  
**Severity:** Low  

**Description:**  
The `copy()` method (aliased as `with_options()`) uses the `or` operator to fall back to the existing credential when the new value is `None`:

```python
return self.__class__(
    api_key=api_key or self.api_key,       # cannot pass None to clear key
    auth_token=auth_token or self.auth_token,
    ...
)
```

This means it is **impossible** to create a credential-free copy of a client (e.g. for unauthenticated endpoints or credential rotation where the new key is not yet available). Passing `api_key=None` silently inherits the parent key instead of raising or creating an unauthenticated client.

**Recommendation:** Use an explicit sentinel (`NOT_GIVEN`) to distinguish "not specified" from "explicitly None":
```python
api_key=self.api_key if api_key is not_given else api_key,
```

---

### SEC-03 — `Retry-After` header accepted without floor validation  
**Location:** `src/anthropic/_base_client.py`, lines 791–792  
**Category:** Security / Robustness  
**Severity:** Low  

**Description:**  
The `_parse_retry_after_header` method returns negative values without clamping:

```python
retry_date = email.utils.mktime_tz(retry_date_tuple)
return float(retry_date - time.time())   # can be negative
```

A `Retry-After` header containing a past date returns a negative float. The caller in `_calculate_retry_timeout` handles this with `return timeout if timeout >= 0 else 0`, but only for the *final* computed timeout after jitter, not for the raw `retry_after` value returned from `_parse_retry_after_header`. If the check `0 < retry_after <= 60` is changed in future, a negative value could cause `anyio.sleep(negative)` which raises a `ValueError`.

A malicious or misconfigured server could also attempt to send a very large `Retry-After-Ms` header (the non-standard millisecond variant is accepted without upper-bound validation on the millisecond path):

```python
retry_ms_header = response_headers.get("retry-after-ms", None)
return float(retry_ms_header) / 1000   # no upper-bound check
```

The 60-second cap on the final result of `_parse_retry_after_header` does protect against large values in `_calculate_retry_timeout`, but `_parse_retry_after_header` itself is a public-ish method that could be called elsewhere.

**Recommendation:**  
1. Clamp the return value of `_parse_retry_after_header` to `[0, some_maximum]` before returning, e.g. `max(0.0, min(float(retry_date - time.time()), 60.0))`.  
2. Add upper-bound validation on the `retry-after-ms` path.

---

### SEC-04 — API key logged at DEBUG level  
**Location:** `src/anthropic/_base_client.py`, lines 496–509  
**Category:** Security  
**Severity:** Low  

**Description:**  
At DEBUG log level, the SDK logs full request options including headers:

```python
if log.isEnabledFor(logging.DEBUG):
    log.debug(
        "Request options: %s",
        model_dump(options, exclude_unset=True, ...),
    )
```

The `options` object includes `headers`, which contain the `X-Api-Key` and `Authorization` headers (i.e., the user's API key and auth token). If a developer enables DEBUG logging in a production environment or test pipeline, credentials will appear in logs.

**Recommendation:**  
Redact sensitive headers before logging:
```python
safe_options = model_dump(options, exclude_unset=True, ...)
if "headers" in safe_options:
    safe_options["headers"] = {
        k: "**redacted**" if k.lower() in ("x-api-key", "authorization") else v
        for k, v in safe_options["headers"].items()
    }
log.debug("Request options: %s", safe_options)
```

---

### SEC-05 — Proxy configuration sourced from environment without validation  
**Location:** `src/anthropic/_utils/_httpx.py`, `get_environment_proxies()`  
**Category:** Security  
**Severity:** Informational  

**Description:**  
The SDK reads proxy configuration from environment variables (`HTTP_PROXY`, `HTTPS_PROXY`, etc.) via `urllib.request.getproxies()`. Proxy URLs are used directly to configure `httpx.Proxy` and transport objects without further validation. This is standard practice, but in containerised environments where environment variables can be injected by an orchestrator, a misconfigured or malicious proxy setting could route all API traffic through an attacker-controlled host.

**Recommendation:** Document this behaviour explicitly in the README security section. Consider providing a `trust_env=False` shortcut option in the client constructor to disable environment-based proxy configuration, similar to httpx's own `trust_env` parameter.

---

## Non-Issues (Considered and Dismissed)

- **`uuid.uuid4()` for idempotency keys** — Cryptographically random UUID4 is appropriate here; no concern.
- **`random.random()` for jitter** — Not a security issue; jitter does not need to be cryptographically random.
- **API base URL hardcoded as `https://api.anthropic.com`** — TLS is enforced by httpx; no plaintext fallback. Fine.
- **`distro` library** — Used only for telemetry header construction; no PII concern.
