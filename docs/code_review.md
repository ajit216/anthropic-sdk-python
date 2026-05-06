# Code Review: anthropic-sdk-python

**Reviewer:** Senior Software Developer (Automated Review)  
**Review Date:** 2026-05-06  
**SDK Version:** 0.89.0  
**Scope:** Full codebase review of `src/anthropic/`  

---

## Summary

The Anthropic Python SDK is a well-structured, auto-generated SDK with solid patterns for HTTP client management, streaming, retry logic, and Pydantic-based model construction. The core architecture is sound; most issues identified are in the medium-to-low severity range, with one confirmed bug (missing `f`-string prefix) and several code-quality concerns worth addressing.

---

## Issues by Category

### 🐛 Bugs

---

#### BUG-001 — Missing `f`-string Prefix Produces Uninformative Error Message

- **Location:** `src/anthropic/_files.py`, line 100  
- **Category:** Bug  
- **Severity:** Medium  

**Description:**  
The error message string in `async_to_httpx_files()` is missing the `f` prefix, meaning `{type(files)}` is never interpolated. The actual type of the invalid input is never shown to the user.

```python
# Current (line 100) — {type(files)} is a literal string, NOT interpolated
raise TypeError("Unexpected file type input {type(files)}, expected mapping or sequence")
```

Compare with the synchronous equivalent in `to_httpx_files()` at line 58, which is correct:
```python
raise TypeError(f"Expected file input `{obj!r}`")
```

**Recommendation:**  
Add the `f` prefix:
```python
raise TypeError(f"Unexpected file type input {type(files)}, expected mapping or sequence")
```

---

#### BUG-002 — Redundant / Unreachable Conditions in `_validate_headers`

- **Location:** `src/anthropic/_client.py`, lines 185–198 (Anthropic) and 425–438 (AsyncAnthropic)  
- **Category:** Bug / Code Quality  
- **Severity:** Low  

**Description:**  
The `_validate_headers` method has three consecutive `if` blocks that check for the same headers. The first block already returns if either `Authorization` or `X-Api-Key` are present. The second and third blocks therefore can never be reached when those headers exist, making lines 190–194 dead code. This is a logic error: the intent appears to be checking for the `Omit()` sentinel (which means the user explicitly removed the header) but the first condition short-circuits it.

```python
def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
    if headers.get("Authorization") or headers.get("X-Api-Key"):
        # valid
        return

    # These two blocks are UNREACHABLE when either header exists (already returned above):
    if headers.get("X-Api-Key") or isinstance(custom_headers.get("X-Api-Key"), Omit):
        return

    if headers.get("Authorization") or isinstance(custom_headers.get("Authorization"), Omit):
        return

    raise TypeError(...)
```

**Recommendation:**  
Restructure the checks so the `Omit()` sentinel detection is not hidden behind an already-checked condition:

```python
def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
    if headers.get("Authorization") or headers.get("X-Api-Key"):
        return  # valid: a real auth header is set

    # Allow explicit omission via Omit() sentinel
    if isinstance(custom_headers.get("X-Api-Key"), Omit):
        return
    if isinstance(custom_headers.get("Authorization"), Omit):
        return

    raise TypeError(
        "Could not resolve authentication method. Expected either api_key or auth_token "
        "to be set. Or for one of the `X-Api-Key` or `Authorization` headers to be "
        "explicitly omitted"
    )
```

---

### ⚙️ Code Quality

---

#### CQ-001 — Error Message String Wrapped in Redundant Quotes

- **Location:** `src/anthropic/_client.py`, lines 197 and 437  
- **Category:** Code Quality  
- **Severity:** Low  

**Description:**  
The `TypeError` message passed to `raise TypeError(...)` is itself a Python string that wraps the actual message in double-quote characters. When a user catches this exception and prints its message, the displayed text will start and end with `"`, which looks unintentional:

```python
raise TypeError(
    '"Could not resolve authentication method. Expected either api_key or auth_token ...'
)
# User sees: '"Could not resolve authentication method...'
```

**Recommendation:**  
Remove the wrapping quotes from the string literal. Both occurrences (Anthropic and AsyncAnthropic classes) need to be updated.

---

#### CQ-002 — Unnecessary `f`-string for Static `base_url` Literal

- **Location:** `src/anthropic/_client.py`, lines 103 and 343  
- **Category:** Code Quality / Style  
- **Severity:** Low  

**Description:**  
The default `base_url` is assigned using an f-string with no interpolation:
```python
base_url = f"https://api.anthropic.com"
```
This is a no-op—no variables are interpolated—and is misleading to readers who expect `f"..."` to contain substitution expressions.

**Recommendation:**  
Use a plain string literal:
```python
base_url = "https://api.anthropic.com"
```

---

#### CQ-003 — Mutable Default Argument `options: RequestOptions = {}`

- **Location:** `src/anthropic/_base_client.py`, multiple method signatures (lines 1263, 1273, 1284, 1294, 1311, 1324, 1338, 1351, 1380, 1406, 1431, 1452, 1893, 1903, 1914, 1924, 1940, 1953, 1967, 1980)  
- **Category:** Code Quality  
- **Severity:** Low  

**Description:**  
Many method signatures use a mutable dict as a default argument value:
```python
def get(self, path: str, *, cast_to: Type[ResponseT], options: RequestOptions = {}, ...) -> ...:
```
In Python, mutable default arguments are evaluated once at function definition time and shared across all calls. While this does not cause a practical bug here (since `options` is only read, not mutated directly), it is a well-known Python anti-pattern that may cause subtle bugs if the implementation ever changes to mutate the dict.

**Recommendation:**  
Use `None` as the default and create an empty dict inside the function body:
```python
def get(self, path: str, *, cast_to: Type[ResponseT], options: RequestOptions | None = None, ...) -> ...:
    if options is None:
        options = {}
    ...
```
Alternatively, use `_extra_kwargs: Mapping[str, Any] = {}` pattern that already exists (which is safe because `Mapping` is immutable at the interface level), but for `RequestOptions` (a `TypedDict`, which is a `dict` at runtime), switching to `None` is safer.

---

#### CQ-004 — `copy()` Uses `or` to Merge Credential Parameters (Falsy-Value Bug)

- **Location:** `src/anthropic/_client.py`, lines 238–239 (Anthropic) and 478–479 (AsyncAnthropic)  
- **Category:** Code Quality / Bug Risk  
- **Severity:** Medium  

**Description:**  
The `copy()` method uses Python's `or` operator to decide whether to use the caller-provided credential or the existing one:

```python
return self.__class__(
    api_key=api_key or self.api_key,
    auth_token=auth_token or self.auth_token,
    ...
)
```

This pattern silently ignores any falsy value passed for `api_key` or `auth_token`. If a caller passes `api_key=""` (an empty string) or `api_key=None` explicitly intending to unset the key, the call falls back to `self.api_key` instead, which is the *old* value. This makes it impossible to explicitly clear a credential via `copy()`.

**Recommendation:**  
Use explicit `None`-checks to distinguish "not provided" from "intentionally falsy":
```python
api_key=api_key if api_key is not None else self.api_key,
auth_token=auth_token if auth_token is not None else self.auth_token,
```

---

#### CQ-005 — `ServiceUnavailableError`, `DeadlineExceededError`, and `RequestTooLargeError` Not Exported

- **Location:** `src/anthropic/_exceptions.py` (lines 115, 127, 135) and `src/anthropic/__init__.py`  
- **Category:** Code Quality / Usability  
- **Severity:** Medium  

**Description:**  
Three exception classes are defined in `_exceptions.py` but are not included in the module's `__all__` list and are not re-exported from `src/anthropic/__init__.py`:

- `RequestTooLargeError` (HTTP 413)
- `ServiceUnavailableError` (HTTP 503)
- `DeadlineExceededError` (HTTP 504)

Additionally, `_make_status_error` in `_client.py` does not handle HTTP 503 or 504 explicitly—they fall through to the generic `>= 500` → `InternalServerError` mapping. This means a 503 response raises `InternalServerError`, not `ServiceUnavailableError`, and users cannot catch these by their specific type.

```python
# _client.py _make_status_error — 503 and 504 not handled
if response.status_code >= 500:
    return _exceptions.InternalServerError(err_msg, response=response, body=body)
# ServiceUnavailableError and DeadlineExceededError are never returned
```

**Recommendation:**  
1. Add explicit handling for 503 and 504 in `_make_status_error`.
2. Add `RequestTooLargeError`, `ServiceUnavailableError`, and `DeadlineExceededError` to `__all__` in `_exceptions.py`.
3. Export all three from `__init__.py` so users can `from anthropic import ServiceUnavailableError`.

---

#### CQ-006 — `qs` Property Instantiates a New `Querystring` on Every Call

- **Location:** `src/anthropic/_base_client.py`, lines 675–676  
- **Category:** Performance / Code Quality  
- **Severity:** Low  

**Description:**  
The `qs` property on `BaseClient` creates a new `Querystring()` object on every invocation:
```python
@property
def qs(self) -> Querystring:
    return Querystring()
```
`Querystring` has no mutable state between requests, so this allocation is unnecessary.

**Recommendation:**  
Cache the instance at class level or use `functools.cached_property`:
```python
@cached_property
def qs(self) -> Querystring:
    return Querystring()
```
Note: the `Anthropic` and `AsyncAnthropic` subclasses override `qs` to use `Querystring(array_format="comma")`, so caching works correctly since the subclass override also only runs once.

---

#### CQ-007 — `AsyncHttpxClientWrapper.__del__` Has Fragile Async Cleanup

- **Location:** `src/anthropic/_base_client.py`, lines 1543–1551  
- **Category:** Code Quality / Reliability  
- **Severity:** Medium  

**Description:**  
The `__del__` method on `AsyncHttpxClientWrapper` attempts to schedule the async close operation using `asyncio.get_running_loop().create_task()`. This approach has multiple problems:
1. `asyncio.get_running_loop()` raises a `RuntimeError` if there is no running event loop (which is common when the interpreter is shutting down or when the object is garbage collected outside of an async context). The bare `except Exception: pass` silently swallows this, meaning the connection is never properly closed.
2. The internal `TODO` comment acknowledges it doesn't support non-asyncio runtimes (`trio`, `uvloop` configurations).
3. Tasks created via `create_task()` during interpreter shutdown may never execute.

```python
def __del__(self) -> None:
    if self.is_closed:
        return
    try:
        # TODO(someday): support non asyncio runtimes here
        asyncio.get_running_loop().create_task(self.aclose())
    except Exception:
        pass
```

**Recommendation:**  
The async client should be explicitly closed using `async with` or `await client.close()`. The `__del__` approach is inherently unreliable for async resources. Consider issuing a `ResourceWarning` when the client is garbage collected without being closed, similar to how Python's built-in file objects work:

```python
def __del__(self) -> None:
    if self.is_closed:
        return
    import warnings
    warnings.warn(
        "Unclosed AsyncAnthropic client. Use 'async with' or call 'await client.close()' explicitly.",
        ResourceWarning,
        source=self,
    )
    try:
        asyncio.get_running_loop().create_task(self.aclose())
    except Exception:
        pass
```

---

#### CQ-008 — Unresolved `TODO` Comments in Critical Code Paths

- **Location:** Multiple files  
- **Category:** Code Quality  
- **Severity:** Low  

**Description:**  
There are multiple unresolved `TODO` comments in production code paths, some of which indicate known gaps or uncertainties:

| File | Line | Comment |
|------|------|---------|
| `_base_client.py` | 98 | `# TODO: make base page type vars covariant` |
| `_base_client.py` | 201 | `# TODO: do we have to preprocess params here?` |
| `_base_client.py` | 1548 | `# TODO(someday): support non asyncio runtimes here` |
| `_base_client.py` | 2247, 2254 | `# TODO: untested` (architecture detection) |
| `_models.py` | 432 | `# TODO` (missing Pydantic v1 extra fields type support) |
| `_models.py` | 802 | `# TODO: condition is weird` |
| `_transform.py` | 37–38 | `# TODO: support for drilling globals() and locals()` |
| `_transform.py` | 214, 380 | `# TODO: there may be edge cases...` |
| `_utils/_utils.py` | 275 | `# TODO: this error message is not deterministic` |
| `lib/streaming/_messages.py` | 457 | `# TODO: check index` |

**Recommendation:**  
Track all TODOs in a dedicated issue tracker. Those marked "untested" or flagging non-deterministic behavior are particularly risky and should be addressed or explicitly accepted as known limitations in documentation.

---

#### CQ-009 — `is_body_allowed` Only Excludes `GET` — HEAD/OPTIONS/TRACE May Get Unexpected Bodies

- **Location:** `src/anthropic/_base_client.py`, line 566  
- **Category:** Code Quality  
- **Severity:** Low  

**Description:**  
The request builder determines whether to include a body only by checking if the method is not `GET`:
```python
is_body_allowed = options.method.lower() != "get"
```
HTTP semantics dictate that `HEAD`, `OPTIONS`, and `TRACE` requests should not have a body. While the SDK does not appear to use these methods currently, the logic is broader than intended.

**Recommendation:**  
Use an allowlist approach for methods that may carry a body:
```python
BODY_METHODS = frozenset({"post", "put", "patch", "delete"})
is_body_allowed = options.method.lower() in BODY_METHODS
```

---

### 🔒 Security

---

#### SEC-001 — HTTP Response Headers Logged at DEBUG Level May Contain Sensitive Data

- **Location:** `src/anthropic/_base_client.py`, line 1118  
- **Category:** Security  
- **Severity:** Low  

**Description:**  
The full set of HTTP response headers is logged at the DEBUG level:
```python
log.debug(
    'HTTP Response: %s %s "%i %s" %s',
    request.method,
    request.url,
    response.status_code,
    response.reason_phrase,
    response.headers,  # ← All headers, including any sensitive ones
)
```
In most cases this is fine (Anthropic response headers are not sensitive), but if a proxy or custom transport injects credential headers into the response, they would be captured in logs.

**Recommendation:**  
This is a low-risk, low-priority concern given the SDK's context. As a best practice, consider filtering out specific sensitive header names (e.g., `Authorization`, `X-Api-Key`, `Set-Cookie`) before logging, or document that debug logging should not be enabled in production environments.

---

#### SEC-002 — Non-Cryptographic `random` Used for Retry Jitter

- **Location:** `src/anthropic/_base_client.py`, line 16 and 801  
- **Category:** Security  
- **Severity:** Low  

**Description:**  
Retry jitter is computed using `from random import random`, which uses Python's Mersenne Twister PRNG — not a cryptographically secure source. For retry timing, this is entirely acceptable since the purpose is just load distribution, not security. However, having `random` imported explicitly alongside security-sensitive code paths could mislead future maintainers.

**Recommendation:**  
No action required. The use of `random` for jitter is appropriate. However, ensuring that idempotency key generation (which uses `uuid.uuid4()` at line 841 — correctly using `os.urandom()` internally) remains separate from the `random` import is good practice.

---

## Summary Table

| ID | File | Line(s) | Category | Severity | Title |
|----|------|---------|----------|----------|-------|
| BUG-001 | `_files.py` | 100 | Bug | **Medium** | Missing `f`-string prefix in error message |
| BUG-002 | `_client.py` | 185–194, 425–434 | Bug | Low | Redundant/unreachable conditions in `_validate_headers` |
| CQ-001 | `_client.py` | 197, 437 | Code Quality | Low | Error message string wrapped in redundant quotes |
| CQ-002 | `_client.py` | 103, 343 | Style | Low | Unnecessary `f`-string for static base URL |
| CQ-003 | `_base_client.py` | ~20 locations | Code Quality | Low | Mutable default argument `options: RequestOptions = {}` |
| CQ-004 | `_client.py` | 238–239, 478–479 | Code Quality | **Medium** | `copy()` uses `or` losing explicit `None`/empty credentials |
| CQ-005 | `_exceptions.py`, `__init__.py` | 115, 127, 135 | Code Quality | **Medium** | 3 exception classes not exported; 2 not raised on correct status codes |
| CQ-006 | `_base_client.py` | 675–676 | Performance | Low | `qs` property allocates new object on every call |
| CQ-007 | `_base_client.py` | 1543–1551 | Reliability | **Medium** | `AsyncHttpxClientWrapper.__del__` has fragile async cleanup |
| CQ-008 | Multiple | Various | Code Quality | Low | Multiple unresolved TODO comments in production code |
| CQ-009 | `_base_client.py` | 566 | Code Quality | Low | `is_body_allowed` only excludes GET, not HEAD/OPTIONS/TRACE |
| SEC-001 | `_base_client.py` | 1118 | Security | Low | Full response headers logged at DEBUG level |
| SEC-002 | `_base_client.py` | 16, 801 | Security | Low | Non-cryptographic `random` for retry jitter (acceptable) |

---

## Positive Observations

The following aspects of the codebase deserve recognition:

- **Retry logic** (`_should_retry`, `_calculate_retry_timeout`) is well-implemented with proper exponential backoff, jitter, and respect for `Retry-After` headers.
- **Stream handling** correctly uses `finally` blocks to guarantee resource cleanup even if an exception occurs mid-stream.
- **Pydantic v1/v2 compatibility** is handled thoroughly throughout `_models.py` and `_compat.py`.
- **Idempotency keys** are generated using `uuid.uuid4()`, which is cryptographically secure.
- **Context manager support** is properly implemented for both sync and async clients.
- **Type safety** is taken seriously with extensive use of TypeVar, Generic, overload, and Literal types.
- **Platform/architecture detection** for telemetry headers (`platform_headers`) is cached properly using `lru_cache`.
- **File upload handling** correctly uses `anyio.Path` for async reads.
