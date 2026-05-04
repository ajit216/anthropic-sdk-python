# Code Review — anthropic-sdk-python

**Reviewed version:** 0.89.0  
**Review date:** 2026-05-04  
**Scope:** Full codebase under `src/anthropic/`  

---

## Summary

The SDK is overall well-structured and makes good use of Pydantic, httpx, and anyio. The review identified **one confirmed bug**, several **code-quality issues**, and a handful of **security observations**. Issues are grouped by category below.

See [`security_issues.md`](./security_issues.md) for security-specific findings.

---

## 1. Bugs

### BUG-001 — Missing `f`-string prefix in `async_to_httpx_files` error message

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_files.py`, line 100 |
| **Severity** | Medium |
| **Category** | Bug |

**Description**  
The `async_to_httpx_files` function raises a `TypeError` whose message contains an unformatted `{type(files)}` placeholder because the string literal is missing the `f` prefix. The equivalent synchronous function (`to_httpx_files`, line 58) has the same message correctly prefixed with `f`.

```python
# async version — BUG: missing f-prefix, prints literal "{type(files)}"
raise TypeError("Unexpected file type input {type(files)}, expected mapping or sequence")

# sync version — correct
raise TypeError(f"Unexpected file type input {type(files)}, expected mapping or sequence")
```

**Recommendation**  
Add `f` prefix to line 100:

```python
raise TypeError(f"Unexpected file type input {type(files)}, expected mapping or sequence")
```

---

### BUG-002 — Redundant / dead-code checks in `_validate_headers`

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_client.py`, lines 185–198 and 425–438 |
| **Severity** | Low |
| **Category** | Bug / Code Quality |

**Description**  
`_validate_headers` (duplicated in both `Anthropic` and `AsyncAnthropic`) contains logically dead branches. The first check (line 186) already covers both `Authorization` **and** `X-Api-Key`, making lines 190 and 193 unreachable under the same conditions:

```python
def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
    if headers.get("Authorization") or headers.get("X-Api-Key"):
        return                                         # covers both keys already

    if headers.get("X-Api-Key") or ...:               # DEAD: X-Api-Key already checked above
        return

    if headers.get("Authorization") or ...:           # DEAD: Authorization already checked above
        return
```

The `isinstance(custom_headers.get(...), Omit)` branches are only reachable, but the `headers.get(...)` halves of lines 190 and 193 are dead.

**Recommendation**  
Simplify the guard to remove the duplicate header lookups:

```python
def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
    if headers.get("Authorization") or headers.get("X-Api-Key"):
        return

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

### BUG-003 — Unguarded index access on compaction response content

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/lib/tools/_beta_runner.py`, lines 237 and 518 |
| **Severity** | Medium |
| **Category** | Bug |

**Description**  
After calling the API for compaction, the code accesses `list(response.content)[0]` without first checking whether `response.content` is non-empty. If the API returns a message with an empty content list, this raises an `IndexError` rather than a meaningful error.

```python
first_content = list(response.content)[0]   # IndexError if content is empty
if first_content.type != "text":
    raise ValueError("Compaction response content is not of type 'text'")
```

**Recommendation**  
Guard the access explicitly:

```python
content_list = list(response.content)
if not content_list:
    raise ValueError("Compaction response returned an empty content list")
first_content = content_list[0]
if first_content.type != "text":
    raise ValueError("Compaction response content is not of type 'text'")
```

---

## 2. Code Quality

### CQ-001 — Mutable default argument `_extra_kwargs: Mapping[str, Any] = {}`

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_client.py`, lines 213 and 453 |
| **Severity** | Low |
| **Category** | Code Quality |

**Description**  
Both `Anthropic.copy()` and `AsyncAnthropic.copy()` use a mutable dict literal `{}` as a default argument. Although `Mapping` is used as the type annotation (which is read-only), the default object itself is a `dict` shared across all call sites:

```python
def copy(self, ..., _extra_kwargs: Mapping[str, Any] = {}) -> Self:
```

Ruff rule **B006** (mutable defaults) is explicitly disabled in `pyproject.toml`, which silences the warning for the entire project. While this specific usage is unlikely to cause a bug because `_extra_kwargs` is only read (via `**_extra_kwargs`), it's still a code-smell that could mislead contributors.

**Recommendation**  
Use `None` as the default and normalise inside the function:

```python
def copy(self, ..., _extra_kwargs: Mapping[str, Any] | None = None) -> Self:
    extra = _extra_kwargs or {}
    ...
    return self.__class__(..., **extra)
```

Alternatively, re-enable Ruff B006 once this and any other genuine mutable defaults are fixed.

---

### CQ-002 — `copy()` cannot explicitly clear `api_key` or `auth_token`

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_client.py`, lines 238–239 and 478–479 |
| **Severity** | Low |
| **Category** | Code Quality |

**Description**  
The `copy()` / `with_options()` method merges credentials with `or`:

```python
api_key=api_key or self.api_key,
auth_token=auth_token or self.auth_token,
```

This means passing `api_key=None` is silently ignored and the parent client's key is always inherited. A caller switching from API-key auth to bearer-token auth by passing `api_key=None, auth_token="..."` will still have the old `api_key` in the copied client. The same issue applies to `base_url`.

**Recommendation**  
Use a sentinel (e.g. `NotGiven`) rather than `None` as the "unchanged" signal:

```python
_UNSET: Any = object()

def copy(self, *, api_key: str | None = _UNSET, ...):
    resolved_api_key = self.api_key if api_key is _UNSET else api_key
    ...
```

---

### CQ-003 — `AsyncHttpxClientWrapper.__del__` silently loses async close errors

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_base_client.py`, lines 1542–1551 |
| **Severity** | Low |
| **Category** | Code Quality |

**Description**  
The `__del__` method schedules `aclose()` via `create_task()` but does not await it. If the event loop is already closed (e.g. after the interpreter shuts down), `get_running_loop()` raises `RuntimeError`, which is silently swallowed:

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

The result is that the underlying HTTP connection may not be cleanly closed in all scenarios.

**Recommendation**  
Prefer `asyncio.get_event_loop()` with a fallback to a new loop if no running loop is available, or log a warning when the task cannot be scheduled:

```python
def __del__(self) -> None:
    if self.is_closed:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(self.aclose())
    except RuntimeError:
        # No running loop — best-effort close via a new loop
        try:
            asyncio.run(self.aclose())
        except Exception:
            pass
```

---

### CQ-004 — `ServiceUnavailableError`, `DeadlineExceededError`, `RequestTooLargeError`, and `OverloadedError` missing from public API

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_exceptions.py`, lines 13–22; `src/anthropic/__init__.py`, lines 29–44 |
| **Severity** | Medium |
| **Category** | Code Quality |

**Description**  
Four concrete error classes defined in `_exceptions.py` are absent from both the module-level `__all__` list and the public `__init__.py` import:

- `ServiceUnavailableError` (HTTP 503)
- `DeadlineExceededError` (HTTP 504)
- `RequestTooLargeError` (HTTP 413)
- `OverloadedError` (HTTP 529)

They **are** used internally (Vertex and Bedrock clients raise `ServiceUnavailableError` and `DeadlineExceededError`) and users who catch `anthropic.ServiceUnavailableError` will get a `NameError` at runtime.

**Recommendation**  
Add the missing classes to `__all__` in `_exceptions.py` and export them from `__init__.py`:

```python
# _exceptions.py
__all__ = [
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "RequestTooLargeError",
    "UnprocessableEntityError",
    "RateLimitError",
    "ServiceUnavailableError",
    "OverloadedError",
    "DeadlineExceededError",
    "InternalServerError",
]
```

---

### CQ-005 — `Querystring` object created on every `qs` property access

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_base_client.py`, lines 674–676 |
| **Severity** | Low |
| **Category** | Performance / Code Quality |

**Description**  
The base `qs` property returns a new `Querystring()` object on every call. The subclass overrides in `Anthropic` and `AsyncAnthropic` also do this. Every request that uses query parameters calls `self.qs.stringify(...)` or `self.qs.stringify_items(...)`, creating and discarding a short-lived object unnecessarily.

```python
@property
def qs(self) -> Querystring:
    return Querystring()          # new instance every call
```

**Recommendation**  
Cache the instance, either as a class-level attribute or via `functools.cached_property` / `lru_cache`:

```python
@lru_cache(maxsize=1)
def _get_qs(self) -> Querystring:
    return Querystring()

@property
def qs(self) -> Querystring:
    return self._get_qs()
```

Or simply use a class-level default:

```python
_qs: Querystring = Querystring()

@property
def qs(self) -> Querystring:
    return self._qs
```

---

### CQ-006 — F-string formatting in `log.info` / `log.exception` calls

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/lib/tools/_beta_runner.py`, lines 200, 235, 356, 481, 516, 657 |
| **Severity** | Low |
| **Category** | Code Quality / Performance |

**Description**  
Several logging calls use f-strings for interpolation:

```python
log.info(f"Token usage {tokens_used} has exceeded the threshold of {threshold}. Performing compaction.")
log.exception(f"Error occurred while calling tool: {tool.name}", exc_info=exc)
```

Python's logging module supports lazy `%`-style formatting; f-strings are evaluated eagerly even when the log level means the message is never emitted. Additionally, passing `exc_info=exc` to `log.exception` is incorrect — `log.exception` already captures the current exception; passing an explicit `exc_info` overrides that behaviour in potentially unexpected ways. For `log.exception` the correct pattern is either `log.exception(msg)` (no `exc_info`) or `log.error(msg, exc_info=exc)`.

**Recommendation**  
Use `%`-style lazy formatting and fix the `exc_info` usage:

```python
log.info("Token usage %d has exceeded the threshold of %d. Performing compaction.", tokens_used, threshold)
log.error("Error occurred while calling tool: %s", tool.name, exc_info=exc)
```

---

### CQ-007 — Multiple unresolved `TODO` comments indicating incomplete or untested code

| Field | Detail |
|-------|--------|
| **Location** | Multiple files (see list below) |
| **Severity** | Low |
| **Category** | Code Quality |

**Description**  
The following `TODO` comments flag known gaps that have not been addressed:

| File | Line | Comment |
|------|------|---------|
| `_base_client.py` | 98 | `# TODO: make base page type vars covariant` |
| `_base_client.py` | 201 | `# TODO: do we have to preprocess params here?` |
| `_base_client.py` | 1548 | `# TODO(someday): support non asyncio runtimes here` |
| `_base_client.py` | 2247 | `# TODO: untested` (arm detection) |
| `_base_client.py` | 2254 | `# TODO: untested` (x32 detection) |
| `_models.py` | 432 | `# TODO` (bare, no description) |
| `_models.py` | 802 | `# TODO: condition is weird` |
| `_compat.py` | 73 | `# TODO: provide an error message here?` |
| `_utils/_transform.py` | 37–38 | Forward reference handling |
| `_utils/_transform.py` | 214, 380 | Field name collision edge-cases |
| `_qs.py` | 81 | `# TODO: error if unknown format` |
| `lib/streaming/_messages.py` | 457 | `# TODO: check index` |
| `lib/streaming/_beta_messages.py` | 477 | `# TODO: check index` |

**Recommendation**  
Create GitHub issues for each TODO to track them formally, or resolve them where straightforward (e.g. the bare `# TODO` at `_models.py:432` and the missing error message at `_compat.py:73`).

---

### CQ-008 — Debug logging dumps full response headers

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_base_client.py`, lines 1112–1119 |
| **Severity** | Low (see Security section for elevated concern) |
| **Category** | Code Quality |

**Description**  
The full `response.headers` object is logged at DEBUG level:

```python
log.debug(
    'HTTP Response: %s %s "%i %s" %s',
    request.method,
    request.url,
    response.status_code,
    response.reason_phrase,
    response.headers,          # all headers, including Set-Cookie, auth material, etc.
)
```

In test or CI environments where `ANTHROPIC_LOG=debug` is set, this may expose sensitive header values in log files. See also `security_issues.md`.

**Recommendation**  
Either redact known-sensitive headers before logging, or limit the logged information to header names only:

```python
log.debug(
    'HTTP Response: %s %s "%i %s" headers=%s',
    ...,
    list(response.headers.keys()),
)
```

---

### CQ-009 — `FinalRequestOptions` uses mutable `{}` as Pydantic field default

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_models.py`, line 859 |
| **Severity** | Low |
| **Category** | Code Quality |

**Description**  
```python
class FinalRequestOptions(pydantic.BaseModel):
    params: Query = {}
```

Pydantic v2 internally copies mutable defaults correctly, but Pydantic v1 does not always protect against shared state. This may not cause a runtime bug today (because `FinalRequestOptions` is always constructed via `construct()` which strips `NotGiven` and then re-builds), but it is fragile and flagged by linters.

**Recommendation**  
Use `default_factory`:

```python
from pydantic import Field
params: Query = Field(default_factory=dict)
```

---

## 3. Style

### STY-001 — `raise TypeError('"...')` — extra inner quotes in error messages

| Field | Detail |
|-------|--------|
| **Location** | `src/anthropic/_client.py`, lines 196–198 and 436–438 |
| **Severity** | Low |
| **Category** | Style |

**Description**  
The `TypeError` raised from `_validate_headers` includes spurious double-quote characters around the message:

```python
raise TypeError(
    '"Could not resolve authentication method. Expected either api_key ...'
    # ^ unnecessary leading double-quote
)
```

**Recommendation**  
Remove the extra quotes so the message reads cleanly:

```python
raise TypeError(
    "Could not resolve authentication method. Expected either api_key or auth_token "
    "to be set. Or for one of the `X-Api-Key` or `Authorization` headers to be "
    "explicitly omitted."
)
```

---

## 4. Open Questions / Design Observations

| # | Topic | Note |
|---|-------|------|
| 1 | `max_retries=math.inf` advertised but not validated | The error message suggests `math.inf` is supported, but there is no guard against non-integer `max_retries`. Passing `float('inf')` would cause `min(max_retries - remaining_retries, 1000)` to return `float` instead of `int`, potentially breaking downstream arithmetic. |
| 2 | `thinking` dict access without type guard | `messages.py` accesses `thinking["type"]` directly on the `ThinkingConfigParam` union type. Since `ThinkingConfigParam` is a `Union[TypedDict, ...]`, mypy/pyright accept this, but runtime access on an `Omit` instance (if passed) would raise `TypeError`. The guard `thinking and thinking["type"] == "enabled"` may mask this. |
| 3 | `MODELS_TO_WARN_WITH_THINKING_ENABLED = ["claude-opus-4-6"]` | This list contains a single model with a numeric suffix that does not match standard Anthropic model naming conventions (`claude-opus-4-5`, `claude-3-5-sonnet-*`, etc.). Verify whether this model identifier is accurate. |
