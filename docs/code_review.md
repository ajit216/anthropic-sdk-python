# Code Review: anthropic-sdk-python

**Reviewed version:** 0.89.0  
**Review date:** 2025-05-12  
**Reviewer:** Senior Software Developer (automated review)

---

## Summary

This document captures all issues identified during a systematic review of the `anthropic-sdk-python` SDK codebase. Issues are organized by category and severity.

For security-specific concerns see [docs/security_issues.md](./security_issues.md).

---

## Table of Contents

1. [Bugs](#1-bugs)
2. [Code Quality](#2-code-quality)
3. [Performance](#3-performance)
4. [Security](#4-security)
5. [Style / Maintainability](#5-style--maintainability)
6. [Unresolved TODOs](#6-unresolved-todos)

---

## 1. Bugs

### BUG-001 — Missing f-string prefix in `async_to_httpx_files` error message

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_files.py`, line 100 |
| **Category** | Bug |
| **Severity** | Medium |

**Description:**  
`async_to_httpx_files()` raises a `TypeError` with a hard-coded literal string that contains `{type(files)}` but is **not** prefixed with `f`. The synchronous counterpart `to_httpx_files()` at line 58 correctly uses an f-string. The result is that async callers receive the useless error message `"Unexpected file type input {type(files)}, expected mapping or sequence"` instead of the actual type.

```python
# line 100 (WRONG — missing f-prefix)
raise TypeError("Unexpected file type input {type(files)}, expected mapping or sequence")

# line 58 (CORRECT)
raise TypeError(f"Unexpected file type input {type(files)}, expected mapping or sequence")
```

**Recommendation:**  
Add the `f` prefix to the string on line 100.

```python
raise TypeError(f"Unexpected file type input {type(files)}, expected mapping or sequence")
```

---

### BUG-002 — Extra outer double-quotes in authentication error message

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_client.py`, lines 197 and 437 |
| **Category** | Bug |
| **Severity** | Low |

**Description:**  
Both `Anthropic._validate_headers()` and `AsyncAnthropic._validate_headers()` raise a `TypeError` where the error message string is itself **wrapped in double-quotes**, producing a message that begins and ends with a literal `"` character:

```python
raise TypeError(
    '"Could not resolve authentication method. Expected either api_key …"'
)
```

Users and log aggregators receive: `"Could not resolve authentication method. …"` (with extra quotes), which is confusing.

**Recommendation:**  
Remove the outer double-quotes from the message string in both occurrences:

```python
raise TypeError(
    "Could not resolve authentication method. Expected either api_key or auth_token to be set. "
    "Or for one of the `X-Api-Key` or `Authorization` headers to be explicitly omitted"
)
```

---

### BUG-003 — Redundant / dead conditions in `_validate_headers`

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_client.py`, lines 185-198 and 425-438 |
| **Category** | Bug |
| **Severity** | Low |

**Description:**  
`_validate_headers` in both `Anthropic` and `AsyncAnthropic` has three `return` guards:

```python
def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
    if headers.get("Authorization") or headers.get("X-Api-Key"):   # (1)
        return

    if headers.get("X-Api-Key") or isinstance(...):                 # (2) — X-Api-Key already checked above
        return

    if headers.get("Authorization") or isinstance(...):             # (3) — Authorization already checked above
        return
```

Conditions (2) and (3) can **never** return early via `headers.get(...)` because those same checks are covered (and would have returned) at guard (1). The only meaningful parts of guards (2) and (3) are the `isinstance(custom_headers.get(...), Omit)` checks, which could be combined into a single guard.

**Recommendation:**  
Simplify to:

```python
def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
    if headers.get("Authorization") or headers.get("X-Api-Key"):
        return
    if isinstance(custom_headers.get("X-Api-Key"), Omit):
        return
    if isinstance(custom_headers.get("Authorization"), Omit):
        return
    raise TypeError(
        "Could not resolve authentication method. ..."
    )
```

---

### BUG-004 — `copy()` / `with_options()` cannot clear `api_key` or `auth_token` to `None`

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_client.py`, lines 238-239 and 478-479 |
| **Category** | Bug |
| **Severity** | Medium |

**Description:**  
The `copy()` method in both `Anthropic` and `AsyncAnthropic` uses Python's `or` operator to fall back to the existing value:

```python
api_key=api_key or self.api_key,
auth_token=auth_token or self.auth_token,
```

Because `None` is falsy, a caller that explicitly passes `api_key=None` intending to create a client with no API key (e.g., switching to bearer auth only) will silently receive the **old** `api_key` value. The same applies to `auth_token`. This is inconsistent with how `base_url` is handled (uses `or`) and with how `timeout` and `max_retries` are handled (use sentinel `NotGiven`).

**Recommendation:**  
Use a sentinel (e.g., `NotGiven`) to distinguish "not provided" from `None`, matching the approach used elsewhere in the SDK:

```python
def copy(
    self,
    *,
    api_key: str | None | NotGiven = not_given,
    auth_token: str | None | NotGiven = not_given,
    ...
):
    ...
    return self.__class__(
        api_key=self.api_key if isinstance(api_key, NotGiven) else api_key,
        auth_token=self.auth_token if isinstance(auth_token, NotGiven) else auth_token,
        ...
    )
```

---

### BUG-005 — Unnecessary f-string on static string literal for base URL

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_client.py`, lines 103 and 343 |
| **Category** | Bug / Style |
| **Severity** | Low |

**Description:**  
Both `Anthropic.__init__` and `AsyncAnthropic.__init__` use an f-string on a plain static string:

```python
base_url = f"https://api.anthropic.com"
```

There are no interpolated values, so the `f` prefix is unnecessary. While harmless, this may confuse readers and triggers linter warnings in some configurations.

**Recommendation:**  
Remove the `f` prefix:

```python
base_url = "https://api.anthropic.com"
```

---

### BUG-006 — `assert` statements used as control flow guards in production code

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py`, lines 1147 and 1787; `src/anthropic/lib/streaming/_messages.py`, lines 94, 125, 242, 273 |
| **Category** | Bug |
| **Severity** | Low |

**Description:**  
`assert` statements are used in production code paths. Python's optimized mode (`python -O`) disables assertions, which would silently hide `None` dereferences and snapshot access violations:

```python
# _base_client.py
assert response is not None, "could not resolve response (should never happen)"

# lib/streaming/_messages.py
assert self.__final_message_snapshot is not None
```

**Recommendation:**  
Replace `assert` with explicit `if`/`raise` guards using appropriate exception types (e.g., `RuntimeError`):

```python
if response is None:
    raise RuntimeError("could not resolve response (should never happen)")
```

---

## 2. Code Quality

### CQ-001 — Mutable default argument `{}` used for `options` parameters

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py`, ~25 occurrences in `SyncAPIClient` and `AsyncAPIClient` HTTP methods |
| **Category** | Code Quality |
| **Severity** | Low |

**Description:**  
Multiple method signatures use `options: RequestOptions = {}` as a default argument. In Python, mutable default arguments are shared across all calls; if any code path mutates the dict, subsequent calls with the default will see the modified value. The ruff lint rule `B006` is intentionally suppressed (`"B006"` in `ruff.lint.ignore`), suggesting the team is aware. However, it is worth confirming that no mutation of `options` occurs when the default is used.

**Recommendation:**  
The safest fix is to change defaults to `None` and replace with `{}` inside the function body:

```python
def get(self, path: str, *, cast_to: Type[ResponseT], options: RequestOptions | None = None, ...) -> ResponseT:
    opts = FinalRequestOptions.construct(method="get", url=path, **(options or {}))
```

Alternatively, confirm in a comment that the passed dict is never mutated and keep B006 suppressed.

---

### CQ-002 — Inconsistent error handling: bare `except Exception` swallows all exceptions

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py`, lines 426, 912, 1550; `src/anthropic/lib/streaming/_messages.py` (indirect) |
| **Category** | Code Quality |
| **Severity** | Low |

**Description:**  
Several `__del__` methods and platform-detection helpers use bare `except Exception: pass` which silently swallows all errors, including `KeyboardInterrupt` (in Python 3 `Exception` does not catch `BaseException`, but `Exception` can still swallow important errors):

```python
class SyncHttpxClientWrapper(DefaultHttpxClient):
    def __del__(self) -> None:
        ...
        try:
            self.close()
        except Exception:
            pass   # no logging
```

**Recommendation:**  
For `__del__` finalizers this is generally acceptable, but consider at least logging the exception at DEBUG level to aid diagnostics:

```python
except Exception:
    log.debug("Error closing HTTP client in __del__", exc_info=True)
```

---

### CQ-003 — Duplicate code between `Anthropic` and `AsyncAnthropic`

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_client.py` |
| **Category** | Code Quality |
| **Severity** | Low |

**Description:**  
`Anthropic` and `AsyncAnthropic` share nearly identical implementations for:
- `__init__` (environment variable resolution, base URL defaulting)
- `_api_key_auth`, `_bearer_auth` properties
- `_validate_headers`
- `copy()` / `with_options()`
- `_make_status_error()`

Any bug fixed in one class (e.g., BUG-001, BUG-002 above) must be manually replicated in the other.

**Recommendation:**  
Extract shared logic into the `BaseClient` class or a shared mixin, leaving only async/sync specific overrides in the subclasses.

---

### CQ-004 — `TODO` comments left in production code without tracking issues

| Attribute   | Value |
|-------------|-------|
| **Location** | Multiple files: `_base_client.py` (lines 98, 201, 586, 602, 1548, 2247, 2254), `_models.py` (line 432, 802), `_utils/_transform.py` (lines 37–38, 214, 380), `_compat.py` (line 73), `_qs.py` (line 81), `lib/streaming/_messages.py` (line 457), `lib/streaming/_beta_messages.py` (line 477) |
| **Category** | Code Quality |
| **Severity** | Low |

**Description:**  
The codebase contains numerous `# TODO` comments noting unresolved design decisions or missing implementations without tracking references (e.g., GitHub issue numbers). Examples:

- `# TODO: untested` (`_base_client.py:2247, 2254`) — architecture for ARM and x32 detection is untested.
- `# TODO: check index` (`_messages.py:457`) — content-block index is not validated.
- `# TODO: condition is weird` (`_models.py:802`) — acknowledged code smell.

**Recommendation:**  
Create GitHub issues for each TODO and reference the issue number in the comment:

```python
# TODO(#123): check index
```

---

### CQ-005 — `AsyncHttpxClientWrapper.__del__` may silently fail to close connections

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py`, lines 1542–1552 |
| **Category** | Code Quality |
| **Severity** | Medium |

**Description:**  
`AsyncHttpxClientWrapper.__del__` attempts to schedule `aclose()` on the running event loop:

```python
def __del__(self) -> None:
    if self.is_closed:
        return
    try:
        asyncio.get_running_loop().create_task(self.aclose())
    except Exception:
        pass
```

This approach has several issues:
1. **No running loop**: If the client outlives the event loop (e.g., in tests or synchronous cleanup), `get_running_loop()` raises `RuntimeError`. The `except Exception` silently eats this and the connection is leaked.
2. **Comment acknowledges incompleteness**: `# TODO(someday): support non asyncio runtimes here` — anyio runtimes (trio) are not handled.
3. **`create_task` is fire-and-forget**: Even if a loop is running, there is no guarantee the task will complete before program exit.

**Recommendation:**  
Document clearly that callers are expected to use the async context manager (`async with`) to guarantee cleanup. Optionally emit a `ResourceWarning` when the object is garbage-collected unclosed (mirroring the stdlib `socket` pattern).

---

### CQ-006 — `_construct_field` returns `field_get_default` when `value is None`

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_models.py`, lines 415-427 |
| **Category** | Code Quality |
| **Severity** | Medium |

**Description:**  
`_construct_field()` treats `None` values as absent:

```python
def _construct_field(value: object, field: FieldInfo, key: str) -> object:
    if value is None:
        return field_get_default(field)
    ...
```

This means an API response field that is explicitly `null` will be silently replaced by the field's default value rather than `None`. For Optional fields this produces incorrect results where `None` (explicitly set by the API) is indistinguishable from "not set".

**Recommendation:**  
Use a sentinel (`NotGiven`) to distinguish "absent" from `None`, or only substitute the default when the key was truly absent in the source data. This is a deeper design issue that requires careful analysis of how `construct` is called throughout the SDK.

---

## 3. Performance

### PERF-001 — `_get_annotated_type` LRU cache key is the full type; large union types may thrash

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_utils/_transform.py`, line 115 |
| **Category** | Performance |
| **Severity** | Low |

**Description:**  
`_get_annotated_type` is decorated with `@lru_cache(maxsize=8096)`. The cache key is the Python `type` object, which uses identity (`id()`). For dynamically created generic aliases (e.g., `List[Union[FooType, BarType, ...]]` generated per-request), each call creates a new object that is not cached and pollutes the cache, eventually evicting useful entries.

**Recommendation:**  
This is likely already well-understood given the explicit `maxsize=8096` value. Consider monitoring cache hit rate in performance-sensitive workloads and document the cache strategy.

---

### PERF-002 — `_transform_typeddict` calls `get_type_hints()` on every transform

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/_utils/_transform.py`, lines 268, 434 |
| **Category** | Performance |
| **Severity** | Low |

**Description:**  
`get_type_hints()` is already wrapped in an `lru_cache` (line 450), which mitigates the cost. However, the inner `_transform_recursive` function iterates over all subtypes of a union and recursively calls `_transform_typeddict`, which re-invokes `get_type_hints()`. For deeply nested payloads, this may result in redundant cache lookups.

**Recommendation:**  
This is generally acceptable for SDK use cases. No immediate action required; note for future profiling.

---

## 4. Security

See the dedicated [docs/security_issues.md](./security_issues.md) file.

---

## 5. Style / Maintainability

### STY-001 — Inconsistent use of `Union[X, Y]` vs `X | Y` syntax

| Attribute   | Value |
|-------------|-------|
| **Location** | Throughout `src/anthropic/` |
| **Category** | Style |
| **Severity** | Low |

**Description:**  
The codebase uses both the older `Union[X, Y]` / `Optional[X]` form and the newer `X | Y` / `X | None` form. Since the package requires Python 3.9+ (`requires-python = ">= 3.9"`) and uses `from __future__ import annotations`, the PEP 604 `|` syntax is forward-compatible but `from __future__ import annotations` is required for it to work at runtime.

**Recommendation:**  
Standardise on the `X | Y` syntax across the codebase for new code. Existing code can be migrated incrementally.

---

### STY-002 — `Plan-836.md` artifact file committed at wrong path

| Attribute   | Value |
|-------------|-------|
| **Location** | `src/anthropic/docs/Plan-836.md` and `docs/Plan-836.md` |
| **Category** | Style |
| **Severity** | Low |

**Description:**  
A planning document (`Plan-836.md`) for a future `ConversationManager` helper is committed inside `src/anthropic/docs/` as well as `docs/`. This is an internal planning artifact and should not be part of the published package (it is included in the wheel by the `src/*` glob in `pyproject.toml`).

**Recommendation:**  
Move the file to `docs/` only, and add an exclusion rule in `pyproject.toml` under `[tool.hatch.build]` to prevent it from being shipped in the wheel:

```toml
[tool.hatch.build]
exclude = [
  "src/anthropic/docs/",
]
```

---

## 6. Unresolved TODOs

The following table lists unresolved TODO items in the source code. Each should be converted into a tracked GitHub issue.

| File | Line | TODO Text | Priority |
|------|------|-----------|----------|
| `_base_client.py` | 98 | `make base page type vars covariant` | Low |
| `_base_client.py` | 201 | `do we have to preprocess params here?` | Low |
| `_base_client.py` | 1548 | `support non asyncio runtimes here` (anyio/trio) | Medium |
| `_base_client.py` | 2247, 2254 | `untested` ARM/x32 platform detection | Low |
| `_models.py` | 432 | Pydantic v1 extra fields type annotation | Low |
| `_models.py` | 802 | `condition is weird` — `elif not TYPE_CHECKING` | Medium |
| `_utils/_transform.py` | 37–38 | Forward reference support | Medium |
| `_utils/_transform.py` | 214, 380 | Union field name collision edge cases | Medium |
| `_compat.py` | 73 | Missing error message | Low |
| `_qs.py` | 81 | Unknown array format not raising error | Low |
| `lib/streaming/_messages.py` | 457 | Content block index validation | Medium |
| `lib/streaming/_beta_messages.py` | 477 | Content block index validation | Medium |

---

*For security-specific findings, see [docs/security_issues.md](./security_issues.md).*
