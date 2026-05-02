# Code Review: anthropic-sdk-python

**Review Date:** 2026-05-02  
**Reviewer:** Automated Senior Python Developer Review  
**Codebase Version:** 0.89.0  
**Scope:** Full codebase review — `src/anthropic/` and related modules

---

## Executive Summary

The Anthropic Python SDK is a well-structured, auto-generated SDK with solid retry logic, streaming support, AWS/Vertex/Bedrock integrations, and Pydantic v1/v2 compatibility. The review identified **4 bugs** (including 2 high-severity), **5 code quality issues**, **3 security concerns**, and **4 incomplete/TODO areas**. None of the identified bugs would cause a catastrophic failure, but two of them produce silent, hard-to-debug mismatches between user expectations and runtime behaviour.

---

## Findings

### BUG-001 — Missing `f` prefix in error message string

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_files.py`, line 100 |
| **Category** | Bug |
| **Severity** | Medium |

**Description:**  
In `async_to_httpx_files()`, the `TypeError` message is a plain string literal that looks like an f-string but lacks the `f` prefix. The `{type(files)}` expression is **never interpolated**; it is emitted as literal text.

```python
# Line 58 (sync version — correct):
raise TypeError(f"Unexpected file type input {type(files)}, expected mapping or sequence")

# Line 100 (async version — BUG, missing f prefix):
raise TypeError("Unexpected file type input {type(files)}, expected mapping or sequence")
```

**Recommendation:**  
Add the `f` prefix to the string on line 100 so it matches the sync counterpart on line 58.

---

### BUG-002 — `ServiceUnavailableError` (503) and `DeadlineExceededError` (504) are never raised

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_client.py`, `_make_status_error()` (lines 261–290 and 501–530) |
| **Category** | Bug |
| **Severity** | High |

**Description:**  
`_exceptions.py` defines `ServiceUnavailableError` (status 503) and `DeadlineExceededError` (status 504) as distinct exception subclasses of `APIStatusError`. However, `_make_status_error()` in both `Anthropic` and `AsyncAnthropic` never maps these status codes to those classes. The check is:

```python
if response.status_code >= 500:
    return _exceptions.InternalServerError(err_msg, response=response, body=body)
```

Any 503 or 504 response is silently caught as `InternalServerError`, making it impossible for users to specifically catch `ServiceUnavailableError` or `DeadlineExceededError`.

**Recommendation:**  
Add explicit checks before the catch-all `>= 500` block:

```python
if response.status_code == 503:
    return _exceptions.ServiceUnavailableError(err_msg, response=response, body=body)

if response.status_code == 504:
    return _exceptions.DeadlineExceededError(err_msg, response=response, body=body)

if response.status_code >= 500:
    return _exceptions.InternalServerError(err_msg, response=response, body=body)
```

---

### BUG-003 — `copy()` uses falsy-or-fallback for `api_key` / `auth_token`

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_client.py`, lines 238–239 and 478–479 |
| **Category** | Bug |
| **Severity** | High |

**Description:**  
The `copy()` method on both `Anthropic` and `AsyncAnthropic` uses Python's `or` operator to determine the new client's credentials:

```python
return self.__class__(
    api_key=api_key or self.api_key,
    auth_token=auth_token or self.auth_token,
    ...
)
```

If a caller explicitly passes `api_key=""` (an empty string, which is falsy) to intentionally clear the key, the fallback to `self.api_key` silently overrides the intent. While an empty API key would eventually fail at the API level, the silent override means the behaviour is different from what the user intended and makes the `copy()` contract confusing. The same applies to `auth_token`.

**Recommendation:**  
Use a `None`-sentinel check instead of the `or` operator, matching how other arguments are handled:

```python
api_key=api_key if api_key is not None else self.api_key,
auth_token=auth_token if auth_token is not None else self.auth_token,
```

---

### BUG-004 — `_construct_field` silently drops explicit `None` values

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_models.py`, lines 415–427 (`_construct_field`) |
| **Category** | Bug |
| **Severity** | Medium |

**Description:**  
The `_construct_field()` helper used by `BaseModel.construct()` replaces an explicit `None` value with the field's default:

```python
def _construct_field(value: object, field: FieldInfo, key: str) -> object:
    if value is None:
        return field_get_default(field)
    ...
```

For optional/nullable fields (e.g., `Optional[str]`), a caller that passes `None` explicitly (to set the field to null) will instead get the field's default. This silently discards intentional `None` values and can produce unexpected model states.

**Recommendation:**  
Only fall back to the default when the key was not present in the `values` dict. The existing code in `BaseModel.construct()` already tracks which keys are present in `values`:

```python
def _construct_field(value: object, field: FieldInfo, key: str) -> object:
    # Do not replace None with default — None is a valid value for Optional fields.
    if PYDANTIC_V1:
        type_ = cast(type, field.outer_type_)
    else:
        type_ = field.annotation
    if type_ is None:
        raise RuntimeError(f"Unexpected field type is None for {key}")
    return construct_type(value=value, type_=type_, metadata=getattr(field, "metadata", None))
```

The default-filling logic should be applied at the call site in `construct()` only when `key not in values`, which is already the case for the `else` branch — the `if value is None` guard inside `_construct_field` is redundant and incorrect.

---

### BUG-005 — `PageInfo.__repr__` uses truthiness on `URL` / `Body`, not `is not NotGiven`

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py`, lines 166–171 |
| **Category** | Bug |
| **Severity** | Low |

**Description:**  
`PageInfo.__repr__` tests `if self.url:` and `if self.json:`, which evaluates truthiness rather than checking whether the sentinel `NotGiven` was set. An `httpx.URL` object or a `Body` value of `{}` (empty dict) would both evaluate to `False`, making `__repr__` fall through to `params=` even when a `url` or `json` was actually provided.

```python
def __repr__(self) -> str:
    if self.url:          # BUG: empty URL evaluates False
        return f"{self.__class__.__name__}(url={self.url})"
    if self.json:         # BUG: empty dict evaluates False
        return f"{self.__class__.__name__}(json={self.json})"
    return f"{self.__class__.__name__}(params={self.params})"
```

**Recommendation:**  
```python
def __repr__(self) -> str:
    if not isinstance(self.url, NotGiven):
        return f"{self.__class__.__name__}(url={self.url})"
    if not isinstance(self.json, NotGiven):
        return f"{self.__class__.__name__}(json={self.json})"
    return f"{self.__class__.__name__}(params={self.params})"
```

---

### QUALITY-001 — Redundant (unreachable) checks in `_validate_headers`

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_client.py`, lines 185–198 and 425–438 (both `Anthropic` and `AsyncAnthropic`) |
| **Category** | Code Quality |
| **Severity** | Low |

**Description:**  
The `_validate_headers` method performs duplicate checks. The first `if` already handles both `Authorization` and `X-Api-Key`. The two subsequent checks for each header individually are unreachable because they repeat what the first condition already handled:

```python
def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
    if headers.get("Authorization") or headers.get("X-Api-Key"):
        return  # catches both

    # These two are UNREACHABLE — the cases are already covered above:
    if headers.get("X-Api-Key") or isinstance(custom_headers.get("X-Api-Key"), Omit):
        return
    if headers.get("Authorization") or isinstance(custom_headers.get("Authorization"), Omit):
        return
    ...
```

However, the `isinstance(custom_headers.get(...), Omit)` branches **are** reachable and meaningful. The first `if` block (lines 186–188) should be removed or merged to avoid the confusion.

**Recommendation:**  
Consolidate into:

```python
def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
    if headers.get("Authorization") or headers.get("X-Api-Key"):
        return
    if isinstance(custom_headers.get("X-Api-Key"), Omit):
        return
    if isinstance(custom_headers.get("Authorization"), Omit):
        return
    raise TypeError(
        "Could not resolve authentication method. Expected either api_key or auth_token to be set. "
        "Or for one of the `X-Api-Key` or `Authorization` headers to be explicitly omitted"
    )
```

Note also: the error message is wrapped in an extra pair of double-quotes (`'"...'"`), which appears to be a formatting mistake.

---

### QUALITY-002 — Unnecessary f-strings for static string literals

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_client.py`, lines 103 and 343 |
| **Category** | Code Quality |
| **Severity** | Low |

**Description:**  
Both `Anthropic.__init__` and `AsyncAnthropic.__init__` assign:

```python
base_url = f"https://api.anthropic.com"
```

The `f` prefix has no effect — there is no expression to interpolate. This is misleading.

**Recommendation:**  
Remove the `f` prefix:

```python
base_url = "https://api.anthropic.com"
```

---

### QUALITY-003 — Mutable default argument `options: RequestOptions = {}`

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py` and various resource files (25+ occurrences) |
| **Category** | Code Quality |
| **Severity** | Low |

**Description:**  
Multiple method signatures use a mutable empty dict as a default argument:

```python
def get(self, path: str, *, cast_to: ..., options: RequestOptions = {}, ...) -> ...:
```

In Python, mutable default arguments are shared across all calls. If any code path mutates `options`, the default would be corrupted for subsequent calls. While the current implementation does not appear to mutate the `options` dict directly, this is a fragile pattern that violates PEP 8 best practices.

**Recommendation:**  
Use `None` as the default and create an empty dict inside the function body:

```python
def get(self, path: str, *, cast_to: ..., options: RequestOptions | None = None, ...) -> ...:
    if options is None:
        options = {}
```

---

### QUALITY-004 — `qs` property creates new `Querystring` instance on every request

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py`, lines 674–676 |
| **Category** | Performance |
| **Severity** | Low |

**Description:**  
The `qs` property is called on every outgoing request to stringify query parameters. Each call constructs a new `Querystring()` object:

```python
@property
def qs(self) -> Querystring:
    return Querystring()
```

**Recommendation:**  
Cache the instance (e.g., as a private attribute set in `__init__`, or use `functools.cached_property`) to avoid repeated instantiation.

---

### QUALITY-005 — Deprecated model list contains potentially stale EOL dates

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/resources/messages/messages.py`, lines 59–73 |
| **Category** | Code Quality |
| **Severity** | Low |

**Description:**  
`DEPRECATED_MODELS` is a hardcoded dict mapping model names to EOL date strings. Several EOL dates are in the past (e.g., `"claude-3-7-sonnet-latest"` and `"claude-3-7-sonnet-20250219"` with EOL `"February 19th, 2026"`, and `"claude-3-opus-20240229"` with EOL `"January 5th, 2026"`). The deprecation warning fires regardless of whether the EOL date has passed, which means users may be told a model is "deprecated and will reach end-of-life" when it already has. The warning message should distinguish between "will be deprecated" and "is already past end-of-life".

**Recommendation:**  
Parse EOL dates and distinguish the warning message based on whether the EOL is in the future or past:

```python
import datetime

eol_str = DEPRECATED_MODELS[model]
# ... parse eol_str and check against datetime.date.today() ...
if today >= eol_date:
    warnings.warn(f"The model '{model}' reached end-of-life on {eol_str} ...")
else:
    warnings.warn(f"The model '{model}' is deprecated and will reach end-of-life on {eol_str} ...")
```

---
