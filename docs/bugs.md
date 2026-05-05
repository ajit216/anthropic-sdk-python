# Confirmed Bugs: anthropic-sdk-python

**Review Date:** 2025-05-05  
**Reviewer:** Senior Python Developer (automated review)  
**Codebase Version:** 0.89.0  

This document lists confirmed bugs with reproduction steps and proposed fixes. See [`code_review.md`](./code_review.md) for the full review context.

---

## BUG-01 — Missing `f` prefix on f-string in `async_to_httpx_files`

**File:** `src/anthropic/_files.py`  
**Line:** 100  
**Severity:** High  
**Category:** Bug  

### Description

The `async_to_httpx_files` function raises a `TypeError` when an unsupported `files` argument type is supplied. The error message is intended to display the type of the invalid input, but the `f` prefix is absent — making `{type(files)}` a literal string instead of an interpolated expression.

### Affected Code

```python
# src/anthropic/_files.py  lines 91–101 (async version — BUG)
async def async_to_httpx_files(files: RequestFiles | None) -> HttpxRequestFiles | None:
    if files is None:
        return None

    if is_mapping_t(files):
        files = {key: await _async_transform_file(file) for key, file in files.items()}
    elif is_sequence_t(files):
        files = [(key, await _async_transform_file(file)) for key, file in files]
    else:
        raise TypeError("Unexpected file type input {type(files)}, expected mapping or sequence")
        #                ^^^ MISSING f prefix — {type(files)} is NOT interpolated
```

The synchronous equivalent on line 58 is correct:
```python
raise TypeError(f"Unexpected file type input {type(files)}, expected mapping or sequence")
```

### Reproduction

```python
import asyncio
from anthropic._files import async_to_httpx_files

async def repro():
    try:
        await async_to_httpx_files("not_a_mapping_or_sequence")  # type: ignore
    except TypeError as e:
        print(repr(str(e)))
        # Prints: 'Unexpected file type input {type(files)}, expected mapping or sequence'
        # Expected: 'Unexpected file type input <class 'str'>, expected mapping or sequence'

asyncio.run(repro())
```

### Fix

```python
# src/anthropic/_files.py  line 100 — add `f` prefix
raise TypeError(f"Unexpected file type input {type(files)}, expected mapping or sequence")
```

---

## BUG-02 — Dead code / logic error in `_validate_headers`

**File:** `src/anthropic/_client.py`  
**Lines:** 185–198 (`Anthropic`) and 425–438 (`AsyncAnthropic`)  
**Severity:** Medium  
**Category:** Bug  

### Description

The `_validate_headers` method intends to:
1. Pass when either auth header is present in the resolved headers.
2. Pass when the caller has explicitly set either auth header to `Omit` (to suppress validation).
3. Raise when no auth can be resolved.

The current implementation achieves goal (1) correctly on line 186, but goals (2) and (3) are partially broken because lines 190 and 193 repeat the same header-presence check that already passed on line 186. This means the `isinstance(..., Omit)` check — the only logic that is _new_ on those lines — can only be reached when **neither** `X-Api-Key` nor `Authorization` is set in the resolved headers. The sub-expression `headers.get("X-Api-Key")` on line 190 will therefore always be falsy when reached, making these conditions effectively:

```
if isinstance(custom_headers.get("X-Api-Key"), Omit):   # the only relevant check
if isinstance(custom_headers.get("Authorization"), Omit): # the only relevant check
```

The `headers.get(...)` parts are dead code, which obscures intent and is misleading to future maintainers.

### Affected Code

```python
# src/anthropic/_client.py  lines 185–198 (identical issue in AsyncAnthropic)
def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
    if headers.get("Authorization") or headers.get("X-Api-Key"):
        # valid
        return

    if headers.get("X-Api-Key") or isinstance(custom_headers.get("X-Api-Key"), Omit):
        #  ^^^ DEAD: headers.get("X-Api-Key") is always falsy here
        return

    if headers.get("Authorization") or isinstance(custom_headers.get("Authorization"), Omit):
        #  ^^^ DEAD: headers.get("Authorization") is always falsy here
        return

    raise TypeError(
        '"Could not resolve authentication method. ...'  # note: extra quotes around message
    )
```

### Fix

Remove the dead sub-expressions and clarify the intent:

```python
def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
    if headers.get("Authorization") or headers.get("X-Api-Key"):
        # An auth header is present in the resolved headers — valid.
        return

    if isinstance(custom_headers.get("X-Api-Key"), Omit):
        # Caller explicitly omitted X-Api-Key — respect that decision.
        return

    if isinstance(custom_headers.get("Authorization"), Omit):
        # Caller explicitly omitted Authorization — respect that decision.
        return

    raise TypeError(
        "Could not resolve authentication method. Expected either api_key or auth_token to be set. "
        "Or for one of the `X-Api-Key` or `Authorization` headers to be explicitly omitted"
    )
```

Note: The fix also removes the extraneous double-quotes from the error message string (see CQ-04 in `code_review.md`).

---

## Potential Bug — `_construct_field` discards explicit `None`

**File:** `src/anthropic/_models.py`  
**Lines:** 415–417  
**Severity:** Medium  
**Category:** Possible Bug (depends on design intent)  

### Description

```python
def _construct_field(value: object, field: FieldInfo, key: str) -> object:
    if value is None:
        return field_get_default(field)   # silently replaces None with default
```

When a caller explicitly provides `None` for a field via `BaseModel.construct(field_name=None)`, the function discards `None` and returns the field's default value instead. This deviates from Pydantic's standard behaviour where `None` is a valid field value for optional/nullable fields.

### Reproduction

```python
from anthropic._models import BaseModel
from typing import Optional

class MyModel(BaseModel):
    name: Optional[str] = "default_name"

# Attempt to explicitly set name=None
m = MyModel.construct(name=None)
print(m.name)  # Prints: "default_name"  (expected: None)
```

### Assessment

This may be intentional for Pydantic v1 compatibility (where `None` on a required field could cause issues), but it silently swallows a programmer's explicit intent and may cause subtle bugs for optional fields that should be `None`. If intentional, this behaviour should be clearly documented.

### Recommendation

Either:
1. Remove the `if value is None: return field_get_default(field)` guard and trust the caller (aligns with standard Pydantic `construct` behaviour).
2. Keep the guard but add a docstring/comment explaining why `None` values are replaced with defaults.
