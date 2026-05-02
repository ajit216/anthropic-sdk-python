# Incomplete Implementations & TODO Items: anthropic-sdk-python

**Review Date:** 2026-05-02  
**Reviewer:** Automated Senior Python Developer Review  
**Codebase Version:** 0.89.0

---

## Summary

The codebase contains several TODO comments and acknowledged incomplete implementations. These represent acknowledged technical debt. The following are the most actionable items.

---

## Findings

### TODO-001 — `content_block_start` index validation missing in stream accumulation

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/lib/streaming/_messages.py`, line 457; `src/anthropic/lib/streaming/_beta_messages.py` (corresponding line) |
| **Category** | Bug / Incomplete |
| **Severity** | Medium |

**Description:**  
In `accumulate_event()`, when a `content_block_start` event is received, a comment explicitly notes the index is not validated:

```python
if event.type == "content_block_start":
    # TODO: check index
    current_snapshot.content.append(...)
```

Similarly for `content_block_delta`, the code accesses `current_snapshot.content[event.index]` without bounds-checking. If the API ever emits events out of order or with an unexpected index, this will raise an uncaught `IndexError` that propagates to the user as a non-informative exception rather than a well-formed SDK error.

**Recommendation:**  
Add index validation:

```python
if event.type == "content_block_start":
    if event.index != len(current_snapshot.content):
        raise RuntimeError(
            f"Unexpected content_block_start index {event.index}, "
            f"expected {len(current_snapshot.content)}"
        )
    current_snapshot.content.append(...)
```

For `content_block_delta`, validate before accessing:

```python
if event.index >= len(current_snapshot.content):
    raise RuntimeError(
        f"content_block_delta references out-of-bounds index {event.index} "
        f"(only {len(current_snapshot.content)} blocks accumulated so far)"
    )
```

---

### TODO-002 — `_get_extra_fields_type` not implemented for Pydantic v1

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_models.py`, lines 430–444 |
| **Category** | Code Quality / Incomplete |
| **Severity** | Low |

**Description:**  
The `_get_extra_fields_type()` helper returns `None` for Pydantic v1 with an explicit TODO comment:

```python
def _get_extra_fields_type(cls: type[pydantic.BaseModel]) -> type | None:
    if PYDANTIC_V1:
        # TODO
        return None
    ...
```

This means that for Pydantic v1, extra/unknown fields in model responses will not be coerced to their declared type (if any). While this is unlikely to cause breakage in practice (extra fields are stored as-is), it is an acknowledged gap in parity between Pydantic versions.

**Recommendation:**  
Implement the Pydantic v1 equivalent by inspecting `cls.__fields__` and checking the `Config.extra` setting for additional field type hints if applicable, or document that this behaviour intentionally differs.

---

### TODO-003 — `ARM` architecture detection untested

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_base_client.py`, lines 2247–2249 and 2253–2255 |
| **Category** | Code Quality / Incomplete |
| **Severity** | Low |

**Description:**  
Two branches in `get_architecture()` are explicitly marked as untested:

```python
# TODO: untested
if machine == "arm":
    return "arm"

# TODO: untested
if sys.maxsize <= 2**32:
    return "x32"
```

These are telemetry-only headers (`X-Stainless-Arch`) so incorrect values won't affect functionality, but the annotations indicate missing test coverage.

**Recommendation:**  
Add unit tests that mock `platform.machine()` to return `"arm"` and verify the correct `Arch` literal is returned. Similarly for `sys.maxsize` mocking for the `x32` case.

---

### TODO-004 — Non-deterministic error message in `required_args` validator

| Field       | Value |
|-------------|-------|
| **Location** | `src/anthropic/_utils/_utils.py`, line 275 |
| **Category** | Code Quality / Incomplete |
| **Severity** | Low |

**Description:**  
The `required_args` decorator has a TODO comment noting that the error message for a single-variant mismatch is non-deterministic:

```python
# TODO: this error message is not deterministic
missing = list(set(variants[0]) - given_params)
```

Since `set` iteration order is not guaranteed, the error message listing missing arguments may appear in different orders across Python runs, making it harder to write reliable assertions in tests.

**Recommendation:**  
Sort the missing list before generating the message:

```python
missing = sorted(set(variants[0]) - given_params)
```

---
