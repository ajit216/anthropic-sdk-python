# Code Review: anthropic-sdk-python

**Review Date:** 2026-05-02  
**Reviewer:** Automated Senior Python Developer Review  
**Scope:** Full codebase review of `src/anthropic/`

---

## Summary

This review covers bugs, code quality problems, performance issues, security concerns, and style issues identified across the `anthropic-sdk-python` SDK. Findings are categorised by severity and type. Each issue includes the file location, category, description, and a concrete recommendation.

---

## Table of Contents

1. [Bug: Missing f-string prefix in `_files.py`](#bug-1)
2. [Security: Debug logs expose response headers and body](#bug-security-logging)
3. [Security: Credentials cached with `lru_cache` without expiry](#security-credentials-cache)
4. [Code Quality: Duplicate HTTP client initialisation logic](#code-quality-http-client-dup)
5. [Code Quality: Duplicate `_check_and_compact` in tool runner](#code-quality-compact-dup)
6. [Code Quality: Duplicate auth helpers across bedrock/aws](#code-quality-auth-dup)
7. [Code Quality: Redundant condition logic in `_validate_headers`](#code-quality-validate-headers)
8. [Code Quality: Mutable default argument in HTTP method signatures](#code-quality-mutable-default)
9. [Code Quality: `qs` property instantiates a new object on every call](#code-quality-qs-property)
10. [Bug: `AsyncHttpxClientWrapper.__del__` only supports asyncio](#bug-del-asyncio)
11. [Bug: `log.exception` with `exc_info=exc` outside exception handler](#bug-log-exception)
12. [Style: Unnecessary f-strings for string literals](#style-f-strings)
13. [Style: Jitter comment does not match implementation](#style-jitter-comment)
14. [Code Quality: `coerce_boolean` is case-sensitive](#code-quality-coerce-boolean)
15. [Code Quality: `PageInfo.__repr__` truthiness check on URL](#code-quality-pageinfo-repr)
16. [Code Quality: Stale TODO comments](#code-quality-todos)
17. [Performance: `SSEDecoder._iter_chunks` byte accumulation](#performance-sse-chunks)

---

## Findings

---

### <a name="bug-1"></a>1. Bug: Missing `f`-prefix in error string in `_files.py`

- **Location:** `src/anthropic/_files.py`, line 100
- **Category:** Bug
- **Severity:** High

**Description:**  
The error message in `async_to_httpx_files` contains a `{type(files)}` placeholder but the string literal is missing the `f` prefix. The curly-brace expression will be printed literally, providing no diagnostic value when debugging unexpected file type inputs.

```python
# Current (broken)
raise TypeError("Unexpected file type input {type(files)}, expected mapping or sequence")

# Compare with the correctly formatted sync counterpart at line 58
raise TypeError(f"Expected query input to be a dictionary for multipart requests but got {type(json_data)} instead.")
```

**Recommendation:**  
Add the `f` prefix to the string:
```python
raise TypeError(f"Unexpected file type input {type(files)}, expected mapping or sequence")
```

---

### <a name="bug-security-logging"></a>2. Security: Debug logs expose full response headers and body

- **Location:** `src/anthropic/_base_client.py`, lines 1112–1120 (`SyncAPIClient.request`), lines 1752–1760 (`AsyncAPIClient.request`)
- **Category:** Security
- **Severity:** High

**Description:**  
At `DEBUG` level, the entire response headers dict (including `Set-Cookie`, authorisation tokens, or bearer tokens from the server) and the error body (which may contain user PII, account details, or sensitive error metadata) are written to the log.

```python
log.debug(
    'HTTP Response: %s %s "%i %s" %s',
    request.method,
    request.url,
    response.status_code,
    response.reason_phrase,
    response.headers,   # <— full header dict, may include sensitive fields
)
log.debug("request_id: %s", response.headers.get("request-id"))
```

Additionally, in `_make_status_error_from_response` (line 425), the raw error body is formatted into `err_msg` and logged at `log.debug("Re-raising status error")` indirectly through exception messages.

**Recommendation:**  
- Redact or allowlist headers before logging. Only log safe headers (e.g., `Content-Type`, `request-id`, `x-ratelimit-*`).
- Do not include the full body in error message strings that may reach log handlers. Keep sensitive API error detail in the exception's `body` attribute rather than the exception message.

---

### <a name="security-credentials-cache"></a>3. Security: AWS credentials cached indefinitely via `lru_cache`

- **Location:** `src/anthropic/lib/bedrock/_auth.py`, lines 13–30; `src/anthropic/lib/aws/_auth.py`, lines 13–30
- **Category:** Security
- **Severity:** Medium

**Description:**  
`_get_session` is decorated with `@lru_cache(maxsize=512)`. The boto3 `Session` object can contain short-lived STS credentials that expire. Because the cache has no TTL, a session created with a temporary token that later expires will continue to be returned from the cache, potentially causing request failures or, in edge cases where credential material is compared by identity, incorrect re-use.

```python
@lru_cache(maxsize=512)
def _get_session(
    *,
    aws_access_key: str | None,
    ...
    aws_session_token: str | None,  # <— STS tokens are time-limited
    ...
) -> boto3.Session:
    ...
```

**Recommendation:**  
Use a TTL-aware cache (e.g., `cachetools.TTLCache`) or refresh credentials on each call while caching only the session object itself (not the frozen credentials). Alternatively, cap the cache entry lifetime to a conservative duration (e.g., 15 minutes) consistent with typical STS token lifetimes.

---

### <a name="code-quality-http-client-dup"></a>4. Code Quality: Duplicated HTTP client initialisation logic

- **Location:** `src/anthropic/_base_client.py`, `_DefaultHttpxClient.__init__` (lines 845–890) and `_DefaultAsyncHttpxClient.__init__` (lines 1460–1505)
- **Category:** Code Quality
- **Severity:** Medium

**Description:**  
The entire block that constructs TCP keepalive socket options and configures HTTP transport (including proxy mounts) is copy-pasted almost identically between the sync and async client classes. Any bug fix or enhancement (e.g., new keepalive option) must be applied in two places and can easily diverge.

```python
# Identical logic in both classes:
socket_options = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, True)]
TCP_KEEPINTVL = getattr(socket, "TCP_KEEPINTVL", None)
if TCP_KEEPINTVL is not None:
    socket_options.append((socket.IPPROTO_TCP, TCP_KEEPINTVL, 60))
...
proxy_map = {key: None if url is None else Proxy(url=url) ...}
transport_kwargs = {arg: kwargs[arg] for arg in ("verify", "cert", ...) if arg in kwargs}
...
```

**Recommendation:**  
Extract the shared logic into a module-level helper function, e.g., `_build_transport_kwargs(kwargs)`, that returns the computed `socket_options`, `transport_kwargs`, and `proxy_mounts`. Both classes call this single function.

---

### <a name="code-quality-compact-dup"></a>5. Code Quality: `_check_and_compact` method fully duplicated

- **Location:** `src/anthropic/lib/tools/_beta_runner.py`
  - `BaseSyncToolRunner._check_and_compact` (lines 176–258)
  - `BaseAsyncToolRunner._check_and_compact` (lines 457–539)
- **Category:** Code Quality
- **Severity:** Medium

**Description:**  
The compaction logic (token counting, threshold comparison, message pruning, and summary-prompt creation) is entirely copy-pasted between the sync and async runner classes. The two implementations differ only in whether they `await` the client call. Keeping them separate means any change to compaction logic requires two edits.

**Recommendation:**  
Extract the non-IO parts of the algorithm (token counting from `message.usage`, message list pruning, threshold comparison) into a shared helper on `BaseToolRunner`. Override only the API call itself in each subclass. For example:

```python
class BaseToolRunner:
    def _build_compaction_messages(self, messages): ...  # pure data manipulation
    def _apply_compaction_result(self, summary_text): ...

class BaseSyncToolRunner(BaseToolRunner):
    def _check_and_compact(self):
        if not self._compaction_needed(): return False
        msgs = self._build_compaction_messages(...)
        response = self._client.beta.messages.create(...)  # sync
        self._apply_compaction_result(response)
        return True

class BaseAsyncToolRunner(BaseToolRunner):
    async def _check_and_compact(self):
        if not self._compaction_needed(): return False
        msgs = self._build_compaction_messages(...)
        response = await self._client.beta.messages.create(...)  # async
        self._apply_compaction_result(response)
        return True
```

---

### <a name="code-quality-auth-dup"></a>6. Code Quality: Duplicate auth helpers in `bedrock` and `aws`

- **Location:** `src/anthropic/lib/bedrock/_auth.py` and `src/anthropic/lib/aws/_auth.py`
- **Category:** Code Quality
- **Severity:** Medium

**Description:**  
Both files contain byte-for-byte identical implementations of `_get_session` and `get_auth_headers`. Any future change (e.g., updating the boto3 API surface, fixing a credential refresh bug) needs to be made twice.

```python
# Identical in both files:
@lru_cache(maxsize=512)
def _get_session(*, aws_access_key, aws_secret_key, aws_session_token, region, profile) -> boto3.Session:
    ...

def get_auth_headers(*, method, url, headers, ...) -> dict[str, str]:
    ...
```

**Recommendation:**  
Move the shared implementation into `src/anthropic/lib/aws/_auth.py` (or a new shared module such as `src/anthropic/lib/_sigv4.py`) and import it in `src/anthropic/lib/bedrock/_auth.py`.

---

### <a name="code-quality-validate-headers"></a>7. Code Quality: Redundant condition logic in `_validate_headers`

- **Location:** `src/anthropic/_client.py`, lines 185–198 (`Anthropic._validate_headers`) and lines 424–438 (`AsyncAnthropic._validate_headers`)
- **Category:** Code Quality / Bug
- **Severity:** Medium

**Description:**  
The validation method has a logical structure where lines 190 and 193 repeat conditions that were already checked at line 186, making the intent unclear and the code hard to reason about:

```python
def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
    if headers.get("Authorization") or headers.get("X-Api-Key"):  # line 186 — early return if either is present
        return

    # Lines 190-191: checks headers.get("X-Api-Key") AGAIN — already checked above
    if headers.get("X-Api-Key") or isinstance(custom_headers.get("X-Api-Key"), Omit):
        return

    # Lines 193-194: checks headers.get("Authorization") AGAIN — already checked above
    if headers.get("Authorization") or isinstance(custom_headers.get("Authorization"), Omit):
        return

    raise TypeError(...)
```

The only unique logic in lines 190 and 193 is the `isinstance(..., Omit)` checks on `custom_headers`. The `headers.get(...)` portions are dead code.

**Recommendation:**  
Simplify the conditions to only check `custom_headers` for `Omit` markers on lines 190 and 193:

```python
def _validate_headers(self, headers: Headers, custom_headers: Headers) -> None:
    if headers.get("Authorization") or headers.get("X-Api-Key"):
        return

    if isinstance(custom_headers.get("X-Api-Key"), Omit):
        return

    if isinstance(custom_headers.get("Authorization"), Omit):
        return

    raise TypeError(...)
```

---

### <a name="code-quality-mutable-default"></a>8. Code Quality: Mutable default argument `options: RequestOptions = {}`

- **Location:** `src/anthropic/_base_client.py`, multiple methods including `get()` (line 1295), `post()` (line 1352), `patch()` (line 1381), `put()` (line 1406), `delete()` (line 1432), `get_api_list()` (line 1453), and their async counterparts
- **Category:** Code Quality
- **Severity:** Low

**Description:**  
Using a mutable `{}` as a default argument is a well-known Python anti-pattern. While `RequestOptions` is a `TypedDict` and callers do not mutate this dict in the current code paths, this is fragile. If any code path ever modifies `options` in place, all subsequent calls with the default will see the mutation.

```python
def get(self, path: str, *, cast_to: Type[ResponseT], options: RequestOptions = {}, ...):
    ...
```

**Recommendation:**  
Use `None` as the default and convert to an empty dict inside the function body:

```python
def get(self, path: str, *, cast_to: Type[ResponseT], options: RequestOptions | None = None, ...):
    opts = FinalRequestOptions.construct(method="get", url=path, **(options or {}))
    ...
```

---

### <a name="code-quality-qs-property"></a>9. Code Quality: `qs` property instantiates a new `Querystring` object on every call

- **Location:** `src/anthropic/_base_client.py`, lines 674–676
- **Category:** Performance / Code Quality
- **Severity:** Low

**Description:**  
The `qs` property on `BaseClient` creates a new `Querystring()` instance every time it is accessed. Since `Querystring` is stateless, this is wasteful.

```python
@property
def qs(self) -> Querystring:
    return Querystring()  # new object on every access
```

The overriding implementations in `Anthropic` and `AsyncAnthropic` (`_client.py`) do the same:
```python
@property
@override
def qs(self) -> Querystring:
    return Querystring(array_format="comma")
```

**Recommendation:**  
Either use `@cached_property` (already used elsewhere in the codebase) or instantiate once as a class attribute / instance attribute in `__init__`.

---

### <a name="bug-del-asyncio"></a>10. Bug: `AsyncHttpxClientWrapper.__del__` only supports asyncio event loops

- **Location:** `src/anthropic/_base_client.py`, lines 1542–1551
- **Category:** Bug
- **Severity:** Low

**Description:**  
The `__del__` method hard-codes `asyncio.get_running_loop()`, which will only work when the finaliser is called from within an asyncio event loop. When using `anyio` with a `trio` backend, the destructor silently swallows the exception without closing the client, resulting in a connection leak.

```python
class AsyncHttpxClientWrapper(DefaultAsyncHttpxClient):
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
Use `anyio` to schedule the close task, or attempt a `trio` fallback before falling back to asyncio:

```python
def __del__(self) -> None:
    if self.is_closed:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(self.aclose())
    except RuntimeError:
        pass  # No running loop; connection may leak
```

At minimum, remove the broad `except Exception` so that unexpected errors surface during development, and document the known limitation for trio users.

---

### <a name="bug-log-exception"></a>11. Bug: `log.exception(...)` called with `exc_info=exc` outside `except` block

- **Location:** `src/anthropic/lib/tools/_beta_runner.py`, lines 356, 657
- **Category:** Bug
- **Severity:** Low

**Description:**  
`logging.Logger.exception()` automatically captures the current exception via `sys.exc_info()`. Passing `exc_info=exc` as a keyword argument overrides this with the exception object directly. While this works in CPython, `log.exception(msg, exc_info=exc)` is semantically equivalent to `log.error(msg, exc_info=exc)` — it does not add the "exception was raised" contextual framing that callers of `log.exception` expect. The intent appears to be `log.error(...)` or `log.exception(...)` without the redundant kwarg.

```python
except Exception as exc:
    log.exception(f"Error occurred while calling tool: {tool.name}", exc_info=exc)
```

**Recommendation:**  
Replace with `log.error` with `exc_info=True` (inside an except block the current exception is automatically captured):

```python
except Exception as exc:
    log.error("Error occurred while calling tool: %s", tool.name, exc_info=True)
```

Using `%s`-style formatting (instead of an f-string) also avoids evaluating the format string when the log level is disabled.

---

### <a name="style-f-strings"></a>12. Style: Unnecessary f-strings for plain string literals

- **Location:** `src/anthropic/_client.py`, lines 103 and 343
- **Category:** Style
- **Severity:** Low

**Description:**  
An f-string is used where there is no interpolation, making the intent and the code misleading:

```python
base_url = f"https://api.anthropic.com"  # line 103 (Anthropic.__init__)
base_url = f"https://api.anthropic.com"  # line 343 (AsyncAnthropic.__init__)
```

**Recommendation:**  
```python
base_url = "https://api.anthropic.com"
```

---

### <a name="style-jitter-comment"></a>13. Style: Jitter comment does not match implementation

- **Location:** `src/anthropic/_base_client.py`, lines 800–803
- **Category:** Style
- **Severity:** Low

**Description:**  
The comment "Apply some jitter, plus-or-minus half a second" implies the jitter can increase or decrease the timeout. However the implementation `jitter = 1 - 0.25 * random()` produces values in the range `[0.75, 1.0]`, which can only reduce the timeout, never increase it.

```python
# Apply some jitter, plus-or-minus half a second.
jitter = 1 - 0.25 * random()
timeout = sleep_seconds * jitter
```

**Recommendation:**  
Either update the comment to accurately describe the behaviour ("apply a small random reduction of up to 25%"), or change the jitter formula to `0.5 + random()` to produce symmetric jitter around `1.0` (i.e., ±50%).

---

### <a name="code-quality-coerce-boolean"></a>14. Code Quality: `coerce_boolean` is case-sensitive

- **Location:** `src/anthropic/_utils/_utils.py`, line 325
- **Category:** Code Quality
- **Severity:** Low

**Description:**  
`coerce_boolean` only recognises lower-case values `"true"`, `"1"`, and `"on"`. Common variations like `"True"`, `"TRUE"`, or `"ON"` will silently return `False`:

```python
def coerce_boolean(val: str) -> bool:
    return val == "true" or val == "1" or val == "on"
```

**Recommendation:**  
```python
def coerce_boolean(val: str) -> bool:
    return val.lower() in ("true", "1", "on")
```

---

### <a name="code-quality-pageinfo-repr"></a>15. Code Quality: `PageInfo.__repr__` relies on truthiness of `URL` object

- **Location:** `src/anthropic/_base_client.py`, lines 165–171
- **Category:** Code Quality
- **Severity:** Low

**Description:**  
`PageInfo.__repr__` uses `if self.url:` and `if self.json:` to check whether these optional fields are set, but `self.url` is typed as `URL | NotGiven`. An empty `httpx.URL("")` evaluates as falsy, causing the branch to fall through incorrectly. The idiomatic way in this codebase is to use `isinstance(self.url, NotGiven)` or `is_given(self.url)`.

```python
@override
def __repr__(self) -> str:
    if self.url:       # <— should be: if not isinstance(self.url, NotGiven)
        return ...
    if self.json:      # <— same issue
        return ...
    return ...
```

**Recommendation:**  
```python
@override
def __repr__(self) -> str:
    if not isinstance(self.url, NotGiven):
        return f"{self.__class__.__name__}(url={self.url})"
    if not isinstance(self.json, NotGiven):
        return f"{self.__class__.__name__}(json={self.json})"
    return f"{self.__class__.__name__}(params={self.params})"
```

---

### <a name="code-quality-todos"></a>16. Code Quality: Stale TODO comments indicate unresolved design decisions

- **Location:** Multiple files (see below)
- **Category:** Code Quality
- **Severity:** Low

**Description:**  
The codebase contains numerous TODO and NOTE comments that reference unresolved design decisions. These items are potentially blocking correctness or completeness:

| File | Line | Comment |
|------|------|---------|
| `_base_client.py` | 98 | `# TODO: make base page type vars covariant` |
| `_base_client.py` | 201 | `# TODO: do we have to preprocess params here?` |
| `_base_client.py` | 1548 | `# TODO(someday): support non asyncio runtimes here` |
| `_base_client.py` | 2247, 2254 | `# TODO: untested` (arm/x32 architecture detection) |
| `_models.py` | 802 | `elif not TYPE_CHECKING:  # TODO: condition is weird` |
| `_utils/_utils.py` | 275 | `# TODO: this error message is not deterministic` |
| `_utils/_transform.py` | 37–38 | Forward reference handling is incomplete |
| `_utils/_transform.py` | 214, 380 | Normalised field name collision edge case |
| `lib/streaming/_messages.py` | 457 | `# TODO: check index` |
| `lib/streaming/_beta_messages.py` | 477 | `# TODO: check index` |

**Recommendation:**  
Create GitHub issues for each TODO, especially those marked as "untested" or flagging potential incorrect behaviour. The streaming index validation TODOs (`check index` at lines 457/477) are particularly important as out-of-bounds index access would raise an unhandled `IndexError` at runtime.

---

### <a name="performance-sse-chunks"></a>17. Performance: Byte concatenation in `SSEDecoder._iter_chunks`

- **Location:** `src/anthropic/_streaming.py`, lines 328–336 and 349–357
- **Category:** Performance
- **Severity:** Low

**Description:**  
The `_iter_chunks` method (both sync and async) accumulates bytes using `+=` in a loop:

```python
data = b""
for chunk in iterator:
    for line in chunk.splitlines(keepends=True):
        data += line      # <— O(n²) for large data
        if data.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
            yield data
            data = b""
```

For large SSE events or high-throughput streams, this O(n²) accumulation can become a bottleneck.

**Recommendation:**  
Use a `bytearray` or `io.BytesIO` buffer for accumulation, or collect chunks in a list and call `b"".join(parts)` when the boundary is detected:

```python
parts: list[bytes] = []
for chunk in iterator:
    for line in chunk.splitlines(keepends=True):
        parts.append(line)
        if line.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
            yield b"".join(parts)
            parts = []
if parts:
    yield b"".join(parts)
```

---

## Issue Count Summary

| Severity | Count |
|----------|-------|
| High | 2 |
| Medium | 5 |
| Low | 10 |
| **Total** | **17** |

---

*See also: [`docs/security_issues.md`](./security_issues.md) for a focused view of security-related findings.*
