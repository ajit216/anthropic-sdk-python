# Security Review — anthropic-sdk-python v0.89.0

**Date:** 2026-05-03
**Scope:** Security-focused review of authentication, credential handling, input validation, and data exposure.

---

## Summary

The SDK generally follows good security practices: credentials are read from environment variables, HTTPS is enforced for all default endpoints, and the memory tool has path traversal protections. However, several areas warrant attention.

---

## Findings

### SEC-1: Boto3 Session Caching Risks Credential Staleness [HIGH]

- **Location:** `src/anthropic/lib/bedrock/_auth.py:13-30`, `src/anthropic/lib/aws/_auth.py:13-30`
- **Description:** AWS boto3 sessions are cached indefinitely via `@lru_cache(maxsize=512)`. Session credentials (including temporary STS tokens) are resolved at cache-miss time and never refreshed. If credentials rotate — which is standard for IAM roles, EC2 instance profiles, ECS task roles, and Lambda execution roles — the cached session will continue using expired credentials, leading to authentication failures.
- **Impact:** Authentication failures in long-running processes; potential for stale credential reuse.
- **Recommendation:**
  1. Remove the LRU cache on `_get_session()`, or
  2. Implement a TTL-aware cache that expires entries before typical STS credential lifetimes, or
  3. Keep the session cached but resolve credentials fresh on each `get_auth_headers` call via `session.get_credentials()` without caching the frozen credentials.

---

### SEC-2: User-Controlled Input in URL Construction [MEDIUM]

- **Location:**
  - `src/anthropic/lib/bedrock/_client.py:189` — `aws_region` in URL
  - `src/anthropic/lib/vertex/_client.py:125,270` — `region` in URL
  - `src/anthropic/lib/foundry.py:160,337` — `resource` in URL
- **Description:** User-provided `region`, `aws_region`, and `resource` values are interpolated into base URLs via f-strings without validation. While these typically come from environment variables or explicit user configuration (not untrusted input), there's no validation that they match expected patterns.

  Example attack vector: `region="evil.com/path#"` would produce `https://evil.com/path#-aiplatform.googleapis.com/v1`, sending requests to an attacker-controlled server.
- **Impact:** Potential credential exfiltration (API keys, bearer tokens, or AWS SigV4 signatures sent to wrong host).
- **Recommendation:**
  ```python
  import re
  _REGION_PATTERN = re.compile(r'^[a-z][a-z0-9-]{0,63}$')
  if not _REGION_PATTERN.match(region):
      raise ValueError(f"Invalid region format: {region!r}")
  ```

---

### SEC-3: API Keys and Tokens Exposed in Debug Logging [LOW]

- **Location:** `src/anthropic/_base_client.py:496-509` (request options logging), lines 1112-1120 (response headers logging)
- **Description:** When logging is set to `DEBUG`, the SDK logs request options (including headers) and response headers. The request options dump via `model_dump(options)` could include sensitive headers or body content. Response headers are logged in full, which could include `Set-Cookie` or other sensitive server headers.

  The actual `Authorization` and `X-Api-Key` headers are added in `_build_headers` after the options logging, so they're not directly logged via the options dump. However, custom headers passed by users could contain sensitive values.
- **Impact:** Low — requires DEBUG logging to be enabled, and the most sensitive headers are added after the log point. But custom sensitive headers could be exposed.
- **Recommendation:** Consider redacting known sensitive header patterns (Authorization, X-Api-Key, api-key, Cookie) from debug log output.

---

### SEC-4: Memory Tool File Permissions Are Correctly Restrictive [INFORMATIONAL/POSITIVE]

- **Location:** `src/anthropic/lib/tools/_beta_builtin_memory_tool.py:47-52`
- **Description:** The memory tool correctly uses `0o600` for file creation and `0o700` for directory creation, preventing world-readable files even in environments with permissive umasks (e.g., Docker). The code also uses atomic file writes via `os.O_CREAT | os.O_EXCL` to prevent TOCTOU race conditions. Symlink escape prevention is implemented via path canonicalization.
- **Impact:** Positive finding — good security practices.

---

### SEC-5: Memory Tool Symlink Validation Has Correct TOCTOU Mitigation [INFORMATIONAL/POSITIVE]

- **Location:** `src/anthropic/lib/tools/_beta_builtin_memory_tool.py:306-320`
- **Description:** The `_validate_no_symlink_escape` function walks the path hierarchy from the target up to the root, resolving each component. This provides defense-in-depth against symlink-based path traversal. The `_validate_path` method also calls `.resolve()` to canonicalize paths. Combined with the atomic file operations, this provides reasonable TOCTOU protection.
- **Impact:** Positive finding.

---

### SEC-6: Bedrock Client Reads Full Request Body Into Memory for Signing [LOW]

- **Location:** `src/anthropic/lib/bedrock/_client.py:222`
- **Description:** `data = request.read().decode()` reads the entire request body into memory as a string for AWS SigV4 signing. For large payloads (e.g., base64-encoded images in messages), this could cause high memory usage. The signing process requires the full body, so this is functionally necessary, but it doubles memory usage (once for the original request body, once for the decoded string).
- **Impact:** Memory pressure for large request payloads.
- **Recommendation:** Document the memory implications. For very large payloads, consider streaming signing approaches if supported by botocore in the future.

---

### SEC-7: No Certificate Pinning or Custom CA Bundle Configuration [INFORMATIONAL]

- **Location:** SDK-wide
- **Description:** The SDK relies on system CA certificates for TLS verification. While this is standard practice and httpx supports custom certificate configuration, the SDK doesn't provide a first-class API for certificate pinning or custom CA bundles.
- **Impact:** Standard practice; no action needed unless high-security environments require cert pinning.
- **Recommendation:** Document that users can pass a custom `httpx.Client` with `verify` parameter for custom CA bundle requirements.
