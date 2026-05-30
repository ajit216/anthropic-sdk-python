"""Example: response cache for identical requests."""

import anthropic
from anthropic.helpers.cache import ResponseCache

client = anthropic.Anthropic()
cache = ResponseCache(ttl=300, max_size=128)

params = dict(
    model="claude-opus-4-5",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
)

r1 = cache.get_or_fetch(client, **params)
r2 = cache.get_or_fetch(client, **params)  # from cache — no API call

print(f"r1.id={r1.id}, r2.id={r2.id}, same={r1 is r2}")
print(f"Cache size: {cache.size}")
