"""Example: rate-limited Anthropic client."""

import anthropic
from anthropic.helpers.ratelimit import RateLimitedClient

client = RateLimitedClient(anthropic.Anthropic(), rpm=60, timeout=10.0)

# client.messages, client.api_key, etc. are all forwarded transparently.
# Call _wait() before each API request to consume a token.
client._wait()
print("Token acquired — would call client.messages.create(...) here")
