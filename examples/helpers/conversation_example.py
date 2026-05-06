#!/usr/bin/env python3
"""Example usage of ConversationManager and AsyncConversationManager.

Demonstrates maintaining multi-turn conversation history with automatic
truncation when approaching context window limits.
"""

import asyncio
from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def sync_example() -> None:
    """Demonstrates synchronous ConversationManager."""
    client = Anthropic()

    manager = ConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        system="You are a helpful assistant.",
    )

    print("=== Sync Example ===\n")

    # First turn
    print("User: What is the capital of France?")
    response = manager.get_response("What is the capital of France?")
    print(f"Assistant: {response.content[0].text}\n")

    # Second turn
    print("User: What's its population?")
    response = manager.get_response("What's its population?")
    print(f"Assistant: {response.content[0].text}\n")

    # Display conversation stats
    print(f"Total messages in history: {len(manager.history)}")
    print(f"Last usage: input_tokens={manager.last_usage.input_tokens}, "
          f"output_tokens={manager.last_usage.output_tokens}")

    # Reset for next conversation
    manager.reset()
    print(f"After reset: {len(manager.history)} messages\n")


async def async_example() -> None:
    """Demonstrates asynchronous AsyncConversationManager."""
    client = AsyncAnthropic()

    manager = AsyncConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=256,
        system="You are a helpful assistant.",
    )

    print("=== Async Example ===\n")

    # First turn
    print("User: What is the capital of Japan?")
    response = await manager.get_response("What is the capital of Japan?")
    print(f"Assistant: {response.content[0].text}\n")

    # Second turn
    print("User: Tell me about its culture.")
    response = await manager.get_response("Tell me about its culture.")
    print(f"Assistant: {response.content[0].text}\n")

    # Display conversation stats
    print(f"Total messages in history: {len(manager.history)}")
    print(f"Last usage: input_tokens={manager.last_usage.input_tokens}, "
          f"output_tokens={manager.last_usage.output_tokens}")


def main() -> None:
    """Run both sync and async examples."""
    sync_example()
    asyncio.run(async_example())


if __name__ == "__main__":
    main()
