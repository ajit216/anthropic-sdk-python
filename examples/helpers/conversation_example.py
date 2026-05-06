#!/usr/bin/env python3
"""Example usage of ConversationManager and AsyncConversationManager.

This example demonstrates how to use the conversation helpers
to maintain multi-turn conversation state.
"""

import asyncio
import os

from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def sync_example() -> None:
    """Demonstrate synchronous ConversationManager."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = Anthropic(api_key=api_key)

    # Create a conversation manager
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system="You are a helpful assistant.",
        context_window_limit=200000,
    )

    print("=== Synchronous ConversationManager ===\n")

    # First turn
    print("User: Hello! What is 2 + 2?")
    manager.add_user_message("Hello! What is 2 + 2?")
    response = manager.get_response()
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")
    print(f"Usage: {manager.last_usage}\n")

    # Second turn
    print("User: What about 3 + 3?")
    response = manager.get_response("What about 3 + 3?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")
    print(f"Usage: {manager.last_usage}\n")

    # Show history
    print(f"History: {len(manager.history)} messages")
    for i, msg in enumerate(manager.history):
        print(f"  [{i}] {msg['role']}: {msg['content'][:50]}...")

    # Reset
    print("\nResetting conversation...\n")
    manager.reset()
    print(f"History after reset: {len(manager.history)} messages")


async def async_example() -> None:
    """Demonstrate asynchronous AsyncConversationManager."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = AsyncAnthropic(api_key=api_key)

    # Create an async conversation manager
    manager = AsyncConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system="You are a helpful assistant.",
        context_window_limit=200000,
    )

    print("\n=== Asynchronous AsyncConversationManager ===\n")

    # First turn
    print("User: Hello! What is 5 + 5?")
    manager.add_user_message("Hello! What is 5 + 5?")
    response = await manager.get_response()
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")
    print(f"Usage: {manager.last_usage}\n")

    # Second turn
    print("User: What about 10 - 3?")
    response = await manager.get_response("What about 10 - 3?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")
    print(f"Usage: {manager.last_usage}\n")

    # Show history
    print(f"History: {len(manager.history)} messages")
    for i, msg in enumerate(manager.history):
        print(f"  [{i}] {msg['role']}: {msg['content'][:50]}...")

    # Reset
    print("\nResetting conversation...\n")
    manager.reset()
    print(f"History after reset: {len(manager.history)} messages")


def main() -> None:
    """Run both sync and async examples."""
    sync_example()
    asyncio.run(async_example())


if __name__ == "__main__":
    main()
