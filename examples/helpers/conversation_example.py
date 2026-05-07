#!/usr/bin/env python3
"""Example usage of ConversationManager and AsyncConversationManager.

This example demonstrates:
1. Sync ConversationManager for multi-turn conversations
2. Accessing conversation history and usage stats
3. Async AsyncConversationManager with asyncio
4. Resetting conversation state
"""

import asyncio
import os

from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def sync_example() -> None:
    """Demonstrate sync ConversationManager."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set. Skipping sync example.")
        return

    client = Anthropic(api_key=api_key)

    # Create a conversation manager
    manager = ConversationManager(
        client,
        model="claude-3-5-sonnet-latest",
        max_tokens=256,
        system="You are a helpful assistant that answers questions concisely.",
    )

    print("=== Sync ConversationManager Example ===\n")

    # Turn 1
    print("User: What is 2 + 2?")
    response1 = manager.get_response("What is 2 + 2?")
    print(f"Assistant: {response1.content[0].text}\n")

    # Turn 2
    print("User: What about 3 + 3?")
    response2 = manager.get_response("What about 3 + 3?")
    print(f"Assistant: {response2.content[0].text}\n")

    # Display usage
    print(f"Last API usage: {manager.last_usage}\n")

    # Display history
    print(f"Conversation history ({len(manager.history)} messages):")
    for i, msg in enumerate(manager.history, 1):
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            content = "[complex content]"
        print(f"  {i}. {role}: {content[:50]}...")

    # Reset conversation
    print("\nResetting conversation...")
    manager.reset()
    print(f"History after reset: {len(manager.history)} messages\n")


async def async_example() -> None:
    """Demonstrate async AsyncConversationManager."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set. Skipping async example.")
        return

    client = AsyncAnthropic(api_key=api_key)

    # Create an async conversation manager
    manager = AsyncConversationManager(
        client,
        model="claude-3-5-sonnet-latest",
        max_tokens=256,
        system="You are a helpful assistant that answers questions concisely.",
    )

    print("=== Async AsyncConversationManager Example ===\n")

    # Turn 1
    print("User: What is the capital of France?")
    response1 = await manager.get_response("What is the capital of France?")
    print(f"Assistant: {response1.content[0].text}\n")

    # Turn 2
    print("User: What is its population?")
    response2 = await manager.get_response("What is its population?")
    print(f"Assistant: {response2.content[0].text}\n")

    # Display usage
    print(f"Last API usage: {manager.last_usage}\n")

    # Display history
    print(f"Conversation history ({len(manager.history)} messages):")
    for i, msg in enumerate(manager.history, 1):
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            content = "[complex content]"
        print(f"  {i}. {role}: {content[:50]}...")


def main() -> None:
    """Run both examples."""
    try:
        sync_example()
    except Exception as e:
        print(f"Sync example error: {e}\n")

    try:
        asyncio.run(async_example())
    except Exception as e:
        print(f"Async example error: {e}\n")


if __name__ == "__main__":
    main()
