"""Example demonstrating ConversationManager and AsyncConversationManager.

This example shows how to use the ConversationManager to maintain a
multi-turn conversation with automatic message history management.

Before running, set the ANTHROPIC_API_KEY environment variable.
"""

import asyncio
import os

from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import AsyncConversationManager, ConversationManager


def sync_example() -> None:
    """Demonstrate synchronous ConversationManager."""
    print("=== Synchronous ConversationManager Example ===\n")

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Create a conversation manager
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant that provides concise answers.",
    )

    print("Turn 1: Asking about Python")
    manager.add_user_message("What is Python?")
    response1 = manager.get_response()
    print(f"Assistant: {response1.content[0].text}\n")

    print("Turn 2: Follow-up question")
    manager.add_user_message("Can you give me a simple example?")
    response2 = manager.get_response()
    print(f"Assistant: {response2.content[0].text}\n")

    print("Conversation history:")
    for i, msg in enumerate(manager.history):
        role = msg["role"].upper()
        content = (
            msg["content"]
            if isinstance(msg["content"], str)
            else f"[{len(msg['content'])} content blocks]"
        )
        print(f"  {i + 1}. {role}: {content}")

    print(f"\nLast response usage: {manager.last_usage}")
    print(f"Manager: {manager}")

    # Reset for a new conversation
    print("\n--- Resetting conversation ---\n")
    manager.reset()
    print(f"History after reset: {len(manager.history)} messages")


async def async_example() -> None:
    """Demonstrate asynchronous AsyncConversationManager."""
    print("\n=== Asynchronous AsyncConversationManager Example ===\n")

    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Create an async conversation manager
    manager = AsyncConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant that provides concise answers.",
    )

    print("Turn 1: Asking about JavaScript")
    response1 = await manager.get_response(content="What is JavaScript?")
    print(f"Assistant: {response1.content[0].text}\n")

    print("Turn 2: Follow-up question")
    response2 = await manager.get_response(content="How is it different from Python?")
    print(f"Assistant: {response2.content[0].text}\n")

    print("Conversation history:")
    for i, msg in enumerate(manager.history):
        role = msg["role"].upper()
        content = (
            msg["content"]
            if isinstance(msg["content"], str)
            else f"[{len(msg['content'])} content blocks]"
        )
        print(f"  {i + 1}. {role}: {content}")

    print(f"\nLast response usage: {manager.last_usage}")
    print(f"Manager: {manager}")


def main() -> None:
    """Run both examples."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    # Run sync example
    sync_example()

    # Run async example
    asyncio.run(async_example())


if __name__ == "__main__":
    main()
