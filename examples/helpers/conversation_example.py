#!/usr/bin/env python3
"""Example usage of ConversationManager for multi-turn conversations.

This script demonstrates how to use ConversationManager and AsyncConversationManager
to manage conversation history with automatic context window management.

Requires ANTHROPIC_API_KEY environment variable.
"""

import asyncio
from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def example_sync():
    """Demonstrate synchronous ConversationManager."""
    print("=" * 60)
    print("Synchronous ConversationManager Example")
    print("=" * 60)

    client = Anthropic()
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant that answers questions concisely.",
        context_window_limit=200_000,  # 200k context limit
        token_budget_headroom=0.10,  # Reserve 10% of context
    )

    # First turn
    print("\nUser: What is 2 + 2?")
    response = manager.get_response("What is 2 + 2?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}")
    print(f"Usage: {manager.last_usage.input_tokens} input, {manager.last_usage.output_tokens} output")

    # Second turn
    print("\nUser: What about 3 + 3?")
    response = manager.get_response("What about 3 + 3?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}")
    print(f"Usage: {manager.last_usage.input_tokens} input, {manager.last_usage.output_tokens} output")

    # Print conversation history
    print("\nConversation History:")
    for i, msg in enumerate(manager.history):
        role = msg["role"].upper()
        content = msg["content"]
        if isinstance(content, str):
            print(f"{i+1}. {role}: {content[:50]}...")
        else:
            print(f"{i+1}. {role}: [content blocks]")

    # Reset for a new conversation
    print("\nResetting conversation...")
    manager.reset()
    print(f"History length after reset: {len(manager.history)}")
    print(f"Last usage after reset: {manager.last_usage}")


async def example_async():
    """Demonstrate asynchronous AsyncConversationManager."""
    print("\n" + "=" * 60)
    print("Asynchronous AsyncConversationManager Example")
    print("=" * 60)

    client = AsyncAnthropic()
    manager = AsyncConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a math tutor. Explain answers step by step.",
        context_window_limit=200_000,
        token_budget_headroom=0.10,
    )

    # First turn
    print("\nUser: Explain how to calculate 10 * 5")
    response = await manager.get_response("Explain how to calculate 10 * 5")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message[:100]}...")
    print(f"Usage: {manager.last_usage.input_tokens} input, {manager.last_usage.output_tokens} output")

    # Second turn
    print("\nUser: What about 7 * 8?")
    response = await manager.get_response("What about 7 * 8?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message[:100]}...")
    print(f"Usage: {manager.last_usage.input_tokens} input, {manager.last_usage.output_tokens} output")

    print("\nConversation History:")
    for i, msg in enumerate(manager.history):
        role = msg["role"].upper()
        content = msg["content"]
        if isinstance(content, str):
            print(f"{i+1}. {role}: {content[:50]}...")
        else:
            print(f"{i+1}. {role}: [content blocks]")

    print("\nResetting conversation...")
    manager.reset()
    print(f"History length after reset: {len(manager.history)}")


def main():
    """Run both sync and async examples."""
    print("ConversationManager Examples\n")

    # Run sync example
    example_sync()

    # Run async example (requires Python 3.7+)
    asyncio.run(example_async())

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
