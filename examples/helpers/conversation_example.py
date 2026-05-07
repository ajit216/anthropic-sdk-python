"""Example demonstrating ConversationManager and AsyncConversationManager.

This script shows how to use the ConversationManager helper for managing
multi-turn conversations with automatic context window truncation.
"""

import asyncio
import os

from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def example_sync_conversation():
    """Demonstrate sync ConversationManager with a simple two-turn conversation."""
    print("=== Sync ConversationManager Example ===\n")

    # Initialize client and manager
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful AI assistant that answers questions concisely.",
        context_window_limit=200000,
        token_budget_headroom=0.1,
    )

    # First turn
    print("User: What is the capital of France?")
    response = manager.get_response("What is the capital of France?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")

    # Print usage from first turn
    if manager.last_usage:
        print(
            f"First turn usage: {manager.last_usage.input_tokens} input, "
            f"{manager.last_usage.output_tokens} output tokens\n"
        )

    # Second turn
    print("User: What is its population?")
    response = manager.get_response("What is its population?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")

    # Print usage from second turn
    if manager.last_usage:
        print(
            f"Second turn usage: {manager.last_usage.input_tokens} input, "
            f"{manager.last_usage.output_tokens} output tokens\n"
        )

    # Print conversation history
    print(f"Conversation history has {len(manager.history)} messages")
    for i, msg in enumerate(manager.history):
        role = msg["role"].upper()
        if isinstance(msg["content"], str):
            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            print(f"  {i+1}. {role}: {content}")
        else:
            print(f"  {i+1}. {role}: [complex content]")

    # Reset for next conversation
    print("\nResetting conversation...\n")
    manager.reset()
    print(f"After reset, history has {len(manager.history)} messages")


async def example_async_conversation():
    """Demonstrate AsyncConversationManager with a simple two-turn conversation."""
    print("\n=== Async ConversationManager Example ===\n")

    # Initialize async client and manager
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    manager = AsyncConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful AI assistant that answers questions concisely.",
        context_window_limit=200000,
        token_budget_headroom=0.1,
    )

    # First turn
    print("User: What is the largest planet in our solar system?")
    response = await manager.get_response("What is the largest planet in our solar system?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")

    # Print usage from first turn
    if manager.last_usage:
        print(
            f"First turn usage: {manager.last_usage.input_tokens} input, "
            f"{manager.last_usage.output_tokens} output tokens\n"
        )

    # Second turn
    print("User: How many moons does it have?")
    response = await manager.get_response("How many moons does it have?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")

    # Print usage from second turn
    if manager.last_usage:
        print(
            f"Second turn usage: {manager.last_usage.input_tokens} input, "
            f"{manager.last_usage.output_tokens} output tokens\n"
        )

    # Print conversation history
    print(f"Conversation history has {len(manager.history)} messages")
    for i, msg in enumerate(manager.history):
        role = msg["role"].upper()
        if isinstance(msg["content"], str):
            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            print(f"  {i+1}. {role}: {content}")
        else:
            print(f"  {i+1}. {role}: [complex content]")

    # Reset for next conversation
    print("\nResetting conversation...\n")
    manager.reset()
    print(f"After reset, history has {len(manager.history)} messages")


def main():
    """Run both sync and async examples."""
    # Sync example
    example_sync_conversation()

    # Async example (requires Python 3.7+)
    asyncio.run(example_async_conversation())

    print("\n=== Examples completed ===")


if __name__ == "__main__":
    main()
