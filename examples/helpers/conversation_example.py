#!/usr/bin/env python3
"""Example usage of ConversationManager and AsyncConversationManager.

This example demonstrates how to use the ConversationManager helper to manage
multi-turn conversations with automatic context window management.

Requires ANTHROPIC_API_KEY environment variable to be set.
"""

import asyncio
from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def sync_example():
    """Demonstrate sync ConversationManager."""
    print("=== Sync ConversationManager Example ===\n")
    
    client = Anthropic()
    
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
        context_window_limit=200000,
    )
    
    # First turn
    print("User: What is your name?")
    response = manager.get_response("What is your name?")
    print(f"Assistant: {response.content[0].text}\n")
    print(f"Usage: {manager.last_usage}\n")
    
    # Second turn
    print("User: How can you help me?")
    response = manager.get_response("How can you help me?")
    print(f"Assistant: {response.content[0].text}\n")
    
    # Show conversation history
    print("Conversation history:")
    for i, msg in enumerate(manager.history):
        preview = str(msg["content"])[:50]
        print(f"  {i+1}. {msg['role']}: {preview}...")
    
    # Reset conversation
    manager.reset()
    print("\nAfter reset:")
    print(f"  History length: {len(manager.history)}")
    print(f"  Last usage: {manager.last_usage}")


async def async_example():
    """Demonstrate async AsyncConversationManager."""
    print("\n=== Async AsyncConversationManager Example ===\n")
    
    client = AsyncAnthropic()
    
    manager = AsyncConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
    )
    
    # First turn
    print("User: What is a neural network?")
    response = await manager.get_response("What is a neural network?")
    print(f"Assistant: {response.content[0].text[:100]}...\n")
    
    # Second turn
    print("User: How are they used in practice?")
    response = await manager.get_response("How are they used in practice?")
    print(f"Assistant: {response.content[0].text[:100]}...\n")
    
    # Reset
    await manager.reset()
    print(f"After reset, history length: {len(manager.history)}")


def main():
    """Run examples."""
    try:
        sync_example()
    except Exception as e:
        print(f"Sync example skipped (requires ANTHROPIC_API_KEY): {e}\n")
    
    try:
        asyncio.run(async_example())
    except Exception as e:
        print(f"Async example skipped (requires ANTHROPIC_API_KEY): {e}")


if __name__ == "__main__":
    print("ConversationManager Examples")
    print("=" * 60)
    print()
    main()
