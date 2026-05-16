#!/usr/bin/env python3
"""Example usage of ConversationManager and AsyncConversationManager helpers.

This example demonstrates:
1. Creating a ConversationManager for multi-turn conversations
2. Adding messages and getting responses
3. Viewing usage information
4. Resetting the conversation
5. Using the async version with AsyncAnthropic

Requires ANTHROPIC_API_KEY environment variable.
"""

import asyncio
from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def sync_example() -> None:
    """Demonstrate synchronous ConversationManager."""
    print("=== Synchronous ConversationManager Example ===\n")
    
    client = Anthropic()
    
    manager = ConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
        context_window_limit=200000,
    )
    
    print("Turn 1: User asks a question")
    response = manager.get_response("What is the capital of France?")
    print(f"Assistant: {response.content[0].text}\n")
    
    print(f"Usage after turn 1: {manager.last_usage}\n")
    
    print("Turn 2: Follow-up question")
    response = manager.get_response("Tell me about its population.")
    print(f"Assistant: {response.content[0].text}\n")
    
    print(f"Usage after turn 2: {manager.last_usage}\n")
    
    print(f"Conversation history ({len(manager.history)} messages):")
    for i, msg in enumerate(manager.history):
        role = msg["role"].upper()
        content = msg["content"]
        if isinstance(content, str):
            preview = content[:50] + "..." if len(content) > 50 else content
        else:
            preview = f"[{len(content)} content blocks]"
        print(f"  {i+1}. {role}: {preview}")
    
    print("\nResetting conversation...")
    manager.reset()
    print(f"History after reset: {len(manager.history)} messages")


async def async_example() -> None:
    """Demonstrate asynchronous AsyncConversationManager."""
    print("\n=== Asynchronous AsyncConversationManager Example ===\n")
    
    client = AsyncAnthropic()
    
    manager = AsyncConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
        context_window_limit=200000,
    )
    
    print("Turn 1: User asks a question")
    response = await manager.get_response("What is the largest planet in our solar system?")
    print(f"Assistant: {response.content[0].text}\n")
    
    print(f"Usage after turn 1: {manager.last_usage}\n")
    
    print("Turn 2: Follow-up question")
    response = await manager.get_response("How many moons does it have?")
    print(f"Assistant: {response.content[0].text}\n")
    
    print(f"Usage after turn 2: {manager.last_usage}\n")
    
    print(f"Conversation history ({len(manager.history)} messages):")
    for i, msg in enumerate(manager.history):
        role = msg["role"].upper()
        content = msg["content"]
        if isinstance(content, str):
            preview = content[:50] + "..." if len(content) > 50 else content
        else:
            preview = f"[{len(content)} content blocks]"
        print(f"  {i+1}. {role}: {preview}")
    
    print("\nResetting conversation...")
    manager.reset()
    print(f"History after reset: {len(manager.history)} messages")


def main() -> None:
    """Run examples."""
    sync_example()
    asyncio.run(async_example())
    print("\n✓ Examples completed successfully!")


if __name__ == "__main__":
    main()
