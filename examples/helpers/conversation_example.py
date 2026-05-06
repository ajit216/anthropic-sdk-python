#!/usr/bin/env python3
"""Example of using ConversationManager and AsyncConversationManager helpers.

This example demonstrates:
1. Sync ConversationManager for multi-turn conversations
2. Async AsyncConversationManager for async multi-turn conversations
"""

import asyncio
from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def sync_conversation_example() -> None:
    """Example using ConversationManager (sync)."""
    print("=== Sync ConversationManager Example ===\n")
    
    client = Anthropic()
    manager = ConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
    )
    
    # First turn
    print("User: Hello! What's your name?")
    response = manager.get_response("Hello! What's your name?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")
    
    # Second turn
    print("User: Can you help me with Python?")
    response = manager.get_response("Can you help me with Python?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")
    
    # Print usage information
    print(f"Last response usage: {manager.last_usage}\n")
    
    # Print conversation history length
    print(f"Conversation turns: {len(manager.history) // 2}")
    
    # Reset for new conversation
    manager.reset()
    print("Conversation reset.\n")


async def async_conversation_example() -> None:
    """Example using AsyncConversationManager."""
    print("=== Async AsyncConversationManager Example ===\n")
    
    client = AsyncAnthropic()
    manager = AsyncConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
    )
    
    # First turn
    print("User: Hello! What's your name?")
    response = await manager.get_response("Hello! What's your name?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")
    
    # Second turn
    print("User: Can you help me with Python?")
    response = await manager.get_response("Can you help me with Python?")
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")
    
    # Print usage information
    print(f"Last response usage: {manager.last_usage}\n")
    
    # Print conversation history length
    print(f"Conversation turns: {len(manager.history) // 2}")
    
    # Reset for new conversation
    manager.reset()
    print("Conversation reset.\n")


if __name__ == "__main__":
    import os
    
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Note: ANTHROPIC_API_KEY not set. Examples will not run.")
        print("Set ANTHROPIC_API_KEY environment variable to run examples.\n")
    else:
        # Run sync example
        try:
            sync_conversation_example()
        except Exception as e:
            print(f"Sync example error: {e}\n")
        
        # Run async example
        try:
            asyncio.run(async_conversation_example())
        except Exception as e:
            print(f"Async example error: {e}\n")
