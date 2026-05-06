#!/usr/bin/env python3
"""Example demonstrating ConversationManager and AsyncConversationManager."""

import asyncio
import os

from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def sync_example() -> None:
    """Demonstrate sync ConversationManager."""
    print("=== Sync ConversationManager Example ===\n")
    
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    manager = ConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
    )
    
    # First turn
    print("User: Hello! What is 2 + 2?")
    manager.add_user_message("Hello! What is 2 + 2?")
    response = manager.get_response()
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")
    print(f"Usage: {manager.last_usage}\n")
    
    # Second turn
    print("User: Great! What about 3 + 3?")
    manager.add_user_message("Great! What about 3 + 3?")
    response = manager.get_response()
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")
    print(f"Usage: {manager.last_usage}\n")
    
    # Show conversation history
    print("Conversation history:")
    for i, msg in enumerate(manager.history):
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            preview = content[:50]
        else:
            preview = str(content)[:50]
        print(f"  [{i}] {role}: {preview}...\n")
    
    # Reset for new conversation
    manager.reset()
    print("Reset conversation. History is now empty.\n")


async def async_example() -> None:
    """Demonstrate async AsyncConversationManager."""
    print("=== Async ConversationManager Example ===\n")
    
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    manager = AsyncConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
    )
    
    # First turn
    print("User: Hello! What is 5 + 5?")
    manager.add_user_message("Hello! What is 5 + 5?")
    response = await manager.get_response()
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")
    print(f"Usage: {manager.last_usage}\n")
    
    # Second turn
    print("User: What about 10 + 10?")
    manager.add_user_message("What about 10 + 10?")
    response = await manager.get_response()
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}\n")
    print(f"Usage: {manager.last_usage}\n")
    
    # Show conversation history
    print("Conversation history:")
    for i, msg in enumerate(manager.history):
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            preview = content[:50]
        else:
            preview = str(content)[:50]
        print(f"  [{i}] {role}: {preview}...\n")
    
    # Reset for new conversation
    manager.reset()
    print("Reset conversation. History is now empty.\n")


if __name__ == "__main__":
    # Run sync example
    sync_example()
    
    # Run async example
    asyncio.run(async_example())
