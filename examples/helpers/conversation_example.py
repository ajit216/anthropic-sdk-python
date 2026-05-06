#!/usr/bin/env python3
"""Example demonstrating ConversationManager for multi-turn conversations.

This example shows how to use both sync and async ConversationManager
to maintain conversation history across multiple turns.
"""

import asyncio
from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def example_sync():
    """Example of using ConversationManager (sync version)."""
    print("=== Synchronous ConversationManager Example ===\n")
    
    client = Anthropic()
    
    # Create a conversation manager
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant that answers questions concisely.",
    )
    
    # First turn
    print("User: What is the capital of France?")
    manager.add_user_message("What is the capital of France?")
    response = manager.get_response()
    assistant_text = response.content[0].text
    print(f"Assistant: {assistant_text}\n")
    
    # Print usage from first turn
    if manager.last_usage:
        print(f"First turn - Input tokens: {manager.last_usage.input_tokens}, Output tokens: {manager.last_usage.output_tokens}\n")
    
    # Second turn - manager maintains history
    print("User: What is its population?")
    manager.add_user_message("What is its population?")
    response = manager.get_response()
    assistant_text = response.content[0].text
    print(f"Assistant: {assistant_text}\n")
    
    # Print updated usage
    if manager.last_usage:
        print(f"Second turn - Input tokens: {manager.last_usage.input_tokens}, Output tokens: {manager.last_usage.output_tokens}\n")
    
    # Show conversation history
    print(f"Conversation history (turns): {len(manager.history) // 2}")
    for i, message in enumerate(manager.history):
        role = message["role"].upper()
        content = message["content"]
        if isinstance(content, str):
            preview = content[:50]
        else:
            preview = content[0].text[:50] if content else ""
        print(f"  {i+1}. {role}: {preview}...")
    
    # Reset for new conversation
    print("\nResetting conversation...")
    manager.reset()
    print(f"History after reset: {len(manager.history)} messages\n")


async def example_async():
    """Example of using AsyncConversationManager (async version)."""
    print("\n=== Asynchronous AsyncConversationManager Example ===\n")
    
    client = AsyncAnthropic()
    
    # Create an async conversation manager
    manager = AsyncConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant that answers questions concisely.",
    )
    
    # First turn
    print("User: What is the capital of Japan?")
    manager.add_user_message("What is the capital of Japan?")
    response = await manager.get_response()
    assistant_text = response.content[0].text
    print(f"Assistant: {assistant_text}\n")
    
    # Print usage from first turn
    if manager.last_usage:
        print(f"First turn - Input tokens: {manager.last_usage.input_tokens}, Output tokens: {manager.last_usage.output_tokens}\n")
    
    # Second turn - manager maintains history
    print("User: What is its population?")
    manager.add_user_message("What is its population?")
    response = await manager.get_response()
    assistant_text = response.content[0].text
    print(f"Assistant: {assistant_text}\n")
    
    # Print updated usage
    if manager.last_usage:
        print(f"Second turn - Input tokens: {manager.last_usage.input_tokens}, Output tokens: {manager.last_usage.output_tokens}\n")
    
    # Show conversation history
    print(f"Conversation history (turns): {len(manager.history) // 2}")
    for i, message in enumerate(manager.history):
        role = message["role"].upper()
        content = message["content"]
        if isinstance(content, str):
            preview = content[:50]
        else:
            preview = content[0].text[:50] if content else ""
        print(f"  {i+1}. {role}: {preview}...")


if __name__ == "__main__":
    # Run sync example
    example_sync()
    
    # Run async example
    asyncio.run(example_async())
    
    print("\nExamples completed!")
