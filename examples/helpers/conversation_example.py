"""Example demonstrating ConversationManager and AsyncConversationManager.

This example shows how to use the ConversationManager helper to maintain
multi-turn conversation history with automatic context window management.

Requires ANTHROPIC_API_KEY environment variable.
"""

import asyncio
import os
from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def sync_example():
    """Demonstrate synchronous ConversationManager."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY environment variable not set")
        return
    
    client = Anthropic(api_key=api_key)
    
    # Create a ConversationManager
    manager = ConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
        context_window_limit=200000,
    )
    
    print("=== Synchronous ConversationManager Example ===\n")
    
    # First turn
    print("User: What is the capital of France?")
    manager.add_user_message("What is the capital of France?")
    response1 = manager.get_response()
    assistant_text = response1.content[0].text
    print(f"Assistant: {assistant_text}\n")
    print(f"Usage: input={response1.usage.input_tokens}, output={response1.usage.output_tokens}\n")
    
    # Second turn
    print("User: Tell me more about it.")
    manager.add_user_message("Tell me more about it.")
    response2 = manager.get_response()
    assistant_text = response2.content[0].text
    print(f"Assistant: {assistant_text}\n")
    print(f"Usage: input={response2.usage.input_tokens}, output={response2.usage.output_tokens}\n")
    
    # Show conversation history
    print(f"Conversation turns: {len([m for m in manager.history if m['role'] == 'user'])}")
    print(f"Total messages: {len(manager.history)}\n")
    
    # Reset conversation
    manager.reset()
    print("Conversation reset. History cleared.\n")


async def async_example():
    """Demonstrate asynchronous AsyncConversationManager."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY environment variable not set")
        return
    
    client = AsyncAnthropic(api_key=api_key)
    
    # Create an AsyncConversationManager
    manager = AsyncConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
        context_window_limit=200000,
    )
    
    print("=== Asynchronous AsyncConversationManager Example ===\n")
    
    # First turn
    print("User: What is the capital of France?")
    manager.add_user_message("What is the capital of France?")
    response1 = await manager.get_response()
    assistant_text = response1.content[0].text
    print(f"Assistant: {assistant_text}\n")
    print(f"Usage: input={response1.usage.input_tokens}, output={response1.usage.output_tokens}\n")
    
    # Second turn
    print("User: Tell me more about it.")
    manager.add_user_message("Tell me more about it.")
    response2 = await manager.get_response()
    assistant_text = response2.content[0].text
    print(f"Assistant: {assistant_text}\n")
    print(f"Usage: input={response2.usage.input_tokens}, output={response2.usage.output_tokens}\n")
    
    # Show conversation history
    print(f"Conversation turns: {len([m for m in manager.history if m['role'] == 'user'])}")
    print(f"Total messages: {len(manager.history)}\n")
    
    # Reset conversation
    manager.reset()
    print("Conversation reset. History cleared.\n")


if __name__ == "__main__":
    # Run sync example
    sync_example()
    
    # Run async example
    asyncio.run(async_example())
