"""Example demonstrating ConversationManager and AsyncConversationManager."""

import os
import asyncio

from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def example_sync():
    """Demonstrate sync ConversationManager."""
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    conversation = ConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
    )
    
    # First turn
    print("User: Hello, how are you?")
    conversation.add_user_message("Hello, how are you?")
    response = conversation.get_response()
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}")
    print(f"Usage: {conversation.last_usage}")
    print()
    
    # Second turn
    print("User: What is the capital of France?")
    conversation.add_user_message("What is the capital of France?")
    response = conversation.get_response()
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}")
    print(f"Usage: {conversation.last_usage}")
    print()
    
    # Reset conversation
    conversation.reset()
    print("Conversation reset. History is now empty.")
    print(f"History length: {len(conversation.history)}")


async def example_async():
    """Demonstrate async AsyncConversationManager."""
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    conversation = AsyncConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
    )
    
    # First turn
    print("User: Hello, how are you?")
    conversation.add_user_message("Hello, how are you?")
    response = await conversation.get_response()
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}")
    print(f"Usage: {conversation.last_usage}")
    print()
    
    # Second turn
    print("User: What is the capital of France?")
    conversation.add_user_message("What is the capital of France?")
    response = await conversation.get_response()
    assistant_message = response.content[0].text
    print(f"Assistant: {assistant_message}")
    print(f"Usage: {conversation.last_usage}")
    print()
    
    # Reset conversation
    conversation.reset()
    print("Conversation reset. History is now empty.")
    print(f"History length: {len(conversation.history)}")


if __name__ == "__main__":
    print("=== Sync ConversationManager Example ===\n")
    example_sync()
    
    print("\n=== Async AsyncConversationManager Example ===\n")
    asyncio.run(example_async())
