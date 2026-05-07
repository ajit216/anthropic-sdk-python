#!/usr/bin/env python3
"""Example demonstrating ConversationManager and AsyncConversationManager."""

import asyncio
import os

import anthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def sync_conversation_example():
    """Demonstrate synchronous ConversationManager usage."""
    print("=" * 60)
    print("Synchronous ConversationManager Example")
    print("=" * 60)

    # Initialize client and manager
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        context_window_limit=200000,
        system="You are a helpful assistant.",
    )

    # First turn
    print("\nUser: What is the capital of France?")
    manager.add_user_message("What is the capital of France?")
    response = manager.get_response()
    assistant_text = response.content[0].text
    print(f"Assistant: {assistant_text}")
    print(f"Usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}")

    # Second turn
    print("\nUser: What about Germany?")
    manager.add_user_message("What about Germany?")
    response = manager.get_response()
    assistant_text = response.content[0].text
    print(f"Assistant: {assistant_text}")
    print(f"Usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}")

    # Print conversation history
    print("\n" + "-" * 60)
    print(f"Conversation history ({len(manager.history)} messages):")
    for i, msg in enumerate(manager.history):
        role = msg["role"].upper()
        content = msg["content"]
        if isinstance(content, str):
            preview = content[:50] + "..." if len(content) > 50 else content
        else:
            preview = f"[{len(content)} content blocks]"
        print(f"  {i+1}. {role}: {preview}")

    # Reset conversation
    print("\nResetting conversation...")
    manager.reset()
    print(f"History after reset: {len(manager.history)} messages")
    print()


async def async_conversation_example():
    """Demonstrate asynchronous AsyncConversationManager usage."""
    print("=" * 60)
    print("Asynchronous AsyncConversationManager Example")
    print("=" * 60)

    # Initialize async client and manager
    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    manager = AsyncConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        context_window_limit=200000,
        system="You are a helpful assistant.",
    )

    # First turn
    print("\nUser: What is the capital of Spain?")
    manager.add_user_message("What is the capital of Spain?")
    response = await manager.get_response()
    assistant_text = response.content[0].text
    print(f"Assistant: {assistant_text}")
    print(f"Usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}")

    # Second turn
    print("\nUser: What about Italy?")
    manager.add_user_message("What about Italy?")
    response = await manager.get_response()
    assistant_text = response.content[0].text
    print(f"Assistant: {assistant_text}")
    print(f"Usage: input={response.usage.input_tokens}, output={response.usage.output_tokens}")

    # Print conversation history
    print("\n" + "-" * 60)
    print(f"Conversation history ({len(manager.history)} messages):")
    for i, msg in enumerate(manager.history):
        role = msg["role"].upper()
        content = msg["content"]
        if isinstance(content, str):
            preview = content[:50] + "..." if len(content) > 50 else content
        else:
            preview = f"[{len(content)} content blocks]"
        print(f"  {i+1}. {role}: {preview}")

    # Reset conversation
    print("\nResetting conversation...")
    manager.reset()
    print(f"History after reset: {len(manager.history)} messages")
    print()


if __name__ == "__main__":
    # Run synchronous example
    sync_conversation_example()

    # Run asynchronous example
    asyncio.run(async_conversation_example())
