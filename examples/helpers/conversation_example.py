#!/usr/bin/env python3
"""Example demonstrating ConversationManager and AsyncConversationManager.

This example shows how to use the conversation helpers to manage multi-turn
conversations with automatic history truncation.

Usage:
    python conversation_example.py
    python conversation_example.py --async
"""

import asyncio
import sys
from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def example_sync_conversation() -> None:
    """Example of using ConversationManager for synchronous conversations."""
    print("=" * 60)
    print("Synchronous Conversation Example")
    print("=" * 60)

    client = Anthropic()
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        system="You are a helpful assistant.",
    )

    # Example conversation
    messages = [
        "What is the capital of France?",
        "Tell me more about its history.",
        "What are some famous landmarks there?",
    ]

    for user_message in messages:
        print(f"\nUser: {user_message}")
        response = manager.add_user_message(user_message)

        # Extract text from response
        if response.content and len(response.content) > 0:
            text_content = response.content[0]
            if hasattr(text_content, "text"):
                print(f"Assistant: {text_content.text}")
            else:
                print(f"Assistant: {response.content}")

    # Display conversation stats
    print("\n" + "=" * 60)
    print("Conversation Statistics")
    print("=" * 60)
    history = manager.get_conversation_history()
    print(f"Total messages: {len(history)}")
    print(f"User messages: {sum(1 for m in history if m['role'] == 'user')}")
    print(f"Assistant messages: {sum(1 for m in history if m['role'] == 'assistant')}")


async def example_async_conversation() -> None:
    """Example of using AsyncConversationManager for asynchronous conversations."""
    print("=" * 60)
    print("Asynchronous Conversation Example")
    print("=" * 60)

    client = AsyncAnthropic()
    manager = AsyncConversationManager(
        client=client,
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        system="You are a helpful assistant.",
    )

    # Example conversation
    messages = [
        "What is the capital of Japan?",
        "Tell me about its geography.",
        "What is it famous for?",
    ]

    for user_message in messages:
        print(f"\nUser: {user_message}")
        response = await manager.add_user_message(user_message)

        # Extract text from response
        if response.content and len(response.content) > 0:
            text_content = response.content[0]
            if hasattr(text_content, "text"):
                print(f"Assistant: {text_content.text}")
            else:
                print(f"Assistant: {response.content}")

    # Display conversation stats
    print("\n" + "=" * 60)
    print("Conversation Statistics")
    print("=" * 60)
    history = manager.get_conversation_history()
    print(f"Total messages: {len(history)}")
    print(f"User messages: {sum(1 for m in history if m['role'] == 'user')}")
    print(f"Assistant messages: {sum(1 for m in history if m['role'] == 'assistant')}")


def example_with_context_limit() -> None:
    """Example showing auto-truncation with a small context window."""
    print("=" * 60)
    print("Context Truncation Example")
    print("=" * 60)

    client = Anthropic()
    # Create manager with smaller context window to demonstrate truncation
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-latest",
        max_tokens=512,
        context_window=2000,  # Small context window
        system="You are a helpful assistant.",
    )

    print("\nAdding messages to demonstrate auto-truncation...")
    print("Creating manager with context_window=2000 for demonstration")

    # Add several messages
    messages = [
        "Tell me about Python programming",
        "What are the benefits of using Python?",
        "How does Python compare to Java?",
        "What is the Python standard library?",
        "Tell me about decorators in Python",
    ]

    for i, user_message in enumerate(messages, 1):
        print(f"\nMessage {i}: Adding user message...")
        print(f"User: {user_message[:50]}...")

        try:
            response = manager.add_user_message(user_message)
            history_size = len(manager.get_conversation_history())
            print(f"History size: {history_size} messages")
        except Exception as e:
            print(f"Error: {e}")

    # Show final conversation state
    print("\n" + "=" * 60)
    print("Final Conversation State")
    print("=" * 60)
    history = manager.get_conversation_history()
    print(f"Final message count: {len(history)}")
    print("Note: Messages may have been truncated to fit context window")


def example_clearing_history() -> None:
    """Example showing how to clear conversation history."""
    print("=" * 60)
    print("Clear History Example")
    print("=" * 60)

    client = Anthropic()
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
    )

    # Add a message
    print("\nAdding a test message...")
    print(f"History before: {len(manager.get_conversation_history())} messages")

    try:
        # Note: This will attempt to call the API
        response = manager.add_user_message("Hello!")
        print(f"History after: {len(manager.get_conversation_history())} messages")

        # Clear history
        print("\nClearing history...")
        manager.clear_history()
        print(f"History after clear: {len(manager.get_conversation_history())} messages")
        print(f"Last response: {manager.get_last_response()}")
    except Exception as e:
        print(f"Note: API call failed (expected in examples): {e}")
        print("History management still works without API calls")


def main() -> None:
    """Run examples based on command line arguments."""
    if "--async" in sys.argv or "-a" in sys.argv:
        print("Running asynchronous example...")
        asyncio.run(example_async_conversation())
    elif "--context" in sys.argv or "-c" in sys.argv:
        print("Running context limit example...")
        example_with_context_limit()
    elif "--clear" in sys.argv:
        print("Running clear history example...")
        example_clearing_history()
    else:
        print("Running synchronous example...")
        print("\nNote: This example requires a valid ANTHROPIC_API_KEY")
        print("Examples available:")
        print("  --async/-a: Run asynchronous example")
        print("  --context/-c: Run context truncation example")
        print("  --clear: Run clear history example")
        print()

        try:
            example_sync_conversation()
        except Exception as e:
            print(f"Example error (expected without API key): {type(e).__name__}")
            print("To run this example, set your ANTHROPIC_API_KEY environment variable")


if __name__ == "__main__":
    main()
