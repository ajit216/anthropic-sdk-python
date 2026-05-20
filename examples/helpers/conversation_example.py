#!/usr/bin/env python3
"""Example demonstrating ConversationManager for multi-turn conversations.

This example shows how to use the ConversationManager helper to maintain
conversation history with automatic context window management.
"""

from anthropic import Anthropic
from anthropic.helpers import ConversationManager


def main():
    """Run an interactive conversation with automatic history management."""
    # Initialize the client and conversation manager
    client = Anthropic()
    
    manager = ConversationManager(
        client=client,
        model="claude-opus-4-6",
        max_tokens=1024,
        system="You are a helpful assistant.",
        context_window_size=200000,
        reserve_tokens=2000,
    )

    print("Conversation Manager Example")
    print("=" * 50)
    print("Type 'quit' to exit, 'clear' to clear history")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "clear":
            manager.clear_history()
            print("Conversation history cleared.")
            continue
        
        if not user_input:
            continue

        # Add user message to history
        manager.add_user_message(user_input)

        # Get response from Claude
        response = manager.create_response()

        # Print the response
        for content_block in response.content:
            if hasattr(content_block, "text"):
                print(f"\nAssistant: {content_block.text}")

        # Display token usage
        if response.usage:
            print(
                f"\n[Tokens: input={response.usage.input_tokens}, "
                f"output={response.usage.output_tokens}]"
            )


def demo_conversation():
    """Run a demo conversation without user input."""
    client = Anthropic()
    
    manager = ConversationManager(
        client=client,
        model="claude-opus-4-6",
        max_tokens=512,
        system="You are a helpful assistant specializing in Python programming.",
    )

    # Simulate a multi-turn conversation
    conversations = [
        "What are the benefits of using type hints in Python?",
        "Can you give me a simple example?",
        "How would I use this in a real project?",
    ]

    print("Demo Conversation with ConversationManager")
    print("=" * 50)

    for user_message in conversations:
        print(f"\nUser: {user_message}")
        manager.add_user_message(user_message)

        # Get response
        response = manager.create_response()

        # Print response
        for content_block in response.content:
            if hasattr(content_block, "text"):
                print(f"\nAssistant: {content_block.text[:200]}...")  # Truncate for display

        print(f"(Conversation history: {len(manager.get_messages())} messages)")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo_conversation()
    else:
        main()
