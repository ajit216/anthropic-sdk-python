#!/usr/bin/env python3
"""Example of using the ConversationManager helper for multi-turn conversations.

This example demonstrates:
- Creating a ConversationManager instance
- Conducting a multi-turn conversation with automatic history management
- Using custom system prompts
- Accessing conversation history
"""

from anthropic import Anthropic
from anthropic.helpers import ConversationManager


def main():
    # Initialize the Anthropic client
    client = Anthropic()

    # Create a conversation manager with a custom system prompt
    conversation = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        context_window=200000,
        system="You are a helpful AI assistant. Keep responses concise and friendly.",
    )

    print("🤖 Multi-turn Conversation Example")
    print("=" * 60)
    print("Type 'quit' to exit, 'history' to see conversation history\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "history":
            print("\n📋 Conversation History:")
            print("-" * 60)
            messages = conversation.get_messages()
            for i, msg in enumerate(messages, 1):
                role = msg["role"].upper()
                content = msg["content"]
                preview = content[:100] + "..." if len(content) > 100 else content
                print(f"{i}. {role}: {preview}")
            print("-" * 60 + "\n")
            continue

        # Get response from the model
        print("Claude: ", end="", flush=True)
        try:
            response = conversation.get_response(user_input)
            print()
        except Exception as e:
            print(f"\nError: {e}")
            continue

        # Show token usage if available
        if hasattr(response, "usage"):
            print(f"(tokens: {response.usage.output_tokens} output)")


if __name__ == "__main__":
    main()
