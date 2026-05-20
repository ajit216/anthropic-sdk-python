#!/usr/bin/env python3
"""Example of using the AsyncConversationManager helper for async multi-turn conversations.

This example demonstrates:
- Creating an AsyncConversationManager instance
- Conducting an async multi-turn conversation
- Handling async/await patterns
- Context window management in async flows
"""

import asyncio
from anthropic import AsyncAnthropic
from anthropic.helpers import AsyncConversationManager


async def main():
    # Initialize the async Anthropic client
    client = AsyncAnthropic()

    # Create an async conversation manager
    conversation = AsyncConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        context_window=200000,
        system="You are a helpful AI assistant. Keep responses concise and friendly.",
    )

    print("🤖 Async Multi-turn Conversation Example")
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

        # Get response from the model asynchronously
        print("Claude: ", end="", flush=True)
        try:
            response = await conversation.get_response(user_input)
            print()
        except Exception as e:
            print(f"\nError: {e}")
            continue

        # Show token usage if available
        if hasattr(response, "usage"):
            print(f"(tokens: {response.usage.output_tokens} output)")


async def example_multi_turn():
    """Example of a programmatic multi-turn conversation."""
    print("\n🤖 Programmatic Multi-turn Conversation")
    print("=" * 60)

    client = AsyncAnthropic()
    conversation = AsyncConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system="You are a math tutor. Explain concepts step by step.",
    )

    questions = [
        "What is a prime number?",
        "Can you give me an example?",
        "How would you check if 17 is prime?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        response = await conversation.get_response(question)
        if response.content and response.content[0].type == "text":
            print(f"Answer: {response.content[0].text[:200]}...")


if __name__ == "__main__":
    # Run interactive mode
    # Uncomment the line below to run the interactive example
    asyncio.run(main())

    # Or run the programmatic example
    # Uncomment the line below to run the programmatic example
    # asyncio.run(example_multi_turn())
