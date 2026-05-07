"""Example demonstrating ConversationManager for multi-turn conversations."""

import asyncio
from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager


def sync_example() -> None:
    """Demonstrate synchronous ConversationManager."""
    client = Anthropic()

    conversation = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
    )

    print("=== Sync Conversation Example ===\n")

    print("User: What is the capital of France?")
    response = conversation.get_response(content="What is the capital of France?")
    print(f"Assistant: {response.content[0].text}\n")

    print("User: What about Germany?")
    response = conversation.get_response(content="What about Germany?")
    print(f"Assistant: {response.content[0].text}\n")

    print(f"Total messages in history: {len(conversation.history)}")
    print(f"Last usage: {conversation.last_usage}\n")

    print("Resetting conversation...\n")
    conversation.reset()
    print(f"History after reset: {len(conversation.history)}")


async def async_example() -> None:
    """Demonstrate asynchronous AsyncConversationManager."""
    client = AsyncAnthropic()

    conversation = AsyncConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
    )

    print("=== Async Conversation Example ===\n")

    print("User: What is the capital of France?")
    response = await conversation.get_response(content="What is the capital of France?")
    print(f"Assistant: {response.content[0].text}\n")

    print("User: What about Germany?")
    response = await conversation.get_response(content="What about Germany?")
    print(f"Assistant: {response.content[0].text}\n")

    print(f"Total messages in history: {len(conversation.history)}")
    print(f"Last usage: {conversation.last_usage}\n")

    print("Resetting conversation...\n")
    conversation.reset()
    print(f"History after reset: {len(conversation.history)}")


if __name__ == "__main__":
    # Uncomment to run sync example
    # sync_example()

    # Uncomment to run async example
    # asyncio.run(async_example())

    print("Examples available:")
    print("- sync_example() - for synchronous usage")
    print("- asyncio.run(async_example()) - for asynchronous usage")
    print("\nNote: Requires ANTHROPIC_API_KEY environment variable")
