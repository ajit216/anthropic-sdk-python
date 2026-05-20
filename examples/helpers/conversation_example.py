"""Example usage of ConversationManager helper for multi-turn conversations.

This example demonstrates how to use the ConversationManager to maintain
conversation history with automatic context window management.
"""

import os
from anthropic import Anthropic
from anthropic.helpers import ConversationManager


def synchronous_example():
    """Example of synchronous conversation management."""
    # Initialize the Anthropic client
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Create a conversation manager
    conversation = ConversationManager(
        client=client,
        max_tokens=2000,  # Max tokens for conversation history
        model="claude-3-5-sonnet-20241022",
        system="You are a helpful assistant that explains topics clearly and concisely.",
    )
    
    # Example 1: Simple single-turn interaction
    print("=== Example 1: Single Turn ===")
    response = conversation.create_message("What is Python?")
    print(f"User: What is Python?")
    print(f"Assistant: {response.content[0].text}\n")
    
    # Example 2: Multi-turn conversation
    print("=== Example 2: Multi-Turn Conversation ===")
    
    # Turn 1
    response = conversation.create_message("What are the key features of Python?")
    print(f"User: What are the key features of Python?")
    print(f"Assistant: {response.content[0].text[:100]}...\n")
    
    # Turn 2 - The conversation manager remembers the previous context
    response = conversation.create_message("Can you give me an example of one of those features?")
    print(f"User: Can you give me an example of one of those features?")
    print(f"Assistant: {response.content[0].text[:100]}...\n")
    
    # Turn 3
    response = conversation.create_message("How is this different from JavaScript?")
    print(f"User: How is this different from JavaScript?")
    print(f"Assistant: {response.content[0].text[:100]}...\n")
    
    # Example 3: Check conversation history and token count
    print("=== Example 3: Conversation Metadata ===")
    messages = conversation.get_messages()
    tokens = conversation.get_conversation_tokens()
    print(f"Total messages in history: {len(messages)}")
    print(f"Estimated tokens used: {tokens}")
    print(f"Max tokens allowed: {conversation.max_tokens}\n")
    
    # Example 4: Manually add messages (without API calls)
    print("=== Example 4: Manual Message Management ===")
    conversation.add_user_message("I have a follow-up question.")
    print(f"Added manual user message. Total messages: {len(conversation.get_messages())}\n")
    
    # Example 5: Clear history and start fresh
    print("=== Example 5: Clear History ===")
    conversation.clear_history()
    print(f"History cleared. Total messages: {len(conversation.get_messages())}")
    
    # Start a new conversation
    response = conversation.create_message("Let's talk about a different topic. What is machine learning?")
    print(f"User: Let's talk about a different topic. What is machine learning?")
    print(f"Assistant: {response.content[0].text[:100]}...\n")


async def asynchronous_example():
    """Example of asynchronous conversation management."""
    import asyncio
    from anthropic import AsyncAnthropic
    from anthropic.helpers import AsyncConversationManager
    
    # Initialize the async Anthropic client
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Create an async conversation manager
    conversation = AsyncConversationManager(
        client=client,
        max_tokens=2000,
        model="claude-3-5-sonnet-20241022",
        system="You are a helpful assistant for programming questions.",
    )
    
    print("=== Async Example: Multi-Turn Conversation ===")
    
    # Turn 1
    response = await conversation.create_message("What is async/await in Python?")
    print(f"User: What is async/await in Python?")
    print(f"Assistant: {response.content[0].text[:100]}...\n")
    
    # Turn 2
    response = await conversation.create_message("Can you show me a practical example?")
    print(f"User: Can you show me a practical example?")
    print(f"Assistant: {response.content[0].text[:100]}...\n")
    
    # Turn 3
    response = await conversation.create_message("What are the benefits over threading?")
    print(f"User: What are the benefits over threading?")
    print(f"Assistant: {response.content[0].text[:100]}...\n")


def advanced_example():
    """Example with custom parameters and configuration."""
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    # Create a conversation with smaller context window
    # Useful for testing truncation behavior
    conversation = ConversationManager(
        client=client,
        max_tokens=500,  # Smaller limit to demonstrate truncation
        model="claude-3-5-sonnet-20241022",
        system="Be concise and brief in your responses.",
    )
    
    print("=== Advanced Example: Context Window Management ===")
    
    # Add multiple messages to show truncation
    messages_to_send = [
        "Tell me about artificial intelligence",
        "What about machine learning?",
        "How is deep learning different?",
        "What are neural networks?",
        "Can you explain transformers?",
    ]
    
    for user_msg in messages_to_send:
        response = conversation.create_message(user_msg)
        tokens = conversation.get_conversation_tokens()
        msg_count = len(conversation.get_messages())
        print(f"User: {user_msg}")
        print(f"  -> History: {msg_count} messages, ~{tokens} tokens")
        if response.content:
            print(f"Assistant: {response.content[0].text[:80]}...\n")


if __name__ == "__main__":
    print("ConversationManager Examples\n")
    
    # Run synchronous example
    try:
        synchronous_example()
    except Exception as e:
        print(f"Note: Synchronous example requires ANTHROPIC_API_KEY: {e}\n")
    
    # Run advanced example
    try:
        advanced_example()
    except Exception as e:
        print(f"Note: Advanced example requires ANTHROPIC_API_KEY: {e}\n")
    
    # Async example would require: asyncio.run(asynchronous_example())
    # Uncomment below if running in an async context
    # import asyncio
    # try:
    #     asyncio.run(asynchronous_example())
    # except Exception as e:
    #     print(f"Note: Async example requires ANTHROPIC_API_KEY: {e}\n")
