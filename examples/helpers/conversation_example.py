#!/usr/bin/env python3
"""Example demonstrating ConversationManager usage.

This example shows how to use ConversationManager to maintain conversation
history and automatically manage context window limits during multi-turn
conversations with Claude.
"""

from anthropic import Anthropic
from anthropic.helpers import ConversationManager


def synchronous_example():
    """Example using ConversationManager with synchronous client."""
    client = Anthropic()
    
    # Initialize the conversation manager with a system prompt
    manager = ConversationManager(
        max_context_window=4096,
        system_prompt="You are a helpful assistant. Answer questions concisely."
    )
    
    print("=== Synchronous Conversation Example ===\n")
    
    # Simulate a multi-turn conversation
    questions = [
        "What is Python?",
        "What are its main features?",
        "How do I install Python?",
    ]
    
    for question in questions:
        print(f"User: {question}")
        
        # Add user message to history
        manager.add_message("user", question)
        
        # Get current history to send to API
        messages = manager.get_history()
        
        # Filter out system prompt from messages (API handles it separately)
        api_messages = [msg for msg in messages if msg["role"] != "system"]
        
        # Call the API
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=manager.system_prompt,
            messages=api_messages
        )
        
        # Extract response text
        response_text = response.content[0].text
        print(f"Assistant: {response_text}\n")
        
        # Add response to history
        manager.add_message("assistant", response_text)
        
        # Check if we're approaching context limits
        token_count = manager.get_token_count()
        print(f"[Context: {token_count}/{manager.max_context_window} tokens]\n")
    
    # Show final conversation history
    print("=== Final Conversation History ===")
    history = manager.get_history()
    for msg in history:
        role = msg["role"].upper()
        content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
        print(f"{role}: {content}")


async def async_example():
    """Example using AsyncConversationManager with async client."""
    import asyncio
    from anthropic import AsyncAnthropic
    from anthropic.helpers import AsyncConversationManager
    
    client = AsyncAnthropic()
    
    # Initialize the async conversation manager
    manager = AsyncConversationManager(
        max_context_window=4096,
        system_prompt="You are a helpful Python expert."
    )
    
    print("\n=== Asynchronous Conversation Example ===\n")
    
    questions = [
        "What is a list comprehension?",
        "Can you give me an example?",
    ]
    
    for question in questions:
        print(f"User: {question}")
        
        # Add user message to history
        await manager.add_message("user", question)
        
        # Get current history
        messages = await manager.get_history()
        
        # Filter out system prompt
        api_messages = [msg for msg in messages if msg["role"] != "system"]
        
        # Call the async API
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=manager.system_prompt,
            messages=api_messages
        )
        
        # Extract response text
        response_text = response.content[0].text
        print(f"Assistant: {response_text}\n")
        
        # Add response to history
        await manager.add_message("assistant", response_text)
        
        # Check context usage
        token_count = await manager.get_token_count()
        print(f"[Context: {token_count}/{manager.max_context_window} tokens]\n")


def context_window_management_example():
    """Example showing automatic context window management."""
    client = Anthropic()
    
    # Use a smaller context window for demonstration
    manager = ConversationManager(
        max_context_window=500,
        system_prompt="You are a helpful assistant."
    )
    
    print("\n=== Context Window Management Example ===\n")
    print("Using a small 500-token context window to demonstrate truncation.\n")
    
    # Add messages until truncation occurs
    messages = [
        "Hello!",
        "Tell me about machine learning.",
        "What are neural networks?",
        "How do transformers work?",
    ]
    
    for msg in messages:
        print(f"User: {msg}")
        manager.add_message("user", msg)
        
        # Simulate a long response
        response = "This is a simulated response from Claude. " * 20
        print(f"Assistant: {response[:80]}...\n")
        manager.add_message("assistant", response)
        
        # Show history status
        history = manager.get_history()
        token_count = manager.get_token_count()
        print(f"History size: {len(history)} messages")
        print(f"Token count: {token_count}/{manager.max_context_window}")
        print(f"Will truncate: {manager.should_truncate()}\n")


def reset_and_reuse_example():
    """Example showing how to reset and reuse a manager."""
    manager = ConversationManager(
        system_prompt="You are a helpful assistant."
    )
    
    print("\n=== Reset and Reuse Example ===\n")
    
    # First conversation
    print("--- First Conversation ---")
    manager.add_message("user", "What is Python?")
    manager.add_message("assistant", "Python is a programming language.")
    print(f"Messages in history: {len(manager.get_history())}")
    
    # Reset for a new conversation
    print("\n--- After Reset ---")
    manager.reset()
    print(f"Messages in history: {len(manager.get_history())}")
    print(f"System prompt preserved: {manager.system_prompt is not None}")
    
    # Start a new conversation with the same manager
    print("\n--- New Conversation ---")
    manager.add_message("user", "What is JavaScript?")
    manager.add_message("assistant", "JavaScript is a scripting language.")
    print(f"Messages in history: {len(manager.get_history())}")


if __name__ == "__main__":
    # Run the synchronous example
    synchronous_example()
    
    # Run the context window management example
    context_window_management_example()
    
    # Run the reset example
    reset_and_reuse_example()
    
    # Note: To run the async example, you would call:
    # import asyncio
    # asyncio.run(async_example())
