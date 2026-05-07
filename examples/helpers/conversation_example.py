#!/usr/bin/env python3
"""Example demonstrating the ConversationManager helper for managing multi-turn conversations.

This example shows how to use ConversationManager to maintain conversation history
and automatically handle context window limits.
"""

from unittest.mock import Mock
from anthropic.lib.conversation import ConversationManager, AsyncConversationManager


def example_basic_conversation():
    """Basic example of using ConversationManager."""
    print("=" * 60)
    print("Example 1: Basic Conversation")
    print("=" * 60)
    
    # Create a conversation manager with a 200 token limit for demo purposes
    manager = ConversationManager(
        model="claude-3-5-sonnet-20241022",
        max_context_tokens=200
    )
    
    # Add a system message to set the context
    manager.add_message("system", "You are a helpful programming assistant.")
    print("✓ Added system message")
    
    # Add a user message
    manager.add_message("user", "What is Python?")
    print("✓ Added user message: 'What is Python?'")
    
    # Display current conversation
    print(f"\nCurrent conversation ({len(manager.messages)} messages):")
    for i, msg in enumerate(manager.get_messages(), 1):
        print(f"  {i}. [{msg['role'].upper()}]: {msg['content'][:50]}...")
    
    print()


def example_conversation_with_mock_client():
    """Example of sending messages through a mock client."""
    print("=" * 60)
    print("Example 2: Sending Messages with Mock Client")
    print("=" * 60)
    
    manager = ConversationManager(
        model="claude-3-5-sonnet-20241022",
        max_context_tokens=300
    )
    
    manager.add_message("system", "You are a helpful assistant.")
    manager.add_message("user", "Explain quantum computing.")
    print("✓ Added initial messages")
    
    # Create a mock client to simulate API responses
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock(text="Quantum computing leverages quantum mechanics for computation.")]
    mock_client.messages.create.return_value = mock_response
    
    # Send message through the manager
    response = manager.send_message(mock_client, max_tokens=256)
    print("✓ Sent message and received response")
    
    # The response is automatically added to history
    print(f"\nConversation after response ({len(manager.messages)} messages):")
    for i, msg in enumerate(manager.get_messages(), 1):
        role = msg['role'].upper()
        content = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
        print(f"  {i}. [{role}]: {content}")
    
    print()


def example_context_overflow_handling():
    """Example showing automatic truncation when context limit is exceeded."""
    print("=" * 60)
    print("Example 3: Context Overflow Handling")
    print("=" * 60)
    
    # Use a very small context limit to demonstrate truncation
    manager = ConversationManager(
        model="claude-3-5-sonnet-20241022",
        max_context_tokens=100  # Very small for demo
    )
    
    # Add a system message (should not be truncated)
    manager.add_message("system", "You are a helpful assistant answering questions.")
    print("✓ Added system message")
    
    # Add multiple conversation turns
    questions_and_answers = [
        ("What is Python?", "Python is a high-level programming language."),
        ("What about Java?", "Java is a statically-typed compiled language."),
        ("What makes Python different?", "Python emphasizes readability and simplicity."),
        ("How about C++?", "C++ is low-level with direct memory management."),
    ]
    
    for i, (question, answer) in enumerate(questions_and_answers, 1):
        manager.add_message("user", question)
        manager.add_message("assistant", answer)
        print(f"✓ Added exchange {i}")
        
        # Check if truncation occurred
        print(f"  Current message count: {len(manager.messages)}")
    
    print(f"\nFinal conversation ({len(manager.messages)} messages):")
    for i, msg in enumerate(manager.get_messages(), 1):
        role = msg['role'].upper()
        content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"  {i}. [{role}]: {content}")
    
    # Note: System message is always kept
    system_msgs = [msg for msg in manager.messages if msg['role'] == 'system']
    print(f"\nSystem messages preserved: {len(system_msgs)}")
    
    print()


def example_history_management():
    """Example of managing conversation history."""
    print("=" * 60)
    print("Example 4: History Management")
    print("=" * 60)
    
    manager = ConversationManager(
        model="claude-3-5-sonnet-20241022",
        max_context_tokens=500
    )
    
    # Build a conversation
    manager.add_message("system", "You are a math tutor.")
    manager.add_message("user", "How do I solve quadratic equations?")
    manager.add_message("assistant", "Use the quadratic formula...")
    manager.add_message("user", "Can you show an example?")
    
    print(f"Conversation has {len(manager.messages)} messages")
    
    # Get a copy of messages (useful for inspection)
    current_history = manager.get_messages()
    print(f"\nCurrent history:")
    for msg in current_history:
        print(f"  - [{msg['role']}]: {msg['content'][:40]}...")
    
    # Clear history and start fresh
    print("\n✓ Clearing history...")
    manager.clear_history()
    print(f"After clear: {len(manager.messages)} messages")
    
    # Start a new conversation
    manager.add_message("system", "You are a language teacher.")
    manager.add_message("user", "How do I say hello in French?")
    
    print(f"\nNew conversation: {len(manager.messages)} messages")
    for msg in manager.get_messages():
        print(f"  - [{msg['role']}]: {msg['content'][:40]}...")
    
    print()


def example_token_counting():
    """Example of token counting behavior."""
    print("=" * 60)
    print("Example 5: Token Counting")
    print("=" * 60)
    
    manager = ConversationManager(
        model="claude-3-5-sonnet-20241022",
        max_context_tokens=1000
    )
    
    # Add messages and show token counts
    messages_content = [
        ("system", "Be helpful."),
        ("user", "Hello how are you today?"),
        ("assistant", "I am doing well thank you for asking me."),
    ]
    
    total_tokens = 0
    for role, content in messages_content:
        manager.add_message(role, content)
        tokens = manager._count_tokens({"role": role, "content": content})
        total_tokens += tokens
        print(f"[{role.upper()}] '{content}'")
        print(f"  → {tokens} tokens (estimated)")
    
    print(f"\nTotal tokens used: {total_tokens}")
    print(f"Max context tokens: {manager.max_context_tokens}")
    print(f"Remaining capacity: {manager.max_context_tokens - total_tokens}")
    
    print()


async def example_async_conversation():
    """Example of using AsyncConversationManager (conceptual - doesn't make real API calls)."""
    print("=" * 60)
    print("Example 6: AsyncConversationManager (Conceptual)")
    print("=" * 60)
    
    manager = AsyncConversationManager(
        model="claude-3-5-sonnet-20241022",
        max_context_tokens=500
    )
    
    manager.add_message("system", "You are a helpful assistant.")
    manager.add_message("user", "Hello!")
    
    print("✓ Created AsyncConversationManager")
    print(f"  Current messages: {len(manager.messages)}")
    
    # Note: In real usage, you would use an AsyncAnthropic client
    # response = await manager.send_message(async_client, max_tokens=256)
    
    print("\n  In production, you would use:")
    print("  from anthropic import AsyncAnthropic")
    print("  client = AsyncAnthropic()")
    print("  response = await manager.send_message(client, max_tokens=256)")
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " ConversationManager Examples ".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Run synchronous examples
    example_basic_conversation()
    example_conversation_with_mock_client()
    example_context_overflow_handling()
    example_history_management()
    example_token_counting()
    
    # Note about async example
    print("=" * 60)
    print("Example 6: AsyncConversationManager")
    print("=" * 60)
    print("AsyncConversationManager is available for use with AsyncAnthropic client.")
    print("It provides the same interface as ConversationManager with async send_message().")
    print()
    
    print("\n" + "=" * 60)
    print("Examples completed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
