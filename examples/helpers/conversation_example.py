#!/usr/bin/env python3
"""
Example demonstrating ConversationManager for multi-turn conversations.

This example shows how to:
1. Create a ConversationManager
2. Maintain conversation history across multiple turns
3. Handle auto-truncation with token limits
4. Use system prompts
"""

from anthropic import Anthropic
from anthropic.helpers import ConversationManager


def basic_conversation_example() -> None:
    """Demonstrate basic multi-turn conversation."""
    print("=" * 60)
    print("Basic Conversation Example")
    print("=" * 60)
    
    client = Anthropic()
    
    # Create a conversation manager
    manager = ConversationManager(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
    )
    
    # First turn
    print("\n[User] What is Python?")
    response1 = manager.create_message(
        "What is Python?",
        client,
        max_tokens=256,
    )
    assistant_msg = response1.content[0].text
    print(f"[Assistant] {assistant_msg}")
    
    # Second turn - context is maintained
    print("\n[User] How is it used in data science?")
    response2 = manager.create_message(
        "How is it used in data science?",
        client,
        max_tokens=256,
    )
    assistant_msg = response2.content[0].text
    print(f"[Assistant] {assistant_msg}")
    
    # Show conversation history
    print("\n" + "-" * 60)
    print(f"Conversation history ({len(manager.history)} messages):")
    for msg in manager.history:
        role = msg["role"].upper()
        print(f"  [{role}] {msg['content'][:50]}...")


def conversation_with_system_prompt() -> None:
    """Demonstrate conversation with system prompt."""
    print("\n" + "=" * 60)
    print("Conversation with System Prompt")
    print("=" * 60)
    
    client = Anthropic()
    
    # Create a manager with a system prompt
    manager = ConversationManager(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system_prompt="You are a helpful coding assistant. Keep responses concise.",
    )
    
    # First turn
    print("\n[User] How do I read a file in Python?")
    response1 = manager.create_message(
        "How do I read a file in Python?",
        client,
        max_tokens=256,
    )
    assistant_msg = response1.content[0].text
    print(f"[Assistant] {assistant_msg}")
    
    # Second turn
    print("\n[User] What about writing to a file?")
    response2 = manager.create_message(
        "What about writing to a file?",
        client,
        max_tokens=256,
    )
    assistant_msg = response2.content[0].text
    print(f"[Assistant] {assistant_msg}")
    
    print("\nSystem prompt was used: ", manager.system_prompt is not None)


def token_limit_management() -> None:
    """Demonstrate automatic token limit management."""
    print("\n" + "=" * 60)
    print("Token Limit Management")
    print("=" * 60)
    
    # Create a manager with a tight token limit
    manager = ConversationManager(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,  # Very limited for demo purposes
    )
    
    print(f"Max tokens: {manager.max_tokens}")
    print("Token estimation: ~4 characters = 1 token")
    
    # Add messages that will trigger truncation
    messages = [
        "This is the first user message. " * 5,
        "This is the first assistant response. " * 5,
        "This is the second user message. " * 5,
        "This is the second assistant response. " * 5,
        "This is the third user message. " * 5,
        "This is the third assistant response. " * 5,
    ]
    
    for i, msg in enumerate(messages):
        if i % 2 == 0:
            manager.add_user_message(msg)
            msg_type = "USER"
        else:
            manager.add_assistant_message(msg)
            msg_type = "ASSISTANT"
        
        print(f"\nAdded {msg_type} message #{(i // 2) + 1}")
        print(f"  Content length: {len(msg)} chars")
        print(f"  Estimated tokens: {manager._estimate_tokens(msg)}")
        print(f"  History size: {len(manager.history)} messages")
    
    print("\n" + "-" * 60)
    print("Final conversation history (after auto-truncation):")
    for i, msg in enumerate(manager.history):
        role = msg["role"].upper()
        preview = msg["content"][:40].replace("\n", " ")
        print(f"  {i+1}. [{role}] {preview}...")


def manual_message_management() -> None:
    """Demonstrate manual message addition without API calls."""
    print("\n" + "=" * 60)
    print("Manual Message Management")
    print("=" * 60)
    
    manager = ConversationManager(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
    )
    
    # Add messages manually
    manager.add_user_message("What's the weather?")
    manager.add_assistant_message("I don't have access to real-time weather data.")
    manager.add_user_message("Can you help me write code?")
    manager.add_assistant_message("Sure! I'd be happy to help with coding.")
    
    print(f"Total messages: {len(manager.history)}")
    print("\nConversation:")
    for msg in manager.history:
        role = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
        print(f"  {role}: {msg['content']}")
    
    # Clear history
    print(f"\nClearing history...")
    manager.clear_history()
    print(f"Messages after clear: {len(manager.history)}")


if __name__ == "__main__":
    # Run examples
    try:
        basic_conversation_example()
        conversation_with_system_prompt()
        token_limit_management()
        manual_message_management()
        
        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60)
    except Exception as e:
        print(f"Error running examples: {e}")
        print("\nNote: Make sure you have ANTHROPIC_API_KEY set in your environment.")
