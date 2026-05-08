#!/usr/bin/env python3
"""Example usage of ConversationManager for managing multi-turn conversations.

This example demonstrates how to use ConversationManager to build a simple
chatbot with automatic conversation history management and context window limits.
"""

from anthropic.helpers import ConversationManager

def main() -> None:
    """Run the conversation example."""
    # Initialize the conversation manager with a model and context window size
    manager = ConversationManager(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200000,  # Context window of 200k tokens
        response_token_budget=1000,  # Reserve 1000 tokens for the response
    )
    
    # Add a system message to set the assistant's behavior
    manager.add_message(
        "system",
        "You are a helpful assistant. Answer questions concisely and accurately."
    )
    
    # Simulate a multi-turn conversation
    conversations = [
        ("user", "What is the capital of France?"),
        ("assistant", "The capital of France is Paris."),
        ("user", "Tell me more about Paris."),
        ("assistant", 
         "Paris is the capital and largest city of France. It is known for landmarks "
         "like the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. "
         "It's also famous for its art, culture, cuisine, and fashion."),
        ("user", "What is the population of Paris?"),
        ("assistant",
         "The city of Paris proper has a population of about 2.2 million people. "
         "The greater Paris metropolitan area (Île-de-France) has over 12 million people."),
    ]
    
    print("=" * 60)
    print("ConversationManager Example")
    print("=" * 60)
    print()
    
    for role, content in conversations:
        # Add the message to the conversation history
        manager.add_message(role, content)
        
        # Display the message
        print(f"{role.upper()}: {content[:80]}..." if len(content) > 80 else f"{role.upper()}: {content}")
        print()
    
    # Display statistics
    messages = manager.get_messages()
    token_count = manager.get_token_count()
    
    print("=" * 60)
    print("Conversation Statistics")
    print("=" * 60)
    print(f"Total messages: {len(messages)}")
    print(f"Estimated token count: {token_count}")
    print(f"Available context window: {manager.max_tokens - manager.response_token_budget} tokens")
    print(f"Response budget: {manager.response_token_budget} tokens")
    print()
    
    # Check if we have space for more messages
    print(f"Space available for 500 token message: {manager.has_space(500)}")
    print(f"Space available for 1000 token message: {manager.has_space(1000)}")
    print()
    
    # Display the conversation history
    print("=" * 60)
    print("Full Conversation History")
    print("=" * 60)
    for i, message in enumerate(messages, 1):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        print(f"{i}. [{role.upper()}] {content[:60]}..." if len(content) > 60 else f"{i}. [{role.upper()}] {content}")
    print()
    
    # Demonstrate automatic truncation
    print("=" * 60)
    print("Demonstration: Small Context Window with Truncation")
    print("=" * 60)
    print()
    
    small_manager = ConversationManager(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,  # Very small context window
        response_token_budget=100,
    )
    
    print("Adding messages to manager with small context window (500 tokens)...")
    print()
    
    for i in range(10):
        message = f"This is message {i+1} with some content to increase token count."
        small_manager.add_message("user" if i % 2 == 0 else "assistant", message)
        print(f"Added message {i+1} - Total messages in history: {len(small_manager.get_messages())}")
    
    print()
    print(f"Final message count: {len(small_manager.get_messages())} (out of 10 added)")
    print("Note: Older messages were automatically removed to stay within context limits.")


if __name__ == "__main__":
    main()
