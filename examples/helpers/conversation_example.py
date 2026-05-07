#!/usr/bin/env python3
"""Example of using ConversationManager for multi-turn conversations.

This example demonstrates how to use the ConversationManager helper to maintain
conversation history across multiple turns with automatic context window management.
"""

import os
from anthropic import Anthropic
from anthropic.helpers import ConversationManager


def main() -> None:
    """Run a simple multi-turn conversation example."""
    # Initialize the client and conversation manager
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    manager = ConversationManager(model="claude-3-5-sonnet-20241022")
    
    print("Conversation Manager Example")
    print("=" * 50)
    print("This example shows how to use ConversationManager for multi-turn conversations.")
    print("Type 'exit' to end the conversation.\n")
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Add user message to the manager
        manager.add_user_message(user_input)
        
        # Ensure we're within context limits before making the API call
        manager.ensure_within_context()
        
        try:
            # Create a message with the conversation history
            response = client.messages.create(
                model=manager.model,
                max_tokens=1024,
                messages=manager.get_messages(),
            )
            
            # Extract the assistant's response
            assistant_message = response.content[0].text
            
            # Add assistant message to the manager
            manager.add_assistant_message(assistant_message)
            
            print(f"\nAssistant: {assistant_message}\n")
            
        except Exception as e:
            print(f"Error: {e}")
            # Remove the last user message if the API call failed
            messages = manager.get_messages()
            if messages and messages[-1]["role"] == "user":
                manager._messages.pop()
            break


def example_context_window_management() -> None:
    """Example showing automatic context window management."""
    print("\nContext Window Management Example")
    print("=" * 50)
    
    # Create a manager with a small context window for demonstration
    manager = ConversationManager(
        model="claude-3-5-sonnet-20241022",
        context_window=1000  # Very small for demo purposes
    )
    
    # Add many messages to demonstrate auto-pruning
    for i in range(10):
        manager.add_user_message(f"Question {i}: This is a test message number {i}.")
        manager.add_assistant_message(
            f"Answer {i}: This is a response to test message number {i}."
        )
    
    print(f"Messages before pruning: {len(manager.get_messages())}")
    
    # Ensure we're within context limits
    manager.ensure_within_context()
    
    print(f"Messages after pruning: {len(manager.get_messages())}")
    print("Note: Oldest messages are automatically removed when approaching context limit.")


def example_simple_conversation() -> None:
    """Simple example of a single-turn conversation."""
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    manager = ConversationManager()
    
    # Add a user question
    manager.add_user_message("What is the capital of France?")
    
    # Get response from Claude
    response = client.messages.create(
        model=manager.model,
        max_tokens=100,
        messages=manager.get_messages(),
    )
    
    assistant_response = response.content[0].text
    manager.add_assistant_message(assistant_response)
    
    print("Simple Conversation Example")
    print("=" * 50)
    print(f"User: What is the capital of France?")
    print(f"Assistant: {assistant_response}")


if __name__ == "__main__":
    # Run the interactive example
    main()
