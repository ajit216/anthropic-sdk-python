#!/usr/bin/env python3
"""Example of using the ConversationManager helper for multi-turn conversations.

This example demonstrates how to use the ConversationManager to handle
multi-turn conversations with automatic context window management.
"""

from anthropic import Anthropic
from anthropic.helpers import ConversationManager


def main() -> None:
    """Run a simple multi-turn conversation using ConversationManager."""
    client = Anthropic()
    
    # Create a conversation manager with a system prompt
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        system_prompt="You are a helpful assistant that answers questions concisely."
    )
    
    print("Conversation Manager Example")
    print("=" * 50)
    print("Type 'quit' to exit the conversation.\n")
    
    # Multi-turn conversation loop
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            # Send message and get response
            response = manager.send_message(user_input)
            
            # Extract and print the assistant's response
            if response.content and len(response.content) > 0:
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                
                print(f"\nAssistant: {assistant_text}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
