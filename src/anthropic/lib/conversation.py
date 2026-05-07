"""Conversation Manager for managing multi-turn conversation history with context window limits."""

import copy
from typing import Any


class ConversationManager:
    """Manages multi-turn conversation history with automatic context window management.
    
    This helper maintains conversation state across multiple turns and automatically
    truncates the oldest messages when approaching the context window limit.
    """

    def __init__(self, model: str, max_context_tokens: int = 2048) -> None:
        """Initialize the ConversationManager.
        
        Args:
            model: The model name to use for message creation (e.g., 'claude-3-5-sonnet-20241022')
            max_context_tokens: Maximum tokens allowed in the context window. Defaults to 2048.
            
        Raises:
            ValueError: If max_context_tokens is less than or equal to 0.
        """
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than 0")
        
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.messages: list[dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ('user', 'assistant', or 'system')
            content: The message content
            
        Raises:
            ValueError: If role is not one of 'user', 'assistant', or 'system'
        """
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"Invalid role: {role}. Must be 'user', 'assistant', or 'system'")
        
        self.messages.append({"role": role, "content": content})

    def get_messages(self) -> list[dict[str, str]]:
        """Get the current conversation history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return copy.deepcopy(self.messages)

    def send_message(self, client: Any, **kwargs: Any) -> Any:
        """Send a message and add the response to history.
        
        Automatically truncates the oldest non-system messages if the total
        tokens exceed the context window limit.
        
        Args:
            client: The Anthropic client instance
            **kwargs: Additional arguments to pass to client.messages.create()
            
        Returns:
            The Message response object from the API
        """
        self._truncate_messages()
        
        response = client.messages.create(
            model=self.model,
            messages=self.messages,
            **kwargs
        )
        
        # Add the response to history
        if response.content and len(response.content) > 0:
            # Handle both text content and other content types
            content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
            self.add_message("assistant", content)
        
        return response

    def clear_history(self) -> None:
        """Clear all messages from the conversation history."""
        self.messages.clear()

    def _count_tokens(self, message: dict[str, str]) -> int:
        """Count approximate tokens in a message.
        
        Uses a simple heuristic: 4 tokens for role overhead + 4 tokens per word in content.
        
        Args:
            message: Message dictionary with 'role' and 'content' keys
            
        Returns:
            Approximate token count
        """
        # 4 tokens overhead per message for role and formatting
        # 4 tokens per word in content
        return 4 + len(message["content"].split()) * 4

    def _truncate_messages(self) -> None:
        """Truncate oldest messages if total tokens exceed the limit.
        
        Removes the oldest non-system messages first, keeping all system messages intact.
        """
        total_tokens = sum(self._count_tokens(msg) for msg in self.messages)
        
        if total_tokens > self.max_context_tokens:
            # Find indices of non-system messages
            non_system_indices = [
                i for i, msg in enumerate(self.messages) 
                if msg["role"] != "system"
            ]
            
            # Remove oldest non-system messages until we're under the limit
            while non_system_indices and total_tokens > self.max_context_tokens:
                # Remove the first (oldest) non-system message
                idx = non_system_indices.pop(0)
                total_tokens -= self._count_tokens(self.messages[idx])
                self.messages.pop(idx)
                
                # Adjust remaining indices after removal
                non_system_indices = [
                    i - 1 if i > idx else i 
                    for i in non_system_indices 
                    if i != idx
                ]


class AsyncConversationManager(ConversationManager):
    """Async variant of ConversationManager for use with AsyncAnthropic client.
    
    Identical to ConversationManager except send_message is async.
    """

    async def send_message(self, client: Any, **kwargs: Any) -> Any:
        """Send a message asynchronously and add the response to history.
        
        Automatically truncates the oldest non-system messages if the total
        tokens exceed the context window limit.
        
        Args:
            client: The AsyncAnthropic client instance
            **kwargs: Additional arguments to pass to client.messages.create()
            
        Returns:
            The Message response object from the API
        """
        self._truncate_messages()
        
        response = await client.messages.create(
            model=self.model,
            messages=self.messages,
            **kwargs
        )
        
        # Add the response to history
        if response.content and len(response.content) > 0:
            # Handle both text content and other content types
            content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
            self.add_message("assistant", content)
        
        return response
