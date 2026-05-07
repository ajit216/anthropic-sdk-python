"""ConversationManager helpers for managing multi-turn conversation state.

This module provides ConversationManager (sync) and AsyncConversationManager (async) 
helpers to maintain conversation history and automatically manage context window limits.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Sequence, Union
from typing_extensions import TypeAlias

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import MessageParam

__all__ = ["ConversationManager", "AsyncConversationManager"]

# Type for a message in the conversation
Message: TypeAlias = MessageParam

# Model context window sizes (in tokens)
DEFAULT_MODEL_CONTEXT_WINDOWS = {
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet": 200000,
    "claude-3-opus-20250219": 200000,
    "claude-3-opus": 200000,
    "claude-3-sonnet-20250229": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku-20250307": 200000,
    "claude-3-haiku": 200000,
    "claude-3-5-haiku-20241022": 100000,
    "claude-3-5-haiku": 100000,
}

# Reserved tokens for model response
RESERVED_TOKENS_FOR_RESPONSE = 1024


class ConversationManager:
    """Manages a multi-turn conversation with automatic context window management.
    
    This helper maintains a list of messages and automatically removes the oldest
    messages when approaching the model's context window limit.
    
    Example:
        ```python
        from anthropic import Anthropic
        from anthropic.helpers import ConversationManager
        
        client = Anthropic()
        manager = ConversationManager(model="claude-3-5-sonnet-20241022")
        
        manager.add_user_message("Hello, what's the capital of France?")
        response = client.messages.create(
            model=manager.model,
            max_tokens=1024,
            messages=manager.get_messages()
        )
        
        manager.add_assistant_message(response.content[0].text)
        ```
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        context_window: Optional[int] = None,
    ) -> None:
        """Initialize the ConversationManager.
        
        Args:
            model: The model name. Used to determine context window size.
            context_window: Optional custom context window size in tokens.
                If not provided, defaults will be looked up from known models.
        """
        self.model = model
        self._messages: list[Message] = []
        
        if context_window is not None:
            self.context_window = context_window
        else:
            # Try to get from known models
            self.context_window = DEFAULT_MODEL_CONTEXT_WINDOWS.get(
                model, 200000  # Default fallback
            )

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.
        
        Args:
            content: The user message content.
        """
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.
        
        Args:
            content: The assistant message content.
        """
        self._messages.append({"role": "assistant", "content": content})

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation.
        
        Args:
            role: The message role ("user" or "assistant").
            content: The message content.
            
        Raises:
            ValueError: If role is not "user" or "assistant".
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        self._messages.append({"role": role, "content": content})  # type: ignore

    def get_messages(self) -> list[Message]:
        """Get the current list of messages.
        
        Returns:
            A copy of the current message list.
        """
        return list(self._messages)

    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self._messages = []

    def _estimate_tokens(self, messages: Sequence[Message]) -> int:
        """Estimate the number of tokens in a list of messages.
        
        This is a simple estimation that doesn't require API calls.
        Real token counting would use the token counter API.
        
        Args:
            messages: The messages to estimate tokens for.
            
        Returns:
            Estimated token count.
        """
        # Rough estimation: ~4 tokens per word, with overhead per message
        total_tokens = 0
        for msg in messages:
            # Add overhead for role and formatting
            total_tokens += 8
            
            # Count tokens in content
            content = msg.get("content", "")
            if isinstance(content, str):
                # Rough: 1 token per ~4 characters
                total_tokens += len(content) // 4
            elif isinstance(content, list):
                # Count tokens in content blocks
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        total_tokens += len(block["text"]) // 4
        
        return total_tokens

    def _prune_messages(self) -> None:
        """Remove oldest messages if approaching context window limit.
        
        This method is called automatically before API calls to ensure
        we don't exceed the context window.
        """
        # Calculate available tokens
        available_tokens = self.context_window - RESERVED_TOKENS_FOR_RESPONSE
        
        # Estimate current token usage
        current_tokens = self._estimate_tokens(self._messages)
        
        # Remove oldest messages if needed (but keep at least 1 message)
        while current_tokens > available_tokens * 0.8 and len(self._messages) > 1:
            # Remove the oldest message
            removed = self._messages.pop(0)
            current_tokens = self._estimate_tokens(self._messages)

    def ensure_within_context(self) -> None:
        """Ensure the conversation is within context window limits.
        
        This method should be called before making API calls to ensure
        the conversation fits within the model's context window.
        """
        self._prune_messages()


class AsyncConversationManager:
    """Async version of ConversationManager.
    
    Manages a multi-turn conversation with automatic context window management.
    
    Example:
        ```python
        from anthropic import AsyncAnthropic
        from anthropic.helpers import AsyncConversationManager
        
        async def main():
            client = AsyncAnthropic()
            manager = AsyncConversationManager(model="claude-3-5-sonnet-20241022")
            
            manager.add_user_message("Hello!")
            response = await client.messages.create(
                model=manager.model,
                max_tokens=1024,
                messages=manager.get_messages()
            )
            
            manager.add_assistant_message(response.content[0].text)
        
        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        context_window: Optional[int] = None,
    ) -> None:
        """Initialize the AsyncConversationManager.
        
        Args:
            model: The model name. Used to determine context window size.
            context_window: Optional custom context window size in tokens.
                If not provided, defaults will be looked up from known models.
        """
        self.model = model
        self._messages: list[Message] = []
        
        if context_window is not None:
            self.context_window = context_window
        else:
            # Try to get from known models
            self.context_window = DEFAULT_MODEL_CONTEXT_WINDOWS.get(
                model, 200000  # Default fallback
            )

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.
        
        Args:
            content: The user message content.
        """
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.
        
        Args:
            content: The assistant message content.
        """
        self._messages.append({"role": "assistant", "content": content})

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation.
        
        Args:
            role: The message role ("user" or "assistant").
            content: The message content.
            
        Raises:
            ValueError: If role is not "user" or "assistant".
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        self._messages.append({"role": role, "content": content})  # type: ignore

    def get_messages(self) -> list[Message]:
        """Get the current list of messages.
        
        Returns:
            A copy of the current message list.
        """
        return list(self._messages)

    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self._messages = []

    def _estimate_tokens(self, messages: Sequence[Message]) -> int:
        """Estimate the number of tokens in a list of messages.
        
        This is a simple estimation that doesn't require API calls.
        
        Args:
            messages: The messages to estimate tokens for.
            
        Returns:
            Estimated token count.
        """
        # Rough estimation: ~4 tokens per word, with overhead per message
        total_tokens = 0
        for msg in messages:
            # Add overhead for role and formatting
            total_tokens += 8
            
            # Count tokens in content
            content = msg.get("content", "")
            if isinstance(content, str):
                # Rough: 1 token per ~4 characters
                total_tokens += len(content) // 4
            elif isinstance(content, list):
                # Count tokens in content blocks
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        total_tokens += len(block["text"]) // 4
        
        return total_tokens

    def _prune_messages(self) -> None:
        """Remove oldest messages if approaching context window limit.
        
        This method is called automatically before API calls to ensure
        we don't exceed the context window.
        """
        # Calculate available tokens
        available_tokens = self.context_window - RESERVED_TOKENS_FOR_RESPONSE
        
        # Estimate current token usage
        current_tokens = self._estimate_tokens(self._messages)
        
        # Remove oldest messages if needed (but keep at least 1 message)
        while current_tokens > available_tokens * 0.8 and len(self._messages) > 1:
            # Remove the oldest message
            removed = self._messages.pop(0)
            current_tokens = self._estimate_tokens(self._messages)

    def ensure_within_context(self) -> None:
        """Ensure the conversation is within context window limits.
        
        This method should be called before making API calls to ensure
        the conversation fits within the model's context window.
        """
        self._prune_messages()
