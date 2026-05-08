"""Conversation management helpers for multi-turn context window management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Message, MessageParam

__all__ = ["ConversationManager", "AsyncConversationManager"]


class ConversationManager:
    """Manages conversation history with automatic context window management.
    
    Maintains a history of messages and automatically truncates older messages
    when approaching the context window limit.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize a ConversationManager.
        
        Args:
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022").
            max_tokens: Maximum context window size in tokens.
            system_prompt: Optional system message for all conversations.
            
        Raises:
            ValueError: If model is empty or max_tokens is invalid.
        """
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._messages: list[dict[str, str]] = []

    @property
    def model(self) -> str:
        """The model being used."""
        return self._model

    @property
    def max_tokens(self) -> int:
        """The maximum context window size in tokens."""
        return self._max_tokens

    @property
    def system_prompt(self) -> str | None:
        """The system prompt if set."""
        return self._system_prompt

    @property
    def history(self) -> list[dict[str, str]]:
        """Get the current message history."""
        return self._messages.copy()

    @property
    def messages(self) -> list[dict[str, str]]:
        """Alias for history property."""
        return self.history

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message author ("user" or "assistant").
            content: The message content.
            
        Raises:
            ValueError: If role is invalid or content is empty.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', not '{role}'")
        if not content or not isinstance(content, str):
            raise ValueError("content must be a non-empty string")
        
        self._messages.append({"role": role, "content": content})
        self._truncate_history()

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.
        
        Args:
            content: The message content.
        """
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.
        
        Args:
            content: The message content.
        """
        self.add_message("assistant", content)

    def get_messages(self) -> list[dict[str, str]]:
        """Get all messages formatted for API calls.
        
        Returns:
            List of messages with role and content fields.
        """
        return self.history

    def clear_history(self) -> None:
        """Clear all messages from the conversation history."""
        self._messages = []

    def create_message(
        self,
        user_message: str,
        client: Anthropic,
        **kwargs: Any,
    ) -> Message:
        """Create a message by adding user input and calling the API.
        
        Args:
            user_message: The user's message to add and send.
            client: The Anthropic client instance.
            **kwargs: Additional arguments to pass to client.messages.create().
            
        Returns:
            The API response message.
        """
        self.add_user_message(user_message)
        
        messages = self.get_messages()
        
        create_kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
            **kwargs,
        }
        
        if self._system_prompt:
            create_kwargs["system"] = self._system_prompt
        
        response = client.messages.create(**create_kwargs)
        
        # Add assistant response to history
        if response.content and len(response.content) > 0:
            content = response.content[0]
            if hasattr(content, "text"):
                self.add_assistant_message(content.text)
        
        return response

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string.
        
        Uses a simple approximation: 1 token ≈ 4 characters.
        
        Args:
            text: The text to estimate tokens for.
            
        Returns:
            Estimated number of tokens.
        """
        return max(1, len(text) // 4)

    def _truncate_history(self) -> None:
        """Remove oldest messages if history exceeds context window.
        
        Keeps the system prompt and most recent messages.
        """
        # Calculate total token count
        system_tokens = 0
        if self._system_prompt:
            system_tokens = self._estimate_tokens(self._system_prompt)
        
        # Reserve 10% of tokens as safety buffer
        safe_token_limit = int(self._max_tokens * 0.9)
        
        # Calculate message tokens
        total_tokens = system_tokens
        for msg in self._messages:
            total_tokens += self._estimate_tokens(msg.get("content", ""))
        
        # Remove oldest messages if we exceed the limit
        while total_tokens > safe_token_limit and len(self._messages) > 1:
            removed_msg = self._messages.pop(0)
            total_tokens -= self._estimate_tokens(removed_msg.get("content", ""))


class AsyncConversationManager:
    """Async version of ConversationManager.
    
    Manages conversation history with automatic context window management
    for async operations.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize an AsyncConversationManager.
        
        Args:
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022").
            max_tokens: Maximum context window size in tokens.
            system_prompt: Optional system message for all conversations.
            
        Raises:
            ValueError: If model is empty or max_tokens is invalid.
        """
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._messages: list[dict[str, str]] = []

    @property
    def model(self) -> str:
        """The model being used."""
        return self._model

    @property
    def max_tokens(self) -> int:
        """The maximum context window size in tokens."""
        return self._max_tokens

    @property
    def system_prompt(self) -> str | None:
        """The system prompt if set."""
        return self._system_prompt

    @property
    def history(self) -> list[dict[str, str]]:
        """Get the current message history."""
        return self._messages.copy()

    @property
    def messages(self) -> list[dict[str, str]]:
        """Alias for history property."""
        return self.history

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message author ("user" or "assistant").
            content: The message content.
            
        Raises:
            ValueError: If role is invalid or content is empty.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', not '{role}'")
        if not content or not isinstance(content, str):
            raise ValueError("content must be a non-empty string")
        
        self._messages.append({"role": role, "content": content})
        self._truncate_history()

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.
        
        Args:
            content: The message content.
        """
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.
        
        Args:
            content: The message content.
        """
        self.add_message("assistant", content)

    def get_messages(self) -> list[dict[str, str]]:
        """Get all messages formatted for API calls.
        
        Returns:
            List of messages with role and content fields.
        """
        return self.history

    def clear_history(self) -> None:
        """Clear all messages from the conversation history."""
        self._messages = []

    async def create_message(
        self,
        user_message: str,
        client: AsyncAnthropic,
        **kwargs: Any,
    ) -> Message:
        """Create a message by adding user input and calling the API asynchronously.
        
        Args:
            user_message: The user's message to add and send.
            client: The AsyncAnthropic client instance.
            **kwargs: Additional arguments to pass to client.messages.create().
            
        Returns:
            The API response message.
        """
        self.add_user_message(user_message)
        
        messages = self.get_messages()
        
        create_kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
            **kwargs,
        }
        
        if self._system_prompt:
            create_kwargs["system"] = self._system_prompt
        
        response = await client.messages.create(**create_kwargs)
        
        # Add assistant response to history
        if response.content and len(response.content) > 0:
            content = response.content[0]
            if hasattr(content, "text"):
                self.add_assistant_message(content.text)
        
        return response

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text string.
        
        Uses a simple approximation: 1 token ≈ 4 characters.
        
        Args:
            text: The text to estimate tokens for.
            
        Returns:
            Estimated number of tokens.
        """
        return max(1, len(text) // 4)

    def _truncate_history(self) -> None:
        """Remove oldest messages if history exceeds context window.
        
        Keeps the system prompt and most recent messages.
        """
        # Calculate total token count
        system_tokens = 0
        if self._system_prompt:
            system_tokens = self._estimate_tokens(self._system_prompt)
        
        # Reserve 10% of tokens as safety buffer
        safe_token_limit = int(self._max_tokens * 0.9)
        
        # Calculate message tokens
        total_tokens = system_tokens
        for msg in self._messages:
            total_tokens += self._estimate_tokens(msg.get("content", ""))
        
        # Remove oldest messages if we exceed the limit
        while total_tokens > safe_token_limit and len(self._messages) > 1:
            removed_msg = self._messages.pop(0)
            total_tokens -= self._estimate_tokens(removed_msg.get("content", ""))
