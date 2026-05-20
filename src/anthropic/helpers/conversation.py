"""Conversation management helpers for multi-turn chat interactions.

This module provides ConversationManager and AsyncConversationManager classes
that automatically manage conversation history and handle context truncation
to prevent overflow errors.

Example:
    Synchronous usage:
        >>> from anthropic import Anthropic
        >>> from anthropic.helpers import ConversationManager
        >>> client = Anthropic()
        >>> manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=1024)
        >>> response = manager.add_user_message("Hello!")
        >>> response = manager.add_user_message("How are you?")

    Asynchronous usage:
        >>> from anthropic import AsyncAnthropic
        >>> from anthropic.helpers import AsyncConversationManager
        >>> client = AsyncAnthropic()
        >>> manager = AsyncConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=1024)
        >>> response = await manager.add_user_message("Hello!")
        >>> response = await manager.add_user_message("How are you?")
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union
from typing_extensions import TypedDict

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, MessageParam


class ConversationConfig(TypedDict, total=False):
    """Configuration for conversation management.

    Attributes:
        max_tokens: Maximum tokens for model responses (required)
        model: Model identifier (required)
        context_window: Total context window size (default: 200,000)
        system: System prompt for all messages
        temperature: Sampling temperature (0.0-1.0)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
    """

    max_tokens: int
    model: str
    context_window: int
    system: str
    temperature: float
    top_p: float
    top_k: int


class ConversationManager:
    """Manages multi-turn conversations with automatic history truncation.

    This class maintains conversation history and automatically removes the oldest
    messages when approaching the context window limit, preventing overflow errors.

    Args:
        client: An Anthropic client instance
        model: Model identifier (e.g., "claude-3-5-sonnet-latest")
        max_tokens: Maximum tokens for model responses
        context_window: Total context window size (default: 200,000)
        system: Optional system prompt for the conversation
        **kwargs: Additional parameters (temperature, top_p, top_k)

    Raises:
        ValueError: If required parameters are missing or invalid
    """

    def __init__(
        self,
        client: Anthropic,
        model: str,
        max_tokens: int,
        context_window: int = 200_000,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the ConversationManager.

        Args:
            client: An Anthropic client instance
            model: Model identifier
            max_tokens: Maximum tokens for responses
            context_window: Total context window size
            system: Optional system prompt
            **kwargs: Additional parameters for API calls

        Raises:
            ValueError: If client, model, or max_tokens are invalid
        """
        if not isinstance(client, Anthropic):
            raise ValueError("client must be an Anthropic instance")
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(context_window, int) or context_window <= 0:
            raise ValueError("context_window must be a positive integer")

        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.system = system
        self.extra_params = kwargs
        self.messages: list[MessageParam] = []
        self.last_response: Optional[Message] = None

    def add_user_message(self, content: str) -> Message:
        """Add a user message and get the model response.

        Args:
            content: The user message content

        Returns:
            The assistant's response as a Message object

        Raises:
            ValueError: If content is invalid
        """
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string")

        self.messages.append({"role": "user", "content": content})
        self._truncate_history()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=self.messages,
            **self.extra_params,
        )

        self.messages.append({"role": "assistant", "content": response.content})
        self.last_response = response
        return response

    def get_conversation_history(self) -> list[MessageParam]:
        """Get the current conversation history.

        Returns:
            List of messages in the conversation
        """
        return self.messages.copy()

    def get_last_response(self) -> Optional[Message]:
        """Get the last assistant response.

        Returns:
            The last Message object or None if no responses yet
        """
        return self.last_response

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.messages = []
        self.last_response = None

    def _truncate_history(self) -> None:
        """Truncate conversation history if approaching context limit.

        This method removes the oldest messages (starting after system prompt)
        to keep the total token count under the context window limit.
        """
        # Rough estimation: 1 token ≈ 4 characters
        # Reserve space for max_tokens, formatting, and system prompt
        reserved_tokens = self.max_tokens + 500

        if self.system:
            reserved_tokens += len(self.system) // 4

        available_tokens = self.context_window - reserved_tokens

        # Calculate current usage
        current_tokens = sum(self._estimate_tokens(msg) for msg in self.messages)

        # Remove oldest messages until we fit
        while current_tokens > available_tokens and len(self.messages) > 1:
            removed_msg = self.messages.pop(0)
            current_tokens -= self._estimate_tokens(removed_msg)

    def _estimate_tokens(self, message: MessageParam) -> int:
        """Estimate token count for a message.

        Args:
            message: Message to estimate

        Returns:
            Estimated token count
        """
        content = message.get("content", "")
        if isinstance(content, str):
            return len(content) // 4 + 4
        return 50  # Rough estimate for complex content


class AsyncConversationManager:
    """Asynchronous conversation manager for multi-turn conversations.

    This class is the async variant of ConversationManager, providing the same
    functionality with async/await support.

    Args:
        client: An AsyncAnthropic client instance
        model: Model identifier (e.g., "claude-3-5-sonnet-latest")
        max_tokens: Maximum tokens for model responses
        context_window: Total context window size (default: 200,000)
        system: Optional system prompt for the conversation
        **kwargs: Additional parameters (temperature, top_p, top_k)

    Raises:
        ValueError: If required parameters are missing or invalid
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        max_tokens: int,
        context_window: int = 200_000,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the AsyncConversationManager.

        Args:
            client: An AsyncAnthropic client instance
            model: Model identifier
            max_tokens: Maximum tokens for responses
            context_window: Total context window size
            system: Optional system prompt
            **kwargs: Additional parameters for API calls

        Raises:
            ValueError: If client, model, or max_tokens are invalid
        """
        if not isinstance(client, AsyncAnthropic):
            raise ValueError("client must be an AsyncAnthropic instance")
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(context_window, int) or context_window <= 0:
            raise ValueError("context_window must be a positive integer")

        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.system = system
        self.extra_params = kwargs
        self.messages: list[MessageParam] = []
        self.last_response: Optional[Message] = None

    async def add_user_message(self, content: str) -> Message:
        """Add a user message and get the model response.

        Args:
            content: The user message content

        Returns:
            The assistant's response as a Message object

        Raises:
            ValueError: If content is invalid
        """
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string")

        self.messages.append({"role": "user", "content": content})
        self._truncate_history()

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=self.messages,
            **self.extra_params,
        )

        self.messages.append({"role": "assistant", "content": response.content})
        self.last_response = response
        return response

    def get_conversation_history(self) -> list[MessageParam]:
        """Get the current conversation history.

        Returns:
            List of messages in the conversation
        """
        return self.messages.copy()

    def get_last_response(self) -> Optional[Message]:
        """Get the last assistant response.

        Returns:
            The last Message object or None if no responses yet
        """
        return self.last_response

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.messages = []
        self.last_response = None

    def _truncate_history(self) -> None:
        """Truncate conversation history if approaching context limit.

        This method removes the oldest messages to keep the total token count
        under the context window limit.
        """
        # Rough estimation: 1 token ≈ 4 characters
        # Reserve space for max_tokens, formatting, and system prompt
        reserved_tokens = self.max_tokens + 500

        if self.system:
            reserved_tokens += len(self.system) // 4

        available_tokens = self.context_window - reserved_tokens

        # Calculate current usage
        current_tokens = sum(self._estimate_tokens(msg) for msg in self.messages)

        # Remove oldest messages until we fit
        while current_tokens > available_tokens and len(self.messages) > 1:
            removed_msg = self.messages.pop(0)
            current_tokens -= self._estimate_tokens(removed_msg)

    def _estimate_tokens(self, message: MessageParam) -> int:
        """Estimate token count for a message.

        Args:
            message: Message to estimate

        Returns:
            Estimated token count
        """
        content = message.get("content", "")
        if isinstance(content, str):
            return len(content) // 4 + 4
        return 50  # Rough estimate for complex content


__all__ = ["ConversationManager", "AsyncConversationManager", "ConversationConfig"]
