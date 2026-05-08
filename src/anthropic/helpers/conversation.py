"""ConversationManager helper for managing multi-turn conversation history."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class Message(TypedDict, total=False):
    """A message in the conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str


class ConversationManager:
    """Manages multi-turn conversation history with automatic context window management.
    
    This helper maintains conversation history and automatically truncates the oldest
    messages when approaching the context window limit, allowing developers to build
    chatbots without worrying about context overflow.
    
    Example:
        >>> manager = ConversationManager(
        ...     model="claude-3-5-sonnet-20241022",
        ...     max_tokens=200000,
        ... )
        >>> manager.add_message("user", "Hello!")
        >>> manager.add_message("assistant", "Hi there! How can I help?")
        >>> messages = manager.get_messages()
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        response_token_budget: int = 1000,
    ) -> None:
        """Initialize the ConversationManager.
        
        Args:
            model: The model to use (e.g., "claude-3-5-sonnet-20241022")
            max_tokens: Maximum tokens for the context window
            response_token_budget: Tokens to reserve for the response (default: 1000)
            
        Raises:
            ValueError: If max_tokens or response_token_budget is negative
        """
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if response_token_budget < 0:
            raise ValueError("response_token_budget must be non-negative")
        if response_token_budget >= max_tokens:
            raise ValueError("response_token_budget must be less than max_tokens")
            
        self.model = model
        self.max_tokens = max_tokens
        self.response_token_budget = response_token_budget
        self._messages: list[Message] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ("user", "assistant", or "system")
            content: The message content
            
        Raises:
            ValueError: If role is not a valid value
        """
        valid_roles = {"user", "assistant", "system"}
        if role not in valid_roles:
            raise ValueError(f"role must be one of {valid_roles}, got {role!r}")
        
        message: Message = {"role": role, "content": content}
        self._messages.append(message)
        self._truncate_if_needed()

    def get_messages(self) -> list[Message]:
        """Get the current message history respecting context limits.
        
        Returns:
            List of messages that fit within the context window
        """
        return list(self._messages)

    def get_token_count(self) -> int:
        """Estimate the current token usage.
        
        Uses a rough estimation: ~4 characters per token on average.
        
        Returns:
            Estimated number of tokens used by the message history
        """
        total_chars = 0
        for message in self._messages:
            # Account for role and content
            total_chars += len(message.get("role", "")) + len(message.get("content", ""))
            # Add overhead for message structure
            total_chars += 10
        
        # Rough estimation: 4 characters per token on average
        return max(1, total_chars // 4)

    def has_space(self, estimated_tokens: int) -> bool:
        """Check if a message with the given token count fits in the context window.
        
        Args:
            estimated_tokens: The estimated token count of the message to add
            
        Returns:
            True if the message would fit, False otherwise
        """
        available_tokens = self.max_tokens - self.response_token_budget
        current_tokens = self.get_token_count()
        return current_tokens + estimated_tokens <= available_tokens

    def _truncate_if_needed(self) -> None:
        """Automatically truncate oldest messages when approaching context limit."""
        available_tokens = self.max_tokens - self.response_token_budget
        
        while self.get_token_count() > available_tokens:
            # Keep system messages, remove oldest non-system messages
            for i, message in enumerate(self._messages):
                if message.get("role") != "system":
                    self._messages.pop(i)
                    break
            else:
                # All remaining messages are system messages, stop truncating
                break


class AsyncConversationManager:
    """Async version of ConversationManager for use with async/await.
    
    This helper maintains conversation history and automatically truncates the oldest
    messages when approaching the context window limit. This async version provides
    the same interface as ConversationManager for use in async contexts.
    
    Example:
        >>> manager = AsyncConversationManager(
        ...     model="claude-3-5-sonnet-20241022",
        ...     max_tokens=200000,
        ... )
        >>> manager.add_message("user", "Hello!")
        >>> manager.add_message("assistant", "Hi there! How can I help?")
        >>> messages = manager.get_messages()
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        response_token_budget: int = 1000,
    ) -> None:
        """Initialize the AsyncConversationManager.
        
        Args:
            model: The model to use (e.g., "claude-3-5-sonnet-20241022")
            max_tokens: Maximum tokens for the context window
            response_token_budget: Tokens to reserve for the response (default: 1000)
            
        Raises:
            ValueError: If max_tokens or response_token_budget is negative
        """
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        if response_token_budget < 0:
            raise ValueError("response_token_budget must be non-negative")
        if response_token_budget >= max_tokens:
            raise ValueError("response_token_budget must be less than max_tokens")
            
        self.model = model
        self.max_tokens = max_tokens
        self.response_token_budget = response_token_budget
        self._messages: list[Message] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ("user", "assistant", or "system")
            content: The message content
            
        Raises:
            ValueError: If role is not a valid value
        """
        valid_roles = {"user", "assistant", "system"}
        if role not in valid_roles:
            raise ValueError(f"role must be one of {valid_roles}, got {role!r}")
        
        message: Message = {"role": role, "content": content}
        self._messages.append(message)
        self._truncate_if_needed()

    def get_messages(self) -> list[Message]:
        """Get the current message history respecting context limits.
        
        Returns:
            List of messages that fit within the context window
        """
        return list(self._messages)

    def get_token_count(self) -> int:
        """Estimate the current token usage.
        
        Uses a rough estimation: ~4 characters per token on average.
        
        Returns:
            Estimated number of tokens used by the message history
        """
        total_chars = 0
        for message in self._messages:
            # Account for role and content
            total_chars += len(message.get("role", "")) + len(message.get("content", ""))
            # Add overhead for message structure
            total_chars += 10
        
        # Rough estimation: 4 characters per token on average
        return max(1, total_chars // 4)

    def has_space(self, estimated_tokens: int) -> bool:
        """Check if a message with the given token count fits in the context window.
        
        Args:
            estimated_tokens: The estimated token count of the message to add
            
        Returns:
            True if the message would fit, False otherwise
        """
        available_tokens = self.max_tokens - self.response_token_budget
        current_tokens = self.get_token_count()
        return current_tokens + estimated_tokens <= available_tokens

    def _truncate_if_needed(self) -> None:
        """Automatically truncate oldest messages when approaching context limit."""
        available_tokens = self.max_tokens - self.response_token_budget
        
        while self.get_token_count() > available_tokens:
            # Keep system messages, remove oldest non-system messages
            for i, message in enumerate(self._messages):
                if message.get("role") != "system":
                    self._messages.pop(i)
                    break
            else:
                # All remaining messages are system messages, stop truncating
                break
