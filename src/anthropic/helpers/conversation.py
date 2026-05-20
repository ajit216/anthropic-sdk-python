"""ConversationManager for managing multi-turn conversations with automatic history truncation."""

from __future__ import annotations

from typing import Optional
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, MessageParam


class ConversationManager:
    """Manages multi-turn conversation history with automatic context window management.
    
    This helper maintains conversation state across turns and automatically truncates
    the oldest messages when approaching the context window limit.
    
    Args:
        client: The Anthropic client instance to use for API calls.
        model: The model to use for completions (e.g., "claude-opus-4-6").
        max_tokens: Maximum tokens to generate in each response.
        system: Optional system prompt or list of system blocks.
        context_window_size: The context window size for the model. Defaults to 200000.
        reserve_tokens: Number of tokens to reserve for the response. Defaults to 2000.
    
    Raises:
        ValueError: If invalid parameters are provided.
    """

    def __init__(
        self,
        client: Anthropic,
        model: str,
        max_tokens: int,
        system: str | list | None = None,
        context_window_size: int = 200000,
        reserve_tokens: int = 2000,
    ):
        """Initialize the ConversationManager."""
        if not isinstance(client, Anthropic):
            raise ValueError("client must be an Anthropic instance")
        if not model:
            raise ValueError("model cannot be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if context_window_size <= 0:
            raise ValueError("context_window_size must be positive")
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens cannot be negative")
        if max_tokens > context_window_size:
            raise ValueError("max_tokens cannot exceed context_window_size")
        if reserve_tokens + max_tokens > context_window_size:
            raise ValueError("reserve_tokens + max_tokens cannot exceed context_window_size")

        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        self.context_window_size = context_window_size
        self.reserve_tokens = reserve_tokens
        self.conversation_history: list[MessageParam] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history.
        
        Args:
            content: The user message content.
            
        Raises:
            ValueError: If content is empty.
        """
        if not content:
            raise ValueError("content cannot be empty")
        self.conversation_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history.
        
        Args:
            content: The assistant message content.
            
        Raises:
            ValueError: If content is empty.
        """
        if not content:
            raise ValueError("content cannot be empty")
        self.conversation_history.append({"role": "assistant", "content": content})

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text. Uses simple approximation: ~4 chars per token."""
        return len(text) // 4 + 1

    def _truncate_history(self) -> None:
        """Truncate the oldest messages when approaching context window limit."""
        if not self.conversation_history:
            return

        # Calculate available tokens for conversation
        available_tokens = self.context_window_size - self.max_tokens - self.reserve_tokens

        # Estimate system prompt tokens
        system_tokens = 0
        if self.system:
            if isinstance(self.system, str):
                system_tokens = self._estimate_tokens(self.system)
            elif isinstance(self.system, list):
                for item in self.system:
                    if isinstance(item, dict) and "text" in item:
                        system_tokens += self._estimate_tokens(item["text"])

        available_tokens -= system_tokens

        # Calculate current conversation tokens
        current_tokens = 0
        for msg in self.conversation_history:
            if isinstance(msg["content"], str):
                current_tokens += self._estimate_tokens(msg["content"])

        # Remove oldest messages if necessary
        while current_tokens > available_tokens and len(self.conversation_history) > 1:
            removed_msg = self.conversation_history.pop(0)
            if isinstance(removed_msg["content"], str):
                current_tokens -= self._estimate_tokens(removed_msg["content"])

    def get_messages(self) -> list[MessageParam]:
        """Get the current conversation history."""
        return self.conversation_history.copy()

    def create_response(self, **kwargs) -> Message:
        """Create a response using the conversation history.
        
        This method automatically truncates the history if needed and sends
        the messages to the API.
        
        Args:
            **kwargs: Additional arguments to pass to client.messages.create()
            
        Returns:
            The Message object returned by the API.
        """
        self._truncate_history()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=self.conversation_history,
            **kwargs,
        )

        # Extract text content from response and add to history
        response_text = ""
        for content_block in response.content:
            if hasattr(content_block, "text"):
                response_text += content_block.text

        if response_text:
            self.add_assistant_message(response_text)

        return response

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []


class AsyncConversationManager:
    """Async version of ConversationManager for managing multi-turn conversations.
    
    This helper maintains conversation state across turns and automatically truncates
    the oldest messages when approaching the context window limit.
    
    Args:
        client: The AsyncAnthropic client instance to use for API calls.
        model: The model to use for completions (e.g., "claude-opus-4-6").
        max_tokens: Maximum tokens to generate in each response.
        system: Optional system prompt or list of system blocks.
        context_window_size: The context window size for the model. Defaults to 200000.
        reserve_tokens: Number of tokens to reserve for the response. Defaults to 2000.
    
    Raises:
        ValueError: If invalid parameters are provided.
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        max_tokens: int,
        system: str | list | None = None,
        context_window_size: int = 200000,
        reserve_tokens: int = 2000,
    ):
        """Initialize the AsyncConversationManager."""
        if not isinstance(client, AsyncAnthropic):
            raise ValueError("client must be an AsyncAnthropic instance")
        if not model:
            raise ValueError("model cannot be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if context_window_size <= 0:
            raise ValueError("context_window_size must be positive")
        if reserve_tokens < 0:
            raise ValueError("reserve_tokens cannot be negative")
        if max_tokens > context_window_size:
            raise ValueError("max_tokens cannot exceed context_window_size")
        if reserve_tokens + max_tokens > context_window_size:
            raise ValueError("reserve_tokens + max_tokens cannot exceed context_window_size")

        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        self.context_window_size = context_window_size
        self.reserve_tokens = reserve_tokens
        self.conversation_history: list[MessageParam] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history.
        
        Args:
            content: The user message content.
            
        Raises:
            ValueError: If content is empty.
        """
        if not content:
            raise ValueError("content cannot be empty")
        self.conversation_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history.
        
        Args:
            content: The assistant message content.
            
        Raises:
            ValueError: If content is empty.
        """
        if not content:
            raise ValueError("content cannot be empty")
        self.conversation_history.append({"role": "assistant", "content": content})

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text. Uses simple approximation: ~4 chars per token."""
        return len(text) // 4 + 1

    def _truncate_history(self) -> None:
        """Truncate the oldest messages when approaching context window limit."""
        if not self.conversation_history:
            return

        # Calculate available tokens for conversation
        available_tokens = self.context_window_size - self.max_tokens - self.reserve_tokens

        # Estimate system prompt tokens
        system_tokens = 0
        if self.system:
            if isinstance(self.system, str):
                system_tokens = self._estimate_tokens(self.system)
            elif isinstance(self.system, list):
                for item in self.system:
                    if isinstance(item, dict) and "text" in item:
                        system_tokens += self._estimate_tokens(item["text"])

        available_tokens -= system_tokens

        # Calculate current conversation tokens
        current_tokens = 0
        for msg in self.conversation_history:
            if isinstance(msg["content"], str):
                current_tokens += self._estimate_tokens(msg["content"])

        # Remove oldest messages if necessary
        while current_tokens > available_tokens and len(self.conversation_history) > 1:
            removed_msg = self.conversation_history.pop(0)
            if isinstance(removed_msg["content"], str):
                current_tokens -= self._estimate_tokens(removed_msg["content"])

    def get_messages(self) -> list[MessageParam]:
        """Get the current conversation history."""
        return self.conversation_history.copy()

    async def create_response(self, **kwargs) -> Message:
        """Create a response using the conversation history.
        
        This method automatically truncates the history if needed and sends
        the messages to the API.
        
        Args:
            **kwargs: Additional arguments to pass to client.messages.create()
            
        Returns:
            The Message object returned by the API.
        """
        self._truncate_history()

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=self.conversation_history,
            **kwargs,
        )

        # Extract text content from response and add to history
        response_text = ""
        for content_block in response.content:
            if hasattr(content_block, "text"):
                response_text += content_block.text

        if response_text:
            self.add_assistant_message(response_text)

        return response

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
