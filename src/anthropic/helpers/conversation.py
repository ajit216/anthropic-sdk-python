"""Conversation manager helpers for managing multi-turn conversation history."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from anthropic import Anthropic, AsyncAnthropic

if TYPE_CHECKING:
    from anthropic.types import Message, MessageParam


class ConversationManager:
    """Manages multi-turn conversation history with automatic context window management.

    This helper maintains conversation state across turns and automatically truncates
    the oldest messages when approaching the model's context window limit.

    Args:
        client: An Anthropic client instance.
        model: The model to use for API calls (e.g., "claude-3-5-sonnet-20241022").
        max_tokens: Maximum tokens for model responses.
        context_window: The context window size for the model in tokens.
                       Defaults to 200000 for Claude 3.5 Sonnet.
        system: Optional system prompt to use for all API calls.

    Raises:
        ValueError: If invalid parameters are provided.
    """

    def __init__(
        self,
        client: Anthropic,
        model: str,
        max_tokens: int,
        context_window: int = 200000,
        system: Optional[str] = None,
    ):
        if not isinstance(client, Anthropic):
            raise ValueError("client must be an instance of Anthropic")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(context_window, int) or context_window <= 0:
            raise ValueError("context_window must be a positive integer")
        if max_tokens > context_window:
            raise ValueError("max_tokens cannot exceed context_window")
        # FIX #4: Validate system parameter
        if system is not None and (not isinstance(system, str) or not system.strip()):
            raise ValueError("system must be a non-empty string or None")

        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.system = system
        self.messages: list[MessageParam] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The user message content.

        Raises:
            ValueError: If content is not a non-empty string.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history.

        Args:
            content: The assistant message content.

        Raises:
            ValueError: If content is not a non-empty string.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> list[MessageParam]:
        """Get the current conversation history.

        Returns:
            A copy of the current message list.
        """
        return list(self.messages)

    def clear(self) -> None:
        """Clear all messages from the conversation history."""
        self.messages = []

    def _estimate_tokens(self, content) -> int:  # FIX #5: Accept any content type
        """Estimate token count for content using simple heuristic.

        This uses a rough approximation: ~4 characters per token for strings,
        ~100 tokens per list item for complex content.
        For production use, consider using the tokenizer or count_tokens API.

        Args:
            content: The content to estimate tokens for (str or list).

        Returns:
            Estimated token count.
        """
        if isinstance(content, str):
            return max(1, len(content) // 4)
        elif isinstance(content, (list, tuple)):
            # For list content (e.g., multiple blocks), estimate ~100 tokens per block
            # as a conservative estimate for tool results, images, etc.
            return max(1, len(content) * 100)
        else:
            # Unknown content type - conservative estimate
            return 100

    def _truncate_messages(self) -> None:
        """Truncate oldest messages if approaching context window limit."""
        # Reserve tokens for: system prompt, max response, and conversation buffer
        reserved_tokens = self.max_tokens + 1000
        if self.system:
            reserved_tokens += self._estimate_tokens(self.system)

        # Calculate tokens used by current messages
        current_tokens = reserved_tokens
        for msg in self.messages:
            current_tokens += self._estimate_tokens(msg["content"])

        # If we exceed context window, remove oldest messages
        while current_tokens > self.context_window and len(self.messages) > 0:
            removed_msg = self.messages.pop(0)
            # FIX #3: Handle all content types
            current_tokens -= self._estimate_tokens(removed_msg["content"])

    def get_response(self, user_message: str) -> Message:
        """Get a response from the model for the given user message.

        This method adds the user message to history, calls the API, and stores
        the assistant response. Messages are automatically truncated if needed.

        Args:
            user_message: The user's message.

        Returns:
            The Message response from the API.

        Raises:
            ValueError: If user_message is not a non-empty string or exceeds reasonable bounds.
        """
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty string")

        # FIX #3: Validate user message fits in context window
        user_msg_tokens = self._estimate_tokens(user_message)
        reserved = self.max_tokens + 1000
        if self.system:
            reserved += self._estimate_tokens(self.system)
        
        if user_msg_tokens + reserved > self.context_window:
            raise ValueError(
                f"user_message ({user_msg_tokens} tokens) + reserved tokens ({reserved}) "
                f"exceeds context_window ({self.context_window})"
            )

        self._truncate_messages()  # FIX #2: Truncate BEFORE adding user message
        self.add_user_message(user_message)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=self.messages,
        )

        # FIX #1: Handle all response content types, not just text
        if response.content:
            # Extract all content and add to history
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    # For text blocks, extract text
                    assistant_content.append(block.text)
                else:
                    # For non-text blocks (tool_use, etc.), we still need to maintain history
                    # Store a reference to the block type for tracking
                    assistant_content.append(f"[{block.type}]")
            
            # Only add to history if we have text content
            # Non-text responses (tool_use) need different handling by the caller
            if any(block.type == "text" for block in response.content):
                text_content = next(
                    (block.text for block in response.content if block.type == "text"),
                    None
                )
                if text_content:
                    self.add_assistant_message(text_content)

        return response


class AsyncConversationManager:
    """Async version of ConversationManager for managing multi-turn conversations.

    This helper maintains conversation state across turns and automatically truncates
    the oldest messages when approaching the model's context window limit.
    All operations are async/await compatible.

    Args:
        client: An AsyncAnthropic client instance.
        model: The model to use for API calls (e.g., "claude-3-5-sonnet-20241022").
        max_tokens: Maximum tokens for model responses.
        context_window: The context window size for the model in tokens.
                       Defaults to 200000 for Claude 3.5 Sonnet.
        system: Optional system prompt to use for all API calls.

    Raises:
        ValueError: If invalid parameters are provided.
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        max_tokens: int,
        context_window: int = 200000,
        system: Optional[str] = None,
    ):
        if not isinstance(client, AsyncAnthropic):
            raise ValueError("client must be an instance of AsyncAnthropic")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not isinstance(context_window, int) or context_window <= 0:
            raise ValueError("context_window must be a positive integer")
        if max_tokens > context_window:
            raise ValueError("max_tokens cannot exceed context_window")
        # FIX #4: Validate system parameter
        if system is not None and (not isinstance(system, str) or not system.strip()):
            raise ValueError("system must be a non-empty string or None")

        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.system = system
        self.messages: list[MessageParam] = []

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The user message content.

        Raises:
            ValueError: If content is not a non-empty string.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history.

        Args:
            content: The assistant message content.

        Raises:
            ValueError: If content is not a non-empty string.
        """
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> list[MessageParam]:
        """Get the current conversation history.

        Returns:
            A copy of the current message list.
        """
        return list(self.messages)

    def clear(self) -> None:
        """Clear all messages from the conversation history."""
        self.messages = []

    def _estimate_tokens(self, content) -> int:  # FIX #5: Accept any content type
        """Estimate token count for content using simple heuristic.

        This uses a rough approximation: ~4 characters per token for strings,
        ~100 tokens per list item for complex content.
        For production use, consider using the tokenizer or count_tokens API.

        Args:
            content: The content to estimate tokens for (str or list).

        Returns:
            Estimated token count.
        """
        if isinstance(content, str):
            return max(1, len(content) // 4)
        elif isinstance(content, (list, tuple)):
            # For list content (e.g., multiple blocks), estimate ~100 tokens per block
            # as a conservative estimate for tool results, images, etc.
            return max(1, len(content) * 100)
        else:
            # Unknown content type - conservative estimate
            return 100

    def _truncate_messages(self) -> None:
        """Truncate oldest messages if approaching context window limit."""
        # Reserve tokens for: system prompt, max response, and conversation buffer
        reserved_tokens = self.max_tokens + 1000
        if self.system:
            reserved_tokens += self._estimate_tokens(self.system)

        # Calculate tokens used by current messages
        current_tokens = reserved_tokens
        for msg in self.messages:
            current_tokens += self._estimate_tokens(msg["content"])

        # If we exceed context window, remove oldest messages
        while current_tokens > self.context_window and len(self.messages) > 0:
            removed_msg = self.messages.pop(0)
            # FIX #3: Handle all content types
            current_tokens -= self._estimate_tokens(removed_msg["content"])

    async def get_response(self, user_message: str) -> Message:
        """Get a response from the model for the given user message.

        This method adds the user message to history, calls the API, and stores
        the assistant response. Messages are automatically truncated if needed.

        Args:
            user_message: The user's message.

        Returns:
            The Message response from the API.

        Raises:
            ValueError: If user_message is not a non-empty string or exceeds reasonable bounds.
        """
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty string")

        # FIX #3: Validate user message fits in context window
        user_msg_tokens = self._estimate_tokens(user_message)
        reserved = self.max_tokens + 1000
        if self.system:
            reserved += self._estimate_tokens(self.system)
        
        if user_msg_tokens + reserved > self.context_window:
            raise ValueError(
                f"user_message ({user_msg_tokens} tokens) + reserved tokens ({reserved}) "
                f"exceeds context_window ({self.context_window})"
            )

        self._truncate_messages()  # FIX #2: Truncate BEFORE adding user message
        self.add_user_message(user_message)

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            messages=self.messages,
        )

        # FIX #1: Handle all response content types, not just text
        if response.content:
            # Extract all content and add to history
            assistant_content = []
            for block in response.content:
                if block.type == "text":
                    # For text blocks, extract text
                    assistant_content.append(block.text)
                else:
                    # For non-text blocks (tool_use, etc.), we still need to maintain history
                    # Store a reference to the block type for tracking
                    assistant_content.append(f"[{block.type}]")
            
            # Only add to history if we have text content
            # Non-text responses (tool_use) need different handling by the caller
            if any(block.type == "text" for block in response.content):
                text_content = next(
                    (block.text for block in response.content if block.type == "text"),
                    None
                )
                if text_content:
                    self.add_assistant_message(text_content)

        return response
