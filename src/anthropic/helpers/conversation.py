"""Conversation management helper for handling multi-turn conversation history."""

import asyncio
from collections import deque
from typing import Optional, List, Any
import anthropic
from anthropic.types import MessageParam, Message


def _estimate_token_count(text: str) -> int:
    """Estimate token count for text using simple heuristic.
    
    Claude models typically use roughly 1 token per 4 characters.
    """
    return max(1, len(text) // 4)


def _count_message_tokens(message: MessageParam) -> int:
    """Count estimated tokens for a message."""
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    
    if isinstance(content, str):
        return _estimate_token_count(content)
    elif isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                total += _estimate_token_count(text)
            else:
                text = getattr(block, "text", "")
                total += _estimate_token_count(text)
        return total
    return 1


class ConversationManager:
    """Manages multi-turn conversation history with automatic context window management.
    
    This helper maintains conversation history and automatically truncates older messages
    when approaching the context window limit, allowing developers to build chatbots
    without manually managing conversation state.
    
    Note: This class is not designed for concurrent access. For concurrent use,
    use AsyncConversationManager which provides async-safe message handling.
    
    Args:
        client: The Anthropic client to use for API calls. Must not be None.
        max_tokens: Maximum tokens for the conversation history. Must be > 0.
        model: The model to use for API calls (e.g., "claude-3-5-sonnet-20241022").
               Must not be None or empty string.
        system: Optional system prompt to prepend to all requests.
    
    Raises:
        ValueError: If max_tokens <= 0, client is None, model is None/empty, or other invalid parameters.
    """
    
    def __init__(
        self,
        client: anthropic.Anthropic,
        max_tokens: int,
        model: str,
        system: Optional[str] = None,
    ):
        # Validate client
        if client is None:
            raise ValueError("client must not be None")
        
        # Validate max_tokens
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be an integer greater than 0")
        
        # Validate model
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
        
        # Validate system prompt if provided
        if system is not None and not isinstance(system, str):
            raise ValueError("system must be a string or None")
        
        self.client = client
        self.max_tokens = max_tokens
        self.model = model
        self.system = system
        self.messages: deque[MessageParam] = deque()
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history.
        
        Args:
            content: The user message content.
        """
        self.messages.append({"role": "user", "content": content})
        self._truncate_if_needed()
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history.
        
        Args:
            content: The assistant message content.
        """
        self.messages.append({"role": "assistant", "content": content})
        self._truncate_if_needed()
    
    def get_messages(self) -> List[MessageParam]:
        """Get the current conversation history.
        
        Returns:
            List of message dictionaries.
        """
        return list(self.messages)
    
    def get_conversation_tokens(self) -> int:
        """Get the estimated token count of current conversation history.
        
        Returns:
            Estimated token count.
        """
        return sum(_count_message_tokens(msg) for msg in self.messages)
    
    def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if conversation exceeds max_tokens.
        
        Preserves minimum viable context by keeping at least 2 messages
        (e.g., user request + assistant response) to maintain conversation flow.
        """
        # Keep at least 2 messages to preserve minimum conversation context
        while self.get_conversation_tokens() > self.max_tokens and len(self.messages) > 2:
            self.messages.popleft()
    
    def create_message(
        self,
        user_message: str,
        **kwargs: Any,
    ) -> Message:
        """Send a user message and get assistant response.
        
        This method adds the user message, calls the API, and adds the response
        to the conversation history automatically.
        
        Note: Streaming is not supported. Do not pass stream=True in kwargs.
        
        Args:
            user_message: The user message content.
            **kwargs: Additional arguments to pass to client.messages.create().
                     Do not use stream=True as it is not supported.
        
        Returns:
            The Message response from the API.
        
        Raises:
            ValueError: If stream=True is passed in kwargs.
        """
        if kwargs.get("stream") is True:
            raise ValueError("Streaming is not supported by ConversationManager. Use client.messages.stream() directly.")
        
        self.add_user_message(user_message)
        
        # Build the messages list
        messages = self.get_messages()
        
        # Prepare kwargs
        api_kwargs = {
            "model": self.model,
            "max_tokens": 2048,  # Default response max tokens
            "messages": messages,
        }
        
        # Add system prompt if provided
        if self.system:
            api_kwargs["system"] = self.system
        
        # Override with any user-provided kwargs
        api_kwargs.update(kwargs)
        
        # Call the API
        response = self.client.messages.create(**api_kwargs)
        
        # Extract assistant response and add to history
        if response.content:
            # Concatenate all text content blocks
            assistant_content = "".join(
                block.text for block in response.content 
                if hasattr(block, "text")
            )
            # Only add if we have non-empty content
            if assistant_content:
                self.add_assistant_message(assistant_content)
        
        return response
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.messages.clear()


class AsyncConversationManager:
    """Async version of ConversationManager for managing multi-turn conversations.
    
    This helper maintains conversation history and automatically truncates older messages
    when approaching the context window limit, allowing developers to build chatbots
    without manually managing conversation state.
    
    This implementation is async-safe for concurrent calls through the use of asyncio.Lock.
    All message modifications (add_user_message, add_assistant_message, create_message)
    are protected from concurrent access conflicts.
    
    Args:
        client: The AsyncAnthropic client to use for API calls. Must not be None.
        max_tokens: Maximum tokens for the conversation history. Must be > 0.
        model: The model to use for API calls (e.g., "claude-3-5-sonnet-20241022").
               Must not be None or empty string.
        system: Optional system prompt to prepend to all requests.
    
    Raises:
        ValueError: If max_tokens <= 0, client is None, model is None/empty, or other invalid parameters.
    """
    
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        max_tokens: int,
        model: str,
        system: Optional[str] = None,
    ):
        # Validate client
        if client is None:
            raise ValueError("client must not be None")
        
        # Validate max_tokens
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be an integer greater than 0")
        
        # Validate model
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
        
        # Validate system prompt if provided
        if system is not None and not isinstance(system, str):
            raise ValueError("system must be a string or None")
        
        self.client = client
        self.max_tokens = max_tokens
        self.model = model
        self.system = system
        self.messages: deque[MessageParam] = deque()
        self._lock = asyncio.Lock()
    
    async def _add_user_message_locked(self, content: str) -> None:
        """Add a user message with async-safe locking.
        
        Args:
            content: The user message content.
        """
        async with self._lock:
            self.messages.append({"role": "user", "content": content})
            self._truncate_if_needed()
    
    async def _add_assistant_message_locked(self, content: str) -> None:
        """Add an assistant message with async-safe locking.
        
        Args:
            content: The assistant message content.
        """
        async with self._lock:
            self.messages.append({"role": "assistant", "content": content})
            self._truncate_if_needed()
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history (non-async).
        
        Warning: This method is not async-safe. For concurrent use, use await
        inside create_message() which handles locking automatically.
        
        Args:
            content: The user message content.
        """
        self.messages.append({"role": "user", "content": content})
        self._truncate_if_needed()
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history (non-async).
        
        Warning: This method is not async-safe. For concurrent use, use await
        inside create_message() which handles locking automatically.
        
        Args:
            content: The assistant message content.
        """
        self.messages.append({"role": "assistant", "content": content})
        self._truncate_if_needed()
    
    def get_messages(self) -> List[MessageParam]:
        """Get the current conversation history.
        
        Note: This is a snapshot and may be slightly stale in concurrent scenarios.
        
        Returns:
            List of message dictionaries.
        """
        return list(self.messages)
    
    def get_conversation_tokens(self) -> int:
        """Get the estimated token count of current conversation history.
        
        Note: This is an estimate and may vary slightly due to async updates.
        
        Returns:
            Estimated token count.
        """
        return sum(_count_message_tokens(msg) for msg in self.messages)
    
    def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if conversation exceeds max_tokens.
        
        Preserves minimum viable context by keeping at least 2 messages
        (e.g., user request + assistant response) to maintain conversation flow.
        
        Note: This method assumes the caller holds the async lock.
        """
        # Keep at least 2 messages to preserve minimum conversation context
        while self.get_conversation_tokens() > self.max_tokens and len(self.messages) > 2:
            self.messages.popleft()
    
    async def create_message(
        self,
        user_message: str,
        **kwargs: Any,
    ) -> Message:
        """Send a user message and get assistant response asynchronously.
        
        This method adds the user message, calls the API, and adds the response
        to the conversation history automatically with async-safe locking.
        
        Note: Streaming is not supported. Do not pass stream=True in kwargs.
        
        Args:
            user_message: The user message content.
            **kwargs: Additional arguments to pass to client.messages.create().
                     Do not use stream=True as it is not supported.
        
        Returns:
            The Message response from the API.
        
        Raises:
            ValueError: If stream=True is passed in kwargs.
        """
        if kwargs.get("stream") is True:
            raise ValueError("Streaming is not supported by AsyncConversationManager. Use client.messages.stream() directly.")
        
        # Add user message with locking
        await self._add_user_message_locked(user_message)
        
        # Build the messages list (snapshot with lock held)
        async with self._lock:
            messages = self.get_messages()
        
        # Prepare kwargs
        api_kwargs = {
            "model": self.model,
            "max_tokens": 2048,  # Default response max tokens
            "messages": messages,
        }
        
        # Add system prompt if provided
        if self.system:
            api_kwargs["system"] = self.system
        
        # Override with any user-provided kwargs
        api_kwargs.update(kwargs)
        
        # Call the API (non-blocking, no lock held)
        response = await self.client.messages.create(**api_kwargs)
        
        # Extract assistant response and add to history with locking
        if response.content:
            # Concatenate all text content blocks
            assistant_content = "".join(
                block.text for block in response.content 
                if hasattr(block, "text")
            )
            # Only add if we have non-empty content
            if assistant_content:
                await self._add_assistant_message_locked(assistant_content)
        
        return response
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.messages.clear()
