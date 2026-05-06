"""ConversationManager helper for maintaining multi-turn conversation history.

Example::

    from anthropic import Anthropic
    from anthropic.helpers import ConversationManager

    client = Anthropic()
    manager = ConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
        context_window_limit=200000,
    )

    manager.add_user_message("Hello!")
    response = manager.get_response()
    print(response.content[0].text)

    # For async:
    from anthropic.helpers import AsyncConversationManager
    import asyncio

    async def main():
        async_manager = AsyncConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        response = await async_manager.get_response("Hello!")
        print(response.content[0].text)

    asyncio.run(main())
"""

from __future__ import annotations

from typing import Any, Optional


class ConversationManager:
    """Manages multi-turn conversation history with automatic context truncation.
    
    This helper maintains conversation state and automatically truncates the oldest
    messages when approaching the model's context window limit.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        context_window_limit: int | None = None,
        token_budget_headroom: float = 0.10,
        accurate_token_counting: bool = False,
    ) -> None:
        """Initialize ConversationManager.
        
        Args:
            client: Anthropic client instance
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022")
            max_tokens: Maximum tokens for response generation
            system: Optional system prompt
            context_window_limit: Optional context window limit in tokens
            token_budget_headroom: Fraction of context to reserve (0.0 to 1.0)
            accurate_token_counting: Whether to use accurate token counting via API
            
        Raises:
            ValueError: If model is empty, max_tokens < 1, context_window_limit < 1,
                       or token_budget_headroom not in [0.0, 1.0)
        """
        if not model:
            raise ValueError("model must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if context_window_limit is not None and context_window_limit < 1:
            raise ValueError("context_window_limit must be >= 1")
        if not (0.0 <= token_budget_headroom < 1.0):
            raise ValueError("token_budget_headroom must be in [0.0, 1.0)")

        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._context_window_limit = context_window_limit
        self._token_budget_headroom = token_budget_headroom
        self._accurate_token_counting = accurate_token_counting
        self._history: list[dict[str, Any]] = []
        self._last_usage: Any | None = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to conversation history.
        
        Args:
            content: User message content (string or list of content blocks)
            
        Raises:
            ValueError: If content is empty
        """
        if isinstance(content, str) and not content:
            raise ValueError("User message content cannot be empty")
        if isinstance(content, list) and not content:
            raise ValueError("User message content cannot be empty")

        self._history.append({"role": "user", "content": content})

    def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model.
        
        Args:
            content: Optional user message to add before getting response
            **kwargs: Additional arguments passed to messages.create()
            
        Returns:
            The API response
            
        Raises:
            ValueError: If no user message is staged or if truncation fails
        """
        # Add user message if provided
        if content is not None:
            self.add_user_message(content)

        # Ensure last message is from user
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No user message staged for response generation")

        # Truncate if needed
        if self._context_window_limit is not None:
            self._truncate_if_needed()

        # Build request arguments
        request_args = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }

        # Add system prompt if provided
        if self._system is not None:
            request_args["system"] = self._system

        # Merge with user kwargs
        request_args.update(kwargs)

        # Get response
        response = self._client.messages.create(**request_args)

        # Update history and usage
        self._history.append({"role": "assistant", "content": response.content})
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Clear conversation history and usage."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Get a shallow copy of conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Get usage statistics from the last response."""
        return self._last_usage

    def __repr__(self) -> str:
        """String representation of the ConversationManager."""
        turn_count = len(self._history) // 2
        return (
            f"ConversationManager(model={self._model!r}, "
            f"turns={turn_count}, "
            f"context_limit={self._context_window_limit})"
        )

    def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if approaching context window limit."""
        if self._context_window_limit is None:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate tokens
        estimated_tokens = self._estimate_tokens()

        # If no usage data yet, skip truncation
        if estimated_tokens is None:
            return

        # Truncate pairs until under threshold
        while estimated_tokens >= threshold:
            # Check if we can remove another pair (need at least 2 messages remaining)
            if len(self._history) <= 2:
                raise ValueError(
                    f"Cannot truncate further — single message pair exceeds limit. "
                    f"Model: {self._model}, Context limit: {self._context_window_limit} tokens. "
                    f"Consider increasing context_window_limit or reducing max_tokens."
                )

            # Remove oldest user+assistant pair
            self._history.pop(0)
            self._history.pop(0)

            # Re-estimate tokens
            if self._accurate_token_counting:
                estimated_tokens = self._estimate_tokens()
            else:
                # Use heuristic: reduce by proportion of removed messages
                pair_fraction = 2.0 / (len(self._history) + 2)
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _estimate_tokens(self) -> int | None:
        """Estimate tokens in current history + response tokens.
        
        Returns:
            Estimated token count or None if no usage data available
        """
        if self._accurate_token_counting:
            # Call API for accurate count
            count_args = {
                "model": self._model,
                "messages": list(self._history),
            }
            if self._system is not None:
                count_args["system"] = self._system

            response = self._client.messages.count_tokens(**count_args)
            return response.input_tokens + self._max_tokens
        else:
            # Use last usage data
            if self._last_usage is None:
                return None
            return self._last_usage.input_tokens + self._last_usage.output_tokens


class AsyncConversationManager:
    """Async version of ConversationManager for maintaining multi-turn conversations.
    
    This helper maintains conversation state and automatically truncates the oldest
    messages when approaching the model's context window limit.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        max_tokens: int,
        system: str | None = None,
        context_window_limit: int | None = None,
        token_budget_headroom: float = 0.10,
        accurate_token_counting: bool = False,
    ) -> None:
        """Initialize AsyncConversationManager.
        
        Args:
            client: AsyncAnthropic client instance
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022")
            max_tokens: Maximum tokens for response generation
            system: Optional system prompt
            context_window_limit: Optional context window limit in tokens
            token_budget_headroom: Fraction of context to reserve (0.0 to 1.0)
            accurate_token_counting: Whether to use accurate token counting via API
            
        Raises:
            ValueError: If model is empty, max_tokens < 1, context_window_limit < 1,
                       or token_budget_headroom not in [0.0, 1.0)
        """
        if not model:
            raise ValueError("model must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if context_window_limit is not None and context_window_limit < 1:
            raise ValueError("context_window_limit must be >= 1")
        if not (0.0 <= token_budget_headroom < 1.0):
            raise ValueError("token_budget_headroom must be in [0.0, 1.0)")

        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._context_window_limit = context_window_limit
        self._token_budget_headroom = token_budget_headroom
        self._accurate_token_counting = accurate_token_counting
        self._history: list[dict[str, Any]] = []
        self._last_usage: Any | None = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to conversation history.
        
        Args:
            content: User message content (string or list of content blocks)
            
        Raises:
            ValueError: If content is empty
        """
        if isinstance(content, str) and not content:
            raise ValueError("User message content cannot be empty")
        if isinstance(content, list) and not content:
            raise ValueError("User message content cannot be empty")

        self._history.append({"role": "user", "content": content})

    async def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model asynchronously.
        
        Args:
            content: Optional user message to add before getting response
            **kwargs: Additional arguments passed to messages.create()
            
        Returns:
            The API response
            
        Raises:
            ValueError: If no user message is staged or if truncation fails
        """
        # Add user message if provided
        if content is not None:
            self.add_user_message(content)

        # Ensure last message is from user
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No user message staged for response generation")

        # Truncate if needed
        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        # Build request arguments
        request_args = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }

        # Add system prompt if provided
        if self._system is not None:
            request_args["system"] = self._system

        # Merge with user kwargs
        request_args.update(kwargs)

        # Get response
        response = await self._client.messages.create(**request_args)

        # Update history and usage
        self._history.append({"role": "assistant", "content": response.content})
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Clear conversation history and usage."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Get a shallow copy of conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Get usage statistics from the last response."""
        return self._last_usage

    def __repr__(self) -> str:
        """String representation of the AsyncConversationManager."""
        turn_count = len(self._history) // 2
        return (
            f"AsyncConversationManager(model={self._model!r}, "
            f"turns={turn_count}, "
            f"context_limit={self._context_window_limit})"
        )

    async def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if approaching context window limit."""
        if self._context_window_limit is None:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate tokens
        estimated_tokens = await self._estimate_tokens()

        # If no usage data yet, skip truncation
        if estimated_tokens is None:
            return

        # Truncate pairs until under threshold
        while estimated_tokens >= threshold:
            # Check if we can remove another pair (need at least 2 messages remaining)
            if len(self._history) <= 2:
                raise ValueError(
                    f"Cannot truncate further — single message pair exceeds limit. "
                    f"Model: {self._model}, Context limit: {self._context_window_limit} tokens. "
                    f"Consider increasing context_window_limit or reducing max_tokens."
                )

            # Remove oldest user+assistant pair
            self._history.pop(0)
            self._history.pop(0)

            # Re-estimate tokens
            if self._accurate_token_counting:
                estimated_tokens = await self._estimate_tokens()
            else:
                # Use heuristic: reduce by proportion of removed messages
                pair_fraction = 2.0 / (len(self._history) + 2)
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    async def _estimate_tokens(self) -> int | None:
        """Estimate tokens in current history + response tokens.
        
        Returns:
            Estimated token count or None if no usage data available
        """
        if self._accurate_token_counting:
            # Call API for accurate count
            count_args = {
                "model": self._model,
                "messages": list(self._history),
            }
            if self._system is not None:
                count_args["system"] = self._system

            response = await self._client.messages.count_tokens(**count_args)
            return response.input_tokens + self._max_tokens
        else:
            # Use last usage data
            if self._last_usage is None:
                return None
            return self._last_usage.input_tokens + self._last_usage.output_tokens
