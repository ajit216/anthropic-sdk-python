"""ConversationManager helper for managing multi-turn conversations.

This module provides the ConversationManager and AsyncConversationManager classes
for maintaining conversation history and auto-truncating messages when approaching
the model's context window limit.

Example::

    from anthropic import Anthropic
    from anthropic.helpers import ConversationManager

    client = Anthropic()
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
        system="You are a helpful assistant.",
        context_window_limit=200000,
    )

    manager.add_user_message("Hello!")
    response = manager.get_response()
    print(response.content[0].text)
"""

from __future__ import annotations

from typing import Any

__all__ = ["ConversationManager", "AsyncConversationManager"]


class ConversationManager:
    """Manages multi-turn conversation history with auto-truncation.

    This helper maintains a conversation history and automatically truncates
    the oldest messages when approaching the model's context window limit.
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
        """Initialize the ConversationManager.

        Args:
            client: The Anthropic client instance.
            model: The model ID to use for API calls.
            max_tokens: Maximum tokens to request in responses.
            system: Optional system prompt for the conversation.
            context_window_limit: Optional context window limit in tokens.
                If set, the manager will auto-truncate history when approaching this limit.
            token_budget_headroom: Fraction of context window to reserve (0.0-1.0).
                Default is 0.10 (10%).
            accurate_token_counting: If True, use count_tokens API for accurate estimates.
                If False (default), use heuristic based on last response usage.

        Raises:
            ValueError: If validation fails.
        """
        if not model or model == "":
            raise ValueError("model cannot be empty")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if context_window_limit is not None and context_window_limit < 1:
            raise ValueError(
                f"context_window_limit must be >= 1, got {context_window_limit}"
            )
        if not (0.0 <= token_budget_headroom < 1.0):
            raise ValueError(
                f"token_budget_headroom must be in [0.0, 1.0), got {token_budget_headroom}"
            )

        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._context_window_limit = context_window_limit
        self._token_budget_headroom = token_budget_headroom
        self._accurate_token_counting = accurate_token_counting
        self._history: list[dict[str, Any]] = []
        self._last_usage: Any = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The message content (text or content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str):
            if not content:
                raise ValueError("content cannot be empty")
            self._history.append({"role": "user", "content": content})
        else:
            if not content:
                raise ValueError("content cannot be empty")
            self._history.append({"role": "user", "content": content})

    def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model.

        If content is provided, it will be added as a user message first.

        Args:
            content: Optional message content to add before getting response.
            **kwargs: Additional arguments to pass to messages.create().

        Returns:
            The model's response.

        Raises:
            ValueError: If no user message is staged, or if truncation fails.
        """
        # Add content if provided
        if content is not None:
            self.add_user_message(content)

        # Validate history
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No user message staged. Call add_user_message() first.")

        # Truncate if needed
        if self._context_window_limit is not None:
            self._truncate_if_needed()

        # Build request kwargs
        request_kwargs = {
            "model": self._model,
            "messages": list(self._history),
            "max_tokens": self._max_tokens,
        }

        if self._system:
            request_kwargs["system"] = self._system

        # Add any additional kwargs
        request_kwargs.update(kwargs)

        # Call API
        response = self._client.messages.create(**request_kwargs)

        # Append assistant response to history
        self._history.append({"role": "assistant", "content": response.content})

        # Store usage
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Reset the conversation history and last usage."""
        self._history.clear()
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return the usage from the last response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return a string representation of the manager."""
        turn_count = sum(1 for msg in self._history if msg["role"] == "user")
        limit_str = (
            f", limit={self._context_window_limit}"
            if self._context_window_limit
            else ""
        )
        return f"ConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    def _truncate_if_needed(self) -> None:
        """Truncate history if approaching context window limit."""
        if not self._context_window_limit:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate current tokens
        if self._accurate_token_counting:
            estimated_tokens = self._count_tokens()
        else:
            # Heuristic: use last usage or skip truncation
            if self._last_usage is None:
                return
            estimated_tokens = (
                self._last_usage.input_tokens + self._last_usage.output_tokens
            )

        # Truncate pairs until under threshold
        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                model_str = self._model
                limit_str = self._context_window_limit
                raise ValueError(
                    f"Cannot truncate further — single message pair "
                    f"exceeds context window. Model: {model_str}, "
                    f"Limit: {limit_str} tokens. "
                    f"Consider increasing context_window_limit or max_tokens."
                )

            # Remove oldest user+assistant pair
            self._history.pop(0)
            self._history.pop(0)

            # Re-estimate tokens
            if self._accurate_token_counting:
                estimated_tokens = self._count_tokens()
            else:
                # Heuristic: estimate reduction from removing 2 messages
                pair_fraction = 2.0 / (len(self._history) + 2)
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _count_tokens(self) -> int:
        """Count tokens in the current history using the API."""
        response = self._client.messages.count_tokens(
            model=self._model,
            messages=list(self._history),
            system=self._system if self._system else None,
        )
        return response.input_tokens


class AsyncConversationManager:
    """Async version of ConversationManager.

    Manages multi-turn conversation history with auto-truncation for async code.
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
        """Initialize the AsyncConversationManager.

        Args:
            client: The async Anthropic client instance.
            model: The model ID to use for API calls.
            max_tokens: Maximum tokens to request in responses.
            system: Optional system prompt for the conversation.
            context_window_limit: Optional context window limit in tokens.
                If set, the manager will auto-truncate history when approaching this limit.
            token_budget_headroom: Fraction of context window to reserve (0.0-1.0).
                Default is 0.10 (10%).
            accurate_token_counting: If True, use count_tokens API for accurate estimates.
                If False (default), use heuristic based on last response usage.

        Raises:
            ValueError: If validation fails.
        """
        if not model or model == "":
            raise ValueError("model cannot be empty")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if context_window_limit is not None and context_window_limit < 1:
            raise ValueError(
                f"context_window_limit must be >= 1, got {context_window_limit}"
            )
        if not (0.0 <= token_budget_headroom < 1.0):
            raise ValueError(
                f"token_budget_headroom must be in [0.0, 1.0), got {token_budget_headroom}"
            )

        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._context_window_limit = context_window_limit
        self._token_budget_headroom = token_budget_headroom
        self._accurate_token_counting = accurate_token_counting
        self._history: list[dict[str, Any]] = []
        self._last_usage: Any = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The message content (text or content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str):
            if not content:
                raise ValueError("content cannot be empty")
            self._history.append({"role": "user", "content": content})
        else:
            if not content:
                raise ValueError("content cannot be empty")
            self._history.append({"role": "user", "content": content})

    async def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model asynchronously.

        If content is provided, it will be added as a user message first.

        Args:
            content: Optional message content to add before getting response.
            **kwargs: Additional arguments to pass to messages.create().

        Returns:
            The model's response.

        Raises:
            ValueError: If no user message is staged, or if truncation fails.
        """
        # Add content if provided
        if content is not None:
            self.add_user_message(content)

        # Validate history
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No user message staged. Call add_user_message() first.")

        # Truncate if needed
        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        # Build request kwargs
        request_kwargs = {
            "model": self._model,
            "messages": list(self._history),
            "max_tokens": self._max_tokens,
        }

        if self._system:
            request_kwargs["system"] = self._system

        # Add any additional kwargs
        request_kwargs.update(kwargs)

        # Call API
        response = await self._client.messages.create(**request_kwargs)

        # Append assistant response to history
        self._history.append({"role": "assistant", "content": response.content})

        # Store usage
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Reset the conversation history and last usage."""
        self._history.clear()
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return the usage from the last response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return a string representation of the manager."""
        turn_count = sum(1 for msg in self._history if msg["role"] == "user")
        limit_str = (
            f", limit={self._context_window_limit}"
            if self._context_window_limit
            else ""
        )
        return f"AsyncConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    async def _truncate_if_needed(self) -> None:
        """Truncate history if approaching context window limit."""
        if not self._context_window_limit:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate current tokens
        if self._accurate_token_counting:
            estimated_tokens = await self._count_tokens()
        else:
            # Heuristic: use last usage or skip truncation
            if self._last_usage is None:
                return
            estimated_tokens = (
                self._last_usage.input_tokens + self._last_usage.output_tokens
            )

        # Truncate pairs until under threshold
        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                model_str = self._model
                limit_str = self._context_window_limit
                raise ValueError(
                    f"Cannot truncate further — single message pair "
                    f"exceeds context window. Model: {model_str}, "
                    f"Limit: {limit_str} tokens. "
                    f"Consider increasing context_window_limit or max_tokens."
                )

            # Remove oldest user+assistant pair
            self._history.pop(0)
            self._history.pop(0)

            # Re-estimate tokens
            if self._accurate_token_counting:
                estimated_tokens = await self._count_tokens()
            else:
                # Heuristic: estimate reduction from removing 2 messages
                pair_fraction = 2.0 / (len(self._history) + 2)
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    async def _count_tokens(self) -> int:
        """Count tokens in the current history using the API."""
        response = await self._client.messages.count_tokens(
            model=self._model,
            messages=list(self._history),
            system=self._system if self._system else None,
        )
        return response.input_tokens
