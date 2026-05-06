"""ConversationManager helper for maintaining multi-turn conversation history.

Example::

    from anthropic import Anthropic
    from anthropic.helpers import ConversationManager

    client = Anthropic()
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        context_window_limit=200000,
    )

    # First turn
    manager.add_user_message("Hello, what is 2+2?")
    response = manager.get_response()
    print(response.content[0].text)

    # Second turn
    manager.add_user_message("What about 3+3?")
    response = manager.get_response()
    print(response.content[0].text)

    # View history
    print(manager.history)
    print(manager.last_usage)

    # Reset for new conversation
    manager.reset()
"""

from __future__ import annotations

from typing import Any

__all__ = ["ConversationManager", "AsyncConversationManager"]


class ConversationManager:
    """Maintains multi-turn conversation history with auto-truncation.

    The ConversationManager helps manage conversation state across multiple
    turns, automatically truncating older messages when approaching the
    model's context window limit.

    Args:
        client: The Anthropic client instance.
        model: The model identifier (e.g., "claude-3-5-sonnet-20241022").
        max_tokens: Maximum tokens for the response.
        system: Optional system prompt to prepend to all requests.
        context_window_limit: Optional limit for total context window.
            If set, messages will be truncated when approaching this limit.
        token_budget_headroom: Fraction of context_window_limit to reserve
            as headroom before truncation. Must be in [0.0, 1.0).
        accurate_token_counting: If True, use client.messages.count_tokens()
            for precise token estimates. If False, use heuristic based on
            last response usage.
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
        """Initialize the ConversationManager."""
        # Validate inputs
        if not model:
            raise ValueError("model cannot be an empty string")
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
        self._history: list[Any] = []
        self._last_usage: Any = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The user message content (string or list of content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if not content:
            raise ValueError("User message content cannot be empty")
        self._history.append({"role": "user", "content": content})

    def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model.

        If content is provided, it will be added as a user message first.
        Otherwise, the last message in history must be from the user.

        Args:
            content: Optional user message content to add before getting response.
            **kwargs: Additional keyword arguments to pass to client.messages.create().

        Returns:
            The Message response from the API.

        Raises:
            ValueError: If no user message is staged or if history cannot be truncated.
        """
        # Add user message if content is provided
        if content is not None:
            self.add_user_message(content)

        # Ensure there's a user message to respond to
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError(
                "No user message to respond to. Either provide content to get_response() "
                "or call add_user_message() first."
            )

        # Truncate if needed
        if self._context_window_limit is not None:
            self._truncate_if_needed()

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        if self._system is not None:
            request_kwargs["system"] = self._system
        request_kwargs.update(kwargs)

        # Get response from API
        response = self._client.messages.create(**request_kwargs)

        # Append assistant response to history
        self._history.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )

        # Store usage information
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Reset the conversation history and usage tracking.

        Model and system prompt are preserved.
        """
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[Any]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return the usage information from the last response, if any."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return a string representation of the ConversationManager."""
        turn_count = len([m for m in self._history if m["role"] == "assistant"])
        limit_str = (
            f", limit={self._context_window_limit}"
            if self._context_window_limit
            else ""
        )
        return f"ConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    def _truncate_if_needed(self) -> None:
        """Truncate conversation history if approaching context window limit.

        Raises:
            ValueError: If a single message pair exceeds the limit.
        """
        if self._context_window_limit is None:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate current token count
        estimated_tokens = self._estimate_tokens()

        # If we can't estimate (first call), skip truncation
        if estimated_tokens is None:
            return

        # Truncate until under threshold
        while estimated_tokens >= threshold:
            # Need at least 2 messages (user + assistant pair)
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further: conversation history is too large "
                    f"for context_window_limit={self._context_window_limit}. "
                    f"Even a single message pair exceeds the limit "
                    f"(threshold={threshold}). "
                    f"Consider increasing context_window_limit or decreasing "
                    f"token_budget_headroom for model {self._model!r}."
                )

            # Remove oldest message pair (user + assistant)
            pair_fraction = 2.0 / len(self._history)
            self._history.pop(0)  # oldest user
            self._history.pop(0)  # oldest assistant

            # Re-estimate tokens
            if self._accurate_token_counting:
                estimated_tokens = self._estimate_tokens()
                if estimated_tokens is None:
                    # Fallback if counting fails
                    break
            else:
                # Use heuristic: assume tokens scale proportionally
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _estimate_tokens(self) -> int | None:
        """Estimate the token count of the current history.

        Returns:
            Estimated token count, or None if cannot estimate.
        """
        if self._accurate_token_counting:
            # Use the client's count_tokens method
            try:
                result = self._client.messages.count_tokens(
                    messages=self._history,
                    model=self._model,
                    system=self._system,
                )
                return result.input_tokens
            except Exception:
                return None
        else:
            # Use last_usage as heuristic
            if self._last_usage is None:
                return None
            return self._last_usage.input_tokens + self._last_usage.output_tokens


class AsyncConversationManager:
    """Async version of ConversationManager.

    Maintains multi-turn conversation history with auto-truncation,
    with support for async/await.

    Args:
        client: The Anthropic async client instance.
        model: The model identifier (e.g., "claude-3-5-sonnet-20241022").
        max_tokens: Maximum tokens for the response.
        system: Optional system prompt to prepend to all requests.
        context_window_limit: Optional limit for total context window.
            If set, messages will be truncated when approaching this limit.
        token_budget_headroom: Fraction of context_window_limit to reserve
            as headroom before truncation. Must be in [0.0, 1.0).
        accurate_token_counting: If True, use client.messages.count_tokens()
            for precise token estimates. If False, use heuristic based on
            last response usage.
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
        """Initialize the AsyncConversationManager."""
        # Validate inputs
        if not model:
            raise ValueError("model cannot be an empty string")
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
        self._history: list[Any] = []
        self._last_usage: Any = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The user message content (string or list of content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if not content:
            raise ValueError("User message content cannot be empty")
        self._history.append({"role": "user", "content": content})

    async def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model asynchronously.

        If content is provided, it will be added as a user message first.
        Otherwise, the last message in history must be from the user.

        Args:
            content: Optional user message content to add before getting response.
            **kwargs: Additional keyword arguments to pass to client.messages.create().

        Returns:
            The Message response from the API.

        Raises:
            ValueError: If no user message is staged or if history cannot be truncated.
        """
        # Add user message if content is provided
        if content is not None:
            self.add_user_message(content)

        # Ensure there's a user message to respond to
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError(
                "No user message to respond to. Either provide content to get_response() "
                "or call add_user_message() first."
            )

        # Truncate if needed
        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        if self._system is not None:
            request_kwargs["system"] = self._system
        request_kwargs.update(kwargs)

        # Get response from API
        response = await self._client.messages.create(**request_kwargs)

        # Append assistant response to history
        self._history.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )

        # Store usage information
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Reset the conversation history and usage tracking.

        Model and system prompt are preserved.
        """
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[Any]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return the usage information from the last response, if any."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return a string representation of the AsyncConversationManager."""
        turn_count = len([m for m in self._history if m["role"] == "assistant"])
        limit_str = (
            f", limit={self._context_window_limit}"
            if self._context_window_limit
            else ""
        )
        return f"AsyncConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    async def _truncate_if_needed(self) -> None:
        """Truncate conversation history if approaching context window limit.

        Raises:
            ValueError: If a single message pair exceeds the limit.
        """
        if self._context_window_limit is None:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate current token count
        estimated_tokens = await self._estimate_tokens()

        # If we can't estimate (first call), skip truncation
        if estimated_tokens is None:
            return

        # Truncate until under threshold
        while estimated_tokens >= threshold:
            # Need at least 2 messages (user + assistant pair)
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further: conversation history is too large "
                    f"for context_window_limit={self._context_window_limit}. "
                    f"Even a single message pair exceeds the limit "
                    f"(threshold={threshold}). "
                    f"Consider increasing context_window_limit or decreasing "
                    f"token_budget_headroom for model {self._model!r}."
                )

            # Remove oldest message pair (user + assistant)
            pair_fraction = 2.0 / len(self._history)
            self._history.pop(0)  # oldest user
            self._history.pop(0)  # oldest assistant

            # Re-estimate tokens
            if self._accurate_token_counting:
                estimated_tokens = await self._estimate_tokens()
                if estimated_tokens is None:
                    # Fallback if counting fails
                    break
            else:
                # Use heuristic: assume tokens scale proportionally
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    async def _estimate_tokens(self) -> int | None:
        """Estimate the token count of the current history.

        Returns:
            Estimated token count, or None if cannot estimate.
        """
        if self._accurate_token_counting:
            # Use the client's count_tokens method
            try:
                result = await self._client.messages.count_tokens(
                    messages=self._history,
                    model=self._model,
                    system=self._system,
                )
                return result.input_tokens
            except Exception:
                return None
        else:
            # Use last_usage as heuristic
            if self._last_usage is None:
                return None
            return self._last_usage.input_tokens + self._last_usage.output_tokens
