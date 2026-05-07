"""ConversationManager: Manages multi-turn conversations with automatic context window management.

This module provides helpers for maintaining conversation history and automatically
truncating messages when approaching a model's context window limit.

Example::

    import anthropic

    client = anthropic.Anthropic()
    manager = anthropic.helpers.ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        context_window_limit=200000,
    )

    manager.add_user_message("What is 2+2?")
    response = manager.get_response()
    print(response.content[0].text)

    manager.add_user_message("What about 3+3?")
    response = manager.get_response()
    print(response.content[0].text)
"""

from __future__ import annotations

from typing import Any


class ConversationManager:
    """Manages multi-turn conversations with automatic context window management.

    Maintains conversation history and automatically truncates the oldest messages
    when approaching the model's context window limit.
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
            client: The Anthropic client instance.
            model: The model to use for API calls.
            max_tokens: Maximum tokens for each response.
            system: Optional system prompt.
            context_window_limit: Maximum context window size. If None, no truncation.
            token_budget_headroom: Fraction of context to reserve (0.0-1.0).
            accurate_token_counting: If True, use count_tokens API. If False, use heuristics.

        Raises:
            ValueError: If model is empty, max_tokens < 1, context_window_limit < 1,
                       or token_budget_headroom not in [0.0, 1.0).
        """
        # Validate inputs
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
        self._last_usage: Any = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The user message content (string or list of content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if not content:
            raise ValueError("content must not be empty")
        self._history.append({"role": "user", "content": content})

    def get_response(
        self, content: str | list[Any] | None = None, **kwargs: Any
    ) -> Any:
        """Get a response from the model.

        If content is provided, adds it as a user message first.
        Automatically truncates history if approaching context window limit.

        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional arguments to pass to client.messages.create().

        Returns:
            The model's response (Message object).

        Raises:
            ValueError: If no user message is staged, or if single message pair
                       exceeds context limit.
        """
        # Add user message if provided
        if content is not None:
            self.add_user_message(content)

        # Validate that history ends with a user message
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No user message staged for get_response()")

        # Truncate if needed
        if self._context_window_limit is not None:
            self._truncate_if_needed()

        # Build request kwargs
        request_kwargs = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        if self._system:
            request_kwargs["system"] = self._system
        request_kwargs.update(kwargs)

        # Call API
        response = self._client.messages.create(**request_kwargs)

        # Append assistant response to history
        self._history.append({"role": "assistant", "content": response.content})

        # Store usage
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Clear conversation history and last usage.

        Preserves model and system prompt settings.
        """
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return the usage from the last API response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return string representation showing model, turn count, and limit."""
        turn_count = len(self._history) // 2
        limit_str = (
            f", limit={self._context_window_limit}"
            if self._context_window_limit
            else ""
        )
        return (
            f"ConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"
        )

    def _truncate_if_needed(self) -> None:
        """Truncate history if approaching context window limit.

        Removes oldest user+assistant message pairs to maintain role alternation.

        Raises:
            ValueError: If single message pair exceeds the limit.
        """
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate current tokens
        if self._accurate_token_counting:
            estimated_tokens = self._count_tokens()
        else:
            # Heuristic mode: use last_usage or skip on first call
            if self._last_usage is None:
                return
            estimated_tokens = (
                self._last_usage.input_tokens + self._last_usage.output_tokens
            )

        # Truncate until under threshold
        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further: single message pair exceeds "
                    f"context limit. Model: {self._model}, "
                    f"Limit: {self._context_window_limit}, "
                    f"Headroom: {self._token_budget_headroom}. "
                    f"Consider increasing context_window_limit or max_tokens."
                )

            # Remove oldest pair (user + assistant)
            pair_fraction = 2.0 / len(self._history)
            self._history.pop(0)  # Remove oldest user
            self._history.pop(0)  # Remove oldest assistant

            # Recalculate tokens
            if self._accurate_token_counting:
                estimated_tokens = self._count_tokens()
            else:
                # Heuristic: scale down by the fraction removed
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _count_tokens(self) -> int:
        """Count tokens in current history using the API.

        Returns:
            Estimated token count.
        """
        result = self._client.messages.count_tokens(
            model=self._model,
            messages=self._history,
            system=self._system if self._system else None,
        )
        return result.input_tokens


class AsyncConversationManager:
    """Async version of ConversationManager for managing multi-turn conversations.

    Maintains conversation history and automatically truncates the oldest messages
    when approaching the model's context window limit.
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
            client: The async Anthropic client instance.
            model: The model to use for API calls.
            max_tokens: Maximum tokens for each response.
            system: Optional system prompt.
            context_window_limit: Maximum context window size. If None, no truncation.
            token_budget_headroom: Fraction of context to reserve (0.0-1.0).
            accurate_token_counting: If True, use count_tokens API. If False, use heuristics.

        Raises:
            ValueError: If model is empty, max_tokens < 1, context_window_limit < 1,
                       or token_budget_headroom not in [0.0, 1.0).
        """
        # Validate inputs
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
        self._last_usage: Any = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The user message content (string or list of content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if not content:
            raise ValueError("content must not be empty")
        self._history.append({"role": "user", "content": content})

    async def get_response(
        self, content: str | list[Any] | None = None, **kwargs: Any
    ) -> Any:
        """Get a response from the model asynchronously.

        If content is provided, adds it as a user message first.
        Automatically truncates history if approaching context window limit.

        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional arguments to pass to client.messages.create().

        Returns:
            The model's response (Message object).

        Raises:
            ValueError: If no user message is staged, or if single message pair
                       exceeds context limit.
        """
        # Add user message if provided
        if content is not None:
            self.add_user_message(content)

        # Validate that history ends with a user message
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No user message staged for get_response()")

        # Truncate if needed
        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        # Build request kwargs
        request_kwargs = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        if self._system:
            request_kwargs["system"] = self._system
        request_kwargs.update(kwargs)

        # Call API
        response = await self._client.messages.create(**request_kwargs)

        # Append assistant response to history
        self._history.append({"role": "assistant", "content": response.content})

        # Store usage
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Clear conversation history and last usage.

        Preserves model and system prompt settings.
        """
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return the usage from the last API response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return string representation showing model, turn count, and limit."""
        turn_count = len(self._history) // 2
        limit_str = (
            f", limit={self._context_window_limit}"
            if self._context_window_limit
            else ""
        )
        return (
            f"AsyncConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"
        )

    async def _truncate_if_needed(self) -> None:
        """Truncate history if approaching context window limit.

        Removes oldest user+assistant message pairs to maintain role alternation.

        Raises:
            ValueError: If single message pair exceeds the limit.
        """
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate current tokens
        if self._accurate_token_counting:
            estimated_tokens = await self._count_tokens()
        else:
            # Heuristic mode: use last_usage or skip on first call
            if self._last_usage is None:
                return
            estimated_tokens = (
                self._last_usage.input_tokens + self._last_usage.output_tokens
            )

        # Truncate until under threshold
        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further: single message pair exceeds "
                    f"context limit. Model: {self._model}, "
                    f"Limit: {self._context_window_limit}, "
                    f"Headroom: {self._token_budget_headroom}. "
                    f"Consider increasing context_window_limit or max_tokens."
                )

            # Remove oldest pair (user + assistant)
            pair_fraction = 2.0 / len(self._history)
            self._history.pop(0)  # Remove oldest user
            self._history.pop(0)  # Remove oldest assistant

            # Recalculate tokens
            if self._accurate_token_counting:
                estimated_tokens = await self._count_tokens()
            else:
                # Heuristic: scale down by the fraction removed
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    async def _count_tokens(self) -> int:
        """Count tokens in current history using the API.

        Returns:
            Estimated token count.
        """
        result = await self._client.messages.count_tokens(
            model=self._model,
            messages=self._history,
            system=self._system if self._system else None,
        )
        return result.input_tokens
