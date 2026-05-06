"""Conversation management helpers for maintaining multi-turn conversation history with auto-truncation.

Example::

    from anthropic import Anthropic
    from anthropic.helpers import ConversationManager

    client = Anthropic()
    manager = ConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        context_window_limit=100000,
    )

    # First turn
    manager.add_user_message("What is the capital of France?")
    response = manager.get_response()
    print(response.content[0].text)

    # Second turn
    response = manager.get_response("What's its population?")
    print(response.content[0].text)

    # View history
    print(manager.history)
    print(manager.last_usage)
"""

from __future__ import annotations

from typing import Any


class ConversationManager:
    """Manages multi-turn conversation history with automatic truncation.

    Maintains message history across turns and automatically truncates oldest messages
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
            client: Anthropic client instance.
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022").
            max_tokens: Maximum tokens to request in each response.
            system: Optional system prompt.
            context_window_limit: Optional context window limit in tokens.
            token_budget_headroom: Fraction of context window to reserve (0.0 to 1.0).
            accurate_token_counting: If True, use count_tokens API for accurate estimates.

        Raises:
            ValueError: If model is empty, max_tokens < 1, context_window_limit < 1, or
                       token_budget_headroom not in [0.0, 1.0).
        """
        if not model:
            raise ValueError("model must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if context_window_limit is not None and context_window_limit < 1:
            raise ValueError("context_window_limit must be >= 1 if provided")
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
            ValueError: If content is empty string or empty list.
        """
        if isinstance(content, str) and not content:
            raise ValueError("content must not be empty")
        if isinstance(content, list) and not content:
            raise ValueError("content must not be empty")

        self._history.append({"role": "user", "content": content})

    def get_response(
        self, content: str | list[Any] | None = None, **kwargs: Any
    ) -> Any:
        """Get a response from the model, optionally adding a user message first.

        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional arguments to forward to client.messages.create.

        Returns:
            The API response message object.

        Raises:
            ValueError: If no staged user message exists, or if truncation fails.
        """
        if content is not None:
            self.add_user_message(content)

        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("no staged user message")

        if self._context_window_limit is not None:
            self._truncate_if_needed()

        api_kwargs = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        if self._system:
            api_kwargs["system"] = self._system
        api_kwargs.update(kwargs)

        response = self._client.messages.create(**api_kwargs)

        self._history.append({"role": "assistant", "content": response.content})
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Reset conversation history and usage tracking."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return usage information from the last response, or None."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return string representation of the conversation manager."""
        turn_count = len([m for m in self._history if m["role"] == "user"])
        limit_str = (
            f", limit={self._context_window_limit}"
            if self._context_window_limit
            else ""
        )
        return f"ConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if approaching context window limit."""
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        if self._accurate_token_counting:
            estimated_tokens = self._count_tokens()
        else:
            if self._last_usage is None:
                return
            estimated_tokens = (
                self._last_usage.input_tokens + self._last_usage.output_tokens
            )

        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"cannot truncate further — single message pair exceeds limit "
                    f"(model={self._model}, limit={self._context_window_limit}). "
                    f"Consider increasing context_window_limit or decreasing max_tokens."
                )

            pair_fraction = 2 / len(self._history)
            self._history.pop(0)
            self._history.pop(0)

            if self._accurate_token_counting:
                estimated_tokens = self._count_tokens()
            else:
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _count_tokens(self) -> int:
        """Count tokens in current history using the API."""
        api_kwargs = {
            "messages": list(self._history),
            "model": self._model,
        }
        if self._system:
            api_kwargs["system"] = self._system

        response = self._client.messages.count_tokens(**api_kwargs)
        return response.input_tokens


class AsyncConversationManager:
    """Async version of ConversationManager for maintaining multi-turn conversations.

    Maintains message history across turns and automatically truncates oldest messages
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
            client: Async Anthropic client instance.
            model: Model identifier (e.g., "claude-3-5-sonnet-20241022").
            max_tokens: Maximum tokens to request in each response.
            system: Optional system prompt.
            context_window_limit: Optional context window limit in tokens.
            token_budget_headroom: Fraction of context window to reserve (0.0 to 1.0).
            accurate_token_counting: If True, use count_tokens API for accurate estimates.

        Raises:
            ValueError: If model is empty, max_tokens < 1, context_window_limit < 1, or
                       token_budget_headroom not in [0.0, 1.0).
        """
        if not model:
            raise ValueError("model must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if context_window_limit is not None and context_window_limit < 1:
            raise ValueError("context_window_limit must be >= 1 if provided")
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
            ValueError: If content is empty string or empty list.
        """
        if isinstance(content, str) and not content:
            raise ValueError("content must not be empty")
        if isinstance(content, list) and not content:
            raise ValueError("content must not be empty")

        self._history.append({"role": "user", "content": content})

    async def get_response(
        self, content: str | list[Any] | None = None, **kwargs: Any
    ) -> Any:
        """Get a response from the model, optionally adding a user message first.

        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional arguments to forward to client.messages.create.

        Returns:
            The API response message object.

        Raises:
            ValueError: If no staged user message exists, or if truncation fails.
        """
        if content is not None:
            self.add_user_message(content)

        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("no staged user message")

        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        api_kwargs = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        if self._system:
            api_kwargs["system"] = self._system
        api_kwargs.update(kwargs)

        response = await self._client.messages.create(**api_kwargs)

        self._history.append({"role": "assistant", "content": response.content})
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Reset conversation history and usage tracking."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return usage information from the last response, or None."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return string representation of the conversation manager."""
        turn_count = len([m for m in self._history if m["role"] == "user"])
        limit_str = (
            f", limit={self._context_window_limit}"
            if self._context_window_limit
            else ""
        )
        return f"AsyncConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    async def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if approaching context window limit."""
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        if self._accurate_token_counting:
            estimated_tokens = await self._count_tokens()
        else:
            if self._last_usage is None:
                return
            estimated_tokens = (
                self._last_usage.input_tokens + self._last_usage.output_tokens
            )

        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"cannot truncate further — single message pair exceeds limit "
                    f"(model={self._model}, limit={self._context_window_limit}). "
                    f"Consider increasing context_window_limit or decreasing max_tokens."
                )

            pair_fraction = 2 / len(self._history)
            self._history.pop(0)
            self._history.pop(0)

            if self._accurate_token_counting:
                estimated_tokens = await self._count_tokens()
            else:
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    async def _count_tokens(self) -> int:
        """Count tokens in current history using the API."""
        api_kwargs = {
            "messages": list(self._history),
            "model": self._model,
        }
        if self._system:
            api_kwargs["system"] = self._system

        response = await self._client.messages.count_tokens(**api_kwargs)
        return response.input_tokens
