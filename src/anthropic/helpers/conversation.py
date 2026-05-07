"""Multi-turn conversation manager with context window management.

Example::

    from anthropic import Anthropic

    client = Anthropic()
    manager = client.helpers.conversation(
        client=client,
        model="claude-3-5-sonnet-latest",
        max_tokens=512,
        system="You are a helpful assistant."
    )

    manager.add_user_message("What is 2+2?")
    response = manager.get_response()
    print(response.content[0].text)

    # Multi-turn conversation
    manager.add_user_message("What about 3+3?")
    response = manager.get_response()
    print(response.content[0].text)

    # Access conversation history
    print(manager.history)
"""

from __future__ import annotations

from typing import Any


class ConversationManager:
    """Manages multi-turn conversations with automatic context window truncation.

    This helper maintains a message history and automatically truncates old messages
    when approaching the model's context window limit, preventing context overflow errors.

    Attributes:
        model: The model ID to use for API calls
        max_tokens: Maximum tokens for each response
        system: Optional system prompt
        context_window_limit: Maximum context window size (None = no truncation)
        token_budget_headroom: Fraction of context window to reserve (0.0-1.0)
        accurate_token_counting: Use count_tokens API vs heuristic estimation
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
            model: Model ID to use
            max_tokens: Maximum tokens per response
            system: Optional system prompt
            context_window_limit: Context window size limit for truncation
            token_budget_headroom: Fraction to reserve (0.0-1.0), default 0.10
            accurate_token_counting: Use count_tokens API if True

        Raises:
            ValueError: If model is empty, max_tokens < 1, context_window_limit < 1,
                       or token_budget_headroom not in [0.0, 1.0)
        """
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
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
        self._history: list[Any] = []
        self._last_usage: Any | None = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to conversation history.

        Args:
            content: Message content (string or list of content blocks)

        Raises:
            ValueError: If content is empty
        """
        if isinstance(content, str) and not content:
            raise ValueError("content cannot be empty")
        if isinstance(content, list) and not content:
            raise ValueError("content cannot be empty")

        self._history.append({"role": "user", "content": content})

    def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the API.

        If content is provided, adds it as a user message first.
        Truncates history if needed before calling the API.

        Args:
            content: Optional content to add as user message before getting response
            **kwargs: Additional arguments to pass to messages.create()

        Returns:
            Message object from API response

        Raises:
            ValueError: If history is empty or last message isn't a user message
            ValueError: If single message pair exceeds context window limit
        """
        if content is not None:
            self.add_user_message(content)

        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No staged user message for get_response()")

        if self._context_window_limit is not None:
            self._truncate_if_needed()

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=list(self._history),
            **({"system": self._system} if self._system else {}),
            **kwargs,
        )

        self._history.append({"role": "assistant", "content": response.content})
        self._last_usage = response.usage
        return response

    def reset(self) -> None:
        """Clear conversation history and usage stats."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[Any]:
        """Return a shallow copy of conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return usage stats from the last API response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return string representation."""
        turn_count = len([m for m in self._history if m["role"] == "user"])
        limit_str = f", limit={self._context_window_limit}" if self._context_window_limit else ""
        return f"ConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if approaching context window limit.

        Drops oldest user+assistant pairs to maintain role alternation.
        Uses count_tokens API in accurate mode, or estimates from last_usage otherwise.

        Raises:
            ValueError: If single message pair still exceeds the threshold
        """
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        if self._accurate_token_counting:
            estimated_tokens = self._client.messages.count_tokens(
                model=self._model,
                messages=self._history,
                system=self._system,
            ).input_tokens
        else:
            if self._last_usage is None:
                return
            estimated_tokens = self._last_usage.input_tokens + self._last_usage.output_tokens

        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further: single message pair "
                    f"({estimated_tokens} tokens) exceeds context window limit "
                    f"({self._context_window_limit} tokens). "
                    f"Consider increasing context_window_limit or reducing max_tokens."
                )

            pair_fraction = 2.0 / len(self._history)
            self._history.pop(0)
            self._history.pop(0)

            if self._accurate_token_counting:
                estimated_tokens = self._client.messages.count_tokens(
                    model=self._model,
                    messages=self._history,
                    system=self._system,
                ).input_tokens
            else:
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))


class AsyncConversationManager:
    """Async version of ConversationManager for use with AsyncAnthropic client.

    All methods and properties are identical to ConversationManager except
    get_response() is async.
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
            model: Model ID to use
            max_tokens: Maximum tokens per response
            system: Optional system prompt
            context_window_limit: Context window size limit for truncation
            token_budget_headroom: Fraction to reserve (0.0-1.0), default 0.10
            accurate_token_counting: Use count_tokens API if True

        Raises:
            ValueError: If model is empty, max_tokens < 1, context_window_limit < 1,
                       or token_budget_headroom not in [0.0, 1.0)
        """
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
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
        self._history: list[Any] = []
        self._last_usage: Any | None = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to conversation history.

        Args:
            content: Message content (string or list of content blocks)

        Raises:
            ValueError: If content is empty
        """
        if isinstance(content, str) and not content:
            raise ValueError("content cannot be empty")
        if isinstance(content, list) and not content:
            raise ValueError("content cannot be empty")

        self._history.append({"role": "user", "content": content})

    async def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the API (async).

        If content is provided, adds it as a user message first.
        Truncates history if needed before calling the API.

        Args:
            content: Optional content to add as user message before getting response
            **kwargs: Additional arguments to pass to messages.create()

        Returns:
            Message object from API response

        Raises:
            ValueError: If history is empty or last message isn't a user message
            ValueError: If single message pair exceeds context window limit
        """
        if content is not None:
            self.add_user_message(content)

        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No staged user message for get_response()")

        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=list(self._history),
            **({"system": self._system} if self._system else {}),
            **kwargs,
        )

        self._history.append({"role": "assistant", "content": response.content})
        self._last_usage = response.usage
        return response

    def reset(self) -> None:
        """Clear conversation history and usage stats."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[Any]:
        """Return a shallow copy of conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return usage stats from the last API response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return string representation."""
        turn_count = len([m for m in self._history if m["role"] == "user"])
        limit_str = f", limit={self._context_window_limit}" if self._context_window_limit else ""
        return f"AsyncConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    async def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if approaching context window limit (async).

        Drops oldest user+assistant pairs to maintain role alternation.
        Uses count_tokens API in accurate mode, or estimates from last_usage otherwise.

        Raises:
            ValueError: If single message pair still exceeds the threshold
        """
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        if self._accurate_token_counting:
            estimated_tokens = (
                await self._client.messages.count_tokens(
                    model=self._model,
                    messages=self._history,
                    system=self._system,
                )
            ).input_tokens
        else:
            if self._last_usage is None:
                return
            estimated_tokens = self._last_usage.input_tokens + self._last_usage.output_tokens

        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further: single message pair "
                    f"({estimated_tokens} tokens) exceeds context window limit "
                    f"({self._context_window_limit} tokens). "
                    f"Consider increasing context_window_limit or reducing max_tokens."
                )

            pair_fraction = 2.0 / len(self._history)
            self._history.pop(0)
            self._history.pop(0)

            if self._accurate_token_counting:
                estimated_tokens = (
                    await self._client.messages.count_tokens(
                        model=self._model,
                        messages=self._history,
                        system=self._system,
                    )
                ).input_tokens
            else:
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))
