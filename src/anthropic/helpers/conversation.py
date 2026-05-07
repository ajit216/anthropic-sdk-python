"""ConversationManager helper for maintaining multi-turn conversation history.

The ConversationManager maintains message history across turns and automatically
truncates the oldest messages when approaching the model's context window limit.

Example::

    from anthropic import Anthropic
    from anthropic.helpers import ConversationManager

    client = Anthropic()
    conversation = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
        context_window_limit=200000,
    )

    conversation.add_user_message("What is the capital of France?")
    response = conversation.get_response()
    print(response.content[0].text)

    # For async:
    from anthropic import AsyncAnthropic
    from anthropic.helpers import AsyncConversationManager

    client = AsyncAnthropic()
    conversation = AsyncConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
    )

    conversation.add_user_message("Hello!")
    response = await conversation.get_response()
"""

from __future__ import annotations

from typing import Any, Optional


class ConversationManager:
    """Manages multi-turn conversations with automatic context window management.

    Maintains message history and automatically truncates the oldest messages
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
            max_tokens: Maximum tokens in each response.
            system: Optional system prompt.
            context_window_limit: Optional context window limit in tokens.
                If set, messages will be truncated when approaching this limit.
            token_budget_headroom: Fraction of context window to reserve as buffer.
                Must be in [0.0, 1.0). Default is 0.10 (10%).
            accurate_token_counting: If True, uses client.messages.count_tokens()
                for precise token counting. If False (default), uses estimated tokens
                from last response usage.

        Raises:
            ValueError: If model is empty, max_tokens < 1, context_window_limit < 1,
                or token_budget_headroom not in [0.0, 1.0).
        """
        if not model:
            raise ValueError("model cannot be empty")
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
            raise ValueError("content cannot be empty")
        self._history.append({"role": "user", "content": content})

    def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model.

        If content is provided, it is added as a user message first.
        Automatically truncates history if approaching context window limit.

        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional arguments to pass to messages.create().

        Returns:
            The Message response from the API.

        Raises:
            ValueError: If there's no user message to respond to, or if
                truncation would remove all messages.
        """
        if content is not None:
            self.add_user_message(content)

        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No user message to respond to")

        if self._context_window_limit is not None:
            self._truncate_if_needed()

        api_kwargs = {}
        if self._system is not None:
            api_kwargs["system"] = self._system

        response = self._client.messages.create(
            messages=list(self._history),
            model=self._model,
            max_tokens=self._max_tokens,
            **api_kwargs,
            **kwargs,
        )

        self._history.append({"role": "assistant", "content": response.content})
        self._last_usage = response.usage
        return response

    def reset(self) -> None:
        """Reset conversation history and last usage.

        Model and system prompt are preserved.
        """
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Get a shallow copy of the conversation history.

        Returns:
            A shallow copy of the message history.
        """
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Get usage information from the last response.

        Returns:
            Usage object from the last API response, or None if no response yet.
        """
        return self._last_usage

    def _truncate_if_needed(self) -> None:
        """Truncate history if it exceeds the context window threshold.

        Removes oldest user-assistant message pairs until under threshold.
        Uses either accurate token counting or heuristic estimation.
        """
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        if self._accurate_token_counting:
            estimated_tokens = self._count_tokens(self._history)
        else:
            if self._last_usage is None:
                return
            estimated_tokens = self._last_usage.input_tokens + self._last_usage.output_tokens

        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further — single message pair exceeds context "
                    f"window limit of {self._context_window_limit}. Consider increasing "
                    f"context_window_limit or decreasing max_tokens."
                )

            pair_fraction = 2 / len(self._history)
            self._history.pop(0)
            self._history.pop(0)

            if self._accurate_token_counting:
                estimated_tokens = self._count_tokens(self._history)
            else:
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in the given messages using the API.

        Args:
            messages: The messages to count tokens for.

        Returns:
            The estimated token count.
        """
        response = self._client.messages.count_tokens(
            model=self._model,
            messages=messages,
            system=self._system,
        )
        return response.input_tokens

    def __repr__(self) -> str:
        """Return string representation of the manager."""
        turn_count = len([m for m in self._history if m["role"] == "user"])
        limit_str = f", limit={self._context_window_limit}" if self._context_window_limit else ""
        return f"ConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"


class AsyncConversationManager:
    """Async version of ConversationManager for managing multi-turn conversations.

    Maintains message history and automatically truncates the oldest messages
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
            client: The AsyncAnthropic client instance.
            model: The model to use for API calls.
            max_tokens: Maximum tokens in each response.
            system: Optional system prompt.
            context_window_limit: Optional context window limit in tokens.
                If set, messages will be truncated when approaching this limit.
            token_budget_headroom: Fraction of context window to reserve as buffer.
                Must be in [0.0, 1.0). Default is 0.10 (10%).
            accurate_token_counting: If True, uses client.messages.count_tokens()
                for precise token counting. If False (default), uses estimated tokens
                from last response usage.

        Raises:
            ValueError: If model is empty, max_tokens < 1, context_window_limit < 1,
                or token_budget_headroom not in [0.0, 1.0).
        """
        if not model:
            raise ValueError("model cannot be empty")
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
            raise ValueError("content cannot be empty")
        self._history.append({"role": "user", "content": content})

    async def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model asynchronously.

        If content is provided, it is added as a user message first.
        Automatically truncates history if approaching context window limit.

        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional arguments to pass to messages.create().

        Returns:
            The Message response from the API.

        Raises:
            ValueError: If there's no user message to respond to, or if
                truncation would remove all messages.
        """
        if content is not None:
            self.add_user_message(content)

        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No user message to respond to")

        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        api_kwargs = {}
        if self._system is not None:
            api_kwargs["system"] = self._system

        response = await self._client.messages.create(
            messages=list(self._history),
            model=self._model,
            max_tokens=self._max_tokens,
            **api_kwargs,
            **kwargs,
        )

        self._history.append({"role": "assistant", "content": response.content})
        self._last_usage = response.usage
        return response

    def reset(self) -> None:
        """Reset conversation history and last usage.

        Model and system prompt are preserved.
        """
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Get a shallow copy of the conversation history.

        Returns:
            A shallow copy of the message history.
        """
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Get usage information from the last response.

        Returns:
            Usage object from the last API response, or None if no response yet.
        """
        return self._last_usage

    async def _truncate_if_needed(self) -> None:
        """Truncate history if it exceeds the context window threshold.

        Removes oldest user-assistant message pairs until under threshold.
        Uses either accurate token counting or heuristic estimation.
        """
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        if self._accurate_token_counting:
            estimated_tokens = await self._count_tokens(self._history)
        else:
            if self._last_usage is None:
                return
            estimated_tokens = self._last_usage.input_tokens + self._last_usage.output_tokens

        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further — single message pair exceeds context "
                    f"window limit of {self._context_window_limit}. Consider increasing "
                    f"context_window_limit or decreasing max_tokens."
                )

            pair_fraction = 2 / len(self._history)
            self._history.pop(0)
            self._history.pop(0)

            if self._accurate_token_counting:
                estimated_tokens = await self._count_tokens(self._history)
            else:
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    async def _count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in the given messages using the API.

        Args:
            messages: The messages to count tokens for.

        Returns:
            The estimated token count.
        """
        response = await self._client.messages.count_tokens(
            model=self._model,
            messages=messages,
            system=self._system,
        )
        return response.input_tokens

    def __repr__(self) -> str:
        """Return string representation of the manager."""
        turn_count = len([m for m in self._history if m["role"] == "user"])
        limit_str = f", limit={self._context_window_limit}" if self._context_window_limit else ""
        return f"AsyncConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"
