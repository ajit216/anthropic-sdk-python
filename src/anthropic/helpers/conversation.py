"""ConversationManager helper for managing multi-turn conversations with auto-truncation.

Example::

    from anthropic import Anthropic
    from anthropic.helpers import ConversationManager

    client = Anthropic()
    manager = ConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        context_window_limit=200000,
    )

    manager.add_user_message("Hello!")
    response = manager.get_response()
    print(response.content[0].text)

    manager.add_user_message("How are you?")
    response = manager.get_response()
    print(response.content[0].text)
"""

from __future__ import annotations

from typing import Any


class ConversationManager:
    """Manages multi-turn conversations with automatic context window truncation.
    
    Maintains conversation history and automatically removes oldest message pairs
    when approaching the model's context window limit.
    
    Each instance is single-threaded and not thread-safe.
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
            max_tokens: Maximum tokens to request in each response.
            system: Optional system prompt to prepend to all requests.
            context_window_limit: Optional limit on context window size.
                If set, triggers auto-truncation when approaching this limit.
            token_budget_headroom: Fraction of context_window_limit to reserve
                as headroom (default 0.10 = 10%). Must be in [0.0, 1.0).
            accurate_token_counting: If True, use client.messages.count_tokens()
                for precise truncation. If False, use last_usage for estimation.

        Raises:
            ValueError: If model is empty string, max_tokens < 1,
                context_window_limit < 1, or token_budget_headroom not in [0.0, 1.0).
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
        self._last_usage: Any | None = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The message content (string or list of content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str) and not content:
            raise ValueError("content must not be empty")
        if isinstance(content, list) and not content:
            raise ValueError("content must not be empty")

        self._history.append({"role": "user", "content": content})

    def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model, optionally adding user content first.

        Args:
            content: Optional user message content to add before getting response.
            **kwargs: Additional arguments to pass to client.messages.create().

        Returns:
            The API response from client.messages.create().

        Raises:
            ValueError: If history is empty, last message is not from user,
                or truncation fails.
        """
        # Step 1: Add user message if content provided
        if content is not None:
            self.add_user_message(content)

        # Step 2: Validate history state
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No staged user message. Add one with add_user_message() or pass content to get_response().")

        # Step 3: Truncate if needed
        if self._context_window_limit is not None:
            self._truncate_if_needed()

        # Step 4: Call API
        call_kwargs = {
            "model": self._model,
            "messages": list(self._history),
            "max_tokens": self._max_tokens,
        }
        if self._system is not None:
            call_kwargs["system"] = self._system

        call_kwargs.update(kwargs)
        response = self._client.messages.create(**call_kwargs)

        # Step 5: Append assistant response
        self._history.append({"role": "assistant", "content": response.content})

        # Step 6: Store usage
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Clear conversation history and last usage stats.
        
        Model and system prompt remain unchanged.
        """
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return usage stats from the last API response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return a string representation of the manager."""
        turn_count = sum(1 for msg in self._history if msg["role"] == "user")
        limit_str = f", limit={self._context_window_limit}" if self._context_window_limit else ""
        return f"ConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    def _truncate_if_needed(self) -> None:
        """Truncate oldest message pairs if history exceeds threshold."""
        if self._context_window_limit is None:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate tokens
        if self._accurate_token_counting:
            estimated_tokens = self._count_tokens_accurate()
        else:
            estimated_tokens = self._estimate_tokens_heuristic()
            if estimated_tokens is None:
                # First call, no usage data yet
                return

        # Truncate pairs until under threshold
        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further—conversation has single message pair "
                    f"(history too long for context limit {self._context_window_limit}). "
                    f"Consider increasing context_window_limit or decreasing max_tokens."
                )

            pair_fraction = 2.0 / len(self._history)
            self._history.pop(0)  # Remove oldest user message
            self._history.pop(0)  # Remove oldest assistant response

            if self._accurate_token_counting:
                estimated_tokens = self._count_tokens_accurate()
            else:
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _estimate_tokens_heuristic(self) -> int | None:
        """Estimate token count using last API response usage stats."""
        if self._last_usage is None:
            return None

        # Use last response's input + output tokens as rough estimate
        return self._last_usage.input_tokens + self._last_usage.output_tokens

    def _count_tokens_accurate(self) -> int:
        """Count tokens accurately using the client's count_tokens method."""
        response = self._client.messages.count_tokens(
            model=self._model,
            messages=self._history,
            **({"system": self._system} if self._system else {}),
        )
        return response.input_tokens


class AsyncConversationManager:
    """Async version of ConversationManager for use with async client.
    
    Maintains conversation history and automatically removes oldest message pairs
    when approaching the model's context window limit.
    
    Each instance is single-threaded and not thread-safe.
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
            max_tokens: Maximum tokens to request in each response.
            system: Optional system prompt to prepend to all requests.
            context_window_limit: Optional limit on context window size.
                If set, triggers auto-truncation when approaching this limit.
            token_budget_headroom: Fraction of context_window_limit to reserve
                as headroom (default 0.10 = 10%). Must be in [0.0, 1.0).
            accurate_token_counting: If True, use client.messages.count_tokens()
                for precise truncation. If False, use last_usage for estimation.

        Raises:
            ValueError: If model is empty string, max_tokens < 1,
                context_window_limit < 1, or token_budget_headroom not in [0.0, 1.0).
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
        self._last_usage: Any | None = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The message content (string or list of content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str) and not content:
            raise ValueError("content must not be empty")
        if isinstance(content, list) and not content:
            raise ValueError("content must not be empty")

        self._history.append({"role": "user", "content": content})

    async def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model, optionally adding user content first.

        Args:
            content: Optional user message content to add before getting response.
            **kwargs: Additional arguments to pass to client.messages.create().

        Returns:
            The API response from client.messages.create().

        Raises:
            ValueError: If history is empty, last message is not from user,
                or truncation fails.
        """
        # Step 1: Add user message if content provided
        if content is not None:
            self.add_user_message(content)

        # Step 2: Validate history state
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No staged user message. Add one with add_user_message() or pass content to get_response().")

        # Step 3: Truncate if needed
        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        # Step 4: Call API
        call_kwargs = {
            "model": self._model,
            "messages": list(self._history),
            "max_tokens": self._max_tokens,
        }
        if self._system is not None:
            call_kwargs["system"] = self._system

        call_kwargs.update(kwargs)
        response = await self._client.messages.create(**call_kwargs)

        # Step 5: Append assistant response
        self._history.append({"role": "assistant", "content": response.content})

        # Step 6: Store usage
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Clear conversation history and last usage stats.
        
        Model and system prompt remain unchanged.
        """
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return usage stats from the last API response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return a string representation of the manager."""
        turn_count = sum(1 for msg in self._history if msg["role"] == "user")
        limit_str = f", limit={self._context_window_limit}" if self._context_window_limit else ""
        return f"AsyncConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    async def _truncate_if_needed(self) -> None:
        """Truncate oldest message pairs if history exceeds threshold."""
        if self._context_window_limit is None:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate tokens
        if self._accurate_token_counting:
            estimated_tokens = await self._count_tokens_accurate()
        else:
            estimated_tokens = self._estimate_tokens_heuristic()
            if estimated_tokens is None:
                # First call, no usage data yet
                return

        # Truncate pairs until under threshold
        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further—conversation has single message pair "
                    f"(history too long for context limit {self._context_window_limit}). "
                    f"Consider increasing context_window_limit or decreasing max_tokens."
                )

            pair_fraction = 2.0 / len(self._history)
            self._history.pop(0)  # Remove oldest user message
            self._history.pop(0)  # Remove oldest assistant response

            if self._accurate_token_counting:
                estimated_tokens = await self._count_tokens_accurate()
            else:
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _estimate_tokens_heuristic(self) -> int | None:
        """Estimate token count using last API response usage stats."""
        if self._last_usage is None:
            return None

        # Use last response's input + output tokens as rough estimate
        return self._last_usage.input_tokens + self._last_usage.output_tokens

    async def _count_tokens_accurate(self) -> int:
        """Count tokens accurately using the client's count_tokens method."""
        response = await self._client.messages.count_tokens(
            model=self._model,
            messages=self._history,
            **({"system": self._system} if self._system else {}),
        )
        return response.input_tokens
