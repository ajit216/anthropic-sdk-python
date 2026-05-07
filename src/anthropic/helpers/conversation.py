"""ConversationManager helper for managing multi-turn conversations with automatic context window truncation.

Example::

    from anthropic import Anthropic
    from anthropic.helpers import ConversationManager

    client = Anthropic()
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        context_window_limit=200000,
        token_budget_headroom=0.1,
    )

    manager.add_user_message("What is the capital of France?")
    response = manager.get_response()
    print(response.content[0].text)

    # Multi-turn conversation
    response = manager.get_response("And what is its population?")
    print(response.content[0].text)
"""

from __future__ import annotations

from typing import Any, Optional


class ConversationManager:
    """Manages multi-turn conversation history with automatic context window truncation.

    This helper maintains conversation state and automatically truncates the oldest messages
    when approaching the model's context window limit. Thread safety is not guaranteed;
    each instance should be used from a single thread.

    Args:
        client: An Anthropic client instance.
        model: The model ID to use for API calls (required, cannot be empty).
        max_tokens: Maximum tokens in the API response (required, must be >= 1).
        system: Optional system prompt to send with every message.
        context_window_limit: Optional context window size of the model. If not set,
                             truncation is disabled.
        token_budget_headroom: Fraction of context window to keep as headroom (default 0.10).
                              Must be in [0.0, 1.0).
        accurate_token_counting: If True, use client.messages.count_tokens() for precise
                                token estimation. If False (default), use last_usage for
                                heuristic estimation (zero API calls, but less accurate).
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
        # Validate inputs
        if not model:
            raise ValueError("model cannot be empty")
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
        """Add a user message to the conversation history.

        Args:
            content: The user message content (string or list of content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str):
            if not content:
                raise ValueError("content cannot be empty")
            self._history.append({"role": "user", "content": content})
        elif isinstance(content, list):
            if not content:
                raise ValueError("content cannot be empty")
            self._history.append({"role": "user", "content": content})
        else:
            raise ValueError("content must be a string or list")

    def get_response(
        self, content: str | list[Any] | None = None, **kwargs: Any
    ) -> Any:
        """Get a response from the model, managing conversation history.

        If content is provided, it will be added as a user message first.
        Then truncation is applied if necessary, and the model is called.
        The assistant response is appended to history.

        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional arguments to pass to client.messages.create().

        Returns:
            The Message response from the API.

        Raises:
            ValueError: If no user message is staged or if history cannot be truncated
                       enough to fit the model's context window.
        """
        # Step 1: Add user message if content provided
        if content is not None:
            self.add_user_message(content)

        # Step 2: Validate that last message is from user
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No staged user message. Call add_user_message() first.")

        # Step 3: Truncate if needed
        if self._context_window_limit is not None:
            self._truncate_if_needed()

        # Step 4: Call the API
        call_kwargs = {
            "messages": list(self._history),
            "model": self._model,
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

        # Step 7: Return response
        return response

    def reset(self) -> None:
        """Clear conversation history and reset usage tracking.

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
        """Return the usage from the last API response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return a string representation of the manager."""
        turn_count = len(
            [m for m in self._history if m["role"] == "assistant"]
        )
        return f"ConversationManager(model={self._model!r}, turns={turn_count}, limit={self._context_window_limit})"

    def _truncate_if_needed(self) -> None:
        """Truncate history if it exceeds the context window threshold."""
        if not self._history:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate current token count
        if self._accurate_token_counting:
            estimated_tokens = self._client.messages.count_tokens(
                messages=self._history,
                model=self._model,
                system=self._system,
            )
        else:
            # Heuristic: use last_usage if available
            if self._last_usage is None:
                # Skip truncation on first call (no usage data yet)
                return
            estimated_tokens = (
                self._last_usage.input_tokens + self._last_usage.output_tokens
            )

        # Truncate oldest pairs until under threshold
        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further: single message pair ({self._model}) "
                    f"exceeds context window limit ({self._context_window_limit} tokens). "
                    f"Consider increasing context_window_limit or token_budget_headroom."
                )

            # Drop oldest user + assistant pair
            self._history.pop(0)  # oldest user
            self._history.pop(0)  # oldest assistant

            # Re-estimate tokens
            if self._accurate_token_counting:
                estimated_tokens = self._client.messages.count_tokens(
                    messages=self._history,
                    model=self._model,
                    system=self._system,
                )
            else:
                # Heuristic: estimate based on pair removal
                pair_fraction = 2.0 / (len(self._history) + 2)  # +2 for removed pair
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))


class AsyncConversationManager:
    """Async version of ConversationManager for use with async clients.

    All synchronous methods remain sync. Only get_response() is async.
    See ConversationManager for detailed documentation.
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
        # Validate inputs
        if not model:
            raise ValueError("model cannot be empty")
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
        """Add a user message to the conversation history."""
        if isinstance(content, str):
            if not content:
                raise ValueError("content cannot be empty")
            self._history.append({"role": "user", "content": content})
        elif isinstance(content, list):
            if not content:
                raise ValueError("content cannot be empty")
            self._history.append({"role": "user", "content": content})
        else:
            raise ValueError("content must be a string or list")

    async def get_response(
        self, content: str | list[Any] | None = None, **kwargs: Any
    ) -> Any:
        """Async version of get_response. See ConversationManager for details."""
        # Step 1: Add user message if content provided
        if content is not None:
            self.add_user_message(content)

        # Step 2: Validate that last message is from user
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No staged user message. Call add_user_message() first.")

        # Step 3: Truncate if needed
        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        # Step 4: Call the API
        call_kwargs = {
            "messages": list(self._history),
            "model": self._model,
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

        # Step 7: Return response
        return response

    def reset(self) -> None:
        """Clear conversation history and reset usage tracking."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[Any]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return the usage from the last API response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return a string representation of the manager."""
        turn_count = len(
            [m for m in self._history if m["role"] == "assistant"]
        )
        return f"AsyncConversationManager(model={self._model!r}, turns={turn_count}, limit={self._context_window_limit})"

    async def _truncate_if_needed(self) -> None:
        """Truncate history if it exceeds the context window threshold."""
        if not self._history:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate current token count
        if self._accurate_token_counting:
            estimated_tokens = await self._client.messages.count_tokens(
                messages=self._history,
                model=self._model,
                system=self._system,
            )
        else:
            # Heuristic: use last_usage if available
            if self._last_usage is None:
                # Skip truncation on first call (no usage data yet)
                return
            estimated_tokens = (
                self._last_usage.input_tokens + self._last_usage.output_tokens
            )

        # Truncate oldest pairs until under threshold
        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further: single message pair ({self._model}) "
                    f"exceeds context window limit ({self._context_window_limit} tokens). "
                    f"Consider increasing context_window_limit or token_budget_headroom."
                )

            # Drop oldest user + assistant pair
            self._history.pop(0)  # oldest user
            self._history.pop(0)  # oldest assistant

            # Re-estimate tokens
            if self._accurate_token_counting:
                estimated_tokens = await self._client.messages.count_tokens(
                    messages=self._history,
                    model=self._model,
                    system=self._system,
                )
            else:
                # Heuristic: estimate based on pair removal
                pair_fraction = 2.0 / (len(self._history) + 2)  # +2 for removed pair
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))
