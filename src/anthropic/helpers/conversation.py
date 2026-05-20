"""ConversationManager helper for managing multi-turn conversation history.

Example::

    from anthropic import Anthropic
    from anthropic.helpers import ConversationManager

    client = Anthropic()
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        context_window_limit=200_000,
    )

    # Single turn
    manager.add_user_message("Hello, what's your name?")
    response = manager.get_response()
    print(response.content[0].text)

    # Multi-turn
    manager.add_user_message("What's 2 + 2?")
    response = manager.get_response()
    print(response.content[0].text)

    # Or in a loop
    manager.add_user_message("Reset my knowledge.")
    manager.get_response()
    manager.reset()
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Message, MessageParam, Usage


class ConversationManager:
    """Manages multi-turn conversation history with automatic context window management.

    This helper maintains conversation state across multiple turns and automatically
    truncates the oldest messages when approaching the model's context window limit.

    Args:
        client: The Anthropic client instance.
        model: The model ID to use for API calls (e.g., "claude-3-5-sonnet-20241022").
        max_tokens: Maximum tokens to generate per response.
        system: Optional system prompt to prepend to all requests.
        context_window_limit: Maximum context window size. If None, no truncation occurs.
        token_budget_headroom: Fraction of context window to reserve (0.0 to 1.0). Default is 0.10 (10%).
        accurate_token_counting: If True, use count_tokens API for precise measurements.
            If False (default), use heuristic based on usage stats.

    Raises:
        ValueError: If model is empty, max_tokens < 1, context_window_limit < 1,
            or token_budget_headroom is not in [0.0, 1.0).
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
        if not model or (isinstance(model, str) and model.strip() == ""):
            raise ValueError("model cannot be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if context_window_limit is not None and context_window_limit < 1:
            raise ValueError("context_window_limit must be at least 1 or None")
        if not (0.0 <= token_budget_headroom < 1.0):
            raise ValueError("token_budget_headroom must be in [0.0, 1.0)")

        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._context_window_limit = context_window_limit
        self._token_budget_headroom = token_budget_headroom
        self._accurate_token_counting = accurate_token_counting
        self._history: list[MessageParam] = []
        self._last_usage: Usage | None = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The message content (string or list of content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str) and not content.strip():
            raise ValueError("content cannot be empty")
        if isinstance(content, list) and len(content) == 0:
            raise ValueError("content cannot be empty")

        self._history.append({"role": "user", "content": content})

    def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Message:
        """Get a response from the model and append it to history.

        Args:
            content: Optional user message content. If provided, it will be added to history
                before requesting a response.
            **kwargs: Additional arguments to pass to messages.create() (e.g., temperature, top_p).

        Returns:
            The Message response from the API.

        Raises:
            ValueError: If no user message is staged or if truncation fails.
        """
        # Add message if content provided
        if content is not None:
            self.add_user_message(content)

        # Validate that we have a staged user message
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No staged user message. Call add_user_message() first.")

        # Truncate if needed
        if self._context_window_limit is not None:
            self._truncate_if_needed()

        # Build API call arguments
        api_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": list(self._history),
            "max_tokens": self._max_tokens,
        }

        if self._system is not None:
            api_kwargs["system"] = self._system

        # Merge user-provided kwargs
        api_kwargs.update(kwargs)

        # Call API
        response = self._client.messages.create(**api_kwargs)

        # Append assistant response to history
        self._history.append({"role": "assistant", "content": response.content})

        # Store usage for truncation heuristic
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Clear conversation history and reset usage tracking.

        Model and system prompt are preserved.
        """
        self._history.clear()
        self._last_usage = None

    @property
    def history(self) -> list[MessageParam]:
        """Return a shallow copy of the conversation history.

        Returns:
            A list of message dicts with 'role' and 'content' keys.
        """
        return list(self._history)

    @property
    def last_usage(self) -> Usage | None:
        """Return usage statistics from the last API call, or None if no calls made yet."""
        return self._last_usage

    def _truncate_if_needed(self) -> None:
        """Truncate conversation history if it exceeds the context window threshold.

        Raises:
            ValueError: If a single message pair exceeds the limit and cannot be truncated further.
        """
        if self._context_window_limit is None:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate current token usage
        if self._accurate_token_counting:
            estimated_tokens = self._count_tokens()
        else:
            # Heuristic: use last_usage; skip on first call
            if self._last_usage is None:
                return
            estimated_tokens = self._last_usage.input_tokens + self._last_usage.output_tokens

        # Truncate oldest pairs until under threshold
        while estimated_tokens >= threshold:
            # Prevent truncation below a single user/assistant pair
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate conversation further. A single message pair "
                    f"requires {estimated_tokens} tokens but the context window limit is "
                    f"{self._context_window_limit} tokens (headroom: {self._token_budget_headroom * 100:.0f}%). "
                    f"Consider increasing context_window_limit or reducing token_budget_headroom."
                )

            # Remove oldest user and assistant messages
            self._history.pop(0)  # user
            self._history.pop(0)  # assistant

            # Re-estimate
            if self._accurate_token_counting:
                estimated_tokens = self._count_tokens()
            else:
                # Heuristic: scale down proportionally
                pair_fraction = 2.0 / (len(self._history) + 2)
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _count_tokens(self) -> int:
        """Count tokens in the current history using the API.

        Returns:
            Total estimated token count.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": list(self._history),
        }

        if self._system is not None:
            kwargs["system"] = self._system

        token_response = self._client.messages.count_tokens(**kwargs)
        return token_response.input_tokens

    def __repr__(self) -> str:
        turn_count = len(self._history) // 2
        return (
            f"ConversationManager(model={self._model!r}, turns={turn_count}, "
            f"context_limit={self._context_window_limit})"
        )


class AsyncConversationManager:
    """Async version of ConversationManager for use with AsyncAnthropic client.

    See ConversationManager for full documentation.
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
        if not model or (isinstance(model, str) and model.strip() == ""):
            raise ValueError("model cannot be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if context_window_limit is not None and context_window_limit < 1:
            raise ValueError("context_window_limit must be at least 1 or None")
        if not (0.0 <= token_budget_headroom < 1.0):
            raise ValueError("token_budget_headroom must be in [0.0, 1.0)")

        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system = system
        self._context_window_limit = context_window_limit
        self._token_budget_headroom = token_budget_headroom
        self._accurate_token_counting = accurate_token_counting
        self._history: list[MessageParam] = []
        self._last_usage: Usage | None = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to the conversation history.

        Args:
            content: The message content (string or list of content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str) and not content.strip():
            raise ValueError("content cannot be empty")
        if isinstance(content, list) and len(content) == 0:
            raise ValueError("content cannot be empty")

        self._history.append({"role": "user", "content": content})

    async def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Message:
        """Get a response from the model and append it to history.

        Args:
            content: Optional user message content. If provided, it will be added to history
                before requesting a response.
            **kwargs: Additional arguments to pass to messages.create() (e.g., temperature, top_p).

        Returns:
            The Message response from the API.

        Raises:
            ValueError: If no user message is staged or if truncation fails.
        """
        # Add message if content provided
        if content is not None:
            self.add_user_message(content)

        # Validate that we have a staged user message
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No staged user message. Call add_user_message() first.")

        # Truncate if needed
        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        # Build API call arguments
        api_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": list(self._history),
            "max_tokens": self._max_tokens,
        }

        if self._system is not None:
            api_kwargs["system"] = self._system

        # Merge user-provided kwargs
        api_kwargs.update(kwargs)

        # Call API
        response = await self._client.messages.create(**api_kwargs)

        # Append assistant response to history
        self._history.append({"role": "assistant", "content": response.content})

        # Store usage for truncation heuristic
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Clear conversation history and reset usage tracking.

        Model and system prompt are preserved.
        """
        self._history.clear()
        self._last_usage = None

    @property
    def history(self) -> list[MessageParam]:
        """Return a shallow copy of the conversation history.

        Returns:
            A list of message dicts with 'role' and 'content' keys.
        """
        return list(self._history)

    @property
    def last_usage(self) -> Usage | None:
        """Return usage statistics from the last API call, or None if no calls made yet."""
        return self._last_usage

    async def _truncate_if_needed(self) -> None:
        """Truncate conversation history if it exceeds the context window threshold.

        Raises:
            ValueError: If a single message pair exceeds the limit and cannot be truncated further.
        """
        if self._context_window_limit is None:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate current token usage
        if self._accurate_token_counting:
            estimated_tokens = await self._count_tokens()
        else:
            # Heuristic: use last_usage; skip on first call
            if self._last_usage is None:
                return
            estimated_tokens = self._last_usage.input_tokens + self._last_usage.output_tokens

        # Truncate oldest pairs until under threshold
        while estimated_tokens >= threshold:
            # Prevent truncation below a single user/assistant pair
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate conversation further. A single message pair "
                    f"requires {estimated_tokens} tokens but the context window limit is "
                    f"{self._context_window_limit} tokens (headroom: {self._token_budget_headroom * 100:.0f}%). "
                    f"Consider increasing context_window_limit or reducing token_budget_headroom."
                )

            # Remove oldest user and assistant messages
            self._history.pop(0)  # user
            self._history.pop(0)  # assistant

            # Re-estimate
            if self._accurate_token_counting:
                estimated_tokens = await self._count_tokens()
            else:
                # Heuristic: scale down proportionally
                pair_fraction = 2.0 / (len(self._history) + 2)
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    async def _count_tokens(self) -> int:
        """Count tokens in the current history using the API.

        Returns:
            Total estimated token count.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": list(self._history),
        }

        if self._system is not None:
            kwargs["system"] = self._system

        token_response = await self._client.messages.count_tokens(**kwargs)
        return token_response.input_tokens

    def __repr__(self) -> str:
        turn_count = len(self._history) // 2
        return (
            f"AsyncConversationManager(model={self._model!r}, turns={turn_count}, "
            f"context_limit={self._context_window_limit})"
        )
