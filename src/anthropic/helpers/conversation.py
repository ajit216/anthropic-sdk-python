"""
Conversation history management helpers for multi-turn conversations.

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

    manager.add_user_message("What is 2 + 2?")
    response = manager.get_response()
    print(response.content[0].text)
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional


class ConversationManager:
    """Manages conversation history with automatic context window truncation.
    
    This helper maintains a list of messages across multiple turns and automatically
    truncates the oldest messages when approaching the context window limit.
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
            client: An Anthropic client instance.
            model: The model to use for API calls (e.g., "claude-3-5-sonnet-20241022").
            max_tokens: Maximum tokens to request in each response.
            system: Optional system prompt to include in all API calls.
            context_window_limit: Maximum context window size. If None, no truncation.
            token_budget_headroom: Fraction of context_window_limit to reserve (0.0-1.0).
            accurate_token_counting: If True, use count_tokens API for accurate counts.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not model:
            raise ValueError("model cannot be an empty string")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if context_window_limit is not None and context_window_limit < 1:
            raise ValueError(
                f"context_window_limit must be >= 1 or None, got {context_window_limit}"
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
            content: The message content (string or list of content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str):
            if not content:
                raise ValueError("User message content cannot be empty")
            message = {"role": "user", "content": content}
        else:
            if not content:
                raise ValueError("User message content cannot be empty")
            message = {"role": "user", "content": content}

        self._history.append(message)

    def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model.

        If content is provided, it's added as a user message first. Then the method
        truncates history if needed, calls the API, and appends the assistant response.

        Args:
            content: Optional user message content to add before getting response.
            **kwargs: Additional arguments to pass to messages.create().

        Returns:
            The API Message response object.

        Raises:
            ValueError: If history is empty or doesn't end with a user message.
        """
        if content is not None:
            self.add_user_message(content)

        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError(
                "No staged user message. Either provide content to get_response() "
                "or call add_user_message() first."
            )

        if self._context_window_limit is not None:
            self._truncate_if_needed()

        api_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": list(self._history),
        }
        if self._system:
            api_kwargs["system"] = self._system
        api_kwargs.update(kwargs)

        response = self._client.messages.create(**api_kwargs)

        self._history.append({"role": "assistant", "content": response.content})
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Clear conversation history and last usage."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return usage info from the last API response, or None."""
        return self._last_usage

    def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if approaching context window limit."""
        if self._context_window_limit is None:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        if self._accurate_token_counting:
            estimated_tokens = self._count_tokens_accurate()
        else:
            estimated_tokens = self._count_tokens_heuristic()
            if estimated_tokens is None:
                return

        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further — conversation has only {len(self._history)} "
                    f"message(s) but estimated tokens ({estimated_tokens}) exceed "
                    f"threshold ({threshold}). Try increasing context_window_limit "
                    f"or decreasing max_tokens for model {self._model}."
                )

            pair_fraction = 2.0 / len(self._history)
            self._history.pop(0)
            self._history.pop(0)

            if self._accurate_token_counting:
                estimated_tokens = self._count_tokens_accurate()
            else:
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _count_tokens_accurate(self) -> int:
        """Get accurate token count using the count_tokens API."""
        count_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": list(self._history),
        }
        if self._system:
            count_kwargs["system"] = self._system

        response = self._client.messages.count_tokens(**count_kwargs)
        return response.input_tokens

    def _count_tokens_heuristic(self) -> int | None:
        """Estimate token count using last_usage (heuristic mode)."""
        if self._last_usage is None:
            return None
        return self._last_usage.input_tokens + self._last_usage.output_tokens


class AsyncConversationManager:
    """Async version of ConversationManager for use with AsyncAnthropic client.
    
    This helper maintains a list of messages across multiple turns and automatically
    truncates the oldest messages when approaching the context window limit.
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
            client: An AsyncAnthropic client instance.
            model: The model to use for API calls (e.g., "claude-3-5-sonnet-20241022").
            max_tokens: Maximum tokens to request in each response.
            system: Optional system prompt to include in all API calls.
            context_window_limit: Maximum context window size. If None, no truncation.
            token_budget_headroom: Fraction of context_window_limit to reserve (0.0-1.0).
            accurate_token_counting: If True, use count_tokens API for accurate counts.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not model:
            raise ValueError("model cannot be an empty string")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if context_window_limit is not None and context_window_limit < 1:
            raise ValueError(
                f"context_window_limit must be >= 1 or None, got {context_window_limit}"
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
            content: The message content (string or list of content blocks).

        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str):
            if not content:
                raise ValueError("User message content cannot be empty")
            message = {"role": "user", "content": content}
        else:
            if not content:
                raise ValueError("User message content cannot be empty")
            message = {"role": "user", "content": content}

        self._history.append(message)

    async def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model asynchronously.

        If content is provided, it's added as a user message first. Then the method
        truncates history if needed, calls the API, and appends the assistant response.

        Args:
            content: Optional user message content to add before getting response.
            **kwargs: Additional arguments to pass to messages.create().

        Returns:
            The API Message response object.

        Raises:
            ValueError: If history is empty or doesn't end with a user message.
        """
        if content is not None:
            self.add_user_message(content)

        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError(
                "No staged user message. Either provide content to get_response() "
                "or call add_user_message() first."
            )

        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        api_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": list(self._history),
        }
        if self._system:
            api_kwargs["system"] = self._system
        api_kwargs.update(kwargs)

        response = await self._client.messages.create(**api_kwargs)

        self._history.append({"role": "assistant", "content": response.content})
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Clear conversation history and last usage."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return usage info from the last API response, or None."""
        return self._last_usage

    async def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if approaching context window limit."""
        if self._context_window_limit is None:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        if self._accurate_token_counting:
            estimated_tokens = await self._count_tokens_accurate()
        else:
            estimated_tokens = self._count_tokens_heuristic()
            if estimated_tokens is None:
                return

        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further — conversation has only {len(self._history)} "
                    f"message(s) but estimated tokens ({estimated_tokens}) exceed "
                    f"threshold ({threshold}). Try increasing context_window_limit "
                    f"or decreasing max_tokens for model {self._model}."
                )

            pair_fraction = 2.0 / len(self._history)
            self._history.pop(0)
            self._history.pop(0)

            if self._accurate_token_counting:
                estimated_tokens = await self._count_tokens_accurate()
            else:
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    async def _count_tokens_accurate(self) -> int:
        """Get accurate token count using the count_tokens API."""
        count_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": list(self._history),
        }
        if self._system:
            count_kwargs["system"] = self._system

        response = await self._client.messages.count_tokens(**count_kwargs)
        return response.input_tokens

    def _count_tokens_heuristic(self) -> int | None:
        """Estimate token count using last_usage (heuristic mode)."""
        if self._last_usage is None:
            return None
        return self._last_usage.input_tokens + self._last_usage.output_tokens
