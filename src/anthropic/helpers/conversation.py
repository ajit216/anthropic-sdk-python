"""ConversationManager helper for managing multi-turn conversations.

Example::

    import anthropic

    client = anthropic.Anthropic()
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant.",
        context_window_limit=200000,
    )

    manager.add_user_message("What is 2+2?")
    response = manager.get_response()
    print(response.content[0].text)
"""

from __future__ import annotations

from typing import Any, Optional


class ConversationManager:
    """Manages multi-turn conversation history with automatic context window management.
    
    Maintains conversation state across turns and automatically truncates the oldest
    messages when approaching the model's context window limit.
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
            client: An Anthropic client instance.
            model: The model to use for API calls (e.g., "claude-3-5-sonnet-20241022").
            max_tokens: Maximum tokens to request in each API response.
            system: Optional system prompt to include with all requests.
            context_window_limit: Optional context window limit to trigger truncation.
            token_budget_headroom: Fraction (0.0-1.0) of context_window_limit to reserve
                as safety margin. Default: 0.10 (10%).
            accurate_token_counting: If True, use client.messages.count_tokens() for
                accurate truncation decisions. If False (default), use heuristic based
                on last_usage.
                
        Raises:
            ValueError: If model is empty, max_tokens < 1, context_window_limit < 1,
                or token_budget_headroom not in [0.0, 1.0).
        """
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
            content: String message or list of content blocks.
            
        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str):
            if not content:
                raise ValueError("content must not be empty")
            content_to_add = content
        elif isinstance(content, list):
            if not content:
                raise ValueError("content must not be empty")
            content_to_add = content
        else:
            if not content:
                raise ValueError("content must not be empty")
            content_to_add = content

        self._history.append({"role": "user", "content": content_to_add})

    def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get an assistant response, optionally adding a user message first.
        
        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional keyword arguments to pass to messages.create().
            
        Returns:
            The Message response from the API.
            
        Raises:
            ValueError: If no user message is staged (last message is not from user).
        """
        # Add user message if content provided
        if content is not None:
            self.add_user_message(content)

        # Validate that we have a staged user message
        if not self._history or self._history[-1].get("role") != "user":
            raise ValueError(
                "No user message staged. Either call add_user_message() first or pass content to get_response()."
            )

        # Truncate if needed
        if self._context_window_limit is not None:
            self._truncate_if_needed()

        # Prepare API call arguments
        api_kwargs = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }

        # Add system prompt if set
        if self._system:
            api_kwargs["system"] = self._system

        # Add any user-provided kwargs
        api_kwargs.update(kwargs)

        # Call the API
        response = self._client.messages.create(**api_kwargs)

        # Append assistant response to history
        self._history.append({"role": "assistant", "content": response.content})

        # Store usage for heuristic truncation
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Clear conversation history and last_usage, but keep model and system prompt."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return the usage data from the last API response, or None if no response yet."""
        return self._last_usage

    def _truncate_if_needed(self) -> None:
        """Truncate history if estimated tokens exceed threshold."""
        if self._context_window_limit is None:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate current tokens
        if self._accurate_token_counting:
            estimated_tokens = self._count_tokens()
        else:
            # Heuristic: use last_usage
            if self._last_usage is None:
                # First call, no usage data yet - skip truncation
                return
            estimated_tokens = (
                self._last_usage.input_tokens + self._last_usage.output_tokens
            )

        # Truncate pairs until under threshold
        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further: history too short to continue conversation. "
                    f"Model {self._model!r} with context_window_limit={self._context_window_limit} "
                    f"and token_budget_headroom={self._token_budget_headroom} may be incompatible with "
                    f"your max_tokens setting. Consider increasing context_window_limit, decreasing "
                    f"token_budget_headroom, or reducing max_tokens."
                )

            # Remove oldest pair (user + assistant)
            pair_fraction = 2 / len(self._history)
            self._history.pop(0)  # Remove oldest user message
            self._history.pop(0)  # Remove oldest assistant message

            # Re-estimate tokens
            if self._accurate_token_counting:
                estimated_tokens = self._count_tokens()
            else:
                # Heuristic: scale down by the fraction removed
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _count_tokens(self) -> int:
        """Count tokens in current history using client.messages.count_tokens()."""
        count_kwargs = {
            "messages": list(self._history),
            "model": self._model,
        }
        if self._system:
            count_kwargs["system"] = self._system

        return self._client.messages.count_tokens(**count_kwargs).input_tokens

    def __repr__(self) -> str:
        """Return a string representation of the ConversationManager."""
        turn_count = len(self._history) // 2
        limit_str = (
            f", limit={self._context_window_limit}" if self._context_window_limit else ""
        )
        return f"ConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"


class AsyncConversationManager:
    """Async version of ConversationManager for managing multi-turn conversations.
    
    Maintains conversation state across turns and automatically truncates the oldest
    messages when approaching the model's context window limit.
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
            client: An async Anthropic client instance.
            model: The model to use for API calls (e.g., "claude-3-5-sonnet-20241022").
            max_tokens: Maximum tokens to request in each API response.
            system: Optional system prompt to include with all requests.
            context_window_limit: Optional context window limit to trigger truncation.
            token_budget_headroom: Fraction (0.0-1.0) of context_window_limit to reserve
                as safety margin. Default: 0.10 (10%).
            accurate_token_counting: If True, use client.messages.count_tokens() for
                accurate truncation decisions. If False (default), use heuristic based
                on last_usage.
                
        Raises:
            ValueError: If model is empty, max_tokens < 1, context_window_limit < 1,
                or token_budget_headroom not in [0.0, 1.0).
        """
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
            content: String message or list of content blocks.
            
        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str):
            if not content:
                raise ValueError("content must not be empty")
            content_to_add = content
        elif isinstance(content, list):
            if not content:
                raise ValueError("content must not be empty")
            content_to_add = content
        else:
            if not content:
                raise ValueError("content must not be empty")
            content_to_add = content

        self._history.append({"role": "user", "content": content_to_add})

    async def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get an assistant response, optionally adding a user message first.
        
        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional keyword arguments to pass to messages.create().
            
        Returns:
            The Message response from the API.
            
        Raises:
            ValueError: If no user message is staged (last message is not from user).
        """
        # Add user message if content provided
        if content is not None:
            self.add_user_message(content)

        # Validate that we have a staged user message
        if not self._history or self._history[-1].get("role") != "user":
            raise ValueError(
                "No user message staged. Either call add_user_message() first or pass content to get_response()."
            )

        # Truncate if needed
        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        # Prepare API call arguments
        api_kwargs = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }

        # Add system prompt if set
        if self._system:
            api_kwargs["system"] = self._system

        # Add any user-provided kwargs
        api_kwargs.update(kwargs)

        # Call the API
        response = await self._client.messages.create(**api_kwargs)

        # Append assistant response to history
        self._history.append({"role": "assistant", "content": response.content})

        # Store usage for heuristic truncation
        self._last_usage = response.usage

        return response

    def reset(self) -> None:
        """Clear conversation history and last_usage, but keep model and system prompt."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return the usage data from the last API response, or None if no response yet."""
        return self._last_usage

    async def _truncate_if_needed(self) -> None:
        """Truncate history if estimated tokens exceed threshold."""
        if self._context_window_limit is None:
            return

        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        # Estimate current tokens
        if self._accurate_token_counting:
            estimated_tokens = await self._count_tokens()
        else:
            # Heuristic: use last_usage
            if self._last_usage is None:
                # First call, no usage data yet - skip truncation
                return
            estimated_tokens = (
                self._last_usage.input_tokens + self._last_usage.output_tokens
            )

        # Truncate pairs until under threshold
        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further: history too short to continue conversation. "
                    f"Model {self._model!r} with context_window_limit={self._context_window_limit} "
                    f"and token_budget_headroom={self._token_budget_headroom} may be incompatible with "
                    f"your max_tokens setting. Consider increasing context_window_limit, decreasing "
                    f"token_budget_headroom, or reducing max_tokens."
                )

            # Remove oldest pair (user + assistant)
            pair_fraction = 2 / len(self._history)
            self._history.pop(0)  # Remove oldest user message
            self._history.pop(0)  # Remove oldest assistant message

            # Re-estimate tokens
            if self._accurate_token_counting:
                estimated_tokens = await self._count_tokens()
            else:
                # Heuristic: scale down by the fraction removed
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    async def _count_tokens(self) -> int:
        """Count tokens in current history using client.messages.count_tokens()."""
        count_kwargs = {
            "messages": list(self._history),
            "model": self._model,
        }
        if self._system:
            count_kwargs["system"] = self._system

        return (await self._client.messages.count_tokens(**count_kwargs)).input_tokens

    def __repr__(self) -> str:
        """Return a string representation of the AsyncConversationManager."""
        turn_count = len(self._history) // 2
        limit_str = (
            f", limit={self._context_window_limit}" if self._context_window_limit else ""
        )
        return f"AsyncConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"
