"""ConversationManager helper for maintaining multi-turn conversations with auto-truncation.

Example::

    from anthropic import Anthropic
    from anthropic.helpers import ConversationManager

    client = Anthropic()
    conversation = ConversationManager(
        client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        context_window_limit=200000,
    )

    conversation.add_user_message("Hello, how are you?")
    response = conversation.get_response()
    print(response.content[0].text)
"""

from __future__ import annotations

from typing import Any


class ConversationManager:
    """Manages multi-turn conversations with automatic history truncation.
    
    Maintains conversation state across turns and automatically truncates
    the oldest messages when approaching the model's context window limit.
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
            model: The model ID to use (e.g., "claude-3-5-sonnet-20241022").
            max_tokens: Maximum tokens to request in API calls.
            system: Optional system prompt.
            context_window_limit: Optional context window size limit for auto-truncation.
            token_budget_headroom: Fraction of context_window_limit to keep as headroom (0.0-1.0).
            accurate_token_counting: Whether to use count_tokens API for accurate estimates.
            
        Raises:
            ValueError: If any parameter validation fails.
        """
        # Validate parameters
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
            content: The user message content.
            
        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str):
            if not content:
                raise ValueError("content must not be empty")
        elif isinstance(content, list):
            if not content:
                raise ValueError("content must not be empty")
        
        self._history.append({
            "role": "user",
            "content": content,
        })

    def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model.
        
        If content is provided, it's added as a user message first.
        
        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional arguments to pass to messages.create().
            
        Returns:
            The API response.
            
        Raises:
            ValueError: If no user message is staged or history validation fails.
        """
        # Add content if provided
        if content is not None:
            self.add_user_message(content)
        
        # Validate that we have a user message staged
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No user message staged. Call add_user_message() first.")
        
        # Truncate if needed
        if self._context_window_limit is not None:
            self._truncate_if_needed()
        
        # Build kwargs for API call
        api_kwargs: dict[str, Any] = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        
        # Add system prompt if set
        if self._system is not None:
            api_kwargs["system"] = self._system
        
        # Add any additional kwargs
        api_kwargs.update(kwargs)
        
        # Get response
        response = self._client.messages.create(**api_kwargs)
        
        # Add assistant response to history
        self._history.append({
            "role": "assistant",
            "content": response.content,
        })
        
        # Update last usage
        self._last_usage = response.usage
        
        return response

    def _truncate_if_needed(self) -> None:
        """Truncate the oldest messages if we're approaching the context limit."""
        if self._context_window_limit is None:
            return
        
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)
        
        # Estimate tokens
        estimated_tokens = self._estimate_tokens()
        
        # Skip truncation on first call if using heuristic mode
        if estimated_tokens is None:
            return
        
        # Truncate oldest pairs until under threshold
        while estimated_tokens >= threshold:
            # Need at least 2 messages (one user, one assistant pair)
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further. Single message pair exceeds context limit. "
                    f"Model: {self._model}, Context limit: {self._context_window_limit}, "
                    f"Consider increasing context_window_limit or using a model with larger context."
                )
            
            # Remove oldest user message
            self._history.pop(0)
            # Remove oldest assistant message
            if self._history:
                self._history.pop(0)
            
            # Re-estimate tokens
            estimated_tokens = self._estimate_tokens()

    def _estimate_tokens(self) -> int | None:
        """Estimate the token count for the current history.
        
        Returns:
            Token count estimate, or None if unable to estimate (first call in heuristic mode).
        """
        if self._accurate_token_counting:
            # Use count_tokens API for accurate count
            response = self._client.messages.count_tokens(
                model=self._model,
                messages=self._history,
                system=self._system,
            )
            return response.input_tokens
        else:
            # Use heuristic based on last API response
            if self._last_usage is None:
                return None
            
            # Estimate as previous input + output tokens
            return int(self._last_usage.input_tokens + self._last_usage.output_tokens)

    def reset(self) -> None:
        """Reset the conversation history and usage stats.
        
        Model and system prompt are preserved.
        """
        self._history.clear()
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Get a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Get the usage stats from the last API response."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return a string representation of the ConversationManager."""
        turn_count = len([m for m in self._history if m["role"] == "user"])
        limit_str = f", limit={self._context_window_limit}" if self._context_window_limit else ""
        return f"ConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"


class AsyncConversationManager:
    """Async version of ConversationManager for async client usage.
    
    Manages multi-turn conversations with automatic history truncation.
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
            model: The model ID to use (e.g., "claude-3-5-sonnet-20241022").
            max_tokens: Maximum tokens to request in API calls.
            system: Optional system prompt.
            context_window_limit: Optional context window size limit for auto-truncation.
            token_budget_headroom: Fraction of context_window_limit to keep as headroom (0.0-1.0).
            accurate_token_counting: Whether to use count_tokens API for accurate estimates.
            
        Raises:
            ValueError: If any parameter validation fails.
        """
        # Validate parameters
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
            content: The user message content.
            
        Raises:
            ValueError: If content is empty.
        """
        if isinstance(content, str):
            if not content:
                raise ValueError("content must not be empty")
        elif isinstance(content, list):
            if not content:
                raise ValueError("content must not be empty")
        
        self._history.append({
            "role": "user",
            "content": content,
        })

    async def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model asynchronously.
        
        If content is provided, it's added as a user message first.
        
        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional arguments to pass to messages.create().
            
        Returns:
            The API response.
            
        Raises:
            ValueError: If no user message is staged or history validation fails.
        """
        # Add content if provided
        if content is not None:
            self.add_user_message(content)
        
        # Validate that we have a user message staged
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("No user message staged. Call add_user_message() first.")
        
        # Truncate if needed
        if self._context_window_limit is not None:
            await self._truncate_if_needed()
        
        # Build kwargs for API call
        api_kwargs: dict[str, Any] = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        
        # Add system prompt if set
        if self._system is not None:
            api_kwargs["system"] = self._system
        
        # Add any additional kwargs
        api_kwargs.update(kwargs)
        
        # Get response
        response = await self._client.messages.create(**api_kwargs)
        
        # Add assistant response to history
        self._history.append({
            "role": "assistant",
            "content": response.content,
        })
        
        # Update last usage
        self._last_usage = response.usage
        
        return response

    async def _truncate_if_needed(self) -> None:
        """Truncate the oldest messages if we're approaching the context limit."""
        if self._context_window_limit is None:
            return
        
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)
        
        # Estimate tokens
        estimated_tokens = await self._estimate_tokens()
        
        # Skip truncation on first call if using heuristic mode
        if estimated_tokens is None:
            return
        
        # Truncate oldest pairs until under threshold
        while estimated_tokens >= threshold:
            # Need at least 2 messages (one user, one assistant pair)
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further. Single message pair exceeds context limit. "
                    f"Model: {self._model}, Context limit: {self._context_window_limit}, "
                    f"Consider increasing context_window_limit or using a model with larger context."
                )
            
            # Remove oldest user message
            self._history.pop(0)
            # Remove oldest assistant message
            if self._history:
                self._history.pop(0)
            
            # Re-estimate tokens
            estimated_tokens = await self._estimate_tokens()

    async def _estimate_tokens(self) -> int | None:
        """Estimate the token count for the current history.
        
        Returns:
            Token count estimate, or None if unable to estimate (first call in heuristic mode).
        """
        if self._accurate_token_counting:
            # Use count_tokens API for accurate count
            response = await self._client.messages.count_tokens(
                model=self._model,
                messages=self._history,
                system=self._system,
            )
            return response.input_tokens
        else:
            # Use heuristic based on last API response
            if self._last_usage is None:
                return None
            
            # Estimate as previous input + output tokens
            return int(self._last_usage.input_tokens + self._last_usage.output_tokens)

    def reset(self) -> None:
        """Reset the conversation history and usage stats.
        
        Model and system prompt are preserved.
        """
        self._history.clear()
        self._last_usage = None

    @property
    def history(self) -> list[dict[str, Any]]:
        """Get a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Get the usage stats from the last API response."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return a string representation of the AsyncConversationManager."""
        turn_count = len([m for m in self._history if m["role"] == "user"])
        limit_str = f", limit={self._context_window_limit}" if self._context_window_limit else ""
        return f"AsyncConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"
