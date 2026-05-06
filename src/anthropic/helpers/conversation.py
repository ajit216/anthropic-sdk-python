"""ConversationManager helper for maintaining multi-turn conversation history.

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
    """Maintains multi-turn conversation history with context window management.
    
    This helper manages message history and automatically truncates oldest messages
    when approaching the model's context window limit.
    
    Thread safety: Each instance is single-threaded and not thread-safe.
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
            client: The Anthropic client instance
            model: The model to use (cannot be empty)
            max_tokens: Maximum tokens in response (must be >= 1)
            system: System prompt (optional)
            context_window_limit: Max tokens for context window (optional, must be >= 1 if provided)
            token_budget_headroom: Fraction of context to reserve as headroom (must be in [0.0, 1.0))
            accurate_token_counting: Use count_tokens API for accurate counts vs heuristic
            
        Raises:
            ValueError: If inputs are invalid
        """
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
        self._history: list[dict[str, Any]] = []
        self._last_usage: Any | None = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to the conversation history.
        
        Args:
            content: The message content (string or list of content blocks)
        """
        self._history.append({"role": "user", "content": content})

    def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model.
        
        Args:
            content: Optional message to add as user message before getting response
            **kwargs: Additional arguments to pass to messages.create
            
        Returns:
            The Message response from the API
            
        Raises:
            ValueError: If history doesn't end with a user message or if truncation fails
        """
        # Step 1: Add user message if content provided
        if content is not None:
            self.add_user_message(content)
        
        # Step 2: Validate history ends with user message
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("history must contain at least one user message before getting response")
        
        # Step 3: Truncate if needed
        if self._context_window_limit is not None:
            self._truncate_if_needed()
        
        # Step 4: Create message
        create_kwargs = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        if self._system is not None:
            create_kwargs["system"] = self._system
        create_kwargs.update(kwargs)
        
        response = self._client.messages.create(**create_kwargs)
        
        # Step 5: Append assistant response
        self._history.append({"role": "assistant", "content": response.content})
        
        # Step 6: Store usage
        self._last_usage = response.usage
        
        return response

    def reset(self) -> None:
        """Reset the conversation history and last usage."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[Any]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return the usage from the last response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return string representation of the manager."""
        turn_count = len([m for m in self._history if m["role"] == "user"])
        limit_str = f", limit={self._context_window_limit}" if self._context_window_limit else ""
        return f"ConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if approaching context window limit.
        
        Raises:
            ValueError: If single message pair exceeds the limit
        """
        if not self._context_window_limit:
            return
        
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)
        
        # Estimate tokens
        if self._accurate_token_counting:
            estimated_tokens = self._get_token_count()
        else:
            if self._last_usage is None:
                # Skip truncation on first call in heuristic mode
                return
            estimated_tokens = self._last_usage.input_tokens + self._last_usage.output_tokens
        
        # Truncate while over threshold
        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"cannot truncate further — single message pair exceeds limit "
                    f"(model={self._model}, limit={self._context_window_limit}). "
                    f"Consider increasing context_window_limit or reducing message size."
                )
            
            # Remove oldest user and assistant pair
            self._history.pop(0)
            self._history.pop(0)
            
            # Re-estimate tokens
            if self._accurate_token_counting:
                estimated_tokens = self._get_token_count()
            else:
                # Heuristic: estimate reduction based on pair fraction
                pair_fraction = 2.0 / (len(self._history) + 2)
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _get_token_count(self) -> int:
        """Get accurate token count using the API."""
        kwargs = {
            "messages": self._history,
            "model": self._model,
        }
        if self._system is not None:
            kwargs["system"] = self._system
        
        response = self._client.messages.count_tokens(**kwargs)
        return response.input_tokens


class AsyncConversationManager:
    """Async version of ConversationManager for maintaining multi-turn conversation history.
    
    This helper manages message history and automatically truncates oldest messages
    when approaching the model's context window limit.
    
    Thread safety: Each instance is single-threaded and not thread-safe.
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
            client: The async Anthropic client instance
            model: The model to use (cannot be empty)
            max_tokens: Maximum tokens in response (must be >= 1)
            system: System prompt (optional)
            context_window_limit: Max tokens for context window (optional, must be >= 1 if provided)
            token_budget_headroom: Fraction of context to reserve as headroom (must be in [0.0, 1.0))
            accurate_token_counting: Use count_tokens API for accurate counts vs heuristic
            
        Raises:
            ValueError: If inputs are invalid
        """
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
        self._history: list[dict[str, Any]] = []
        self._last_usage: Any | None = None

    def add_user_message(self, content: str | list[Any]) -> None:
        """Add a user message to the conversation history.
        
        Args:
            content: The message content (string or list of content blocks)
        """
        self._history.append({"role": "user", "content": content})

    async def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get a response from the model asynchronously.
        
        Args:
            content: Optional message to add as user message before getting response
            **kwargs: Additional arguments to pass to messages.create
            
        Returns:
            The Message response from the API
            
        Raises:
            ValueError: If history doesn't end with a user message or if truncation fails
        """
        # Step 1: Add user message if content provided
        if content is not None:
            self.add_user_message(content)
        
        # Step 2: Validate history ends with user message
        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError("history must contain at least one user message before getting response")
        
        # Step 3: Truncate if needed
        if self._context_window_limit is not None:
            await self._truncate_if_needed()
        
        # Step 4: Create message
        create_kwargs = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        if self._system is not None:
            create_kwargs["system"] = self._system
        create_kwargs.update(kwargs)
        
        response = await self._client.messages.create(**create_kwargs)
        
        # Step 5: Append assistant response
        self._history.append({"role": "assistant", "content": response.content})
        
        # Step 6: Store usage
        self._last_usage = response.usage
        
        return response

    def reset(self) -> None:
        """Reset the conversation history and last usage."""
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[Any]:
        """Return a shallow copy of the conversation history."""
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return the usage from the last response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return string representation of the manager."""
        turn_count = len([m for m in self._history if m["role"] == "user"])
        limit_str = f", limit={self._context_window_limit}" if self._context_window_limit else ""
        return f"AsyncConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    async def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if approaching context window limit.
        
        Raises:
            ValueError: If single message pair exceeds the limit
        """
        if not self._context_window_limit:
            return
        
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)
        
        # Estimate tokens
        if self._accurate_token_counting:
            estimated_tokens = await self._get_token_count()
        else:
            if self._last_usage is None:
                # Skip truncation on first call in heuristic mode
                return
            estimated_tokens = self._last_usage.input_tokens + self._last_usage.output_tokens
        
        # Truncate while over threshold
        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"cannot truncate further — single message pair exceeds limit "
                    f"(model={self._model}, limit={self._context_window_limit}). "
                    f"Consider increasing context_window_limit or reducing message size."
                )
            
            # Remove oldest user and assistant pair
            self._history.pop(0)
            self._history.pop(0)
            
            # Re-estimate tokens
            if self._accurate_token_counting:
                estimated_tokens = await self._get_token_count()
            else:
                # Heuristic: estimate reduction based on pair fraction
                pair_fraction = 2.0 / (len(self._history) + 2)
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    async def _get_token_count(self) -> int:
        """Get accurate token count using the API."""
        kwargs = {
            "messages": self._history,
            "model": self._model,
        }
        if self._system is not None:
            kwargs["system"] = self._system
        
        response = await self._client.messages.count_tokens(**kwargs)
        return response.input_tokens
