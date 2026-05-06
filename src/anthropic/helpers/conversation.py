"""ConversationManager helper for managing multi-turn conversations with auto-truncation.

Example::

    from anthropic import Anthropic
    from anthropic.helpers import ConversationManager
    
    client = Anthropic()
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=2048,
        system="You are a helpful assistant.",
        context_window_limit=200000,
    )
    
    manager.add_user_message("Hello!")
    response = manager.get_response()
    print(response.content[0].text)
    
    response = manager.get_response("How are you?")
    print(manager.history)
"""

from __future__ import annotations

from typing import Any


class ConversationManager:
    """Manages multi-turn conversation history with auto-truncation.
    
    This helper maintains message history across multiple turns and automatically
    truncates the oldest message pairs when approaching the model's context window limit.
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
            max_tokens: Max output tokens for each API call.
            system: Optional system prompt.
            context_window_limit: Optional context window size. If set, enables auto-truncation.
            token_budget_headroom: Fraction [0.0, 1.0) of context to reserve. Default 0.10 (10%).
            accurate_token_counting: If True, call count_tokens() for precise estimates.
                                    If False, use heuristic from last response (faster).

        Raises ValueError:
            - If model is empty string
            - If max_tokens < 1
            - If context_window_limit provided but < 1
            - If token_budget_headroom not in [0.0, 1.0)
        """
        if not model:
            raise ValueError("model cannot be empty")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if context_window_limit is not None and context_window_limit < 1:
            raise ValueError(f"context_window_limit must be >= 1, got {context_window_limit}")
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
        """Add a user message to conversation history.

        Args:
            content: Text string or list of content blocks.

        Raises ValueError:
            If content is empty.
        """
        if isinstance(content, str):
            if not content:
                raise ValueError("content cannot be empty string")
        elif isinstance(content, list):
            if not content:
                raise ValueError("content list cannot be empty")
        
        self._history.append({"role": "user", "content": content})

    def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get assistant response for the conversation.

        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional arguments passed to client.messages.create()
                     (e.g., temperature, tools, etc.)

        Returns:
            Message object from API.

        Raises ValueError:
            If no user message is staged (history empty or last role != "user").
        """
        if content is not None:
            self.add_user_message(content)

        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError(
                "No user message staged. Either pass content to get_response() "
                "or call add_user_message() first."
            )

        if self._context_window_limit is not None:
            self._truncate_if_needed()

        api_kwargs = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        if self._system is not None:
            api_kwargs["system"] = self._system
        api_kwargs.update(kwargs)

        response = self._client.messages.create(**api_kwargs)
        
        self._history.append({"role": "assistant", "content": response.content})
        self._last_usage = response.usage
        
        return response

    def reset(self) -> None:
        """Clear conversation history and last_usage.

        Model, system prompt, and configuration remain unchanged.
        """
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[Any]:
        """Return a shallow copy of the message history.

        Modifications to the returned list do not affect the internal state.
        """
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return usage information from the last API response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return representation showing model, turn count, and context limit."""
        turn_count = len([m for m in self._history if m["role"] == "assistant"])
        limit_str = f", limit={self._context_window_limit}" if self._context_window_limit else ""
        return f"ConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    def _truncate_if_needed(self) -> None:
        """Auto-truncate oldest message pairs when approaching context limit.

        Raises ValueError:
            If a single user-assistant pair exceeds the context window.
        """
        assert self._context_window_limit is not None
        
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        if self._accurate_token_counting:
            estimated_tokens = self._count_tokens()
        else:
            if self._last_usage is None:
                return
            estimated_tokens = self._last_usage.input_tokens + self._last_usage.output_tokens

        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further — conversation with model {self._model!r} "
                    f"exceeds context window limit of {self._context_window_limit} tokens. "
                    f"Consider increasing context_window_limit or reducing message content."
                )

            pair_fraction = 2 / len(self._history)
            self._history.pop(0)
            self._history.pop(0)

            if self._accurate_token_counting:
                estimated_tokens = self._count_tokens()
            else:
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    def _count_tokens(self) -> int:
        """Count tokens in current history."""
        response = self._client.messages.count_tokens(
            messages=self._history,
            model=self._model,
            system=self._system,
        )
        return response.input_tokens


class AsyncConversationManager:
    """Async version of ConversationManager for managing multi-turn conversations.
    
    This helper maintains message history across multiple turns and automatically
    truncates the oldest message pairs when approaching the model's context window limit.
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
            max_tokens: Max output tokens for each API call.
            system: Optional system prompt.
            context_window_limit: Optional context window size. If set, enables auto-truncation.
            token_budget_headroom: Fraction [0.0, 1.0) of context to reserve. Default 0.10 (10%).
            accurate_token_counting: If True, call count_tokens() for precise estimates.
                                    If False, use heuristic from last response (faster).

        Raises ValueError:
            - If model is empty string
            - If max_tokens < 1
            - If context_window_limit provided but < 1
            - If token_budget_headroom not in [0.0, 1.0)
        """
        if not model:
            raise ValueError("model cannot be empty")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if context_window_limit is not None and context_window_limit < 1:
            raise ValueError(f"context_window_limit must be >= 1, got {context_window_limit}")
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
        """Add a user message to conversation history.

        Args:
            content: Text string or list of content blocks.

        Raises ValueError:
            If content is empty.
        """
        if isinstance(content, str):
            if not content:
                raise ValueError("content cannot be empty string")
        elif isinstance(content, list):
            if not content:
                raise ValueError("content list cannot be empty")
        
        self._history.append({"role": "user", "content": content})

    async def get_response(self, content: str | list[Any] | None = None, **kwargs: Any) -> Any:
        """Get assistant response for the conversation.

        Args:
            content: Optional user message to add before getting response.
            **kwargs: Additional arguments passed to client.messages.create()
                     (e.g., temperature, tools, etc.)

        Returns:
            Message object from API.

        Raises ValueError:
            If no user message is staged (history empty or last role != "user").
        """
        if content is not None:
            self.add_user_message(content)

        if not self._history or self._history[-1]["role"] != "user":
            raise ValueError(
                "No user message staged. Either pass content to get_response() "
                "or call add_user_message() first."
            )

        if self._context_window_limit is not None:
            await self._truncate_if_needed()

        api_kwargs = {
            "messages": list(self._history),
            "model": self._model,
            "max_tokens": self._max_tokens,
        }
        if self._system is not None:
            api_kwargs["system"] = self._system
        api_kwargs.update(kwargs)

        response = await self._client.messages.create(**api_kwargs)
        
        self._history.append({"role": "assistant", "content": response.content})
        self._last_usage = response.usage
        
        return response

    async def reset(self) -> None:
        """Clear conversation history and last_usage.

        Model, system prompt, and configuration remain unchanged.
        """
        self._history = []
        self._last_usage = None

    @property
    def history(self) -> list[Any]:
        """Return a shallow copy of the message history.

        Modifications to the returned list do not affect the internal state.
        """
        return list(self._history)

    @property
    def last_usage(self) -> Any | None:
        """Return usage information from the last API response, or None if no response yet."""
        return self._last_usage

    def __repr__(self) -> str:
        """Return representation showing model, turn count, and context limit."""
        turn_count = len([m for m in self._history if m["role"] == "assistant"])
        limit_str = f", limit={self._context_window_limit}" if self._context_window_limit else ""
        return f"AsyncConversationManager(model={self._model!r}, turns={turn_count}{limit_str})"

    async def _truncate_if_needed(self) -> None:
        """Auto-truncate oldest message pairs when approaching context limit.

        Raises ValueError:
            If a single user-assistant pair exceeds the context window.
        """
        assert self._context_window_limit is not None
        
        threshold = self._context_window_limit * (1.0 - self._token_budget_headroom)

        if self._accurate_token_counting:
            estimated_tokens = await self._count_tokens()
        else:
            if self._last_usage is None:
                return
            estimated_tokens = self._last_usage.input_tokens + self._last_usage.output_tokens

        while estimated_tokens >= threshold:
            if len(self._history) < 2:
                raise ValueError(
                    f"Cannot truncate further — conversation with model {self._model!r} "
                    f"exceeds context window limit of {self._context_window_limit} tokens. "
                    f"Consider increasing context_window_limit or reducing message content."
                )

            pair_fraction = 2 / len(self._history)
            self._history.pop(0)
            self._history.pop(0)

            if self._accurate_token_counting:
                estimated_tokens = await self._count_tokens()
            else:
                estimated_tokens = int(estimated_tokens * (1.0 - pair_fraction))

    async def _count_tokens(self) -> int:
        """Count tokens in current history."""
        response = await self._client.messages.count_tokens(
            messages=self._history,
            model=self._model,
            system=self._system,
        )
        return response.input_tokens
