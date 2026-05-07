"""Tests for ConversationManager and AsyncConversationManager."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from anthropic.helpers import ConversationManager, AsyncConversationManager


def _make_sync_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock sync client for testing."""
    client = MagicMock()
    
    # Mock messages.create response
    response = MagicMock()
    response.content = [MagicMock(text=content_text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    client.messages.create = MagicMock(return_value=response)
    
    # Mock messages.count_tokens response
    count_response = MagicMock()
    count_response.input_tokens = input_tokens
    client.messages.count_tokens = MagicMock(return_value=count_response)
    
    return client


def _make_async_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock async client for testing."""
    client = MagicMock()
    
    # Mock messages.create response
    response = MagicMock()
    response.content = [MagicMock(text=content_text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    client.messages.create = AsyncMock(return_value=response)
    
    # Mock messages.count_tokens response
    count_response = MagicMock()
    count_response.input_tokens = input_tokens
    client.messages.count_tokens = AsyncMock(return_value=count_response)
    
    return client


class TestConversationManager:
    """Test suite for ConversationManager (sync)."""

    def test_constructor_empty_model(self) -> None:
        """Constructor raises ValueError for empty model."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ConversationManager(client, model="", max_tokens=100)

    def test_constructor_zero_max_tokens(self) -> None:
        """Constructor raises ValueError for max_tokens < 1."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=0)

    def test_constructor_negative_max_tokens(self) -> None:
        """Constructor raises ValueError for negative max_tokens."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=-1)

    def test_constructor_negative_context_window_limit(self) -> None:
        """Constructor raises ValueError for negative context_window_limit."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client,
                model="claude-3-5-sonnet-latest",
                max_tokens=100,
                context_window_limit=-1,
            )

    def test_constructor_zero_context_window_limit(self) -> None:
        """Constructor raises ValueError for zero context_window_limit."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client,
                model="claude-3-5-sonnet-latest",
                max_tokens=100,
                context_window_limit=0,
            )

    def test_constructor_invalid_token_budget_headroom_too_high(self) -> None:
        """Constructor raises ValueError for token_budget_headroom >= 1.0."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client,
                model="claude-3-5-sonnet-latest",
                max_tokens=100,
                token_budget_headroom=1.0,
            )

    def test_constructor_invalid_token_budget_headroom_negative(self) -> None:
        """Constructor raises ValueError for negative token_budget_headroom."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client,
                model="claude-3-5-sonnet-latest",
                max_tokens=100,
                token_budget_headroom=-0.1,
            )

    def test_constructor_valid(self) -> None:
        """Constructor succeeds with valid arguments."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-latest",
            max_tokens=512,
            system="You are helpful.",
            context_window_limit=4096,
            token_budget_headroom=0.15,
        )
        assert manager._model == "claude-3-5-sonnet-latest"
        assert manager._max_tokens == 512
        assert manager._system == "You are helpful."
        assert manager._context_window_limit == 4096
        assert manager._token_budget_headroom == 0.15

    def test_add_user_message_string(self) -> None:
        """add_user_message appends user message to history."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0] == {"role": "user", "content": "Hello"}

    def test_add_user_message_list(self) -> None:
        """add_user_message handles list content."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        content = [{"type": "text", "text": "Hello"}]
        manager.add_user_message(content)
        assert len(manager.history) == 1
        assert manager.history[0] == {"role": "user", "content": content}

    def test_add_user_message_empty_string(self) -> None:
        """add_user_message raises ValueError for empty string."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_user_message("")

    def test_add_user_message_empty_list(self) -> None:
        """add_user_message raises ValueError for empty list."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_user_message([])

    def test_get_response_with_content(self) -> None:
        """get_response with content arg adds message and calls API."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        response = manager.get_response("What is 2+2?")
        
        # Should add user message, call API, append assistant message
        assert len(manager.history) == 2
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "What is 2+2?"
        assert manager.history[1]["role"] == "assistant"
        assert response is not None
        client.messages.create.assert_called_once()

    def test_get_response_without_content(self) -> None:
        """get_response without content uses pre-staged message."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        manager.add_user_message("What is 2+2?")
        response = manager.get_response()
        
        # Should use existing message
        assert len(manager.history) == 2
        assert manager.history[1]["role"] == "assistant"
        assert response is not None

    def test_get_response_no_user_message_raises(self) -> None:
        """get_response raises if no user message staged."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        with pytest.raises(ValueError, match="No staged user message"):
            manager.get_response()

    def test_get_response_last_message_not_user_raises(self) -> None:
        """get_response raises if last message is assistant, not user."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        manager.add_user_message("Hello")
        manager.get_response()  # This adds assistant message
        
        with pytest.raises(ValueError, match="No staged user message"):
            manager.get_response()

    def test_multi_turn_conversation(self) -> None:
        """Multi-turn conversation maintains history correctly."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        # Turn 1
        r1 = manager.get_response("Hello")
        assert len(manager.history) == 2  # user + assistant
        
        # Turn 2
        r2 = manager.get_response("How are you?")
        assert len(manager.history) == 4  # 2 + 2 more
        
        # Verify roles alternate
        roles = [m["role"] for m in manager.history]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_last_usage_none_initially(self) -> None:
        """last_usage is None before any API call."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        assert manager.last_usage is None

    def test_last_usage_populated(self) -> None:
        """last_usage is populated after API call."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        manager.get_response("Hello")
        
        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100
        assert manager.last_usage.output_tokens == 50

    def test_kwargs_forwarded_to_api(self) -> None:
        """Additional kwargs are forwarded to messages.create."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        manager.get_response("Hello", temperature=0.5, top_p=0.9)
        
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    def test_system_prompt_included(self) -> None:
        """System prompt is included in API call."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-latest",
            max_tokens=100,
            system="You are helpful.",
        )
        
        manager.get_response("Hello")
        
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful."

    def test_system_prompt_omitted_when_none(self) -> None:
        """System prompt is omitted from API call when None."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-latest", max_tokens=100, system=None
        )
        
        manager.get_response("Hello")
        
        call_kwargs = client.messages.create.call_args.kwargs
        assert "system" not in call_kwargs

    def test_history_returns_copy(self) -> None:
        """history property returns a copy, not reference."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        manager.add_user_message("Hello")
        history1 = manager.history
        history1.append({"role": "test", "content": "mutated"})
        
        history2 = manager.history
        assert len(history2) == 1  # Not mutated
        assert len(manager._history) == 1  # Internal state unchanged

    def test_reset(self) -> None:
        """reset() clears history and last_usage."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-latest", max_tokens=100)
        
        manager.get_response("Hello")
        assert len(manager.history) > 0
        assert manager.last_usage is not None
        
        manager.reset()
        assert len(manager.history) == 0
        assert manager.last_usage is None

    def test_reset_preserves_settings(self) -> None:
        """reset() does not affect model/system settings."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-latest",
            max_tokens=256,
            system="You are helpful.",
        )
        
        manager.get_response("Hello")
        manager.reset()
        
        assert manager._model == "claude-3-5-sonnet-latest"
        assert manager._max_tokens == 256
        assert manager._system == "You are helpful."

    def test_no_truncation_when_limit_none(self) -> None:
        """No truncation occurs when context_window_limit is None."""
        client = _make_sync_client(input_tokens=1000, output_tokens=100)
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-latest", max_tokens=100, context_window_limit=None
        )
        
        # Add many messages
        for i in range(10):
            manager.get_response(f"Message {i}")
        
        # Should have 20 messages (10 user + 10 assistant)
        assert len(manager.history) == 20

    def test_no_truncation_when_under_threshold(self) -> None:
        """No truncation when token count is below threshold."""
        client = _make_sync_client(input_tokens=50, output_tokens=10)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-latest",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.1,
        )
        
        manager.get_response("Hello")
        initial_history_len = len(manager.history)
        
        manager.get_response("Hi again")
        # Should not truncate since 60 + 60 < 900 (threshold)
        assert len(manager.history) == 4

    def test_no_truncation_on_first_call_heuristic(self) -> None:
        """No truncation on first call when using heuristic mode (last_usage is None)."""
        client = _make_sync_client(input_tokens=5000, output_tokens=1000)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-latest",
            max_tokens=100,
            context_window_limit=1000,
            accurate_token_counting=False,
        )
        
        # First call has last_usage=None, should skip truncation
        manager.get_response("Hello")
        assert len(manager.history) == 2  # user + assistant, not truncated

    def test_truncation_drops_oldest_pair(self) -> None:
        """Truncation drops oldest user+assistant pair."""
        client = _make_sync_client(input_tokens=500, output_tokens=100)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-latest",
            max_tokens=100,
            context_window_limit=600,
            token_budget_headroom=0.0,
        )
        
        # Add first turn
        manager.get_response("First")
        first_history = manager.history.copy()
        
        # Add second turn - should trigger truncation
        manager.get_response("Second")
        
        # Should have dropped the first pair (indices 0, 1)
        assert len(manager.history) == 2
        # First message should be "Second", not "First"
        assert "Second" in str(manager.history[0])

    def test_truncation_raises_on_single_pair_exceeds(self) -> None:
        """Truncation raises ValueError when single pair exceeds limit."""
        client = _make_sync_client(input_tokens=2000, output_tokens=1000)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-latest",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.0,
        )
        
        # First call doesn't truncate (last_usage is None), but second call will
        manager.get_response("First message")
        
        with pytest.raises(ValueError, match="Cannot truncate further"):
            manager.get_response("This message pair still exceeds the limit")

    def test_truncation_accurate_mode(self) -> None:
        """Truncation uses count_tokens in accurate mode."""
        client = _make_sync_client(input_tokens=500, output_tokens=100)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-latest",
            max_tokens=100,
            context_window_limit=600,
            token_budget_headroom=0.0,
            accurate_token_counting=True,
        )
        
        manager.get_response("First")
        manager.get_response("Second")
        
        # count_tokens should have been called
        assert client.messages.count_tokens.called

    def test_repr(self) -> None:
        """__repr__ returns informative string."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-latest",
            max_tokens=512,
            context_window_limit=4096,
        )
        
        manager.get_response("Hello")
        repr_str = repr(manager)
        
        assert "ConversationManager" in repr_str
        assert "claude-3-5-sonnet-latest" in repr_str
        assert "4096" in repr_str


class TestAsyncConversationManager:
    """Test suite for AsyncConversationManager."""

    @pytest.mark.asyncio
    async def test_constructor_valid(self) -> None:
        """Constructor succeeds with valid arguments."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client,
            model="claude-3-5-sonnet-latest",
            max_tokens=512,
            system="You are helpful.",
            context_window_limit=4096,
        )
        assert manager._model == "claude-3-5-sonnet-latest"
        assert manager._max_tokens == 512

    @pytest.mark.asyncio
    async def test_constructor_empty_model(self) -> None:
        """Constructor raises ValueError for empty model."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            AsyncConversationManager(client, model="", max_tokens=100)

    @pytest.mark.asyncio
    async def test_get_response_with_content(self) -> None:
        """get_response with content arg adds message and calls API."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-latest", max_tokens=100
        )
        
        response = await manager.get_response("What is 2+2?")
        
        assert len(manager.history) == 2
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"
        assert response is not None
        client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_response_no_user_message_raises(self) -> None:
        """get_response raises if no user message staged."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-latest", max_tokens=100
        )
        
        with pytest.raises(ValueError, match="No staged user message"):
            await manager.get_response()

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self) -> None:
        """Multi-turn conversation maintains history correctly."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-latest", max_tokens=100
        )
        
        await manager.get_response("Hello")
        assert len(manager.history) == 2
        
        await manager.get_response("How are you?")
        assert len(manager.history) == 4

    @pytest.mark.asyncio
    async def test_last_usage_populated(self) -> None:
        """last_usage is populated after API call."""
        client = _make_async_client(input_tokens=100, output_tokens=50)
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-latest", max_tokens=100
        )
        
        await manager.get_response("Hello")
        
        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """reset() clears history and last_usage."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-latest", max_tokens=100
        )
        
        await manager.get_response("Hello")
        assert len(manager.history) > 0
        assert manager.last_usage is not None
        
        manager.reset()
        assert len(manager.history) == 0
        assert manager.last_usage is None

    @pytest.mark.asyncio
    async def test_truncation_accurate_mode(self) -> None:
        """Truncation uses count_tokens in accurate mode."""
        client = _make_async_client(input_tokens=500, output_tokens=100)
        manager = AsyncConversationManager(
            client,
            model="claude-3-5-sonnet-latest",
            max_tokens=100,
            context_window_limit=600,
            token_budget_headroom=0.0,
            accurate_token_counting=True,
        )
        
        await manager.get_response("First")
        await manager.get_response("Second")
        
        # count_tokens should have been called
        assert client.messages.count_tokens.called

    @pytest.mark.asyncio
    async def test_repr(self) -> None:
        """__repr__ returns informative string."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client,
            model="claude-3-5-sonnet-latest",
            max_tokens=512,
            context_window_limit=4096,
        )
        
        await manager.get_response("Hello")
        repr_str = repr(manager)
        
        assert "AsyncConversationManager" in repr_str
        assert "claude-3-5-sonnet-latest" in repr_str
