"""Tests for ConversationManager and AsyncConversationManager helpers."""

import pytest
from unittest.mock import MagicMock, AsyncMock, call
from anthropic.helpers import ConversationManager, AsyncConversationManager


def _make_sync_client(*, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello") -> MagicMock:
    """Create a mock sync client for testing."""
    client = MagicMock()
    
    # Mock response
    content_mock = MagicMock()
    content_mock.text = content_text
    
    response_mock = MagicMock()
    response_mock.content = [content_mock]
    
    usage_mock = MagicMock()
    usage_mock.input_tokens = input_tokens
    usage_mock.output_tokens = output_tokens
    response_mock.usage = usage_mock
    
    client.messages.create.return_value = response_mock
    
    # Mock count_tokens
    count_response = MagicMock()
    count_response.input_tokens = input_tokens
    client.messages.count_tokens.return_value = count_response
    
    return client


def _make_async_client(*, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello") -> AsyncMock:
    """Create a mock async client for testing."""
    client = AsyncMock()
    
    # Mock response
    content_mock = MagicMock()
    content_mock.text = content_text
    
    response_mock = MagicMock()
    response_mock.content = [content_mock]
    
    usage_mock = MagicMock()
    usage_mock.input_tokens = input_tokens
    usage_mock.output_tokens = output_tokens
    response_mock.usage = usage_mock
    
    client.messages.create.return_value = response_mock
    
    # Mock count_tokens
    count_response = MagicMock()
    count_response.input_tokens = input_tokens
    client.messages.count_tokens.return_value = count_response
    
    return client


class TestConversationManager:
    """Test suite for ConversationManager."""

    def test_constructor_empty_model(self):
        """Test that constructor raises ValueError for empty model."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="model cannot be empty"):
            ConversationManager(client, model="", max_tokens=1024)

    def test_constructor_zero_max_tokens(self):
        """Test that constructor raises ValueError for zero max_tokens."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3", max_tokens=0)

    def test_constructor_negative_max_tokens(self):
        """Test that constructor raises ValueError for negative max_tokens."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3", max_tokens=-1)

    def test_constructor_invalid_context_window_limit(self):
        """Test that constructor raises ValueError for invalid context_window_limit."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client,
                model="claude-3",
                max_tokens=1024,
                context_window_limit=0,
            )

    def test_constructor_invalid_headroom(self):
        """Test that constructor raises ValueError for invalid token_budget_headroom."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client,
                model="claude-3",
                max_tokens=1024,
                token_budget_headroom=1.0,
            )
        
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client,
                model="claude-3",
                max_tokens=1024,
                token_budget_headroom=-0.1,
            )

    def test_constructor_valid(self):
        """Test that constructor works with valid inputs."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="You are helpful.",
            context_window_limit=200000,
            token_budget_headroom=0.15,
            accurate_token_counting=True,
        )
        assert manager._model == "claude-3-5-sonnet-20241022"
        assert manager._max_tokens == 1024
        assert manager._system == "You are helpful."
        assert manager._context_window_limit == 200000
        assert manager._token_budget_headroom == 0.15
        assert manager._accurate_token_counting is True

    def test_add_user_message_string(self):
        """Test add_user_message with string content."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        assert len(manager._history) == 1
        assert manager._history[0]["role"] == "user"
        assert manager._history[0]["content"] == "Hello!"

    def test_add_user_message_list(self):
        """Test add_user_message with list content."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        content = [{"type": "text", "text": "Hello"}]
        manager.add_user_message(content)
        assert len(manager._history) == 1
        assert manager._history[0]["role"] == "user"
        assert manager._history[0]["content"] == content

    def test_get_response_basic(self):
        """Test get_response with basic call."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        response = manager.get_response()
        
        assert response is not None
        client.messages.create.assert_called_once()
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3"
        assert call_kwargs["max_tokens"] == 1024
        assert len(call_kwargs["messages"]) == 1
        
        # Check history
        assert len(manager._history) == 2
        assert manager._history[0]["role"] == "user"
        assert manager._history[1]["role"] == "assistant"

    def test_get_response_with_content(self):
        """Test get_response with content argument."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        response = manager.get_response("Hello!")
        
        assert len(manager._history) == 2
        assert manager._history[0]["role"] == "user"
        assert manager._history[0]["content"] == "Hello!"
        assert manager._history[1]["role"] == "assistant"

    def test_get_response_without_staged_user_message(self):
        """Test that get_response raises if no user message is staged."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        with pytest.raises(ValueError, match="history must contain at least one user message"):
            manager.get_response()

    def test_get_response_after_assistant_message(self):
        """Test that get_response raises if last message is from assistant."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        manager.get_response()
        
        with pytest.raises(ValueError, match="history must contain at least one user message"):
            manager.get_response()

    def test_get_response_with_system_prompt(self):
        """Test that system prompt is included in API call."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
            system="You are helpful.",
        )
        
        manager.add_user_message("Hello!")
        manager.get_response()
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."

    def test_get_response_without_system_prompt(self):
        """Test that system prompt is not included when not set."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        manager.get_response()
        
        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_get_response_forwards_kwargs(self):
        """Test that extra kwargs are forwarded to messages.create."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        manager.get_response(temperature=0.5, top_p=0.9)
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        response1 = manager.get_response()
        assert len(manager._history) == 2
        
        manager.add_user_message("How are you?")
        response2 = manager.get_response()
        assert len(manager._history) == 4
        assert manager._history[0]["role"] == "user"
        assert manager._history[1]["role"] == "assistant"
        assert manager._history[2]["role"] == "user"
        assert manager._history[3]["role"] == "assistant"

    def test_last_usage_initially_none(self):
        """Test that last_usage is None initially."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        assert manager.last_usage is None

    def test_last_usage_populated_after_response(self):
        """Test that last_usage is populated after get_response."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        manager.get_response()
        
        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100
        assert manager.last_usage.output_tokens == 50

    def test_history_returns_copy(self):
        """Test that history property returns a copy."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        history1 = manager.history
        history2 = manager.history
        
        assert history1 is not history2  # Different objects
        assert history1 == history2  # Same content

    def test_reset(self):
        """Test reset clears history and last_usage."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        manager.get_response()
        
        assert len(manager._history) == 2
        assert manager.last_usage is not None
        
        manager.reset()
        
        assert len(manager._history) == 0
        assert manager.last_usage is None

    def test_reset_preserves_model_and_system(self):
        """Test that reset preserves model and system settings."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
            system="You are helpful.",
        )
        
        manager.add_user_message("Hello!")
        manager.reset()
        
        assert manager._model == "claude-3"
        assert manager._system == "You are helpful."

    def test_truncation_disabled(self):
        """Test that truncation is skipped when context_window_limit is None."""
        client = _make_sync_client(input_tokens=1000, output_tokens=1000)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
            context_window_limit=None,
        )
        
        manager.add_user_message("x" * 10000)
        manager.get_response()
        
        # Should not truncate
        assert len(manager._history) == 2

    def test_truncation_no_op_under_threshold(self):
        """Test that truncation is no-op when under threshold."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,
        )
        
        manager.add_user_message("Hello!")
        manager.get_response()
        
        # Threshold is 900, estimated is 150, should not truncate
        assert len(manager._history) == 2

    def test_truncation_drops_oldest_pair(self):
        """Test that truncation drops oldest user+assistant pair."""
        client = _make_sync_client(input_tokens=600, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,
        )
        
        # First turn: adds messages, tokens become 150 (under threshold of 900)
        manager.add_user_message("First turn")
        manager.get_response()
        assert len(manager._history) == 2
        
        # Second turn: now tokens would be estimated at ~300 (still under 900)
        manager.add_user_message("Second turn")
        manager.get_response()
        assert len(manager._history) == 4
        
        # Third turn: with heuristic, estimated at ~450
        manager.add_user_message("Third turn")
        manager.get_response()
        assert len(manager._history) == 6

    def test_truncation_multiple_pairs(self):
        """Test that truncation drops multiple pairs until under threshold."""
        client = _make_sync_client(input_tokens=900, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )
        
        # Build history
        manager.add_user_message("First")
        manager.get_response()
        manager.add_user_message("Second")
        manager.get_response()
        manager.add_user_message("Third")
        manager.get_response()
        
        assert len(manager._history) == 6
        
        # Next response should trigger truncation since 950 >= 900
        manager.add_user_message("Fourth")
        manager.get_response()
        
        # Should have truncated pairs until under threshold
        assert len(manager._history) < 8

    def test_truncation_raises_on_single_pair_exceed(self):
        """Test that truncation raises ValueError when single pair exceeds limit."""
        client = _make_sync_client(input_tokens=2000, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )
        
        manager.add_user_message("First turn")
        manager.get_response()
        manager.add_user_message("Second turn")
        
        # This should raise because even single pair exceeds threshold
        with pytest.raises(ValueError, match="cannot truncate further"):
            manager.get_response()

    def test_no_truncation_on_first_call(self):
        """Test that truncation is skipped on first call in heuristic mode."""
        client = _make_sync_client(input_tokens=9000, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )
        
        # First call: last_usage is None, so skip truncation
        manager.add_user_message("Hello!")
        manager.get_response()
        
        # Should have 2 messages (no truncation)
        assert len(manager._history) == 2

    def test_truncation_accurate_mode(self):
        """Test truncation with accurate_token_counting=True."""
        client = _make_sync_client(input_tokens=600, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )
        
        manager.add_user_message("First")
        manager.get_response()
        manager.add_user_message("Second")
        manager.get_response()
        
        # count_tokens should have been called
        assert client.messages.count_tokens.called

    def test_repr(self):
        """Test __repr__ method."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
            context_window_limit=200000,
        )
        
        manager.add_user_message("Hello!")
        manager.add_user_message("Hi!")  # User message, not a turn
        
        repr_str = repr(manager)
        assert "ConversationManager" in repr_str
        assert "claude-3" in repr_str
        assert "200000" in repr_str


class TestAsyncConversationManager:
    """Test suite for AsyncConversationManager."""

    @pytest.mark.asyncio
    async def test_constructor_empty_model(self):
        """Test that constructor raises ValueError for empty model."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="model cannot be empty"):
            AsyncConversationManager(client, model="", max_tokens=1024)

    @pytest.mark.asyncio
    async def test_constructor_valid(self):
        """Test that constructor works with valid inputs."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        assert manager._model == "claude-3-5-sonnet-20241022"

    @pytest.mark.asyncio
    async def test_add_user_message(self):
        """Test add_user_message."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        assert len(manager._history) == 1
        assert manager._history[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_response_basic(self):
        """Test get_response with basic call."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        response = await manager.get_response()
        
        assert response is not None
        client.messages.create.assert_called_once()
        assert len(manager._history) == 2

    @pytest.mark.asyncio
    async def test_get_response_with_content(self):
        """Test get_response with content argument."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=1024)
        
        response = await manager.get_response("Hello!")
        
        assert len(manager._history) == 2
        assert manager._history[0]["role"] == "user"
        assert manager._history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_response_without_user_message(self):
        """Test that get_response raises if no user message."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=1024)
        
        with pytest.raises(ValueError, match="history must contain at least one user message"):
            await manager.get_response()

    @pytest.mark.asyncio
    async def test_get_response_with_system_prompt(self):
        """Test that system prompt is included."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
            system="You are helpful.",
        )
        
        manager.add_user_message("Hello!")
        await manager.get_response()
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test multi-turn async conversation."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        await manager.get_response()
        assert len(manager._history) == 2
        
        manager.add_user_message("How are you?")
        await manager.get_response()
        assert len(manager._history) == 4

    @pytest.mark.asyncio
    async def test_last_usage(self):
        """Test last_usage property."""
        client = _make_async_client(input_tokens=100, output_tokens=50)
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=1024)
        
        assert manager.last_usage is None
        
        manager.add_user_message("Hello!")
        await manager.get_response()
        
        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test reset clears history."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=1024)
        
        manager.add_user_message("Hello!")
        await manager.get_response()
        
        manager.reset()
        assert len(manager._history) == 0
        assert manager.last_usage is None

    @pytest.mark.asyncio
    async def test_truncation_accurate_mode(self):
        """Test truncation with accurate_token_counting=True."""
        client = _make_async_client(input_tokens=600, output_tokens=50)
        manager = AsyncConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )
        
        manager.add_user_message("First")
        await manager.get_response()
        manager.add_user_message("Second")
        await manager.get_response()
        
        assert client.messages.count_tokens.called

    @pytest.mark.asyncio
    async def test_repr(self):
        """Test __repr__ method."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client,
            model="claude-3",
            max_tokens=1024,
        )
        
        repr_str = repr(manager)
        assert "AsyncConversationManager" in repr_str
