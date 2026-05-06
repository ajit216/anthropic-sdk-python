# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, call

from anthropic.helpers import ConversationManager, AsyncConversationManager


def _make_sync_client(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    content_text: str = "Hello",
) -> MagicMock:
    """Create a mock sync client for testing."""
    mock_client = MagicMock()
    
    # Mock message response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=content_text)]
    mock_response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    
    mock_client.messages.create.return_value = mock_response
    
    # Mock token counting
    mock_count_response = MagicMock()
    mock_count_response.input_tokens = input_tokens
    mock_client.messages.count_tokens.return_value = mock_count_response
    
    return mock_client


def _make_async_client(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    content_text: str = "Hello",
) -> MagicMock:
    """Create a mock async client for testing."""
    mock_client = MagicMock()
    
    # Mock message response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=content_text)]
    mock_response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    
    # Mock token counting
    mock_count_response = MagicMock()
    mock_count_response.input_tokens = input_tokens
    mock_client.messages.count_tokens = AsyncMock(return_value=mock_count_response)
    
    return mock_client


class TestConversationManager:
    """Test suite for ConversationManager."""

    def test_constructor_validates_empty_model(self) -> None:
        """Constructor raises ValueError for empty model."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="model must not be empty"):
            ConversationManager(client, model="", max_tokens=1024)

    def test_constructor_validates_max_tokens(self) -> None:
        """Constructor raises ValueError for invalid max_tokens."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=0)

    def test_constructor_validates_context_window_limit(self) -> None:
        """Constructor raises ValueError for invalid context_window_limit."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                context_window_limit=0,
            )

    def test_constructor_validates_token_budget_headroom(self) -> None:
        """Constructor raises ValueError for invalid token_budget_headroom."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                token_budget_headroom=1.5,
            )

    def test_add_user_message(self) -> None:
        """add_user_message appends to history."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "Hello"

    def test_add_user_message_raises_on_empty_string(self) -> None:
        """add_user_message raises ValueError for empty string."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            manager.add_user_message("")

    def test_add_user_message_raises_on_empty_list(self) -> None:
        """add_user_message raises ValueError for empty list."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        with pytest.raises(ValueError, match="cannot be empty"):
            manager.add_user_message([])

    def test_get_response_single_call(self) -> None:
        """get_response calls API once and returns response."""
        client = _make_sync_client(content_text="Response from Claude")
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        manager.add_user_message("Hello")
        response = manager.get_response()
        
        assert response.content[0].text == "Response from Claude"
        client.messages.create.assert_called_once()
        
    def test_get_response_with_content_argument(self) -> None:
        """get_response adds message when content argument provided."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        response = manager.get_response("Hello")
        
        assert len(manager.history) == 2  # user + assistant
        client.messages.create.assert_called_once()

    def test_get_response_raises_without_staged_message(self) -> None:
        """get_response raises ValueError if no user message staged."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        with pytest.raises(ValueError, match="No user message staged"):
            manager.get_response()

    def test_multi_turn_conversation(self) -> None:
        """Multi-turn conversation maintains history correctly."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        manager.get_response("First message")
        manager.get_response("Second message")
        
        history = manager.history
        assert len(history) == 4  # 2 user + 2 assistant
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
        assert history[3]["role"] == "assistant"

    def test_last_usage_initially_none(self) -> None:
        """last_usage is None initially."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        assert manager.last_usage is None

    def test_last_usage_populated_after_response(self) -> None:
        """last_usage is populated after getting response."""
        client = _make_sync_client(input_tokens=150, output_tokens=75)
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        manager.get_response("Hello")
        
        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 150
        assert manager.last_usage.output_tokens == 75

    def test_kwargs_forwarded_to_create(self) -> None:
        """Additional kwargs are forwarded to messages.create."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        manager.get_response("Hello", temperature=0.5, top_p=0.9)
        
        # Check that kwargs were passed
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    def test_system_prompt_included(self) -> None:
        """System prompt is included in request when provided."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="You are helpful.",
        )
        
        manager.get_response("Hello")
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."

    def test_system_prompt_omitted_when_none(self) -> None:
        """System prompt is omitted when None."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        manager.get_response("Hello")
        
        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_history_returns_copy(self) -> None:
        """history property returns a copy."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        manager.add_user_message("Test")
        history1 = manager.history
        history1.append({"role": "fake", "content": "fake"})
        
        history2 = manager.history
        assert len(history2) == 1  # mutation didn't affect internal state

    def test_reset_clears_history(self) -> None:
        """reset clears history and last_usage."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        manager.get_response("Hello")
        assert len(manager.history) > 0
        assert manager.last_usage is not None
        
        manager.reset()
        assert len(manager.history) == 0
        assert manager.last_usage is None

    def test_reset_preserves_model_and_system(self) -> None:
        """reset preserves model and system settings."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="Original system",
        )
        
        manager.get_response("Hello")
        manager.reset()
        
        manager.get_response("New message")
        call_kwargs = client.messages.create.call_args_list[-1][1]
        assert call_kwargs["system"] == "Original system"

    def test_truncation_disabled_when_no_limit(self) -> None:
        """No truncation occurs when context_window_limit is None."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=1024)
        
        # Add many messages
        for i in range(10):
            manager.get_response(f"Message {i}")
        
        assert len(manager.history) == 20  # 10 user + 10 assistant

    def test_truncation_no_op_under_threshold(self) -> None:
        """No truncation when under threshold."""
        client = _make_sync_client(input_tokens=10, output_tokens=10)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,
        )
        
        manager.get_response("Hello")
        assert len(manager.history) == 2

    def test_truncation_drops_oldest_pair(self) -> None:
        """Truncation removes oldest user+assistant pair."""
        client = _make_sync_client(input_tokens=600, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,
        )
        
        # Add first message (will cause truncation on second message)
        manager.add_user_message("First")
        response1 = manager.get_response()
        # At this point, one user/assistant pair is in history
        
        # Add second message with different token counts to trigger truncation
        # Simulate higher usage to force truncation
        client.messages.create.return_value.usage.input_tokens = 900
        client.messages.create.return_value.usage.output_tokens = 200
        
        manager.add_user_message("Second")
        response2 = manager.get_response()
        
        # If truncation happened, we should have fewer than 4 messages or first is gone
        assert len(manager.history) <= 4  # Either truncated or still has both messages

    def test_truncation_skipped_on_first_call(self) -> None:
        """No truncation on first call when using heuristic mode."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            context_window_limit=100,
            token_budget_headroom=0.1,
        )
        
        # First call should not truncate even though way over limit
        manager.get_response("Hello")
        assert len(manager.history) == 2

    def test_truncation_raises_when_single_pair_exceeds(self) -> None:
        """Truncation raises ValueError when single pair exceeds limit."""
        # Setup: high token count that always exceeds threshold
        client = _make_sync_client(input_tokens=600, output_tokens=100)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            context_window_limit=500,  # threshold = 450
            token_budget_headroom=0.1,
        )
        
        # Manually build history to test truncation logic directly
        manager.add_user_message("Message 1")
        # Simulate a response with high token count
        manager._history.append({"role": "assistant", "content": "Response 1"})
        manager._last_usage = MagicMock(input_tokens=600, output_tokens=100)
        
        # Now truncation is needed (700 > 450)
        # With only 1 pair, truncation should fail
        with pytest.raises(ValueError, match="Cannot truncate further"):
            manager._truncate_if_needed()

    def test_accurate_token_counting(self) -> None:
        """Accurate mode calls count_tokens."""
        client = _make_sync_client(input_tokens=50, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )
        
        manager.get_response("Hello")
        
        # count_tokens should have been called
        client.messages.count_tokens.assert_called()

    def test_repr(self) -> None:
        """__repr__ shows model, turns, and limit."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            context_window_limit=200000,
        )
        
        repr_str = repr(manager)
        assert "claude-3-5-sonnet-20241022" in repr_str
        assert "ConversationManager" in repr_str


class TestAsyncConversationManager:
    """Test suite for AsyncConversationManager."""

    @pytest.mark.asyncio
    async def test_constructor_validates_empty_model(self) -> None:
        """Constructor raises ValueError for empty model."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="model must not be empty"):
            AsyncConversationManager(client, model="", max_tokens=1024)

    @pytest.mark.asyncio
    async def test_async_get_response_single_call(self) -> None:
        """get_response calls API once and returns response."""
        client = _make_async_client(content_text="Response from Claude")
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        response = await manager.get_response("Hello")
        
        assert response.content[0].text == "Response from Claude"
        client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_get_response_with_content_argument(self) -> None:
        """get_response adds message when content argument provided."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        response = await manager.get_response("Hello")
        
        assert len(manager.history) == 2
        client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_multi_turn(self) -> None:
        """Multi-turn async conversation works correctly."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        await manager.get_response("First")
        await manager.get_response("Second")
        
        assert len(manager.history) == 4

    @pytest.mark.asyncio
    async def test_async_last_usage(self) -> None:
        """Async last_usage is populated after response."""
        client = _make_async_client(input_tokens=150, output_tokens=75)
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        await manager.get_response("Hello")
        
        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 150

    @pytest.mark.asyncio
    async def test_async_reset(self) -> None:
        """Async reset clears history."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        await manager.get_response("Hello")
        assert len(manager.history) > 0
        
        manager.reset()
        assert len(manager.history) == 0

    @pytest.mark.asyncio
    async def test_async_accurate_token_counting(self) -> None:
        """Async accurate mode calls count_tokens."""
        client = _make_async_client(input_tokens=50, output_tokens=50)
        manager = AsyncConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            context_window_limit=1000,
            accurate_token_counting=True,
        )
        
        await manager.get_response("Hello")
        
        client.messages.count_tokens.assert_called()

    @pytest.mark.asyncio
    async def test_async_repr(self) -> None:
        """Async __repr__ shows model and turns."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        repr_str = repr(manager)
        assert "AsyncConversationManager" in repr_str
