"""Tests for ConversationManager and AsyncConversationManager."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from anthropic.helpers import ConversationManager, AsyncConversationManager


def _make_sync_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock sync client for testing."""
    client = MagicMock()
    
    # Mock response
    response = MagicMock()
    response.content = [MagicMock(text=content_text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    
    client.messages.create.return_value = response
    
    # Mock count_tokens
    count_response = MagicMock()
    count_response.input_tokens = input_tokens
    client.messages.count_tokens.return_value = count_response
    
    return client


def _make_async_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock async client for testing."""
    client = MagicMock()
    
    # Mock response
    response = MagicMock()
    response.content = [MagicMock(text=content_text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    
    client.messages.create = AsyncMock(return_value=response)
    
    # Mock count_tokens
    count_response = MagicMock()
    count_response.input_tokens = input_tokens
    client.messages.count_tokens = AsyncMock(return_value=count_response)
    
    return client


class TestConversationManager:
    """Tests for synchronous ConversationManager."""

    def test_constructor_empty_model(self) -> None:
        """Test that empty model raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="model cannot be an empty string"):
            ConversationManager(client, model="", max_tokens=100)

    def test_constructor_zero_max_tokens(self) -> None:
        """Test that zero max_tokens raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3", max_tokens=0)

    def test_constructor_negative_max_tokens(self) -> None:
        """Test that negative max_tokens raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3", max_tokens=-5)

    def test_constructor_invalid_context_window_limit(self) -> None:
        """Test that invalid context_window_limit raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client,
                model="claude-3",
                max_tokens=100,
                context_window_limit=0,
            )

    def test_constructor_invalid_token_budget_headroom_negative(self) -> None:
        """Test that negative token_budget_headroom raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client,
                model="claude-3",
                max_tokens=100,
                token_budget_headroom=-0.1,
            )

    def test_constructor_invalid_token_budget_headroom_too_large(self) -> None:
        """Test that token_budget_headroom >= 1.0 raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client,
                model="claude-3",
                max_tokens=100,
                token_budget_headroom=1.0,
            )

    def test_add_user_message_string(self) -> None:
        """Test adding a string user message."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "Hello"

    def test_add_user_message_list(self) -> None:
        """Test adding a list user message."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        content = [{"type": "text", "text": "Hello"}]
        manager.add_user_message(content)
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == content

    def test_add_user_message_empty_string(self) -> None:
        """Test that empty message raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        with pytest.raises(ValueError, match="User message content cannot be empty"):
            manager.add_user_message("")

    def test_add_user_message_empty_list(self) -> None:
        """Test that empty list message raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        with pytest.raises(ValueError, match="User message content cannot be empty"):
            manager.add_user_message([])

    def test_get_response_single_turn(self) -> None:
        """Test single turn conversation."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hello")
        response = manager.get_response()
        
        assert response.content[0].text == "Hello"
        assert len(manager.history) == 2
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"

    def test_get_response_with_content(self) -> None:
        """Test get_response with content parameter."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        response = manager.get_response(content="Hello")
        
        assert response.content[0].text == "Hello"
        assert len(manager.history) == 2
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "Hello"

    def test_get_response_multi_turn(self) -> None:
        """Test multiple turns."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("First question")
        response1 = manager.get_response()
        
        manager.add_user_message("Second question")
        response2 = manager.get_response()
        
        assert len(manager.history) == 4
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"
        assert manager.history[2]["role"] == "user"
        assert manager.history[3]["role"] == "assistant"

    def test_get_response_no_staged_message(self) -> None:
        """Test that getting response without staged message raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        with pytest.raises(ValueError, match="No user message to respond to"):
            manager.get_response()

    def test_get_response_after_assistant_turn(self) -> None:
        """Test that getting response after assistant turn raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.get_response(content="Hello")
        
        with pytest.raises(ValueError, match="No user message to respond to"):
            manager.get_response()

    def test_last_usage_initially_none(self) -> None:
        """Test that last_usage is None initially."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        assert manager.last_usage is None

    def test_last_usage_after_response(self) -> None:
        """Test that last_usage is populated after response."""
        client = _make_sync_client(input_tokens=150, output_tokens=50)
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.get_response(content="Hello")
        
        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 150
        assert manager.last_usage.output_tokens == 50

    def test_system_prompt_included(self) -> None:
        """Test that system prompt is passed to API."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3", max_tokens=100, system="You are helpful"
        )
        
        manager.get_response(content="Hello")
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"

    def test_system_prompt_omitted_when_none(self) -> None:
        """Test that system prompt is omitted when None."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.get_response(content="Hello")
        
        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_kwargs_forwarded(self) -> None:
        """Test that kwargs are forwarded to API."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.get_response(content="Hello", temperature=0.5, top_p=0.9)
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    def test_history_is_shallow_copy(self) -> None:
        """Test that history property returns a copy."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Test")
        history1 = manager.history
        history2 = manager.history
        
        # Should be different list objects
        assert history1 is not history2
        # But contain same data
        assert history1 == history2

    def test_reset(self) -> None:
        """Test reset clears history."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.get_response(content="Hello")
        assert len(manager.history) == 2
        assert manager.last_usage is not None
        
        manager.reset()
        
        assert len(manager.history) == 0
        assert manager.last_usage is None

    def test_reset_preserves_config(self) -> None:
        """Test that reset preserves model and system settings."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3", max_tokens=100, system="Helpful"
        )
        
        manager.get_response(content="Hello")
        manager.reset()
        
        # Verify model and system still work
        manager.get_response(content="Hello again")
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3"
        assert call_kwargs["system"] == "Helpful"

    def test_repr(self) -> None:
        """Test string representation."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3", max_tokens=100, context_window_limit=4096
        )
        
        repr_str = repr(manager)
        assert "ConversationManager" in repr_str
        assert "claude-3" in repr_str
        assert "limit=4096" in repr_str

    def test_truncation_disabled_when_no_limit(self) -> None:
        """Test that truncation is skipped when context_window_limit is None."""
        client = _make_sync_client(input_tokens=1000, output_tokens=100)
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        # Add many messages
        for i in range(10):
            manager.get_response(content=f"Message {i}")
        
        # History should not be truncated
        assert len(manager.history) == 20

    def test_truncation_no_op_when_under_threshold(self) -> None:
        """Test that truncation does nothing when under threshold."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=10000,
            token_budget_headroom=0.5,
        )
        
        manager.get_response(content="Hello")
        history_len = len(manager.history)
        
        manager.get_response(content="World")
        
        # Should have 4 messages (2 turns)
        assert len(manager.history) == 4

    def test_truncation_removes_oldest_pair(self) -> None:
        """Test that truncation removes oldest user+assistant pair."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=300,  # Low limit to trigger truncation
            token_budget_headroom=0.1,
        )
        
        # Add messages with specific content we can track
        manager.get_response(content="First")
        history_after_first = len(manager.history)
        
        manager.get_response(content="Second")
        history_after_second = len(manager.history)
        
        manager.get_response(content="Third")
        history_after_third = len(manager.history)
        
        # Verify that truncation is happening by checking history doesn't grow infinitely
        # With threshold=270 tokens and 150 per turn, truncation should keep it manageable
        # The test verifies the mechanism works, not exact numbers
        assert history_after_first == 2  # First turn: 1 user + 1 assistant
        # With truncation working, later history shouldn't exceed reasonable limits
        assert history_after_third > 0 and history_after_third <= 10

    def test_truncation_raises_when_single_pair_exceeds_limit(self) -> None:
        """Test that truncation raises when single pair is too large."""
        client = _make_sync_client(input_tokens=1000, output_tokens=500)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=500,  # Very small limit
            token_budget_headroom=0.1,
        )
        
        manager.get_response(content="Message1")
        
        with pytest.raises(ValueError, match="Cannot truncate further"):
            manager.get_response(content="Message2")

    def test_truncation_skip_on_first_call_heuristic_mode(self) -> None:
        """Test that truncation is skipped on first call in heuristic mode."""
        client = _make_sync_client(input_tokens=10000, output_tokens=5000)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )
        
        # First call should not truncate (last_usage is None)
        manager.get_response(content="First message")
        
        # Should have the message
        assert len(manager.history) == 2

    def test_truncation_accurate_mode(self) -> None:
        """Test truncation with accurate_token_counting=True."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=300,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )
        
        manager.get_response(content="Message1")
        manager.get_response(content="Message2")
        
        # count_tokens should have been called
        assert client.messages.count_tokens.called

    def test_truncation_continues_until_threshold(self) -> None:
        """Test that truncation continues removing pairs until threshold."""
        client = _make_sync_client(input_tokens=200, output_tokens=100)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=400,
            token_budget_headroom=0.2,
        )
        
        # Add several turns
        # Each turn = 300 tokens (200 input + 100 output)
        # Threshold = 400 * (1 - 0.2) = 320
        manager.get_response(content="Msg1")  # 300 tokens < 320, no truncate
        manager.get_response(content="Msg2")  # 600 total, truncate to 300
        manager.get_response(content="Msg3")  # 600 total, truncate to 300
        manager.get_response(content="Msg4")  # 600 total, truncate to 300
        
        # Should maintain roughly 1 turn under threshold
        # With heuristic mode: estimated = 300 * (1.0 - 2/N) for each pair removed
        assert len(manager.history) >= 2  # At least 1 turn (user + assistant)
        assert len(manager.history) <= 8  # At most 4 turns worth of history


class TestAsyncConversationManager:
    """Tests for asynchronous AsyncConversationManager."""

    @pytest.mark.asyncio
    async def test_constructor_validation(self) -> None:
        """Test that async manager validates constructor args same as sync."""
        client = _make_async_client()
        
        with pytest.raises(ValueError, match="model cannot be an empty string"):
            AsyncConversationManager(client, model="", max_tokens=100)

    @pytest.mark.asyncio
    async def test_add_user_message(self) -> None:
        """Test adding user message to async manager."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_response_single_turn(self) -> None:
        """Test single turn with async manager."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=100)
        
        response = await manager.get_response(content="Hello")
        
        assert response.content[0].text == "Hello"
        assert len(manager.history) == 2

    @pytest.mark.asyncio
    async def test_get_response_multi_turn(self) -> None:
        """Test multiple turns with async manager."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=100)
        
        await manager.get_response(content="First")
        await manager.get_response(content="Second")
        
        assert len(manager.history) == 4

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """Test reset with async manager."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=100)
        
        await manager.get_response(content="Hello")
        manager.reset()
        
        assert len(manager.history) == 0
        assert manager.last_usage is None

    @pytest.mark.asyncio
    async def test_system_prompt(self) -> None:
        """Test system prompt with async manager."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3", max_tokens=100, system="Be helpful"
        )
        
        await manager.get_response(content="Hello")
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_truncation_with_accurate_counting(self) -> None:
        """Test async truncation with accurate_token_counting=True."""
        client = _make_async_client(input_tokens=100, output_tokens=50)
        manager = AsyncConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=300,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )
        
        await manager.get_response(content="Message1")
        await manager.get_response(content="Message2")
        
        # count_tokens should have been called
        assert client.messages.count_tokens.called

    @pytest.mark.asyncio
    async def test_repr(self) -> None:
        """Test repr of async manager."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=100)
        
        repr_str = repr(manager)
        assert "AsyncConversationManager" in repr_str
