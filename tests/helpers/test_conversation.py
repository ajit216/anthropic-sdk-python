"""Tests for ConversationManager and AsyncConversationManager."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from anthropic.helpers import ConversationManager, AsyncConversationManager


def _make_sync_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock sync Anthropic client."""
    client = MagicMock()
    
    # Mock content response
    content_mock = MagicMock()
    content_mock.text = content_text
    
    # Mock response
    response_mock = MagicMock()
    response_mock.content = [content_mock]
    
    # Mock usage
    usage_mock = MagicMock()
    usage_mock.input_tokens = input_tokens
    usage_mock.output_tokens = output_tokens
    response_mock.usage = usage_mock
    
    client.messages.create.return_value = response_mock
    
    # Mock count_tokens
    count_mock = MagicMock()
    count_mock.input_tokens = input_tokens
    client.messages.count_tokens.return_value = count_mock
    
    return client


def _make_async_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock async Anthropic client."""
    client = MagicMock()
    
    # Mock content response
    content_mock = MagicMock()
    content_mock.text = content_text
    
    # Mock response
    response_mock = MagicMock()
    response_mock.content = [content_mock]
    
    # Mock usage
    usage_mock = MagicMock()
    usage_mock.input_tokens = input_tokens
    usage_mock.output_tokens = output_tokens
    response_mock.usage = usage_mock
    
    client.messages.create = AsyncMock(return_value=response_mock)
    
    # Mock count_tokens
    count_mock = MagicMock()
    count_mock.input_tokens = input_tokens
    client.messages.count_tokens = AsyncMock(return_value=count_mock)
    
    return client


class TestConversationManager:
    """Test suite for ConversationManager."""

    def test_constructor_validation_empty_model(self) -> None:
        """Constructor raises ValueError for empty model."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="model must not be empty"):
            ConversationManager(client, model="", max_tokens=100)

    def test_constructor_validation_zero_max_tokens(self) -> None:
        """Constructor raises ValueError for zero max_tokens."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3", max_tokens=0)

    def test_constructor_validation_negative_context_limit(self) -> None:
        """Constructor raises ValueError for negative context_window_limit."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client,
                model="claude-3",
                max_tokens=100,
                context_window_limit=-1,
            )

    def test_constructor_validation_invalid_headroom(self) -> None:
        """Constructor raises ValueError for invalid token_budget_headroom."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client,
                model="claude-3",
                max_tokens=100,
                token_budget_headroom=1.5,
            )

    def test_add_user_message_string(self) -> None:
        """add_user_message appends string content."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "Hello"

    def test_add_user_message_empty_string_raises(self) -> None:
        """add_user_message raises on empty string."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        with pytest.raises(ValueError, match="content must not be empty"):
            manager.add_user_message("")

    def test_add_user_message_empty_list_raises(self) -> None:
        """add_user_message raises on empty list."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        with pytest.raises(ValueError, match="content must not be empty"):
            manager.add_user_message([])

    def test_get_response_single_turn(self) -> None:
        """get_response makes API call and appends response."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hi")
        response = manager.get_response()
        
        # Check API was called
        client.messages.create.assert_called_once()
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3"
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        
        # Check response appended to history
        assert len(manager.history) == 2
        assert manager.history[1]["role"] == "assistant"

    def test_get_response_with_content_arg(self) -> None:
        """get_response with content arg adds message then calls API."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        response = manager.get_response("Hi there")
        
        # Check user message was added
        assert len(manager.history) == 2
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "Hi there"
        
        # Check response was added
        assert manager.history[1]["role"] == "assistant"

    def test_get_response_no_staged_message_raises(self) -> None:
        """get_response without staged message raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        with pytest.raises(ValueError, match="No staged user message"):
            manager.get_response()

    def test_get_response_with_system_prompt(self) -> None:
        """System prompt passed to API when set."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            system="You are helpful",
        )
        
        manager.add_user_message("Hi")
        manager.get_response()
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"

    def test_get_response_system_prompt_omitted_when_none(self) -> None:
        """System prompt omitted from API call when None."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hi")
        manager.get_response()
        
        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_get_response_forwards_kwargs(self) -> None:
        """Additional kwargs forwarded to messages.create()."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hi")
        manager.get_response(temperature=0.5, top_p=0.9)
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    def test_multi_turn_conversation(self) -> None:
        """Multiple turns maintain correct message history."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        # First turn
        manager.add_user_message("Hi")
        manager.get_response()
        assert len(manager.history) == 2
        
        # Second turn
        manager.add_user_message("How are you?")
        manager.get_response()
        assert len(manager.history) == 4
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"
        assert manager.history[2]["role"] == "user"
        assert manager.history[3]["role"] == "assistant"

    def test_last_usage_none_initially(self) -> None:
        """last_usage is None before first API call."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        assert manager.last_usage is None

    def test_last_usage_populated_after_response(self) -> None:
        """last_usage populated after get_response()."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hi")
        manager.get_response()
        
        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100
        assert manager.last_usage.output_tokens == 50

    def test_history_returns_copy(self) -> None:
        """history property returns a shallow copy."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hi")
        history1 = manager.history
        history1.pop(0)  # Modify the copy
        
        # Original history unchanged
        assert len(manager.history) == 1

    def test_reset_clears_history_and_usage(self) -> None:
        """reset() clears history and last_usage."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hi")
        manager.get_response()
        
        assert len(manager.history) > 0
        assert manager.last_usage is not None
        
        manager.reset()
        
        assert len(manager.history) == 0
        assert manager.last_usage is None

    def test_truncation_disabled_when_no_limit(self) -> None:
        """Truncation does not occur when context_window_limit=None."""
        client = _make_sync_client(input_tokens=1000, output_tokens=1000)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=None,  # No limit
        )
        
        # Add messages that would exceed threshold
        for i in range(10):
            manager.add_user_message(f"Message {i}")
            manager.get_response()
        
        # All messages should still be in history
        assert len(manager.history) == 20

    def test_truncation_no_op_under_threshold(self) -> None:
        """No truncation when tokens under threshold."""
        client = _make_sync_client(input_tokens=50, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.1,
        )
        
        manager.add_user_message("Hi")
        manager.get_response()
        
        # History should be intact (threshold is 900, usage is 100)
        assert len(manager.history) == 2

    def test_truncation_drops_oldest_pair(self) -> None:
        """Truncation removes oldest user+assistant pair."""
        client = _make_sync_client(input_tokens=600, output_tokens=600)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )
        
        # First conversation
        manager.add_user_message("Message 1")
        manager.get_response()
        assert len(manager.history) == 2
        
        # Second conversation - should trigger truncation
        manager.add_user_message("Message 2")
        manager.get_response()
        
        # First pair should be removed (oldest pair dropped)
        history = manager.history
        assert len(history) == 2
        assert "Message 2" in str(history[-2]["content"])

    def test_truncation_no_op_on_first_call(self) -> None:
        """Heuristic truncation skipped on first call (no last_usage)."""
        client = _make_sync_client(input_tokens=600, output_tokens=600)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )
        
        manager.add_user_message("Hi")
        manager.get_response()
        
        # Should be 2 messages despite high token estimate
        assert len(manager.history) == 2

    def test_truncation_accurate_mode(self) -> None:
        """Accurate token counting triggers truncation correctly."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )
        
        # Setup: count_tokens returns low value for first call
        count_response = MagicMock()
        count_response.input_tokens = 100
        client.messages.count_tokens.return_value = count_response
        
        # First turn
        manager.add_user_message("Message 1")
        manager.get_response()
        assert len(manager.history) == 2
        assert client.messages.count_tokens.called
        
        # Reset call count to track second call
        client.messages.count_tokens.reset_mock()
        
        # Second turn with low token count should work without truncation
        manager.add_user_message("Message 2")
        manager.get_response()
        
        # count_tokens should be called in truncation check
        assert client.messages.count_tokens.called

    def test_truncation_raises_when_single_pair_exceeds_limit(self) -> None:
        """Truncation raises ValueError when single pair exceeds limit."""
        client = _make_sync_client(input_tokens=1500, output_tokens=1500)
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )
        
        # First turn establishes baseline
        manager.add_user_message("Hi")
        manager.get_response()
        
        # Second turn - token estimate so high that even single pair exceeds threshold
        with pytest.raises(ValueError, match="Cannot truncate further"):
            manager.add_user_message("Message 2")
            manager.get_response()

    def test_repr(self) -> None:
        """__repr__ shows model and turn count."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
        )
        
        repr_str = repr(manager)
        assert "claude-3" in repr_str
        assert "turns=0" in repr_str
        
        manager.add_user_message("Hi")
        manager.get_response()
        
        repr_str = repr(manager)
        assert "turns=1" in repr_str


class TestAsyncConversationManager:
    """Test suite for AsyncConversationManager."""

    @pytest.mark.asyncio
    async def test_constructor_validation_empty_model(self) -> None:
        """Constructor raises ValueError for empty model."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="model must not be empty"):
            AsyncConversationManager(client, model="", max_tokens=100)

    @pytest.mark.asyncio
    async def test_add_user_message_string(self) -> None:
        """add_user_message appends string content."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_response_single_turn(self) -> None:
        """get_response makes async API call and appends response."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hi")
        response = await manager.get_response()
        
        # Check API was called
        client.messages.create.assert_called_once()
        
        # Check response appended
        assert len(manager.history) == 2
        assert manager.history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_response_with_content_arg(self) -> None:
        """get_response with content arg adds message then calls API."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=100)
        
        await manager.get_response("Hi there")
        
        # Check user message was added and response received
        assert len(manager.history) == 2
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_response_no_staged_message_raises(self) -> None:
        """get_response without staged message raises ValueError."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=100)
        
        with pytest.raises(ValueError, match="No staged user message"):
            await manager.get_response()

    @pytest.mark.asyncio
    async def test_get_response_with_system_prompt(self) -> None:
        """System prompt passed to API when set."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            system="You are helpful",
        )
        
        manager.add_user_message("Hi")
        await manager.get_response()
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self) -> None:
        """Multiple async turns maintain correct history."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hi")
        await manager.get_response()
        
        manager.add_user_message("How are you?")
        await manager.get_response()
        
        assert len(manager.history) == 4
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"
        assert manager.history[2]["role"] == "user"
        assert manager.history[3]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_last_usage_populated_after_response(self) -> None:
        """last_usage populated after get_response()."""
        client = _make_async_client(input_tokens=100, output_tokens=50)
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hi")
        await manager.get_response()
        
        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100

    @pytest.mark.asyncio
    async def test_reset_clears_history_and_usage(self) -> None:
        """reset() clears history and last_usage."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3", max_tokens=100)
        
        manager.add_user_message("Hi")
        await manager.get_response()
        
        manager.reset()
        
        assert len(manager.history) == 0
        assert manager.last_usage is None

    @pytest.mark.asyncio
    async def test_truncation_accurate_mode(self) -> None:
        """Accurate token counting in async mode."""
        client = _make_async_client(input_tokens=100, output_tokens=50)
        manager = AsyncConversationManager(
            client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )
        
        # Setup: count_tokens returns low value for first call
        count_response = MagicMock()
        count_response.input_tokens = 100
        client.messages.count_tokens = AsyncMock(return_value=count_response)
        
        manager.add_user_message("Message 1")
        await manager.get_response()
        assert client.messages.count_tokens.called
        
        # Reset call count to track second call
        client.messages.count_tokens.reset_mock()
        
        # Second turn with low token count should work without truncation
        manager.add_user_message("Message 2")
        await manager.get_response()
        
        # count_tokens should be called in truncation check
        assert client.messages.count_tokens.called
