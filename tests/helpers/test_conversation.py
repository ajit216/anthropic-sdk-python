"""Tests for ConversationManager and AsyncConversationManager."""

import pytest
from unittest.mock import MagicMock, AsyncMock, call

from anthropic.helpers import ConversationManager, AsyncConversationManager


def _make_sync_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock sync client for testing."""
    client = MagicMock()

    # Mock the response
    response = MagicMock()
    response.content = [MagicMock(text=content_text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)

    client.messages.create.return_value = response

    # Mock count_tokens
    count_response = MagicMock()
    count_response.input_tokens = input_tokens + 50  # Simulate some overhead
    client.messages.count_tokens.return_value = count_response

    return client


def _make_async_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> AsyncMock:
    """Create a mock async client for testing."""
    client = AsyncMock()

    # Mock the response
    response = MagicMock()
    response.content = [MagicMock(text=content_text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)

    client.messages.create = AsyncMock(return_value=response)

    # Mock count_tokens
    count_response = MagicMock()
    count_response.input_tokens = input_tokens + 50
    client.messages.count_tokens = AsyncMock(return_value=count_response)

    return client


class TestConversationManager:
    """Tests for synchronous ConversationManager."""

    def test_constructor_validates_model(self) -> None:
        """Test that constructor validates model."""
        client = _make_sync_client()

        with pytest.raises(ValueError, match="model cannot be empty"):
            ConversationManager(client=client, model="", max_tokens=512)

    def test_constructor_validates_max_tokens(self) -> None:
        """Test that constructor validates max_tokens."""
        client = _make_sync_client()

        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client=client, model="claude-3-5-sonnet-20241022", max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client=client, model="claude-3-5-sonnet-20241022", max_tokens=-1)

    def test_constructor_validates_context_window_limit(self) -> None:
        """Test that constructor validates context_window_limit."""
        client = _make_sync_client()

        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client=client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=512,
                context_window_limit=0,
            )

    def test_constructor_validates_token_budget_headroom(self) -> None:
        """Test that constructor validates token_budget_headroom."""
        client = _make_sync_client()

        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client=client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=512,
                token_budget_headroom=-0.1,
            )

        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client=client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=512,
                token_budget_headroom=1.0,
            )

    def test_add_user_message_with_string(self) -> None:
        """Test adding a user message as a string."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "Hello"

    def test_add_user_message_validates_empty_string(self) -> None:
        """Test that adding empty string raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_user_message("")

    def test_add_user_message_validates_empty_list(self) -> None:
        """Test that adding empty list raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_user_message([])

    def test_get_response_calls_api(self) -> None:
        """Test that get_response calls the API."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        manager.add_user_message("Hello")
        response = manager.get_response()

        assert response is not None
        client.messages.create.assert_called_once()
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert call_kwargs["max_tokens"] == 512

    def test_get_response_with_content(self) -> None:
        """Test get_response with content parameter."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        response = manager.get_response("Hello")

        assert len(manager.history) >= 1
        assert manager.history[-2]["role"] == "user"
        assert manager.history[-1]["role"] == "assistant"

    def test_get_response_requires_user_message(self) -> None:
        """Test that get_response requires a user message."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        with pytest.raises(ValueError, match="No user message staged"):
            manager.get_response()

    def test_get_response_appends_assistant_message(self) -> None:
        """Test that get_response appends assistant message to history."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        manager.add_user_message("Hello")
        manager.get_response()

        assert len(manager.history) == 2
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"

    def test_get_response_stores_usage(self) -> None:
        """Test that get_response stores usage information."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        assert manager.last_usage is None

        manager.add_user_message("Hello")
        manager.get_response()

        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100
        assert manager.last_usage.output_tokens == 50

    def test_get_response_forwards_kwargs(self) -> None:
        """Test that get_response forwards kwargs to messages.create."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        manager.add_user_message("Hello")
        manager.get_response(temperature=0.5, top_p=0.9)

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    def test_system_prompt_included_when_set(self) -> None:
        """Test that system prompt is included in API call when set."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            system="You are helpful.",
        )

        manager.add_user_message("Hello")
        manager.get_response()

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."

    def test_system_prompt_omitted_when_none(self) -> None:
        """Test that system prompt is omitted when None."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            system=None,
        )

        manager.add_user_message("Hello")
        manager.get_response()

        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_multi_turn_conversation(self) -> None:
        """Test a multi-turn conversation."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        manager.add_user_message("First question")
        manager.get_response()

        manager.add_user_message("Second question")
        manager.get_response()

        assert len(manager.history) == 4
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"
        assert manager.history[2]["role"] == "user"
        assert manager.history[3]["role"] == "assistant"

    def test_history_returns_copy(self) -> None:
        """Test that history property returns a copy."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        manager.add_user_message("Hello")
        history = manager.history
        history.append({"role": "fake", "content": "should not affect manager"})

        assert len(manager.history) == 1
        assert len(history) == 2

    def test_reset(self) -> None:
        """Test that reset clears history and usage."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        manager.add_user_message("Hello")
        manager.get_response()

        assert len(manager.history) > 0
        assert manager.last_usage is not None

        manager.reset()

        assert len(manager.history) == 0
        assert manager.last_usage is None

    def test_truncation_no_op_when_limit_none(self) -> None:
        """Test that truncation is skipped when context_window_limit is None."""
        client = _make_sync_client(input_tokens=1000000, output_tokens=500000)
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            context_window_limit=None,
        )

        manager.add_user_message("Hello")
        # Should not raise even though usage is huge
        manager.get_response()

        assert len(manager.history) == 2

    def test_truncation_no_op_when_under_threshold(self) -> None:
        """Test that truncation is skipped when under threshold."""
        client = _make_sync_client(input_tokens=10, output_tokens=5)
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            context_window_limit=1000,
            token_budget_headroom=0.1,
        )

        manager.add_user_message("Hello")
        manager.get_response()

        assert len(manager.history) == 2

    def test_truncation_drops_oldest_pair(self) -> None:
        """Test that truncation drops oldest user+assistant pair."""
        client = _make_sync_client(input_tokens=900, output_tokens=100)
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            context_window_limit=1000,
            token_budget_headroom=0.1,
        )

        # Add first exchange
        manager.add_user_message("First")
        manager.get_response()

        assert len(manager.history) == 2

        # Add second exchange - this should trigger truncation
        manager.add_user_message("Second")
        manager.get_response()

        # Should have truncated the first pair
        assert len(manager.history) == 2
        assert manager.history[0]["role"] == "user"
        assert "Second" in str(manager.history[0]["content"])

    def test_truncation_raises_when_single_pair_exceeds(self) -> None:
        """Test that truncation raises ValueError when single pair exceeds limit."""
        client = _make_sync_client(input_tokens=900, output_tokens=100)
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            context_window_limit=1000,
            token_budget_headroom=0.1,
        )

        manager.add_user_message("Hello")
        manager.get_response()

        # Try to add another turn - should fail because can't truncate further
        manager.add_user_message("Another question")

        with pytest.raises(ValueError, match="Cannot truncate further"):
            manager.get_response()

    def test_truncation_skipped_on_first_call(self) -> None:
        """Test that truncation is skipped on first call (last_usage is None)."""
        client = _make_sync_client(input_tokens=900, output_tokens=100)
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )

        manager.add_user_message("Hello")
        # Should not raise even though usage is high
        manager.get_response()

        assert len(manager.history) == 2

    def test_accurate_token_counting(self) -> None:
        """Test that accurate token counting uses count_tokens API."""
        client = _make_sync_client(input_tokens=900, output_tokens=100)
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )

        manager.add_user_message("Hello")
        manager.get_response()

        # count_tokens should have been called
        client.messages.count_tokens.assert_called()

    def test_repr(self) -> None:
        """Test the string representation."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            context_window_limit=200000,
        )

        repr_str = repr(manager)
        assert "ConversationManager" in repr_str
        assert "claude-3-5-sonnet-20241022" in repr_str
        assert "limit=200000" in repr_str


class TestAsyncConversationManager:
    """Tests for asynchronous AsyncConversationManager."""

    @pytest.mark.asyncio
    async def test_constructor_validates_model(self) -> None:
        """Test that constructor validates model."""
        client = _make_async_client()

        with pytest.raises(ValueError, match="model cannot be empty"):
            AsyncConversationManager(client=client, model="", max_tokens=512)

    @pytest.mark.asyncio
    async def test_add_user_message_with_string(self) -> None:
        """Test adding a user message as a string."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_response_calls_api(self) -> None:
        """Test that get_response calls the API."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        manager.add_user_message("Hello")
        response = await manager.get_response()

        assert response is not None
        client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_response_appends_assistant_message(self) -> None:
        """Test that get_response appends assistant message to history."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        manager.add_user_message("Hello")
        await manager.get_response()

        assert len(manager.history) == 2
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self) -> None:
        """Test a multi-turn conversation."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        manager.add_user_message("First")
        await manager.get_response()

        manager.add_user_message("Second")
        await manager.get_response()

        assert len(manager.history) == 4

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """Test that reset clears history and usage."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client, model="claude-3-5-sonnet-20241022", max_tokens=512
        )

        manager.add_user_message("Hello")
        await manager.get_response()

        assert len(manager.history) > 0
        assert manager.last_usage is not None

        manager.reset()

        assert len(manager.history) == 0
        assert manager.last_usage is None

    @pytest.mark.asyncio
    async def test_truncation_with_accurate_counting(self) -> None:
        """Test truncation with accurate token counting."""
        client = _make_async_client(input_tokens=900, output_tokens=100)
        manager = AsyncConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )

        manager.add_user_message("Hello")
        await manager.get_response()

        # count_tokens should have been called
        client.messages.count_tokens.assert_called()

    @pytest.mark.asyncio
    async def test_repr(self) -> None:
        """Test the string representation."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=512,
            context_window_limit=200000,
        )

        repr_str = repr(manager)
        assert "AsyncConversationManager" in repr_str
        assert "claude-3-5-sonnet-20241022" in repr_str
