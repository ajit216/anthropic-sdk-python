"""Tests for ConversationManager and AsyncConversationManager."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, Mock

from anthropic.helpers import ConversationManager, AsyncConversationManager


def _make_sync_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock synchronous Anthropic client."""
    client = MagicMock()
    content = Mock(text=content_text)
    usage = Mock(input_tokens=input_tokens, output_tokens=output_tokens)
    response = Mock(content=[content], usage=usage)
    client.messages.create.return_value = response
    count_response = Mock(input_tokens=input_tokens)
    client.messages.count_tokens.return_value = count_response
    return client


def _make_async_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> AsyncMock:
    """Create a mock asynchronous Anthropic client."""
    client = AsyncMock()
    content = Mock(text=content_text)
    usage = Mock(input_tokens=input_tokens, output_tokens=output_tokens)
    response = Mock(content=[content], usage=usage)
    client.messages.create.return_value = response
    count_response = Mock(input_tokens=input_tokens)
    client.messages.count_tokens.return_value = count_response
    return client


class TestConversationManager:
    """Tests for synchronous ConversationManager."""

    def test_constructor_validation_empty_model(self) -> None:
        """Test that empty model raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="model must not be empty"):
            ConversationManager(client, model="", max_tokens=100)

    def test_constructor_validation_zero_max_tokens(self) -> None:
        """Test that max_tokens < 1 raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3-5", max_tokens=0)

    def test_constructor_validation_negative_context_window(self) -> None:
        """Test that negative context_window_limit raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client, model="claude-3-5", max_tokens=100, context_window_limit=-1
            )

    def test_constructor_validation_invalid_headroom(self) -> None:
        """Test that token_budget_headroom outside [0.0, 1.0) raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client,
                model="claude-3-5",
                max_tokens=100,
                token_budget_headroom=1.0,
            )

    def test_constructor_valid(self) -> None:
        """Test that valid constructor arguments work."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3-5",
            max_tokens=100,
            system="You are helpful.",
            context_window_limit=2000,
            token_budget_headroom=0.1,
        )
        assert manager._model == "claude-3-5"
        assert manager._max_tokens == 100
        assert manager._system == "You are helpful."

    def test_add_user_message(self) -> None:
        """Test adding user message to history."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        manager.add_user_message("Hello")
        assert len(manager._history) == 1
        assert manager._history[0] == {"role": "user", "content": "Hello"}

    def test_add_user_message_empty_string_raises(self) -> None:
        """Test that empty string raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        with pytest.raises(ValueError, match="content must not be empty"):
            manager.add_user_message("")

    def test_add_user_message_empty_list_raises(self) -> None:
        """Test that empty list raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        with pytest.raises(ValueError, match="content must not be empty"):
            manager.add_user_message([])

    def test_get_response_no_staged_message_raises(self) -> None:
        """Test that get_response without staged user message raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        with pytest.raises(ValueError, match="no staged user message"):
            manager.get_response()

    def test_get_response_basic(self) -> None:
        """Test basic get_response flow."""
        client = _make_sync_client(content_text="Response from API")
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        manager.add_user_message("Hello")
        response = manager.get_response()

        assert response.content[0].text == "Response from API"
        assert len(manager._history) == 2
        assert manager._history[1]["role"] == "assistant"
        assert manager._last_usage is not None

    def test_get_response_with_content_arg(self) -> None:
        """Test get_response with content argument."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        response = manager.get_response("Hello")
        assert len(manager._history) == 2
        assert manager._history[0]["role"] == "user"
        assert manager._history[0]["content"] == "Hello"

    def test_get_response_multi_turn(self) -> None:
        """Test multi-turn conversation."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        manager.get_response("First question")
        manager.get_response("Second question")

        assert len(manager._history) == 4
        assert [m["role"] for m in manager._history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

    def test_last_usage_initially_none(self) -> None:
        """Test that last_usage is None before first response."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        assert manager.last_usage is None

    def test_last_usage_populated_after_response(self) -> None:
        """Test that last_usage is populated after get_response."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        manager.get_response("Hello")

        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100
        assert manager.last_usage.output_tokens == 50

    def test_get_response_forwards_kwargs(self) -> None:
        """Test that **kwargs are forwarded to client.messages.create."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        manager.get_response("Hello", temperature=0.5, top_p=0.9)

        client.messages.create.assert_called_once()
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    def test_get_response_includes_system_when_set(self) -> None:
        """Test that system prompt is included in API call when set."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5", max_tokens=100, system="You are helpful."
        )

        manager.get_response("Hello")

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."

    def test_get_response_excludes_system_when_none(self) -> None:
        """Test that system prompt is excluded when None."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        manager.get_response("Hello")

        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_history_returns_copy(self) -> None:
        """Test that history property returns a copy."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        manager.add_user_message("Hello")
        history = manager.history
        history.append({"role": "user", "content": "Extra"})

        assert len(manager._history) == 1
        assert len(history) == 2

    def test_reset(self) -> None:
        """Test reset clears history and last_usage."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5", max_tokens=100)

        manager.get_response("Hello")
        assert len(manager._history) == 2
        assert manager.last_usage is not None

        manager.reset()

        assert len(manager._history) == 0
        assert manager.last_usage is None

    def test_repr(self) -> None:
        """Test __repr__ method."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5", max_tokens=100, context_window_limit=2000
        )
        manager.add_user_message("Hello")

        repr_str = repr(manager)
        assert "ConversationManager" in repr_str
        assert "claude-3-5" in repr_str
        assert "2000" in repr_str

    def test_truncation_disabled_when_no_limit(self) -> None:
        """Test that truncation is skipped when context_window_limit is None."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client, model="claude-3-5", max_tokens=100, context_window_limit=None
        )

        manager.get_response("Message 1")
        manager.get_response("Message 2")

        assert len(manager._history) == 4

    def test_truncation_no_op_under_threshold(self) -> None:
        """Test that truncation is skipped when under threshold."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5",
            max_tokens=100,
            context_window_limit=500,
            token_budget_headroom=0.2,
        )

        manager.get_response("Message 1")
        initial_history_len = len(manager._history)

        manager.get_response("Message 2")

        assert len(manager._history) == initial_history_len + 2

    def test_truncation_drops_oldest_pair(self) -> None:
        """Test that truncation drops oldest user+assistant pair."""
        client = _make_sync_client(input_tokens=150, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5",
            max_tokens=100,
            context_window_limit=200,
            token_budget_headroom=0.0,
        )

        manager.get_response("Message 1")
        first_user_msg = manager._history[0]["content"]

        manager.get_response("Message 2")

        assert first_user_msg not in [m["content"] for m in manager._history]

    def test_truncation_multiple_pairs(self) -> None:
        """Test that truncation drops multiple pairs when needed."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5",
            max_tokens=100,
            context_window_limit=150,
            token_budget_headroom=0.0,
        )

        manager.get_response("Message 1")
        manager.get_response("Message 2")

        initial_len = len(manager._history)
        manager.get_response("Message 3")

        assert len(manager._history) < initial_len + 2

    def test_truncation_raises_when_single_pair_exceeds(self) -> None:
        """Test that truncation raises when single pair still exceeds limit."""
        client = _make_sync_client(input_tokens=500, output_tokens=500)
        manager = ConversationManager(
            client,
            model="claude-3-5",
            max_tokens=100,
            context_window_limit=200,
            token_budget_headroom=0.0,
        )

        manager.get_response("Message 1")

        with pytest.raises(ValueError, match="cannot truncate further"):
            manager.get_response("Message 2")

    def test_truncation_no_op_on_first_call_heuristic(self) -> None:
        """Test that truncation is skipped on first call in heuristic mode."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5",
            max_tokens=100,
            context_window_limit=100,
            token_budget_headroom=0.0,
            accurate_token_counting=False,
        )

        manager.add_user_message("Message 1")
        response = manager.get_response()

        assert len(manager._history) == 2
        assert response is not None

    def test_truncation_accurate_mode(self) -> None:
        """Test accurate truncation mode with count_tokens."""
        client = _make_sync_client(input_tokens=150, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5",
            max_tokens=100,
            context_window_limit=200,
            token_budget_headroom=0.0,
            accurate_token_counting=True,
        )

        manager.get_response("Message 1")
        client.messages.count_tokens.assert_called()

        manager.get_response("Message 2")
        assert client.messages.count_tokens.call_count > 1


class TestAsyncConversationManager:
    """Tests for asynchronous AsyncConversationManager."""

    @pytest.mark.asyncio
    async def test_constructor_validation_empty_model(self) -> None:
        """Test that empty model raises ValueError."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="model must not be empty"):
            AsyncConversationManager(client, model="", max_tokens=100)

    @pytest.mark.asyncio
    async def test_constructor_validation_zero_max_tokens(self) -> None:
        """Test that max_tokens < 1 raises ValueError."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            AsyncConversationManager(client, model="claude-3-5", max_tokens=0)

    @pytest.mark.asyncio
    async def test_constructor_valid(self) -> None:
        """Test that valid constructor arguments work."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client,
            model="claude-3-5",
            max_tokens=100,
            system="You are helpful.",
        )
        assert manager._model == "claude-3-5"

    @pytest.mark.asyncio
    async def test_add_user_message(self) -> None:
        """Test adding user message to history."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3-5", max_tokens=100)

        manager.add_user_message("Hello")
        assert len(manager._history) == 1
        assert manager._history[0] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_get_response_basic(self) -> None:
        """Test basic get_response flow."""
        client = _make_async_client(content_text="Response from API")
        manager = AsyncConversationManager(client, model="claude-3-5", max_tokens=100)

        response = await manager.get_response("Hello")

        assert response.content[0].text == "Response from API"
        assert len(manager._history) == 2
        assert manager._history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_response_multi_turn(self) -> None:
        """Test multi-turn conversation."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3-5", max_tokens=100)

        await manager.get_response("First question")
        await manager.get_response("Second question")

        assert len(manager._history) == 4
        assert [m["role"] for m in manager._history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

    @pytest.mark.asyncio
    async def test_last_usage_populated(self) -> None:
        """Test that last_usage is populated after response."""
        client = _make_async_client(input_tokens=100, output_tokens=50)
        manager = AsyncConversationManager(client, model="claude-3-5", max_tokens=100)

        await manager.get_response("Hello")

        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100

    @pytest.mark.asyncio
    async def test_get_response_forwards_kwargs(self) -> None:
        """Test that **kwargs are forwarded to client.messages.create."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3-5", max_tokens=100)

        await manager.get_response("Hello", temperature=0.5)

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_get_response_includes_system(self) -> None:
        """Test that system prompt is included when set."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5", max_tokens=100, system="You are helpful."
        )

        await manager.get_response("Hello")

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_history_returns_copy(self) -> None:
        """Test that history property returns a copy."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3-5", max_tokens=100)

        manager.add_user_message("Hello")
        history = manager.history
        history.append({"role": "user", "content": "Extra"})

        assert len(manager._history) == 1
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """Test reset clears history and last_usage."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude-3-5", max_tokens=100)

        await manager.get_response("Hello")
        manager.reset()

        assert len(manager._history) == 0
        assert manager.last_usage is None

    @pytest.mark.asyncio
    async def test_repr(self) -> None:
        """Test __repr__ method."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5", max_tokens=100, context_window_limit=2000
        )

        repr_str = repr(manager)
        assert "AsyncConversationManager" in repr_str
        assert "claude-3-5" in repr_str

    @pytest.mark.asyncio
    async def test_truncation_accurate_mode(self) -> None:
        """Test accurate truncation mode with count_tokens."""
        client = _make_async_client(input_tokens=150, output_tokens=50)
        manager = AsyncConversationManager(
            client,
            model="claude-3-5",
            max_tokens=100,
            context_window_limit=200,
            token_budget_headroom=0.0,
            accurate_token_counting=True,
        )

        await manager.get_response("Message 1")
        client.messages.count_tokens.assert_called()
