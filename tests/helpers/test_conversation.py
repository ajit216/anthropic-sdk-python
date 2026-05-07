"""Tests for ConversationManager and AsyncConversationManager."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock

from anthropic.helpers import ConversationManager, AsyncConversationManager


def _make_sync_client(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    content_text: str = "Hello",
) -> MagicMock:
    """Create a mock sync client for testing."""
    mock_client = MagicMock()
    mock_usage = MagicMock()
    mock_usage.input_tokens = input_tokens
    mock_usage.output_tokens = output_tokens

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=content_text)]
    mock_response.usage = mock_usage

    mock_client.messages.create.return_value = mock_response

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
    mock_usage = MagicMock()
    mock_usage.input_tokens = input_tokens
    mock_usage.output_tokens = output_tokens

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=content_text)]
    mock_response.usage = mock_usage

    mock_client.messages.create = AsyncMock(return_value=mock_response)
    mock_client.messages.count_tokens = AsyncMock()

    mock_count_response = MagicMock()
    mock_count_response.input_tokens = input_tokens
    mock_client.messages.count_tokens.return_value = mock_count_response

    return mock_client


class TestConversationManager:
    """Tests for sync ConversationManager."""

    def test_constructor_empty_model_raises(self) -> None:
        """Constructor should raise ValueError for empty model."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="model cannot be empty"):
            ConversationManager(client=client, model="", max_tokens=100)

    def test_constructor_zero_max_tokens_raises(self) -> None:
        """Constructor should raise ValueError for zero max_tokens."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client=client, model="claude-3-5-sonnet", max_tokens=0)

    def test_constructor_negative_context_limit_raises(self) -> None:
        """Constructor should raise ValueError for negative context_window_limit."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client=client,
                model="claude-3-5-sonnet",
                max_tokens=100,
                context_window_limit=-1,
            )

    def test_constructor_invalid_headroom_raises(self) -> None:
        """Constructor should raise ValueError for invalid token_budget_headroom."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client=client,
                model="claude-3-5-sonnet",
                max_tokens=100,
                token_budget_headroom=1.5,
            )

    def test_add_user_message_single_turn(self) -> None:
        """Should append user message to history."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "Hello"

    def test_add_user_message_empty_raises(self) -> None:
        """Should raise ValueError for empty content."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_user_message("")

    def test_get_response_basic(self) -> None:
        """Should call API and append assistant response."""
        client = _make_sync_client(content_text="Response")
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        manager.add_user_message("Hello")
        response = manager.get_response()

        assert response is not None
        assert len(manager.history) == 2
        assert manager.history[1]["role"] == "assistant"
        client.messages.create.assert_called_once()

    def test_get_response_with_content(self) -> None:
        """Should add content as user message before getting response."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        response = manager.get_response(content="Hello")

        assert len(manager.history) == 2
        assert manager.history[0]["content"] == "Hello"
        assert manager.history[1]["role"] == "assistant"

    def test_get_response_no_user_message_raises(self) -> None:
        """Should raise ValueError if no user message to respond to."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        with pytest.raises(ValueError, match="No user message to respond to"):
            manager.get_response()

    def test_get_response_multi_turn(self) -> None:
        """Should maintain history across multiple turns."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        manager.get_response(content="First")
        manager.get_response(content="Second")

        assert len(manager.history) == 4
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"
        assert manager.history[2]["role"] == "user"
        assert manager.history[3]["role"] == "assistant"

    def test_last_usage_none_initially(self) -> None:
        """last_usage should be None before first response."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        assert manager.last_usage is None

    def test_last_usage_populated_after_response(self) -> None:
        """last_usage should be populated after getting response."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        manager.get_response(content="Hello")
        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100
        assert manager.last_usage.output_tokens == 50

    def test_kwargs_forwarded_to_api(self) -> None:
        """Should forward kwargs to messages.create."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        manager.get_response(content="Hello", temperature=0.5)

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    def test_system_prompt_included(self) -> None:
        """Should include system prompt when set."""
        client = _make_sync_client()
        system = "You are helpful"
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            system=system,
        )
        manager.get_response(content="Hello")

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == system

    def test_system_prompt_omitted_when_none(self) -> None:
        """Should not include system in kwargs when None."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            system=None,
        )
        manager.get_response(content="Hello")

        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_history_returns_copy(self) -> None:
        """history property should return a copy."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        manager.add_user_message("Hello")

        history1 = manager.history
        history1.append({"role": "test", "content": "mutated"})

        history2 = manager.history
        assert len(history2) == 1

    def test_reset_clears_history(self) -> None:
        """reset() should clear history and last_usage."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        manager.get_response(content="Hello")
        assert len(manager.history) == 2
        assert manager.last_usage is not None

        manager.reset()
        assert len(manager.history) == 0
        assert manager.last_usage is None

    def test_reset_preserves_model_and_system(self) -> None:
        """reset() should preserve model and system settings."""
        client = _make_sync_client()
        system = "You are helpful"
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            system=system,
        )
        manager.add_user_message("Hello")
        manager.reset()

        manager.get_response(content="Hi")
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == system

    def test_truncation_noop_when_limit_none(self) -> None:
        """Should not truncate when context_window_limit is None."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=None,
        )
        # Add many messages
        for _ in range(10):
            manager.get_response(content="Message")

        assert len(manager.history) == 20

    def test_truncation_noop_when_under_threshold(self) -> None:
        """Should not truncate when under threshold."""
        client = _make_sync_client(input_tokens=10, output_tokens=5)
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.1,
        )
        manager.get_response(content="Hello")

        assert len(manager.history) == 2

    def test_truncation_drops_oldest_pair(self) -> None:
        """Should drop oldest user-assistant pair when over threshold."""
        client = _make_sync_client(input_tokens=600, output_tokens=400)
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.1,
        )

        manager.get_response(content="First")
        first_message = manager.history[0]["content"]
        manager.get_response(content="Second")

        assert manager.history[0]["content"] != first_message

    def test_truncation_raises_when_single_pair_exceeds(self) -> None:
        """Should raise ValueError when single pair exceeds limit."""
        client = _make_sync_client(input_tokens=2000, output_tokens=1000)
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.1,
        )
        manager.get_response(content="Hello")

        with pytest.raises(ValueError, match="Cannot truncate further"):
            manager.get_response(content="World")

    def test_truncation_no_op_on_first_call_heuristic(self) -> None:
        """Should skip truncation on first call in heuristic mode."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=100,
            accurate_token_counting=False,
        )
        manager.get_response(content="Hello")

        assert len(manager.history) == 2

    def test_accurate_token_counting(self) -> None:
        """Should use count_tokens when accurate_token_counting=True."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=200,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )
        manager.get_response(content="Hello")
        manager.get_response(content="World")

        assert client.messages.count_tokens.called

    def test_repr(self) -> None:
        """Should return informative string representation."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=200000,
        )
        repr_str = repr(manager)
        assert "ConversationManager" in repr_str
        assert "claude-3-5-sonnet" in repr_str
        assert "200000" in repr_str

        manager.get_response(content="Hello")
        repr_str = repr(manager)
        assert "turns=1" in repr_str


class TestAsyncConversationManager:
    """Tests for async AsyncConversationManager."""

    @pytest.mark.asyncio
    async def test_constructor_validation(self) -> None:
        """Constructor should validate inputs."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="model cannot be empty"):
            AsyncConversationManager(client=client, model="", max_tokens=100)

    @pytest.mark.asyncio
    async def test_add_user_message(self) -> None:
        """Should append user message to history."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_response_basic(self) -> None:
        """Should call API and append response."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        response = await manager.get_response(content="Hello")

        assert response is not None
        assert len(manager.history) == 2
        assert manager.history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_response_no_user_message_raises(self) -> None:
        """Should raise ValueError if no user message."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        with pytest.raises(ValueError, match="No user message to respond to"):
            await manager.get_response()

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self) -> None:
        """Should handle multi-turn conversations."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        await manager.get_response(content="First")
        await manager.get_response(content="Second")

        assert len(manager.history) == 4

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        """Should reset history and usage."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
        )
        await manager.get_response(content="Hello")
        assert manager.last_usage is not None

        manager.reset()
        assert len(manager.history) == 0
        assert manager.last_usage is None

    @pytest.mark.asyncio
    async def test_system_prompt(self) -> None:
        """Should include system prompt."""
        client = _make_async_client()
        system = "You are helpful"
        manager = AsyncConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            system=system,
        )
        await manager.get_response(content="Hello")

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == system

    @pytest.mark.asyncio
    async def test_repr(self) -> None:
        """Should return informative string representation."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=200000,
        )
        repr_str = repr(manager)
        assert "AsyncConversationManager" in repr_str
        assert "claude-3-5-sonnet" in repr_str
        assert "200000" in repr_str

        await manager.get_response(content="Hello")
        repr_str = repr(manager)
        assert "turns=1" in repr_str
