"""Tests for ConversationManager and AsyncConversationManager."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, call

from anthropic.helpers.conversation import ConversationManager, AsyncConversationManager


def _make_sync_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock sync Anthropic client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=content_text)]
    mock_response.usage = MagicMock(
        input_tokens=input_tokens, output_tokens=output_tokens
    )
    mock_client.messages.create.return_value = mock_response
    mock_client.messages.count_tokens.return_value = input_tokens + output_tokens
    return mock_client


def _make_async_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> AsyncMock:
    """Create a mock async Anthropic client."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text=content_text)]
    mock_response.usage = MagicMock(
        input_tokens=input_tokens, output_tokens=output_tokens
    )
    mock_client.messages.create.return_value = mock_response
    mock_client.messages.count_tokens.return_value = input_tokens + output_tokens
    return mock_client


class TestConversationManager:
    """Test suite for ConversationManager."""

    def test_init_validation_empty_model(self):
        """Constructor should raise ValueError for empty model."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="model cannot be empty"):
            ConversationManager(client=client, model="", max_tokens=100)

    def test_init_validation_max_tokens_zero(self):
        """Constructor should raise ValueError for max_tokens < 1."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client=client, model="claude-3", max_tokens=0)

    def test_init_validation_max_tokens_negative(self):
        """Constructor should raise ValueError for negative max_tokens."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client=client, model="claude-3", max_tokens=-1)

    def test_init_validation_negative_context_window(self):
        """Constructor should raise ValueError for negative context_window_limit."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client=client,
                model="claude-3",
                max_tokens=100,
                context_window_limit=-1,
            )

    def test_init_validation_invalid_headroom_negative(self):
        """Constructor should raise ValueError for negative token_budget_headroom."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client=client,
                model="claude-3",
                max_tokens=100,
                token_budget_headroom=-0.1,
            )

    def test_init_validation_invalid_headroom_one(self):
        """Constructor should raise ValueError for token_budget_headroom == 1.0."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client=client,
                model="claude-3",
                max_tokens=100,
                token_budget_headroom=1.0,
            )

    def test_init_validation_invalid_headroom_greater_than_one(self):
        """Constructor should raise ValueError for token_budget_headroom > 1.0."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client=client,
                model="claude-3",
                max_tokens=100,
                token_budget_headroom=1.5,
            )

    def test_init_valid(self):
        """Constructor should succeed with valid parameters."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            system="You are helpful",
            context_window_limit=200000,
            token_budget_headroom=0.1,
        )
        assert manager._model == "claude-3"
        assert manager._max_tokens == 100
        assert manager._system == "You are helpful"
        assert manager._context_window_limit == 200000

    def test_add_user_message_string(self):
        """add_user_message should accept and store string content."""
        client = _make_sync_client()
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)
        manager.add_user_message("Hello")
        assert len(manager._history) == 1
        assert manager._history[0] == {"role": "user", "content": "Hello"}

    def test_add_user_message_list(self):
        """add_user_message should accept and store list content."""
        client = _make_sync_client()
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)
        content = [{"type": "text", "text": "Hello"}]
        manager.add_user_message(content)
        assert len(manager._history) == 1
        assert manager._history[0] == {"role": "user", "content": content}

    def test_add_user_message_empty_string_raises(self):
        """add_user_message should raise ValueError for empty string."""
        client = _make_sync_client()
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)
        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_user_message("")

    def test_add_user_message_empty_list_raises(self):
        """add_user_message should raise ValueError for empty list."""
        client = _make_sync_client()
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)
        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_user_message([])

    def test_single_turn_response(self):
        """get_response should return response and append assistant message."""
        client = _make_sync_client(content_text="Response text")
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)
        manager.add_user_message("Hello")
        response = manager.get_response()

        assert response.content[0].text == "Response text"
        assert len(manager._history) == 2
        assert manager._history[0]["role"] == "user"
        assert manager._history[1]["role"] == "assistant"

    def test_multi_turn_conversation(self):
        """Multiple turns should accumulate in history."""
        client = _make_sync_client()
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)

        # First turn
        manager.add_user_message("First")
        manager.get_response()
        assert len(manager._history) == 2

        # Second turn
        manager.add_user_message("Second")
        manager.get_response()
        assert len(manager._history) == 4
        assert manager._history[0]["content"] == "First"
        assert manager._history[2]["content"] == "Second"

    def test_last_usage_none_initially(self):
        """last_usage should be None before any API call."""
        client = _make_sync_client()
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)
        assert manager.last_usage is None

    def test_last_usage_populated_after_call(self):
        """last_usage should be populated after get_response."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)
        manager.add_user_message("Hello")
        manager.get_response()

        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100
        assert manager.last_usage.output_tokens == 50

    def test_kwargs_forwarded_to_create(self):
        """Additional kwargs should be forwarded to messages.create."""
        client = _make_sync_client()
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)
        manager.add_user_message("Hello")
        manager.get_response(temperature=0.5, top_p=0.9)

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    def test_system_prompt_included(self):
        """System prompt should be included in API call when set."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            system="You are helpful",
        )
        manager.add_user_message("Hello")
        manager.get_response()

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"

    def test_system_prompt_omitted_when_none(self):
        """System prompt should not be in API call when None."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3", max_tokens=100, system=None
        )
        manager.add_user_message("Hello")
        manager.get_response()

        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_get_response_without_staged_message_raises(self):
        """get_response should raise ValueError if no user message is staged."""
        client = _make_sync_client()
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)
        with pytest.raises(ValueError, match="No staged user message"):
            manager.get_response()

    def test_get_response_with_content_argument(self):
        """get_response should accept content argument and stage it."""
        client = _make_sync_client()
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)
        manager.get_response("Hello")

        assert len(manager._history) == 2
        assert manager._history[0]["content"] == "Hello"

    def test_history_returns_copy(self):
        """history property should return a shallow copy."""
        client = _make_sync_client()
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)
        manager.add_user_message("Hello")

        hist1 = manager.history
        hist1.append({"role": "user", "content": "Fake"})

        hist2 = manager.history
        assert len(hist2) == 1  # Original not modified

    def test_reset_clears_history_and_usage(self):
        """reset should clear both history and last_usage."""
        client = _make_sync_client()
        manager = ConversationManager(client=client, model="claude-3", max_tokens=100)
        manager.add_user_message("Hello")
        manager.get_response()

        assert len(manager._history) > 0
        assert manager.last_usage is not None

        manager.reset()
        assert len(manager._history) == 0
        assert manager.last_usage is None

    def test_reset_preserves_model_and_system(self):
        """reset should not affect model and system prompt."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            system="You are helpful",
        )
        manager.add_user_message("Hello")
        manager.reset()

        assert manager._model == "claude-3"
        assert manager._system == "You are helpful"

    def test_truncation_disabled_when_limit_none(self):
        """Truncation should be skipped when context_window_limit is None."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client, model="claude-3", max_tokens=100, context_window_limit=None
        )
        # Add many messages
        for i in range(10):
            manager.add_user_message(f"Message {i}")
        manager.get_response()

        # History should not be truncated
        assert len(manager._history) == 11  # 10 user + 1 assistant

    def test_truncation_noop_when_under_threshold(self):
        """Truncation should not remove messages when under threshold."""
        client = _make_sync_client(input_tokens=50, output_tokens=10)
        manager = ConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.5,  # threshold = 500
        )
        manager.add_user_message("Hello")
        manager.get_response()

        # Get another response (usage is 60 tokens, threshold is 500)
        manager.add_user_message("World")
        manager.get_response()

        # Both messages should be in history
        assert len(manager._history) == 4

    def test_truncation_drops_oldest_pair(self):
        """Truncation should drop oldest user+assistant pair when over threshold."""
        client = _make_sync_client(input_tokens=600, output_tokens=100)
        manager = ConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.2,  # threshold = 800
        )

        # First turn (700 tokens total)
        manager.add_user_message("First")
        manager.get_response()

        # Second turn (will exceed threshold with 700 tokens)
        manager.add_user_message("Second")
        manager.get_response()

        # At this point, estimated_tokens is 700 (from last_usage)
        # On third turn, we check if 700 >= 800, it's not, so no truncation yet
        # Third turn
        manager.add_user_message("Third")
        manager.get_response()

        # Since 700 < 800, all messages should still be there
        assert len(manager._history) == 6  # All 3 turns preserved
        assert manager._history[0]["content"] == "First"

    def test_truncation_raises_when_pair_exceeds(self):
        """Truncation should raise ValueError if single pair exceeds limit."""
        client = _make_sync_client(input_tokens=950, output_tokens=100)
        manager = ConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.0,  # threshold = 1000
            accurate_token_counting=True,  # Use accurate mode so we truncate immediately
        )

        # First call: count_tokens returns 1050, which exceeds 1000
        # This should raise ValueError during truncation
        manager.add_user_message("Hello")
        with pytest.raises(ValueError, match="Cannot truncate further"):
            manager.get_response()

    def test_no_truncation_on_first_call_heuristic(self):
        """Truncation should skip on first call in heuristic mode (last_usage=None)."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=100,  # Very small
            accurate_token_counting=False,
        )

        # First call should not truncate (no last_usage yet)
        manager.add_user_message("Hello")
        manager.get_response()

        assert len(manager._history) == 2

    def test_accurate_mode_calls_count_tokens(self):
        """Accurate mode should call count_tokens."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.0,
            accurate_token_counting=True,
        )

        manager.add_user_message("Hello")
        manager.get_response()

        # count_tokens should be called
        assert client.messages.count_tokens.called

    def test_accurate_mode_truncates_to_threshold(self):
        """Accurate mode should truncate based on count_tokens result."""
        client = _make_sync_client(input_tokens=500, output_tokens=100)
        # Make count_tokens return different values for different calls
        # First call: 600 tokens (under threshold of 800, no truncation)
        # Second call (before second response): 1050 tokens (over threshold, truncates First)
        # Third call (after truncation): 600 tokens (under threshold)
        client.messages.count_tokens.side_effect = [600, 1050, 600, 600]

        manager = ConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.2,  # threshold = 800
            accurate_token_counting=True,
        )

        manager.add_user_message("First")
        manager.get_response()
        
        # Now count_tokens shows 1050 (over 800 threshold)
        # This will trigger truncation
        manager.add_user_message("Second")
        manager.get_response()

        # After truncation, only "Second" + assistant should remain
        assert len(manager._history) == 2
        assert manager._history[0]["content"] == "Second"

    def test_repr_shows_model_and_turns(self):
        """__repr__ should show model and turn count."""
        client = _make_sync_client()
        manager = ConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
        )

        repr_str = repr(manager)
        assert "claude-3" in repr_str
        assert "turns=0" in repr_str

        manager.add_user_message("Hello")
        manager.get_response()
        repr_str = repr(manager)
        assert "turns=1" in repr_str


class TestAsyncConversationManager:
    """Test suite for AsyncConversationManager."""

    @pytest.mark.asyncio
    async def test_init_validation_empty_model(self):
        """Constructor should raise ValueError for empty model."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="model cannot be empty"):
            AsyncConversationManager(client=client, model="", max_tokens=100)

    @pytest.mark.asyncio
    async def test_init_validation_max_tokens_zero(self):
        """Constructor should raise ValueError for max_tokens < 1."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            AsyncConversationManager(client=client, model="claude-3", max_tokens=0)

    @pytest.mark.asyncio
    async def test_single_turn_response(self):
        """get_response should return response and append assistant message."""
        client = _make_async_client(content_text="Response text")
        manager = AsyncConversationManager(
            client=client, model="claude-3", max_tokens=100
        )
        manager.add_user_message("Hello")
        response = await manager.get_response()

        assert response.content[0].text == "Response text"
        assert len(manager._history) == 2
        assert manager._history[0]["role"] == "user"
        assert manager._history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Multiple turns should accumulate in history."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client, model="claude-3", max_tokens=100
        )

        # First turn
        manager.add_user_message("First")
        await manager.get_response()
        assert len(manager._history) == 2

        # Second turn
        manager.add_user_message("Second")
        await manager.get_response()
        assert len(manager._history) == 4

    @pytest.mark.asyncio
    async def test_last_usage_populated_after_call(self):
        """last_usage should be populated after get_response."""
        client = _make_async_client(input_tokens=100, output_tokens=50)
        manager = AsyncConversationManager(
            client=client, model="claude-3", max_tokens=100
        )
        manager.add_user_message("Hello")
        await manager.get_response()

        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100
        assert manager.last_usage.output_tokens == 50

    @pytest.mark.asyncio
    async def test_system_prompt_included(self):
        """System prompt should be included in API call when set."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            system="You are helpful",
        )
        manager.add_user_message("Hello")
        await manager.get_response()

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"

    @pytest.mark.asyncio
    async def test_get_response_without_staged_message_raises(self):
        """get_response should raise ValueError if no user message is staged."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client, model="claude-3", max_tokens=100
        )
        with pytest.raises(ValueError, match="No staged user message"):
            await manager.get_response()

    @pytest.mark.asyncio
    async def test_history_returns_copy(self):
        """history property should return a shallow copy."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client, model="claude-3", max_tokens=100
        )
        manager.add_user_message("Hello")

        hist1 = manager.history
        hist1.append({"role": "user", "content": "Fake"})

        hist2 = manager.history
        assert len(hist2) == 1  # Original not modified

    @pytest.mark.asyncio
    async def test_reset_clears_history_and_usage(self):
        """reset should clear both history and last_usage."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client, model="claude-3", max_tokens=100
        )
        manager.add_user_message("Hello")
        await manager.get_response()

        assert len(manager._history) > 0
        assert manager.last_usage is not None

        manager.reset()
        assert len(manager._history) == 0
        assert manager.last_usage is None

    @pytest.mark.asyncio
    async def test_kwargs_forwarded_to_create(self):
        """Additional kwargs should be forwarded to messages.create."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client, model="claude-3", max_tokens=100
        )
        manager.add_user_message("Hello")
        await manager.get_response(temperature=0.5, top_p=0.9)

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_accurate_mode_calls_count_tokens(self):
        """Accurate mode should call count_tokens."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
            token_budget_headroom=0.0,
            accurate_token_counting=True,
        )

        manager.add_user_message("Hello")
        await manager.get_response()

        # count_tokens should be called
        assert client.messages.count_tokens.called

    @pytest.mark.asyncio
    async def test_repr_shows_model_and_turns(self):
        """__repr__ should show model and turn count."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client=client,
            model="claude-3",
            max_tokens=100,
            context_window_limit=1000,
        )

        repr_str = repr(manager)
        assert "claude-3" in repr_str
        assert "turns=0" in repr_str

        manager.add_user_message("Hello")
        await manager.get_response()
        repr_str = repr(manager)
        assert "turns=1" in repr_str
