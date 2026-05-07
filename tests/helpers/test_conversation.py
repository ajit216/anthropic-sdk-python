"""Tests for ConversationManager and AsyncConversationManager."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from anthropic.helpers import ConversationManager, AsyncConversationManager


def _make_sync_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock sync client."""
    client = MagicMock()

    # Create response mock
    response = MagicMock()
    response.content = [MagicMock(text=content_text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)

    # Create count_tokens mock
    count_result = MagicMock()
    count_result.input_tokens = input_tokens

    client.messages.create.return_value = response
    client.messages.count_tokens.return_value = count_result

    return client


def _make_async_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock async client."""
    client = MagicMock()

    # Create response mock
    response = MagicMock()
    response.content = [MagicMock(text=content_text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)

    # Create count_tokens async mock
    count_result = MagicMock()
    count_result.input_tokens = input_tokens

    client.messages.create = AsyncMock(return_value=response)
    client.messages.count_tokens = AsyncMock(return_value=count_result)

    return client


class TestConversationManager:
    """Tests for ConversationManager (sync)."""

    def test_init_validates_empty_model(self):
        """Constructor should raise ValueError for empty model."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="model must not be empty"):
            ConversationManager(client, model="", max_tokens=100)

    def test_init_validates_zero_max_tokens(self):
        """Constructor should raise ValueError for max_tokens < 1."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3-5-sonnet", max_tokens=0)

    def test_init_validates_negative_max_tokens(self):
        """Constructor should raise ValueError for negative max_tokens."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3-5-sonnet", max_tokens=-1)

    def test_init_validates_negative_context_window_limit(self):
        """Constructor should raise ValueError for context_window_limit < 1."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client,
                model="claude-3-5-sonnet",
                max_tokens=100,
                context_window_limit=0,
            )

    def test_init_validates_zero_context_window_limit(self):
        """Constructor should raise ValueError for zero context_window_limit."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client,
                model="claude-3-5-sonnet",
                max_tokens=100,
                context_window_limit=-5,
            )

    def test_init_validates_token_budget_headroom_lower(self):
        """Constructor should raise ValueError for token_budget_headroom < 0."""
        client = _make_sync_client()
        with pytest.raises(
            ValueError, match="token_budget_headroom must be in \\[0.0, 1.0\\)"
        ):
            ConversationManager(
                client,
                model="claude-3-5-sonnet",
                max_tokens=100,
                token_budget_headroom=-0.1,
            )

    def test_init_validates_token_budget_headroom_upper(self):
        """Constructor should raise ValueError for token_budget_headroom >= 1.0."""
        client = _make_sync_client()
        with pytest.raises(
            ValueError, match="token_budget_headroom must be in \\[0.0, 1.0\\)"
        ):
            ConversationManager(
                client,
                model="claude-3-5-sonnet",
                max_tokens=100,
                token_budget_headroom=1.0,
            )

    def test_add_user_message_basic(self):
        """add_user_message should append message to history."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        manager.add_user_message("Hello")

        assert len(manager.history) == 1
        assert manager.history[0] == {"role": "user", "content": "Hello"}

    def test_add_user_message_list_content(self):
        """add_user_message should handle list content."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        content = [{"type": "text", "text": "Hello"}]
        manager.add_user_message(content)

        assert len(manager.history) == 1
        assert manager.history[0] == {"role": "user", "content": content}

    def test_add_user_message_empty_string_raises(self):
        """add_user_message should raise ValueError for empty string."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        with pytest.raises(ValueError, match="content must not be empty"):
            manager.add_user_message("")

    def test_add_user_message_empty_list_raises(self):
        """add_user_message should raise ValueError for empty list."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        with pytest.raises(ValueError, match="content must not be empty"):
            manager.add_user_message([])

    def test_get_response_single_turn(self):
        """get_response should handle single turn conversation."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        manager.add_user_message("Hello")
        response = manager.get_response()

        assert response is not None
        assert len(manager.history) == 2
        assert manager.history[0] == {"role": "user", "content": "Hello"}
        assert manager.history[1]["role"] == "assistant"
        client.messages.create.assert_called_once()

    def test_get_response_with_content_argument(self):
        """get_response with content should add user message first."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        response = manager.get_response("Hello")

        assert response is not None
        assert len(manager.history) == 2
        assert manager.history[0] == {"role": "user", "content": "Hello"}
        assert manager.history[1]["role"] == "assistant"

    def test_get_response_without_staged_message_raises(self):
        """get_response without staged message should raise ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        with pytest.raises(ValueError, match="No user message staged"):
            manager.get_response()

    def test_get_response_after_assistant_response_raises(self):
        """get_response after assistant response without new user message raises."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        manager.add_user_message("Hello")
        manager.get_response()

        with pytest.raises(ValueError, match="No user message staged"):
            manager.get_response()

    def test_get_response_multi_turn(self):
        """get_response should handle multi-turn conversation."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        # First turn
        manager.add_user_message("Hello")
        response1 = manager.get_response()

        # Second turn
        manager.add_user_message("How are you?")
        response2 = manager.get_response()

        assert len(manager.history) == 4
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"
        assert manager.history[2]["role"] == "user"
        assert manager.history[3]["role"] == "assistant"
        assert client.messages.create.call_count == 2

    def test_get_response_forwards_kwargs(self):
        """get_response should forward kwargs to messages.create."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        manager.add_user_message("Hello")
        manager.get_response(temperature=0.5, top_p=0.9)

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    def test_get_response_includes_system_prompt(self):
        """get_response should include system prompt when set."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=100, system="You are helpful"
        )

        manager.add_user_message("Hello")
        manager.get_response()

        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"

    def test_get_response_omits_system_prompt_when_none(self):
        """get_response should omit system prompt when None."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        manager.add_user_message("Hello")
        manager.get_response()

        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_last_usage_initially_none(self):
        """last_usage should be None before first response."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        assert manager.last_usage is None

    def test_last_usage_populated_after_response(self):
        """last_usage should be populated after first response."""
        client = _make_sync_client(input_tokens=150, output_tokens=75)
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        manager.add_user_message("Hello")
        manager.get_response()

        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 150
        assert manager.last_usage.output_tokens == 75

    def test_history_returns_copy(self):
        """history property should return a copy."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        manager.add_user_message("Hello")
        history1 = manager.history
        history2 = manager.history

        assert history1 == history2
        assert history1 is not history2  # Different objects

        # Mutating returned history should not affect internal state
        history1.append({"role": "fake", "content": "test"})
        assert len(manager.history) == 1

    def test_reset_clears_history(self):
        """reset should clear history."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        manager.add_user_message("Hello")
        manager.get_response()

        manager.reset()

        assert len(manager.history) == 0
        assert manager.last_usage is None

    def test_reset_preserves_settings(self):
        """reset should preserve model and system prompt."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=100, system="You are helpful"
        )

        manager.add_user_message("Hello")
        manager.get_response()
        manager.reset()

        # Model and system should be unchanged
        assert manager._model == "claude-3-5-sonnet"
        assert manager._system == "You are helpful"

    def test_repr(self):
        """__repr__ should show model, turn count, and limit."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=200000,
        )

        manager.add_user_message("Hello")
        manager.get_response()

        repr_str = repr(manager)
        assert "ConversationManager" in repr_str
        assert "claude-3-5-sonnet" in repr_str
        assert "turns=1" in repr_str
        assert "limit=200000" in repr_str

    def test_truncation_disabled_when_no_limit(self):
        """Truncation should not occur when context_window_limit is None."""
        client = _make_sync_client(input_tokens=100000, output_tokens=50000)
        manager = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=100)

        manager.add_user_message("Message 1")
        manager.get_response()

        # Should not truncate (no limit)
        assert len(manager.history) == 2

    def test_truncation_skipped_on_first_call_heuristic(self):
        """Truncation should be skipped on first call in heuristic mode."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=200,
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )

        # First call: last_usage is None, so truncation skipped
        manager.add_user_message("Hello")
        manager.get_response()

        assert len(manager.history) == 2
        client.messages.count_tokens.assert_not_called()

    def test_truncation_no_op_when_under_threshold(self):
        """Truncation should not occur when under threshold."""
        client = _make_sync_client(input_tokens=50, output_tokens=25)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=10000,  # Large limit
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )

        manager.add_user_message("Hello")
        manager.get_response()

        # Add more messages (below threshold)
        manager.add_user_message("Hi again")
        manager.get_response()

        # Should have 4 messages (no truncation)
        assert len(manager.history) == 4

    def test_truncation_drops_oldest_pair(self):
        """Truncation should drop oldest user+assistant pair."""
        # First response: 100 input, 50 output = 150 total
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=200,  # Threshold = 200 * 0.9 = 180
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )

        # First turn
        manager.add_user_message("Message 1")
        manager.get_response()
        assert len(manager.history) == 2

        # Second turn - triggers truncation
        # Setup: next response will be 200 tokens (over threshold)
        client.messages.create.return_value.usage.input_tokens = 200
        client.messages.create.return_value.usage.output_tokens = 50

        manager.add_user_message("Message 2")
        manager.get_response()

        # Should have truncated oldest pair
        # History should have: Message 2 (user) + Response (assistant)
        assert len(manager.history) == 2
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "Message 2"

    def test_truncation_multiple_pairs(self):
        """Truncation should drop multiple pairs if needed."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=300,  # Threshold = 300 * 0.9 = 270
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )

        # Add 3 turns (6 messages total)
        for i in range(3):
            manager.add_user_message(f"Message {i+1}")
            manager.get_response()

        assert len(manager.history) == 6

        # Trigger truncation with large response
        # estimated_tokens will be 400 + 100 = 500, which is > threshold (270)
        client.messages.create.return_value.usage.input_tokens = 400
        client.messages.create.return_value.usage.output_tokens = 100

        manager.add_user_message("Message 4")
        manager.get_response()

        # With 6 initial messages + 1 user = 7 messages
        # First iteration: remove 2 messages, estimated_tokens = 500 * (1 - 2/7) ≈ 357
        # Second iteration: remove 2 more, estimated_tokens = 357 * (1 - 2/5) ≈ 214
        # Now under threshold. Should have: Message 3, response, Message 4, response = 4
        assert len(manager.history) == 4

    def test_truncation_raises_when_single_pair_exceeds_limit(self):
        """Truncation should raise ValueError when single pair exceeds limit."""
        # This tests accurate_token_counting mode where we can detect
        # that a single pair exceeds the limit during truncation
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=300,  # Threshold = 300 * 0.9 = 270
            token_budget_headroom=0.1,
            accurate_token_counting=True,  # Use accurate mode
        )

        # Add first turn
        manager.add_user_message("Message 1")
        manager.get_response()
        
        # Add second turn
        manager.add_user_message("Message 2")
        
        # Setup count_tokens to always return very large number
        # This simulates a scenario where even after truncating all old messages,
        # the remaining pair is still over threshold
        count_result = MagicMock()
        count_result.input_tokens = 1000  # Huge, over threshold
        client.messages.count_tokens.return_value = count_result
        
        # Try to get response - will fail because count_tokens returns
        # a number that's always over threshold
        with pytest.raises(ValueError, match="Cannot truncate further"):
            manager.get_response()

    def test_truncation_accurate_mode(self):
        """Truncation should use count_tokens in accurate mode."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=200,  # Threshold = 180
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )

        manager.add_user_message("Message 1")
        manager.get_response()

        # Setup: count_tokens will return values
        # First call returns 150 (under threshold, no truncation)
        # Subsequent calls return 120 (still under threshold)
        count_result_under = MagicMock()
        count_result_under.input_tokens = 120
        client.messages.count_tokens.return_value = count_result_under

        # Next response - should use count_tokens but not truncate
        manager.add_user_message("Message 2")
        manager.get_response()

        # count_tokens should have been called at least once
        assert client.messages.count_tokens.call_count >= 1


class TestAsyncConversationManager:
    """Tests for AsyncConversationManager."""

    @pytest.mark.asyncio
    async def test_init_validates_empty_model(self):
        """Constructor should raise ValueError for empty model."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="model must not be empty"):
            AsyncConversationManager(client, model="", max_tokens=100)

    @pytest.mark.asyncio
    async def test_init_validates_zero_max_tokens(self):
        """Constructor should raise ValueError for max_tokens < 1."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            AsyncConversationManager(client, model="claude-3-5-sonnet", max_tokens=0)

    @pytest.mark.asyncio
    async def test_add_user_message_basic(self):
        """add_user_message should append message to history."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=100
        )

        manager.add_user_message("Hello")

        assert len(manager.history) == 1
        assert manager.history[0] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_get_response_single_turn(self):
        """get_response should handle single turn conversation."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=100
        )

        manager.add_user_message("Hello")
        response = await manager.get_response()

        assert response is not None
        assert len(manager.history) == 2
        assert manager.history[0] == {"role": "user", "content": "Hello"}
        assert manager.history[1]["role"] == "assistant"
        client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_response_with_content_argument(self):
        """get_response with content should add user message first."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=100
        )

        response = await manager.get_response("Hello")

        assert response is not None
        assert len(manager.history) == 2
        assert manager.history[0] == {"role": "user", "content": "Hello"}
        assert manager.history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_response_without_staged_message_raises(self):
        """get_response without staged message should raise ValueError."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=100
        )

        with pytest.raises(ValueError, match="No user message staged"):
            await manager.get_response()

    @pytest.mark.asyncio
    async def test_get_response_multi_turn(self):
        """get_response should handle multi-turn conversation."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=100
        )

        # First turn
        manager.add_user_message("Hello")
        response1 = await manager.get_response()

        # Second turn
        manager.add_user_message("How are you?")
        response2 = await manager.get_response()

        assert len(manager.history) == 4
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"
        assert manager.history[2]["role"] == "user"
        assert manager.history[3]["role"] == "assistant"
        assert client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_last_usage_populated_after_response(self):
        """last_usage should be populated after first response."""
        client = _make_async_client(input_tokens=150, output_tokens=75)
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=100
        )

        manager.add_user_message("Hello")
        await manager.get_response()

        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 150
        assert manager.last_usage.output_tokens == 75

    @pytest.mark.asyncio
    async def test_history_returns_copy(self):
        """history property should return a copy."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=100
        )

        manager.add_user_message("Hello")
        history1 = manager.history
        history2 = manager.history

        assert history1 == history2
        assert history1 is not history2

    @pytest.mark.asyncio
    async def test_reset_clears_history(self):
        """reset should clear history."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=100
        )

        manager.add_user_message("Hello")
        await manager.get_response()

        manager.reset()

        assert len(manager.history) == 0
        assert manager.last_usage is None

    @pytest.mark.asyncio
    async def test_repr(self):
        """__repr__ should show model, turn count, and limit."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=200000,
        )

        manager.add_user_message("Hello")
        await manager.get_response()

        repr_str = repr(manager)
        assert "AsyncConversationManager" in repr_str
        assert "claude-3-5-sonnet" in repr_str
        assert "turns=1" in repr_str

    @pytest.mark.asyncio
    async def test_truncation_accurate_mode(self):
        """Truncation should use count_tokens in accurate mode."""
        client = _make_async_client(input_tokens=100, output_tokens=50)
        manager = AsyncConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=100,
            context_window_limit=200,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )

        manager.add_user_message("Message 1")
        await manager.get_response()

        # Setup: count_tokens will return values under threshold
        count_result = MagicMock()
        count_result.input_tokens = 120  # Under threshold of 180
        client.messages.count_tokens.return_value = count_result

        # Next response - should use count_tokens but not truncate
        manager.add_user_message("Message 2")
        await manager.get_response()

        # count_tokens should have been called at least once
        assert client.messages.count_tokens.call_count >= 1
