"""Tests for ConversationManager and AsyncConversationManager."""

import pytest
from unittest.mock import MagicMock, AsyncMock, call

from anthropic.helpers import ConversationManager, AsyncConversationManager


def _make_sync_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock sync Anthropic client."""
    response = MagicMock()
    response.content = [MagicMock(text=content_text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)

    client = MagicMock()
    client.messages.create.return_value = response
    client.messages.count_tokens.return_value = MagicMock(input_tokens=input_tokens)
    return client


def _make_async_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> AsyncMock:
    """Create a mock async Anthropic client."""
    response = MagicMock()
    response.content = [MagicMock(text=content_text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)

    client = AsyncMock()
    client.messages.create.return_value = response
    client.messages.count_tokens.return_value = MagicMock(input_tokens=input_tokens)
    return client


class TestConversationManager:
    """Test suite for ConversationManager."""

    def test_init_validates_empty_model(self):
        """Constructor raises ValueError for empty model."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="model must not be empty"):
            ConversationManager(client, model="", max_tokens=1024)

    def test_init_validates_max_tokens(self):
        """Constructor raises ValueError for invalid max_tokens."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude", max_tokens=0)

        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude", max_tokens=-1)

    def test_init_validates_context_window_limit(self):
        """Constructor raises ValueError for invalid context_window_limit."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client, model="claude", max_tokens=1024, context_window_limit=0
            )

        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client, model="claude", max_tokens=1024, context_window_limit=-1
            )

    def test_init_validates_token_budget_headroom(self):
        """Constructor raises ValueError for invalid token_budget_headroom."""
        client = _make_sync_client()
        with pytest.raises(
            ValueError, match="token_budget_headroom must be in \\[0.0, 1.0\\)"
        ):
            ConversationManager(
                client,
                model="claude",
                max_tokens=1024,
                token_budget_headroom=-0.1,
            )

        with pytest.raises(
            ValueError, match="token_budget_headroom must be in \\[0.0, 1.0\\)"
        ):
            ConversationManager(
                client, model="claude", max_tokens=1024, token_budget_headroom=1.0
            )

    def test_add_user_message(self):
        """add_user_message appends to history."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "Hello"

    def test_add_user_message_rejects_empty_string(self):
        """add_user_message raises ValueError for empty string."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        with pytest.raises(ValueError, match="content must not be empty"):
            manager.add_user_message("")

    def test_add_user_message_rejects_empty_list(self):
        """add_user_message raises ValueError for empty list."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        with pytest.raises(ValueError, match="content must not be empty"):
            manager.add_user_message([])

    def test_get_response_calls_api(self):
        """get_response calls the API once and returns Message."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("What is 2+2?")
        response = manager.get_response()

        assert response.content[0].text == "Hello"
        client.messages.create.assert_called_once()

    def test_get_response_with_content_argument(self):
        """get_response with content argument stages message before API call."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        response = manager.get_response("What is 2+2?")

        assert response.content[0].text == "Hello"
        client.messages.create.assert_called_once()
        assert len(manager.history) == 2  # user + assistant

    def test_get_response_appends_assistant_turn(self):
        """get_response appends assistant message to history."""
        client = _make_sync_client(content_text="4")
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("What is 2+2?")
        response = manager.get_response()

        assert len(manager.history) == 2
        assert manager.history[1]["role"] == "assistant"
        assert manager.history[1]["content"][0].text == "4"

    def test_get_response_multi_turn(self):
        """Multi-turn conversation maintains history."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("First question")
        response1 = manager.get_response()
        assert response1 is not None

        manager.add_user_message("Second question")
        response2 = manager.get_response()
        assert response2 is not None

        assert len(manager.history) == 4  # 2 user + 2 assistant messages

    def test_last_usage_none_initially(self):
        """last_usage is None before any API call."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        assert manager.last_usage is None

    def test_last_usage_populated_after_response(self):
        """last_usage is populated after get_response."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("Hello")
        manager.get_response()

        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100
        assert manager.last_usage.output_tokens == 50

    def test_get_response_forwards_kwargs(self):
        """get_response forwards kwargs to messages.create."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("Hello")
        manager.get_response(temperature=0.5, top_p=0.9)

        call_args = client.messages.create.call_args
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["top_p"] == 0.9

    def test_system_prompt_included_when_set(self):
        """System prompt is passed to API when set."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude", max_tokens=1024, system="You are helpful"
        )

        manager.add_user_message("Hello")
        manager.get_response()

        call_args = client.messages.create.call_args
        assert call_args.kwargs["system"] == "You are helpful"

    def test_system_prompt_omitted_when_none(self):
        """System prompt is not passed to API when None."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024, system=None)

        manager.add_user_message("Hello")
        manager.get_response()

        call_args = client.messages.create.call_args
        assert "system" not in call_args.kwargs

    def test_get_response_without_staged_message_raises(self):
        """get_response raises ValueError if no user message is staged."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        with pytest.raises(ValueError, match="No user message staged"):
            manager.get_response()

    def test_history_returns_copy(self):
        """history property returns a copy, not a reference."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("Hello")
        history1 = manager.history
        history2 = manager.history

        assert history1 is not history2
        assert history1 == history2

    def test_history_copy_mutation_does_not_affect_manager(self):
        """Mutating returned history copy doesn't affect internal state."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("Hello")
        history = manager.history
        history.clear()

        assert len(manager.history) == 1

    def test_reset_clears_history(self):
        """reset() clears history and last_usage."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("Hello")
        manager.get_response()

        assert len(manager.history) > 0
        assert manager.last_usage is not None

        manager.reset()

        assert len(manager.history) == 0
        assert manager.last_usage is None

    def test_reset_preserves_model_and_system(self):
        """reset() preserves model and system prompt configuration."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude", max_tokens=1024, system="You are helpful"
        )

        manager.add_user_message("Hello")
        manager.get_response()
        manager.reset()

        # Verify model and system are still configured
        manager.add_user_message("Hi again")
        manager.get_response()

        call_args = client.messages.create.call_args
        assert call_args.kwargs["system"] == "You are helpful"
        assert call_args.kwargs["model"] == "claude"

    def test_truncation_noop_without_limit(self):
        """Truncation is no-op when context_window_limit is None."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude", max_tokens=1024, context_window_limit=None
        )

        manager.add_user_message("Hello")
        manager.get_response()

        history_len = len(manager.history)
        manager.add_user_message("More messages")
        manager.get_response()

        assert len(manager.history) == history_len + 2

    def test_truncation_noop_under_threshold(self):
        """Truncation is no-op when under threshold."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,  # threshold = 900
        )

        manager.add_user_message("Hello")
        manager.get_response()

        original_history = list(manager.history)
        manager.add_user_message("How are you")
        manager.get_response()

        # With tokens well under threshold, no truncation
        assert len(manager.history) == 4

    def test_truncation_drops_oldest_pair_when_over_threshold(self):
        """Truncation drops oldest pair when over threshold."""
        # Setup: Use accurate mode to test truncation
        client = MagicMock()
        
        response = MagicMock()
        response.content = [MagicMock(text="Response")]
        response.usage = MagicMock(input_tokens=100, output_tokens=50)
        client.messages.create.return_value = response
        
        # Setup count_tokens to return different values depending on history size
        def count_tokens_side_effect(**kwargs):
            messages = kwargs.get("messages", [])
            # Return 950 for 5 messages, 150 for 3 messages, etc.
            # More messages = more tokens
            token_per_message = 190
            return MagicMock(input_tokens=len(messages) * token_per_message)
        
        client.messages.count_tokens.side_effect = count_tokens_side_effect

        manager = ConversationManager(
            client,
            model="claude",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,  # threshold = 900
            accurate_token_counting=True,  # Use accurate mode
        )

        # Add first pair: 2 messages, 380 tokens < 900
        manager.add_user_message("Q1")
        manager.get_response()
        # Add second pair: 4 messages, 760 tokens < 900
        manager.add_user_message("Q2")
        manager.get_response()
        # Add third user message: 5 messages staged (before get_response)
        manager.add_user_message("Q3")

        # This should trigger truncation:
        # Before API call, count_tokens on 5 messages = 950 >= 900
        # So it truncates: removes Q1 + response (2 messages)
        # Now 3 messages left, count_tokens = 570 < 900, so done truncating
        manager.get_response()

        # Should have removed the first pair (Q1 + response)
        # Remaining: Q2, response, Q3, response (4 messages)
        assert len(manager.history) == 4
        # The oldest message should be from Q2, not Q1
        assert manager.history[0]["content"] == "Q2"

    def test_truncation_raises_when_single_pair_exceeds_limit(self):
        """Truncation raises ValueError when single pair still exceeds limit."""
        client = MagicMock()
        
        response = MagicMock()
        response.content = [MagicMock(text="Response")]
        response.usage = MagicMock(input_tokens=100, output_tokens=50)
        client.messages.create.return_value = response
        
        # Setup count_tokens to always return very high value (over threshold)
        client.messages.count_tokens.return_value = MagicMock(input_tokens=950)

        manager = ConversationManager(
            client,
            model="claude",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,  # threshold = 900
            accurate_token_counting=True,  # Use accurate mode so we always see high tokens
        )

        manager.add_user_message("Question")
        # First call should fail because even a single message pair exceeds the threshold
        with pytest.raises(ValueError, match="Cannot truncate further"):
            manager.get_response()

    def test_no_truncation_on_first_call_heuristic_mode(self):
        """No truncation on first call when using heuristic mode (last_usage is None)."""
        client = _make_sync_client()
        manager = ConversationManager(
            client,
            model="claude",
            max_tokens=1024,
            context_window_limit=100,  # Very low limit
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )

        # First call should not truncate since last_usage is None
        manager.add_user_message("Hello")
        response = manager.get_response()

        assert response is not None
        assert len(manager.history) == 2

    def test_accurate_mode_calls_count_tokens(self):
        """Accurate mode uses count_tokens for precise truncation."""
        client = MagicMock()
        response = MagicMock()
        response.content = [MagicMock(text="Response")]
        response.usage = MagicMock(input_tokens=100, output_tokens=50)
        client.messages.create.return_value = response
        client.messages.count_tokens.return_value = MagicMock(input_tokens=200)

        manager = ConversationManager(
            client,
            model="claude",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )

        manager.add_user_message("Hello")
        manager.get_response()

        # count_tokens should have been called at least once
        assert client.messages.count_tokens.called

    def test_repr_shows_model_and_turns(self):
        """__repr__ shows model and turn count."""
        client = _make_sync_client()
        manager = ConversationManager(client, model="claude-3", max_tokens=1024)

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
    async def test_init_validates_empty_model(self):
        """Constructor raises ValueError for empty model."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="model must not be empty"):
            AsyncConversationManager(client, model="", max_tokens=1024)

    @pytest.mark.asyncio
    async def test_add_user_message(self):
        """add_user_message appends to history."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_get_response_calls_api(self):
        """get_response calls the API once and returns Message."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("What is 2+2?")
        response = await manager.get_response()

        assert response.content[0].text == "Hello"
        client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_response_with_content_argument(self):
        """get_response with content argument stages message before API call."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude", max_tokens=1024)

        response = await manager.get_response("What is 2+2?")

        assert response.content[0].text == "Hello"
        client.messages.create.assert_called_once()
        assert len(manager.history) == 2  # user + assistant

    @pytest.mark.asyncio
    async def test_get_response_appends_assistant_turn(self):
        """get_response appends assistant message to history."""
        client = _make_async_client(content_text="4")
        manager = AsyncConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("What is 2+2?")
        response = await manager.get_response()

        assert len(manager.history) == 2
        assert manager.history[1]["role"] == "assistant"
        assert manager.history[1]["content"][0].text == "4"

    @pytest.mark.asyncio
    async def test_get_response_multi_turn(self):
        """Multi-turn conversation maintains history."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("First question")
        response1 = await manager.get_response()
        assert response1 is not None

        manager.add_user_message("Second question")
        response2 = await manager.get_response()
        assert response2 is not None

        assert len(manager.history) == 4  # 2 user + 2 assistant messages

    @pytest.mark.asyncio
    async def test_last_usage_none_initially(self):
        """last_usage is None before any API call."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude", max_tokens=1024)

        assert manager.last_usage is None

    @pytest.mark.asyncio
    async def test_last_usage_populated_after_response(self):
        """last_usage is populated after get_response."""
        client = _make_async_client(input_tokens=100, output_tokens=50)
        manager = AsyncConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("Hello")
        await manager.get_response()

        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 100
        assert manager.last_usage.output_tokens == 50

    @pytest.mark.asyncio
    async def test_reset_clears_history(self):
        """reset() clears history and last_usage."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude", max_tokens=1024)

        manager.add_user_message("Hello")
        await manager.get_response()

        assert len(manager.history) > 0
        assert manager.last_usage is not None

        manager.reset()

        assert len(manager.history) == 0
        assert manager.last_usage is None

    @pytest.mark.asyncio
    async def test_get_response_without_staged_message_raises(self):
        """get_response raises ValueError if no user message is staged."""
        client = _make_async_client()
        manager = AsyncConversationManager(client, model="claude", max_tokens=1024)

        with pytest.raises(ValueError, match="No user message staged"):
            await manager.get_response()

    @pytest.mark.asyncio
    async def test_accurate_mode_calls_count_tokens(self):
        """Accurate mode uses count_tokens for precise truncation."""
        client = AsyncMock()
        response = MagicMock()
        response.content = [MagicMock(text="Response")]
        response.usage = MagicMock(input_tokens=100, output_tokens=50)
        client.messages.create.return_value = response
        client.messages.count_tokens.return_value = MagicMock(input_tokens=200)

        manager = AsyncConversationManager(
            client,
            model="claude",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )

        manager.add_user_message("Hello")
        await manager.get_response()

        # count_tokens should have been called at least once
        assert client.messages.count_tokens.called
