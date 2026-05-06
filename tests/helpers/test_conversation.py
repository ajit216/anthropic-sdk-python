"""Tests for ConversationManager and AsyncConversationManager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from anthropic.helpers import ConversationManager, AsyncConversationManager


def _make_sync_client(*, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello") -> MagicMock:
    """Create a mock sync client."""
    client = MagicMock()
    
    # Mock response object
    response = MagicMock()
    response.content = [MagicMock(text=content_text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    
    client.messages.create.return_value = response
    
    # Mock count_tokens
    count_response = MagicMock()
    count_response.input_tokens = input_tokens
    client.messages.count_tokens.return_value = count_response
    
    return client


def _make_async_client(*, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello") -> MagicMock:
    """Create a mock async client."""
    client = MagicMock()
    
    # Mock response object
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
    """Tests for ConversationManager."""

    def test_constructor_raises_on_empty_model(self):
        """Constructor should raise on empty model."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="model must not be empty"):
            ConversationManager(client, model="", max_tokens=1024)

    def test_constructor_raises_on_zero_max_tokens(self):
        """Constructor should raise on zero max_tokens."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3-5-sonnet", max_tokens=0)

    def test_constructor_raises_on_negative_max_tokens(self):
        """Constructor should raise on negative max_tokens."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3-5-sonnet", max_tokens=-1)

    def test_constructor_raises_on_zero_context_window_limit(self):
        """Constructor should raise on zero context_window_limit."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1"):
            ConversationManager(
                client, model="claude-3-5-sonnet", max_tokens=1024, context_window_limit=0
            )

    def test_constructor_raises_on_invalid_token_budget_headroom(self):
        """Constructor should raise on invalid token_budget_headroom."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client, model="claude-3-5-sonnet", max_tokens=1024, token_budget_headroom=-0.1
            )
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client, model="claude-3-5-sonnet", max_tokens=1024, token_budget_headroom=1.0
            )

    def test_add_user_message_appends_to_history(self):
        """add_user_message should append to history."""
        client = _make_sync_client()
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        conversation.add_user_message("Hello")
        
        assert len(conversation.history) == 1
        assert conversation.history[0]["role"] == "user"
        assert conversation.history[0]["content"] == "Hello"

    def test_add_user_message_raises_on_empty_string(self):
        """add_user_message should raise on empty string."""
        client = _make_sync_client()
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        with pytest.raises(ValueError, match="content must not be empty"):
            conversation.add_user_message("")

    def test_add_user_message_raises_on_empty_list(self):
        """add_user_message should raise on empty list."""
        client = _make_sync_client()
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        with pytest.raises(ValueError, match="content must not be empty"):
            conversation.add_user_message([])

    def test_get_response_calls_api_once(self):
        """get_response should call API once."""
        client = _make_sync_client()
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        conversation.add_user_message("Hello")
        response = conversation.get_response()
        
        assert client.messages.create.call_count == 1
        assert len(response.content) == 1
        assert response.content[0].text == "Hello"

    def test_get_response_appends_assistant_turn(self):
        """get_response should append assistant turn to history."""
        client = _make_sync_client(content_text="Hi there")
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        conversation.add_user_message("Hello")
        conversation.get_response()
        
        assert len(conversation.history) == 2
        assert conversation.history[1]["role"] == "assistant"

    def test_get_response_with_content_arg(self):
        """get_response with content should add message and get response."""
        client = _make_sync_client()
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        response = conversation.get_response("Hello")
        
        assert len(conversation.history) == 2
        assert conversation.history[0]["role"] == "user"
        assert conversation.history[1]["role"] == "assistant"

    def test_get_response_without_staged_message_raises(self):
        """get_response without staged message should raise."""
        client = _make_sync_client()
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        with pytest.raises(ValueError, match="No user message staged"):
            conversation.get_response()

    def test_multi_turn_conversation(self):
        """Multi-turn conversation should maintain history."""
        client = _make_sync_client()
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        # First turn
        conversation.add_user_message("Hello")
        conversation.get_response()
        
        # Second turn
        conversation.add_user_message("How are you?")
        conversation.get_response()
        
        assert len(conversation.history) == 4
        assert conversation.history[0]["role"] == "user"
        assert conversation.history[1]["role"] == "assistant"
        assert conversation.history[2]["role"] == "user"
        assert conversation.history[3]["role"] == "assistant"

    def test_last_usage_initially_none(self):
        """last_usage should be None initially."""
        client = _make_sync_client()
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        assert conversation.last_usage is None

    def test_last_usage_populated_after_response(self):
        """last_usage should be populated after get_response."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        conversation.get_response("Hello")
        
        assert conversation.last_usage is not None
        assert conversation.last_usage.input_tokens == 100
        assert conversation.last_usage.output_tokens == 50

    def test_kwargs_forwarded_to_api(self):
        """Additional kwargs should be forwarded to messages.create."""
        client = _make_sync_client()
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        conversation.get_response("Hello", temperature=0.5, top_p=0.9)
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    def test_system_prompt_passed_when_set(self):
        """System prompt should be passed to API when set."""
        client = _make_sync_client()
        conversation = ConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=1024, system="You are helpful"
        )
        
        conversation.get_response("Hello")
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"

    def test_system_prompt_omitted_when_none(self):
        """System prompt should be omitted when None."""
        client = _make_sync_client()
        conversation = ConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=1024, system=None
        )
        
        conversation.get_response("Hello")
        
        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_history_returns_copy(self):
        """history property should return a copy."""
        client = _make_sync_client()
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        conversation.add_user_message("Hello")
        
        hist = conversation.history
        original_len = len(hist)
        hist.append({"role": "test", "content": "test"})  # Modify copy
        
        # Internal history should not be affected
        assert len(conversation.history) == original_len

    def test_reset_clears_history(self):
        """reset should clear history."""
        client = _make_sync_client()
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        conversation.get_response("Hello")
        conversation.reset()
        
        assert len(conversation.history) == 0

    def test_reset_clears_last_usage(self):
        """reset should clear last_usage."""
        client = _make_sync_client()
        conversation = ConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        conversation.get_response("Hello")
        conversation.reset()
        
        assert conversation.last_usage is None

    def test_reset_preserves_model_and_system(self):
        """reset should preserve model and system prompt."""
        client = _make_sync_client()
        system_prompt = "You are helpful"
        conversation = ConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=1024, system=system_prompt
        )
        
        conversation.get_response("Hello")
        conversation.reset()
        conversation.add_user_message("Hi again")
        conversation.get_response()
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == system_prompt

    def test_truncation_noop_without_limit(self):
        """Truncation should be no-op without context_window_limit."""
        client = _make_sync_client()
        conversation = ConversationManager(
            client, model="claude-3-5-sonnet", max_tokens=1024, context_window_limit=None
        )
        
        for i in range(5):
            conversation.get_response(f"Message {i}")
        
        # Should have 10 messages (5 user + 5 assistant)
        assert len(conversation.history) == 10

    def test_truncation_noop_when_under_threshold(self):
        """Truncation should be no-op when under threshold."""
        client = _make_sync_client(input_tokens=100, output_tokens=50)
        conversation = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=1024,
            context_window_limit=10000,
            token_budget_headroom=0.1,
        )
        
        conversation.get_response("Hello")
        
        # Total estimated tokens: 150, threshold: 9000 * 0.9 = 8100
        # Should not truncate
        assert len(conversation.history) == 2

    def test_truncation_drops_oldest_pair_when_over_threshold(self):
        """Truncation should drop oldest pair when over threshold."""
        client = _make_sync_client(input_tokens=50, output_tokens=50)
        conversation = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=1024,
            context_window_limit=1000,  # Larger limit to allow messages
            token_budget_headroom=0.1,  # threshold = 900
        )
        
        # First message: estimated tokens = 100
        conversation.get_response("First")
        assert len(conversation.history) == 2
        
        # Create a new client with higher token estimates
        client2 = _make_sync_client(input_tokens=500, output_tokens=500)
        conversation._client = client2  # Replace client
        
        # Second message should still not truncate (1000 > 900)
        conversation.get_response("Second")
        
        # Should still have 4 messages (no truncation needed)
        assert len(conversation.history) == 4

    def test_truncation_raises_when_single_pair_exceeds_limit(self):
        """Truncation should raise when single pair exceeds limit."""
        client = _make_sync_client(input_tokens=1000, output_tokens=1000)
        conversation = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=1024,
            context_window_limit=100,  # Very small
            token_budget_headroom=0.1,
            accurate_token_counting=True,  # Use accurate mode to check on first call
        )
        
        with pytest.raises(ValueError, match="Cannot truncate further"):
            conversation.get_response("Hello")

    def test_no_truncation_on_first_call_heuristic_mode(self):
        """No truncation on first call in heuristic mode."""
        client = _make_sync_client(input_tokens=50, output_tokens=50)
        conversation = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=1024,
            context_window_limit=100,
            token_budget_headroom=0.1,
            accurate_token_counting=False,
        )
        
        # First call has no last_usage, so no truncation
        conversation.get_response("Hello")
        assert len(conversation.history) == 2

    def test_accurate_token_counting(self):
        """Accurate token counting should call count_tokens."""
        client = _make_sync_client(input_tokens=50, output_tokens=50)
        conversation = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=1024,
            context_window_limit=200,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )
        
        conversation.get_response("Hello")
        
        # count_tokens should have been called
        assert client.messages.count_tokens.called

    def test_repr(self):
        """repr should show model, turns, and limit."""
        client = _make_sync_client()
        conversation = ConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=1024,
            context_window_limit=200000,
        )
        
        repr_str = repr(conversation)
        assert "claude-3-5-sonnet" in repr_str
        assert "turns=0" in repr_str
        assert "limit=200000" in repr_str


class TestAsyncConversationManager:
    """Tests for AsyncConversationManager."""

    @pytest.mark.asyncio
    async def test_constructor_raises_on_empty_model(self):
        """Constructor should raise on empty model."""
        client = _make_async_client()
        with pytest.raises(ValueError, match="model must not be empty"):
            AsyncConversationManager(client, model="", max_tokens=1024)

    @pytest.mark.asyncio
    async def test_get_response_calls_api_once(self):
        """get_response should call API once."""
        client = _make_async_client()
        conversation = AsyncConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        conversation.add_user_message("Hello")
        response = await conversation.get_response()
        
        assert client.messages.create.call_count == 1
        assert len(response.content) == 1
        assert response.content[0].text == "Hello"

    @pytest.mark.asyncio
    async def test_get_response_appends_assistant_turn(self):
        """get_response should append assistant turn to history."""
        client = _make_async_client(content_text="Hi there")
        conversation = AsyncConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        conversation.add_user_message("Hello")
        await conversation.get_response()
        
        assert len(conversation.history) == 2
        assert conversation.history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_response_with_content_arg(self):
        """get_response with content should add message and get response."""
        client = _make_async_client()
        conversation = AsyncConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        response = await conversation.get_response("Hello")
        
        assert len(conversation.history) == 2
        assert conversation.history[0]["role"] == "user"
        assert conversation.history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_response_without_staged_message_raises(self):
        """get_response without staged message should raise."""
        client = _make_async_client()
        conversation = AsyncConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        with pytest.raises(ValueError, match="No user message staged"):
            await conversation.get_response()

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Multi-turn conversation should maintain history."""
        client = _make_async_client()
        conversation = AsyncConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        # First turn
        conversation.add_user_message("Hello")
        await conversation.get_response()
        
        # Second turn
        conversation.add_user_message("How are you?")
        await conversation.get_response()
        
        assert len(conversation.history) == 4
        assert conversation.history[0]["role"] == "user"
        assert conversation.history[1]["role"] == "assistant"
        assert conversation.history[2]["role"] == "user"
        assert conversation.history[3]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_reset_clears_history(self):
        """reset should clear history."""
        client = _make_async_client()
        conversation = AsyncConversationManager(client, model="claude-3-5-sonnet", max_tokens=1024)
        
        await conversation.get_response("Hello")
        conversation.reset()
        
        assert len(conversation.history) == 0

    @pytest.mark.asyncio
    async def test_accurate_token_counting(self):
        """Accurate token counting should call count_tokens."""
        client = _make_async_client(input_tokens=50, output_tokens=50)
        conversation = AsyncConversationManager(
            client,
            model="claude-3-5-sonnet",
            max_tokens=1024,
            context_window_limit=200,
            token_budget_headroom=0.1,
            accurate_token_counting=True,
        )
        
        await conversation.get_response("Hello")
        
        # count_tokens should have been called
        assert client.messages.count_tokens.called
