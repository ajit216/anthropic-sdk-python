"""Tests for ConversationManager and AsyncConversationManager helpers."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from anthropic.helpers import ConversationManager, AsyncConversationManager
from anthropic.types import Message, TextBlock, Usage


def _make_sync_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> MagicMock:
    """Create a mock sync client for testing."""
    client = MagicMock()
    
    # Mock the messages.create response
    response = MagicMock()
    response.content = [TextBlock(type="text", text=content_text)]
    response.usage = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    
    client.messages.create.return_value = response
    
    # Mock count_tokens response
    count_response = MagicMock()
    count_response.input_tokens = input_tokens
    client.messages.count_tokens.return_value = count_response
    
    return client


def _make_async_client(
    *, input_tokens: int = 100, output_tokens: int = 50, content_text: str = "Hello"
) -> AsyncMock:
    """Create a mock async client for testing."""
    client = AsyncMock()
    
    # Mock the messages.create response
    response = MagicMock()
    response.content = [TextBlock(type="text", text=content_text)]
    response.usage = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    
    client.messages.create.return_value = response
    
    # Mock count_tokens response
    count_response = MagicMock()
    count_response.input_tokens = input_tokens
    client.messages.count_tokens.return_value = count_response
    
    return client


class TestConversationManager:
    """Tests for ConversationManager."""

    def test_constructor_empty_model(self):
        """Test that empty model raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="model cannot be an empty string"):
            ConversationManager(client, model="", max_tokens=1024)

    def test_constructor_invalid_max_tokens(self):
        """Test that invalid max_tokens raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=0)
        
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ConversationManager(client, model="claude-3-5-sonnet-20241022", max_tokens=-1)

    def test_constructor_invalid_context_window_limit(self):
        """Test that invalid context_window_limit raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="context_window_limit must be >= 1 or None"):
            ConversationManager(
                client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                context_window_limit=0,
            )

    def test_constructor_invalid_token_budget_headroom(self):
        """Test that invalid token_budget_headroom raises ValueError."""
        client = _make_sync_client()
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                token_budget_headroom=-0.1,
            )
        
        with pytest.raises(ValueError, match="token_budget_headroom must be in"):
            ConversationManager(
                client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                token_budget_headroom=1.0,
            )

    def test_add_user_message_string(self):
        """Test adding a user message as string."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "Hello"

    def test_add_user_message_list(self):
        """Test adding a user message as content list."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        content = [{"type": "text", "text": "Hello"}]
        manager.add_user_message(content)
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == content

    def test_add_user_message_empty_string(self):
        """Test that empty string raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        with pytest.raises(ValueError, match="cannot be empty"):
            manager.add_user_message("")

    def test_add_user_message_empty_list(self):
        """Test that empty list raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        with pytest.raises(ValueError, match="cannot be empty"):
            manager.add_user_message([])

    def test_get_response_with_content(self):
        """Test getting response with content argument."""
        client = _make_sync_client(content_text="Response text")
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        response = manager.get_response("Hello")
        
        assert len(manager.history) == 2
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "Hello"
        assert manager.history[1]["role"] == "assistant"
        assert manager.last_usage is not None

    def test_get_response_without_content(self):
        """Test getting response with pre-staged message."""
        client = _make_sync_client(content_text="Response text")
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        manager.add_user_message("Hello")
        response = manager.get_response()
        
        assert len(manager.history) == 2
        assert manager.history[1]["role"] == "assistant"

    def test_get_response_no_staged_message(self):
        """Test that get_response without staged message raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        with pytest.raises(ValueError, match="No staged user message"):
            manager.get_response()

    def test_get_response_after_assistant_message(self):
        """Test that get_response after assistant message raises ValueError."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        manager.add_user_message("Hello")
        manager.get_response()
        
        with pytest.raises(ValueError, match="No staged user message"):
            manager.get_response()

    def test_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        manager.get_response("First question")
        assert len(manager.history) == 2
        
        manager.get_response("Second question")
        assert len(manager.history) == 4
        assert manager.history[0]["role"] == "user"
        assert manager.history[1]["role"] == "assistant"
        assert manager.history[2]["role"] == "user"
        assert manager.history[3]["role"] == "assistant"

    def test_last_usage_initially_none(self):
        """Test that last_usage is None initially."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        assert manager.last_usage is None

    def test_last_usage_set_after_response(self):
        """Test that last_usage is set after getting response."""
        client = _make_sync_client(input_tokens=150, output_tokens=75)
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        manager.get_response("Hello")
        assert manager.last_usage is not None
        assert manager.last_usage.input_tokens == 150
        assert manager.last_usage.output_tokens == 75

    def test_kwargs_forwarded_to_api(self):
        """Test that kwargs are forwarded to messages.create."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        manager.get_response("Hello", temperature=0.5)
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    def test_system_prompt_included(self):
        """Test that system prompt is included in API call."""
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

    def test_system_prompt_omitted_when_none(self):
        """Test that system prompt is omitted when None."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024, system=None
        )
        
        manager.get_response("Hello")
        
        call_kwargs = client.messages.create.call_args[1]
        assert "system" not in call_kwargs

    def test_history_returns_copy(self):
        """Test that history property returns a shallow copy."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        manager.add_user_message("Hello")
        history1 = manager.history
        history2 = manager.history
        
        assert history1 == history2
        assert history1 is not history2

    def test_reset_clears_history(self):
        """Test that reset clears history and last_usage."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        manager.get_response("Hello")
        assert len(manager.history) == 2
        assert manager.last_usage is not None
        
        manager.reset()
        
        assert len(manager.history) == 0
        assert manager.last_usage is None

    def test_truncation_disabled_when_limit_none(self):
        """Test that truncation is skipped when context_window_limit is None."""
        client = _make_sync_client()
        manager = ConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        manager.get_response("Message 1")
        manager.get_response("Message 2")
        
        assert len(manager.history) == 4
        client.messages.count_tokens.assert_not_called()

    def test_truncation_when_over_threshold(self):
        """Test truncation when over threshold with accurate mode."""
        client = _make_sync_client(input_tokens=600, output_tokens=50)
        
        # Set up count_tokens to return values that trigger truncation on second call
        # First call: threshold = 1000 * 0.5 = 500; we need < 500 so it passes
        count_response1 = MagicMock()
        count_response1.input_tokens = 300  # Under threshold, no truncation
        # Second call: now we have 2 messages; token estimate should be > threshold
        count_response2 = MagicMock()
        count_response2.input_tokens = 600  # Over threshold of 500, will truncate
        # After truncation: token count goes down
        count_response3 = MagicMock()
        count_response3.input_tokens = 350  # Under threshold after truncation
        
        client.messages.count_tokens.side_effect = [count_response1, count_response2, count_response3]
        
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.5,
            accurate_token_counting=True,
        )
        
        manager.get_response("Message 1")
        initial_count = len(manager.history)
        
        manager.get_response("Message 2")
        
        # Should have truncated some old messages
        assert len(manager.history) < initial_count + 2

    def test_truncation_raises_when_impossible(self):
        """Test that truncation raises ValueError when single pair exceeds limit."""
        client = _make_sync_client(input_tokens=800, output_tokens=500)
        
        # Set up count_tokens: first call passes, second call triggers truncation but fails
        count_response1 = MagicMock()
        count_response1.input_tokens = 500  # Under threshold of 1000, OK
        # Second call: high token count that triggers truncation attempt
        count_response2 = MagicMock()
        count_response2.input_tokens = 1100  # Over threshold
        # After first truncation (removing oldest pair), still too high
        count_response3 = MagicMock()
        count_response3.input_tokens = 1100  # Still over, will raise error
        
        client.messages.count_tokens.side_effect = [count_response1, count_response2, count_response3]
        
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.0,
            accurate_token_counting=True,
        )
        
        manager.get_response("Message 1")
        
        with pytest.raises(ValueError, match="Cannot truncate further"):
            manager.get_response("Message 2")

    def test_no_truncation_on_first_call(self):
        """Test no truncation on first call in heuristic mode."""
        client = _make_sync_client(input_tokens=900, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.0,
        )
        
        manager.get_response("Message 1")
        assert len(manager.history) == 2

    def test_truncation_accurate_mode(self):
        """Test truncation in accurate mode."""
        client = _make_sync_client(input_tokens=200, output_tokens=50)
        
        count_sequence = [200, 150]
        client.messages.count_tokens.side_effect = [
            MagicMock(input_tokens=count_seq) for count_seq in count_sequence
        ]
        
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.5,
            accurate_token_counting=True,
        )
        
        manager.get_response("Message 1")
        manager.get_response("Message 2")
        
        assert client.messages.count_tokens.call_count > 0

    def test_truncation_multiple_pairs(self):
        """Test truncation removes multiple pairs."""
        client = _make_sync_client(input_tokens=500, output_tokens=50)
        manager = ConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            context_window_limit=500,
            token_budget_headroom=0.1,
        )
        
        manager.get_response("Message 1")
        manager.get_response("Message 2")
        manager.get_response("Message 3")
        
        assert len(manager.history) >= 2


class TestAsyncConversationManager:
    """Tests for AsyncConversationManager."""

    @pytest.mark.asyncio
    async def test_constructor_validation(self):
        """Test constructor validation."""
        client = _make_async_client()
        
        with pytest.raises(ValueError, match="model cannot be an empty string"):
            AsyncConversationManager(client, model="", max_tokens=1024)

    @pytest.mark.asyncio
    async def test_add_user_message(self):
        """Test adding user message."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        manager.add_user_message("Hello")
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_response_with_content(self):
        """Test getting response with content."""
        client = _make_async_client(content_text="Response")
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        response = await manager.get_response("Hello")
        
        assert len(manager.history) == 2
        assert manager.last_usage is not None

    @pytest.mark.asyncio
    async def test_get_response_without_content(self):
        """Test getting response with pre-staged message."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        manager.add_user_message("Hello")
        await manager.get_response()
        
        assert len(manager.history) == 2

    @pytest.mark.asyncio
    async def test_get_response_no_staged_message(self):
        """Test that get_response without staged message raises ValueError."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        with pytest.raises(ValueError, match="No staged user message"):
            await manager.get_response()

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        await manager.get_response("First")
        await manager.get_response("Second")
        
        assert len(manager.history) == 4

    @pytest.mark.asyncio
    async def test_reset(self):
        """Test reset clears history."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        await manager.get_response("Hello")
        manager.reset()
        
        assert len(manager.history) == 0
        assert manager.last_usage is None

    @pytest.mark.asyncio
    async def test_truncation_accurate_mode(self):
        """Test truncation in accurate mode."""
        client = _make_async_client(input_tokens=200, output_tokens=50)
        
        count_response = MagicMock()
        count_response.input_tokens = 150
        client.messages.count_tokens.return_value = count_response
        
        manager = AsyncConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            context_window_limit=1000,
            token_budget_headroom=0.5,
            accurate_token_counting=True,
        )
        
        await manager.get_response("Message 1")
        await manager.get_response("Message 2")
        
        assert client.messages.count_tokens.call_count > 0

    @pytest.mark.asyncio
    async def test_system_prompt_included(self):
        """Test that system prompt is included in API call."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="You are helpful.",
        )
        
        await manager.get_response("Hello")
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_kwargs_forwarded(self):
        """Test that kwargs are forwarded to API."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        await manager.get_response("Hello", temperature=0.7)
        
        call_kwargs = client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_history_returns_copy(self):
        """Test that history returns a shallow copy."""
        client = _make_async_client()
        manager = AsyncConversationManager(
            client, model="claude-3-5-sonnet-20241022", max_tokens=1024
        )
        
        manager.add_user_message("Hello")
        history1 = manager.history
        history2 = manager.history
        
        assert history1 == history2
        assert history1 is not history2
