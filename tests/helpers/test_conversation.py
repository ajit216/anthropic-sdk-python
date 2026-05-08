"""Tests for ConversationManager and AsyncConversationManager helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import pytest

from anthropic.helpers import ConversationManager, AsyncConversationManager

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic


# ============================================================================
# ConversationManager Tests
# ============================================================================


class TestConversationManagerInit:
    """Test ConversationManager initialization and validation."""

    def test_init_valid(self) -> None:
        """Test successful initialization."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        assert manager.model == "claude-3-5-sonnet-20241022"
        assert manager.max_tokens == 4096
        assert manager.system_prompt is None

    def test_init_with_system_prompt(self) -> None:
        """Test initialization with system prompt."""
        system_msg = "You are a helpful assistant."
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system_prompt=system_msg,
        )
        assert manager.system_prompt == system_msg

    def test_init_empty_model(self) -> None:
        """Test that empty model raises ValueError."""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ConversationManager(model="", max_tokens=4096)

    def test_init_invalid_model_type(self) -> None:
        """Test that non-string model raises ValueError."""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ConversationManager(model=None, max_tokens=4096)  # type: ignore

    def test_init_invalid_max_tokens_zero(self) -> None:
        """Test that zero max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=0)

    def test_init_invalid_max_tokens_negative(self) -> None:
        """Test that negative max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=-100)

    def test_init_invalid_max_tokens_type(self) -> None:
        """Test that non-integer max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens="4096")  # type: ignore


class TestConversationManagerMessages:
    """Test message addition and retrieval."""

    def test_add_user_message(self) -> None:
        """Test adding a user message."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        manager.add_user_message("Hello")
        
        messages = manager.history
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_add_assistant_message(self) -> None:
        """Test adding an assistant message."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        manager.add_assistant_message("Hi there!")
        
        messages = manager.history
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hi there!"

    def test_add_message_with_role(self) -> None:
        """Test adding message using generic add_message method."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        manager.add_message("user", "Test message")
        manager.add_message("assistant", "Test response")
        
        messages = manager.history
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_add_message_invalid_role(self) -> None:
        """Test that invalid role raises ValueError."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        with pytest.raises(ValueError, match="role must be 'user' or 'assistant'"):
            manager.add_message("invalid", "content")

    def test_add_message_empty_content(self) -> None:
        """Test that empty content raises ValueError."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        with pytest.raises(ValueError, match="content must be a non-empty string"):
            manager.add_message("user", "")

    def test_add_message_invalid_content_type(self) -> None:
        """Test that non-string content raises ValueError."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        with pytest.raises(ValueError, match="content must be a non-empty string"):
            manager.add_message("user", None)  # type: ignore

    def test_get_messages_returns_copy(self) -> None:
        """Test that get_messages returns a copy, not reference."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        manager.add_user_message("Test")
        
        messages1 = manager.get_messages()
        messages2 = manager.get_messages()
        
        assert messages1 == messages2
        assert messages1 is not messages2  # Different objects

    def test_messages_property_alias(self) -> None:
        """Test that messages property is an alias for history."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        manager.add_user_message("Test")
        
        assert manager.messages == manager.history


class TestConversationManagerHistory:
    """Test history management."""

    def test_clear_history(self) -> None:
        """Test clearing message history."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        manager.add_user_message("Message 1")
        manager.add_assistant_message("Response 1")
        
        assert len(manager.history) == 2
        
        manager.clear_history()
        assert len(manager.history) == 0

    def test_history_is_copy(self) -> None:
        """Test that history property returns a copy."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        manager.add_user_message("Test")
        
        history = manager.history
        history.append({"role": "test", "content": "test"})
        
        # Original history should not be modified
        assert len(manager.history) == 1


class TestConversationManagerTruncation:
    """Test message truncation logic."""

    def test_token_estimation(self) -> None:
        """Test token estimation (1 token ≈ 4 chars)."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        
        # 4 characters should be ~1 token
        assert manager._estimate_tokens("test") == 1
        # 400 characters should be ~100 tokens
        assert manager._estimate_tokens("x" * 400) == 100
        # Empty string should be at least 1 token
        assert manager._estimate_tokens("") == 1

    def test_truncation_keeps_recent_messages(self) -> None:
        """Test that truncation keeps recent messages."""
        # Create manager with small token limit
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=100)
        
        # Add messages that will exceed the limit
        manager.add_user_message("a" * 80)  # ~20 tokens
        manager.add_assistant_message("b" * 80)  # ~20 tokens
        manager.add_user_message("c" * 80)  # ~20 tokens
        manager.add_assistant_message("d" * 80)  # ~20 tokens
        manager.add_user_message("e" * 80)  # ~20 tokens
        
        # History should be truncated
        assert len(manager.history) < 5
        
        # Most recent messages should be preserved
        last_message = manager.history[-1]
        assert last_message["content"] == "e" * 80

    def test_truncation_with_system_prompt(self) -> None:
        """Test that truncation accounts for system prompt tokens."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            system_prompt="x" * 80,  # ~20 tokens
        )
        
        # Add messages
        manager.add_user_message("y" * 80)  # ~20 tokens
        manager.add_assistant_message("z" * 80)  # ~20 tokens
        manager.add_user_message("a" * 80)  # ~20 tokens
        manager.add_assistant_message("b" * 80)  # ~20 tokens
        
        # With system prompt taking 20 tokens and 90% limit,
        # we have 90 tokens for messages
        messages = manager.history
        total_tokens = manager._estimate_tokens(manager.system_prompt or "")
        for msg in messages:
            total_tokens += manager._estimate_tokens(msg["content"])
        
        assert total_tokens <= 100


class TestConversationManagerAPICall:
    """Test API integration."""

    def test_create_message_basic(self) -> None:
        """Test creating a message with API call."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        
        # Mock the client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Assistant response")]
        mock_client.messages.create.return_value = mock_response
        
        response = manager.create_message("User input", mock_client)
        
        assert response == mock_response
        assert len(manager.history) == 2
        assert manager.history[0]["content"] == "User input"
        assert manager.history[1]["content"] == "Assistant response"

    def test_create_message_with_system_prompt(self) -> None:
        """Test that system prompt is passed to API."""
        system_msg = "You are helpful."
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system_prompt=system_msg,
        )
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        
        manager.create_message("Hello", mock_client)
        
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == system_msg

    def test_create_message_with_extra_kwargs(self) -> None:
        """Test that extra kwargs are passed to API."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        
        manager.create_message("Hello", mock_client, temperature=0.5, top_p=0.9)
        
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["top_p"] == 0.9

    def test_create_message_empty_response(self) -> None:
        """Test handling of empty response."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = []
        mock_client.messages.create.return_value = mock_response
        
        response = manager.create_message("Hello", mock_client)
        
        assert response == mock_response
        # User message added but not assistant response
        assert len(manager.history) == 1


# ============================================================================
# AsyncConversationManager Tests
# ============================================================================


class TestAsyncConversationManagerInit:
    """Test AsyncConversationManager initialization."""

    def test_init_valid(self) -> None:
        """Test successful initialization."""
        manager = AsyncConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        assert manager.model == "claude-3-5-sonnet-20241022"
        assert manager.max_tokens == 4096

    def test_init_with_system_prompt(self) -> None:
        """Test initialization with system prompt."""
        system_msg = "You are helpful."
        manager = AsyncConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system_prompt=system_msg,
        )
        assert manager.system_prompt == system_msg

    def test_init_validation_errors(self) -> None:
        """Test that validation errors work."""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            AsyncConversationManager(model="", max_tokens=4096)
        
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            AsyncConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=0)


class TestAsyncConversationManagerMessages:
    """Test async conversation message handling."""

    def test_add_user_message(self) -> None:
        """Test adding user message."""
        manager = AsyncConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        manager.add_user_message("Hello")
        
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "user"

    def test_add_assistant_message(self) -> None:
        """Test adding assistant message."""
        manager = AsyncConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        manager.add_assistant_message("Hi!")
        
        assert len(manager.history) == 1
        assert manager.history[0]["role"] == "assistant"

    def test_add_message_validation(self) -> None:
        """Test message validation."""
        manager = AsyncConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        
        with pytest.raises(ValueError, match="role must be 'user' or 'assistant'"):
            manager.add_message("invalid", "content")
        
        with pytest.raises(ValueError, match="content must be a non-empty string"):
            manager.add_message("user", "")

    def test_clear_history(self) -> None:
        """Test clearing history."""
        manager = AsyncConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        manager.add_user_message("Test")
        manager.clear_history()
        
        assert len(manager.history) == 0


class TestAsyncConversationManagerAPICall:
    """Test async API integration."""

    @pytest.mark.asyncio
    async def test_create_message_basic(self) -> None:
        """Test async message creation."""
        manager = AsyncConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        
        response = await manager.create_message("Hello", mock_client)
        
        assert response == mock_response
        assert len(manager.history) == 2

    @pytest.mark.asyncio
    async def test_create_message_with_system_prompt(self) -> None:
        """Test system prompt with async."""
        system_msg = "You are helpful."
        manager = AsyncConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system_prompt=system_msg,
        )
        
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        
        await manager.create_message("Hello", mock_client)
        
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == system_msg

    @pytest.mark.asyncio
    async def test_create_message_with_kwargs(self) -> None:
        """Test passing extra kwargs."""
        manager = AsyncConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response
        
        await manager.create_message("Hello", mock_client, temperature=0.5)
        
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5


class TestConversationManagerIntegration:
    """Integration tests for ConversationManager."""

    def test_multi_turn_conversation(self) -> None:
        """Test a complete multi-turn conversation."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=4096)
        mock_client = Mock()
        
        def create_response(content: str) -> Mock:
            response = Mock()
            response.content = [Mock(text=content)]
            return response
        
        mock_client.messages.create.side_effect = [
            create_response("I'll help with that."),
            create_response("Sure, here's the answer."),
            create_response("That makes sense."),
        ]
        
        # Turn 1
        manager.create_message("What can you help with?", mock_client)
        assert len(manager.history) == 2
        
        # Turn 2
        manager.create_message("Can you explain this?", mock_client)
        assert len(manager.history) == 4
        
        # Turn 3
        manager.create_message("I understand now.", mock_client)
        assert len(manager.history) == 6
        
        # Verify conversation flow
        assert manager.history[0]["role"] == "user"
        assert manager.history[0]["content"] == "What can you help with?"
        assert manager.history[-1]["role"] == "assistant"
        assert manager.history[-1]["content"] == "That makes sense."

    def test_conversation_with_token_limits(self) -> None:
        """Test conversation respects token limits through truncation."""
        manager = ConversationManager(model="claude-3-5-sonnet-20241022", max_tokens=200)
        
        # Add enough messages to trigger truncation
        for i in range(10):
            manager.add_user_message(f"User message {i}: " + "x" * 50)
            manager.add_assistant_message(f"Assistant response {i}: " + "y" * 50)
        
        # Should have truncated some messages
        assert len(manager.history) < 20
        
        # Recent messages should be preserved
        assert "User message 9" in manager.history[-2]["content"]
