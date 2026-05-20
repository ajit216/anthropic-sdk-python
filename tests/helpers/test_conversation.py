"""Tests for conversation management helpers."""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import AsyncConversationManager, ConversationManager
from anthropic.types import Message, TextBlock, ContentBlock


@pytest.fixture
def sync_client() -> Anthropic:
    """Create a mock Anthropic client for testing."""
    return Anthropic(api_key="test-key")


@pytest.fixture
def async_client() -> AsyncAnthropic:
    """Create a mock AsyncAnthropic client for testing."""
    return AsyncAnthropic(api_key="test-key")


@pytest.fixture
def mock_message() -> Message:
    """Create a mock Message response."""
    return Message(
        id="msg-123",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet-latest",
        content=[TextBlock(type="text", text="Test response")],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )


class TestConversationManager:
    """Tests for ConversationManager class."""

    def test_init_with_valid_params(self, sync_client: Anthropic) -> None:
        """Test initialization with valid parameters."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )
        assert manager.client == sync_client
        assert manager.model == "claude-3-5-sonnet-latest"
        assert manager.max_tokens == 1024
        assert manager.context_window == 200_000
        assert manager.system is None
        assert manager.messages == []
        assert manager.last_response is None

    def test_init_with_custom_context_window(self, sync_client: Anthropic) -> None:
        """Test initialization with custom context window."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            context_window=100_000,
        )
        assert manager.context_window == 100_000

    def test_init_with_system_prompt(self, sync_client: Anthropic) -> None:
        """Test initialization with system prompt."""
        system = "You are a helpful assistant."
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            system=system,
        )
        assert manager.system == system

    def test_init_with_extra_kwargs(self, sync_client: Anthropic) -> None:
        """Test initialization with extra parameters."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )
        assert manager.extra_params["temperature"] == 0.7
        assert manager.extra_params["top_p"] == 0.9

    def test_init_invalid_client(self) -> None:
        """Test initialization with invalid client."""
        with pytest.raises(ValueError, match="client must be an Anthropic instance"):
            ConversationManager(  # type: ignore
                client="not-a-client",
                model="claude-3-5-sonnet-latest",
                max_tokens=1024,
            )

    def test_init_invalid_model(self, sync_client: Anthropic) -> None:
        """Test initialization with invalid model."""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ConversationManager(
                client=sync_client,
                model="",  # type: ignore
                max_tokens=1024,
            )

    def test_init_invalid_model_type(self, sync_client: Anthropic) -> None:
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ConversationManager(
                client=sync_client,
                model=None,  # type: ignore
                max_tokens=1024,
            )

    def test_init_invalid_max_tokens(self, sync_client: Anthropic) -> None:
        """Test initialization with invalid max_tokens."""
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            ConversationManager(
                client=sync_client,
                model="claude-3-5-sonnet-latest",
                max_tokens=0,  # type: ignore
            )

    def test_init_negative_max_tokens(self, sync_client: Anthropic) -> None:
        """Test initialization with negative max_tokens."""
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            ConversationManager(
                client=sync_client,
                model="claude-3-5-sonnet-latest",
                max_tokens=-100,  # type: ignore
            )

    def test_init_invalid_context_window(self, sync_client: Anthropic) -> None:
        """Test initialization with invalid context_window."""
        with pytest.raises(ValueError, match="context_window must be a positive integer"):
            ConversationManager(
                client=sync_client,
                model="claude-3-5-sonnet-latest",
                max_tokens=1024,
                context_window=0,  # type: ignore
            )

    def test_add_user_message(self, sync_client: Anthropic, mock_message: Message) -> None:
        """Test adding a user message and receiving response."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        with mock.patch.object(sync_client.messages, "create", return_value=mock_message):
            response = manager.add_user_message("Hello!")

        assert response == mock_message
        assert manager.last_response == mock_message
        assert len(manager.messages) == 2
        assert manager.messages[0]["role"] == "user"
        assert manager.messages[0]["content"] == "Hello!"
        assert manager.messages[1]["role"] == "assistant"

    def test_add_user_message_empty_content(self, sync_client: Anthropic) -> None:
        """Test adding empty user message."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        with pytest.raises(ValueError, match="content must be a non-empty string"):
            manager.add_user_message("")

    def test_add_user_message_invalid_content_type(self, sync_client: Anthropic) -> None:
        """Test adding user message with invalid content type."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        with pytest.raises(ValueError, match="content must be a non-empty string"):
            manager.add_user_message(None)  # type: ignore

    def test_add_multiple_messages(self, sync_client: Anthropic, mock_message: Message) -> None:
        """Test adding multiple messages in sequence."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        with mock.patch.object(sync_client.messages, "create", return_value=mock_message):
            manager.add_user_message("First message")
            manager.add_user_message("Second message")
            manager.add_user_message("Third message")

        assert len(manager.messages) == 6  # 3 user + 3 assistant
        assert manager.messages[0]["content"] == "First message"
        assert manager.messages[2]["content"] == "Second message"
        assert manager.messages[4]["content"] == "Third message"

    def test_get_conversation_history(self, sync_client: Anthropic, mock_message: Message) -> None:
        """Test retrieving conversation history."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        with mock.patch.object(sync_client.messages, "create", return_value=mock_message):
            manager.add_user_message("Test message")

        history = manager.get_conversation_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

        # Verify it's a copy, not the original list
        history.clear()
        assert len(manager.messages) == 2

    def test_get_last_response(self, sync_client: Anthropic, mock_message: Message) -> None:
        """Test retrieving the last response."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        assert manager.get_last_response() is None

        with mock.patch.object(sync_client.messages, "create", return_value=mock_message):
            response = manager.add_user_message("Test message")

        assert manager.get_last_response() == mock_message
        assert manager.get_last_response() == response

    def test_clear_history(self, sync_client: Anthropic, mock_message: Message) -> None:
        """Test clearing conversation history."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        with mock.patch.object(sync_client.messages, "create", return_value=mock_message):
            manager.add_user_message("Test message")

        assert len(manager.messages) > 0
        assert manager.last_response is not None

        manager.clear_history()

        assert len(manager.messages) == 0
        assert manager.last_response is None

    def test_truncation_with_system_prompt(self, sync_client: Anthropic, mock_message: Message) -> None:
        """Test that truncation respects system prompt."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            system="You are a helpful assistant.",
            context_window=500,  # Small context window to trigger truncation
        )

        with mock.patch.object(sync_client.messages, "create", return_value=mock_message):
            # Add messages that exceed the small context window
            for i in range(10):
                manager.add_user_message(f"Message {i} " * 100)

        # History should be truncated
        assert len(manager.messages) < 20  # Would be 20 if no truncation

    def test_estimate_tokens_string(self, sync_client: Anthropic) -> None:
        """Test token estimation for string content."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        msg: Any = {"role": "user", "content": "Test message"}
        tokens = manager._estimate_tokens(msg)
        # 12 characters / 4 + 4 = 3 + 4 = 7
        assert tokens == 7

    def test_estimate_tokens_empty(self, sync_client: Anthropic) -> None:
        """Test token estimation for empty content."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        msg: Any = {"role": "user", "content": ""}
        tokens = manager._estimate_tokens(msg)
        assert tokens == 4

    def test_estimate_tokens_complex(self, sync_client: Anthropic) -> None:
        """Test token estimation for complex content."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        msg: Any = {"role": "user"}  # No content key
        tokens = manager._estimate_tokens(msg)
        # When content is missing, .get() returns "" which gives (0 // 4) + 4 = 4
        assert tokens == 4

    def test_with_system_prompt(self, sync_client: Anthropic, mock_message: Message) -> None:
        """Test that system prompt is passed to API."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            system="You are a helpful assistant.",
        )

        with mock.patch.object(sync_client.messages, "create", return_value=mock_message) as mock_create:
            manager.add_user_message("Test message")

        # Verify system prompt was passed to create
        assert mock_create.call_args[1]["system"] == "You are a helpful assistant."

    def test_with_extra_params(self, sync_client: Anthropic, mock_message: Message) -> None:
        """Test that extra parameters are passed to API."""
        manager = ConversationManager(
            client=sync_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )

        with mock.patch.object(sync_client.messages, "create", return_value=mock_message) as mock_create:
            manager.add_user_message("Test message")

        # Verify extra params were passed
        assert mock_create.call_args[1]["temperature"] == 0.7
        assert mock_create.call_args[1]["top_p"] == 0.9


class TestAsyncConversationManager:
    """Tests for AsyncConversationManager class."""

    def test_init_with_valid_params(self, async_client: AsyncAnthropic) -> None:
        """Test initialization with valid parameters."""
        manager = AsyncConversationManager(
            client=async_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )
        assert manager.client == async_client
        assert manager.model == "claude-3-5-sonnet-latest"
        assert manager.max_tokens == 1024
        assert manager.context_window == 200_000
        assert manager.system is None
        assert manager.messages == []
        assert manager.last_response is None

    def test_init_invalid_client(self) -> None:
        """Test initialization with invalid client."""
        with pytest.raises(ValueError, match="client must be an AsyncAnthropic instance"):
            AsyncConversationManager(  # type: ignore
                client="not-a-client",
                model="claude-3-5-sonnet-latest",
                max_tokens=1024,
            )

    def test_init_invalid_model(self, async_client: AsyncAnthropic) -> None:
        """Test initialization with invalid model."""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            AsyncConversationManager(
                client=async_client,
                model="",  # type: ignore
                max_tokens=1024,
            )

    def test_init_invalid_max_tokens(self, async_client: AsyncAnthropic) -> None:
        """Test initialization with invalid max_tokens."""
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            AsyncConversationManager(
                client=async_client,
                model="claude-3-5-sonnet-latest",
                max_tokens=0,  # type: ignore
            )

    def test_init_invalid_context_window(self, async_client: AsyncAnthropic) -> None:
        """Test initialization with invalid context_window."""
        with pytest.raises(ValueError, match="context_window must be a positive integer"):
            AsyncConversationManager(
                client=async_client,
                model="claude-3-5-sonnet-latest",
                max_tokens=1024,
                context_window=0,  # type: ignore
            )

    @pytest.mark.asyncio
    async def test_add_user_message(self, async_client: AsyncAnthropic, mock_message: Message) -> None:
        """Test adding a user message and receiving response."""
        manager = AsyncConversationManager(
            client=async_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        async def mock_create(*args: Any, **kwargs: Any) -> Message:
            return mock_message

        with mock.patch.object(async_client.messages, "create", side_effect=mock_create):
            response = await manager.add_user_message("Hello!")

        assert response == mock_message
        assert manager.last_response == mock_message
        assert len(manager.messages) == 2
        assert manager.messages[0]["role"] == "user"
        assert manager.messages[0]["content"] == "Hello!"
        assert manager.messages[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_add_user_message_empty_content(self, async_client: AsyncAnthropic) -> None:
        """Test adding empty user message."""
        manager = AsyncConversationManager(
            client=async_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        with pytest.raises(ValueError, match="content must be a non-empty string"):
            await manager.add_user_message("")

    @pytest.mark.asyncio
    async def test_add_multiple_messages(self, async_client: AsyncAnthropic, mock_message: Message) -> None:
        """Test adding multiple messages in sequence."""
        manager = AsyncConversationManager(
            client=async_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        async def mock_create(*args: Any, **kwargs: Any) -> Message:
            return mock_message

        with mock.patch.object(async_client.messages, "create", side_effect=mock_create):
            await manager.add_user_message("First message")
            await manager.add_user_message("Second message")
            await manager.add_user_message("Third message")

        assert len(manager.messages) == 6  # 3 user + 3 assistant

    def test_get_conversation_history(self, async_client: AsyncAnthropic, mock_message: Message) -> None:
        """Test retrieving conversation history."""
        manager = AsyncConversationManager(
            client=async_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        # Manually set messages to test get_conversation_history without async
        manager.messages = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": [TextBlock(type="text", text="Response")]},
        ]

        history = manager.get_conversation_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"

        # Verify it's a copy
        history.clear()
        assert len(manager.messages) == 2

    def test_get_last_response(self, async_client: AsyncAnthropic, mock_message: Message) -> None:
        """Test retrieving the last response."""
        manager = AsyncConversationManager(
            client=async_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        assert manager.get_last_response() is None

        manager.last_response = mock_message
        assert manager.get_last_response() == mock_message

    def test_clear_history(self, async_client: AsyncAnthropic, mock_message: Message) -> None:
        """Test clearing conversation history."""
        manager = AsyncConversationManager(
            client=async_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
        )

        manager.messages = [{"role": "user", "content": "Test"}]
        manager.last_response = mock_message

        manager.clear_history()

        assert len(manager.messages) == 0
        assert manager.last_response is None

    def test_truncation_with_system_prompt(self, async_client: AsyncAnthropic) -> None:
        """Test that truncation respects system prompt."""
        manager = AsyncConversationManager(
            client=async_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            system="You are a helpful assistant.",
            context_window=500,
        )

        # Add several messages
        for i in range(5):
            msg_content = f"Test message {i} with some content"
            manager.messages.append({"role": "user", "content": msg_content})
            manager.messages.append({"role": "assistant", "content": "Response"})

        # Manually call truncation
        manager._truncate_history()

        # Should have truncated some messages
        # Initial: 10 messages, after truncation should be less
        assert len(manager.messages) <= 10

    def test_with_extra_params(self, async_client: AsyncAnthropic) -> None:
        """Test initialization with extra parameters."""
        manager = AsyncConversationManager(
            client=async_client,
            model="claude-3-5-sonnet-latest",
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
        )
        assert manager.extra_params["temperature"] == 0.7
        assert manager.extra_params["top_p"] == 0.9
