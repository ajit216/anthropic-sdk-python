"""Tests for ConversationManager and AsyncConversationManager."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, TextBlock
from anthropic.helpers import ConversationManager, AsyncConversationManager


class TestConversationManager:
    """Tests for ConversationManager."""

    def test_constructor_validation_invalid_client(self):
        """Test that invalid client raises ValueError."""
        with pytest.raises(ValueError, match="client must be an Anthropic instance"):
            ConversationManager(
                client="not_a_client",
                model="claude-opus-4-6",
                max_tokens=1024,
            )

    def test_constructor_validation_empty_model(self):
        """Test that empty model raises ValueError."""
        client = Mock(spec=Anthropic)
        with pytest.raises(ValueError, match="model cannot be empty"):
            ConversationManager(
                client=client,
                model="",
                max_tokens=1024,
            )

    def test_constructor_validation_invalid_max_tokens(self):
        """Test that non-positive max_tokens raises ValueError."""
        client = Mock(spec=Anthropic)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ConversationManager(
                client=client,
                model="claude-opus-4-6",
                max_tokens=0,
            )

    def test_constructor_validation_invalid_context_window(self):
        """Test that non-positive context_window_size raises ValueError."""
        client = Mock(spec=Anthropic)
        with pytest.raises(ValueError, match="context_window_size must be positive"):
            ConversationManager(
                client=client,
                model="claude-opus-4-6",
                max_tokens=1024,
                context_window_size=0,
            )

    def test_constructor_validation_invalid_reserve_tokens(self):
        """Test that negative reserve_tokens raises ValueError."""
        client = Mock(spec=Anthropic)
        with pytest.raises(ValueError, match="reserve_tokens cannot be negative"):
            ConversationManager(
                client=client,
                model="claude-opus-4-6",
                max_tokens=1024,
                reserve_tokens=-1,
            )

    def test_constructor_validation_max_tokens_exceeds_context_window(self):
        """Test that max_tokens > context_window_size raises ValueError."""
        client = Mock(spec=Anthropic)
        with pytest.raises(ValueError, match="max_tokens cannot exceed context_window_size"):
            ConversationManager(
                client=client,
                model="claude-opus-4-6",
                max_tokens=200000,
                context_window_size=10000,
            )

    def test_constructor_validation_reserve_plus_max_exceeds_context_window(self):
        """Test that reserve_tokens + max_tokens > context_window_size raises ValueError."""
        client = Mock(spec=Anthropic)
        with pytest.raises(ValueError, match="reserve_tokens \\+ max_tokens cannot exceed context_window_size"):
            ConversationManager(
                client=client,
                model="claude-opus-4-6",
                max_tokens=900,
                context_window_size=1000,
                reserve_tokens=150,
            )

    def test_constructor_success(self):
        """Test successful constructor."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(
            client=client,
            model="claude-opus-4-6",
            max_tokens=1024,
            system="You are helpful",
            context_window_size=200000,
            reserve_tokens=2000,
        )

        assert manager.client is client
        assert manager.model == "claude-opus-4-6"
        assert manager.max_tokens == 1024
        assert manager.system == "You are helpful"
        assert manager.context_window_size == 200000
        assert manager.reserve_tokens == 2000
        assert manager.conversation_history == []

    def test_add_user_message(self):
        """Test adding user messages."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client, model="claude-opus-4-6", max_tokens=1024)

        manager.add_user_message("Hello, Claude!")
        assert len(manager.conversation_history) == 1
        assert manager.conversation_history[0] == {"role": "user", "content": "Hello, Claude!"}

    def test_add_user_message_empty_content(self):
        """Test that empty user message raises ValueError."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client, model="claude-opus-4-6", max_tokens=1024)

        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_user_message("")

    def test_add_assistant_message(self):
        """Test adding assistant messages."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client, model="claude-opus-4-6", max_tokens=1024)

        manager.add_assistant_message("Hi there!")
        assert len(manager.conversation_history) == 1
        assert manager.conversation_history[0] == {"role": "assistant", "content": "Hi there!"}

    def test_add_assistant_message_empty_content(self):
        """Test that empty assistant message raises ValueError."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client, model="claude-opus-4-6", max_tokens=1024)

        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_assistant_message("")

    def test_get_messages(self):
        """Test getting conversation history."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client, model="claude-opus-4-6", max_tokens=1024)

        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")

        messages = manager.get_messages()
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi"}

        # Verify it's a copy
        messages.append({"role": "user", "content": "test"})
        assert len(manager.get_messages()) == 2

    def test_clear_history(self):
        """Test clearing conversation history."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client, model="claude-opus-4-6", max_tokens=1024)

        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        assert len(manager.conversation_history) == 2

        manager.clear_history()
        assert len(manager.conversation_history) == 0

    def test_estimate_tokens(self):
        """Test token estimation."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client, model="claude-opus-4-6", max_tokens=1024)

        # Rough approximation: 4 chars per token
        assert manager._estimate_tokens("hello") == 2  # 5 // 4 + 1
        assert manager._estimate_tokens("hello world test") == 5  # 16 // 4 + 1

    def test_truncate_history_with_single_message(self):
        """Test that truncation preserves at least one message."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(
            client=client,
            model="claude-opus-4-6",
            max_tokens=100,
            context_window_size=5000,
            reserve_tokens=0,
        )

        # Add a very long message that would exceed available tokens
        long_message = "x" * 100000
        manager.add_user_message(long_message)
        manager.add_assistant_message("response")

        manager._truncate_history()
        # Should keep at least the last message even if it exceeds limit
        assert len(manager.conversation_history) >= 1

    def test_create_response(self):
        """Test creating a response with the conversation history."""
        client = Mock(spec=Anthropic)
        mock_message = Mock(spec=Message)
        mock_content = Mock(spec=TextBlock)
        mock_content.text = "Response from Claude"
        mock_message.content = [mock_content]

        client.messages.create = Mock(return_value=mock_message)

        manager = ConversationManager(client=client, model="claude-opus-4-6", max_tokens=1024)
        manager.add_user_message("Hello")

        response = manager.create_response()

        assert response is mock_message
        # Verify the assistant message was added to history after the API call
        assert len(manager.conversation_history) == 2
        assert manager.conversation_history[0]["content"] == "Hello"
        assert manager.conversation_history[1]["content"] == "Response from Claude"

        # Verify the API was called with correct parameters
        client.messages.create.assert_called_once()
        call_args = client.messages.create.call_args
        assert call_args[1]["model"] == "claude-opus-4-6"
        assert call_args[1]["max_tokens"] == 1024
        # The message sent to the API should have been just the user message
        # But after getting the response, it's added to history
        assert len(call_args[1]["messages"]) >= 1

    def test_create_response_with_system_prompt(self):
        """Test creating a response with system prompt."""
        client = Mock(spec=Anthropic)
        mock_message = Mock(spec=Message)
        mock_content = Mock(spec=TextBlock)
        mock_content.text = "Response"
        mock_message.content = [mock_content]
        client.messages.create = Mock(return_value=mock_message)

        manager = ConversationManager(
            client=client,
            model="claude-opus-4-6",
            max_tokens=1024,
            system="You are helpful",
        )
        manager.add_user_message("Hello")

        response = manager.create_response()

        call_args = client.messages.create.call_args
        assert call_args[1]["system"] == "You are helpful"


class TestAsyncConversationManager:
    """Tests for AsyncConversationManager."""

    def test_constructor_validation_invalid_client(self):
        """Test that invalid client raises ValueError."""
        with pytest.raises(ValueError, match="client must be an AsyncAnthropic instance"):
            AsyncConversationManager(
                client="not_a_client",
                model="claude-opus-4-6",
                max_tokens=1024,
            )

    def test_constructor_validation_empty_model(self):
        """Test that empty model raises ValueError."""
        client = AsyncMock(spec=AsyncAnthropic)
        with pytest.raises(ValueError, match="model cannot be empty"):
            AsyncConversationManager(
                client=client,
                model="",
                max_tokens=1024,
            )

    def test_constructor_validation_invalid_max_tokens(self):
        """Test that non-positive max_tokens raises ValueError."""
        client = AsyncMock(spec=AsyncAnthropic)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            AsyncConversationManager(
                client=client,
                model="claude-opus-4-6",
                max_tokens=0,
            )

    def test_constructor_validation_invalid_context_window(self):
        """Test that non-positive context_window_size raises ValueError."""
        client = AsyncMock(spec=AsyncAnthropic)
        with pytest.raises(ValueError, match="context_window_size must be positive"):
            AsyncConversationManager(
                client=client,
                model="claude-opus-4-6",
                max_tokens=1024,
                context_window_size=0,
            )

    def test_constructor_validation_invalid_reserve_tokens(self):
        """Test that negative reserve_tokens raises ValueError."""
        client = AsyncMock(spec=AsyncAnthropic)
        with pytest.raises(ValueError, match="reserve_tokens cannot be negative"):
            AsyncConversationManager(
                client=client,
                model="claude-opus-4-6",
                max_tokens=1024,
                reserve_tokens=-1,
            )

    def test_constructor_validation_max_tokens_exceeds_context_window(self):
        """Test that max_tokens > context_window_size raises ValueError."""
        client = AsyncMock(spec=AsyncAnthropic)
        with pytest.raises(ValueError, match="max_tokens cannot exceed context_window_size"):
            AsyncConversationManager(
                client=client,
                model="claude-opus-4-6",
                max_tokens=200000,
                context_window_size=10000,
            )

    def test_constructor_validation_reserve_plus_max_exceeds_context_window(self):
        """Test that reserve_tokens + max_tokens > context_window_size raises ValueError."""
        client = AsyncMock(spec=AsyncAnthropic)
        with pytest.raises(ValueError, match="reserve_tokens \\+ max_tokens cannot exceed context_window_size"):
            AsyncConversationManager(
                client=client,
                model="claude-opus-4-6",
                max_tokens=900,
                context_window_size=1000,
                reserve_tokens=150,
            )

    def test_constructor_success(self):
        """Test successful constructor."""
        client = AsyncMock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(
            client=client,
            model="claude-opus-4-6",
            max_tokens=1024,
            system="You are helpful",
            context_window_size=200000,
            reserve_tokens=2000,
        )

        assert manager.client is client
        assert manager.model == "claude-opus-4-6"
        assert manager.max_tokens == 1024
        assert manager.system == "You are helpful"
        assert manager.context_window_size == 200000
        assert manager.reserve_tokens == 2000
        assert manager.conversation_history == []

    def test_add_user_message(self):
        """Test adding user messages."""
        client = AsyncMock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(
            client=client, model="claude-opus-4-6", max_tokens=1024
        )

        manager.add_user_message("Hello, Claude!")
        assert len(manager.conversation_history) == 1
        assert manager.conversation_history[0] == {"role": "user", "content": "Hello, Claude!"}

    def test_add_user_message_empty_content(self):
        """Test that empty user message raises ValueError."""
        client = AsyncMock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(
            client=client, model="claude-opus-4-6", max_tokens=1024
        )

        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_user_message("")

    def test_add_assistant_message(self):
        """Test adding assistant messages."""
        client = AsyncMock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(
            client=client, model="claude-opus-4-6", max_tokens=1024
        )

        manager.add_assistant_message("Hi there!")
        assert len(manager.conversation_history) == 1
        assert manager.conversation_history[0] == {"role": "assistant", "content": "Hi there!"}

    def test_add_assistant_message_empty_content(self):
        """Test that empty assistant message raises ValueError."""
        client = AsyncMock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(
            client=client, model="claude-opus-4-6", max_tokens=1024
        )

        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_assistant_message("")

    def test_get_messages(self):
        """Test getting conversation history."""
        client = AsyncMock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(
            client=client, model="claude-opus-4-6", max_tokens=1024
        )

        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")

        messages = manager.get_messages()
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi"}

        # Verify it's a copy
        messages.append({"role": "user", "content": "test"})
        assert len(manager.get_messages()) == 2

    def test_clear_history(self):
        """Test clearing conversation history."""
        client = AsyncMock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(
            client=client, model="claude-opus-4-6", max_tokens=1024
        )

        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        assert len(manager.conversation_history) == 2

        manager.clear_history()
        assert len(manager.conversation_history) == 0

    def test_estimate_tokens(self):
        """Test token estimation."""
        client = AsyncMock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(
            client=client, model="claude-opus-4-6", max_tokens=1024
        )

        # Rough approximation: 4 chars per token
        assert manager._estimate_tokens("hello") == 2  # 5 // 4 + 1
        assert manager._estimate_tokens("hello world test") == 5  # 16 // 4 + 1

    @pytest.mark.asyncio
    async def test_create_response(self):
        """Test creating a response with the conversation history."""
        client = AsyncMock(spec=AsyncAnthropic)
        mock_message = Mock(spec=Message)
        mock_content = Mock(spec=TextBlock)
        mock_content.text = "Response from Claude"
        mock_message.content = [mock_content]

        client.messages.create = AsyncMock(return_value=mock_message)

        manager = AsyncConversationManager(
            client=client, model="claude-opus-4-6", max_tokens=1024
        )
        manager.add_user_message("Hello")

        response = await manager.create_response()

        assert response is mock_message
        # Verify the assistant message was added to history after the API call
        assert len(manager.conversation_history) == 2
        assert manager.conversation_history[0]["content"] == "Hello"
        assert manager.conversation_history[1]["content"] == "Response from Claude"

        # Verify the API was called with correct parameters
        client.messages.create.assert_called_once()
        call_args = client.messages.create.call_args
        assert call_args[1]["model"] == "claude-opus-4-6"
        assert call_args[1]["max_tokens"] == 1024
        # The message sent to the API should have been just the user message
        # But after getting the response, it's added to history
        assert len(call_args[1]["messages"]) >= 1

    @pytest.mark.asyncio
    async def test_create_response_with_system_prompt(self):
        """Test creating a response with system prompt."""
        client = AsyncMock(spec=AsyncAnthropic)
        mock_message = Mock(spec=Message)
        mock_content = Mock(spec=TextBlock)
        mock_content.text = "Response"
        mock_message.content = [mock_content]
        client.messages.create = AsyncMock(return_value=mock_message)

        manager = AsyncConversationManager(
            client=client,
            model="claude-opus-4-6",
            max_tokens=1024,
            system="You are helpful",
        )
        manager.add_user_message("Hello")

        response = await manager.create_response()

        call_args = client.messages.create.call_args
        assert call_args[1]["system"] == "You are helpful"
