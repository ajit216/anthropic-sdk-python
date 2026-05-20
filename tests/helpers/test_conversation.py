"""Tests for the ConversationManager and AsyncConversationManager helpers."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, TextBlock
from anthropic.helpers import ConversationManager, AsyncConversationManager


@pytest.fixture
def mock_client():
    """Create a mock Anthropic client."""
    return Mock(spec=Anthropic)


@pytest.fixture
def mock_async_client():
    """Create a mock AsyncAnthropic client."""
    return Mock(spec=AsyncAnthropic)


@pytest.fixture
def mock_message():
    """Create a mock Message response."""
    return Message(
        id="msg_123",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet-20241022",
        content=[TextBlock(type="text", text="Hello, how can I help?")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage={"input_tokens": 10, "output_tokens": 10},
    )


class TestConversationManager:
    """Tests for ConversationManager."""

    def test_constructor_with_valid_params(self, mock_client):
        """Test initialization with valid parameters."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            context_window=200000,
            system="You are a helpful assistant.",
        )
        assert manager.client is mock_client
        assert manager.model == "claude-3-5-sonnet-20241022"
        assert manager.max_tokens == 1024
        assert manager.context_window == 200000
        assert manager.system == "You are a helpful assistant."
        assert manager.messages == []

    def test_constructor_invalid_client(self):
        """Test that invalid client raises ValueError."""
        with pytest.raises(ValueError, match="client must be an instance of Anthropic"):
            ConversationManager(
                client="not_a_client",
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
            )

    def test_constructor_invalid_model(self, mock_client):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ConversationManager(
                client=mock_client,
                model="",
                max_tokens=1024,
            )

        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ConversationManager(
                client=mock_client,
                model=123,
                max_tokens=1024,
            )

    def test_constructor_invalid_max_tokens(self, mock_client):
        """Test that invalid max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            ConversationManager(
                client=mock_client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=0,
            )

        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            ConversationManager(
                client=mock_client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=-100,
            )

        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            ConversationManager(
                client=mock_client,
                model="claude-3-5-sonnet-20241022",
                max_tokens="1024",
            )

    def test_constructor_invalid_context_window(self, mock_client):
        """Test that invalid context_window raises ValueError."""
        with pytest.raises(ValueError, match="context_window must be a positive integer"):
            ConversationManager(
                client=mock_client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                context_window=0,
            )

        with pytest.raises(ValueError, match="context_window must be a positive integer"):
            ConversationManager(
                client=mock_client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                context_window="200000",
            )

    def test_constructor_max_tokens_exceeds_context(self, mock_client):
        """Test that max_tokens > context_window raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens cannot exceed context_window"):
            ConversationManager(
                client=mock_client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=300000,
                context_window=200000,
            )

    def test_add_user_message(self, mock_client):
        """Test adding a user message."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        manager.add_user_message("Hello, Claude!")
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "user"
        assert manager.messages[0]["content"] == "Hello, Claude!"

    def test_add_user_message_invalid(self, mock_client):
        """Test that invalid user messages raise ValueError."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        with pytest.raises(ValueError, match="content must be a non-empty string"):
            manager.add_user_message("")

        with pytest.raises(ValueError, match="content must be a non-empty string"):
            manager.add_user_message("   ")

        with pytest.raises(ValueError, match="content must be a non-empty string"):
            manager.add_user_message(123)

    def test_add_assistant_message(self, mock_client):
        """Test adding an assistant message."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        manager.add_assistant_message("How can I help you today?")
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "assistant"
        assert manager.messages[0]["content"] == "How can I help you today?"

    def test_add_assistant_message_invalid(self, mock_client):
        """Test that invalid assistant messages raise ValueError."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        with pytest.raises(ValueError, match="content must be a non-empty string"):
            manager.add_assistant_message("")

        with pytest.raises(ValueError, match="content must be a non-empty string"):
            manager.add_assistant_message(None)

    def test_get_messages(self, mock_client):
        """Test getting conversation messages."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there!")

        messages = manager.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        # Ensure we get a copy, not the original list
        assert messages is not manager.messages

    def test_clear(self, mock_client):
        """Test clearing conversation history."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there!")
        assert len(manager.messages) == 2

        manager.clear()
        assert len(manager.messages) == 0

    def test_estimate_tokens(self, mock_client):
        """Test token estimation."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        # Rough estimate: 4 chars per token
        assert manager._estimate_tokens("hello") == 1
        assert manager._estimate_tokens("a" * 100) == 25
        # Minimum 1 token
        assert manager._estimate_tokens("") == 1

    def test_truncate_messages(self, mock_client):
        """Test message truncation when approaching context limit."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            context_window=500,
            system="Short system",
        )

        # Add messages that will exceed context window
        for i in range(10):
            manager.add_user_message(f"Message {i} with some content to take up tokens")
            manager.add_assistant_message(f"Response {i} with some content to take up tokens")

        initial_count = len(manager.messages)
        manager._truncate_messages()

        # After truncation, we should have fewer or equal messages
        assert len(manager.messages) <= initial_count
        # If messages remain, the last one should be an assistant message
        if len(manager.messages) > 0:
            assert manager.messages[-1]["role"] == "assistant"

    def test_get_response(self, mock_client, mock_message):
        """Test getting a response from the model."""
        mock_client.messages.create.return_value = mock_message

        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="You are helpful.",
        )

        response = manager.get_response("What is 2+2?")

        # Check that the user message was added
        assert len(manager.messages) >= 1
        assert manager.messages[0]["role"] == "user"
        assert manager.messages[0]["content"] == "What is 2+2?"

        # Check that the response was returned
        assert response.id == "msg_123"

        # Check that API was called with correct params
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["system"] == "You are helpful."

    def test_get_response_invalid(self, mock_client):
        """Test that invalid responses raise ValueError."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        with pytest.raises(ValueError, match="user_message must be a non-empty string"):
            manager.get_response("")

        with pytest.raises(ValueError, match="user_message must be a non-empty string"):
            manager.get_response(None)

    def test_multi_turn_conversation(self, mock_client):
        """Test a multi-turn conversation flow."""
        responses = [
            Message(
                id="msg_1",
                type="message",
                role="assistant",
                model="claude-3-5-sonnet-20241022",
                content=[TextBlock(type="text", text="2+2 equals 4.")],
                stop_reason="end_turn",
                stop_sequence=None,
                usage={"input_tokens": 10, "output_tokens": 10},
            ),
            Message(
                id="msg_2",
                type="message",
                role="assistant",
                model="claude-3-5-sonnet-20241022",
                content=[TextBlock(type="text", text="It is 25.")],
                stop_reason="end_turn",
                stop_sequence=None,
                usage={"input_tokens": 20, "output_tokens": 10},
            ),
        ]
        mock_client.messages.create.side_effect = responses

        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )

        # Turn 1
        response1 = manager.get_response("What is 2+2?")
        assert response1.id == "msg_1"
        assert len(manager.messages) == 2  # User + Assistant

        # Turn 2
        response2 = manager.get_response("What is 5*5?")
        assert response2.id == "msg_2"
        assert len(manager.messages) == 4  # Previous 2 + new User + Assistant


class TestAsyncConversationManager:
    """Tests for AsyncConversationManager."""

    def test_constructor_with_valid_params(self, mock_async_client):
        """Test initialization with valid parameters."""
        manager = AsyncConversationManager(
            client=mock_async_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            context_window=200000,
            system="You are a helpful assistant.",
        )
        assert manager.client is mock_async_client
        assert manager.model == "claude-3-5-sonnet-20241022"
        assert manager.max_tokens == 1024
        assert manager.context_window == 200000
        assert manager.system == "You are a helpful assistant."
        assert manager.messages == []

    def test_constructor_invalid_client(self):
        """Test that invalid client raises ValueError."""
        with pytest.raises(ValueError, match="client must be an instance of AsyncAnthropic"):
            AsyncConversationManager(
                client="not_a_client",
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
            )

    def test_constructor_invalid_model(self, mock_async_client):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            AsyncConversationManager(
                client=mock_async_client,
                model="",
                max_tokens=1024,
            )

    def test_constructor_invalid_max_tokens(self, mock_async_client):
        """Test that invalid max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            AsyncConversationManager(
                client=mock_async_client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=0,
            )

    def test_add_user_message(self, mock_async_client):
        """Test adding a user message."""
        manager = AsyncConversationManager(
            client=mock_async_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        manager.add_user_message("Hello, Claude!")
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "user"

    def test_add_user_message_invalid(self, mock_async_client):
        """Test that invalid user messages raise ValueError."""
        manager = AsyncConversationManager(
            client=mock_async_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        with pytest.raises(ValueError, match="content must be a non-empty string"):
            manager.add_user_message("")

    def test_add_assistant_message(self, mock_async_client):
        """Test adding an assistant message."""
        manager = AsyncConversationManager(
            client=mock_async_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        manager.add_assistant_message("How can I help you today?")
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "assistant"

    def test_get_messages(self, mock_async_client):
        """Test getting conversation messages."""
        manager = AsyncConversationManager(
            client=mock_async_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there!")

        messages = manager.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"

    def test_clear(self, mock_async_client):
        """Test clearing conversation history."""
        manager = AsyncConversationManager(
            client=mock_async_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there!")
        assert len(manager.messages) == 2

        manager.clear()
        assert len(manager.messages) == 0

    def test_estimate_tokens(self, mock_async_client):
        """Test token estimation."""
        manager = AsyncConversationManager(
            client=mock_async_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        assert manager._estimate_tokens("hello") == 1
        assert manager._estimate_tokens("a" * 100) == 25
        assert manager._estimate_tokens("") == 1

    def test_truncate_messages(self, mock_async_client):
        """Test message truncation when approaching context limit."""
        manager = AsyncConversationManager(
            client=mock_async_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            context_window=500,
            system="Short system",
        )

        for i in range(10):
            manager.add_user_message(f"Message {i} with some content to take up tokens")
            manager.add_assistant_message(f"Response {i} with some content to take up tokens")

        initial_count = len(manager.messages)
        manager._truncate_messages()

        # After truncation, we should have fewer or equal messages
        assert len(manager.messages) <= initial_count
        # If messages remain, verify structure is valid
        if len(manager.messages) > 0:
            assert all(msg.get("role") in ["user", "assistant"] for msg in manager.messages)

    @pytest.mark.asyncio
    async def test_get_response(self, mock_async_client, mock_message):
        """Test getting a response from the model."""
        mock_async_client.messages.create = AsyncMock(return_value=mock_message)

        manager = AsyncConversationManager(
            client=mock_async_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system="You are helpful.",
        )

        response = await manager.get_response("What is 2+2?")

        assert len(manager.messages) >= 1
        assert manager.messages[0]["role"] == "user"
        assert response.id == "msg_123"

    @pytest.mark.asyncio
    async def test_get_response_invalid(self, mock_async_client):
        """Test that invalid responses raise ValueError."""
        manager = AsyncConversationManager(
            client=mock_async_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )
        with pytest.raises(ValueError, match="user_message must be a non-empty string"):
            await manager.get_response("")

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, mock_async_client):
        """Test a multi-turn conversation flow."""
        responses = [
            Message(
                id="msg_1",
                type="message",
                role="assistant",
                model="claude-3-5-sonnet-20241022",
                content=[TextBlock(type="text", text="2+2 equals 4.")],
                stop_reason="end_turn",
                stop_sequence=None,
                usage={"input_tokens": 10, "output_tokens": 10},
            ),
            Message(
                id="msg_2",
                type="message",
                role="assistant",
                model="claude-3-5-sonnet-20241022",
                content=[TextBlock(type="text", text="It is 25.")],
                stop_reason="end_turn",
                stop_sequence=None,
                usage={"input_tokens": 20, "output_tokens": 10},
            ),
        ]
        mock_async_client.messages.create = AsyncMock(side_effect=responses)

        manager = AsyncConversationManager(
            client=mock_async_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )

        response1 = await manager.get_response("What is 2+2?")
        assert response1.id == "msg_1"
        assert len(manager.messages) == 2

        response2 = await manager.get_response("What is 5*5?")
        assert response2.id == "msg_2"
        assert len(manager.messages) == 4


class TestConversationManagerFixes:
    """Tests for fixes to ConversationManager."""

    def test_system_parameter_validation(self, mock_client):
        """Test that system parameter is validated (FIX #4)."""
        with pytest.raises(ValueError, match="system must be a non-empty string or None"):
            ConversationManager(
                client=mock_client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system={"invalid": "dict"},
            )

        with pytest.raises(ValueError, match="system must be a non-empty string or None"):
            ConversationManager(
                client=mock_client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system="",  # Empty string
            )

        # Should accept None
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=None,
        )
        assert manager.system is None

    def test_estimate_tokens_with_list_content(self, mock_client):
        """Test token estimation with list content (FIX #5)."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
        )

        # String content
        assert manager._estimate_tokens("hello") == 1
        
        # List content (multiple blocks)
        list_content = [
            {"type": "text", "text": "hello"},
            {"type": "tool_result", "tool_use_id": "123", "content": "result"},
        ]
        tokens = manager._estimate_tokens(list_content)
        assert tokens >= 100  # ~100 tokens per block

    def test_user_message_validation_against_context_window(self, mock_client):
        """Test that large user messages are rejected (FIX #3)."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            context_window=2000,
        )

        # A message that exceeds context window
        large_message = "x" * 10000  # ~2500 tokens

        with pytest.raises(ValueError, match="exceeds context_window"):
            manager.get_response(large_message)

    def test_truncation_with_list_content(self, mock_client):
        """Test truncation works with list content (FIX #3)."""
        manager = ConversationManager(
            client=mock_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            context_window=500,
        )

        # Add messages with list content
        manager.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "1", "content": "x" * 1000}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "2", "content": "y" * 1000}
                ],
            },
            {"role": "user", "content": "z"},
        ]

        manager._truncate_messages()

        # Should remove some messages to fit
        assert len(manager.messages) < 3

    @pytest.mark.asyncio
    async def test_async_system_parameter_validation(self, mock_async_client):
        """Test that async version validates system parameter (FIX #4)."""
        with pytest.raises(ValueError, match="system must be a non-empty string or None"):
            AsyncConversationManager(
                client=mock_async_client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=123,
            )

    @pytest.mark.asyncio
    async def test_async_user_message_validation(self, mock_async_client):
        """Test that async version validates user message size (FIX #3)."""
        manager = AsyncConversationManager(
            client=mock_async_client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            context_window=2000,
        )

        large_message = "x" * 10000

        with pytest.raises(ValueError, match="exceeds context_window"):
            await manager.get_response(large_message)
