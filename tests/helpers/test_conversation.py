"""Tests for the ConversationManager and AsyncConversationManager helpers."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from anthropic import Anthropic, AsyncAnthropic
from anthropic.helpers import ConversationManager, AsyncConversationManager
from anthropic.types import Message, ContentBlock
from anthropic.types.text_block import TextBlock


class TestConversationManager:
    """Tests for the synchronous ConversationManager."""

    def test_init_valid_params(self) -> None:
        """Test ConversationManager initialization with valid parameters."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            system_prompt="You are helpful."
        )
        
        assert manager._client is client
        assert manager._model == "claude-3-5-sonnet-20241022"
        assert manager._max_tokens == 8000
        assert manager._system_prompt == "You are helpful."
        assert manager._messages == []

    def test_init_default_model(self) -> None:
        """Test that default model is set correctly."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client)
        
        assert manager._model == "claude-3-5-sonnet-20241022"
        assert manager._max_tokens == 8000

    def test_init_invalid_max_tokens_zero(self) -> None:
        """Test that ValueError is raised for max_tokens=0."""
        client = Mock(spec=Anthropic)
        
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            ConversationManager(client=client, max_tokens=0)

    def test_init_invalid_max_tokens_negative(self) -> None:
        """Test that ValueError is raised for negative max_tokens."""
        client = Mock(spec=Anthropic)
        
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            ConversationManager(client=client, max_tokens=-100)

    def test_init_invalid_model_empty(self) -> None:
        """Test that ValueError is raised for empty model."""
        client = Mock(spec=Anthropic)
        
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ConversationManager(client=client, model="")

    def test_init_invalid_model_not_string(self) -> None:
        """Test that ValueError is raised for non-string model."""
        client = Mock(spec=Anthropic)
        
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ConversationManager(client=client, model=123)

    def test_add_user_message(self) -> None:
        """Test adding a user message."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client)
        
        manager.add_user_message("Hello!")
        
        assert len(manager._messages) == 1
        assert manager._messages[0]["role"] == "user"
        assert manager._messages[0]["content"] == "Hello!"

    def test_add_assistant_message(self) -> None:
        """Test adding an assistant message."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client)
        
        manager.add_assistant_message("Hi there!")
        
        assert len(manager._messages) == 1
        assert manager._messages[0]["role"] == "assistant"
        assert manager._messages[0]["content"] == "Hi there!"

    def test_add_message_invalid_role(self) -> None:
        """Test that ValueError is raised for invalid role."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client)
        
        with pytest.raises(ValueError, match='role must be either "user" or "assistant"'):
            manager.add_message("system", "test")

    def test_add_message_empty_content(self) -> None:
        """Test that ValueError is raised for empty content."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client)
        
        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_message("user", "")

    def test_add_message_non_string_content(self) -> None:
        """Test that ValueError is raised for non-string content."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client)
        
        with pytest.raises(ValueError, match="content must be a string"):
            manager.add_message("user", 123)

    def test_get_messages_empty(self) -> None:
        """Test getting messages when history is empty."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client)
        
        messages = manager.get_messages()
        
        assert messages == []

    def test_get_messages_multiple(self) -> None:
        """Test getting multiple messages."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client)
        
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        manager.add_user_message("How are you?")
        
        messages = manager.get_messages()
        
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi"}
        assert messages[2] == {"role": "user", "content": "How are you?"}

    def test_send_message(self) -> None:
        """Test sending a message and receiving a response."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client, max_tokens=1000)
        
        # Mock the API response
        mock_message = Mock(spec=Message)
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = "Hello from assistant!"
        mock_message.content = [mock_text_block]
        client.messages.create.return_value = mock_message
        
        response = manager.send_message("Hello!")
        
        assert response is mock_message
        assert len(manager._messages) == 2  # User message + assistant response
        assert manager._messages[0]["role"] == "user"
        assert manager._messages[1]["role"] == "assistant"
        assert manager._messages[1]["content"] == "Hello from assistant!"
        
        # Check that create was called with correct arguments
        client.messages.create.assert_called_once()
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == manager._model
        assert call_kwargs["max_tokens"] == manager._max_tokens
        assert len(call_kwargs["messages"]) == 1  # Only the user message is sent

    def test_truncation_basic(self) -> None:
        """Test that messages are truncated when exceeding threshold."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client, max_tokens=200)
        
        # Add messages with total tokens > 80% of 200 = 160 tokens
        manager.add_user_message("x" * 100)  # ~25 tokens
        assert len(manager._messages) == 1
        
        manager.add_user_message("y" * 100)  # ~25 tokens (total ~50)
        assert len(manager._messages) == 2
        
        manager.add_user_message("z" * 400)  # ~100 tokens (total ~150)
        assert len(manager._messages) == 3
        
        manager.add_user_message("a" * 200)  # ~50 tokens (total ~200, exceeds 160)
        # Should truncate oldest messages to stay under threshold
        assert len(manager._messages) == 2
        assert manager._messages[0]["content"].startswith("z")
        assert manager._messages[1]["content"].startswith("a")

    def test_truncation_respects_system_prompt(self) -> None:
        """Test that truncation accounts for system prompt tokens."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(
            client=client,
            max_tokens=100,
            system_prompt="x" * 200  # ~50 tokens
        )
        
        # With system prompt taking 50 tokens, threshold is 80 tokens
        # So messages can take ~30 tokens before truncation
        manager.add_user_message("a" * 100)  # ~25 tokens
        assert len(manager._messages) == 1
        
        manager.add_user_message("b" * 100)  # ~25 tokens (total ~50)
        # Should truncate oldest message since system prompt + messages > 80
        assert len(manager._messages) == 1

    def test_token_estimation(self) -> None:
        """Test token estimation calculation."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(client=client)
        
        # Test the estimation: ~4 chars per token
        assert manager._estimate_tokens("abcd") == 1
        assert manager._estimate_tokens("abcdefgh") == 2
        assert manager._estimate_tokens("x") == 1  # At least 1 token
        assert manager._estimate_tokens("") == 1  # At least 1 token

    def test_total_tokens_calculation(self) -> None:
        """Test total token calculation."""
        client = Mock(spec=Anthropic)
        manager = ConversationManager(
            client=client,
            system_prompt="system"  # ~2 tokens
        )
        
        manager.add_user_message("user")  # ~1 token
        manager.add_assistant_message("assistant")  # ~2 tokens
        
        total = manager._calculate_total_tokens()
        # Total should be ~5 tokens (2 + 1 + 2)
        assert total >= 4 and total <= 6


class TestAsyncConversationManager:
    """Tests for the asynchronous AsyncConversationManager."""

    def test_init_valid_params(self) -> None:
        """Test AsyncConversationManager initialization with valid parameters."""
        client = Mock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            system_prompt="You are helpful."
        )
        
        assert manager._client is client
        assert manager._model == "claude-3-5-sonnet-20241022"
        assert manager._max_tokens == 8000
        assert manager._system_prompt == "You are helpful."
        assert manager._messages == []

    def test_init_invalid_max_tokens(self) -> None:
        """Test that ValueError is raised for invalid max_tokens."""
        client = Mock(spec=AsyncAnthropic)
        
        with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
            AsyncConversationManager(client=client, max_tokens=0)

    def test_init_invalid_model(self) -> None:
        """Test that ValueError is raised for invalid model."""
        client = Mock(spec=AsyncAnthropic)
        
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            AsyncConversationManager(client=client, model="")

    def test_add_user_message(self) -> None:
        """Test adding a user message."""
        client = Mock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(client=client)
        
        manager.add_user_message("Hello!")
        
        assert len(manager._messages) == 1
        assert manager._messages[0]["role"] == "user"
        assert manager._messages[0]["content"] == "Hello!"

    def test_add_assistant_message(self) -> None:
        """Test adding an assistant message."""
        client = Mock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(client=client)
        
        manager.add_assistant_message("Hi there!")
        
        assert len(manager._messages) == 1
        assert manager._messages[0]["role"] == "assistant"
        assert manager._messages[0]["content"] == "Hi there!"

    def test_add_message_invalid_role(self) -> None:
        """Test that ValueError is raised for invalid role."""
        client = Mock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(client=client)
        
        with pytest.raises(ValueError, match='role must be either "user" or "assistant"'):
            manager.add_message("system", "test")

    def test_add_message_empty_content(self) -> None:
        """Test that ValueError is raised for empty content."""
        client = Mock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(client=client)
        
        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_message("user", "")

    def test_get_messages_empty(self) -> None:
        """Test getting messages when history is empty."""
        client = Mock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(client=client)
        
        messages = manager.get_messages()
        
        assert messages == []

    def test_get_messages_multiple(self) -> None:
        """Test getting multiple messages."""
        client = Mock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(client=client)
        
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        manager.add_user_message("How are you?")
        
        messages = manager.get_messages()
        
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi"}
        assert messages[2] == {"role": "user", "content": "How are you?"}

    @pytest.mark.asyncio
    async def test_send_message(self) -> None:
        """Test sending a message and receiving a response."""
        client = AsyncMock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(client=client, max_tokens=1000)
        
        # Mock the API response
        mock_message = Mock(spec=Message)
        mock_text_block = Mock(spec=TextBlock)
        mock_text_block.text = "Hello from assistant!"
        mock_message.content = [mock_text_block]
        
        # Setup async mock
        async def async_create(*args, **kwargs):
            return mock_message
        
        client.messages.create.side_effect = async_create
        
        response = await manager.send_message("Hello!")
        
        assert response is mock_message
        assert len(manager._messages) == 2  # User message + assistant response
        assert manager._messages[0]["role"] == "user"
        assert manager._messages[1]["role"] == "assistant"
        assert manager._messages[1]["content"] == "Hello from assistant!"
        
        # Check that create was called with correct arguments
        client.messages.create.assert_called_once()
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == manager._model
        assert call_kwargs["max_tokens"] == manager._max_tokens
        assert len(call_kwargs["messages"]) == 1  # Only the user message is sent

    def test_truncation_basic(self) -> None:
        """Test that messages are truncated when exceeding threshold."""
        client = Mock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(client=client, max_tokens=200)
        
        # Add messages with total tokens > 80% of 200 = 160 tokens
        manager.add_user_message("x" * 100)  # ~25 tokens
        assert len(manager._messages) == 1
        
        manager.add_user_message("y" * 100)  # ~25 tokens (total ~50)
        assert len(manager._messages) == 2
        
        manager.add_user_message("z" * 400)  # ~100 tokens (total ~150)
        assert len(manager._messages) == 3
        
        manager.add_user_message("a" * 200)  # ~50 tokens (total ~200, exceeds 160)
        # Should truncate oldest messages to stay under threshold
        assert len(manager._messages) == 2
        assert manager._messages[0]["content"].startswith("z")
        assert manager._messages[1]["content"].startswith("a")

    def test_token_estimation(self) -> None:
        """Test token estimation calculation."""
        client = Mock(spec=AsyncAnthropic)
        manager = AsyncConversationManager(client=client)
        
        # Test the estimation: ~4 chars per token
        assert manager._estimate_tokens("abcd") == 1
        assert manager._estimate_tokens("abcdefgh") == 2
        assert manager._estimate_tokens("x") == 1  # At least 1 token
