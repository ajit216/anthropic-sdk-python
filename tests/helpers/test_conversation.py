"""Tests for ConversationManager and AsyncConversationManager helpers."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from anthropic.helpers.conversation import (
    ConversationManager,
    AsyncConversationManager,
    _estimate_token_count,
    _count_message_tokens,
)
from anthropic.types import Message, ContentBlock
from anthropic.types.text_block import TextBlock


class TestTokenEstimation:
    """Test token counting utilities."""
    
    def test_estimate_token_count_empty_string(self):
        """Empty string should count as at least 1 token."""
        assert _estimate_token_count("") == 1
    
    def test_estimate_token_count_four_characters(self):
        """Four characters should be 1 token."""
        assert _estimate_token_count("test") == 1
    
    def test_estimate_token_count_longer_text(self):
        """Longer text should scale appropriately."""
        # 100 characters should be ~25 tokens
        text = "a" * 100
        assert _estimate_token_count(text) == 25
    
    def test_count_message_tokens_string_content(self):
        """Count tokens for message with string content."""
        message = {"role": "user", "content": "Hello world"}
        tokens = _count_message_tokens(message)
        assert tokens >= 1
    
    def test_count_message_tokens_list_content(self):
        """Count tokens for message with list content."""
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]
        }
        tokens = _count_message_tokens(message)
        assert tokens >= 1


class TestConversationManager:
    """Test synchronous ConversationManager."""
    
    def test_constructor_valid_params(self):
        """Constructor accepts valid parameters."""
        client = Mock()
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
            system="You are a helpful assistant.",
        )
        assert cm.client is client
        assert cm.max_tokens == 2000
        assert cm.model == "claude-3-5-sonnet-20241022"
        assert cm.system == "You are a helpful assistant."
    
    def test_constructor_invalid_max_tokens_zero(self):
        """Constructor raises ValueError for max_tokens = 0."""
        client = Mock()
        with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
            ConversationManager(
                client=client,
                max_tokens=0,
                model="claude-3-5-sonnet-20241022",
            )
    
    def test_constructor_invalid_max_tokens_negative(self):
        """Constructor raises ValueError for negative max_tokens."""
        client = Mock()
        with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
            ConversationManager(
                client=client,
                max_tokens=-100,
                model="claude-3-5-sonnet-20241022",
            )
    
    def test_add_user_message(self):
        """Add user message to conversation history."""
        client = Mock()
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        cm.add_user_message("Hello")
        
        assert len(cm.messages) == 1
        assert cm.messages[0]["role"] == "user"
        assert cm.messages[0]["content"] == "Hello"
    
    def test_add_assistant_message(self):
        """Add assistant message to conversation history."""
        client = Mock()
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        cm.add_assistant_message("Hi there!")
        
        assert len(cm.messages) == 1
        assert cm.messages[0]["role"] == "assistant"
        assert cm.messages[0]["content"] == "Hi there!"
    
    def test_get_messages(self):
        """Get current conversation history."""
        client = Mock()
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        cm.add_user_message("Hello")
        cm.add_assistant_message("Hi!")
        
        messages = cm.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
    
    def test_get_conversation_tokens(self):
        """Get estimated token count."""
        client = Mock()
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        cm.add_user_message("Hello world")
        
        tokens = cm.get_conversation_tokens()
        assert tokens >= 1
    
    def test_truncation_removes_oldest_messages(self):
        """Truncation removes oldest messages when exceeding max_tokens."""
        client = Mock()
        cm = ConversationManager(
            client=client,
            max_tokens=10,  # Very small limit to trigger truncation
            model="claude-3-5-sonnet-20241022",
        )
        
        # Add messages that will exceed the limit
        cm.add_user_message("Message 1")
        cm.add_assistant_message("Response 1")
        cm.add_user_message("Message 2")
        cm.add_assistant_message("Response 2")
        
        # Should have truncated some older messages
        assert len(cm.messages) < 4
    
    def test_clear_history(self):
        """Clear conversation history."""
        client = Mock()
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        cm.add_user_message("Hello")
        cm.add_assistant_message("Hi!")
        
        assert len(cm.messages) == 2
        cm.clear_history()
        assert len(cm.messages) == 0
    
    def test_create_message(self):
        """Create message via API."""
        client = Mock()
        
        # Mock the response
        mock_response = Mock(spec=Message)
        mock_response.content = [Mock(spec=TextBlock)]
        mock_response.content[0].text = "Response text"
        client.messages.create.return_value = mock_response
        
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
            system="Test system",
        )
        
        response = cm.create_message("Hello, Claude!")
        
        assert response is mock_response
        # Should have added user message and assistant response to history
        assert len(cm.messages) == 2
        assert cm.messages[0]["role"] == "user"
        assert cm.messages[1]["role"] == "assistant"
        
        # Verify API was called with correct parameters
        call_args = client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-5-sonnet-20241022"
        assert call_args[1]["system"] == "Test system"
    
    def test_create_message_without_system(self):
        """Create message without system prompt."""
        client = Mock()
        mock_response = Mock(spec=Message)
        mock_response.content = []
        client.messages.create.return_value = mock_response
        
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        
        response = cm.create_message("Hello!")
        
        # Verify system prompt is not included
        call_args = client.messages.create.call_args
        assert "system" not in call_args[1]
    
    def test_create_message_multiple_content_blocks(self):
        """Handle response with multiple content blocks."""
        client = Mock()
        
        # Mock response with multiple text blocks
        mock_response = Mock(spec=Message)
        block1 = Mock(spec=TextBlock)
        block1.text = "Part 1"
        block2 = Mock(spec=TextBlock)
        block2.text = " Part 2"
        mock_response.content = [block1, block2]
        client.messages.create.return_value = mock_response
        
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        
        response = cm.create_message("Hello!")
        
        # Assistant message should contain concatenated text
        assert len(cm.messages) == 2
        assistant_msg = cm.messages[1]
        assert assistant_msg["content"] == "Part 1 Part 2"
    
    def test_create_message_with_custom_kwargs(self):
        """Create message with custom kwargs passed to API."""
        client = Mock()
        mock_response = Mock(spec=Message)
        mock_response.content = []
        client.messages.create.return_value = mock_response
        
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        
        response = cm.create_message(
            "Hello!",
            max_tokens=1024,
            temperature=0.7,
        )
        
        # Verify custom kwargs were passed
        call_args = client.messages.create.call_args
        assert call_args[1]["max_tokens"] == 1024
        assert call_args[1]["temperature"] == 0.7
    
    def test_constructor_invalid_client_none(self):
        """Constructor raises ValueError for None client."""
        with pytest.raises(ValueError, match="client must not be None"):
            ConversationManager(
                client=None,
                max_tokens=2000,
                model="claude-3-5-sonnet-20241022",
            )
    
    def test_constructor_invalid_model_empty(self):
        """Constructor raises ValueError for empty model."""
        client = Mock()
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ConversationManager(
                client=client,
                max_tokens=2000,
                model="",
            )
    
    def test_constructor_invalid_model_none(self):
        """Constructor raises ValueError for None model."""
        client = Mock()
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            ConversationManager(
                client=client,
                max_tokens=2000,
                model=None,
            )
    
    def test_create_message_empty_response_content(self):
        """Response with empty content should not add empty message."""
        client = Mock()
        mock_response = Mock(spec=Message)
        mock_response.content = []
        client.messages.create.return_value = mock_response
        
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        
        response = cm.create_message("Hello!")
        
        # Should NOT have added empty assistant message
        assert len(cm.messages) == 1  # Only user message
    
    def test_create_message_non_text_blocks_only(self):
        """Response with only non-text blocks should not add empty message."""
        client = Mock()
        mock_response = Mock(spec=Message)
        
        # Create a block without text attribute
        tool_block = Mock()
        # Don't set text attribute - hasattr will return False
        mock_response.content = [tool_block]
        client.messages.create.return_value = mock_response
        
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        
        response = cm.create_message("Use a tool")
        
        # Should NOT have added empty assistant message
        assert len(cm.messages) == 1  # Only user message
    
    def test_create_message_stream_not_supported(self):
        """create_message should raise ValueError if stream=True is passed."""
        client = Mock()
        cm = ConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        
        with pytest.raises(ValueError, match="Streaming is not supported"):
            cm.create_message("Hello!", stream=True)
    
    def test_truncation_preserves_minimum_context(self):
        """Truncation should preserve at least 2 messages for minimum context."""
        client = Mock()
        cm = ConversationManager(
            client=client,
            max_tokens=5,  # Very small to force truncation
            model="claude-3-5-sonnet-20241022",
        )
        
        # Add enough messages to exceed limit
        cm.add_user_message("message 1")
        cm.add_assistant_message("response 1")
        cm.add_user_message("message 2")
        cm.add_assistant_message("response 2")
        
        # Should have at least 2 messages preserved
        assert len(cm.messages) >= 2


class TestAsyncConversationManager:
    """Test asynchronous AsyncConversationManager."""
    
    def test_constructor_valid_params(self):
        """Constructor accepts valid parameters."""
        client = AsyncMock()
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
            system="You are a helpful assistant.",
        )
        assert acm.client is client
        assert acm.max_tokens == 2000
        assert acm.model == "claude-3-5-sonnet-20241022"
        assert acm.system == "You are a helpful assistant."
    
    def test_constructor_invalid_max_tokens_zero(self):
        """Constructor raises ValueError for max_tokens = 0."""
        client = AsyncMock()
        with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
            AsyncConversationManager(
                client=client,
                max_tokens=0,
                model="claude-3-5-sonnet-20241022",
            )
    
    def test_constructor_invalid_max_tokens_negative(self):
        """Constructor raises ValueError for negative max_tokens."""
        client = AsyncMock()
        with pytest.raises(ValueError, match="max_tokens must be greater than 0"):
            AsyncConversationManager(
                client=client,
                max_tokens=-100,
                model="claude-3-5-sonnet-20241022",
            )
    
    def test_add_user_message(self):
        """Add user message to conversation history."""
        client = AsyncMock()
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        acm.add_user_message("Hello")
        
        assert len(acm.messages) == 1
        assert acm.messages[0]["role"] == "user"
        assert acm.messages[0]["content"] == "Hello"
    
    def test_add_assistant_message(self):
        """Add assistant message to conversation history."""
        client = AsyncMock()
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        acm.add_assistant_message("Hi there!")
        
        assert len(acm.messages) == 1
        assert acm.messages[0]["role"] == "assistant"
        assert acm.messages[0]["content"] == "Hi there!"
    
    def test_get_messages(self):
        """Get current conversation history."""
        client = AsyncMock()
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        acm.add_user_message("Hello")
        acm.add_assistant_message("Hi!")
        
        messages = acm.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
    
    def test_get_conversation_tokens(self):
        """Get estimated token count."""
        client = AsyncMock()
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        acm.add_user_message("Hello world")
        
        tokens = acm.get_conversation_tokens()
        assert tokens >= 1
    
    def test_truncation_removes_oldest_messages(self):
        """Truncation removes oldest messages when exceeding max_tokens."""
        client = AsyncMock()
        acm = AsyncConversationManager(
            client=client,
            max_tokens=10,  # Very small limit to trigger truncation
            model="claude-3-5-sonnet-20241022",
        )
        
        # Add messages that will exceed the limit
        acm.add_user_message("Message 1")
        acm.add_assistant_message("Response 1")
        acm.add_user_message("Message 2")
        acm.add_assistant_message("Response 2")
        
        # Should have truncated some older messages
        assert len(acm.messages) < 4
    
    def test_clear_history(self):
        """Clear conversation history."""
        client = AsyncMock()
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        acm.add_user_message("Hello")
        acm.add_assistant_message("Hi!")
        
        assert len(acm.messages) == 2
        acm.clear_history()
        assert len(acm.messages) == 0
    
    @pytest.mark.asyncio
    async def test_create_message(self):
        """Create message via async API."""
        client = AsyncMock()
        
        # Mock the response
        mock_response = Mock(spec=Message)
        mock_response.content = [Mock(spec=TextBlock)]
        mock_response.content[0].text = "Response text"
        client.messages.create = AsyncMock(return_value=mock_response)
        
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
            system="Test system",
        )
        
        response = await acm.create_message("Hello, Claude!")
        
        assert response is mock_response
        # Should have added user message and assistant response to history
        assert len(acm.messages) == 2
        assert acm.messages[0]["role"] == "user"
        assert acm.messages[1]["role"] == "assistant"
        
        # Verify API was called with correct parameters
        call_args = client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-5-sonnet-20241022"
        assert call_args[1]["system"] == "Test system"
    
    @pytest.mark.asyncio
    async def test_create_message_without_system(self):
        """Create message without system prompt."""
        client = AsyncMock()
        mock_response = Mock(spec=Message)
        mock_response.content = []
        client.messages.create = AsyncMock(return_value=mock_response)
        
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        
        response = await acm.create_message("Hello!")
        
        # Verify system prompt is not included
        call_args = client.messages.create.call_args
        assert "system" not in call_args[1]
    
    @pytest.mark.asyncio
    async def test_create_message_multiple_content_blocks(self):
        """Handle response with multiple content blocks."""
        client = AsyncMock()
        
        # Mock response with multiple text blocks
        mock_response = Mock(spec=Message)
        block1 = Mock(spec=TextBlock)
        block1.text = "Part 1"
        block2 = Mock(spec=TextBlock)
        block2.text = " Part 2"
        mock_response.content = [block1, block2]
        client.messages.create = AsyncMock(return_value=mock_response)
        
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        
        response = await acm.create_message("Hello!")
        
        # Assistant message should contain concatenated text
        assert len(acm.messages) == 2
        assistant_msg = acm.messages[1]
        assert assistant_msg["content"] == "Part 1 Part 2"
    
    @pytest.mark.asyncio
    async def test_create_message_with_custom_kwargs(self):
        """Create message with custom kwargs passed to API."""
        client = AsyncMock()
        mock_response = Mock(spec=Message)
        mock_response.content = []
        client.messages.create = AsyncMock(return_value=mock_response)
        
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        
        response = await acm.create_message(
            "Hello!",
            max_tokens=1024,
            temperature=0.7,
        )
        
        # Verify custom kwargs were passed
        call_args = client.messages.create.call_args
        assert call_args[1]["max_tokens"] == 1024
        assert call_args[1]["temperature"] == 0.7
    
    def test_constructor_invalid_client_none_async(self):
        """Constructor raises ValueError for None client."""
        with pytest.raises(ValueError, match="client must not be None"):
            AsyncConversationManager(
                client=None,
                max_tokens=2000,
                model="claude-3-5-sonnet-20241022",
            )
    
    def test_constructor_invalid_model_empty_async(self):
        """Constructor raises ValueError for empty model."""
        client = AsyncMock()
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            AsyncConversationManager(
                client=client,
                max_tokens=2000,
                model="",
            )
    
    def test_constructor_invalid_model_none_async(self):
        """Constructor raises ValueError for None model."""
        client = AsyncMock()
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            AsyncConversationManager(
                client=client,
                max_tokens=2000,
                model=None,
            )
    
    @pytest.mark.asyncio
    async def test_create_message_empty_response_content_async(self):
        """Response with empty content should not add empty message."""
        client = AsyncMock()
        mock_response = Mock(spec=Message)
        mock_response.content = []
        client.messages.create = AsyncMock(return_value=mock_response)
        
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        
        response = await acm.create_message("Hello!")
        
        # Should NOT have added empty assistant message
        assert len(acm.messages) == 1  # Only user message
    
    @pytest.mark.asyncio
    async def test_create_message_non_text_blocks_only_async(self):
        """Response with only non-text blocks should not add empty message."""
        client = AsyncMock()
        mock_response = Mock(spec=Message)
        
        # Create a block without text attribute
        tool_block = Mock()
        # Don't set text attribute - hasattr will return False
        mock_response.content = [tool_block]
        client.messages.create = AsyncMock(return_value=mock_response)
        
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        
        response = await acm.create_message("Use a tool")
        
        # Should NOT have added empty assistant message
        assert len(acm.messages) == 1  # Only user message
    
    @pytest.mark.asyncio
    async def test_create_message_stream_not_supported_async(self):
        """create_message should raise ValueError if stream=True is passed."""
        client = AsyncMock()
        acm = AsyncConversationManager(
            client=client,
            max_tokens=2000,
            model="claude-3-5-sonnet-20241022",
        )
        
        with pytest.raises(ValueError, match="Streaming is not supported"):
            await acm.create_message("Hello!", stream=True)
    
    def test_truncation_preserves_minimum_context_async(self):
        """Truncation should preserve at least 2 messages for minimum context."""
        client = AsyncMock()
        acm = AsyncConversationManager(
            client=client,
            max_tokens=5,  # Very small to force truncation
            model="claude-3-5-sonnet-20241022",
        )
        
        # Add enough messages to exceed limit (using non-async methods for setup)
        acm.add_user_message("message 1")
        acm.add_assistant_message("response 1")
        acm.add_user_message("message 2")
        acm.add_assistant_message("response 2")
        
        # Should have at least 2 messages preserved
        assert len(acm.messages) >= 2
