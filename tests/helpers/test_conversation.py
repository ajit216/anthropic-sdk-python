"""Tests for ConversationManager and AsyncConversationManager."""

import pytest
from anthropic.helpers import ConversationManager, AsyncConversationManager


class TestConversationManager:
    """Tests for ConversationManager."""
    
    def test_init_default(self):
        """Test default initialization."""
        manager = ConversationManager()
        assert manager.max_context_window == 4096
        assert manager.system_prompt is None
        assert manager.messages == []
        assert manager.get_history() == []
    
    def test_init_with_system_prompt(self):
        """Test initialization with system prompt."""
        system_prompt = "You are a helpful assistant."
        manager = ConversationManager(system_prompt=system_prompt)
        assert manager.system_prompt == system_prompt
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "system"
        assert manager.messages[0]["content"] == system_prompt
    
    def test_init_with_max_context_window(self):
        """Test initialization with custom context window."""
        manager = ConversationManager(max_context_window=8192)
        assert manager.max_context_window == 8192
    
    def test_init_invalid_context_window(self):
        """Test that zero or negative context window raises ValueError."""
        with pytest.raises(ValueError, match="max_context_window must be greater than 0"):
            ConversationManager(max_context_window=0)
        
        with pytest.raises(ValueError, match="max_context_window must be greater than 0"):
            ConversationManager(max_context_window=-100)
    
    def test_init_empty_system_prompt(self):
        """Test that empty system prompt raises ValueError."""
        with pytest.raises(ValueError, match="system_prompt cannot be empty"):
            ConversationManager(system_prompt="")
        
        with pytest.raises(ValueError, match="system_prompt cannot be empty"):
            ConversationManager(system_prompt="   ")
    
    def test_add_message_user(self):
        """Test adding a user message."""
        manager = ConversationManager()
        manager.add_message("user", "Hello!")
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "user"
        assert manager.messages[0]["content"] == "Hello!"
    
    def test_add_message_assistant(self):
        """Test adding an assistant message."""
        manager = ConversationManager()
        manager.add_message("assistant", "Hi there!")
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "assistant"
        assert manager.messages[0]["content"] == "Hi there!"
    
    def test_add_message_invalid_role(self):
        """Test that invalid role raises ValueError."""
        manager = ConversationManager()
        with pytest.raises(ValueError, match="role must be 'user' or 'assistant'"):
            manager.add_message("system", "test")
        
        with pytest.raises(ValueError, match="role must be 'user' or 'assistant'"):
            manager.add_message("invalid", "test")
    
    def test_add_message_empty_content(self):
        """Test that empty content raises ValueError."""
        manager = ConversationManager()
        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_message("user", "")
        
        with pytest.raises(ValueError, match="content cannot be empty"):
            manager.add_message("user", "   ")
    
    def test_get_history(self):
        """Test getting conversation history."""
        manager = ConversationManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        
        history = manager.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    def test_get_history_returns_copy(self):
        """Test that get_history returns a copy, not the original list."""
        manager = ConversationManager()
        manager.add_message("user", "Hello")
        
        history = manager.get_history()
        history.append({"role": "assistant", "content": "test"})
        
        # Original should not be modified
        assert len(manager.messages) == 1
    
    def test_token_count_estimation(self):
        """Test token count estimation."""
        manager = ConversationManager()
        manager.add_message("user", "Hello world!")  # ~3 tokens
        
        token_count = manager.get_token_count()
        assert token_count > 0
        # Should be roughly 3 tokens from content + overhead
        assert token_count >= 3
    
    def test_should_truncate_false_below_threshold(self):
        """Test should_truncate returns False when below threshold."""
        manager = ConversationManager(max_context_window=10000)
        manager.add_message("user", "Short")
        assert not manager.should_truncate()
    
    def test_should_truncate_true_above_threshold(self):
        """Test should_truncate returns True when above threshold."""
        manager = ConversationManager(max_context_window=100)
        # Add enough content to exceed 80% of 100 tokens
        long_text = "a" * 500
        manager.add_message("user", long_text)
        assert manager.should_truncate()
    
    def test_truncate_removes_oldest_message(self):
        """Test that truncate removes the oldest message."""
        manager = ConversationManager(max_context_window=100)
        manager.add_message("user", "First")
        manager.add_message("assistant", "Second")
        manager.add_message("user", "Third")
        
        # Manually call truncate to remove oldest
        manager.truncate()
        
        history = manager.get_history()
        assert len(history) == 2
        # Should have kept the last two messages
        assert history[0]["content"] == "Second"
        assert history[1]["content"] == "Third"
    
    def test_truncate_preserves_system_prompt(self):
        """Test that truncate preserves the system prompt."""
        manager = ConversationManager(system_prompt="You are helpful.")
        manager.add_message("user", "Message 1")
        manager.add_message("assistant", "Response 1")
        manager.add_message("user", "Message 2")
        
        manager.truncate()
        
        history = manager.get_history()
        # System prompt should still be first
        assert history[0]["role"] == "system"
        assert len(history) >= 1
    
    def test_truncate_does_not_remove_below_minimum(self):
        """Test that truncate doesn't remove messages below minimum."""
        manager = ConversationManager()
        manager.add_message("user", "Only message")
        
        original_count = len(manager.messages)
        manager.truncate()
        
        # Should not truncate a single message
        assert len(manager.messages) == original_count
    
    def test_add_response_with_dict(self):
        """Test add_response with dict-like object."""
        manager = ConversationManager()
        response = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello!"}
            ]
        }
        manager.add_response(response)
        
        history = manager.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "assistant"
        assert "Hello!" in history[0]["content"]
    
    def test_add_response_with_dict_string_content(self):
        """Test add_response with dict containing string content."""
        manager = ConversationManager()
        response = {
            "role": "assistant",
            "content": "Hello!"
        }
        manager.add_response(response)
        
        history = manager.get_history()
        assert len(history) == 1
        assert "Hello!" in history[0]["content"]
    
    def test_add_response_with_object(self):
        """Test add_response with object having content and role attributes."""
        manager = ConversationManager()
        
        # Create a mock response object
        class MockResponse:
            def __init__(self):
                self.role = "assistant"
                self.content = "Hello from mock!"
        
        response = MockResponse()
        manager.add_response(response)
        
        history = manager.get_history()
        assert len(history) == 1
        assert "Hello from mock!" in history[0]["content"]
    
    def test_add_response_with_content_blocks(self):
        """Test add_response with object having content blocks."""
        manager = ConversationManager()
        
        # Create mock content blocks
        class MockTextBlock:
            def __init__(self, text):
                self.text = text
        
        class MockResponse:
            def __init__(self):
                self.role = "assistant"
                self.content = [
                    MockTextBlock("Hello "),
                    MockTextBlock("World!")
                ]
        
        response = MockResponse()
        manager.add_response(response)
        
        history = manager.get_history()
        assert len(history) == 1
        assert "Hello World!" in history[0]["content"]
    
    def test_reset_clears_history(self):
        """Test that reset clears the conversation."""
        manager = ConversationManager()
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        
        manager.reset()
        
        assert manager.get_history() == []
    
    def test_reset_preserves_system_prompt(self):
        """Test that reset preserves the system prompt."""
        system_prompt = "You are helpful."
        manager = ConversationManager(system_prompt=system_prompt)
        manager.add_message("user", "Hello")
        manager.add_message("assistant", "Hi")
        
        manager.reset()
        
        history = manager.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "system"
        assert history[0]["content"] == system_prompt
    
    def test_multi_turn_conversation(self):
        """Test a multi-turn conversation flow."""
        manager = ConversationManager(system_prompt="You are a helpful assistant.")
        
        # First turn
        manager.add_message("user", "What is Python?")
        manager.add_message("assistant", "Python is a programming language.")
        
        # Second turn
        manager.add_message("user", "How do I use it?")
        manager.add_message("assistant", "You can use Python for many things.")
        
        history = manager.get_history()
        assert len(history) == 5  # system + 4 messages
        
        # Verify order
        assert history[0]["role"] == "system"
        assert history[1]["role"] == "user"
        assert history[2]["role"] == "assistant"
        assert history[3]["role"] == "user"
        assert history[4]["role"] == "assistant"
    
    def test_large_context_auto_truncation(self):
        """Test that auto-truncation works when context gets large."""
        manager = ConversationManager(max_context_window=200)
        
        # Add messages that will exceed the threshold
        manager.add_message("user", "a" * 300)
        # This should trigger truncation due to being above 80% of 200
        
        # Verify the context didn't grow unbounded
        token_count = manager.get_token_count()
        assert token_count <= manager.max_context_window


class TestAsyncConversationManager:
    """Tests for AsyncConversationManager."""
    
    def test_init_default(self):
        """Test default initialization."""
        manager = AsyncConversationManager()
        assert manager.max_context_window == 4096
        assert manager.system_prompt is None
        assert manager.messages == []
    
    def test_init_with_system_prompt(self):
        """Test initialization with system prompt."""
        system_prompt = "You are a helpful assistant."
        manager = AsyncConversationManager(system_prompt=system_prompt)
        assert manager.system_prompt == system_prompt
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "system"
    
    def test_init_invalid_context_window(self):
        """Test that zero or negative context window raises ValueError."""
        with pytest.raises(ValueError, match="max_context_window must be greater than 0"):
            AsyncConversationManager(max_context_window=0)
    
    def test_init_empty_system_prompt(self):
        """Test that empty system prompt raises ValueError."""
        with pytest.raises(ValueError, match="system_prompt cannot be empty"):
            AsyncConversationManager(system_prompt="")
    
    @pytest.mark.asyncio
    async def test_add_message_user(self):
        """Test adding a user message."""
        manager = AsyncConversationManager()
        await manager.add_message("user", "Hello!")
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "user"
    
    @pytest.mark.asyncio
    async def test_add_message_assistant(self):
        """Test adding an assistant message."""
        manager = AsyncConversationManager()
        await manager.add_message("assistant", "Hi there!")
        assert len(manager.messages) == 1
        assert manager.messages[0]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_add_message_invalid_role(self):
        """Test that invalid role raises ValueError."""
        manager = AsyncConversationManager()
        with pytest.raises(ValueError, match="role must be 'user' or 'assistant'"):
            await manager.add_message("system", "test")
    
    @pytest.mark.asyncio
    async def test_add_message_empty_content(self):
        """Test that empty content raises ValueError."""
        manager = AsyncConversationManager()
        with pytest.raises(ValueError, match="content cannot be empty"):
            await manager.add_message("user", "")
    
    @pytest.mark.asyncio
    async def test_get_history(self):
        """Test getting conversation history."""
        manager = AsyncConversationManager()
        await manager.add_message("user", "Hello")
        await manager.add_message("assistant", "Hi")
        
        history = await manager.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_get_history_returns_copy(self):
        """Test that get_history returns a copy."""
        manager = AsyncConversationManager()
        await manager.add_message("user", "Hello")
        
        history = await manager.get_history()
        history.append({"role": "assistant", "content": "test"})
        
        # Original should not be modified
        assert len(manager.messages) == 1
    
    @pytest.mark.asyncio
    async def test_token_count_estimation(self):
        """Test token count estimation."""
        manager = AsyncConversationManager()
        await manager.add_message("user", "Hello world!")
        
        token_count = await manager.get_token_count()
        assert token_count > 0
    
    @pytest.mark.asyncio
    async def test_should_truncate(self):
        """Test should_truncate method."""
        manager = AsyncConversationManager(max_context_window=10000)
        await manager.add_message("user", "Short")
        assert not await manager.should_truncate()
    
    @pytest.mark.asyncio
    async def test_truncate_removes_oldest_message(self):
        """Test that truncate removes the oldest message."""
        manager = AsyncConversationManager(max_context_window=100)
        await manager.add_message("user", "First")
        await manager.add_message("assistant", "Second")
        await manager.add_message("user", "Third")
        
        await manager.truncate()
        
        history = await manager.get_history()
        assert len(history) == 2
    
    @pytest.mark.asyncio
    async def test_truncate_preserves_system_prompt(self):
        """Test that truncate preserves the system prompt."""
        manager = AsyncConversationManager(system_prompt="You are helpful.")
        await manager.add_message("user", "Message 1")
        await manager.add_message("assistant", "Response 1")
        
        await manager.truncate()
        
        history = await manager.get_history()
        assert history[0]["role"] == "system"
    
    @pytest.mark.asyncio
    async def test_add_response_with_dict(self):
        """Test add_response with dict-like object."""
        manager = AsyncConversationManager()
        response = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello!"}
            ]
        }
        await manager.add_response(response)
        
        history = await manager.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "assistant"
    
    @pytest.mark.asyncio
    async def test_reset_clears_history(self):
        """Test that reset clears the conversation."""
        manager = AsyncConversationManager()
        await manager.add_message("user", "Hello")
        await manager.add_message("assistant", "Hi")
        
        await manager.reset()
        
        history = await manager.get_history()
        assert history == []
    
    @pytest.mark.asyncio
    async def test_reset_preserves_system_prompt(self):
        """Test that reset preserves the system prompt."""
        system_prompt = "You are helpful."
        manager = AsyncConversationManager(system_prompt=system_prompt)
        await manager.add_message("user", "Hello")
        await manager.add_message("assistant", "Hi")
        
        await manager.reset()
        
        history = await manager.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "system"
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test a multi-turn conversation flow."""
        manager = AsyncConversationManager(system_prompt="You are a helpful assistant.")
        
        await manager.add_message("user", "What is Python?")
        await manager.add_message("assistant", "Python is a programming language.")
        await manager.add_message("user", "How do I use it?")
        await manager.add_message("assistant", "You can use Python for many things.")
        
        history = await manager.get_history()
        assert len(history) == 5  # system + 4 messages
        assert history[0]["role"] == "system"
