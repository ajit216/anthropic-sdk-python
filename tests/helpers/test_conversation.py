"""Tests for ConversationManager helpers."""

import pytest

from anthropic.helpers import ConversationManager, AsyncConversationManager


class TestConversationManager:
    """Tests for the ConversationManager sync class."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        manager = ConversationManager()
        assert manager.model == "claude-3-5-sonnet-20241022"
        assert manager.context_window == 200000
        assert manager.get_messages() == []

    def test_initialization_custom_model(self) -> None:
        """Test initialization with custom model."""
        manager = ConversationManager(model="claude-3-opus-20250219")
        assert manager.model == "claude-3-opus-20250219"
        assert manager.context_window == 200000

    def test_initialization_custom_context_window(self) -> None:
        """Test initialization with custom context window."""
        manager = ConversationManager(context_window=100000)
        assert manager.context_window == 100000

    def test_initialization_unknown_model(self) -> None:
        """Test initialization with unknown model defaults to fallback."""
        manager = ConversationManager(model="unknown-model-xyz")
        assert manager.model == "unknown-model-xyz"
        assert manager.context_window == 200000  # Default fallback

    def test_add_user_message(self) -> None:
        """Test adding a user message."""
        manager = ConversationManager()
        manager.add_user_message("Hello")
        
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_add_assistant_message(self) -> None:
        """Test adding an assistant message."""
        manager = ConversationManager()
        manager.add_assistant_message("Hi there")
        
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hi there"

    def test_add_message_user(self) -> None:
        """Test adding a message with user role."""
        manager = ConversationManager()
        manager.add_message("user", "Hello")
        
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_add_message_assistant(self) -> None:
        """Test adding a message with assistant role."""
        manager = ConversationManager()
        manager.add_message("assistant", "Hi")
        
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hi"

    def test_add_message_invalid_role(self) -> None:
        """Test that invalid role raises ValueError."""
        manager = ConversationManager()
        with pytest.raises(ValueError, match="role must be"):
            manager.add_message("invalid", "content")

    def test_multi_turn_conversation(self) -> None:
        """Test a multi-turn conversation."""
        manager = ConversationManager()
        
        manager.add_user_message("What's the capital of France?")
        manager.add_assistant_message("The capital of France is Paris.")
        manager.add_user_message("And the capital of Germany?")
        manager.add_assistant_message("The capital of Germany is Berlin.")
        
        messages = manager.get_messages()
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"

    def test_get_messages_returns_copy(self) -> None:
        """Test that get_messages returns a copy."""
        manager = ConversationManager()
        manager.add_user_message("Hello")
        
        messages1 = manager.get_messages()
        messages2 = manager.get_messages()
        
        # Should be equal but not the same object
        assert messages1 == messages2
        assert messages1 is not messages2

    def test_clear(self) -> None:
        """Test clearing the conversation."""
        manager = ConversationManager()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        
        assert len(manager.get_messages()) == 2
        
        manager.clear()
        assert manager.get_messages() == []

    def test_estimate_tokens_string_content(self) -> None:
        """Test token estimation for string content."""
        manager = ConversationManager()
        
        # Add a message with known content
        messages = [{"role": "user", "content": "Hello world"}]
        tokens = manager._estimate_tokens(messages)
        
        # Should be > 0
        assert tokens > 0

    def test_estimate_tokens_multiple_messages(self) -> None:
        """Test token estimation with multiple messages."""
        manager = ConversationManager()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        tokens = manager._estimate_tokens(messages)
        
        # Should account for both messages
        assert tokens > 0

    def test_ensure_within_context(self) -> None:
        """Test ensuring conversation is within context."""
        manager = ConversationManager()
        manager.add_user_message("Hello")
        
        # Should not raise
        manager.ensure_within_context()
        
        # Messages should still be there
        assert len(manager.get_messages()) > 0

    def test_prune_messages_keeps_minimum(self) -> None:
        """Test that pruning always keeps at least one message."""
        manager = ConversationManager(context_window=100)  # Very small window
        
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        manager.add_user_message("World")
        
        manager._prune_messages()
        
        # Should keep at least one message even with tiny context window
        assert len(manager.get_messages()) >= 1

    def test_model_context_windows_known_models(self) -> None:
        """Test context window sizes for known models."""
        test_cases = [
            ("claude-3-5-sonnet-20241022", 200000),
            ("claude-3-opus-20250219", 200000),
            ("claude-3-5-haiku-20241022", 100000),
        ]
        
        for model, expected_window in test_cases:
            manager = ConversationManager(model=model)
            assert manager.context_window == expected_window, f"Failed for {model}"


class TestAsyncConversationManager:
    """Tests for the AsyncConversationManager async class."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        manager = AsyncConversationManager()
        assert manager.model == "claude-3-5-sonnet-20241022"
        assert manager.context_window == 200000
        assert manager.get_messages() == []

    def test_initialization_custom_model(self) -> None:
        """Test initialization with custom model."""
        manager = AsyncConversationManager(model="claude-3-opus-20250219")
        assert manager.model == "claude-3-opus-20250219"
        assert manager.context_window == 200000

    def test_initialization_custom_context_window(self) -> None:
        """Test initialization with custom context window."""
        manager = AsyncConversationManager(context_window=100000)
        assert manager.context_window == 100000

    def test_add_user_message(self) -> None:
        """Test adding a user message."""
        manager = AsyncConversationManager()
        manager.add_user_message("Hello")
        
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_add_assistant_message(self) -> None:
        """Test adding an assistant message."""
        manager = AsyncConversationManager()
        manager.add_assistant_message("Hi there")
        
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hi there"

    def test_add_message_user(self) -> None:
        """Test adding a message with user role."""
        manager = AsyncConversationManager()
        manager.add_message("user", "Hello")
        
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"

    def test_add_message_assistant(self) -> None:
        """Test adding a message with assistant role."""
        manager = AsyncConversationManager()
        manager.add_message("assistant", "Hi")
        
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Hi"

    def test_add_message_invalid_role(self) -> None:
        """Test that invalid role raises ValueError."""
        manager = AsyncConversationManager()
        with pytest.raises(ValueError, match="role must be"):
            manager.add_message("invalid", "content")

    def test_multi_turn_conversation(self) -> None:
        """Test a multi-turn conversation."""
        manager = AsyncConversationManager()
        
        manager.add_user_message("What's the capital of France?")
        manager.add_assistant_message("The capital of France is Paris.")
        manager.add_user_message("And the capital of Germany?")
        manager.add_assistant_message("The capital of Germany is Berlin.")
        
        messages = manager.get_messages()
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"

    def test_get_messages_returns_copy(self) -> None:
        """Test that get_messages returns a copy."""
        manager = AsyncConversationManager()
        manager.add_user_message("Hello")
        
        messages1 = manager.get_messages()
        messages2 = manager.get_messages()
        
        # Should be equal but not the same object
        assert messages1 == messages2
        assert messages1 is not messages2

    def test_clear(self) -> None:
        """Test clearing the conversation."""
        manager = AsyncConversationManager()
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        
        assert len(manager.get_messages()) == 2
        
        manager.clear()
        assert manager.get_messages() == []

    def test_estimate_tokens_string_content(self) -> None:
        """Test token estimation for string content."""
        manager = AsyncConversationManager()
        
        # Add a message with known content
        messages = [{"role": "user", "content": "Hello world"}]
        tokens = manager._estimate_tokens(messages)
        
        # Should be > 0
        assert tokens > 0

    def test_estimate_tokens_multiple_messages(self) -> None:
        """Test token estimation with multiple messages."""
        manager = AsyncConversationManager()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        tokens = manager._estimate_tokens(messages)
        
        # Should account for both messages
        assert tokens > 0

    def test_ensure_within_context(self) -> None:
        """Test ensuring conversation is within context."""
        manager = AsyncConversationManager()
        manager.add_user_message("Hello")
        
        # Should not raise
        manager.ensure_within_context()
        
        # Messages should still be there
        assert len(manager.get_messages()) > 0

    def test_prune_messages_keeps_minimum(self) -> None:
        """Test that pruning always keeps at least one message."""
        manager = AsyncConversationManager(context_window=100)  # Very small window
        
        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi")
        manager.add_user_message("World")
        
        manager._prune_messages()
        
        # Should keep at least one message even with tiny context window
        assert len(manager.get_messages()) >= 1

    def test_model_context_windows_known_models(self) -> None:
        """Test context window sizes for known models."""
        test_cases = [
            ("claude-3-5-sonnet-20241022", 200000),
            ("claude-3-opus-20250219", 200000),
            ("claude-3-5-haiku-20241022", 100000),
        ]
        
        for model, expected_window in test_cases:
            manager = AsyncConversationManager(model=model)
            assert manager.context_window == expected_window, f"Failed for {model}"
