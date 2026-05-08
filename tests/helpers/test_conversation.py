"""Tests for ConversationManager and AsyncConversationManager."""

import pytest
from anthropic.helpers import ConversationManager, AsyncConversationManager


class TestConversationManagerConstructor:
    """Test ConversationManager constructor validation."""

    def test_valid_construction(self) -> None:
        """Test valid construction with proper parameters."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        assert manager.model == "claude-3-5-sonnet-20241022"
        assert manager.max_tokens == 10000
        assert manager.response_token_budget == 1000

    def test_custom_response_budget(self) -> None:
        """Test construction with custom response token budget."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
            response_token_budget=2000,
        )
        assert manager.response_token_budget == 2000

    def test_negative_max_tokens(self) -> None:
        """Test that negative max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be non-negative"):
            ConversationManager(
                model="claude-3-5-sonnet-20241022",
                max_tokens=-1,
            )

    def test_negative_response_budget(self) -> None:
        """Test that negative response_token_budget raises ValueError."""
        with pytest.raises(ValueError, match="response_token_budget must be non-negative"):
            ConversationManager(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10000,
                response_token_budget=-1,
            )

    def test_budget_exceeds_max_tokens(self) -> None:
        """Test that response_token_budget >= max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="response_token_budget must be less than max_tokens"):
            ConversationManager(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                response_token_budget=1000,
            )

        with pytest.raises(ValueError, match="response_token_budget must be less than max_tokens"):
            ConversationManager(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                response_token_budget=2000,
            )


class TestConversationManagerMessageHandling:
    """Test message addition and retrieval."""

    def test_add_single_message(self) -> None:
        """Test adding a single message."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        manager.add_message("user", "Hello!")
        
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello!"

    def test_add_multiple_messages(self) -> None:
        """Test adding multiple messages."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        manager.add_message("user", "Hello!")
        manager.add_message("assistant", "Hi there!")
        manager.add_message("user", "How are you?")
        
        messages = manager.get_messages()
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_invalid_role(self) -> None:
        """Test that invalid role raises ValueError."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        with pytest.raises(ValueError, match="role must be one of"):
            manager.add_message("invalid_role", "Hello!")

    def test_system_message(self) -> None:
        """Test adding a system message."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        manager.add_message("system", "You are a helpful assistant.")
        manager.add_message("user", "Hello!")
        
        messages = manager.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"


class TestConversationManagerTokenCounting:
    """Test token counting and space checking."""

    def test_get_token_count_empty(self) -> None:
        """Test token count for empty history."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        # Empty history should return at least 1 token
        assert manager.get_token_count() >= 1

    def test_get_token_count_with_messages(self) -> None:
        """Test token counting with messages."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        manager.add_message("user", "Hello!")
        
        count1 = manager.get_token_count()
        manager.add_message("assistant", "Hi there!")
        count2 = manager.get_token_count()
        
        # Token count should increase
        assert count2 > count1

    def test_has_space_sufficient(self) -> None:
        """Test has_space when there is sufficient space."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
            response_token_budget=1000,
        )
        manager.add_message("user", "Hello!")
        
        # Should have space for a reasonable message
        assert manager.has_space(100)

    def test_has_space_insufficient(self) -> None:
        """Test has_space when there is insufficient space."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            response_token_budget=50,
        )
        # Fill it up with a large message
        manager.add_message("user", "x" * 1000)
        
        # Should not have space for another large message
        assert not manager.has_space(500)


class TestConversationManagerTruncation:
    """Test automatic message truncation."""

    def test_truncation_removes_oldest_non_system(self) -> None:
        """Test that truncation removes oldest non-system messages."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            response_token_budget=20,
        )
        
        # Add a system message
        manager.add_message("system", "You are helpful.")
        # Add messages that will exceed the limit
        manager.add_message("user", "x" * 200)
        manager.add_message("assistant", "y" * 200)
        manager.add_message("user", "z" * 200)
        
        messages = manager.get_messages()
        
        # System message should still be there
        assert any(m["role"] == "system" for m in messages)
        # Some messages should have been removed
        assert len(messages) < 4

    def test_no_truncation_below_limit(self) -> None:
        """Test that no truncation occurs when under the limit."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
            response_token_budget=1000,
        )
        
        manager.add_message("user", "Hello!")
        manager.add_message("assistant", "Hi there!")
        manager.add_message("user", "How are you?")
        
        messages = manager.get_messages()
        # All messages should be kept
        assert len(messages) == 3

    def test_truncation_keeps_recent_messages(self) -> None:
        """Test that truncation keeps the most recent messages."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=150,
            response_token_budget=50,
        )
        
        manager.add_message("user", "Message 1 " + "x" * 200)
        manager.add_message("user", "Message 2 " + "y" * 200)
        manager.add_message("user", "Message 3 " + "z" * 200)
        manager.add_message("user", "Message 4 " + "w" * 200)
        
        messages = manager.get_messages()
        
        # Should have some messages left (not all 4)
        assert len(messages) > 0
        assert len(messages) < 4
        # Most recent message should always be there
        assert messages[-1]["content"].startswith("Message 4")

    def test_truncation_preserves_system_messages(self) -> None:
        """Test that system messages are preserved during truncation."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            response_token_budget=50,
        )
        
        manager.add_message("system", "You are a helpful assistant. Keep this!")
        manager.add_message("user", "x" * 100)
        manager.add_message("assistant", "y" * 100)
        manager.add_message("user", "z" * 100)
        
        messages = manager.get_messages()
        
        # System message should still be present
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) > 0
        assert "You are a helpful assistant. Keep this!" in system_messages[0]["content"]


class TestConversationManagerMultipleTurns:
    """Test state management across multiple turns."""

    def test_state_across_turns(self) -> None:
        """Test that conversation state is maintained across turns."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        
        # Turn 1
        manager.add_message("user", "What is 2+2?")
        manager.add_message("assistant", "2+2 equals 4.")
        
        messages = manager.get_messages()
        assert len(messages) == 2
        
        # Turn 2
        manager.add_message("user", "What is 3+3?")
        manager.add_message("assistant", "3+3 equals 6.")
        
        messages = manager.get_messages()
        assert len(messages) == 4

    def test_context_window_respected_across_turns(self) -> None:
        """Test that context window limit is respected across multiple turns."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            response_token_budget=50,
        )
        
        # Add multiple turns
        for i in range(10):
            manager.add_message("user", f"Question {i} " + "x" * 50)
            manager.add_message("assistant", f"Answer {i} " + "y" * 50)
        
        messages = manager.get_messages()
        # Not all 20 messages should be kept
        assert len(messages) < 20

    def test_empty_history_starts_fresh(self) -> None:
        """Test that conversation starts with empty history."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        
        messages = manager.get_messages()
        assert len(messages) == 0


class TestAsyncConversationManager:
    """Test AsyncConversationManager."""

    def test_async_manager_construction(self) -> None:
        """Test AsyncConversationManager construction."""
        manager = AsyncConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        assert manager.model == "claude-3-5-sonnet-20241022"
        assert manager.max_tokens == 10000

    def test_async_manager_same_interface(self) -> None:
        """Test that AsyncConversationManager has same interface as ConversationManager."""
        sync_manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        async_manager = AsyncConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        
        # Both should support same operations
        sync_manager.add_message("user", "Hello!")
        async_manager.add_message("user", "Hello!")
        
        sync_messages = sync_manager.get_messages()
        async_messages = async_manager.get_messages()
        
        assert len(sync_messages) == len(async_messages)
        assert sync_messages[0]["role"] == async_messages[0]["role"]

    def test_async_manager_validation(self) -> None:
        """Test that AsyncConversationManager validates inputs."""
        with pytest.raises(ValueError, match="max_tokens must be non-negative"):
            AsyncConversationManager(
                model="claude-3-5-sonnet-20241022",
                max_tokens=-1,
            )

    def test_async_manager_token_counting(self) -> None:
        """Test token counting in AsyncConversationManager."""
        manager = AsyncConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        manager.add_message("user", "Hello!")
        manager.add_message("assistant", "Hi!")
        
        count = manager.get_token_count()
        assert count > 0

    def test_async_manager_truncation(self) -> None:
        """Test truncation in AsyncConversationManager."""
        manager = AsyncConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            response_token_budget=20,
        )
        
        manager.add_message("user", "x" * 200)
        manager.add_message("assistant", "y" * 200)
        manager.add_message("user", "z" * 200)
        
        messages = manager.get_messages()
        # Messages should be truncated
        assert len(messages) < 3


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_response_budget(self) -> None:
        """Test with zero response budget."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            response_token_budget=0,
        )
        assert manager.response_token_budget == 0

    def test_empty_message_content(self) -> None:
        """Test adding message with empty content."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        manager.add_message("user", "")
        
        messages = manager.get_messages()
        assert len(messages) == 1
        assert messages[0]["content"] == ""

    def test_very_long_message(self) -> None:
        """Test handling of very long messages."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        long_content = "x" * 10000
        manager.add_message("user", long_content)
        
        messages = manager.get_messages()
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 10000

    def test_special_characters_in_content(self) -> None:
        """Test handling special characters in message content."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        special_content = "Hello! 你好 🎉 \n\t\r"
        manager.add_message("user", special_content)
        
        messages = manager.get_messages()
        assert messages[0]["content"] == special_content

    def test_repeated_role_additions(self) -> None:
        """Test adding messages with the same role repeatedly."""
        manager = ConversationManager(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10000,
        )
        manager.add_message("user", "First user message")
        manager.add_message("user", "Second user message")
        manager.add_message("user", "Third user message")
        
        messages = manager.get_messages()
        assert len(messages) == 3
        assert all(m["role"] == "user" for m in messages)
