"""Conversation Manager helpers for multi-turn context window management."""

from typing import Optional, Any, Literal
import json


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text using a simple approximation.
    
    Approximation: ~1 token per 4 characters (common for English text).
    """
    return max(1, len(text) // 4)


class ConversationManager:
    """Manages multi-turn conversation history with automatic context window management.
    
    This helper maintains conversation state across turns and automatically truncates
    the oldest messages when approaching the context window limit.
    
    Args:
        max_context_window: Maximum number of tokens allowed in the context window.
            Defaults to 4096. Must be greater than 0.
        system_prompt: Optional system prompt/instructions to include at the start
            of the conversation.
    
    Raises:
        ValueError: If max_context_window is not positive or if system_prompt is empty.
    
    Example:
        >>> manager = ConversationManager(max_context_window=4096)
        >>> manager.add_message("user", "Hello!")
        >>> manager.add_message("assistant", "Hi there! How can I help?")
        >>> history = manager.get_history()
    """
    
    def __init__(
        self,
        max_context_window: int = 4096,
        system_prompt: Optional[str] = None,
    ) -> None:
        if max_context_window <= 0:
            raise ValueError("max_context_window must be greater than 0")
        
        if system_prompt is not None and not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")
        
        self.max_context_window = max_context_window
        self.messages: list[dict[str, str]] = []
        self.system_prompt = system_prompt
        
        # Add system prompt as first message if provided
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message sender. Must be 'user' or 'assistant'.
            content: The message content. Cannot be empty.
        
        Raises:
            ValueError: If role is invalid or content is empty.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        
        if not content or not content.strip():
            raise ValueError("content cannot be empty")
        
        self.messages.append({"role": role, "content": content})
        
        # Automatically truncate if we're approaching the limit
        if self.should_truncate():
            self.truncate()
    
    def get_history(self) -> list[dict[str, str]]:
        """Get the current conversation history.
        
        Returns:
            A list of message dictionaries with 'role' and 'content' keys.
        """
        return self.messages.copy()
    
    def add_response(self, response: Any) -> None:
        """Process an API response and add it to history.
        
        Supports both Message objects and dict-like responses.
        
        Args:
            response: The API response object or dict with message content.
        """
        # Handle Message objects from the API
        if hasattr(response, "content") and hasattr(response, "role"):
            # Extract text content from content blocks
            text_content = ""
            if isinstance(response.content, list):
                for block in response.content:
                    if hasattr(block, "text"):
                        text_content += block.text
            elif isinstance(response.content, str):
                text_content = response.content
            
            if text_content:
                self.add_message("assistant", text_content)
        elif isinstance(response, dict) and "content" in response:
            # Handle dict-like responses
            content = response["content"]
            if isinstance(content, list):
                text_content = ""
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content += block.get("text", "")
                if text_content:
                    self.add_message("assistant", text_content)
            elif isinstance(content, str):
                self.add_message("assistant", content)
    
    def get_token_count(self) -> int:
        """Get the current total token count of the conversation.
        
        Returns:
            Estimated token count for all messages in the history.
        """
        total = 0
        for message in self.messages:
            # Account for message structure overhead
            total += _estimate_tokens(message.get("content", ""))
            total += 4  # Approximate overhead for role and formatting
        return total
    
    def should_truncate(self) -> bool:
        """Check if the conversation is approaching the context window limit.
        
        Returns:
            True if token count exceeds 80% of the context window limit.
        """
        token_count = self.get_token_count()
        threshold = int(self.max_context_window * 0.8)
        return token_count > threshold
    
    def truncate(self) -> None:
        """Remove the oldest message(s) to stay within context limits.
        
        Preserves the system prompt (first message if it exists with role 'system')
        and removes oldest user/assistant message pairs first.
        """
        # Find the starting index (skip system message if it exists)
        start_index = 0
        if self.messages and self.messages[0].get("role") == "system":
            start_index = 1
        
        if len(self.messages) <= start_index + 1:
            # Don't truncate if we only have system prompt and one message
            return
        
        # Remove oldest message (after system message)
        if len(self.messages) > start_index:
            del self.messages[start_index]
    
    def reset(self) -> None:
        """Clear the conversation history, preserving only the system prompt."""
        if self.system_prompt:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        else:
            self.messages = []


class AsyncConversationManager:
    """Async version of ConversationManager.
    
    Manages multi-turn conversation history with automatic context window management.
    This async version provides the same functionality as ConversationManager but
    is designed to integrate with async workflows.
    
    Args:
        max_context_window: Maximum number of tokens allowed in the context window.
            Defaults to 4096. Must be greater than 0.
        system_prompt: Optional system prompt/instructions to include at the start
            of the conversation.
    
    Raises:
        ValueError: If max_context_window is not positive or if system_prompt is empty.
    
    Example:
        >>> manager = AsyncConversationManager(max_context_window=4096)
        >>> await manager.add_message("user", "Hello!")
        >>> await manager.add_message("assistant", "Hi there! How can I help?")
        >>> history = await manager.get_history()
    """
    
    def __init__(
        self,
        max_context_window: int = 4096,
        system_prompt: Optional[str] = None,
    ) -> None:
        if max_context_window <= 0:
            raise ValueError("max_context_window must be greater than 0")
        
        if system_prompt is not None and not system_prompt.strip():
            raise ValueError("system_prompt cannot be empty")
        
        self.max_context_window = max_context_window
        self.messages: list[dict[str, str]] = []
        self.system_prompt = system_prompt
        
        # Add system prompt as first message if provided
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    
    async def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message sender. Must be 'user' or 'assistant'.
            content: The message content. Cannot be empty.
        
        Raises:
            ValueError: If role is invalid or content is empty.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")
        
        if not content or not content.strip():
            raise ValueError("content cannot be empty")
        
        self.messages.append({"role": role, "content": content})
        
        # Automatically truncate if we're approaching the limit
        if await self.should_truncate():
            await self.truncate()
    
    async def get_history(self) -> list[dict[str, str]]:
        """Get the current conversation history.
        
        Returns:
            A list of message dictionaries with 'role' and 'content' keys.
        """
        return self.messages.copy()
    
    async def add_response(self, response: Any) -> None:
        """Process an API response and add it to history.
        
        Supports both Message objects and dict-like responses.
        
        Args:
            response: The API response object or dict with message content.
        """
        # Handle Message objects from the API
        if hasattr(response, "content") and hasattr(response, "role"):
            # Extract text content from content blocks
            text_content = ""
            if isinstance(response.content, list):
                for block in response.content:
                    if hasattr(block, "text"):
                        text_content += block.text
            elif isinstance(response.content, str):
                text_content = response.content
            
            if text_content:
                await self.add_message("assistant", text_content)
        elif isinstance(response, dict) and "content" in response:
            # Handle dict-like responses
            content = response["content"]
            if isinstance(content, list):
                text_content = ""
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content += block.get("text", "")
                if text_content:
                    await self.add_message("assistant", text_content)
            elif isinstance(content, str):
                await self.add_message("assistant", content)
    
    async def get_token_count(self) -> int:
        """Get the current total token count of the conversation.
        
        Returns:
            Estimated token count for all messages in the history.
        """
        total = 0
        for message in self.messages:
            # Account for message structure overhead
            total += _estimate_tokens(message.get("content", ""))
            total += 4  # Approximate overhead for role and formatting
        return total
    
    async def should_truncate(self) -> bool:
        """Check if the conversation is approaching the context window limit.
        
        Returns:
            True if token count exceeds 80% of the context window limit.
        """
        token_count = await self.get_token_count()
        threshold = int(self.max_context_window * 0.8)
        return token_count > threshold
    
    async def truncate(self) -> None:
        """Remove the oldest message(s) to stay within context limits.
        
        Preserves the system prompt (first message if it exists with role 'system')
        and removes oldest user/assistant message pairs first.
        """
        # Find the starting index (skip system message if it exists)
        start_index = 0
        if self.messages and self.messages[0].get("role") == "system":
            start_index = 1
        
        if len(self.messages) <= start_index + 1:
            # Don't truncate if we only have system prompt and one message
            return
        
        # Remove oldest message (after system message)
        if len(self.messages) > start_index:
            del self.messages[start_index]
    
    async def reset(self) -> None:
        """Clear the conversation history, preserving only the system prompt."""
        if self.system_prompt:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        else:
            self.messages = []
