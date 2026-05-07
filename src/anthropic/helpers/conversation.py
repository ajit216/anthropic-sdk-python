"""Conversation manager for multi-turn conversation context window management."""

from typing import Any, List, Optional, Dict, Union
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message, MessageParam


class ConversationManager:
    """Manages conversation history with automatic context window truncation.
    
    This helper maintains a conversation history and automatically truncates older
    messages when approaching the context window limit, allowing developers to build
    multi-turn chatbots without manual context management.
    
    Example:
        ```python
        from anthropic import Anthropic
        from anthropic.helpers import ConversationManager
        
        client = Anthropic()
        manager = ConversationManager(
            client=client,
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000
        )
        
        manager.add_user_message("Hello!")
        response = manager.send_message("How are you?")
        print(response.content[0].text)
        ```
    """

    def __init__(
        self,
        client: Anthropic,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 8000,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initialize the ConversationManager.
        
        Args:
            client: An Anthropic client instance.
            model: The model to use for API calls. Defaults to claude-3-5-sonnet-20241022.
            max_tokens: Maximum tokens to use for the API call. Must be positive.
            system_prompt: Optional system prompt to include in all API calls.
            
        Raises:
            ValueError: If max_tokens is not positive or model is empty.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
        
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ("user" or "assistant").
            content: The message content.
            
        Raises:
            ValueError: If role is not "user" or "assistant", or content is not a string.
        """
        if role not in ("user", "assistant"):
            raise ValueError('role must be either "user" or "assistant"')
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        if not content:
            raise ValueError("content cannot be empty")
        
        self._messages.append({"role": role, "content": content})
        self._truncate_if_needed()

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.
        
        Args:
            content: The message content.
        """
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.
        
        Args:
            content: The message content.
        """
        self.add_message("assistant", content)

    def get_messages(self) -> List[MessageParam]:
        """Get all messages in the conversation history.
        
        Returns:
            A list of MessageParam dicts in the format expected by the API.
        """
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self._messages
        ]

    def send_message(self, message: str) -> Message:
        """Send a user message and get a response from the model.
        
        Args:
            message: The user message to send.
            
        Returns:
            The Message object returned by the API.
        """
        self.add_user_message(message)
        
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_prompt,
            messages=self.get_messages(),
        )
        
        # Add assistant response to history
        if response.content and len(response.content) > 0:
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text
            if assistant_text:
                self.add_assistant_message(assistant_text)
        
        return response

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using a simple heuristic.
        
        Uses approximately 4 characters per token as a rough estimate.
        
        Args:
            text: The text to estimate tokens for.
            
        Returns:
            Estimated token count.
        """
        return max(1, len(text) // 4)

    def _calculate_total_tokens(self) -> int:
        """Calculate the total estimated tokens in the current message history.
        
        Returns:
            Total estimated token count.
        """
        total = 0
        for msg in self._messages:
            total += self._estimate_tokens(msg["content"])
        
        # Add system prompt tokens if present
        if self._system_prompt:
            total += self._estimate_tokens(self._system_prompt)
        
        return total

    def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if approaching the context window limit.
        
        Removes messages from the beginning of the history when the total estimated
        tokens exceed 80% of max_tokens. This ensures there's buffer space for
        new messages and the API response.
        """
        threshold = int(self._max_tokens * 0.8)
        
        while len(self._messages) > 0 and self._calculate_total_tokens() > threshold:
            self._messages.pop(0)


class AsyncConversationManager:
    """Async version of ConversationManager for use with AsyncAnthropic client.
    
    Provides the same interface as ConversationManager but with async/await support.
    
    Example:
        ```python
        from anthropic import AsyncAnthropic
        from anthropic.helpers import AsyncConversationManager
        
        async def main():
            client = AsyncAnthropic()
            manager = AsyncConversationManager(
                client=client,
                model="claude-3-5-sonnet-20241022",
                max_tokens=8000
            )
            
            manager.add_user_message("Hello!")
            response = await manager.send_message("How are you?")
            print(response.content[0].text)
        
        asyncio.run(main())
        ```
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 8000,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Initialize the AsyncConversationManager.
        
        Args:
            client: An AsyncAnthropic client instance.
            model: The model to use for API calls. Defaults to claude-3-5-sonnet-20241022.
            max_tokens: Maximum tokens to use for the API call. Must be positive.
            system_prompt: Optional system prompt to include in all API calls.
            
        Raises:
            ValueError: If max_tokens is not positive or model is empty.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if not model or not isinstance(model, str):
            raise ValueError("model must be a non-empty string")
        
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._messages: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ("user" or "assistant").
            content: The message content.
            
        Raises:
            ValueError: If role is not "user" or "assistant", or content is not a string.
        """
        if role not in ("user", "assistant"):
            raise ValueError('role must be either "user" or "assistant"')
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        if not content:
            raise ValueError("content cannot be empty")
        
        self._messages.append({"role": role, "content": content})
        self._truncate_if_needed()

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation.
        
        Args:
            content: The message content.
        """
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation.
        
        Args:
            content: The message content.
        """
        self.add_message("assistant", content)

    def get_messages(self) -> List[MessageParam]:
        """Get all messages in the conversation history.
        
        Returns:
            A list of MessageParam dicts in the format expected by the API.
        """
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self._messages
        ]

    async def send_message(self, message: str) -> Message:
        """Send a user message and get a response from the model.
        
        Args:
            message: The user message to send.
            
        Returns:
            The Message object returned by the API.
        """
        self.add_user_message(message)
        
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_prompt,
            messages=self.get_messages(),
        )
        
        # Add assistant response to history
        if response.content and len(response.content) > 0:
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text
            if assistant_text:
                self.add_assistant_message(assistant_text)
        
        return response

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text using a simple heuristic.
        
        Uses approximately 4 characters per token as a rough estimate.
        
        Args:
            text: The text to estimate tokens for.
            
        Returns:
            Estimated token count.
        """
        return max(1, len(text) // 4)

    def _calculate_total_tokens(self) -> int:
        """Calculate the total estimated tokens in the current message history.
        
        Returns:
            Total estimated token count.
        """
        total = 0
        for msg in self._messages:
            total += self._estimate_tokens(msg["content"])
        
        # Add system prompt tokens if present
        if self._system_prompt:
            total += self._estimate_tokens(self._system_prompt)
        
        return total

    def _truncate_if_needed(self) -> None:
        """Truncate oldest messages if approaching the context window limit.
        
        Removes messages from the beginning of the history when the total estimated
        tokens exceed 80% of max_tokens. This ensures there's buffer space for
        new messages and the API response.
        """
        threshold = int(self._max_tokens * 0.8)
        
        while len(self._messages) > 0 and self._calculate_total_tokens() > threshold:
            self._messages.pop(0)
