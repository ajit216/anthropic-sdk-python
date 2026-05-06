"""Helper utilities for working with messages in the Anthropic SDK."""

from typing import Optional

from anthropic.types import Message, TextBlock


def extract_text_from_message(message: Message) -> str:
    """
    Extract all text content from a message.
    
    This helper function concatenates all text blocks from a message's content,
    making it easier to work with the full text response from the API.
    
    Args:
        message: The message object to extract text from.
        
    Returns:
        A string containing all text content from the message, with blocks
        separated by newlines.
        
    Example:
        >>> from anthropic import Anthropic
        >>> from anthropic.lib.message_helpers import extract_text_from_message
        >>> client = Anthropic()
        >>> message = client.messages.create(
        ...     model="claude-3-5-sonnet-20241022",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> text = extract_text_from_message(message)
        >>> print(text)
    """
    text_parts = []
    for block in message.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
    return "\n".join(text_parts)


def get_message_stop_reason(message: Message) -> Optional[str]:
    """
    Get the stop reason from a message.
    
    Args:
        message: The message object to check.
        
    Returns:
        The stop reason as a string, or None if not available.
        
    Example:
        >>> from anthropic import Anthropic
        >>> from anthropic.lib.message_helpers import get_message_stop_reason
        >>> client = Anthropic()
        >>> message = client.messages.create(
        ...     model="claude-3-5-sonnet-20241022",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> stop_reason = get_message_stop_reason(message)
        >>> print(f"Stopped because: {stop_reason}")
    """
    if hasattr(message, 'stop_reason'):
        return message.stop_reason
    return None
