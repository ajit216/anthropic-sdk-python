"""Helper utilities for the Anthropic SDK.

Example::

    from anthropic.helpers import ConversationManager
    from anthropic import Anthropic

    client = Anthropic()
    manager = ConversationManager(
        client=client,
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        context_window_limit=200_000,
    )

    manager.add_user_message("What is 2 + 2?")
    response = manager.get_response()
    print(response.content[0].text)
"""

from .conversation import ConversationManager, AsyncConversationManager

__all__ = [
    "ConversationManager",
    "AsyncConversationManager",
]
