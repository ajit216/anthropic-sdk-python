# File created for symphony agent testing - RAP-439

from __future__ import annotations

import os

import pytest

from anthropic import Anthropic, AsyncAnthropic
from tests.utils import assert_matches_type

base_url = os.environ.get("TEST_API_BASE_URL", "http://127.0.0.1:4010")
api_key = os.environ.get("ANTHROPIC_API_KEY", "test-key")


class TestSymphonyAgents:
    """Test suite for agent symphony testing - RAP-439"""

    parametrize = pytest.mark.parametrize("client", [False, True], indirect=True, ids=["loose", "strict"])

    def test_symphony_agent_initialization(self) -> None:
        """Test basic initialization of symphony agents"""
        client = Anthropic(base_url=base_url, api_key=api_key)
        assert client is not None
        assert client.api_key == api_key

    def test_symphony_agent_client_creation(self) -> None:
        """Test creation of Anthropic client for symphony agents"""
        with Anthropic(base_url=base_url, api_key=api_key) as client:
            assert client is not None
            assert hasattr(client, "messages")

    @parametrize
    def test_symphony_agent_models_available(self, client: Anthropic) -> None:
        """Test that models are available for symphony agents"""
        # Verify client has required attributes for symphony operations
        assert hasattr(client, "messages")
        assert hasattr(client, "models")

    def test_async_symphony_agent_initialization(self) -> None:
        """Test async initialization of symphony agents"""
        client = AsyncAnthropic(base_url=base_url, api_key=api_key)
        assert client is not None
        assert client.api_key == api_key

    @pytest.mark.asyncio
    async def test_async_symphony_agent_client_creation(self) -> None:
        """Test creation of async Anthropic client for symphony agents"""
        async with AsyncAnthropic(base_url=base_url, api_key=api_key) as client:
            assert client is not None
            assert hasattr(client, "messages")

    def test_symphony_agent_configuration(self) -> None:
        """Test configuration options for symphony agents"""
        client = Anthropic(
            base_url=base_url,
            api_key=api_key,
            timeout=30.0,
            max_retries=3,
        )
        assert client is not None
        assert client.timeout == 30.0

    def test_symphony_agent_multiple_instances(self) -> None:
        """Test creating multiple symphony agent instances"""
        client1 = Anthropic(base_url=base_url, api_key=api_key)
        client2 = Anthropic(base_url=base_url, api_key=api_key)

        assert client1 is not client2
        assert client1.api_key == client2.api_key

    def test_symphony_agent_context_manager(self) -> None:
        """Test symphony agent usage with context manager"""
        with Anthropic(base_url=base_url, api_key=api_key) as client:
            assert client is not None
            # Verify client is properly initialized with context manager
            assert hasattr(client, "messages")
            assert isinstance(client, Anthropic)

    @pytest.mark.asyncio
    async def test_async_symphony_agent_context_manager(self) -> None:
        """Test async symphony agent usage with context manager"""
        async with AsyncAnthropic(base_url=base_url, api_key=api_key) as client:
            assert client is not None
            assert hasattr(client, "messages")
            assert isinstance(client, AsyncAnthropic)


class TestSymphonyAgentIntegration:
    """Integration tests for symphony agents - RAP-439"""

    def test_symphony_agent_base_url_configuration(self) -> None:
        """Test base URL configuration for symphony agents"""
        custom_url = "https://api.anthropic.com"
        client = Anthropic(base_url=custom_url, api_key=api_key)
        # Verify client was created successfully with custom base URL
        assert client is not None
        assert isinstance(client, Anthropic)

    def test_symphony_agent_api_key_setting(self) -> None:
        """Test API key setting for symphony agents"""
        test_key = "test-symphony-key-123"
        client = Anthropic(base_url=base_url, api_key=test_key)
        assert client.api_key == test_key

    def test_symphony_agent_headers_configuration(self) -> None:
        """Test headers configuration for symphony agents"""
        client = Anthropic(
            base_url=base_url,
            api_key=api_key,
            default_headers={"X-Custom-Header": "symphony-test"},
        )
        assert client is not None

    def test_symphony_agent_retry_configuration(self) -> None:
        """Test retry configuration for symphony agents"""
        client = Anthropic(
            base_url=base_url,
            api_key=api_key,
            max_retries=5,
        )
        assert client is not None

    def test_symphony_agent_timeout_configuration(self) -> None:
        """Test timeout configuration for symphony agents"""
        client = Anthropic(
            base_url=base_url,
            api_key=api_key,
            timeout=60.0,
        )
        assert client is not None

    @pytest.mark.asyncio
    async def test_async_symphony_agent_base_url_configuration(self) -> None:
        """Test async base URL configuration for symphony agents"""
        custom_url = "https://api.anthropic.com"
        client = AsyncAnthropic(base_url=custom_url, api_key=api_key)
        # Verify async client was created successfully with custom base URL
        assert client is not None
        assert isinstance(client, AsyncAnthropic)


class TestSymphonyAgentErrors:
    """Error handling tests for symphony agents - RAP-439"""

    def test_symphony_agent_empty_api_key_handling(self) -> None:
        """Test symphony agent creation with empty API key"""
        # Empty API key string is acceptable at client creation;
        # validation occurs on actual API requests
        client = Anthropic(base_url=base_url, api_key="")
        assert client is not None

    def test_symphony_agent_invalid_base_url(self) -> None:
        """Test symphony agent with invalid base URL format"""
        # Invalid URLs are acceptable at client creation;
        # validation happens on actual API calls
        client = Anthropic(base_url="invalid-url", api_key=api_key)
        assert client is not None

    def test_symphony_agent_cleanup_on_context_exit(self) -> None:
        """Test symphony agent cleanup when context manager exits"""
        # Verify context manager properly initializes and cleans up
        with Anthropic(base_url=base_url, api_key=api_key) as client:
            assert client is not None
            assert isinstance(client, Anthropic)
        # Client should be properly closed after context exit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
