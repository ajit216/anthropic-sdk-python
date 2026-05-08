"""Tests for system health monitoring utilities."""

from __future__ import annotations

import pytest

from anthropic._utils import SystemHealthMonitor


class TestSystemHealthMonitor:
    """Tests for SystemHealthMonitor class."""

    def test_initialization(self) -> None:
        """Test that SystemHealthMonitor can be initialized."""
        monitor = SystemHealthMonitor()
        assert monitor is not None

    def test_get_cpu_usage(self) -> None:
        """Test CPU usage reporting."""
        monitor = SystemHealthMonitor()
        cpu_usage = monitor.get_cpu_usage()

        # CPU usage should be either None (if unavailable) or a percentage
        assert cpu_usage is None or (isinstance(cpu_usage, float) and 0 <= cpu_usage <= 100)

    def test_get_memory_usage(self) -> None:
        """Test memory usage reporting."""
        monitor = SystemHealthMonitor()
        memory = monitor.get_memory_usage()

        if memory is not None:
            assert "used_mb" in memory
            assert "total_mb" in memory
            assert "percent" in memory
            assert memory["used_mb"] >= 0
            assert memory["total_mb"] > 0
            assert 0 <= memory["percent"] <= 100

    def test_get_connection_count(self) -> None:
        """Test connection counting."""
        monitor = SystemHealthMonitor()
        conn_count = monitor.get_connection_count()

        # Connection count should be either None (if unavailable) or a non-negative integer
        assert conn_count is None or (isinstance(conn_count, int) and conn_count >= 0)

    def test_get_diagnostics(self) -> None:
        """Test comprehensive diagnostics output."""
        monitor = SystemHealthMonitor()
        diagnostics = monitor.get_diagnostics()

        assert "timestamp" in diagnostics
        assert "cpu_percent" in diagnostics
        assert "memory" in diagnostics
        assert "process_count" in diagnostics

        # Memory should always be present
        assert diagnostics["memory"] is not None
        assert isinstance(diagnostics["memory"], dict)

    def test_is_high_cpu_usage(self) -> None:
        """Test CPU usage threshold checking."""
        monitor = SystemHealthMonitor()
        result = monitor.is_high_cpu_usage(threshold=1.0)  # Very low threshold

        # Result should be a boolean
        assert isinstance(result, bool)

    def test_is_high_memory_usage(self) -> None:
        """Test memory usage threshold checking."""
        monitor = SystemHealthMonitor()
        result = monitor.is_high_memory_usage(threshold=1.0)  # Very low threshold

        # Result should be a boolean
        assert isinstance(result, bool)

    def test_to_dict(self) -> None:
        """Test converting diagnostics to dictionary."""
        monitor = SystemHealthMonitor()
        diag_dict = monitor.to_dict()

        # Should return a dictionary
        assert isinstance(diag_dict, dict)
        assert "timestamp" in diag_dict

    def test_thread_safety(self) -> None:
        """Test that monitor has thread safety mechanisms."""
        monitor = SystemHealthMonitor()
        # The lock should be initialized
        assert monitor._lock is not None
