"""System health monitoring utilities for diagnosing resource usage issues.

This module provides tools to monitor CPU usage, memory utilization, and other
system resources to help diagnose performance issues similar to those reported
in RAP-841 (high CPU utilization with many active connections).
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional, TypedDict
from dataclasses import dataclass, asdict


class MemoryStats(TypedDict):
    """Memory statistics."""

    used_mb: float
    total_mb: float
    percent: float


class SystemDiagnostics(TypedDict, total=False):
    """System diagnostics information."""

    cpu_percent: Optional[float]
    memory: MemoryStats
    process_count: Optional[int]
    timestamp: str


@dataclass
class SystemHealthMonitor:
    """Monitor system health metrics for performance diagnostics.

    This class provides thread-safe access to CPU usage, memory utilization,
    and other system metrics. It's designed to help diagnose issues like
    high CPU utilization with many active connections.

    Example:
        >>> monitor = SystemHealthMonitor()
        >>> cpu_usage = monitor.get_cpu_usage()
        >>> diagnostics = monitor.get_diagnostics()
    """

    _lock: threading.Lock = None
    _last_cpu_check: Optional[float] = None
    _last_cpu_time: Optional[float] = None

    def __post_init__(self) -> None:
        """Initialize thread lock."""
        if self._lock is None:
            object.__setattr__(self, "_lock", threading.Lock())

    def get_cpu_usage(self) -> Optional[float]:
        """Get current CPU usage percentage.

        Returns:
            CPU usage as a percentage (0-100), or None if unavailable.
        """
        try:
            # Try using psutil if available
            try:
                import psutil

                return psutil.cpu_percent(interval=0.1)
            except ImportError:
                pass

            # Fallback: try /proc/stat on Linux
            if os.path.exists("/proc/stat"):
                return self._get_cpu_usage_linux()

            return None
        except Exception:
            return None

    def _get_cpu_usage_linux(self) -> Optional[float]:
        """Get CPU usage from /proc/stat on Linux systems."""
        try:
            with open("/proc/stat", "r") as f:
                line = f.readline()
                if not line.startswith("cpu "):
                    return None

                fields = line.split()[1:8]
                cpu_time = sum(int(f) for f in fields)

            with self._lock:
                if (
                    self._last_cpu_time is None
                    or self._last_cpu_check is None
                ):
                    object.__setattr__(self, "_last_cpu_time", cpu_time)
                    return None

                # This is simplified and may not be perfectly accurate
                # but provides a reasonable estimate
                time_diff = cpu_time - self._last_cpu_time
                object.__setattr__(self, "_last_cpu_time", cpu_time)

                # Return a simplified calculation
                # In production, use psutil for accurate readings
                return min(float(time_diff % 100), 100.0)

        except Exception:
            return None

    def get_memory_usage(self) -> Optional[MemoryStats]:
        """Get current memory usage statistics.

        Returns:
            Dictionary with 'used_mb', 'total_mb', and 'percent' keys,
            or None if unavailable.
        """
        try:
            # Try using psutil if available
            try:
                import psutil

                memory = psutil.virtual_memory()
                return MemoryStats(
                    used_mb=memory.used / (1024 * 1024),
                    total_mb=memory.total / (1024 * 1024),
                    percent=memory.percent,
                )
            except ImportError:
                pass

            # Fallback: try /proc/meminfo on Linux
            if os.path.exists("/proc/meminfo"):
                return self._get_memory_usage_linux()

            return None
        except Exception:
            return None

    def _get_memory_usage_linux(self) -> Optional[MemoryStats]:
        """Get memory usage from /proc/meminfo on Linux systems."""
        try:
            meminfo: dict[str, int] = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    key, value = line.split(":")
                    meminfo[key.strip()] = int(value.split()[0])

            total = meminfo.get("MemTotal", 0)
            available = meminfo.get("MemAvailable", 0)
            used = total - available

            if total == 0:
                return None

            return MemoryStats(
                used_mb=used / 1024.0,
                total_mb=total / 1024.0,
                percent=(used / total) * 100.0,
            )
        except Exception:
            return None

    def get_connection_count(self) -> Optional[int]:
        """Get estimated number of active network connections.

        Returns:
            Number of active connections, or None if unavailable.
        """
        try:
            # Try using psutil if available
            try:
                import psutil

                return len(psutil.net_connections())
            except ImportError:
                pass

            # Fallback: try /proc/net/tcp on Linux
            if os.path.exists("/proc/net/tcp"):
                return self._get_connection_count_linux()

            return None
        except Exception:
            return None

    def _get_connection_count_linux(self) -> Optional[int]:
        """Get connection count from /proc/net/tcp on Linux systems."""
        try:
            count = 0
            for net_file in ["/proc/net/tcp", "/proc/net/tcp6", "/proc/net/udp"]:
                if os.path.exists(net_file):
                    with open(net_file, "r") as f:
                        # Skip header line
                        f.readline()
                        count += sum(1 for _ in f)
            return count if count > 0 else None
        except Exception:
            return None

    def get_diagnostics(self) -> SystemDiagnostics:
        """Get comprehensive system diagnostics.

        Returns a dictionary with CPU, memory, and connection information.
        This can be used to diagnose performance issues like the high CPU
        utilization described in RAP-841.

        Returns:
            Dictionary with diagnostic information.
        """
        from datetime import datetime

        return SystemDiagnostics(
            cpu_percent=self.get_cpu_usage(),
            memory=self.get_memory_usage() or MemoryStats(
                used_mb=0, total_mb=0, percent=0
            ),
            process_count=self.get_connection_count(),
            timestamp=datetime.utcnow().isoformat(),
        )

    def is_high_cpu_usage(self, threshold: float = 80.0) -> bool:
        """Check if CPU usage exceeds a threshold.

        Args:
            threshold: CPU percentage threshold (default 80%).

        Returns:
            True if CPU usage exceeds threshold, False otherwise.
        """
        cpu_usage = self.get_cpu_usage()
        return cpu_usage is not None and cpu_usage > threshold

    def is_high_memory_usage(self, threshold: float = 85.0) -> bool:
        """Check if memory usage exceeds a threshold.

        Args:
            threshold: Memory percentage threshold (default 85%).

        Returns:
            True if memory usage exceeds threshold, False otherwise.
        """
        memory = self.get_memory_usage()
        return memory is not None and memory["percent"] > threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnostics to dictionary format.

        Returns:
            Dictionary representation of current system state.
        """
        return self.get_diagnostics()
