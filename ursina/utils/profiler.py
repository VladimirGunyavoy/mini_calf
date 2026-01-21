"""
Performance profiler for measuring operation timings in real-time

Provides context manager for timing operations with minimal overhead (<1ms).
Tracks statistics: min, max, avg, EMA (exponential moving average).

Usage:
    profiler = PerformanceProfiler()

    with profiler.measure('operation_name'):
        # ... code to measure ...
        pass

    # Get report
    report = profiler.get_report(top_n=5)

    # Export to CSV
    profiler.export_to_csv('profile.csv')
"""

import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import csv


class PerformanceProfiler:
    """Lightweight performance profiler with context manager interface"""

    def __init__(self, ema_alpha: float = 0.1):
        """
        Initialize profiler

        Parameters:
        -----------
        ema_alpha : float
            EMA smoothing factor (0.1 = slow response, 0.5 = fast response)
        """
        self.ema_alpha = ema_alpha

        # Statistics storage
        self._stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'count': 0,
            'total': 0.0,
            'min': float('inf'),
            'max': 0.0,
            'ema': 0.0,
            'last': 0.0
        })

        # History for CSV export (operation, timestamp, duration_ms)
        self._history: List[Tuple[str, float, float]] = []

        # Internal state
        self._start_time: Optional[float] = None
        self._current_operation: Optional[str] = None
        self._enabled = True

    def enable(self):
        """Enable profiling"""
        self._enabled = True

    def disable(self):
        """Disable profiling (zero overhead)"""
        self._enabled = False

    def reset(self):
        """Reset all statistics"""
        self._stats.clear()
        self._history.clear()

    @contextmanager
    def measure(self, operation: str):
        """
        Context manager for measuring operation time

        Parameters:
        -----------
        operation : str
            Name of the operation to measure

        Example:
        --------
        with profiler.measure('physics_update'):
            simulation_engine.update()
        """
        if not self._enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self._record(operation, duration)

    def _record(self, operation: str, duration: float):
        """Record a measurement (internal)"""
        duration_ms = duration * 1000.0  # Convert to milliseconds

        stats = self._stats[operation]
        stats['count'] += 1
        stats['total'] += duration_ms
        stats['min'] = min(stats['min'], duration_ms)
        stats['max'] = max(stats['max'], duration_ms)
        stats['last'] = duration_ms

        # Update EMA (exponential moving average)
        if stats['count'] == 1:
            stats['ema'] = duration_ms
        else:
            stats['ema'] = (self.ema_alpha * duration_ms +
                           (1 - self.ema_alpha) * stats['ema'])

        # Add to history
        self._history.append((operation, time.time(), duration_ms))

    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a specific operation

        Returns:
        --------
        dict with keys: count, total, min, max, avg, ema, last
        or None if operation not found
        """
        if operation not in self._stats:
            return None

        stats = self._stats[operation].copy()
        stats['avg'] = stats['total'] / stats['count'] if stats['count'] > 0 else 0.0
        return stats

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all operations

        Returns:
        --------
        dict mapping operation names to their statistics
        """
        result = {}
        for op in self._stats:
            result[op] = self.get_stats(op)
        return result

    def get_report(self, top_n: int = 5, sort_by: str = 'ema') -> str:
        """
        Generate text report of top N slowest operations

        Parameters:
        -----------
        top_n : int
            Number of operations to include
        sort_by : str
            Metric to sort by: 'ema', 'avg', 'max', 'last'

        Returns:
        --------
        str : Formatted text report
        """
        if not self._stats:
            return "No profiling data available"

        # Calculate averages and sort
        operations = []
        for op_name in self._stats:
            stats = self.get_stats(op_name)
            operations.append((op_name, stats))

        # Sort by specified metric
        operations.sort(key=lambda x: x[1][sort_by], reverse=True)

        # Build report
        lines = []
        lines.append("Performance Profile (times in ms)")
        lines.append("=" * 60)
        lines.append(f"{'Operation':<25} {'EMA':>8} {'Last':>8} {'Avg':>8} {'Min':>8} {'Max':>8}")
        lines.append("-" * 60)

        for op_name, stats in operations[:top_n]:
            # Color indicator based on EMA
            if stats['ema'] < 5.0:
                indicator = '[+]'
            elif stats['ema'] < 10.0:
                indicator = '[!]'
            else:
                indicator = '[X]'

            lines.append(
                f"{indicator} {op_name:<23} "
                f"{stats['ema']:>7.2f} "
                f"{stats['last']:>7.2f} "
                f"{stats['avg']:>7.2f} "
                f"{stats['min']:>7.2f} "
                f"{stats['max']:>7.2f}"
            )

        lines.append("-" * 60)
        lines.append(f"Total operations tracked: {len(self._stats)}")

        return '\n'.join(lines)

    def get_compact_report(self, top_n: int = 3) -> str:
        """
        Get compact one-line report for UI display

        Parameters:
        -----------
        top_n : int
            Number of operations to include

        Returns:
        --------
        str : Compact report (e.g., "physics: 2.3ms | training: 1.8ms | visual: 4.5ms")
        """
        if not self._stats:
            return "No profiling data"

        # Get top N by EMA
        operations = []
        for op_name in self._stats:
            stats = self.get_stats(op_name)
            operations.append((op_name, stats['ema']))

        operations.sort(key=lambda x: x[1], reverse=True)

        # Build compact string
        parts = []
        for op_name, ema_ms in operations[:top_n]:
            parts.append(f"{op_name}: {ema_ms:.1f}ms")

        return " | ".join(parts)

    def export_to_csv(self, filename: str):
        """
        Export profiling history to CSV file

        Parameters:
        -----------
        filename : str
            Output CSV file path

        CSV format:
        -----------
        timestamp, operation, duration_ms
        """
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'operation', 'duration_ms'])

            for operation, timestamp, duration_ms in self._history:
                writer.writerow([timestamp, operation, duration_ms])

        print(f"[Profiler] Exported {len(self._history)} measurements to {filename}")

    def get_total_frame_time(self) -> float:
        """
        Get approximate total frame time (sum of all EMA times)

        Returns:
        --------
        float : Total time in milliseconds
        """
        total = 0.0
        for stats in self._stats.values():
            total += stats['ema']
        return total

    def get_fps_estimate(self) -> float:
        """
        Estimate FPS based on total frame time

        Returns:
        --------
        float : Estimated FPS (1000 / total_frame_time_ms)
        """
        total_time = self.get_total_frame_time()
        if total_time <= 0:
            return 0.0
        return 1000.0 / total_time


# Global instance for convenience
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def reset_profiler():
    """Reset global profiler"""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.reset()
