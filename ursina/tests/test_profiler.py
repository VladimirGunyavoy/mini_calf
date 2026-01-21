"""
Unit tests for PerformanceProfiler

Tests basic functionality of the profiler without running the full app.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from utils.profiler import PerformanceProfiler


def test_basic_measurement():
    """Test basic measurement functionality"""
    profiler = PerformanceProfiler()

    # Measure a simple operation
    with profiler.measure('test_op'):
        time.sleep(0.01)  # Sleep 10ms

    stats = profiler.get_stats('test_op')
    assert stats is not None, "Stats should exist"
    assert stats['count'] == 1, "Count should be 1"
    assert 8 < stats['last'] < 15, f"Duration should be ~10ms, got {stats['last']:.2f}ms"
    print(f"[OK] Basic measurement: {stats['last']:.2f}ms (expected ~10ms)")


def test_multiple_measurements():
    """Test multiple measurements and EMA calculation"""
    profiler = PerformanceProfiler(ema_alpha=0.5)

    # First measurement: 10ms
    with profiler.measure('multi_op'):
        time.sleep(0.01)

    # Second measurement: 20ms
    with profiler.measure('multi_op'):
        time.sleep(0.02)

    stats = profiler.get_stats('multi_op')
    assert stats['count'] == 2, "Count should be 2"
    assert stats['min'] < stats['max'], "Min should be less than max"
    assert stats['avg'] > 0, "Average should be positive"
    print(f"[OK] Multiple measurements: count={stats['count']}, avg={stats['avg']:.2f}ms, ema={stats['ema']:.2f}ms")


def test_report_generation():
    """Test report generation"""
    profiler = PerformanceProfiler()

    # Measure multiple operations
    with profiler.measure('op1'):
        time.sleep(0.005)  # 5ms

    with profiler.measure('op2'):
        time.sleep(0.010)  # 10ms

    with profiler.measure('op3'):
        time.sleep(0.015)  # 15ms

    report = profiler.get_report(top_n=3)
    assert 'op1' in report or 'op2' in report or 'op3' in report, "Report should contain operations"
    print(f"[OK] Report generation:\n{report}")


def test_compact_report():
    """Test compact report for UI display"""
    profiler = PerformanceProfiler()

    with profiler.measure('physics'):
        time.sleep(0.002)

    with profiler.measure('rendering'):
        time.sleep(0.005)

    compact = profiler.get_compact_report(top_n=2)
    assert 'physics' in compact or 'rendering' in compact, "Compact report should contain operations"
    print(f"[OK] Compact report: {compact}")


def test_enable_disable():
    """Test enabling/disabling profiler"""
    profiler = PerformanceProfiler()

    # Measure with profiler enabled
    with profiler.measure('enabled_op'):
        time.sleep(0.001)

    stats1 = profiler.get_stats('enabled_op')
    assert stats1 is not None, "Stats should exist when enabled"

    # Disable profiler
    profiler.disable()

    # Measure with profiler disabled (should have no overhead)
    start = time.perf_counter()
    with profiler.measure('disabled_op'):
        time.sleep(0.001)
    overhead = (time.perf_counter() - start) * 1000

    stats2 = profiler.get_stats('disabled_op')
    assert stats2 is None, "Stats should not exist when disabled"
    print(f"[OK] Enable/disable: overhead when disabled = {overhead:.3f}ms")


def test_csv_export():
    """Test CSV export functionality"""
    profiler = PerformanceProfiler()

    # Measure some operations
    for i in range(5):
        with profiler.measure('export_test'):
            time.sleep(0.001)

    # Export to CSV
    import tempfile
    import os
    import csv

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        temp_path = f.name

    try:
        profiler.export_to_csv(temp_path)

        # Verify CSV content
        with open(temp_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 6, f"Should have 6 rows (header + 5 measurements), got {len(rows)}"
            assert rows[0] == ['timestamp', 'operation', 'duration_ms'], "Header should be correct"
            assert rows[1][1] == 'export_test', "Operation name should be correct"
        print(f"[OK] CSV export: {len(rows)-1} measurements exported")
    finally:
        os.remove(temp_path)


def test_overhead():
    """Test profiler overhead"""
    profiler = PerformanceProfiler()

    # Measure 1000 empty operations
    start = time.perf_counter()
    for i in range(1000):
        with profiler.measure('overhead_test'):
            pass
    total_time = (time.perf_counter() - start) * 1000

    per_measurement = total_time / 1000
    print(f"[OK] Overhead: {per_measurement:.3f}ms per measurement (1000 measurements in {total_time:.2f}ms)")
    assert per_measurement < 0.1, f"Overhead too high: {per_measurement:.3f}ms (should be < 0.1ms)"


def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("PERFORMANCE PROFILER TESTS")
    print("="*70)

    tests = [
        test_basic_measurement,
        test_multiple_measurements,
        test_report_generation,
        test_compact_report,
        test_enable_disable,
        test_csv_export,
        test_overhead
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n{test.__name__}:")
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] ERROR: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
