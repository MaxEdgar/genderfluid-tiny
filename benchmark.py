#!/usr/bin/env python3
"""Benchmark script for genderfluid-tiny model."""

import os
import sys
import time
import resource

from genderfluid.inference import predict_name, predict_names


def format_size(size_bytes: float) -> str:
    """Format bytes to human readable."""
    if size_bytes < 1024:
        return f"{size_bytes:.0f} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def get_memory_usage() -> float:
    """Get current process RSS in MB."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # Linux: bytes -> MB
    except Exception:
        return 0.0


def benchmark():
    """Run benchmark suite."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "genderfluid-tiny.bin")
    if not os.path.exists(model_path):
        print("Error: no model found. Run: python train.py")
        sys.exit(1)

    print("=" * 50)
    print("GENDERFLUID-TINY BENCHMARK")
    print("=" * 50)
    print()

    # Model size
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Model size bytes: {size_bytes}")
    print()

    # Model loading time
    print("Model loading time...")
    times = []
    for _ in range(3):
        # Reset cache
        import genderfluid.inference as inf
        inf._model_cache = None
        inf._model_path_cache = None

        start = time.perf_counter()
        predict_name("test")
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg_load = sum(times) / len(times)
    print(f"  Average: {avg_load * 1000:.1f} ms")
    print()

    # Single name inference
    test_names = [
        "Emma", "James", "Alex", "Sophia", "Michael",
        "Elva Retta", "Michelle Renatta Chan", "Max",
        "Taylor Swift", "Jordan Peterson", "Chris Evans",
    ]

    print("Single name inference (10 iterations each)...")
    single_times = []
    for name in test_names:
        times = []
        for _ in range(10):
            start = time.perf_counter()
            predict_name(name)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg = sum(times) / len(times)
        single_times.append(avg)
    avg_single = sum(single_times) / len(single_times)
    print(f"  Average single name: {avg_single * 1000:.2f} ms")
    print()

    # Batch inference
    batch_sizes = [10, 100, 1000]
    all_names = test_names * 100  # 1100 names

    print("Batch inference...")
    for batch_size in batch_sizes:
        batch = all_names[:batch_size]
        times = []
        for _ in range(3):
            start = time.perf_counter()
            predict_names(batch)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        avg = sum(times) / len(times)
        throughput = batch_size / avg
        print(f"  {batch_size:>5} names: {avg * 1000:.1f} ms ({throughput:.0f} names/sec)")
    print()

    # Memory usage
    mem = get_memory_usage()
    print(f"Memory usage (RSS): {mem:.1f} MB")
    print()

    # Summary
    print("=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Loading time: {avg_load * 1000:.1f} ms")
    print(f"Single name: {avg_single * 1000:.2f} ms")
    print(f"Throughput: {1.0 / avg_single:.0f} names/sec")
    print(f"Memory: {mem:.1f} MB")
    print(f"GPU required: NO")
    print(f"Internet required: NO")
    print()


if __name__ == "__main__":
    benchmark()
