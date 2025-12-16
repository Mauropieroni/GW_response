#!/usr/bin/env python
"""
Script to compare performance across different hardware and optimization levels.

Usage:
    # Run benchmarks on current hardware
    python benchmark_comparison.py

    # Run with specific settings
    python benchmark_comparison.py --warmup 5 --iterations 20

    # Compare two saved reports
    python benchmark_comparison.py --compare baseline.json optimized.json

    # Save to specific file
    python benchmark_comparison.py --output my_benchmark.json
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp


def main():
    parser = argparse.ArgumentParser(
        description="GW Response Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of timing iterations (default: 10)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output file path for results"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "OPTIMIZED"),
        help="Compare two saved reports",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark (fewer iterations, smaller sizes)",
    )
    parser.add_argument(
        "--full", action="store_true", help="Run full benchmark suite (all sizes)"
    )

    args = parser.parse_args()

    # Handle comparison mode
    if args.compare:
        from gw_response.benchmark import compare_reports

        print(compare_reports(args.compare[0], args.compare[1]))
        return

    # Import after checking compare mode (faster startup for compare)
    from gw_response.benchmark import BenchmarkSuite, BenchmarkConfig

    # Print system info
    devices = jax.devices()
    print("=" * 60)
    print("GW_RESPONSE BENCHMARK")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {len(devices)}x {devices[0].device_kind if devices else 'cpu'}")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d.device_kind} (id={d.id})")
    print("=" * 60)
    print()

    # Configure benchmark
    if args.quick:
        config = BenchmarkConfig(warmup_iterations=1, timing_iterations=3)
    else:
        config = BenchmarkConfig(
            warmup_iterations=args.warmup, timing_iterations=args.iterations
        )

    # Run benchmarks
    suite = BenchmarkSuite(config)
    report = suite.run_all()

    # Print summary
    print()
    print(report.summary())

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        device_type = devices[0].device_kind if devices else "cpu"
        output_path = f"benchmark_{device_type}_{timestamp}.json"

    report.save(output_path)


if __name__ == "__main__":
    main()
