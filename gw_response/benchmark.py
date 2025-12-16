"""
Benchmarking infrastructure for gw_response performance analysis.

Example usage:
    from gw_response.benchmark import Benchmark, BenchmarkSuite

    bench = Benchmark()

    # Time a single function
    result = bench.time_function(
        compute_response,
        frequencies=freqs,
        name="compute_response_default"
    )
    print(result)

    # Run full benchmark suite
    suite = BenchmarkSuite()
    report = suite.run_all()
    report.save("benchmark_results.json")
"""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Any
from functools import wraps
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class TimingResult:
    """Result from a single timing run."""

    name: str
    wall_time_ms: float
    compile_time_ms: float
    execution_time_ms: float
    n_iterations: int
    input_shapes: Dict[str, tuple]
    output_shape: tuple
    device_type: str
    n_devices: int

    # Optional performance metrics
    estimated_flops: Optional[float] = None
    memory_bytes: Optional[int] = None
    throughput_elements_per_sec: Optional[float] = None

    def __repr__(self):
        lines = [
            f"TimingResult({self.name})",
            f"  Device: {self.device_type} x {self.n_devices}",
            f"  Compile: {self.compile_time_ms:.2f} ms",
            f"  Execute: {self.execution_time_ms:.2f} ms (avg of {self.n_iterations})",
            f"  Total wall: {self.wall_time_ms:.2f} ms",
        ]
        if self.estimated_flops:
            gflops_per_sec = (self.estimated_flops / 1e9) / (self.execution_time_ms / 1000)
            lines.append(f"  Performance: {gflops_per_sec:.2f} GFLOP/s")
        if self.throughput_elements_per_sec:
            lines.append(f"  Throughput: {self.throughput_elements_per_sec:.2e} elem/s")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert tuple to list for JSON
        d["input_shapes"] = {k: list(v) for k, v in d["input_shapes"].items()}
        d["output_shape"] = list(d["output_shape"]) if d["output_shape"] else []
        return d


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    warmup_iterations: int = 3
    timing_iterations: int = 10
    sync_device: bool = True  # Block until computation complete


class Benchmark:
    """Core benchmarking class for timing JAX functions."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[TimingResult] = []

    def _get_device_info(self) -> tuple:
        """Get device type and count."""
        devices = jax.devices()
        device_type = devices[0].device_kind if devices else "cpu"
        return device_type, len(devices)

    def _get_input_shapes(self, args, kwargs) -> Dict[str, tuple]:
        """Extract shapes from function inputs."""
        shapes = {}
        for i, arg in enumerate(args):
            if hasattr(arg, "shape"):
                shapes[f"arg_{i}"] = tuple(arg.shape)
        for key, val in kwargs.items():
            if hasattr(val, "shape"):
                shapes[key] = tuple(val.shape)
        return shapes

    def _estimate_flops(self, func_name: str, shapes: Dict) -> Optional[float]:
        """Estimate FLOPS for known functions."""
        # Extract common dimensions
        configs = 1
        freq = 100
        pixels = 768
        arms = 6

        for key, shape in shapes.items():
            if "positions" in key and len(shape) >= 1:
                configs = shape[0]
            if "x_vector" in key or "frequencies" in key:
                freq = shape[0] if len(shape) >= 1 else freq
            if "wavevector" in key and len(shape) >= 2:
                pixels = shape[1]

        if "single_link" in func_name.lower():
            # Complex multiply ~ 6 flops, exp ~ 20 flops
            pos_exp_flops = configs * freq * 3 * pixels * 20
            arm_exp_flops = configs * freq * arms * 20
            einsum_flops = configs * freq * arms * pixels * 12
            return pos_exp_flops + arm_exp_flops + einsum_flops

        if "quadratic" in func_name.lower():
            tdi = 3
            linear_flops = configs * freq * tdi * arms * pixels * 12
            quadratic_flops = configs * freq * tdi * tdi * pixels * 12
            return linear_flops + quadratic_flops

        if "compute_response" in func_name.lower():
            # Full pipeline: 2x single_link + quadratic
            single_link_flops = 2 * (
                configs * freq * 3 * pixels * 20
                + configs * freq * arms * 20
                + configs * freq * arms * pixels * 12
            )
            quadratic_flops = configs * freq * 3 * 3 * pixels * 12
            return single_link_flops + quadratic_flops

        return None

    def time_function(
        self, func: Callable, *args, name: Optional[str] = None, **kwargs
    ) -> TimingResult:
        """Time a function call with warmup and multiple iterations.

        Args:
            func: Function to time
            *args: Positional arguments to pass to func
            name: Optional name for this benchmark
            **kwargs: Keyword arguments to pass to func

        Returns:
            TimingResult with timing and performance metrics
        """
        name = name or func.__name__
        device_type, n_devices = self._get_device_info()
        input_shapes = self._get_input_shapes(args, kwargs)

        # Warmup iterations (includes compilation)
        compile_start = time.perf_counter()
        for _ in range(self.config.warmup_iterations):
            result = func(*args, **kwargs)
            if self.config.sync_device:
                jax.block_until_ready(result)
        compile_end = time.perf_counter()
        compile_time_ms = (
            (compile_end - compile_start) * 1000 / self.config.warmup_iterations
        )

        # Timed iterations
        exec_start = time.perf_counter()
        for _ in range(self.config.timing_iterations):
            result = func(*args, **kwargs)
            if self.config.sync_device:
                jax.block_until_ready(result)
        exec_end = time.perf_counter()

        execution_time_ms = (
            (exec_end - exec_start) * 1000 / self.config.timing_iterations
        )
        wall_time_ms = compile_time_ms + execution_time_ms

        # Get output shape
        if hasattr(result, "shape"):
            output_shape = tuple(result.shape)
        elif hasattr(result, "quadratic"):
            # ResponseResult object
            key = list(result.quadratic.keys())[0]
            output_shape = tuple(result.quadratic[key].shape)
        else:
            output_shape = ()

        # Estimate performance metrics
        estimated_flops = self._estimate_flops(name, input_shapes)

        # Calculate throughput
        total_elements = int(np.prod(output_shape)) if output_shape else 1
        throughput = total_elements / (execution_time_ms / 1000)

        # Estimate memory
        memory_bytes = total_elements * 16  # complex128 = 16 bytes

        timing_result = TimingResult(
            name=name,
            wall_time_ms=wall_time_ms,
            compile_time_ms=compile_time_ms,
            execution_time_ms=execution_time_ms,
            n_iterations=self.config.timing_iterations,
            input_shapes=input_shapes,
            output_shape=output_shape,
            device_type=device_type,
            n_devices=n_devices,
            estimated_flops=estimated_flops,
            memory_bytes=memory_bytes,
            throughput_elements_per_sec=throughput,
        )

        self.results.append(timing_result)
        return timing_result

    def compare(self, name1: str, name2: str) -> Dict:
        """Compare two benchmark results by name."""
        r1 = next((r for r in self.results if r.name == name1), None)
        r2 = next((r for r in self.results if r.name == name2), None)

        if not r1 or not r2:
            raise ValueError(f"Results not found: {name1}, {name2}")

        return {
            "speedup": r1.execution_time_ms / r2.execution_time_ms,
            "compile_speedup": r1.compile_time_ms / r2.compile_time_ms,
            "baseline": name1,
            "optimized": name2,
        }

    def clear(self):
        """Clear stored results."""
        self.results = []


@dataclass
class BenchmarkReport:
    """Report containing all benchmark results."""

    results: List[TimingResult]
    system_info: Dict[str, Any]
    timestamp: str

    def save(self, filepath: str):
        """Save report to JSON file."""
        data = {
            "results": [r.to_dict() for r in self.results],
            "system_info": self.system_info,
            "timestamp": self.timestamp,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Report saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "BenchmarkReport":
        """Load report from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        results = []
        for r in data["results"]:
            # Convert lists back to tuples
            r["input_shapes"] = {k: tuple(v) for k, v in r["input_shapes"].items()}
            r["output_shape"] = tuple(r["output_shape"])
            results.append(TimingResult(**r))

        return cls(
            results=results,
            system_info=data["system_info"],
            timestamp=data["timestamp"],
        )

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "GW_RESPONSE BENCHMARK REPORT",
            f"Timestamp: {self.timestamp}",
            f"Platform: {self.system_info.get('platform', 'unknown')}",
            f"JAX version: {self.system_info.get('jax_version', 'unknown')}",
            f"Devices: {self.system_info.get('n_devices', 'unknown')}x {self.system_info.get('device_type', 'unknown')}",
            "=" * 60,
            "",
        ]

        for r in self.results:
            lines.append(f"{r.name}:")
            lines.append(f"  Execution: {r.execution_time_ms:.3f} ms")
            lines.append(f"  Compile: {r.compile_time_ms:.3f} ms")
            if r.throughput_elements_per_sec:
                lines.append(f"  Throughput: {r.throughput_elements_per_sec:.2e} elem/s")
            if r.estimated_flops:
                gflops = r.estimated_flops / 1e9
                gflops_per_sec = gflops / (r.execution_time_ms / 1000)
                lines.append(f"  Performance: {gflops_per_sec:.2f} GFLOP/s")
            lines.append("")

        return "\n".join(lines)


class BenchmarkSuite:
    """Pre-defined benchmark suite for gw_response."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.benchmark = Benchmark(config)

    def _get_system_info(self) -> Dict:
        """Collect system information."""
        import platform

        devices = jax.devices()
        device_type = devices[0].device_kind if devices else "cpu"

        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "jax_version": jax.__version__,
            "device_type": device_type,
            "n_devices": len(devices),
            "devices": [{"kind": d.device_kind, "id": d.id} for d in devices],
        }

    def run_single_link_benchmarks(self) -> List[TimingResult]:
        """Benchmark single link response at various sizes."""
        from . import (
            LISA,
            Pixel,
            unit_vec,
            uv_analytical,
            polarization_tensors_LR,
            get_single_link_response,
        )

        results = []
        lisa = LISA()

        # Test different sizes
        test_configs = [
            {"nside": 8, "n_freq": 100, "n_times": 1, "label": "small"},
            {"nside": 8, "n_freq": 300, "n_times": 1, "label": "medium_freq"},
            {"nside": 16, "n_freq": 100, "n_times": 1, "label": "medium_pixels"},
            {"nside": 8, "n_freq": 100, "n_times": 10, "label": "multi_time"},
        ]

        for cfg in test_configs:
            pixel = Pixel(NSIDE=cfg["nside"])
            freqs = jnp.logspace(-4, -1, cfg["n_freq"])
            times = jnp.linspace(0, 1, cfg["n_times"])

            k_vector = unit_vec(pixel.theta_pixel, pixel.phi_pixel)
            u, v = uv_analytical(pixel.theta_pixel, pixel.phi_pixel)
            p1, _ = polarization_tensors_LR(u, v)

            positions = lisa.satellite_positions(times) / lisa.armlength
            arms_matrix = lisa.detector_arms(times) / lisa.armlength
            x_array = lisa.x(freqs)

            name = f"single_link_{cfg['label']}"
            result = self.benchmark.time_function(
                get_single_link_response,
                p1,
                arms_matrix,
                k_vector,
                x_array,
                positions,
                name=name,
            )
            results.append(result)

        return results

    def run_full_pipeline_benchmarks(self) -> List[TimingResult]:
        """Benchmark full compute_response pipeline."""
        from . import compute_response

        results = []

        test_configs = [
            {"n_freq": 100, "nside": 8, "label": "small"},
            {"n_freq": 300, "nside": 8, "label": "medium_freq"},
            {"n_freq": 100, "nside": 16, "label": "medium_pixels"},
        ]

        for cfg in test_configs:
            freqs = jnp.logspace(-4, -1, cfg["n_freq"])

            name = f"compute_response_{cfg['label']}"
            result = self.benchmark.time_function(
                compute_response, freqs, nside=cfg["nside"], name=name
            )
            results.append(result)

        return results

    def run_all(self) -> BenchmarkReport:
        """Run complete benchmark suite."""
        print("Running single_link benchmarks...")
        self.run_single_link_benchmarks()

        print("Running full pipeline benchmarks...")
        self.run_full_pipeline_benchmarks()

        return BenchmarkReport(
            results=self.benchmark.results,
            system_info=self._get_system_info(),
            timestamp=datetime.now().isoformat(),
        )


class DeviceProfiler:
    """Profile device utilization during computation."""

    def profile_memory(self) -> Dict:
        """Get current memory usage on all devices."""
        devices = jax.devices()
        memory_stats = {}

        for device in devices:
            try:
                stats = device.memory_stats()
                memory_stats[f"{device.device_kind}:{device.id}"] = {
                    "bytes_in_use": stats.get("bytes_in_use", 0),
                    "peak_bytes_in_use": stats.get("peak_bytes_in_use", 0),
                    "bytes_limit": stats.get("bytes_limit", 0),
                }
            except Exception:
                memory_stats[f"{device.device_kind}:{device.id}"] = None

        return memory_stats

    def estimate_bandwidth(
        self,
        input_shapes: Dict[str, tuple],
        output_shape: tuple,
        execution_time_ms: float,
        dtype_bytes: int = 16,  # complex128
    ) -> Dict:
        """Estimate memory bandwidth usage."""
        input_bytes = sum(int(np.prod(shape)) * dtype_bytes for shape in input_shapes.values())
        output_bytes = int(np.prod(output_shape)) * dtype_bytes
        total_bytes = input_bytes + output_bytes

        bandwidth_gb_s = (total_bytes / 1e9) / (execution_time_ms / 1000)

        return {
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "total_bytes": total_bytes,
            "bandwidth_gb_s": bandwidth_gb_s,
        }


def timed(func):
    """Decorator to add timing to any function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}: {elapsed*1000:.2f} ms")
        return result

    return wrapper


def compare_reports(baseline_path: str, optimized_path: str) -> str:
    """Compare two benchmark reports and return summary."""
    baseline = BenchmarkReport.load(baseline_path)
    optimized = BenchmarkReport.load(optimized_path)

    lines = [
        "=" * 60,
        "BENCHMARK COMPARISON",
        f"Baseline: {baseline_path}",
        f"Optimized: {optimized_path}",
        "=" * 60,
        "",
    ]

    for r1 in baseline.results:
        r2 = next((r for r in optimized.results if r.name == r1.name), None)
        if r2:
            speedup = r1.execution_time_ms / r2.execution_time_ms
            lines.append(f"{r1.name}:")
            lines.append(f"  Baseline: {r1.execution_time_ms:.2f} ms")
            lines.append(f"  Optimized: {r2.execution_time_ms:.2f} ms")
            lines.append(f"  Speedup: {speedup:.2f}x")
            lines.append("")

    return "\n".join(lines)
