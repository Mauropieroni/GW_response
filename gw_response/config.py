"""
Configuration utilities for JAX and XLA optimization.

This module provides functions to configure JAX and XLA for optimal
performance on different hardware (CPU, GPU, TPU).

Example usage:
    from gw_response.config import configure_for_performance

    # Call before any JAX operations
    configure_for_performance()
"""

import os
from typing import Optional


def configure_xla_flags(
    gpu: bool = True,
    tpu: bool = False,
    enable_fast_math: bool = True,
    enable_async: bool = True,
):
    """
    Set XLA compilation flags for optimal performance.

    Call this BEFORE importing JAX or running any JAX operations.

    Args:
        gpu: Enable GPU-specific optimizations
        tpu: Enable TPU-specific optimizations
        enable_fast_math: Enable fast math operations (may reduce precision slightly)
        enable_async: Enable asynchronous execution

    Note:
        These flags affect XLA compilation behavior. Some flags may not be
        available on all hardware/software versions.
    """
    flags = []

    if gpu:
        if enable_fast_math:
            flags.append("--xla_gpu_enable_fast_min_max=true")
        if enable_async:
            flags.append("--xla_gpu_enable_async_all_reduce=true")
        # Enable latency hiding for better GPU utilization
        flags.append("--xla_gpu_enable_latency_hiding_scheduler=true")

    if tpu:
        flags.append("--xla_tpu_enable_data_parallel_all_reduce_opt=true")

    if flags:
        current_flags = os.environ.get("XLA_FLAGS", "")
        new_flags = " ".join(flags)
        if current_flags:
            os.environ["XLA_FLAGS"] = f"{current_flags} {new_flags}"
        else:
            os.environ["XLA_FLAGS"] = new_flags


def configure_jax_memory(
    preallocate_fraction: Optional[float] = None,
    enable_compilation_cache: bool = True,
    cache_dir: str = "/tmp/jax_cache",
):
    """
    Configure JAX memory allocation and compilation caching.

    Args:
        preallocate_fraction: Fraction of GPU memory to preallocate (0.0-1.0).
            None uses default (no preallocation). Set to 0.9 for production
            to reduce memory fragmentation.
        enable_compilation_cache: Enable persistent compilation cache
        cache_dir: Directory for compilation cache
    """
    import jax

    if preallocate_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(preallocate_fraction)

    if enable_compilation_cache:
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)


def configure_for_performance(
    device: str = "auto",
    preallocate_memory: bool = False,
    enable_cache: bool = True,
):
    """
    One-stop configuration for optimal JAX performance.

    Args:
        device: Device type - "auto", "cpu", "gpu", or "tpu"
        preallocate_memory: Whether to preallocate GPU memory
        enable_cache: Whether to enable compilation caching

    Example:
        >>> from gw_response.config import configure_for_performance
        >>> configure_for_performance()  # Auto-detect and configure
    """
    import jax

    # Detect device type if auto
    if device == "auto":
        devices = jax.devices()
        if devices:
            device = devices[0].device_kind
        else:
            device = "cpu"

    # Apply XLA flags based on device
    if device in ("gpu", "cuda"):
        configure_xla_flags(gpu=True, tpu=False)
    elif device == "tpu":
        configure_xla_flags(gpu=False, tpu=True)

    # Configure memory
    memory_fraction = 0.9 if preallocate_memory else None
    configure_jax_memory(
        preallocate_fraction=memory_fraction,
        enable_compilation_cache=enable_cache,
    )


def get_device_info() -> dict:
    """
    Get information about available JAX devices.

    Returns:
        Dictionary with device information
    """
    import jax

    devices = jax.devices()
    return {
        "n_devices": len(devices),
        "device_type": devices[0].device_kind if devices else "none",
        "devices": [
            {"id": d.id, "kind": d.device_kind, "platform": d.platform}
            for d in devices
        ],
        "default_backend": jax.default_backend(),
    }


def print_device_info():
    """Print device information to console."""
    info = get_device_info()
    print(f"JAX Devices: {info['n_devices']}x {info['device_type']}")
    print(f"Default backend: {info['default_backend']}")
    for d in info["devices"]:
        print(f"  [{d['id']}] {d['kind']} ({d['platform']})")
