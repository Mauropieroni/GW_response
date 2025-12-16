"""
Multi-device parallelization utilities using JAX pmap.

This module provides parallel implementations of key gw_response functions
that can run across multiple GPUs/TPUs using JAX's pmap.

Example usage:
    from gw_response.parallel import parallel_compute_response

    # Automatically uses all available devices
    result = parallel_compute_response(frequencies, tdi="AET")
"""

import jax
import jax.numpy as jnp
from jax import pmap
from functools import partial
from typing import Optional, Union

from .constants import PhysicalConstants
from .lisa import LISA
from .detector import Detector
from .utils import Pixel
from .single_link import (
    unit_vec,
    uv_analytical,
    polarization_tensors_LR,
    polarization_tensors_PC,
    geometrical_factor,
    xi_k_Avec_func,
    single_link_response,
    quadratic_integrand,
    quadratic_response_integrated,
)
from .tdi import TDI_map
from .results import ResponseResult


def get_device_count() -> int:
    """Get number of available JAX devices."""
    return len(jax.devices())


def pad_to_device_count(array: jnp.ndarray, axis: int, n_devices: int):
    """Pad array along axis to be divisible by n_devices.

    Args:
        array: Input array
        axis: Axis along which to pad
        n_devices: Number of devices

    Returns:
        Tuple of (padded array, original size along axis)
    """
    current_size = array.shape[axis]
    remainder = current_size % n_devices
    if remainder == 0:
        return array, current_size

    pad_size = n_devices - remainder
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, pad_size)
    padded = jnp.pad(array, pad_width, mode="edge")
    return padded, current_size


def unpad_result(array: jnp.ndarray, original_size: int, axis: int) -> jnp.ndarray:
    """Remove padding from result array."""
    slices = [slice(None)] * array.ndim
    slices[axis] = slice(0, original_size)
    return array[tuple(slices)]


def reshape_for_pmap(array: jnp.ndarray, axis: int, n_devices: int) -> jnp.ndarray:
    """Reshape array to have device dimension first.

    Args:
        array: Input array
        axis: Axis to distribute across devices
        n_devices: Number of devices

    Returns:
        Reshaped array with shape (n_devices, size_per_device, ...)
    """
    # Move target axis to position 0, then split
    array = jnp.moveaxis(array, axis, 0)
    size_per_device = array.shape[0] // n_devices
    new_shape = (n_devices, size_per_device) + array.shape[1:]
    return array.reshape(new_shape)


def reshape_from_pmap(
    array: jnp.ndarray, axis: int, original_ndim: int
) -> jnp.ndarray:
    """Reshape pmap result back to original layout.

    Args:
        array: Array from pmap with shape (n_devices, size_per_device, ...)
        axis: Original axis that was distributed
        original_ndim: Number of dimensions in original array

    Returns:
        Array with device dimension merged and moved back to original axis
    """
    # Merge first two dimensions
    merged = array.reshape((-1,) + array.shape[2:])
    # Move back to original axis
    return jnp.moveaxis(merged, 0, axis)


def _create_single_link_pmap_kernel():
    """Create pmap kernel for single_link_response over pixels."""

    # in_axes specifies which axis to map for each argument:
    # - 0 means map over axis 0 (sharded across devices)
    # - None means broadcast (replicate to all devices)
    @partial(pmap, axis_name="pixel", in_axes=(0, None, 0, None, None))
    def kernel(polarization_shard, arms_matrix, k_vector_shard, x_array, positions):
        """Compute single_link_response for a shard of pixels.

        Args:
            polarization_shard: (pixels_per_device, 3, 3) - sharded
            arms_matrix: (configs, 3, 6) - broadcast
            k_vector_shard: (3, pixels_per_device) - sharded
            x_array: (freq,) - broadcast
            positions: (configs, 3, 3) - broadcast
        """
        geometrical = geometrical_factor(arms_matrix, polarization_shard)
        xi_k_vec = xi_k_Avec_func(arms_matrix, k_vector_shard, x_array, geometrical)
        return single_link_response(
            positions, arms_matrix, k_vector_shard, x_array, xi_k_vec
        )

    return kernel


# Pre-create the kernel
_single_link_pmap = _create_single_link_pmap_kernel()


def parallel_single_link_response(
    polarization_tensor: jnp.ndarray,
    arms_matrix: jnp.ndarray,
    k_vector: jnp.ndarray,
    x_array: jnp.ndarray,
    positions: jnp.ndarray,
) -> jnp.ndarray:
    """Compute single_link_response using pmap over pixels.

    Args:
        polarization_tensor: (pixels, 3, 3)
        arms_matrix: (configs, 3, 6)
        k_vector: (3, pixels)
        x_array: (freq,)
        positions: (configs, 3, 3)

    Returns:
        Single link response array (configs, freq, arms, pixels)
    """
    n_devices = get_device_count()
    n_pixels = polarization_tensor.shape[0]

    if n_devices == 1:
        # Fall back to serial computation
        geometrical = geometrical_factor(arms_matrix, polarization_tensor)
        xi_k_vec = xi_k_Avec_func(arms_matrix, k_vector, x_array, geometrical)
        return single_link_response(
            positions, arms_matrix, k_vector, x_array, xi_k_vec
        )

    # Pad arrays for even distribution
    pol_padded, orig_pixels = pad_to_device_count(polarization_tensor, 0, n_devices)
    k_padded, _ = pad_to_device_count(k_vector, 1, n_devices)

    # Reshape for pmap
    pol_sharded = reshape_for_pmap(pol_padded, 0, n_devices)
    k_sharded = reshape_for_pmap(k_padded.T, 0, n_devices)  # Transpose for pixel axis
    k_sharded = jnp.moveaxis(k_sharded, -1, 1)  # Back to (n_devices, 3, pixels_per_device)

    # Run parallel computation
    result = _single_link_pmap(pol_sharded, arms_matrix, k_sharded, x_array, positions)

    # Result shape: (n_devices, configs, freq, arms, pixels_per_device)
    # Reshape to (configs, freq, arms, n_devices * pixels_per_device)
    result = jnp.moveaxis(result, 0, -1)  # Move device dim to end
    result = result.reshape(result.shape[:-2] + (-1,))  # Merge last two dims

    # Remove padding
    result = result[..., :orig_pixels]

    return result


def parallel_compute_response(
    frequencies: jnp.ndarray,
    *,
    times: Union[float, jnp.ndarray] = 0.0,
    tdi: str = "AET",
    polarization: str = "LR",
    detector: Optional[Detector] = None,
    nside: int = 8,
) -> ResponseResult:
    """Compute response using multi-device parallelization.

    This function parallelizes computation across available devices (GPUs/TPUs)
    by distributing the pixel dimension.

    Args:
        frequencies: Array of frequencies in Hz
        times: Observation time(s) in years
        tdi: TDI combination (XYZ, AET, etc.)
        polarization: Polarization basis (LR or PC)
        detector: Detector configuration (default: LISA)
        nside: HEALPix NSIDE parameter

    Returns:
        ResponseResult with computed response
    """
    n_devices = get_device_count()

    # Set up detector
    if detector is None:
        detector = LISA()

    # Normalize times
    if isinstance(times, (int, float)):
        times_array = jnp.array([float(times)])
    else:
        times_array = jnp.asarray(times)

    # Set up pixel grid
    pixel = Pixel(NSIDE=nside)
    theta = pixel.theta_pixel
    phi = pixel.phi_pixel

    frequencies = jnp.asarray(frequencies)

    # Compute unit vectors and polarization tensors
    k_vector = unit_vec(theta, phi)
    u, v = uv_analytical(theta, phi)

    pol = polarization.upper()
    if pol == "PC":
        p1, p2 = polarization_tensors_PC(u, v)
    elif pol == "LR":
        p1, p2 = polarization_tensors_LR(u, v)
    else:
        raise ValueError(f"Unknown polarization: {polarization}")

    # Get detector parameters
    positions = detector.satellite_positions(times_array) / detector.armlength
    arms_matrix = detector.detector_arms(times_array) / detector.armlength
    x_array = detector.x(frequencies)

    # Compute single-link response for each polarization
    # For multi-device, we use pmap over pixels
    single_link_results = {}
    for p_name, p_tensor in [(pol[0], p1), (pol[1], p2)]:
        if n_devices > 1:
            single_link_results[p_name] = parallel_single_link_response(
                p_tensor, arms_matrix, k_vector, x_array, positions
            )
        else:
            # Single device: use standard computation
            geometrical = geometrical_factor(arms_matrix, p_tensor)
            xi_k_vec = xi_k_Avec_func(arms_matrix, k_vector, x_array, geometrical)
            single_link_results[p_name] = single_link_response(
                positions, arms_matrix, k_vector, x_array, xi_k_vec
            )

    # Compute quadratic integrand and integration
    quadratic_data = {}
    tdi_idx = TDI_map[tdi]

    for p_name in [pol[0], pol[1]]:
        quad_int = quadratic_integrand(
            tdi_idx, single_link_results[p_name], arms_matrix, x_array
        )
        quad_integrated = quadratic_response_integrated(quad_int)
        quadratic_data[2 * p_name] = quad_integrated

    return ResponseResult(
        frequencies=frequencies,
        times=times_array,
        tdi=tdi,
        polarization=polarization,
        quadratic=quadratic_data,
        detector_name=detector.name,
        nside=nside,
    )


def get_parallel_info() -> dict:
    """Get information about parallel configuration.

    Returns:
        Dictionary with device count and types
    """
    devices = jax.devices()
    return {
        "n_devices": len(devices),
        "device_type": devices[0].device_kind if devices else "none",
        "parallel_enabled": len(devices) > 1,
        "devices": [{"id": d.id, "kind": d.device_kind} for d in devices],
    }
