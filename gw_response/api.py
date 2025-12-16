"""
High-level API for GW response computation.

This module provides convenience functions that wrap the lower-level
Response class with sensible defaults for common use cases.

Example:
    >>> import gw_response as gwr
    >>> import jax.numpy as jnp
    >>>
    >>> # Simple one-liner for LISA response
    >>> result = gwr.compute_response(frequencies=jnp.logspace(-4, -1, 100))
    >>>
    >>> # Access results easily
    >>> print(result.AA)  # A-A TDI response
    >>> print(result.diagonal())  # All diagonal elements
"""

from typing import Optional, Union

import jax.numpy as jnp

from .response import Response
from .lisa import LISA
from .detector import Detector
from .constants import PhysicalConstants
from .utils import Pixel
from .results import ResponseResult
from .presets import TDIBasis, PolarizationBasis, list_tdi_options


def compute_response(
    frequencies: jnp.ndarray,
    *,
    times: Union[float, jnp.ndarray] = 0.0,
    tdi: TDIBasis = "AET",
    polarization: PolarizationBasis = "LR",
    detector: Optional[Detector] = None,
    nside: int = 8,
    include_intermediate: bool = False,
    parallel: bool = False,
) -> ResponseResult:
    """
    Compute the GW detector response function.

    This is the recommended high-level entry point for computing response
    functions. It handles all the boilerplate of setting up the detector,
    pixel grid, and coordinate transformations.

    Args:
        frequencies: Array of frequencies in Hz at which to compute the response.
            Use jnp.logspace(-4, -1, 100) for a typical range covering
            0.1 mHz to 0.1 Hz.

        times: Observation time(s) in years. Can be a single float (default: 0.0)
            or an array for time-dependent response. Time 0.0 corresponds to
            the initial LISA constellation orientation.

        tdi: TDI combination to use. Options:
            - "XYZ": First-generation Michelson
            - "AET": Orthogonalized, decorrelated noise (RECOMMENDED)
            - "Sagnac": Sagnac interferometer
            - "AET_Sagnac": Orthogonalized Sagnac
            - "AE_zeta": A, E plus symmetric Sagnac
            - "AE_Sagnac_zeta": Sagnac A, E plus zeta

            Use gwr.list_tdi_options() for detailed guidance.

        polarization: Polarization basis. Options:
            - "LR": Left/Right circular (default)
            - "PC": Plus/Cross linear

        detector: Detector configuration. Defaults to LISA() with standard
            parameters. Pass a custom LISA() or other Detector subclass
            for modified configurations.

        nside: HEALPix NSIDE parameter for sky discretization. Higher values
            give more accurate sky integration at the cost of computation time.
            Default: 8 (768 pixels). Use 16 (3072 pixels) for high precision.

        include_intermediate: If True, include single_link and linear response
            in the result. Useful for debugging or advanced analyses.
            Default: False (saves memory).

        parallel: If True, use multi-device parallelization via JAX pmap.
            Automatically distributes computation across available GPUs/TPUs.
            Provides significant speedup with multiple devices but has overhead
            on single-device systems. Default: False.

    Returns:
        ResponseResult: Immutable dataclass containing the computed response.
            Access quadratic response via result.LL, result.AA, result.diagonal(), etc.

    Example:
        Basic usage for unpolarized stochastic background::

            >>> import gw_response as gwr
            >>> import jax.numpy as jnp
            >>>
            >>> freqs = jnp.logspace(-4, -1, 200)
            >>> result = gwr.compute_response(freqs, tdi="AET")
            >>>
            >>> # Response for LL polarization
            >>> response_LL = result.LL
            >>>
            >>> # Sum over polarizations (for unpolarized background)
            >>> total_response = result.sum_polarizations()
            >>>
            >>> # Just the diagonal TDI elements (AA, EE, TT)
            >>> diag = result.diagonal()

        Time-dependent response::

            >>> times = jnp.linspace(0, 1, 12)  # Monthly snapshots over 1 year
            >>> result = gwr.compute_response(freqs, times=times, tdi="AET")
            >>> print(result.shape)  # (12, 200, 3, 3)

        Custom detector configuration::

            >>> lisa_short = gwr.LISA(armlength=1.5e9)  # Shorter arms
            >>> result = gwr.compute_response(freqs, detector=lisa_short)

    See Also:
        Response: Lower-level class for step-by-step computation
        list_tdi_options: Print guide to TDI combinations
        ResponseResult: Documentation of result object
    """
    # Set up detector
    if detector is None:
        detector = LISA()

    # Normalize times to array
    if isinstance(times, (int, float)):
        times_array = jnp.array([float(times)])
    else:
        times_array = jnp.asarray(times)

    # Ensure frequencies is a JAX array
    frequencies = jnp.asarray(frequencies)

    # Use parallel implementation if requested
    if parallel:
        from .parallel import parallel_compute_response

        return parallel_compute_response(
            frequencies,
            times=times_array,
            tdi=tdi,
            polarization=polarization,
            detector=detector,
            nside=nside,
        )

    # Set up pixel grid
    pixel = Pixel(NSIDE=nside)
    theta = pixel.theta_pixel
    phi = pixel.phi_pixel

    # Create response object and compute
    response = Response(
        ps=PhysicalConstants(),
        det=detector,
    )

    response.compute_detector(
        times_in_years=times_array,
        theta_array=theta,
        phi_array=phi,
        frequency_array=frequencies,
        TDI=tdi,
        polarization=polarization,
    )

    # Extract results into proper structure
    pol = polarization.upper()
    quadratic_data = {}

    for p in pol:
        key = 2 * p  # "LL", "RR" or "PP", "CC"
        quadratic_data[key] = response.quadratic_integrated[tdi][key]

    # Build result object
    result_kwargs = dict(
        frequencies=frequencies,
        times=times_array,
        tdi=tdi,
        polarization=polarization,
        quadratic=quadratic_data,
        detector_name=detector.name,
        nside=nside,
    )

    if include_intermediate:
        linear_data = {}
        for p in pol:
            linear_data[p] = response.linear_integrand[tdi][p]
        result_kwargs["linear"] = linear_data
        result_kwargs["single_link"] = dict(response.single_link_response)

        quad_integrand_data = {}
        for p in pol:
            quad_integrand_data[2 * p] = response.quadratic_integrand[tdi][2 * p]
        result_kwargs["quadratic_integrand"] = quad_integrand_data

    return ResponseResult(**result_kwargs)


def quick_response(
    freq_min: float = 1e-4,
    freq_max: float = 1e-1,
    n_freq: int = 100,
    tdi: TDIBasis = "AET",
) -> ResponseResult:
    """
    Compute response with minimal configuration.

    This is the simplest possible interface - just specify a frequency range.
    Uses all default settings (LISA, time=0, LR polarization, nside=8).

    Args:
        freq_min: Minimum frequency in Hz. Default: 0.1 mHz
        freq_max: Maximum frequency in Hz. Default: 0.1 Hz
        n_freq: Number of frequency points. Default: 100
        tdi: TDI combination. Default: "AET"

    Returns:
        ResponseResult: The computed response.

    Example:
        >>> result = gwr.quick_response()  # All defaults
        >>> result = gwr.quick_response(1e-3, 1e-2, 50)  # Custom range
    """
    frequencies = jnp.logspace(jnp.log10(freq_min), jnp.log10(freq_max), n_freq)
    return compute_response(frequencies, tdi=tdi)


# Re-export for convenience at module level
__all__ = ["compute_response", "quick_response", "list_tdi_options"]
