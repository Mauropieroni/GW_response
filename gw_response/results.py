"""
Result dataclass for GW response computations.

This module provides the ResponseResult dataclass which replaces the previous
nested dictionary approach with a typed, immutable result object.
"""

import chex
import jax.numpy as jnp
from typing import Dict, Optional, Tuple


@chex.dataclass(frozen=True)
class ResponseResult:
    """
    Immutable result object from GW response computation.

    Provides structured access to computed response functions with convenient
    properties for common access patterns.

    Attributes:
        frequencies: Array of frequencies (Hz) at which response was computed.
        times: Array of times (years) at which response was computed.
        tdi: TDI combination used (e.g., "XYZ", "AET").
        polarization: Polarization basis used ("LR" or "PC").
        quadratic: Dict mapping polarization pairs to response arrays.
            Keys are "LL", "RR" for LR basis or "PP", "CC" for PC basis.
            Each value has shape (n_times, n_frequencies, n_tdi, n_tdi).
        linear: Optional dict of linear response (angular integrand).
        single_link: Optional dict of single-link response.
        quadratic_integrand: Optional dict of quadratic integrand before integration.
        detector_name: Name of the detector (e.g., "LISA").
        nside: HEALPix NSIDE used for sky integration.

    Example:
        >>> result = gwr.compute_response(frequencies=freqs)
        >>> result.LL  # Get LL polarization response
        >>> result.AA  # Get A-A TDI diagonal element
        >>> result.diagonal()  # Get all diagonal TDI elements
    """

    # Core data
    frequencies: jnp.ndarray
    times: jnp.ndarray
    tdi: str
    polarization: str

    # Response data (required)
    quadratic: Dict[str, jnp.ndarray]

    # Optional intermediate results (for expert users)
    linear: Optional[Dict[str, jnp.ndarray]] = None
    single_link: Optional[Dict[str, jnp.ndarray]] = None
    quadratic_integrand: Optional[Dict[str, jnp.ndarray]] = None

    # Metadata
    detector_name: str = "LISA"
    nside: int = 8

    @property
    def LL(self) -> Optional[jnp.ndarray]:
        """Left-Left polarization response (LR basis only)."""
        return self.quadratic.get("LL")

    @property
    def RR(self) -> Optional[jnp.ndarray]:
        """Right-Right polarization response (LR basis only)."""
        return self.quadratic.get("RR")

    @property
    def PP(self) -> Optional[jnp.ndarray]:
        """Plus-Plus polarization response (PC basis only)."""
        return self.quadratic.get("PP")

    @property
    def CC(self) -> Optional[jnp.ndarray]:
        """Cross-Cross polarization response (PC basis only)."""
        return self.quadratic.get("CC")

    @property
    def XX(self) -> jnp.ndarray:
        """X-X TDI response (first diagonal element, for XYZ basis)."""
        key = list(self.quadratic.keys())[0]
        return self.quadratic[key][..., 0, 0]

    @property
    def YY(self) -> jnp.ndarray:
        """Y-Y TDI response (second diagonal element, for XYZ basis)."""
        key = list(self.quadratic.keys())[0]
        return self.quadratic[key][..., 1, 1]

    @property
    def ZZ(self) -> jnp.ndarray:
        """Z-Z TDI response (third diagonal element, for XYZ basis)."""
        key = list(self.quadratic.keys())[0]
        return self.quadratic[key][..., 2, 2]

    @property
    def AA(self) -> jnp.ndarray:
        """A-A TDI response (first diagonal element, for AET basis)."""
        return self.XX

    @property
    def EE(self) -> jnp.ndarray:
        """E-E TDI response (second diagonal element, for AET basis)."""
        return self.YY

    @property
    def TT(self) -> jnp.ndarray:
        """T-T TDI response (third diagonal element, for AET basis)."""
        return self.ZZ

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the quadratic response arrays."""
        key = list(self.quadratic.keys())[0]
        return self.quadratic[key].shape

    def diagonal(self, polarization: str = None) -> jnp.ndarray:
        """
        Extract diagonal TDI elements (XX, YY, ZZ or AA, EE, TT).

        Args:
            polarization: Which polarization to use. Defaults to first available.

        Returns:
            Array of shape (n_times, n_frequencies, n_tdi) with diagonal elements.
        """
        if polarization is None:
            polarization = list(self.quadratic.keys())[0]
        data = self.quadratic[polarization]
        return jnp.diagonal(data, axis1=-2, axis2=-1)

    def sum_polarizations(self) -> jnp.ndarray:
        """
        Sum over polarization states (for unpolarized backgrounds).

        Returns:
            Array of shape (n_times, n_frequencies, n_tdi, n_tdi).
        """
        return sum(self.quadratic.values())

    def at_frequency(self, freq: float) -> "ResponseResult":
        """
        Extract response at a specific frequency (nearest neighbor).

        Args:
            freq: Target frequency in Hz.

        Returns:
            New ResponseResult at the specified frequency.
        """
        idx = jnp.argmin(jnp.abs(self.frequencies - freq))
        new_quadratic = {k: v[:, idx : idx + 1, ...] for k, v in self.quadratic.items()}

        return ResponseResult(
            frequencies=self.frequencies[idx : idx + 1],
            times=self.times,
            tdi=self.tdi,
            polarization=self.polarization,
            quadratic=new_quadratic,
            detector_name=self.detector_name,
            nside=self.nside,
        )

    def __repr__(self) -> str:
        key = list(self.quadratic.keys())[0]
        shape = self.quadratic[key].shape
        freq_min = float(self.frequencies[0])
        freq_max = float(self.frequencies[-1])
        return (
            f"ResponseResult(\n"
            f"  detector={self.detector_name},\n"
            f"  tdi={self.tdi!r},\n"
            f"  polarization={self.polarization!r},\n"
            f"  frequencies={len(self.frequencies)} pts from {freq_min:.2e} to {freq_max:.2e} Hz,\n"
            f"  times={len(self.times)} pts,\n"
            f"  shape={shape}\n"
            f")"
        )
