from __future__ import annotations

# Global imports
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from jax.typing import ArrayLike

# Local imports
from gw_response.utils import as_time_array

if TYPE_CHECKING:
    from gw_response.noise import Noise
    from gw_response.response import Response


class Detector(ABC):
    """Abstract base class for a gravitational wave detector.

    Concrete subclasses (e.g. ``LISA``, ``LIGO``) must provide the
    detector's basic characteristics as class attributes, and implement the
    methods that describe its geometry, response and noise.

    Attributes:
        name: Human-readable name of the detector.
        fmin: Minimum frequency of the detector's sensitive band, in Hz.
        fmax: Maximum frequency of the detector's sensitive band, in Hz.
        armlength: Nominal detector arm length, in meters.
        res: Expected relative resolution/precision of the detector.
        ps: Detector-specific physical/instrumental parameters.
        default_combination: Name of the default readout combination (e.g.
            a TDI variable for LISA) used when none is specified.
        response: Response object used to compute the detector's response
            to gravitational waves.
        noise: Noise object used to compute the detector's noise budget.
    """

    name: str
    fmin: float
    fmax: float
    armlength: float
    res: float
    ps: Any
    default_combination: str
    # Concrete detectors (LISA, LIGO) will instantiate real Response/Noise objects
    response: Response
    noise: Noise

    def vertex_positions(self, time_in_years: ArrayLike) -> jax.Array:
        """Computes satellite/vertex positions at the given time(s).

        Args:
            time_in_years: Time(s), in years, at which to evaluate the
                vertex positions.

        Returns:
            Array of vertex positions.
        """
        return self._vertex_positions(as_time_array(time_in_years))

    @abstractmethod
    def _vertex_positions(self, time_in_years: ArrayLike) -> jax.Array:
        pass

    def detector_arms(self, time_in_years: ArrayLike) -> jax.Array:
        """Computes the detector's arm matrix at the given time(s).

        Args:
            time_in_years: Time(s), in years, at which to evaluate the
                detector arms.

        Returns:
            Array representing the vector between each pair of vertices
            (i.e. each detector arm).
        """
        return self._detector_arms(as_time_array(time_in_years))

    @abstractmethod
    def _detector_arms(self, time_in_years: ArrayLike) -> jax.Array:
        pass

    def frequency_vec(self, freq_pts):
        """
        Generates a frequency vector within the LISA frequency range.

        Args:
            freq_pts (int): The number of frequency points to generate.

        Returns:
            jnp.ndarray: A linearly spaced array of frequency points within
            LISA's operational frequency range, starting from LISA_fmin to
            LISA_fmax.
        """
        return jnp.linspace(self.fmin, self.fmax, freq_pts)

    def klvector(self, frequency_vec):
        """
        Computes the kl-vector for a given frequency vector in the context of
        the LISA configuration.

        Args:
            frequency_vec (jnp.ndarray): An array of frequency values for which
            the kl-vector is to be computed.

        Returns:
            jnp.ndarray: An array representing the kl-vector, which is a product
            of the frequency vector, the LISA arm length, and the inverse of the
            speed of light.
        """
        return frequency_vec * self.armlength / self.ps.light_speed

    def x(self, frequency_vec):
        """
        Computes the x-parameter for a given frequency vector based on the LISA
        configuration.

        Args:
            frequency_vec (jnp.ndarray): An array of frequency values for which
            the x-parameter is to be computed.

        Returns:
            jnp.ndarray: An array representing the x-parameter, calculated as
            2π times the kl-vector for the given frequency vector.
        """
        return 2 * jnp.pi * self.klvector(frequency_vec)

    @abstractmethod
    def combination_matrix(
        self, combination, arms_matrix_rescaled, x_vector
    ) -> jax.Array:
        """
        Builds the mixing matrix that turns per-link responses into the
        detector's readout channel(s) for the requested combination (e.g. a
        TDI variable for LISA, the Michelson combination for LIGO).

        Returns an array of shape (..., x_vector, channels, arms).
        """

    @abstractmethod
    def linear_response_from_single_link(
        self, single_link: dict, combination_matrix: jax.Array
    ) -> dict:
        """
        Applies `combination_matrix` (as built by `combination_matrix`) to
        each polarization's single-link response, returning the dict of
        linear integrands (keyed by polarization) for this detector.
        """

    @abstractmethod
    def quadratic_response_from_single_link(self, linear_integrand: dict) -> dict:
        """
        Given the dict of linear integrands (keyed by polarization), returns
        the dict of quadratic integrands for this detector.
        """

    @abstractmethod
    def integrate_quadratic_response(self, quadratic_integrand: dict) -> dict:
        """
        Given the dict of quadratic integrands (keyed by polarization) for a
        single combination, returns the corresponding integrated response.
        """

    @abstractmethod
    def single_link_noise(
        self, frequency_array, arms_matrix_rescaled, x_vector, **noise_parameters
    ) -> jax.Array:
        """
        Builds the per-link noise covariance (or, for detectors with no
        per-link decomposition, the already-combined noise) before
        projection into a readout combination. `noise_parameters` are
        whatever detector-specific noise parameters this needs (e.g. LISA's
        TM_acceleration_parameters/OMS_parameters); LIGO takes none.
        """

    @abstractmethod
    def project_noise(
        self, combination_matrix: jax.Array, single_link_noise
    ) -> jax.Array:
        """
        Projects `single_link_noise` (as built by `single_link_noise`) into
        the readout basis defined by `combination_matrix` (as built by
        `combination_matrix`), e.g. via the congruence transform
        combination_matrix @ single_link_noise @ combination_matrix^H.
        Detectors without a per-link decomposition (e.g. LIGO) can simply
        return `single_link_noise` unchanged.
        """
