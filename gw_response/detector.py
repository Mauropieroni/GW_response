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
        """
        Concrete-detector implementation of :meth:`vertex_positions`.

        `time_in_years` is already normalized to an array (of at least 1
        dimension) by :func:`gw_response.utils.as_time_array` before this is
        called, so implementations don't need to handle bare scalars
        themselves.

        Args:
            time_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the vertex positions.

        Returns:
            jax.Array: Array of vertex positions.
        """

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
        """
        Concrete-detector implementation of :meth:`detector_arms`.

        `time_in_years` is already normalized to an array (of at least 1
        dimension) by :func:`gw_response.utils.as_time_array` before this is
        called, so implementations don't need to handle bare scalars
        themselves.

        Args:
            time_in_years (ArrayLike): Time(s), in years, at which to
                evaluate the detector arms.

        Returns:
            jax.Array: Array representing the vector between each pair of
                vertices (i.e. each detector arm).
        """

    def frequency_vec(self, freq_pts: int) -> jax.Array:
        """
        Generates a frequency vector within the detector's frequency range.

        Args:
            freq_pts (int): The number of frequency points to generate.

        Returns:
            jax.Array: A linearly spaced array of frequency points within
                the detector's operational frequency range, starting from
                ``self.fmin`` to ``self.fmax``.
        """
        return jnp.linspace(self.fmin, self.fmax, freq_pts)

    def klvector(self, frequency_vec: ArrayLike) -> jax.Array:
        """
        Computes the kl-vector for a given frequency vector, i.e. the
        detector's arm length in units of the reduced wavelength.

        Args:
            frequency_vec (ArrayLike): An array of frequency values, in Hz,
                for which the kl-vector is to be computed.

        Returns:
            jax.Array: An array representing the kl-vector, which is a
                product of the frequency vector, the detector arm length, and
                the inverse of the speed of light.
        """
        return frequency_vec * self.armlength / self.ps.light_speed

    def x(self, frequency_vec: ArrayLike) -> jax.Array:
        """
        Computes the x-parameter (``2 pi f L / c``) for a given frequency
        vector.

        Args:
            frequency_vec (ArrayLike): An array of frequency values, in Hz,
                for which the x-parameter is to be computed.

        Returns:
            jax.Array: An array representing the x-parameter, calculated as
                2π times the kl-vector for the given frequency vector.
        """
        return 2 * jnp.pi * self.klvector(frequency_vec)

    @abstractmethod
    def combination_matrix(
        self, combination: str, arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike
    ) -> jax.Array:
        """
        Builds the mixing matrix that turns per-link responses into the
        detector's readout channel(s) for the requested combination (e.g. a
        TDI variable for LISA, the Michelson combination for LIGO).

        Args:
            combination (str): Name of the readout combination to build the
                mixing matrix for.
            arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled
                by the arm length.
            x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
                frequency.

        Returns:
            jax.Array: The mixing matrix, of shape (..., x_vector, channels,
                arms).
        """

    @abstractmethod
    def linear_response_from_single_link(
        self, single_link: dict[str, jax.Array], combination_matrix: ArrayLike
    ) -> dict[str, jax.Array]:
        """
        Applies `combination_matrix` (as built by `combination_matrix`) to
        each polarization's single-link response, returning the dict of
        linear integrands (keyed by polarization) for this detector.

        Args:
            single_link (dict): Single-link response per polarization, e.g.
                as returned by
                :meth:`gw_response.response.Response.get_single_link_response`.
            combination_matrix (ArrayLike): Mixing matrix as built by
                :meth:`combination_matrix`.

        Returns:
            dict: The linear response integrand per polarization.
        """

    @abstractmethod
    def quadratic_response_from_single_link(
        self, linear_integrand: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """
        Given the dict of linear integrands (keyed by polarization), returns
        the dict of quadratic integrands for this detector.

        Args:
            linear_integrand (dict): Linear response integrand per
                polarization, as returned by
                :meth:`linear_response_from_single_link`.

        Returns:
            dict: The quadratic response integrand per doubled polarization
                letter (e.g. "LL", "RR").
        """

    @abstractmethod
    def integrate_quadratic_response(
        self, quadratic_integrand: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """
        Given the dict of quadratic integrands (keyed by polarization) for a
        single combination, returns the corresponding integrated response.

        Args:
            quadratic_integrand (dict): Quadratic response integrand per
                doubled polarization letter, as returned by
                :meth:`quadratic_response_from_single_link`.

        Returns:
            dict: The integrated (e.g. sky-averaged) quadratic response per
                doubled polarization letter.
        """

    @abstractmethod
    def single_link_noise(
        self,
        frequency_array: ArrayLike,
        arms_matrix_rescaled: ArrayLike,
        x_vector: ArrayLike,
        **noise_parameters,
    ) -> jax.Array:
        """
        Builds the per-link noise covariance (or, for detectors with no
        per-link decomposition, the already-combined noise) before
        projection into a readout combination. `noise_parameters` are
        whatever detector-specific noise parameters this needs (e.g. LISA's
        TM_acceleration_parameters/OMS_parameters); LIGO takes none.

        Args:
            frequency_array (ArrayLike): Frequency values, in Hz, at which
                to evaluate the noise.
            arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled
                by the arm length.
            x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
                frequency.
            **noise_parameters: Detector-specific noise parameters.

        Returns:
            jax.Array: The per-link (or already-combined) noise covariance.
        """

    @abstractmethod
    def project_noise(
        self, combination_matrix: ArrayLike, single_link_noise: jax.Array
    ) -> jax.Array:
        """
        Projects `single_link_noise` (as built by `single_link_noise`) into
        the readout basis defined by `combination_matrix` (as built by
        `combination_matrix`), e.g. via the congruence transform
        combination_matrix @ single_link_noise @ combination_matrix^H.
        Detectors without a per-link decomposition (e.g. LIGO) can simply
        return `single_link_noise` unchanged.

        Args:
            combination_matrix (ArrayLike): Mixing matrix as built by
                :meth:`combination_matrix`.
            single_link_noise (jax.Array): Per-link noise covariance, as
                built by :meth:`single_link_noise`.

        Returns:
            jax.Array: The noise covariance, projected into the readout
                basis.
        """
