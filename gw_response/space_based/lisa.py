# Global imports
import jax
import jax.numpy as jnp

import chex
from dataclasses import field

from jax.typing import ArrayLike

# Local imports
from gw_response.constants import PhysicalConstants
from gw_response.detector import Detector
from gw_response.noise import Noise
from gw_response.response import Response
from gw_response.response_utils import (
    quadratic_response_integrated,
    quadratic_from_linear,
)
from gw_response.utils import (
    as_time_array,
    combine_single_link,
    load_numerical_orbits,
    project_noise_matrix,
)
from gw_response.space_based.noise import (
    single_link_OMS_noise_variance,
    single_link_TM_acceleration_noise_variance,
)
from gw_response.space_based.orbits import LISA_arms_matrix, LISA_satellite_positions
from gw_response.space_based.single_link_geometry import _SINGLE_LINK_ARM_LABELS
from gw_response.space_based.tdi import TDI_map, tdi_matrix

# Update jax configuration to enable 64-bit precision for numerical computations
jax.config.update("jax_enable_x64", True)


@chex.dataclass
class LISA(Detector):
    """
    A data class representing the configuration of the Laser Interferometer Space
    Antenna (LISA).

    This class encapsulates the key parameters and settings used in simulating LISA's
    operation and response in astronomical studies, particularly related to
    gravitational wave detection.

    Attributes:
        ps (PhysicalConstants): Physical constants used in the response/noise
            computations.
        fmin (float): The minimum frequency sensitivity for LISA, set to 3.0e-5 Hz.
        fmax (float): The maximum frequency sensitivity for LISA, set to 5.0e-1 Hz.
        armlength (float): The length of LISA's arm, set to 2.5e9 meters.
        deg (float): The angular displacement of LISA behind the Earth, set to 20
            degrees.
        res (float): The expected resolution of LISA, set to 1e-6.
        orbit_approximant (str): The orbit model to use: 'rigid' (a perfectly rigid,
            non-flexing constellation), 'keplerian' (each satellite on its own
            heliocentric Keplerian ellipse, following Martens & Joffre 2021,
            arXiv:2101.03040), or 'numeric' (interpolated from precomputed orbit data).
            Default is 'rigid'.
        orbit_file (str or None): Path to a file with numerical orbit data (see
            `load_numerical_orbits`). Required when `orbit_approximant` is 'numeric',
            ignored otherwise.
        orbit_interpolation_method (str): The interpolation method used to evaluate the
            numerical orbit data (see `load_numerical_orbits`), e.g. 'linear',
            'nearest', 'cubic', 'cubic2', 'cardinal', 'catmull-rom', 'monotonic',
            'monotonic-0', or 'akima'. Default is 'linear'. Only used when
            `orbit_approximant` is 'numeric'.
        keplerian_tilt_parameter (float): Dimensionless inclination parameter delta_1
            used by the 'keplerian' orbit approximant. Default is 5/8, which minimizes
            arm length flexing. Ignored otherwise.
        keplerian_initial_clocking_angle (float): Initial clocking angle sigma_0 used by
            the 'keplerian' orbit approximant. Default is 0.0. Ignored otherwise.
        keplerian_chirality (float): +1.0 for a counter-clockwise, -1.0 for a clockwise
            constellation rotation, used by the 'keplerian' orbit approximant. Default
            is +1.0. Ignored otherwise.
        default_combination (str): The default TDI combination, "XYZ".
        response (Response): Response object used to compute LISA's response to
            gravitational waves.
        noise (Noise): Noise object used to compute LISA's noise budget.
    """

    name: str = "LISA"
    ps: PhysicalConstants = PhysicalConstants()
    fmin: float = 3.0e-5
    fmax: float = 5.0e-1
    armlength: float = 2.5e9
    deg: float = 20
    res: float = 1e-6
    orbit_approximant: str = "rigid"
    orbit_file: str | None = None
    orbit_interpolation_method: str = "linear"
    keplerian_tilt_parameter: float = 5.0 / 8.0
    keplerian_initial_clocking_angle: float = 0.0
    keplerian_chirality: float = 1.0
    default_combination: str = "XYZ"
    response: Response = field(init=False, default_factory=Response)
    noise: Noise = field(init=False, default_factory=Noise)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __post_init__(self) -> None:
        """
        Post-initialization method to compute additional LISA configuration parameters.

        This method is invoked automatically after the class is instantiated. It
        computes the observational period of LISA, the orbit eccentricity, and the
        characteristic frequency of LISA based on the provided configuration settings.
        If `orbit_approximant` is 'numeric', it also builds an interpolator over the
        numerical orbit data loaded from `orbit_file`.

        The observational period is calculated as three times the duration of a year,
        derived from the PhysicalConstants class. The orbit eccentricity is derived from
        LISA's arm length and astronomical unit. The characteristic frequency is
        calculated based on the light speed and LISA's arm length.
        """
        self.obs = 3 * self.ps.yr
        self.ecc = self.armlength / (2 * self.ps.AU * jnp.sqrt(3))
        self._f_star = self.ps.light_speed / (2 * jnp.pi * self.armlength)

        self.orbit_interpolator = None
        if self.orbit_approximant == "numeric":
            if self.orbit_file is None:
                raise ValueError(
                    "orbit_approximant='numeric' requires an orbit_file "
                    "pointing to the numerical orbit data."
                )
            self.orbit_interpolator = load_numerical_orbits(
                self.orbit_file, self.orbit_interpolation_method
            )

    def vertex_positions(self, time_in_years: ArrayLike) -> jax.Array:
        """
        Calculates the positions of LISA satellites at a given time in years.

        Args:
            time_in_years (ArrayLike): The time at which the positions are to be
                calculated, in years. Normalized via
                :func:`gw_response.utils.as_time_array` (accepts a bare scalar or an
                array).

        Returns:
            jax.Array: The positions of LISA satellites as calculated by the
                :func:`gw_response.space_based.orbits.LISA_satellite_positions`
                function, using `self.orbit_approximant` (and the related
                `self.orbit_interpolator`/`self.keplerian_*` attributes as required by
                that model).
        """
        time_in_years = as_time_array(time_in_years)
        return LISA_satellite_positions(
            time_in_years,
            self.ps.AU,
            self.ecc,
            self.orbit_approximant,
            self.orbit_interpolator,
            self.keplerian_tilt_parameter,
            self.keplerian_initial_clocking_angle,
            self.keplerian_chirality,
        )

    def detector_arms(self, time_in_years: ArrayLike) -> jax.Array:
        """
        Computes the arm matrix of the LISA detector for a given time in years.

        Args:
            time_in_years (ArrayLike): The time at which the arm matrix is to be
                computed, in years. Normalized via
                :func:`gw_response.utils.as_time_array` (accepts a bare scalar or an
                array).

        Returns:
            jax.Array: The arm matrix of the LISA detector as calculated by the
                :func:`gw_response.space_based.orbits.LISA_arms_matrix` function, using
                `self.orbit_approximant` (and the related
                `self.orbit_interpolator`/`self.keplerian_*` attributes as required by
                that model).
        """
        time_in_years = as_time_array(time_in_years)
        return LISA_arms_matrix(
            time_in_years,
            self.ps.AU,
            self.ecc,
            self.orbit_approximant,
            self.orbit_interpolator,
            self.keplerian_tilt_parameter,
            self.keplerian_initial_clocking_angle,
            self.keplerian_chirality,
        )

    @property
    def arm_vertex_pairs(self) -> tuple[tuple[int, int], ...]:
        """
        Derived from :data:`_SINGLE_LINK_ARM_LABELS` (12, 23, 31, 21, 32, 13): for each
        two-digit label "ij", satellites are 1-indexed there, so the 0-indexed
        (receiver, emitter) pair is (i - 1, j - 1).

        Returns:
            tuple[tuple[int, int], ...]: The 6 (receiver, emitter) satellite index
                pairs, in :data:`_SINGLE_LINK_ARM_LABELS` order.
        """
        return tuple(
            (label // 10 - 1, label % 10 - 1) for label in _SINGLE_LINK_ARM_LABELS
        )

    def combination_matrix(
        self, combination: str, arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike
    ) -> jax.Array:
        """
        Builds the TDI projection matrix for the requested combination. See
        :func:`gw_response.space_based.tdi.tdi_matrix`.

        Args:
            combination (str): Name of the TDI combination (a key of
                :data:`gw_response.space_based.tdi.TDI_map`).
            arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
                length.
            x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

        Returns:
            jax.Array: The TDI projection matrix for the requested combination.
        """
        return tdi_matrix(TDI_map[combination], arms_matrix_rescaled, x_vector)

    def linear_response_from_single_link(
        self, single_link: dict[str, jax.Array], combination_matrix: ArrayLike
    ) -> dict[str, jax.Array]:
        """
        Projects the single-link response onto the TDI combination. See
        :func:`gw_response.utils.combine_single_link`.

        Args:
            single_link (dict): Single-link response per polarization.
            combination_matrix (ArrayLike): TDI projection matrix as built by
                :meth:`combination_matrix`.

        Returns:
            dict: The linear TDI response per polarization.
        """
        return {
            p: combine_single_link(combination_matrix, sl)
            for p, sl in single_link.items()
        }

    def quadratic_response_from_single_link(
        self, linear_integrand: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """
        Computes the sky-resolved quadratic TDI response as the cross-spectrum of the
        linear response with its own conjugate, summed over polarizations and Hermitian
        conjugation.

        Args:
            linear_integrand (dict): Linear TDI response per polarization, as returned
                by :meth:`linear_response_from_single_link`.

        Returns:
            dict: The sky-resolved quadratic TDI response per doubled polarization
                letter (e.g. "LL", "RR").
        """
        return {
            2 * p: quadratic_from_linear(linear)
            for p, linear in linear_integrand.items()
        }

    def integrate_quadratic_response(
        self, quadratic_integrand: dict[str, jax.Array]
    ) -> dict[str, jax.Array]:
        """
        Averages the sky-resolved quadratic TDI response over the sky. See
        :func:`gw_response.response_utils.quadratic_response_integrated`.

        Args:
            quadratic_integrand (dict): Sky-resolved quadratic TDI response per doubled
                polarization letter, as returned by
                :meth:`quadratic_response_from_single_link`.

        Returns:
            dict: The sky-averaged quadratic TDI response per doubled polarization
                letter.
        """
        return {
            p: quadratic_response_integrated(quad)
            for p, quad in quadratic_integrand.items()
        }

    def single_link_noise(
        self,
        frequency_array: ArrayLike,
        arms_matrix_rescaled: ArrayLike,
        x_vector: ArrayLike,
        **noise_parameters,
    ) -> jax.Array:
        """
        Computes the total (test-mass + OMS) single-link noise covariance matrix. See
        :func:`gw_response.space_based.noise.single_link_TM_acceleration_noise_variance`
        and :func:`gw_response.space_based.noise.single_link_OMS_noise_variance`.

        Args:
            frequency_array (ArrayLike): Frequency values, in Hz, at which to evaluate
                the noise.
            arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
                length.
            x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.
                **noise_parameters: Must contain ``TM_acceleration_parameters`` and
                ``OMS_parameters`` (each a per-arm noise amplitude vector of length 6).

        Returns:
            jax.Array: The single-link test-mass + OMS noise covariance matrix.
        """
        TM_acceleration_parameters = noise_parameters["TM_acceleration_parameters"]
        OMS_parameters = noise_parameters["OMS_parameters"]
        return single_link_TM_acceleration_noise_variance(
            frequency_array, TM_acceleration_parameters, arms_matrix_rescaled, x_vector
        ) + single_link_OMS_noise_variance(
            frequency_array, OMS_parameters, arms_matrix_rescaled, x_vector
        )

    def project_noise(
        self, combination_matrix: ArrayLike, single_link_noise: jax.Array
    ) -> jax.Array:
        """
        Projects the single-link noise covariance into the TDI basis. See
        :func:`gw_response.utils.project_noise_matrix`.

        Args:
            combination_matrix (ArrayLike): TDI projection matrix as built by
                :meth:`combination_matrix`.
            single_link_noise (jax.Array): Single-link noise covariance as built by
                :meth:`single_link_noise`.

        Returns:
            jax.Array: The TDI noise covariance matrix.
        """
        return project_noise_matrix(combination_matrix, single_link_noise)
