# Global imports
import jax
import jax.numpy as jnp

import chex
from dataclasses import field

# Local imports
from gw_response.constants import PhysicalConstants
from gw_response.detector import Detector
from gw_response.noise import Noise
from gw_response.response import Response, quadratic_response_integrated
from gw_response.utils import (
    combine_single_link,
    load_numerical_orbits,
    project_noise_matrix,
)
from gw_response.space_based.noise import (
    single_link_OMS_noise_variance,
    single_link_TM_acceleration_noise_variance,
)
from gw_response.space_based.orbits import LISA_arms_matrix, LISA_satellite_positions
from gw_response.space_based.tdi import TDI_map, tdi_matrix

# Update jax configuration to enable 64-bit precision for numerical computations
jax.config.update("jax_enable_x64", True)


@chex.dataclass
class LISA(Detector):
    """
    A data class representing the configuration of the Laser Interferometer
    Space Antenna (LISA).

    This class encapsulates the key parameters and settings used in simulating
    LISA's operation and response in astronomical studies, particularly related
    to gravitational wave detection.

    Attributes:
        ps (chex.dataclass): An instance of the PhysicalConstants class
        providing essential physical constants.
        fmin (float): The minimum frequency sensitivity for LISA, set to
        3.0e-5 Hz. Below this threshold, LISA's minimum frequency sensitivity is
        considered.
        fmax (float): The maximum frequency sensitivity for LISA, set to 5.0e-1
        Hz. Above this threshold, LISA's maximum frequency sensitivity is
        considered.
        arm (float): The length of LISA's arm, set to 2.5e9 meters.
        deg (float): The angular displacement of LISA after Earth, set to 20
        degrees.
        res (float): The expected resolution of LISA, set to 1e-6.
        orbit_approximant (str): The orbit model to use: 'rigid' (a perfectly
        rigid, non-flexing constellation), 'keplerian' (each satellite on its
        own heliocentric Keplerian ellipse, following Martens & Joffre 2021,
        arXiv:2101.03040), or 'numeric' (interpolated from precomputed orbit
        data). Default is 'rigid'.
        orbit_file (str): Path to a file with numerical orbit data (see
        `load_numerical_orbits`). Required when `orbit_approximant` is
        'numeric', ignored otherwise.
        orbit_interpolation_method (str): The interpolation method used to
        evaluate the numerical orbit data (see `load_numerical_orbits`), e.g.
        'linear', 'nearest', 'cubic', 'cubic2', 'cardinal', 'catmull-rom',
        'monotonic', 'monotonic-0', or 'akima'. Default is 'linear'. Only
        used when `orbit_approximant` is 'numeric'.
        keplerian_tilt_parameter (float): Dimensionless inclination parameter
        delta_1 used by the 'keplerian' orbit approximant. Default is 5/8,
        which minimizes arm length flexing. Ignored otherwise.
        keplerian_initial_clocking_angle (float): Initial clocking angle
        sigma_0 used by the 'keplerian' orbit approximant. Default is 0.0.
        Ignored otherwise.
        keplerian_chirality (float): +1.0 for a counter-clockwise, -1.0 for a
        clockwise constellation rotation, used by the 'keplerian' orbit
        approximant. Default is +1.0. Ignored otherwise.

    The class provides methods for initializing the configuration and generating
    a frequency vector for analysis.
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

    def __post_init__(self):
        """
        Post-initialization method to compute additional LISA configuration
        parameters.

        This method is invoked automatically after the class is instantiated.
        It computes the observational period of LISA, the orbit eccentricity,
        and the characteristic frequency of LISA based on the provided
        configuration settings. If `orbit_approximant` is 'numeric', it also
        builds an interpolator over the numerical orbit data loaded from
        `orbit_file`.

        The observational period is calculated as three times the duration of a
        year, derived from the PhysicalConstants class. The orbit eccentricity
        is derived from LISA's arm length and astronomical unit. The
        characteristic frequency is calculated based on the light speed and
        LISA's arm length.
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

    def _vertex_positions(self, time_in_years):
        """
        Calculates the positions of LISA satellites at a given time in years.

        Args:
            time_in_years (float): The time at which the positions are to be
            calculated, in years.
            orbit_approximant (str): The method of orbit calculation,
            defaulting to 'rigid'.

        Returns:
            The positions of LISA satellites as calculated by the
            LISA_satellites_positions function, with parameters including time
            in years, astronomical unit, orbit eccentricity, and orbit
            calculation method.
        """
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

    def _detector_arms(self, time_in_years):
        """
        Computes the arm matrix of the LISA detector for a given time in years.

        Args:
            time_in_years (float): The time at which the arm matrix is to be
            computed, in years.
            orbit_approximant (str): The method of orbit calculation,
            defaulting to 'rigid'.

        Returns:
            The arm matrix of the LISA detector as calculated by the
            LISA_arms_matrix function, with parameters including time in years,
            astronomical unit, orbit eccentricity, and orbit calculation method.
        """
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

    def combination_matrix(self, combination, arms_matrix_rescaled, x_vector):
        return tdi_matrix(TDI_map[combination], arms_matrix_rescaled, x_vector)

    def linear_response_from_single_link(self, single_link, combination_matrix):
        return {
            p: combine_single_link(combination_matrix, sl)
            for p, sl in single_link.items()
        }

    def quadratic_response_from_single_link(self, linear_integrand):
        quadratic_integrand = {}
        for p, linear in linear_integrand.items():
            # The first 2 is sum over polarization the second is for the h.c. sum
            quadratic_integrand[2 * p] = (
                2
                * 2
                / jnp.pi
                / 4
                * jnp.einsum("...ijl,...ikl->...ijkl", linear, jnp.conjugate(linear))
            )
        return quadratic_integrand

    def integrate_quadratic_response(self, quadratic_integrand):
        return {
            p: quadratic_response_integrated(quad)
            for p, quad in quadratic_integrand.items()
        }

    def single_link_noise(
        self, frequency_array, arms_matrix_rescaled, x_vector, **noise_parameters
    ):
        TM_acceleration_parameters = noise_parameters["TM_acceleration_parameters"]
        OMS_parameters = noise_parameters["OMS_parameters"]
        return single_link_TM_acceleration_noise_variance(
            frequency_array, TM_acceleration_parameters, arms_matrix_rescaled, x_vector
        ) + single_link_OMS_noise_variance(
            frequency_array, OMS_parameters, arms_matrix_rescaled, x_vector
        )

    def project_noise(self, combination_matrix, single_link_noise):
        return project_noise_matrix(combination_matrix, single_link_noise)
