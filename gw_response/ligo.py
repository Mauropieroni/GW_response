import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import chex
from functools import partial

from .constants import PhysicalConstants
from .detector import Detector
from .utils import coordinates_numerical, arms_matrix_numerical


@partial(jax.jit, static_argnums=(3,))
def LISA_satellite_positions(
    time_in_years, orbit_radius, eccentricity, which_orbits="analytic"
):
    """
    Calculates the positions of LISA satellites based on the specified orbit
    model.

    Args:
        time_in_years (float): The time in years for which the positions are to
        be calculated.
        orbit_radius (float): The radius of the satellites' orbits.
        eccentricity (float): The eccentricity of the satellites' orbits.
        which_orbits (str, optional): The orbit model to be used. Default is
        'analytic'.

    Returns:
        jnp.ndarray: A numpy array representing the positions of the LISA
        satellites. Each row corresponds to the position of a satellite.

    This function allows for the selection of different orbit models to
    accommodate various analytical and simulation needs.
    """
    return jax.lax.switch(
        LISA_orbits_map[which_orbits],
        LISA_positions_functions,
        jnp.array([1, 2, 3]),
        time_in_years,
        orbit_radius,
        eccentricity,
    ).T


@partial(jax.jit, static_argnums=(3,))
def LISA_arms_matrix(
    time_in_years, orbit_radius, eccentricity, which_orbits="analytic"
):
    """
    Computes the arm matrix for the LISA constellation based on a specified
    orbit model.

    Args:
        time_in_years (float): The time in years at which to calculate the arm
        matrix.
        orbit_radius (float): The radius of the satellite's orbit.
        eccentricity (float): The eccentricity of the satellite's orbit.
        which_orbits (str, optional): The orbit model to be used. Default is
        'analytic'.

    Returns:
        jnp.ndarray: A numpy array representing the arm matrix of the LISA
        constellation based on the chosen orbit model.

    This function provides flexibility in simulating the LISA constellation by
    allowing the selection of different orbit models.
    """
    return jax.lax.switch(
        LISA_orbits_map[which_orbits],
        LISA_arms_functions,
        time_in_years,
        orbit_radius,
        eccentricity,
    )


@chex.dataclass
class LIGO(Detector):
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

    The class provides methods for initializing the configuration and generating
    a frequency vector for analysis.
    """

    name: str = "LIGO Hanford"
    ps: chex.dataclass = PhysicalConstants()  # type: ignore
    fmin: float = 1.0
    fmax: float = 2e3
    armlength: float = 4e3
    res: float = 1e-1
    which_detector: str = "Hanford"

    def __post_init__(self):
        """
        Post-initialization method to compute additional LISA configuration
        parameters.

        This method is invoked automatically after the class is instantiated.
        It computes the observational period of LISA, the orbit eccentricity,
        and the characteristic frequency of LISA based on the provided
        configuration settings.

        The observational period is calculated as three times the duration of a
        year, derived from the PhysicalConstants class. The orbit eccentricity
        is derived from LISA's arm length and astronomical unit. The
        characteristic frequency is calculated based on the light speed and
        LISA's arm length.
        """

        self._f_star = self.ps.light_speed / (2 * jnp.pi * self.armlength)

    def satellite_positions(self, time_in_years):
        """
        Calculates the positions of LISA satellites at a given time in years.

        Args:
            time_in_years (float): The time at which the positions are to be
            calculated, in years.
            which_orbits (str): The method of orbit calculation, defaulting to
            'analytic'.

        Returns:
            The positions of LISA satellites as calculated by the
            LISA_satellites_positions function, with parameters including time
            in years, astronomical unit, orbit eccentricity, and orbit
            calculation method.
        """
        time_in_years = (
            jnp.array([time_in_years])
            if isinstance(time_in_years, float)
            else time_in_years
        )
        return LISA_satellite_positions(
            time_in_years, self.ps.AU, self.ecc, self.which_orbits
        )

    def detector_arms(self, time_in_years):
        """
        Computes the arm matrix of the LISA detector for a given time in years.

        Args:
            time_in_years (float): The time at which the arm matrix is to be
            computed, in years.
            which_orbits (str): The method of orbit calculation, defaulting to
            'analytic'.

        Returns:
            The arm matrix of the LISA detector as calculated by the
            LISA_arms_matrix function, with parameters including time in years,
            astronomical unit, orbit eccentricity, and orbit calculation method.
        """
        time_in_years = (
            jnp.array([time_in_years])
            if isinstance(time_in_years, float)
            else time_in_years
        )

        return LISA_arms_matrix(time_in_years, self.ps.AU, self.ecc, self.which_orbits)
