import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import chex
from functools import partial

from .constants import PhysicalConstants
from .detector import Detector
from .utils import coordinates_numerical, arms_matrix_numerical


@jax.jit
def LISA_alpha_par(time_in_years):
    """
    Computes the alpha parameter for the LISA satellite configuration based on
    time.

    This function calculates the alpha parameter, which is crucial in
    determining the orientation of the LISA constellation over time. It is used
    in the calculation of LISA satellite positions and their relative movements.

    Args:
        time_in_years (float or jnp.ndarray): Time in years at which to
        calculate the alpha parameter. Can be a single value or an array of
        values.

    Returns:
        jnp.ndarray: The computed alpha parameter, given as 2π times the input
        time(s) in years. If an array of times is provided, an array
        of corresponding alpha values is returned.

    Note:
        The input time is expected to be provided in years but internally
        converted to seconds for computation.
    """
    return 2 * jnp.pi * jnp.array(time_in_years)


@jax.jit
def LISA_beta_par(index):
    """
    Computes the beta parameter for the LISA satellite configuration based on
    the satellite index.

    This function calculates the beta parameter, which is a key component in
    determining the position and orientation of each LISA satellite. It is
    integral to the calculation of LISA satellite positions and their relative
    movements within the constellation.

    Args:
        index (int): The index of the LISA satellite (1, 2, or 3) for which the
        beta parameter is to be computed.

    Returns:
        float: The computed beta parameter for the given satellite index,
        calculated as (index - 1) times 2π/3.

    Note:
        There is a known typo in Equation 17 of the reference manual regarding
        this calculation. This function corrects that typo and provides the
        accurate computation for the beta parameter.
    """
    return (index - 1) * 2 * jnp.pi / 3


@jax.jit
def LISA_satellite_x_coordinate_analytical(
    index, time_in_years, orbit_radius, eccentricity
):
    """
    Computes the x-coordinate of a LISA satellite in an analytical orbit model.

    This function calculates the x-coordinate of a specific LISA satellite based
    on its index, time, orbit radius, and eccentricity. It uses an analytical
    model to determine the satellite's position in space.

    Args:
        index (int): The index of the LISA satellite (1, 2, or 3).
        time_in_years (float): The time in years at which to calculate the
        satellite's position.
        orbit_radius (float): The radius of the satellite's orbit.
        eccentricity (float): The eccentricity of the satellite's orbit.

    Returns:
        float: The x-coordinate of the specified LISA satellite at the given
        time.

    The calculation incorporates both the alpha and beta parameters (derived
    from `LISA_alpha_par` and `LISA_beta_par` functions), which represent the
    satellite's orientation and position in its orbit respectively. This
    function is crucial for simulating and understanding the spatial
    configuration of the LISA constellation over time.
    """
    alpha = LISA_alpha_par(time_in_years)
    beta = LISA_beta_par(index)
    return orbit_radius * (
        jnp.cos(alpha)
        + eccentricity
        * (
            jnp.sin(alpha) * jnp.cos(alpha) * jnp.sin(beta)
            - (1 + jnp.sin(alpha) ** 2) * jnp.cos(beta)
        )
    )


@jax.jit
def LISA_satellite_y_coordinate_analytical(
    index, time_in_years, orbit_radius, eccentricity
):
    """
    Computes the y-coordinate of a LISA satellite in an analytical orbit model.

    This function calculates the y-coordinate of a specific LISA satellite based
    on its index, time, orbit radius, and eccentricity. It uses an analytical
    model to determine the satellite's position in space.

    Args:
        index (int): The index of the LISA satellite (1, 2, or 3).
        time_in_years (float): The time in years at which to calculate the
        satellite's position.
        orbit_radius (float): The radius of the satellite's orbit.
        eccentricity (float): The eccentricity of the satellite's orbit.

    Returns:
        float: The y-coordinate of the specified LISA satellite at the given
        time.

    Note:
        There is a known typo in Equation 15 of the reference manual regarding
        this calculation. This function corrects that typo (a sin is replaced
        with a cos) and provides the accurate computation for the y-coordinate.

    This function is crucial for simulating and understanding the spatial
    configuration of the LISA constellation over time, particularly in the
    y-axis.
    """
    alpha = LISA_alpha_par(time_in_years)
    beta = LISA_beta_par(index)
    return orbit_radius * (
        jnp.sin(alpha)
        + eccentricity
        * (
            jnp.sin(alpha) * jnp.cos(alpha) * jnp.cos(beta)
            - (1 + jnp.cos(alpha) ** 2) * jnp.sin(beta)
        )
    )


@jax.jit
def LISA_satellite_z_coordinate_analytical(
    index, time_in_years, orbit_radius, eccentricity
):
    """
    Computes the z-coordinate of a LISA satellite in an analytical orbit model.

    This function calculates the z-coordinate of a specific LISA satellite based
    on its index, time, orbit radius, and eccentricity. It uses an analytical
    model to determine the satellite's position in space.

    Args:
        index (int): The index of the LISA satellite (1, 2, or 3).
        time_in_years (float): The time in years at which to calculate the
        satellite's position.
        orbit_radius (float): The radius of the satellite's orbit.
        eccentricity (float): The eccentricity of the satellite's orbit.

    Returns:
        float: The z-coordinate of the specified LISA satellite at the given
        time.

    The function computes the z-coordinate by applying the alpha and beta
    parameters (derived from `LISA_alpha_par` and `LISA_beta_par` functions),
    which are crucial for modeling the satellite's position in its orbit. This
    computation is essential for simulating and understanding the spatial
    configuration of the LISA constellation over time, especially in the z-axis.
    """
    alpha = LISA_alpha_par(time_in_years)
    beta = LISA_beta_par(index)
    return -jnp.sqrt(3) * orbit_radius * eccentricity * jnp.cos(alpha - beta)


@jax.jit
def LISA_satellite_coordinates_analytical(
    index, time_in_years, orbit_radius, eccentricity
):
    """
    Computes the three-dimensional coordinates of a LISA satellite in an
    analytical orbit model.

    This function integrates the analytical calculations for the x, y, and z
    coordinates of a specific LISA satellite, based on its index, time, orbit
    radius, and eccentricity. It employs analytical models to determine the
    satellite's precise position in space.

    Args:
        index (int): The index of the LISA satellite (1, 2, or 3).
        time_in_years (float): The time in years at which to calculate the
        satellite's position.
        orbit_radius (float): The radius of the satellite's orbit.
        eccentricity (float): The eccentricity of the satellite's orbit.

    Returns:
        jnp.ndarray: A three-dimensional numpy array representing the
        coordinates of the specified LISA satellite at the given time. The array
        contains the x, y, and z coordinates, computed by the
        `LISA_satellite_x_coordinate_analytical`,
        `LISA_satellite_y_coordinate_analytical`, and
        `LISA_satellite_z_coordinate_analytical` functions, respectively.

    This function is essential for simulating the spatial configuration of the
    LISA constellation over time, providing a comprehensive view of each
    satellite's position in three-dimensional space.
    """
    return jnp.array(
        [
            LISA_satellite_x_coordinate_analytical(
                index, time_in_years, orbit_radius, eccentricity
            ),
            LISA_satellite_y_coordinate_analytical(
                index, time_in_years, orbit_radius, eccentricity
            ),
            LISA_satellite_z_coordinate_analytical(
                index, time_in_years, orbit_radius, eccentricity
            ),
        ]
    )


LISA_satellite_coordinates_analytical_vm = jax.vmap(
    LISA_satellite_coordinates_analytical, in_axes=(0, None, None, None)
)


@jax.jit
def LISA_arms_matrix_analytical(time_in_years, orbit_radius, eccentricity):
    """
    Computes the arm matrix for the LISA constellation using an analytical
    model.

    Args:
        time_in_years (float): The time in years at which to calculate the arm
        matrix.
        orbit_radius (float): The radius of the satellite's orbit.
        eccentricity (float): The eccentricity of the satellite's orbit.

    Returns:
        jnp.ndarray: A numpy array representing the arm matrix of the LISA
        constellation. Each row of the array corresponds to the vector
        difference between pairs of LISA satellites.

    This function is vital for analyzing the spatial arrangement and relative
    positions of the LISA satellites in their orbits.
    """
    m1, m2, m3 = LISA_satellite_coordinates_analytical_vm(
        jnp.array([1, 2, 3]),
        time_in_years,
        orbit_radius,
        eccentricity,
    )

    fa1 = 1  # + 0.01 * jnp.sin(2 * jnp.pi * time_in_years)
    fa2 = 1  # + 0.01 * jnp.sin(2 * jnp.pi * time_in_years + 2 * jnp.pi / 3)
    fa3 = 1  # + 0.01 * jnp.sin(2 * jnp.pi * time_in_years + 4 * jnp.pi / 3)

    return jnp.array(
        [
            fa1 * (m2 - m1),
            fa2 * (m3 - m2),
            fa3 * (m1 - m3),
            fa1 * (m1 - m2),
            fa2 * (m2 - m3),
            fa3 * (m3 - m1),
        ]
    ).T


LISA_orbits_map = {
    "analytic": 0,
    # "numeric": 1,
}

LISA_positions_functions = [
    LISA_satellite_coordinates_analytical_vm,
    # LISA_satellite_coordinates_numerical_vm,
]
LISA_arms_functions = [
    LISA_arms_matrix_analytical,  # arms_matrix_numerical
]


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

    The class provides methods for initializing the configuration and generating
    a frequency vector for analysis.
    """

    name: str = "LISA"
    ps: chex.dataclass = PhysicalConstants()  # type: ignore
    fmin: float = 3.0e-5
    fmax: float = 5.0e-1
    armlength: float = 2.5e9
    deg: float = 20
    res: float = 1e-6
    which_orbits: str = "analytic"

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
        self.obs = 3 * self.ps.yr
        self.ecc = self.armlength / (2 * self.ps.AU * jnp.sqrt(3))
        self._f_star = self.ps.light_speed / (2 * jnp.pi * self.armlength)

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
