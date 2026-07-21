# Global imports
import jax
import jax.numpy as jnp

import chex
from functools import partial

# Local imports
from .constants import PhysicalConstants
from .detector import Detector
from .utils import (
    arms_matrix_from_satellite_positions,
    as_time_array,
    load_numerical_orbits,
)

# Update jax configuration to enable 64-bit precision for numerical computations
jax.config.update("jax_enable_x64", True)


def _as_time_array(time_in_years):
    """
    Wraps a bare float `time_in_years` into a length-1 jnp array so that
    downstream orbit functions can always assume an array-like of times.
    """
    return (
        jnp.array([time_in_years])
        if isinstance(time_in_years, float)
        else time_in_years
    )


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
    return arms_matrix_from_satellite_positions(m1, m2, m3)


@jax.jit
def _solve_kepler_equation(mean_anomaly, orbit_eccentricity):
    """
    Solves the Kepler equation M = E - e*sin(E) for the eccentric anomaly E
    given the mean anomaly M and the orbit eccentricity e, using
    Newton-Raphson iterations.

    Note: Martens & Joffre 2021 (arXiv:2101.03040, Eq. 4) state this equation
    with the opposite sign convention, E + e*sin(E) = M. That convention is
    only consistent with the standard perifocal position formulas used here
    (and with the paper's own Table 1 linear-model elements, and with the arm
    length "breathing" shown in the paper's Fig. 3) if paired with a mirrored
    perifocal frame; using the standard convention below with the standard
    formulas reproduces both, so it is used directly instead.

    Args:
        mean_anomaly (float or jnp.ndarray): The mean anomaly (radians).
        orbit_eccentricity (float): The orbit eccentricity.

    Returns:
        jnp.ndarray: The eccentric anomaly solving the Kepler equation above.
    """

    def newton_step(_, eccentric_anomaly):
        residual = (
            eccentric_anomaly
            - orbit_eccentricity * jnp.sin(eccentric_anomaly)
            - mean_anomaly
        )
        derivative = 1 - orbit_eccentricity * jnp.cos(eccentric_anomaly)
        return eccentric_anomaly - residual / derivative

    return jax.lax.fori_loop(0, 10, newton_step, mean_anomaly)


@jax.jit
def LISA_keplerian_orbital_elements(
    index, armlength_ratio, tilt_parameter, initial_clocking_angle
):
    """
    Computes the per-satellite orbital elements (eccentricity, inclination,
    longitude of the ascending node) of the exact Keplerian cartwheel model
    of Martens & Joffre 2021 (arXiv:2101.03040), section 2.2, Table 2.

    Unlike `LISA_satellite_coordinates_analytical`, which builds a perfectly
    rigid (non-flexing) constellation directly in Cartesian coordinates, this
    model places each satellite on its own heliocentric Keplerian ellipse.
    The resulting triangle is only approximately equilateral and "breathes"
    over the course of a year, which is closer to the actual behaviour of
    LISA.

    Args:
        index (int): The index of the LISA satellite (1, 2, or 3).
        armlength_ratio (float): The ratio alpha = armlength / (2 *
        orbit_radius) between the (nominal) constellation arm length and
        twice the orbit radius.
        tilt_parameter (float): The dimensionless inclination parameter
        delta_1 (delta = alpha * delta_1 in the reference paper). The value
        5/8 minimizes the arm length flexing in the Keplerian model.
        initial_clocking_angle (float): The initial clocking angle sigma_0,
        which sets the orientation of the constellation relative to the
        Earth at t=0.

    Returns:
        tuple: (orbit_eccentricity, inclination, ascending_node), the
        Keplerian orbital elements of the requested satellite. The
        eccentricity and inclination are the same for all three satellites;
        only the ascending node differs.
    """
    tilt = armlength_ratio * tilt_parameter
    phase = jnp.pi / 3 + tilt
    orbit_eccentricity = -1 + jnp.sqrt(
        1
        + 4 / 3 * armlength_ratio**2
        + 4 / jnp.sqrt(3) * armlength_ratio * jnp.cos(phase)
    )
    inclination = jnp.arctan2(
        2 / jnp.sqrt(3) * armlength_ratio * jnp.sin(phase),
        1 + 2 / jnp.sqrt(3) * armlength_ratio * jnp.cos(phase),
    )
    ascending_node = initial_clocking_angle - jnp.pi / 2 + (index - 1) * 2 * jnp.pi / 3
    return orbit_eccentricity, inclination, ascending_node


@jax.jit
def LISA_satellite_coordinates_keplerian(
    index,
    time_in_years,
    orbit_radius,
    eccentricity,
    tilt_parameter=5.0 / 8.0,
    initial_clocking_angle=0.0,
    chirality=1.0,
):
    """
    Computes the three-dimensional heliocentric coordinates of a LISA
    satellite using the exact Keplerian cartwheel model of Martens & Joffre
    2021 (arXiv:2101.03040), section 2.2.

    Each satellite is placed on its own heliocentric Keplerian ellipse (with
    elements from `LISA_keplerian_orbital_elements`), and its position is
    obtained by solving Kepler's equation exactly (via
    `_solve_kepler_equation`) rather than by an expansion in the constellation
    arm length as in `LISA_satellite_coordinates_analytical`. Because of this,
    the resulting triangle is not perfectly rigid: its arm lengths and corner
    angles "breathe" slightly over a year.

    Args:
        index (int): The index of the LISA satellite (1, 2, or 3).
        time_in_years (float or jnp.ndarray): Time in years at which to
        calculate the satellite's position. The satellites are assumed to
        complete one heliocentric orbit per year.
        orbit_radius (float): The (approximate) radius of the satellite's
        orbit, i.e. its semi-major axis.
        eccentricity (float): Here this parametrises the ratio between the
        constellation arm length and the orbit radius (as `LISA().ecc` does
        for the rigid model), from which the armlength_ratio alpha =
        sqrt(3) * eccentricity and the actual orbit eccentricity are derived.
        Kept with this name so this function is interchangeable with
        `LISA_satellite_coordinates_analytical`.
        tilt_parameter (float, optional): The dimensionless inclination
        parameter delta_1. Default is 5/8, which minimizes arm length
        flexing. See `LISA_keplerian_orbital_elements`.
        initial_clocking_angle (float, optional): The initial clocking angle
        sigma_0. Default is 0.0.
        chirality (float, optional): +1.0 for a counter-clockwise, -1.0 for a
        clockwise constellation rotation (as seen from ecliptic north).
        Default is +1.0.

    Returns:
        jnp.ndarray: A three-dimensional array with the x, y, z coordinates
        of the specified LISA satellite at the given time(s).
    """
    armlength_ratio = jnp.sqrt(3) * eccentricity
    orbit_eccentricity, inclination, ascending_node = LISA_keplerian_orbital_elements(
        index, armlength_ratio, tilt_parameter, initial_clocking_angle
    )
    argument_of_perihelion = chirality * jnp.pi / 2
    mean_anomaly = (
        jnp.pi
        - initial_clocking_angle
        - (index - 1) * 2 * jnp.pi / 3
        + 2 * jnp.pi * jnp.array(time_in_years)
    )
    eccentric_anomaly = _solve_kepler_equation(mean_anomaly, orbit_eccentricity)
    denominator = 1 - orbit_eccentricity * jnp.cos(eccentric_anomaly)
    radius = orbit_radius * denominator
    x_perifocal = (
        radius * (jnp.cos(eccentric_anomaly) - orbit_eccentricity) / denominator
    )
    y_perifocal = (
        radius
        * jnp.sqrt(1 - orbit_eccentricity**2)
        * jnp.sin(eccentric_anomaly)
        / denominator
    )

    cos_Omega = jnp.cos(ascending_node)
    sin_Omega = jnp.sin(ascending_node)
    cos_omega = jnp.cos(argument_of_perihelion)
    sin_omega = jnp.sin(argument_of_perihelion)
    cos_i = jnp.cos(inclination)
    sin_i = jnp.sin(inclination)

    x = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * x_perifocal - (
        cos_Omega * sin_omega + sin_Omega * cos_omega * cos_i
    ) * y_perifocal
    y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * x_perifocal + (
        cos_Omega * cos_omega * cos_i - sin_Omega * sin_omega
    ) * y_perifocal
    z = (sin_omega * sin_i) * x_perifocal + (cos_omega * sin_i) * y_perifocal

    return jnp.array([x, y, z])


LISA_satellite_coordinates_keplerian_vm = jax.vmap(
    LISA_satellite_coordinates_keplerian,
    in_axes=(0, None, None, None, None, None, None),
)


@jax.jit
def LISA_arms_matrix_keplerian(
    time_in_years,
    orbit_radius,
    eccentricity,
    tilt_parameter=5.0 / 8.0,
    initial_clocking_angle=0.0,
    chirality=1.0,
):
    """
    Computes the arm matrix for the LISA constellation using the exact
    Keplerian cartwheel model of Martens & Joffre 2021 (arXiv:2101.03040).

    Args:
        time_in_years (float or jnp.ndarray): Time(s) in years at which to
        calculate the arm matrix.
        orbit_radius (float): The radius of the satellite's orbit.
        eccentricity (float): See `LISA_satellite_coordinates_keplerian`.
        tilt_parameter (float, optional): See
        `LISA_satellite_coordinates_keplerian`. Default is 5/8.
        initial_clocking_angle (float, optional): See
        `LISA_satellite_coordinates_keplerian`. Default is 0.0.
        chirality (float, optional): See
        `LISA_satellite_coordinates_keplerian`. Default is +1.0.

    Returns:
        jnp.ndarray: A numpy array representing the arm matrix of the LISA
        constellation. Each row corresponds to the vector difference between
        pairs of LISA satellites.
    """
    m1, m2, m3 = LISA_satellite_coordinates_keplerian_vm(
        jnp.array([1, 2, 3]),
        time_in_years,
        orbit_radius,
        eccentricity,
        tilt_parameter,
        initial_clocking_angle,
        chirality,
    )
    return arms_matrix_from_satellite_positions(m1, m2, m3)


@jax.jit
def LISA_satellite_coordinates_numerical(index, time_in_years, orbit_interpolator):
    """
    Computes the coordinates of a LISA satellite by evaluating a precomputed
    interpolator built from numerical orbit data.

    This function evaluates, at the requested times, the satellite position
    interpolator built by `load_numerical_orbits`. It mirrors the signature
    of `LISA_satellite_coordinates_analytical` so that both orbit models can
    be used interchangeably.

    Args:
        index (int): The index of the LISA satellite (1, 2, or 3).
        time_in_years (float or jnp.ndarray): Time(s) in years at which to
        evaluate the satellite's position. Values outside the range covered
        by the interpolator's time grid are clamped to the boundary values.
        orbit_interpolator (interpax.Interpolator1D): Interpolator built by
        `load_numerical_orbits`.

    Returns:
        jnp.ndarray: The x, y, z coordinates of the requested satellite at the
        requested time(s).
    """
    clipped_time = jnp.clip(
        time_in_years, orbit_interpolator.x[0], orbit_interpolator.x[-1]
    )
    satellite_positions = orbit_interpolator(clipped_time)[..., index - 1, :]
    return jnp.moveaxis(satellite_positions, -1, 0)


LISA_satellite_coordinates_numerical_vm = jax.vmap(
    LISA_satellite_coordinates_numerical, in_axes=(0, None, None)
)


@jax.jit
def LISA_arms_matrix_numerical(time_in_years, orbit_interpolator):
    """
    Computes the arm matrix for the LISA constellation from numerical orbit
    data, by evaluating the satellite position interpolator and taking
    differences.

    Args:
        time_in_years (float or jnp.ndarray): Time(s) in years at which to
        calculate the arm matrix.
        orbit_interpolator (interpax.Interpolator1D): Interpolator built by
        `load_numerical_orbits`.

    Returns:
        jnp.ndarray: A numpy array representing the arm matrix of the LISA
        constellation. Each row of the array corresponds to the vector
        difference between pairs of LISA satellites.
    """
    m1, m2, m3 = LISA_satellite_coordinates_numerical_vm(
        jnp.array([1, 2, 3]), time_in_years, orbit_interpolator
    )
    return arms_matrix_from_satellite_positions(m1, m2, m3)


def _LISA_satellite_coordinates_vm(
    time_in_years,
    orbit_radius,
    eccentricity,
    orbit_approximant,
    orbit_interpolator,
    keplerian_tilt_parameter,
    keplerian_initial_clocking_angle,
    keplerian_chirality,
):
    """
    Dispatches to the vmapped per-satellite coordinate function for the
    requested orbit approximant, returning the stacked (3, 3, ...) array
    indexed as [satellite, coordinate, ...].

    This is the one place the 'rigid' / 'keplerian' / 'numeric' branching
    happens; it is shared by `LISA_satellite_positions` (which transposes the
    result) and `LISA_arms_matrix` (which differences the three satellites).
    """
    indices = jnp.array([1, 2, 3])
    if orbit_approximant == "rigid":
        return LISA_satellite_coordinates_analytical_vm(
            indices, time_in_years, orbit_radius, eccentricity
        )
    elif orbit_approximant == "keplerian":
        return LISA_satellite_coordinates_keplerian_vm(
            indices,
            time_in_years,
            orbit_radius,
            eccentricity,
            keplerian_tilt_parameter,
            keplerian_initial_clocking_angle,
            keplerian_chirality,
        )
    elif orbit_approximant == "numeric":
        return LISA_satellite_coordinates_numerical_vm(
            indices, time_in_years, orbit_interpolator
        )
    raise ValueError(
        f"Unknown orbit approximant '{orbit_approximant}'. Must be 'rigid', "
        "'keplerian', or 'numeric'."
    )


@partial(jax.jit, static_argnums=(3,))
def LISA_satellite_positions(
    time_in_years,
    orbit_radius,
    eccentricity,
    orbit_approximant="rigid",
    orbit_interpolator=None,
    keplerian_tilt_parameter=5.0 / 8.0,
    keplerian_initial_clocking_angle=0.0,
    keplerian_chirality=1.0,
):
    """
    Calculates the positions of LISA satellites based on the specified orbit
    approximant.

    Args:
        time_in_years (float): The time in years for which the positions are to
        be calculated.
        orbit_radius (float): The radius of the satellites' orbits. Not used
        when `orbit_approximant` is 'numeric'.
        eccentricity (float): The eccentricity of the satellites' orbits. Not
        used when `orbit_approximant` is 'numeric'.
        orbit_approximant (str, optional): The orbit model to be used: 'rigid'
        (a perfectly rigid, non-flexing constellation, see
        `LISA_satellite_coordinates_analytical`), 'keplerian' (each satellite
        on its own heliocentric Keplerian ellipse, see
        `LISA_satellite_coordinates_keplerian`), or 'numeric' (interpolated
        from precomputed orbit data). Default is 'rigid'.
        orbit_interpolator (interpax.Interpolator1D, optional): Interpolator
        built by `load_numerical_orbits`. Required when `orbit_approximant` is
        'numeric'.
        keplerian_tilt_parameter (float, optional): See
        `LISA_satellite_coordinates_keplerian`. Only used when
        `orbit_approximant` is 'keplerian'.
        keplerian_initial_clocking_angle (float, optional): See
        `LISA_satellite_coordinates_keplerian`. Only used when
        `orbit_approximant` is 'keplerian'.
        keplerian_chirality (float, optional): See
        `LISA_satellite_coordinates_keplerian`. Only used when
        `orbit_approximant` is 'keplerian'.

    Returns:
        jnp.ndarray: A numpy array representing the positions of the LISA
        satellites. Each row corresponds to the position of a satellite.

    This function allows for the selection of different orbit models to
    accommodate various analytical and simulation needs. Because
    `orbit_approximant` is a static argument, the branch selection happens at
    trace time, so the different orbit models can have different data
    requirements.
    """
    return _LISA_satellite_coordinates_vm(
        time_in_years,
        orbit_radius,
        eccentricity,
        orbit_approximant,
        orbit_interpolator,
        keplerian_tilt_parameter,
        keplerian_initial_clocking_angle,
        keplerian_chirality,
    ).T


@partial(jax.jit, static_argnums=(3,))
def LISA_arms_matrix(
    time_in_years,
    orbit_radius,
    eccentricity,
    orbit_approximant="rigid",
    orbit_interpolator=None,
    keplerian_tilt_parameter=5.0 / 8.0,
    keplerian_initial_clocking_angle=0.0,
    keplerian_chirality=1.0,
):
    """
    Computes the arm matrix for the LISA constellation based on a specified
    orbit approximant.

    Args:
        time_in_years (float): The time in years at which to calculate the arm
        matrix.
        orbit_radius (float): The radius of the satellite's orbit. Not used
        when `orbit_approximant` is 'numeric'.
        eccentricity (float): The eccentricity of the satellite's orbit. Not
        used when `orbit_approximant` is 'numeric'.
        orbit_approximant (str, optional): The orbit model to be used: 'rigid',
        'keplerian', or 'numeric'. Default is 'rigid'. See
        `LISA_satellite_positions` for details.
        orbit_interpolator (interpax.Interpolator1D, optional): Interpolator
        built by `load_numerical_orbits`. Required when `orbit_approximant` is
        'numeric'.
        keplerian_tilt_parameter (float, optional): See
        `LISA_satellite_coordinates_keplerian`. Only used when
        `orbit_approximant` is 'keplerian'.
        keplerian_initial_clocking_angle (float, optional): See
        `LISA_satellite_coordinates_keplerian`. Only used when
        `orbit_approximant` is 'keplerian'.
        keplerian_chirality (float, optional): See
        `LISA_satellite_coordinates_keplerian`. Only used when
        `orbit_approximant` is 'keplerian'.

    Returns:
        jnp.ndarray: A numpy array representing the arm matrix of the LISA
        constellation based on the chosen orbit model.

    This function provides flexibility in simulating the LISA constellation by
    allowing the selection of different orbit models.
    """
    m1, m2, m3 = _LISA_satellite_coordinates_vm(
        time_in_years,
        orbit_radius,
        eccentricity,
        orbit_approximant,
        orbit_interpolator,
        keplerian_tilt_parameter,
        keplerian_initial_clocking_angle,
        keplerian_chirality,
    )
    return arms_matrix_from_satellite_positions(m1, m2, m3)


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
    ps: chex.dataclass = PhysicalConstants()  # type: ignore
    fmin: float = 3.0e-5
    fmax: float = 5.0e-1
    armlength: float = 2.5e9
    deg: float = 20
    res: float = 1e-6
    orbit_approximant: str = "rigid"
    orbit_file: str = None
    orbit_interpolation_method: str = "linear"
    keplerian_tilt_parameter: float = 5.0 / 8.0
    keplerian_initial_clocking_angle: float = 0.0
    keplerian_chirality: float = 1.0

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
            orbit_approximant (str): The method of orbit calculation,
            defaulting to 'rigid'.

        Returns:
            The positions of LISA satellites as calculated by the
            LISA_satellites_positions function, with parameters including time
            in years, astronomical unit, orbit eccentricity, and orbit
            calculation method.
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

    def detector_arms(self, time_in_years):
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
