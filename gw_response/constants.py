import jax
import jax.numpy as jnp
import chex
from dataclasses import field

jax.config.update("jax_enable_x64", True)


@chex.dataclass(frozen=True)
class PhysicalConstants:
    """
    A data class for storing physical constants used in unit conversions within
    astronomical computations.

    This class provides a convenient way to access commonly used physical
    constants, ensuring consistency and clarity across different parts of the
    code. The constants are set as class attributes with predefined values.

    Attributes:
        light_speed (float): The speed of light in meters per second (m/s).
        Default is 299792458.0.
        hour (float): The duration of an hour in seconds. Default is 3600
        seconds.
        day (float): The duration of a day in seconds, calculated as 24 hours.
        Default is 86400 seconds.
        yr (float): The duration of a year in seconds, accounting for leap
        years. Default is 365.25 days.
        Hubble_over_h (float): Hubble constant divided by the dimensionless
        Hubble parameter 'h'. Units are 1/second (1/s).

    The class is frozen using chex.dataclass, meaning its instances are
    immutable and cannot be modified after creation.
    """

    light_speed: float = 299792458.0  # speed of light in m/s
    hour: float = 3600.0
    day: float = 24 * hour
    yr: float = 365.25 * day  # year in s
    Hubble_over_h: float = 3.24e-18  # H0 divided by h in units of 1/s
    AU: float = 1.495978707e11  # Astronomical unit in meters
    cmb_dipole: jnp.array = field(
        default_factory=lambda: jnp.array([-0.972, 0.137, -0.191])
    )
    # Direction on the CMB dipole


@chex.dataclass
class BasisTransformations:
    """
    A data class for managing basis transformations in astronomical
    computations.

    This class provides a predefined transformation matrix for converting
    coordinates from the XYZ coordinate system to the AET (Arm, Ecliptic,
    Transverse) coordinate system, which is commonly used in the context of
    LISA (Laser Interferometer Space Antenna) and similar astronomical studies.

    Attributes:
        XYZ_to_AET (jnp.array): A numpy array representing the transformation
        matrix from the XYZ coordinate system to the AET coordinate system. The
        transformation matrix is defined as:
            [
                [-1 / sqrt(2), 0, 1 / sqrt(2)],
                [1 / sqrt(6), -2 / sqrt(6), 1 / sqrt(6)],
                [1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)]
            ]

    This class facilitates the conversion of coordinates between different
    systems, which is crucial for accurate representation and analysis in
    astronomical models.
    """

    XYZ_to_AET: jnp.array = field(
        default_factory=lambda: jnp.array(
            [
                [-1 / jnp.sqrt(2), 0, 1 / jnp.sqrt(2)],
                [1 / jnp.sqrt(6), -2 / jnp.sqrt(6), 1 / jnp.sqrt(6)],
                [1 / jnp.sqrt(3), 1 / jnp.sqrt(3), 1 / jnp.sqrt(3)],
            ]
        )
    )
