# Global imports
import chex
import h5py
import interpax
import jax
import jax.numpy as jnp
import jax_healpy as hp
import numpy as np
from jax.typing import ArrayLike

# Local imports
from gw_response.constants import PhysicalConstants

# Update jax configuration to enable 64-bit precision for numerical computations
jax.config.update("jax_enable_x64", True)


def as_time_array(time_in_years: ArrayLike) -> jax.Array:
    """
    Wraps a bare scalar `time_in_years` (a Python int/float, or a 0-d array)
    into a length-1 jnp array so that downstream functions can always assume
    an array-like of times.

    Args:
        time_in_years (ArrayLike): A scalar or array of time(s), in years.

    Returns:
        jax.Array: `time_in_years` as an array with at least 1 dimension.
    """
    return (
        jnp.array([time_in_years])
        if jnp.ndim(time_in_years) == 0
        else jnp.asarray(time_in_years)
    )


@chex.dataclass
class Pixel:
    """
    A data class for handling the pixelization of the sky in astronomical
    observations.

    Attributes:
        NSIDE (int): The number of sides of each pixel in the HEALPix
            pixelization. Default is 8.
        NPIX (int): The total number of pixels, computed based on NSIDE.
        angular_map (jax.Array): An array representing the angular position
            of each pixel.
        theta_pixel (jax.Array): An array of theta (colatitude) values for
            each pixel.
        phi_pixel (jax.Array): An array of phi (longitude) values for each
            pixel.

    The class automatically computes the pixelization upon instantiation or when
    the NSIDE value is changed.
    """

    NSIDE: int = 8
    NPIX: int | None = None
    angular_map: jax.Array | None = None
    theta_pixel: jax.Array | None = None
    phi_pixel: jax.Array | None = None

    def __post_init__(self) -> None:
        """
        Post-initialization method to compute the pixelization of the sky.

        This method is automatically called after the class initialization. It
        computes the total number of pixels (NPIX), the angular map, and the
        theta and phi values for each pixel based on the NSIDE value.
        """
        (
            self.NPIX,
            self.angular_map,
            self.theta_pixel,
            self.phi_pixel,
        ) = self.compute_pixelisation()

    def compute_pixelisation(self) -> tuple[int, jax.Array, jax.Array, jax.Array]:
        """
        Computes the pixelization parameters of the sky.

        Returns:
            tuple: A tuple containing:
                - NPIX (int): The total number of pixels.
                - angular_map (jax.Array): The angular map array.
                - theta_pixel (jax.Array): The theta values for each pixel.
                - phi_pixel (jax.Array): The phi values for each pixel.
        """
        NPIX = hp.nside2npix(self.NSIDE)
        theta_pixel, phi_pixel = hp.pix2ang(self.NSIDE, jnp.arange(NPIX))
        theta_pixel = jnp.asarray(theta_pixel)
        phi_pixel = jnp.asarray(phi_pixel)
        angular_map = jnp.asarray(jnp.stack([theta_pixel, phi_pixel], axis=-1))
        return NPIX, angular_map, theta_pixel, phi_pixel

    def change_NSIDE(self, NSIDE: int) -> None:
        """
        Changes the NSIDE attribute and recomputes the pixelization parameters.

        Args:
            NSIDE (int): The new NSIDE value for pixelization.

        This method updates the NSIDE attribute and recomputes the NPIX,
        angular_map, theta_pixel, and phi_pixel attributes.
        """
        self.NSIDE = NSIDE
        (
            self.NPIX,
            self.angular_map,
            self.theta_pixel,
            self.phi_pixel,
        ) = self.compute_pixelisation()


@jax.jit
def arm_length_exponential(
    arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike
) -> jax.Array:
    """
    Compute the exponential factor for the Time Delay Interferometry (TDI).

    The function calculates the exponential factors used in TDI computations for
    a laser interferometer space antenna (LISA) setup. It is part of the process
    of accounting for the time delay in the arms of the interferometer due to
    the finite speed of light.

    Args:
        arms_matrix_rescaled (ArrayLike): Rescaled arm matrices of the
            interferometer, with shape (configurations, vectorial_index (3),
            arms (6)). Ordering: [12, 23, 31, 21, 32, 13].
        x_vector (ArrayLike): Vector of the x values over frequency, specific
            to the LISA interferometer's configuration and operational
            characteristics.

    Returns:
        jax.Array: A complex-valued 3D array representing the exponential
            factors, with shape [configurations, x_vector, arms]. These
            factors are used in further calculations of the TDI response.
    """
    arm_lengths = jnp.sqrt(
        jnp.einsum("...ij,...ij->...j", arms_matrix_rescaled, arms_matrix_rescaled)
    )
    xij = jnp.einsum("i,...j->...ij", -1j * x_vector, arm_lengths)
    return jnp.exp(xij)


@jax.jit
def combine_single_link(
    combination_matrix: ArrayLike, single_link: ArrayLike
) -> jax.Array:
    """
    Applies a detector's channel-combination matrix to per-link responses.

    Args:
        combination_matrix (ArrayLike): Mixing matrix turning per-link
            responses into readout channels, with shape (..., x_vector,
            channels, arms).
        single_link (ArrayLike): Per-link response, with shape (...,
            x_vector, arms, pixels).

    Returns:
        jax.Array: The per-channel response, with shape (..., x_vector,
            channels, pixels).
    """
    return jnp.einsum("...ijk,...ikl->...ijl", combination_matrix, single_link)


@jax.jit
def project_noise_matrix(
    combination_matrix: ArrayLike, single_link_noise: ArrayLike
) -> jax.Array:
    """
    Congruence-transforms a per-link noise covariance into a detector's
    channel-combination basis: ``combination_matrix @ single_link_noise @
    combination_matrix^H``.

    Args:
        combination_matrix (ArrayLike): Mixing matrix turning per-link
            responses into readout channels, with shape (..., x_vector,
            channels, arms).
        single_link_noise (ArrayLike): Per-link noise covariance, with shape
            (..., x_vector, arms, arms).

    Returns:
        jax.Array: The projected noise covariance, with shape (...,
            x_vector, channels, channels).
    """
    first_contraction = jnp.einsum(
        "...ijk,...ikl->...ijl", combination_matrix, single_link_noise
    )
    return jnp.einsum(
        "...ijk,...ilk->...ijl", jnp.conjugate(combination_matrix), first_contraction
    )


@jax.jit
def shift_to_center(
    first: jax.Array, second: jax.Array, third: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Adjusts the positions of three points (or vectors) so that their barycenter
    is at the origin.

    This function is used in the context of astronomical computations where it's
    necessary to centralize a system of points, such as adjusting the positions
    of satellites or celestial bodies.

    Args:
        first (jnp.ndarray): The coordinates of the first point or vector.
        second (jnp.ndarray): The coordinates of the second point or vector.
        third (jnp.ndarray): The coordinates of the third point or vector.

    Returns:
        tuple: A tuple of three jnp.ndarrays representing the adjusted
            coordinates of the first, second, and third points (or vectors),
            respectively.

    Each output array has the same shape as the input arrays, and their
    collective barycenter is shifted to the origin.
    """
    center = (first + second + third) / 3
    first_mass = first - center
    second_mass = second - center
    third_mass = third - center

    return first_mass, second_mass, third_mass


@jax.jit
def arms_matrix_from_vertex_positions(
    m1: jax.Array, m2: jax.Array, m3: jax.Array
) -> jax.Array:
    """
    Builds a constellation arm matrix (the vector difference between each
    ordered pair of vertices) from three vertices' Cartesian positions.

    This differencing is identical regardless of which orbit model produced
    the positions, so it is shared by the analytical, Keplerian, and
    numerical orbit models in `gw_response.space_based.orbits`.

    Args:
        m1 (jax.Array): The Cartesian position of vertex 1.
        m2 (jax.Array): The Cartesian position of vertex 2.
        m3 (jax.Array): The Cartesian position of vertex 3.

    Returns:
        jax.Array: A numpy array representing the arm matrix of the
            constellation. Each row of the array corresponds to the vector
            difference between pairs of vertices, ordered [12, 23, 31, 21,
            32, 13].
    """
    return jnp.array(
        [
            m2 - m1,
            m3 - m2,
            m1 - m3,
            m1 - m2,
            m2 - m3,
            m3 - m1,
        ]
    ).T


def _load_numerical_orbits_text(orbit_file: str) -> tuple[jax.Array, jax.Array]:
    """
    Loads numerical satellite orbit data from a plain-text file.

    The file is expected to be readable by `numpy.loadtxt` and to contain 10
    columns: time_in_years, x1, y1, z1, x2, y2, z2, x3, y3, z3, where
    (xi, yi, zi) are the coordinates (in meters) of satellite i at the given
    time. One row per time sample, with rows sorted by increasing time.

    Args:
        orbit_file (str): Path to the plain-text numerical orbit data file.

    Returns:
        tuple: (time_grid, positions_grid), where time_grid has shape
            (samples,) in years and positions_grid has shape
            (samples, 3, 3), indexed as [time, satellite, coordinate].
    """
    data = np.atleast_2d(np.loadtxt(orbit_file))
    if data.shape[1] != 10:
        raise ValueError(
            "Numerical orbit files must have 10 columns: time_in_years, x1, "
            "y1, z1, x2, y2, z2, x3, y3, z3. Got "
            f"{data.shape[1]} columns instead."
        )
    time_grid = jnp.array(data[:, 0])
    positions_grid = jnp.array(data[:, 1:]).reshape(data.shape[0], 3, 3)
    return time_grid, positions_grid


def _load_numerical_orbits_lisaorbits(orbit_file: str) -> tuple[jax.Array, jax.Array]:
    """
    Loads numerical satellite orbit data from an HDF5 orbit file produced by
    the `lisaorbits` package (https://pypi.org/project/lisaorbits/).

    Only the spacecraft positions (dataset `tcb/x`, shape (size, 3, 3) for
    (time, satellite, xyz), in meters) and the TCB time grid (attributes
    `t0` and `dt`, both in seconds, and `size`) are used; velocities,
    accelerations, light travel times, and pseudoranges are ignored. The
    time grid is converted from seconds to years to match this module's
    convention.

    Args:
        orbit_file (str): Path to the HDF5 orbit file.

    Returns:
        tuple: (time_grid, positions_grid), where time_grid has shape
            (samples,) in years and positions_grid has shape
            (samples, 3, 3), indexed as [time, satellite, coordinate].
    """
    with h5py.File(orbit_file, "r") as hdf5:
        version = str(hdf5.attrs["version"])
        if int(version.split(".", 1)[0]) < 2:
            raise ValueError(
                f"Unsupported lisaorbits file version {version!r}; "
                "gw_response requires lisaorbits format version >= 2.0."
            )
        t0 = float(np.asarray(hdf5.attrs["t0"]).item())
        dt = float(np.asarray(hdf5.attrs["dt"]).item())
        size = int(np.asarray(hdf5.attrs["size"]).item())
        dataset = hdf5["tcb/x"]
        assert isinstance(dataset, h5py.Dataset)
        positions_grid = jnp.array(dataset[:])
    time_grid = (t0 + np.arange(size) * dt) / PhysicalConstants().yr
    return jnp.array(time_grid), positions_grid


def load_numerical_orbits(
    orbit_file: str, interpolation_method: str = "linear"
) -> interpax.Interpolator1D:
    """
    Loads numerical satellite orbit data from an external file and builds an
    interpolator for the satellite positions.

    Two file formats are supported, auto-detected from the file content:
      - Plain-text files readable by `numpy.loadtxt`, with 10 columns:
        time_in_years, x1, y1, z1, x2, y2, z2, x3, y3, z3. See
        `_load_numerical_orbits_text`.
      - HDF5 orbit files produced by the `lisaorbits` package
        (https://pypi.org/project/lisaorbits/), format version >= 2.0. See
        `_load_numerical_orbits_lisaorbits`.

    The interpolation coefficients are computed once here (similar in spirit
    to `scipy.interpolate.interp1d`), so evaluating the returned interpolator
    at query times - even repeatedly inside a jit/vmap - only needs to
    evaluate the precomputed spline rather than re-deriving it.

    Args:
        orbit_file (str): Path to the numerical orbit data file.
        interpolation_method (str, optional): The interpolation method
            passed to `interpax.Interpolator1D`, e.g. 'linear', 'nearest',
            'cubic', 'cubic2', 'cardinal', 'catmull-rom', 'monotonic',
            'monotonic-0', or 'akima'. Default is 'linear'.

    Returns:
        interpax.Interpolator1D: An interpolator mapping time (in years) to
            satellite positions. Calling it with query time(s) returns an
            array of shape (..., 3, 3), indexed as [satellite, coordinate].
    """
    if h5py.is_hdf5(orbit_file):
        time_grid, positions_grid = _load_numerical_orbits_lisaorbits(orbit_file)
    else:
        time_grid, positions_grid = _load_numerical_orbits_text(orbit_file)
    return interpax.Interpolator1D(
        time_grid, positions_grid, method=interpolation_method
    )
