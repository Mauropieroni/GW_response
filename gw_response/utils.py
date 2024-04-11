import os

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import chex

import numpy as np
import healpy as hp


@chex.dataclass
class Pixel:
    """
    A data class for handling the pixelization of the sky in astronomical
    observations.

    Attributes:
        NSIDE (int): The number of sides of each pixel in the HEALPix
        pixelization. Default is 8.
        NPIX (int): The total number of pixels, computed based on NSIDE.
        angular_map (jnp.ndarray): An array representing the angular position
        of each pixel.
        theta_pixel (jnp.ndarray): An array of theta (colatitude) values for
        each pixel.
        phi_pixel (jnp.ndarray): An array of phi (longitude) values for each
        pixel.

    The class automatically computes the pixelization upon instantiation or when
    the NSIDE value is changed.
    """

    NSIDE: int = 8
    NPIX: int = None
    angular_map: jnp.ndarray = None
    theta_pixel: jnp.ndarray = None
    phi_pixel: jnp.ndarray = None

    def __post_init__(self):
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

    def compute_pixelisation(self):
        """
        Computes the pixelization parameters of the sky.

        Returns:
            tuple: A tuple containing:
                - NPIX (int): The total number of pixels.
                - angular_map (jnp.ndarray): The angular map array.
                - theta_pixel (jnp.ndarray): The theta values for each pixel.
                - phi_pixel (jnp.ndarray): The phi values for each pixel.
        """
        NPIX = hp.nside2npix(self.NSIDE)
        angular_map = jnp.array(
            jnp.rollaxis(jnp.array(hp.pix2ang(self.NSIDE, jnp.arange(NPIX))), -1)
        )
        theta_pixel = jnp.array(angular_map[:, 0])
        phi_pixel = jnp.array(angular_map[:, 1])
        return NPIX, angular_map, theta_pixel, phi_pixel

    def change_NSIDE(self, NSIDE):
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
def arm_length_exponential(arms_matrix_rescaled, x_vector):
    """
    Compute the exponential factor for the Time Delay Interferometry (TDI).

    The function calculates the exponential factors used in TDI computations for
    a laser interferometer space antenna (LISA) setup. It is part of the process
    of accounting for the time delay in the arms of the interferometer due to
    the finite speed of light.

    Parameters:
    arms_matrix_rescaled (array): A 3D array representing the rescaled arm
    matrices of the interferometer. The dimensions are [configurations,
    vectorial_index (3), arms (6)]. Ordering: [12, 23, 31, 21, 32, 13].
    x_vector (array): A 1D array representing the x values over frequency.
    These values are specific to the LISA interferometer's configuration and
    operational characteristics.

    Returns:
    array: A complex-valued 3D array representing the exponential factors.
    The dimensions are [configurations, x_vector, arms]. These factors are used
    in further calculations of the TDI response.
    """
    arm_lengths = jnp.sqrt(
        jnp.einsum("...ij,...ij->...j", arms_matrix_rescaled, arms_matrix_rescaled)
    )
    xij = jnp.einsum("i,...j->...ij", -1j * x_vector, arm_lengths)
    return jnp.exp(xij)


@jax.jit
def shift_to_center(first, second, third):
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


def coordinates_numerical(*args, **kwargs):
    path = os.path.dirname(os.path.abspath(__file__))
    data = jnp.array(np.loadtxt(path + "/input_data/test_positions.txt"))
    return jnp.reshape(data, (data.shape[0], 3, 3)).T


def arms_matrix_numerical(*args, **kwargs):
    path = os.path.dirname(os.path.abspath(__file__))
    data = np.loadtxt(path + "/input_data/test_armlengths.txt")
    return jnp.reshape(data, (data.shape[0], 3, 6))
