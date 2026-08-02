# Global imports
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Local imports
from gw_response.utils import arm_length_exponential
from gw_response.polarization import (
    unit_vec,
    polarization_tensors_PC_angles,
    polarization_tensors_LR_angles,
)

# Update jax to use 64 bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def geometrical_factor(
    arms_matrix_rescaled: jax.Array, polarization_tensor: jax.Array
) -> jax.Array:
    """
    Projects the gravitational wave polarization tensor onto each detector
    arm, giving the geometrical antenna-pattern factor of the single-link
    response.

    Args:
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        polarization_tensor (jax.Array): Polarization tensor(s) as returned
            by e.g. :func:`gw_response.polarization.polarization_tensors_LR`,
            with shape (pixels, vectorial_index (3), vectorial_index (3)).

    Returns:
        jax.Array: The geometrical factor, with shape (configurations, arms,
            pixels).
    """
    # arms_matrix_rescaled is configurations, vectorial_index, arms
    # polarization_tensor is pixels, vectorial_index, vectorial_index

    arms_tensor = jnp.einsum(
        "...ik,...jk->...ijk", arms_matrix_rescaled, arms_matrix_rescaled / 2
    )

    # the output is configurations, arms, pixels
    return jnp.einsum("...ijk,...ijl->...kl", arms_tensor, polarization_tensor.T)


@jax.jit
def geometrical_factor_PC_angles(
    arms_matrix_rescaled: jax.Array, theta: ArrayLike, phi: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the plus/cross geometrical antenna-pattern factors for each detector arm,
    given the sky position.

    Args:
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
        length, with shape (configurations, vectorial_index (3), arms (6)).
        theta (float or ArrayLike): Colatitude(s) of the sky position(s), in radians.
        phi (float or ArrayLike): Longitude(s) of the sky position(s), in radians.

    Returns:
        tuple: A tuple ``(G_plus, G_cross)`` of jax.Array, each with shape
            (configurations, arms, pixels), giving the plus and cross
            geometrical factors for every arm and sky position.
    """
    e_plus, e_cross = polarization_tensors_PC_angles(theta, phi)

    return (
        geometrical_factor(arms_matrix_rescaled, e_plus),
        geometrical_factor(arms_matrix_rescaled, e_cross),
    )


@jax.jit
def geometrical_factor_LR_angles(
    arms_matrix_rescaled: jax.Array, theta: ArrayLike, phi: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the left/right geometrical antenna-pattern factors for each detector arm,
    given the sky position.

    Args:
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
        length, with shape (configurations, vectorial_index (3), arms (6)).
        theta (float or ArrayLike): Colatitude(s) of the sky position(s), in radians.
        phi (float or ArrayLike): Longitude(s) of the sky position(s), in radians.

    Returns:
        tuple: A tuple ``(G_L, G_R)`` of jax.Array, each with shape
            (configurations, arms, pixels), giving the left and right
            geometrical factors for every arm and sky position.
    """
    e_L, e_R = polarization_tensors_LR_angles(theta, phi)

    return (
        geometrical_factor(arms_matrix_rescaled, e_L),
        geometrical_factor(arms_matrix_rescaled, e_R),
    )


@jax.jit
def xi_k_no_G(
    unit_wavevector: jax.Array, x_vector: jax.Array, arms_matrix_rescaled: jax.Array
) -> jax.Array:
    """
    Computes the finite-arm-length transfer function of the single-link
    response, before the geometrical antenna-pattern factor is applied (see
    :func:`xi_k_A`, which combines this with :func:`geometrical_factor`).

    Args:
        unit_wavevector (jax.Array): Unit wavevector(s), with shape
            (vectorial_index (3), pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over
            frequency.
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).

    Returns:
        jax.Array: The finite-arm-length transfer function, with shape
            (configurations, x_vector, arms, pixels).
    """

    k_dot_arms = jnp.einsum("...ij,ik->...jk", arms_matrix_rescaled, unit_wavevector)

    # These guys will be configurations, arms, pixel
    comb_plus = 1.0 + k_dot_arms
    comb_minus = 1.0 - k_dot_arms

    # These guys are  configurations, x_vector, arms, pixels
    prod_plus = jnp.einsum("i,...kl->...ikl", x_vector, comb_plus)
    prod_minus = jnp.einsum("i,...kl->...ikl", x_vector, comb_minus)

    # These guys are configurations, x_vector, arms, pixels
    return jnp.exp(0.5j * prod_minus) * jnp.sinc(prod_plus / 2.0 / jnp.pi)


@jax.jit
def position_exponential(
    positions_detector_frame_rescaled: jax.Array,
    unit_wavevector: jax.Array,
    x_vector: jax.Array,
) -> jax.Array:
    """
    Computes the plane-wave phase factor picked up by each satellite due to
    its position relative to the detector-frame center.

    Args:
        positions_detector_frame_rescaled (jax.Array): Satellite positions
            relative to the detector-frame center (already shifted for
            numerical precision in the dot product below), rescaled by the
            arm length, with shape (configurations, vectorial_index (3),
            satellite (3)).
        unit_wavevector (jax.Array): Unit wavevector(s), with shape
            (vectorial_index (3), pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The position phase factor, with shape (configurations,
            x_vector, satellite, pixels).
    """

    # This is configurations, satellite, pixels
    scalar = jnp.einsum(
        "...ij,ik->...jk", positions_detector_frame_rescaled, unit_wavevector
    )
    exponent = jnp.einsum("i,...jk->...ijk", -1j * x_vector, scalar)

    # Output is configurations, x_vector, satellite, pixels
    return jnp.exp(exponent)


@jax.jit
def xi_k_A(
    arms_matrix_rescaled: jax.Array,
    unit_wavevector: jax.Array,
    x_vector: jax.Array,
    geometrical: jax.Array,
) -> jax.Array:
    """
    Combines the finite-arm-length transfer function (:func:`xi_k_no_G`)
    with the geometrical antenna-pattern factor to give the single-link
    response kernel, prior to the light-travel-time and position phase
    factors.

    Args:
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        unit_wavevector (jax.Array): Unit wavevector(s), with shape
            (vectorial_index (3), pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over
            frequency.
        geometrical (jax.Array): Geometrical antenna-pattern factor as
            returned by :func:`geometrical_factor`, with shape (configurations,
            arms, pixels).

    Returns:
        jax.Array: The single-link response kernel, with shape
            (configurations, x_vector, arms, pixels).
    """

    # xi_vec is configurations, x_vector, arms, pixels
    xi_vec = xi_k_no_G(unit_wavevector, x_vector, arms_matrix_rescaled)

    # The output is configurations, x_vector, arms, pixels
    return jnp.einsum("...ijk,...jk->...ijk", xi_vec, geometrical)


def xi_k_A_PC_angles(
    arms_matrix_rescaled: jax.Array,
    theta: ArrayLike,
    phi: ArrayLike,
    x_vector: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the single-link response kernel for plus/cross polarizations, given the sky
    position.

    Args:
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        theta (ArrayLike): Polar angle(s) of the sky position, with shape
            (pixels,).
        phi (ArrayLike): Azimuthal angle(s) of the sky position, with shape
            (pixels,).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The single-link response kernel, with shape
            (configurations, x_vector, arms, pixels).
    """
    G_plus, G_cross = geometrical_factor_PC_angles(arms_matrix_rescaled, theta, phi)

    xi_k_P = xi_k_A(arms_matrix_rescaled, unit_vec(theta, phi), x_vector, G_plus)
    xi_k_C = xi_k_A(arms_matrix_rescaled, unit_vec(theta, phi), x_vector, G_cross)

    return xi_k_P, xi_k_C


def xi_k_A_LR_angles(
    arms_matrix_rescaled: jax.Array,
    theta: ArrayLike,
    phi: ArrayLike,
    x_vector: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the single-link response kernel for left/right polarizations, given the sky
    position.

    Args:
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
        length, with shape (configurations, vectorial_index (3), arms (6)).
        theta (ArrayLike): Polar angle(s) of the sky position, with shape (pixels,).
        phi (ArrayLike): Azimuthal angle(s) of the sky position, with shape (pixels,).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        tuple: A tuple ``(xi_k_L, xi_k_R)`` of jax.Array, each with shape
            (configurations, x_vector, arms, pixels), giving the single-link
            response kernel for left and right polarizations for every arm and
            sky position.
    """
    G_left, G_right = geometrical_factor_LR_angles(arms_matrix_rescaled, theta, phi)

    xi_k_L = xi_k_A(arms_matrix_rescaled, unit_vec(theta, phi), x_vector, G_left)
    xi_k_R = xi_k_A(arms_matrix_rescaled, unit_vec(theta, phi), x_vector, G_right)

    return xi_k_L, xi_k_R


@jax.jit
def single_link_response(
    positions_rescaled: jax.Array,
    arms_matrix_rescaled: jax.Array,
    wavevector: jax.Array,
    x_vector: jax.Array,
    xi_k_A: jax.Array,
) -> jax.Array:
    """
    Computes the full single-link (arm) strain response, combining the
    response kernel with the light-travel-time delay and the satellite
    position phase factors.

    Args:
        positions_rescaled (jax.Array): Satellite positions rescaled by the
            arm length, with shape (configurations, vectorial_index (3),
            satellite (3)).
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)).
        wavevector (jax.Array): Unit wavevector(s), with shape
            (vectorial_index (3), pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over
            frequency.
        xi_k_A (jax.Array): Single-link response kernel as returned by
            :func:`xi_k_A`, with shape (configurations, x_vector, arms,
            pixels).

    Returns:
        jax.Array: The single-link strain response, with shape
            (configurations, x_vector, arms, pixels).
    """
    # positions_rescaled is configuration, vector, masses (masses are 1,2,3)
    all_positions_rescaled = jnp.concatenate(
        (positions_rescaled, jnp.roll(positions_rescaled, -1, axis=-1)), axis=-1
    )

    # exp has shape configurations, x_vector, arms, pixels
    position_exp_factor = position_exponential(
        all_positions_rescaled, wavevector, x_vector
    )

    # this guy will be configurations, x_vector, arms
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)

    # this guy will be configurations, arms
    arm_lengths = jnp.sqrt(
        jnp.einsum("...ij,...ij->...j", arms_matrix_rescaled, arms_matrix_rescaled)
    )

    # This will be configurations, x_vector, arms
    prefactor = jnp.einsum("...j,...ij->...ij", arm_lengths, t_retarded_factor)
    # Need to pre-multiply by x to convert to fractional frequency in single
    # link response
    prefactor = jnp.einsum("i,...ij->...ij", x_vector, prefactor)

    # This will be configurations, x_vector, arms, pixels
    return jnp.einsum("...ij,...ijk->...ijk", prefactor, position_exp_factor * xi_k_A)


@jax.jit
def single_link_response_PC_angles(
    positions_rescaled: jax.Array,
    arms_matrix_rescaled: jax.Array,
    theta: ArrayLike,
    phi: ArrayLike,
    x_vector: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the full single-link (arm) strain response for plus/cross polarizations,
    given the sky position.

    Args:
        positions_rescaled (jax.Array): Satellite positions rescaled by the arm length,
        with shape (configurations, vectorial_index (3), satellite (3)).
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
        length, with shape (configurations, vectorial_index (3), arms (6)).
        theta (ArrayLike): Polar angle(s) of the sky position, with shape (pixels,).
        phi (ArrayLike): Azimuthal angle(s) of the sky position, with shape (pixels,).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        tuple: A tuple ``(response_P, response_C)`` of jax.Array, each with shape
        (configurations, x_vector, arms, pixels), giving the single-link strain response
        for plus and cross polarizations for every arm and sky position.
    """
    xi_k_P, xi_k_C = xi_k_A_PC_angles(arms_matrix_rescaled, theta, phi, x_vector)

    response_P = single_link_response(
        positions_rescaled, arms_matrix_rescaled, unit_vec(theta, phi), x_vector, xi_k_P
    )
    response_C = single_link_response(
        positions_rescaled, arms_matrix_rescaled, unit_vec(theta, phi), x_vector, xi_k_C
    )

    return response_P, response_C


@jax.jit
def single_link_response_LR_angles(
    positions_rescaled: jax.Array,
    arms_matrix_rescaled: jax.Array,
    theta: ArrayLike,
    phi: ArrayLike,
    x_vector: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the full single-link (arm) strain response for left/right polarizations,
    given the sky position.

    Args:
        positions_rescaled (jax.Array): Satellite positions rescaled by the arm length,
        with shape (configurations, vectorial_index (3), satellite (3)).
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
        length, with shape (configurations, vectorial_index (3), arms (6)).
        theta (ArrayLike): Polar angle(s) of the sky position, with shape (pixels,).
        phi (ArrayLike): Azimuthal angle(s) of the sky position, with shape (pixels,).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        tuple: A tuple ``(response_L, response_R)`` of jax.Array, each with shape
        (configurations, x_vector, arms, pixels), giving the single-link strain response
        for left and right polarizations for every arm and sky position.
    """
    xi_k_L, xi_k_R = xi_k_A_LR_angles(arms_matrix_rescaled, theta, phi, x_vector)

    response_L = single_link_response(
        positions_rescaled, arms_matrix_rescaled, unit_vec(theta, phi), x_vector, xi_k_L
    )
    response_R = single_link_response(
        positions_rescaled, arms_matrix_rescaled, unit_vec(theta, phi), x_vector, xi_k_R
    )

    return response_L, response_R


@jax.jit
def get_single_link_response(
    polarization_tensor: jax.Array,
    arms_matrix_rescaled: jax.Array,
    wavevector: jax.Array,
    x_vector: jax.Array,
    positions_rescaled: jax.Array,
) -> jax.Array:
    """
    Computes the single-link strain response for a given polarization
    tensor, tying together the geometrical factor, the response kernel, and
    the phase factors.

    Args:
        polarization_tensor (jax.Array): Polarization tensor, with shape (pixels,
            vectorial_index (3), vectorial_index (3)), e.g. one of the tensors
            returned by :func:`gw_response.polarization.polarization_tensors_LR` or
            :func:`gw_response.polarization.polarization_tensors_PC`.
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3), arms (6)).
        wavevector (jax.Array): Unit wavevector(s), with shape (vectorial_index (3),
            pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.
        positions_rescaled (jax.Array): Satellite positions rescaled by the arm length,
            with shape (configurations, vectorial_index (3), satellite (3)).

    Returns:
        jax.Array: The single-link strain response, with shape (configurations,
            x_vector, arms, pixels).
    """
    # This computes the geometrical factor
    geometrical = geometrical_factor(arms_matrix_rescaled, polarization_tensor)

    # This computes the xi vectors
    xi_k_vec = xi_k_A(arms_matrix_rescaled, wavevector, x_vector, geometrical)

    # This will be configurations, x_vector, arms, pixels
    return single_link_response(
        positions_rescaled, arms_matrix_rescaled, wavevector, x_vector, xi_k_vec
    )
