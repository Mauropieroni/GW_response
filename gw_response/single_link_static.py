# Global imports
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Local imports
from gw_response.utils import arm_length_exponential, arm_lengths_from_matrix
from gw_response.polarization import (
    unit_vec,
    polarization_tensors_PC_angles,
    polarization_tensors_LR_angles,
)
from gw_response.single_link_utils import (
    geometrical_factor,
    position_exponential,
    finite_arm_transfer_function,
)

# Update jax to use 64 bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def geometrical_factor_PC_angles_static(
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
            (configurations, arms, pixels), giving the plus and cross geometrical
            factors for every arm and sky position.
    """
    e_plus, e_cross = polarization_tensors_PC_angles(theta, phi)

    return (
        geometrical_factor(arms_matrix_rescaled, e_plus),
        geometrical_factor(arms_matrix_rescaled, e_cross),
    )


@jax.jit
def geometrical_factor_LR_angles_static(
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
        tuple: A tuple ``(G_L, G_R)`` of jax.Array, each with shape (configurations,
            arms, pixels), giving the left and right geometrical factors for every arm
            and sky position.
    """
    e_L, e_R = polarization_tensors_LR_angles(theta, phi)

    return (
        geometrical_factor(arms_matrix_rescaled, e_L),
        geometrical_factor(arms_matrix_rescaled, e_R),
    )


@jax.jit
def xi_k_no_G_static(
    unit_wavevector: jax.Array, x_vector: jax.Array, arms_matrix_rescaled: jax.Array
) -> jax.Array:
    """
    Computes the finite-arm-length transfer function of the single-link response, before
    the geometrical antenna-pattern factor is applied (see :func:`xi_k_A_static`, which
    combines this with :func:`gw_response.single_link_utils.geometrical_factor`). Static
    (simultaneous-arm-geometry) counterpart of
    :func:`gw_response.single_link_retarded.xi_k_no_G_retarded`. The ``M_ij`` factor of
    Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929) eq. 2.14 -- see
    :func:`gw_response.single_link_utils.finite_arm_transfer_function` regarding the
    phase-splitting caveat.

    Args:
        unit_wavevector (jax.Array): Unit wavevector(s), with shape (vectorial_index
            (3), pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)).

    Returns:
        jax.Array: The finite-arm-length transfer function, with shape (configurations,
            x_vector, arms, pixels).
    """

    k_dot_arms = jnp.einsum("...ij,ik->...jk", arms_matrix_rescaled, unit_wavevector)

    # These guys will be configurations, arms, pixel
    comb_plus = 1.0 + k_dot_arms
    comb_minus = 1.0 - k_dot_arms

    return finite_arm_transfer_function(comb_plus, comb_minus, x_vector)


@jax.jit
def xi_k_A_static(
    arms_matrix_rescaled: jax.Array,
    unit_wavevector: jax.Array,
    x_vector: jax.Array,
    geometrical: jax.Array,
) -> jax.Array:
    """
    Combines the finite-arm-length transfer function (:func:`xi_k_no_G_static`) with the
    geometrical antenna-pattern factor to give the single-link response kernel, prior to
    the light-travel-time and position phase factors. The
    ``ξ_ij^A(f,k̂) = e^{-2πifk̂·L⃗_ij} M_ij(f,k̂) G^A(k̂,l̂_ij)`` kernel of Hartwig,
    Lilley, Muratore & Pieroni (arXiv:2303.15929) eq. 2.13 (the ``e^{-2πifk̂·L⃗_ij}``
    arm-vector phase factor is folded into :func:`xi_k_no_G_static`'s own ``M_ij``
    here rather than kept separate).

    Args:
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)).
        unit_wavevector (jax.Array): Unit wavevector(s), with shape (vectorial_index
            (3), pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.
        geometrical (jax.Array): Geometrical antenna-pattern factor as returned by
            :func:`gw_response.single_link_utils.geometrical_factor`, with shape
            (configurations, arms, pixels).

    Returns:
        jax.Array: The single-link response kernel, with shape (configurations,
            x_vector, arms, pixels).
    """

    # xi_vec is configurations, x_vector, arms, pixels
    xi_vec = xi_k_no_G_static(unit_wavevector, x_vector, arms_matrix_rescaled)

    # The output is configurations, x_vector, arms, pixels
    return jnp.einsum("...ijk,...jk->...ijk", xi_vec, geometrical)


def xi_k_A_PC_angles_static(
    arms_matrix_rescaled: jax.Array,
    theta: ArrayLike,
    phi: ArrayLike,
    x_vector: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the single-link response kernel for plus/cross polarizations, given the sky
    position.

    Args:
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)).
        theta (ArrayLike): Polar angle(s) of the sky position, with shape (pixels,).
        phi (ArrayLike): Azimuthal angle(s) of the sky position, with shape (pixels,).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The single-link response kernel, with shape (configurations,
            x_vector, arms, pixels).
    """

    G_plus, G_cross = geometrical_factor_PC_angles_static(
        arms_matrix_rescaled, theta, phi
    )

    k_vec = unit_vec(theta, phi)
    xi_k_P = xi_k_A_static(arms_matrix_rescaled, k_vec, x_vector, G_plus)
    xi_k_C = xi_k_A_static(arms_matrix_rescaled, k_vec, x_vector, G_cross)

    return xi_k_P, xi_k_C


def xi_k_A_LR_angles_static(
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
            (configurations, x_vector, arms, pixels), giving the single-link response
            kernel for left and right polarizations for every arm and sky position.
    """
    G_left, G_right = geometrical_factor_LR_angles_static(
        arms_matrix_rescaled, theta, phi
    )

    k_vec = unit_vec(theta, phi)
    xi_k_L = xi_k_A_static(arms_matrix_rescaled, k_vec, x_vector, G_left)
    xi_k_R = xi_k_A_static(arms_matrix_rescaled, k_vec, x_vector, G_right)

    return xi_k_L, xi_k_R


@jax.jit
def single_link_response_static(
    positions_rescaled: jax.Array,
    arms_matrix_rescaled: jax.Array,
    wavevector: jax.Array,
    x_vector: jax.Array,
    xi_k_A_static: jax.Array,
) -> jax.Array:
    """
    Computes the full single-link (arm) strain response, combining the response kernel
    with the light-travel-time delay and the satellite position phase factors. The
    full frequency-domain single-link transfer function of Hartwig, Lilley, Muratore &
    Pieroni (arXiv:2303.15929) eq. 2.12 (the ``(f/f_ij) e^{2πif(t-L_ij)}`` prefactor and
    ``e^{-2πifk̂·x⃗_i}`` position phase combined with :func:`xi_k_A_static`'s own
    ``ξ_ij^A``), for the symmetric-arm (static) case; static-arm counterpart of
    :func:`gw_response.single_link_retarded.single_link_response_retarded`.

    Args:
        positions_rescaled (jax.Array): Satellite positions rescaled by the arm length,
            with shape (configurations, vectorial_index (3), satellite (3)).
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)).
        wavevector (jax.Array): Unit wavevector(s), with shape (vectorial_index (3),
            pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.
        xi_k_A_static (jax.Array): Single-link response kernel as returned by
            :func:`xi_k_A_static`, with shape (configurations, x_vector, arms, pixels).

    Returns:
        jax.Array: The single-link strain response, with shape (configurations,
            x_vector, arms, pixels).
    """
    # Pairs each arm's receiver position with its emitter's (satellite i, then i+1
    # cyclically), doubling the trailing axis from 3 satellites to 6 arms.
    all_positions_rescaled = jnp.concatenate(
        (positions_rescaled, jnp.roll(positions_rescaled, -1, axis=-1)), axis=-1
    )
    position_exp_factor = position_exponential(
        all_positions_rescaled, wavevector, x_vector
    )
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)
    arm_lengths = arm_lengths_from_matrix(arms_matrix_rescaled)
    prefactor = jnp.einsum("...j,...ij->...ij", arm_lengths, t_retarded_factor)
    # Need to pre-multiply by -i*x to convert to fractional frequency in single
    # link response -- see the matching comment in single_link_retarded.py's
    # single_link_response_retarded (this static pipeline's own counterpart)
    # for the derivation/cross-check; kept consistent with it here so the two
    # pipelines agree in the frozen-geometry limit they're meant to share.
    prefactor = jnp.einsum("i,...ij->...ij", -1j * x_vector, prefactor)

    return jnp.einsum(
        "...ij,...ijk->...ijk", prefactor, position_exp_factor * xi_k_A_static
    )


@jax.jit
def single_link_response_PC_angles_static(
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
            (configurations, x_vector, arms, pixels), giving the single-link strain
            response for plus and cross polarizations for every arm and sky position.
    """
    xi_k_P, xi_k_C = xi_k_A_PC_angles_static(arms_matrix_rescaled, theta, phi, x_vector)

    response_P = single_link_response_static(
        positions_rescaled, arms_matrix_rescaled, unit_vec(theta, phi), x_vector, xi_k_P
    )
    response_C = single_link_response_static(
        positions_rescaled, arms_matrix_rescaled, unit_vec(theta, phi), x_vector, xi_k_C
    )

    return response_P, response_C


@jax.jit
def single_link_response_LR_angles_static(
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
            (configurations, x_vector, arms, pixels), giving the single-link strain
            response for left and right polarizations for every arm and sky position.
    """
    xi_k_L, xi_k_R = xi_k_A_LR_angles_static(arms_matrix_rescaled, theta, phi, x_vector)

    response_L = single_link_response_static(
        positions_rescaled, arms_matrix_rescaled, unit_vec(theta, phi), x_vector, xi_k_L
    )
    response_R = single_link_response_static(
        positions_rescaled, arms_matrix_rescaled, unit_vec(theta, phi), x_vector, xi_k_R
    )

    return response_L, response_R


@jax.jit
def get_single_link_response_static(
    polarization_tensor: jax.Array,
    arms_matrix_rescaled: jax.Array,
    wavevector: jax.Array,
    x_vector: jax.Array,
    positions_rescaled: jax.Array,
) -> jax.Array:
    """
    Computes the single-link strain response for a given polarization tensor, tying
    together the geometrical factor, the response kernel, and the phase factors.

    Args:
        polarization_tensor (jax.Array): Polarization tensor, with shape (pixels,
            vectorial_index (3), vectorial_index (3)), e.g. one of the tensors returned
            by :func:`gw_response.polarization.polarization_tensors_LR` or
            :func:`gw_response.polarization.polarization_tensors_PC`.
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)).
        wavevector (jax.Array): Unit wavevector(s), with shape (vectorial_index (3),
            pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.
        positions_rescaled (jax.Array): Satellite positions rescaled by the arm length,
            with shape (configurations, vectorial_index (3), satellite (3)).

    Returns:
        jax.Array: The single-link strain response, with shape (configurations,
            x_vector, arms, pixels).
    """
    geometrical = geometrical_factor(arms_matrix_rescaled, polarization_tensor)
    xi_k_vec = xi_k_A_static(arms_matrix_rescaled, wavevector, x_vector, geometrical)
    return single_link_response_static(
        positions_rescaled, arms_matrix_rescaled, wavevector, x_vector, xi_k_vec
    )
