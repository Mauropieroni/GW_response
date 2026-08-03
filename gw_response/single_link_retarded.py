# Global imports
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Local imports
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
from gw_response.utils import delay_factor

# Update jax to use 64 bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def xi_k_no_G_retarded(
    arm_vector_retarded_rescaled: jax.Array,
    ltt_rescaled: jax.Array,
    wavevector: jax.Array,
    x_vector: jax.Array,
) -> jax.Array:
    """
    Retarded counterpart of :func:`gw_response.single_link_static.xi_k_no_G_static`: the
    finite-arm-length transfer function before the geometrical antenna-pattern factor is
    applied, for a genuinely asymmetric arm.

    Args:
        arm_vector_retarded_rescaled (jax.Array): Vector from the receiver's current
            position to the emitter's position at the retarded (emission) time, rescaled
            by the nominal arm length, with shape (configurations, vectorial_index (3),
            arms).
        ltt_rescaled (jax.Array): The light-travel-time estimate used to find the
            retarded emission time, rescaled to the same dimensionless units as
            `arm_vector_retarded_rescaled`'s magnitude, with shape (configurations,
            arms).
        wavevector (jax.Array): Unit wavevector(s), with shape (vectorial_index (3),
            pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency (`L` the
            nominal arm length).

    Returns:
        jax.Array: The retarded finite-arm-length transfer function, with shape
            (configurations, x_vector, arms, pixels).
    """
    ltt_rescaled = jnp.asarray(ltt_rescaled)
    k_dot_arm = jnp.einsum("...ij,ik->...jk", arm_vector_retarded_rescaled, wavevector)
    comb_plus = ltt_rescaled[..., None] + k_dot_arm
    comb_minus = ltt_rescaled[..., None] - k_dot_arm

    return finite_arm_transfer_function(comb_plus, comb_minus, x_vector)


@jax.jit
def xi_k_A_retarded(
    polarization_tensor: jax.Array,
    arm_vector_retarded_rescaled: jax.Array,
    ltt_rescaled: jax.Array,
    wavevector: jax.Array,
    x_vector: jax.Array,
) -> jax.Array:
    """
    Retarded counterpart of :func:`gw_response.single_link_static.xi_k_A_static`: the
    single-link response kernel for a genuinely asymmetric arm (``12 != 21``), where the
    emitter's retarded arm length differs from the light-travel-time-derived
    `ltt_rescaled`.

    Args:
        polarization_tensor (jax.Array): Polarization tensor, with shape (pixels,
            vectorial_index (3), vectorial_index (3)).
        arm_vector_retarded_rescaled (jax.Array): Vector from the receiver's current
            position to the emitter's position at the retarded (emission) time, rescaled
            by the nominal arm length, with shape (configurations, vectorial_index (3),
            arms).
        ltt_rescaled (jax.Array): The light-travel-time estimate used to find the
            retarded emission time, rescaled to the same dimensionless units as
            `arm_vector_retarded_rescaled`'s magnitude, with shape (configurations,
            arms).
        wavevector (jax.Array): Unit wavevector(s), with shape (vectorial_index (3),
            pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency (`L` the
            nominal arm length).

    Returns:
        jax.Array: The retarded single-link response kernel, with shape (configurations,
            x_vector, arms, pixels).
    """
    ltt_rescaled = jnp.asarray(ltt_rescaled)
    arm_length_retarded_rescaled = jnp.linalg.norm(arm_vector_retarded_rescaled, axis=1)
    unit_arm = arm_vector_retarded_rescaled / arm_length_retarded_rescaled[:, None, :]

    xi_no_G = xi_k_no_G_retarded(
        arm_vector_retarded_rescaled, ltt_rescaled, wavevector, x_vector
    )

    # k_dot_arm/comb_plus are also needed (unlike the static case) for the
    # length_correction term below, so they're recomputed here rather than
    # threaded out of xi_k_no_G_retarded.
    k_dot_arm = jnp.einsum("...ij,ik->...jk", arm_vector_retarded_rescaled, wavevector)
    comb_plus = ltt_rescaled[..., None] + k_dot_arm

    geometrical_unit = geometrical_factor(unit_arm, polarization_tensor)
    length_correction = (arm_length_retarded_rescaled[..., None] * comb_plus) / (
        ltt_rescaled[..., None] * (arm_length_retarded_rescaled[..., None] + k_dot_arm)
    )
    geometrical = geometrical_unit * length_correction

    return jnp.einsum("...ijk,...jk->...ijk", xi_no_G, geometrical)


@jax.jit
def single_link_response_retarded(
    receiver_positions_rescaled: jax.Array,
    ltt_rescaled: jax.Array,
    wavevector: jax.Array,
    x_vector: jax.Array,
    xi_k_A_retarded_value: jax.Array,
) -> jax.Array:
    """
    Retarded counterpart of
    :func:`gw_response.single_link_static.single_link_response_static`: combines the
    retarded response kernel with the light-travel-time delay and receiver-position
    phase factors.

    Args:
        receiver_positions_rescaled (jax.Array): Receiver's position at each arm's own
            reception time, rescaled by the nominal arm length, with shape
            (configurations, vectorial_index (3), arms).
        ltt_rescaled (jax.Array): The light-travel-time estimate used to find the
            retarded emission time, rescaled to the same dimensionless units as the arm
            vector's magnitude, with shape (configurations, arms).
        wavevector (jax.Array): Unit wavevector(s), with shape (vectorial_index (3),
            pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency (`L` the
            nominal arm length).
        xi_k_A_retarded_value (jax.Array): Retarded response kernel as returned by
            :func:`xi_k_A_retarded`, with shape (configurations, x_vector, arms,
            pixels).

    Returns:
        jax.Array: The single-link strain response, with shape (configurations,
            x_vector, arms, pixels).
    """
    position_exp_factor = position_exponential(
        receiver_positions_rescaled, wavevector, x_vector
    )

    t_retarded_factor = delay_factor(ltt_rescaled, x_vector)
    prefactor = jnp.einsum("...j,...ij->...ij", ltt_rescaled, t_retarded_factor)
    prefactor = jnp.einsum("i,...ij->...ij", x_vector, prefactor)

    return jnp.einsum(
        "...ij,...ijk->...ijk", prefactor, position_exp_factor * xi_k_A_retarded_value
    )


@jax.jit
def get_single_link_response_retarded(
    polarization_tensor: jax.Array,
    arm_vector_retarded_rescaled: jax.Array,
    ltt_rescaled: jax.Array,
    wavevector: jax.Array,
    x_vector: jax.Array,
    receiver_positions_rescaled: jax.Array,
) -> jax.Array:
    """
    Single-link strain response for a genuinely asymmetric arm (``12 != 21``), i.e. the
    retarded counterpart of
    :func:`gw_response.single_link_static.get_single_link_response_static` for a moving
    detector.

    Args:
        polarization_tensor (jax.Array): Polarization tensor, with shape (pixels,
            vectorial_index (3), vectorial_index (3)).
        arm_vector_retarded_rescaled (jax.Array): Vector from the receiver's current
            position to the emitter's position at the retarded (emission) time, rescaled
            by the nominal arm length, with shape (configurations, vectorial_index (3),
            arms).
        ltt_rescaled (jax.Array): The light-travel-time estimate used to find the
            retarded emission time, rescaled to the same dimensionless units as
            `arm_vector_retarded_rescaled`'s magnitude, with shape (configurations,
            arms).
        wavevector (jax.Array): Unit wavevector(s), with shape (vectorial_index (3),
            pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency (`L` the
            nominal arm length).
        receiver_positions_rescaled (jax.Array): Receiver's position at each arm's own
            reception time, rescaled by the nominal arm length, with shape
            (configurations, vectorial_index (3), arms).

    Returns:
        jax.Array: The single-link strain response, with shape (configurations,
            x_vector, arms, pixels).
    """
    xi_k_A_retarded_value = xi_k_A_retarded(
        polarization_tensor,
        arm_vector_retarded_rescaled,
        ltt_rescaled,
        wavevector,
        x_vector,
    )
    return single_link_response_retarded(
        receiver_positions_rescaled,
        ltt_rescaled,
        wavevector,
        x_vector,
        xi_k_A_retarded_value,
    )


@jax.jit
def get_single_link_response_retarded_PC_angles(
    arm_vector_retarded_rescaled: jax.Array,
    ltt_rescaled: jax.Array,
    theta: ArrayLike,
    phi: ArrayLike,
    x_vector: jax.Array,
    receiver_positions_rescaled: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the retarded single-link strain response for plus/cross polarizations,
    given the sky position.

    Args:
        arm_vector_retarded_rescaled (jax.Array): Vector from the receiver's current
            position to the emitter's position at the retarded (emission) time, rescaled
            by the nominal arm length, with shape (configurations, vectorial_index (3),
            arms).
        ltt_rescaled (jax.Array): The light-travel-time estimate used to find the
            retarded emission time, rescaled to the same dimensionless units as
            `arm_vector_retarded_rescaled`'s magnitude, with shape (configurations,
            arms).
        theta (ArrayLike): Polar angle(s) of the sky position, with shape (pixels,).
        phi (ArrayLike): Azimuthal angle(s) of the sky position, with shape (pixels,).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency (`L` the
            nominal arm length).
        receiver_positions_rescaled (jax.Array): Receiver's position at each arm's own
            reception time, rescaled by the nominal arm length, with shape
            (configurations, vectorial_index (3), arms).

    Returns:
        tuple: A tuple ``(response_P, response_C)`` of jax.Array, each with shape
            (configurations, x_vector, arms, pixels), giving the single-link strain
            response for plus and cross polarizations for every arm and sky position.
    """
    e_plus, e_cross = polarization_tensors_PC_angles(theta, phi)
    wavevector = unit_vec(theta, phi)

    response_P = get_single_link_response_retarded(
        e_plus,
        arm_vector_retarded_rescaled,
        ltt_rescaled,
        wavevector,
        x_vector,
        receiver_positions_rescaled,
    )
    response_C = get_single_link_response_retarded(
        e_cross,
        arm_vector_retarded_rescaled,
        ltt_rescaled,
        wavevector,
        x_vector,
        receiver_positions_rescaled,
    )

    return response_P, response_C


@jax.jit
def get_single_link_response_retarded_LR_angles(
    arm_vector_retarded_rescaled: jax.Array,
    ltt_rescaled: jax.Array,
    theta: ArrayLike,
    phi: ArrayLike,
    x_vector: jax.Array,
    receiver_positions_rescaled: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Computes the retarded single-link strain response for left/right polarizations,
    given the sky position.

    Args:
        arm_vector_retarded_rescaled (jax.Array): Vector from the receiver's current
            position to the emitter's position at the retarded (emission) time, rescaled
            by the nominal arm length, with shape (configurations, vectorial_index (3),
            arms).
        ltt_rescaled (jax.Array): The light-travel-time estimate used to find the
            retarded emission time, rescaled to the same dimensionless units as
            `arm_vector_retarded_rescaled`'s magnitude, with shape (configurations,
            arms).
        theta (ArrayLike): Polar angle(s) of the sky position, with shape (pixels,).
        phi (ArrayLike): Azimuthal angle(s) of the sky position, with shape (pixels,).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency (`L` the
            nominal arm length).
        receiver_positions_rescaled (jax.Array): Receiver's position at each arm's own
            reception time, rescaled by the nominal arm length, with shape
            (configurations, vectorial_index (3), arms).

    Returns:
        tuple: A tuple ``(response_L, response_R)`` of jax.Array, each with shape
            (configurations, x_vector, arms, pixels), giving the single-link strain
            response for left and right polarizations for every arm and sky position.
    """
    e_L, e_R = polarization_tensors_LR_angles(theta, phi)
    wavevector = unit_vec(theta, phi)

    response_L = get_single_link_response_retarded(
        e_L,
        arm_vector_retarded_rescaled,
        ltt_rescaled,
        wavevector,
        x_vector,
        receiver_positions_rescaled,
    )
    response_R = get_single_link_response_retarded(
        e_R,
        arm_vector_retarded_rescaled,
        ltt_rescaled,
        wavevector,
        x_vector,
        receiver_positions_rescaled,
    )

    return response_L, response_R
