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
from gw_response.single_link import geometrical_factor, position_exponential

# Update jax to use 64 bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def xi_k_A_retarded(
    polarization_tensor: jax.Array,
    arm_vector_retarded_rescaled: jax.Array,
    ltt_rescaled: jax.Array,
    wavevector: jax.Array,
    x_vector: jax.Array,
) -> jax.Array:
    """
    Retarded counterpart of :func:`gw_response.single_link.xi_k_A`: the
    single-link response kernel for a genuinely asymmetric arm (``12 !=
    21``), combining the finite-arm-length transfer function with the
    geometrical antenna-pattern factor.

    This can't just reuse :func:`gw_response.single_link.xi_k_no_G`/
    :func:`gw_response.single_link.geometrical_factor`/
    :func:`gw_response.single_link.xi_k_A` unchanged because those assume
    one shared, simultaneous arm geometry; here, `ltt_rescaled` (the
    light-travel-time estimate `arm_vector_retarded_rescaled`'s own
    emission time was computed from) and
    ``jnp.linalg.norm(arm_vector_retarded_rescaled, axis=1)`` (the *true*
    retarded arm length) genuinely differ -- by exactly the distance the
    emitter moved during the transit -- so :func:`geometrical_factor` is
    fed a unit arm vector instead of the rescaled one, and the exact
    scaling this formula's own retardation structure requires
    (`length_correction`) is applied explicitly here. Conflating the two
    lengths (as :func:`gw_response.single_link.xi_k_A` does, fine for a
    static arm where they coincide) would leave a residual error that does
    not shrink with a shorter observation window, since it is not a
    "segment span" effect at all.

    Args:
        polarization_tensor (jax.Array): Polarization tensor, with shape (pixels,
            vectorial_index (3), vectorial_index (3)).
        arm_vector_retarded_rescaled (jax.Array): Vector from the receiver's
            current position to the emitter's position at the retarded
            (emission) time, rescaled by the nominal arm length, with shape
            (configurations, vectorial_index (3), arms).
        ltt_rescaled (jax.Array): The light-travel-time estimate used to
            find the retarded emission time, rescaled to the same
            dimensionless units as `arm_vector_retarded_rescaled`'s
            magnitude (i.e. light-travel-time * light_speed / nominal arm
            length), with shape (configurations, arms).
        wavevector (jax.Array): Unit wavevector(s), with shape
            (vectorial_index (3), pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over
            frequency (`L` the nominal arm length).

    Returns:
        jax.Array: The retarded single-link response kernel, with shape
            (configurations, x_vector, arms, pixels).
    """
    ltt_rescaled = jnp.asarray(ltt_rescaled)
    arm_length_retarded_rescaled = jnp.linalg.norm(arm_vector_retarded_rescaled, axis=1)
    unit_arm = arm_vector_retarded_rescaled / arm_length_retarded_rescaled[:, None, :]

    k_dot_arm = jnp.einsum("...ij,ik->...jk", arm_vector_retarded_rescaled, wavevector)
    comb_plus = ltt_rescaled[..., None] + k_dot_arm
    comb_minus = ltt_rescaled[..., None] - k_dot_arm

    prod_plus = jnp.einsum("i,...kl->...ikl", x_vector, comb_plus)
    prod_minus = jnp.einsum("i,...kl->...ikl", x_vector, comb_minus)
    xi_no_G = jnp.exp(0.5j * prod_minus) * jnp.sinc(prod_plus / 2.0 / jnp.pi)

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
    :func:`gw_response.single_link.single_link_response`: combines the
    retarded response kernel (:func:`xi_k_A_retarded`) with the light-
    travel-time delay and receiver-position phase factors.

    Args:
        receiver_positions_rescaled (jax.Array): Receiver's position at each
            arm's own reception time, rescaled by the nominal arm length,
            with shape (configurations, vectorial_index (3), arms).
        ltt_rescaled (jax.Array): The light-travel-time estimate used to
            find the retarded emission time, rescaled to the same
            dimensionless units as the arm vector's magnitude, with shape
            (configurations, arms).
        wavevector (jax.Array): Unit wavevector(s), with shape
            (vectorial_index (3), pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over
            frequency (`L` the nominal arm length).
        xi_k_A_retarded_value (jax.Array): Retarded response kernel as
            returned by :func:`xi_k_A_retarded`, with shape (configurations,
            x_vector, arms, pixels).

    Returns:
        jax.Array: The single-link strain response, with shape
            (configurations, x_vector, arms, pixels).
    """
    position_exp_factor = position_exponential(
        receiver_positions_rescaled, wavevector, x_vector
    )

    t_retarded_factor = jnp.exp(
        jnp.einsum("i,...j->...ij", -1j * x_vector, ltt_rescaled)
    )
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
    Single-link strain response for a genuinely asymmetric arm (``12 !=
    21``): the emitter's position is taken at its own light-travel-time-
    retarded instant rather than simultaneously with the receiver, so the
    arm vector -- and this transfer function -- differs by propagation
    direction. :func:`gw_response.single_link.get_single_link_response`
    assumes one shared, simultaneous geometry for a truly static arm; this
    is its exact counterpart for a moving one, built the same way from
    :func:`xi_k_A_retarded` (the retarded response kernel) and
    :func:`single_link_response_retarded` (the delay/position phase
    factors) -- see those for the derivation.

    Args:
        polarization_tensor (jax.Array): Polarization tensor, with shape (pixels,
            vectorial_index (3), vectorial_index (3)).
        arm_vector_retarded_rescaled (jax.Array): Vector from the receiver's
            current position to the emitter's position at the retarded
            (emission) time, rescaled by the nominal arm length, with shape
            (configurations, vectorial_index (3), arms).
        ltt_rescaled (jax.Array): The light-travel-time estimate used to
            find the retarded emission time, rescaled to the same
            dimensionless units as `arm_vector_retarded_rescaled`'s
            magnitude (i.e. light-travel-time * light_speed / nominal arm
            length), with shape (configurations, arms).
        wavevector (jax.Array): Unit wavevector(s), with shape
            (vectorial_index (3), pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over
            frequency (`L` the nominal arm length).
        receiver_positions_rescaled (jax.Array): Receiver's position at each
            arm's own reception time, rescaled by the nominal arm length,
            with shape (configurations, vectorial_index (3), arms).

    Returns:
        jax.Array: The single-link strain response, with shape
            (configurations, x_vector, arms, pixels).
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
    Computes the single-link strain response for a genuinely asymmetric
    (retarded) arm, for plus/cross polarizations, given the sky position.

    The ``_angles`` sibling of :func:`get_single_link_response_retarded`:
    builds the plus/cross polarization tensors and wavevector from
    ``(theta, phi)`` internally, so the caller doesn't need to build them
    (via
    :func:`gw_response.polarization.polarization_tensors_PC_angles`/
    :func:`gw_response.polarization.unit_vec`) first.

    Args:
        arm_vector_retarded_rescaled (jax.Array): Vector from the
            receiver's current position to the emitter's position at the
            retarded (emission) time, rescaled by the nominal arm length,
            with shape (configurations, vectorial_index (3), arms).
        ltt_rescaled (jax.Array): The light-travel-time estimate used to
            find the retarded emission time, rescaled to the same
            dimensionless units as `arm_vector_retarded_rescaled`'s
            magnitude, with shape (configurations, arms).
        theta (ArrayLike): Polar angle(s) of the sky position, with shape
            (pixels,).
        phi (ArrayLike): Azimuthal angle(s) of the sky position, with shape
            (pixels,).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over
            frequency (`L` the nominal arm length).
        receiver_positions_rescaled (jax.Array): Receiver's position at
            each arm's own reception time, rescaled by the nominal arm
            length, with shape (configurations, vectorial_index (3), arms).

    Returns:
        tuple: A tuple ``(response_P, response_C)`` of jax.Array, each with
            shape (configurations, x_vector, arms, pixels), giving the
            single-link strain response for plus and cross polarizations
            for every arm and sky position.
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
    Computes the single-link strain response for a genuinely asymmetric
    (retarded) arm, for left/right polarizations, given the sky position.

    The ``_angles`` sibling of :func:`get_single_link_response_retarded`:
    builds the left/right polarization tensors and wavevector from
    ``(theta, phi)`` internally, so the caller doesn't need to build them
    (via
    :func:`gw_response.polarization.polarization_tensors_LR_angles`/
    :func:`gw_response.polarization.unit_vec`) first.

    Args:
        arm_vector_retarded_rescaled (jax.Array): Vector from the
            receiver's current position to the emitter's position at the
            retarded (emission) time, rescaled by the nominal arm length,
            with shape (configurations, vectorial_index (3), arms).
        ltt_rescaled (jax.Array): The light-travel-time estimate used to
            find the retarded emission time, rescaled to the same
            dimensionless units as `arm_vector_retarded_rescaled`'s
            magnitude, with shape (configurations, arms).
        theta (ArrayLike): Polar angle(s) of the sky position, with shape
            (pixels,).
        phi (ArrayLike): Azimuthal angle(s) of the sky position, with shape
            (pixels,).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over
            frequency (`L` the nominal arm length).
        receiver_positions_rescaled (jax.Array): Receiver's position at
            each arm's own reception time, rescaled by the nominal arm
            length, with shape (configurations, vectorial_index (3), arms).

    Returns:
        tuple: A tuple ``(response_L, response_R)`` of jax.Array, each with
            shape (configurations, x_vector, arms, pixels), giving the
            single-link strain response for left and right polarizations
            for every arm and sky position.
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
