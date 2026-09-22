# Global imports
import jax
import jax.numpy as jnp

# Update jax to use 64 bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def armlength_suppression_function(
    comb_plus: jax.Array, comb_minus: jax.Array, x_vector: jax.Array
) -> jax.Array:
    """
    Computes the sinc/phase suppression factor (induced by several oscillations in a
    single arm light-travel time). Structurally it corresponds to the M_{ij}(f,\\hat{k})
    factor in eq. 2.14's of Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929)

        ``M_{ij}(f,\\hat{k}) = e^{i π f L_{ij}(1 + \\hat{k} · \\hat{l}_{ij})}
                            \times sinc(π f L_{ij}(1 + \\hat{k} · \\hat{l}_{ij}))``

    Args:
        comb_plus (jax.Array): `comb_plus`, with shape (configurations, arms, pixels).
        comb_minus (jax.Array): `comb_minus`, same shape as `comb_plus`.
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The finite-arm-length transfer function, with shape (configurations,
            x_vector, arms, pixels).
    """
    prod_plus = jnp.einsum("i,...kl->...ikl", x_vector, comb_plus)
    prod_minus = jnp.einsum("i,...kl->...ikl", x_vector, comb_minus)
    return jnp.exp(0.5j * prod_minus) * jnp.sinc(prod_plus / 2.0 / jnp.pi)


@jax.jit
def geometrical_factor(
    arms_matrix_rescaled: jax.Array, polarization_tensor: jax.Array
) -> jax.Array:
    """
    Projects the gravitational wave polarization tensor onto each detector arm, giving
    the geometrical antenna-pattern factor of the single-link response. Implements
    eq. 2.14's of Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929)

        ``G^A(\\hat{k},\\hat{l}_{ij}) =
                \\hat{l}_{ij}^a \\hat{l}_{ij}^b e^A_{ab}(\\hat{k}) / 2``

    for `\\hat{l}_{ij}` = a unit arm direction and `e^A` = `polarization_tensor`.

    Args:
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)).
        polarization_tensor (jax.Array): Polarization tensor(s) as returned by e.g.
            :func:`gw_response.polarization.polarization_tensors_LR`, with shape
            (pixels, vectorial_index (3), vectorial_index (3)).

    Returns:
        jax.Array: The geometrical factor, with shape (configurations, arms, pixels).
    """
    arms_tensor = jnp.einsum(
        "...ik,...jk->...ijk", arms_matrix_rescaled, arms_matrix_rescaled
    )
    return jnp.einsum("...ijk,...ijl->...kl", arms_tensor, polarization_tensor.T) / 2.0


@jax.jit
def position_exponential(
    positions_detector_frame_rescaled: jax.Array,
    unit_wavevector: jax.Array,
    x_vector: jax.Array,
) -> jax.Array:
    """
    Computes the plane-wave phase factor picked up by each satellite due to its position
    relative to the detector-frame center. It matches the exponential factor

        ``e^{- 2 π i f \\hat{k} · \\vec{x}_i}``

    in eq. 2.12 of Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929).

    Args:
        positions_detector_frame_rescaled (jax.Array): Satellite positions relative to
            the detector-frame center (already shifted for numerical precision in the
            dot product below), rescaled by the arm length, with shape (configurations,
            vectorial_index (3), satellite (3)).
        unit_wavevector (jax.Array): Unit wavevector(s), with shape (vectorial_index
            (3), pixels).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The position phase factor, with shape (configurations, x_vector,
            satellite, pixels).
    """

    scalar = jnp.einsum(
        "...ij,ik->...jk", positions_detector_frame_rescaled, unit_wavevector
    )
    exponent = jnp.einsum("i,...jk->...ijk", -1j * x_vector, scalar)
    return jnp.exp(exponent)
