# Global imports
import jax
import jax.numpy as jnp

# Update jax to use 64 bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def finite_arm_transfer_function(
    comb_plus: jax.Array, comb_minus: jax.Array, x_vector: jax.Array
) -> jax.Array:
    """
    Shared tail of :func:`gw_response.single_link_static.xi_k_no_G_static`/
    :func:`gw_response.single_link_retarded.xi_k_no_G_retarded`: the finite-arm-length
    sinc/phase factor, given each pipeline's own `comb_plus`/`comb_minus` (its
    light-travel-time-like term, plus/minus the wavevector dotted with the arm).

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
    the geometrical antenna-pattern factor of the single-link response. Shared by the
    static (:mod:`gw_response.single_link_static`) and retarded
    (:mod:`gw_response.single_link_retarded`) pipelines -- this contraction doesn't
    depend on the arm-length symmetry assumption either one makes.

    Args:
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)).
        polarization_tensor (jax.Array): Polarization tensor(s) as returned by e.g.
            :func:`gw_response.polarization.polarization_tensors_LR`, with shape
            (pixels, vectorial_index (3), vectorial_index (3)).

    Returns:
        jax.Array: The geometrical factor, with shape (configurations, arms, pixels).
    """
    # arms_matrix_rescaled is configurations, vectorial_index, arms
    # polarization_tensor is pixels, vectorial_index, vectorial_index

    arms_tensor = jnp.einsum(
        "...ik,...jk->...ijk", arms_matrix_rescaled, arms_matrix_rescaled / 2
    )

    # the output is configurations, arms, pixels
    return jnp.einsum("...ijk,...ijl->...kl", arms_tensor, polarization_tensor.T)


@jax.jit
def get_single_link_response_long_wavelength(
    polarization_tensor: jax.Array,
    arms_matrix_rescaled: jax.Array,
    x_vector: jax.Array,
) -> jax.Array:
    """
    Long-wavelength-limit single-link strain response: the antenna-pattern geometrical
    factor alone (see :func:`geometrical_factor`), with no
    finite-arm-length/light-travel-time correction -- the same simplification used by
    parameter-estimation codes like `gw_fast`/ `gw_fish`, valid when the signal's
    wavelength is much longer than the arm length. Broadcast to carry a
    (frequency-independent) x_vector axis so it stays shape-compatible with
    :func:`gw_response.single_link_static.get_single_link_response_static`/
    :func:`gw_response.single_link_retarded.get_single_link_response_retarded`.

    Args:
        polarization_tensor (jax.Array): Polarization tensor(s), with shape (pixels,
            vectorial_index (3), vectorial_index (3)).
        arms_matrix_rescaled (jax.Array): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms).
        x_vector (jax.Array): Vector of ``2 pi f L / c`` values over frequency; only its
            length is used, to broadcast the (frequency-independent) result to a
            matching shape.

    Returns:
        jax.Array: The single-link antenna-pattern response, with shape (configurations,
            x_vector, arms, pixels).
    """
    G = geometrical_factor(arms_matrix_rescaled, polarization_tensor)
    return jnp.broadcast_to(
        G[:, None, :, :], (G.shape[0], x_vector.shape[-1], *G.shape[1:])
    )


@jax.jit
def position_exponential(
    positions_detector_frame_rescaled: jax.Array,
    unit_wavevector: jax.Array,
    x_vector: jax.Array,
) -> jax.Array:
    """
    Computes the plane-wave phase factor picked up by each satellite due to its position
    relative to the detector-frame center. Shared by the static
    (:mod:`gw_response.single_link_static`) and retarded
    (:mod:`gw_response.single_link_retarded`) pipelines.

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

    # This is configurations, satellite, pixels
    scalar = jnp.einsum(
        "...ij,ik->...jk", positions_detector_frame_rescaled, unit_wavevector
    )
    exponent = jnp.einsum("i,...jk->...ijk", -1j * x_vector, scalar)

    # Output is configurations, x_vector, satellite, pixels
    return jnp.exp(exponent)
