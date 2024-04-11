import jax

jax.config.update("jax_enable_x64", True)
# jax.checking_leaks = True

import jax.numpy as jnp


from .constants import *
from .tdi import *


@jax.jit
def unit_vec(theta, phi):
    theta = jax.lax.cond(
        isinstance(theta, float),
        lambda theta: jnp.reshape(jnp.array([theta]), (-1,)),
        lambda theta: jnp.reshape(jnp.array(theta), (-1,)),
        theta,
    )
    phi = jax.lax.cond(
        isinstance(phi, float),
        lambda phi: jnp.reshape(jnp.array([phi]), (-1,)),
        lambda phi: jnp.reshape(jnp.array(phi), (-1,)),
        phi,
    )
    ### The output will be vectorial index, pixels
    return jnp.array(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ]
    )


@jax.jit
def uv_analytical(theta, phi):
    theta = jax.lax.cond(
        isinstance(theta, float),
        lambda theta: jnp.reshape(jnp.array([theta]), (-1,)),
        lambda theta: jnp.reshape(jnp.array(theta), (-1,)),
        theta,
    )
    phi = jax.lax.cond(
        isinstance(phi, float),
        lambda phi: jnp.reshape(jnp.array([phi]), (-1,)),
        lambda phi: jnp.reshape(jnp.array(phi), (-1,)),
        phi,
    )
    ### The output will be pixels, vectorial index
    return (
        jnp.array(
            [
                jnp.cos(theta) * jnp.cos(phi),
                jnp.cos(theta) * jnp.sin(phi),
                -jnp.sin(theta),
            ]
        ).T,
        jnp.array([jnp.sin(phi), -jnp.cos(phi), 0.0 * phi]).T,
    )


@jax.jit
def polarization_vectors(u, v):
    ### The output will be pixels, vectorial index
    return (u - 1j * v) / jnp.sqrt(2), (u + 1j * v) / jnp.sqrt(2)


@jax.jit
def polarization_tensors_PC(u, v):
    e1p = jnp.einsum("...i,...j->...ij", u, u) - jnp.einsum("...i,...j->...ij", v, v)
    e1c = jnp.einsum("...i,...j->...ij", u, v) + jnp.einsum("...i,...j->...ij", v, u)

    ### The output will be pixels, vectorial index, vectorial index
    return e1p / jnp.sqrt(2), e1c / jnp.sqrt(2)


@jax.jit
def polarization_tensors_LR(u, v):
    first, second = polarization_vectors(u, v)
    e1L = jnp.einsum("...i,...j->...ij", first, first)
    e1R = jnp.einsum("...i,...j->...ij", second, second)

    ### The output will be pixels, vectorial index, vectorial index
    return e1L, e1R


@jax.jit
def xi_k_no_G(unit_wavevector, x_vector, arms_mat_rescaled):
    """x_vector is a vector over frequency (2 pi f L / c) unit_wavevector is
    vectorial index (3), pixels arms_mat_norm is (configurations (num time
    slices),) vectorial_index (3), arms (6)

    This returns an object with shape:
    configurations, x_vector, arms, pixels

    2.5 of the present draft without G
    """

    k_dot_arms = jnp.einsum("...ij,ik->...jk", arms_mat_rescaled, unit_wavevector)
    ### These guys will be configurations, arms, pixel
    comb_plus = 1 + k_dot_arms
    comb_minus = 1 - k_dot_arms

    ### These guys are  configurations, x_vector, arms, pixels
    prod_plus = jnp.einsum("i,...kl->...ikl", x_vector, comb_plus)
    prod_minus = jnp.einsum("i,...kl->...ikl", x_vector, comb_minus)

    ### These guys are configurations, x_vector, arms, pixels
    return jnp.exp(0.5j * prod_minus) * jnp.sinc(prod_plus / 2 / jnp.pi)


@jax.jit
def position_exponential(positions_detector_frame, unit_wavevector, x_vector):
    """x_vector is a vector unit_wavevector is vectorial index, pixels
    positions_detector_frame is configurations, vectorial_index, satellite (3)
    # Need shifted positions to get the numerical precision in the dot
    product."""

    ### This is configurations, satellite, pixels
    scalar = jnp.einsum("...ij,ik->...jk", positions_detector_frame, unit_wavevector)
    exponent = jnp.einsum("i,...jk->...ijk", -1j * x_vector, scalar)

    ### Output is configurations, x_vector, satellite, pixels
    return jnp.exp(exponent)


@jax.jit
def geometrical_factor(arms_matrix, polarization_tensor):
    #### arms_matrix is configurations, vectorial_index, arms
    #### polarization_tensor is pixels, vectorial_index, vectorial_index

    aux = jnp.einsum("...ik,...jk->...ijk", arms_matrix, arms_matrix / 2)

    ### aux is configurations, arms, vectorial_index, pixels
    # aux = jnp.einsum("...ij,ilk->...jlk", arms_matrix, polarization_tensor.T)

    ### the output is configurations, arms, pixels
    return jnp.einsum("...ijk,...ijl->...kl", aux, polarization_tensor.T)


@jax.jit
def xi_k_Avec_func(arms_matrix_rescaled, unit_wavevector, x_vector, geometrical):
    """
    Returns something which shape:
    configurations, x_vector, arms, pixels
    """

    ### xi_vec is configurations, x_vector, arms, pixels
    xi_vec = xi_k_no_G(unit_wavevector, x_vector, arms_matrix_rescaled)

    ### The output is configurations, x_vector, arms, pixels
    return jnp.einsum("...ijk,...jk->...ijk", xi_vec, geometrical)


@jax.jit
def single_link_response(
    positions, arms_matrix_rescaled, wavevector, x_vector, xi_k_Avec
):
    ### positions is configuration, vector, masses (masses are 1,2,3)
    all_position = jnp.concatenate(
        (positions, jnp.roll(positions, -1, axis=-1)), axis=-1
    )

    ### exp has shape configurations, x_vector, arms, pixels
    position_exp_factor = position_exponential(all_position, wavevector, x_vector)

    ### this guy will be configurations, x_vector, arms
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)

    ### this guy will be configurations, arms
    arm_lengths = jnp.sqrt(
        jnp.einsum("...ij,...ij->...j", arms_matrix_rescaled, arms_matrix_rescaled)
    )

    ### This will be configurations, x_vector, arms
    prefactor = jnp.einsum("...j,...ij->...ij", arm_lengths, t_retarded_factor)
    ### Need to pre-multiply by x to convert to fractional frequency in single
    ### link response
    prefactor = jnp.einsum("i,...ij->...ij", x_vector, prefactor)

    ### This will be configurations, x_vector, arms, pixels
    return jnp.einsum(
        "...ij,...ijk->...ijk", prefactor, position_exp_factor * xi_k_Avec
    )


@jax.jit
def get_single_link_response(
    polarization, arms_matrix_rescaled, wavevector, x_vector, positions
):
    ### This computes the geometrical factor
    geometrical = geometrical_factor(arms_matrix_rescaled, polarization)

    ### This computes the xi vectors
    xi_k_vec = xi_k_Avec_func(arms_matrix_rescaled, wavevector, x_vector, geometrical)

    ### This will be configurations, x_vector, arms, pixels
    return single_link_response(
        positions, arms_matrix_rescaled, wavevector, x_vector, xi_k_vec
    )


@jax.jit
def linear_response_angular(TDI_idx, single_link, arms_matrix_rescaled, x_vector):
    ### tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_matrix(TDI_idx, arms_matrix_rescaled, x_vector)

    ### single_link has shape configuration, x_vector, arms, pixels

    ### linear response is configuration, x_vector, TDI, pixels
    return jnp.einsum("...ijk,...ikl->...ijl", tdi_mat, single_link)


@jax.jit
def response_angular(TDI_idx, single_link, arms_matrix_rescaled, x_vector):
    ### linear response is configuration, x_vector, TDI, pixels
    linear_response = linear_response_angular(
        TDI_idx, single_link, arms_matrix_rescaled, x_vector
    )

    ### quadratic response is configuration, x_vector, TDI, TDI, pixels
    quadratic_response = jnp.einsum(
        "...ijl,...ikl->...ijkl",
        linear_response,
        jnp.conjugate(linear_response),
    )

    ### The first 2 is sum over polarization the second is for the h.c. sum
    return 2 * 2 * quadratic_response / jnp.pi / 4


@jax.jit
def integrand(
    TDI_idx,
    single_link,
    arms_matrix_rescaled,
    x_vector,
):
    ### Defines the integrand using the TDI factors
    return response_angular(TDI_idx, single_link, arms_matrix_rescaled, x_vector)


@jax.jit
def response_integrated(angular_response):
    return 4 * jnp.pi * jnp.mean(angular_response, axis=-1)
