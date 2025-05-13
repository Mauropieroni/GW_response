import jax

jax.config.update("jax_enable_x64", True)
# jax.checking_leaks = True

import jax.numpy as jnp

from .constants import BasisTransformations
from .utils import arm_length_exponential


@jax.jit
def sin_factors(arms_matrix_rescaled, x_vector):
    ### arms_matrix_rescaled is configurations, vectorial_index, arms
    ### arm_lengths has shape configurations, arms
    ### arms ordered as 2-1, 3-2, 1-3, 2-1, 2-3, 3-1
    arm_lengths = jnp.sqrt(
        jnp.sqrt(
            jnp.einsum("...ij,...ij->...j", arms_matrix_rescaled, arms_matrix_rescaled)
        )
    )

    ### xij is configurations, x_vector, arms
    xij = jnp.einsum("i,...j->...ij", x_vector, arm_lengths)

    ### This is averaging ij, ji
    single_arm_mean = (xij + jnp.roll(xij, 3, axis=-1)) / 2

    ### the output is configurations, x_vector, arms / 2
    return 2j * jnp.sin(single_arm_mean) * jnp.exp(-1j * single_arm_mean)


@jax.jit
def detector_output(arms_matrix_rescaled, x_vector):
    # this guy should be some linear algebra transformation with shape (1, 4)

    ### this guy will be configurations, x_vector, arms
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)

    ### this guy will be configurations, x_vector, arms
    sin_fac = sin_factors(arms_matrix_rescaled, x_vector)

    ### With this roll 12, 23, 31 --> 31, 12, 23
    permuted_sin_fac = jnp.roll(sin_fac, 1, axis=-1)

    ### This would be a diag matrix on the last 2 indexes,
    ### the shape is configurations, x_vector, TDI, arms (now 3 not 6!!)
    t_retarded_coeffs = jnp.einsum(
        "ij,...kj->...kij", jnp.identity(3), t_retarded_factor[..., :3]
    )

    ### This would be a diag matrix on the last 2 indexes,
    ### the shape is configurations, x_vector, TDI, arms (now 3 not 6!!)
    flipped_t_retarded_coeffs = jnp.einsum(
        "ij,...kj->...kij", jnp.identity(3), t_retarded_factor[..., 3:]
    )

    ### This takes ij + retarded (using ij) ji
    ### this guy will be configurations, x_vector, TDI, arms (back to 6!)
    ones = jnp.ones_like(t_retarded_coeffs)
    identity = jnp.einsum("ij,...lij->...lij", jnp.identity(3), ones)

    ### To test if the next 2 things are equal !!!!
    # rolled_identity1 = jnp.einsum(
    #    "ij,...lij->...lij", jnp.roll(jnp.identity(3), 1, axis=-2), ones
    # )
    rolled_identity = jnp.roll(identity, 1, axis=-2)

    single_arm = jnp.concatenate((identity, t_retarded_coeffs), axis=-1)

    ### This takes ji + retarded (using ji) ij
    ### this guy will be configurations, x_vector, TDI, arms (back to 6!)
    flipped_single_arm = jnp.concatenate(
        (jnp.roll(flipped_t_retarded_coeffs, 1, axis=-2), rolled_identity),
        axis=-1,
    )

    ### These two guys are configurations, x_vector, TDI, arms (back to 6!)
    retarded_arm1 = jnp.einsum("...ik,...ijk->...ijk", permuted_sin_fac, single_arm)

    retarded_arm2 = jnp.einsum("...ik,...ijk->...ijk", sin_fac, flipped_single_arm)

    # the output has shape configurations, x_vector, TDI_indexes, arms
    return retarded_arm1 - retarded_arm2


@jax.jit
def build_datastream(TDI_idx, single_link, arms_matrix_rescaled, x_vector):
    ### tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = detector_output(TDI_idx, arms_matrix_rescaled, x_vector)

    ### single_link has shape configuration, x_vector, arms, pixels

    ### linear response is configuration, x_vector, TDI, pixels
    return jnp.einsum("...ijk,...ikl->...ijl", tdi_mat, single_link)
