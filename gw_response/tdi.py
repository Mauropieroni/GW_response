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
            jnp.einsum(
                "...ij,...ij->...j", arms_matrix_rescaled, arms_matrix_rescaled
            )
        )
    )

    ### xij is configurations, x_vector, arms
    xij = jnp.einsum("i,...j->...ij", x_vector, arm_lengths)

    ### This is averaging ij, ji
    single_arm_mean = (xij + jnp.roll(xij, 3, axis=-1)) / 2

    ### the output is configurations, x_vector, arms / 2
    return 2j * jnp.sin(single_arm_mean) * jnp.exp(-1j * single_arm_mean)


@jax.jit
def tdi_XYZ_matrix(arms_matrix_rescaled, x_vector):
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
    retarded_arm1 = jnp.einsum(
        "...ik,...ijk->...ijk", permuted_sin_fac, single_arm
    )

    retarded_arm2 = jnp.einsum(
        "...ik,...ijk->...ijk", sin_fac, flipped_single_arm
    )

    # the output has shape configurations, x_vector, TDI_indexes, arms
    return retarded_arm1 - retarded_arm2


@jax.jit
def tdi_zeta_matrix(arms_matrix_rescaled, x_vector):
    ### zeta is D12 (η31 − η32) + D23(η12 − η13) + D31(η23 − η21)

    ### This guy will be configurations, x_vector, arms / 2
    ### arms are ordered as (12, 23, 31, 21, 32, 13)
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)[
        ..., :3
    ]

    ### This is 3 x 6
    identities = jnp.concatenate(
        (
            jnp.roll(jnp.identity(3), -1, axis=-1),
            -jnp.roll(jnp.identity(3), 1, axis=-1),
        ),
        axis=-1,
    )

    ### This will be configurations, x_vector, arms / 2, arms
    to_sum = jnp.einsum("ij,...ki->...kij", identities, t_retarded_factor)

    return jnp.sum(to_sum, axis=-1)


@jax.jit
def tdi_Sagnac_matrix(arms_matrix_rescaled, x_vector):
    ### This is configurations, x_vector, arms
    ### arms are ordered as (12, 23, 31, 21, 32, 13)
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)

    first_three_arms = t_retarded_factor[..., :3]
    permuted_first_three = jnp.roll(first_three_arms, 1, axis=-1)

    flipped_arms = t_retarded_factor[..., 3:]
    permuted_flipped_arms = jnp.roll(flipped_arms, 1, axis=-1)
    permuted_two_flipped_arms = jnp.roll(flipped_arms, 2, axis=-1)

    ones = jnp.ones_like(first_three_arms)

    ### These will have shapes configuration, x_vector, TDI, arms (just 3 here)
    identity = jnp.einsum("ij,...lj->...lij", jnp.identity(3), ones)
    rolled_identity = jnp.roll(identity, 1, axis=-2)
    rolled_two_identity = jnp.roll(identity, 2, axis=-2)

    ### α = η12 + D12η23 + D12D23η31 − (η13 + D13η32 + D13D32η21)
    ### The other 2 are cyclic permutations
    ### Term 1 builds the plus part, term 2 the - part

    term1 = (
        identity
        + jnp.einsum("...ijk,...ij->...ijk", rolled_identity, first_three_arms)
        + jnp.einsum(
            "...ijk,...ij->...ijk",
            rolled_two_identity,
            first_three_arms * permuted_first_three,
        )
    )

    term2 = (
        rolled_two_identity
        + jnp.einsum(
            "...ijk,...ij->...ijk", rolled_identity, permuted_two_flipped_arms
        )
        + jnp.einsum(
            "...ijk,...ij->...ijk",
            identity,
            permuted_two_flipped_arms * permuted_flipped_arms,
        )
    )

    # The output is configuration, x_vector, TDI, arms (just 3 here)
    return jnp.concatenate((term1, -term2), axis=-1)


@jax.jit
def tdi_AET_matrix(arms_matrix_rescaled, x_vector):
    ### tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_XYZ_matrix(arms_matrix_rescaled, x_vector)

    ### XYZ_to_AET is TDI TDI, we have to rotate the TDI index
    return jnp.einsum(
        "jk,...ikl->...ijl", BasisTransformations().XYZ_to_AET, tdi_mat
    )


@jax.jit
def tdi_AET_Sagnac_matrix(arms_matrix_rescaled, x_vector):
    ### tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_Sagnac_matrix(arms_matrix_rescaled, x_vector)

    ### XYZ_to_AET is TDI TDI, we have to rotate the TDI index
    return jnp.einsum(
        "jk,...ikl->...ijl", BasisTransformations().XYZ_to_AET, tdi_mat
    )


@jax.jit
def tdi_AE_zeta_matrix(arms_matrix_rescaled, x_vector):
    ### tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_AET_matrix(arms_matrix_rescaled, x_vector)

    ### Change last column to zeta
    tdi_mat.at[:, :, 2, :].set(
        tdi_zeta_matrix(arms_matrix_rescaled, x_vector)[:, :, 0, :]
    )

    return tdi_mat


@jax.jit
def tdi_AE_Sagnac_zeta_matrix(arms_matrix_rescaled, x_vector):
    ### tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_AET_Sagnac_matrix(arms_matrix_rescaled, x_vector)

    ### Change last column to zeta
    tdi_mat.at[:, :, 2, :].set(
        tdi_zeta_matrix(arms_matrix_rescaled, x_vector)[:, :, 0, :]
    )
    return tdi_mat


TDI_map = {
    "XYZ": 0,
    "AET": 1,
    "Sagnac": 2,
    "AET_Sagnac": 3,
    # "AE_zeta": 4,
    # "AE_Sagnac_zeta": 5,
}

tdi_fun_list = [
    tdi_XYZ_matrix,
    tdi_AET_matrix,
    tdi_Sagnac_matrix,
    tdi_AET_Sagnac_matrix,
    # tdi_AE_zeta_matrix,
    # tdi_AE_Sagnac_zeta_matrix,
]


@jax.jit
def tdi_matrix(TDI_idx, arms_matrix_rescaled, x_vector):
    return jax.lax.switch(TDI_idx, tdi_fun_list, arms_matrix_rescaled, x_vector)


@jax.jit
def build_tdi(TDI_idx, single_link, arms_matrix_rescaled, x_vector):
    ### tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_matrix(TDI_idx, arms_matrix_rescaled, x_vector)

    ### single_link has shape configuration, x_vector, arms, pixels

    ### linear response is configuration, x_vector, TDI, pixels
    return jnp.einsum("...ijk,...ikl->...ijl", tdi_mat, single_link)
