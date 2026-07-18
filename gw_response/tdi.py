# Global imports

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

# Local imports
from .constants import BasisTransformations
from .utils import arm_length_exponential

# Update jax configuration to enable 64-bit precision for numerical computations
jax.config.update("jax_enable_x64", True)


@jax.jit
def sin_factors(arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike) -> jax.Array:
    """
    Computes the sine factors used to build the single-arm Time Delay
    Interferometry (TDI) combinations from the (rescaled) arm lengths.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)). Arms are ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: Complex sine-factor array, with shape (configurations,
            x_vector, arms / 2 (3)), obtained by averaging each arm with its
            reverse-direction counterpart (e.g. 12 with 21).
    """
    # arms_matrix_rescaled is configurations, vectorial_index, arms
    # arm_lengths has shape configurations, arms
    # arms ordered as 2-1, 3-2, 1-3, 2-1, 2-3, 3-1
    arm_lengths = jnp.sqrt(
        jnp.sqrt(
            jnp.einsum("...ij,...ij->...j", arms_matrix_rescaled, arms_matrix_rescaled)
        )
    )

    # xij is configurations, x_vector, arms
    xij = jnp.einsum("i,...j->...ij", x_vector, arm_lengths)

    # This is averaging ij, ji
    single_arm_mean = (xij + jnp.roll(xij, 3, axis=-1)) / 2

    # the output is configurations, x_vector, arms / 2
    return 2j * jnp.sin(single_arm_mean) * jnp.exp(-1j * single_arm_mean)


@jax.jit
def tdi_XYZ_matrix(arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike) -> jax.Array:
    """
    Builds the matrix projecting single-link responses onto the first-
    generation Michelson TDI variables X, Y, Z.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)). Arms are ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The XYZ TDI projection matrix, with shape
            (configurations, x_vector, TDI (3), arms (6)).
    """
    # this guy will be configurations, x_vector, arms
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)

    # this guy will be configurations, x_vector, arms
    sin_fac = sin_factors(arms_matrix_rescaled, x_vector)

    # With this roll 12, 23, 31 --> 31, 12, 23
    permuted_sin_fac = jnp.roll(sin_fac, 1, axis=-1)

    # This would be a diag matrix on the last 2 indexes,
    # the shape is configurations, x_vector, TDI, arms (now 3 not 6!!)
    t_retarded_coeffs = jnp.einsum(
        "ij,...kj->...kij", jnp.identity(3), t_retarded_factor[..., :3]
    )

    # This would be a diag matrix on the last 2 indexes,
    # the shape is configurations, x_vector, TDI, arms (now 3 not 6!!)
    flipped_t_retarded_coeffs = jnp.einsum(
        "ij,...kj->...kij", jnp.identity(3), t_retarded_factor[..., 3:]
    )

    # This takes ij + retarded (using ij) ji
    # this guy will be configurations, x_vector, TDI, arms (back to 6!)
    ones = jnp.ones_like(t_retarded_coeffs)
    identity = jnp.einsum("ij,...lij->...lij", jnp.identity(3), ones)

    # To test if the next 2 things are equal !!!!
    # rolled_identity1 = jnp.einsum(
    #    "ij,...lij->...lij", jnp.roll(jnp.identity(3), 1, axis=-2), ones
    # )
    rolled_identity = jnp.roll(identity, 1, axis=-2)

    single_arm = jnp.concatenate((identity, t_retarded_coeffs), axis=-1)

    # This takes ji + retarded (using ji) ij
    # this guy will be configurations, x_vector, TDI, arms (back to 6!)
    flipped_single_arm = jnp.concatenate(
        (jnp.roll(flipped_t_retarded_coeffs, 1, axis=-2), rolled_identity),
        axis=-1,
    )

    # These two guys are configurations, x_vector, TDI, arms (back to 6!)
    retarded_arm1 = jnp.einsum("...ik,...ijk->...ijk", permuted_sin_fac, single_arm)

    retarded_arm2 = jnp.einsum("...ik,...ijk->...ijk", sin_fac, flipped_single_arm)

    # the output has shape configurations, x_vector, TDI_indexes, arms
    return retarded_arm1 - retarded_arm2


@jax.jit
def tdi_zeta_matrix(arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike) -> jax.Array:
    """
    Builds the matrix projecting single-link responses onto the zeta
    ("Sagnac-like") symmetric TDI combination.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)). Arms are ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The zeta TDI projection matrix, with shape
            (configurations, x_vector, TDI (1), arms (6)).
    """
    # zeta is (D21 η31− D31 η21) +(D32 η12 −D12 η32) +(D13 η23 −D23 η13)

    # This guy will be configurations, x_vector, arms
    # arms are ordered as (12, 23, 31, 21, 32, 13)
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)

    # This will be configurations, x_vector, arms (just 3 arms) 32, 13, 21
    plus_terms = jnp.einsum(
        "i,...ki->...ki", jnp.ones(3), jnp.roll(t_retarded_factor[..., 3:], -1, axis=-1)
    )

    # This will be configurations, x_vector, arms (just 3 arms) 31, 12, 23
    minus_terms = jnp.einsum(
        "i,...ki->...ki", -jnp.ones(3), jnp.roll(t_retarded_factor[..., :3], 1, axis=-1)
    )

    # This guy will be configurations, x_vector, TDI, arms
    return jnp.concatenate((plus_terms, minus_terms), axis=-1)[..., jnp.newaxis, :]


@jax.jit
def tdi_Sagnac_matrix(
    arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike
) -> jax.Array:
    """
    Builds the matrix projecting single-link responses onto the (first-
    generation) Sagnac TDI variables alpha, beta, gamma.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)). Arms are ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The Sagnac TDI projection matrix, with shape
            (configurations, x_vector, TDI (3), arms (6)).
    """
    # This is configurations, x_vector, arms
    # arms are ordered as (12, 23, 31, 21, 32, 13)
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)

    first_three_arms = t_retarded_factor[..., :3]
    permuted_first_three = jnp.roll(first_three_arms, 1, axis=-1)

    flipped_arms = t_retarded_factor[..., 3:]
    permuted_flipped_arms = jnp.roll(flipped_arms, 1, axis=-1)
    permuted_two_flipped_arms = jnp.roll(flipped_arms, 2, axis=-1)

    ones = jnp.ones_like(first_three_arms)

    # These will have shapes configuration, x_vector, TDI, arms (just 3 here)
    identity = jnp.einsum("ij,...lj->...lij", jnp.identity(3), ones)
    rolled_identity = jnp.roll(identity, 1, axis=-2)
    rolled_two_identity = jnp.roll(identity, 2, axis=-2)

    # α = η12 + D12η23 + D12D23η31 − (η13 + D13η32 + D13D32η21)
    # The other 2 are cyclic permutations
    # Term 1 builds the plus part, term 2 the - part

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
        + jnp.einsum("...ijk,...ij->...ijk", rolled_identity, permuted_two_flipped_arms)
        + jnp.einsum(
            "...ijk,...ij->...ijk",
            identity,
            permuted_two_flipped_arms * permuted_flipped_arms,
        )
    )

    # The output is configuration, x_vector, TDI, arms (just 3 here)
    return jnp.concatenate((term1, -term2), axis=-1)


@jax.jit
def tdi_AET_matrix(arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike) -> jax.Array:
    """
    Builds the matrix projecting single-link responses onto the A, E, T TDI
    variables, obtained by rotating the XYZ TDI basis.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)). Arms are ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The AET TDI projection matrix, with shape
            (configurations, x_vector, TDI (3), arms (6)).
    """
    # tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_XYZ_matrix(arms_matrix_rescaled, x_vector)

    # XYZ_to_AET is TDI TDI, we have to rotate the TDI index
    return jnp.einsum("jk,...ikl->...ijl", BasisTransformations().XYZ_to_AET, tdi_mat)


@jax.jit
def tdi_AET_Sagnac_matrix(
    arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike
) -> jax.Array:
    """
    Builds the matrix projecting single-link responses onto the A, E, T TDI
    variables built from the Sagnac (rather than Michelson) combinations.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)). Arms are ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The Sagnac-based AET TDI projection matrix, with shape
            (configurations, x_vector, TDI (3), arms (6)).
    """
    # tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_Sagnac_matrix(arms_matrix_rescaled, x_vector)

    # XYZ_to_AET is TDI TDI, we have to rotate the TDI index
    return jnp.einsum("jk,...ikl->...ijl", BasisTransformations().XYZ_to_AET, tdi_mat)


@jax.jit
def tdi_AE_zeta_matrix(
    arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike
) -> jax.Array:
    """
    Builds the matrix projecting single-link responses onto the A, E TDI
    variables together with the (Sagnac-like) zeta combination.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)). Arms are ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The A, E, zeta TDI projection matrix, with shape
            (configurations, x_vector, TDI (3), arms (6)).
    """
    # tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat_AET = tdi_AET_matrix(arms_matrix_rescaled, x_vector)

    # zeta has shape configuration, x_vector, TDI, arms
    zeta = tdi_zeta_matrix(arms_matrix_rescaled, x_vector)

    return jnp.concatenate((tdi_mat_AET[..., :2, :], zeta), axis=-2)


@jax.jit
def tdi_AE_Sagnac_zeta_matrix(
    arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike
) -> jax.Array:
    """
    Builds the matrix projecting single-link responses onto the Sagnac-based
    A, E TDI variables together with the zeta combination.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)). Arms are ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The Sagnac-based A, E, zeta TDI projection matrix, with
            shape (configurations, x_vector, TDI (3), arms (6)).
    """
    # tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat_AET = tdi_AET_Sagnac_matrix(arms_matrix_rescaled, x_vector)

    # zeta has shape configuration, x_vector, TDI, arms
    zeta = tdi_zeta_matrix(arms_matrix_rescaled, x_vector)

    return jnp.concatenate((tdi_mat_AET[..., :2, :], zeta), axis=-2)


# Maps a TDI combination name to its index in `tdi_fun_list` (used with
# `jax.lax.switch`, which requires a static/traced integer, not a string).
TDI_map: dict[str, int] = {
    "XYZ": 0,
    "AET": 1,
    "Sagnac": 2,
    "AET_Sagnac": 3,
    "AE_zeta": 4,
    "AE_Sagnac_zeta": 5,
}

# The TDI projection-matrix function for each entry of `TDI_map`, indexed in
# the same order.
tdi_fun_list: list = [
    tdi_XYZ_matrix,
    tdi_AET_matrix,
    tdi_Sagnac_matrix,
    tdi_AET_Sagnac_matrix,
    tdi_AE_zeta_matrix,
    tdi_AE_Sagnac_zeta_matrix,
]

# Human-readable (LaTeX-friendly) labels for the 3 TDI channels of each
# combination in `TDI_map`, e.g. for use in plot legends.
TDI_labels: dict[str, list[str]] = {
    "XYZ": ["XX", "YY", "ZZ"],
    "AET": ["AA", "EE", "TT"],
    "Sagnac": [r"$\alpha \alpha$", r"$\beta \beta$", r"$\gamma \gamma$"],
    "AET_Sagnac": [r"$\mathcal{AA}$", r"$\mathcal{EE}$", r"$\mathcal{TT}$"],
    "AE_zeta": ["AA", "EE", r"$\zeta \zeta$"],
    "AE_Sagnac_zeta": [r"$\mathcal{AA}$", r"$\mathcal{EE}$", r"$\zeta \zeta$"],
}


@jax.jit
def tdi_matrix(
    TDI_idx: ArrayLike, arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike
) -> jax.Array:
    """
    Dispatches to the projection matrix for the requested TDI combination.

    Args:
        TDI_idx (ArrayLike): Index into :data:`TDI_map` (and
            :data:`tdi_fun_list`) selecting the TDI combination to build.
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)). Arms are ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The TDI projection matrix for the selected combination,
            with shape (configurations, x_vector, TDI (3), arms (6)).
    """
    return jax.lax.switch(TDI_idx, tdi_fun_list, arms_matrix_rescaled, x_vector)


@jax.jit
def build_tdi(
    TDI_idx: ArrayLike,
    single_link: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Projects a single-link response onto the requested TDI combination.

    This mirrors :func:`gw_response.single_link.linear_response_angular`,
    exposed here for convenience when only TDI-related quantities are
    needed.

    Args:
        TDI_idx (ArrayLike): Index into :data:`TDI_map` (and
            :data:`tdi_fun_list`) selecting the TDI combination to project onto.
        single_link (ArrayLike): Single-link strain response, with shape
            (configurations, x_vector, arms, pixels).
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by
            the arm length, with shape (configurations, vectorial_index (3),
            arms (6)). Arms are ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over
            frequency.

    Returns:
        jax.Array: The linear TDI response, with shape (configurations,
            x_vector, TDI, pixels).
    """
    # tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_matrix(TDI_idx, arms_matrix_rescaled, x_vector)

    # single_link has shape configuration, x_vector, arms, pixels

    # linear response is configuration, x_vector, TDI, pixels
    return jnp.einsum("...ijk,...ikl->...ijl", tdi_mat, single_link)
