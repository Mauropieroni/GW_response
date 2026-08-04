# Global imports
import jax
import jax.numpy as jnp
from typing import Any, Callable, TYPE_CHECKING

from jax.typing import ArrayLike

# Local imports
from gw_response.constants import BasisTransformations, PhysicalConstants
from gw_response.utils import arm_length_exponential
from gw_response.space_based.single_link_geometry import (
    _SINGLE_LINK_ARM_LABELS,
    single_link_response_delay_td,
    single_link_response_segmented_td,
)

if TYPE_CHECKING:
    from gw_response.detector import Detector

# single_link_geometry imports response_utils (for contract_with_h), which
# itself imports this module (for build_tdi) -- importing
# single_link_response_delay_td/_SINGLE_LINK_ARM_LABELS at module level here
# would close that into a circular import, so the functions below that need
# them import lazily instead.

# Update jax configuration to enable 64-bit precision for numerical computations
jax.config.update("jax_enable_x64", True)


@jax.jit
def sin_factors(arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike) -> jax.Array:
    """
    Computes the sine factors used to build the single-arm Time Delay Interferometry
    (TDI) combinations from the (rescaled) arm lengths.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: Complex sine-factor array, with shape (configurations, x_vector, arms
            / 2 (3)), obtained by averaging each arm with its reverse-direction
            counterpart (e.g. 12 with 21).
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
    Builds the matrix projecting single-link responses onto the first-generation
    Michelson TDI variables X, Y, Z: Hartwig, Lilley, Muratore & Pieroni
    (arXiv:2303.15929) eq. 2.24a's
    ``X = (1-D13D31)(η12+D12η21) + (D12D21-1)(η13+D13η31)`` (Y, Z cyclic
    permutations), as a frequency-domain projection matrix -- each delay operator
    ``D_ij`` replaced by ``e^{-i2πfL_ij}`` per eq. 2.27's own prescription for
    reading off ``c^V_ij`` this way.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The XYZ TDI projection matrix, with shape (configurations, x_vector,
            TDI (3), arms (6)).
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
    Builds the matrix projecting single-link responses onto the zeta ("Sagnac-like")
    symmetric TDI combination: Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929)
    eq. 2.24c's ``ζ = D12(η31-η32) + D23(η12-η13) + D31(η23-η21)`` (algebraically
    identical to the ``D_ji``-labeled form in the comment below, under the reciprocal-
    arm ``D_ij = D_ji`` this module assumes), as a frequency-domain projection matrix
    (see :func:`tdi_XYZ_matrix` regarding the ``D_ij -> e^{-i2πfL_ij}`` replacement).

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The zeta TDI projection matrix, with shape (configurations, x_vector,
            TDI (1), arms (6)).
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
    Builds the matrix projecting single-link responses onto the (first-generation)
    Sagnac TDI variables alpha, beta, gamma: Hartwig, Lilley, Muratore & Pieroni
    (arXiv:2303.15929) eq. 2.24b (quoted verbatim in the comment below), as a
    frequency-domain projection matrix (see :func:`tdi_XYZ_matrix` regarding the
    ``D_ij -> e^{-i2πfL_ij}`` replacement).

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The Sagnac TDI projection matrix, with shape (configurations,
            x_vector, TDI (3), arms (6)).
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
    Builds the matrix projecting single-link responses onto the A, E, T TDI variables,
    obtained by rotating the XYZ TDI basis: Hartwig, Lilley, Muratore & Pieroni
    (arXiv:2303.15929) eq. 2.26's ``A = (Z-X)/√2``, ``E = (X-2Y+Z)/√6``,
    ``T = (X+Y+Z)/√3`` (see :class:`gw_response.constants.BasisTransformations` for the
    exact matrix).

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The AET TDI projection matrix, with shape (configurations, x_vector,
            TDI (3), arms (6)).
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
    Builds the matrix projecting single-link responses onto the A, E, T TDI variables
    built from the Sagnac (rather than Michelson) combinations: Hartwig, Lilley,
    Muratore & Pieroni (arXiv:2303.15929) eq. 2.25's ``𝒜 = (γ-α)/√2``,
    ``ℰ = (α-2β+γ)/√6``, ``𝒯 = (α+β+γ)/√3`` -- structurally the same rotation matrix as
    :func:`tdi_AET_matrix`'s eq. 2.26, applied to the Sagnac base variables instead of
    the Michelson ones.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

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
    Builds the matrix projecting single-link responses onto the A, E TDI variables
    together with the (Sagnac-like) zeta combination -- this paper's own "AEζ" basis
    (Hartwig, Lilley, Muratore & Pieroni, arXiv:2303.15929, Sec. II B 1), combining
    :func:`tdi_AET_matrix`'s eq. 2.26 A/E with :func:`tdi_zeta_matrix`'s eq. 2.24c ζ.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The A, E, zeta TDI projection matrix, with shape (configurations,
            x_vector, TDI (3), arms (6)).
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
    Builds the matrix projecting single-link responses onto the Sagnac-based A, E TDI
    variables together with the zeta combination -- this paper's own "𝒜ℰζ" basis
    (Hartwig, Lilley, Muratore & Pieroni, arXiv:2303.15929, Sec. II B 1), combining
    :func:`tdi_AET_Sagnac_matrix`'s eq. 2.25 𝒜/ℰ with :func:`tdi_zeta_matrix`'s eq.
    2.24c ζ.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The Sagnac-based A, E, zeta TDI projection matrix, with shape
            (configurations, x_vector, TDI (3), arms (6)).
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
        TDI_idx (ArrayLike): Index into :data:`TDI_map` (and :data:`tdi_fun_list`)
            selecting the TDI combination to build.
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The TDI projection matrix for the selected combination, with shape
            (configurations, x_vector, TDI (3), arms (6)).
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
    Projects a single-link response onto the requested TDI combination: Hartwig,
    Lilley, Muratore & Pieroni (arXiv:2303.15929) eq. 2.27's
    ``V(f) = Σ c^V_ij η_ij(f)``.

    This mirrors :func:`gw_response.response_utils.linear_response_angular`, exposed
    here for convenience when only TDI-related quantities are needed.

    Args:
        TDI_idx (ArrayLike): Index into :data:`TDI_map` (and :data:`tdi_fun_list`)
            selecting the TDI combination to project onto.
        single_link (ArrayLike): Single-link strain response, with shape
            (configurations, x_vector, arms, pixels).
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The linear TDI response, with shape (configurations, x_vector, TDI,
            pixels).
    """
    # tdi_mat has shape configuration, x_vector, TDI, arms
    tdi_mat = tdi_matrix(TDI_idx, arms_matrix_rescaled, x_vector)

    # single_link is configuration, x_vector, arms, and optionally a
    # trailing pixels axis if it hasn't been integrated over the sky yet.
    # The trailing "..." picks up that pixels axis when present and
    # contributes nothing when it isn't, so this one contraction handles
    # both shapes without branching (same idiom as the leading "..." used
    # for the configuration axis elsewhere in this module).
    return jnp.einsum("cijk,cik...->cij...", tdi_mat, single_link)


# TDI 1.5 (unequal but locally-constant arms) time-domain combination
# formulas, from Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929), eq. (2.24),
# each expressed as a tuple of (sign, arm_label, delay_arm_labels) terms: a
# term contributes ``sign * eta_{arm_label}(t - sum(ltt[d] for d in
# delay_arm_labels))``, i.e. a delay operator "D_ij" becomes a time shift by
# arm ij's own (current) light-travel-time, composed by summing when several
# delays are nested. `Y`/`Z` and `beta`/`gamma` are cyclic satellite
# permutations of `X`/`alpha` (see :func:`_cyclic_permute_terms`); `zeta` (the
# fully symmetric Sagnac combination) has no such siblings. A genuinely
# evolving geometry -- unlike the frequency-domain `tdi_XYZ_matrix`/
# `tdi_Sagnac_matrix`, which assume one arm length per (undirected) arm pair
# -- means `X` here can differ from those by using the arm's own two
# (possibly unequal) directional light-travel-times directly, since it's
# built from exact per-arm data via
# :func:`gw_response.space_based.single_link_geometry.single_link_response_delay_td`.
_X_TERMS = (
    (1, 12, ()),
    (1, 21, (12,)),
    (-1, 12, (13, 31)),
    (-1, 21, (13, 31, 12)),
    (1, 13, (12, 21)),
    (1, 31, (12, 21, 13)),
    (-1, 13, ()),
    (-1, 31, (13,)),
)
_ALPHA_TERMS = (
    (1, 12, ()),
    (1, 23, (12,)),
    (1, 31, (12, 23)),
    (-1, 13, ()),
    (-1, 32, (13,)),
    (-1, 21, (13, 32)),
)
_ZETA_TERMS = (
    (1, 31, (12,)),
    (-1, 32, (12,)),
    (1, 12, (23,)),
    (-1, 13, (23,)),
    (1, 23, (31,)),
    (-1, 21, (31,)),
)

_CYCLIC_SATELLITE = {1: 2, 2: 3, 3: 1}


def _relabel_satellite(label: int, shift: int) -> int:
    """Cyclically relabels satellites (1->2->3->1, applied `shift` times) in a 2-digit
    arm label."""
    d1, d2 = label // 10, label % 10
    for _ in range(shift % 3):
        d1, d2 = _CYCLIC_SATELLITE[d1], _CYCLIC_SATELLITE[d2]
    return d1 * 10 + d2


def _cyclic_permute_terms(
    terms: tuple[tuple[int, int, tuple[int, ...]], ...], shift: int
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Cyclically relabels satellites (1->2->3->1, applied `shift` times) in every arm
    label of `terms` -- builds `Y`/`Z` from `X` (or `beta`/`gamma` from `alpha`)."""
    return tuple(
        (
            sign,
            _relabel_satellite(arm, shift),
            tuple(_relabel_satellite(d, shift) for d in delays),
        )
        for sign, arm, delays in terms
    )


# TDI 2.0 pre-factors (Hartwig, Lilley, Muratore & Pieroni, arXiv:2303.15929, eq. 2.23),
# each a tuple of (sign, delay_arm_labels) terms applied to the *already-built*
# TDI 1.5 channel (see :func:`_apply_tdi2_prefactor`): X2 = (1 - D31^2 D12^2) X,
# alpha2 = (1 - D12 D23 D31) alpha, zeta2 = (D31 - D12 D23) zeta. `Y2`/`Z2` and
# `beta2`/`gamma2` reuse the same cyclic relabeling as their 1.5-generation
# counterparts; `zeta2` has no such siblings.
_TDI2_PREFACTOR = {
    "X": ((1, ()), (-1, (31, 31, 12, 12))),
    "alpha": ((1, ()), (-1, (12, 23, 31))),
    "zeta": ((1, (31,)), (-1, (12, 23))),
}


def _apply_tdi2_prefactor(
    prefactor: tuple[tuple[int, tuple[int, ...]], ...],
    channel_1_5: Callable[[jax.Array], jax.Array],
    ps: PhysicalConstants,
    times_in_years: jax.Array,
    ltt_by_arm: dict[int, jax.Array],
) -> jax.Array:
    """
    Promotes an already-built TDI 1.5 channel (`channel_1_5`, a function of reception
    time) to TDI 2.0 by applying `prefactor`: for each term, shifts `times_in_years` by
    the term's own cumulative delay (built from `ltt_by_arm`, evaluated at the
    *unshifted* `times_in_years` -- the same "current geometry" convention every other
    delay in this module uses) and calls `channel_1_5` at the shifted times (which
    re-evaluates its own geometry fresh there, for a genuinely evolving-geometry inner
    computation), accumulating with the term's sign.
    """
    result = jnp.zeros_like(times_in_years)
    for sign, delay_labels in prefactor:
        if delay_labels:
            delay_seconds = sum(ltt_by_arm[label] for label in delay_labels)
            shifted_times = times_in_years - delay_seconds / ps.yr
        else:
            shifted_times = times_in_years
        result = result + sign * channel_1_5(shifted_times)
    return result


def _tdi_channel_delay_td(
    terms: tuple[tuple[int, int, tuple[int, ...]], ...],
    det: "Detector",
    ps: PhysicalConstants,
    times_in_years: jax.Array,
    theta: ArrayLike,
    phi: ArrayLike,
    strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
    waveform_params: Any,
    ltt_by_arm: dict[int, jax.Array],
) -> jax.Array:
    """
    Evaluates one TDI channel from `terms`: for each term, shifts `times_in_years` by
    the term's own cumulative delay (built from `ltt_by_arm`, each arm's current
    light-travel-time) and calls
    :func:`gw_response.space_based.single_link_geometry.single_link_response_delay_td`
    (reusing its exact per-arm evaluation and geometry), picking out just that term's
    arm and accumulating with its sign. `jnp.real` (with the library's own natural
    convention baked in) is applied per term inside `single_link_response_delay_td`;
    since `Re` is linear over the (real) signs summed here, this is exactly equivalent
    to combining the complex per-arm terms first.

    Returns:
        jax.Array: shape (time,).
    """
    channel = jnp.zeros_like(times_in_years)
    for sign, arm_label, delay_labels in terms:
        if delay_labels:
            delay_seconds = sum(ltt_by_arm[label] for label in delay_labels)
            shifted_times = times_in_years - delay_seconds / ps.yr
        else:
            shifted_times = times_in_years

        y_all_arms = single_link_response_delay_td(
            det,
            ps,
            shifted_times,
            theta,
            phi,
            strain_td,
            waveform_params,
        )
        arm_idx = _SINGLE_LINK_ARM_LABELS.index(arm_label)
        channel = channel + sign * y_all_arms[:, arm_idx]
    return channel


def _tdi_channel_segmented_td(
    terms: tuple[tuple[int, int, tuple[int, ...]], ...],
    det: "Detector",
    ps: PhysicalConstants,
    times_in_years: jax.Array,
    theta: ArrayLike,
    phi: ArrayLike,
    strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
    waveform_params: Any,
    segment_length: int,
    ltt_by_arm: dict[int, jax.Array],
) -> jax.Array:
    """
    Evaluates one TDI channel from `terms` via segment-stacking: for each term, shifts
    `times_in_years` by the term's own cumulative delay (built from `ltt_by_arm`, each
    arm's current light-travel-time) and calls `single_link_response_segmented_td`
    (in :mod:`gw_response.space_based.single_link_geometry`, reusing its segment-local
    linearized evaluation), picking out just that term's arm and accumulating with its
    sign -- the segmented analog of :func:`_tdi_channel_delay_td`.

    Returns:
        jax.Array: shape (time,).
    """
    channel = jnp.zeros_like(times_in_years)
    for sign, arm_label, delay_labels in terms:
        if delay_labels:
            delay_seconds = sum(ltt_by_arm[label] for label in delay_labels)
            shifted_times = times_in_years - delay_seconds / ps.yr
        else:
            shifted_times = times_in_years

        y_all_arms = single_link_response_segmented_td(
            det,
            ps,
            shifted_times,
            theta,
            phi,
            strain_td,
            waveform_params,
            segment_length,
        )
        arm_idx = _SINGLE_LINK_ARM_LABELS.index(arm_label)
        channel = channel + sign * y_all_arms[:, arm_idx]
    return channel


def tdi_response_segmented_td(
    det: "Detector",
    ps: PhysicalConstants,
    times_in_years: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
    waveform_params: Any,
    segment_length: int,
    combination: str = "XYZ",
) -> jax.Array:
    """
    TDI 1.5 (unequal but locally-constant arms) time-domain response for a LISA-like
    constellation, via segment-stacking -- reuses `single_link_response_segmented_td`
    (in :mod:`gw_response.space_based.single_link_geometry`) for each combination
    term's own (shifted) segment-local evaluation, the same delay-operator term tables
    as :func:`tdi_response_delay_td` (see the module-level comments above
    :data:`_X_TERMS`), and :meth:`gw_response.detector.Detector.detector_arms_retarded`
    for the light-travel-times the delay operators need. TDI 2.0 isn't supported here
    (its nested-delay prefactor re-evaluates the 1.5-generation channel at genuinely
    different times, which doesn't mesh with segment-stacking's fixed segment grid).
    Backs ``Response.get_response_segmented_td``.

    Args:
        det, ps, theta, phi, strain_td, waveform_params: see
            :func:`tdi_response_delay_td`.
        times_in_years (ArrayLike): Reception time(s), in years, uniformly spaced.
        segment_length (int): Number of samples per segment; must evenly divide
            `times_in_years`'s length -- see `single_link_response_segmented_td` (in
            :mod:`gw_response.space_based.single_link_geometry`).
        combination (str, optional): One of "XYZ", "AET", "Sagnac", "AET_Sagnac",
            "AE_zeta", "AE_Sagnac_zeta" (matching :data:`TDI_map`'s keys). Default is
            "XYZ".

    Returns:
        jax.Array: The real TDI-combined time-domain response, with shape (time,
            channels=3).

    Raises:
        ValueError: If `combination` isn't one of the supported values.
    """
    times_in_years = jnp.atleast_1d(times_in_years)
    _, ltt, _ = det.detector_arms_retarded(times_in_years, ps)  # (time, arms)
    ltt_by_arm = {label: ltt[:, i] for i, label in enumerate(_SINGLE_LINK_ARM_LABELS)}

    def channel_group(base_terms: tuple) -> jax.Array:
        channels = [
            _tdi_channel_segmented_td(
                _cyclic_permute_terms(base_terms, shift),
                det,
                ps,
                times_in_years,
                theta,
                phi,
                strain_td,
                waveform_params,
                segment_length,
                ltt_by_arm,
            )
            for shift in range(3)
        ]
        return jnp.stack(channels, axis=0)  # (3, time)

    def zeta_channel() -> jax.Array:
        return _tdi_channel_segmented_td(
            _ZETA_TERMS,
            det,
            ps,
            times_in_years,
            theta,
            phi,
            strain_td,
            waveform_params,
            segment_length,
            ltt_by_arm,
        )

    xyz_to_aet = BasisTransformations().XYZ_to_AET

    if combination in ("XYZ", "AET", "AE_zeta"):
        xyz = channel_group(_X_TERMS)
        if combination == "XYZ":
            result = xyz
        else:
            aet = xyz_to_aet @ xyz
            result = (
                aet
                if combination == "AET"
                else jnp.concatenate([aet[:2], zeta_channel()[None]], axis=0)
            )
    elif combination in ("Sagnac", "AET_Sagnac", "AE_Sagnac_zeta"):
        sagnac = channel_group(_ALPHA_TERMS)
        if combination == "Sagnac":
            result = sagnac
        else:
            aet_sagnac = xyz_to_aet @ sagnac
            result = (
                aet_sagnac
                if combination == "AET_Sagnac"
                else jnp.concatenate([aet_sagnac[:2], zeta_channel()[None]], axis=0)
            )
    else:
        raise ValueError(
            f"Unknown TDI combination '{combination}'; expected one of 'XYZ', "
            "'AET', 'Sagnac', 'AET_Sagnac', 'AE_zeta', 'AE_Sagnac_zeta'."
        )

    return jnp.moveaxis(result, -1, 0)  # (time, channels=3)


def tdi_response_delay_td(
    det: "Detector",
    ps: PhysicalConstants,
    times_in_years: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
    waveform_params: Any,
    combination: str = "XYZ",
    tdi_order: float = 1.5,
) -> jax.Array:
    """
    TDI 1.5 or 2.0 (unequal, and for 2.0 also evolving-during-the-nested-delays, arms)
    time-domain response for a LISA-like constellation, computed exactly for genuinely
    evolving geometry -- reuses
    :func:`gw_response.space_based.single_link_geometry.single_link_response_delay_td`
    for each combination term's own (shifted) exact single-link evaluation and
    :meth:`gw_response.detector.Detector.detector_arms_retarded` for the
    light-travel-times the delay operators need, rather than any new geometry code.
    `tdi_order=2.0` additionally reuses the 1.5-generation channel itself (see
    :func:`_apply_tdi2_prefactor`), rather than any new per-arm derivation. See the
    module-level comments above :data:`_X_TERMS`/:data:`_TDI2_PREFACTOR` for the
    delay-operator convention and the Hartwig, Lilley, Muratore & Pieroni reference
    (arXiv:2303.15929, eqs. 2.24 and 2.23) this implements. Backs
    ``Response.get_response_delay_td``.

    Args:
        det (Detector): The detector (e.g. LISA, Taiji) the response is computed for.
        ps (PhysicalConstants): Physical constants used to convert between distance and
            time units.
        times_in_years (ArrayLike): Reception time(s), in years, at which to evaluate
            the response.
        theta (ArrayLike): Colatitude of the single sky position the signal arrives
            from, in radians.
        phi (ArrayLike): Longitude of the single sky position the signal arrives from,
            in radians.
        strain_td (Callable): Maps a time, in seconds, and `waveform_params` to the
            complex ``(h_plus, h_cross)`` quadratures at that time -- see
            :class:`gw_response.response_utils.Waveform`.
        waveform_params (Any): Source parameters passed through to `strain_td`.
        combination (str, optional): One of "XYZ", "AET", "Sagnac", "AET_Sagnac",
            "AE_zeta", "AE_Sagnac_zeta" (matching :data:`TDI_map`'s keys). Default is
            "XYZ".
        tdi_order (float, optional): 1.5 or 2.0. Default is 1.5.

    Returns:
        jax.Array: The real TDI-combined time-domain response, with shape (time,
            channels=3). Uses the library's own natural convention internally, same as
            `single_link_response_delay_td` -- see there.

    Raises:
        ValueError: If `combination` or `tdi_order` isn't one of the supported values.
    """
    from gw_response.space_based.single_link_geometry import _SINGLE_LINK_ARM_LABELS

    if tdi_order not in (1.5, 2.0):
        raise ValueError(f"Unsupported tdi_order '{tdi_order}'; expected 1.5 or 2.0.")

    times_in_years = jnp.atleast_1d(times_in_years)
    _, ltt, _ = det.detector_arms_retarded(times_in_years, ps)  # (time, arms)
    ltt_by_arm = {label: ltt[:, i] for i, label in enumerate(_SINGLE_LINK_ARM_LABELS)}

    def channel_1_5_at(
        terms: tuple, t: jax.Array, ltt_by_arm_t: dict[int, jax.Array]
    ) -> jax.Array:
        return _tdi_channel_delay_td(
            terms,
            det,
            ps,
            t,
            theta,
            phi,
            strain_td,
            waveform_params,
            ltt_by_arm_t,
        )

    def channel_group(base_terms: tuple, prefactor_name: str) -> jax.Array:
        channels = []
        for shift in range(3):
            terms = _cyclic_permute_terms(base_terms, shift)
            if tdi_order == 1.5:
                channels.append(channel_1_5_at(terms, times_in_years, ltt_by_arm))
                continue

            def channel_1_5(t: jax.Array, terms: tuple = terms) -> jax.Array:
                _, ltt_t, _ = det.detector_arms_retarded(t, ps)
                ltt_by_arm_t = {
                    label: ltt_t[:, i]
                    for i, label in enumerate(_SINGLE_LINK_ARM_LABELS)
                }
                return channel_1_5_at(terms, t, ltt_by_arm_t)

            prefactor = tuple(
                (sign, tuple(_relabel_satellite(d, shift) for d in delays))
                for sign, delays in _TDI2_PREFACTOR[prefactor_name]
            )
            channels.append(
                _apply_tdi2_prefactor(
                    prefactor, channel_1_5, ps, times_in_years, ltt_by_arm
                )
            )
        return jnp.stack(channels, axis=0)  # (3, time)

    def zeta_channel() -> jax.Array:
        if tdi_order == 1.5:
            return channel_1_5_at(_ZETA_TERMS, times_in_years, ltt_by_arm)

        def channel_1_5(t: jax.Array) -> jax.Array:
            _, ltt_t, _ = det.detector_arms_retarded(t, ps)
            ltt_by_arm_t = {
                label: ltt_t[:, i] for i, label in enumerate(_SINGLE_LINK_ARM_LABELS)
            }
            return channel_1_5_at(_ZETA_TERMS, t, ltt_by_arm_t)

        return _apply_tdi2_prefactor(
            _TDI2_PREFACTOR["zeta"], channel_1_5, ps, times_in_years, ltt_by_arm
        )  # (time,)

    xyz_to_aet = BasisTransformations().XYZ_to_AET

    if combination in ("XYZ", "AET", "AE_zeta"):
        xyz = channel_group(_X_TERMS, "X")
        if combination == "XYZ":
            result = xyz
        else:
            aet = xyz_to_aet @ xyz
            result = (
                aet
                if combination == "AET"
                else jnp.concatenate([aet[:2], zeta_channel()[None]], axis=0)
            )
    elif combination in ("Sagnac", "AET_Sagnac", "AE_Sagnac_zeta"):
        sagnac = channel_group(_ALPHA_TERMS, "alpha")
        if combination == "Sagnac":
            result = sagnac
        else:
            aet_sagnac = xyz_to_aet @ sagnac
            result = (
                aet_sagnac
                if combination == "AET_Sagnac"
                else jnp.concatenate([aet_sagnac[:2], zeta_channel()[None]], axis=0)
            )
    else:
        raise ValueError(
            f"Unknown TDI combination '{combination}'; expected one of 'XYZ', "
            "'AET', 'Sagnac', 'AET_Sagnac', 'AE_zeta', 'AE_Sagnac_zeta'."
        )

    return jnp.moveaxis(result, -1, 0)  # (time, channels=3)
