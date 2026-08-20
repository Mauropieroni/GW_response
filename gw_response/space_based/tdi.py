# Global imports
import jax
import jax.numpy as jnp
from functools import partial
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
    arm_lengths = jnp.sqrt(
        jnp.sqrt(
            jnp.einsum("...ij,...ij->...j", arms_matrix_rescaled, arms_matrix_rescaled)
        )
    )
    xij = jnp.einsum("i,...j->...ij", x_vector, arm_lengths)
    # Average each arm (ij) with its reverse-direction counterpart (ji, 3 slots away
    # in the 12/23/31/21/32/13 ordering) -- see the Returns docstring above.
    single_arm_mean = (xij + jnp.roll(xij, 3, axis=-1)) / 2
    return 2j * jnp.sin(single_arm_mean) * jnp.exp(-1j * single_arm_mean)


@jax.jit
def tdi_XYZ_matrix(arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike) -> jax.Array:
    """
    Builds the matrix projecting single-link responses onto the first-generation
    Michelson TDI variables X, Y, Z: Hartwig, Lilley, Muratore & Pieroni
    (arXiv:2303.15929) eq. 2.24a's ``X = (1-D13D31)(η12+D12η21) +
    (D12D21-1)(η13+D13η31)`` (Y, Z cyclic permutations), as a frequency-domain
    projection matrix -- each delay operator ``D_ij`` replaced by ``e^{-i2πfL_ij}`` per
    eq. 2.27's own prescription for reading off ``c^V_ij`` this way.

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The XYZ TDI projection matrix, with shape (configurations, x_vector,
            TDI (3), arms (6)).
    """
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)
    sin_fac = sin_factors(arms_matrix_rescaled, x_vector)
    # 12, 23, 31 -> 31, 12, 23: aligns each TDI channel with the *other* arm in its
    # own Michelson term (X's own D12/D13, not D12/D12).
    permuted_sin_fac = jnp.roll(sin_fac, 1, axis=-1)

    # Diagonal in the TDI/arm indices (arms restricted to the first 3, 12/23/31).
    t_retarded_coeffs = jnp.einsum(
        "ij,...kj->...kij", jnp.identity(3), t_retarded_factor[..., :3]
    )
    # Same, for the reverse-direction arms (21/32/13).
    flipped_t_retarded_coeffs = jnp.einsum(
        "ij,...kj->...kij", jnp.identity(3), t_retarded_factor[..., 3:]
    )

    # eta_ij + D_ij eta_ji: identity term plus the ij-delayed ji term, concatenated
    # back to all 6 arms.
    ones = jnp.ones_like(t_retarded_coeffs)
    identity = jnp.einsum("ij,...lij->...lij", jnp.identity(3), ones)
    rolled_identity = jnp.roll(identity, 1, axis=-2)
    single_arm = jnp.concatenate((identity, t_retarded_coeffs), axis=-1)

    # eta_ji + D_ji eta_ij: the mirror term.
    flipped_single_arm = jnp.concatenate(
        (jnp.roll(flipped_t_retarded_coeffs, 1, axis=-2), rolled_identity),
        axis=-1,
    )

    retarded_arm1 = jnp.einsum("...ik,...ijk->...ijk", permuted_sin_fac, single_arm)
    retarded_arm2 = jnp.einsum("...ik,...ijk->...ijk", sin_fac, flipped_single_arm)
    return retarded_arm1 - retarded_arm2


@jax.jit
def tdi_zeta_matrix(arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike) -> jax.Array:
    """
    Builds the matrix projecting single-link responses onto the zeta ("Sagnac-like")
    symmetric TDI combination: Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929)
    eq. 2.24c's ``ζ = D12(η31-η32) + D23(η12-η13) + D31(η23-η21)`` (algebraically
    identical to the ``D_ji``-labeled form in the comment below, under the
    reciprocal-arm ``D_ij = D_ji`` this module assumes), as a frequency-domain
    projection matrix (see :func:`tdi_XYZ_matrix` regarding the ``D_ij ->
    e^{-i2πfL_ij}`` replacement).

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The zeta TDI projection matrix, with shape (configurations, x_vector,
            TDI (1), arms (6)).
    """
    # zeta is (D21 η31− D31 η21) +(D32 η12 −D12 η32) +(D13 η23 −D23 η13) -- the
    # D_ji-labeled form the class docstring above refers to.
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)

    plus_terms = jnp.einsum(  # D21, D32, D13 (the "+" arms, 32/13/21 after the roll)
        "i,...ki->...ki", jnp.ones(3), jnp.roll(t_retarded_factor[..., 3:], -1, axis=-1)
    )
    minus_terms = jnp.einsum(  # D31, D12, D23 (the "-" arms, 31/12/23 after the roll)
        "i,...ki->...ki", -jnp.ones(3), jnp.roll(t_retarded_factor[..., :3], 1, axis=-1)
    )
    return jnp.concatenate((plus_terms, minus_terms), axis=-1)[..., jnp.newaxis, :]


@jax.jit
def tdi_Sagnac_matrix(
    arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike
) -> jax.Array:
    """
    Builds the matrix projecting single-link responses onto the (first-generation)
    Sagnac TDI variables alpha, beta, gamma: Hartwig, Lilley, Muratore & Pieroni
    (arXiv:2303.15929) eq. 2.24b (quoted verbatim in the comment below), as a
    frequency-domain projection matrix (see :func:`tdi_XYZ_matrix` regarding the ``D_ij
    -> e^{-i2πfL_ij}`` replacement).

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The Sagnac TDI projection matrix, with shape (configurations,
            x_vector, TDI (3), arms (6)).
    """
    t_retarded_factor = arm_length_exponential(arms_matrix_rescaled, x_vector)

    first_three_arms = t_retarded_factor[..., :3]
    permuted_first_three = jnp.roll(first_three_arms, 1, axis=-1)

    flipped_arms = t_retarded_factor[..., 3:]
    permuted_flipped_arms = jnp.roll(flipped_arms, 1, axis=-1)
    permuted_two_flipped_arms = jnp.roll(flipped_arms, 2, axis=-1)

    ones = jnp.ones_like(first_three_arms)
    identity = jnp.einsum("ij,...lj->...lij", jnp.identity(3), ones)
    rolled_identity = jnp.roll(identity, 1, axis=-2)
    rolled_two_identity = jnp.roll(identity, 2, axis=-2)

    # α = η12 + D12η23 + D12D23η31 − (η13 + D13η32 + D13D32η21); beta/gamma are cyclic
    # permutations. term1 builds the "+" part, term2 the "-" part.
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

    return jnp.concatenate((term1, -term2), axis=-1)


@jax.jit
def tdi_AET_matrix(arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike) -> jax.Array:
    """
    Builds the matrix projecting single-link responses onto the A, E, T TDI variables,
    obtained by rotating the XYZ TDI basis: Hartwig, Lilley, Muratore & Pieroni
    (arXiv:2303.15929) eq. 2.26's ``A = (Z-X)/√2``, ``E = (X-2Y+Z)/√6``, ``T =
    (X+Y+Z)/√3`` (see :class:`gw_response.constants.BasisTransformations` for the exact
    matrix).

    Args:
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)). Arms are
            ordered as 12, 23, 31, 21, 32, 13.
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The AET TDI projection matrix, with shape (configurations, x_vector,
            TDI (3), arms (6)).
    """
    tdi_mat = tdi_XYZ_matrix(arms_matrix_rescaled, x_vector)
    # XYZ_to_AET (TDI x TDI) rotates tdi_mat's own TDI index, leaving arms untouched.
    return jnp.einsum("jk,...ikl->...ijl", BasisTransformations().XYZ_to_AET, tdi_mat)


@jax.jit
def tdi_AET_Sagnac_matrix(
    arms_matrix_rescaled: ArrayLike, x_vector: ArrayLike
) -> jax.Array:
    """
    Builds the matrix projecting single-link responses onto the A, E, T TDI variables
    built from the Sagnac (rather than Michelson) combinations: Hartwig, Lilley,
    Muratore & Pieroni (arXiv:2303.15929) eq. 2.25's ``𝒜 = (γ-α)/√2``, ``ℰ =
    (α-2β+γ)/√6``, ``𝒯 = (α+β+γ)/√3`` -- structurally the same rotation matrix as
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
    tdi_mat = tdi_Sagnac_matrix(arms_matrix_rescaled, x_vector)
    # Same rotation as tdi_AET_matrix, applied to the Sagnac base variables instead.
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
    tdi_mat_AET = tdi_AET_matrix(arms_matrix_rescaled, x_vector)
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
    tdi_mat_AET = tdi_AET_Sagnac_matrix(arms_matrix_rescaled, x_vector)
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
    Projects a single-link response onto the requested TDI combination: Hartwig, Lilley,
    Muratore & Pieroni (arXiv:2303.15929) eq. 2.27's ``V(f) = Σ c^V_ij η_ij(f)``.

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
    tdi_mat = tdi_matrix(TDI_idx, arms_matrix_rescaled, x_vector)
    # The trailing "..." picks up single_link's optional pixels axis when present and
    # contributes nothing when it isn't, so this one contraction handles both shapes
    # without branching (same idiom as the leading "..." used for the configuration
    # axis elsewhere in this module).
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
    arm label.
    """
    d1, d2 = label // 10, label % 10
    for _ in range(shift % 3):
        d1, d2 = _CYCLIC_SATELLITE[d1], _CYCLIC_SATELLITE[d2]
    return d1 * 10 + d2


def _cyclic_permute_terms(
    terms: tuple[tuple[int, int, tuple[int, ...]], ...], shift: int
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Cyclically relabels satellites (1->2->3->1, applied `shift` times) in every arm
    label of `terms` -- builds `Y`/`Z` from `X` (or `beta`/`gamma` from `alpha`).
    """
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
    ltt_by_arm: dict[int, jax.Array] | None = None,
) -> jax.Array:
    """
    Evaluates one TDI channel from `terms`: for each term, shifts `times_in_years` by
    the term's own cumulative delay (built from `ltt_by_arm`, each arm's current
    light-travel-time) and calls
    :func:`gw_response.space_based.single_link_geometry.single_link_response_delay_td`
    (reusing its exact per-arm evaluation and geometry) for just that term's arm -- via
    `arm_indices`, so the other 5 arms' geometry and waveform evaluations are skipped
    entirely rather than computed and discarded -- and accumulating with its sign.
    `jnp.real` (with the library's own natural convention baked in) is applied per term
    inside `single_link_response_delay_td`; since `Re` is linear over the (real) signs
    summed here, this is exactly equivalent to combining the complex per-arm terms
    first. `ltt_by_arm` defaults to `None`, recomputed at `times_in_years` if so -- used
    by `_apply_tdi2_prefactor`, which calls this at shifted times with their own,
    different light-travel-times.

    Returns:
        jax.Array: shape (time,).
    """
    if ltt_by_arm is None:
        _, ltt, _ = det.detector_arms_retarded(times_in_years, ps)
        ltt_by_arm = {
            label: ltt[:, i] for i, label in enumerate(_SINGLE_LINK_ARM_LABELS)
        }
    channel = jnp.zeros_like(times_in_years)
    for sign, arm_label, delay_labels in terms:
        if delay_labels:
            delay_seconds = sum(ltt_by_arm[label] for label in delay_labels)
            shifted_times = times_in_years - delay_seconds / ps.yr
        else:
            shifted_times = times_in_years

        arm_idx = _SINGLE_LINK_ARM_LABELS.index(arm_label)
        y_one_arm = single_link_response_delay_td(
            det,
            ps,
            shifted_times,
            theta,
            phi,
            strain_td,
            waveform_params,
            False,  # freeze_geometry
            (arm_idx,),  # arm_indices -- only this term's own arm
        )
        channel = channel + sign * y_one_arm[:, 0]
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
    arm's current light-travel-time) and calls `single_link_response_segmented_td` (in
    :mod:`gw_response.space_based.single_link_geometry`, reusing its segment-local
    linearized evaluation) for just that term's arm -- via `arm_indices`, same as
    :func:`_tdi_channel_delay_td` -- and accumulating with its sign.

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

        arm_idx = _SINGLE_LINK_ARM_LABELS.index(arm_label)
        y_one_arm = single_link_response_segmented_td(
            det,
            ps,
            shifted_times,
            theta,
            phi,
            strain_td,
            waveform_params,
            segment_length,
            (arm_idx,),  # arm_indices -- only this term's own arm
        )
        channel = channel + sign * y_one_arm[:, 0]
    return channel


def _tdi_channel_group_segmented_td(
    base_terms: tuple[tuple[int, int, tuple[int, ...]], ...],
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
    """One 3-channel group (X/Y/Z or alpha/beta/gamma), cyclically permuted from
    `base_terms`, via segment-stacking. Returns shape (3, time).
    """
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


@partial(jax.jit, static_argnums=(0, 5, 7, 8))
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
    (in :mod:`gw_response.space_based.single_link_geometry`) for each combination term's
    own (shifted) segment-local evaluation, the same delay-operator term tables as
    :func:`tdi_response_delay_td` (see the module-level comments above
    :data:`_X_TERMS`), and :meth:`gw_response.detector.Detector.detector_arms_retarded`
    for the light-travel-times the delay operators need. TDI 2.0 isn't supported here
    (its nested-delay prefactor re-evaluates the 1.5-generation channel at genuinely
    different times, which doesn't mesh with segment-stacking's fixed segment grid).
    Backs ``Response.get_response_segmented_td``.

    Jitted as one fused program (`det`/`strain_td`/`segment_length`/`combination` static
    -- `segment_length` has to be, for `single_link_response_segmented_td`'s reshape).
    Unjitted, this paid a large per-call dispatch tax (24 eager term evaluations, each
    re-tracing an internal `jax.vmap`); see ``examples/compare_with_pytdi.ipynb``
    section 9c.

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

    # Shared by every branch below -- both `_tdi_channel_group_segmented_td` and
    # `_tdi_channel_segmented_td` (called directly for `zeta`, which has no cyclic
    # siblings to group) take exactly this, after their own leading `terms`/
    # `base_terms` argument.
    shared_args = (
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
        xyz = _tdi_channel_group_segmented_td(_X_TERMS, *shared_args)
        if combination == "XYZ":
            result = xyz
        else:
            aet = xyz_to_aet @ xyz
            result = (
                aet
                if combination == "AET"
                else jnp.concatenate(
                    [
                        aet[:2],
                        _tdi_channel_segmented_td(_ZETA_TERMS, *shared_args)[None],
                    ],
                    axis=0,
                )
            )
    elif combination in ("Sagnac", "AET_Sagnac", "AE_Sagnac_zeta"):
        sagnac = _tdi_channel_group_segmented_td(_ALPHA_TERMS, *shared_args)
        if combination == "Sagnac":
            result = sagnac
        else:
            aet_sagnac = xyz_to_aet @ sagnac
            result = (
                aet_sagnac
                if combination == "AET_Sagnac"
                else jnp.concatenate(
                    [
                        aet_sagnac[:2],
                        _tdi_channel_segmented_td(_ZETA_TERMS, *shared_args)[None],
                    ],
                    axis=0,
                )
            )
    else:
        raise ValueError(
            f"Unknown TDI combination '{combination}'; expected one of 'XYZ', "
            "'AET', 'Sagnac', 'AET_Sagnac', 'AE_zeta', 'AE_Sagnac_zeta'."
        )

    return jnp.moveaxis(result, -1, 0)  # (time, channels=3)


def _tdi_channel_group_delay_td(
    base_terms: tuple[tuple[int, int, tuple[int, ...]], ...],
    prefactor_name: str,
    det: "Detector",
    ps: PhysicalConstants,
    times_in_years: jax.Array,
    theta: ArrayLike,
    phi: ArrayLike,
    strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
    waveform_params: Any,
    tdi_order: float,
    ltt_by_arm: dict[int, jax.Array],
) -> jax.Array:
    """One 3-channel group (X/Y/Z or alpha/beta/gamma), cyclically permuted from
    `base_terms` -- each channel at TDI 1.5, or promoted to TDI 2.0 via
    `_apply_tdi2_prefactor`. Returns shape (3, time).
    """
    channels = []
    for shift in range(3):
        terms = _cyclic_permute_terms(base_terms, shift)
        if tdi_order == 1.5:
            channels.append(
                _tdi_channel_delay_td(
                    terms,
                    det,
                    ps,
                    times_in_years,
                    theta,
                    phi,
                    strain_td,
                    waveform_params,
                    ltt_by_arm,
                )
            )
            continue

        prefactor = tuple(
            (sign, tuple(_relabel_satellite(d, shift) for d in delays))
            for sign, delays in _TDI2_PREFACTOR[prefactor_name]
        )
        channel_1_5 = partial(
            _tdi_channel_delay_td,
            terms,
            det,
            ps,
            theta=theta,
            phi=phi,
            strain_td=strain_td,
            waveform_params=waveform_params,
        )
        channels.append(
            _apply_tdi2_prefactor(
                prefactor, channel_1_5, ps, times_in_years, ltt_by_arm
            )
        )
    return jnp.stack(channels, axis=0)  # (3, time)


def _tdi_zeta_channel_delay_td(
    det: "Detector",
    ps: PhysicalConstants,
    times_in_years: jax.Array,
    theta: ArrayLike,
    phi: ArrayLike,
    strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
    waveform_params: Any,
    tdi_order: float,
    ltt_by_arm: dict[int, jax.Array],
) -> jax.Array:
    """The fully-symmetric zeta combination -- TDI 1.5, or promoted to TDI 2.0 via
    `_apply_tdi2_prefactor(_TDI2_PREFACTOR["zeta"], ...)`. Returns shape (time,).
    """
    if tdi_order == 1.5:
        return _tdi_channel_delay_td(
            _ZETA_TERMS,
            det,
            ps,
            times_in_years,
            theta,
            phi,
            strain_td,
            waveform_params,
            ltt_by_arm,
        )
    channel_1_5 = partial(
        _tdi_channel_delay_td,
        _ZETA_TERMS,
        det,
        ps,
        theta=theta,
        phi=phi,
        strain_td=strain_td,
        waveform_params=waveform_params,
    )
    return _apply_tdi2_prefactor(
        _TDI2_PREFACTOR["zeta"], channel_1_5, ps, times_in_years, ltt_by_arm
    )


@partial(jax.jit, static_argnums=(0, 5, 7, 8))
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

    Jitted as one fused program (`det`/`strain_td`/`combination`/`tdi_order` static,
    same convention as `single_link_response_delay_td`'s own jit). Unjitted, this paid a
    large per-call dispatch tax (24 separately-dispatched eager term evaluations); see
    ``examples/compare_with_pytdi.ipynb`` section 9a.

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
            complex ``(h_plus, h_cross)`` at that time -- see
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

    # Shared by every branch below -- both `_tdi_channel_group_delay_td` (with a
    # `base_terms`/`prefactor_name` pair prepended) and `_tdi_zeta_channel_delay_td`
    # (no `base_terms`/`prefactor_name` of its own, `zeta` being fully symmetric) take
    # exactly this.
    shared_args = (
        det,
        ps,
        times_in_years,
        theta,
        phi,
        strain_td,
        waveform_params,
        tdi_order,
        ltt_by_arm,
    )
    xyz_to_aet = BasisTransformations().XYZ_to_AET

    if combination in ("XYZ", "AET", "AE_zeta"):
        xyz = _tdi_channel_group_delay_td(_X_TERMS, "X", *shared_args)
        if combination == "XYZ":
            result = xyz
        else:
            aet = xyz_to_aet @ xyz
            result = (
                aet
                if combination == "AET"
                else jnp.concatenate(
                    [aet[:2], _tdi_zeta_channel_delay_td(*shared_args)[None]], axis=0
                )
            )
    elif combination in ("Sagnac", "AET_Sagnac", "AE_Sagnac_zeta"):
        sagnac = _tdi_channel_group_delay_td(_ALPHA_TERMS, "alpha", *shared_args)
        if combination == "Sagnac":
            result = sagnac
        else:
            aet_sagnac = xyz_to_aet @ sagnac
            result = (
                aet_sagnac
                if combination == "AET_Sagnac"
                else jnp.concatenate(
                    [aet_sagnac[:2], _tdi_zeta_channel_delay_td(*shared_args)[None]],
                    axis=0,
                )
            )
    else:
        raise ValueError(
            f"Unknown TDI combination '{combination}'; expected one of 'XYZ', "
            "'AET', 'Sagnac', 'AET_Sagnac', 'AE_zeta', 'AE_Sagnac_zeta'."
        )

    return jnp.moveaxis(result, -1, 0)  # (time, channels=3)
