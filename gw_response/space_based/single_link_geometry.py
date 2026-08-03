# Global imports
import jax
import jax.numpy as jnp
from functools import partial
from typing import Any, Callable, TYPE_CHECKING

from jax.typing import ArrayLike

# Local imports
from gw_response.constants import BasisTransformations, PhysicalConstants
from gw_response.polarization import pc_tensors_and_signed_wavevector
from gw_response.response_utils import contract_with_h
from gw_response.single_link_utils import geometrical_factor

if TYPE_CHECKING:
    from gw_response.detector import Detector

# Update jax to use 64 bit precision
jax.config.update("jax_enable_x64", True)


# Single-link arm labels and their (receiver, emitter) satellite indices, for
# a LISA-like 3-satellite, one-way-laser-link constellation. Matches
# gw_response.utils.arms_matrix_from_vertex_positions's fixed arm ordering
# and vertex-index convention (satellites 1, 2, 3): for arm label "ij" (a
# two-digit int with digits i, j), the arm vector points from satellite i to
# satellite j, and -- as validated against the independent `lisagwresponse`
# package -- satellite j is the emitter and satellite i is the receiver.
_SINGLE_LINK_ARM_LABELS = (12, 23, 31, 21, 32, 13)


def per_arm_retarded_geometry_rescaled(
    det: "Detector", time_in_years: ArrayLike, ps: PhysicalConstants
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    :meth:`gw_response.detector.Detector.detector_arms_retarded`, rescaled to the
    dimensionless units
    :func:`gw_response.single_link_retarded.get_single_link_response_retarded` expects:
    lengths by `det.armlength`, and the light-travel-time by `ps.light_speed /
    det.armlength` to match.

    Args:
        det (Detector): The detector (e.g. LISA, LIGO) the geometry is computed for.
        time_in_years (ArrayLike): Time(s), in years; see
            :meth:`gw_response.detector.Detector.detector_arms_retarded`.
        ps (PhysicalConstants): Physical constants (`light_speed`, `yr`).

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: `(arm_vector_retarded_rescaled,
            ltt_rescaled, receiver_positions_rescaled)`, shapes as in
            :meth:`gw_response.detector.Detector.detector_arms_retarded`.
    """
    arm_vector_retarded, ltt, receiver_position = det.detector_arms_retarded(
        time_in_years, ps
    )
    return (
        arm_vector_retarded / det.armlength,
        ltt * ps.light_speed / det.armlength,
        receiver_position / det.armlength,
    )


def all_arms_geometry(
    det: "Detector",
    ps: PhysicalConstants,
    k: jax.Array,
    t_years: ArrayLike,
    freeze_geometry: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    For all of the detector's arms at once, backdates each emitter's position by that
    arm's own light-travel-time estimate (unless `freeze_geometry`) and projects both
    endpoints onto the wavevector -- built on
    :meth:`gw_response.detector.Detector.detector_arms_retarded` (shared with the
    frequency-domain retarded pipeline), batched over arms instead of the per-arm
    computation this replaces. The geometry shared by
    :func:`single_link_response_delay_td` (which evaluates the waveform at the
    resulting shifted times directly) and :func:`per_arm_linearized_geometry` (which
    Taylor-expands these same quantities around a reference time via `jax.jvp`).

    Args:
        det (Detector): The detector the geometry is computed for.
        ps (PhysicalConstants): Physical constants (`light_speed`, `yr`).
        k (jax.Array): Unit wavevector, with shape (vectorial_index (3),).
        t_years (ArrayLike): Reception/geometry time, in years (scalar).
        freeze_geometry (bool): If True, evaluate each emitter at the same (reception)
            time as the receiver instead of backdating it.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]: `(n_vec, shift_rec,
            shift_emi, denom)`: each arm's unit direction (receiver to emitter), with
            shape (vectorial_index (3), arms), and each endpoint's wavevector-projected
            time shift plus the antenna-pattern normalization ``2 * (1 - n_vec . k)``,
            each with shape (arms,) -- so that e.g. the emitter's own retarded phase
            argument is ``t_rec_seconds - shift_emi``. Arms are in
            `det.arm_vertex_pairs` (== :data:`_SINGLE_LINK_ARM_LABELS`) order.
    """
    arm_vector, ltt, receiver = det.detector_arms_retarded(t_years, ps, freeze_geometry)
    # t_years is a scalar, so the leading "configurations" axis is trivial here.
    arm_vector = arm_vector[0]  # (3, arms), receiver -> emitter
    ltt = ltt[0]  # (arms,)
    receiver = receiver[0]  # (3, arms)
    emitter = receiver + arm_vector

    n_vec = -arm_vector / jnp.linalg.norm(arm_vector, axis=0)  # emitter -> receiver
    shift_rec = jnp.einsum("i,ij->j", k, receiver) / ps.light_speed
    shift_emi = ltt + jnp.einsum("i,ij->j", k, emitter) / ps.light_speed
    denom = 2 * (1 - jnp.einsum("i,ij->j", k, n_vec))
    return n_vec, shift_rec, shift_emi, denom


def _xi_plus_cross(
    n_vec: jax.Array, p_plus_mat: jax.Array, p_cross_mat: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Contracts each arm's unit direction (`n_vec`, from :func:`all_arms_geometry`) with
    the plus/cross polarization tensors, via
    :func:`gw_response.single_link_utils.geometrical_factor` (shared with the
    frequency-domain static/retarded pipelines, which already combine this same
    contraction with a finite-arm-length transfer function) -- `geometrical_factor`
    bakes in a factor of ``1/2`` (from its own outer-product convention) that's
    undone here to match this module's `xiplus`/`xicross` convention.

    Args:
        n_vec (jax.Array): Unit arm direction(s), with shape (vectorial_index (3),
            arms).
        p_plus_mat (jax.Array): Plus-polarization tensor, with shape (3, 3).
        p_cross_mat (jax.Array): Cross-polarization tensor, with shape (3, 3).

    Returns:
        tuple[jax.Array, jax.Array]: `(xiplus, xicross)`, each with shape (arms,).
    """
    xiplus = 2 * geometrical_factor(n_vec[None], p_plus_mat[None])[0, :, 0]
    xicross = 2 * geometrical_factor(n_vec[None], p_cross_mat[None])[0, :, 0]
    return xiplus, xicross


def per_arm_linearized_geometry(
    det: "Detector",
    t_ref_years: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    wavevector_sign: ArrayLike,
    ps: PhysicalConstants,
) -> tuple[jax.Array, jax.Array]:
    """
    For all 6 arms at once, computes the delay formula's geometric quantities
    (`shift_rec`, `shift_emi`, `xiplus`, `xicross`, `denom`) at `t_ref_years` plus their
    time derivatives, via a single `jax.jvp` over the batched
    :func:`all_arms_geometry` -- used by :func:`single_link_response_segmented_td` to
    build a first-order Taylor expansion per segment.

    Args:
        det (Detector): The detector (e.g. LISA) the geometry is computed for.
        t_ref_years (ArrayLike): Time, in years, around which to linearize.
        theta (ArrayLike): Colatitude of the single sky position the signal arrives
            from, in radians.
        phi (ArrayLike): Longitude of the single sky position the signal arrives from,
            in radians.
        wavevector_sign (ArrayLike): Multiplies `unit_vec(theta, phi)`.
        ps (PhysicalConstants): Physical constants (`light_speed`, `yr`).

    Returns:
        tuple[jax.Array, jax.Array]: `(values, derivatives)`, each shape (5, arms)
            stacked in :data:`_SINGLE_LINK_ARM_LABELS` order, rows `(shift_rec,
            shift_emi, xiplus, xicross, denom)`.
    """
    wavevector, p_plus, p_cross = pc_tensors_and_signed_wavevector(
        theta, phi, wavevector_sign
    )
    p_plus_mat = p_plus[0]
    p_cross_mat = p_cross[0]
    k = wavevector[:, 0]

    def geometry_at(t_years: ArrayLike) -> jax.Array:
        n_vec, shift_rec, shift_emi, denom = all_arms_geometry(det, ps, k, t_years)
        xiplus, xicross = _xi_plus_cross(n_vec, p_plus_mat, p_cross_mat)
        return jnp.stack([shift_rec, shift_emi, xiplus, xicross, denom])

    v_mid, dv_dyear = jax.jvp(geometry_at, (t_ref_years,), (1.0,))
    return v_mid, dv_dyear / ps.yr


@partial(jax.jit, static_argnums=(0, 7))
def single_link_response_linearized(
    det: "Detector",
    ps: PhysicalConstants,
    t_ref_years: ArrayLike,
    t_ref_seconds: ArrayLike,
    t_phase_seconds: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
    params: Any,
    wavevector_sign: ArrayLike,
    final_factor: ArrayLike,
) -> jax.Array:
    """
    Shared computation behind
    :meth:`gw_response.response.Response.get_single_link_response_segmented_td`, for one
    segment: evaluates :func:`per_arm_linearized_geometry` once at the segment's own
    `t_ref_years`, then reconstructs each geometric quantity as a first-order Taylor
    expansion around it (``value(t) = value(t_ref) + derivative(t_ref) * (t - t_ref)``)
    at every sample in `t_phase_seconds`, and combines them exactly as
    :func:`single_link_response_delay_td` does -- differing only in using this linear
    model instead of a fresh `det.vertex_positions` lookup per sample.

    Args:
        det (Detector): The detector (e.g. LISA) the response is computed for.
        ps (PhysicalConstants): Physical constants (`light_speed`, `yr`).
        t_ref_years (ArrayLike): Time, in years, around which to linearize (this
            segment's own midpoint).
        t_ref_seconds (ArrayLike): `t_ref_years` in seconds.
        t_phase_seconds (ArrayLike): Time(s), in seconds, within this segment at which
            to evaluate the response.
        theta (ArrayLike): Colatitude of the single sky position the signal arrives
            from, in radians.
        phi (ArrayLike): Longitude of the single sky position the signal arrives from,
            in radians.
        strain_td (Callable): Maps a time, in seconds, and `params` to the complex
            ``(h_plus, h_cross)`` quadratures at that time -- see
            :class:`gw_response.response_utils.Waveform`.
        params (Any): Source parameters passed through to `strain_td`.
        wavevector_sign (ArrayLike): Multiplies `unit_vec(theta, phi)`.
        final_factor (ArrayLike): Complex factor applied to the result just before
            taking its real part.

    Returns:
        jax.Array: shape (time, arms=6), arm order :data:`_SINGLE_LINK_ARM_LABELS`.
    """
    values, derivatives = per_arm_linearized_geometry(
        det, t_ref_years, theta, phi, wavevector_sign, ps
    )
    shift_rec_ref, shift_emi_ref, xiplus_ref, xicross_ref, denom_ref = values
    dshift_rec, dshift_emi, dxiplus, dxicross, ddenom = derivatives

    t_phase_seconds = jnp.asarray(t_phase_seconds)
    dt_arr = (t_phase_seconds - t_ref_seconds)[:, None]  # (time, 1)
    shift_rec = shift_rec_ref[None, :] + dshift_rec[None, :] * dt_arr
    shift_emi = shift_emi_ref[None, :] + dshift_emi[None, :] * dt_arr
    xiplus = xiplus_ref[None, :] + dxiplus[None, :] * dt_arr
    xicross = xicross_ref[None, :] + dxicross[None, :] * dt_arr
    denom = denom_ref[None, :] + ddenom[None, :] * dt_arr

    t_rec_shifted = t_phase_seconds[:, None] - shift_rec  # (time, arms)
    t_emi_shifted = t_phase_seconds[:, None] - shift_emi

    h_plus_emi, h_cross_emi = strain_td(t_emi_shifted, params)
    h_plus_rec, h_cross_rec = strain_td(t_rec_shifted, params)
    termplus = h_plus_emi - h_plus_rec
    termcross = h_cross_emi - h_cross_rec
    y_complex = contract_with_h(xiplus, xicross, termplus, termcross) / denom
    return jnp.real(y_complex * final_factor)


@partial(jax.jit, static_argnums=(0, 5, 8))
def single_link_response_delay_td(
    det: "Detector",
    ps: PhysicalConstants,
    times_in_years: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
    params: Any,
    times_geometry_years: ArrayLike | None = None,
    freeze_geometry: bool = False,
    wavevector_sign: ArrayLike = 1.0,
    final_factor: ArrayLike = 1j,
) -> jax.Array:
    """
    Single-link (not TDI-combined) time-domain response for a one-way-laser-link
    constellation (LISA, Taiji), computed by directly evaluating the waveform at each
    link's retarded emission/reception times and differencing -- ``h(t_emission) -
    h(t_reception)`` -- rather than building a frequency-domain transfer function. No
    narrowband approximation: exact for arbitrarily evolving geometry. Cross-checked
    against the independent `lisagwresponse` package
    (https://gitlab.in2p3.fr/lisa-simulation/gw-response) to floating-point precision
    (via `wavevector_sign`/`final_factor`); see
    ``examples/compare_with_lisagwresponse.ipynb``. Backs
    ``Response.get_single_link_response_delay_td``.

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
        strain_td (Callable): Maps a time, in seconds, and `params` to the complex
            ``(h_plus, h_cross)`` quadratures at that time -- see
            :class:`gw_response.response_utils.Waveform`.
        params (Any): Source parameters passed through to `strain_td`.
        times_geometry_years (ArrayLike, optional): Time(s), in years, at which to
            evaluate the *detector geometry* (broadcast against `times_in_years`, which
            always supplies the *phase*/reception time). Defaults to `times_in_years`
            itself.
        freeze_geometry (bool): If True, also evaluates each emitter's *position* at the
            same reception time as the receiver, instead of backdating it by the
            light-travel time. The light-travel-time delay in the *phase* argument is
            still applied either way.
        wavevector_sign (ArrayLike): Multiplies `unit_vec(theta, phi)` before it's used
            as the wavevector; -1.0 reproduces `lisagwresponse`'s antiparallel
            convention (see above).
        final_factor (ArrayLike): Complex factor applied to the result just before
            taking its real part; see above.

    Returns:
        jax.Array: The real single-link time-domain response, with shape (time, arms=6),
            in arm order :data:`_SINGLE_LINK_ARM_LABELS` (12, 23, 31, 21, 32, 13).

    Raises:
        ValueError: If `theta`/`phi` resolve to more than one sky position.
    """
    if jnp.atleast_1d(theta).shape[0] != 1 or jnp.atleast_1d(phi).shape[0] != 1:
        raise ValueError(
            "single_link_response_delay_td requires a single sky position."
        )

    times_in_years = jnp.atleast_1d(times_in_years)
    times_phase_seconds = times_in_years * ps.yr
    times_geometry_years = (
        times_in_years
        if times_geometry_years is None
        else jnp.atleast_1d(times_geometry_years)
    )
    times_geometry_years, times_phase_seconds = jnp.broadcast_arrays(
        times_geometry_years, times_phase_seconds
    )

    wavevector, p_plus, p_cross = pc_tensors_and_signed_wavevector(
        theta, phi, wavevector_sign
    )
    p_plus_mat = p_plus[0]  # (3, 3), single sky position
    p_cross_mat = p_cross[0]
    k = wavevector[:, 0]  # (3,), single sky position

    def all_arms_at_sample(
        t_geometry_years: ArrayLike, t_rec_seconds: ArrayLike
    ) -> jax.Array:
        n_vec, shift_rec, shift_emi, denom = all_arms_geometry(
            det, ps, k, t_geometry_years, freeze_geometry
        )
        xiplus, xicross = _xi_plus_cross(n_vec, p_plus_mat, p_cross_mat)

        t_emi_shifted = jnp.asarray(t_rec_seconds - shift_emi)  # (arms,)
        t_rec_shifted = jnp.asarray(t_rec_seconds - shift_rec)  # (arms,)
        h_plus_emi, h_cross_emi = strain_td(t_emi_shifted, params)
        h_plus_rec, h_cross_rec = strain_td(t_rec_shifted, params)
        termplus = h_plus_emi - h_plus_rec
        termcross = h_cross_emi - h_cross_rec
        signal = contract_with_h(xiplus, xicross, termplus, termcross)
        return signal / denom

    y_complex = jax.vmap(all_arms_at_sample)(
        times_geometry_years, times_phase_seconds
    )  # (time, arms)
    return jnp.real(y_complex * final_factor)  # (time, arms)


# TDI 1.5 (unequal but locally-constant arms) time-domain combination
# formulas, from Muratore, Vetrugno & Vitale (arXiv:2303.15929), eq. (2.24),
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
# built from exact per-arm data via :func:`single_link_response_delay_td`.
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


def _cyclic_permute_terms(
    terms: tuple[tuple[int, int, tuple[int, ...]], ...], shift: int
) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
    """Cyclically relabels satellites (1->2->3->1, applied `shift` times) in every
    arm label of `terms` -- builds `Y`/`Z` from `X` (or `beta`/`gamma` from `alpha`)."""

    def relabel(label: int) -> int:
        d1, d2 = label // 10, label % 10
        for _ in range(shift % 3):
            d1, d2 = _CYCLIC_SATELLITE[d1], _CYCLIC_SATELLITE[d2]
        return d1 * 10 + d2

    return tuple(
        (sign, relabel(arm), tuple(relabel(d) for d in delays))
        for sign, arm, delays in terms
    )


def _tdi_channel_delay_td(
    terms: tuple[tuple[int, int, tuple[int, ...]], ...],
    det: "Detector",
    ps: PhysicalConstants,
    times_in_years: jax.Array,
    theta: ArrayLike,
    phi: ArrayLike,
    strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
    params: Any,
    ltt_by_arm: dict[int, jax.Array],
    wavevector_sign: ArrayLike,
    final_factor: ArrayLike,
) -> jax.Array:
    """
    Evaluates one TDI channel from `terms`: for each term, shifts `times_in_years` by
    the term's own cumulative delay (built from `ltt_by_arm`, each arm's current
    light-travel-time) and calls :func:`single_link_response_delay_td` (reusing its
    exact per-arm evaluation and geometry), picking out just that term's arm and
    accumulating with its sign. `final_factor`/`jnp.real` are applied per term inside
    `single_link_response_delay_td`; since `Re` is linear over the (real) signs summed
    here, this is exactly equivalent to combining the complex per-arm terms first.

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
            params,
            wavevector_sign=wavevector_sign,
            final_factor=final_factor,
        )
        arm_idx = _SINGLE_LINK_ARM_LABELS.index(arm_label)
        channel = channel + sign * y_all_arms[:, arm_idx]
    return channel


def tdi_response_delay_td(
    det: "Detector",
    ps: PhysicalConstants,
    times_in_years: ArrayLike,
    theta: ArrayLike,
    phi: ArrayLike,
    strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
    params: Any,
    combination: str = "XYZ",
    wavevector_sign: ArrayLike = 1.0,
    final_factor: ArrayLike = 1j,
) -> jax.Array:
    """
    TDI 1.5 (unequal but locally-constant arms) time-domain response for a LISA-like
    constellation, computed exactly for genuinely evolving geometry -- reuses
    :func:`single_link_response_delay_td` for each combination term's own (shifted)
    exact single-link evaluation and :meth:`gw_response.detector.Detector.
    detector_arms_retarded` for the light-travel-times the delay operators need, rather
    than any new geometry code. See the module-level comment above :data:`_X_TERMS` for
    the delay-operator convention and the Muratore, Vetrugno & Vitale reference
    (arXiv:2303.15929, eq. 2.24) this implements. Backs
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
        strain_td (Callable): Maps a time, in seconds, and `params` to the complex
            ``(h_plus, h_cross)`` quadratures at that time -- see
            :class:`gw_response.response_utils.Waveform`.
        params (Any): Source parameters passed through to `strain_td`.
        combination (str, optional): One of "XYZ", "AET", "Sagnac", "AET_Sagnac",
            "AE_zeta", "AE_Sagnac_zeta" (matching
            :data:`gw_response.space_based.tdi.TDI_map`'s keys). Default is "XYZ".
        wavevector_sign (ArrayLike): Multiplies `unit_vec(theta, phi)`; see
            :func:`single_link_response_delay_td`.
        final_factor (ArrayLike): Complex factor applied to each term's result just
            before taking its real part; see :func:`single_link_response_delay_td`.

    Returns:
        jax.Array: The real TDI-combined time-domain response, with shape (time,
            channels=3).

    Raises:
        ValueError: If `combination` isn't one of the supported names.
    """
    times_in_years = jnp.atleast_1d(times_in_years)
    _, ltt, _ = det.detector_arms_retarded(times_in_years, ps)  # (time, arms)
    ltt_by_arm = {
        label: ltt[:, i] for i, label in enumerate(_SINGLE_LINK_ARM_LABELS)
    }

    def channel_group(base_terms: tuple) -> jax.Array:
        return jnp.stack(
            [
                _tdi_channel_delay_td(
                    _cyclic_permute_terms(base_terms, shift),
                    det,
                    ps,
                    times_in_years,
                    theta,
                    phi,
                    strain_td,
                    params,
                    ltt_by_arm,
                    wavevector_sign,
                    final_factor,
                )
                for shift in range(3)
            ],
            axis=0,
        )  # (3, time)

    def zeta_channel() -> jax.Array:
        return _tdi_channel_delay_td(
            _ZETA_TERMS,
            det,
            ps,
            times_in_years,
            theta,
            phi,
            strain_td,
            params,
            ltt_by_arm,
            wavevector_sign,
            final_factor,
        )  # (time,)

    xyz_to_aet = BasisTransformations().XYZ_to_AET

    if combination in ("XYZ", "AET", "AE_zeta"):
        xyz = channel_group(_X_TERMS)
        if combination == "XYZ":
            result = xyz
        else:
            aet = xyz_to_aet @ xyz
            result = aet if combination == "AET" else jnp.concatenate(
                [aet[:2], zeta_channel()[None]], axis=0
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
