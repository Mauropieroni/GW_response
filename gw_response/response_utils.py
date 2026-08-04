# Global imports
import chex
import jax
import jax.numpy as jnp

from typing import Any, Callable
from jax.typing import ArrayLike

# Local imports
from gw_response.space_based.tdi import build_tdi

# Update jax to use 64 bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def linear_response_angular(
    TDI_idx: ArrayLike,
    single_link: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Projects the single-link strain response onto a TDI combination, giving the
    (sky-resolved) linear response of that TDI variable: Hartwig, Lilley, Muratore &
    Pieroni (arXiv:2303.15929) eq. 2.27's ``V(f) = Σ c^V_ij η_ij(f)``.

    Args:
        TDI_idx (ArrayLike): Index into :data:`gw_response.space_based.tdi.TDI_map`
            selecting the TDI combination to project onto.
        single_link (ArrayLike): Single-link strain response, as returned by
            :func:`gw_response.single_link_static.get_single_link_response_static`, with
            shape (configurations, x_vector, arms, pixels).
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The linear TDI response, with shape (configurations, x_vector, TDI,
            pixels).
    """
    # Delegates to build_tdi, which also handles single_link with no
    # trailing pixels axis (already sky-integrated).
    return build_tdi(TDI_idx, single_link, arms_matrix_rescaled, x_vector)


@jax.jit
def quadratic_from_linear(linear_response: ArrayLike) -> jax.Array:
    """
    Squares a linear (TDI- or Michelson-projected) response into its quadratic
    cross-spectrum, summed over polarizations and Hermitian conjugation -- the
    per-sky-pixel ``C^U_ij C^V*_mn`` product underlying Hartwig, Lilley, Muratore &
    Pieroni (arXiv:2303.15929) eq. 2.29's ``S^UV(f)``, before the ``dΩ_k̂`` sky
    integral (eq. 2.15-2.16) that :func:`quadratic_response_integrated` performs.

    Args:
        linear_response (ArrayLike): Linear response, with shape (configurations,
            x_vector, TDI, pixels).

    Returns:
        jax.Array: The quadratic response, with shape (configurations, x_vector, TDI,
            TDI, pixels).
    """
    quadratic_response = jnp.einsum(
        "...ijl,...ikl->...ijkl",
        linear_response,
        jnp.conjugate(linear_response),
    )
    # The first 2 is sum over polarization the second is for the h.c. sum
    return 2 * 2 * quadratic_response / jnp.pi / 4


@jax.jit
def quadratic_response_angular(
    TDI_idx: ArrayLike,
    single_link: ArrayLike,
    arms_matrix_rescaled: ArrayLike,
    x_vector: ArrayLike,
) -> jax.Array:
    """
    Computes the (sky-resolved) quadratic response of a TDI combination:
    :func:`linear_response_angular` squared via :func:`quadratic_from_linear`.

    Args:
        TDI_idx (ArrayLike): Index into :data:`gw_response.space_based.tdi.TDI_map`
            selecting the TDI combination to project onto.
        single_link (ArrayLike): Single-link strain response, with shape
            (configurations, x_vector, arms, pixels).
        arms_matrix_rescaled (ArrayLike): Detector arm vectors rescaled by the arm
            length, with shape (configurations, vectorial_index (3), arms (6)).
        x_vector (ArrayLike): Vector of ``2 pi f L / c`` values over frequency.

    Returns:
        jax.Array: The quadratic TDI response, with shape (configurations, x_vector,
            TDI, TDI, pixels).
    """
    linear_response = linear_response_angular(
        TDI_idx, single_link, arms_matrix_rescaled, x_vector
    )
    return quadratic_from_linear(linear_response)


@jax.jit
def quadratic_response_integrated(angular_response: ArrayLike) -> jax.Array:
    """
    Averages the sky-resolved quadratic response over the sky (pixels) to give the
    quadratic TDI response as a function of frequency -- the sky-averaged response
    ``R(f)`` of Hartwig, Lilley, Muratore & Pieroni (arXiv:2303.15929) eq. 2.4-2.5
    (``S^GW(f) = R(f) P_h(f)``), via a pixelized ``dΩ_k̂`` average (eq. 2.16).

    Args:
        angular_response (ArrayLike): Sky-resolved quadratic response, as returned by
            :func:`quadratic_integrand`, with shape (configurations, x_vector, TDI, TDI,
            pixels).

    Returns:
        jax.Array: The sky-averaged quadratic response, with shape (configurations,
            x_vector, TDI, TDI), normalized by ``4 * pi`` to account for the solid angle
            of the sphere.
    """
    return 4 * jnp.pi * jnp.mean(angular_response, axis=-1)


def contract_with_h(
    R_plus: jax.Array,
    R_cross: jax.Array,
    h_plus: jax.Array,
    h_cross: jax.Array,
) -> jax.Array:
    """
    Combines a plus/cross transfer function (or antenna-pattern coefficient) with the
    plus/cross waveform quadrature components: ``R_plus * h_plus + R_cross * h_cross``
    -- the polarization sum ``Σ_A ξ_ij^A(f,k̂) h̃_A(f,k̂)`` of Hartwig, Lilley, Muratore
    & Pieroni (arXiv:2303.15929) eq. 2.12, restricted to the two polarizations `A`
    this package always uses (``+``/``×`` or ``L``/``R``). The shared contraction
    behind every time-domain response method in this package -- callers are
    responsible for shaping `R_plus`/`R_cross`/`h_plus`/`h_cross` so plain
    broadcasting lines up the axes being contracted (e.g. via `moveaxis` or a
    trailing `None` index), so this stays a single elementwise op.

    Args:
        R_plus (jax.Array): Plus-polarization transfer function/coefficient.
        R_cross (jax.Array): Cross-polarization transfer function/coefficient.
        h_plus (jax.Array): Plus-polarization waveform component.
        h_cross (jax.Array): Cross-polarization waveform component.

    Returns:
        jax.Array: ``R_plus * h_plus + R_cross * h_cross``.
    """
    return R_plus * h_plus + R_cross * h_cross


def h_from_amplitudes_phase(
    amplitude_plus: jax.Array, amplitude_cross: jax.Array, phase_value: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    The standard plus/cross quadrature decomposition of a waveform from its (real)
    amplitude(s) and phase: ``h_plus = amplitude_plus * exp(i*phase)``, ``h_cross =
    amplitude_cross * exp(i*(phase - pi/2))``. Shared by every time-domain response
    method that builds a waveform this way, given already-evaluated amplitude/phase
    values.

    Args:
        amplitude_plus (jax.Array): Plus-polarization amplitude(s).
        amplitude_cross (jax.Array): Cross-polarization amplitude(s).
        phase_value (jax.Array): Phase(s), in radians.

    Returns:
        tuple[jax.Array, jax.Array]: ``(h_plus, h_cross)``.
    """
    h_plus = amplitude_plus * jnp.exp(1j * phase_value)
    h_cross = -1j * amplitude_cross * jnp.exp(1j * phase_value)
    return h_plus, h_cross


def rotate_polarizations_by_psi(
    h_plus_source: jax.Array, h_cross_source: jax.Array, psi: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """
    Rotates a waveform's plus/cross quadratures from the source frame to the observer
    (detector-sky) frame by the polarization angle psi, via the standard
    ``cos(2*psi)``/``sin(2*psi)`` rotation (e.g. LDC Manual LISA-LCST-SGS-MAN-001, Sec.
    6.1.2, eq. 20): ``h_plus = h_plus_source*cos(2*psi) - h_cross_source*sin(2*psi)``,
    ``h_cross = h_plus_source*sin(2*psi) + h_cross_source*cos(2*psi)``.

    This is deliberately kept at the waveform layer rather than inside
    `Response`/geometry code: psi is a rotation *of the waveform itself* around the
    propagation direction, independent of (and applied before) the detector's own
    geometric projection -- matching how `lisagwresponse` itself applies psi entirely
    inside its own ``Waveform.strain()``, never touching its response/geometry code.
    Compose this with :func:`h_from_amplitudes_phase` (or any other source-frame
    ``(h_plus_source, h_cross_source)`` construction) to build a `Waveform.strain_td`/
    `strain_fd` callable that includes psi -- see :meth:`Waveform.from_source_frame` for
    the common case.

    Args:
        h_plus_source (jax.Array): Source-frame plus-polarization quadrature.
        h_cross_source (jax.Array): Source-frame cross-polarization quadrature.
        psi (ArrayLike): Polarization angle, in radians.

    Returns:
        tuple[jax.Array, jax.Array]: ``(h_plus, h_cross)``, rotated into the observer
            frame. ``psi=0`` is a no-op: returns `h_plus_source`/`h_cross_source`
            unchanged.
    """
    cos_2psi = jnp.cos(2 * psi)
    sin_2psi = jnp.sin(2 * psi)
    h_plus = h_plus_source * cos_2psi - h_cross_source * sin_2psi
    h_cross = h_plus_source * sin_2psi + h_cross_source * cos_2psi
    return h_plus, h_cross


@chex.dataclass
class Waveform:
    """
    Bundles a waveform as callables returning both polarizations at once, given a time/
    frequency and the source parameters -- matching how waveform models actually work
    (e.g. `ripplegw`'s `model(frequency, params) -> {"p": h_plus(f), "c": h_cross(f)}`),
    rather than recomputing shared amplitude/phase evolution twice. Taking
    `waveform_params` as its own argument (rather than baking specific values into the
    callable via a closure) lets `Response`'s methods stay jit-compiled once and reused
    across many parameter values -- e.g. the repeated likelihood evaluations of an
    inference run -- instead of retracing per call. `strain_td`/`strain_fd` could
    interpolate a
    densely-sampled model output (see ``examples/ripple_interface_prototype.py``) or be
    any other scalar-evaluable model. Attach one to `Response.waveform` so its methods
    don't need it passed again at every call.

    A model native to one domain need only set that one field -- callers that need the
    other domain handle deriving it themselves; `Waveform` doesn't do that conversion
    itself, since it would need a sample grid (`n`, `dt`) that isn't known until call
    time.

    Attributes:
        strain_td (Callable, optional): Maps a time, in seconds, and the source
            parameters to the complex ``(h_plus, h_cross)`` quadratures at that time.
            Needed by the single-link methods that evaluate the waveform at run-time-
            determined (retarded) times (`get_single_link_response_delay_td` and
            `get_single_link_response_segmented_td`).
        strain_fd (Callable, optional): Maps a frequency, in Hz, and the source
            parameters to the complex ``(h_f_plus, h_f_cross)`` quadratures at that
            frequency. Used directly (no FFT) by `Response.get_response_fd`.
    """

    strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]] | None = None
    strain_fd: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]] | None = None

    @classmethod
    def from_amplitude_phase(
        cls,
        amplitude_plus: Callable[[jax.Array, Any], jax.Array],
        amplitude_cross: Callable[[jax.Array, Any], jax.Array],
        phase: Callable[[jax.Array, Any], jax.Array],
    ) -> "Waveform":
        """
        Builds a :class:`Waveform` from separate amplitude/phase callables (setting
        `strain_td` only, each mapping a time and the source parameters to a value), via
        :func:`h_from_amplitudes_phase` -- convenient for simple analytic models that
        don't naturally return both polarizations together (unlike e.g. `ripplegw`, see
        the class docstring).
        """
        return cls(
            strain_td=lambda tau, waveform_params: h_from_amplitudes_phase(
                amplitude_plus(tau, waveform_params),
                amplitude_cross(tau, waveform_params),
                phase(tau, waveform_params),
            )
        )

    @classmethod
    def from_source_frame(
        cls,
        amplitude_plus: Callable[[jax.Array, Any], jax.Array],
        amplitude_cross: Callable[[jax.Array, Any], jax.Array],
        phase: Callable[[jax.Array, Any], jax.Array],
        psi: ArrayLike,
    ) -> "Waveform":
        """
        Builds a :class:`Waveform` the same way as :meth:`from_amplitude_phase`
        (source-frame amplitude/phase callables via :func:`h_from_amplitudes_phase`),
        additionally rotating the result into the observer frame by the polarization
        angle `psi` via :func:`rotate_polarizations_by_psi` -- so callers with a
        source-frame model and a separate `psi` don't have to hand-roll that rotation
        themselves. `psi` is fixed at construction time (like a source parameter baked
        into a specific `Waveform` instance), not threaded through `waveform_params`;
        build a new `Waveform` (e.g. via a thin wrapper following this same pattern) if
        `psi` itself needs to vary per call without retracing.

        Args:
            amplitude_plus (Callable): Maps a time, in seconds, and the source
                parameters to the source-frame plus-polarization amplitude.
            amplitude_cross (Callable): Maps a time, in seconds, and the source
                parameters to the source-frame cross-polarization amplitude.
            phase (Callable): Maps a time, in seconds, and the source parameters to the
                phase, in radians.
            psi (ArrayLike): Polarization angle, in radians.
        """
        return cls(
            strain_td=lambda tau, waveform_params: rotate_polarizations_by_psi(
                *h_from_amplitudes_phase(
                    amplitude_plus(tau, waveform_params),
                    amplitude_cross(tau, waveform_params),
                    phase(tau, waveform_params),
                ),
                psi,
            )
        )
