# Global imports
import chex
import jax
import jax.numpy as jnp

from typing import Any, Callable
from jax.typing import ArrayLike

# Update jax to use 64 bit precision
jax.config.update("jax_enable_x64", True)


def contract_with_h(
    R_plus: jax.Array,
    R_cross: jax.Array,
    h_plus: jax.Array,
    h_cross: jax.Array,
) -> jax.Array:
    """
    Combines a plus/cross transfer function (or antenna-pattern coefficient) with the
    plus/cross waveform components, returning the total response.

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
    amplitudes: tuple[jax.Array, jax.Array], phase_value: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Complex embedding of the plus/cross waveform from (real) amplitudes and phase:
    ``h_plus = amplitude_plus * exp(i*phase)``, ``h_cross = amplitude_cross *
    exp(i*(phase - pi/2))``. Not physically complex strain: ``Re(h_plus) =
    amplitude_plus * cos(phase)`` and ``Re(h_cross) = amplitude_cross * sin(phase)`` are
    the real waveforms; callers take ``jnp.real`` once, downstream (see
    :class:`Waveform`'s own `strain_td`).

    Args:
        amplitudes (tuple[jax.Array, jax.Array]): ``(amplitude_plus, amplitude_cross)``
            -- always defined together (e.g. both from the same inclination), so taken
            as one pair rather than two separate arguments.
        phase_value (jax.Array): Phase(s), in radians.

    Returns:
        tuple[jax.Array, jax.Array]: ``(h_plus, h_cross)``, complex-embedded as above.
    """
    amplitude_plus, amplitude_cross = amplitudes
    h_plus = amplitude_plus * jnp.exp(1j * phase_value)
    h_cross = -1j * amplitude_cross * jnp.exp(1j * phase_value)
    return h_plus, h_cross


def rotate_polarizations_by_psi(
    h_plus_source: jax.Array, h_cross_source: jax.Array, psi: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """
    Rotates a waveform's plus/cross from the source frame to the observer (detector-sky)
    frame by the polarization angle psi, via the standard ``cos(2*psi)``/``sin(2*psi)``
    rotation (e.g. LDC Manual LISA-LCST-SGS-MAN-001, Sec. 6.1.2, eq. 20): ``h_plus =
    h_plus_source*cos(2*psi) - h_cross_source*sin(2*psi)``, ``h_cross =
    h_plus_source*sin(2*psi) + h_cross_source*cos(2*psi)``.

    This is deliberately kept at the waveform layer rather than inside the response.

    Args:
        h_plus_source (jax.Array): Source-frame plus-polarization.
        h_cross_source (jax.Array): Source-frame cross-polarization.
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
    Bundles a waveform as callables returning both polarizations together for a given
    time/frequency and source parameters, matching how waveform models work (e.g.
    `ripplegw`'s `model(frequency, params) -> {"p": h_plus(f), "c": h_cross(f)}`). This
    approach avoids recomputing shared amplitude/phase evolution twice. Keeping
    `waveform_params` separate lets `Response` methods remain jit-compiled and reusable
    across parameter values, e.g. (repeated likelihood evaluations).

    A model native to one domain only requires that field; callers needing the other
    domain must derive it themselves. `Waveform` does not perform this conversion
    because it would require a sample grid (`n`, `dt`) unavailable until call time.

    Attributes:
        strain_td (Callable, optional): Maps time in seconds and source parameters to
            complex ``(h_plus, h_cross)`` -- a complex embedding (see
            :func:`h_from_amplitudes_phase`), not physically complex strain: the real
            waveform is ``Re(h_plus)``/``Re(h_cross)``, recovered via `jnp.real` once,
            downstream (see `single_link_response_delay_td`). Required by single-link
            methods that evaluate the waveform at run-time-determined retarded times
            (`get_single_link_response_delay_td` and
            `get_single_link_response_segmented_td`).
        strain_fd (Callable, optional): Maps frequency in Hz and source parameters to
            complex ``(h_f_plus, h_f_cross)`` -- genuinely complex (ordinary
            Fourier-domain strain, no embedding). Used directly (without an FFT) by
            `Response.get_response_fd`.
    """

    strain_td: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]] | None = None
    strain_fd: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]] | None = None

    @classmethod
    def from_amplitude_phase(
        cls,
        amplitudes: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
        phase: Callable[[jax.Array, Any], jax.Array],
    ) -> "Waveform":
        """
        Builds a :class:`Waveform` from an amplitudes/phase callable pair (setting
        `strain_td` only), via :func:`h_from_amplitudes_phase` -- convenient for simple
        analytic models that don't naturally return both polarizations together
        (unlike e.g. `ripplegw`, see the class docstring). `amplitudes` returns both
        ``(amplitude_plus, amplitude_cross)`` together, since they're always defined
        from the same source parameters (e.g. inclination), never independently.
        """
        return cls(
            strain_td=lambda tau, waveform_params: h_from_amplitudes_phase(
                amplitudes(tau, waveform_params), phase(tau, waveform_params)
            )
        )

    @classmethod
    def from_source_frame(
        cls,
        amplitudes: Callable[[jax.Array, Any], tuple[jax.Array, jax.Array]],
        phase: Callable[[jax.Array, Any], jax.Array],
        psi: ArrayLike,
    ) -> "Waveform":
        """
        Builds a :class:`Waveform` the same way as :meth:`from_amplitude_phase`
        (source-frame `amplitudes`/`phase` via :func:`h_from_amplitudes_phase`),
        additionally rotating the result into the observer frame by the polarization
        angle `psi` via :func:`rotate_polarizations_by_psi` -- so callers with a
        source-frame model and a separate `psi` don't have to hand-roll that rotation
        themselves. `psi` is fixed at construction time (like a source parameter baked
        into a specific `Waveform` instance), not threaded through `waveform_params`;
        build a new `Waveform` (e.g. via a thin wrapper following this same pattern) if
        `psi` itself needs to vary per call without retracing.

        Args:
            amplitudes (Callable): Maps a time, in seconds, and the source parameters
                to the source-frame ``(amplitude_plus, amplitude_cross)``.
            phase (Callable): Maps a time, in seconds, and the source parameters to the
                phase, in radians.
            psi (ArrayLike): Polarization angle, in radians.
        """
        return cls(
            strain_td=lambda tau, waveform_params: rotate_polarizations_by_psi(
                *h_from_amplitudes_phase(
                    amplitudes(tau, waveform_params), phase(tau, waveform_params)
                ),
                psi,
            )
        )
