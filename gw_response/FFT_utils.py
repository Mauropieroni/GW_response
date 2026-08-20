# Global imports
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable
from jax.typing import ArrayLike

# Update jax configuration to enable 64-bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def strain_to_frequency_domain(h_time: ArrayLike, dt: ArrayLike) -> jax.Array:
    """
    Converts a real time-domain signal to its non-negative-frequency spectrum: ``H(f) =
    dt * rfft(h(t))``, the exact inverse of :func:`frequency_domain_to_time_domain`.

    Args:
        h_time (ArrayLike): Real time-domain signal, with shape (..., time).
        dt (ArrayLike): Sample spacing, in seconds.

    Returns:
        jax.Array: The complex non-negative-frequency spectrum, with shape (..., time //
            2 + 1).
    """
    return jnp.fft.rfft(h_time, axis=-1) * dt


@partial(jax.jit, static_argnums=(1,))
def frequency_domain_to_time_domain(
    h_freq: ArrayLike, n: int, dt: ArrayLike
) -> jax.Array:
    """
    Inverts :func:`strain_to_frequency_domain`: ``h(t) = irfft(H(f), n=n) / dt``.

    Args:
        h_freq (ArrayLike): Complex non-negative-frequency spectrum, with shape (...,
            frequency).
        n (int): Number of time-domain samples to reconstruct (needed since a
            non-negative-frequency spectrum alone doesn't determine whether the original
            signal had an even or odd length).
        dt (ArrayLike): Sample spacing, in seconds.

    Returns:
        jax.Array: The real time-domain signal, with shape (..., n).
    """
    return jnp.fft.irfft(h_freq, n=n, axis=-1) / dt


@jax.jit
def fft_positive_time_and_freqs(h_time: jax.Array) -> jax.Array:
    """
    Complex analytic signal of a real time series (the standard Hilbert-transform
    construction, e.g. ``scipy.signal.hilbert``), computed via FFT: keeps only the
    non-negative-frequency content (doubled, DC/Nyquist untouched), so ``h(t) =
    Re[fft_positive_time_and_freqs(h)(t)]`` exactly. Used to extract an instantaneous
    amplitude/phase envelope from a real waveform without assuming a particular
    carrier/envelope decomposition upfront.

    Args:
        h_time (ArrayLike): Real time-domain signal, with shape (time,).

    Returns:
        jax.Array: The complex analytic signal, with shape (time,).
    """
    n = h_time.shape[-1]
    spectrum = jnp.fft.fft(h_time, axis=-1)
    mask = jnp.zeros(n).at[0].set(1.0)
    if n % 2 == 0:
        mask = mask.at[n // 2].set(1.0).at[1 : n // 2].set(2.0)
    else:
        mask = mask.at[1 : (n + 1) // 2].set(2.0)
    return jnp.fft.ifft(spectrum * mask, axis=-1)


@jax.jit
def spectral_derivative(x_time: jax.Array, dt: ArrayLike) -> jax.Array:
    """
    Exact derivative of a uniformly-sampled (real or complex) time series, computed by
    spectral differentiation: FFT, multiply by ``i * omega``, inverse FFT. Exact for
    band-limited signals, unlike finite differences.

    Args:
        x_time (ArrayLike): Time series, with shape (..., time).
        dt (ArrayLike): Sample spacing, in seconds.

    Returns:
        jax.Array: The derivative, with shape (..., time).
    """
    n = x_time.shape[-1]
    omega = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dt)
    return jnp.fft.ifft(1j * omega * jnp.fft.fft(x_time, axis=-1), axis=-1)


@jax.jit
def instantaneous_frequency(h_time: jax.Array, dt: ArrayLike) -> jax.Array:
    """
    Instantaneous frequency of a real time series, via its analytic signal's derivative:
    ``f(t) = (1 / 2 pi) Im[d(fft_positive_time_and_freqs)/dt /
    fft_positive_time_and_freqs(t)]``. The derivative is computed by exact spectral
    differentiation (:func:`spectral_derivative`) rather than finite differences, since
    `h_time` is a fixed array of samples with no closed-form function to autodiff.

    Args:
        h_time (ArrayLike): Real time-domain signal, with shape (time,).
        dt (ArrayLike): Sample spacing, in seconds.

    Returns:
        jax.Array: The instantaneous frequency, in Hz, with shape (time,).
    """
    h_of_f = fft_positive_time_and_freqs(h_time)
    return jnp.imag(spectral_derivative(h_of_f, dt) / h_of_f) / (2.0 * jnp.pi)


def instantaneous_phase_and_frequency(
    phase: Callable[[ArrayLike], jax.Array], t_seconds: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """
    Instantaneous phase and frequency of a waveform given as a function of time, via one
    `jax.jvp` call -- the autodiff counterpart of :func:`instantaneous_frequency`, which
    spectrally differentiates a fixed array of samples instead.

    Args:
        phase (Callable): Maps a time, in seconds, to the waveform phase, in radians
            (scalar, or one per mode).
        t_seconds (ArrayLike): A single time, in seconds.

    Returns:
        tuple: ``(phase_value, frequency)``, in radians and Hz, both evaluated at
            `t_seconds`.
    """
    phase_value, phase_dot = jax.jvp(phase, (t_seconds,), (1.0,))
    return phase_value, phase_dot / (2.0 * jnp.pi)
