# Global imports
import jax
import jax.numpy as jnp
from functools import partial

# Update jax configuration to enable 64-bit precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def strain_to_frequency_domain(h_time: jax.Array, dt: jax.Array) -> jax.Array:
    """
    Converts a real time-domain signal to its non-negative-frequency spectrum: ``H(f) =
    dt * rfft(h(t))``, the exact inverse of :func:`frequency_domain_to_time_domain`. The
    "derive it themselves" conversion :class:`gw_response.response_utils.Waveform`'s own
    docstring refers to, for a deterministic source with a `strain_td` model but no
    native `strain_fd`.

    Args:
        h_time (jax.Array): Real time-domain signal, with shape (..., time).
        dt (jax.Array): Sample spacing, in seconds.

    Returns:
        jax.Array: The complex non-negative-frequency spectrum, with shape (..., time //
            2 + 1).
    """
    return jnp.fft.rfft(h_time, axis=-1) * dt


@partial(jax.jit, static_argnums=(1,))
def frequency_domain_to_time_domain(
    h_freq: jax.Array, n: int, dt: jax.Array
) -> jax.Array:
    """
    Inverts :func:`strain_to_frequency_domain`: ``h(t) = irfft(H(f), n=n) / dt``.

    Args:
        h_freq (jax.Array): Complex non-negative-frequency spectrum, with shape (...,
            frequency).
        n (int): Number of time-domain samples to reconstruct (needed since a
            non-negative-frequency spectrum alone doesn't determine whether the original
            signal had an even or odd length).
        dt (jax.Array): Sample spacing, in seconds.

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
        h_time (jax.Array): Real time-domain signal, with shape (time,).

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
def spectral_derivative(x_time: jax.Array, dt: jax.Array) -> jax.Array:
    """
    Exact derivative of a uniformly-sampled (real or complex) time series, computed by
    spectral differentiation: FFT, multiply by ``i * omega``, inverse FFT. Exact for
    band-limited signals, unlike finite differences.

    Args:
        x_time (jax.Array): Time series, with shape (..., time).
        dt (jax.Array): Sample spacing, in seconds.

    Returns:
        jax.Array: The derivative, with shape (..., time).
    """
    n = x_time.shape[-1]
    omega = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dt)
    return jnp.fft.ifft(1.0j * omega * jnp.fft.fft(x_time, axis=-1), axis=-1)
