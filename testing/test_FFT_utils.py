import unittest
import jax.numpy as jnp
import numpy as np
import gw_response as gwr


class TestTimeDomainRoundTrip(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        dt = 0.1
        n = 256
        h = jnp.asarray(rng.normal(size=n))
        H = gwr.strain_to_frequency_domain(h, dt)
        h_rec = gwr.frequency_domain_to_time_domain(H, n, dt)
        self.assertAlmostEqual(float(jnp.max(jnp.abs(h_rec - h))), 0.0, places=10)

    def test_round_trip_odd_length(self):
        rng = np.random.default_rng(1)
        dt = 0.25
        n = 257
        h = jnp.asarray(rng.normal(size=n))
        H = gwr.strain_to_frequency_domain(h, dt)
        self.assertEqual(H.shape[0], n // 2 + 1)
        h_rec = gwr.frequency_domain_to_time_domain(H, n, dt)
        self.assertAlmostEqual(float(jnp.max(jnp.abs(h_rec - h))), 0.0, places=10)

    def test_strain_to_frequency_domain_matches_scaled_rfft(self):
        # Independent check of strain_to_frequency_domain's normalization convention
        # (H(f) = dt * rfft(h(t))) against numpy's rfft
        rng = np.random.default_rng(4)
        dt = 0.05
        n = 128
        h = jnp.asarray(rng.normal(size=n))
        H = gwr.strain_to_frequency_domain(h, dt)
        H_expected = np.fft.rfft(np.asarray(h)) * dt
        self.assertAlmostEqual(float(jnp.max(jnp.abs(H - H_expected))), 0.0, places=12)

    def test_frequency_domain_to_time_domain_matches_scaled_irfft(self):
        rng = np.random.default_rng(5)
        dt = 0.05
        n = 128
        H = jnp.asarray(
            rng.normal(size=n // 2 + 1) + 1.0j * rng.normal(size=n // 2 + 1)
        )
        h = gwr.frequency_domain_to_time_domain(H, n, dt)
        h_expected = np.fft.irfft(np.asarray(H), n=n) / dt
        self.assertAlmostEqual(float(jnp.max(jnp.abs(h - h_expected))), 0.0, places=12)


class TestFftPositiveTimeAndFreqs(unittest.TestCase):
    def test_matches_complex_exponential_for_exact_bin_tone(self):
        # For an exact-bin cosine tone (no spectral leakage), the analytic
        # signal is exactly the complex exponential at that frequency.
        dt = 0.01
        n = 512
        df = 1.0 / (n * dt)
        f0 = 12.0 * df
        t = jnp.arange(n) * dt
        h = jnp.cos(2.0 * jnp.pi * f0 * t)

        analytic = gwr.fft_positive_time_and_freqs(h)
        expected = jnp.exp(1.0j * 2.0 * jnp.pi * f0 * t)

        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(analytic - expected))), 0.0, places=9
        )

    def test_real_part_reconstructs_input(self):
        # h(t) = Re[fft_positive_time_and_freqs(h)(t)] exactly, by
        # construction -- true for any real input, not just an exact-bin
        # tone.
        rng = np.random.default_rng(6)
        h = jnp.asarray(rng.normal(size=100))
        analytic = gwr.fft_positive_time_and_freqs(h)
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(jnp.real(analytic) - h))), 0.0, places=10
        )


class TestSpectralDerivative(unittest.TestCase):
    def test_matches_closed_form_derivative_for_exact_bin_tone(self):
        # d/dt exp(i*2*pi*f0*t) = i*2*pi*f0*exp(i*2*pi*f0*t) exactly, for an
        # exact-bin complex tone (band-limited to a single FFT bin, so
        # spectral differentiation is exact rather than approximate).
        dt = 0.01
        n = 512
        df = 1.0 / (n * dt)
        f0 = 9.0 * df
        t = jnp.arange(n) * dt
        x = jnp.exp(1.0j * 2.0 * jnp.pi * f0 * t)

        derivative = gwr.spectral_derivative(x, dt)
        expected = 1.0j * 2.0 * jnp.pi * f0 * x

        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(derivative - expected))), 0.0, places=9
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
