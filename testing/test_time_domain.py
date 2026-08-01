import unittest
import jax
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


class TestInstantaneousFrequency(unittest.TestCase):
    def test_matches_analytical_derivative_for_exact_bin_tone(self):
        # Only exact for a tone that's exactly periodic in the FFT window
        # (an exact bin frequency) -- otherwise spectral leakage
        # contaminates the analytic-signal envelope/phase. The expected
        # frequency is the phase's own analytical time derivative
        # (autodiff'd via jax.grad), not just a hardcoded constant, so this
        # is a genuine cross-check of instantaneous_frequency's spectral
        # differentiation against a closed-form derivative.
        dt = 0.01
        n = 2048
        df = 1.0 / (n * dt)
        f0 = 76 * df

        def phase(t):
            return 2 * jnp.pi * f0 * t

        t = jnp.arange(n) * dt
        h = jnp.cos(phase(t))
        f_inst = gwr.instantaneous_frequency(h, dt)

        f_analytical = jax.vmap(jax.grad(phase))(t) / (2 * jnp.pi)
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(f_inst - f_analytical))), 0.0, places=9
        )


class TestTimeDomainResponseFrozen_LIGO(unittest.TestCase):
    def test_shapes_and_consistency(self):
        ligo = gwr.LIGO()
        response = ligo.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        dt = 1.0 / (4 * ligo.fmax)
        n = 4096
        t = jnp.arange(n) * dt
        h_plus = jnp.sin(2 * jnp.pi * 100.0 * t)
        h_cross = jnp.zeros(n)

        d_t = response.get_response_frozen_td(
            ligo, 0.0, theta, phi, h_plus, h_cross, dt
        )
        self.assertEqual(d_t.shape, (1, 1, n))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_t))))

        # Manual cross-check via the (separately tested) linear_integrand
        freqs = jnp.fft.rfftfreq(n, d=dt)
        H_plus = gwr.strain_to_frequency_domain(h_plus, dt)
        H_cross = gwr.strain_to_frequency_domain(h_cross, dt)

        linear = response.get_linear_integrand_fd(
            ligo, 0.0, theta, phi, freqs, polarization="PC"
        )
        R_plus = jnp.moveaxis(linear["P"], 1, -1)
        R_cross = jnp.moveaxis(linear["C"], 1, -1)
        expected_f = R_plus * H_plus + R_cross * H_cross
        expected_t = gwr.frequency_domain_to_time_domain(expected_f, n, dt)
        self.assertAlmostEqual(float(jnp.max(jnp.abs(d_t - expected_t))), 0.0, places=8)

    def test_multiple_times_and_pixels(self):
        # Each (time, pixel) slice of a batched call should match calling
        # get_response_frozen_td individually for just that configuration.
        ligo = gwr.LIGO()
        response = ligo.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:3], pixel.phi_pixel[:3]
        times = jnp.array([0.0, 0.1])

        dt = 1.0 / (4 * ligo.fmax)
        n = 512
        t = jnp.arange(n) * dt
        h_plus = jnp.sin(2 * jnp.pi * 100.0 * t)
        h_cross = jnp.zeros(n)

        d_batched = response.get_response_frozen_td(
            ligo, times, theta, phi, h_plus, h_cross, dt
        )
        self.assertEqual(d_batched.shape, (2, 3, n))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_batched))))

        d_single = response.get_response_frozen_td(
            ligo, times[1:2], theta[2:3], phi[2:3], h_plus, h_cross, dt
        )
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(d_batched[1, 2] - d_single[0, 0]))), 0.0, places=10
        )


class TestTimeDomainResponseFrozen_LISA(unittest.TestCase):
    def test_shapes(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        dt = 10.0
        n = 2048
        t = jnp.arange(n) * dt
        h_plus = jnp.sin(2 * jnp.pi * 1e-2 * t)
        h_cross = jnp.cos(2 * jnp.pi * 1e-2 * t)

        d_t = response.get_response_frozen_td(
            lisa, 0.0, theta, phi, h_plus, h_cross, dt, combination="XYZ"
        )
        self.assertEqual(d_t.shape, (1, 3, 1, n))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_t))))


class TestInstantaneousTimeDomainResponseSpectral(unittest.TestCase):
    """
    Cross-checks Response.get_response_spectral_td (which
    evaluates the detector's own evolving configuration and instantaneous
    frequency at every sample, from the Hilbert-transform/analytic-signal
    envelope of real h_plus/h_cross arrays) against the already-tested
    FFT-based get_response_frozen_td: per the requirement that the two
    genuinely-different methods agree whenever the detector's configuration
    doesn't change appreciably over the signal's duration.
    """

    def test_ligo_matches_fft_method_exactly(self):
        # LIGO's geometry doesn't depend on time_in_years at all (a static,
        # non-rotating frame in this codebase), so any uniformly-spaced
        # times_in_years grid is, by construction, the frozen-geometry
        # case -- an exact (not just approximate) cross-check. f0 is
        # chosen on an exact FFT bin so h_plus/h_cross are exactly periodic
        # in the sample window, with no analytic-signal leakage either.
        ligo = gwr.LIGO()
        response = ligo.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        dt = 1.0 / (4 * ligo.fmax)
        n = 4096
        df = 1.0 / (n * dt)
        f0 = 101 * df
        t = jnp.arange(n) * dt
        h_plus = jnp.cos(2 * jnp.pi * f0 * t)
        h_cross = 0.3 * jnp.sin(2 * jnp.pi * f0 * t)

        times_in_years = t / ligo.ps.yr
        d_instantaneous = response.get_response_spectral_td(
            ligo, times_in_years, theta, phi, h_plus, h_cross
        )
        self.assertEqual(d_instantaneous.shape, (n,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_instantaneous))))

        d_fft = response.get_response_frozen_td(
            ligo, 0.0, theta, phi, h_plus, h_cross, dt
        )[0, 0]

        max_err = float(jnp.max(jnp.abs(d_instantaneous - d_fft)))
        scale = float(jnp.max(jnp.abs(d_fft)))
        self.assertLess(max_err / scale, 1e-6)

    def test_two_tones_as_separate_modes_matches_fft(self):
        # Two well-separated tones, on exact FFT bins so each is exactly
        # periodic in the sample window (no leakage). Passed as two
        # independent modes (shape (2, n)) rather than summed into one
        # h_plus array, each mode's Hilbert-transform envelope and
        # instantaneous frequency are computed independently, so this
        # matches the FFT method just as exactly as the single-tone case
        # above -- unlike summing them first, which breaks the analytic
        # signal's instantaneous frequency (see the next test).
        ligo = gwr.LIGO()
        response = ligo.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        dt = 1.0 / (4 * ligo.fmax)
        n = 4096
        df = 1.0 / (n * dt)
        f1 = 101 * df
        f2 = 401 * df
        t = jnp.arange(n) * dt

        h_plus_modes = jnp.stack(
            [jnp.cos(2 * jnp.pi * f1 * t), jnp.cos(2 * jnp.pi * f2 * t)]
        )
        h_cross_modes = jnp.zeros((2, n))

        times_in_years = t / ligo.ps.yr
        d_modes = response.get_response_spectral_td(
            ligo, times_in_years, theta, phi, h_plus_modes, h_cross_modes
        )
        self.assertEqual(d_modes.shape, (n,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_modes))))

        h_plus_summed = h_plus_modes[0] + h_plus_modes[1]
        h_cross_summed = jnp.zeros(n)
        d_fft = response.get_response_frozen_td(
            ligo, 0.0, theta, phi, h_plus_summed, h_cross_summed, dt
        )[0, 0]

        max_err = float(jnp.max(jnp.abs(d_modes - d_fft)))
        scale = float(jnp.max(jnp.abs(d_fft)))
        self.assertLess(max_err / scale, 1e-6)

    def test_two_tones_summed_into_one_mode_breaks_down(self):
        # The same two tones as above, but summed into a single (time,)
        # h_plus before extracting one instantaneous frequency from it --
        # demonstrates why the previous test passes them as separate modes
        # instead. LIGO's geometry never moves, so any disagreement here is
        # entirely due to this method's narrowband assumption, not orbital
        # motion.
        ligo = gwr.LIGO()
        response = ligo.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        dt = 1.0 / (4 * ligo.fmax)
        n = 4096
        df = 1.0 / (n * dt)
        f1 = 101 * df
        f2 = 401 * df
        t = jnp.arange(n) * dt

        h_plus = jnp.cos(2 * jnp.pi * f1 * t) + jnp.cos(2 * jnp.pi * f2 * t)
        h_cross = jnp.zeros(n)

        times_in_years = t / ligo.ps.yr
        d_instantaneous = response.get_response_spectral_td(
            ligo, times_in_years, theta, phi, h_plus, h_cross
        )
        d_fft = response.get_response_frozen_td(
            ligo, 0.0, theta, phi, h_plus, h_cross, dt
        )[0, 0]

        max_err = float(jnp.max(jnp.abs(d_instantaneous - d_fft)))
        scale = float(jnp.max(jnp.abs(d_fft)))
        self.assertGreater(max_err / scale, 0.1)  # a real, large breakdown

    @staticmethod
    def _lisa_relative_error(response, lisa, theta, phi, t_ref_years, n, dt, f0):
        t = jnp.arange(n) * dt
        h_plus = jnp.cos(2 * jnp.pi * f0 * t)
        h_cross = 0.5 * jnp.sin(2 * jnp.pi * f0 * t)
        times_in_years = t_ref_years + t / lisa.ps.yr

        d_instantaneous = response.get_response_spectral_td(
            lisa, times_in_years, theta, phi, h_plus, h_cross, combination="XYZ"
        )
        d_fft = response.get_response_frozen_td(
            lisa, t_ref_years, theta, phi, h_plus, h_cross, dt, combination="XYZ"
        )[0, :, 0]

        max_err = float(jnp.max(jnp.abs(d_instantaneous - d_fft)))
        scale = float(jnp.max(jnp.abs(d_fft)))
        return max_err / scale, d_instantaneous

    def test_lisa_matches_fft_method_short_duration(self):
        # Over a duration much shorter than LISA's orbital period, the
        # constellation's motion is negligible, so the two methods should
        # still agree to high precision.
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        dt = 0.5
        n = 128  # ~1 minute total
        df = 1.0 / (n * dt)
        f0 = round(0.05 / df) * df

        rel_err, d_instantaneous = self._lisa_relative_error(
            response, lisa, theta, phi, 0.3, n, dt, f0
        )
        self.assertEqual(d_instantaneous.shape, (3, n))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_instantaneous))))
        self.assertLess(rel_err, 1e-3)

    def test_lisa_deviation_scales_linearly_with_duration(self):
        # The only difference from get_response_frozen_td in this
        # monochromatic, otherwise-frozen setup is that the instantaneous
        # method evaluates the detector's actual (slightly) evolving
        # configuration over times_in_years' span instead of freezing it
        # at t_ref_years -- so to leading order in the (tiny) orbital-phase
        # change over a short span, the discrepancy between the two methods
        # should scale linearly with duration. Checking that scaling
        # (rather than just bounding the error at one duration) confirms
        # it's a genuine, well-behaved geometric effect, not a numerical
        # artifact -- see the shape of :meth:`_lisa_relative_error`'s
        # output at several durations, which was used to establish this
        # scaling empirically.
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        dt = 0.5
        target_f0 = 0.05

        def rel_err_at(n):
            df = 1.0 / (n * dt)
            f0 = round(target_f0 / df) * df
            err, _ = self._lisa_relative_error(
                response, lisa, theta, phi, 0.3, n, dt, f0
            )
            return err

        err_long = rel_err_at(2048)  # 1024 s
        err_short = rel_err_at(1024)  # 512 s -- half the duration

        self.assertGreater(err_long, 1e-4)  # a real, measurable effect
        ratio = err_short / err_long
        self.assertGreater(ratio, 0.35)  # halving duration ~halves the error
        self.assertLess(ratio, 0.65)

    def test_lisa_real_motion_gives_small_but_nonzero_deviation(self):
        # Now span a duration long enough for LISA's orbital motion to
        # matter (half a year) -- the instantaneous method should visibly
        # deviate from the frozen-geometry FFT method, but only by a
        # modest amount.
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        n = 2048
        dt = (0.5 * lisa.ps.yr) / n
        df = 1.0 / (n * dt)
        f0 = 101 * df
        t = jnp.arange(n) * dt
        h_plus = jnp.cos(2 * jnp.pi * f0 * t)
        h_cross = 0.5 * jnp.sin(2 * jnp.pi * f0 * t)

        t_ref_years = 0.0
        times_in_years = t_ref_years + t / lisa.ps.yr
        d_instantaneous = response.get_response_spectral_td(
            lisa, times_in_years, theta, phi, h_plus, h_cross, combination="XYZ"
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_instantaneous))))

        d_fft = response.get_response_frozen_td(
            lisa, t_ref_years, theta, phi, h_plus, h_cross, dt, combination="XYZ"
        )[0, :, 0]
        scale = float(jnp.max(jnp.abs(d_fft)))
        deviation = float(jnp.max(jnp.abs(d_instantaneous - d_fft))) / scale
        self.assertGreater(deviation, 1e-3)  # actually captures the motion
        self.assertLess(deviation, 1.0)  # but stays a modest correction

    def test_multiple_sky_positions_raises(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:3], pixel.phi_pixel[:3]

        dt = 100.0
        n = 256
        t = jnp.arange(n) * dt
        h_plus = jnp.sin(2 * jnp.pi * 1e-3 * t)
        h_cross = jnp.cos(2 * jnp.pi * 1e-3 * t)
        times_in_years = t / lisa.ps.yr

        with self.assertRaises(ValueError):
            response.get_response_spectral_td(
                lisa, times_in_years, theta, phi, h_plus, h_cross, combination="XYZ"
            )


class TestInstantaneousTimeDomainResponseAutodiff(unittest.TestCase):
    """
    Cross-checks Response.get_response_autodiff_td --
    the autodiff-exact sibling of get_response_spectral_td, taking
    amplitude/phase as JAX callables instead of sampled arrays, with no FFT
    anywhere -- against both the array-based method (fed samples of the
    same closed-form waveform) and the FFT-based get_response_frozen_td.
    """

    def test_ligo_matches_array_and_fft_methods(self):
        ligo = gwr.LIGO()
        response = ligo.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        dt = 1.0 / (4 * ligo.fmax)
        n = 4096
        df = 1.0 / (n * dt)
        f0 = 101 * df

        def phase(t):
            return 2 * jnp.pi * f0 * t

        def amplitude_plus(t):
            return 1.0

        def amplitude_cross(t):
            return 0.3

        t = jnp.arange(n) * dt
        times_in_years = t / ligo.ps.yr

        d_autodiff = response.get_response_autodiff_td(
            ligo, times_in_years, theta, phi, amplitude_plus, amplitude_cross, phase
        )
        self.assertEqual(d_autodiff.shape, (n,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_autodiff))))

        h_plus = jnp.cos(phase(t))
        h_cross = 0.3 * jnp.sin(phase(t))
        d_array = response.get_response_spectral_td(
            ligo, times_in_years, theta, phi, h_plus, h_cross
        )
        d_fft = response.get_response_frozen_td(
            ligo, 0.0, theta, phi, h_plus, h_cross, dt
        )[0, 0]

        scale = float(jnp.max(jnp.abs(d_fft)))
        self.assertLess(float(jnp.max(jnp.abs(d_autodiff - d_array))) / scale, 1e-6)
        self.assertLess(float(jnp.max(jnp.abs(d_autodiff - d_fft))) / scale, 1e-6)

    def test_lisa_chirp_captures_real_motion(self):
        # A genuinely time-varying (chirping) phase -- exercises jax.grad
        # giving a per-sample instantaneous frequency that itself varies,
        # with no FFT/windowing anywhere in the computation.
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        n = 2048
        dt = (0.5 * lisa.ps.yr) / n
        df = 1.0 / (n * dt)
        f0 = 101 * df
        fdot = f0 / (n * dt)  # a modest frequency drift over the signal

        def phase(t):
            return 2 * jnp.pi * (f0 * t + 0.5 * fdot * t**2)

        def amplitude_plus(t):
            return 1.0

        def amplitude_cross(t):
            return 0.5

        t = jnp.arange(n) * dt
        times_in_years = t / lisa.ps.yr

        d_autodiff = response.get_response_autodiff_td(
            lisa,
            times_in_years,
            theta,
            phi,
            amplitude_plus,
            amplitude_cross,
            phase,
            combination="XYZ",
        )
        self.assertEqual(d_autodiff.shape, (3, n))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_autodiff))))

        d_fft = response.get_response_frozen_td(
            lisa,
            0.0,
            theta,
            phi,
            jnp.cos(phase(t)),
            0.5 * jnp.sin(phase(t)),
            dt,
            combination="XYZ",
        )[0, :, 0]
        scale = float(jnp.max(jnp.abs(d_fft)))
        deviation = float(jnp.max(jnp.abs(d_autodiff - d_fft))) / scale
        self.assertGreater(deviation, 1e-3)  # actually captures the motion
        self.assertLess(deviation, 1.0)  # but stays a modest correction

    def test_multiple_sky_positions_raises(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:3], pixel.phi_pixel[:3]

        n = 256
        dt = 100.0
        t = jnp.arange(n) * dt
        times_in_years = t / lisa.ps.yr

        with self.assertRaises(ValueError):
            response.get_response_autodiff_td(
                lisa,
                times_in_years,
                theta,
                phi,
                lambda t: 1.0,
                lambda t: 0.5,
                lambda t: 2 * jnp.pi * 1e-3 * t,
                combination="XYZ",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
