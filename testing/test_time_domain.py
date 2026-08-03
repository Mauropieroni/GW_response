import os
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import gw_response as gwr

TEST_DATA_PATH_lisa = os.path.join(os.path.dirname(__file__), "test_data_lisa/")


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
        response.waveform = gwr.Waveform(
            strain_td=lambda t, params: (
                jnp.sin(2 * jnp.pi * 100.0 * t) + 0j,
                jnp.zeros_like(t) + 0j,
            )
        )

        d_t = response.get_response_frozen_td(ligo, 0.0, theta, phi, None, n, dt)
        self.assertEqual(d_t.shape, (n, 1, 1))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_t))))

        # Manual cross-check via the (separately tested) linear_integrand
        t = jnp.arange(n) * dt
        h_plus = jnp.sin(2 * jnp.pi * 100.0 * t)
        h_cross = jnp.zeros(n)
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
        expected_t = jnp.moveaxis(expected_t, -1, 0)
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
        response.waveform = gwr.Waveform(
            strain_td=lambda t, params: (
                jnp.sin(2 * jnp.pi * 100.0 * t) + 0j,
                jnp.zeros_like(t) + 0j,
            )
        )

        d_batched = response.get_response_frozen_td(
            ligo, times, theta, phi, None, n, dt
        )
        self.assertEqual(d_batched.shape, (n, 2, 3))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_batched))))

        d_single = response.get_response_frozen_td(
            ligo, times[1:2], theta[2:3], phi[2:3], None, n, dt
        )
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(d_batched[:, 1, 2] - d_single[:, 0, 0]))),
            0.0,
            places=10,
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
        response.waveform = gwr.Waveform(
            strain_td=lambda t, params: (
                jnp.sin(2 * jnp.pi * 1e-2 * t) + 0j,
                jnp.cos(2 * jnp.pi * 1e-2 * t) + 0j,
            )
        )

        d_t = response.get_response_frozen_td(
            lisa, 0.0, theta, phi, None, n, dt, combination="XYZ"
        )
        self.assertEqual(d_t.shape, (n, 1, 3, 1))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_t))))

    def test_strain_fd_matches_strain_td(self):
        # A Waveform with both strain_td and strain_fd set to the same
        # underlying signal (one exact-FFT of the other, on this method's
        # own jnp.fft.rfftfreq(n, d=dt) grid) should give the same response
        # regardless of which path get_response_frozen_td takes -- strain_fd
        # (used directly, no FFT) should win when both are set.
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        dt = 10.0
        n = 2048
        t_grid = jnp.arange(n) * dt
        h_plus_td = jnp.sin(2 * jnp.pi * 1e-2 * t_grid)
        h_cross_td = jnp.cos(2 * jnp.pi * 1e-2 * t_grid)
        h_f_plus = gwr.strain_to_frequency_domain(h_plus_td, dt)
        h_f_cross = gwr.strain_to_frequency_domain(h_cross_td, dt)

        response.waveform = gwr.Waveform(
            strain_td=lambda t, params: (h_plus_td + 0j, h_cross_td + 0j)
        )
        d_from_td = response.get_response_frozen_td(
            lisa, 0.0, theta, phi, None, n, dt, combination="XYZ"
        )

        response.waveform = gwr.Waveform(
            strain_fd=lambda f, params: (h_f_plus, h_f_cross)
        )
        d_from_fd = response.get_response_frozen_td(
            lisa, 0.0, theta, phi, None, n, dt, combination="XYZ"
        )

        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(d_from_td - d_from_fd))), 0.0, places=10
        )

    def test_waveform_without_strain_raises(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        response.waveform = gwr.Waveform()  # neither strain_td nor strain_fd set
        with self.assertRaises(ValueError):
            response.get_response_frozen_td(lisa, 0.0, theta, phi, None, 128, 10.0)


class TestSingleLinkDelayRetardedSegmentedTD(unittest.TestCase):
    """
    Regression coverage for Response.get_single_link_response_delay_td (including its
    frozen-geometry special case, via a constant `times_geometry_years`) and
    get_single_link_response_segmented_td, against golden-snapshot fixtures of their
    own output, confirmed exact to machine precision.
    """

    def test_delay_td_matches_reference(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        times = jnp.linspace(0.0, 0.01, 50)

        def amplitude_plus(t, params):
            return jnp.asarray(1e-21)

        def amplitude_cross(t, params):
            return jnp.asarray(0.5e-21)

        def phase(t, params):
            return 2 * jnp.pi * 1e-2 * t

        response.waveform = gwr.Waveform.from_amplitude_phase(
            amplitude_plus, amplitude_cross, phase
        )

        d_t = response.get_single_link_response_delay_td(
            lisa, times, theta, phi, None
        )
        self.assertEqual(d_t.shape, (50, 6))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_t))))

        save_arr = np.load(TEST_DATA_PATH_lisa + "single_link_response_delay_td.npy")
        self.assertAlmostEqual(float(jnp.max(jnp.abs(d_t - save_arr))), 0.0, places=12)

    def test_reassigning_waveform_is_not_stale_cached(self):
        # get_single_link_response_delay_td/get_single_link_response_segmented_td
        # read self.waveform via a deliberately unjitted wrapper specifically
        # so that reassigning it between calls is always picked up, rather
        # than risking a jax.jit cache hit keyed off self's identity (which
        # doesn't change when only self.waveform's *content* does) silently
        # returning the first call's stale result. This exercises exactly
        # that: same response object, same call arguments, two different
        # waveforms.
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        times = jnp.linspace(0.0, 0.01, 50)
        waveform_a = gwr.Waveform.from_amplitude_phase(
            amplitude_plus=lambda t, params: jnp.asarray(1e-21),
            amplitude_cross=lambda t, params: jnp.asarray(0.5e-21),
            phase=lambda t, params: jnp.asarray(2 * jnp.pi * 1e-2 * t),
        )
        waveform_b = gwr.Waveform.from_amplitude_phase(
            amplitude_plus=lambda t, params: jnp.asarray(3e-21),
            amplitude_cross=lambda t, params: jnp.asarray(2e-21),
            phase=lambda t, params: jnp.asarray(2 * jnp.pi * 3e-2 * t + 0.7),
        )

        for method_name, kwargs in (
            ("get_single_link_response_delay_td", {}),
            ("get_single_link_response_segmented_td", {"segment_length": 10}),
        ):
            method = getattr(response, method_name)

            response.waveform = waveform_a
            d_a_first = method(lisa, times, theta, phi, None, **kwargs)
            response.waveform = waveform_b
            d_b = method(lisa, times, theta, phi, None, **kwargs)
            response.waveform = waveform_a
            d_a_second = method(lisa, times, theta, phi, None, **kwargs)

            with self.subTest(method=method_name):
                # Re-running with the original waveform reproduces the
                # original output exactly (no cross-call contamination).
                self.assertAlmostEqual(
                    float(jnp.max(jnp.abs(d_a_first - d_a_second))), 0.0, places=12
                )
                # A different waveform gives a genuinely different result
                # (not the first call's cached output).
                self.assertGreater(float(jnp.max(jnp.abs(d_b - d_a_first))), 0.0)

    def test_waveform_not_set_raises(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]
        times = jnp.linspace(0.0, 0.01, 50)

        response.waveform = None
        with self.assertRaises(ValueError):
            response.get_single_link_response_delay_td(lisa, times, theta, phi, None)
        with self.assertRaises(ValueError):
            response.get_single_link_response_segmented_td(
                lisa, times, theta, phi, None, segment_length=10
            )

    def test_segmented_td_matches_reference(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        times = jnp.linspace(0.0, 0.01, 50)

        def amplitude_plus(t, params):
            return jnp.asarray(1e-21)

        def amplitude_cross(t, params):
            return jnp.asarray(0.5e-21)

        def phase(t, params):
            return 2 * jnp.pi * 1e-2 * t

        response.waveform = gwr.Waveform.from_amplitude_phase(
            amplitude_plus, amplitude_cross, phase
        )

        d_t = response.get_single_link_response_segmented_td(
            lisa,
            times,
            theta,
            phi,
            None,
            segment_length=10,
        )
        self.assertEqual(d_t.shape, (50, 6))
        self.assertTrue(bool(jnp.all(jnp.isfinite(d_t))))

        save_arr = np.load(
            TEST_DATA_PATH_lisa + "single_link_response_segmented_td.npy"
        )
        self.assertAlmostEqual(float(jnp.max(jnp.abs(d_t - save_arr))), 0.0, places=12)

        # Also cross-checks against the exact delay_td reference it
        # approximates -- segment_length=10 should already track it closely.
        delay_reference = np.load(
            TEST_DATA_PATH_lisa + "single_link_response_delay_td.npy"
        )
        rel_err = float(
            jnp.max(jnp.abs(d_t - delay_reference)) / jnp.max(jnp.abs(delay_reference))
        )
        self.assertLess(rel_err, 1e-4)


class TestTDIResponseDelayTD(unittest.TestCase):
    """
    Regression coverage for Response.get_response_delay_td -- the TDI 1.5
    (unequal but locally-constant arms) time-domain combination, exact for evolving
    geometry, from Muratore, Vetrugno & Vitale (arXiv:2303.15929, eq. 2.24).
    """

    def _lisa_with_waveform(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        def amplitude_plus(t, params):
            return jnp.asarray(1e-21)

        def amplitude_cross(t, params):
            return jnp.asarray(0.5e-21)

        def phase(t, params):
            return 2 * jnp.pi * 3e-3 * t

        response.waveform = gwr.Waveform.from_amplitude_phase(
            amplitude_plus, amplitude_cross, phase
        )
        return lisa, response, theta, phi

    def test_zeta_matches_independent_manual_construction(self):
        # Rebuilds zeta = D12(eta31-eta32) + D23(eta12-eta13) + D31(eta23-eta21)
        # (eq. 2.24c) from scratch here, using only the already-tested
        # get_single_link_response_delay_td/detector_arms_retarded -- an
        # independent check of get_response_delay_td's own internal term
        # tables/orchestration, not just a re-run of the same code.
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.005, 40)

        _, ltt, _ = lisa.detector_arms_retarded(times, lisa.response.ps)
        arm_order = (12, 23, 31, 21, 32, 13)
        ltt_by_arm = {label: ltt[:, i] for i, label in enumerate(arm_order)}

        def arm_at(shift_labels, arm_label):
            delay = (
                sum(ltt_by_arm[label] for label in shift_labels)
                if shift_labels
                else 0.0
            )
            shifted = times - delay / lisa.ps.yr
            y = response.get_single_link_response_delay_td(
                lisa, shifted, theta, phi, None
            )
            return y[:, arm_order.index(arm_label)]

        zeta_manual = (
            (arm_at((12,), 31) - arm_at((12,), 32))
            + (arm_at((23,), 12) - arm_at((23,), 13))
            + (arm_at((31,), 23) - arm_at((31,), 21))
        )

        d_t = response.get_response_delay_td(
            lisa, times, theta, phi, None, combination="AE_zeta"
        )
        zeta_from_method = d_t[:, 2]

        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(zeta_manual - zeta_from_method))), 0.0, places=10
        )

    def test_shapes_and_finite_for_all_combinations(self):
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.005, 30)

        for combination in (
            "XYZ",
            "AET",
            "Sagnac",
            "AET_Sagnac",
            "AE_zeta",
            "AE_Sagnac_zeta",
        ):
            with self.subTest(combination=combination):
                d_t = response.get_response_delay_td(
                    lisa, times, theta, phi, None, combination=combination
                )
                self.assertEqual(d_t.shape, (30, 3))
                self.assertTrue(bool(jnp.all(jnp.isfinite(d_t))))

    def test_unknown_combination_raises(self):
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.005, 10)
        with self.assertRaises(ValueError):
            response.get_response_delay_td(
                lisa, times, theta, phi, None, combination="not_a_combination"
            )

    def test_waveform_not_set_raises(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]
        times = jnp.linspace(0.0, 0.005, 10)

        response.waveform = None
        with self.assertRaises(ValueError):
            response.get_response_delay_td(lisa, times, theta, phi, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
