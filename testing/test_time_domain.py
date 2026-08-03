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


class TestSingleLinkDelayRetardedSegmentedTD(unittest.TestCase):
    """
    Regression coverage for Response.get_single_link_response_delay_td and
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

        def amplitude_plus(t, waveform_params):
            return jnp.asarray(1e-21)

        def amplitude_cross(t, waveform_params):
            return jnp.asarray(0.5e-21)

        def phase(t, waveform_params):
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
            amplitude_plus=lambda t, waveform_params: jnp.asarray(1e-21),
            amplitude_cross=lambda t, waveform_params: jnp.asarray(0.5e-21),
            phase=lambda t, waveform_params: jnp.asarray(2 * jnp.pi * 1e-2 * t),
        )
        waveform_b = gwr.Waveform.from_amplitude_phase(
            amplitude_plus=lambda t, waveform_params: jnp.asarray(3e-21),
            amplitude_cross=lambda t, waveform_params: jnp.asarray(2e-21),
            phase=lambda t, waveform_params: jnp.asarray(2 * jnp.pi * 3e-2 * t + 0.7),
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

        def amplitude_plus(t, waveform_params):
            return jnp.asarray(1e-21)

        def amplitude_cross(t, waveform_params):
            return jnp.asarray(0.5e-21)

        def phase(t, waveform_params):
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
    Regression coverage for Response.get_response_delay_td -- the TDI 1.5 (unequal
    but locally-constant arms) and 2.0 time-domain combinations, exact for evolving
    geometry, from Muratore, Vetrugno & Vitale (arXiv:2303.15929, eqs. 2.24 and 2.23).
    """

    def _lisa_with_waveform(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        def amplitude_plus(t, waveform_params):
            return jnp.asarray(1e-21)

        def amplitude_cross(t, waveform_params):
            return jnp.asarray(0.5e-21)

        def phase(t, waveform_params):
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

    def test_unknown_tdi_order_raises(self):
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.005, 10)
        with self.assertRaises(ValueError):
            response.get_response_delay_td(
                lisa, times, theta, phi, None, combination="XYZ", tdi_order=1.0
            )

    def test_tdi2_X_matches_independent_manual_construction(self):
        # Rebuilds X2 = X(t) - X(t - 2*(T31+T12)) (eq. 2.23a) from scratch
        # here, using only the already-tested tdi_order=1.5 path and
        # detector_arms_retarded -- an independent check of
        # get_response_delay_td's own TDI 2.0 prefactor machinery, not just
        # a re-run of the same code.
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.001, 15)

        _, ltt, _ = lisa.detector_arms_retarded(times, lisa.response.ps)
        arm_order = (12, 23, 31, 21, 32, 13)
        ltt_by_arm = {label: ltt[:, i] for i, label in enumerate(arm_order)}

        delay = 2 * (ltt_by_arm[31] + ltt_by_arm[12])
        shifted_times = times - delay / lisa.ps.yr

        X_now = response.get_response_delay_td(
            lisa, times, theta, phi, None, combination="XYZ", tdi_order=1.5
        )[:, 0]
        X_shifted = response.get_response_delay_td(
            lisa, shifted_times, theta, phi, None, combination="XYZ", tdi_order=1.5
        )[:, 0]
        X2_manual = X_now - X_shifted

        d_t = response.get_response_delay_td(
            lisa, times, theta, phi, None, combination="XYZ", tdi_order=2.0
        )
        X2_from_method = d_t[:, 0]

        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(X2_manual - X2_from_method))), 0.0, places=10
        )

    def test_shapes_and_finite_for_tdi_order_2(self):
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.001, 15)

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
                    lisa,
                    times,
                    theta,
                    phi,
                    None,
                    combination=combination,
                    tdi_order=2.0,
                )
                self.assertEqual(d_t.shape, (15, 3))
                self.assertTrue(bool(jnp.all(jnp.isfinite(d_t))))

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


class TestTDIResponseSegmentedTD(unittest.TestCase):
    """
    Regression coverage for Response.get_response_segmented_td -- the TDI 1.5
    combination built from segment-stacked single-link terms.
    """

    def _lisa_with_waveform(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        def amplitude_plus(t, waveform_params):
            return jnp.asarray(1e-21)

        def amplitude_cross(t, waveform_params):
            return jnp.asarray(0.5e-21)

        def phase(t, waveform_params):
            return 2 * jnp.pi * 3e-3 * t

        response.waveform = gwr.Waveform.from_amplitude_phase(
            amplitude_plus, amplitude_cross, phase
        )
        return lisa, response, theta, phi

    def test_zeta_matches_independent_manual_construction(self):
        # Same independent reconstruction as
        # TestTDIResponseDelayTD.test_zeta_matches_independent_manual_construction,
        # but built from get_single_link_response_segmented_td instead -- an
        # independent check of get_response_segmented_td's own term-table
        # orchestration, not just a re-run of the same code.
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.005, 40)
        segment_length = 4

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
            y = response.get_single_link_response_segmented_td(
                lisa, shifted, theta, phi, None, segment_length=segment_length
            )
            return y[:, arm_order.index(arm_label)]

        zeta_manual = (
            (arm_at((12,), 31) - arm_at((12,), 32))
            + (arm_at((23,), 12) - arm_at((23,), 13))
            + (arm_at((31,), 23) - arm_at((31,), 21))
        )

        d_t = response.get_response_segmented_td(
            lisa, times, theta, phi, None, segment_length, combination="AE_zeta"
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
                d_t = response.get_response_segmented_td(
                    lisa, times, theta, phi, None, 10, combination=combination
                )
                self.assertEqual(d_t.shape, (30, 3))
                self.assertTrue(bool(jnp.all(jnp.isfinite(d_t))))

    def test_matches_delay_td_reference(self):
        # Cross-checks against the exact delay_td path it approximates --
        # segment_length=10 should already track it closely.
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.005, 30)

        segmented = response.get_response_segmented_td(
            lisa, times, theta, phi, None, 10, combination="XYZ"
        )
        delay = response.get_response_delay_td(
            lisa, times, theta, phi, None, combination="XYZ"
        )
        rel_err = float(
            jnp.max(jnp.abs(segmented - delay)) / jnp.max(jnp.abs(delay))
        )
        self.assertLess(rel_err, 1e-4)

    def test_unknown_combination_raises(self):
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.005, 10)
        with self.assertRaises(ValueError):
            response.get_response_segmented_td(
                lisa, times, theta, phi, None, 5, combination="not_a_combination"
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
            response.get_response_segmented_td(lisa, times, theta, phi, None, 5)


class TestGetResponseDispatcher(unittest.TestCase):
    """
    Regression coverage for Response.get_response -- a thin dispatcher, so each
    case checks it reproduces the direct call to the underlying method exactly.
    """

    def _lisa_with_waveform(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        def amplitude_plus(t, waveform_params):
            return jnp.asarray(1e-21)

        def amplitude_cross(t, waveform_params):
            return jnp.asarray(0.5e-21)

        def phase(t, waveform_params):
            return 2 * jnp.pi * 3e-3 * t

        response.waveform = gwr.Waveform.from_amplitude_phase(
            amplitude_plus, amplitude_cross, phase
        )
        return lisa, response, theta, phi

    def test_td_delay_matches_direct_call(self):
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.001, 15)

        via_dispatcher = response.get_response(
            lisa,
            theta,
            phi,
            None,
            which_domain="TD",
            which_TDI="AET",
            TDI_order=2.0,
            times_in_years=times,
        )
        direct = response.get_response_delay_td(
            lisa, times, theta, phi, None, combination="AET", tdi_order=2.0
        )
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(via_dispatcher - direct))), 0.0, places=12
        )

    def test_td_segmented_matches_direct_call(self):
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.001, 20)

        via_dispatcher = response.get_response(
            lisa,
            theta,
            phi,
            None,
            which_domain="TD",
            which_method="segmented",
            which_TDI="AET",
            times_in_years=times,
            segment_length=5,
        )
        direct = response.get_response_segmented_td(
            lisa, times, theta, phi, None, 5, combination="AET"
        )
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(via_dispatcher - direct))), 0.0, places=12
        )

    def test_segmented_missing_segment_length_raises(self):
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.001, 15)
        with self.assertRaises(ValueError):
            response.get_response(
                lisa,
                theta,
                phi,
                None,
                which_domain="TD",
                which_method="segmented",
                times_in_years=times,
            )

    def test_segmented_with_tdi_order_2_raises(self):
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.001, 15)
        with self.assertRaises(ValueError):
            response.get_response(
                lisa,
                theta,
                phi,
                None,
                which_domain="TD",
                which_method="segmented",
                times_in_years=times,
                segment_length=5,
                TDI_order=2.0,
            )

    def test_unknown_which_method_raises(self):
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.001, 15)
        with self.assertRaises(ValueError):
            response.get_response(
                lisa,
                theta,
                phi,
                None,
                which_domain="TD",
                which_method="not_a_method",
                times_in_years=times,
            )

    def test_fd_matches_direct_call(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:1], pixel.phi_pixel[:1]

        freqs = jnp.logspace(-4, -1, 50)
        h_f_plus = jnp.ones_like(freqs, dtype=complex) * 1e-21
        h_f_cross = jnp.ones_like(freqs, dtype=complex) * 0.5e-21
        response.waveform = gwr.Waveform(
            strain_fd=lambda f, waveform_params: (h_f_plus, h_f_cross)
        )

        via_dispatcher = response.get_response(
            lisa,
            theta,
            phi,
            None,
            which_domain="FD",
            which_TDI="XYZ",
            times_in_years=0.0,
            frequency_array=freqs,
        )
        direct = response.get_response_fd(
            lisa, 0.0, theta, phi, None, freqs, combination="XYZ"
        )
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(via_dispatcher - direct))), 0.0, places=12
        )

    def test_missing_required_args_raise(self):
        lisa, response, theta, phi = self._lisa_with_waveform()
        with self.assertRaises(ValueError):
            response.get_response(lisa, theta, phi, None, which_domain="TD")
        with self.assertRaises(ValueError):
            response.get_response(lisa, theta, phi, None, which_domain="FD")

    def test_unknown_which_domain_raises(self):
        lisa, response, theta, phi = self._lisa_with_waveform()
        times = jnp.linspace(0.0, 0.001, 15)
        with self.assertRaises(ValueError):
            response.get_response(
                lisa,
                theta,
                phi,
                None,
                which_domain="not_a_domain",
                times_in_years=times,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
