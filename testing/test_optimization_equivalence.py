"""
Tests to ensure optimized implementations produce identical results.

These tests verify that performance optimizations (vmap, static_argnums, etc.)
do not change the numerical output of the computations.
"""

import unittest
import jax.numpy as jnp
import numpy.testing as npt
import gw_response as gwr


class TestVectorizedEquivalence(unittest.TestCase):
    """Test that vectorized (vmap) implementations match loop-based ones."""

    def setUp(self):
        """Set up test fixtures."""
        self.lisa = gwr.LISA()
        self.pixel = gwr.Pixel(NSIDE=8)
        self.freqs = jnp.logspace(-4, -1, 50)
        self.times = jnp.array([0.0])
        self.response = gwr.Response(ps=gwr.PhysicalConstants(), det=self.lisa)

    def test_single_link_vectorized_matches_loop(self):
        """Test that vmap single_link matches loop version."""
        # Loop-based (original)
        result_loop = self.response.get_single_link_response(
            self.times,
            self.pixel.theta_pixel,
            self.pixel.phi_pixel,
            self.freqs,
            polarization="LR",
        )

        # Vectorized
        result_vmap = self.response.get_single_link_response_vectorized(
            self.times,
            self.pixel.theta_pixel,
            self.pixel.phi_pixel,
            self.freqs,
            polarization="LR",
        )

        # Compare
        for key in result_loop:
            npt.assert_allclose(
                result_loop[key],
                result_vmap[key],
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Mismatch in polarization {key}",
            )

    def test_linear_integrand_vectorized_matches_loop(self):
        """Test that vmap linear_integrand matches loop version."""
        single_link = self.response.get_single_link_response(
            self.times,
            self.pixel.theta_pixel,
            self.pixel.phi_pixel,
            self.freqs,
            polarization="LR",
        )

        # Loop-based
        result_loop = self.response.get_linear_integrand(
            self.times, single_link, self.freqs, TDI="XYZ", polarization="LR"
        )

        # Vectorized
        result_vmap = self.response.get_linear_integrand_vectorized(
            self.times, single_link, self.freqs, TDI="XYZ", polarization="LR"
        )

        for key in result_loop:
            npt.assert_allclose(
                result_loop[key],
                result_vmap[key],
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Mismatch in linear integrand {key}",
            )

    def test_quadratic_integrand_vectorized_matches_loop(self):
        """Test that vmap quadratic_integrand matches loop version."""
        single_link = self.response.get_single_link_response(
            self.times,
            self.pixel.theta_pixel,
            self.pixel.phi_pixel,
            self.freqs,
            polarization="LR",
        )

        # Loop-based
        result_loop = self.response.get_quadratic_integrand(
            self.times, single_link, self.freqs, TDI="XYZ", polarization="LR"
        )

        # Vectorized
        result_vmap = self.response.get_quadratic_integrand_vectorized(
            self.times, single_link, self.freqs, TDI="XYZ", polarization="LR"
        )

        for key in result_loop:
            npt.assert_allclose(
                result_loop[key],
                result_vmap[key],
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Mismatch in quadratic integrand {key}",
            )

    def test_compute_detector_vectorized_matches_original(self):
        """Test full pipeline equivalence."""
        # Original
        response_orig = gwr.Response(ps=gwr.PhysicalConstants(), det=self.lisa)
        response_orig.compute_detector(
            self.times,
            self.pixel.theta_pixel,
            self.pixel.phi_pixel,
            self.freqs,
            TDI="XYZ",
            polarization="LR",
        )

        # Vectorized
        response_vmap = gwr.Response(ps=gwr.PhysicalConstants(), det=self.lisa)
        response_vmap.compute_detector_vectorized(
            self.times,
            self.pixel.theta_pixel,
            self.pixel.phi_pixel,
            self.freqs,
            TDI="XYZ",
            polarization="LR",
        )

        # Compare final results
        for key in response_orig.quadratic_integrated["XYZ"]:
            npt.assert_allclose(
                response_orig.quadratic_integrated["XYZ"][key],
                response_vmap.quadratic_integrated["XYZ"][key],
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"Mismatch in quadratic_integrated[{key}]",
            )


class TestTDIOptions(unittest.TestCase):
    """Test that all TDI options work with optimizations."""

    def setUp(self):
        """Set up test fixtures."""
        self.freqs = jnp.logspace(-4, -1, 30)

    def test_all_tdi_options(self):
        """Test each TDI option produces valid results."""
        tdi_options = ["XYZ", "AET", "Sagnac", "AET_Sagnac", "AE_zeta", "AE_Sagnac_zeta"]

        for tdi in tdi_options:
            with self.subTest(tdi=tdi):
                result = gwr.compute_response(self.freqs, tdi=tdi)

                # Check for NaN/Inf
                for key, val in result.quadratic.items():
                    self.assertFalse(
                        jnp.any(jnp.isnan(val)), f"NaN found in {tdi}/{key}"
                    )
                    self.assertFalse(
                        jnp.any(jnp.isinf(val)), f"Inf found in {tdi}/{key}"
                    )


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of optimized implementations."""

    def test_high_frequency_stability(self):
        """Test stability at high frequencies."""
        # High frequencies near upper bound
        freqs = jnp.logspace(-2, -0.5, 50)

        result = gwr.compute_response(freqs, nside=8)

        for key, val in result.quadratic.items():
            self.assertFalse(jnp.any(jnp.isnan(val)), f"NaN found in {key}")
            self.assertFalse(jnp.any(jnp.isinf(val)), f"Inf found in {key}")

    def test_many_time_steps(self):
        """Test stability with many time configurations."""
        freqs = jnp.logspace(-4, -1, 30)
        times = jnp.linspace(0, 1, 50)

        result = gwr.compute_response(freqs, times=times, nside=8)

        for key, val in result.quadratic.items():
            self.assertFalse(jnp.any(jnp.isnan(val)), f"NaN found in {key}")
            self.assertFalse(jnp.any(jnp.isinf(val)), f"Inf found in {key}")

    def test_repeated_calls_consistency(self):
        """Ensure repeated calls produce identical results."""
        freqs = jnp.logspace(-4, -1, 30)

        results = []
        for _ in range(3):
            result = gwr.compute_response(freqs)
            results.append(result.LL)

        # All results should be identical
        for i in range(1, len(results)):
            npt.assert_array_equal(
                results[0], results[i], err_msg=f"Result {i} differs from result 0"
            )


class TestPolarizationOptions(unittest.TestCase):
    """Test both polarization options work correctly."""

    def setUp(self):
        """Set up test fixtures."""
        self.freqs = jnp.logspace(-4, -1, 30)

    def test_lr_polarization(self):
        """Test LR (circular) polarization."""
        result = gwr.compute_response(self.freqs, polarization="LR")

        self.assertIn("LL", result.quadratic)
        self.assertIn("RR", result.quadratic)
        self.assertIsNotNone(result.LL)
        self.assertIsNotNone(result.RR)

    def test_pc_polarization(self):
        """Test PC (linear) polarization."""
        result = gwr.compute_response(self.freqs, polarization="PC")

        self.assertIn("PP", result.quadratic)
        self.assertIn("CC", result.quadratic)
        self.assertIsNotNone(result.PP)
        self.assertIsNotNone(result.CC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
