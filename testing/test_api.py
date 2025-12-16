"""Tests for the high-level API."""

import unittest
import jax.numpy as jnp
import gw_response as gwr


class TestComputeResponse(unittest.TestCase):
    """Tests for compute_response function."""

    def test_basic_call(self):
        """Test basic compute_response call with defaults."""
        freqs = jnp.logspace(-4, -1, 50)
        result = gwr.compute_response(freqs)

        self.assertIsInstance(result, gwr.ResponseResult)
        self.assertEqual(len(result.frequencies), 50)
        self.assertEqual(result.tdi, "AET")
        self.assertEqual(result.polarization, "LR")
        self.assertIn("LL", result.quadratic)
        self.assertIn("RR", result.quadratic)

    def test_xyz_tdi(self):
        """Test XYZ TDI option."""
        freqs = jnp.logspace(-4, -1, 50)
        result = gwr.compute_response(freqs, tdi="XYZ")

        self.assertEqual(result.tdi, "XYZ")

    def test_pc_polarization(self):
        """Test PC polarization option."""
        freqs = jnp.logspace(-4, -1, 50)
        result = gwr.compute_response(freqs, polarization="PC")

        self.assertEqual(result.polarization, "PC")
        self.assertIn("PP", result.quadratic)
        self.assertIn("CC", result.quadratic)

    def test_time_float(self):
        """Test that single float time works."""
        freqs = jnp.logspace(-4, -1, 50)
        result = gwr.compute_response(freqs, times=0.5)

        self.assertEqual(len(result.times), 1)
        self.assertAlmostEqual(float(result.times[0]), 0.5)

    def test_time_array(self):
        """Test that time array works."""
        freqs = jnp.logspace(-4, -1, 50)
        times = jnp.array([0.0, 0.25, 0.5])
        result = gwr.compute_response(freqs, times=times)

        self.assertEqual(len(result.times), 3)

    def test_numerical_equivalence(self):
        """Ensure new API gives same results as old API."""
        freqs = jnp.logspace(-4, -1, 50)

        # New API
        new_result = gwr.compute_response(freqs, tdi="XYZ", polarization="LR")

        # Old API
        response = gwr.Response(ps=gwr.PhysicalConstants(), det=gwr.LISA())
        pixel = gwr.Pixel()
        response.compute_detector(
            times_in_years=jnp.array([0.0]),
            theta_array=pixel.theta_pixel,
            phi_array=pixel.phi_pixel,
            frequency_array=freqs,
            TDI="XYZ",
            polarization="LR",
        )
        old_result = response.quadratic_integrated["XYZ"]["LL"]

        # Compare
        diff = jnp.sum(jnp.abs(new_result.LL - old_result))
        self.assertAlmostEqual(float(diff), 0.0)

    def test_include_intermediate(self):
        """Test that include_intermediate provides extra data."""
        freqs = jnp.logspace(-4, -1, 30)
        result = gwr.compute_response(freqs, include_intermediate=True)

        self.assertIsNotNone(result.linear)
        self.assertIsNotNone(result.single_link)
        self.assertIsNotNone(result.quadratic_integrand)


class TestQuickResponse(unittest.TestCase):
    """Tests for quick_response function."""

    def test_defaults(self):
        """Test quick_response with all defaults."""
        result = gwr.quick_response()

        self.assertEqual(len(result.frequencies), 100)
        self.assertEqual(result.tdi, "AET")

    def test_custom_range(self):
        """Test quick_response with custom frequency range."""
        result = gwr.quick_response(1e-3, 1e-2, 20)

        self.assertEqual(len(result.frequencies), 20)
        self.assertAlmostEqual(float(result.frequencies[0]), 1e-3, places=10)
        self.assertAlmostEqual(float(result.frequencies[-1]), 1e-2, places=10)


class TestResponseResult(unittest.TestCase):
    """Tests for ResponseResult dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        freqs = jnp.logspace(-4, -1, 30)
        self.result = gwr.compute_response(freqs)

    def test_polarization_properties(self):
        """Test LL and RR properties."""
        self.assertIsNotNone(self.result.LL)
        self.assertIsNotNone(self.result.RR)
        self.assertEqual(self.result.LL.shape, self.result.RR.shape)

    def test_tdi_properties(self):
        """Test AA, EE, TT properties."""
        self.assertIsNotNone(self.result.AA)
        self.assertIsNotNone(self.result.EE)
        self.assertIsNotNone(self.result.TT)

    def test_diagonal(self):
        """Test diagonal extraction."""
        diag = self.result.diagonal()
        # Should be (n_times, n_freq, n_tdi)
        self.assertEqual(len(diag.shape), 3)
        self.assertEqual(diag.shape[-1], 3)  # 3 TDI channels

    def test_sum_polarizations(self):
        """Test polarization sum."""
        total = self.result.sum_polarizations()
        expected = self.result.LL + self.result.RR

        diff = jnp.sum(jnp.abs(total - expected))
        self.assertAlmostEqual(float(diff), 0.0)

    def test_at_frequency(self):
        """Test frequency extraction."""
        freq_target = float(self.result.frequencies[15])
        result_at_freq = self.result.at_frequency(freq_target)

        self.assertEqual(len(result_at_freq.frequencies), 1)

    def test_shape_property(self):
        """Test shape property."""
        shape = self.result.shape
        self.assertEqual(len(shape), 4)
        self.assertEqual(shape[0], 1)  # n_times
        self.assertEqual(shape[1], 30)  # n_freq
        self.assertEqual(shape[2], 3)  # n_tdi
        self.assertEqual(shape[3], 3)  # n_tdi

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.result)
        self.assertIn("ResponseResult", repr_str)
        self.assertIn("LISA", repr_str)
        self.assertIn("AET", repr_str)


class TestPresets(unittest.TestCase):
    """Tests for presets module."""

    def test_tdi_guide_keys(self):
        """Test that all TDI options are documented."""
        expected_keys = {"XYZ", "AET", "Sagnac", "AET_Sagnac", "AE_zeta", "AE_Sagnac_zeta"}
        self.assertEqual(set(gwr.TDI_GUIDE.keys()), expected_keys)

    def test_get_tdi_info(self):
        """Test get_tdi_info function."""
        info = gwr.get_tdi_info("AET")
        self.assertEqual(info.name, "AET (Orthogonal)")
        self.assertEqual(info.channels, ("A", "E", "T"))

    def test_get_tdi_info_invalid(self):
        """Test get_tdi_info with invalid TDI."""
        with self.assertRaises(ValueError):
            gwr.get_tdi_info("INVALID")

    def test_lisa_band(self):
        """Test LISA_BAND constant."""
        self.assertEqual(gwr.LISA_BAND, (3e-5, 0.5))


class TestParallelOption(unittest.TestCase):
    """Tests for parallel computation option."""

    def test_parallel_false(self):
        """Test that parallel=False (default) works."""
        freqs = jnp.logspace(-4, -1, 30)
        result = gwr.compute_response(freqs, parallel=False)

        self.assertIsInstance(result, gwr.ResponseResult)
        self.assertIn("LL", result.quadratic)

    def test_parallel_true(self):
        """Test that parallel=True works (may use single device)."""
        freqs = jnp.logspace(-4, -1, 30)
        result = gwr.compute_response(freqs, parallel=True)

        self.assertIsInstance(result, gwr.ResponseResult)
        self.assertIn("LL", result.quadratic)

    def test_parallel_equivalence(self):
        """Test that parallel and serial produce same results."""
        freqs = jnp.logspace(-4, -1, 30)

        result_serial = gwr.compute_response(freqs, parallel=False)
        result_parallel = gwr.compute_response(freqs, parallel=True)

        # Check results are numerically equivalent
        for key in result_serial.quadratic:
            diff = jnp.max(jnp.abs(result_serial.quadratic[key] - result_parallel.quadratic[key]))
            self.assertLess(float(diff), 1e-10, f"Mismatch in {key}")


class TestPerformanceExports(unittest.TestCase):
    """Tests for performance module exports."""

    def test_config_exports(self):
        """Test config module exports are available."""
        self.assertTrue(callable(gwr.configure_for_performance))
        self.assertTrue(callable(gwr.get_device_info))
        self.assertTrue(callable(gwr.print_device_info))

    def test_parallel_exports(self):
        """Test parallel module exports are available."""
        self.assertTrue(callable(gwr.parallel_compute_response))
        self.assertTrue(callable(gwr.get_device_count))
        self.assertTrue(callable(gwr.get_parallel_info))

    def test_benchmark_module(self):
        """Test benchmark module is importable."""
        self.assertTrue(hasattr(gwr, 'benchmark'))
        self.assertTrue(hasattr(gwr.benchmark, 'Benchmark'))
        self.assertTrue(hasattr(gwr.benchmark, 'BenchmarkSuite'))

    def test_get_device_info(self):
        """Test get_device_info returns valid structure."""
        info = gwr.get_device_info()
        self.assertIn('n_devices', info)
        self.assertIn('device_type', info)
        self.assertIn('devices', info)
        self.assertGreaterEqual(info['n_devices'], 1)

    def test_get_parallel_info(self):
        """Test get_parallel_info returns valid structure."""
        info = gwr.get_parallel_info()
        self.assertIn('n_devices', info)
        self.assertIn('parallel_enabled', info)
        self.assertIn('devices', info)


if __name__ == "__main__":
    unittest.main()
