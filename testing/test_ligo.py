import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np
from gw_response.ground_based.ligo import _SITE_GEOMETRIES
from gw_response.ground_based.datastream import detector_output

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data_ligo/")
LIGOHAN_ARM1 = _SITE_GEOMETRIES["Hanford"]["arm1"]
LIGOHAN_ARM2 = _SITE_GEOMETRIES["Hanford"]["arm2"]
LIGOHAN_C = _SITE_GEOMETRIES["Hanford"]["center"]


class TestLIGO(unittest.TestCase):

    def test_ligo_positions(self):
        ligo_analytical_positions = gwr.LIGO_positions(
            LIGOHAN_C, LIGOHAN_ARM1, LIGOHAN_ARM2, 4.0e3, jnp.zeros(1)
        )

        save_arr = np.load(TEST_DATA_PATH + "ligo_positions.npy")
        self.assertAlmostEqual(
            jnp.max(jnp.abs(ligo_analytical_positions - save_arr)) / np.max(save_arr),
            0.0,
        )

    def test_ligo_arms_matrix(self):
        ligo_arms_matrix = gwr.LIGO_arms_matrix(
            LIGOHAN_ARM1, LIGOHAN_ARM2, 4.0e3, jnp.zeros(1)
        )
        save_arr = np.load(TEST_DATA_PATH + "arms_matrix.npy")
        self.assertAlmostEqual(
            jnp.max(jnp.abs(ligo_arms_matrix - save_arr)) / np.max(save_arr),
            0.0,
        )
        arm_lengths = jnp.sqrt(
            jnp.einsum("tij,tij->tj", ligo_arms_matrix, ligo_arms_matrix)
        )

        save_arr = np.load(TEST_DATA_PATH + "arm_lengths.npy")
        self.assertAlmostEqual(
            jnp.max(jnp.abs(arm_lengths - save_arr)) / np.max(save_arr),
            0.0,
        )

    def test_ligo_detector_output(self):
        # LIGO's Michelson-combination mixing matrix, the analog of LISA's
        # tdi_*_matrix functions (see test_tdi.py::test_TDI_matrices).
        ligo = gwr.LIGO()
        freqs = jnp.logspace(1, 5, 1000)
        mix_matrix = detector_output(
            arms_matrix_rescaled=ligo.detector_arms(0.0) / ligo.armlength,
            x_vector=ligo.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "detector_output.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(mix_matrix - save_arr))), 0.0)

    def test_ligo_class(self):
        ligo = gwr.LIGO()
        frequencies = ligo.frequency_vec(10)
        save_arr = np.load(TEST_DATA_PATH + "frequencies.npy")
        self.assertAlmostEqual(float(jnp.max(jnp.abs(frequencies - save_arr))), 0.0)
        kl_vector = ligo.klvector(frequencies)
        save_arr = np.load(TEST_DATA_PATH + "kl_vector.npy")
        self.assertAlmostEqual(float(jnp.max(jnp.abs(kl_vector - save_arr))), 0.0)
        x_vector = ligo.x(frequencies)
        save_arr = np.load(TEST_DATA_PATH + "x_vector.npy")
        self.assertAlmostEqual(float(jnp.max(jnp.abs(x_vector - save_arr))), 0.0)
        vertex_positions = ligo.vertex_positions(1)
        save_arr = np.load(TEST_DATA_PATH + "ligo_positions.npy")
        self.assertAlmostEqual(
            jnp.max(jnp.abs(vertex_positions - save_arr)) / np.max(save_arr),
            0.0,
        )
        arms_matrix = ligo.detector_arms(1)
        save_arr = np.load(TEST_DATA_PATH + "arms_matrix.npy")
        self.assertAlmostEqual(
            jnp.max(jnp.abs(arms_matrix - save_arr)) / np.max(save_arr),
            0.0,
        )


class TestLIGOEarthRotation(unittest.TestCase):
    def test_disabled_by_default_and_time_independent(self):
        ligo = gwr.LIGO()
        self.assertFalse(ligo.include_earth_rotation)
        p0 = ligo.vertex_positions(0.0)
        p1 = ligo.vertex_positions(1.0)
        self.assertAlmostEqual(float(jnp.max(jnp.abs(p0 - p1))), 0.0, places=12)

    def test_agrees_with_static_at_t_zero(self):
        ligo_static = gwr.LIGO(include_earth_rotation=False)
        ligo_rot = gwr.LIGO(include_earth_rotation=True)
        p_static = ligo_static.vertex_positions(0.0)
        p_rot = ligo_rot.vertex_positions(0.0)
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(p_static - p_rot))), 0.0, places=12
        )

    def test_returns_to_start_after_one_sidereal_day(self):
        ligo = gwr.LIGO(include_earth_rotation=True)
        sidereal_day_years = ligo.ps.sidereal_day / ligo.ps.yr
        p0 = ligo.vertex_positions(0.0)
        p1 = ligo.vertex_positions(sidereal_day_years)
        self.assertAlmostEqual(float(jnp.max(jnp.abs(p0 - p1))), 0.0, places=6)

    def test_half_day_rotates_180_degrees_about_z(self):
        # A rotation by pi about the ECEF z-axis flips the x/y components
        # and leaves z unchanged.
        ligo = gwr.LIGO(include_earth_rotation=True)
        half_day_years = 0.5 * ligo.ps.sidereal_day / ligo.ps.yr
        p0 = ligo.vertex_positions(0.0)[0]
        p_half = ligo.vertex_positions(half_day_years)[0]
        expected = p0 * jnp.array([-1.0, -1.0, 1.0])[:, None]
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(p_half - expected))), 0.0, places=6
        )


class TestLIGOLongWavelengthApproximation(unittest.TestCase):
    def test_matches_standard_antenna_pattern(self):
        # The long-wavelength response should equal the standard
        # 0.5*(arm1 outer arm1 - arm2 outer arm2) : e_pol antenna-pattern
        # contraction (the same convention used by gw_fast/gw_fish), to
        # machine precision.
        ligo = gwr.LIGO(long_wavelength_approximation=True)
        response = ligo.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        freqs = jnp.array([50.0])

        linear = response.get_linear_integrand_fd(
            ligo, 0.0, theta, phi, freqs, polarization="PC", combination="Michelson"
        )
        P_response = linear["P"][0, 0]
        C_response = linear["C"][0, 0]

        D = 0.5 * (jnp.outer(ligo.arm1, ligo.arm1) - jnp.outer(ligo.arm2, ligo.arm2))
        u, v = gwr.uv_analytical(theta, phi)
        e_plus, e_cross = gwr.polarization_tensors_PC(u, v)
        F_plus = jnp.einsum("ij,pij->p", D, e_plus)
        F_cross = jnp.einsum("ij,pij->p", D, e_cross)

        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(P_response - F_plus))), 0.0, places=12
        )
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(C_response - F_cross))), 0.0, places=12
        )

    def test_is_frequency_independent(self):
        ligo = gwr.LIGO(long_wavelength_approximation=True)
        response = ligo.response
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:10], pixel.phi_pixel[:10]

        linear_lo = response.get_linear_integrand_fd(
            ligo, 0.0, theta, phi, jnp.array([1.0]), polarization="PC"
        )
        linear_hi = response.get_linear_integrand_fd(
            ligo, 0.0, theta, phi, jnp.array([1900.0]), polarization="PC"
        )
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(linear_lo["P"] - linear_hi["P"]))), 0.0, places=12
        )

    def test_disabled_by_default(self):
        ligo = gwr.LIGO()
        self.assertFalse(ligo.long_wavelength_approximation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
