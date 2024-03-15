import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")


class TestSingleLink(unittest.TestCase):
    def test_unit_vector(self):
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        save_arr = np.load(TEST_DATA_PATH + "unit_vector.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(unit_vector - save_arr)), 0.0)

    def test_uv(self):
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        u, v = gwr.uv_analytical(theta, phi)
        save_arr = np.load(TEST_DATA_PATH + "u.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(u - save_arr)), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "v.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(v - save_arr)), 0.0)
        e1, e2 = gwr.polarization_vectors(u, v)
        save_arr = np.load(TEST_DATA_PATH + "e1.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(e1 - save_arr)), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "e2.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(e2 - save_arr)), 0.0)
        e1p, e1c = gwr.polarization_tensors_PC(u, v)
        e1L, e1R = gwr.polarization_tensors_LR(u, v)
        save_arr = np.load(TEST_DATA_PATH + "e1p.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(e1p - save_arr)), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "e1c.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(e1c - save_arr)), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "e1L.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(e1L - save_arr)), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "e1R.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(e1R - save_arr)), 0.0)

    def test_xi_k(self):
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        time_in_years = jnp.linspace(0, 1.0, 100)
        xi_k = gwr.xi_k_no_G(
            unit_wavevector=unit_vector,
            x_vector=lisa.x(freqs),
            arms_mat_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
        )
        save_arr = np.load(TEST_DATA_PATH + "xi_k.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(xi_k - save_arr)), 0.0)

    def test_position_exp(self):
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        sat_positions = lisa.satellite_positions(0.0)[0]
        p1, p2, p3 = (
            sat_positions[:, 0],
            sat_positions[:, 1],
            sat_positions[:, 2],
        )
        sp1, sp2, sp3 = gwr.shift_to_center(p1, p2, p3)
        position_exp = gwr.position_exponential(
            positions_detector_frame=jnp.array([[sp1, sp2, sp3]])
            / lisa.armlength,
            unit_wavevector=unit_vector,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "position_exp.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(position_exp - save_arr)),
            0.0,
        )

    def test_geometrical_factor(self):
        lisa = gwr.LISA()
        time_in_years = jnp.linspace(0, 1.0, 100)
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        u, v = gwr.uv_analytical(theta, phi)
        e1L, _ = gwr.polarization_tensors_LR(u, v)
        geomtrical_factor = gwr.geometrical_factor(
            arms_matrix=lisa.detector_arms(0.0) / lisa.armlength,
            polarization_tensor=e1L,
        )
        save_arr = np.load(TEST_DATA_PATH + "geometrical_factor.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(geomtrical_factor - save_arr)), 0.0
        )

    def test_xi_k_Avec(self):
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        time_in_years = jnp.linspace(0, 1.0, 100)
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        u, v = gwr.uv_analytical(theta, phi)
        e1L, _ = gwr.polarization_tensors_LR(u, v)
        geomtrical_factor = gwr.geometrical_factor(
            arms_matrix=lisa.detector_arms(0.0) / lisa.armlength,
            polarization_tensor=e1L,
        )
        xi_k_Avec = gwr.xi_k_Avec_func(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            unit_wavevector=unit_vector,
            x_vector=lisa.x(freqs),
            geometrical=geomtrical_factor,
        )
        save_arr = np.load(TEST_DATA_PATH + "xi_k_Avec.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(xi_k_Avec - save_arr)), 0.0)

    def test_single_link_response(self):
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        time_in_years = jnp.linspace(0, 1.0, 100)
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        u, v = gwr.uv_analytical(theta, phi)
        e1L, _ = gwr.polarization_tensors_LR(u, v)
        geomtrical_factor = gwr.geometrical_factor(
            arms_matrix=lisa.detector_arms(0.0) / lisa.armlength,
            polarization_tensor=e1L,
        )
        xi_k_Avec = gwr.xi_k_Avec_func(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            unit_wavevector=unit_vector,
            x_vector=lisa.x(freqs),
            geometrical=geomtrical_factor,
        )
        single_link_response = gwr.single_link_response(
            positions=lisa.satellite_positions(0.0) / lisa.armlength,
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            wavevector=unit_vector,
            x_vector=lisa.x(freqs),
            xi_k_Avec=xi_k_Avec,
        )
        save_arr = np.load(TEST_DATA_PATH + "single_link_response.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(single_link_response - save_arr)), 0.0
        )
        linear_response_tdi = gwr.linear_response_angular(
            TDI_idx=0,  # XYZ basis
            single_link=single_link_response,
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "linear_response_tdi.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(linear_response_tdi - save_arr)), 0.0
        )
        quadratic_angular_response = gwr.response_angular(
            TDI_idx=0,  # XYZ basis
            single_link=single_link_response,
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "quadratic_angular_response.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(quadratic_angular_response - save_arr)), 0.0
        )
        integrated_response = gwr.response_integrated(
            quadratic_angular_response
        )
        save_arr = np.load(TEST_DATA_PATH + "integrated_response.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(integrated_response - save_arr)), 0.0
        )
        quadratic_angular_response_AET = gwr.response_angular(
            TDI_idx=1,  # AET basis
            single_link=single_link_response,
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        integrated_response = gwr.response_integrated(
            quadratic_angular_response_AET
        )
        save_arr = np.load(TEST_DATA_PATH + "integrated_response_AET.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(integrated_response - save_arr)), 0.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
