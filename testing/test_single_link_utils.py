import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH_lisa = os.path.join(os.path.dirname(__file__), "test_data_lisa/")
TEST_DATA_PATH_ligo = os.path.join(os.path.dirname(__file__), "test_data_ligo/")


class TestSingleLinkUtils(unittest.TestCase):
    def test_position_exp(self):
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        sat_positions = lisa.vertex_positions(0.0)[0]
        p1, p2, p3 = (
            sat_positions[:, 0],
            sat_positions[:, 1],
            sat_positions[:, 2],
        )
        sp1, sp2, sp3 = gwr.shift_to_center(p1, p2, p3)
        position_exp = gwr.position_exponential(
            positions_detector_frame_rescaled=jnp.array([[sp1, sp2, sp3]])
            / lisa.armlength,
            unit_wavevector=unit_vector,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH_lisa + "position_exp.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(position_exp - save_arr))),
            0.0,
        )

    def test_geometrical_factor(self):
        lisa = gwr.LISA()
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        u, v = gwr.uv_analytical(theta, phi)
        e1L, _ = gwr.polarization_tensors_LR(u, v)
        geomtrical_factor = gwr.geometrical_factor(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            polarization_tensor=e1L,
        )
        save_arr = np.load(TEST_DATA_PATH_lisa + "geometrical_factor.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(geomtrical_factor - save_arr))), 0.0
        )


class TestSingleLinkUtils_ligo(unittest.TestCase):
    def test_position_exp(self):
        ligo = gwr.LIGO()
        freqs = jnp.logspace(1, 5, 1000)
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        sat_positions = ligo.vertex_positions(0.0)[0]
        p1, p2, p3 = sat_positions[:, 0], sat_positions[:, 1], sat_positions[:, 2]
        sp1, sp2, sp3 = gwr.shift_to_center(p1, p2, p3)

        position_exp = gwr.position_exponential(
            positions_detector_frame_rescaled=jnp.array([[sp1, sp2, sp3]])
            / ligo.armlength,
            unit_wavevector=unit_vector,
            x_vector=ligo.x(freqs),
        )

        save_arr = np.load(TEST_DATA_PATH_ligo + "position_exp.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(position_exp - save_arr))), 0.0)

    def test_geometrical_factor(self):
        ligo = gwr.LIGO()
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        u, v = gwr.uv_analytical(theta, phi)
        e1L, _ = gwr.polarization_tensors_LR(u, v)
        geomtrical_factor = gwr.geometrical_factor(
            arms_matrix_rescaled=ligo.detector_arms(0.0) / ligo.armlength,
            polarization_tensor=e1L,
        )
        save_arr = np.load(TEST_DATA_PATH_ligo + "geometrical_factor.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(geomtrical_factor - save_arr))), 0.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
