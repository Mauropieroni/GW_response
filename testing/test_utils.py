import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")


class TestUtils(unittest.TestCase):
    def test_pixels(self):
        pixel_class = gwr.Pixel(NSIDE=8)
        theta, phi = pixel_class.theta_pixel, pixel_class.phi_pixel
        save_arr = np.load(TEST_DATA_PATH + "pixel_theta.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(theta - save_arr)), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "pixel_phi.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(phi - save_arr)), 0.0)
        angular_map = pixel_class.angular_map
        save_arr = np.load(TEST_DATA_PATH + "pixel_angular_map.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(angular_map - save_arr)), 0.0)
        pixel_class.change_NSIDE(16)
        theta, phi = pixel_class.theta_pixel, pixel_class.phi_pixel
        save_arr = np.load(TEST_DATA_PATH + "pixel_theta_16.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(theta - save_arr)), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "pixel_phi_16.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(phi - save_arr)), 0.0)
        angular_map = pixel_class.angular_map
        save_arr = np.load(TEST_DATA_PATH + "pixel_angular_map_16.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(angular_map - save_arr)), 0.0)

    def test_arm_length_exp(self):
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        arm_length_exp = gwr.arm_length_exponential(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "arm_length_exp.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(arm_length_exp - save_arr)), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
