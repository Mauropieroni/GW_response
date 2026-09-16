import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")


class TestPolarization(unittest.TestCase):
    def test_unit_vector(self):
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        save_arr = np.load(TEST_DATA_PATH + "unit_vector.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(unit_vector - save_arr))), 0.0)

    def test_uv(self):
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        u, v = gwr.uv_analytical(theta, phi)
        save_arr = np.load(TEST_DATA_PATH + "u.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(u - save_arr))), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "v.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(v - save_arr))), 0.0)
        e1, e2 = gwr.polarization_vectors(u, v)
        save_arr = np.load(TEST_DATA_PATH + "e1.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(e1 - save_arr))), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "e2.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(e2 - save_arr))), 0.0)
        e1p, e1c = gwr.polarization_tensors_PC(u, v)
        e1L, e1R = gwr.polarization_tensors_LR(u, v)
        save_arr = np.load(TEST_DATA_PATH + "e1p.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(e1p - save_arr))), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "e1c.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(e1c - save_arr))), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "e1L.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(e1L - save_arr))), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "e1R.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(e1R - save_arr))), 0.0)

    def test_polarization_angles_convenience_functions(self):
        # The "_angles" functions are convenience wrappers that build (u, v)
        # from (theta, phi) internally, so a caller never has to touch that
        # intermediate representation. They must match chaining the
        # already-tested raw functions by hand.
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        u, v = gwr.uv_analytical(theta, phi)

        e1, e2 = gwr.polarization_vectors(u, v)
        e1_angles, e2_angles = gwr.polarization_vectors_angles(theta, phi)
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(e1_angles - e1))), 0.0)
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(e2_angles - e2))), 0.0)

        e1p, e1c = gwr.polarization_tensors_PC(u, v)
        e1p_angles, e1c_angles = gwr.polarization_tensors_PC_angles(theta, phi)
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(e1p_angles - e1p))), 0.0)
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(e1c_angles - e1c))), 0.0)

        e1L, e1R = gwr.polarization_tensors_LR(u, v)
        e1L_angles, e1R_angles = gwr.polarization_tensors_LR_angles(theta, phi)
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(e1L_angles - e1L))), 0.0)
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(e1R_angles - e1R))), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
