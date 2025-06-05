import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np
from gw_response.ligo import _SITE_GEOMETRIES

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data_ligo/")
LIGOHAN_ARM1   = _SITE_GEOMETRIES['Hanford']['arm1']
LIGOHAN_ARM2   = _SITE_GEOMETRIES['Hanford']['arm2']
LIGOHAN_C      = _SITE_GEOMETRIES['Hanford']['center']

class TestLIGO(unittest.TestCase):

    def test_ligo_satellite_positions(self):
        ligo_analytical_positions = gwr.LIGO_satellite_positions(
        1, LIGOHAN_C , LIGOHAN_ARM1, LIGOHAN_ARM2, 4.0e3)
        
        save_arr = np.load(TEST_DATA_PATH + "ligo_satellite_positions.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(ligo_analytical_positions  - save_arr)) / np.max(save_arr),
            0.0,
        )

    def test_ligo_arms_matrix(self):
        ligo_arms_matrix=gwr.LIGO_arms_matrix(
            1, LIGOHAN_ARM1, LIGOHAN_ARM2, 4.0e3
        )
        save_arr = np.load(TEST_DATA_PATH + "arms_matrix.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(ligo_arms_matrix - save_arr)) / np.max(save_arr),
            0.0,
        )
        arm_lengths = jnp.sqrt(jnp.einsum("tij,tij->tj", ligo_arms_matrix, ligo_arms_matrix))

        save_arr = np.load(TEST_DATA_PATH + "arm_lengths.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(arm_lengths - save_arr)) / np.max(save_arr),
            0.0,
        )

    def test_lisa_class(self):
        ligo = gwr.LIGO()
        frequencies = ligo.frequency_vec(10)
        save_arr = np.load(TEST_DATA_PATH + "frequencies.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(frequencies - save_arr)), 0.0)
        kl_vector = ligo.klvector(frequencies)
        save_arr = np.load(TEST_DATA_PATH + "kl_vector.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(kl_vector - save_arr)), 0.0)
        x_vector = ligo.x(frequencies)
        save_arr = np.load(TEST_DATA_PATH + "x_vector.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(x_vector - save_arr)), 0.0)
        satellite_positions = ligo.satellite_positions(1)
        save_arr = np.load(TEST_DATA_PATH + "ligo_satellite_positions.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(satellite_positions - save_arr)) / np.max(save_arr),
            0.0,
        )
        arms_matrix = ligo.detector_arms(1)
        save_arr = np.load(TEST_DATA_PATH + "arms_matrix.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(arms_matrix - save_arr)) / np.max(save_arr),
            0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
