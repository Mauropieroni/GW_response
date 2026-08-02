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
            1, LIGOHAN_C, LIGOHAN_ARM1, LIGOHAN_ARM2, 4.0e3
        )

        save_arr = np.load(TEST_DATA_PATH + "ligo_positions.npy")
        self.assertAlmostEqual(
            jnp.max(jnp.abs(ligo_analytical_positions - save_arr)) / np.max(save_arr),
            0.0,
        )

    def test_ligo_arms_matrix(self):
        ligo_arms_matrix = gwr.LIGO_arms_matrix(1, LIGOHAN_ARM1, LIGOHAN_ARM2, 4.0e3)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
