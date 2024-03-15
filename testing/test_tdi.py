import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")


class TestTDI(unittest.TestCase):
    def test_sine_factors(self):
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        sine_factors = gwr.sin_factors(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "sine_factors.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(sine_factors - save_arr)), 0.0)

    def test_TDI_matrices(self):
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        tdi_XYZ = gwr.tdi_XYZ_matrix(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_XYZ.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_XYZ - save_arr)), 0.0)
        tdi_zeta = gwr.tdi_zeta_matrix(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_zeta.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_zeta - save_arr)), 0.0)
        tdi_Sagnac = gwr.tdi_Sagnac_matrix(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_Sagnac.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_Sagnac - save_arr)), 0.0)
        tdi_AET = gwr.tdi_AET_matrix(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_AET.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_AET - save_arr)), 0.0)
        tdi_AET_Sagnac = gwr.tdi_AET_Sagnac_matrix(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_AET_Sagnac.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_AET_Sagnac - save_arr)), 0.0)

        tdi_matrix = gwr.tdi_matrix(
            TDI_idx=0,  # XYZ basis
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_matrix_XYZ.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_matrix - save_arr)), 0.0)

        tdi_matrix = gwr.tdi_matrix(
            TDI_idx=gwr.TDI_map["AET_Sagnac"],  # AET Sagnac basis
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_matrix_AET_Sagnac.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_matrix - save_arr)), 0.0)

    def test_TDI_projection(self):
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
        tdi_projection = gwr.build_tdi(
            TDI_idx=gwr.TDI_map["XYZ"],  # XYZ basis
            single_link=single_link_response,
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_projection_XYZ.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_projection - save_arr)), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
