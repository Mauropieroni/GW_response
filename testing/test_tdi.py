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
        tdi_AE_zeta = gwr.tdi_AE_zeta_matrix(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_AE_zeta.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_AE_zeta - save_arr)), 0.0)
        tdi_AE_Sagnac_zeta = gwr.tdi_AE_Sagnac_zeta_matrix(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_AE_Sagnac_zeta.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_AE_Sagnac_zeta - save_arr)), 0.0)
        tdi_matrix = gwr.tdi_matrix(
            TDI_idx=0,  # XYZ basis
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_matrix_XYZ.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_matrix - save_arr)), 0.0)
        tdi_matrix = gwr.tdi_matrix(
            TDI_idx=gwr.TDI_map["AET"],  # AET basis
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_matrix_AET.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_matrix - save_arr)), 0.0)
        tdi_matrix = gwr.tdi_matrix(
            TDI_idx=gwr.TDI_map["AET_Sagnac"],  # AET Sagnac basis
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_matrix_AET_Sagnac.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_matrix - save_arr)), 0.0)
        tdi_matrix = gwr.tdi_matrix(
            TDI_idx=gwr.TDI_map["AE_zeta"],  # AE_zeta basis
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_matrix_AE_zeta.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_matrix - save_arr)), 0.0)
        tdi_matrix = gwr.tdi_matrix(
            TDI_idx=gwr.TDI_map["AE_Sagnac_zeta"],  # AE_Sagnac_zeta basis
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_matrix_AE_Sagnac_zeta.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_matrix - save_arr)), 0.0)

    def test_TDI_projection(self):
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
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
        tdi_projection = gwr.build_tdi(
            TDI_idx=gwr.TDI_map["AET"],  # AET basis
            single_link=single_link_response,
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_projection_AET.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_projection - save_arr)), 0.0)
        tdi_projection = gwr.build_tdi(
            TDI_idx=gwr.TDI_map["AE_zeta"],  # AEZ basis
            single_link=single_link_response,
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_projection_AE_zeta.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_projection - save_arr)), 0.0)
        tdi_projection = gwr.build_tdi(
            TDI_idx=gwr.TDI_map["AE_Sagnac_zeta"],  # AE_Sagnac_Z basis
            single_link=single_link_response,
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tdi_projection_AE_Sagnac_zeta.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(tdi_projection - save_arr)), 0.0)

    def test_TDI_projection_no_pixels_axis(self):
        # build_tdi should also accept an already sky-integrated single_link
        # (shape configurations, x_vector, arms, with no trailing pixels
        # axis), without the caller having to do
        # single_link[..., jnp.newaxis] / result[..., 0] by hand.
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
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
        arms_matrix_rescaled = lisa.detector_arms(0.0) / lisa.armlength
        x_vector = lisa.x(freqs)

        pixel_resolved = gwr.build_tdi(
            TDI_idx=gwr.TDI_map["XYZ"],
            single_link=single_link_response,
            arms_matrix_rescaled=arms_matrix_rescaled,
            x_vector=x_vector,
        )

        integrated_single_link = jnp.mean(single_link_response, axis=-1)
        self.assertEqual(integrated_single_link.ndim, 3)

        integrated_projection = gwr.build_tdi(
            TDI_idx=gwr.TDI_map["XYZ"],
            single_link=integrated_single_link,
            arms_matrix_rescaled=arms_matrix_rescaled,
            x_vector=x_vector,
        )

        # No pixels axis went in, so none should come out.
        self.assertEqual(integrated_projection.ndim, 3)
        self.assertEqual(integrated_projection.shape, pixel_resolved.shape[:-1])

        # The TDI projection is linear in single_link, so integrating over
        # pixels before or after projecting onto the TDI basis must agree.
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(integrated_projection - jnp.mean(pixel_resolved, axis=-1))),
            0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
