import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH_lisa = os.path.join(os.path.dirname(__file__), "test_data_lisa/")
TEST_DATA_PATH_ligo = os.path.join(os.path.dirname(__file__), "test_data_ligo/")


class TestSingleLinkStatic(unittest.TestCase):
    def test_xi_k(self):
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        xi_k = gwr.xi_k_no_G_static(
            unit_wavevector=unit_vector,
            x_vector=lisa.x(freqs),
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
        )
        save_arr = np.load(TEST_DATA_PATH_lisa + "xi_k.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(xi_k - save_arr))), 0.0)

    def test_xi_k_A(self):
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        u, v = gwr.uv_analytical(theta, phi)
        e1L, _ = gwr.polarization_tensors_LR(u, v)
        geomtrical_factor = gwr.geometrical_factor(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            polarization_tensor=e1L,
        )
        xi_k_A_static = gwr.xi_k_A_static(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            unit_wavevector=unit_vector,
            x_vector=lisa.x(freqs),
            geometrical=geomtrical_factor,
        )
        save_arr = np.load(TEST_DATA_PATH_lisa + "xi_k_A.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(xi_k_A_static - save_arr))), 0.0)

    def test_single_link_response(self):
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        u, v = gwr.uv_analytical(theta, phi)
        e1L, _ = gwr.polarization_tensors_LR(u, v)
        geomtrical_factor = gwr.geometrical_factor(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            polarization_tensor=e1L,
        )
        xi_k_A_static = gwr.xi_k_A_static(
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            unit_wavevector=unit_vector,
            x_vector=lisa.x(freqs),
            geometrical=geomtrical_factor,
        )
        single_link_response_static = gwr.single_link_response_static(
            positions_rescaled=lisa.vertex_positions(0.0) / lisa.armlength,
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            wavevector=unit_vector,
            x_vector=lisa.x(freqs),
            xi_k_A_static=xi_k_A_static,
        )
        save_arr = np.load(TEST_DATA_PATH_lisa + "single_link_response.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(single_link_response_static - save_arr))), 0.0
        )

    def test_geometrical_and_xi_k_angles_convenience_functions(self):
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        assert theta is not None and phi is not None
        u, v = gwr.uv_analytical(theta, phi)
        arms_matrix_rescaled = lisa.detector_arms(0.0) / lisa.armlength
        x_vector = lisa.x(freqs)
        wavevector = gwr.unit_vec(theta, phi)

        e1p, e1c = gwr.polarization_tensors_PC(u, v)
        G_plus = gwr.geometrical_factor(arms_matrix_rescaled, e1p)
        G_cross = gwr.geometrical_factor(arms_matrix_rescaled, e1c)
        G_plus_angles, G_cross_angles = gwr.geometrical_factor_PC_angles_static(
            arms_matrix_rescaled, theta, phi
        )
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(G_plus_angles - G_plus))), 0.0)
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(G_cross_angles - G_cross))), 0.0)

        e1L, e1R = gwr.polarization_tensors_LR(u, v)
        G_L = gwr.geometrical_factor(arms_matrix_rescaled, e1L)
        G_R = gwr.geometrical_factor(arms_matrix_rescaled, e1R)
        G_L_angles, G_R_angles = gwr.geometrical_factor_LR_angles_static(
            arms_matrix_rescaled, theta, phi
        )
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(G_L_angles - G_L))), 0.0)
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(G_R_angles - G_R))), 0.0)

        xi_k_P = gwr.xi_k_A_static(arms_matrix_rescaled, wavevector, x_vector, G_plus)
        xi_k_C = gwr.xi_k_A_static(arms_matrix_rescaled, wavevector, x_vector, G_cross)
        xi_k_P_angles, xi_k_C_angles = gwr.xi_k_A_PC_angles_static(
            arms_matrix_rescaled, theta, phi, x_vector
        )
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(xi_k_P_angles - xi_k_P))), 0.0)
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(xi_k_C_angles - xi_k_C))), 0.0)

        xi_k_L = gwr.xi_k_A_static(arms_matrix_rescaled, wavevector, x_vector, G_L)
        xi_k_R = gwr.xi_k_A_static(arms_matrix_rescaled, wavevector, x_vector, G_R)
        xi_k_L_angles, xi_k_R_angles = gwr.xi_k_A_LR_angles_static(
            arms_matrix_rescaled, theta, phi, x_vector
        )
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(xi_k_L_angles - xi_k_L))), 0.0)
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(xi_k_R_angles - xi_k_R))), 0.0)

    def test_single_link_response_angles_convenience_functions(self):
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        u, v = gwr.uv_analytical(theta, phi)
        arms_matrix_rescaled = lisa.detector_arms(0.0) / lisa.armlength
        positions_rescaled = lisa.vertex_positions(0.0) / lisa.armlength
        x_vector = lisa.x(freqs)
        wavevector = gwr.unit_vec(theta, phi)

        e1p, e1c = gwr.polarization_tensors_PC(u, v)
        response_P = gwr.get_single_link_response_static(
            e1p, arms_matrix_rescaled, wavevector, x_vector, positions_rescaled
        )
        response_C = gwr.get_single_link_response_static(
            e1c, arms_matrix_rescaled, wavevector, x_vector, positions_rescaled
        )
        response_P_angles, response_C_angles = (
            gwr.single_link_response_PC_angles_static(
                positions_rescaled, arms_matrix_rescaled, theta, phi, x_vector
            )
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(response_P_angles - response_P))), 0.0
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(response_C_angles - response_C))), 0.0
        )

        e1L, e1R = gwr.polarization_tensors_LR(u, v)
        response_L = gwr.get_single_link_response_static(
            e1L, arms_matrix_rescaled, wavevector, x_vector, positions_rescaled
        )
        response_R = gwr.get_single_link_response_static(
            e1R, arms_matrix_rescaled, wavevector, x_vector, positions_rescaled
        )
        response_L_angles, response_R_angles = (
            gwr.single_link_response_LR_angles_static(
                positions_rescaled, arms_matrix_rescaled, theta, phi, x_vector
            )
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(response_L_angles - response_L))), 0.0
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(response_R_angles - response_R))), 0.0
        )


class TestSingleLinkStatic_ligo(unittest.TestCase):
    def test_xi_k(self):
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        ligo = gwr.LIGO()
        freqs = jnp.logspace(1, 5, 1000)
        xi_k = gwr.xi_k_no_G_static(
            unit_wavevector=unit_vector,
            x_vector=ligo.x(freqs),
            arms_matrix_rescaled=ligo.detector_arms(0.0) / ligo.armlength,
        )
        save_arr = np.load(TEST_DATA_PATH_ligo + "xi_k.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(xi_k - save_arr))), 0.0)

    def test_xi_k_A(self):
        ligo = gwr.LIGO()
        freqs = jnp.logspace(1, 5, 1000)
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        u, v = gwr.uv_analytical(theta, phi)
        e1L, _ = gwr.polarization_tensors_LR(u, v)
        geomtrical_factor = gwr.geometrical_factor(
            arms_matrix_rescaled=ligo.detector_arms(0.0) / ligo.armlength,
            polarization_tensor=e1L,
        )
        xi_k_A_static = gwr.xi_k_A_static(
            arms_matrix_rescaled=ligo.detector_arms(0.0) / ligo.armlength,
            unit_wavevector=unit_vector,
            x_vector=ligo.x(freqs),
            geometrical=geomtrical_factor,
        )
        save_arr = np.load(TEST_DATA_PATH_ligo + "xi_k_A.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(xi_k_A_static - save_arr))), 0.0)

    def test_single_link_response(self):
        ligo = gwr.LIGO()
        freqs = jnp.logspace(1, 5, 1000)
        pixel = gwr.Pixel()
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        u, v = gwr.uv_analytical(theta, phi)
        e1L, _ = gwr.polarization_tensors_LR(u, v)
        geomtrical_factor = gwr.geometrical_factor(
            arms_matrix_rescaled=ligo.detector_arms(0.0) / ligo.armlength,
            polarization_tensor=e1L,
        )
        xi_k_A_static = gwr.xi_k_A_static(
            arms_matrix_rescaled=ligo.detector_arms(0.0) / ligo.armlength,
            unit_wavevector=unit_vector,
            x_vector=ligo.x(freqs),
            geometrical=geomtrical_factor,
        )
        # single_link_response_static's roll-trick expects exactly arms/2
        # positions (the 2 end mirrors); LIGO.vertex_positions also carries
        # the corner station (vertex 0, needed by the retarded pipeline's
        # arm_vertex_pairs), so that's sliced off here.
        single_link_response_static = gwr.single_link_response_static(
            positions_rescaled=ligo.vertex_positions(0.0)[..., 1:] / ligo.armlength,
            arms_matrix_rescaled=ligo.detector_arms(0.0) / ligo.armlength,
            wavevector=unit_vector,
            x_vector=ligo.x(freqs),
            xi_k_A_static=xi_k_A_static,
        )
        save_arr = np.load(TEST_DATA_PATH_ligo + "single_link_response.npy")
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(single_link_response_static - save_arr))), 0.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
