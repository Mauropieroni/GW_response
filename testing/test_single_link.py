import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")
TEST_DATA_PATH_lisa = os.path.join(os.path.dirname(__file__), "test_data_lisa/")
TEST_DATA_PATH_ligo = os.path.join(os.path.dirname(__file__), "test_data_ligo/")


class TestSingleLink(unittest.TestCase):
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
        linear_response_tdi = gwr.linear_response_angular(
            TDI_idx=0,  # XYZ basis
            single_link=single_link_response_static,
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH_lisa + "linear_response_tdi.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(linear_response_tdi - save_arr))), 0.0
        )
        quadratic_angular_response = gwr.quadratic_response_angular(
            TDI_idx=0,  # XYZ basis
            single_link=single_link_response_static,
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH_lisa + "quadratic_angular_response.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(quadratic_angular_response - save_arr))), 0.0
        )
        quadratic_response_integrated = gwr.quadratic_response_integrated(
            quadratic_angular_response
        )
        save_arr = np.load(TEST_DATA_PATH_lisa + "quadratic_response_integrated.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(quadratic_response_integrated - save_arr))), 0.0
        )
        quadratic_angular_response_AET = gwr.quadratic_response_angular(
            TDI_idx=1,  # AET basis
            single_link=single_link_response_static,
            arms_matrix_rescaled=lisa.detector_arms(0.0) / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        quadratic_response_integrated = gwr.quadratic_response_integrated(
            quadratic_angular_response_AET
        )
        save_arr = np.load(
            TEST_DATA_PATH_lisa + "quadratic_response_integrated_AET.npy"
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(quadratic_response_integrated - save_arr))), 0.0
        )

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

    def test_get_single_link_response_retarded_angles(self):
        # Same "_angles" convenience pattern, for the retarded (genuinely
        # asymmetric arm) single-link response used by
        # Response.get_single_link_response_fd.
        lisa = gwr.LISA()
        freqs = jnp.logspace(-5, 0, 300)
        pixel = gwr.Pixel()
        assert pixel.theta_pixel is not None and pixel.phi_pixel is not None
        theta, phi = pixel.theta_pixel[:5], pixel.phi_pixel[:5]
        u, v = gwr.uv_analytical(theta, phi)
        x_vector = lisa.x(freqs)
        wavevector = gwr.unit_vec(theta, phi)

        arm_vector_retarded, ltt, receiver_position = lisa.detector_arms_retarded(
            0.0, lisa.response.ps
        )
        arm_vector_retarded_rescaled = arm_vector_retarded[None] / lisa.armlength
        ltt_rescaled = ltt[None] * lisa.ps.light_speed / lisa.armlength
        receiver_positions_rescaled = receiver_position[None] / lisa.armlength

        e1p, e1c = gwr.polarization_tensors_PC(u, v)
        response_P = gwr.get_single_link_response_retarded(
            e1p,
            arm_vector_retarded_rescaled,
            ltt_rescaled,
            wavevector,
            x_vector,
            receiver_positions_rescaled,
        )
        response_C = gwr.get_single_link_response_retarded(
            e1c,
            arm_vector_retarded_rescaled,
            ltt_rescaled,
            wavevector,
            x_vector,
            receiver_positions_rescaled,
        )
        response_P_angles, response_C_angles = (
            gwr.get_single_link_response_retarded_PC_angles(
                arm_vector_retarded_rescaled,
                ltt_rescaled,
                theta,
                phi,
                x_vector,
                receiver_positions_rescaled,
            )
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(response_P_angles - response_P))), 0.0
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(response_C_angles - response_C))), 0.0
        )

        e1L, e1R = gwr.polarization_tensors_LR(u, v)
        response_L = gwr.get_single_link_response_retarded(
            e1L,
            arm_vector_retarded_rescaled,
            ltt_rescaled,
            wavevector,
            x_vector,
            receiver_positions_rescaled,
        )
        response_R = gwr.get_single_link_response_retarded(
            e1R,
            arm_vector_retarded_rescaled,
            ltt_rescaled,
            wavevector,
            x_vector,
            receiver_positions_rescaled,
        )
        response_L_angles, response_R_angles = (
            gwr.get_single_link_response_retarded_LR_angles(
                arm_vector_retarded_rescaled,
                ltt_rescaled,
                theta,
                phi,
                x_vector,
                receiver_positions_rescaled,
            )
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(response_L_angles - response_L))), 0.0
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(response_R_angles - response_R))), 0.0
        )


class TestSingleLink_ligo(unittest.TestCase):
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

    def test_position_exp(self):
        ligo = gwr.LIGO()
        freqs = jnp.logspace(1, 5, 1000)
        pixel = gwr.Pixel()
        # print(pixel)
        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        unit_vector = gwr.unit_vec(theta, phi)
        # print(unit_vector)
        sat_positions = ligo.vertex_positions(0.0)[0]
        # print(sat_positions)
        p1, p2, p3 = sat_positions[:, 0], sat_positions[:, 1], sat_positions[:, 2]
        sp1, sp2, sp3 = gwr.shift_to_center(p1, p2, p3)
        # print(sp1, sp2, sp3)

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
