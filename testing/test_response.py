import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")
TEST_DATA_PATH_ligo = os.path.join(os.path.dirname(__file__), "test_data_ligo/")


class TestResponse_LISA(unittest.TestCase):
    def test_response(self):
        lisa = gwr.LISA()
        response = lisa.response
        pixel = gwr.Pixel()
        freqs = jnp.logspace(-5, 0, 300)

        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        assert theta is not None and phi is not None
        single_link_response = response.get_single_link_response_fd(
            lisa,
            times_in_years=jnp.array([0.0]),
            theta_array=theta,
            phi_array=phi,
            frequency_array=freqs,
            polarization="LR",
        )
        save_arr = np.load(TEST_DATA_PATH + "single_link_response_L.npy")
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(single_link_response["L"] - save_arr))), 0.0
        )
        save_arr = np.load(TEST_DATA_PATH + "single_link_response_R.npy")
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(single_link_response["R"] - save_arr))), 0.0
        )
        linear_integrand = response.get_linear_integrand_fd(
            lisa,
            times_in_years=jnp.array([0.0]),
            theta_array=theta,
            phi_array=phi,
            frequency_array=freqs,
            combination="XYZ",
            polarization="LR",
        )
        save_arr = np.load(TEST_DATA_PATH + "linear_integrand_L.npy")
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(linear_integrand["L"] - save_arr))), 0.0
        )
        save_arr = np.load(TEST_DATA_PATH + "linear_integrand_R.npy")
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(linear_integrand["R"] - save_arr))), 0.0
        )
        response.compute_detector(
            lisa,
            times_in_years=jnp.array([0.0]),
            theta_array=theta,
            phi_array=phi,
            frequency_array=freqs,
            combination="XYZ",
            polarization="LR",
        )
        save_arr = np.load(TEST_DATA_PATH + "quadratic_response_XYZ_LL.npy")
        self.assertAlmostEqual(
            float(
                jnp.max(jnp.abs(response.quadratic_integrated["XYZ"]["LL"] - save_arr))
            ),
            0.0,
        )
        save_arr = np.load(TEST_DATA_PATH + "quadratic_response_XYZ_RR.npy")
        self.assertAlmostEqual(
            float(
                jnp.max(jnp.abs(response.quadratic_integrated["XYZ"]["RR"] - save_arr))
            ),
            0.0,
        )
        response.compute_detector(
            lisa,
            times_in_years=jnp.array([0.0]),
            theta_array=theta,
            phi_array=phi,
            frequency_array=freqs,
            combination="AET",
            polarization="LR",
        )
        save_arr = np.load(TEST_DATA_PATH + "quadratic_response_AET_LL.npy")
        self.assertAlmostEqual(
            float(
                jnp.max(jnp.abs(response.quadratic_integrated["AET"]["LL"] - save_arr))
            ),
            0.0,
        )
        save_arr = np.load(TEST_DATA_PATH + "quadratic_response_AET_RR.npy")
        self.assertAlmostEqual(
            float(
                jnp.max(jnp.abs(response.quadratic_integrated["AET"]["RR"] - save_arr))
            ),
            0.0,
        )


class TestResponse_LIGO(unittest.TestCase):
    def test_response(self):
        ligo = gwr.LIGO()
        response = ligo.response
        pixel = gwr.Pixel()
        freqs = jnp.logspace(1, 5, 1000)

        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        assert theta is not None and phi is not None

        single_link_response = response.get_single_link_response_fd(
            ligo,
            times_in_years=jnp.array([0.0]),
            theta_array=theta,
            phi_array=phi,
            frequency_array=freqs,
            polarization="LR",
        )
        save_arr = np.load(TEST_DATA_PATH_ligo + "single_link_response_L.npy")
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(single_link_response["L"] - save_arr))), 0.0
        )
        save_arr = np.load(TEST_DATA_PATH_ligo + "single_link_response_R.npy")
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(single_link_response["R"] - save_arr))), 0.0
        )

        linear_integrand = response.get_linear_integrand_fd(
            ligo,
            times_in_years=jnp.array([0.0]),
            theta_array=theta,
            phi_array=phi,
            frequency_array=freqs,
            polarization="LR",
        )

        save_arr = np.load(TEST_DATA_PATH_ligo + "linear_integrand_L.npy")
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(linear_integrand["L"] - save_arr))), 0.0
        )
        save_arr = np.load(TEST_DATA_PATH_ligo + "linear_integrand_R.npy")
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(linear_integrand["R"] - save_arr))), 0.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
