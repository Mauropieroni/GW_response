import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")


class TestResponse(unittest.TestCase):
    def test_response(self):
        response = gwr.Response(
            ps=gwr.PhysicalConstants(),
            det=gwr.LISA(),
        )
        pixel = gwr.Pixel()
        freqs = jnp.logspace(-5, 0, 300)

        theta, phi = pixel.theta_pixel, pixel.phi_pixel
        single_link_response = response.get_single_link_response(
            times_in_years=jnp.array([0.0]),
            theta_array=theta,
            phi_array=phi,
            frequency_array=freqs,
            polarization="LR",
        )
        save_arr = np.load(TEST_DATA_PATH + "single_link_response_LL.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(single_link_response["LL"] - save_arr)), 0.0
        )
        save_arr = np.load(TEST_DATA_PATH + "single_link_response_RR.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(single_link_response["RR"] - save_arr)), 0.0
        )
        linear_integrand = response.get_linear_integrand(
            times_in_years=jnp.array([0.0]),
            single_link=single_link_response,
            frequency_array=freqs,
            TDI="XYZ",
            polarization="LR",
        )
        save_arr = np.load(TEST_DATA_PATH + "linear_integrand_LL.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(linear_integrand["LL"] - save_arr)), 0.0)
        save_arr = np.load(TEST_DATA_PATH + "linear_integrand_RR.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(linear_integrand["RR"] - save_arr)), 0.0)
        response.compute_detector(
            times_in_years=jnp.array([0.0]),
            theta_array=theta,
            phi_array=phi,
            frequency_array=freqs,
            TDI="XYZ",
            polarization="LR",
        )
        save_arr = np.load(TEST_DATA_PATH + "response_XYZ_LL.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(response.integrated["XYZ"]["LL"] - save_arr)), 0.0
        )
        save_arr = np.load(TEST_DATA_PATH + "response_XYZ_RR.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(response.integrated["XYZ"]["RR"] - save_arr)), 0.0
        )
        response.compute_detector(
            times_in_years=jnp.array([0.0]),
            theta_array=theta,
            phi_array=phi,
            frequency_array=freqs,
            TDI="AET",
            polarization="LR",
        )
        save_arr = np.load(TEST_DATA_PATH + "response_AET_LL.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(response.integrated["AET"]["LL"] - save_arr)), 0.0
        )
        save_arr = np.load(TEST_DATA_PATH + "response_AET_RR.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(response.integrated["AET"]["RR"] - save_arr)), 0.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
