import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH_lisa = os.path.join(os.path.dirname(__file__), "test_data_lisa/")
TEST_DATA_PATH_ligo = os.path.join(os.path.dirname(__file__), "test_data_ligo/")


class TestNoise(unittest.TestCase):
    def test_noise_class(self):
        freqs = jnp.logspace(-5, 0, 300)
        time_in_years = jnp.linspace(0, 1.0, 100)
        TM_params = jnp.ones(shape=(100, 6))
        OMS_params = jnp.ones(shape=(100, 6))

        lisa = gwr.LISA()
        noise = lisa.noise

        noise.compute_detector(
            lisa,
            times_in_years=time_in_years,
            frequency_array=freqs,
            TM_acceleration_parameters=TM_params,
            OMS_parameters=OMS_params,
            combination="XYZ",
        )

        tm_single_link = np.load(TEST_DATA_PATH_lisa + "tm_noise_single_link.npy")
        oms_single_link = np.load(TEST_DATA_PATH_lisa + "oms_noise_single_link.npy")
        self.assertAlmostEqual(
            float(
                jnp.sum(
                    jnp.abs(
                        noise.single_link_noise - (tm_single_link + oms_single_link)
                    )
                )
            ),
            0.0,
        )
        save_arr = np.load(TEST_DATA_PATH_lisa + "noise_matrix.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(noise.noise_matrix["XYZ"] - save_arr))),
            0.0,
        )


class TestNoise_ligo(unittest.TestCase):
    def test_noise_class_ligo(self):
        freqs = jnp.logspace(1, 5, 1000)

        ligo = gwr.LIGO()
        noise = ligo.noise
        noise.compute_detector(
            ligo,
            times_in_years=jnp.array([0.0]),
            frequency_array=freqs,
            combination="Michelson",
        )

        save_arr = np.load(TEST_DATA_PATH_ligo + "ligo_psd.npy")
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(noise.noise_matrix["Michelson"] - save_arr))),
            0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
