import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")
TEST_DATA_PATH_ligo = os.path.join(os.path.dirname(__file__), "test_data_ligo/")


class TestNoise(unittest.TestCase):
    def test_acc_noise(self):
        freqs = jnp.logspace(-5, 0, 300)
        acc_noise = gwr.LISA_acceleration_noise(freqs, acc_param=1.0)
        save_arr = np.load(TEST_DATA_PATH + "acc_noise.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(acc_noise - save_arr))), 0.0)
        int_noise = gwr.LISA_interferometric_noise(freqs, inter_param=1.0)
        save_arr = np.load(TEST_DATA_PATH + "int_noise.npy")
        self.assertAlmostEqual(float(jnp.sum(jnp.abs(int_noise - save_arr))), 0.0)

    def test_tm_noise_single_link(self):
        freqs = jnp.logspace(-5, 0, 300)
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa = gwr.LISA()
        TM_params = jnp.ones(shape=(100, 6))
        tm_noise_single_link = gwr.single_link_TM_acceleration_noise_variance(
            freqs,
            TM_acceleration_parameters=TM_params,  # Should be length 6 (for each of
            # the arms)
            arms_matrix_rescaled=lisa.detector_arms(time_in_years=time_in_years)
            / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tm_noise_single_link.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(tm_noise_single_link - save_arr))),
            0.0,
        )
        TM_tdi_projection = gwr.tdi_projection(
            TDI_idx=0,  # 0 = XTZ, 1 = AET, 2 = Sagnac, 3 = AET_Sagnac etc.
            single_link_mat=tm_noise_single_link,
            arms_matrix_rescaled=lisa.detector_arms(time_in_years=time_in_years)
            / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tm_tdi_projection.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(TM_tdi_projection - save_arr))),
            0.0,
        )
        TM_tdi_projection = gwr.tdi_projection(
            TDI_idx=1,  # 0 = XTZ, 1 = AET, 2 = Sagnac, 3 = AET_Sagnac etc.
            single_link_mat=tm_noise_single_link,
            arms_matrix_rescaled=lisa.detector_arms(time_in_years=time_in_years)
            / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tm_tdi_projection_aet.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(TM_tdi_projection - save_arr))),
            0.0,
        )
        TM_tdi_matrix = gwr.noise_TM_matrix(
            TDI_idx=0,
            frequency=freqs,
            TM_acceleration_parameters=TM_params,
            arms_matrix_rescaled=lisa.detector_arms(time_in_years=time_in_years)
            / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "tm_tdi_matrix.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(TM_tdi_matrix - save_arr))),
            0.0,
        )

    def test_oms_noise_single_link(self):
        freqs = jnp.logspace(-5, 0, 300)
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa = gwr.LISA()
        OMS_params = jnp.ones(shape=(100, 6))
        oms_noise_single_link = gwr.single_link_OMS_noise_variance(
            freqs,
            OMS_parameters=OMS_params,  # Should be length 6 (for each of the arms)
            arms_matrix_rescaled=lisa.detector_arms(time_in_years=time_in_years)
            / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "oms_noise_single_link.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(oms_noise_single_link - save_arr))),
            0.0,
        )
        OMS_tdi_projection = gwr.tdi_projection(
            TDI_idx=0,  # 0 = XTZ, 1 = AET, 2 = Sagnac, 3 = AET_Sagnac etc.
            single_link_mat=oms_noise_single_link,
            arms_matrix_rescaled=lisa.detector_arms(time_in_years=time_in_years)
            / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "oms_tdi_projection.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(OMS_tdi_projection - save_arr))),
            0.0,
        )
        OMS_tdi_projection = gwr.tdi_projection(
            TDI_idx=1,  # 0 = XTZ, 1 = AET, 2 = Sagnac, 3 = AET_Sagnac etc.
            single_link_mat=oms_noise_single_link,
            arms_matrix_rescaled=lisa.detector_arms(time_in_years=time_in_years)
            / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "oms_tdi_projection_aet.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(OMS_tdi_projection - save_arr))),
            0.0,
        )
        OMS_tdi_matrix = gwr.noise_OMS_matrix(
            TDI_idx=0,
            frequency=freqs,
            OMS_parameters=OMS_params,
            arms_matrix_rescaled=lisa.detector_arms(time_in_years=time_in_years)
            / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "oms_tdi_matrix.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(OMS_tdi_matrix - save_arr))),
            0.0,
        )

    def test_noise_matrix(self):
        freqs = jnp.logspace(-5, 0, 300)
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa = gwr.LISA()
        TM_params = jnp.ones(shape=(100, 6))
        OMS_params = jnp.ones(shape=(100, 6))
        noise_matrix = gwr.noise_matrix(
            TDI_idx=0,
            frequency=freqs,
            TM_acceleration_parameters=TM_params,
            OMS_parameters=OMS_params,
            arms_matrix_rescaled=lisa.detector_arms(time_in_years=time_in_years)
            / lisa.armlength,
            x_vector=lisa.x(freqs),
        )
        save_arr = np.load(TEST_DATA_PATH + "noise_matrix.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(noise_matrix - save_arr))),
            0.0,
        )

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

        tm_single_link = np.load(TEST_DATA_PATH + "tm_noise_single_link.npy")
        oms_single_link = np.load(TEST_DATA_PATH + "oms_noise_single_link.npy")
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
        save_arr = np.load(TEST_DATA_PATH + "noise_matrix.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(noise.noise_matrix["XYZ"] - save_arr))),
            0.0,
        )


class TestNoise_ligo(unittest.TestCase):
    def test_noise_ligo(self):
        freqs = jnp.logspace(1, 5, 1000)
        ligo_noise = gwr.LIGO_noise(freqs)
        save_arr = np.load(TEST_DATA_PATH_ligo + "ligo_psd.npy")
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(ligo_noise / save_arr - 1.0))),
            0.0,
        )

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

    def test_noise_class(self):
        freqs = jnp.logspace(-5, 0, 300)
        time_in_years = jnp.linspace(0, 1.0, 100)
        TM_params = jnp.ones(shape=(100, 6))
        OMS_params = jnp.ones(shape=(100, 6))

        lisa = gwr.LISA()
        noise = gwr.Noise()
        noise.compute_detector(
            lisa,
            time_in_years,
            freqs,
            combination="XYZ",
            TM_acceleration_parameters=TM_params,
            OMS_parameters=OMS_params,
        )

        arms_matrix_rescaled = lisa.detector_arms(time_in_years) / lisa.armlength
        x_vector = lisa.x(freqs)

        save_arr = np.load(TEST_DATA_PATH + "tm_tdi_matrix.npy")
        TM_noise_matrix = gwr.noise_TM_matrix(
            gwr.TDI_map["XYZ"], freqs, TM_params, arms_matrix_rescaled, x_vector
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(TM_noise_matrix - save_arr))),
            0.0,
        )
        save_arr = np.load(TEST_DATA_PATH + "oms_tdi_matrix.npy")
        OMS_noise_matrix = gwr.noise_OMS_matrix(
            gwr.TDI_map["XYZ"], freqs, OMS_params, arms_matrix_rescaled, x_vector
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(OMS_noise_matrix - save_arr))),
            0.0,
        )
        save_arr = np.load(TEST_DATA_PATH + "noise_matrix.npy")
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(noise.noise_matrix["XYZ"] - save_arr))),
            0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
