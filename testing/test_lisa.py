import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")


class TestLISA(unittest.TestCase):
    def test_lisa_analytical_positions(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa_analytical_x_positions = (
            gwr.LISA_satellite_x_coordinate_analytical(
                index=1,
                time_in_years=time_in_years,
                orbit_radius=gwr.PhysicalConstants().AU,
                eccentricity=gwr.LISA().ecc,
            )
        )
        save_arr = np.load(TEST_DATA_PATH + "lisa_analytical_x_positions.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(lisa_analytical_x_positions - save_arr))
            / np.max(save_arr),
            0.0,
        )

    def test_lisa_analytical_y_positions(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa_analytical_y_positions = (
            gwr.LISA_satellite_y_coordinate_analytical(
                index=1,
                time_in_years=time_in_years,
                orbit_radius=gwr.PhysicalConstants().AU,
                eccentricity=gwr.LISA().ecc,
            )
        )
        save_arr = np.load(TEST_DATA_PATH + "lisa_analytical_y_positions.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(lisa_analytical_y_positions - save_arr))
            / np.max(save_arr),
            0.0,
        )

    def test_lisa_analytical_z_positions(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa_analytical_z_positions = (
            gwr.LISA_satellite_z_coordinate_analytical(
                index=1,
                time_in_years=time_in_years,
                orbit_radius=gwr.PhysicalConstants().AU,
                eccentricity=gwr.LISA().ecc,
            )
        )
        save_arr = np.load(TEST_DATA_PATH + "lisa_analytical_z_positions.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(lisa_analytical_z_positions - save_arr))
            / np.max(save_arr),
            0.0,
        )

    def test_lisa_analytical_positions(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa_analytical_positions = gwr.LISA_satellite_coordinates_analytical(
            index=1,
            time_in_years=time_in_years,
            orbit_radius=gwr.PhysicalConstants().AU,
            eccentricity=gwr.LISA().ecc,
        )
        save_arr = np.load(
            TEST_DATA_PATH + "lisa_analytical_positions_sat_1.npy"
        )
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(lisa_analytical_positions - save_arr))
            / np.max(save_arr),
            0.0,
        )

    def test_lisa_analytical_positions_vm(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa_analytical_positions = (
            gwr.LISA_satellite_coordinates_analytical_vm(
                jnp.array([1, 2, 3]),
                time_in_years,
                gwr.PhysicalConstants().AU,
                gwr.LISA().ecc,
            )
        )
        save_arr = np.load(TEST_DATA_PATH + "lisa_analytical_positions_vm.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(lisa_analytical_positions - save_arr))
            / np.max(save_arr),
            0.0,
        )

    def test_lisa_satellite_positions(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa_analytical_positions = gwr.LISA_satellite_positions(
            time_in_years,
            gwr.PhysicalConstants().AU,
            eccentricity=gwr.LISA().ecc,
            which_orbits="analytic",  # Currently only analytic implemented
        )
        save_arr = np.load(TEST_DATA_PATH + "lisa_satellite_positions.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(lisa_analytical_positions - save_arr))
            / np.max(save_arr),
            0.0,
        )

    def test_lisa_arms_matrix(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa_arms_matrix = gwr.LISA_arms_matrix_analytical(
            time_in_years,
            gwr.PhysicalConstants().AU,
            eccentricity=gwr.LISA().ecc,
        )
        save_arr = np.load(TEST_DATA_PATH + "lisa_arms_matrix_analytical.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(lisa_arms_matrix - save_arr)) / np.max(save_arr),
            0.0,
        )
        arm_lengths = jnp.sqrt(
            jnp.einsum("tij,tij->tj", lisa_arms_matrix, lisa_arms_matrix)
        )
        save_arr = np.load(TEST_DATA_PATH + "arm_lengths.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(arm_lengths - save_arr)) / np.max(save_arr),
            0.0,
        )

    def test_lisa_class(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa = gwr.LISA()
        frequencies = lisa.frequency_vec(10)
        save_arr = np.load(TEST_DATA_PATH + "frequencies.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(frequencies - save_arr)), 0.0)
        kl_vector = lisa.klvector(frequencies)
        save_arr = np.load(TEST_DATA_PATH + "kl_vector.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(kl_vector - save_arr)), 0.0)
        x_vector = lisa.x(frequencies)
        save_arr = np.load(TEST_DATA_PATH + "x_vector.npy")
        self.assertAlmostEqual(jnp.sum(jnp.abs(x_vector - save_arr)), 0.0)
        satellite_positions = lisa.satellite_positions(time_in_years)
        save_arr = np.load(TEST_DATA_PATH + "satellite_positions.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(satellite_positions - save_arr)) / np.max(save_arr),
            0.0,
        )
        arms_matrix = lisa.detector_arms(time_in_years)
        save_arr = np.load(TEST_DATA_PATH + "arms_matrix.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(arms_matrix - save_arr)) / np.max(save_arr),
            0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
