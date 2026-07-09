import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import tempfile
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")


def write_numerical_orbit_file(path, time_grid, orbit_radius, eccentricity):
    positions = gwr.LISA_satellite_coordinates_analytical_vm(
        jnp.array([1, 2, 3]), time_grid, orbit_radius, eccentricity
    )  # shape (3 satellites, 3 coordinates, N times)
    positions = jnp.transpose(positions, (2, 0, 1))  # (N times, 3 satellites, 3 coords)
    data = np.column_stack(
        [np.array(time_grid), np.array(positions).reshape(len(time_grid), 9)]
    )
    np.savetxt(path, data)


class TestLISA(unittest.TestCase):
    def test_lisa_analytical_positions(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa_analytical_x_positions = gwr.LISA_satellite_x_coordinate_analytical(
            index=1,
            time_in_years=time_in_years,
            orbit_radius=gwr.PhysicalConstants().AU,
            eccentricity=gwr.LISA().ecc,
        )
        save_arr = np.load(TEST_DATA_PATH + "lisa_analytical_x_positions.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(lisa_analytical_x_positions - save_arr)) / np.max(save_arr),
            0.0,
        )

    def test_lisa_analytical_y_positions(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa_analytical_y_positions = gwr.LISA_satellite_y_coordinate_analytical(
            index=1,
            time_in_years=time_in_years,
            orbit_radius=gwr.PhysicalConstants().AU,
            eccentricity=gwr.LISA().ecc,
        )
        save_arr = np.load(TEST_DATA_PATH + "lisa_analytical_y_positions.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(lisa_analytical_y_positions - save_arr)) / np.max(save_arr),
            0.0,
        )

    def test_lisa_analytical_z_positions(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa_analytical_z_positions = gwr.LISA_satellite_z_coordinate_analytical(
            index=1,
            time_in_years=time_in_years,
            orbit_radius=gwr.PhysicalConstants().AU,
            eccentricity=gwr.LISA().ecc,
        )
        save_arr = np.load(TEST_DATA_PATH + "lisa_analytical_z_positions.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(lisa_analytical_z_positions - save_arr)) / np.max(save_arr),
            0.0,
        )

    def test_lisa_analytical_positions_vm(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa_analytical_positions = gwr.LISA_satellite_coordinates_analytical_vm(
            jnp.array([1, 2, 3]),
            time_in_years,
            gwr.PhysicalConstants().AU,
            gwr.LISA().ecc,
        )
        save_arr = np.load(TEST_DATA_PATH + "lisa_analytical_positions_vm.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(lisa_analytical_positions - save_arr)) / np.max(save_arr),
            0.0,
        )

    def test_lisa_satellite_positions(self):
        time_in_years = jnp.linspace(0, 1.0, 100)
        lisa_analytical_positions = gwr.LISA_satellite_positions(
            time_in_years,
            gwr.PhysicalConstants().AU,
            eccentricity=gwr.LISA().ecc,
            which_orbits="analytic",
        )
        save_arr = np.load(TEST_DATA_PATH + "lisa_satellite_positions.npy")
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(lisa_analytical_positions - save_arr)) / np.max(save_arr),
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

    def test_load_numerical_orbits(self):
        orbit_radius = gwr.PhysicalConstants().AU
        eccentricity = gwr.LISA().ecc
        time_grid = jnp.linspace(0, 1.0, 50)
        with tempfile.TemporaryDirectory() as tmp_dir:
            orbit_file = os.path.join(tmp_dir, "orbits.txt")
            write_numerical_orbit_file(
                orbit_file, time_grid, orbit_radius, eccentricity
            )
            loaded_time_grid, loaded_positions_grid = gwr.load_numerical_orbits(
                orbit_file
            )
        self.assertEqual(loaded_positions_grid.shape, (50, 3, 3))
        self.assertAlmostEqual(jnp.sum(jnp.abs(loaded_time_grid - time_grid)), 0.0)

    def test_load_numerical_orbits_wrong_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            orbit_file = os.path.join(tmp_dir, "orbits.txt")
            np.savetxt(orbit_file, np.zeros((10, 4)))
            with self.assertRaises(ValueError):
                gwr.load_numerical_orbits(orbit_file)

    def test_lisa_numerical_orbits_matches_analytical(self):
        orbit_radius = gwr.PhysicalConstants().AU
        eccentricity = gwr.LISA().ecc
        # Dense grid so linear interpolation closely matches the analytical
        # orbit.
        dense_time_grid = jnp.linspace(-0.1, 1.1, 2000)
        query_time = jnp.linspace(0, 1.0, 100)

        lisa_analytic = gwr.LISA(which_orbits="analytic")
        analytic_positions = lisa_analytic.satellite_positions(query_time)
        analytic_arms = lisa_analytic.detector_arms(query_time)

        with tempfile.TemporaryDirectory() as tmp_dir:
            orbit_file = os.path.join(tmp_dir, "orbits.txt")
            write_numerical_orbit_file(
                orbit_file, dense_time_grid, orbit_radius, eccentricity
            )
            lisa_numeric = gwr.LISA(which_orbits="numeric", orbit_file=orbit_file)
            numeric_positions = lisa_numeric.satellite_positions(query_time)
            numeric_arms = lisa_numeric.detector_arms(query_time)

        self.assertLess(
            jnp.max(jnp.abs(numeric_positions - analytic_positions))
            / jnp.max(jnp.abs(analytic_positions)),
            1e-5,
        )
        self.assertLess(
            jnp.max(jnp.abs(numeric_arms - analytic_arms))
            / jnp.max(jnp.abs(analytic_arms)),
            1e-5,
        )

    def test_lisa_numeric_orbits_requires_orbit_file(self):
        with self.assertRaises(ValueError):
            gwr.LISA(which_orbits="numeric")

    def test_lisa_unknown_orbit_model(self):
        lisa = gwr.LISA(which_orbits="bogus")
        with self.assertRaises(ValueError):
            lisa.satellite_positions(jnp.linspace(0, 1.0, 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
