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
            orbit_approximant="rigid",
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
            orbit_interpolator = gwr.load_numerical_orbits(orbit_file)
        self.assertEqual(orbit_interpolator.f.shape, (50, 3, 3))
        self.assertAlmostEqual(jnp.sum(jnp.abs(orbit_interpolator.x - time_grid)), 0.0)

    def test_load_numerical_orbits_interpolation_method(self):
        orbit_radius = gwr.PhysicalConstants().AU
        eccentricity = gwr.LISA().ecc
        time_grid = jnp.linspace(0, 1.0, 50)
        with tempfile.TemporaryDirectory() as tmp_dir:
            orbit_file = os.path.join(tmp_dir, "orbits.txt")
            write_numerical_orbit_file(
                orbit_file, time_grid, orbit_radius, eccentricity
            )
            linear_interpolator = gwr.load_numerical_orbits(orbit_file)
            cubic_interpolator = gwr.load_numerical_orbits(
                orbit_file, interpolation_method="cubic"
            )
        self.assertEqual(linear_interpolator.method, "linear")
        self.assertEqual(cubic_interpolator.method, "cubic")

    def test_load_numerical_orbits_wrong_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            orbit_file = os.path.join(tmp_dir, "orbits.txt")
            np.savetxt(orbit_file, np.zeros((10, 4)))
            with self.assertRaises(ValueError):
                gwr.load_numerical_orbits(orbit_file)

    def test_lisa_numerical_orbits_matches_analytical(self):
        # The fixture file stores analytical ("rigid") positions on a coarse
        # grid (100 points over 3 years, i.e. about 11 days apart). Querying
        # on a much finer grid forces the loaded interpolator to genuinely
        # interpolate between stored samples, rather than trivially landing
        # on stored points, so this exercises real interpolation error
        # (small, but not vanishing) rather than a machine-precision
        # round-trip.
        orbit_file = os.path.join(TEST_DATA_PATH, "lisa_numerical_orbit_data.txt")
        query_time = jnp.linspace(0.01, 2.99, 5000)

        lisa_analytic = gwr.LISA(orbit_approximant="rigid")
        analytic_positions = lisa_analytic.satellite_positions(query_time)
        analytic_arms = lisa_analytic.detector_arms(query_time)

        lisa_numeric = gwr.LISA(orbit_approximant="numeric", orbit_file=orbit_file)
        numeric_positions = lisa_numeric.satellite_positions(query_time)
        numeric_arms = lisa_numeric.detector_arms(query_time)

        # The numeric path must actually interpolate rather than silently
        # falling through to the analytical formulas.
        self.assertFalse(jnp.array_equal(numeric_positions, analytic_positions))
        self.assertFalse(jnp.array_equal(numeric_arms, analytic_arms))

        # Linear interpolation between coarse samples of a smooth orbit
        # should still stay close to the analytical model.
        position_error = jnp.max(
            jnp.abs(numeric_positions - analytic_positions)
        ) / jnp.max(jnp.abs(analytic_positions))
        arms_error = jnp.max(jnp.abs(numeric_arms - analytic_arms)) / jnp.max(
            jnp.abs(analytic_arms)
        )
        self.assertLess(position_error, 1e-2)
        self.assertLess(arms_error, 1e-2)

    def test_lisa_numeric_orbits_requires_orbit_file(self):
        with self.assertRaises(ValueError):
            gwr.LISA(orbit_approximant="numeric")

    def test_lisa_orbit_interpolation_method(self):
        orbit_radius = gwr.PhysicalConstants().AU
        eccentricity = gwr.LISA().ecc
        dense_time_grid = jnp.linspace(-0.1, 1.1, 2000)
        with tempfile.TemporaryDirectory() as tmp_dir:
            orbit_file = os.path.join(tmp_dir, "orbits.txt")
            write_numerical_orbit_file(
                orbit_file, dense_time_grid, orbit_radius, eccentricity
            )
            lisa_cubic = gwr.LISA(
                orbit_approximant="numeric",
                orbit_file=orbit_file,
                orbit_interpolation_method="cubic",
            )
        self.assertEqual(lisa_cubic.orbit_interpolator.method, "cubic")

    def test_lisa_unknown_orbit_model(self):
        lisa = gwr.LISA(orbit_approximant="bogus")
        with self.assertRaises(ValueError):
            lisa.satellite_positions(jnp.linspace(0, 1.0, 10))

    def test_lisa_keplerian_orbits_are_not_rigid(self):
        # Unlike the rigid model, the exact Keplerian model (Martens & Joffre
        # 2021, arXiv:2101.03040) should show a small "breathing" of the
        # constellation triangle: arm lengths and corner angles are not
        # exactly constant.
        time_in_years = jnp.linspace(0, 3.0, 3000)
        lisa_keplerian = gwr.LISA(orbit_approximant="keplerian")
        arms = lisa_keplerian.detector_arms(time_in_years)
        arm_lengths = jnp.linalg.norm(arms[:, :, 0], axis=1)

        self.assertGreater(arm_lengths.max() - arm_lengths.min(), 1e6)
        self.assertAlmostEqual(
            float(jnp.mean(arm_lengths)) / lisa_keplerian.armlength, 1.0, places=2
        )

    def test_lisa_keplerian_tilt_parameter_minimizes_flexing(self):
        # The reference paper reports that a tilt parameter of 5/8 minimizes
        # the arm length flexing of the Keplerian model.
        time_in_years = jnp.linspace(0, 3.0, 3000)

        def arm_length_range(tilt_parameter):
            lisa_keplerian = gwr.LISA(
                orbit_approximant="keplerian",
                keplerian_tilt_parameter=tilt_parameter,
            )
            arms = lisa_keplerian.detector_arms(time_in_years)
            arm_lengths = jnp.linalg.norm(arms[:, :, 0], axis=1)
            return arm_lengths.max() - arm_lengths.min()

        optimal_range = arm_length_range(5.0 / 8.0)
        for tilt_parameter in [0.0, 0.3, 1.0, 1.5]:
            self.assertLess(optimal_range, arm_length_range(tilt_parameter))


if __name__ == "__main__":
    unittest.main(verbosity=2)
