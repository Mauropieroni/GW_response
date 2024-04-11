import unittest
import jax.numpy as jnp
import gw_response as gwr


class TestConstant(unittest.TestCase):
    def test_constants(self):
        constants = gwr.PhysicalConstants()
        self.assertEqual(constants.light_speed, 299792458.0)
        self.assertEqual(constants.hour, 3600.0)
        self.assertEqual(constants.day, 86400.0)
        self.assertEqual(constants.yr, 31557600.0)
        self.assertEqual(constants.Hubble_over_h, 3.24e-18)
        self.assertEqual(constants.AU, 149597870700.0)
        for i in range(3):
            self.assertEqual(constants.cmb_dipole[i], [-0.972, 0.137, -0.191][i])

    def test_basis_transformations(self):
        basis_transformations = gwr.BasisTransformations()
        self.assertEqual(basis_transformations.XYZ_to_AET.shape, (3, 3))
        self.assertAlmostEqual(
            jnp.sum(
                jnp.abs(
                    basis_transformations.XYZ_to_AET
                    - jnp.array(
                        [
                            [-1 / jnp.sqrt(2), 0, 1 / jnp.sqrt(2)],
                            [
                                1 / jnp.sqrt(6),
                                -2 / jnp.sqrt(6),
                                1 / jnp.sqrt(6),
                            ],
                            [1 / jnp.sqrt(3), 1 / jnp.sqrt(3), 1 / jnp.sqrt(3)],
                        ]
                    ),
                )
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
