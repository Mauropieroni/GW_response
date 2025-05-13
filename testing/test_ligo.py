import unittest
import jax.numpy as jnp
import gw_response as gwr
import os
import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data/")


class TestLISA(unittest.TestCase):
    def test_ligo_1(self):
        return 0


if __name__ == "__main__":
    unittest.main(verbosity=2)
