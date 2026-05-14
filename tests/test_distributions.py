"""Basic tests for the package scaffold."""

import unittest

from spherics import spherical_distribution


class TestDistributions(unittest.TestCase):
    def test_spherical_distribution_returns_list(self) -> None:
        values = (1.0, 2.0, 3.0)
        result = spherical_distribution(values)
        self.assertEqual(result, [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
