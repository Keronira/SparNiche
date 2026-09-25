import unittest

import numpy as np

from src.plotting import _estimate_spot_diameter


class SpatialSpotDiameterTests(unittest.TestCase):
    def test_normalized_coordinates_keep_diameter_below_one(self):
        coordinates = np.array(
            [[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.01, 0.01]]
        )

        self.assertAlmostEqual(_estimate_spot_diameter(coordinates), 0.0085)

    def test_pixel_coordinates_retain_existing_scale(self):
        coordinates = np.array(
            [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
        )

        self.assertAlmostEqual(_estimate_spot_diameter(coordinates), 8.5)


if __name__ == "__main__":
    unittest.main()
