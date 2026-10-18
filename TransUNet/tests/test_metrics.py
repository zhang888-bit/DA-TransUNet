import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import calculate_metric_percase


class MetricTests(unittest.TestCase):
    def test_false_positive_does_not_receive_perfect_dice(self):
        prediction = np.array([[1, 0], [0, 0]], dtype=bool)
        ground_truth = np.zeros((2, 2), dtype=bool)
        dice, hd95 = calculate_metric_percase(prediction, ground_truth)
        self.assertEqual(dice, 0.0)
        self.assertEqual(hd95, 0.0)

    def test_empty_prediction_with_empty_ground_truth(self):
        empty = np.zeros((2, 2), dtype=bool)
        self.assertEqual(calculate_metric_percase(empty, empty.copy()), (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
