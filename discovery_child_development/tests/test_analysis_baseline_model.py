import numpy as np
import pandas as pd
from typing import List
import unittest

import discovery_child_development.analysis.baseline_model as bm


class TestFindMostCommonRow(unittest.TestCase):
    def test_find_most_common_row(self):
        # Test case with a DataFrame
        df = pd.DataFrame(
            {"Label_A": [1, 0, 1, 1], "Label_B": [0, 1, 0, 0], "Label_C": [1, 1, 1, 1]}
        )
        expected_result = ("101", 3)
        result = bm.find_most_common_row(df)
        self.assertEqual(result, expected_result)


class TestGeneratePredictions(unittest.TestCase):
    def test_return_type(self):
        labels = ["a", "b", "c"]
        label_probabilities = {"a": 0.7, "b": 0.2, "c": 0.5}

        result = bm.generate_predictions(labels, label_probabilities)

        # Check if the result is a list
        self.assertIsInstance(result, List, "The result should be a list")

        # Check if all elements in the result are integers
        for element in result:
            self.assertIsInstance(
                element,
                (int, np.int64),
                "All elements in the result list should be integers",
            )

    def test_output_length(self):
        labels = ["a", "b", "c"]
        label_probabilities = {"a": 0.7, "b": 0.2, "c": 0.5}

        result = bm.generate_predictions(labels, label_probabilities)

        # Check if the length of the output list matches the length of the input labels
        self.assertEqual(
            len(result),
            len(labels),
            "The length of the output list should match the length of the input labels",
        )
