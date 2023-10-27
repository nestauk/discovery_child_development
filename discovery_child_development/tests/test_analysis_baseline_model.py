import unittest
import pandas as pd
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
