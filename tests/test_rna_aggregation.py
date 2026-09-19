import unittest

import pandas as pd

from scripts.aggregate_rna_experiments import build_summary


class RNAAggregationTests(unittest.TestCase):
    def test_summary_contains_mean_std_per_case(self):
        frame = pd.DataFrame(
            {
                "case": ["anchor", "anchor", "gated_fusion"],
                "ari": [0.4, 0.6, 0.7],
                "nmi": [0.5, 0.7, 0.8],
                "ami": [0.3, 0.5, 0.6],
            }
        )
        summary = build_summary(frame)
        anchor = summary[summary["case"] == "anchor"].iloc[0]
        self.assertAlmostEqual(anchor["ari_mean"], 0.5)
        self.assertAlmostEqual(anchor["ari_std"], 0.1414213562, places=5)
        self.assertEqual(int(anchor["n_records"]), 2)
