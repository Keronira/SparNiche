import unittest

import pandas as pd

from scripts.tmp_run_source27_config_search import SCREEN_SPECS, select_validation_specs


class Source27ConfigSearchTests(unittest.TestCase):
    def test_screen_specs_respect_gene_count_and_neighbor_pairing(self):
        self.assertEqual(len(SCREEN_SPECS), 6)
        self.assertTrue(all(spec["hvg"] <= 1122 for spec in SCREEN_SPECS))
        self.assertEqual({spec["name"] for spec in SCREEN_SPECS}, set("ABCDEF"))

    def test_selects_top_two_completed_by_ari_then_nmi(self):
        rows = pd.DataFrame([
            {"config_id": "A", "status": "completed", "ari": 0.3, "nmi": 0.4},
            {"config_id": "B", "status": "completed", "ari": 0.4, "nmi": 0.3},
            {"config_id": "C", "status": "completed", "ari": 0.4, "nmi": 0.5},
            {"config_id": "D", "status": "failed", "ari": 0.9, "nmi": 0.9},
        ])
        self.assertEqual([spec["config_id"] for spec in select_validation_specs(rows, specs=[
            {"config_id": name} for name in "ABCD"
        ])], ["C", "B"])


if __name__ == "__main__":
    unittest.main()
