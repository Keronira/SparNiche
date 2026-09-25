import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.tmp_run_layer_recovery_sensitivity_step1 import (
    STEP1_SAMPLES,
    STEP1_SEED,
    STEP1_SET_OVERRIDES,
    build_step1_config,
    write_step1_tables,
)


class LayerSensitivityStep1Tests(unittest.TestCase):
    def test_step1_has_six_balanced_samples_and_one_seed(self):
        self.assertEqual(
            STEP1_SAMPLES,
            ("151507", "151510", "151669", "151670", "151673", "151676"),
        )
        self.assertEqual(STEP1_SEED, 1234)

    def test_step1_has_24_deduplicated_sets(self):
        self.assertEqual(len(STEP1_SET_OVERRIDES), 24)
        self.assertIn("anchor", STEP1_SET_OVERRIDES)
        self.assertIn("neighbors_08_attention_16", STEP1_SET_OVERRIDES)
        self.assertIn("latent_128_norm_off", STEP1_SET_OVERRIDES)
        self.assertIn("epochs_800", STEP1_SET_OVERRIDES)

    def test_step1_config_uses_layer_aware_leiden_window(self):
        base = {
            "model": {
                "latent_dim": 64,
                "local_graph_normalize": True,
                "sparniche_view1": {"attention_neighbors": 12},
                "sparniche": {"lr": 0.005},
            },
            "data": {"n_neighbors": 12},
            "training": {"epochs": 550},
        }
        config = build_step1_config(
            base,
            Path("151670.h5ad"),
            set_name="neighbors_08_attention_16",
            device="cpu",
            input_kind="raw_counts",
        )
        self.assertEqual(config["data"]["n_neighbors"], 8)
        self.assertEqual(config["model"]["sparniche_view1"]["attention_neighbors"], 16)
        self.assertEqual(config["evaluation"]["leiden_cluster_lower_offset"], 0)
        self.assertEqual(config["evaluation"]["leiden_cluster_upper_offset"], 2)
        self.assertNotIn("leiden_selection_strategy", config["evaluation"])

    def test_fallback_run_is_ineligible_for_ranking(self):
        metrics = {
            "ari": 0.8,
            "nmi": 0.8,
            "macro_layer_iou": 0.8,
            "worst_layer_iou": 0.8,
            "layer_recovery_rate": 1.0,
        }
        bundles = []
        for set_name, fallback in (("anchor", False), ("fallback", True)):
            bundles.append(
                {
                    "run": {
                        "set_name": set_name,
                        "sample": "151670",
                        "seed": 1234,
                        "status": "completed",
                        "leiden_window_fallback": fallback,
                        **metrics,
                    },
                    "per_layer_iou": {"Layer3": 0.8},
                    "leiden_audit": {},
                }
            )
        with TemporaryDirectory() as directory:
            write_step1_tables(bundles, Path(directory))
            ranking = pd.read_csv(Path(directory) / "configuration_ranking.csv").set_index(
                "set_name"
            )
            self.assertFalse(bool(ranking.loc["fallback", "eligible"]))
            self.assertEqual(int(ranking.loc["fallback", "window_fallback_runs"]), 1)

    def test_failed_first_run_still_writes_incremental_tables(self):
        bundles = [
            {
                "run": {
                    "set_name": "anchor",
                    "sample": "151507",
                    "seed": 1234,
                    "status": "failed",
                    "error": "synthetic failure",
                },
                "per_layer_iou": {},
                "leiden_audit": {},
            }
        ]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_step1_tables(bundles, root)
            self.assertTrue((root / "run_metrics.csv").exists())
            self.assertTrue((root / "configuration_ranking.csv").exists())


if __name__ == "__main__":
    unittest.main()
