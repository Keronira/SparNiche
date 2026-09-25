import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.tmp_validate_source26_dynamics_3sample import (
    CANDIDATES,
    SAMPLE_NAMES,
    SEEDS,
    build_config,
    build_run_specs,
    paired_differences,
    summarize_runs,
)


class Source26DynamicThreeSampleValidationTests(unittest.TestCase):
    def test_run_specs_cover_three_configs_samples_and_seeds_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for sample in SAMPLE_NAMES:
                (root / sample).touch()
            specs = build_run_specs(root)

        self.assertEqual(len(specs), 27)
        self.assertEqual(len({item["run_id"] for item in specs}), 27)
        self.assertEqual({item["seed"] for item in specs}, {1234, 1235, 1236})
        self.assertEqual(
            {item["input_path"].name for item in specs}, set(SAMPLE_NAMES)
        )
        self.assertEqual(
            {item["config_id"] for item in specs}, set(CANDIDATES)
        )

    def test_build_config_preserves_source26_fixed_values(self):
        base = {
            "data": {
                "n_neighbors": 8,
                "preprocessing": {"n_top_genes": 3000, "pca_n_components": 64},
            },
            "model": {
                "latent_dim": 8,
                "local_graph_hops": 1,
                "sparniche_view1": {
                    "attention_neighbors": 8,
                    "dec_cluster_n": 14,
                },
                "sparniche": {
                    "dec_interval": 20,
                    "dec_kl_w": 0.5,
                    "lr": 0.01,
                    "gcn_w": 0.1,
                    "rec_w": 10.0,
                    "weight_decay": 0.01,
                },
            },
            "training": {"epochs": 550, "seed": 2023, "device": "auto"},
            "evaluation": {},
            "benchmark": {},
            "paths": {},
        }
        config = build_config(
            base,
            input_path=Path("E9.5_E1S1.MOSTA.h5ad"),
            seed=1235,
            device="cuda",
            candidate=CANDIDATES["int40_ep300_hops2"],
        )

        self.assertEqual(config["model"]["sparniche"]["dec_interval"], 40)
        self.assertEqual(config["training"]["epochs"], 300)
        self.assertEqual(config["model"]["local_graph_hops"], 2)
        self.assertEqual(config["model"]["sparniche"]["weight_decay"], 0.01)
        self.assertEqual(config["model"]["sparniche_view1"]["attention_neighbors"], 8)
        self.assertEqual(config["model"]["sparniche_view1"]["dec_cluster_n"], 14)
        self.assertEqual(config["model"]["latent_dim"], 8)
        self.assertEqual(config["training"]["seed"], 1235)
        self.assertEqual(config["evaluation"]["leiden_cluster_lower_offset"], 0)
        self.assertEqual(config["evaluation"]["leiden_cluster_upper_offset"], 2)

    def test_summary_weights_samples_equally(self):
        rows = []
        for sample, values in {"s1": (0.2, 0.4), "s2": (0.6, 0.8)}.items():
            for seed, ari in zip((1234, 1235), values):
                rows.append(
                    {
                        "config_id": "anchor",
                        "sample": sample,
                        "seed": seed,
                        "status": "completed",
                        "ari": ari,
                        "nmi": ari,
                        "fmi": ari,
                        "accuracy": ari,
                        "macro_f1": ari,
                    }
                )
        by_sample, overall = summarize_runs(pd.DataFrame(rows), expected_seeds=2)

        self.assertEqual(len(by_sample), 2)
        self.assertAlmostEqual(float(overall.iloc[0]["ari_mean"]), 0.5)
        self.assertAlmostEqual(float(overall.iloc[0]["ari_min_sample"]), 0.3)

    def test_paired_differences_match_sample_and_seed(self):
        frame = pd.DataFrame(
            [
                {"config_id": "anchor", "sample": "s1", "seed": 1, "status": "completed", "ari": 0.3, "nmi": 0.4, "fmi": 0.5, "accuracy": 0.6, "macro_f1": 0.7},
                {"config_id": "candidate", "sample": "s1", "seed": 1, "status": "completed", "ari": 0.5, "nmi": 0.5, "fmi": 0.6, "accuracy": 0.4, "macro_f1": 0.8},
            ]
        )
        details, summary = paired_differences(frame, anchor_id="anchor")

        self.assertAlmostEqual(float(details.iloc[0]["ari_delta"]), 0.2)
        self.assertAlmostEqual(float(details.iloc[0]["accuracy_delta"]), -0.2)
        self.assertAlmostEqual(float(summary.iloc[0]["ari_win_rate"]), 1.0)


if __name__ == "__main__":
    unittest.main()
