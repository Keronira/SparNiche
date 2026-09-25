import unittest
from pathlib import Path

import pandas as pd

from scripts.tmp_run_source26_loss_search import (
    FIXED_EMBEDDING,
    build_config,
    build_stage1a_specs,
    build_stage1b_specs,
    build_stage2_specs,
    select_best_levels,
    select_best_structures,
    summarize_and_rank,
)


class Source26LossSearchTests(unittest.TestCase):
    def test_stage1a_contains_one_no_dec_and_nine_dec_configs(self):
        specs = build_stage1a_specs()
        self.assertEqual(len(specs), 10)
        self.assertEqual(len({spec["config_id"] for spec in specs}), 10)
        no_dec = [spec for spec in specs if spec["factors"]["dec_kl_w"] == 0.0]
        self.assertEqual(len(no_dec), 1)
        self.assertEqual(no_dec[0]["factors"]["dec_cluster_n"], 10)

    def test_stage1b_contains_anchor_and_two_changes_per_factor(self):
        dec_structure = {"dec_cluster_n": 12, "dec_kl_w": 0.5}
        specs = build_stage1b_specs(dec_structure)
        self.assertEqual(len(specs), 7)
        self.assertEqual(len({spec["config_id"] for spec in specs}), 7)
        anchor = {"lr": 0.01, "gcn_w": 0.1, "rec_w": 10.0}
        for spec in specs:
            factors = spec["factors"]
            self.assertEqual(factors["dec_cluster_n"], 12)
            self.assertEqual(factors["dec_kl_w"], 0.5)
            changed = [key for key, value in anchor.items() if factors[key] != value]
            self.assertLessEqual(len(changed), 1)

    def test_stage2_crosses_two_levels_of_each_selected_factor(self):
        structures = [
            {"dec_cluster_n": 10, "dec_kl_w": 0.0},
            {"dec_cluster_n": 12, "dec_kl_w": 0.5},
        ]
        levels = {
            "lr": [0.005, 0.01],
            "gcn_w": [0.1, 0.4],
            "rec_w": [5.0, 10.0],
        }
        specs = build_stage2_specs(structures, levels)
        self.assertEqual(len(specs), 16)
        self.assertEqual(len({spec["config_id"] for spec in specs}), 16)

    def test_build_config_fixes_embedding_and_applies_loss_factors(self):
        base = {
            "data": {"preprocessing": {}},
            "model": {"sparniche_view1": {}, "sparniche": {}},
            "training": {},
            "evaluation": {},
            "benchmark": {},
        }
        factors = {
            "dec_cluster_n": 14,
            "dec_kl_w": 0.25,
            "lr": 0.005,
            "gcn_w": 0.4,
            "rec_w": 5.0,
        }
        config = build_config(
            base,
            input_path=Path("E9.5_E1S1.MOSTA.h5ad"),
            seed=1234,
            device="cuda",
            factors=factors,
        )
        self.assertEqual(config["data"]["preprocessing"]["n_top_genes"], 3000)
        self.assertEqual(config["data"]["preprocessing"]["pca_n_components"], 64)
        self.assertEqual(config["data"]["n_neighbors"], 8)
        self.assertEqual(config["model"]["sparniche_view1"]["attention_neighbors"], 8)
        self.assertEqual(config["model"]["latent_dim"], 8)
        self.assertEqual(FIXED_EMBEDDING["latent"], 8)
        self.assertEqual(config["model"]["sparniche_view1"]["dec_cluster_n"], 14)
        self.assertEqual(config["model"]["sparniche"]["dec_kl_w"], 0.25)
        self.assertEqual(config["model"]["sparniche"]["lr"], 0.005)
        self.assertEqual(config["model"]["sparniche"]["gcn_w"], 0.4)
        self.assertEqual(config["model"]["sparniche"]["rec_w"], 5.0)
        self.assertEqual(config["evaluation"]["leiden_cluster_lower_offset"], 0)
        self.assertEqual(config["evaluation"]["leiden_cluster_upper_offset"], 2)

    def test_ranking_prefers_worst_case_within_ari_tolerance(self):
        frame = pd.DataFrame(
            [
                {"config_id": "unstable", "seed": 1, "status": "completed", "ari": 0.43, "nmi": 0.5, "fmi": 0.5, "accuracy": 0.5, "macro_f1": 0.5},
                {"config_id": "unstable", "seed": 2, "status": "completed", "ari": 0.27, "nmi": 0.5, "fmi": 0.5, "accuracy": 0.5, "macro_f1": 0.5},
                {"config_id": "stable", "seed": 1, "status": "completed", "ari": 0.346, "nmi": 0.6, "fmi": 0.5, "accuracy": 0.5, "macro_f1": 0.6},
                {"config_id": "stable", "seed": 2, "status": "completed", "ari": 0.346, "nmi": 0.6, "fmi": 0.5, "accuracy": 0.5, "macro_f1": 0.6},
                {"config_id": "lower", "seed": 1, "status": "completed", "ari": 0.30, "nmi": 0.7, "fmi": 0.5, "accuracy": 0.5, "macro_f1": 0.7},
                {"config_id": "lower", "seed": 2, "status": "completed", "ari": 0.30, "nmi": 0.7, "fmi": 0.5, "accuracy": 0.5, "macro_f1": 0.7},
            ]
        )
        ranking = summarize_and_rank(frame, ari_tolerance=0.01)
        self.assertEqual(ranking["config_id"].tolist(), ["stable", "unstable", "lower"])

    def test_selection_helpers_return_requested_candidates(self):
        stage1a = pd.DataFrame(
            [
                {"config_id": "a", "dec_cluster_n": 10, "dec_kl_w": 0.0, "rank": 2},
                {"config_id": "b", "dec_cluster_n": 12, "dec_kl_w": 0.5, "rank": 1},
                {"config_id": "c", "dec_cluster_n": 14, "dec_kl_w": 1.0, "rank": 3},
            ]
        )
        self.assertEqual(
            select_best_structures(stage1a, count=2),
            [
                {"dec_cluster_n": 12, "dec_kl_w": 0.5},
                {"dec_cluster_n": 10, "dec_kl_w": 0.0},
            ],
        )

        stage1b = pd.DataFrame(
            [
                {"lr": 0.005, "gcn_w": 0.1, "rec_w": 10.0, "rank": 1},
                {"lr": 0.01, "gcn_w": 0.1, "rec_w": 10.0, "rank": 2},
                {"lr": 0.02, "gcn_w": 0.1, "rec_w": 10.0, "rank": 3},
                {"lr": 0.01, "gcn_w": 0.025, "rec_w": 10.0, "rank": 4},
                {"lr": 0.01, "gcn_w": 0.4, "rec_w": 10.0, "rank": 5},
                {"lr": 0.01, "gcn_w": 0.1, "rec_w": 2.5, "rank": 6},
                {"lr": 0.01, "gcn_w": 0.1, "rec_w": 5.0, "rank": 7},
            ]
        )
        selected = select_best_levels(stage1b)
        self.assertEqual(selected["lr"], [0.005, 0.01])
        self.assertEqual(selected["gcn_w"], [0.1, 0.025])
        self.assertEqual(selected["rec_w"], [10.0, 2.5])


if __name__ == "__main__":
    unittest.main()
