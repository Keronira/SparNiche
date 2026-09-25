import unittest
from pathlib import Path

import pandas as pd

from scripts.tmp_run_source26_config_search import (
    ANCHOR,
    FACTOR_LEVELS,
    build_config,
    build_stage1_specs,
    build_stage2_specs,
    select_best_levels,
    select_stage3_specs,
)


class Source26ConfigSearchTests(unittest.TestCase):
    def test_stage1_has_fourteen_unique_one_factor_configs(self):
        specs = build_stage1_specs()
        self.assertEqual(len(specs), 14)
        self.assertEqual(len({spec["config_id"] for spec in specs}), 14)
        self.assertEqual(specs[0]["factors"], ANCHOR)
        for spec in specs[1:]:
            changed = [key for key in ANCHOR if spec["factors"][key] != ANCHOR[key]]
            self.assertEqual(len(changed), 1)

    def test_build_config_changes_only_four_search_factors(self):
        base = {
            "data": {
                "n_neighbors": 12,
                "preprocessing": {
                    "n_top_genes": 2000,
                    "pca_n_components": 200,
                },
            },
            "model": {
                "latent_dim": 32,
                "sparniche_view1": {
                    "attention_neighbors": None,
                    "dropout": 0.2,
                },
                "sparniche": {"lr": 0.01, "weight_decay": 0.01},
            },
            "training": {"epochs": 550},
            "evaluation": {},
            "benchmark": {"external_only": True},
        }
        factors = {"hvg": 5000, "pca": 64, "neighbors": 8, "latent": 16}
        config = build_config(
            base,
            input_path=Path("E9.5_E1S1.MOSTA.h5ad"),
            seed=1234,
            device="cuda",
            factors=factors,
        )
        self.assertEqual(config["data"]["preprocessing"]["n_top_genes"], 5000)
        self.assertEqual(config["data"]["preprocessing"]["pca_n_components"], 64)
        self.assertEqual(config["data"]["n_neighbors"], 8)
        self.assertEqual(config["model"]["sparniche_view1"]["attention_neighbors"], 8)
        self.assertEqual(config["model"]["latent_dim"], 16)
        self.assertEqual(config["model"]["sparniche_view1"]["dropout"], 0.2)
        self.assertEqual(config["model"]["sparniche"]["lr"], 0.01)
        self.assertEqual(config["model"]["sparniche"]["weight_decay"], 0.01)
        self.assertEqual(config["training"]["epochs"], 550)
        self.assertEqual(config["evaluation"]["leiden_cluster_lower_offset"], 0)
        self.assertEqual(config["evaluation"]["leiden_cluster_upper_offset"], 2)

    def test_stage2_uses_two_best_levels_per_factor(self):
        rows = []
        score = 0.1
        for spec in build_stage1_specs():
            rows.append({
                "status": "completed",
                "stage": 1,
                "seed": 1234,
                "ari": score,
                **spec["factors"],
            })
            score += 0.1
        frame = pd.DataFrame(rows)
        selected = select_best_levels(frame)
        self.assertEqual(set(selected), set(FACTOR_LEVELS))
        self.assertTrue(all(len(levels) == 2 for levels in selected.values()))
        specs = build_stage2_specs(selected)
        self.assertEqual(len(specs), 16)
        self.assertEqual(len({spec["config_id"] for spec in specs}), 16)

    def test_stage3_selects_four_highest_ari_stage2_configs(self):
        rows = []
        for index in range(6):
            rows.append({
                "stage": 2,
                "status": "completed",
                "seed": 1234,
                "ari": index / 10,
                "hvg": 2000 + index,
                "pca": 32,
                "neighbors": 8,
                "latent": 16,
                "config_id": f"config_{index}",
            })
        selected = select_stage3_specs(pd.DataFrame(rows), count=4)
        self.assertEqual(
            [spec["config_id"] for spec in selected],
            ["config_5", "config_4", "config_3", "config_2"],
        )


if __name__ == "__main__":
    unittest.main()
