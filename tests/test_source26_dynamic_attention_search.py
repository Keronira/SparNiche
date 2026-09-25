import unittest
from pathlib import Path

from scripts.tmp_run_source26_dynamic_attention_search import (
    ATTENTION_LEVELS,
    DYNAMIC_LEVELS,
    WEIGHT_DECAY_LEVELS,
    build_attention_specs,
    build_config,
    build_dynamic_specs,
)


class Source26DynamicAttentionSearchTests(unittest.TestCase):
    def test_dynamic_grid_is_full_factorial(self):
        specs = build_dynamic_specs()
        self.assertEqual(len(specs), 4 * 3 * 3)
        self.assertEqual(
            len({spec["config_id"] for spec in specs}), len(specs)
        )
        for spec in specs:
            self.assertTrue(set(DYNAMIC_LEVELS).issubset(spec["factors"]))
            self.assertEqual(spec["factors"]["weight_decay"], 0.01)
            self.assertEqual(spec["factors"]["attention_neighbors"], 8)

    def test_attention_stage_is_weight_decay_by_attention(self):
        dynamic = {"dec_interval": 20, "epochs": 550, "local_graph_hops": 1}
        specs = build_attention_specs(dynamic)
        self.assertEqual(len(specs), len(WEIGHT_DECAY_LEVELS) * len(ATTENTION_LEVELS))
        for spec in specs:
            self.assertEqual(spec["factors"]["dec_interval"], 20)
            self.assertEqual(spec["factors"]["epochs"], 550)
            self.assertEqual(spec["factors"]["local_graph_hops"], 1)

    def test_build_config_only_changes_requested_search_fields(self):
        base = {
            "data": {"n_neighbors": 8},
            "model": {
                "latent_dim": 8,
                "local_graph_hops": 1,
                "sparniche_view1": {"attention_neighbors": 8},
                "sparniche": {
                    "dec_interval": 20,
                    "weight_decay": 0.01,
                },
            },
            "training": {"epochs": 550, "seed": 2023, "device": "auto"},
        }
        config = build_config(
            base,
            input_path=Path("sample.h5ad"),
            seed=1234,
            device="cuda",
            factors={
                "dec_interval": 10,
                "epochs": 300,
                "local_graph_hops": 2,
                "weight_decay": 0.001,
                "attention_neighbors": 16,
            },
        )
        self.assertEqual(config["model"]["sparniche"]["dec_interval"], 10)
        self.assertEqual(config["training"]["epochs"], 300)
        self.assertEqual(config["model"]["local_graph_hops"], 2)
        self.assertEqual(config["model"]["sparniche"]["weight_decay"], 0.001)
        self.assertEqual(
            config["model"]["sparniche_view1"]["attention_neighbors"], 16
        )
        self.assertEqual(config["model"]["latent_dim"], 8)
        self.assertEqual(config["data"]["n_neighbors"], 8)


if __name__ == "__main__":
    unittest.main()
