import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.tmp_run_source2_neighbor_latent import (
    EXPERIMENT_SPECS,
    QUALITY_METRICS,
    build_config,
    write_summaries,
)


class Source2NeighborLatentExperimentTests(unittest.TestCase):
    def test_factorial_registry_has_one_anchor_and_twelve_unique_sets(self):
        self.assertEqual(len(EXPERIMENT_SPECS), 12)
        self.assertEqual(len({row["set_name"] for row in EXPERIMENT_SPECS}), 12)
        anchors = [row for row in EXPERIMENT_SPECS if row["is_anchor"]]
        self.assertTrue(EXPERIMENT_SPECS[0]["is_anchor"])
        self.assertEqual(
            anchors,
            [{"set_name": "neighbors_12_latent_32", "n_neighbors": 12,
              "latent_dim": 32, "is_anchor": True}],
        )

    def test_build_config_changes_only_requested_neighbor_and_latent_fields(self):
        base = {
            "data": {"n_neighbors": 12},
            "model": {
                "latent_dim": 32,
                "sparniche_view1": {
                    "attention_mode": "spatial_local", "attention_neighbors": None,
                    "attention_chunk_size": None,
                },
                "sparniche": {"lr": 0.01},
            },
            "training": {"epochs": 550},
            "evaluation": {},
        }
        config = build_config(
            base, input_path=Path("200727_09.h5ad"), seed=1235,
            device="cuda", n_neighbors=16, latent_dim=64,
        )
        self.assertEqual(config["data"]["n_neighbors"], 16)
        self.assertEqual(config["model"]["sparniche_view1"]["attention_neighbors"], 16)
        self.assertEqual(config["model"]["latent_dim"], 64)
        self.assertEqual(config["model"]["sparniche"]["lr"], 0.01)
        self.assertEqual(config["training"]["epochs"], 550)
        self.assertIsNone(config["model"]["sparniche_view1"]["attention_chunk_size"])
        self.assertEqual(config["evaluation"]["leiden_cluster_lower_offset"], 0)
        self.assertEqual(config["evaluation"]["leiden_cluster_upper_offset"], 2)

    def test_write_summaries_reports_mean_sd_ranks_and_anchor_deltas(self):
        rows = []
        for set_name, neighbor, latent, values in (
            ("neighbors_12_latent_32", 12, 32, (0.4, 0.5, 0.6)),
            ("neighbors_08_latent_64", 8, 64, (0.5, 0.6, 0.7)),
        ):
            for seed, value in zip((1234, 1235, 1236), values):
                rows.append({
                    "set_name": set_name, "n_neighbors": neighbor,
                    "latent_dim": latent, "is_anchor": set_name == "neighbors_12_latent_32",
                    "seed": seed, "status": "completed",
                    **{metric: value for metric in QUALITY_METRICS},
                    "runtime_seconds": 10.0, "peak_cuda_memory_mb": 100.0,
                })
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_summaries(pd.DataFrame(rows), root)
            summary = pd.read_csv(root / "summary_by_config.csv")
            ranks = pd.read_csv(root / "quality_ranks.csv")
            deltas = pd.read_csv(root / "paired_deltas_vs_anchor.csv")
            effects = pd.read_csv(root / "factor_effects.csv")

        candidate = summary[summary["set_name"] == "neighbors_08_latent_64"].iloc[0]
        self.assertAlmostEqual(candidate["ari_mean"], 0.6)
        self.assertAlmostEqual(candidate["ari_sd"], 0.1)
        self.assertEqual(ranks.iloc[0]["set_name"], "neighbors_08_latent_64")
        ari_delta = deltas[(deltas["set_name"] == "neighbors_08_latent_64")
                           & (deltas["metric"] == "ari")].iloc[0]
        self.assertAlmostEqual(ari_delta["mean_delta_vs_anchor"], 0.1)
        self.assertEqual(set(effects["factor"]), {"n_neighbors", "latent_dim"})


if __name__ == "__main__":
    unittest.main()
