from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anndata
import numpy as np

from src.pipeline import train_adata


class SingleViewSmokeTests(unittest.TestCase):
    def test_three_stage_cpu_training_writes_single_embedding(self) -> None:
        rng = np.random.default_rng(7)
        adata = anndata.AnnData(rng.normal(size=(12, 8)).astype(np.float32))
        adata.obsm["spatial"] = np.column_stack(
            (np.arange(12, dtype=np.float32), np.arange(12, dtype=np.float32) % 3)
        )
        config = {
            "data": {
                "n_neighbors": 2,
                "preprocessing": {"enabled": False},
                "view1": {"source": "X"},
            },
            "model": {
                "latent_dim": 4,
                "feature_graph_fusion_mode": "gated",
                "local_graph_mode": "normalized",
                "sparniche_view1": {
                    "hidden_dims": [64, 16],
                    "num_heads": 1,
                    "dropout": 0.2,
                    "dec_cluster_n": 10,
                    "attention_mode": "spatial_local",
                    "attention_neighbors": 2,
                    "attention_chunk_size": 8,
                },
                "sparniche": {
                    "gan_epochs": 1,
                    "gan_lr": 0.0001,
                    "pretrain_epochs": 1,
                    "lr": 0.001,
                    "weight_decay": 0.0,
                    "negative_repeats": 1,
                    "rec_w": 1.0,
                    "gcn_w": 0.1,
                    "self_w": 0.1,
                    "dec_kl_w": 0.1,
                    "dec_interval": 1,
                    "dec_tol": 0.0,
                },
            },
            "training": {
                "epochs": 1,
                "checkpoint_every": 1,
                "seed": 7,
                "device": "cpu",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = train_adata(adata, config, Path(temp_dir))
            self.assertEqual(artifacts.adata.obsm["sparniche"].shape, (12, 4))
            self.assertTrue((Path(temp_dir) / "checkpoint.pt").is_file())
            self.assertTrue((Path(temp_dir) / "trained.h5ad").is_file())
            self.assertNotIn("sparniche_router_weights", artifacts.adata.obsm)
            self.assertNotIn("sparniche_reliability", artifacts.adata.obsm)


if __name__ == "__main__":
    unittest.main()
