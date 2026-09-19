from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import anndata
import numpy as np
import torch

from src import data
from src.config import load_yaml
from src.data import build_sparniche_graph
from src.models import SparNiche, SparNicheEncoder
from src.pipeline import PipelineArtifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SingleViewContractTests(unittest.TestCase):
    def test_data_module_exposes_single_view_resolver(self) -> None:
        self.assertTrue(hasattr(data, "resolve_view1"))

    def test_resolve_view1_ignores_image_features(self) -> None:
        adata = anndata.AnnData(np.ones((4, 2), dtype=np.float32))
        adata.obsm["feat"] = np.arange(12, dtype=np.float32).reshape(4, 3)
        adata.obsm["image_features"] = np.full((4, 5), 99.0, dtype=np.float32)

        features = data.resolve_view1(
            adata, {"view1": {"source": "obsm", "key": "feat"}}
        )

        self.assertEqual(tuple(features.shape), (4, 3))
        np.testing.assert_array_equal(features.numpy(), adata.obsm["feat"])

    def test_model_constructor_and_forward_are_single_view(self) -> None:
        constructor = inspect.signature(SparNiche.__init__)
        forward = inspect.signature(SparNiche.forward)
        self.assertIn("input_dim", constructor.parameters)
        self.assertNotIn("input_dims", constructor.parameters)
        self.assertEqual(
            [name for name in forward.parameters if name != "self"],
            ["view1", "neighbor_idx"],
        )

    def test_model_forward_returns_one_embedding_per_spot(self) -> None:
        model = SparNiche(
            input_dim=8,
            latent_dim=4,
            sparniche_config={
                "attention_mode": "spatial_local",
                "attention_neighbors": 2,
                "attention_chunk_size": 8,
            },
        ).eval()
        view1 = torch.randn(6, 8)
        neighbors = torch.tensor(
            [[1, 2], [0, 2], [1, 3], [2, 4], [3, 5], [3, 4]],
            dtype=torch.long,
        )
        model.configure_graph(
            build_sparniche_graph(
                np.asarray(
                    [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
                    dtype=np.float32,
                ),
                n_neighbors=2,
            )
        )

        with torch.no_grad():
            output = model(view1, neighbors)

        self.assertEqual(tuple(output.embedding.shape), (6, 4))
        self.assertEqual(tuple(output.reconstruction.shape), (6, 8))
        self.assertEqual(tuple(output.q.shape), (6, 10))

    def test_default_config_has_no_view2_or_fusion_stage(self) -> None:
        config = load_yaml(PROJECT_ROOT / "configs" / "config.yaml")
        self.assertNotIn("view2", config["data"])
        self.assertNotIn("fusion_epochs", config["training"])
        for key in (
            "expert_hidden_dim",
            "reliability_threshold",
            "router_temperature",
            "hard_fallback",
            "joint_expert_enabled",
            "use_spatial_quality",
        ):
            self.assertNotIn(key, config["model"])

    def test_encoder_accepts_configured_dropout_sweep_values(self) -> None:
        for dropout in (0.0, 0.2, 0.4):
            encoder = SparNicheEncoder(
                input_dim=8,
                latent_dim=32,
                hidden_dims=(64, 16),
                num_heads=1,
                dropout=dropout,
                dec_cluster_n=10,
            )
            self.assertEqual(encoder.dropout, dropout)

    def test_pipeline_artifacts_expose_one_feature_tensor(self) -> None:
        fields = set(PipelineArtifacts.__dataclass_fields__)
        self.assertIn("features", fields)
        self.assertNotIn("views", fields)


if __name__ == "__main__":
    unittest.main()
