from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import anndata
import numpy as np
import torch
from scipy import sparse

from src.data import build_sparniche_graph, resolve_view2, resolve_view2_key
from src.models import SparNiche
from src.trainer import _sparniche_losses
from src.pipeline import train_adata
from scripts import run_experiments


class DoubleViewAdtTests(unittest.TestCase):
    def test_existing_single_view_command_keeps_double_view_disabled(self) -> None:
        with patch("sys.argv", [
            "run_experiments.py", "--experiment-set", "single",
            "--source", "source29", "--device", "cuda", "--continue-on-error",
        ]):
            args = run_experiments.parse_args()
        self.assertFalse(args.double_view)
        self.assertEqual(args.view2_key, "auto")
        self.assertEqual(args.source, [Path("source29")])

    def test_single_file_runs_have_distinct_sample_artifact_ids(self) -> None:
        self.assertEqual(
            run_experiments.sample_run_id(Path("/data/source30/S1.h5ad"), "single-seed1234"),
            "S1--single-seed1234",
        )
        self.assertNotEqual(
            run_experiments.sample_run_id(Path("/data/source30/S1.h5ad"), "single-seed1234"),
            run_experiments.sample_run_id(Path("/data/source30/S2.h5ad"), "single-seed1234"),
        )

    def test_resolve_view2_uses_configured_obsm_key_and_standardizes_adt(self) -> None:
        adata = anndata.AnnData(np.ones((4, 3), dtype=np.float32))
        adata.obsm["adt"] = np.array(
            [[0.0, 2.0], [1.0, 4.0], [3.0, 8.0], [7.0, 16.0]],
            dtype=np.float32,
        )
        values = resolve_view2(adata, {"view2": {"source": "obsm", "key": "adt"}})
        self.assertEqual(tuple(values.shape), (4, 2))
        np.testing.assert_allclose(values.mean(dim=0).numpy(), [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(values.std(dim=0, unbiased=False).numpy(), [1.0, 1.0], atol=1e-6)

    def test_resolve_view2_rejects_missing_adt(self) -> None:
        adata = anndata.AnnData(np.ones((4, 3), dtype=np.float32))
        with self.assertRaisesRegex(KeyError, "adt"):
            resolve_view2(adata, {"view2": {"source": "obsm", "key": "adt"}})

    def test_auto_selects_adt_then_atac_and_allows_rna_only(self) -> None:
        adata = anndata.AnnData(np.ones((6, 3), dtype=np.float32))
        self.assertIsNone(resolve_view2_key(adata, {"view2": {"key": "auto"}}))
        adata.obsm["atac"] = sparse.csr_matrix(np.eye(6, 20, dtype=np.float32))
        self.assertEqual(resolve_view2_key(adata, {"view2": {"key": "auto"}}), "atac")
        adata.obsm["adt"] = np.ones((6, 2), dtype=np.float32)
        self.assertEqual(resolve_view2_key(adata, {"view2": {"key": "auto"}}), "adt")

    def test_sparse_atac_is_reduced_before_encoder(self) -> None:
        adata = anndata.AnnData(np.ones((8, 3), dtype=np.float32))
        adata.obsm["atac"] = sparse.csr_matrix(np.eye(8, 1000, dtype=np.float32))
        values = resolve_view2(adata, {"view2": {"key": "atac", "atac_n_components": 4}})
        self.assertEqual(tuple(values.shape), (8, 4))
        self.assertTrue(torch.isfinite(values).all())

    def test_double_view_fusion_and_reconstruction_train_adt_encoder(self) -> None:
        model = SparNiche(
            input_dim=8,
            latent_dim=4,
            double_view=True,
            view2_dim=3,
            sparniche_config={
                "attention_mode": "spatial_local",
                "attention_neighbors": 2,
                "attention_chunk_size": 8,
            },
        )
        model.configure_graph(build_sparniche_graph(
            np.asarray([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float32),
            n_neighbors=2,
        ))
        model.eval()
        rna = torch.randn(6, 8)
        adt = torch.randn(6, 3)
        neighbors = torch.tensor([[1, 2], [0, 2], [1, 3], [2, 4], [3, 5], [3, 4]])

        first = model(rna, neighbors, adt)
        second = model(rna, neighbors, adt + 2.0)
        self.assertEqual(tuple(first.embedding.shape), (6, 4))
        self.assertEqual(tuple(first.aux["adt_reconstruction"].shape), (6, 3))
        self.assertFalse(torch.allclose(first.embedding, second.embedding))

        losses = _sparniche_losses(first.aux, rna, None, clean_adt=adt)
        losses["adt_reconstruction"].backward()
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in model.adt_encoder.parameters()
        ))

    def test_single_view_does_not_require_adt(self) -> None:
        model = SparNiche(input_dim=8, latent_dim=4)
        self.assertIsNone(model.adt_encoder)

    def test_double_view_three_stage_training_writes_fused_embedding(self) -> None:
        rng = np.random.default_rng(7)
        adata = anndata.AnnData(rng.normal(size=(12, 8)).astype(np.float32))
        adata.obsm["spatial"] = np.column_stack((np.arange(12), np.arange(12) % 3))
        adata.obsm["adt"] = rng.poisson(5, size=(12, 3)).astype(np.float32)
        config = {
            "data": {"n_neighbors": 2, "preprocessing": {"enabled": False}, "view1": {"source": "X"}},
            "model": {
                "double_view": True,
                "latent_dim": 4,
                "sparniche_view1": {"attention_mode": "spatial_local", "attention_neighbors": 2, "attention_chunk_size": 8},
                "sparniche": {"gan_epochs": 1, "pretrain_epochs": 1, "lr": 0.001,
                              "weight_decay": 0.0, "negative_repeats": 1, "dec_interval": 1,
                              "adt_rec_w": 1.0},
            },
            "training": {"epochs": 1, "checkpoint_every": 1, "seed": 7, "device": "cpu"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = train_adata(adata, config, Path(temp_dir))
            self.assertEqual(artifacts.adata.obsm["sparniche"].shape, (12, 4))
            self.assertEqual(artifacts.adata.uns["sparniche_training"]["view_contract"]["double_view"], True)
            self.assertIn("loss_adt_reconstruction", artifacts.training_result.history[-1])

    def test_skip_visual_artifacts_does_not_duplicate_multimodal_h5ad(self) -> None:
        rng = np.random.default_rng(9)
        adata = anndata.AnnData(rng.normal(size=(12, 8)).astype(np.float32))
        adata.obsm["spatial"] = np.column_stack((np.arange(12), np.arange(12) % 3))
        adata.obsm["adt"] = rng.poisson(5, size=(12, 3)).astype(np.float32)
        config = {
            "data": {"n_neighbors": 2, "preprocessing": {"enabled": False},
                     "view1": {"source": "X"}, "view2": {"key": "auto"}},
            "model": {"double_view": True, "latent_dim": 4,
                      "sparniche_view1": {"attention_mode": "spatial_local", "attention_neighbors": 2},
                      "sparniche": {"gan_epochs": 1, "pretrain_epochs": 1, "lr": 0.001,
                                    "dec_interval": 1}},
            "benchmark": {"write_h5ad": False},
            "training": {"epochs": 1, "seed": 9, "device": "cpu"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            train_adata(adata, config, Path(temp_dir))
            self.assertFalse((Path(temp_dir) / "trained.h5ad").exists())

    def test_auto_view2_falls_back_to_single_rna_for_unpaired_sample(self) -> None:
        rng = np.random.default_rng(8)
        adata = anndata.AnnData(rng.normal(size=(12, 8)).astype(np.float32))
        adata.obsm["spatial"] = np.column_stack((np.arange(12), np.arange(12) % 3))
        config = {
            "data": {"n_neighbors": 2, "preprocessing": {"enabled": False},
                     "view1": {"source": "X"}, "view2": {"source": "obsm", "key": "auto"}},
            "model": {"double_view": True, "latent_dim": 4,
                      "sparniche_view1": {"attention_mode": "spatial_local", "attention_neighbors": 2},
                      "sparniche": {"gan_epochs": 1, "pretrain_epochs": 1, "lr": 0.001,
                                    "dec_interval": 1}},
            "training": {"epochs": 1, "seed": 8, "device": "cpu"},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = train_adata(adata, config, Path(temp_dir))
        self.assertEqual(artifacts.adata.uns["sparniche_training"]["view_contract"]["double_view"], False)


if __name__ == "__main__":
    unittest.main()
