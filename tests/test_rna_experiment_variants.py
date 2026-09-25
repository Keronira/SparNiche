import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import anndata
import numpy as np
import yaml

import torch

from src.models import SparNicheEncoder
from src.trainer import _sparniche_contrastive_loss
from src.runner import (
    _cached_leiden_labels,
    _leiden_resolution_midpoints,
    _leiden_resolution_probe_points,
    _leiden_search_direction,
    _leiden_directional_bounds,
    _leiden_update_directional_bounds,
    _leiden_labels,
    _select_resolution_candidate,
    _should_interrupt_leiden_search,
)
from scripts.run_experiments import EXPERIMENT_CONFIGS, resolved_config


class RNAExperimentVariantTests(unittest.TestCase):
    def test_h5ad_export_serializes_nested_leiden_candidates(self):
        from src.runner import _write_h5ad_with_serializable_leiden_audit

        adata = anndata.AnnData(X=np.ones((2, 1), dtype=np.float32))
        candidates = [
            {
                "resolution": 0.12,
                "cluster_count": 7,
                "per_layer_iou": {"Layer4": 0.5, "Layer6": 0.25},
                "eligible": True,
                "selected": True,
            },
            {
                "resolution": 0.13,
                "cluster_count": 8,
                "eligible": True,
                "selected": False,
            },
        ]
        adata.uns["sparniche_leiden_search"] = {
            "selected_resolution": 0.12,
            "candidates": candidates,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "trained.h5ad"
            _write_h5ad_with_serializable_leiden_audit(adata, output_path)
            restored = anndata.read_h5ad(output_path)

        stored_audit = restored.uns["sparniche_leiden_search"]
        self.assertNotIn("candidates", stored_audit)
        self.assertEqual(json.loads(stored_audit["candidates_json"]), candidates)
        self.assertEqual(
            adata.uns["sparniche_leiden_search"]["candidates"], candidates
        )

    def test_rna_ablation_is_not_a_public_experiment_set(self):
        self.assertNotIn("rna_ablation", EXPERIMENT_CONFIGS)

    def test_default_model_is_normalized_local_graph(self):
        root = Path(__file__).resolve().parents[1]
        base = yaml.safe_load((root / "configs" / "config.yaml").read_text())

        self.assertEqual(base["model"]["variant"], "local_graph_normalized")
        self.assertEqual(base["model"]["feature_graph_fusion_mode"], "gated")
        self.assertEqual(base["model"]["local_graph_mode"], "normalized")

    def test_attention_mode_cli_override_is_written_to_resolved_config(self):
        base = {"model": {"sparniche_view1": {"attention_mode": "global"}}}
        args = Namespace(
            device="cpu",
            label_key="annotation_final",
            input_kind=None,
            epochs=None,
            experiment_set="single",
            attention_mode="spatial_local",
        )
        resolved = resolved_config(
            base,
            args,
            {"seed": 2023, "overrides": {}},
            Path("sample.h5ad"),
            3,
        )
        self.assertEqual(
            resolved["model"]["sparniche_view1"]["attention_mode"], "spatial_local"
        )

    def test_leiden_max_resolution_cli_override_is_written_to_resolved_config(self):
        base = {"evaluation": {"leiden_max_resolution": 1.99}}
        args = Namespace(
            device="cpu",
            label_key="annotation_final",
            input_kind=None,
            epochs=None,
            experiment_set="single",
            attention_mode=None,
            leiden_min_resolution=6.0,
            leiden_max_resolution=8.0,
            leiden_cluster_lower_offset=0,
            leiden_cluster_upper_offset=2,
        )
        resolved = resolved_config(
            base,
            args,
            {"seed": 2023, "overrides": {}},
            Path("sample.h5ad"),
            81,
        )
        self.assertEqual(resolved["evaluation"]["leiden_min_resolution"], 6.0)
        self.assertEqual(resolved["evaluation"]["leiden_max_resolution"], 8.0)
        self.assertEqual(resolved["evaluation"]["leiden_cluster_lower_offset"], 0)
        self.assertEqual(resolved["evaluation"]["leiden_cluster_upper_offset"], 2)

    def test_gated_fusion_keeps_embedding_contract(self):
        encoder = SparNicheEncoder(200, fusion_mode="gated")
        self.assertEqual(encoder.fusion_mode, "gated")
        self.assertEqual(encoder.latent_dim, 32)

    def test_spatial_plot_default_point_size_is_one(self):
        from src.plotting import DEFAULT_BASELINE_POINT_SIZE

        self.assertEqual(DEFAULT_BASELINE_POINT_SIZE, 1.0)


    def test_contrastive_loss_is_finite_and_disableable(self):
        left = torch.randn(6, 32)
        right = torch.randn(6, 32)
        loss = _sparniche_contrastive_loss(left, right, temperature=0.2)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(_sparniche_contrastive_loss(left, right, temperature=0.2, weight=0.0).item(), 0.0)


    def test_adaptive_graph_gate_is_opt_in(self):
        encoder = SparNicheEncoder(200, adaptive_graph=True)
        self.assertTrue(encoder.adaptive_graph)

    def test_only_normalized_local_graph_experiment_is_public(self):
        from src.config import load_yaml
        from src.manifest import expand_manifest
        root = Path(__file__).resolve().parents[1]
        rows = expand_manifest(load_yaml(root / "configs" / "experiment_local_graph.yaml"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["overrides"]["model.variant"], "local_graph_normalized")

    def test_local_graph_encoder_contains_residual_gcn(self):
        encoder = SparNicheEncoder(200, local_graph_mode="residual")
        self.assertEqual(len(encoder.local_residual_gcns), 1)

    def test_graph_loss_ablation_matrix_has_two_cases_and_three_seeds(self):
        from src.config import load_yaml
        from src.manifest import expand_manifest
        root = Path(__file__).resolve().parents[1]
        rows = expand_manifest(load_yaml(root / "configs" / "experiment_ablation_local_graph.yaml"))
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["case"] for row in rows}, {"anchor", "graph_loss_0"})
        self.assertEqual({row["seed"] for row in rows}, {2024, 2025, 2026})

    def test_leiden_reuses_cached_search_labels(self):
        labels = _cached_leiden_labels({108: ["0", "1", "0"]}, 108)
        self.assertEqual(labels.tolist(), ["0", "1", "0"])
        self.assertIsNone(_cached_leiden_labels({}, 108))

    def test_resolution_grid_excludes_k_minus_one_and_k_plus_three(self):
        records = [
            {
                "resolution": 0.04,
                "cluster_count": 5,
                "labels": list("0123401234"),
            },
            {
                "resolution": 0.16,
                "cluster_count": 8,
                "labels": list("0123456701"),
            },
            {
                "resolution": 0.20,
                "cluster_count": 9,
                "labels": list("aabbccddee"),
            },
        ]
        selected = _select_resolution_candidate(
            records,
            5,
            list("aabbccddee"),
            cluster_lower_offset=0,
            cluster_upper_offset=2,
        )
        self.assertEqual(selected["resolution"], 0.04)
        self.assertFalse(selected["window_fallback"])

    def test_resolution_grid_can_require_at_least_target_clusters(self):
        records = [
            {"resolution": 7.44, "cluster_count": 80, "labels": list("aabb")},
            {"resolution": 7.52, "cluster_count": 82, "labels": list("abcd")},
        ]
        selected = _select_resolution_candidate(
            records,
            81,
            list("aabb"),
            cluster_lower_offset=0,
            cluster_upper_offset=2,
        )
        self.assertEqual(selected["cluster_count"], 82)

    def test_leiden_selection_uses_ari_best_within_window(self):
        truth = np.asarray(list("AAAABBBBCCCC"))
        records = [
            {
                "resolution": 0.2,
                "cluster_count": 3,
                "labels": np.asarray(list("XXZZYYYYYYYY")),
            },
            {
                "resolution": 0.3,
                "cluster_count": 3,
                "labels": np.asarray(list("XXXYYYYZZZZX")),
            },
        ]
        selected = _select_resolution_candidate(
            records,
            3,
            truth,
            cluster_lower_offset=0,
            cluster_upper_offset=2,
        )
        self.assertEqual(selected["resolution"], 0.2)
        self.assertIn("ari", selected)
        self.assertNotIn("layer_recovery_rate", selected)
        self.assertNotIn("worst_layer_iou", selected)

    def test_resolution_grid_falls_back_to_closest_count(self):
        records = [
            {"resolution": 0.4, "cluster_count": 5, "labels": ["0"]},
            {"resolution": 0.8, "cluster_count": 8, "labels": ["0"]},
        ]
        selected = _select_resolution_candidate(records, 7, None)
        self.assertEqual(selected["resolution"], 0.8)

    def test_leiden_search_interrupt_threshold(self):
        self.assertFalse(_should_interrupt_leiden_search(9, 7, 2))
        self.assertTrue(_should_interrupt_leiden_search(10, 7, 2))

    def test_leiden_resolution_grid_uses_configured_upper_bound(self):
        self.assertEqual(_leiden_resolution_midpoints(0.01, 0.03), [1, 2, 3])
        self.assertEqual(_leiden_resolution_midpoints(6.0, 8.0), list(range(600, 801)))

    def test_leiden_probe_order_is_upper_middle_lower(self):
        self.assertEqual(_leiden_resolution_probe_points(1, 9), [9, 5, 1])
        self.assertEqual(_leiden_resolution_probe_points(1, 2), [2, 1])

    def test_leiden_midpoint_direction_uses_cluster_window(self):
        self.assertEqual(_leiden_search_direction(6, 7, 0, 2), "higher")
        self.assertEqual(_leiden_search_direction(10, 7, 0, 2), "lower")
        self.assertEqual(_leiden_search_direction(9, 7, 0, 2), "midpoint")

    def test_leiden_directional_bounds_start_on_target_side(self):
        self.assertEqual(_leiden_directional_bounds(1, 5, 9, "higher"), (6, 9))
        self.assertEqual(_leiden_directional_bounds(1, 5, 9, "lower"), (1, 4))
        self.assertEqual(_leiden_directional_bounds(1, 5, 9, "midpoint"), (5, 5))

    def test_leiden_directional_bounds_update_after_midpoint_probe(self):
        self.assertEqual(
            _leiden_update_directional_bounds(6, 9, 7, 5, 7, "higher", 0, 2),
            (8, 9),
        )
        self.assertEqual(
            _leiden_update_directional_bounds(1, 4, 3, 11, 7, "lower", 0, 2),
            (1, 2),
        )

    def test_leiden_search_traverses_entire_valid_resolution_window(self):
        adata = anndata.AnnData(X=np.zeros((80, 2), dtype=np.float32))
        resolutions = []

        def fake_leiden(adata, resolution, key_added, **kwargs):
            resolutions.append(round(float(resolution), 2))
            cluster_count = max(1, min(79, int(round(float(resolution) * 20))))
            adata.obs[key_added] = np.asarray(
                [str(index % cluster_count) for index in range(adata.n_obs)],
                dtype=str,
            )

        with patch("src.runner.sc.pp.neighbors"), patch(
            "src.runner.sc.tl.leiden", side_effect=fake_leiden
        ):
            labels = _leiden_labels(
                np.zeros((80, 2), dtype=np.float32),
                adata,
                target_clusters=12,
                seed=1234,
                min_resolution=0.01,
                max_resolution=0.99,
            )

        self.assertEqual(len(np.unique(labels)), 12)
        valid_window = {round(value / 100, 2) for value in range(58, 73)}
        self.assertTrue(valid_window.issubset(set(resolutions)))
        self.assertIn(0.75, resolutions)

    def test_leiden_traversal_crosses_nonmonotonic_high_side_misses(self):
        adata = anndata.AnnData(X=np.zeros((80, 2), dtype=np.float32))
        resolutions = []

        def fake_leiden(adata, resolution, key_added, **kwargs):
            value = round(float(resolution), 2)
            resolutions.append(value)
            if value >= 0.5:
                cluster_count = 10
            elif value >= 0.2:
                cluster_count = 11
            elif value >= 0.1:
                cluster_count = 8
            else:
                cluster_count = 4
            adata.obs[key_added] = np.asarray(
                [str(index % cluster_count) for index in range(adata.n_obs)],
                dtype=str,
            )

        with patch("src.runner.sc.pp.neighbors"), patch(
            "src.runner.sc.tl.leiden", side_effect=fake_leiden
        ):
            labels = _leiden_labels(
                np.zeros((80, 2), dtype=np.float32),
                adata,
                target_clusters=7,
                seed=1234,
                min_resolution=0.01,
                max_resolution=0.99,
            )

        self.assertIn(0.2, resolutions)
        self.assertEqual(len(np.unique(labels)), 8)

    def test_leiden_search_falls_back_when_no_resolution_hits_window(self):
        adata = anndata.AnnData(X=np.zeros((80, 2), dtype=np.float32))

        def fake_leiden(adata, resolution, key_added, **kwargs):
            cluster_count = 1 if float(resolution) < 0.5 else 20
            adata.obs[key_added] = np.asarray(
                [str(index % cluster_count) for index in range(adata.n_obs)],
                dtype=str,
            )

        with patch("src.runner.sc.pp.neighbors"), patch(
            "src.runner.sc.tl.leiden", side_effect=fake_leiden
        ):
            labels = _leiden_labels(
                np.zeros((80, 2), dtype=np.float32),
                adata,
                target_clusters=12,
                seed=1234,
                min_resolution=0.01,
                max_resolution=0.99,
            )

        self.assertEqual(len(np.unique(labels)), 20)
