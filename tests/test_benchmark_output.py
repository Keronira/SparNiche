import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import anndata

from src.benchmark import benchmark_output_paths, export_benchmark_result
from src.data import resolve_spatial_coordinates
from src.data import prepare_sparniche_rna_features
from src.plotting import _metric_subtitle, save_benchmark_combined_plot


class BenchmarkOutputTests(unittest.TestCase):
    def test_normalized_input_preprocessing_builds_pca_features(self):
        adata = anndata.AnnData(np.full((6, 5), 0.5, dtype="float32"))
        config = {
            "enabled": True, "reuse_existing": False, "feature_key": "feat",
            "input_kind": "normalized", "hvg_enabled": False,
            "min_cells": 1, "min_counts": 1, "n_top_genes": 5,
            "hvg_flavor": "seurat_v3", "normalize_total": False,
            "target_sum": 1e6, "scale": False, "pca_enabled": True,
            "pca_n_components": 2, "pca_random_state": 42,
        }
        result = prepare_sparniche_rna_features(adata, config)
        self.assertEqual(result.obsm["feat"].shape, (6, 2))
    def test_spatial_coordinate_key_falls_back_to_x_spatial(self):
        class Adata:
            obsm = {"X_spatial": np.asarray([[1.0, 2.0]])}
        np.testing.assert_array_equal(resolve_spatial_coordinates(Adata()), [[1.0, 2.0]])
    def test_metric_subtitle_contains_ari_and_nmi(self):
        frame = pd.DataFrame({"ground_truth": ["A", "A", "B", "B"], "pred": ["0", "0", "1", "1"]})
        subtitle = _metric_subtitle(frame)
        self.assertIn("ARI=1.0000", subtitle)
        self.assertIn("NMI=1.0000", subtitle)

    def test_external_benchmark_writes_combined_plot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adata = anndata.AnnData(np.ones((2, 2), dtype="float32"))
            adata.obs["truth"] = ["A", "B"]
            adata.obs["pred"] = ["B", "A"]
            adata.obsm["spatial"] = np.asarray([[0.0, 0.0], [1.0, 1.0]])
            output = Path(temp_dir) / "plots" / "sample_combined.png"
            save_benchmark_combined_plot(
                adata, output, label_key="truth", cluster_key="pred"
            )
            self.assertTrue(output.is_file())

    def test_export_accepts_partial_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            method_root = Path(temp_dir) / "dataset" / "method"
            export_benchmark_result(
                method_root,
                "sample",
                ["s1", "s2"],
                ["A", None],
                ["B", "A"],
                np.ones((2, 2)),
                runtime_seconds=1.0,
            )
            frame = pd.read_csv(method_root / "results" / "sample.csv")
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["spot_id"], "s1")

    def test_benchmark_paths_use_shared_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = benchmark_output_paths(
                Path(temp_dir), "dlpfc", "SparNiche", "sample1"
            )
            self.assertEqual(
                paths.results_dir,
                Path(temp_dir) / "dlpfc" / "SparNiche" / "results",
            )
            self.assertEqual(paths.prediction_csv.name, "sample1.csv")

    def test_export_has_calculate_bench_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            method_root = Path(temp_dir) / "dlpfc" / "SparNiche"
            export_benchmark_result(
                method_root,
                "sample1",
                ["s1", "s2"],
                ["A", "B"],
                ["B", "A"],
                np.asarray([[0.1, 0.2], [0.3, 0.4]]),
                runtime_seconds=1.5,
                cpu_rss_peak_mb=12.0,
            )
            frame = pd.read_csv(method_root / "results" / "sample1.csv")
            self.assertEqual(list(frame.columns), ["spot_id", "ground_truth", "pred"])
            self.assertTrue((method_root / "results" / "sample1_embedding.csv").is_file())
            profile = json.loads((method_root / "results" / "sample1_profile.json").read_text())
            self.assertEqual(profile["runtime_seconds"], 1.5)
            self.assertEqual(profile["cpu_rss_peak_mb"], 12.0)
