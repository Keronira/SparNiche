import unittest
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from scripts.plot_results import align_result_to_adata, plot_results_directory, result_output_path


class ResultsPlottingTests(unittest.TestCase):
    def test_result_rows_are_aligned_to_h5ad_and_invalid_truth_is_excluded(self):
        adata = anndata.AnnData(np.ones((3, 2), dtype="float32"))
        adata.obs_names = ["spot-b", "spot-a", "spot-c"]
        adata.obs["annotation_final"] = ["B", "A", "unknown"]
        adata.obsm["X_spatial"] = np.asarray([[1, 1], [0, 0], [2, 2]], dtype=float)
        result = pd.DataFrame(
            {
                "spot_id": ["spot-a", "spot-b", "spot-c"],
                "ground_truth": ["A", "B", "unknown"],
                "pred": ["0", "1", "2"],
            }
        )

        aligned = align_result_to_adata(adata, result, ground_truth_key="annotation_final")

        self.assertEqual(aligned.obs_names.tolist(), ["spot-a", "spot-b"])
        self.assertEqual(aligned.obs["plot_ground_truth"].tolist(), ["A", "B"])
        self.assertEqual(aligned.obs["plot_prediction"].tolist(), ["0", "1"])
        self.assertIn("spatial", aligned.obsm)

    def test_result_output_path_defaults_to_sibling_plots_directory(self):
        result_path = Path("/tmp/method/results/sample1.csv")
        self.assertEqual(
            result_output_path(result_path),
            Path("/tmp/method/plots/sample1_combined.png"),
        )

    def test_results_directory_writes_combined_plot_with_configurable_size(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            results_dir = root / "method" / "results"
            data_dir.mkdir()
            results_dir.mkdir(parents=True)
            adata = anndata.AnnData(np.ones((4, 2), dtype="float32"))
            adata.obs_names = ["a", "b", "c", "d"]
            adata.obsm["X_spatial"] = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
            adata.write_h5ad(data_dir / "sample1.h5ad")
            pd.DataFrame(
                {
                    "spot_id": ["a", "b", "c", "d"],
                    "ground_truth": ["A", "A", "B", "B"],
                    "pred": ["0", "0", "1", "1"],
                }
            ).to_csv(results_dir / "sample1.csv", index=False)

            outputs = plot_results_directory(results_dir, data_dir, point_size=0.7)

            self.assertEqual(outputs, [root / "method" / "plots" / "sample1_combined.png"])
            self.assertTrue(outputs[0].is_file())


if __name__ == "__main__":
    unittest.main()
