import unittest
from pathlib import Path

import numpy as np

from scripts.run_embedding_neighbor_sweep import SWEEP_SPECS, select_tail_samples
from src.embedding_metrics import compute_embedding_metrics


class NeighborSweepTests(unittest.TestCase):
    def test_full_is_the_reference_condition(self):
        self.assertEqual(SWEEP_SPECS["full"], {"n_neighbors": 12, "attention_neighbors": 12})

    def test_one_factor_at_a_time_grid(self):
        self.assertEqual(
            SWEEP_SPECS,
            {
                "full": {"n_neighbors": 12, "attention_neighbors": 12},
                "graph_6": {"n_neighbors": 6, "attention_neighbors": 6},
                "graph_24": {"n_neighbors": 24, "attention_neighbors": 12},
                "attention_6": {"n_neighbors": 12, "attention_neighbors": 6},
                "attention_24": {"n_neighbors": 24, "attention_neighbors": 24},
            },
        )

    def test_tail_selection(self):
        paths = [Path(f"{name}.h5ad") for name in ("151676", "151671", "151674", "151673", "151675", "151672", "151670")]
        self.assertEqual([path.stem for path in select_tail_samples(paths, 6)], ["151671", "151672", "151673", "151674", "151675", "151676"])

    def test_macro_f1_is_reported(self):
        embedding = np.asarray([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]], dtype=np.float32)
        labels = np.asarray(["a", "a", "b", "b"])
        spatial = embedding.copy()
        metrics = compute_embedding_metrics(embedding, labels, spatial, seed=1, n_neighbors=1)
        self.assertIn("linear_probe_macro_f1", metrics)


if __name__ == "__main__":
    unittest.main()
