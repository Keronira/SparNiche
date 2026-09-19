from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from scripts.run_sparse_scaling_ablation import (
    build_dense_sparniche_graph,
    build_dense_sparniche_negative_mask,
    CONDITIONS,
    build_sparse_sparniche_graph,
    build_sparse_sparniche_negative_mask,
    select_input_samples,
)
from src.data import build_sparniche_graph
from src.data import build_sparniche_negative_mask


class SparseScalingAblationTests(unittest.TestCase):
    def test_ablation_anchor_keeps_dense_reference_separate_from_sparse_default(self) -> None:
        self.assertIsNot(build_dense_sparniche_graph, build_sparse_sparniche_graph)
        self.assertIsNot(build_dense_sparniche_negative_mask, build_sparse_sparniche_negative_mask)

    def test_formal_graph_default_does_not_materialize_pairwise_distances(self) -> None:
        coordinates = np.asarray(
            [[0.0, 0.0], [1.0, 0.2], [2.5, 0.1], [0.3, 2.0]],
            dtype=np.float32,
        )
        with patch("src.data.NearestNeighbors", wraps=NearestNeighbors) as knn:
            graph = build_sparniche_graph(coordinates, n_neighbors=2)
        self.assertTrue(knn.called)
        self.assertEqual(tuple(graph["adj_label"].shape), (4, 4))

    def test_formal_negative_default_does_not_enumerate_dense_permutations(self) -> None:
        coordinates = np.asarray(
            [[0.0, 0.0], [1.0, 0.2], [2.5, 0.1], [0.3, 2.0]],
            dtype=np.float32,
        )
        positive = build_sparniche_graph(coordinates, n_neighbors=1)["adj_label"]
        with patch(
            "src.data.torch.randperm",
            side_effect=AssertionError("dense negative-sampling path was used"),
        ):
            mask = build_sparniche_negative_mask(positive, repeats=2, seed=17)
        self.assertEqual(tuple(mask.shape), (4, 4))

    def test_four_conditions_change_only_graph_and_negative_sampler(self) -> None:
        self.assertEqual(
            CONDITIONS,
            {
                "anchor": (False, False),
                "sparse_graph": (True, False),
                "sparse_negative": (False, True),
                "sparse_both": (True, True),
            },
        )

    def test_select_input_samples_uses_first_three_sorted_h5ad_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            source1 = data_root / "source1"
            source1.mkdir()
            for name in ("c.h5ad", "a.h5ad", "ignore.txt", "b.h5ad", "d.h5ad"):
                (source1 / name).touch()

            selected = select_input_samples(data_root, ["source1"], 3)

        self.assertEqual(
            [path.name for _, path in selected],
            ["a.h5ad", "b.h5ad", "c.h5ad"],
        )

    def test_sparse_graph_matches_anchor_on_unique_distances(self) -> None:
        coordinates = np.asarray(
            [[0.0, 0.0], [1.0, 0.2], [2.5, 0.1], [0.3, 2.0], [3.2, 2.7]],
            dtype=np.float32,
        )
        anchor = build_sparniche_graph(coordinates, n_neighbors=2)
        sparse_graph = build_sparse_sparniche_graph(coordinates, n_neighbors=2)

        torch.testing.assert_close(
            sparse_graph["adj_norm"].to_dense(), anchor["adj_norm"].to_dense()
        )
        torch.testing.assert_close(
            sparse_graph["adj_label"].to_dense(), anchor["adj_label"].to_dense()
        )
        self.assertAlmostEqual(sparse_graph["norm_value"], anchor["norm_value"])

    def test_sparse_negative_sampler_is_valid_counted_and_deterministic(self) -> None:
        coordinates = np.asarray(
            [[0.0, 0.0], [1.0, 0.2], [2.5, 0.1], [0.3, 2.0], [3.2, 2.7]],
            dtype=np.float32,
        )
        positive = build_sparse_sparniche_graph(coordinates, n_neighbors=1)[
            "adj_label"
        ].coalesce()
        first = build_sparse_sparniche_negative_mask(positive, repeats=2, seed=17).coalesce()
        second = build_sparse_sparniche_negative_mask(positive, repeats=2, seed=17).coalesce()
        torch.testing.assert_close(first.indices(), second.indices())
        torch.testing.assert_close(first.values(), second.values())

        positive_pairs = {
            tuple(pair) for pair in positive.indices().t().tolist()
        }
        sampled_pairs = first.indices().t().tolist()
        sampled_values = first.values().tolist()
        negative_pairs = {
            tuple(pair)
            for pair, value in zip(sampled_pairs, sampled_values)
            if value == 0.0
        }
        self.assertTrue(positive_pairs.isdisjoint(negative_pairs))
        self.assertEqual(len(negative_pairs), len(sampled_pairs) - len(positive_pairs))

        expected_negative_count = 0
        n_spots = positive.shape[0]
        positive_by_source = [set() for _ in range(n_spots)]
        for source, target in positive_pairs:
            positive_by_source[source].add(target)
        for neighbors in positive_by_source:
            expected_negative_count += min(
                n_spots - len(neighbors), len(neighbors) * 2
            )
        self.assertEqual(len(negative_pairs), expected_negative_count)

    def test_sparse_negative_sampler_rejects_invalid_repeats(self) -> None:
        positive = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]]),
            torch.ones(2),
            (2, 2),
        )
        with self.assertRaisesRegex(ValueError, "repeats"):
            build_sparse_sparniche_negative_mask(positive, repeats=-1)


if __name__ == "__main__":
    unittest.main()
