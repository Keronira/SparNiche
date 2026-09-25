from __future__ import annotations

import torch
import unittest

from src.models import SparNicheEncoder
from src.trainer import _resolve_sparniche_attention_config


def _encoder(**kwargs) -> SparNicheEncoder:
    return SparNicheEncoder(
        input_dim=8,
        latent_dim=32,
        hidden_dims=(64, 16),
        num_heads=1,
        dropout=0.2,
        dec_cluster_n=10,
        **kwargs,
    )


class SparNicheAttentionTest(unittest.TestCase):
    def test_dec_cluster_count_accepts_positive_configured_values(self) -> None:
        for cluster_count in (10, 12, 14):
            encoder = SparNicheEncoder(
                input_dim=8,
                latent_dim=32,
                hidden_dims=(64, 16),
                num_heads=1,
                dropout=0.2,
                dec_cluster_n=cluster_count,
            )
            self.assertEqual(encoder.dec_cluster_n, cluster_count)
            self.assertEqual(encoder.cluster_layer.shape, (cluster_count, 32))

    def test_dec_cluster_count_rejects_non_positive_values(self) -> None:
        for cluster_count in (0, -1):
            with self.assertRaisesRegex(ValueError, "dec_cluster_n"):
                SparNicheEncoder(
                    input_dim=8,
                    latent_dim=32,
                    hidden_dims=(64, 16),
                    num_heads=1,
                    dropout=0.2,
                    dec_cluster_n=cluster_count,
                )

    def test_spatial_local_replaces_null_defaults_from_base_config(self) -> None:
        resolved = _resolve_sparniche_attention_config(
            {"attention_mode": "spatial_local", "attention_neighbors": None, "attention_chunk_size": None},
            data_neighbors=12,
            available_neighbors=12,
        )
        self.assertEqual(resolved["attention_neighbors"], 12)
        self.assertEqual(resolved["attention_chunk_size"], 4096)

    def test_global_allows_null_local_attention_settings(self) -> None:
        resolved = _resolve_sparniche_attention_config(
            {"attention_mode": "global", "attention_neighbors": None, "attention_chunk_size": None},
            data_neighbors=12,
            available_neighbors=12,
        )
        self.assertIsNone(resolved["attention_neighbors"])
        self.assertIsNone(resolved["attention_chunk_size"])

    def test_global_attention_matches_legacy_path(self) -> None:
        torch.manual_seed(7)
        encoder = _encoder(attention_mode="global").eval()
        features = torch.randn(6, 8)
        with torch.no_grad():
            expected, _ = encoder.layer[0](
                features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1)
            )
            expected = expected.squeeze(1) + features
            for layer in encoder.layers:
                expected = layer(expected)
            actual = encoder._complex_encode(features)
        torch.testing.assert_close(actual, expected)

    def test_spatial_local_returns_one_embedding_per_spot(self) -> None:
        encoder = _encoder(
            attention_mode="spatial_local",
            attention_neighbors=2,
            attention_chunk_size=2,
        ).eval()
        features = torch.randn(5, 8)
        neighbors = torch.tensor([[1, 2], [0, 2], [1, 3], [2, 4], [2, 3]])
        self.assertEqual(encoder._complex_encode(features, neighbors).shape, (5, 16))

    def test_spatial_local_chunk_size_preserves_outputs_and_gradients(self) -> None:
        torch.manual_seed(11)
        one_chunk = _encoder(attention_mode="spatial_local", attention_neighbors=2, attention_chunk_size=8).eval()
        many_chunks = _encoder(attention_mode="spatial_local", attention_neighbors=2, attention_chunk_size=2).eval()
        many_chunks.load_state_dict(one_chunk.state_dict())
        features_a = torch.randn(6, 8, requires_grad=True)
        features_b = features_a.detach().clone().requires_grad_(True)
        neighbors = torch.tensor([[1, 2], [0, 2], [1, 3], [2, 4], [3, 5], [3, 4]])
        loss_a = one_chunk._complex_encode(features_a, neighbors).sum()
        loss_b = many_chunks._complex_encode(features_b, neighbors).sum()
        loss_a.backward()
        loss_b.backward()
        torch.testing.assert_close(loss_a, loss_b)
        torch.testing.assert_close(features_a.grad, features_b.grad)

    def test_invalid_attention_configuration_is_rejected(self) -> None:
        for kwargs, message in [
            ({"attention_mode": "invalid"}, "attention_mode"),
            ({"attention_mode": "spatial_local", "attention_neighbors": 0}, "attention_neighbors"),
            ({"attention_mode": "spatial_local", "attention_neighbors": 2, "attention_chunk_size": 0}, "attention_chunk_size"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                _encoder(**kwargs)

    def test_spatial_local_rejects_invalid_neighbor_indices(self) -> None:
        encoder = _encoder(attention_mode="spatial_local", attention_neighbors=2, attention_chunk_size=2)
        for neighbors, message in [
            (torch.zeros(4, 2, dtype=torch.float32), "integer"),
            (torch.zeros(3, 2, dtype=torch.long), "shape"),
            (torch.tensor([[0, 1], [0, 1], [0, 9], [0, 1]]), "outside"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                encoder._complex_encode(torch.randn(4, 8), neighbors)
