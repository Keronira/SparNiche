import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import anndata
import numpy as np
import pandas as pd

from scripts.tmp_run_layer_recovery_sensitivity_step2 import (
    BASELINE_METHODS,
    STEP2_SAMPLES,
    STEP2_SEEDS,
    STEP2_SET_OVERRIDES,
    build_step2_config,
    load_reclustered_anchor_runs,
    load_reference_runs,
    write_step2_tables,
)


class LayerSensitivityStep2Tests(unittest.TestCase):
    def test_step2_retains_three_candidates_across_three_seeds(self):
        self.assertEqual(
            STEP2_SAMPLES,
            ("151507", "151510", "151669", "151670", "151673", "151676"),
        )
        self.assertEqual(STEP2_SEEDS, (1234, 1235, 1236))
        self.assertEqual(
            STEP2_SET_OVERRIDES,
            {
                "lr_0025": {"model.sparniche.lr": 0.0025},
                "latent_32_norm_on": {
                    "model.latent_dim": 32,
                    "model.local_graph_normalize": True,
                },
                "latent_32_norm_off": {
                    "model.latent_dim": 32,
                    "model.local_graph_normalize": False,
                },
            },
        )

    def test_step2_config_uses_seed_and_k_to_k_plus_two_window(self):
        base = {
            "model": {
                "latent_dim": 64,
                "local_graph_normalize": True,
                "sparniche_view1": {"attention_neighbors": 12},
                "sparniche": {"lr": 0.005},
            },
            "data": {"n_neighbors": 12},
            "training": {"epochs": 550},
        }
        config = build_step2_config(
            base,
            Path("151670.h5ad"),
            set_name="latent_32_norm_off",
            seed=1236,
            device="cpu",
            input_kind="raw_counts",
        )
        self.assertEqual(config["model"]["latent_dim"], 32)
        self.assertFalse(config["model"]["local_graph_normalize"])
        self.assertEqual(config["training"]["seed"], 1236)
        self.assertEqual(config["evaluation"]["sparniche_leiden_seed"], 1236)
        self.assertEqual(config["evaluation"]["leiden_cluster_lower_offset"], 0)
        self.assertEqual(config["evaluation"]["leiden_cluster_upper_offset"], 2)
        self.assertNotIn("leiden_selection_strategy", config["evaluation"])

    def test_reference_loader_reads_all_registered_baselines_but_not_old_anchor_predictions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            methods = ("BANKSY", "CellCharter")
            for name in ("SparNiche_rep1", "BANKSY__rep1", "CellCharter__rep1"):
                results = root / name / "results"
                results.mkdir(parents=True)
                pd.DataFrame(
                    {
                        "spot_id": ["a", "b", "c", "d"],
                        "ground_truth": ["Layer4", "Layer4", "Layer6", "Layer6"],
                        "pred": ["0", "0", "1", "1"],
                    }
                ).to_csv(results / "151507.csv", index=False)

            metrics, layers, failures = load_reference_runs(
                root,
                samples=("151507",),
                reps=(1,),
                baseline_methods=methods,
            )

        self.assertEqual(BASELINE_METHODS[:2], ("BANKSY", "CellCharter"))
        self.assertEqual(set(metrics["model"]), {"BANKSY", "CellCharter"})
        self.assertEqual(set(metrics["model_type"]), {"baseline"})
        self.assertEqual(set(metrics["seed"]), {1234})
        self.assertEqual(set(layers["layer"]), {"Layer4", "Layer6"})
        self.assertTrue(failures.empty)

    def test_anchor_is_reclustered_from_trained_embedding_with_step2_window(self):
        base = {
            "data": {"label_key": "annotation_final", "n_neighbors": 2},
            "model": {
                "variant": "local_graph_normalized",
                "latent_dim": 64,
                "local_graph_normalize": True,
                "sparniche_view1": {"attention_neighbors": 2},
                "sparniche": {"lr": 0.005},
            },
            "training": {"epochs": 1},
            "evaluation": {"n_clusters": 0},
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = (
                root
                / "SparNiche_rep1"
                / "artifacts"
                / "151507--single-local_graph_normalized-seed1234"
            )
            artifact.mkdir(parents=True)
            adata = anndata.AnnData(
                X=np.ones((4, 2)),
                obs=pd.DataFrame(
                    {"annotation_final": ["Layer4", "Layer4", "Layer6", "Layer6"]},
                    index=["a", "b", "c", "d"],
                ),
            )
            adata.obsm["spatial"] = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
            adata.obsm["sparniche"] = np.asarray([[0, 0], [0, 1], [2, 0], [2, 1]])
            adata.write_h5ad(artifact / "trained.h5ad")

            captured = {}

            def fake_predict(embedding, evaluation_adata, config):
                captured["config"] = config
                return np.asarray(["0", "0", "1", "1"])

            with patch(
                "scripts.tmp_run_layer_recovery_sensitivity_step2._predict_clusters",
                side_effect=fake_predict,
            ):
                metrics, layers, failures = load_reclustered_anchor_runs(
                    root,
                    base,
                    samples=("151507",),
                    reps=(1,),
                    device="cpu",
                )

        config = captured["config"]
        self.assertEqual(config["evaluation"]["leiden_cluster_lower_offset"], 0)
        self.assertEqual(config["evaluation"]["leiden_cluster_upper_offset"], 2)
        self.assertNotIn("leiden_selection_strategy", config["evaluation"])
        self.assertEqual(metrics.iloc[0]["source"], "reclustered_anchor_embedding")
        self.assertAlmostEqual(metrics.iloc[0]["ari"], 1.0)
        self.assertEqual(set(layers["layer"]), {"Layer4", "Layer6"})
        self.assertTrue(failures.empty)

    def test_layer46_summary_and_candidate_deltas_are_written(self):
        metrics = pd.DataFrame(
            [
                {
                    "model": "anchor",
                    "model_type": "anchor",
                    "sample": "151507",
                    "seed": 1234,
                    "status": "completed",
                    "ari": 0.50,
                    "nmi": 0.60,
                    "macro_layer_iou": 0.40,
                    "worst_layer_iou": 0.00,
                    "layer_recovery_rate": 0.50,
                },
                {
                    "model": "lr_0025",
                    "model_type": "sparniche_candidate",
                    "sample": "151507",
                    "seed": 1234,
                    "status": "completed",
                    "ari": 0.48,
                    "nmi": 0.62,
                    "macro_layer_iou": 0.55,
                    "worst_layer_iou": 0.20,
                    "layer_recovery_rate": 0.75,
                },
            ]
        )
        layers = pd.DataFrame(
            [
                {"model": "anchor", "model_type": "anchor", "sample": "151507", "seed": 1234, "layer": "Layer4", "iou": 0.0},
                {"model": "anchor", "model_type": "anchor", "sample": "151507", "seed": 1234, "layer": "Layer6", "iou": 0.6},
                {"model": "lr_0025", "model_type": "sparniche_candidate", "sample": "151507", "seed": 1234, "layer": "Layer4", "iou": 0.4},
                {"model": "lr_0025", "model_type": "sparniche_candidate", "sample": "151507", "seed": 1234, "layer": "Layer6", "iou": 0.8},
            ]
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_step2_tables(metrics, layers, pd.DataFrame(), root)
            layer46 = pd.read_csv(root / "layer46_summary.csv")
            deltas = pd.read_csv(root / "paired_deltas_vs_anchor.csv")

        candidate_layer4 = layer46[
            (layer46["model"] == "lr_0025") & (layer46["layer"] == "Layer4")
        ].iloc[0]
        self.assertAlmostEqual(candidate_layer4["mean_iou"], 0.4)
        self.assertAlmostEqual(candidate_layer4["zero_rate"], 0.0)
        self.assertAlmostEqual(candidate_layer4["recovery_rate_iou_ge_0_5"], 0.0)
        ari_delta = deltas[
            (deltas["model"] == "lr_0025") & (deltas["metric"] == "ari")
        ].iloc[0]
        self.assertAlmostEqual(ari_delta["delta_candidate_minus_anchor"], -0.02)


if __name__ == "__main__":
    unittest.main()
