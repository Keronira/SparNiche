import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import anndata
import numpy as np
import pandas as pd

from scripts.run_final_embedding_experiment import (
    FINAL_SET_NAMES,
    SCREENING_SET_OVERRIDES,
    _full_config,
    load_anchor_rep_metrics,
    load_existing_metrics,
    select_samples,
    select_tail_samples,
    set_output_dir,
    summarize_sets,
)
from src.final_metrics import compute_final_metrics


class FinalEmbeddingExperimentTests(unittest.TestCase):
    def test_tail_samples(self):
        paths = [Path(f"{name}.h5ad") for name in ("151676", "151671", "151674", "151673", "151675", "151672", "151670")]
        self.assertEqual([p.stem for p in select_tail_samples(paths, 6)], ["151671", "151672", "151673", "151674", "151675", "151676"])

    def test_metrics_perfect_embedding(self):
        labels = np.asarray(["Layer1", "Layer1", "Layer2", "Layer2", "WM", "WM"])
        embedding = np.asarray([[0, 0], [0.1, 0], [5, 0], [5.1, 0], [10, 0], [10.1, 0]], dtype=np.float32)
        predicted = np.asarray([0, 0, 1, 1, 2, 2])
        metrics = compute_final_metrics(
            embedding,
            labels,
            embedding,
            predicted=predicted,
            seed=1,
            n_neighbors=1,
        )
        self.assertAlmostEqual(metrics["ari"], 1.0)
        self.assertAlmostEqual(metrics["nmi"], 1.0)
        self.assertAlmostEqual(metrics["fmi"], 1.0)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["layer_order_score"], 1.0)
        self.assertGreater(metrics["fidelity"], 0.99)

    def test_full_config_keeps_leiden_k_minus_1_to_k_plus_3(self):
        config = _full_config({}, Path("sample.h5ad"), 1234, "cpu")
        self.assertFalse(config["benchmark"]["external_only"])
        self.assertEqual(config["evaluation"]["leiden_cluster_lower_offset"], -1)
        self.assertEqual(config["evaluation"]["leiden_cluster_upper_offset"], 3)

    def test_screening_registry_has_fifteen_deduplicated_sets(self):
        expected = {
            "anchor",
            "neighbors_08",
            "neighbors_16",
            "neighbors_24",
            "latent_16",
            "latent_64",
            "norm_off",
            "dropout_00",
            "dropout_40",
            "lr_low",
            "lr_high",
            "wd_none",
            "wd_strong",
            "epochs_300",
            "epochs_800",
        }
        self.assertEqual(set(SCREENING_SET_OVERRIDES), expected)
        self.assertEqual(len(SCREENING_SET_OVERRIDES), 15)
        self.assertTrue(all(isinstance(value, dict) for value in SCREENING_SET_OVERRIDES.values()))

    def test_screening_config_changes_only_declared_fields(self):
        base = {
            "model": {"latent_dim": 32, "sparniche_view1": {"dropout": 0.2}},
            "data": {"n_neighbors": 12},
            "training": {"epochs": 550, "lr": 0.001, "weight_decay": 0.0},
        }
        config = _full_config(base, Path("sample.h5ad"), 1234, "cpu", set_name="dropout_40")
        self.assertEqual(config["model"]["sparniche_view1"]["dropout"], 0.4)
        self.assertEqual(config["model"]["latent_dim"], 32)
        self.assertEqual(config["data"]["n_neighbors"], 12)
        self.assertEqual(config["training"]["epochs"], 550)

    def test_explicit_sample_selection(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            paths = [tmp_path / "151671.h5ad", tmp_path / "151672.h5ad", tmp_path / "151673.h5ad"]
            selected = select_samples(paths, ["151672", "151671"])
            self.assertEqual([path.stem for path in selected], ["151672", "151671"])

    def test_final_sets_include_selected_configs_and_interaction(self):
        self.assertEqual(
            FINAL_SET_NAMES,
            (
                "neighbors_16",
                "neighbors_08",
                "lr_low",
                "latent_64",
                "lr_low_latent_64",
            ),
        )

    def test_lr_low_latent_64_combines_both_overrides(self):
        config = _full_config(
            {},
            Path("sample.h5ad"),
            1234,
            "cpu",
            set_name="lr_low_latent_64",
        )
        self.assertEqual(config["model"]["latent_dim"], 64)
        self.assertEqual(config["model"]["sparniche"]["lr"], 0.005)

    def test_final_mode_uses_one_output_directory_per_set(self):
        root = Path("results")
        self.assertEqual(
            set_output_dir(root, "final", "neighbors_16"),
            root / "neighbors_16",
        )

    def test_load_existing_metrics_adds_set_name(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.csv").write_text(
                "sample,seed,status,ari,nmi,fmi,accuracy,macro_f1,layer_order_score,fidelity\n"
                "151671,1234,completed,0.5,0.6,0.7,0.8,0.4,0.9,0.3\n",
                encoding="utf-8",
            )
            rows = load_existing_metrics(root, set_name="anchor")
            self.assertEqual(rows[0]["set_name"], "anchor")
            self.assertEqual(rows[0]["sample"], "151671")

    def test_summarize_sets_writes_one_row_per_set_and_metric(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "summary.csv"
            rows = [
                {
                    "set_name": "anchor",
                    "sample": "151671",
                    "seed": "1234",
                    "status": "completed",
                    "ari": "0.5",
                    "nmi": "0.6",
                    "fmi": "0.7",
                    "accuracy": "0.8",
                    "macro_f1": "0.4",
                    "layer_order_score": "0.9",
                    "fidelity": "0.3",
                }
            ]
            summarize_sets(rows, output)
            text = output.read_text(encoding="utf-8")
            self.assertIn("set_name,metric,mean,sd,n", text)
            self.assertIn("anchor,ari,0.5", text)

    def test_load_anchor_rep_metrics_recomputes_metrics_from_rep_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source24"
            rep = root / "SparNiche_rep1" / "results"
            source.mkdir(parents=True)
            rep.mkdir(parents=True)
            obs_names = [f"spot{i}" for i in range(6)]
            adata = anndata.AnnData(
                X=np.ones((6, 2), dtype=np.float32),
                obs=pd.DataFrame(index=obs_names),
            )
            adata.obsm["spatial"] = np.asarray(
                [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=np.float32
            )
            adata.write_h5ad(source / "151671.h5ad")
            labels = ["Layer1", "Layer1", "Layer2", "Layer2", "WM", "WM"]
            pd.DataFrame(
                {"spot_id": obs_names, "ground_truth": labels, "pred": [0, 0, 1, 1, 2, 2]}
            ).to_csv(rep / "151671.csv", index=False)
            pd.DataFrame(
                {"spot_id": obs_names, "0": [0, 0, 1, 1, 2, 2], "1": [0, 0, 0, 0, 0, 0]}
            ).to_csv(rep / "151671_embedding.csv", index=False)
            (rep / "151671_profile.json").write_text(
                json.dumps({"runtime_seconds": 1.5}), encoding="utf-8"
            )
            rows = load_anchor_rep_metrics(
                str(root / "SparNiche_rep*"),
                source,
                [source / "151671.h5ad"],
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["set_name"], "anchor")
            self.assertEqual(rows[0]["seed"], 1234)
            self.assertAlmostEqual(rows[0]["ari"], 1.0)


if __name__ == "__main__":
    unittest.main()
