import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import pandas as pd

from scripts.run_experiments import (
    aggregate_summary_records,
    input_paths_from_source,
    _method_name_for_row,
    summarize_prediction_csv,
    write_experiment_summary,
    resolve_source_arguments,
    resolve_run_seeds,
    resolve_repeat_output_root,
    load_run_config,
    selected_rows,
)


class ExperimentSummaryTests(unittest.TestCase):
    def test_default_runs_use_the_three_baseline_repeat_seeds(self):
        args = Namespace(seeds=None, repeats=3, no_repeats=False)
        self.assertEqual(resolve_run_seeds(args), [1234, 1235, 1236])

    def test_default_repeat_output_roots_follow_seed_order(self):
        base = Path("/tmp/bench_results")
        seeds = [1234, 1235, 1236]
        self.assertEqual(
            resolve_repeat_output_root(base, 1234, seeds),
            base / "SparNiche_rep1",
        )
        self.assertEqual(
            resolve_repeat_output_root(base, 1236, seeds),
            base / "SparNiche_rep3",
        )

    def test_custom_seed_order_still_maps_to_repeat_numbers(self):
        base = Path("/tmp/bench_results")
        self.assertEqual(
            resolve_repeat_output_root(
                base, 2025, [2024, 2025, 2026]
            ),
            base / "SparNiche_rep2",
        )

    def test_no_repeats_uses_only_the_first_baseline_seed(self):
        args = Namespace(seeds=None, repeats=3, no_repeats=True)
        self.assertEqual(resolve_run_seeds(args), [1234])

    def test_explicit_seeds_override_automatic_repeats(self):
        args = Namespace(seeds=[2024, 2025], repeats=3, no_repeats=False)
        self.assertEqual(resolve_run_seeds(args), [2024, 2025])

    def test_default_single_experiment_creates_three_seeded_runs(self):
        args = Namespace(
            experiment_set="single",
            variants=None,
            seeds=None,
            repeats=3,
            no_repeats=False,
        )
        rows = selected_rows(args)
        self.assertEqual([row["seed"] for row in rows], [1234, 1235, 1236])

    def test_ablation_cases_use_distinct_method_names(self):
        first = _method_name_for_row("method", {"case": "anchor", "seed": 2023}, "ablation_local_graph", [2023])
        second = _method_name_for_row("method", {"case": "no_dec_kl", "seed": 2023}, "ablation_local_graph", [2023])
        self.assertNotEqual(first, second)
    def test_summarize_prediction_csv_reports_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "dataset" / "method"
            results = root / "results"
            results.mkdir(parents=True)
            pd.DataFrame(
                {"spot_id": ["a", "b", "c", "d"], "ground_truth": ["A", "A", "B", "B"], "pred": ["0", "0", "1", "1"]}
            ).to_csv(results / "sample.csv", index=False)
            row = {"run_id": "run", "dataset": "dataset", "variant": "view1", "case": "single", "seed": 1, "sample": "sample", "method_root": str(root)}
            summary = summarize_prediction_csv(row)
            self.assertEqual(summary["n"], 4)
            self.assertAlmostEqual(summary["ari"], 1.0)
            self.assertAlmostEqual(summary["nmi"], 1.0)
            self.assertAlmostEqual(summary["ami"], 1.0)

    def test_write_summary_creates_csv_and_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            write_experiment_summary(output, "seq", [{"run_id": "r", "dataset": "d", "variant": "v", "case": "single", "seed": 1, "sample": "s", "n": 2, "ari": 1.0, "nmi": 1.0, "ami": 1.0, "runtime_seconds": 2.0}])
            self.assertTrue((output / "experiment_summary.csv").is_file())
            self.assertTrue((output / "experiment_summary.json").is_file())

    def test_aggregate_summary_records_reports_mean_and_std(self):
        records = [
            {"dataset": "d", "variant": "v", "case": "anchor", "ari": 0.2, "nmi": 0.4, "ami": 0.3, "runtime_seconds": 1.0, "n": 2},
            {"dataset": "other", "variant": "v", "case": "anchor", "ari": 0.4, "nmi": 0.6, "ami": 0.5, "runtime_seconds": 3.0, "n": 2},
        ]
        summary = aggregate_summary_records(records)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["ari_mean"], 0.3)
        self.assertAlmostEqual(summary[0]["ari_std"], 0.1414213562)
        self.assertNotIn("dataset", summary[0])
        self.assertEqual(summary[0]["sample_count"], 2)

    def test_input_paths_can_limit_to_first_five(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(7):
                (root / f"sample{index}.h5ad").touch()
            self.assertEqual(len(input_paths_from_source(root, max_samples=5)), 5)

    def test_multiple_sources_are_flattened_in_path_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "source1"
            second = root / "source2"
            first.mkdir()
            second.mkdir()
            (first / "a.h5ad").touch()
            (second / "b.h5ad").touch()
            paths = input_paths_from_source([first, second])
            self.assertEqual([path.name for path in paths], ["a.h5ad", "b.h5ad"])

    def test_source_names_resolve_under_data_root(self):
        resolved = resolve_source_arguments(["source1", "source2"], data_root=Path("/root/autodl-fs/data"))
        self.assertEqual(resolved, [Path("/root/autodl-fs/data/source1"), Path("/root/autodl-fs/data/source2")])

    def test_multiple_source_directories_include_all_h5ad_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for source_name, filenames in {"source1": ["a.h5ad", "b.h5ad"], "source2": ["c.h5ad", "d.h5ad"]}.items():
                directory = root / source_name
                directory.mkdir()
                for filename in filenames:
                    (directory / filename).touch()
            sources = resolve_source_arguments(["source1", "source2"], data_root=root)
            paths = input_paths_from_source(sources)
            self.assertEqual([path.name for path in paths], ["a.h5ad", "b.h5ad", "c.h5ad", "d.h5ad"])

    def test_source24_uses_its_default_overlay(self):
        config = load_run_config(
            None,
            Path("/root/autodl-fs/data/source24/151671.h5ad"),
        )
        self.assertEqual(config["model"]["latent_dim"], 64)
        self.assertEqual(config["model"]["sparniche"]["lr"], 0.005)
        self.assertEqual(config["data"]["n_neighbors"], 12)

    def test_other_sources_keep_the_shared_default(self):
        config = load_run_config(
            None,
            Path("/root/autodl-fs/data/source1/sample.h5ad"),
        )
        self.assertEqual(config["model"]["latent_dim"], 32)
        self.assertEqual(config["model"]["sparniche"]["lr"], 0.01)

    def test_explicit_config_disables_source24_overlay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "custom.yaml"
            config_path.write_text(
                "model:\n  latent_dim: 16\n  sparniche:\n    lr: 0.02\n",
                encoding="utf-8",
            )
            config = load_run_config(
                config_path,
                Path("/root/autodl-fs/data/source24/151671.h5ad"),
            )
        self.assertEqual(config["model"]["latent_dim"], 16)
        self.assertEqual(config["model"]["sparniche"]["lr"], 0.02)


if __name__ == "__main__":
    unittest.main()
