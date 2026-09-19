#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import apply_overrides, deep_merge, load_yaml, save_yaml  # noqa: E402
from src.manifest import expand_manifest  # noqa: E402
from src.evaluation import clustering_metrics  # noqa: E402
from src.benchmark import filter_ambiguous_ground_truth  # noqa: E402


EXPERIMENT_CONFIGS = {
    "local_graph": PROJECT_ROOT / "configs" / "experiment_local_graph.yaml",
    "ablation_local_graph": PROJECT_ROOT / "configs" / "experiment_ablation_local_graph.yaml",
}
SINGLE_VARIANTS = ("local_graph_normalized",)
DEFAULT_LABEL_KEY = "annotation_final"
DEFAULT_EPOCHS = 550
DEFAULT_SEED = 1234
DEFAULT_REPEAT_COUNT = 3
DEFAULT_DATA_ROOT = Path("/root/autodl-fs/data")
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
SOURCE_CONFIGS = {
    "source24": PROJECT_ROOT / "configs" / "source24.yaml",
}


def detect_input_kind(adata) -> str:
    """Classify an AnnData expression matrix without densifying sparse input."""
    matrix = adata.X
    if sparse.issparse(matrix):
        values = matrix.data
    elif isinstance(matrix, np.ndarray):
        values = matrix
    elif hasattr(matrix, "shape"):
        # AnnData backed datasets expose row slicing rather than a SciPy
        # sparse object. A bounded sample avoids loading multi-GB matrices.
        sample = matrix[: min(int(matrix.shape[0]), 256), :]
        values = sample.data if sparse.issparse(sample) else np.asarray(sample)
    else:
        values = np.asarray(matrix)
    values = np.asarray(values)
    if values.size == 0:
        return "raw_counts"
    is_counts = (
        np.isfinite(values).all()
        and np.all(values >= 0.0)
        and np.allclose(values, np.rint(values), rtol=0.0, atol=1e-6)
    )
    return "raw_counts" if is_counts else "normalized"


def infer_input_kind(input_path: Path) -> str:
    adata = anndata.read_h5ad(input_path, backed="r")
    try:
        return detect_input_kind(adata)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def resolve_source_arguments(
    sources: list[Path | str], data_root: Path = DEFAULT_DATA_ROOT
) -> list[Path]:
    """Resolve shorthand dataset names against the shared data directory."""
    resolved: list[Path] = []
    for value in sources:
        path = Path(value).expanduser()
        if not path.is_absolute() and len(path.parts) == 1 and path.suffix == "":
            path = Path(data_root) / path
        resolved.append(path)
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and run SparNiche formal experiments or ablations."
    )
    parser.add_argument(
        "--experiment-set", choices=("single", "local_graph", "ablation_local_graph"), default="single"
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--source",
        type=Path,
        nargs="+",
        help="Input .h5ad file or directory; relative paths are supported.",
    )
    source_group.add_argument(
        "--input-h5ad",
        dest="source",
        type=Path,
        nargs="+",
        help="Deprecated alias for --source.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-fs/bench_results"),
        help="Shared benchmark output root.",
    )
    parser.add_argument(
        "--method-name",
        default="SparNiche",
        help="Method directory name consumed by calculate_bench.py.",
    )
    parser.add_argument(
        "--sequence-id",
        help="Reuse an existing output sequence directory (required for an explicit resume target).",
    )
    parser.add_argument("--variants", nargs="+")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Explicit seed list. Overrides the default baseline-aligned repeats.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEAT_COUNT,
        help="Independent repeats per experiment. Default: 3.",
    )
    parser.add_argument(
        "--no-repeats",
        action="store_true",
        help="Shortcut for one run using the first selected seed.",
    )
    parser.add_argument("--label-key", default=DEFAULT_LABEL_KEY)
    parser.add_argument(
        "--input-kind",
        choices=("raw_counts", "normalized"),
        help="Deprecated compatibility override; input kind is detected automatically by default.",
    )
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument(
        "--epochs",
        type=int,
        help=f"Official SparNiche DEC-stage epochs (config default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--attention-mode",
        choices=("global", "spatial_local"),
        help="Override SparNiche view-1 attention mode for every generated run.",
    )
    parser.add_argument(
        "--leiden-min-resolution",
        type=float,
        help="Override the inclusive lower bound of the Leiden resolution grid.",
    )
    parser.add_argument(
        "--leiden-max-resolution",
        type=float,
        help="Override the inclusive upper bound of the Leiden resolution grid.",
    )
    parser.add_argument(
        "--leiden-cluster-lower-offset",
        type=int,
        choices=(-1, 0),
        help="Allow K-1 clusters (-1, default) or require at least K clusters (0).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit config path; otherwise the shared default plus a registered source overlay is used.",
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of input samples to run.")
    parser.add_argument("--skip-visual-artifacts", action="store_true", help="Do not write plots or trained.h5ad files.")
    return parser.parse_args()


def _variant_name(row: dict, experiment_set: str) -> str:
    if experiment_set == "ablation":
        return str(row["case"])
    return str(row.get("overrides", {}).get("model.variant", "local_graph_normalized"))


def resolve_repeat_output_root(
    output_root: Path,
    seed: int,
    seeds: list[int],
    method_name: str = "SparNiche",
) -> Path:
    """Map each selected seed to a stable one-based repeat directory."""
    ordered_seeds = list(dict.fromkeys(int(value) for value in seeds))
    try:
        repeat_number = ordered_seeds.index(int(seed)) + 1
    except ValueError as error:
        raise ValueError(f"seed {seed} is absent from the selected seed order") from error
    return Path(output_root) / f"{method_name}_rep{repeat_number}"


def _method_name_for_row(
    base_name: str,
    row: dict,
    experiment_set: str,
    seeds: list[int],
) -> str:
    """Give each ablation case an isolated benchmark output namespace."""
    method_name = str(base_name)
    if experiment_set in {"ablation", "ablation_local_graph"}:
        method_name = f"{method_name}__{str(row['case']).replace(' ', '_')}"
    return resolve_repeat_output_root(
        Path(), int(row["seed"]), seeds, method_name=method_name
    ).name


def resolve_run_seeds(args: argparse.Namespace) -> list[int]:
    """Resolve repetitions with the same default seed sequence as baselines."""
    explicit_seeds = getattr(args, "seeds", None)
    if explicit_seeds:
        seeds = [int(seed) for seed in explicit_seeds]
        return seeds[:1] if getattr(args, "no_repeats", False) else seeds

    repeat_count = 1 if getattr(args, "no_repeats", False) else max(
        int(getattr(args, "repeats", DEFAULT_REPEAT_COUNT)), 1
    )
    return [DEFAULT_SEED + repeat_idx for repeat_idx in range(repeat_count)]


def _repeat_manifest_rows(rows: list[dict], seeds: list[int]) -> list[dict]:
    """Replace manifest seed placeholders with the launcher-selected repeat seeds."""
    base_rows: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        identity = (
            str(row.get("dataset", "")),
            str(row.get("case", "")),
            json.dumps(row.get("overrides", {}), sort_keys=True),
        )
        base_rows.setdefault(identity, row)

    repeated: list[dict] = []
    for row in base_rows.values():
        for seed in seeds:
            repeated_row = dict(row)
            repeated_row["seed"] = int(seed)
            repeated_row["run_id"] = f"{row['case']}-seed{seed}"
            repeated.append(repeated_row)
    return repeated


def selected_rows(args: argparse.Namespace) -> list[dict]:
    seeds = resolve_run_seeds(args)
    if args.experiment_set == "single":
        variants = args.variants or ["local_graph_normalized"]
        unknown = sorted(set(variants) - set(SINGLE_VARIANTS))
        if unknown:
            raise ValueError(f"unknown single variant(s): {', '.join(unknown)}")
        return [
            {
                "run_id": f"single-{variant}-seed{seed}",
                "seed": int(seed),
                "case": "single",
                "overrides": {"model.variant": variant},
            }
            for variant in variants
            for seed in seeds
        ]

    rows = _repeat_manifest_rows(
        expand_manifest(load_yaml(EXPERIMENT_CONFIGS[args.experiment_set])), seeds
    )
    wanted_variants = set(args.variants or [])
    return [
        row
        for row in rows
        if (not wanted_variants or _variant_name(row, args.experiment_set) in wanted_variants)
    ]


def infer_n_clusters(input_path: Path, label_key: str) -> int:
    adata = anndata.read_h5ad(input_path)
    if label_key not in adata.obs:
        raise ValueError(
            f"cannot infer n_clusters: AnnData obs does not contain {label_key!r}; "
            "pass --n-clusters explicitly"
        )
    valid_labels = filter_ambiguous_ground_truth(adata.obs[label_key].tolist())
    n_clusters = int(pd.Series(valid_labels, dtype="string").nunique())
    if n_clusters <= 0:
        raise ValueError(
            f"cannot infer n_clusters: AnnData obs[{label_key!r}] has no valid labels; "
            "pass --n-clusters explicitly"
        )
    return n_clusters


def input_paths_from_source(
    source: Path | list[Path], max_samples: int | None = None
) -> list[Path]:
    sources = source if isinstance(source, list) else [source]
    paths: list[Path] = []
    for item in sources:
        item = Path(item)
        if item.is_file():
            if item.suffix != ".h5ad":
                raise ValueError(f"input file is not .h5ad: {item}")
            paths.append(item)
        elif item.is_dir():
            found = sorted(path for path in item.iterdir() if path.suffix == ".h5ad")
            if not found:
                raise FileNotFoundError(f"no .h5ad files found in input source directory: {item}")
            paths.extend(found)
        else:
            raise FileNotFoundError(f"input source does not exist: {item}")
    return paths[: int(max_samples)] if max_samples else paths


def load_run_config(config_path: Path | None, input_path: Path) -> dict:
    """Load an explicit config or the shared default with a source overlay."""
    if config_path is not None:
        return load_yaml(Path(config_path))
    base = load_yaml(DEFAULT_CONFIG_PATH)
    source_config = SOURCE_CONFIGS.get(Path(input_path).parent.name)
    if source_config is None:
        return base
    return deep_merge(base, load_yaml(source_config))


def resolved_config(
    base: dict,
    args: argparse.Namespace,
    row: dict,
    input_path: Path,
    n_clusters: int,
) -> dict:
    overrides = {
        **row["overrides"],
        "paths.input_h5ad": str(input_path),
        "training.seed": int(row["seed"]),
        "training.device": args.device,
    }
    if args.label_key:
        overrides["data.label_key"] = args.label_key
    if args.input_kind:
        overrides["data.preprocessing.input_kind"] = args.input_kind
    overrides["evaluation.n_clusters"] = int(n_clusters)
    if args.epochs is not None:
        overrides["training.epochs"] = int(args.epochs)
    if args.attention_mode is not None:
        overrides["model.sparniche_view1.attention_mode"] = args.attention_mode
    if getattr(args, "leiden_max_resolution", None) is not None:
        overrides["evaluation.leiden_max_resolution"] = float(
            args.leiden_max_resolution
        )
    if getattr(args, "leiden_min_resolution", None) is not None:
        overrides["evaluation.leiden_min_resolution"] = float(
            args.leiden_min_resolution
        )
    if getattr(args, "leiden_cluster_lower_offset", None) is not None:
        overrides["evaluation.leiden_cluster_lower_offset"] = int(
            args.leiden_cluster_lower_offset
        )
    return apply_overrides(base, overrides)


def allocate_sequence_root(output_root: Path, sequence_id: str | None) -> tuple[Path, str]:
    """Create one isolated output directory for this invocation."""
    if sequence_id:
        sequence_root = output_root / sequence_id
        sequence_root.mkdir(parents=True, exist_ok=True)
        return sequence_root, sequence_id

    sequence_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    sequence_root = output_root / sequence_id
    suffix = 1
    while sequence_root.exists():
        sequence_root = output_root / f"{sequence_id}-{suffix:02d}"
        suffix += 1
    sequence_root.mkdir(parents=True)
    return sequence_root, sequence_root.name


def summarize_prediction_csv(manifest_row: dict) -> dict:
    """Compute clustering metrics for one exported benchmark prediction CSV."""
    prediction_path = Path(manifest_row["method_root"]) / "results" / (
        f"{manifest_row['sample']}.csv"
    )
    record = {
        "run_id": manifest_row.get("run_id", ""),
        "dataset": manifest_row.get("dataset", ""),
        "variant": manifest_row.get("variant", ""),
        "case": manifest_row.get("case", ""),
        "seed": manifest_row.get("seed", ""),
        "sample": manifest_row.get("sample", ""),
        "n": 0,
        "ari": float("nan"),
        "nmi": float("nan"),
        "ami": float("nan"),
        "runtime_seconds": float("nan"),
    }
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    frame = pd.read_csv(prediction_path)
    required = {"ground_truth", "pred"}
    if not required.issubset(frame.columns):
        raise ValueError(f"prediction CSV missing columns: {sorted(required - set(frame.columns))}")
    valid = frame["ground_truth"].notna() & frame["pred"].notna()
    valid &= frame["ground_truth"].astype(str).ne("<NA>")
    valid &= frame["pred"].astype(str).ne("<NA>")
    record["n"] = int(valid.sum())
    if record["n"]:
        metrics = clustering_metrics(frame.loc[valid, "ground_truth"], frame.loc[valid, "pred"])
        record.update(metrics)
    profile_path = prediction_path.with_name(f"{manifest_row['sample']}_profile.json")
    if profile_path.is_file():
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            if profile.get("runtime_seconds") is not None:
                record["runtime_seconds"] = float(profile["runtime_seconds"])
        except (OSError, ValueError, TypeError):
            pass
    return record


def write_experiment_summary(output_root: Path, sequence_id: str, records: list[dict]) -> None:
    """Write machine-readable and human-readable sequence metric summaries."""
    columns = ["sequence_id", "run_id", "dataset", "variant", "case", "seed", "sample", "n", "ari", "nmi", "ami", "runtime_seconds"]
    rows = [{"sequence_id": sequence_id, **record} for record in records]
    frame = pd.DataFrame(rows, columns=columns)
    csv_path = output_root / "experiment_summary.csv"
    json_path = output_root / "experiment_summary.json"
    frame.to_csv(csv_path, index=False)
    json_rows = []
    for row in rows:
        json_rows.append(
            {
                key: (None if isinstance(value, float) and pd.isna(value) else value)
                for key, value in row.items()
            }
        )
    json_path.write_text(json.dumps(json_rows, indent=2), encoding="utf-8")
    print(f"[SUMMARY] completed={len(records)} csv={csv_path}")
    for record in records:
        def fmt(value: object) -> str:
            return "NA" if pd.isna(value) else f"{float(value):.4f}"
        print(
            f"[METRIC] dataset={record['dataset']} variant={record['variant']} "
            f"seed={record['seed']} sample={record['sample']} n={record['n']} "
            f"ARI={fmt(record['ari'])} NMI={fmt(record['nmi'])} AMI={fmt(record['ami'])}"
        )


def aggregate_summary_records(records: list[dict]) -> list[dict]:
    """Aggregate all sample and seed results by model variant and experiment case."""
    if not records:
        return []
    frame = pd.DataFrame(records)
    group_columns = [column for column in ("variant", "case") if column in frame]
    metric_columns = [column for column in ("ari", "nmi", "ami", "runtime_seconds", "n") if column in frame]
    grouped = frame.groupby(group_columns, dropna=False)[metric_columns].agg(["mean", "std", "count"]).reset_index()
    output = []
    for _, row in grouped.iterrows():
        item = {
            column: row[(column, "")] if (column, "") in grouped.columns else row[column]
            for column in group_columns
        }
        for metric in metric_columns:
            item[f"{metric}_mean"] = row[(metric, "mean")]
            item[f"{metric}_std"] = row[(metric, "std")]
            item[f"{metric}_count"] = int(row[(metric, "count")])
        item["sample_count"] = int(frame.loc[
            (frame["variant"] == item["variant"]) & (frame["case"] == item["case"]), "sample"
        ].nunique()) if "sample" in frame else int(item["ari_count"])
        output.append(item)
    return output


def main() -> int:
    args = parse_args()
    args.source = resolve_source_arguments(args.source)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    input_paths = input_paths_from_source(args.source, max_samples=args.max_samples)
    source_is_directory = len(args.source) == 1 and args.source[0].is_dir()
    rows = selected_rows(args)
    if not rows:
        raise ValueError("no runs matched the requested experiment set, variants, and seeds")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sequence_id = args.sequence_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[SEQ ] {sequence_id} output={output_root}")
    manifest_rows_by_path: dict[Path, list[dict]] = {}
    failures = []
    manifest_rows = []
    summary_records: list[dict] = []
    stop_after_failure = False
    input_kind_cache: dict[Path, str] = {}
    selected_seeds = list(dict.fromkeys(int(row["seed"]) for row in rows))
    for input_path in input_paths:
        base = load_run_config(args.config, input_path)
        n_clusters = (
            int(args.n_clusters)
            if args.n_clusters is not None
            else infer_n_clusters(input_path, args.label_key)
        )
        if args.input_kind:
            detected_input_kind = args.input_kind
        else:
            detected_input_kind = input_kind_cache.setdefault(
                input_path, infer_input_kind(input_path)
            )
        for row in rows:
            variant = _variant_name(row, args.experiment_set)
            effective_input = input_path
            input_kind = detected_input_kind
            print(f"[INPUT] {effective_input.name}: input_kind={input_kind}")
            dataset_name = input_path.parent.name
            method_name = _method_name_for_row(
                args.method_name, row, args.experiment_set, selected_seeds
            )
            method_root = output_root / dataset_name / method_name
            base_run_id = str(row["run_id"])
            run_id = (
                f"{input_path.stem}--{base_run_id}"
                if source_is_directory
                else base_run_id
            )
            run_dir = method_root / "artifacts" / run_id
            config = resolved_config(base, args, row, effective_input, n_clusters)
            config["data"]["preprocessing"]["input_kind"] = input_kind
            config["benchmark"] = {
                "external_only": True,
                "method_root": str(method_root),
                "sample_name": input_path.stem,
                "write_plot": not args.skip_visual_artifacts,
                "write_h5ad": not args.skip_visual_artifacts,
            }
            temp_config = Path(tempfile.gettempdir()) / (
                f"sparniche-{output_root.name}-{run_id}.yaml"
            )
            save_yaml(config, temp_config)
            manifest_row = {
                "run_id": run_id,
                "dataset": input_path.stem,
                "variant": variant,
                "case": str(row["case"]),
                "seed": int(row["seed"]),
                "overrides": row["overrides"],
                "input_h5ad": str(effective_input),
                "run_dir": str(run_dir),
                "method_root": str(method_root),
                "sample": input_path.stem,
            }
            manifest_rows.append(manifest_row)
            manifest_path = method_root / "run_manifest.jsonl"
            manifest_rows_by_path.setdefault(manifest_path, []).append(manifest_row)
            command = [
                args.python_exe,
                str(PROJECT_ROOT / "scripts" / "run_sparniche.py"),
                "--config", str(temp_config),
                "--output-dir", str(run_dir),
            ]
            if args.resume:
                command.append("--resume")
            if args.force:
                command.append("--force")
            print(f"[RUN ] {run_id} variant={variant} seed={row['seed']}")
            print("       " + subprocess.list2cmdline(command))
            if args.dry_run:
                temp_config.unlink(missing_ok=True)
                continue
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            temp_config.unlink(missing_ok=True)
            if result.returncode:
                failures.append((run_id, result.returncode))
                print(f"[SUMMARY-SKIP] {run_id}: failed run (exit_code={result.returncode})")
                if not args.continue_on_error:
                    stop_after_failure = True
                    break
            else:
                try:
                    summary_records.append(summarize_prediction_csv(manifest_row))
                except (FileNotFoundError, ValueError, OSError) as error:
                    print(f"[SUMMARY-SKIP] {run_id}: {error}")
        if stop_after_failure:
            break

    for manifest_path, rows_for_path in manifest_rows_by_path.items():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as handle:
            for manifest_row in rows_for_path:
                handle.write(json.dumps(manifest_row, sort_keys=True) + "\n")
        print(f"wrote {len(rows_for_path)} run specifications to {manifest_path}")
    if not args.dry_run:
        write_experiment_summary(output_root, sequence_id, summary_records)
        aggregate = aggregate_summary_records(summary_records)
        aggregate_path = output_root / "experiment_summary_aggregate.csv"
        aggregate_frame = pd.DataFrame(aggregate)
        aggregate_frame.to_csv(aggregate_path, index=False)
        print(f"[SUMMARY] aggregate={aggregate_path}")
        if not aggregate_frame.empty:
            table_columns = [
                column for column in (
                    "case", "sample_count", "ari_mean", "ari_std", "nmi_mean", "nmi_std", "ami_mean", "ami_std"
                ) if column in aggregate_frame.columns
            ]
            print("[SUMMARY TABLE] experiment_group x mean +/- std")
            print(aggregate_frame[table_columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if failures:
        for run_id, code in failures:
            print(f"[FAIL] {run_id}: exit_code={code}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
