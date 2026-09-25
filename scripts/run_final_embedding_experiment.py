#!/usr/bin/env python3
"""Run final or first-stage screening SparNiche experiments.

The default single mode evaluates the tail samples from ``source24``.  The
screening mode evaluates the 15 deduplicated, one-factor-at-a-time sets
defined below.  Both modes use the existing Leiden prediction path and write
sample-, seed-, and overall-level mean/SD summaries.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import AMBIGUOUS_GROUND_TRUTH_LABELS  # noqa: E402
from src.config import apply_overrides, load_yaml, normalize_sparniche_config  # noqa: E402
from src.final_metrics import compute_final_metrics  # noqa: E402
from src.pipeline import train_adata  # noqa: E402
from src.runner import _predict_clusters  # noqa: E402


DEFAULT_SEEDS = (1234, 1235, 1236)
UNKNOWN_LABELS = {str(value).strip().lower() for value in AMBIGUOUS_GROUND_TRUTH_LABELS}
METRIC_COLUMNS = (
    "ari",
    "nmi",
    "fmi",
    "accuracy",
    "macro_f1",
    "macro_layer_iou",
    "worst_layer_iou",
    "layer_recovery_rate",
    "layer_order_score",
    "fidelity",
)

# The first-stage screen intentionally excludes duplicate anchor settings.
# Every entry is merged on top of the same full SparNiche anchor configuration.
SCREENING_SET_OVERRIDES: dict[str, dict[str, Any]] = {
    "anchor": {},
    "neighbors_08": {
        "data.n_neighbors": 8,
        "model.sparniche_view1.attention_neighbors": 8,
    },
    "neighbors_16": {
        "data.n_neighbors": 16,
        "model.sparniche_view1.attention_neighbors": 16,
    },
    "neighbors_24": {
        "data.n_neighbors": 24,
        "model.sparniche_view1.attention_neighbors": 24,
    },
    "latent_16": {"model.latent_dim": 16},
    "latent_64": {"model.latent_dim": 64},
    "norm_off": {"model.local_graph_normalize": False},
    "dropout_00": {"model.sparniche_view1.dropout": 0.0},
    "dropout_40": {"model.sparniche_view1.dropout": 0.4},
    "lr_low": {
        "model.sparniche.lr": 0.005,
    },
    "lr_high": {
        "model.sparniche.lr": 0.02,
    },
    "wd_none": {
        "model.sparniche.weight_decay": 0.0,
    },
    "wd_strong": {
        "model.sparniche.weight_decay": 0.03,
    },
    "epochs_300": {"training.epochs": 300},
    "epochs_800": {"training.epochs": 800},
}

INTERACTION_SET_OVERRIDES: dict[str, dict[str, Any]] = {
    "lr_low_latent_64": {
        "model.latent_dim": 64,
        "model.sparniche.lr": 0.005,
    }
}
EXPERIMENT_SET_OVERRIDES = {**SCREENING_SET_OVERRIDES, **INTERACTION_SET_OVERRIDES}
FINAL_SET_NAMES = (
    "neighbors_16",
    "neighbors_08",
    "lr_low",
    "latent_64",
    "lr_low_latent_64",
)


def _sample_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), stem) if stem.isdigit() else (math.inf, stem)


def select_tail_samples(paths: list[Path], count: int) -> list[Path]:
    if int(count) <= 0:
        raise ValueError("tail sample count must be positive")
    return sorted((Path(path) for path in paths), key=_sample_sort_key)[-int(count):]


def select_samples(paths: list[Path], sample_names: list[str]) -> list[Path]:
    """Select samples by exact stem while preserving the requested order."""
    available = {Path(path).stem: Path(path) for path in paths}
    missing = [str(name) for name in sample_names if str(name) not in available]
    if missing:
        raise ValueError(f"requested samples are unavailable: {', '.join(missing)}")
    return [available[str(name)] for name in sample_names]


def set_output_dir(output_root: Path, experiment_set: str, set_name: str) -> Path:
    """Keep multi-set experiment artifacts isolated by set name."""
    if experiment_set in {"screening", "final"}:
        return Path(output_root) / set_name
    return Path(output_root)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def load_existing_metrics(root: Path, *, set_name: str) -> list[dict[str, Any]]:
    """Load a completed set's metrics without retraining it."""
    metrics_path = Path(root) / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"existing metrics file is unavailable: {metrics_path}")
    frame = pd.read_csv(metrics_path)
    if frame.empty:
        raise ValueError(f"existing metrics file is empty: {metrics_path}")
    rows = frame.to_dict(orient="records")
    for row in rows:
        row["set_name"] = str(set_name)
        if "sample" in row:
            row["sample"] = str(row["sample"])
        if "seed" in row:
            row["seed"] = int(row["seed"])
        if "status" in row:
            row["status"] = str(row["status"])
    return rows


def load_anchor_rep_metrics(
    root_pattern: str | Path,
    source: Path,
    samples: list[Path],
) -> list[dict[str, Any]]:
    """Recompute anchor metrics from ``SparNiche_rep*`` result exports.

    Each replicate contains ``results/<sample>.csv`` with ground-truth and
    predicted labels plus ``<sample>_embedding.csv``.  Replicate numbers map
    to the established seeds 1234, 1235, and 1236.
    """
    pattern = str(root_pattern)
    roots = sorted(Path(value) for value in glob.glob(pattern))
    if not roots and Path(pattern).is_dir():
        roots = [Path(pattern)]
    if not roots:
        raise FileNotFoundError(f"no SparNiche replicate directories match {pattern!r}")
    sample_paths = {Path(path).stem: Path(path) for path in samples}
    rows: list[dict[str, Any]] = []
    for rep_root in roots:
        match = re.search(r"rep(\d+)$", rep_root.name)
        if match is None:
            raise ValueError(f"cannot infer replicate number from {rep_root}")
        rep_index = int(match.group(1))
        if rep_index < 1 or rep_index > len(DEFAULT_SEEDS):
            raise ValueError(f"unsupported SparNiche replicate {rep_root.name}")
        seed = int(DEFAULT_SEEDS[rep_index - 1])
        results_dir = rep_root / "results"
        for sample_name, sample_path in sample_paths.items():
            result_path = results_dir / f"{sample_name}.csv"
            embedding_path = results_dir / f"{sample_name}_embedding.csv"
            profile_path = results_dir / f"{sample_name}_profile.json"
            if not result_path.exists() or not embedding_path.exists():
                raise FileNotFoundError(
                    f"missing anchor result or embedding for {sample_name} in {rep_root}"
                )
            result_frame = pd.read_csv(result_path)
            embedding_frame = pd.read_csv(embedding_path)
            if not {"spot_id", "ground_truth", "pred"}.issubset(result_frame.columns):
                raise ValueError(f"invalid anchor result columns: {result_path}")
            if "spot_id" not in embedding_frame.columns:
                raise ValueError(f"invalid anchor embedding columns: {embedding_path}")
            adata = anndata.read_h5ad(sample_path)
            if "spatial" not in adata.obsm:
                raise KeyError(f"sample has no spatial coordinates: {sample_path}")
            spot_ids = result_frame["spot_id"].astype(str).to_numpy()
            obs_index = pd.Index(np.asarray(adata.obs_names).astype(str))
            positions = obs_index.get_indexer(spot_ids)
            if np.any(positions < 0):
                raise ValueError(f"anchor spot IDs do not match {sample_path}")
            embedding_indexed = embedding_frame.copy()
            embedding_indexed["spot_id"] = embedding_indexed["spot_id"].astype(str)
            embedding_indexed = embedding_indexed.set_index("spot_id")
            if not pd.Index(spot_ids).isin(embedding_indexed.index).all():
                raise ValueError(f"anchor embedding IDs do not match {result_path}")
            embedding = embedding_indexed.loc[spot_ids].to_numpy(dtype=np.float32)
            labels = result_frame["ground_truth"].astype("string").fillna("")
            normalized = labels.str.strip().str.lower()
            valid = (~normalized.isin(UNKNOWN_LABELS)).to_numpy()
            if valid.sum() < 3 or np.unique(labels.to_numpy()[valid]).size < 2:
                raise ValueError(f"not enough valid anchor labels in {result_path}")
            profile = {}
            if profile_path.exists():
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
            metrics = compute_final_metrics(
                embedding[valid],
                labels.to_numpy()[valid],
                np.asarray(adata.obsm["spatial"])[positions][valid],
                predicted=result_frame["pred"].to_numpy()[valid],
                seed=seed,
                n_neighbors=12,
            )
            rows.append(
                {
                    "sample": sample_name,
                    "seed": seed,
                    "runtime_seconds": float(profile.get("runtime_seconds", float("nan"))),
                    "status": "completed",
                    "set_name": "anchor",
                    **metrics,
                }
            )
    return rows


def summarize_sets(rows: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    """Write one mean/SD summary row per set and metric."""
    frame = pd.DataFrame(rows)
    completed = frame[frame["status"].astype(str) == "completed"] if not frame.empty else frame
    records: list[dict[str, Any]] = []
    for set_name, group in completed.groupby("set_name", sort=True):
        for metric in METRIC_COLUMNS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            records.append(
                {
                    "set_name": str(set_name),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": int(values.size),
                }
            )
    summary = pd.DataFrame(records, columns=["set_name", "metric", "mean", "sd", "n"])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def _valid_mask(adata: anndata.AnnData, label_key: str) -> np.ndarray:
    if label_key not in adata.obs:
        raise KeyError(f"missing label key {label_key!r}")
    labels = adata.obs[label_key].astype("string").fillna("")
    normalized = labels.str.strip().str.lower()
    mask = (~normalized.isin(UNKNOWN_LABELS)).to_numpy()
    if mask.sum() < 3 or np.unique(labels.to_numpy()[mask]).size < 2:
        raise ValueError("not enough valid labels for final metrics")
    return mask


def _full_config(
    base: dict[str, Any],
    input_path: Path,
    seed: int,
    device: str,
    *,
    set_name: str = "anchor",
) -> dict[str, Any]:
    if set_name not in EXPERIMENT_SET_OVERRIDES:
        raise ValueError(f"unknown experiment set {set_name!r}")
    overrides = {
        "paths.input_h5ad": str(input_path),
        "model.variant": "local_graph_normalized",
        "model.local_graph_mode": "normalized",
        "model.sparniche_view1.attention_mode": "spatial_local",
        "model.sparniche_view1.attention_neighbors": 12,
        "model.sparniche_view1.attention_chunk_size": 4096,
        "data.n_neighbors": 12,
        "training.seed": int(seed),
        "training.device": str(device),
        "benchmark.external_only": False,
        "evaluation.leiden_cluster_lower_offset": 0,
        "evaluation.leiden_cluster_upper_offset": 2,
    }
    overrides.update(EXPERIMENT_SET_OVERRIDES[set_name])
    return normalize_sparniche_config(apply_overrides(normalize_sparniche_config(base), overrides))


def _evaluate(
    adata: anndata.AnnData,
    label_key: str,
    seed: int,
    config: dict[str, Any],
) -> dict[str, float]:
    mask = _valid_mask(adata, label_key)
    embedding = np.asarray(adata.obsm["sparniche"])[mask]
    labels = adata.obs[label_key].astype(str).to_numpy()[mask]
    spatial = np.asarray(adata.obsm["spatial"])[mask]
    predicted_full = _predict_clusters(embedding, adata[mask].copy(), config)
    return compute_final_metrics(
        embedding,
        labels,
        spatial,
        predicted=predicted_full,
        seed=int(seed),
        n_neighbors=12,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    parser.add_argument("--label-key", default="annotation_final")
    parser.add_argument("--tail-samples", type=int, default=6)
    parser.add_argument(
        "--samples",
        nargs="+",
        default=None,
        help="exact sample stems to use; overrides --tail-samples",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--experiment-set",
        choices=("single", "screening", "final"),
        default="single",
        help="single anchor run, all 15 screening sets, or five final candidates",
    )
    parser.add_argument(
        "--anchor-root",
        type=Path,
        default=None,
        help="existing anchor directory used by --experiment-set final",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true", help="overwrite existing run directories")
    return parser.parse_args()


def _run_set(
    args: argparse.Namespace,
    base: dict[str, Any],
    source: Path,
    samples: list[Path],
    output_root: Path,
    set_name: str,
) -> int:
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "manifest.json",
        {
            "source": str(source),
            "set_name": set_name,
            "overrides": EXPERIMENT_SET_OVERRIDES[set_name],
            "samples": [p.stem for p in samples],
            "seeds": [int(s) for s in args.seeds],
            "variant": "full",
            "leiden_used": True,
            "leiden_cluster_window": "k-1..k+3",
            "metrics": list(METRIC_COLUMNS),
        },
    )
    rows: list[dict[str, Any]] = []
    for sample_path in samples:
        for seed in args.seeds:
            run_dir = output_root / f"seed{int(seed)}" / sample_path.stem
            status_path = run_dir / "final_metrics.json"
            if status_path.exists() and not args.force:
                print(f"[SKIP] {sample_path.stem} seed={seed} existing final_metrics.json", flush=True)
                rows.append(json.loads(status_path.read_text(encoding="utf-8")))
                continue
            started = time.perf_counter()
            print(f"[RUN ] set={set_name} sample={sample_path.stem} seed={seed}", flush=True)
            try:
                config = _full_config(base, sample_path, int(seed), args.device, set_name=set_name)
                adata = anndata.read_h5ad(sample_path)
                artifacts = train_adata(adata, config, run_dir, resume=False)
                metrics = _evaluate(artifacts.adata, args.label_key, int(seed), config)
                row: dict[str, Any] = {
                    "sample": sample_path.stem,
                    "seed": int(seed),
                    "runtime_seconds": float(time.perf_counter() - started),
                    "status": "completed",
                    **metrics,
                }
                _write_json(status_path, row)
                rows.append(row)
                print(
                    f"[DONE] set={set_name} sample={sample_path.stem} seed={seed} "
                    f"ARI={row['ari']:.4f} NMI={row['nmi']:.4f} "
                    f"FMI={row['fmi']:.4f} MacroF1={row['macro_f1']:.4f}",
                    flush=True,
                )
            except Exception as error:
                row = {
                    "sample": sample_path.stem,
                    "seed": int(seed),
                    "runtime_seconds": float(time.perf_counter() - started),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                _write_json(status_path, row)
                rows.append(row)
                print(f"[FAIL] set={set_name} {row['error']}", flush=True)
                if not args.continue_on_error:
                    raise

    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "metrics.csv", index=False)
    frame.to_json(output_root / "metrics.jsonl", orient="records", lines=True)
    completed = frame[frame["status"] == "completed"] if not frame.empty else frame
    if not completed.empty:
        overall = pd.DataFrame(
            {
                "metric": list(METRIC_COLUMNS),
                "mean": [completed[m].mean() for m in METRIC_COLUMNS],
                "sd": [completed[m].std(ddof=1) for m in METRIC_COLUMNS],
                "n": [completed[m].notna().sum() for m in METRIC_COLUMNS],
            }
        )
        overall.to_csv(output_root / "summary_overall.csv", index=False)
        completed.groupby("sample")[list(METRIC_COLUMNS)].agg(["mean", "std"]).to_csv(output_root / "summary_by_sample.csv")
        completed.groupby("seed")[list(METRIC_COLUMNS)].agg(["mean", "std"]).to_csv(output_root / "summary_by_seed.csv")
        _write_json(
            output_root / "summary_overall.json",
            {metric: {"mean": float(overall.loc[overall.metric == metric, "mean"].iloc[0]), "sd": float(overall.loc[overall.metric == metric, "sd"].iloc[0])} for metric in METRIC_COLUMNS},
        )
    print(f"[SUMMARY] completed={len(completed)} total={len(frame)} output={output_root}", flush=True)
    return 0 if len(completed) == len(frame) else 1


def main() -> int:
    args = parse_args()
    base = load_yaml(args.config)
    source = Path(args.source)
    all_paths = list(source.glob("*.h5ad"))
    if not all_paths:
        raise ValueError(f"no h5ad files found under {source}")
    if args.samples:
        samples = select_samples(all_paths, [str(name) for name in args.samples])
    else:
        samples = select_tail_samples(all_paths, args.tail_samples)
    output_root = Path(args.output_root)
    if args.experiment_set == "screening":
        set_names = list(SCREENING_SET_OVERRIDES)
    elif args.experiment_set == "final":
        if args.anchor_root is None:
            raise ValueError("--anchor-root is required for --experiment-set final")
        if len(samples) != 6 or len(args.seeds) != 3:
            raise ValueError("final mode requires exactly 6 samples and 3 seeds")
        set_names = list(FINAL_SET_NAMES)
    else:
        set_names = ["anchor"]
    exit_codes = []
    for set_name in set_names:
        set_output = set_output_dir(output_root, args.experiment_set, set_name)
        exit_codes.append(_run_set(args, base, source, samples, set_output, set_name))
    if args.experiment_set == "final":
        anchor_text = str(args.anchor_root)
        if glob.has_magic(anchor_text) or "SparNiche_rep" in Path(anchor_text).name:
            combined_rows = load_anchor_rep_metrics(anchor_text, source, samples)
        else:
            combined_rows = load_existing_metrics(args.anchor_root, set_name="anchor")
        for set_name in FINAL_SET_NAMES:
            combined_rows.extend(load_existing_metrics(output_root / set_name, set_name=set_name))
        pd.DataFrame(combined_rows).to_csv(output_root / "metrics_all_sets.csv", index=False)
        summarize_sets(combined_rows, output_root / "summary_all_sets.csv")
    return 0 if all(code == 0 for code in exit_codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
