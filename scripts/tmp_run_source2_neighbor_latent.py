#!/usr/bin/env python3
"""Temporary source2 factorial sensitivity experiment for neighbors and latent size."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import AMBIGUOUS_GROUND_TRUTH_LABELS  # noqa: E402
from src.config import apply_overrides, load_yaml, normalize_sparniche_config  # noqa: E402
from src.final_metrics import _hungarian_alignment  # noqa: E402
from src.pipeline import train_adata  # noqa: E402
from src.runner import _predict_clusters  # noqa: E402


DEFAULT_INPUT = Path("/root/autodl-fs/data/source2/200727_09.h5ad")
DEFAULT_OUTPUT = Path(
    "/root/autodl-fs/bench_results/_tmp_source2_neighbors_latent_200727_09"
)
DEFAULT_SEEDS = (1234, 1235, 1236)
NEIGHBOR_LEVELS = (8, 12, 16, 24)
LATENT_LEVELS = (16, 32, 64)
ANCHOR_NAME = "neighbors_12_latent_32"
QUALITY_METRICS = ("ari", "nmi", "fmi", "accuracy", "macro_f1")
RESOURCE_METRICS = ("runtime_seconds", "peak_cuda_memory_mb")
UNKNOWN_LABELS = {
    str(value).strip().lower() for value in AMBIGUOUS_GROUND_TRUTH_LABELS
}


EXPERIMENT_SPECS = sorted([
    {
        "set_name": f"neighbors_{n_neighbors:02d}_latent_{latent_dim}",
        "n_neighbors": n_neighbors,
        "latent_dim": latent_dim,
        "is_anchor": n_neighbors == 12 and latent_dim == 32,
    }
    for n_neighbors in NEIGHBOR_LEVELS
    for latent_dim in LATENT_LEVELS
], key=lambda row: (not row["is_anchor"], row["n_neighbors"], row["latent_dim"]))


def build_config(
    base: dict[str, Any],
    *,
    input_path: Path,
    seed: int,
    device: str,
    n_neighbors: int,
    latent_dim: int,
) -> dict[str, Any]:
    """Apply only the two experimental factors plus fixed execution settings."""
    overrides = {
        "paths.input_h5ad": str(input_path),
        "data.n_neighbors": int(n_neighbors),
        "model.latent_dim": int(latent_dim),
        "model.variant": "local_graph_normalized",
        "model.local_graph_mode": "normalized",
        "model.sparniche_view1.attention_mode": "spatial_local",
        "model.sparniche_view1.attention_neighbors": int(n_neighbors),
        "training.seed": int(seed),
        "training.device": str(device),
        "benchmark.external_only": False,
        "evaluation.leiden_cluster_lower_offset": 0,
        "evaluation.leiden_cluster_upper_offset": 2,
    }
    return normalize_sparniche_config(
        apply_overrides(normalize_sparniche_config(base), overrides)
    )


def compute_quality_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels).astype(str)
    predicted = np.asarray(predicted).astype(str)
    aligned = _hungarian_alignment(labels, predicted)
    return {
        "ari": float(adjusted_rand_score(labels, predicted)),
        "nmi": float(normalized_mutual_info_score(labels, predicted)),
        "fmi": float(fowlkes_mallows_score(labels, predicted)),
        "accuracy": float(accuracy_score(labels, aligned)),
        "macro_f1": float(
            f1_score(labels, aligned, average="macro", zero_division=0)
        ),
    }


def _valid_mask(adata: anndata.AnnData, label_key: str) -> np.ndarray:
    if label_key not in adata.obs:
        raise KeyError(f"missing label key {label_key!r}")
    labels = adata.obs[label_key].astype("string").fillna("")
    normalized = labels.str.strip().str.lower()
    valid = (~normalized.isin(UNKNOWN_LABELS)).to_numpy()
    if valid.sum() < 3 or np.unique(labels.to_numpy()[valid]).size < 2:
        raise ValueError("not enough valid labels after excluding ambiguous classes")
    return valid


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _aggregate_wide(completed: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_columns = ["set_name", "n_neighbors", "latent_dim", "is_anchor"]
    for keys, group in completed.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_completed"] = int(len(group))
        for metric in (*QUALITY_METRICS, *RESOURCE_METRICS):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        records.append(row)
    return pd.DataFrame(records)


def write_summaries(frame: pd.DataFrame, output_root: Path) -> None:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    completed = frame[frame["status"].astype(str) == "completed"].copy()
    if completed.empty:
        return

    summary = _aggregate_wide(completed)
    summary.to_csv(output_root / "summary_by_config.csv", index=False)

    ranks = summary[["set_name", "n_neighbors", "latent_dim", "is_anchor"]].copy()
    for metric in QUALITY_METRICS:
        ranks[f"{metric}_rank"] = summary[f"{metric}_mean"].rank(
            ascending=False, method="average"
        )
    rank_columns = [f"{metric}_rank" for metric in QUALITY_METRICS]
    ranks["mean_quality_rank"] = ranks[rank_columns].mean(axis=1)
    ranks.sort_values(["mean_quality_rank", "set_name"]).to_csv(
        output_root / "quality_ranks.csv", index=False
    )

    anchor = completed[completed["set_name"] == ANCHOR_NAME].set_index("seed")
    delta_rows: list[dict[str, Any]] = []
    for set_name, group in completed.groupby("set_name", sort=True):
        if set_name == ANCHOR_NAME:
            continue
        candidate = group.set_index("seed")
        common = sorted(set(anchor.index) & set(candidate.index))
        for metric in QUALITY_METRICS:
            differences = (
                pd.to_numeric(candidate.loc[common, metric], errors="coerce").to_numpy()
                - pd.to_numeric(anchor.loc[common, metric], errors="coerce").to_numpy()
            )
            differences = differences[np.isfinite(differences)]
            delta_rows.append(
                {
                    "set_name": set_name,
                    "metric": metric,
                    "mean_delta_vs_anchor": float(differences.mean()),
                    "sd_delta_vs_anchor": (
                        float(differences.std(ddof=1)) if differences.size > 1 else 0.0
                    ),
                    "n_pairs": int(differences.size),
                }
            )
    pd.DataFrame(delta_rows).to_csv(
        output_root / "paired_deltas_vs_anchor.csv", index=False
    )

    effect_rows: list[dict[str, Any]] = []
    for factor in ("n_neighbors", "latent_dim"):
        for level, group in completed.groupby(factor, sort=True):
            for metric in QUALITY_METRICS:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                effect_rows.append(
                    {
                        "factor": factor,
                        "level": int(level),
                        "metric": metric,
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "n": int(len(values)),
                    }
                )
    pd.DataFrame(effect_rows).to_csv(output_root / "factor_effects.csv", index=False)


def _run_one(
    *,
    base: dict[str, Any],
    input_path: Path,
    output_root: Path,
    spec: dict[str, Any],
    seed: int,
    device: str,
    label_key: str,
    force: bool,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / str(spec["set_name"]) / f"seed{int(seed)}"
    result_path = run_dir / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        config = build_config(
            base,
            input_path=input_path,
            seed=seed,
            device=device,
            n_neighbors=int(spec["n_neighbors"]),
            latent_dim=int(spec["latent_dim"]),
        )
        adata = anndata.read_h5ad(input_path)
        artifacts = train_adata(adata, config, run_dir, resume=False)
        valid = _valid_mask(artifacts.adata, label_key)
        embedding = np.asarray(artifacts.adata.obsm["sparniche"])[valid]
        evaluation_adata = artifacts.adata[valid].copy()
        predicted = _predict_clusters(embedding, evaluation_adata, config)
        labels = artifacts.adata.obs[label_key].astype(str).to_numpy()[valid]
        metrics = compute_quality_metrics(labels, predicted)
        audit = evaluation_adata.uns.get("sparniche_leiden_search", {})
        peak_cuda = (
            float(torch.cuda.max_memory_allocated() / (1024**2))
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else float("nan")
        )
        row: dict[str, Any] = {
            **spec,
            "sample": input_path.stem,
            "seed": int(seed),
            "status": "completed",
            "runtime_seconds": float(time.perf_counter() - started),
            "peak_cuda_memory_mb": peak_cuda,
            "selected_resolution": float(audit.get("selected_resolution", float("nan"))),
            "selected_clusters": int(audit.get("selected_clusters", np.unique(predicted).size)),
            **metrics,
        }
        pd.DataFrame(
            {
                "spot_id": np.asarray(artifacts.adata.obs_names)[valid],
                "ground_truth": labels,
                "pred": predicted,
            }
        ).to_csv(run_dir / "predictions.csv", index=False)
    except Exception as error:
        row = {
            **spec,
            "sample": input_path.stem,
            "seed": int(seed),
            "status": "failed",
            "runtime_seconds": float(time.perf_counter() - started),
            "peak_cuda_memory_mb": float("nan"),
            "error": f"{type(error).__name__}: {error}",
        }
    _write_json(result_path, row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--sets", nargs="+", default=None)
    parser.add_argument("--label-key", default="annotation_final")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_h5ad)
    output_root = Path(args.output_root)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    base = load_yaml(args.config)
    selected = EXPERIMENT_SPECS
    if args.sets:
        requested = set(args.sets)
        unknown = requested - {str(row["set_name"]) for row in EXPERIMENT_SPECS}
        if unknown:
            raise ValueError(f"unknown sets: {', '.join(sorted(unknown))}")
        selected = [row for row in EXPERIMENT_SPECS if row["set_name"] in requested]

    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "manifest.json",
        {
            "input_h5ad": str(input_path),
            "anchor": ANCHOR_NAME,
            "seeds": [int(seed) for seed in args.seeds],
            "sets": selected,
            "quality_metrics": list(QUALITY_METRICS),
            "resource_metrics": list(RESOURCE_METRICS),
            "leiden_window": "K..K+2",
            "leiden_selection": "ARI-best",
        },
    )

    rows: list[dict[str, Any]] = []
    for spec in selected:
        for seed in args.seeds:
            print(
                f"[RUN ] {spec['set_name']} seed={seed} "
                f"neighbors={spec['n_neighbors']} latent={spec['latent_dim']}",
                flush=True,
            )
            row = _run_one(
                base=base,
                input_path=input_path,
                output_root=output_root,
                spec=spec,
                seed=int(seed),
                device=args.device,
                label_key=args.label_key,
                force=bool(args.force),
            )
            rows.append(row)
            if row["status"] == "completed":
                print(
                    f"[DONE] {spec['set_name']} seed={seed} "
                    f"ARI={row['ari']:.4f} NMI={row['nmi']:.4f} "
                    f"FMI={row['fmi']:.4f} Acc={row['accuracy']:.4f} "
                    f"MacroF1={row['macro_f1']:.4f}",
                    flush=True,
                )
            else:
                print(f"[FAIL] {spec['set_name']} seed={seed}: {row['error']}", flush=True)
                if not args.continue_on_error:
                    pd.DataFrame(rows).to_csv(output_root / "metrics.csv", index=False)
                    return 1
            pd.DataFrame(rows).to_csv(output_root / "metrics.csv", index=False)
            write_summaries(pd.DataFrame(rows), output_root)

    completed = sum(row["status"] == "completed" for row in rows)
    print(
        f"[SUMMARY] completed={completed}/{len(rows)} output={output_root}",
        flush=True,
    )
    return 0 if completed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
