#!/usr/bin/env python3
"""Validate three source26 training-dynamic candidates across samples and seeds."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiments import load_run_config  # noqa: E402
from src.benchmark import AMBIGUOUS_GROUND_TRUTH_LABELS  # noqa: E402
from src.config import apply_overrides, normalize_sparniche_config, save_yaml  # noqa: E402
from src.final_metrics import _hungarian_alignment  # noqa: E402
from src.pipeline import train_adata  # noqa: E402
from src.runner import _predict_clusters  # noqa: E402


DEFAULT_DATA_ROOT = Path("/root/autodl-fs/data/source26")
DEFAULT_OUTPUT_ROOT = Path(
    "/root/autodl-fs/bench_results/_tmp_source26_dynamic_3sample_validation"
)
SAMPLE_NAMES = (
    "E9.5_E1S1.MOSTA.h5ad",
    "E10.5_E1S1.MOSTA.h5ad",
    "E11.5_E1S1.MOSTA.h5ad",
)
SEEDS = (1234, 1235, 1236)
ANCHOR_ID = "int20_ep550_hops1"
CANDIDATES = {
    ANCHOR_ID: {"dec_interval": 20, "epochs": 550, "local_graph_hops": 1},
    "int20_ep800_hops1": {
        "dec_interval": 20,
        "epochs": 800,
        "local_graph_hops": 1,
    },
    "int40_ep300_hops2": {
        "dec_interval": 40,
        "epochs": 300,
        "local_graph_hops": 2,
    },
}
QUALITY_METRICS = ("ari", "nmi", "fmi", "accuracy", "macro_f1")
UNKNOWN_LABELS = {
    str(value).strip().lower() for value in AMBIGUOUS_GROUND_TRUTH_LABELS
}


def build_run_specs(data_root: Path) -> list[dict[str, Any]]:
    specs = []
    for sample_name in SAMPLE_NAMES:
        input_path = Path(data_root) / sample_name
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        for config_id, candidate in CANDIDATES.items():
            for seed in SEEDS:
                specs.append(
                    {
                        "run_id": f"{input_path.stem}--{config_id}--seed{seed}",
                        "input_path": input_path,
                        "config_id": config_id,
                        "candidate": dict(candidate),
                        "seed": int(seed),
                    }
                )
    random.Random(20260922).shuffle(specs)
    return specs


def build_config(
    base: dict[str, Any],
    *,
    input_path: Path,
    seed: int,
    device: str,
    candidate: dict[str, int],
) -> dict[str, Any]:
    overrides = {
        "paths.input_h5ad": str(input_path),
        "model.sparniche.dec_interval": int(candidate["dec_interval"]),
        "model.local_graph_hops": int(candidate["local_graph_hops"]),
        "training.epochs": int(candidate["epochs"]),
        "training.seed": int(seed),
        "training.device": str(device),
        "benchmark.external_only": False,
        "evaluation.leiden_cluster_lower_offset": 0,
        "evaluation.leiden_cluster_upper_offset": 2,
    }
    return normalize_sparniche_config(
        apply_overrides(normalize_sparniche_config(base), overrides)
    )


def _valid_mask(adata: anndata.AnnData, label_key: str) -> np.ndarray:
    if label_key not in adata.obs:
        raise KeyError(f"missing label key {label_key!r}")
    labels = adata.obs[label_key].astype("string").fillna("")
    normalized = labels.str.strip().str.lower()
    valid = (~normalized.isin(UNKNOWN_LABELS)).to_numpy()
    if valid.sum() < 3 or np.unique(labels.to_numpy()[valid]).size < 2:
        raise ValueError("not enough valid labels after excluding ambiguous classes")
    return valid


def _quality_metrics(labels: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels).astype(str)
    predicted = np.asarray(predicted).astype(str)
    aligned = _hungarian_alignment(labels, predicted)
    return {
        "ari": float(adjusted_rand_score(labels, predicted)),
        "nmi": float(normalized_mutual_info_score(labels, predicted)),
        "fmi": float(fowlkes_mallows_score(labels, predicted)),
        "accuracy": float(accuracy_score(labels, aligned)),
        "macro_f1": float(f1_score(labels, aligned, average="macro", zero_division=0)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_one(
    *,
    base: dict[str, Any],
    spec: dict[str, Any],
    output_root: Path,
    device: str,
    label_key: str,
    force: bool,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / spec["run_id"]
    result_path = run_dir / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    candidate = spec["candidate"]
    try:
        config = build_config(
            base,
            input_path=spec["input_path"],
            seed=spec["seed"],
            device=device,
            candidate=candidate,
        )
        adata = anndata.read_h5ad(spec["input_path"])
        artifacts = train_adata(adata, config, run_dir, resume=False)
        valid = _valid_mask(artifacts.adata, label_key)
        evaluation_adata = artifacts.adata[valid].copy()
        labels = evaluation_adata.obs[label_key].astype(str).to_numpy()
        embedding = np.asarray(evaluation_adata.obsm["sparniche"])
        predicted = _predict_clusters(embedding, evaluation_adata, config)
        audit = evaluation_adata.uns.get("sparniche_leiden_search", {})
        row: dict[str, Any] = {
            "run_id": spec["run_id"],
            "config_id": spec["config_id"],
            **candidate,
            "sample": spec["input_path"].stem,
            "seed": int(spec["seed"]),
            "status": "completed",
            "runtime_seconds": float(time.perf_counter() - started),
            "selected_resolution": audit.get("selected_resolution"),
            "selected_clusters": int(
                audit.get("selected_clusters", np.unique(predicted).size)
            ),
            **_quality_metrics(labels, predicted),
        }
        pd.DataFrame(
            {
                "spot_id": np.asarray(evaluation_adata.obs_names),
                "ground_truth": labels,
                "pred": predicted,
            }
        ).to_csv(run_dir / "predictions.csv", index=False)
        del evaluation_adata, artifacts, adata
    except Exception as error:
        row = {
            "run_id": spec["run_id"],
            "config_id": spec["config_id"],
            **candidate,
            "sample": spec["input_path"].stem,
            "seed": int(spec["seed"]),
            "status": "failed",
            "runtime_seconds": float(time.perf_counter() - started),
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _write_json(result_path, row)
    return _json_safe(row)


def summarize_runs(
    frame: pd.DataFrame, *, expected_seeds: int = len(SEEDS)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = frame[frame["status"].astype(str).eq("completed")].copy()
    by_sample_rows = []
    for (config_id, sample), group in completed.groupby(
        ["config_id", "sample"], sort=False
    ):
        row: dict[str, Any] = {
            "config_id": config_id,
            "sample": sample,
            "n_completed": int(group["seed"].nunique()),
        }
        for metric in QUALITY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
        by_sample_rows.append(row)
    by_sample = pd.DataFrame(by_sample_rows)
    if by_sample.empty:
        return by_sample, pd.DataFrame()
    by_sample = by_sample[by_sample["n_completed"].eq(int(expected_seeds))].copy()

    overall_rows = []
    for config_id, group in by_sample.groupby("config_id", sort=False):
        row = {
            "config_id": config_id,
            "n_samples": int(group["sample"].nunique()),
        }
        for metric in QUALITY_METRICS:
            values = pd.to_numeric(group[f"{metric}_mean"], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sample_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_min_sample"] = float(values.min())
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows).sort_values(
        ["ari_mean", "ari_min_sample", "nmi_mean", "macro_f1_mean", "config_id"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    overall.insert(0, "rank", np.arange(1, len(overall) + 1))
    return by_sample, overall


def paired_differences(
    frame: pd.DataFrame, *, anchor_id: str = ANCHOR_ID
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = frame[frame["status"].astype(str).eq("completed")].copy()
    keys = ["sample", "seed"]
    anchor = completed[completed["config_id"].eq(anchor_id)][keys + list(QUALITY_METRICS)]
    anchor = anchor.rename(columns={metric: f"{metric}_anchor" for metric in QUALITY_METRICS})
    candidates = completed[~completed["config_id"].eq(anchor_id)]
    details = candidates.merge(anchor, on=keys, how="inner", validate="one_to_one")
    for metric in QUALITY_METRICS:
        details[f"{metric}_delta"] = (
            pd.to_numeric(details[metric]) - pd.to_numeric(details[f"{metric}_anchor"])
        )
    detail_columns = ["config_id", *keys] + [f"{metric}_delta" for metric in QUALITY_METRICS]
    details = details[detail_columns]

    summary_rows = []
    for config_id, group in details.groupby("config_id", sort=False):
        row: dict[str, Any] = {"config_id": config_id, "n_pairs": len(group)}
        for metric in QUALITY_METRICS:
            values = pd.to_numeric(group[f"{metric}_delta"], errors="coerce").dropna()
            row[f"{metric}_delta_mean"] = float(values.mean())
            row[f"{metric}_delta_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_win_rate"] = float((values > 0).mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["ari_delta_mean", "ari_win_rate", "nmi_delta_mean", "config_id"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
    return details, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label-key", default="annotation_final")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    specs = build_run_specs(args.data_root)
    _write_json(
        args.output_root / "manifest.json",
        {
            "samples": SAMPLE_NAMES,
            "seeds": SEEDS,
            "candidates": CANDIDATES,
            "anchor": ANCHOR_ID,
            "run_order": [spec["run_id"] for spec in specs],
            "leiden_window": "K..K+2",
        },
    )
    rows = []
    for index, spec in enumerate(specs, start=1):
        print(f"[RUN ] {index}/{len(specs)} {spec['run_id']}", flush=True)
        base = load_run_config(None, spec["input_path"])
        row = _run_one(
            base=base,
            spec=spec,
            output_root=args.output_root,
            device=args.device,
            label_key=args.label_key,
            force=args.force,
        )
        rows.append(row)
        if row["status"] == "completed":
            print(
                f"[DONE] ARI={row['ari']:.4f} NMI={row['nmi']:.4f} "
                f"FMI={row['fmi']:.4f} MacroF1={row['macro_f1']:.4f}",
                flush=True,
            )
        else:
            print(f"[FAIL] {row['error']}", flush=True)
            if not args.continue_on_error:
                break

    all_runs = pd.DataFrame(rows)
    all_runs.to_csv(args.output_root / "all_runs.csv", index=False)
    by_sample, overall = summarize_runs(all_runs)
    by_sample.to_csv(args.output_root / "by_sample_summary.csv", index=False)
    overall.to_csv(args.output_root / "overall_summary.csv", index=False)
    paired_detail, paired_summary = paired_differences(all_runs)
    paired_detail.to_csv(args.output_root / "paired_differences.csv", index=False)
    paired_summary.to_csv(args.output_root / "paired_summary.csv", index=False)

    if not overall.empty:
        best_id = str(overall.iloc[0]["config_id"])
        best = CANDIDATES[best_id]
        base = load_run_config(None, args.data_root / SAMPLE_NAMES[0])
        save_yaml(
            build_config(
                base,
                input_path=args.data_root / SAMPLE_NAMES[0],
                seed=SEEDS[0],
                device=args.device,
                candidate=best,
            ),
            args.output_root / "best_config.yaml",
        )
    failures = int(all_runs["status"].astype(str).ne("completed").sum())
    print(f"[SUMMARY] completed={len(all_runs)-failures}/27 failures={failures}", flush=True)
    print(f"[SUMMARY] output={args.output_root}", flush=True)
    return 1 if failures and not args.continue_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
