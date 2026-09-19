#!/usr/bin/env python3
"""Tune SparNiche spatial-neighborhood sizes on source24 embeddings.

The ``full`` condition is the existing reference (12 RNA graph neighbors and
12 local-attention neighbors).  Its embeddings are read from the completed
SparNiche rep1/2/3 artifacts.  The trainer requires local-attention neighbors
to be no wider than the graph neighbor index, so the legal comparison grid is
explicitly recorded below.  No Leiden clustering is called.
"""

from __future__ import annotations

import argparse
import json
import math
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
from src.embedding_metrics import compute_embedding_metrics  # noqa: E402
from src.pipeline import train_adata  # noqa: E402


SWEEP_SPECS: dict[str, dict[str, int]] = {
    "full": {"n_neighbors": 12, "attention_neighbors": 12},
    "graph_6": {"n_neighbors": 6, "attention_neighbors": 6},
    "graph_24": {"n_neighbors": 24, "attention_neighbors": 12},
    "attention_6": {"n_neighbors": 12, "attention_neighbors": 6},
    "attention_24": {"n_neighbors": 24, "attention_neighbors": 24},
}
DEFAULT_SEEDS = (1234, 1235, 1236)
UNKNOWN_LABELS = {str(value).strip().lower() for value in AMBIGUOUS_GROUND_TRUTH_LABELS}


def _sample_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), stem) if stem.isdigit() else (math.inf, stem)


def select_tail_samples(paths: list[Path], count: int) -> list[Path]:
    if int(count) <= 0:
        raise ValueError("sample count must be positive")
    return sorted((Path(path) for path in paths), key=_sample_sort_key)[-int(count):]


def _valid_mask(adata: anndata.AnnData, label_key: str) -> np.ndarray:
    if label_key not in adata.obs:
        raise KeyError(f"missing label key {label_key!r}")
    labels = adata.obs[label_key].astype("string").fillna("")
    normalized = labels.str.strip().str.lower()
    mask = (~normalized.isin(UNKNOWN_LABELS)).to_numpy()
    if mask.sum() < 3 or len(np.unique(labels.to_numpy()[mask])) < 2:
        raise ValueError("not enough valid labels for embedding diagnostics")
    return mask


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _condition_config(
    base: dict[str, Any], condition: str, seed: int, input_path: Path, device: str
) -> dict[str, Any]:
    spec = SWEEP_SPECS[condition]
    overrides: dict[str, Any] = {
        "paths.input_h5ad": str(input_path),
        "data.n_neighbors": int(spec["n_neighbors"]),
        "model.variant": "local_graph_normalized",
        "model.local_graph_mode": "normalized",
        "model.sparniche_view1.attention_mode": "spatial_local",
        "model.sparniche_view1.attention_neighbors": int(spec["attention_neighbors"]),
        "model.sparniche_view1.attention_chunk_size": 4096,
        "training.seed": int(seed),
        "training.device": str(device),
        "benchmark.external_only": True,
    }
    return normalize_sparniche_config(apply_overrides(normalize_sparniche_config(base), overrides))


def _embedding_row(
    adata: anndata.AnnData,
    label_key: str,
    seed: int,
    sample: str,
    condition: str,
    runtime_seconds: float,
    status: str = "completed",
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"sample": sample, "condition": condition, "seed": int(seed), "runtime_seconds": float(runtime_seconds), "status": status}
    if error is not None:
        row["error"] = error
        return row
    mask = _valid_mask(adata, label_key)
    embedding = np.asarray(adata.obsm["sparniche"])[mask]
    labels = adata.obs[label_key].astype(str).to_numpy()[mask]
    spatial = np.asarray(adata.obsm["spatial"])[mask]
    row.update(compute_embedding_metrics(embedding, labels, spatial, seed=int(seed), n_neighbors=12))
    return row


def _baseline_path(baseline_root: Path, sample: str, seed: int) -> Path:
    rep = int(seed) - 1233
    return baseline_root / f"SparNiche_rep{rep}" / "artifacts" / f"{sample}--single-local_graph_normalized-seed{seed}" / "trained.h5ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    parser.add_argument("--label-key", default="annotation_final")
    parser.add_argument("--tail-samples", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = load_yaml(args.config)
    source = Path(args.source)
    paths = [path for path in source.glob("*.h5ad")]
    samples = select_tail_samples(paths, args.tail_samples)
    if not samples:
        raise ValueError(f"no h5ad files under {source}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "manifest.json", {"source": str(source), "samples": [p.stem for p in samples], "seeds": [int(s) for s in args.seeds], "conditions": SWEEP_SPECS, "primary_metric": "linear_probe_macro_f1", "leiden_used": False, "baseline_source": str(args.baseline_root)})
    rows: list[dict[str, Any]] = []
    for sample_path in samples:
        for seed in args.seeds:
            sample = sample_path.stem
            baseline = _baseline_path(Path(args.baseline_root), sample, int(seed))
            started = time.perf_counter()
            print(f"[BASELINE] sample={sample} seed={seed} path={baseline}", flush=True)
            try:
                baseline_adata = anndata.read_h5ad(baseline)
                row = _embedding_row(baseline_adata, args.label_key, int(seed), sample, "full", time.perf_counter() - started)
                rows.append(row)
                _write_json(output_root / "baseline" / f"{sample}-seed{seed}.json", row)
                print(f"[DONE] sample={sample} condition=full seed={seed} Macro-F1={row['linear_probe_macro_f1']:.4f}", flush=True)
            except Exception as error:
                failure = {
                    "sample": sample,
                    "condition": "full",
                    "seed": int(seed),
                    "runtime_seconds": float(time.perf_counter() - started),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                rows.append(failure)
                print(f"[FAIL] baseline {failure['error']}", flush=True)
                if not args.continue_on_error:
                    raise
        for condition in (name for name in SWEEP_SPECS if name != "full"):
            for seed in args.seeds:
                sample = sample_path.stem
                run_dir = output_root / "runs" / condition / f"seed{int(seed)}" / sample
                started = time.perf_counter()
                print(f"[RUN ] sample={sample} condition={condition} seed={seed}", flush=True)
                try:
                    config = _condition_config(base, condition, int(seed), sample_path, args.device)
                    adata = anndata.read_h5ad(sample_path)
                    artifacts = train_adata(adata, config, run_dir)
                    row = _embedding_row(artifacts.adata, args.label_key, int(seed), sample, condition, time.perf_counter() - started)
                    np.savez_compressed(run_dir / "embedding.npz", embedding=np.asarray(artifacts.adata.obsm["sparniche"]), labels=artifacts.adata.obs[args.label_key].astype(str).to_numpy(), spatial=np.asarray(artifacts.adata.obsm["spatial"]))
                    _write_json(run_dir / "metrics.json", row)
                    rows.append(row)
                    print(f"[DONE] sample={sample} condition={condition} seed={seed} Macro-F1={row['linear_probe_macro_f1']:.4f}", flush=True)
                except Exception as error:
                    failure = {"sample": sample, "condition": condition, "seed": int(seed), "runtime_seconds": float(time.perf_counter() - started), "status": "failed", "error": f"{type(error).__name__}: {error}"}
                    _write_json(run_dir / "metrics.json", failure)
                    rows.append(failure)
                    print(f"[FAIL] {failure['error']}", flush=True)
                    if not args.continue_on_error:
                        raise
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "embedding_metrics.csv", index=False)
    frame.to_json(output_root / "embedding_metrics.jsonl", orient="records", lines=True)
    completed = frame[frame["status"] == "completed"] if not frame.empty else frame
    if not completed.empty:
        numeric = [c for c in completed.columns if c not in {"sample", "condition", "status", "error"} and pd.api.types.is_numeric_dtype(completed[c])]
        completed.groupby("condition")[numeric].agg(["mean", "std"]).to_csv(output_root / "embedding_metrics_by_condition.csv")
    print(f"[SUMMARY] completed={len(completed)} total={len(frame)} output={output_root}", flush=True)
    return 0 if len(completed) == len(frame) else 1


if __name__ == "__main__":
    raise SystemExit(main())
