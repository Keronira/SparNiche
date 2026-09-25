#!/usr/bin/env python3
"""Search source26 training dynamics, then weight decay and attention locality.

The script is intentionally temporary and does not alter the shared defaults.
It keeps the validated source26 embedding/loss configuration fixed while first
screening ``dec_interval x epochs x local_graph_hops``.  The best dynamic
setting is then validated over three seeds before scanning
``weight_decay x attention_neighbors``.
"""

from __future__ import annotations

import argparse
import gc
import itertools
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


DEFAULT_INPUT = Path("/root/autodl-fs/data/source26/E9.5_E1S1.MOSTA.h5ad")
DEFAULT_OUTPUT = Path("/root/autodl-fs/bench_results/_tmp_source26_dynamic_attention")
SCREEN_SEED = 1234
VALIDATION_SEEDS = (1234, 1235, 1236)
QUALITY_METRICS = ("ari", "nmi", "fmi", "accuracy", "macro_f1")
DYNAMIC_LEVELS = {
    "dec_interval": (5, 10, 20, 40),
    "epochs": (300, 550, 800),
    "local_graph_hops": (0, 1, 2),
}
WEIGHT_DECAY_LEVELS = (0.0, 0.001, 0.01, 0.05)
ATTENTION_LEVELS = (4, 8, 16)
FIXED_SOURCE26 = {
    "hvg": 3000,
    "pca": 64,
    "neighbors": 8,
    "latent": 8,
    "dec_cluster_n": 14,
    "dec_kl_w": 0.5,
    "lr": 0.01,
    "gcn_w": 0.1,
    "rec_w": 10.0,
}
UNKNOWN_LABELS = {
    str(value).strip().lower() for value in AMBIGUOUS_GROUND_TRUTH_LABELS
}


def _token(value: int | float) -> str:
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


def config_id(factors: dict[str, int | float]) -> str:
    return (
        f"int{int(factors['dec_interval'])}_ep{int(factors['epochs'])}_"
        f"hops{int(factors['local_graph_hops'])}_wd{_token(factors['weight_decay'])}_"
        f"att{int(factors['attention_neighbors'])}"
    )


def _spec(factors: dict[str, int | float]) -> dict[str, Any]:
    values = {
        "dec_interval": int(factors["dec_interval"]),
        "epochs": int(factors["epochs"]),
        "local_graph_hops": int(factors["local_graph_hops"]),
        "weight_decay": float(factors["weight_decay"]),
        "attention_neighbors": int(factors["attention_neighbors"]),
    }
    return {"config_id": config_id(values), "factors": values}


def build_dynamic_specs() -> list[dict[str, Any]]:
    base = {"weight_decay": 0.01, "attention_neighbors": 8}
    return [
        _spec({**base, **dict(zip(DYNAMIC_LEVELS, values))})
        for values in itertools.product(*DYNAMIC_LEVELS.values())
    ]


def build_attention_specs(dynamic: dict[str, int | float]) -> list[dict[str, Any]]:
    return [
        _spec(
            {
                **dynamic,
                "weight_decay": weight_decay,
                "attention_neighbors": attention,
            }
        )
        for weight_decay, attention in itertools.product(
            WEIGHT_DECAY_LEVELS, ATTENTION_LEVELS
        )
    ]


def build_config(
    base: dict[str, Any],
    *,
    input_path: Path,
    seed: int,
    device: str,
    factors: dict[str, int | float],
) -> dict[str, Any]:
    overrides = {
        "paths.input_h5ad": str(input_path),
        "data.preprocessing.n_top_genes": FIXED_SOURCE26["hvg"],
        "data.preprocessing.pca_n_components": FIXED_SOURCE26["pca"],
        "data.n_neighbors": FIXED_SOURCE26["neighbors"],
        "model.latent_dim": FIXED_SOURCE26["latent"],
        "model.sparniche_view1.attention_neighbors": int(
            factors["attention_neighbors"]
        ),
        "model.sparniche_view1.dec_cluster_n": FIXED_SOURCE26["dec_cluster_n"],
        "model.sparniche.dec_kl_w": FIXED_SOURCE26["dec_kl_w"],
        "model.sparniche.lr": FIXED_SOURCE26["lr"],
        "model.sparniche.gcn_w": FIXED_SOURCE26["gcn_w"],
        "model.sparniche.rec_w": FIXED_SOURCE26["rec_w"],
        "model.sparniche.dec_interval": int(factors["dec_interval"]),
        "model.sparniche.weight_decay": float(factors["weight_decay"]),
        "model.local_graph_hops": int(factors["local_graph_hops"]),
        "training.epochs": int(factors["epochs"]),
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
    stage: str,
    base: dict[str, Any],
    input_path: Path,
    output_root: Path,
    spec: dict[str, Any],
    seed: int,
    device: str,
    label_key: str,
    force: bool,
) -> dict[str, Any]:
    factors = spec["factors"]
    run_dir = output_root / "runs" / stage / spec["config_id"] / f"seed{seed}"
    result_path = run_dir / "result.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        config = build_config(
            base,
            input_path=input_path,
            seed=seed,
            device=device,
            factors=factors,
        )
        adata = anndata.read_h5ad(input_path)
        artifacts = train_adata(adata, config, run_dir, resume=False)
        valid = _valid_mask(artifacts.adata, label_key)
        evaluation_adata = artifacts.adata[valid].copy()
        labels = evaluation_adata.obs[label_key].astype(str).to_numpy()
        embedding = np.asarray(evaluation_adata.obsm["sparniche"])
        predicted = _predict_clusters(embedding, evaluation_adata, config)
        audit = evaluation_adata.uns.get("sparniche_leiden_search", {})
        row: dict[str, Any] = {
            "stage": stage,
            "config_id": spec["config_id"],
            **factors,
            "sample": input_path.stem,
            "seed": int(seed),
            "status": "completed",
            "runtime_seconds": float(time.perf_counter() - started),
            "selected_resolution": audit.get("selected_resolution"),
            "selected_clusters": int(
                audit.get("selected_clusters", np.unique(predicted).size)
            ),
            **_quality_metrics(labels, predicted),
        }
        pd.DataFrame(
            {"spot_id": np.asarray(evaluation_adata.obs_names), "ground_truth": labels, "pred": predicted}
        ).to_csv(run_dir / "predictions.csv", index=False)
        del evaluation_adata, artifacts, adata
    except Exception as error:
        row = {
            "stage": stage,
            "config_id": spec["config_id"],
            **factors,
            "sample": input_path.stem,
            "seed": int(seed),
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


def _run_specs(
    *, stage: str, specs: list[dict[str, Any]], seeds: tuple[int, ...], args: argparse.Namespace, base: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for seed in seeds:
            print(f"[RUN ] stage={stage} {spec['config_id']} seed={seed}", flush=True)
            row = _run_one(
                stage=stage, base=base, input_path=args.input_h5ad,
                output_root=args.output_root, spec=spec, seed=seed,
                device=args.device, label_key=args.label_key, force=args.force,
            )
            rows.append(row)
            if row["status"] == "completed":
                print(
                    f"[DONE] ARI={row['ari']:.4f} NMI={row['nmi']:.4f} "
                    f"FMI={row['fmi']:.4f} MacroF1={row['macro_f1']:.4f}", flush=True
                )
            else:
                print(f"[FAIL] {row['error']}", flush=True)
                if not args.continue_on_error:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, expected_seeds: int) -> pd.DataFrame:
    completed = frame[frame["status"].astype(str).eq("completed")].copy()
    if completed.empty:
        return pd.DataFrame()
    rows = []
    for config_id_value, group in completed.groupby("config_id", sort=False):
        row: dict[str, Any] = {"config_id": config_id_value, "n_completed": int(group["seed"].nunique())}
        for key in ("dec_interval", "epochs", "local_graph_hops", "weight_decay", "attention_neighbors"):
            row[key] = float(group.iloc[0][key]) if key == "weight_decay" else int(group.iloc[0][key])
        for metric in QUALITY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary[summary["n_completed"].eq(expected_seeds)].sort_values(
        ["ari_mean", "ari_min", "ari_sd", "nmi_mean", "macro_f1_mean", "config_id"],
        ascending=[False, False, True, False, False, True],
    ).reset_index(drop=True).assign(rank=lambda x: np.arange(1, len(x) + 1))


def _all_runs(output_root: Path) -> pd.DataFrame:
    paths = sorted((output_root / "runs").glob("*/ */seed*/result.json"))
    if not paths:
        paths = sorted((output_root / "runs").glob("*/*/seed*/result.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return pd.DataFrame(rows)


def _require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required previous-stage output is missing: {path}")
    return pd.read_csv(path)


def _best_dynamic(ranking: pd.DataFrame) -> dict[str, int | float]:
    if ranking.empty:
        raise RuntimeError("no completed dynamic configuration is available")
    row = ranking.iloc[0]
    return {
        "dec_interval": int(row["dec_interval"]),
        "epochs": int(row["epochs"]),
        "local_graph_hops": int(row["local_graph_hops"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("all", "dynamics", "validate", "attention"), default="all")
    parser.add_argument("--label-key", default="annotation_final")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.input_h5ad = Path(args.input_h5ad)
    args.output_root = Path(args.output_root)
    if not args.input_h5ad.is_file():
        raise FileNotFoundError(args.input_h5ad)
    args.output_root.mkdir(parents=True, exist_ok=True)
    base = load_run_config(None, args.input_h5ad)
    _write_json(
        args.output_root / "manifest.json",
        {"input_h5ad": str(args.input_h5ad), "fixed_source26": FIXED_SOURCE26,
         "dynamic_levels": DYNAMIC_LEVELS, "weight_decay_levels": WEIGHT_DECAY_LEVELS,
         "attention_levels": ATTENTION_LEVELS, "screen_seed": SCREEN_SEED,
         "validation_seeds": VALIDATION_SEEDS, "leiden_window": "K..K+2"},
    )

    if args.stage in {"all", "dynamics"}:
        dynamics = _run_specs(stage="dynamics", specs=build_dynamic_specs(), seeds=(SCREEN_SEED,), args=args, base=base)
        dynamics.to_csv(args.output_root / "stage1_dynamics_runs.csv", index=False)
        dynamics_ranking = summarize(dynamics, expected_seeds=1)
        dynamics_ranking.to_csv(args.output_root / "stage1_dynamics_ranking.csv", index=False)
    else:
        dynamics_ranking = _require_csv(args.output_root / "stage1_dynamics_ranking.csv")
    if args.stage == "dynamics":
        return 0

    dynamic = _best_dynamic(dynamics_ranking)
    _write_json(args.output_root / "selected_dynamic.json", dynamic)
    if args.stage in {"all", "validate"}:
        validation = _run_specs(stage="dynamic_validation", specs=[_spec({**dynamic, "weight_decay": 0.01, "attention_neighbors": 8})], seeds=VALIDATION_SEEDS, args=args, base=base)
        validation.to_csv(args.output_root / "stage2_dynamic_validation.csv", index=False)
        validation_ranking = summarize(validation, expected_seeds=len(VALIDATION_SEEDS))
        validation_ranking.to_csv(args.output_root / "stage2_dynamic_ranking.csv", index=False)
    else:
        validation_ranking = _require_csv(args.output_root / "stage2_dynamic_ranking.csv")
    if args.stage == "validate":
        return 0

    best_dynamic = _best_dynamic(validation_ranking)
    attention_specs = build_attention_specs(best_dynamic)
    attention = _run_specs(stage="attention", specs=attention_specs, seeds=VALIDATION_SEEDS, args=args, base=base)
    attention.to_csv(args.output_root / "stage3_weight_decay_attention_runs.csv", index=False)
    final_ranking = summarize(attention, expected_seeds=len(VALIDATION_SEEDS))
    final_ranking.to_csv(args.output_root / "final_ranking.csv", index=False)
    if final_ranking.empty:
        raise RuntimeError("no fully validated weight-decay/attention configuration is available")
    best = final_ranking.iloc[0]
    best_factors = {"dec_interval": int(best["dec_interval"]), "epochs": int(best["epochs"]),
                    "local_graph_hops": int(best["local_graph_hops"]), "weight_decay": float(best["weight_decay"]),
                    "attention_neighbors": int(best["attention_neighbors"])}
    save_yaml(build_config(base, input_path=args.input_h5ad, seed=VALIDATION_SEEDS[0], device=args.device, factors=best_factors), args.output_root / "best_config.yaml")
    all_runs = _all_runs(args.output_root)
    all_runs.to_csv(args.output_root / "all_runs.csv", index=False)
    failures = int(all_runs["status"].astype(str).ne("completed").sum()) if not all_runs.empty else 0
    print(f"[SUMMARY] failures={failures} output={args.output_root}", flush=True)
    return 1 if failures and not args.continue_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
