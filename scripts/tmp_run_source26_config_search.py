#!/usr/bin/env python3
"""Tune HVG, PCA, spatial neighbors, and latent size on source26 sample 1."""

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

from src.benchmark import AMBIGUOUS_GROUND_TRUTH_LABELS  # noqa: E402
from src.config import apply_overrides, load_yaml, normalize_sparniche_config, save_yaml  # noqa: E402
from src.final_metrics import _hungarian_alignment  # noqa: E402
from src.pipeline import train_adata  # noqa: E402
from src.runner import _predict_clusters  # noqa: E402


DEFAULT_INPUT = Path("/root/autodl-fs/data/source26/E9.5_E1S1.MOSTA.h5ad")
DEFAULT_OUTPUT = Path("/root/autodl-fs/bench_results/_tmp_source26_config_search")
DEFAULT_VALIDATION_SEEDS = (1234, 1235, 1236)
FACTOR_LEVELS = {
    "hvg": (2000, 3000, 5000, 10000),
    "pca": (32, 64, 128, 200),
    "neighbors": (6, 8, 12, 16, 24),
    "latent": (8, 16, 32, 64),
}
ANCHOR = {"hvg": 2000, "pca": 200, "neighbors": 12, "latent": 32}
QUALITY_METRICS = ("ari", "nmi", "fmi", "accuracy", "macro_f1")
UNKNOWN_LABELS = {
    str(value).strip().lower() for value in AMBIGUOUS_GROUND_TRUTH_LABELS
}


def config_id(factors: dict[str, int]) -> str:
    return (
        f"hvg{int(factors['hvg'])}_pca{int(factors['pca'])}_"
        f"neighbors{int(factors['neighbors'])}_latent{int(factors['latent'])}"
    )


def _spec(factors: dict[str, int]) -> dict[str, Any]:
    values = {key: int(factors[key]) for key in FACTOR_LEVELS}
    return {"config_id": config_id(values), "factors": values}


def build_stage1_specs() -> list[dict[str, Any]]:
    specs = [_spec(ANCHOR)]
    for factor, levels in FACTOR_LEVELS.items():
        for level in levels:
            if int(level) == ANCHOR[factor]:
                continue
            factors = dict(ANCHOR)
            factors[factor] = int(level)
            specs.append(_spec(factors))
    return specs


def select_best_levels(stage1: pd.DataFrame) -> dict[str, list[int]]:
    completed = stage1[
        stage1["status"].astype(str).eq("completed")
        & pd.to_numeric(stage1["ari"], errors="coerce").notna()
    ].copy()
    selected: dict[str, list[int]] = {}
    for factor, declared_levels in FACTOR_LEVELS.items():
        mask = np.ones(len(completed), dtype=bool)
        for other in FACTOR_LEVELS:
            if other != factor:
                mask &= pd.to_numeric(completed[other], errors="coerce").eq(
                    ANCHOR[other]
                ).to_numpy()
        candidates = completed.loc[mask].copy()
        candidates[factor] = pd.to_numeric(candidates[factor], errors="coerce")
        candidates["ari"] = pd.to_numeric(candidates["ari"], errors="coerce")
        scores = candidates.groupby(factor, as_index=False)["ari"].mean()
        order = {int(level): index for index, level in enumerate(declared_levels)}
        scores["level_order"] = scores[factor].map(
            lambda value: order.get(int(value), len(order))
        )
        scores = scores.sort_values(
            ["ari", "level_order"], ascending=[False, True]
        )
        levels = [int(value) for value in scores[factor].head(2)]
        if len(levels) != 2:
            raise ValueError(f"stage 1 does not contain two valid levels for {factor}")
        selected[factor] = levels
    return selected


def build_stage2_specs(selected_levels: dict[str, list[int]]) -> list[dict[str, Any]]:
    specs = []
    for values in itertools.product(*(selected_levels[key] for key in FACTOR_LEVELS)):
        specs.append(_spec(dict(zip(FACTOR_LEVELS, values))))
    return specs


def select_stage3_specs(stage2: pd.DataFrame, count: int = 4) -> list[dict[str, Any]]:
    completed = stage2[
        stage2["status"].astype(str).eq("completed")
        & pd.to_numeric(stage2["ari"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(
        ["ari", "config_id"], ascending=[False, True]
    ).drop_duplicates("config_id")
    if len(completed) < int(count):
        raise ValueError(f"stage 2 has only {len(completed)} completed configurations")
    specs = []
    for _, row in completed.head(int(count)).iterrows():
        factors = {key: int(row[key]) for key in FACTOR_LEVELS}
        specs.append({"config_id": str(row["config_id"]), "factors": factors})
    return specs


def build_config(
    base: dict[str, Any],
    *,
    input_path: Path,
    seed: int,
    device: str,
    factors: dict[str, int],
) -> dict[str, Any]:
    overrides = {
        "paths.input_h5ad": str(input_path),
        "data.preprocessing.n_top_genes": int(factors["hvg"]),
        "data.preprocessing.pca_n_components": int(factors["pca"]),
        "data.n_neighbors": int(factors["neighbors"]),
        "model.sparniche_view1.attention_neighbors": int(factors["neighbors"]),
        "model.latent_dim": int(factors["latent"]),
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
        "macro_f1": float(
            f1_score(labels, aligned, average="macro", zero_division=0)
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
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
    input_path: Path,
    output_root: Path,
    spec: dict[str, Any],
    seed: int,
    device: str,
    label_key: str,
    force: bool,
) -> dict[str, Any]:
    factors = spec["factors"]
    run_dir = output_root / "runs" / spec["config_id"] / f"seed{int(seed)}"
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
        embedding = np.asarray(evaluation_adata.obsm["sparniche"])
        labels = evaluation_adata.obs[label_key].astype(str).to_numpy()
        predicted = _predict_clusters(embedding, evaluation_adata, config)
        audit = evaluation_adata.uns.get("sparniche_leiden_search", {})
        row: dict[str, Any] = {
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
            {
                "spot_id": np.asarray(evaluation_adata.obs_names),
                "ground_truth": labels,
                "pred": predicted,
            }
        ).to_csv(run_dir / "predictions.csv", index=False)
        del evaluation_adata, artifacts, adata
    except Exception as error:
        row = {
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
    *,
    stage: int,
    specs: list[dict[str, Any]],
    seeds: tuple[int, ...],
    args: argparse.Namespace,
    base: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for seed in seeds:
            print(
                f"[RUN ] stage={stage} {spec['config_id']} seed={seed}", flush=True
            )
            row = _run_one(
                base=base,
                input_path=args.input_h5ad,
                output_root=args.output_root,
                spec=spec,
                seed=int(seed),
                device=args.device,
                label_key=args.label_key,
                force=args.force,
            )
            row = {"stage": int(stage), **row}
            rows.append(row)
            if row["status"] == "completed":
                print(
                    f"[DONE] ARI={row['ari']:.4f} NMI={row['nmi']:.4f} "
                    f"FMI={row['fmi']:.4f} Acc={row['accuracy']:.4f} "
                    f"MacroF1={row['macro_f1']:.4f}",
                    flush=True,
                )
            else:
                print(f"[FAIL] {row['error']}", flush=True)
                if not args.continue_on_error:
                    return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _stage1_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    completed = frame[frame["status"].astype(str).eq("completed")]
    for factor, levels in FACTOR_LEVELS.items():
        for level in levels:
            mask = pd.to_numeric(completed[factor], errors="coerce").eq(level)
            for other in FACTOR_LEVELS:
                if other != factor:
                    mask &= pd.to_numeric(completed[other], errors="coerce").eq(
                        ANCHOR[other]
                    )
            selected = completed.loc[mask]
            if selected.empty:
                continue
            row = {"factor": factor, "level": int(level)}
            for metric in QUALITY_METRICS:
                row[metric] = float(pd.to_numeric(selected[metric]).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _stage3_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame[frame["status"].astype(str).eq("completed")].copy()
    records = []
    for config_name, group in completed.groupby("config_id", sort=False):
        first = group.iloc[0]
        row = {
            "config_id": config_name,
            **{key: int(first[key]) for key in FACTOR_LEVELS},
            "n_completed": int(len(group)),
        }
        for metric in QUALITY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            row[f"{metric}_min"] = float(values.min())
        records.append(row)
    ranking = pd.DataFrame(records)
    if not ranking.empty:
        ranking = ranking.sort_values(
            ["ari_mean", "ari_min", "ari_sd", "nmi_mean", "macro_f1_mean"],
            ascending=[False, False, True, False, False],
        )
    return ranking


def _all_cached_runs(output_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((output_root / "runs").glob("*/seed*/result.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def _require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required previous-stage output is missing: {path}")
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml"
    )
    parser.add_argument("--screen-seed", type=int, default=1234)
    parser.add_argument(
        "--validation-seeds", nargs="+", type=int,
        default=list(DEFAULT_VALIDATION_SEEDS),
    )
    parser.add_argument("--stage", choices=("all", "1", "2", "3"), default="all")
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
    base = load_yaml(args.config)
    _write_json(
        args.output_root / "manifest.json",
        {
            "input_h5ad": str(args.input_h5ad),
            "base_config": str(args.config),
            "factors": FACTOR_LEVELS,
            "anchor": ANCHOR,
            "screen_seed": int(args.screen_seed),
            "validation_seeds": [int(seed) for seed in args.validation_seeds],
            "leiden_window": "K..K+2",
            "primary_metric": "ARI",
        },
    )

    run_stage1 = args.stage in {"all", "1"}
    run_stage2 = args.stage in {"all", "2"}
    run_stage3 = args.stage in {"all", "3"}

    if run_stage1:
        stage1 = _run_specs(
            stage=1, specs=build_stage1_specs(), seeds=(args.screen_seed,),
            args=args, base=base,
        )
        stage1.to_csv(args.output_root / "stage1_runs.csv", index=False)
        _stage1_summary(stage1).to_csv(
            args.output_root / "stage1_main_effects.csv", index=False
        )
    else:
        stage1 = _require_csv(args.output_root / "stage1_runs.csv")

    if run_stage2:
        selected_levels = select_best_levels(stage1)
        _write_json(args.output_root / "stage2_selected_levels.json", selected_levels)
        stage2 = _run_specs(
            stage=2,
            specs=build_stage2_specs(selected_levels),
            seeds=(args.screen_seed,), args=args, base=base,
        )
        stage2.to_csv(args.output_root / "stage2_interactions.csv", index=False)
    elif run_stage3:
        stage2 = _require_csv(args.output_root / "stage2_interactions.csv")
    else:
        stage2 = pd.DataFrame()

    if run_stage3:
        candidates = select_stage3_specs(stage2, count=4)
        stage3 = _run_specs(
            stage=3, specs=candidates,
            seeds=tuple(dict.fromkeys(int(seed) for seed in args.validation_seeds)),
            args=args, base=base,
        )
        stage3.to_csv(args.output_root / "stage3_seed_validation.csv", index=False)
        ranking = _stage3_ranking(stage3)
        ranking.to_csv(args.output_root / "final_ranking.csv", index=False)
        if ranking.empty or int(ranking.iloc[0]["n_completed"]) < len(
            set(args.validation_seeds)
        ):
            raise RuntimeError("no fully validated best configuration is available")
        best = {
            key: int(ranking.iloc[0][key]) for key in FACTOR_LEVELS
        }
        save_yaml(
            build_config(
                base,
                input_path=args.input_h5ad,
                seed=int(args.validation_seeds[0]),
                device=args.device,
                factors=best,
            ),
            args.output_root / "best_config.yaml",
        )

    all_runs = _all_cached_runs(args.output_root)
    all_runs.to_csv(args.output_root / "all_runs.csv", index=False)
    failures = int(all_runs["status"].astype(str).ne("completed").sum()) if not all_runs.empty else 0
    print(
        f"[SUMMARY] cached_runs={len(all_runs)} failures={failures} "
        f"output={args.output_root}",
        flush=True,
    )
    return 1 if failures and not args.continue_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
