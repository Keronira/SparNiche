#!/usr/bin/env python3
"""Tune DEC structure, optimizer LR, and loss weights on source26 sample 1."""

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
DEFAULT_OUTPUT = Path("/root/autodl-fs/bench_results/_tmp_source26_loss_search")
SCREEN_SEEDS = (1234, 1235)
VALIDATION_SEEDS = (1234, 1235, 1236)
ARI_TOLERANCE = 0.01

FIXED_EMBEDDING = {"hvg": 3000, "pca": 64, "neighbors": 8, "latent": 8}
DEC_CLUSTER_LEVELS = (10, 12, 14)
DEC_KL_LEVELS = (0.0, 0.25, 0.5, 1.0)
LOSS_LEVELS = {
    "lr": (0.005, 0.01, 0.02),
    "gcn_w": (0.025, 0.1, 0.4),
    "rec_w": (2.5, 5.0, 10.0),
}
LOSS_ANCHOR = {"lr": 0.01, "gcn_w": 0.1, "rec_w": 10.0}
FACTOR_COLUMNS = ("dec_cluster_n", "dec_kl_w", "lr", "gcn_w", "rec_w")
QUALITY_METRICS = ("ari", "nmi", "fmi", "accuracy", "macro_f1")
UNKNOWN_LABELS = {
    str(value).strip().lower() for value in AMBIGUOUS_GROUND_TRUTH_LABELS
}


def _number_token(value: int | float) -> str:
    return format(float(value), ".8g").replace("-", "m").replace(".", "p")


def config_id(factors: dict[str, int | float]) -> str:
    if float(factors["dec_kl_w"]) == 0.0:
        dec = "decoff"
    else:
        dec = (
            f"dec{int(factors['dec_cluster_n'])}"
            f"kl{_number_token(factors['dec_kl_w'])}"
        )
    return (
        f"{dec}_lr{_number_token(factors['lr'])}_"
        f"gcn{_number_token(factors['gcn_w'])}_"
        f"rec{_number_token(factors['rec_w'])}"
    )


def _spec(factors: dict[str, int | float]) -> dict[str, Any]:
    values = {
        "dec_cluster_n": int(factors["dec_cluster_n"]),
        "dec_kl_w": float(factors["dec_kl_w"]),
        "lr": float(factors["lr"]),
        "gcn_w": float(factors["gcn_w"]),
        "rec_w": float(factors["rec_w"]),
    }
    return {"config_id": config_id(values), "factors": values}


def build_stage1a_specs() -> list[dict[str, Any]]:
    specs = [_spec({"dec_cluster_n": 10, "dec_kl_w": 0.0, **LOSS_ANCHOR})]
    for dec_kl_w, dec_cluster_n in itertools.product(
        DEC_KL_LEVELS[1:], DEC_CLUSTER_LEVELS
    ):
        specs.append(
            _spec(
                {
                    "dec_cluster_n": dec_cluster_n,
                    "dec_kl_w": dec_kl_w,
                    **LOSS_ANCHOR,
                }
            )
        )
    return specs


def build_stage1b_specs(
    dec_structure: dict[str, int | float],
) -> list[dict[str, Any]]:
    anchor = {**dec_structure, **LOSS_ANCHOR}
    specs = [_spec(anchor)]
    for factor, levels in LOSS_LEVELS.items():
        for level in levels:
            if float(level) == float(LOSS_ANCHOR[factor]):
                continue
            values = dict(anchor)
            values[factor] = float(level)
            specs.append(_spec(values))
    return specs


def build_stage2_specs(
    dec_structures: list[dict[str, int | float]],
    selected_levels: dict[str, list[float]],
) -> list[dict[str, Any]]:
    specs = []
    for dec_structure in dec_structures:
        for lr, gcn_w, rec_w in itertools.product(
            selected_levels["lr"],
            selected_levels["gcn_w"],
            selected_levels["rec_w"],
        ):
            specs.append(
                _spec(
                    {
                        **dec_structure,
                        "lr": lr,
                        "gcn_w": gcn_w,
                        "rec_w": rec_w,
                    }
                )
            )
    return specs


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
        "data.preprocessing.n_top_genes": FIXED_EMBEDDING["hvg"],
        "data.preprocessing.pca_n_components": FIXED_EMBEDDING["pca"],
        "data.n_neighbors": FIXED_EMBEDDING["neighbors"],
        "model.sparniche_view1.attention_neighbors": FIXED_EMBEDDING["neighbors"],
        "model.sparniche_view1.dec_cluster_n": int(factors["dec_cluster_n"]),
        "model.latent_dim": FIXED_EMBEDDING["latent"],
        "model.sparniche.lr": float(factors["lr"]),
        "model.sparniche.dec_kl_w": float(factors["dec_kl_w"]),
        "model.sparniche.gcn_w": float(factors["gcn_w"]),
        "model.sparniche.rec_w": float(factors["rec_w"]),
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
            **FIXED_EMBEDDING,
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
            **FIXED_EMBEDDING,
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
    stage: str,
    specs: list[dict[str, Any]],
    seeds: tuple[int, ...],
    args: argparse.Namespace,
    base: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for seed in seeds:
            print(f"[RUN ] stage={stage} {spec['config_id']} seed={seed}", flush=True)
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
            row = {"stage": stage, **row}
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


def summarize_and_rank(
    frame: pd.DataFrame,
    *,
    ari_tolerance: float = ARI_TOLERANCE,
    expected_seeds: int | None = None,
) -> pd.DataFrame:
    completed = frame[frame["status"].astype(str).eq("completed")].copy()
    records: list[dict[str, Any]] = []
    for config_name, group in completed.groupby("config_id", sort=False):
        first = group.iloc[0]
        row: dict[str, Any] = {
            "config_id": str(config_name),
            "n_completed": int(group["seed"].nunique()),
        }
        for key in FACTOR_COLUMNS:
            if key in first:
                row[key] = int(first[key]) if key == "dec_cluster_n" else float(first[key])
        for metric in QUALITY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min())
        records.append(row)

    summary = pd.DataFrame(records)
    if expected_seeds is not None and not summary.empty:
        summary = summary[summary["n_completed"].eq(int(expected_seeds))].copy()
    if summary.empty:
        return summary

    remaining = summary.copy()
    ranked_parts = []
    while not remaining.empty:
        best_mean = float(remaining["ari_mean"].max())
        band_mask = remaining["ari_mean"].ge(best_mean - float(ari_tolerance) - 1e-12)
        band = remaining.loc[band_mask].sort_values(
            ["ari_min", "ari_sd", "macro_f1_mean", "nmi_mean", "config_id"],
            ascending=[False, True, False, False, True],
        )
        ranked_parts.append(band)
        remaining = remaining.loc[~band_mask]
    ranking = pd.concat(ranked_parts, ignore_index=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def select_best_structures(
    ranking: pd.DataFrame, count: int = 2
) -> list[dict[str, int | float]]:
    ordered = ranking.sort_values("rank") if "rank" in ranking else ranking
    if len(ordered) < int(count):
        raise ValueError(f"only {len(ordered)} complete DEC structures are available")
    return [
        {
            "dec_cluster_n": int(row["dec_cluster_n"]),
            "dec_kl_w": float(row["dec_kl_w"]),
        }
        for _, row in ordered.head(int(count)).iterrows()
    ]


def select_best_levels(ranking: pd.DataFrame) -> dict[str, list[float]]:
    ordered = ranking.sort_values("rank") if "rank" in ranking else ranking
    selected: dict[str, list[float]] = {}
    for factor in LOSS_LEVELS:
        candidates = ordered.copy()
        for other, anchor_value in LOSS_ANCHOR.items():
            if other != factor:
                candidates = candidates[
                    np.isclose(pd.to_numeric(candidates[other]), float(anchor_value))
                ]
        levels: list[float] = []
        for value in pd.to_numeric(candidates[factor], errors="coerce"):
            value = float(value)
            if value not in levels:
                levels.append(value)
            if len(levels) == 2:
                break
        if len(levels) != 2:
            raise ValueError(f"stage 1B does not contain two ranked levels for {factor}")
        selected[factor] = levels
    return selected


def _specs_from_ranking(ranking: pd.DataFrame, count: int) -> list[dict[str, Any]]:
    ordered = ranking.sort_values("rank") if "rank" in ranking else ranking
    if len(ordered) < int(count):
        raise ValueError(f"only {len(ordered)} complete configurations are available")
    return [
        _spec({key: row[key] for key in FACTOR_COLUMNS})
        for _, row in ordered.head(int(count)).iterrows()
    ]


def _require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required previous-stage output is missing: {path}")
    return pd.read_csv(path)


def _all_cached_runs(output_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((output_root / "runs").glob("*/seed*/result.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def _write_all_runs(output_root: Path) -> int:
    all_runs = _all_cached_runs(output_root)
    all_runs.to_csv(output_root / "all_runs.csv", index=False)
    if all_runs.empty:
        return 0
    return int(all_runs["status"].astype(str).ne("completed").sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml"
    )
    parser.add_argument("--stage", choices=("all", "1a", "1b", "2", "3"), default="all")
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
            "fixed_embedding": FIXED_EMBEDDING,
            "dec_cluster_levels": DEC_CLUSTER_LEVELS,
            "dec_kl_levels": DEC_KL_LEVELS,
            "loss_levels": LOSS_LEVELS,
            "screen_seeds": SCREEN_SEEDS,
            "validation_seeds": VALIDATION_SEEDS,
            "ari_tolerance": ARI_TOLERANCE,
            "leiden_window": "K..K+2",
        },
    )

    if args.stage in {"all", "1a"}:
        stage1a = _run_specs(
            stage="1a",
            specs=build_stage1a_specs(),
            seeds=SCREEN_SEEDS,
            args=args,
            base=base,
        )
        stage1a.to_csv(args.output_root / "stage1a_runs.csv", index=False)
        stage1a_ranking = summarize_and_rank(
            stage1a, expected_seeds=len(SCREEN_SEEDS)
        )
        stage1a_ranking.to_csv(args.output_root / "stage1a_ranking.csv", index=False)
    else:
        stage1a_ranking = _require_csv(args.output_root / "stage1a_ranking.csv")

    if args.stage == "1a":
        failures = _write_all_runs(args.output_root)
        return 1 if failures and not args.continue_on_error else 0

    best_structures = select_best_structures(stage1a_ranking, count=2)
    _write_json(args.output_root / "stage1a_selected_structures.json", best_structures)

    if args.stage in {"all", "1b"}:
        stage1b = _run_specs(
            stage="1b",
            specs=build_stage1b_specs(best_structures[0]),
            seeds=SCREEN_SEEDS,
            args=args,
            base=base,
        )
        stage1b.to_csv(args.output_root / "stage1b_runs.csv", index=False)
        stage1b_ranking = summarize_and_rank(
            stage1b, expected_seeds=len(SCREEN_SEEDS)
        )
        stage1b_ranking.to_csv(args.output_root / "stage1b_ranking.csv", index=False)
    else:
        stage1b_ranking = _require_csv(args.output_root / "stage1b_ranking.csv")

    if args.stage == "1b":
        failures = _write_all_runs(args.output_root)
        return 1 if failures and not args.continue_on_error else 0

    selected_levels = select_best_levels(stage1b_ranking)
    _write_json(args.output_root / "stage1b_selected_levels.json", selected_levels)

    if args.stage in {"all", "2"}:
        stage2 = _run_specs(
            stage="2",
            specs=build_stage2_specs(best_structures, selected_levels),
            seeds=SCREEN_SEEDS,
            args=args,
            base=base,
        )
        stage2.to_csv(args.output_root / "stage2_runs.csv", index=False)
        stage2_ranking = summarize_and_rank(
            stage2, expected_seeds=len(SCREEN_SEEDS)
        )
        stage2_ranking.to_csv(
            args.output_root / "stage2_interactions.csv", index=False
        )
    else:
        stage2_ranking = _require_csv(args.output_root / "stage2_interactions.csv")

    if args.stage == "2":
        failures = _write_all_runs(args.output_root)
        return 1 if failures and not args.continue_on_error else 0

    stage3_specs = _specs_from_ranking(stage2_ranking, count=4)
    stage3 = _run_specs(
        stage="3",
        specs=stage3_specs,
        seeds=VALIDATION_SEEDS,
        args=args,
        base=base,
    )
    stage3.to_csv(args.output_root / "stage3_seed_validation.csv", index=False)
    final_ranking = summarize_and_rank(
        stage3, expected_seeds=len(VALIDATION_SEEDS)
    )
    final_ranking.to_csv(args.output_root / "final_ranking.csv", index=False)
    if final_ranking.empty:
        raise RuntimeError("no fully validated best configuration is available")

    best_factors = {
        key: final_ranking.iloc[0][key] for key in FACTOR_COLUMNS
    }
    save_yaml(
        build_config(
            base,
            input_path=args.input_h5ad,
            seed=VALIDATION_SEEDS[0],
            device=args.device,
            factors=best_factors,
        ),
        args.output_root / "best_config.yaml",
    )
    failures = _write_all_runs(args.output_root)
    print(
        f"[SUMMARY] failures={failures} output={args.output_root}",
        flush=True,
    )
    return 1 if failures and not args.continue_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
