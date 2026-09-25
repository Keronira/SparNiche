#!/usr/bin/env python3
"""Validate retained source24 layer-recovery settings and compare baselines."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import anndata
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiments import infer_input_kind  # noqa: E402
from scripts.run_final_embedding_experiment import _valid_mask  # noqa: E402
from src.config import (  # noqa: E402
    apply_overrides,
    deep_merge,
    load_yaml,
    normalize_sparniche_config,
)
from src.final_metrics import compute_final_metrics, compute_layer_recovery_metrics  # noqa: E402
from src.pipeline import train_adata  # noqa: E402
from src.runner import _predict_clusters  # noqa: E402


STEP2_SAMPLES = ("151507", "151510", "151669", "151670", "151673", "151676")
STEP2_SEEDS = (1234, 1235, 1236)
STEP2_SET_OVERRIDES: dict[str, dict[str, Any]] = {
    "lr_0025": {"model.sparniche.lr": 0.0025},
    "latent_32_norm_on": {
        "model.latent_dim": 32,
        "model.local_graph_normalize": True,
    },
    "latent_32_norm_off": {
        "model.latent_dim": 32,
        "model.local_graph_normalize": False,
    },
}
BASELINE_METHODS = (
    "BANKSY",
    "CellCharter",
    "DR-SC",
    "GraphST",
    "HERGAST",
    "STAGATE",
    "SpaGCN",
    "SpaceFlow",
    "scNiche",
)
PRIMARY_METRICS = (
    "ari",
    "nmi",
    "macro_layer_iou",
    "worst_layer_iou",
    "layer_recovery_rate",
)


def build_step2_config(
    base: dict[str, Any],
    input_path: Path,
    *,
    set_name: str,
    seed: int,
    device: str,
    input_kind: str,
) -> dict[str, Any]:
    if set_name == "anchor":
        set_overrides: dict[str, Any] = {}
    elif set_name in STEP2_SET_OVERRIDES:
        set_overrides = STEP2_SET_OVERRIDES[set_name]
    else:
        raise ValueError(f"unknown step2 set {set_name!r}")
    overrides = {
        "paths.input_h5ad": str(input_path),
        "data.preprocessing.input_kind": str(input_kind),
        "model.variant": "local_graph_normalized",
        "model.local_graph_mode": "normalized",
        "model.sparniche_view1.attention_mode": "spatial_local",
        "model.sparniche_view1.attention_chunk_size": 4096,
        "training.seed": int(seed),
        "training.device": str(device),
        "benchmark.external_only": False,
        "evaluation.sparniche_leiden_seed": int(seed),
        "evaluation.leiden_cluster_lower_offset": 0,
        "evaluation.leiden_cluster_upper_offset": 2,
    }
    overrides.update(set_overrides)
    return normalize_sparniche_config(apply_overrides(base, overrides))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _prediction_metrics(
    ground_truth: Iterable[object], predicted: Iterable[object]
) -> tuple[dict[str, float], dict[str, float]]:
    truth = pd.Series(list(ground_truth), dtype="string")
    pred = pd.Series(list(predicted), dtype="string")
    valid = truth.notna() & pred.notna()
    valid &= ~truth.str.strip().str.lower().isin({"", "nan", "na", "none", "unknown"})
    truth_values = truth[valid].astype(str).to_numpy()
    pred_values = pred[valid].astype(str).to_numpy()
    if truth_values.size == 0:
        raise ValueError("prediction table has no labeled rows")
    recovery = compute_layer_recovery_metrics(truth_values, pred_values)
    metrics = {
        "ari": float(adjusted_rand_score(truth_values, pred_values)),
        "nmi": float(normalized_mutual_info_score(truth_values, pred_values)),
        "macro_layer_iou": float(recovery["macro_layer_iou"]),
        "worst_layer_iou": float(recovery["worst_layer_iou"]),
        "layer_recovery_rate": float(recovery["layer_recovery_rate"]),
    }
    return metrics, dict(recovery["per_layer_iou"])


def load_reference_runs(
    bench_root: Path,
    *,
    samples: tuple[str, ...] = STEP2_SAMPLES,
    reps: tuple[int, ...] = (1, 2, 3),
    baseline_methods: tuple[str, ...] = BASELINE_METHODS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    references = [
        (method, "baseline", f"{method}__rep{{rep}}") for method in baseline_methods
    ]
    for model, model_type, directory_template in references:
        for rep in reps:
            seed = 1233 + int(rep)
            for sample in samples:
                path = bench_root / directory_template.format(rep=rep) / "results" / f"{sample}.csv"
                if not path.exists():
                    failure_rows.append(
                        {
                            "model": model,
                            "model_type": model_type,
                            "sample": sample,
                            "seed": seed,
                            "source": "stored_prediction",
                            "error": f"missing prediction file: {path}",
                        }
                    )
                    continue
                try:
                    table = pd.read_csv(path)
                    missing_columns = {"ground_truth", "pred"} - set(table.columns)
                    if missing_columns:
                        raise ValueError(
                            "missing columns: " + ", ".join(sorted(missing_columns))
                        )
                    metrics, per_layer = _prediction_metrics(
                        table["ground_truth"], table["pred"]
                    )
                    run_rows.append(
                        {
                            "model": model,
                            "model_type": model_type,
                            "sample": sample,
                            "seed": seed,
                            "status": "completed",
                            "source": "stored_prediction",
                            **metrics,
                        }
                    )
                    for layer, iou in per_layer.items():
                        layer_rows.append(
                            {
                                "model": model,
                                "model_type": model_type,
                                "sample": sample,
                                "seed": seed,
                                "layer": layer,
                                "iou": float(iou),
                            }
                        )
                except Exception as error:
                    failure_rows.append(
                        {
                            "model": model,
                            "model_type": model_type,
                            "sample": sample,
                            "seed": seed,
                            "source": "stored_prediction",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
    return pd.DataFrame(run_rows), pd.DataFrame(layer_rows), pd.DataFrame(failure_rows)


def load_reclustered_anchor_runs(
    bench_root: Path,
    base: dict[str, Any],
    *,
    samples: tuple[str, ...] = STEP2_SAMPLES,
    reps: tuple[int, ...] = (1, 2, 3),
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for rep in reps:
        seed = 1233 + int(rep)
        artifact_root = bench_root / f"SparNiche_rep{rep}" / "artifacts"
        for sample in samples:
            matches = sorted(
                artifact_root.glob(f"{sample}--*seed{seed}/trained.h5ad")
            )
            if len(matches) != 1:
                failure_rows.append(
                    {
                        "model": "anchor",
                        "model_type": "anchor",
                        "sample": sample,
                        "seed": seed,
                        "source": "reclustered_anchor_embedding",
                        "error": (
                            "expected one trained anchor artifact, found "
                            f"{len(matches)} under {artifact_root}"
                        ),
                    }
                )
                continue
            path = matches[0]
            try:
                adata = anndata.read_h5ad(path)
                config = build_step2_config(
                    base,
                    path,
                    set_name="anchor",
                    seed=seed,
                    device=device,
                    input_kind="raw_counts",
                )
                metrics, per_layer, _, _ = _evaluate_candidate(adata, config, seed)
                run_rows.append(
                    {
                        "model": "anchor",
                        "model_type": "anchor",
                        "sample": sample,
                        "seed": seed,
                        "status": "completed",
                        "source": "reclustered_anchor_embedding",
                        **metrics,
                    }
                )
                for layer, iou in per_layer.items():
                    layer_rows.append(
                        {
                            "model": "anchor",
                            "model_type": "anchor",
                            "sample": sample,
                            "seed": seed,
                            "layer": layer,
                            "iou": float(iou),
                        }
                    )
            except Exception as error:
                failure_rows.append(
                    {
                        "model": "anchor",
                        "model_type": "anchor",
                        "sample": sample,
                        "seed": seed,
                        "source": "reclustered_anchor_embedding",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    return pd.DataFrame(run_rows), pd.DataFrame(layer_rows), pd.DataFrame(failure_rows)


def _long_summary(
    table: pd.DataFrame,
    group_columns: list[str],
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if table.empty:
        return pd.DataFrame(rows)
    for keys, group in table.groupby(group_columns, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        prefix = dict(zip(group_columns, key_values))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    **prefix,
                    "metric": metric,
                    "mean": float(values.mean()) if not values.empty else float("nan"),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def _layer46_summary(layers: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    columns = [
        *group_columns,
        "layer",
        "mean_iou",
        "sd_iou",
        "n",
        "zero_rate",
        "recovery_rate_iou_ge_0_5",
    ]
    if layers.empty:
        return pd.DataFrame(columns=columns)
    selected = layers[layers["layer"].isin(("Layer4", "Layer6"))].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby([*group_columns, "layer"], sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        prefix = dict(zip([*group_columns, "layer"], key_values))
        values = pd.to_numeric(group["iou"], errors="coerce").dropna()
        rows.append(
            {
                **prefix,
                "mean_iou": float(values.mean()) if not values.empty else float("nan"),
                "sd_iou": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n": int(values.size),
                "zero_rate": float((values <= 1e-12).mean()) if not values.empty else float("nan"),
                "recovery_rate_iou_ge_0_5": float((values >= 0.5).mean())
                if not values.empty
                else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _paired_deltas(metrics: pd.DataFrame, layers: pd.DataFrame) -> pd.DataFrame:
    completed = metrics[metrics["status"] == "completed"].copy()
    wide_layers = (
        layers[layers["layer"].isin(("Layer4", "Layer6"))]
        .pivot_table(
            index=["model", "model_type", "sample", "seed"],
            columns="layer",
            values="iou",
            aggfunc="first",
        )
        .rename(columns={"Layer4": "layer4_iou", "Layer6": "layer6_iou"})
        .reset_index()
    )
    completed = completed.merge(
        wide_layers,
        on=["model", "model_type", "sample", "seed"],
        how="left",
    )
    compared_metrics = (*PRIMARY_METRICS, "layer4_iou", "layer6_iou")
    anchor = completed[completed["model"] == "anchor"].set_index(["sample", "seed"])
    rows: list[dict[str, Any]] = []
    candidates = completed[completed["model_type"] == "sparniche_candidate"]
    for _, row in candidates.iterrows():
        key = (row["sample"], row["seed"])
        if key not in anchor.index:
            continue
        anchor_row = anchor.loc[key]
        for metric in compared_metrics:
            if pd.isna(row.get(metric)) or pd.isna(anchor_row.get(metric)):
                continue
            rows.append(
                {
                    "model": row["model"],
                    "sample": row["sample"],
                    "seed": row["seed"],
                    "metric": metric,
                    "anchor": float(anchor_row[metric]),
                    "candidate": float(row[metric]),
                    "delta_candidate_minus_anchor": float(
                        row[metric] - anchor_row[metric]
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_step2_tables(
    metrics: pd.DataFrame,
    layers: pd.DataFrame,
    failures: pd.DataFrame,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_root / "all_run_metrics.csv", index=False)
    layers.to_csv(output_root / "per_layer_iou.csv", index=False)
    failures.to_csv(output_root / "failures.csv", index=False)
    completed = metrics[metrics["status"] == "completed"].copy()
    _long_summary(
        completed, ["model", "model_type"], PRIMARY_METRICS
    ).to_csv(output_root / "summary_overall.csv", index=False)
    _long_summary(
        completed, ["model", "model_type", "sample"], PRIMARY_METRICS
    ).to_csv(output_root / "summary_by_sample.csv", index=False)
    _layer46_summary(layers, ["model", "model_type"]).to_csv(
        output_root / "layer46_summary.csv", index=False
    )
    _layer46_summary(layers, ["model", "model_type", "sample"]).to_csv(
        output_root / "layer46_by_sample.csv", index=False
    )
    _paired_deltas(metrics, layers).to_csv(
        output_root / "paired_deltas_vs_anchor.csv", index=False
    )


def _evaluate_candidate(
    adata: anndata.AnnData, config: dict[str, Any], seed: int
) -> tuple[dict[str, float], dict[str, float], dict[str, Any], pd.DataFrame]:
    label_key = str(config["data"]["label_key"])
    mask = _valid_mask(adata, label_key)
    evaluation_adata = adata[mask].copy()
    embedding = np.asarray(adata.obsm["sparniche"])[mask]
    labels = adata.obs[label_key].astype(str).to_numpy()[mask]
    spatial = np.asarray(adata.obsm["spatial"])[mask]
    predicted = _predict_clusters(embedding, evaluation_adata, config)
    all_metrics = compute_final_metrics(
        embedding,
        labels,
        spatial,
        predicted=predicted,
        seed=int(seed),
        n_neighbors=int(config["data"]["n_neighbors"]),
    )
    recovery = compute_layer_recovery_metrics(labels, predicted)
    metrics = {metric: float(all_metrics[metric]) for metric in PRIMARY_METRICS}
    audit = dict(evaluation_adata.uns.get("sparniche_leiden_search", {}))
    metrics["leiden_window_fallback"] = bool(audit.get("window_fallback", False))
    metrics["selected_resolution"] = float(audit.get("selected_resolution", float("nan")))
    metrics["selected_clusters"] = float(audit.get("selected_clusters", float("nan")))
    predictions = pd.DataFrame(
        {
            "spot_id": adata.obs_names.to_numpy()[mask],
            "ground_truth": labels,
            "pred": np.asarray(predicted).astype(str),
        }
    )
    return metrics, dict(recovery["per_layer_iou"]), audit, predictions


def _select_samples(source: Path) -> list[Path]:
    available = {path.stem: path for path in source.glob("*.h5ad")}
    missing = [sample for sample in STEP2_SAMPLES if sample not in available]
    if missing:
        raise FileNotFoundError(f"missing step2 source24 samples: {', '.join(missing)}")
    return [available[sample] for sample in STEP2_SAMPLES]


def _candidate_frames(
    bundles: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for bundle in bundles:
        run = dict(bundle["run"])
        if run["status"] == "completed":
            runs.append(run)
            for layer, iou in bundle.get("per_layer_iou", {}).items():
                layers.append(
                    {
                        "model": run["model"],
                        "model_type": run["model_type"],
                        "sample": run["sample"],
                        "seed": run["seed"],
                        "layer": layer,
                        "iou": float(iou),
                    }
                )
        else:
            failures.append(
                {
                    "model": run["model"],
                    "model_type": run["model_type"],
                    "sample": run["sample"],
                    "seed": run["seed"],
                    "source": "step2_training",
                    "error": run.get("error", "unknown failure"),
                }
            )
    return pd.DataFrame(runs), pd.DataFrame(layers), pd.DataFrame(failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/root/autodl-fs/data/source24"))
    parser.add_argument(
        "--bench-root",
        type=Path,
        default=Path("/root/autodl-fs/bench_results/source24"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/root/autodl-fs/bench_results/source24/"
            "SparNiche_layer_sensitivity_step2_tmp"
        ),
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    parser.add_argument(
        "--source-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "source24.yaml",
    )
    parser.add_argument("--sets", nargs="+", choices=tuple(STEP2_SET_OVERRIDES), default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    bench_root = Path(args.bench_root)
    output_root = Path(args.output_root)
    samples = _select_samples(source)
    base = deep_merge(load_yaml(args.config), load_yaml(args.source_config))
    set_names = list(args.sets) if args.sets else list(STEP2_SET_OVERRIDES)
    _write_json(
        output_root / "manifest.json",
        {
            "temporary": True,
            "stage": "step2_multiseed_validation",
            "source": str(source),
            "bench_root": str(bench_root),
            "samples": list(STEP2_SAMPLES),
            "seeds": list(STEP2_SEEDS),
            "trained_sets": {name: STEP2_SET_OVERRIDES[name] for name in set_names},
            "anchor": (
                "recluster trained SparNiche_rep1/2/3 embeddings with the Step2 "
                "K..K+2 layer-recovery selection"
            ),
            "baseline_methods": list(BASELINE_METHODS),
            "leiden_cluster_window": "K..K+2",
            "layer_recovery_iou_threshold": 0.5,
        },
    )

    anchor_metrics, anchor_layers, anchor_failures = load_reclustered_anchor_runs(
        bench_root, base, device=args.device
    )
    baseline_metrics, baseline_layers, baseline_failures = load_reference_runs(
        bench_root
    )
    reference_metrics = pd.concat(
        [anchor_metrics, baseline_metrics], ignore_index=True
    )
    reference_layers = pd.concat([anchor_layers, baseline_layers], ignore_index=True)
    reference_failures = pd.concat(
        [anchor_failures, baseline_failures], ignore_index=True
    )
    bundles: list[dict[str, Any]] = []
    failure_count = 0
    for set_name in set_names:
        for seed in STEP2_SEEDS:
            for sample_path in samples:
                run_dir = output_root / "runs" / set_name / sample_path.stem / f"seed{seed}"
                result_path = run_dir / "step2_result.json"
                if result_path.exists() and not args.force:
                    bundle = json.loads(result_path.read_text(encoding="utf-8"))
                    bundles.append(bundle)
                    state = bundle.get("run", {}).get("status", "unknown")
                    print(
                        f"[SKIP] set={set_name} sample={sample_path.stem} seed={seed} status={state}",
                        flush=True,
                    )
                    continue
                started = time.perf_counter()
                print(
                    f"[RUN ] set={set_name} sample={sample_path.stem} seed={seed}",
                    flush=True,
                )
                try:
                    input_kind = infer_input_kind(sample_path)
                    config = build_step2_config(
                        base,
                        sample_path,
                        set_name=set_name,
                        seed=seed,
                        device=args.device,
                        input_kind=input_kind,
                    )
                    adata = anndata.read_h5ad(sample_path)
                    artifacts = train_adata(adata, config, run_dir, resume=False)
                    metrics, per_layer, audit, predictions = _evaluate_candidate(
                        artifacts.adata, config, seed
                    )
                    bundle = {
                        "run": {
                            "model": set_name,
                            "model_type": "sparniche_candidate",
                            "sample": sample_path.stem,
                            "seed": seed,
                            "status": "completed",
                            "source": "step2_training",
                            "runtime_seconds": float(time.perf_counter() - started),
                            "input_kind": input_kind,
                            **metrics,
                        },
                        "per_layer_iou": per_layer,
                        "leiden_audit": audit,
                    }
                    run_dir.mkdir(parents=True, exist_ok=True)
                    predictions.to_csv(run_dir / "prediction.csv", index=False)
                    np.save(run_dir / "embedding.npy", np.asarray(artifacts.adata.obsm["sparniche"]))
                    if not args.keep_artifacts:
                        for name in ("trained.h5ad", "checkpoint.pt"):
                            path = run_dir / name
                            if path.exists():
                                path.unlink()
                    print(
                        f"[DONE] set={set_name} sample={sample_path.stem} seed={seed} "
                        f"ARI={metrics['ari']:.4f} NMI={metrics['nmi']:.4f} "
                        f"worstIoU={metrics['worst_layer_iou']:.4f} "
                        f"LRR={metrics['layer_recovery_rate']:.4f}",
                        flush=True,
                    )
                except Exception as error:
                    failure_count += 1
                    bundle = {
                        "run": {
                            "model": set_name,
                            "model_type": "sparniche_candidate",
                            "sample": sample_path.stem,
                            "seed": seed,
                            "status": "failed",
                            "source": "step2_training",
                            "runtime_seconds": float(time.perf_counter() - started),
                            "error": f"{type(error).__name__}: {error}",
                        },
                        "per_layer_iou": {},
                        "leiden_audit": {},
                    }
                    print(
                        f"[FAIL] set={set_name} sample={sample_path.stem} seed={seed}: {error}",
                        flush=True,
                    )
                _write_json(result_path, bundle)
                bundles.append(bundle)
                candidate_metrics, candidate_layers, candidate_failures = _candidate_frames(
                    bundles
                )
                write_step2_tables(
                    pd.concat([reference_metrics, candidate_metrics], ignore_index=True),
                    pd.concat([reference_layers, candidate_layers], ignore_index=True),
                    pd.concat([reference_failures, candidate_failures], ignore_index=True),
                    output_root,
                )
                if failure_count and args.fail_fast:
                    return 1
    print(
        f"[SUMMARY] completed={len(bundles) - failure_count}/{len(bundles)} "
        f"reference_failures={len(reference_failures)} output={output_root}",
        flush=True,
    )
    return 0 if failure_count == 0 and reference_failures.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
