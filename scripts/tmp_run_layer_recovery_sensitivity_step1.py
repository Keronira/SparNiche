#!/usr/bin/env python3
"""Run step 1 of the temporary source24 layer-recovery sensitivity screen."""

from __future__ import annotations

import argparse
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
from src.final_metrics import (  # noqa: E402
    compute_final_metrics,
    compute_layer_recovery_metrics,
)
from src.pipeline import train_adata  # noqa: E402
from src.runner import _predict_clusters  # noqa: E402


STEP1_SAMPLES = ("151507", "151510", "151669", "151670", "151673", "151676")
STEP1_SEED = 1234
PRIMARY_METRICS = (
    "ari",
    "nmi",
    "macro_layer_iou",
    "worst_layer_iou",
    "layer_recovery_rate",
)


def _build_step1_sets() -> dict[str, dict[str, Any]]:
    sets: dict[str, dict[str, Any]] = {"anchor": {}}
    for graph_neighbors, attention_neighbors in itertools.product((8, 12, 16), repeat=2):
        if (graph_neighbors, attention_neighbors) == (12, 12):
            continue
        sets[f"neighbors_{graph_neighbors:02d}_attention_{attention_neighbors:02d}"] = {
            "data.n_neighbors": graph_neighbors,
            "model.sparniche_view1.attention_neighbors": attention_neighbors,
        }
    for latent_dim, normalize in itertools.product((16, 32, 64, 128), (True, False)):
        if (latent_dim, normalize) == (64, True):
            continue
        state = "on" if normalize else "off"
        sets[f"latent_{latent_dim}_norm_{state}"] = {
            "model.latent_dim": latent_dim,
            "model.local_graph_normalize": normalize,
        }
    sets.update(
        {
            "dropout_00": {"model.sparniche_view1.dropout": 0.0},
            "dropout_40": {"model.sparniche_view1.dropout": 0.4},
            "lr_0025": {"model.sparniche.lr": 0.0025},
            "lr_0100": {"model.sparniche.lr": 0.01},
            "wd_000": {"model.sparniche.weight_decay": 0.0},
            "wd_030": {"model.sparniche.weight_decay": 0.03},
            "epochs_300": {"training.epochs": 300},
            "epochs_800": {"training.epochs": 800},
        }
    )
    return sets


STEP1_SET_OVERRIDES = _build_step1_sets()


def build_step1_config(
    base: dict[str, Any],
    input_path: Path,
    *,
    set_name: str,
    device: str,
    input_kind: str,
) -> dict[str, Any]:
    if set_name not in STEP1_SET_OVERRIDES:
        raise ValueError(f"unknown step1 set {set_name!r}")
    overrides = {
        "paths.input_h5ad": str(input_path),
        "data.preprocessing.input_kind": str(input_kind),
        "model.variant": "local_graph_normalized",
        "model.local_graph_mode": "normalized",
        "model.sparniche_view1.attention_mode": "spatial_local",
        "model.sparniche_view1.attention_chunk_size": 4096,
        "training.seed": STEP1_SEED,
        "training.device": str(device),
        "benchmark.external_only": False,
        "evaluation.leiden_cluster_lower_offset": 0,
        "evaluation.leiden_cluster_upper_offset": 2,
    }
    overrides.update(STEP1_SET_OVERRIDES[set_name])
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


def _evaluate_run(
    adata: anndata.AnnData,
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    label_key = str(config["data"]["label_key"])
    mask = _valid_mask(adata, label_key)
    evaluation_adata = adata[mask].copy()
    embedding = np.asarray(adata.obsm["sparniche"])[mask]
    labels = adata.obs[label_key].astype(str).to_numpy()[mask]
    spatial = np.asarray(adata.obsm["spatial"])[mask]
    predicted = _predict_clusters(embedding, evaluation_adata, config)
    metrics = compute_final_metrics(
        embedding,
        labels,
        spatial,
        predicted=predicted,
        seed=STEP1_SEED,
        n_neighbors=int(config["data"]["n_neighbors"]),
    )
    recovery = compute_layer_recovery_metrics(labels, predicted)
    audit = dict(evaluation_adata.uns.get("sparniche_leiden_search", {}))
    metrics["leiden_window_fallback"] = bool(audit.get("window_fallback", False))
    metrics["selected_resolution"] = float(audit.get("selected_resolution", float("nan")))
    metrics["selected_clusters"] = float(audit.get("selected_clusters", float("nan")))
    return metrics, dict(recovery["per_layer_iou"]), audit


def _candidate_rows(
    set_name: str,
    sample: str,
    audit: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for candidate in audit.get("candidates", []):
        row = {
            "set_name": set_name,
            "sample": sample,
            "seed": STEP1_SEED,
            **{key: value for key, value in candidate.items() if key != "per_layer_iou"},
        }
        row["per_layer_iou_json"] = json.dumps(
            candidate.get("per_layer_iou", {}), sort_keys=True
        )
        rows.append(row)
    return rows


def write_step1_tables(bundles: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    run_rows = []
    layer_rows = []
    candidate_rows = []
    for bundle in bundles:
        run_rows.append(dict(bundle["run"]))
        for layer, iou in bundle.get("per_layer_iou", {}).items():
            layer_rows.append(
                {
                    "set_name": bundle["run"]["set_name"],
                    "sample": bundle["run"]["sample"],
                    "seed": bundle["run"]["seed"],
                    "layer": layer,
                    "iou": iou,
                }
            )
        candidate_rows.extend(
            _candidate_rows(
                bundle["run"]["set_name"],
                bundle["run"]["sample"],
                bundle.get("leiden_audit", {}),
            )
        )
    metrics = pd.DataFrame(run_rows)
    metrics.to_csv(output_root / "run_metrics.csv", index=False)
    layers = pd.DataFrame(layer_rows)
    layers.to_csv(output_root / "layer_iou_long.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output_root / "leiden_candidates.csv", index=False)

    completed = metrics[metrics["status"] == "completed"].copy()
    for metric in PRIMARY_METRICS:
        if metric not in completed:
            completed[metric] = np.nan
    summary_rows = []
    for set_name, group in completed.groupby("set_name", sort=True):
        for metric in PRIMARY_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            summary_rows.append(
                {
                    "set_name": set_name,
                    "metric": metric,
                    "mean": float(values.mean()) if not values.empty else float("nan"),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": int(values.size),
                }
            )
    pd.DataFrame(summary_rows).to_csv(output_root / "summary_by_config.csv", index=False)

    if not layers.empty:
        layer_summary = (
            layers.groupby(["set_name", "layer"], sort=True)["iou"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        layer_summary.to_csv(output_root / "summary_layer_iou.csv", index=False)

    paired_rows = []
    anchor = completed[completed["set_name"] == "anchor"].set_index(["sample", "seed"])
    for _, row in completed[completed["set_name"] != "anchor"].iterrows():
        key = (row["sample"], row["seed"])
        if key not in anchor.index:
            continue
        anchor_row = anchor.loc[key]
        for metric in PRIMARY_METRICS:
            paired_rows.append(
                {
                    "set_name": row["set_name"],
                    "sample": row["sample"],
                    "seed": row["seed"],
                    "metric": metric,
                    "anchor": float(anchor_row[metric]),
                    "candidate": float(row[metric]),
                    "delta_candidate_minus_anchor": float(row[metric] - anchor_row[metric]),
                }
            )
    pd.DataFrame(paired_rows).to_csv(output_root / "paired_deltas_vs_anchor.csv", index=False)

    means = completed.groupby("set_name")[list(PRIMARY_METRICS)].mean()
    fallback_counts = (
        completed.assign(
            leiden_window_fallback=completed.get(
                "leiden_window_fallback", pd.Series(False, index=completed.index)
            ).astype(bool)
        )
        .groupby("set_name")["leiden_window_fallback"]
        .sum()
    )
    ranking_rows = []
    if "anchor" in means.index:
        anchor_ari = float(means.loc["anchor", "ari"])
        anchor_nmi = float(means.loc["anchor", "nmi"])
        for set_name, row in means.iterrows():
            ari_noninferior = float(row["ari"]) >= anchor_ari - 0.02
            nmi_noninferior = float(row["nmi"]) >= anchor_nmi - 0.02
            window_fallback_runs = int(fallback_counts.get(set_name, 0))
            ranking_rows.append(
                {
                    "set_name": set_name,
                    **{metric: float(row[metric]) for metric in PRIMARY_METRICS},
                    "ari_noninferior": ari_noninferior,
                    "nmi_noninferior": nmi_noninferior,
                    "window_fallback_runs": window_fallback_runs,
                    "eligible": (
                        ari_noninferior
                        and nmi_noninferior
                        and window_fallback_runs == 0
                    ),
                }
            )
    ranking = pd.DataFrame(ranking_rows)
    if not ranking.empty:
        ranking = ranking.sort_values(
            [
                "eligible",
                "layer_recovery_rate",
                "worst_layer_iou",
                "macro_layer_iou",
                "ari",
                "nmi",
            ],
            ascending=[False, False, False, False, False, False],
        )
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(output_root / "configuration_ranking.csv", index=False)


def _select_samples(source: Path) -> list[Path]:
    available = {path.stem: path for path in source.glob("*.h5ad")}
    missing = [sample for sample in STEP1_SAMPLES if sample not in available]
    if missing:
        raise FileNotFoundError(f"missing step1 source24 samples: {', '.join(missing)}")
    return [available[sample] for sample in STEP1_SAMPLES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/root/autodl-fs/data/source24"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/root/autodl-fs/bench_results/source24/"
            "SparNiche_layer_sensitivity_step1_tmp"
        ),
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    parser.add_argument(
        "--source-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "source24.yaml",
    )
    parser.add_argument("--sets", nargs="+", choices=tuple(STEP1_SET_OVERRIDES), default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    output_root = Path(args.output_root)
    samples = _select_samples(source)
    base = deep_merge(load_yaml(args.config), load_yaml(args.source_config))
    set_names = list(args.sets) if args.sets else list(STEP1_SET_OVERRIDES)
    _write_json(
        output_root / "manifest.json",
        {
            "temporary": True,
            "stage": "step1_screening",
            "source": str(source),
            "samples": list(STEP1_SAMPLES),
            "seed": STEP1_SEED,
            "sets": {name: STEP1_SET_OVERRIDES[name] for name in set_names},
            "leiden_cluster_window": "K..K+2",
            "leiden_selection_order": [
                "layer_recovery_rate",
                "worst_layer_iou",
                "macro_layer_iou",
                "ari",
                "nmi",
                "closest_cluster_count_to_K",
                "lower_resolution",
            ],
            "layer_recovery_iou_threshold": 0.5,
            "ari_nmi_noninferiority_margin": 0.02,
        },
    )

    bundles: list[dict[str, Any]] = []
    failures = 0
    for set_name in set_names:
        for sample_path in samples:
            run_dir = output_root / "runs" / set_name / sample_path.stem / f"seed{STEP1_SEED}"
            result_path = run_dir / "step1_result.json"
            if result_path.exists() and not args.force:
                bundle = json.loads(result_path.read_text(encoding="utf-8"))
                if bundle.get("run", {}).get("status") == "completed":
                    print(f"[SKIP] set={set_name} sample={sample_path.stem}", flush=True)
                    bundles.append(bundle)
                    continue
            started = time.perf_counter()
            print(f"[RUN ] set={set_name} sample={sample_path.stem} seed={STEP1_SEED}", flush=True)
            try:
                input_kind = infer_input_kind(sample_path)
                config = build_step1_config(
                    base,
                    sample_path,
                    set_name=set_name,
                    device=args.device,
                    input_kind=input_kind,
                )
                adata = anndata.read_h5ad(sample_path)
                artifacts = train_adata(adata, config, run_dir, resume=False)
                metrics, per_layer_iou, leiden_audit = _evaluate_run(artifacts.adata, config)
                bundle = {
                    "run": {
                        "set_name": set_name,
                        "sample": sample_path.stem,
                        "seed": STEP1_SEED,
                        "status": "completed",
                        "runtime_seconds": float(time.perf_counter() - started),
                        "input_kind": input_kind,
                        **metrics,
                    },
                    "per_layer_iou": per_layer_iou,
                    "leiden_audit": leiden_audit,
                }
                np.save(run_dir / "embedding.npy", np.asarray(artifacts.adata.obsm["sparniche"]))
                if not args.keep_artifacts:
                    for name in ("trained.h5ad", "checkpoint.pt"):
                        path = run_dir / name
                        if path.exists():
                            path.unlink()
                print(
                    f"[DONE] set={set_name} sample={sample_path.stem} "
                    f"ARI={metrics['ari']:.4f} NMI={metrics['nmi']:.4f} "
                    f"worstIoU={metrics['worst_layer_iou']:.4f} "
                    f"LRR={metrics['layer_recovery_rate']:.4f}",
                    flush=True,
                )
            except Exception as error:
                failures += 1
                bundle = {
                    "run": {
                        "set_name": set_name,
                        "sample": sample_path.stem,
                        "seed": STEP1_SEED,
                        "status": "failed",
                        "runtime_seconds": float(time.perf_counter() - started),
                        "error": f"{type(error).__name__}: {error}",
                    },
                    "per_layer_iou": {},
                    "leiden_audit": {},
                }
                print(f"[FAIL] set={set_name} sample={sample_path.stem}: {error}", flush=True)
            _write_json(result_path, bundle)
            bundles.append(bundle)
            write_step1_tables(bundles, output_root)
            if failures and args.fail_fast:
                return 1
    print(
        f"[SUMMARY] completed={len(bundles) - failures}/{len(bundles)} "
        f"output={output_root}",
        flush=True,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
