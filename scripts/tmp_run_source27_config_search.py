#!/usr/bin/env python3
"""Temporary two-stage source27 search; never writes to SparNiche_rep* outputs."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tmp_run_source26_config_search import _run_one  # noqa: E402
from src.config import apply_overrides, load_yaml  # noqa: E402


SAMPLES = ("Zhuang-ABCA-2.001", "Zhuang-ABCA-2.002", "Zhuang-ABCA-2.003")
SEEDS = (1234, 1235, 1236)
DEFAULT_DATA_DIR = Path("/root/autodl-fs/data/source27")
DEFAULT_OUTPUT = Path("/root/autodl-fs/bench_results/_tmp_source27_config_search")
SCREEN_SPECS = (
    {"name": "A", "hvg": 800, "pca": 32, "latent": 16, "neighbors": 8},
    {"name": "B", "hvg": 800, "pca": 64, "latent": 16, "neighbors": 8},
    {"name": "C", "hvg": 1000, "pca": 32, "latent": 32, "neighbors": 8},
    {"name": "D", "hvg": 1000, "pca": 64, "latent": 32, "neighbors": 12},
    {"name": "E", "hvg": 1122, "pca": 64, "latent": 16, "neighbors": 12},
    {"name": "F", "hvg": 1122, "pca": 64, "latent": 32, "neighbors": 16},
)


def run_specs():
    return [
        {
            "config_id": spec["name"],
            "factors": {key: spec[key] for key in ("hvg", "pca", "latent", "neighbors")},
        }
        for spec in SCREEN_SPECS
    ]


def select_validation_specs(frame: pd.DataFrame, *, specs: list[dict], count: int = 2):
    completed = frame.loc[frame["status"].eq("completed")].copy()
    completed["ari"] = pd.to_numeric(completed["ari"], errors="coerce")
    completed["nmi"] = pd.to_numeric(completed["nmi"], errors="coerce")
    completed = completed.dropna(subset=["ari", "nmi"])
    winners = completed.sort_values(
        ["ari", "nmi", "config_id"], ascending=[False, False, True]
    )["config_id"].head(count).tolist()
    if len(winners) != count:
        raise RuntimeError(f"only {len(winners)} successful screen runs; need {count}")
    by_id = {spec["config_id"]: spec for spec in specs}
    return [by_id[name] for name in winners]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument("--stage", choices=("all", "screen", "validate"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for sample in SAMPLES:
        if not (args.data_dir / f"{sample}.h5ad").is_file():
            raise FileNotFoundError(args.data_dir / f"{sample}.h5ad")
    if args.output_root.exists() and not (args.output_root / "manifest.json").is_file():
        raise FileExistsError(f"output exists without this experiment's manifest: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    specs = run_specs()
    base = apply_overrides(load_yaml(args.config), {"model.sparniche_view1.dec_cluster_n": 14})
    manifest = {
        "samples": SAMPLES,
        "seeds": SEEDS,
        "screen_specs": SCREEN_SPECS,
        "fixed_overrides": {"model.sparniche_view1.dec_cluster_n": 14},
        "screen": "first sample, seed 1234",
        "validation": "top two by screen ARI then NMI; three samples, three seeds",
        "leiden_window": "K..K+2",
        "output_root": str(args.output_root),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    screen_path = args.output_root / "screen.csv"
    if args.stage in ("all", "screen"):
        rows = []
        for spec in specs:
            print(f"[SCREEN] {spec['config_id']} {spec['factors']}", flush=True)
            row = _run_one(
                base=base,
                input_path=args.data_dir / f"{SAMPLES[0]}.h5ad",
                output_root=args.output_root / "samples" / SAMPLES[0],
                spec=spec, seed=SEEDS[0], device=args.device,
                label_key="annotation_final", force=False,
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(screen_path, index=False)
            print(f"[SCREEN-{row['status'].upper()}] {spec['config_id']} ARI={row.get('ari')}", flush=True)
            if row["status"] != "completed" and not args.continue_on_error:
                return 1
    else:
        if not screen_path.is_file():
            raise FileNotFoundError(screen_path)
        rows = pd.read_csv(screen_path)
    if args.stage == "screen":
        return 0

    winners = select_validation_specs(pd.DataFrame(rows), specs=specs)
    (args.output_root / "selected_specs.json").write_text(
        json.dumps(winners, indent=2), encoding="utf-8"
    )
    validation_path = args.output_root / "validation.csv"
    validation_rows = []
    for spec in winners:
        for sample in SAMPLES:
            for seed in SEEDS:
                print(f"[VALIDATE] {spec['config_id']} {sample} seed={seed}", flush=True)
                row = _run_one(
                    base=base, input_path=args.data_dir / f"{sample}.h5ad",
                    output_root=args.output_root / "samples" / sample,
                    spec=spec, seed=seed, device=args.device,
                    label_key="annotation_final", force=False,
                )
                validation_rows.append(row)
                pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
                print(f"[VALIDATE-{row['status'].upper()}] ARI={row.get('ari')}", flush=True)
                if row["status"] != "completed" and not args.continue_on_error:
                    return 1
    completed = pd.DataFrame(validation_rows)
    completed = completed.loc[completed["status"].eq("completed")]
    if not completed.empty:
        metrics = ("ari", "nmi", "fmi", "accuracy", "macro_f1")
        summary = completed.groupby("config_id")[list(metrics)].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary.to_csv(args.output_root / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
