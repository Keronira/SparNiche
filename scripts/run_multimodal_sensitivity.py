#!/usr/bin/env python3
"""Three-stage RNA/ADT/ATAC sensitivity screen for source29 and source30."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import apply_overrides, load_yaml, save_yaml  # noqa: E402

SAMPLES = (
    ("source29", "E13.5_S1"),
    ("source29", "E15.5_S1"),
    ("source30", "S1"),
    ("source30", "S2"),
)
SEEDS = (1234, 1235, 1236)
BASE = {
    "hvg": 2000, "pca": 200, "epochs": 550,
    "rec_w": 10.0, "gcn_w": 0.1, "dec_kl_w": 1.0,
    "self_w": 1.0, "view2_rec_w": 1.0,
    "latent": 32, "neighbors": 12,
}
LOSS_LEVELS = {
    "epochs": (200, 350, 550),
    "rec_w": (1.0, 10.0, 20.0),
    "gcn_w": (0.05, 0.1, 0.5),
    "dec_kl_w": (0.5, 1.0, 2.0),
    "self_w": (0.5, 1.0, 2.0),
    "view2_rec_w": (0.1, 1.0, 3.0),
}


def stage1_specs() -> list[dict]:
    return [{**BASE, "hvg": hvg, "pca": pca}
            for hvg, pca in itertools.product((1000, 2000, 3000, 5000), (32, 64, 128, 200))]


def stage2_specs(anchor: dict) -> list[dict]:
    result = [dict(anchor)]
    for factor, levels in LOSS_LEVELS.items():
        result.extend({**anchor, factor: level} for level in levels if level != anchor[factor])
    return result


def stage3_specs(anchor: dict) -> list[dict]:
    return [{**anchor, "latent": latent, "neighbors": neighbors}
            for latent, neighbors in itertools.product((16, 32, 64), (6, 8, 12, 16))]


def select_winner(frame: pd.DataFrame) -> str:
    required = len(SAMPLES) * len(SEEDS)
    complete = frame.loc[frame["status"].eq("completed")].copy()
    complete["ari"] = pd.to_numeric(complete["ari"], errors="coerce")
    complete = complete.dropna(subset=["ari"])
    eligible = complete.groupby("config_id").filter(
        lambda group: len(group) == required
        and group.groupby("sample")["seed"].nunique().eq(len(SEEDS)).all()
        and group["sample"].nunique() == len(SAMPLES)
    )
    if eligible.empty:
        raise RuntimeError("no candidate completed all samples and seeds")
    per_sample = eligible.groupby(["config_id", "sample"])["ari"].mean()
    return per_sample.groupby("config_id").mean().sort_values(ascending=False).index[0]


def config_id(stage: int, index: int) -> str:
    return f"stage{stage}_{index:02d}"


def make_config(base: dict, factors: dict) -> dict:
    overrides = {
        "data.preprocessing.n_top_genes": factors["hvg"],
        "data.preprocessing.pca_n_components": factors["pca"],
        "data.n_neighbors": factors["neighbors"],
        "data.view2.source": "obsm",
        "data.view2.key": "auto",
        "data.view2.atac_n_components": 64,
        "model.double_view": True,
        "model.latent_dim": factors["latent"],
        "model.sparniche_view1.attention_neighbors": factors["neighbors"],
        "model.sparniche.rec_w": factors["rec_w"],
        "model.sparniche.gcn_w": factors["gcn_w"],
        "model.sparniche.dec_kl_w": factors["dec_kl_w"],
        "model.sparniche.self_w": factors["self_w"],
        "model.sparniche.adt_rec_w": factors["view2_rec_w"],
        "training.epochs": factors["epochs"],
        "evaluation.leiden_cluster_lower_offset": 0,
        "evaluation.leiden_cluster_upper_offset": 2,
    }
    return apply_overrides(base, overrides)


def run_stage(stage: int, specs: list[dict], args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    rows = []
    by_id = {}
    base = load_yaml(PROJECT_ROOT / "configs" / "config.yaml")
    for index, factors in enumerate(specs, 1):
        name = config_id(stage, index)
        by_id[name] = factors
        run_root = args.output_root / name
        summary = run_root / "experiment_summary.csv"
        existing = pd.read_csv(summary) if summary.is_file() else pd.DataFrame()
        if len(existing) < len(SAMPLES) * len(SEEDS):
            run_root.mkdir(parents=True, exist_ok=True)
            config_path = run_root / "config.yaml"
            save_yaml(make_config(base, factors), config_path)
            command = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_experiments.py"),
                       "--experiment-set", "single", "--source",
                       *(str(args.data_root / source / f"{sample}.h5ad") for source, sample in SAMPLES),
                       "--config", str(config_path), "--double-view", "--view2-key", "auto",
                       "--device", args.device, "--seeds", *(str(seed) for seed in SEEDS),
                       "--output-root", str(run_root), "--method-name", name,
                       "--skip-visual-artifacts", "--continue-on-error", "--resume"]
            print(f"[RUN] {name} {factors}", flush=True)
            with (run_root / "run.log").open("w", encoding="utf-8") as log:
                exit_code = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, check=False).returncode
            print(f"[DONE] {name} exit_code={exit_code}", flush=True)
        if summary.is_file():
            observed = pd.read_csv(summary)
            for _, row in observed.iterrows():
                rows.append({"config_id": name, "stage": stage, "status": "completed",
                             **factors, **row.to_dict()})
        pd.DataFrame(rows).to_csv(args.output_root / f"stage{stage}_results.csv", index=False)
    return pd.DataFrame(rows), by_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-fs/data"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("/root/autodl-fs/bench_results/_tmp_source29_30_multimodal_sensitivity"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for source, sample in SAMPLES:
        if not (args.data_root / source / f"{sample}.h5ad").is_file() and not args.plan_only:
            raise FileNotFoundError(args.data_root / source / f"{sample}.h5ad")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"samples": SAMPLES, "excluded": "source29/E18.5_S1: ATAC-only",
                "seeds": SEEDS, "stage1": "HVG x PCA factorial (16)",
                "stage2": "epochs and five loss weights, one factor at a time (13)",
                "stage3": "latent x neighbors factorial (12)",
                "selection": "all 12 runs required; equal-weight sample mean ARI",
                "view2": "auto: ADT then ATAC; missing second view uses RNA-only",
                "leiden": "ARI-best K..K+2"}
    (args.output_root / "plan.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.plan_only:
        print(json.dumps(manifest, indent=2))
        return 0
    first, lookup = run_stage(1, stage1_specs(), args)
    first_winner = lookup[select_winner(first)]
    second, lookup = run_stage(2, stage2_specs(first_winner), args)
    second_winner = lookup[select_winner(second)]
    third, lookup = run_stage(3, stage3_specs(second_winner), args)
    final_id = select_winner(third)
    result = {"stage1_best": first_winner, "stage2_best": second_winner,
              "final_best": lookup[final_id], "final_id": final_id}
    (args.output_root / "best_config.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[SUMMARY] {args.output_root / 'best_config.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
