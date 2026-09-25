#!/usr/bin/env python3
"""Temporary three-stage scan around source27 configuration E."""

from __future__ import annotations

import argparse
import itertools
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
from src.config import apply_overrides, load_yaml, save_yaml  # noqa: E402


SAMPLES = ("Zhuang-ABCA-2.001", "Zhuang-ABCA-2.002", "Zhuang-ABCA-2.003")
SEEDS = (1234, 1235, 1236)
DATA_DIR = Path("/root/autodl-fs/data/source27")
OUTPUT_ROOT = Path("/root/autodl-fs/bench_results/_tmp_source27_anchorE_scan")
ANCHOR = {
    "hvg": 1122, "pca": 64, "latent": 16, "neighbors": 12,
    "lr": 0.01, "dropout": 0.2, "weight_decay": 0.01, "epochs": 550,
}
LEVELS = {
    "hvg": (800, 1000, 1122),
    "pca": (32, 48, 64, 96),
    "latent": (8, 16, 24, 32),
    "neighbors": (8, 12, 16, 20),
    "lr": (0.005, 0.01, 0.02),
    "dropout": (0.0, 0.1, 0.2),
    "weight_decay": (0.0, 0.00001, 0.0001),
    "epochs": (400, 550, 700),
}
METRICS = ("ari", "nmi", "fmi", "accuracy", "macro_f1")
WEIGHTS = {"ari": 0.30, "nmi": 0.25, "fmi": 0.15,
           "accuracy": 0.15, "macro_f1": 0.15}


def _label(value: int | float) -> str:
    return str(value).replace(".", "p")


def _spec(factors: dict, name: str) -> dict:
    return {"config_id": name, "factors": dict(factors)}


def build_stage1_specs() -> list[dict]:
    specs = [_spec(ANCHOR, "anchor")]
    for factor, levels in LEVELS.items():
        for level in levels:
            if level == ANCHOR[factor]:
                continue
            factors = dict(ANCHOR)
            factors[factor] = level
            specs.append(_spec(factors, f"{factor}_{_label(level)}"))
    return specs


def choose_stage2_specs(screen: pd.DataFrame, specs: list[dict]) -> tuple[list[dict], list[str]]:
    scores = screen.loc[screen["status"].eq("completed")].set_index("config_id")
    if "anchor" not in scores.index:
        raise RuntimeError("anchor did not complete stage 1")
    anchor_ari = float(scores.loc["anchor", "ari"])
    best_by_factor = {}
    for factor in LEVELS:
        candidates = [spec for spec in specs if spec["factors"][factor] != ANCHOR[factor]
                      and sum(spec["factors"][key] != ANCHOR[key] for key in ANCHOR) == 1
                      and spec["config_id"] in scores.index]
        if not candidates:
            continue
        best = max(candidates, key=lambda spec: float(scores.loc[spec["config_id"], "ari"]))
        best_by_factor[factor] = (float(scores.loc[best["config_id"], "ari"]) - anchor_ari,
                                  best["factors"][factor])
    if len(best_by_factor) < 3:
        raise RuntimeError("fewer than three factors completed stage 1")
    factors = sorted(best_by_factor, key=lambda key: (-best_by_factor[key][0], key))[:3]
    selected = [_spec(ANCHOR, "anchor")]
    for count in (1, 2):
        for combination in itertools.combinations(factors, count):
            values = dict(ANCHOR)
            for factor in combination:
                values[factor] = best_by_factor[factor][1]
            name = "__".join(f"{factor}_{_label(values[factor])}" for factor in combination)
            selected.append(_spec(values, name))
    return selected, factors


def _weighted_score(frame: pd.DataFrame) -> pd.Series:
    return sum(WEIGHTS[key] * pd.to_numeric(frame[key], errors="coerce") for key in METRICS)


def select_stage3_specs(validation: pd.DataFrame, specs: list[dict]) -> list[dict]:
    completed = validation.loc[validation["status"].eq("completed")].copy()
    completed["score"] = _weighted_score(completed)
    counts = completed.groupby("config_id")["seed"].nunique()
    eligible = counts[counts.eq(len(SEEDS))].index
    ranking = completed.loc[completed["config_id"].isin(eligible)].groupby("config_id")["score"].mean()
    ranking = ranking.drop(labels="anchor", errors="ignore").sort_values(ascending=False)
    if len(ranking) < 2:
        raise RuntimeError("fewer than two non-anchor configurations completed stage 2")
    by_id = {spec["config_id"]: spec for spec in specs}
    return [by_id[name] for name in ["anchor", *ranking.head(2).index.tolist()]]


def _run_stage(specs: list[dict], samples: tuple[str, ...], seeds: tuple[int, ...],
               args: argparse.Namespace, base: dict, output: Path) -> pd.DataFrame:
    rows = []
    for spec in specs:
        factors = spec["factors"]
        overrides = {
            "model.sparniche.lr": factors["lr"],
            "model.sparniche_view1.dropout": factors["dropout"],
            "model.sparniche.weight_decay": factors["weight_decay"],
            "training.epochs": factors["epochs"],
            "model.sparniche_view1.dec_cluster_n": 14,
        }
        run_base = apply_overrides(base, overrides)
        for sample in samples:
            for seed in seeds:
                print(f"[RUN] {output.stem} {spec['config_id']} {sample} seed={seed}", flush=True)
                row = _run_one(
                    base=run_base, input_path=args.data_dir / f"{sample}.h5ad",
                    output_root=args.output_root / "samples" / sample,
                    spec=spec, seed=seed, device=args.device,
                    label_key="annotation_final", force=False,
                )
                rows.append(row)
                pd.DataFrame(rows).to_csv(output, index=False)
                print(f"[{row['status'].upper()}] ARI={row.get('ari')} NMI={row.get('nmi')}", flush=True)
                if row["status"] != "completed" and not args.continue_on_error:
                    raise RuntimeError(row.get("error", "run failed"))
    return pd.DataFrame(rows)


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame.loc[frame["status"].eq("completed")].copy()
    completed["score"] = _weighted_score(completed)
    rows = []
    for name, group in completed.groupby("config_id"):
        row = {"config_id": name, "n_runs": len(group), "n_samples": group["sample"].nunique()}
        for key in (*METRICS, "score", "runtime_seconds"):
            row[f"{key}_mean"] = group[key].mean()
            row[f"{key}_sd"] = group[key].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score_mean", ascending=False)


def _choose_final(summary: pd.DataFrame, stage3: pd.DataFrame) -> str:
    anchor = stage3.loc[stage3["config_id"].eq("anchor")].groupby("sample")["ari"].mean()
    for name in summary["config_id"]:
        if name == "anchor" or int(summary.loc[summary["config_id"].eq(name), "n_runs"].iloc[0]) != 9:
            continue
        candidate = stage3.loc[stage3["config_id"].eq(name)].groupby("sample")["ari"].mean()
        if int((candidate > anchor).sum()) >= 2:
            return name
    return "anchor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=("all", "1", "2", "3"), default="all")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for sample in SAMPLES:
        if not (args.data_dir / f"{sample}.h5ad").is_file():
            raise FileNotFoundError(args.data_dir / f"{sample}.h5ad")
    if args.output_root.exists() and not (args.output_root / "manifest.json").is_file():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    base = load_yaml(args.config)
    manifest = {"anchor": ANCHOR, "levels": LEVELS, "samples": SAMPLES, "seeds": SEEDS,
                "dec_cluster_n": 14, "leiden_window": "K..K+2", "weights": WEIGHTS,
                "stage1": "single sample and seed, one factor at a time",
                "stage2": "anchor + three singles + three pairwise combinations, three seeds",
                "stage3": "anchor + top two nonanchors, three samples and three seeds"}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    p1 = args.output_root / "stage1.csv"
    p2 = args.output_root / "stage2.csv"
    p3 = args.output_root / "stage3.csv"
    if args.stage in ("all", "1"):
        stage1 = _run_stage(build_stage1_specs(), SAMPLES[:1], SEEDS[:1], args, base, p1)
    else:
        stage1 = pd.read_csv(p1)
    if args.stage == "1":
        return 0
    stage2_specs, chosen_factors = choose_stage2_specs(stage1, build_stage1_specs())
    (args.output_root / "stage2_design.json").write_text(
        json.dumps({"factors": chosen_factors, "specs": stage2_specs}, indent=2), encoding="utf-8")
    if args.stage in ("all", "2"):
        stage2 = _run_stage(stage2_specs, SAMPLES[:1], SEEDS, args, base, p2)
    else:
        stage2 = pd.read_csv(p2)
    if args.stage == "2":
        return 0
    stage3_specs = select_stage3_specs(stage2, stage2_specs)
    (args.output_root / "stage3_design.json").write_text(json.dumps(stage3_specs, indent=2), encoding="utf-8")
    stage3 = _run_stage(stage3_specs, SAMPLES, SEEDS, args, base, p3)
    summary = _summary(stage3)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    selected = _choose_final(summary, stage3)
    factors = next(spec["factors"] for spec in stage3_specs if spec["config_id"] == selected)
    (args.output_root / "best_config.json").write_text(
        json.dumps({"config_id": selected, "factors": factors,
                    "selection": "highest weighted score with ARI improvement on at least 2/3 samples; otherwise anchor"}, indent=2),
        encoding="utf-8")
    print(f"[SUMMARY] selected={selected} output={args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
