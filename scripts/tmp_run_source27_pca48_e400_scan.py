#!/usr/bin/env python3
"""Temporary source27 scan around PCA48 and 400 DEC epochs."""

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
from src.config import apply_overrides, load_yaml  # noqa: E402


SAMPLES = ("Zhuang-ABCA-2.001", "Zhuang-ABCA-2.002", "Zhuang-ABCA-2.003")
SEEDS = (1234, 1235, 1236)
ANCHOR = {
    "hvg": 1122, "pca": 48, "latent": 16, "neighbors": 12,
    "lr": 0.01, "dropout": 0.2, "weight_decay": 0.01, "epochs": 400,
}
LEVELS = {
    "pca": (32, 48, 64),
    "latent": (16, 24, 32),
    "neighbors": (12, 16, 20),
    "epochs": (300, 400, 500),
    "lr": (0.005, 0.01, 0.015),
    "dropout": (0.1, 0.2, 0.3),
    "weight_decay": (0.005, 0.01, 0.02),
}
METRICS = ("ari", "nmi", "fmi", "accuracy", "macro_f1")
WEIGHTS = {"ari": 0.30, "nmi": 0.25, "fmi": 0.15,
           "accuracy": 0.15, "macro_f1": 0.15}
DEFAULT_DATA_DIR = Path("/root/autodl-fs/data/source27")
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-fs/bench_results/_tmp_source27_pca48_e400_scan")


def _label(value: int | float) -> str:
    return str(value).replace(".", "p")


def _spec(factors: dict, name: str) -> dict:
    return {"config_id": name, "factors": dict(factors)}


def build_stage1_specs() -> list[dict]:
    specs = [_spec(ANCHOR, "anchor")]
    for factor, levels in LEVELS.items():
        for value in levels:
            if value == ANCHOR[factor]:
                continue
            factors = dict(ANCHOR)
            factors[factor] = value
            specs.append(_spec(factors, f"{factor}_{_label(value)}"))
    return specs


def select_stage2_specs(frame: pd.DataFrame, specs: list[dict], samples: list[str] | tuple[str, ...]) -> tuple[list[dict], list[str]]:
    completed = frame.loc[frame["status"].eq("completed")].copy()
    completed = completed.loc[completed["sample"].isin(samples)]
    completed["ari"] = pd.to_numeric(completed["ari"], errors="coerce")
    completed["nmi"] = pd.to_numeric(completed["nmi"], errors="coerce")
    grouped = completed.groupby("config_id")
    eligible = grouped.filter(lambda group: group["sample"].nunique() == len(samples) and len(group) == len(samples))
    means = eligible.groupby("config_id")[["ari", "nmi"]].mean()
    if "anchor" not in means.index:
        raise RuntimeError("anchor did not complete all stage 1 samples")
    by_factor = {}
    for factor in LEVELS:
        candidates = [spec for spec in specs if spec["factors"][factor] != ANCHOR[factor]
                      and spec["config_id"] in means.index]
        if candidates:
            best = max(candidates, key=lambda spec: (means.loc[spec["config_id"], "ari"],
                                                    means.loc[spec["config_id"], "nmi"]))
            by_factor[factor] = (means.loc[best["config_id"], "ari"],
                                 means.loc[best["config_id"], "nmi"],
                                 best["factors"][factor])
    if len(by_factor) < 3:
        raise RuntimeError("fewer than three factors completed all stage 1 samples")
    factors = sorted(by_factor, key=lambda key: (-by_factor[key][0], -by_factor[key][1], key))[:3]
    selected = [_spec(ANCHOR, "anchor")]
    for size in (1, 2):
        for combination in itertools.combinations(factors, size):
            values = dict(ANCHOR)
            for factor in combination:
                values[factor] = by_factor[factor][2]
            name = "__".join(f"{factor}_{_label(values[factor])}" for factor in combination)
            selected.append(_spec(values, name))
    return selected, factors


def _weighted_score(frame: pd.DataFrame) -> pd.Series:
    return sum(weight * pd.to_numeric(frame[metric], errors="coerce")
               for metric, weight in WEIGHTS.items())


def choose_final_spec(frame: pd.DataFrame, specs: list[dict],
                      samples: tuple[str, ...], seeds: tuple[int, ...]) -> dict:
    completed = frame.loc[frame["status"].eq("completed")].copy()
    completed = completed.loc[completed["sample"].isin(samples) & completed["seed"].isin(seeds)]
    completed["score"] = _weighted_score(completed)
    expected = {(sample, seed) for sample in samples for seed in seeds}
    eligible = {}
    for name, group in completed.groupby("config_id"):
        pairs = list(zip(group["sample"], group["seed"]))
        if len(pairs) == len(expected) and set(pairs) == expected:
            eligible[name] = group
    if "anchor" not in eligible:
        raise RuntimeError("anchor lacks complete three-sample, three-seed results")
    anchor_ari = eligible["anchor"].groupby("sample")["ari"].mean()
    anchor_score = eligible["anchor"]["score"].mean()
    ranked = sorted(eligible, key=lambda name: (-eligible[name]["score"].mean(), name))
    by_id = {spec["config_id"]: spec for spec in specs}
    for name in ranked:
        if name == "anchor":
            continue
        sample_ari = eligible[name].groupby("sample")["ari"].mean()
        if eligible[name]["score"].mean() > anchor_score and (sample_ari > anchor_ari).sum() >= 2:
            return by_id[name]
    return by_id["anchor"]


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    completed = frame.loc[frame["status"].eq("completed")].copy()
    completed["score"] = _weighted_score(completed)
    rows = []
    for name, group in completed.groupby("config_id"):
        row = {"config_id": name, "n_runs": len(group), "n_samples": group["sample"].nunique()}
        for metric in (*METRICS, "score", "runtime_seconds"):
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score_mean", ascending=False)


def _run_stage(specs: list[dict], seeds: tuple[int, ...], base: dict,
               args: argparse.Namespace, output: Path) -> pd.DataFrame:
    rows = []
    for spec in specs:
        factors = spec["factors"]
        run_base = apply_overrides(base, {
            "model.sparniche.lr": factors["lr"],
            "model.sparniche_view1.dropout": factors["dropout"],
            "model.sparniche.weight_decay": factors["weight_decay"],
            "training.epochs": factors["epochs"],
            "model.sparniche_view1.dec_cluster_n": 14,
        })
        for sample in SAMPLES:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/config.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=("all", "1", "2"), default="all")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def prepare_output_root(root: Path, manifest: dict) -> None:
    manifest_path = root / "manifest.json"
    normalized = json.loads(json.dumps(manifest))
    if root.exists():
        if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")) != normalized:
            raise FileExistsError(f"output directory is not this scan: {root}")
    else:
        root.mkdir(parents=True)
        manifest_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    for sample in SAMPLES:
        if not (args.data_dir / f"{sample}.h5ad").is_file():
            raise FileNotFoundError(args.data_dir / f"{sample}.h5ad")
    manifest = {"anchor": ANCHOR, "levels": LEVELS, "samples": SAMPLES, "seeds": SEEDS,
                "dec_cluster_n": 14, "leiden_window": "K..K+2", "weights": WEIGHTS,
                "stage1": "three samples, seed 1234, one factor at a time",
                "stage2": "anchor, three singles, three pairs; three samples and three seeds"}
    prepare_output_root(args.output_root, manifest)
    base = load_yaml(args.config)
    stage1_path = args.output_root / "stage1.csv"
    if args.stage in ("all", "1"):
        stage1 = _run_stage(build_stage1_specs(), SEEDS[:1], base, args, stage1_path)
        _summarize(stage1).to_csv(args.output_root / "stage1_summary.csv", index=False)
    else:
        stage1 = pd.read_csv(stage1_path)
    if args.stage == "1":
        return 0
    specs, factors = select_stage2_specs(stage1, build_stage1_specs(), SAMPLES)
    (args.output_root / "stage2_design.json").write_text(
        json.dumps({"selected_factors": factors, "specs": specs}, indent=2), encoding="utf-8")
    stage2 = _run_stage(specs, SEEDS, base, args, args.output_root / "stage2.csv")
    summary = _summarize(stage2)
    summary.to_csv(args.output_root / "summary.csv", index=False)
    selected = choose_final_spec(stage2, specs, SAMPLES, SEEDS)
    (args.output_root / "best_config.json").write_text(
        json.dumps({"config_id": selected["config_id"], "factors": selected["factors"],
                    "selection": "highest weighted score with complete results and ARI gain in at least two samples"},
                   indent=2), encoding="utf-8")
    print(f"[SUMMARY] selected={selected['config_id']} output={args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
