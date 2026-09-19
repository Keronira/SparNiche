#!/usr/bin/env python3
"""Summarize RNA encoder experiments in calculate_bench-style tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


METRICS = (
    "ari",
    "nmi",
    "ami",
    "ece",
    "brier",
    "auroc",
    "runtime_seconds",
    "peak_cuda_memory_bytes",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_records(sequence_dir: Path, allow_incomplete: bool = False) -> pd.DataFrame:
    manifest_path = sequence_dir / "run_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    rows: list[dict] = []
    missing: list[str] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        manifest = json.loads(line)
        run_id = str(manifest["run_id"])
        run_dir = sequence_dir / "runs" / run_id
        status_path = run_dir / "status.json"
        condition_path = run_dir / "condition_metrics.jsonl"
        summary_path = run_dir / "summary_metrics.json"
        if not status_path.is_file() or not condition_path.is_file():
            missing.append(run_id)
            continue
        status = _read_json(status_path)
        if status.get("state") != "completed" and not allow_incomplete:
            missing.append(run_id)
            continue
        summary = _read_json(summary_path) if summary_path.is_file() else {}
        conditions = [json.loads(item) for item in condition_path.read_text(encoding="utf-8").splitlines() if item.strip()]
        for condition in conditions:
            row = {
                "run_id": run_id,
                "dataset": manifest.get("dataset"),
                "case": manifest.get("case", manifest.get("variant")),
                "variant": manifest.get("variant"),
                "seed": manifest.get("seed"),
                "regime": condition.get("regime"),
                "target": condition.get("target"),
                "severity": condition.get("severity"),
                "kind": condition.get("kind"),
            }
            row.update({metric: condition.get(metric) for metric in METRICS if metric in condition})
            row["runtime_seconds"] = summary.get("runtime_seconds", row.get("runtime_seconds"))
            row["peak_cuda_memory_bytes"] = summary.get(
                "peak_cuda_memory_bytes", row.get("peak_cuda_memory_bytes")
            )
            rows.append(row)
    if missing and not allow_incomplete:
        print(f"skipped {len(missing)} incomplete run(s)", file=sys.stderr)
    return pd.DataFrame(rows)


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    numeric = [metric for metric in METRICS if metric in frame.columns]
    grouped = frame.groupby(["case"], dropna=False)[numeric]
    mean = grouped.mean(numeric_only=True).add_suffix("_mean")
    std = grouped.std(ddof=1, numeric_only=True).fillna(0.0).add_suffix("_std")
    count = grouped.count().max(axis=1).rename("n_records")
    summary = pd.concat([count, mean, std], axis=1).reset_index()
    return summary.sort_values("case").reset_index(drop=True)


def _plot_metric_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    cases = summary["case"].astype(str).tolist()
    positions = np.arange(len(cases))
    width = 0.36
    figure, axis = plt.subplots(figsize=(max(7.0, len(cases) * 1.8), 5.0))
    for offset, metric, color in [(-width / 2, "ari", "#4c72b0"), (width / 2, "nmi", "#dd8452")]:
        mean = summary[f"{metric}_mean"].to_numpy(float)
        error = summary[f"{metric}_std"].to_numpy(float)
        axis.bar(positions + offset, mean, width, yerr=error, capsize=4, label=metric.upper(), color=color)
    axis.set_xticks(positions, cases, rotation=20, ha="right")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("RNA encoder experiment comparison")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    sequence_dir = args.sequence_dir
    output_dir = args.output_dir or sequence_dir / "rna_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = collect_records(sequence_dir, allow_incomplete=args.allow_incomplete)
    summary = build_summary(frame)
    frame.to_csv(output_dir / "detail_results.csv", index=False)
    summary.to_csv(output_dir / "summary_mean_std.csv", index=False)
    if not summary.empty:
        summary[["case"] + [column for column in summary.columns if column.endswith("_mean")]].to_csv(
            output_dir / "summary.csv", index=False
        )
        _plot_metric_summary(summary, output_dir / "ari_nmi_barplot.png")
    (output_dir / "aggregation_status.json").write_text(
        json.dumps({"records": int(len(frame)), "cases": sorted(frame["case"].dropna().astype(str).unique()) if not frame.empty else []}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(frame)} condition records to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
