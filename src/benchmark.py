from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


AMBIGUOUS_GROUND_TRUTH_LABELS = frozenset(
    {"", "ambiguous", "nan", "na", "n/a", "none", "null", "other", "unknown", "unassigned", "unlabeled"}
)


def filter_ambiguous_ground_truth(values: Sequence[Any]) -> list[Any]:
    """Keep only labels considered valid by the baseline evaluation path."""
    result = []
    for value in values:
        if pd.isna(value):
            continue
        if str(value).strip().lower() in AMBIGUOUS_GROUND_TRUTH_LABELS:
            continue
        result.append(value)
    return result


@dataclass(frozen=True)
class BenchmarkPaths:
    method_root: Path
    artifacts_dir: Path
    results_dir: Path
    prediction_csv: Path
    embedding_csv: Path
    profile_json: Path


def benchmark_output_paths(
    bench_root: Path,
    dataset: str,
    method: str,
    sample: str,
) -> BenchmarkPaths:
    method_root = Path(bench_root) / str(dataset) / str(method)
    results_dir = method_root / "results"
    return BenchmarkPaths(
        method_root=method_root,
        artifacts_dir=method_root / "artifacts",
        results_dir=results_dir,
        prediction_csv=results_dir / f"{sample}.csv",
        embedding_csv=results_dir / f"{sample}_embedding.csv",
        profile_json=results_dir / f"{sample}_profile.json",
    )


def export_benchmark_result(
    method_root: Path,
    sample: str,
    spot_ids: Sequence[Any],
    ground_truth: Sequence[Any],
    predicted: Sequence[Any],
    embedding: np.ndarray,
    *,
    runtime_seconds: float,
    cpu_rss_peak_mb: float | None = None,
) -> BenchmarkPaths:
    method_root = Path(method_root)
    results_dir = method_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    spot_ids = list(spot_ids)
    ground_truth = list(ground_truth)
    predicted = list(predicted)
    if not (len(spot_ids) == len(ground_truth) == len(predicted)):
        raise ValueError("benchmark labels and spot_ids must have equal lengths")
    embedding = np.asarray(embedding)
    if embedding.ndim != 2 or embedding.shape[0] != len(spot_ids):
        raise ValueError("benchmark embedding rows must match spot_ids")
    normalized_truth = np.asarray(
        ["" if pd.isna(value) else str(value).strip().lower() for value in ground_truth],
        dtype=object,
    )
    labeled = np.asarray(pd.notna(ground_truth), dtype=bool)
    labeled &= ~np.isin(normalized_truth, list(AMBIGUOUS_GROUND_TRUTH_LABELS))
    spot_ids = [value for value, keep in zip(spot_ids, labeled) if keep]
    ground_truth = [value for value, keep in zip(ground_truth, labeled) if keep]
    predicted = [value for value, keep in zip(predicted, labeled) if keep]
    embedding = embedding[labeled]
    prediction_csv = results_dir / f"{sample}.csv"
    embedding_csv = results_dir / f"{sample}_embedding.csv"
    profile_json = results_dir / f"{sample}_profile.json"
    pd.DataFrame(
        {"spot_id": spot_ids, "ground_truth": ground_truth, "pred": predicted}
    ).to_csv(prediction_csv, index=False)
    embedding_frame = pd.DataFrame(embedding)
    embedding_frame.insert(0, "spot_id", spot_ids)
    embedding_frame.to_csv(embedding_csv, index=False)
    profile = {"runtime_seconds": float(runtime_seconds)}
    if cpu_rss_peak_mb is not None:
        profile["cpu_rss_peak_mb"] = float(cpu_rss_peak_mb)
    profile_json.write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")
    return BenchmarkPaths(
        method_root=method_root,
        artifacts_dir=method_root / "artifacts",
        results_dir=results_dir,
        prediction_csv=prediction_csv,
        embedding_csv=embedding_csv,
        profile_json=profile_json,
    )
