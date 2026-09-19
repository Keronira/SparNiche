#!/usr/bin/env python3
"""Temporary scaling ablation for SparNiche graph construction and negative sampling.

The formal model code is not modified.  Each worker process patches the two
selected functions in memory, runs one normal SparNiche experiment, and exits.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import anndata
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

try:
    import resource
except ImportError:  # pragma: no cover - Windows only
    resource = None

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_experiments import infer_input_kind, infer_n_clusters  # noqa: E402
from src.config import apply_overrides, load_yaml  # noqa: E402
from src.evaluation import clustering_metrics  # noqa: E402


CONDITIONS: dict[str, tuple[bool, bool]] = {
    "anchor": (False, False),
    "sparse_graph": (True, False),
    "sparse_negative": (False, True),
    "sparse_both": (True, True),
}
DEFAULT_SOURCES = ("source1", "source2", "source7", "source12", "source13", "source24")


def build_dense_sparniche_graph(
    coordinates: np.ndarray,
    n_neighbors: int = 12,
) -> dict[str, Any]:
    """Reference implementation used only for the historical anchor arm."""
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError("coordinates must contain at least two spots")
    n_spots = int(coordinates.shape[0])
    count = min(max(1, int(n_neighbors)), n_spots - 1)
    distances = pairwise_distances(coordinates)
    adjacency = np.zeros((n_spots, n_spots), dtype=np.int64)
    for spot in range(n_spots):
        selected = np.argsort(distances[spot])[: count + 1]
        adjacency[spot, selected] = 1
    np.fill_diagonal(adjacency, 0)
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float32)
    adjacency_no_self = sparse.coo_matrix(adjacency)
    adjacency_no_self = adjacency_no_self - sparse.dia_matrix(
        (adjacency_no_self.diagonal()[np.newaxis, :], [0]),
        shape=adjacency_no_self.shape,
    )
    adjacency_no_self.eliminate_zeros()
    adjacency_with_self = adjacency_no_self + sparse.eye(n_spots)
    row_sum = np.asarray(adjacency_with_self.sum(1)).reshape(-1)
    degree_inv_sqrt = sparse.diags(np.power(row_sum, -0.5))
    adjacency_normalized = (
        adjacency_with_self.dot(degree_inv_sqrt)
        .transpose()
        .dot(degree_inv_sqrt)
        .tocoo()
        .astype(np.float32)
    )
    adjacency_label = adjacency_with_self.tocoo()
    edge_count = float(adjacency_label.sum())
    norm_value = (n_spots * n_spots) / (
        (n_spots * n_spots - edge_count) * 2.0
    )
    return {
        "adj_norm": _torch_sparse(adjacency_normalized, values_are_ones=False),
        "adj_label": _torch_sparse(adjacency_label, values_are_ones=True),
        "norm_value": float(norm_value),
    }


def build_dense_sparniche_negative_mask(
    adj_label: torch.Tensor,
    repeats: int = 1,
    seed: int | None = None,
) -> torch.Tensor:
    """Reference implementation used only for the historical anchor arm."""
    label = adj_label.coalesce()
    n_spots = int(label.shape[0])
    if repeats < 0:
        raise ValueError("repeats must be non-negative")
    generator = (
        torch.Generator(device="cpu").manual_seed(int(seed))
        if seed is not None
        else None
    )
    edge_indices = label.indices()
    neighbors: list[set[int]] = [set() for _ in range(n_spots)]
    for source, target in edge_indices.t().tolist():
        neighbors[source].add(target)
    negative_rows: list[int] = []
    negative_cols: list[int] = []
    all_nodes = set(range(n_spots))
    for source in range(n_spots):
        available = sorted(all_nodes.difference(neighbors[source]))
        count = min(len(available), len(neighbors[source]) * int(repeats))
        if count <= 0:
            continue
        permutation = torch.randperm(len(available), generator=generator)[:count]
        negative_rows.extend([source] * count)
        negative_cols.extend(available[index] for index in permutation.tolist())
    if not negative_rows:
        return label
    negative_indices = torch.tensor(
        [negative_rows, negative_cols], dtype=torch.long
    )
    negative_values = torch.zeros(len(negative_rows), dtype=label.values().dtype)
    return torch.sparse_coo_tensor(
        torch.cat((edge_indices, negative_indices), dim=1),
        torch.cat((label.values(), negative_values)),
        label.shape,
    ).coalesce()


def _torch_sparse(matrix: sparse.spmatrix, *, values_are_ones: bool) -> torch.Tensor:
    coo = matrix.tocoo().astype(np.float32)
    values = (
        torch.ones(coo.nnz, dtype=torch.float32)
        if values_are_ones
        else torch.from_numpy(coo.data)
    )
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)),
        values,
        coo.shape,
    ).coalesce()


def build_sparse_sparniche_graph(
    coordinates: np.ndarray,
    n_neighbors: int = 12,
) -> dict[str, Any]:
    """Build SparNiche's symmetrized KNN graph without an N x N distance matrix."""
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError("coordinates must contain at least two spots")
    n_spots = int(coordinates.shape[0])
    count = min(max(1, int(n_neighbors)), n_spots - 1)

    model = NearestNeighbors(n_neighbors=count + 1).fit(coordinates)
    candidates = model.kneighbors(coordinates, return_distance=False)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for source, row in enumerate(candidates):
        selected = row[row != source][:count]
        if selected.size != count:
            raise RuntimeError(
                f"could not find {count} non-self neighbors for spot {source}"
            )
        rows.append(np.full(count, source, dtype=np.int64))
        cols.append(selected.astype(np.int64, copy=False))
    directed = sparse.coo_matrix(
        (
            np.ones(n_spots * count, dtype=np.float32),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(n_spots, n_spots),
    ).tocsr()
    directed.setdiag(0)
    directed.eliminate_zeros()
    adjacency_no_self = directed.maximum(directed.T).astype(np.float32)

    adjacency_with_self = adjacency_no_self + sparse.eye(
        n_spots, dtype=np.float32, format="csr"
    )
    row_sum = np.asarray(adjacency_with_self.sum(axis=1)).reshape(-1)
    degree_inv_sqrt = sparse.diags(np.power(row_sum, -0.5))
    adjacency_normalized = (
        adjacency_with_self.dot(degree_inv_sqrt)
        .transpose()
        .dot(degree_inv_sqrt)
        .tocoo()
        .astype(np.float32)
    )
    adjacency_label = adjacency_with_self.tocoo()
    edge_count = float(adjacency_label.sum())
    norm_value = (n_spots * n_spots) / (
        (n_spots * n_spots - edge_count) * 2.0
    )
    return {
        "adj_norm": _torch_sparse(adjacency_normalized, values_are_ones=False),
        "adj_label": _torch_sparse(adjacency_label, values_are_ones=True),
        "norm_value": float(norm_value),
    }


def _sample_non_neighbors(
    n_spots: int,
    forbidden: set[int],
    count: int,
    generator: torch.Generator | None,
) -> list[int]:
    if count <= 0:
        return []
    selected: set[int] = set()
    draw_budget = max(n_spots * 2, count * 20)
    draws = 0
    while len(selected) < count and draws < draw_budget:
        batch_size = max(16, (count - len(selected)) * 2)
        candidates = torch.randint(
            n_spots, (batch_size,), generator=generator, device="cpu"
        ).tolist()
        draws += batch_size
        for candidate in candidates:
            if candidate not in forbidden and candidate not in selected:
                selected.add(int(candidate))
                if len(selected) == count:
                    break
    if len(selected) < count:
        start = int(
            torch.randint(n_spots, (1,), generator=generator, device="cpu").item()
        )
        for offset in range(n_spots):
            candidate = (start + offset) % n_spots
            if candidate not in forbidden and candidate not in selected:
                selected.add(candidate)
                if len(selected) == count:
                    break
    if len(selected) != count:
        raise RuntimeError("could not sample the requested number of non-neighbors")
    return sorted(selected)


def build_sparse_sparniche_negative_mask(
    adj_label: torch.Tensor,
    repeats: int = 1,
    seed: int | None = None,
) -> torch.Tensor:
    """Sample SparNiche non-edges without constructing every node complement."""
    if repeats < 0:
        raise ValueError("repeats must be non-negative")
    label = adj_label.coalesce().cpu()
    n_spots = int(label.shape[0])
    generator = (
        torch.Generator(device="cpu").manual_seed(int(seed))
        if seed is not None
        else None
    )
    edge_indices = label.indices()
    neighbors: list[set[int]] = [set() for _ in range(n_spots)]
    for source, target in edge_indices.t().tolist():
        neighbors[source].add(target)

    negative_rows: list[int] = []
    negative_cols: list[int] = []
    for source, forbidden in enumerate(neighbors):
        count = min(n_spots - len(forbidden), len(forbidden) * int(repeats))
        sampled = _sample_non_neighbors(n_spots, forbidden, count, generator)
        negative_rows.extend([source] * len(sampled))
        negative_cols.extend(sampled)
    if not negative_rows:
        return label
    negative_indices = torch.tensor(
        [negative_rows, negative_cols], dtype=torch.long
    )
    negative_values = torch.zeros(len(negative_rows), dtype=label.values().dtype)
    return torch.sparse_coo_tensor(
        torch.cat((edge_indices, negative_indices), dim=1),
        torch.cat((label.values(), negative_values)),
        label.shape,
    ).coalesce()


def select_input_samples(
    data_root: Path,
    sources: list[str] | tuple[str, ...],
    samples_per_source: int,
) -> list[tuple[str, Path]]:
    if samples_per_source <= 0:
        raise ValueError("samples_per_source must be positive")
    selected: list[tuple[str, Path]] = []
    for source in sources:
        source_dir = Path(data_root) / source
        if not source_dir.is_dir():
            raise FileNotFoundError(f"source directory does not exist: {source_dir}")
        paths = sorted(source_dir.glob("*.h5ad"))[: int(samples_per_source)]
        if not paths:
            raise FileNotFoundError(f"no .h5ad files found in {source_dir}")
        selected.extend((source, path) for path in paths)
    return selected


def _timed(
    timings: dict[str, float],
    key: str,
    function: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            timings[key] = timings.get(key, 0.0) + time.perf_counter() - started

    return wrapper


def _cpu_peak_mb() -> float:
    if resource is None:
        return 0.0
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _build_config(args: argparse.Namespace, input_path: Path, run_dir: Path) -> dict[str, Any]:
    n_clusters = infer_n_clusters(input_path, args.label_key)
    input_kind = infer_input_kind(input_path)
    overrides: dict[str, Any] = {
        "paths.input_h5ad": str(input_path),
        "data.label_key": args.label_key,
        "data.preprocessing.input_kind": input_kind,
        "model.variant": "local_graph_normalized",
        "model.sparniche_view1.attention_mode": args.attention_mode,
        "training.seed": int(args.seed),
        "training.device": args.device,
        "evaluation.n_clusters": int(n_clusters),
        "benchmark.external_only": True,
        "benchmark.method_root": str(run_dir),
        "benchmark.sample_name": input_path.stem,
        "benchmark.write_plot": False,
        "benchmark.write_h5ad": False,
    }
    optional_epochs = {
        "model.sparniche.gan_epochs": args.gan_epochs,
        "model.sparniche.pretrain_epochs": args.pretrain_epochs,
        "training.epochs": args.epochs,
    }
    overrides.update(
        {key: int(value) for key, value in optional_epochs.items() if value is not None}
    )
    return apply_overrides(load_yaml(args.config), overrides)


def _metrics_from_prediction(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    scores = clustering_metrics(frame["ground_truth"], frame["pred"])
    return {
        "n": int(len(frame)),
        "ari": float(scores["ari"]),
        "nmi": float(scores["nmi"]),
        "ami": float(scores["ami"]),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_worker(args: argparse.Namespace) -> int:
    from src import pipeline, runner, trainer

    if args.condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {args.condition}")
    sparse_graph, sparse_negative = CONDITIONS[args.condition]
    timings: dict[str, float] = {}
    graph_function = build_sparse_sparniche_graph if sparse_graph else build_dense_sparniche_graph
    negative_function = (
        build_sparse_sparniche_negative_mask
        if sparse_negative
        else build_dense_sparniche_negative_mask
    )
    pipeline.build_sparniche_graph = _timed(timings, "graph_seconds", graph_function)
    trainer.build_sparniche_negative_mask = _timed(
        timings, "negative_sampling_seconds", negative_function
    )
    pipeline.train_tensors = _timed(
        timings, "training_seconds", pipeline.train_tensors
    )
    runner._leiden_labels = _timed(
        timings, "leiden_seconds", runner._leiden_labels
    )

    run_dir = args.run_dir.resolve()
    input_path = args.input_h5ad.resolve()
    record: dict[str, Any] = {
        "source": args.source_name,
        "sample": input_path.stem,
        "input_h5ad": str(input_path),
        "condition": args.condition,
        "seed": int(args.seed),
        "attention_mode": args.attention_mode,
        "run_dir": str(run_dir),
        "status": "failed",
    }
    started = time.perf_counter()
    try:
        config = _build_config(args, input_path, run_dir)
        result = runner.run_experiment(
            config,
            run_dir / "artifacts",
            resume=args.resume,
            force=args.force,
        )
        prediction_path = run_dir / "results" / f"{input_path.stem}.csv"
        record.update(_metrics_from_prediction(prediction_path))
        record["status"] = "completed"
        record["skipped"] = bool(result.get("skipped", False))
    except Exception as error:
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
        record["traceback"] = traceback.format_exc()
    finally:
        record["runtime_seconds"] = time.perf_counter() - started
        record.update(timings)
        record["cpu_rss_peak_mb"] = _cpu_peak_mb()
        record["peak_cuda_memory_mb"] = (
            float(torch.cuda.max_memory_allocated()) / (1024.0**2)
            if torch.cuda.is_available()
            else 0.0
        )
        _write_json(args.result_json, record)
    return 0 if record["status"] == "completed" else 1


def _read_records(records_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(records_dir.glob("*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _add_anchor_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    anchors = frame.loc[frame["condition"] == "anchor"].set_index(
        ["source", "sample"]
    )
    output = frame.copy()
    for index, row in output.iterrows():
        key = (row["source"], row["sample"])
        if key not in anchors.index:
            continue
        anchor = anchors.loc[key]
        if isinstance(anchor, pd.DataFrame):
            anchor = anchor.iloc[0]
        for metric in ("ari", "nmi", "ami"):
            if metric in row and pd.notna(row.get(metric)) and pd.notna(anchor.get(metric)):
                output.loc[index, f"{metric}_delta_vs_anchor"] = (
                    float(row[metric]) - float(anchor[metric])
                )
        if row.get("runtime_seconds", 0) > 0:
            output.loc[index, "speedup_vs_anchor"] = (
                float(anchor["runtime_seconds"]) / float(row["runtime_seconds"])
            )
        for metric in ("cpu_rss_peak_mb", "peak_cuda_memory_mb"):
            if row.get(metric, 0) > 0 and anchor.get(metric, 0) > 0:
                output.loc[index, f"{metric}_reduction_vs_anchor"] = 1.0 - (
                    float(row[metric]) / float(anchor[metric])
                )
    return output


def write_summaries(sequence_dir: Path) -> None:
    records = _read_records(sequence_dir / "run_records")
    frame = _add_anchor_comparisons(pd.DataFrame(records))
    frame.to_csv(sequence_dir / "run_summary.csv", index=False)
    json_records = []
    for record in frame.to_dict("records"):
        json_records.append(
            {
                key: (
                    None
                    if value is None
                    or (isinstance(value, float) and not np.isfinite(value))
                    else value
                )
                for key, value in record.items()
            }
        )
    (sequence_dir / "run_summary.json").write_text(
        json.dumps(json_records, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    completed = frame.loc[frame.get("status", pd.Series(dtype=str)) == "completed"].copy()
    numeric = [
        column
        for column in (
            "ari",
            "nmi",
            "ami",
            "runtime_seconds",
            "graph_seconds",
            "negative_sampling_seconds",
            "training_seconds",
            "leiden_seconds",
            "cpu_rss_peak_mb",
            "peak_cuda_memory_mb",
            "speedup_vs_anchor",
            "ari_delta_vs_anchor",
            "nmi_delta_vs_anchor",
            "ami_delta_vs_anchor",
            "cpu_rss_peak_mb_reduction_vs_anchor",
            "peak_cuda_memory_mb_reduction_vs_anchor",
        )
        if column in completed.columns
    ]
    if completed.empty:
        aggregate = pd.DataFrame()
    else:
        aggregate = completed.groupby("condition", sort=False)[numeric].agg(
            ["mean", "std", "count"]
        )
        aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
        aggregate = aggregate.reset_index()
    aggregate.to_csv(sequence_dir / "condition_summary.csv", index=False)

    lines = [
        "# Sparse scaling ablation summary",
        "",
        f"- Completed runs: {int((frame.get('status') == 'completed').sum()) if not frame.empty else 0}",
        f"- Failed runs: {int((frame.get('status') == 'failed').sum()) if not frame.empty else 0}",
        "",
    ]
    if not aggregate.empty:
        display_columns = [
            column
            for column in (
                "condition",
                "ari_mean",
                "nmi_mean",
                "ami_mean",
                "runtime_seconds_mean",
                "cpu_rss_peak_mb_mean",
                "peak_cuda_memory_mb_mean",
                "speedup_vs_anchor_mean",
            )
            if column in aggregate.columns
        ]
        display = aggregate[display_columns]
        lines.append("| " + " | ".join(display.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(display.columns)) + " |")
        for _, row in display.iterrows():
            values = []
            for column in display.columns:
                value = row[column]
                values.append(
                    f"{float(value):.6g}"
                    if isinstance(value, (float, np.floating)) and pd.notna(value)
                    else str(value)
                )
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    failed = frame.loc[frame.get("status") == "failed"] if not frame.empty else frame
    if not failed.empty:
        lines.extend(["## Failures", ""])
        for _, row in failed.iterrows():
            lines.append(
                f"- {row['source']}/{row['sample']} [{row['condition']}]: "
                f"{row.get('error_type', '')}: {row.get('error', '')}"
            )
        lines.append("")
    (sequence_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _worker_command(
    args: argparse.Namespace,
    source: str,
    input_path: Path,
    condition: str,
    run_dir: Path,
    result_json: Path,
) -> list[str]:
    command = [
        args.python_exe,
        str(Path(__file__).resolve()),
        "--worker",
        "--condition",
        condition,
        "--source-name",
        source,
        "--input-h5ad",
        str(input_path),
        "--run-dir",
        str(run_dir),
        "--result-json",
        str(result_json),
        "--config",
        str(args.config),
        "--label-key",
        args.label_key,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--attention-mode",
        args.attention_mode,
    ]
    for name in ("gan_epochs", "pretrain_epochs", "epochs"):
        value = getattr(args, name)
        if value is not None:
            command.extend([f"--{name.replace('_', '-')}", str(value)])
    if args.resume:
        command.append("--resume")
    if args.force:
        command.append("--force")
    return command


def run_launcher(args: argparse.Namespace) -> int:
    selected = select_input_samples(
        args.data_root.resolve(), args.sources, args.samples_per_source
    )
    sequence_id = args.sequence_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    sequence_dir = args.output_root.resolve() / sequence_id
    records_dir = sequence_dir / "run_records"
    records_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    failures = 0
    for source, input_path in selected:
        for condition in args.conditions:
            run_dir = sequence_dir / "runs" / condition / source / input_path.stem
            result_json = records_dir / f"{source}--{input_path.stem}--{condition}.json"
            command = _worker_command(
                args, source, input_path, condition, run_dir, result_json
            )
            manifest.append(
                {
                    "source": source,
                    "sample": input_path.stem,
                    "condition": condition,
                    "input_h5ad": str(input_path),
                    "run_dir": str(run_dir),
                    "result_json": str(result_json),
                    "command": command,
                }
            )
            print(f"[RUN ] {source}/{input_path.stem} condition={condition}")
            print("      " + subprocess.list2cmdline(command))
            if args.dry_run:
                continue
            if result_json.is_file() and not args.force:
                existing = json.loads(result_json.read_text(encoding="utf-8"))
                if existing.get("status") == "completed":
                    print("[SKIP] completed record already exists")
                    continue
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            write_summaries(sequence_dir)
            if result.returncode:
                failures += 1
                if not args.continue_on_error:
                    break
        if failures and not args.continue_on_error:
            break
    _write_json(sequence_dir / "manifest.json", {"runs": manifest})
    if not args.dry_run:
        write_summaries(sequence_dir)
    print(f"[DONE] output={sequence_dir} failures={failures}")
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run anchor/sparse-graph/sparse-negative SparNiche scaling ablations."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
    parser.add_argument("--source-name")
    parser.add_argument("--input-h5ad", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-fs/data"))
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--samples-per-source", type=int, default=3)
    parser.add_argument(
        "--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS)
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-fs/bench_results/sparse_scaling_ablation"),
    )
    parser.add_argument("--sequence-id")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "config.yaml")
    parser.add_argument("--label-key", default="annotation_final")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attention-mode", choices=("global", "spatial_local"), default="spatial_local"
    )
    parser.add_argument("--gan-epochs", type=int)
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        required = {
            "condition": args.condition,
            "source_name": args.source_name,
            "input_h5ad": args.input_h5ad,
            "run_dir": args.run_dir,
            "result_json": args.result_json,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            parser.error(f"worker mode requires: {', '.join(missing)}")
    return args


def main() -> int:
    args = parse_args()
    return run_worker(args) if args.worker else run_launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
