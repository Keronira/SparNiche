from __future__ import annotations

import hashlib
import json
import platform
try:
    import resource
except ImportError:  # pragma: no cover - Windows development environments
    resource = None
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata
import numpy as np
import scanpy as sc
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .config import expand_environment, load_yaml, normalize_sparniche_config, save_yaml
from .benchmark import AMBIGUOUS_GROUND_TRUTH_LABELS, export_benchmark_result
from .evaluation import clustering_metrics
from .pipeline import train_adata
from .plotting import save_benchmark_combined_plot
from .trainer import _checkpoint_config_identity

warnings.filterwarnings("ignore")

DEFAULT_LEIDEN_CLUSTER_UPPER_OFFSET = 2
LEIDEN_TRAVERSAL_MISS_TOLERANCE = 3


def _cached_leiden_labels(
    labels_by_midpoint: dict[int, np.ndarray], midpoint: int
) -> np.ndarray | None:
    """Return labels from the search pass when the selected point was evaluated."""
    labels = labels_by_midpoint.get(int(midpoint))
    return None if labels is None else np.asarray(labels, dtype=str).copy()


def _select_resolution_candidate(
    records: list[dict],
    target_clusters: int,
    truth,
    *,
    cluster_lower_offset: int = 0,
    cluster_upper_offset: int = DEFAULT_LEIDEN_CLUSTER_UPPER_OFFSET,
) -> dict:
    """Choose the ARI-best candidate within the configured cluster-count range."""
    target = int(target_clusters)
    eligible = [
        row
        for row in records
        if target + int(cluster_lower_offset)
        <= int(row["cluster_count"])
        <= target + int(cluster_upper_offset)
    ]
    pool = eligible if eligible else records
    if truth is not None:
        truth_values = np.asarray(truth).astype(str)
        for row in pool:
            predicted = np.asarray(row["labels"]).astype(str)
            row["ari"] = float(adjusted_rand_score(truth_values, predicted))
            row["nmi"] = float(normalized_mutual_info_score(truth_values, predicted))
        selected = max(
            pool,
            key=lambda row: (
                row["ari"],
                row["nmi"],
                -abs(int(row["cluster_count"]) - target),
                -float(row["resolution"]),
            ),
        )
        selected["window_fallback"] = not bool(eligible)
        return selected
    selected = min(
        pool,
        key=lambda row: (
            abs(int(row["cluster_count"]) - target),
            float(row["resolution"]),
        ),
    )
    selected["window_fallback"] = not bool(eligible)
    return selected


def _should_interrupt_leiden_search(
    cluster_count: int,
    target_clusters: int,
    cluster_upper_offset: int = DEFAULT_LEIDEN_CLUSTER_UPPER_OFFSET,
) -> bool:
    """Stop scanning when the current partition exceeds the cluster-count window."""
    return (
        int(cluster_count)
        > int(target_clusters) + int(cluster_upper_offset)
    )


def _leiden_resolution_midpoints(
    min_resolution: float, max_resolution: float
) -> list[int]:
    """Return the inclusive 0.01-resolution grid within configured limits."""
    minimum = int(round(float(min_resolution) * 100))
    maximum = int(round(float(max_resolution) * 100))
    if minimum < 1:
        raise ValueError("evaluation.leiden_min_resolution must be at least 0.01")
    if maximum < minimum:
        raise ValueError("evaluation.leiden_max_resolution must not be below the minimum")
    return list(range(minimum, maximum + 1))


def _leiden_resolution_probe_points(minimum: int, maximum: int) -> list[int]:
    """Return the mandatory upper, middle, and lower resolution probes."""
    minimum = int(minimum)
    maximum = int(maximum)
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid Leiden resolution bounds")
    midpoint = (minimum + maximum) // 2
    return list(dict.fromkeys((maximum, midpoint, minimum)))


def _leiden_search_direction(
    midpoint_clusters: int,
    target_clusters: int,
    cluster_lower_offset: int = 0,
    cluster_upper_offset: int = DEFAULT_LEIDEN_CLUSTER_UPPER_OFFSET,
) -> str:
    """Infer which half can contain the configured cluster-count window."""
    lower = int(target_clusters) + int(cluster_lower_offset)
    upper = int(target_clusters) + int(cluster_upper_offset)
    clusters = int(midpoint_clusters)
    if clusters < lower:
        return "higher"
    if clusters > upper:
        return "lower"
    return "midpoint"


def _leiden_directional_bounds(
    minimum: int,
    midpoint: int,
    maximum: int,
    direction: str,
) -> tuple[int, int]:
    """Return the initial interval on the side directed toward the target."""
    minimum = int(minimum)
    midpoint = int(midpoint)
    maximum = int(maximum)
    if direction == "higher":
        return midpoint + 1, maximum
    if direction == "lower":
        return minimum, midpoint - 1
    if direction == "midpoint":
        return midpoint, midpoint
    raise ValueError(f"unknown Leiden search direction: {direction!r}")


def _leiden_update_directional_bounds(
    lower: int,
    upper: int,
    probe: int,
    probe_clusters: int,
    target_clusters: int,
    direction: str,
    cluster_lower_offset: int = 0,
    cluster_upper_offset: int = DEFAULT_LEIDEN_CLUSTER_UPPER_OFFSET,
) -> tuple[int, int]:
    """Update a directional interval after one midpoint probe."""
    lower = int(lower)
    upper = int(upper)
    probe = int(probe)
    clusters = int(probe_clusters)
    target_lower = int(target_clusters) + int(cluster_lower_offset)
    target_upper = int(target_clusters) + int(cluster_upper_offset)
    if direction == "higher":
        if clusters < target_lower:
            return probe + 1, upper
        return lower, probe - 1
    if direction == "lower":
        if clusters > target_upper:
            return lower, probe - 1
        return probe + 1, upper
    if direction == "midpoint":
        return probe, probe
    raise ValueError(f"unknown Leiden search direction: {direction!r}")


REQUIRED_RUN_ARTIFACTS = (
    "resolved_config.yaml",
    "run_metadata.json",
    "checkpoint.pt",
    "epoch_metrics.jsonl",
    "trained.h5ad",
)


def _ground_truth_valid_mask(adata, label_key: str) -> np.ndarray:
    if label_key not in adata.obs:
        return np.zeros(adata.n_obs, dtype=bool)
    labels = adata.obs[label_key]
    normalized = labels.astype("string").fillna("").str.strip().str.lower()
    return ~normalized.isin(AMBIGUOUS_GROUND_TRUTH_LABELS).to_numpy()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_value(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _source_identity() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    digest = hashlib.sha256()
    for source in sorted((project_root / "src").glob("*.py")):
        digest.update(source.name.encode("utf-8"))
        digest.update(source.read_bytes())
    return {"git_revision": revision, "source_sha256": digest.hexdigest()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_completed_run(
    run_dir: Path, resolved: dict[str, Any], status: dict[str, Any]
) -> None:
    required = REQUIRED_RUN_ARTIFACTS
    if not bool(resolved.get("benchmark", {}).get("write_h5ad", True)):
        required = tuple(name for name in required if name != "trained.h5ad")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "completed run is missing required artifacts: " + ", ".join(missing)
        )
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("config_identity_sha256") != _checkpoint_config_identity(resolved):
        raise RuntimeError("completed run configuration differs from the requested configuration; use --force")
    input_path = Path(resolved["paths"]["input_h5ad"])
    if not input_path.is_file():
        raise RuntimeError(f"completed run input data is unavailable: {input_path}")
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("input_file_sha256") != _file_sha256(input_path):
        raise RuntimeError("completed run input data fingerprint has changed; use --force")
    if status.get("state") != "completed":
        raise RuntimeError("run status is not completed")




def _leiden_labels(
    embedding: np.ndarray,
    adata,
    *,
    target_clusters: int,
    seed: int,
    label_key: str | None = None,
    min_resolution: float = 0.01,
    max_resolution: float = 1.99,
    cluster_lower_offset: int = 0,
    cluster_upper_offset: int = DEFAULT_LEIDEN_CLUSTER_UPPER_OFFSET,
) -> np.ndarray:
    """Probe bounds, then run at most two directional midpoint refinements."""
    representation_key = "_sparniche_leiden_embedding"
    adata.obsm[representation_key] = np.asarray(embedding, dtype=np.float32)
    sc.pp.neighbors(
        adata,
        use_rep=representation_key,
        key_added=representation_key,
    )
    target = max(1, min(int(target_clusters), int(adata.n_obs)))
    resolution_midpoints = _leiden_resolution_midpoints(
        min_resolution, max_resolution
    )
    minimum_resolution = resolution_midpoints[0]
    maximum_resolution = resolution_midpoints[-1]
    midpoint_resolution = (minimum_resolution + maximum_resolution) // 2
    probe_points = _leiden_resolution_probe_points(
        minimum_resolution, maximum_resolution
    )

    key = "_sparniche_leiden"
    print(
        f"[EVAL] Leiden resolution search: target_clusters={target}, "
        "strategy=probe_then_window_traversal, "
        f"cluster_window=K{int(cluster_lower_offset):+d}.."
        f"K+{int(cluster_upper_offset)}, "
        "selection=ARI-best-in-window, "
        f"resolution_grid={resolution_midpoints[0] / 100.0:.2f}.."
        f"{resolution_midpoints[-1] / 100.0:.2f} step=0.01, "
        "initial_probes=upper,middle,lower",
        flush=True,
    )
    evaluations = 0
    records: list[dict] = []
    truth = None
    truth_mask = None
    if label_key is not None and label_key in adata.obs:
        valid = _ground_truth_valid_mask(adata, label_key)
        if valid.any():
            truth_mask = valid
            truth = adata.obs[label_key].astype(str).to_numpy()

    records_by_midpoint: dict[int, dict] = {}

    def evaluate_resolution(midpoint: int) -> dict:
        nonlocal evaluations
        resolution = midpoint / 100.0
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key,
            neighbors_key=representation_key,
            random_state=int(seed),
        )
        evaluations += 1
        cluster_count = int(adata.obs[key].nunique())
        labels = adata.obs[key].astype(str).to_numpy().copy()
        records.append({
            "resolution": resolution,
            "cluster_count": cluster_count,
            "labels": labels,
        })
        records_by_midpoint[int(midpoint)] = records[-1]
        if truth is not None and truth_mask is not None:
            step_labels = labels[truth_mask]
            step_ari = float(adjusted_rand_score(truth[truth_mask], step_labels))
            step_nmi = float(normalized_mutual_info_score(truth[truth_mask], step_labels))
            metric_suffix = f", ARI={step_ari:.4f}, NMI={step_nmi:.4f}"
        else:
            metric_suffix = ", ARI=NA, NMI=NA"
        print(
            f"[EVAL] Leiden step: resolution={resolution:.2f}, "
            f"labels={cluster_count}{metric_suffix}",
            flush=True,
        )
        return records[-1]

    target_lower = target + int(cluster_lower_offset)
    target_upper = target + int(cluster_upper_offset)

    def in_target_window(cluster_count: int) -> bool:
        return target_lower <= int(cluster_count) <= target_upper

    # Probe the upper, middle, and lower bounds before deciding which half to scan.
    for midpoint in probe_points:
        evaluate_resolution(midpoint)

    midpoint_record = records_by_midpoint[midpoint_resolution]
    direction = _leiden_search_direction(
        midpoint_record["cluster_count"],
        target,
        cluster_lower_offset=cluster_lower_offset,
        cluster_upper_offset=cluster_upper_offset,
    )
    valid_seed: int | None = None
    # Only a true midpoint can terminate the continuous midpoint search.
    if in_target_window(midpoint_record["cluster_count"]):
        valid_seed = int(midpoint_resolution)

    directional_lower, directional_upper = _leiden_directional_bounds(
        minimum_resolution,
        midpoint_resolution,
        maximum_resolution,
        direction,
    )
    directional_rounds = 0
    print(
        f"[EVAL] Leiden directional search: midpoint_labels="
        f"{midpoint_record['cluster_count']}, direction={direction}, "
        "mode=continuous_midpoint_until_window",
        flush=True,
    )
    while valid_seed is None and direction != "midpoint":
        if directional_lower > directional_upper:
            break
        directional_probe = (directional_lower + directional_upper) // 2
        probe_record = evaluate_resolution(directional_probe)
        directional_rounds += 1
        if in_target_window(probe_record["cluster_count"]):
            valid_seed = int(directional_probe)
            print(
                f"[EVAL] Leiden midpoint search hit target window at "
                f"resolution={directional_probe / 100.0:.2f}, "
                f"labels={probe_record['cluster_count']}, rounds={directional_rounds}",
                flush=True,
            )
            break
        directional_lower, directional_upper = _leiden_update_directional_bounds(
            directional_lower,
            directional_upper,
            directional_probe,
            probe_record["cluster_count"],
            target,
            direction,
            cluster_lower_offset=cluster_lower_offset,
            cluster_upper_offset=cluster_upper_offset,
        )
        print(
            f"[EVAL] Leiden midpoint round {directional_rounds}: "
            f"resolution={directional_probe / 100.0:.2f}, "
            f"labels={probe_record['cluster_count']}, "
            f"next_interval={directional_lower / 100.0:.2f}.."
            f"{directional_upper / 100.0:.2f}",
            flush=True,
        )

    # If no midpoint ever enters the window, retain a valid endpoint as a
    # boundary fallback before falling back to the closest cluster count.
    if valid_seed is None:
        for candidate in (maximum_resolution, minimum_resolution):
            record = records_by_midpoint[candidate]
            if in_target_window(record["cluster_count"]):
                valid_seed = int(candidate)
                print(
                    f"[EVAL] Leiden midpoint search missed window; using "
                    f"valid boundary resolution={candidate / 100.0:.2f}",
                    flush=True,
                )
                break

    # Once a valid resolution is found, walk both directions one grid step at a
    # time. Leiden can fluctuate locally, so a single count outside the window
    # is not enough to terminate a directional traversal.
    if valid_seed is not None:
        for step in (-1, 1):
            current = valid_seed + step
            terminal_misses = 0
            while minimum_resolution <= current <= maximum_resolution:
                record = records_by_midpoint.get(current)
                if record is None:
                    record = evaluate_resolution(current)
                cluster_count = int(record["cluster_count"])
                if in_target_window(cluster_count):
                    terminal_misses = 0
                else:
                    past_terminal_side = (
                        step < 0 and cluster_count < target_lower
                    ) or (step > 0 and cluster_count > target_upper)
                    if past_terminal_side:
                        terminal_misses += 1
                    else:
                        # This is a non-monotonic excursion on the opposite
                        # side of the window; keep searching in this direction.
                        terminal_misses = 0
                    if terminal_misses >= LEIDEN_TRAVERSAL_MISS_TOLERANCE:
                        print(
                            f"[EVAL] Leiden window traversal stopped: "
                            f"direction={'lower' if step < 0 else 'upper'}, "
                            f"consecutive_terminal_misses={terminal_misses}",
                            flush=True,
                        )
                        break
                current += step
        print(
            f"[EVAL] Leiden window traversal complete: seed_resolution="
            f"{valid_seed / 100.0:.2f}, rounds={directional_rounds}",
            flush=True,
        )
    else:
        print(
            "[EVAL] Leiden target window not reached; using closest-count fallback",
            flush=True,
        )
    if truth is not None and truth_mask is not None:
        selection_records = [
            {**row, "_record_index": index, "labels": np.asarray(row["labels"])[truth_mask]}
            for index, row in enumerate(records)
        ]
        selected = _select_resolution_candidate(
            selection_records,
            target,
            truth[truth_mask],
            cluster_lower_offset=cluster_lower_offset,
            cluster_upper_offset=cluster_upper_offset,
        )
        selected_index = int(selected["_record_index"])
        selected["labels"] = records[selected_index]["labels"]
    else:
        selected = _select_resolution_candidate(
            records,
            target,
            None,
            cluster_lower_offset=cluster_lower_offset,
            cluster_upper_offset=cluster_upper_offset,
        )
    selected_resolution = float(selected["resolution"])
    print(
        f"[EVAL] Leiden resolution search complete: evaluations={evaluations}, "
        f"selected_resolution={selected_resolution:.2f}, "
        f"selected_clusters={selected['cluster_count']}"
        + (f", best_ARI={selected['ari']:.4f}, best_NMI={selected['nmi']:.4f}" if "ari" in selected else ""),
        flush=True,
    )
    adata.obs[key] = np.asarray(selected["labels"], dtype=str)
    audit_candidates = []
    for row in records:
        candidate = {
            key_name: value
            for key_name, value in row.items()
            if key_name not in {"labels", "per_layer_iou"}
        }
        if "per_layer_iou" in row:
            candidate["per_layer_iou"] = dict(row["per_layer_iou"])
        candidate["eligible"] = target_lower <= int(row["cluster_count"]) <= target_upper
        candidate["selected"] = float(row["resolution"]) == selected_resolution
        audit_candidates.append(candidate)
    adata.uns["sparniche_leiden_search"] = {
        "target_clusters": target,
        "cluster_lower_offset": int(cluster_lower_offset),
        "cluster_upper_offset": int(cluster_upper_offset),
        "selection_rule": "ari_best_in_window",
        "window_fallback": bool(selected.get("window_fallback", False)),
        "selected_resolution": selected_resolution,
        "selected_clusters": int(selected["cluster_count"]),
        "candidates": audit_candidates,
    }
    print(f"[EVAL] Leiden selected cached result: clusters={selected['cluster_count']}", flush=True)
    return adata.obs[key].astype(str).to_numpy()


def _predict_clusters(
    embedding: np.ndarray,
    adata,
    config: dict[str, Any],
) -> np.ndarray:
    evaluation = config.get("evaluation", {})
    label_key = config.get("data", {}).get("label_key")
    if label_key and label_key in adata.obs:
        valid = _ground_truth_valid_mask(adata, label_key)
        truth = adata.obs[label_key].astype(str).to_numpy()[valid]
        inferred_clusters = int(np.unique(truth).size)
    else:
        valid = np.zeros(adata.n_obs, dtype=bool)
        truth = np.asarray([])
        inferred_clusters = 0
    requested_clusters = int(evaluation.get("n_clusters", 0))
    n_clusters = requested_clusters if requested_clusters > 0 else inferred_clusters
    if n_clusters <= 0:
        raise ValueError("evaluation.n_clusters is required when labels are unavailable")
    n_clusters = min(n_clusters, adata.n_obs)
    seed = int(config.get("training", {}).get("seed", 2023))
    model_variant = str(config.get("model", {}).get("variant", "local_graph_normalized"))
    if model_variant == "view1" or model_variant.startswith("local_graph_"):
        predicted = _leiden_labels(
            embedding,
            adata,
            target_clusters=n_clusters,
            seed=int(evaluation.get("sparniche_leiden_seed", 2023)),
            label_key=str(config.get("data", {}).get("label_key", "")) or None,
            min_resolution=float(evaluation.get("leiden_min_resolution", 0.01)),
            max_resolution=float(evaluation.get("leiden_max_resolution", 1.99)),
            cluster_lower_offset=int(
                evaluation.get("leiden_cluster_lower_offset", 0)
            ),
            cluster_upper_offset=int(
                evaluation.get(
                    "leiden_cluster_upper_offset",
                    DEFAULT_LEIDEN_CLUSTER_UPPER_OFFSET,
                )
            ),
        )
    else:
        predicted = KMeans(
            n_clusters=n_clusters,
            n_init=int(evaluation.get("kmeans_n_init", 20)),
            random_state=seed,
        ).fit_predict(embedding)
    return predicted


def _cluster_metrics(
    embedding: np.ndarray,
    adata,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float | None]]:
    predicted = _predict_clusters(embedding, adata, config)
    label_key = config.get("data", {}).get("label_key")
    if label_key and label_key in adata.obs:
        valid = _ground_truth_valid_mask(adata, label_key)
        truth = adata.obs[label_key].astype(str).to_numpy()[valid]
    else:
        valid = np.zeros(adata.n_obs, dtype=bool)
        truth = np.asarray([])
    if valid.any():
        scores = clustering_metrics(truth, predicted[valid])
        return predicted, {key: _json_value(value) for key, value in scores.items()}
    return predicted, {"ari": None, "nmi": None, "ami": None}


def _cpu_peak_mb() -> float:
    # Linux reports ru_maxrss in KiB; keep the exported profile in MiB.
    if resource is None:
        return 0.0
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _write_h5ad_with_serializable_leiden_audit(
    adata: anndata.AnnData, output_path: Path
) -> None:
    """Write H5AD without passing nested candidate records to h5py."""
    audit = adata.uns.get("sparniche_leiden_search")
    if not isinstance(audit, dict) or not isinstance(audit.get("candidates"), list):
        adata.write_h5ad(output_path)
        return

    candidates = audit.pop("candidates")
    missing = object()
    previous_json = audit.get("candidates_json", missing)
    audit["candidates_json"] = json.dumps(
        candidates, ensure_ascii=False, sort_keys=True
    )
    try:
        adata.write_h5ad(output_path)
    finally:
        audit["candidates"] = candidates
        if previous_json is missing:
            audit.pop("candidates_json", None)
        else:
            audit["candidates_json"] = previous_json


def _run_external_benchmark(
    artifacts,
    resolved: dict[str, Any],
    run_dir: Path,
    started_at: str,
    start_time: float,
) -> dict[str, Any]:
    benchmark = resolved.get("benchmark", {})
    label_key = str(resolved.get("data", {}).get("label_key", ""))
    if not label_key or label_key not in artifacts.adata.obs:
        raise KeyError(f"benchmark export requires adata.obs[{label_key!r}]")
    embedding = artifacts.clean_output.embedding.detach().cpu().numpy()
    predicted = _predict_clusters(embedding, artifacts.adata, resolved)
    method_root = Path(benchmark.get("method_root") or run_dir.parents[1])
    sample_name = str(benchmark.get("sample_name") or Path(resolved["paths"]["input_h5ad"]).stem)
    elapsed = time.perf_counter() - start_time
    export_benchmark_result(
        method_root,
        sample_name,
        artifacts.adata.obs_names.astype(str).tolist(),
        artifacts.adata.obs[label_key].astype(str).tolist(),
        predicted.astype(str).tolist(),
        embedding,
        runtime_seconds=elapsed,
        cpu_rss_peak_mb=_cpu_peak_mb(),
    )
    artifacts.adata.obs["sparniche_cluster"] = predicted.astype(str)
    if bool(benchmark.get("write_plot", True)):
        save_benchmark_combined_plot(
            artifacts.adata,
            method_root / "plots" / f"{sample_name}_combined.png",
            label_key=label_key,
            cluster_key="sparniche_cluster",
        )
    if bool(benchmark.get("write_h5ad", True)):
        _write_h5ad_with_serializable_leiden_audit(
            artifacts.adata, run_dir / "trained.h5ad"
        )
    status = {
        "state": "completed",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "runtime_seconds": float(elapsed),
        "benchmark_result": str(method_root / "results" / f"{sample_name}.csv"),
    }
    _atomic_json(run_dir / "status.json", status)
    return {"skipped": False, "status": status}












def run_experiment(
    config: dict[str, Any],
    run_dir: Path,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    if status_path.is_file() and not force:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "completed":
            resolved_for_skip = normalize_sparniche_config(expand_environment(config))
            _validate_completed_run(run_dir, resolved_for_skip, status)
            return {"skipped": True, "status": status}

    started_at = _utc_now()
    start_time = time.perf_counter()
    _atomic_json(status_path, {"state": "running", "started_at": started_at})
    try:
        resolved = normalize_sparniche_config(expand_environment(config))
        save_yaml(resolved, run_dir / "resolved_config.yaml")
        input_path = Path(resolved["paths"]["input_h5ad"])
        if not input_path.is_file():
            raise FileNotFoundError(f"input AnnData does not exist: {input_path}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        metadata = {
            "started_at": started_at,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "seed": int(resolved.get("training", {}).get("seed", 2023)),
            "input_file_sha256": _file_sha256(input_path),
            **_source_identity(),
        }
        _atomic_json(run_dir / "run_metadata.json", metadata)
        adata = anndata.read_h5ad(input_path)
        artifacts = train_adata(adata, resolved, run_dir, resume=resume)
        metadata["data_fingerprint"] = artifacts.adata.uns["sparniche_training"][
            "data_fingerprint"
        ]
        _atomic_json(run_dir / "run_metadata.json", metadata)
        return _run_external_benchmark(
            artifacts, resolved, run_dir, started_at, start_time
        )

    except Exception as error:
        _atomic_json(
            status_path,
            {
                "state": "failed",
                "started_at": started_at,
                "failed_at": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
