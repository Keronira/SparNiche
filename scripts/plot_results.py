#!/usr/bin/env python3
"""Render SparNiche combined plots from existing benchmark results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import anndata
import numpy as np
import pandas as pd


def _import_baseline_common():
    """Load the shared baselines plotting helpers in both repo layouts."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "baselines",
        here.parents[3] / "baselines",
        Path("D:/baselines"),
    ]
    for directory in candidates:
        module_path = directory / "baseline_common.py"
        if module_path.is_file():
            sys.path.insert(0, str(directory))
            import baseline_common  # type: ignore

            return baseline_common
    raise ImportError("Cannot locate baseline_common.py; expected a sibling baselines directory.")


AMBIGUOUS_LABELS = frozenset(
    {"", "ambiguous", "nan", "na", "n/a", "none", "null", "other", "unknown", "unassigned", "unlabeled"}
)


def result_output_path(result_path: Path, output_dir: Path | None = None) -> Path:
    """Return the combined-plot path for one prediction CSV."""
    result_path = Path(result_path)
    destination = Path(output_dir) if output_dir is not None else result_path.parent.parent / "plots"
    return destination / f"{result_path.stem}_combined.png"


def _valid_result_mask(result: pd.DataFrame) -> np.ndarray:
    labels = result["ground_truth"].astype("string")
    normalized = labels.fillna("").str.strip().str.lower()
    return (~normalized.isin(AMBIGUOUS_LABELS)).to_numpy(dtype=bool)


def align_result_to_adata(
    adata,
    result: pd.DataFrame,
    *,
    ground_truth_key: str = "annotation_final",
):
    """Align result rows to source observations and remove invalid labels."""
    required = {"spot_id", "ground_truth", "pred"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Result CSV is missing required columns: {', '.join(sorted(missing))}")
    obs_ids = adata.obs_names.astype(str)
    if obs_ids.duplicated().any():
        raise ValueError("Source AnnData has duplicate observation names")
    lookup = {value: index for index, value in enumerate(obs_ids)}
    result = result.copy()
    result["spot_id"] = result["spot_id"].astype(str)
    valid = _valid_result_mask(result)
    result = result.loc[valid].reset_index(drop=True)
    missing_ids = sorted(set(result["spot_id"]) - set(lookup))
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise KeyError(f"Result CSV contains spot IDs absent from source AnnData: {preview}")
    if result["spot_id"].duplicated().any():
        raise ValueError("Result CSV contains duplicate spot_id values")
    indices = [lookup[value] for value in result["spot_id"]]
    aligned = adata[indices].copy()
    if "spatial" not in aligned.obsm and "X_spatial" in aligned.obsm:
        aligned.obsm["spatial"] = np.asarray(aligned.obsm["X_spatial"]).copy()
    if "spatial" not in aligned.obsm:
        raise KeyError("Source AnnData requires obsm['spatial'] or obsm['X_spatial']")
    aligned.obs[ground_truth_key] = result["ground_truth"].to_numpy()
    aligned.obs["plot_ground_truth"] = result["ground_truth"].to_numpy()
    aligned.obs["plot_prediction"] = result["pred"].astype(str).to_numpy()
    return aligned


def _result_csvs(results_dir: Path, samples: Iterable[str] | None = None) -> list[Path]:
    wanted = set(samples or [])
    auxiliary = ("_embedding", "_profile", "_banksy_origin_")
    paths = [
        path
        for path in sorted(Path(results_dir).glob("*.csv"))
        if not path.stem.endswith(auxiliary) and not any(token in path.stem for token in ("_embedding", "_profile"))
    ]
    if wanted:
        paths = [path for path in paths if path.stem in wanted]
        found = {path.stem for path in paths}
        missing = sorted(wanted - found)
        if missing:
            raise FileNotFoundError(f"Missing result sample(s): {', '.join(missing)}")
    if not paths:
        raise FileNotFoundError(f"No prediction CSV files found in {results_dir}")
    return paths


def plot_results_directory(
    results_dir: Path,
    data_dir: Path,
    *,
    output_dir: Path | None = None,
    point_size: float = 1.0,
    ground_truth_key: str = "annotation_final",
    method_title: str | None = None,
    samples: Iterable[str] | None = None,
) -> list[Path]:
    """Render one combined plot per results CSV using the baselines renderer."""
    if point_size <= 0:
        raise ValueError("point_size must be positive")
    baseline_common = _import_baseline_common()
    results_dir = Path(results_dir)
    data_dir = Path(data_dir)
    title = method_title or results_dir.parent.name
    outputs: list[Path] = []
    for result_path in _result_csvs(results_dir, samples):
        sample = result_path.stem
        source_path = data_dir / f"{sample}.h5ad"
        if not source_path.is_file():
            raise FileNotFoundError(f"Source AnnData not found for {sample}: {source_path}")
        result = pd.read_csv(result_path)
        adata = align_result_to_adata(anndata.read_h5ad(source_path), result, ground_truth_key=ground_truth_key)
        output_path = result_output_path(result_path, output_dir)
        baseline_common.save_combined_plot(
            adata=adata,
            sample_name=sample,
            method_title=title,
            cluster_key="plot_prediction",
            output_path=output_path,
            ground_truth_key="plot_ground_truth",
            point_size=float(point_size),
        )
        outputs.append(output_path)
        print(f"[PLOT] {sample}: {output_path}")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory containing prediction CSV files.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing matching .h5ad files.")
    parser.add_argument("--output-dir", type=Path, help="Destination for combined PNGs. Defaults to results sibling plots/.")
    parser.add_argument(
        "--point-size", "--spot-size", dest="point_size", type=float, default=1.0,
        help="Scanpy marker scale (spot size); default: 1.0.",
    )
    parser.add_argument("--ground-truth-key", default="annotation_final")
    parser.add_argument("--method-title", help="Title shown on prediction panel. Defaults to results parent name.")
    parser.add_argument("--samples", nargs="+", help="Optional sample basenames, without .csv/.h5ad.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_results_directory(
        args.results_dir,
        args.data_dir,
        output_dir=args.output_dir,
        point_size=args.point_size,
        ground_truth_key=args.ground_truth_key,
        method_title=args.method_title,
        samples=args.samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
