from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors


DEFAULT_BASELINE_POINT_SIZE = 1.0
_EMPTY_GROUND_TRUTH_LABELS = frozenset(
    {"", "ambiguous", "nan", "na", "n/a", "none", "null", "other", "unknown", "unassigned", "unlabeled"}
)


def _shared_baseline_plotter():
    """Load baseline_common when SparNiche and baselines share a workspace."""
    import importlib
    import sys

    try:
        return importlib.import_module("baseline_common")
    except ImportError:
        pass
    here = Path(__file__).resolve()
    for directory in (here.parents[2] / "baselines", here.parents[3] / "baselines", Path("D:/baselines")):
        if (directory / "baseline_common.py").is_file():
            sys.path.insert(0, str(directory))
            return importlib.import_module("baseline_common")
    return None


def _spatial_labels(
    adata, key: str, fallback: str, *, normalize_ambiguous: bool = False
) -> np.ndarray:
    if key not in adata.obs:
        return np.full(adata.n_obs, fallback, dtype=object)
    labels = adata.obs[key].astype("string").fillna("<NA>").to_numpy(dtype=object)
    if normalize_ambiguous:
        normalized = np.asarray([str(label).strip().lower() for label in labels], dtype=object)
        labels[np.isin(normalized, list(_EMPTY_GROUND_TRUTH_LABELS))] = ""
    return labels


def _metric_subtitle(matched: pd.DataFrame) -> str:
    if matched.empty:
        return ""
    ari = adjusted_rand_score(matched["ground_truth"], matched["pred"])
    nmi = normalized_mutual_info_score(matched["ground_truth"], matched["pred"])
    return f"ARI={ari:.4f}  NMI={nmi:.4f}"


def _valid_ground_truth_mask(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    return np.asarray(
        [str(label) not in {"unlabeled", "<NA>", ""} for label in labels],
        dtype=bool,
    )


def _estimate_spot_diameter(coordinates: np.ndarray) -> float:
    """Match baseline_common.py's fallback spot diameter estimate."""
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2 or coordinates.shape[1] < 2:
        return 1.0
    coordinates = coordinates[:, :2]
    finite = np.isfinite(coordinates).all(axis=1)
    coordinates = coordinates[finite]
    if coordinates.shape[0] < 2:
        return 1.0
    nearest = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(coordinates)
    distances, _ = nearest.kneighbors(coordinates)
    distances = distances[:, 1]
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if distances.size == 0:
        return 1.0
    return float(max(1.0, np.median(distances) * 0.85))


def _coerce_spatial_plot_metadata(adata, coordinates: np.ndarray) -> float:
    """Resolve the same spot diameter that baseline_common passes to Scanpy.

    ``baseline_common.py`` prefers a valid ``spot_diameter_fullres`` already
    stored in ``adata.uns['spatial']`` and otherwise writes its nearest-neighbor
    estimate. Returning the resolved value lets the custom three-panel plot use
    Scanpy's exact no-image marker-size conversion below.
    """
    estimated_diameter = _estimate_spot_diameter(coordinates)
    spatial_uns = adata.uns.get("spatial")
    if not isinstance(spatial_uns, dict) or not spatial_uns:
        adata.uns["spatial"] = {
            "baseline_spatial": {
                "images": {},
                "scalefactors": {
                    "spot_diameter_fullres": estimated_diameter,
                    "tissue_hires_scalef": 1.0,
                    "tissue_lowres_scalef": 1.0,
                },
            }
        }
        return estimated_diameter

    for library in spatial_uns.values():
        if not isinstance(library, dict):
            continue
        scalefactors = library.setdefault("scalefactors", {})
        if not isinstance(scalefactors, dict):
            library["scalefactors"] = {}
            scalefactors = library["scalefactors"]
        current = scalefactors.get("spot_diameter_fullres")
        try:
            current = float(current)
        except (TypeError, ValueError):
            current = 0.0
        if current <= 0 or current < estimated_diameter * 0.25:
            current = estimated_diameter
            scalefactors["spot_diameter_fullres"] = current
        scalefactors.setdefault("tissue_hires_scalef", 1.0)
        scalefactors.setdefault("tissue_lowres_scalef", 1.0)
        library.setdefault("images", {})
        return float(current)

    adata.uns["spatial"] = {
        "baseline_spatial": {
            "images": {},
            "scalefactors": {
                "spot_diameter_fullres": estimated_diameter,
                "tissue_hires_scalef": 1.0,
                "tissue_lowres_scalef": 1.0,
            },
        }
    }
    return estimated_diameter


def _draw_spatial(
    ax,
    coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
    *,
    point_size: float,
    spot_diameter: float,
) -> None:
    categories = sorted({str(label) for label in labels if str(label)})
    colors = plt.get_cmap("tab20", max(len(categories), 1))
    for index, category in enumerate(categories):
        mask = np.asarray([str(label) == category for label in labels])
        radius = float(point_size) * float(spot_diameter) * 0.5
        patches = [
            Circle((x_value, y_value), radius=radius)
            for x_value, y_value in coordinates[mask, :2]
        ]
        if patches:
            # This is the same data-coordinate circle construction used by
            # scanpy.pl.spatial when img_key=None and scale_factor=1.
            ax.add_collection(
                PatchCollection(
                    patches,
                    facecolor=colors(index),
                    edgecolor="none",
                    label=category,
                )
            )
    ax.autoscale_view()
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    if len(categories) <= 12:
        ax.legend(loc="best", fontsize=7, frameon=False, markerscale=1.2)


def _draw_label_flow(ax, evaluation: pd.DataFrame) -> None:
    ax.set_axis_off()
    if evaluation.empty:
        ax.text(0.5, 0.5, "No matched labels", ha="center", va="center")
        return

    table = pd.crosstab(evaluation["ground_truth"], evaluation["pred"])
    left = table.sum(axis=1).sort_values(ascending=False).index.tolist()
    right = table.sum(axis=0).sort_values(ascending=False).index.tolist()
    table = table.loc[left, right]
    left_y = np.linspace(0.9, 0.1, len(left))
    right_y = np.linspace(0.9, 0.1, len(right))
    left_pos = dict(zip(left, left_y))
    right_pos = dict(zip(right, right_y))
    maximum = max(float(table.to_numpy().max()), 1.0)

    for truth in left:
        for predicted in right:
            value = float(table.loc[truth, predicted])
            if value:
                weight = value / maximum
                ax.plot(
                    [0.22, 0.78],
                    [left_pos[truth], right_pos[predicted]],
                    color="#4c72b0",
                    alpha=0.15 + 0.45 * weight,
                    linewidth=0.35 + 7.5 * weight,
                    solid_capstyle="round",
                )
    for label in left:
        ax.scatter([0.18], [left_pos[label]], s=50, color="#2f5f9f")
        ax.text(0.14, left_pos[label], str(label), ha="right", va="center", fontsize=7)
    for label in right:
        ax.scatter([0.82], [right_pos[label]], s=50, color="#dd8452")
        ax.text(0.86, right_pos[label], str(label), ha="left", va="center", fontsize=7)
    ax.text(0.18, 0.98, "Ground truth", ha="center", va="top", fontsize=10, weight="bold")
    ax.text(0.82, 0.98, "Prediction", ha="center", va="top", fontsize=10, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def save_final_outputs(
    adata,
    output_dir: Path,
    *,
    label_key: str,
    cluster_key: str = "sparniche_cluster",
    point_size: float = DEFAULT_BASELINE_POINT_SIZE,
) -> None:
    """Write v7-style final spatial plots and tabular outputs for one run."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    results_dir = output_dir / "results"
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    coordinates = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    spot_diameter = _coerce_spatial_plot_metadata(adata, coordinates)
    truth = _spatial_labels(adata, label_key, "unlabeled", normalize_ambiguous=True)
    predicted = _spatial_labels(adata, cluster_key, "unassigned")
    valid_spots = _valid_ground_truth_mask(truth)
    evaluation = pd.DataFrame(
        {"ground_truth": truth, "pred": predicted}, index=adata.obs_names
    )
    matched = evaluation[
        valid_spots
    ]
    metric_subtitle = _metric_subtitle(matched)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1.0, 0.95, 1.0]})
    _draw_spatial(
        axes[0],
        coordinates[valid_spots],
        truth[valid_spots],
        "Ground Truth",
        point_size=float(point_size),
        spot_diameter=spot_diameter,
    )
    _draw_label_flow(axes[1], matched)
    _draw_spatial(
        axes[2],
        coordinates[valid_spots],
        predicted[valid_spots],
        "SparNiche Prediction" + (f"\n{metric_subtitle}" if metric_subtitle else ""),
        point_size=float(point_size),
        spot_diameter=spot_diameter,
    )
    figure.tight_layout()
    figure.savefig(plots_dir / "final_combined.png", bbox_inches="tight", dpi=300)
    plt.close(figure)

    evaluation.to_csv(results_dir / "final_clusters.csv", index=True, index_label="spot_id")
    pd.DataFrame(np.asarray(adata.obsm["sparniche"]), index=adata.obs_names).to_csv(
        results_dir / "final_embedding.csv", index=True, index_label="spot_id"
    )


def save_benchmark_combined_plot(
    adata,
    output_path: Path,
    *,
    label_key: str,
    cluster_key: str,
    point_size: float = DEFAULT_BASELINE_POINT_SIZE,
) -> None:
    """Write the standard three-panel plot without calculating metrics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    truth = _spatial_labels(adata, label_key, "unlabeled", normalize_ambiguous=True)
    valid_spots = _valid_ground_truth_mask(truth)
    shared_plotter = _shared_baseline_plotter()
    if shared_plotter is not None and valid_spots.any():
        # Reuse the baselines Scanpy renderer while keeping the benchmark's
        # invalid-spot filtering contract.
        plot_adata = adata[valid_spots].copy()
        shared_plotter.save_combined_plot(
            adata=plot_adata,
            sample_name=output_path.stem.removesuffix("_combined"),
            method_title="SparNiche",
            cluster_key=cluster_key,
            output_path=output_path,
            ground_truth_key=label_key,
            point_size=float(point_size),
        )
        return
    coordinates = np.asarray(adata.obsm["spatial"], dtype=float)[:, :2]
    spot_diameter = _coerce_spatial_plot_metadata(adata, coordinates)
    predicted = _spatial_labels(adata, cluster_key, "unassigned")
    evaluation = pd.DataFrame({"ground_truth": truth, "pred": predicted})
    matched = evaluation[valid_spots]
    metric_subtitle = _metric_subtitle(matched)
    figure, axes = plt.subplots(
        1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1.0, 0.95, 1.0]}
    )
    _draw_spatial(
        axes[0], coordinates[valid_spots], truth[valid_spots], "Ground Truth", point_size=float(point_size), spot_diameter=spot_diameter
    )
    _draw_label_flow(axes[1], matched)
    _draw_spatial(
        axes[2], coordinates[valid_spots], predicted[valid_spots],
        "Prediction" + (f"\n{metric_subtitle}" if metric_subtitle else ""),
        point_size=float(point_size), spot_diameter=spot_diameter
    )
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(figure)
