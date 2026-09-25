"""Metrics for final embedding evaluation using Leiden predictions."""

from __future__ import annotations

import re
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


def _pairwise_spearman(left: np.ndarray, right: np.ndarray, seed: int = 0) -> float:
    n = int(left.shape[0])
    if n < 3:
        return float("nan")
    total = n * (n - 1) // 2
    rng = np.random.default_rng(int(seed))
    if total <= 10000:
        rows, cols = np.triu_indices(n, k=1)
    else:
        rows = rng.integers(0, n, size=10000)
        cols = rng.integers(0, n, size=10000)
        keep = rows != cols
        rows, cols = rows[keep], cols[keep]
    value = spearmanr(
        np.linalg.norm(left[rows] - left[cols], axis=1),
        np.linalg.norm(right[rows] - right[cols], axis=1),
    ).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _neighbor_overlap(embedding: np.ndarray, spatial: np.ndarray, k: int) -> float:
    n = embedding.shape[0]
    if n < 2:
        return float("nan")
    width = min(int(k) + 1, n)
    emb = NearestNeighbors(n_neighbors=width).fit(embedding).kneighbors(
        embedding, return_distance=False
    )[:, 1:]
    spa = NearestNeighbors(n_neighbors=width).fit(spatial).kneighbors(
        spatial, return_distance=False
    )[:, 1:]
    values = []
    for left, right in zip(emb, spa):
        a, b = set(left.tolist()), set(right.tolist())
        union = a | b
        values.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(values))


def _hungarian_alignment(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    truth_values, truth_ids = np.unique(truth, return_inverse=True)
    predicted_values, predicted_ids = np.unique(predicted, return_inverse=True)
    table = np.zeros((len(truth_values), len(predicted_values)), dtype=np.int64)
    np.add.at(table, (truth_ids, predicted_ids), 1)
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(col): int(row) for row, col in zip(rows, cols)}
    fallback = int(np.argmax(table.sum(axis=1)))
    aligned_ids = np.asarray([mapping.get(int(value), fallback) for value in predicted_ids])
    return truth_values[aligned_ids]


def _layer_ordinal(label: object) -> float:
    text = str(label).strip().lower().replace(" ", "")
    match = re.search(r"layer([1-6])", text)
    if match:
        return float(match.group(1))
    if text in {"wm", "whitematter", "white_matter"}:
        return 7.0
    return float("nan")


def compute_layer_recovery_metrics(
    labels: np.ndarray,
    predicted: np.ndarray,
    *,
    recovery_threshold: float = 0.5,
) -> dict[str, object]:
    """Measure class-mask recovery while allowing pure over-segmentation.

    Each predicted cluster is assigned to the true layer with which it has the
    largest overlap. Multiple predicted clusters may therefore reconstruct one
    layer, while one merged cluster can never reconstruct multiple layers.
    """
    labels = np.asarray(labels).astype(str)
    predicted = np.asarray(predicted).astype(str)
    if labels.ndim != 1 or predicted.ndim != 1 or labels.shape != predicted.shape:
        raise ValueError("labels and predicted must be matching one-dimensional arrays")
    if labels.size == 0:
        raise ValueError("at least one labeled spot is required")
    threshold = float(recovery_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("recovery_threshold must be between zero and one")

    true_layers = np.unique(labels)
    reconstructed = np.empty(labels.shape, dtype=object)
    for cluster in np.unique(predicted):
        cluster_mask = predicted == cluster
        overlap = np.asarray(
            [np.count_nonzero(cluster_mask & (labels == layer)) for layer in true_layers]
        )
        reconstructed[cluster_mask] = true_layers[int(np.argmax(overlap))]

    per_layer_iou: dict[str, float] = {}
    for layer in true_layers:
        truth_mask = labels == layer
        predicted_mask = reconstructed == layer
        intersection = int(np.count_nonzero(truth_mask & predicted_mask))
        union = int(np.count_nonzero(truth_mask | predicted_mask))
        per_layer_iou[str(layer)] = float(intersection / union) if union else 0.0
    values = np.asarray(list(per_layer_iou.values()), dtype=float)
    return {
        "per_layer_iou": per_layer_iou,
        "macro_layer_iou": float(values.mean()),
        "worst_layer_iou": float(values.min()),
        "layer_recovery_rate": float(np.mean(values >= threshold)),
        "layer_recovery_threshold": threshold,
    }


def compute_final_metrics(
    embedding: np.ndarray,
    labels: np.ndarray,
    spatial: np.ndarray,
    *,
    predicted: np.ndarray,
    seed: int,
    n_neighbors: int = 12,
) -> dict[str, float]:
    embedding = np.asarray(embedding, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    spatial = np.asarray(spatial, dtype=np.float32)
    if embedding.ndim != 2 or labels.ndim != 1 or spatial.ndim != 2:
        raise ValueError("embedding, labels, and spatial must have valid ranks")
    if embedding.shape[0] != labels.size or spatial.shape[0] != labels.size:
        raise ValueError("embedding, labels, and spatial must have matching rows")
    if labels.size < 3 or np.unique(labels).size < 2:
        raise ValueError("at least two labels and three spots are required")

    predicted = np.asarray(predicted)
    if predicted.ndim != 1 or predicted.shape[0] != labels.size:
        raise ValueError("predicted labels must match the number of spots")
    aligned = _hungarian_alignment(labels, predicted)
    layer_recovery = compute_layer_recovery_metrics(labels, predicted)
    distance_spearman = _pairwise_spearman(embedding, spatial, seed=int(seed))
    overlap = _neighbor_overlap(embedding, spatial, int(n_neighbors))
    fidelity = float(np.mean([overlap, (distance_spearman + 1.0) / 2.0]))

    truth_order = np.asarray([_layer_ordinal(label) for label in labels], dtype=float)
    predicted_order = np.asarray([_layer_ordinal(label) for label in aligned], dtype=float)
    valid_order = np.isfinite(truth_order) & np.isfinite(predicted_order)
    layer_order = (
        float(spearmanr(truth_order[valid_order], predicted_order[valid_order]).statistic)
        if valid_order.sum() >= 3
        else float("nan")
    )
    return {
        "ari": float(adjusted_rand_score(labels, predicted)),
        "nmi": float(normalized_mutual_info_score(labels, predicted)),
        "fmi": float(fowlkes_mallows_score(labels, predicted)),
        "accuracy": float(accuracy_score(labels, aligned)),
        "macro_f1": float(f1_score(labels, aligned, average="macro", zero_division=0)),
        "macro_layer_iou": float(layer_recovery["macro_layer_iou"]),
        "worst_layer_iou": float(layer_recovery["worst_layer_iou"]),
        "layer_recovery_rate": float(layer_recovery["layer_recovery_rate"]),
        "layer_order_score": layer_order,
        "fidelity": fidelity,
        "embedding_spatial_neighbor_overlap": overlap,
        "embedding_spatial_distance_spearman": distance_spearman,
        "n_spots": float(labels.size),
        "n_labels": float(np.unique(labels).size),
        "embedding_dim": float(embedding.shape[1]),
    }
