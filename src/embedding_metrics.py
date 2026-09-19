"""Deterministic embedding-quality metrics used by the neighbor sweep."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


def _safe_float(value: float) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def _neighbors(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        return np.empty((values.shape[0], 0), dtype=np.int64)
    width = min(int(k) + 1, values.shape[0])
    return NearestNeighbors(n_neighbors=width).fit(values).kneighbors(
        values, return_distance=False
    )[:, 1:]


def _purity(indices: np.ndarray, labels: np.ndarray) -> float:
    if indices.size == 0:
        return float("nan")
    return _safe_float(np.mean(np.asarray(labels)[indices] == np.asarray(labels)[:, None]))


def _overlap(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    values = []
    for a, b in zip(left, right):
        sa, sb = set(a.tolist()), set(b.tolist())
        union = sa | sb
        values.append(len(sa & sb) / len(union) if union else 1.0)
    return _safe_float(float(np.mean(values)))


def _pairwise_spearman(left: np.ndarray, right: np.ndarray, seed: int) -> float:
    n = left.shape[0]
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
    return _safe_float(float(value))


def _groups(spatial: np.ndarray, bins: int = 4) -> np.ndarray:
    spatial = np.asarray(spatial, dtype=np.float32)
    if spatial.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    scaled = spatial - np.nanmin(spatial, axis=0, keepdims=True)
    span = np.nanmax(scaled, axis=0, keepdims=True)
    span[span == 0] = 1.0
    cells = np.floor(np.clip(scaled / span, 0.0, 0.999999) * bins).astype(int)
    return cells[:, 0] * bins + cells[:, 1]


def _linear_probe_macro_f1(embedding: np.ndarray, labels: np.ndarray, spatial: np.ndarray) -> float:
    classes = np.unique(labels)
    groups = _groups(spatial)
    n_groups = len(np.unique(groups))
    if len(classes) < 2 or n_groups < 3:
        return float("nan")
    splitter = GroupKFold(n_splits=min(5, n_groups))
    scores = []
    for train, test in splitter.split(embedding, labels, groups):
        if len(np.unique(labels[train])) < 2:
            continue
        model = LogisticRegression(max_iter=1000, random_state=0)
        model.fit(embedding[train], labels[train])
        scores.append(
            f1_score(labels[test], model.predict(embedding[test]), labels=classes, average="macro", zero_division=0)
        )
    return _safe_float(float(np.mean(scores))) if scores else float("nan")


def compute_embedding_metrics(
    embedding: np.ndarray,
    labels: np.ndarray,
    spatial: np.ndarray,
    *,
    seed: int,
    n_neighbors: int = 12,
) -> dict[str, float]:
    embedding = np.asarray(embedding, dtype=np.float32)
    labels = np.asarray(labels).astype(str)
    spatial = np.asarray(spatial, dtype=np.float32)
    if embedding.ndim != 2 or embedding.shape[0] != labels.shape[0]:
        raise ValueError("embedding and labels must have matching rows")
    if spatial.ndim != 2 or spatial.shape[0] != labels.shape[0]:
        raise ValueError("embedding and spatial coordinates must have matching rows")
    embedding_neighbors = _neighbors(embedding, n_neighbors)
    spatial_neighbors = _neighbors(spatial, n_neighbors)
    unique = np.unique(labels)
    predicted = KMeans(n_clusters=len(unique), n_init=20, random_state=int(seed)).fit_predict(embedding)
    result: dict[str, float] = {
        "n_spots": float(embedding.shape[0]),
        "n_labels": float(len(unique)),
        "embedding_dim": float(embedding.shape[1]),
        "fixed_kmeans_ari": _safe_float(adjusted_rand_score(labels, predicted)),
        "embedding_knn_label_purity": _purity(embedding_neighbors, labels),
        "spatial_knn_label_purity": _purity(spatial_neighbors, labels),
        "embedding_spatial_neighbor_overlap": _overlap(embedding_neighbors, spatial_neighbors),
        "embedding_spatial_distance_spearman": _pairwise_spearman(embedding, spatial, seed),
        "linear_probe_macro_f1": _linear_probe_macro_f1(embedding, labels, spatial),
    }
    if len(unique) > 1 and embedding.shape[0] > len(unique):
        result["embedding_silhouette"] = _safe_float(silhouette_score(embedding, labels))
    else:
        result["embedding_silhouette"] = float("nan")
    return result
