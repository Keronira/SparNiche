from __future__ import annotations

from typing import Sequence
import itertools

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def clustering_metrics(
    truth: Sequence[object],
    predicted: Sequence[object],
) -> dict[str, float]:
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("truth and predicted labels must be non-empty and aligned")
    return {
        "ari": float(adjusted_rand_score(truth, predicted)),
        "nmi": float(normalized_mutual_info_score(truth, predicted)),
        "ami": float(adjusted_mutual_info_score(truth, predicted)),
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be a one-dimensional sequence in [0, 1]")
    order = np.argsort(values)
    adjusted_sorted = np.empty(values.size, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (values.size - rank) * values[index]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(1.0, running)
    result = np.empty(values.size, dtype=np.float64)
    result[order] = adjusted_sorted
    return result.tolist()


def _paired_differences(
    left: Sequence[float], right: Sequence[float]
) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.ndim != 1 or left_values.shape != right_values.shape or left_values.size == 0:
        raise ValueError("paired samples must be non-empty, one-dimensional, and aligned")
    if not np.isfinite(left_values).all() or not np.isfinite(right_values).all():
        raise ValueError("paired samples must be finite")
    return left_values - right_values


def paired_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    differences = _paired_differences(left, right)
    if not 0.0 < confidence < 1.0 or n_resamples <= 0:
        raise ValueError("confidence must be in (0, 1) and n_resamples must be positive")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, differences.size, size=(int(n_resamples), differences.size))
    means = differences[indices].mean(axis=1)
    alpha = 1.0 - float(confidence)
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def paired_permutation_pvalue(
    left: Sequence[float],
    right: Sequence[float],
    seed: int = 0,
    n_resamples: int = 100_000,
) -> float:
    differences = _paired_differences(left, right)
    observed = abs(float(differences.mean()))
    if differences.size <= 20:
        means = np.fromiter(
            (
                abs(float(np.mean(differences * np.asarray(signs))))
                for signs in itertools.product((-1.0, 1.0), repeat=differences.size)
            ),
            dtype=np.float64,
            count=2 ** differences.size,
        )
        return float(np.mean(means >= observed - 1e-15))
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    generator = np.random.default_rng(int(seed))
    signs = generator.choice((-1.0, 1.0), size=(int(n_resamples), differences.size))
    permuted = np.abs((signs * differences).mean(axis=1))
    return float((np.sum(permuted >= observed - 1e-15) + 1) / (n_resamples + 1))
