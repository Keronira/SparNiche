from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import scanpy as sc
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def dense_matrix(matrix: Any) -> np.ndarray:
    values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return np.asarray(values, dtype=np.float32)


def _matches_sparniche_preprocessing_metadata(
    metadata: Any,
    config: dict[str, Any],
    feature_key: str,
) -> bool:
    if not isinstance(metadata, dict):
        return False
    expected = {
        "schema_version": 1,
        "profile": "sparniche_github_dlpfc",
        "feature_key": feature_key,
        "input_kind": "raw_counts",
        "min_cells": int(config["min_cells"]),
        "min_counts": int(config["min_counts"]),
        "hvg_enabled": bool(config["hvg_enabled"]),
        "hvg_flavor": str(config["hvg_flavor"]),
        "n_top_genes": int(config["n_top_genes"]),
        "normalize_total": bool(config["normalize_total"]),
        "target_sum": float(config["target_sum"]),
        "scale": bool(config["scale"]),
        "pca_enabled": bool(config["pca_enabled"]),
        "pca_n_components": int(config["pca_n_components"]),
        "pca_random_state": int(config["pca_random_state"]),
    }
    profile_matches = metadata.get("profile") == "sparniche_github_dlpfc"
    return profile_matches and all(
        metadata.get(key) == value for key, value in expected.items() if key != "profile"
    )


def prepare_sparniche_rna_features(
    adata,
    preprocessing_config: dict[str, Any],
):
    """Apply the DLPFC preprocessing sequence from the official SparNiche README."""
    config = {
        "enabled": True,
        "reuse_existing": False,
        "feature_key": "feat",
        "input_kind": "raw_counts",
        "min_cells": 50,
        "min_counts": 10,
        "hvg_enabled": True,
        "n_top_genes": 2000,
        "hvg_flavor": "seurat_v3",
        "normalize_total": True,
        "target_sum": 1_000_000.0,
        "scale": True,
        "pca_enabled": True,
        "pca_n_components": 200,
        "pca_random_state": 42,
        **preprocessing_config,
    }
    if not bool(config["enabled"]):
        return adata.copy()

    result = adata.copy()
    feature_key = str(config["feature_key"])
    preprocessing_metadata = result.uns.get("sparniche_rna_preprocessing")
    reusable = (
        bool(config["reuse_existing"])
        and feature_key in result.obsm
        and _matches_sparniche_preprocessing_metadata(
            preprocessing_metadata,
            config,
            feature_key,
        )
    )
    if reusable:
        features = dense_matrix(result.obsm[feature_key])
        if features.shape[0] != result.n_obs:
            raise ValueError(
                f"preprocessed feature matrix obsm[{feature_key!r}] has the wrong row count"
            )
        if not np.isfinite(features).all():
            raise ValueError(
                f"preprocessed feature matrix obsm[{feature_key!r}] contains non-finite values"
            )
        if bool(config["pca_enabled"]) and features.shape[1] != int(
            config["pca_n_components"]
        ):
            raise ValueError(
                f"preprocessed SparNiche feature matrix must have "
                f"{int(config['pca_n_components'])} columns; got {features.shape[1]}"
            )
        result.obsm[feature_key] = features
        return result

    input_kind = str(config["input_kind"])
    if input_kind == "normalized":
        # Normalized/log-transformed matrices cannot be treated as counts.
        # Use them directly for PCA without count-specific filtering or scaling.
        features = dense_matrix(result.X)
        nan_columns = np.isnan(features).any(axis=0)
        dropped_nan_features = result.var_names[nan_columns].astype(str).tolist()
        if nan_columns.any():
            if nan_columns.all():
                raise ValueError("normalized RNA input has no features after removing NaN columns")
            result = result[:, ~nan_columns].copy()
            features = dense_matrix(result.X)
        if not np.isfinite(features).all():
            raise ValueError("normalized RNA input contains non-finite values")
        pca_applied = False
        actual_pca_components = None
        if bool(config["pca_enabled"]):
            n_components = int(config["pca_n_components"])
            if features.shape[1] >= n_components:
                if features.shape[0] < n_components:
                    raise ValueError(
                        f"normalized-input PCA requires at least {n_components} spots; "
                        f"got {features.shape[0]}"
                    )
                features = PCA(
                    n_components=n_components,
                    random_state=int(config["pca_random_state"]),
                ).fit_transform(features).astype(np.float32, copy=False)
                pca_applied = True
                actual_pca_components = n_components
        result.obsm[feature_key] = features.copy()
        result.uns["sparniche_rna_preprocessing"] = {
            "schema_version": 1,
            "profile": "sparniche_normalized_input",
            "feature_key": feature_key,
            "input_kind": "normalized",
            "pca_enabled": bool(config["pca_enabled"]),
            "pca_applied": pca_applied,
            "pca_n_components_requested": int(config["pca_n_components"]),
            "pca_n_components": actual_pca_components,
            "dropped_nan_features": dropped_nan_features,
        }
        return result
    if input_kind != "raw_counts":
        raise ValueError("SparNiche preprocessing input_kind must be 'raw_counts' or 'normalized'")

    # SparNiche's official script creates a count layer before filtering/HVG
    # selection, then normalizes and scales the selected expression matrix.
    raw_counts = dense_matrix(result.X)
    if np.any(raw_counts < 0.0) or not np.allclose(
        raw_counts,
        np.rint(raw_counts),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError(
            "SparNiche preprocessing expected raw non-negative integer counts in adata.X; "
            "an existing feat is reused only when its sparniche_rna_preprocessing "
            "metadata exactly matches the requested profile"
        )
    result.layers["count"] = raw_counts
    sc.pp.filter_genes(result, min_cells=int(config["min_cells"]))
    sc.pp.filter_genes(result, min_counts=int(config["min_counts"]))
    if bool(config["normalize_total"]):
        sc.pp.normalize_total(result, target_sum=float(config["target_sum"]), inplace=True)
    if bool(config["hvg_enabled"]):
        n_top_genes = min(int(config["n_top_genes"]), result.n_vars)
        sc.pp.highly_variable_genes(
            result,
            flavor=str(config["hvg_flavor"]),
            n_top_genes=n_top_genes,
            layer="count",
        )
        result = result[:, result.var["highly_variable"]].copy()
    result.X = result.X.copy() if sparse.issparse(result.X) else np.asarray(result.X)
    if bool(config["scale"]):
        sc.pp.scale(result)

    features = dense_matrix(result.X)
    if bool(config["pca_enabled"]):
        n_components = int(config["pca_n_components"])
        available_components = min(features.shape[0], features.shape[1])
        if available_components < n_components:
            raise ValueError(
                f"official SparNiche PCA-{n_components} requires at least {n_components} "
                f"spots and selected genes; got {features.shape[0]} spots and "
                f"{features.shape[1]} genes"
            )
        features = PCA(
            n_components=n_components,
            random_state=int(config["pca_random_state"]),
        ).fit_transform(features).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("SparNiche-preprocessed RNA features contain non-finite values")
    result.obsm[feature_key] = features.copy()
    result.uns["sparniche_rna_preprocessing"] = {
        "schema_version": 1,
        "profile": "sparniche_github_dlpfc",
        "feature_key": feature_key,
        "input_kind": "raw_counts",
        "min_cells": int(config["min_cells"]),
        "min_counts": int(config["min_counts"]),
        "hvg_enabled": bool(config["hvg_enabled"]),
        "hvg_flavor": str(config["hvg_flavor"]),
        "n_top_genes": int(config["n_top_genes"]),
        "normalize_total": bool(config["normalize_total"]),
        "target_sum": float(config["target_sum"]),
        "scale": bool(config["scale"]),
        "pca_enabled": bool(config["pca_enabled"]),
        "pca_n_components": int(features.shape[1]) if bool(config["pca_enabled"]) else None,
        "pca_random_state": int(config["pca_random_state"]),
    }
    return result


def build_neighbor_index(
    coordinates: np.ndarray,
    n_neighbors: int = 12,
) -> torch.Tensor:
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[0] == 0:
        raise ValueError("coordinates must be a non-empty rank-2 array")
    if coordinates.shape[0] < 2:
        raise ValueError("at least two spots are required to build non-self neighbors")
    count = min(max(1, int(n_neighbors)), coordinates.shape[0] - 1)
    model = NearestNeighbors(n_neighbors=count + 1).fit(coordinates)
    candidates = model.kneighbors(coordinates, return_distance=False)
    rows = []
    for spot, row in enumerate(candidates):
        non_self = row[row != spot]
        if non_self.size < count:
            raise RuntimeError(f"could not find {count} non-self neighbors for spot {spot}")
        rows.append(non_self[:count])
    return torch.as_tensor(np.stack(rows), dtype=torch.long)


def build_sparniche_graph(
    coordinates: np.ndarray,
    n_neighbors: int = 12,
) -> dict[str, Any]:
    """Build SparNiche's KNN graph without materializing an N x N distance matrix."""
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[0] < 2:
        raise ValueError("coordinates must contain at least two spots")
    n_spots = int(coordinates.shape[0])
    count = min(max(1, int(n_neighbors)), n_spots - 1)

    neighbors = NearestNeighbors(n_neighbors=count + 1).fit(coordinates)
    candidates = neighbors.kneighbors(coordinates, return_distance=False)
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
    adj_norm = torch.sparse_coo_tensor(
        torch.from_numpy(
            np.vstack((adjacency_normalized.row, adjacency_normalized.col)).astype(
                np.int64
            )
        ),
        torch.from_numpy(adjacency_normalized.data),
        adjacency_normalized.shape,
    ).coalesce()

    adjacency_label = adjacency_with_self.tocoo()
    adj_label = torch.sparse_coo_tensor(
        torch.from_numpy(
            np.vstack((adjacency_label.row, adjacency_label.col)).astype(np.int64)
        ),
        torch.ones(adjacency_label.nnz, dtype=torch.float32),
        adjacency_label.shape,
    ).coalesce()
    edge_count = float(adjacency_label.sum())
    norm_value = (n_spots * n_spots) / (
        (n_spots * n_spots - edge_count) * 2.0
    )
    return {
        "adj_norm": adj_norm,
        "adj_label": adj_label,
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


def build_sparniche_negative_mask(
    adj_label: torch.Tensor, repeats: int = 1, seed: int | None = None
) -> torch.Tensor:
    """Append sampled non-edges without enumerating every node complement."""
    label = adj_label.coalesce().cpu()
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


def _resolve_view(adata, view_config: dict[str, Any]) -> torch.Tensor:
    source = str(view_config.get("source", "X"))
    if source == "X":
        values = dense_matrix(adata.X)
    elif source == "obsm":
        key = view_config.get("key")
        if not key or key not in adata.obsm:
            raise KeyError(f"AnnData obsm view {key!r} is unavailable")
        values = dense_matrix(adata.obsm[key])
    else:
        raise ValueError(f"unsupported view source: {source}")
    if values.shape[0] != adata.n_obs:
        raise ValueError(f"view {source!r} row count does not match AnnData")
    if not np.isfinite(values).all():
        raise ValueError(f"view {source!r} contains non-finite values")
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def resolve_view1(
    adata,
    data_config: dict[str, Any],
) -> torch.Tensor:
    """Resolve the RNA feature matrix used by the single-view model."""
    return _resolve_view(
        adata,
        data_config.get("view1", {"source": "obsm", "key": "feat"}),
    )


def _update_array_fingerprint(digest, values: Any) -> None:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def data_fingerprint(
    adata,
    features: torch.Tensor | None = None,
    data_config: dict[str, Any] | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(adata.shape)).encode("utf-8"))
    for name in adata.obs_names.astype(str):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    for name in adata.var_names.astype(str):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
    _update_array_fingerprint(
        digest, np.asarray(adata.obsm["spatial"], dtype=np.float32)
    )
    selected_features = (
        features
        if features is not None
        else torch.from_numpy(dense_matrix(adata.X))
    )
    _update_array_fingerprint(digest, selected_features)
    config = data_config or {}
    digest.update(
        json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )
    label_key = config.get("label_key")
    if label_key and label_key in adata.obs:
        labels = adata.obs[label_key].astype("string").fillna("<NA>").to_numpy()
        for label in labels:
            digest.update(str(label).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()
def resolve_spatial_coordinates(adata) -> np.ndarray:
    """Return spatial coordinates, accepting Scanpy's two common keys."""
    if "spatial" in adata.obsm:
        return np.asarray(adata.obsm["spatial"])
    if "X_spatial" in adata.obsm:
        return np.asarray(adata.obsm["X_spatial"])
    raise KeyError("AnnData requires obsm['spatial'] or obsm['X_spatial']")
