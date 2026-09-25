from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import (
    build_sparniche_graph,
    build_neighbor_index,
    data_fingerprint,
    prepare_sparniche_rna_features,
    resolve_spatial_coordinates,
    resolve_view1,
    resolve_view2,
    resolve_view2_key,
)
from .config import normalize_sparniche_config
from .trainer import train_tensors


@dataclass
class PipelineArtifacts:
    adata: Any
    training_result: Any
    clean_output: Any
    features: torch.Tensor
    neighbor_idx: torch.Tensor


def train_adata(
    adata,
    config: dict[str, Any],
    output_dir: Path,
    resume: bool = False,
) -> PipelineArtifacts:
    config = normalize_sparniche_config(config)
    spatial = resolve_spatial_coordinates(adata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_adata = adata.copy()
    if "spatial" not in result_adata.obsm:
        result_adata.obsm["spatial"] = spatial
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    preprocessing_config = data_config.get("preprocessing")
    if preprocessing_config is not None:
        result_adata = prepare_sparniche_rna_features(result_adata, preprocessing_config)
    default_neighbors = 12
    n_neighbors = int(data_config.get("n_neighbors", default_neighbors))
    neighbor_idx = build_neighbor_index(
        result_adata.obsm["spatial"], n_neighbors
    )
    sparniche_graph = build_sparniche_graph(result_adata.obsm["spatial"], n_neighbors)
    features = resolve_view1(result_adata, data_config)
    requested_double_view = bool(model_config.get("double_view", False))
    view2_key = resolve_view2_key(result_adata, data_config) if requested_double_view else None
    double_view = requested_double_view and view2_key is not None
    if requested_double_view and not double_view and data_config.get("view2", {}).get("key") != "auto":
        raise KeyError("configured second view is unavailable")
    model_config["double_view"] = double_view
    adt_features = resolve_view2(result_adata, data_config) if double_view else None
    fingerprint = data_fingerprint(
        result_adata, features=features, data_config=data_config, view2=adt_features
    )
    trained = train_tensors(
        features,
        neighbor_idx,
        sparniche_graph,
        config,
        output_dir,
        resume=resume,
        data_fingerprint=fingerprint,
        view2=adt_features,
    )
    device = next(trained.model.parameters()).device
    trained.model.eval()
    with torch.no_grad():
        clean_output = trained.model(
            features.to(device),
            neighbor_idx.to(device),
            adt_features.to(device) if adt_features is not None else None,
        )

    result_adata.obsm["sparniche"] = clean_output.embedding.cpu().numpy()
    training_config = config.get("training", {})
    sparniche_encoder = trained.model.encoder
    result_adata.uns["sparniche_training"] = {
        "last_epoch": int(trained.last_epoch),
        "last_metrics": trained.history[-1] if trained.history else {},
        "data_fingerprint": fingerprint,
        "view_contract": {
            "view1": "RNA:obsm[feat]",
            "double_view": double_view,
            "view2": (
                f"{view2_key.upper()}:obsm[{view2_key}]"
                if double_view else "disabled"
            ),
            "adt_reconstruction_weight": (
                float(model_config.get("sparniche", {}).get("adt_rec_w", 1.0))
                if double_view else 0.0
            ),
            "expression_neighbor_mean_input": False,
        },
            "rna_view1_contract": {
            "enabled": True,
            "encoder": (
                "SparNiche full module from official GitHub module.py (ComplexEncoder+GCN+AAE+DEC)"
            ),
            "decoder": (
                "SparNiche GraphConvolution decoder"
            ),
            "loss": (
                "non-DEC: rec_w*MSE + gcn_w*(graph BCE+KLD) + self_w*SCE; "
                "DEC: rec_w*MSE + gcn_w*(graph BCE+KLD) + dec_kl_w*DEC KL"
            ),
            "optimizer": "Adam",
            "learning_rate": float(
                model_config.get("sparniche", {}).get("lr", 0.01)
            ),
            "weight_decay": float(
                model_config.get("sparniche", {}).get("weight_decay", 0.01)
            ),
            "embedding": "official SparNiche feature-graph representation",
            "attention": {
                "mode": sparniche_encoder.attention_mode,
                "neighbors": sparniche_encoder.attention_neighbors,
                "chunk_size": sparniche_encoder.attention_chunk_size,
            },
            "sparniche_official_defaults": {
                "hidden_dims": [64, 16],
                "num_heads": 1,
                "dropout": 0.2,
                "gcn_in_encoder": True,
                "graph_decoder": "inner_product",
                "mask_rate": 0.8,
                "graph_construction": "official pairwise-distance KNN, symmetric, self-loop, D^-1/2 A D^-1/2",
                "training_stages": ["GAN", "non-DEC", "DEC"],
                "dec_cluster_n": int(model_config.get("sparniche_view1", {}).get("dec_cluster_n", 10)),
                },
            "rna_experiment_variant": {
                "feature_graph_fusion_mode": str(
                    model_config.get("feature_graph_fusion_mode", "gated")
                ),
                "adaptive_graph": bool(model_config.get("sparniche_adaptive_graph", False)),
                "contrastive_weight": float(model_config.get("sparniche", {}).get("contrastive_w", 0.0)),
                "contrastive_temperature": float(model_config.get("sparniche", {}).get("contrastive_temperature", 0.2)),
            },
        },
        "configured_encoder": "sparniche",
    }
    if bool(config.get("benchmark", {}).get("write_h5ad", True)):
        result_adata.write_h5ad(output_dir / "trained.h5ad")
    return PipelineArtifacts(
        adata=result_adata,
        training_result=trained,
        clean_output=clean_output,
        features=features,
        neighbor_idx=neighbor_idx,
    )


def run_adata_pipeline(
    adata,
    config: dict[str, Any],
    output_dir: Path,
    resume: bool = False,
):
    return train_adata(adata, config, output_dir, resume=resume).adata
