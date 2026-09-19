from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F



VALID_VARIANTS = {"local_graph_normalized"}

LOCAL_GRAPH_VARIANTS = {name for name in VALID_VARIANTS if name.startswith("local_graph_")}




class SparNicheGraphConvolution(nn.Module):
    """The sparse GCN layer used by SparNiche's official ``module``."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float, activation):
        super().__init__()
        self.dropout = float(dropout)
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(int(input_dim), int(output_dim)))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        dropped = F.dropout(features, self.dropout, self.training)
        support = dropped @ self.weight
        output = torch.sparse.mm(adjacency, support)
        return self.activation(output)


class ResidualGCN(nn.Module):
    """Sparse residual graph convolution used by local_graph variants."""

    def __init__(self, dim: int, dropout: float = 0.2, alpha: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(int(dim), int(dim))
        self.dropout = float(dropout)
        self.alpha = float(alpha)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        message = torch.sparse.mm(adjacency, F.dropout(features, self.dropout, self.training))
        update = F.relu(self.linear(message))
        return features + self.alpha * update


class SparNicheInnerProductDecoder(nn.Module):
    """Decode sampled graph edges with the official inner product decoder."""

    def forward(self, embedding: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        indices = edge_mask.coalesce().indices()
        return torch.sum(embedding[indices[0]] * embedding[indices[1]], dim=1)


class SparNicheGenerator(nn.Module):
    """The 128-unit tanh generator used for SparNiche's adversarial pretraining."""

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(latent_dim), 128),
            nn.ReLU(),
            nn.Linear(128, int(output_dim)),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.model(noise)


class SparNicheSelfAttention(nn.Module):
    """SparNiche's single-token self-attention used before feature concatenation."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc_q = nn.Linear(int(dim), int(dim))
        self.fc_k = nn.Linear(int(dim), int(dim))
        self.fc_v = nn.Linear(int(dim), int(dim))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.fc_q(features)
        key = self.fc_k(features)
        value = self.fc_v(features)
        scores = query @ key.transpose(-2, -1) / (features.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return weights @ value, weights


class SparNicheFeatureFusion(nn.Module):
    """WeightedConcatFusionWithAttention from SparNiche/module.py."""

    def __init__(self, dim: int, mode: str = "official"):
        super().__init__()
        if mode not in {"official", "gated"}:
            raise ValueError("fusion mode must be 'official' or 'gated'")
        self.mode = mode
        self.attention = SparNicheSelfAttention(dim)
        self.gate = nn.Sequential(
            nn.Linear(3 * int(dim), int(dim)),
            nn.ReLU(),
            nn.Linear(int(dim), int(dim)),
            nn.Sigmoid(),
        ) if mode == "gated" else None

    def forward(
        self, feature_embedding: torch.Tensor, graph_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode == "gated":
            gate = self.gate(
                torch.cat(
                    (feature_embedding, graph_embedding,
                     torch.abs(feature_embedding - graph_embedding)), dim=1
                )
            )
            fused = torch.cat((gate * feature_embedding, (1.0 - gate) * graph_embedding), dim=1)
            return fused, gate, 1.0 - gate
        feature_out, feature_weights = self.attention(feature_embedding.unsqueeze(1))
        graph_out, graph_weights = self.attention(graph_embedding.unsqueeze(1))
        return (
            torch.cat((feature_out.squeeze(1), graph_out.squeeze(1)), dim=1),
            feature_weights.squeeze(1),
            graph_weights.squeeze(1),
        )


class SparNicheEncoder(nn.Module):
    """SparNiche's complete expression branch, including graph and DEC heads.

    The implementation mirrors the official GitHub ``module``: ComplexEncoder,
    GCN mu/logvar paths, reparameterization, attention concatenation, graph
    feature decoder, inner-product graph decoder, SCE mask reconstruction,
    adversarial generator/discriminator, and DEC soft assignments.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] | list[int] = (64, 16),
        num_heads: int = 1,
        dropout: float = 0.2,
        dec_cluster_n: int = 10,
        fusion_mode: str = "official",
        adaptive_graph: bool = False,
        local_graph_mode: str = "none",
        local_graph_hops: int = 1,
        local_graph_normalize: bool = True,
        residual_gcn_layers: int = 1,
        mlp_layers: int = 2,
        attention_mode: str = "global",
        attention_neighbors: int | None = None,
        attention_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = tuple(int(value) for value in hidden_dims)
        if len(hidden_dims) != 2 or any(value <= 0 for value in hidden_dims):
            raise ValueError("SparNiche hidden_dims must contain two positive values")
        if int(input_dim) <= 0 or int(latent_dim) <= 0:
            raise ValueError("SparNiche input and latent dimensions must be positive")
        if int(num_heads) <= 0 or int(input_dim) % int(num_heads) != 0:
            raise ValueError("SparNiche num_heads must divide the RNA input dimension")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("SparNiche dropout must be in [0, 1)")
        if hidden_dims != (64, 16) or int(num_heads) != 1 or int(dec_cluster_n) != 10:
            raise ValueError(
                "SparNiche requires hidden_dims=[64,16], num_heads=1, "
                "and dec_cluster_n=10"
            )
        self.input_dim = int(input_dim)
        self.feat_hidden1, self.feat_hidden2 = hidden_dims
        self.gcn_hidden1 = int(self.feat_hidden1)
        self.gcn_hidden2 = int(self.feat_hidden2)
        self.latent_dim = int(latent_dim)
        self.dropout = float(dropout)
        self.alpha = 1.0
        self.dec_cluster_n = int(dec_cluster_n)
        self.num_heads = int(num_heads)

        # Preserve the official module.py parameter initialization order.
        self.fusion_mode = str(fusion_mode)
        self.adaptive_graph = bool(adaptive_graph)
        self.local_graph_mode = str(local_graph_mode)
        self.local_graph_hops = max(0, int(local_graph_hops))
        self.local_graph_normalize = bool(local_graph_normalize)
        self.mlp_layers = max(1, int(mlp_layers))
        self.attention_mode = str(attention_mode)
        self.attention_neighbors = (
            None if attention_neighbors is None else int(attention_neighbors)
        )
        self.attention_chunk_size = (
            None if attention_chunk_size is None else int(attention_chunk_size)
        )
        if self.attention_mode not in {"global", "spatial_local"}:
            raise ValueError("attention_mode must be 'global' or 'spatial_local'")
        if self.attention_mode == "spatial_local":
            if self.attention_neighbors is None or self.attention_neighbors <= 0:
                raise ValueError("attention_neighbors must be positive in spatial_local mode")
            if self.attention_chunk_size is None or self.attention_chunk_size <= 0:
                raise ValueError("attention_chunk_size must be positive in spatial_local mode")
        if self.local_graph_mode not in {"none", "mean", "residual", "gated", "multihop", "normalized", "dropout"}:
            raise ValueError("unsupported local graph mode")
        self.local_graph_gate = (
            nn.Sequential(nn.Linear(2 * self.input_dim, self.input_dim), nn.Sigmoid())
            if self.local_graph_mode == "gated" else None
        )
        self.local_residual_gcns = nn.ModuleList(
            [ResidualGCN(self.gcn_hidden2, dropout, 0.5)
             for _ in range(max(0, int(residual_gcn_layers)))]
        )
        self.feature_fusion = SparNicheFeatureFusion(self.feat_hidden2, mode=self.fusion_mode)
        self.embedding_projection = (
            nn.Linear(2 * self.feat_hidden2, self.latent_dim)
            if self.latent_dim != 32 else nn.Identity()
        )
        self.generator = SparNicheGenerator(self.feat_hidden2, self.input_dim)
        self.discriminator = nn.Linear(self.feat_hidden2, 1)
        self.layers = nn.ModuleList()
        self.layer = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=int(input_dim), num_heads=self.num_heads)]
        )
        in_features = int(input_dim)
        mlp_dims = ([self.feat_hidden2] if self.mlp_layers == 1 else
                    [self.feat_hidden1] * (self.mlp_layers - 1) + [self.feat_hidden2])
        for out_features in mlp_dims:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
                    nn.ELU(),
                    nn.Dropout(p=float(dropout)),
                )
            )
            in_features = out_features

        # Retained because it is part of the official module.  The official
        # forward path leaves this adjustment layer unused.
        self.adjust_layer = nn.Linear(in_features, int(input_dim))
        self.decoder = SparNicheGraphConvolution(
            self.latent_dim, self.input_dim, self.dropout, lambda x: x
        )
        self.gc1 = SparNicheGraphConvolution(
            self.feat_hidden2, self.gcn_hidden1, self.dropout, F.relu
        )
        self.gc2 = SparNicheGraphConvolution(
            self.gcn_hidden1, self.gcn_hidden2, self.dropout, lambda x: x
        )
        self.gc3 = SparNicheGraphConvolution(
            self.gcn_hidden1, self.gcn_hidden2, self.dropout, lambda x: x
        )
        self.graph_decoder = SparNicheInnerProductDecoder()
        self.cluster_layer = nn.Parameter(torch.empty(self.dec_cluster_n, self.latent_dim))
        nn.init.xavier_normal_(self.cluster_layer)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.input_dim))
        self.mask_rate = 0.8
        self._adj_norm: torch.Tensor | None = None
        self._adj_label: torch.Tensor | None = None
        self._graph_mask: torch.Tensor | None = None
        self._graph_norm: float | None = None
        self.edge_gate: nn.Parameter | None = None

    def _apply_local_graph(self, features: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
        if self.local_graph_mode == "none" or self.local_graph_hops == 0:
            return features
        neighbor_mean = features[neighbor_idx].mean(dim=1)
        for _ in range(self.local_graph_hops - 1):
            neighbor_mean = neighbor_mean[neighbor_idx].mean(dim=1)
        if self.local_graph_mode == "mean":
            return neighbor_mean
        if self.local_graph_mode == "residual":
            return features + 0.5 * neighbor_mean
        if self.local_graph_mode == "gated":
            gate = self.local_graph_gate(torch.cat((features, neighbor_mean), dim=1))
            return gate * features + (1.0 - gate) * neighbor_mean
        if self.local_graph_mode == "multihop":
            hop2 = neighbor_mean[neighbor_idx].mean(dim=1)
            return (features + neighbor_mean + hop2) / 3.0
        if self.local_graph_mode == "normalized" and self.local_graph_normalize:
            return F.normalize(features, dim=1) + F.normalize(neighbor_mean, dim=1)
        return features + F.dropout(neighbor_mean, p=0.2, training=self.training)

    def _complex_encode(
        self,
        features: torch.Tensor,
        neighbor_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("SparNiche RNA features must be a rank-2 tensor")
        identity = features
        if self.attention_mode == "global":
            # Official SparNiche uses batch_first=False and unsqueezes dimension 1,
            # making the spot axis the attention sequence dimension.
            attended = features.unsqueeze(1)
            for attention in self.layer:
                attended, _ = attention(attended, attended, attended)
            attended = attended.squeeze(1)
        else:
            if neighbor_idx is None:
                raise ValueError("spatial_local attention requires neighbor_idx")
            if neighbor_idx.dtype != torch.long:
                raise ValueError("neighbor_idx must be an integer torch.long tensor")
            if neighbor_idx.ndim != 2 or neighbor_idx.shape[0] != features.shape[0]:
                raise ValueError("neighbor_idx shape must be [n_spots, n_neighbors]")
            if self.attention_neighbors is None or self.attention_neighbors > neighbor_idx.shape[1]:
                raise ValueError("attention_neighbors exceeds available neighbor_idx width")
            if neighbor_idx.numel() and (
                neighbor_idx.min() < 0 or neighbor_idx.max() >= features.shape[0]
            ):
                raise ValueError("neighbor_idx contains indices outside valid spot range")
            chunks = []
            chunk_size = int(self.attention_chunk_size)
            n_neighbors = int(self.attention_neighbors)
            for start in range(0, features.shape[0], chunk_size):
                end = min(start + chunk_size, features.shape[0])
                query = features[start:end].unsqueeze(0)
                self_idx = torch.arange(start, end, device=features.device).unsqueeze(1)
                local_idx = torch.cat(
                    (self_idx, neighbor_idx[start:end, :n_neighbors]), dim=1
                )
                keys_values = features[local_idx].transpose(0, 1)
                for attention in self.layer:
                    query, _ = attention(
                        query, keys_values, keys_values, need_weights=False
                    )
                chunks.append(query.squeeze(0))
            attended = torch.cat(chunks, dim=0)
        attended = attended + identity
        for layer in self.layers:
            attended = layer(attended)
        return attended

    def configure_graph(
        self,
        graph: dict,
        graph_mask: torch.Tensor | None = None,
    ) -> None:
        """Attach the graph produced by SparNiche's official graph constructor."""
        required = {"adj_norm", "adj_label", "norm_value"}
        missing = required.difference(graph)
        if missing:
            raise ValueError(f"SparNiche graph is missing: {', '.join(sorted(missing))}")
        device = next(self.parameters()).device
        adj_norm = graph["adj_norm"].coalesce().to(device=device, dtype=torch.float32)
        adj_label = graph["adj_label"].coalesce().to(device=device, dtype=torch.float32)
        if adj_norm.shape != adj_label.shape or adj_norm.ndim != 2:
            raise ValueError("SparNiche normalized and label adjacency shapes must match")
        self._adj_norm = adj_norm
        self._adj_label = adj_label
        self._graph_norm = float(graph["norm_value"])
        if self.adaptive_graph:
            self.edge_gate = nn.Parameter(torch.zeros(adj_norm.coalesce().values().numel(), device=device))
        self.set_graph_mask(adj_label if graph_mask is None else graph_mask)

    def _effective_adjacency(self) -> torch.Tensor:
        adjacency = self._adj_norm
        if adjacency is None or not self.adaptive_graph or self.edge_gate is None:
            return adjacency
        adjacency = adjacency.coalesce()
        values = adjacency.values() * torch.sigmoid(self.edge_gate)
        return torch.sparse_coo_tensor(
            adjacency.indices(), values, adjacency.shape, device=adjacency.device
        ).coalesce()

    def set_graph_mask(self, graph_mask: torch.Tensor) -> None:
        device = next(self.parameters()).device
        mask = graph_mask.coalesce().to(device=device, dtype=torch.float32)
        if self._adj_label is not None and mask.shape != self._adj_label.shape:
            raise ValueError("SparNiche graph mask shape does not match adj_label")
        self._graph_mask = mask

    def graph_mask_cpu(self) -> torch.Tensor | None:
        if self._graph_mask is None:
            return None
        return self._graph_mask.detach().cpu().coalesce()

    def decode_graph(
        self, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, graph_mask, _ = self._require_graph(embedding.shape[0])
        return self.graph_decoder(embedding, graph_mask), graph_mask

    def _require_graph(
        self, n_spots: int
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        if self._adj_norm is None or self._graph_mask is None or self._graph_norm is None:
            raise RuntimeError("SparNiche official graph must be configured before forward")
        if self._adj_norm.shape != (n_spots, n_spots):
            raise ValueError("SparNiche graph size does not match the RNA spot count")
        return self._adj_norm, self._graph_mask, self._graph_norm

    def _encode_graph(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        neighbor_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_embedding = self._complex_encode(features, neighbor_idx)
        hidden = self.gc1(feature_embedding, adjacency)
        graph_embedding = self.gc2(hidden, adjacency)
        for residual_gcn in self.local_residual_gcns:
            graph_embedding = residual_gcn(graph_embedding, adjacency)
        return graph_embedding, self.gc3(hidden, adjacency), feature_embedding

    def _mask_noise(
        self, features: torch.Tensor, seed: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=features.device).manual_seed(int(seed))
        n_spots = features.shape[0]
        n_mask = int(self.mask_rate * n_spots)
        permutation = torch.randperm(n_spots, generator=generator, device=features.device)
        mask_nodes = permutation[:n_mask]
        masked = features.clone()
        if mask_nodes.numel():
            masked[mask_nodes] = masked[mask_nodes] + self.enc_mask_token
        return masked, mask_nodes

    def _q_distribution(self, embedding: torch.Tensor) -> torch.Tensor:
        q = 1.0 / (
            1.0
            + torch.sum(
                (embedding.unsqueeze(1) - self.cluster_layer).pow(2), dim=2
            )
            / self.alpha
        )
        q = q.pow((self.alpha + 1.0) / 2.0)
        return (q.t() / q.sum(dim=1).clamp_min(1e-8)).t()

    def forward(
        self,
        features: torch.Tensor,
        neighbor_idx: torch.Tensor,
        *,
        return_aux: bool = False,
        mask_seed: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("SparNiche RNA features have the wrong shape")
        _, graph_mask, norm_value = self._require_graph(features.shape[0])
        adj_norm = self._effective_adjacency()
        masked, mask_nodes = self._mask_noise(features, mask_seed)
        graph_input = self._apply_local_graph(masked, neighbor_idx)
        mu, logvar, feat_x = self._encode_graph(graph_input, adj_norm, neighbor_idx)
        if self.training:
            gnn_z = torch.randn_like(torch.exp(logvar)).mul(torch.exp(logvar)).add_(mu)
        else:
            gnn_z = mu
        z, attention_x, attention_z = self.feature_fusion(feat_x, gnn_z)
        z = self.embedding_projection(z)
        decoded = self.decoder(z, adj_norm)
        q = self._q_distribution(z)
        graph_logits = self.graph_decoder(z, graph_mask)
        masked_target = masked[mask_nodes]
        masked_reconstruction = decoded[mask_nodes]
        aux = {
            "embedding": z,
            "feature_embedding": feat_x,
            "graph_embedding": gnn_z,
            "mu": mu,
            "logvar": logvar,
            "reconstruction": decoded,
            "q": q,
            "graph_logits": graph_logits,
            "graph_mask": graph_mask,
            "graph_norm": torch.as_tensor(norm_value, device=features.device),
            "mask_nodes": mask_nodes,
            "masked_target": masked_target,
            "masked_reconstruction": masked_reconstruction,
            "attention_x": attention_x,
            "attention_z": attention_z,
        }
        return aux if return_aux else z

    def gan_losses(
        self,
        features: torch.Tensor,
        neighbor_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return SparNiche's generator and discriminator losses for pretraining."""
        adj_norm, _, _ = self._require_graph(features.shape[0])
        noise = torch.randn(
            features.shape[0], self.feat_hidden2, device=features.device
        )
        fake_features = self.generator(noise)
        _, _, real_features = self._encode_graph(features, adj_norm, neighbor_idx)
        criterion = nn.BCEWithLogitsLoss()
        real_logits = self.discriminator(real_features)
        fake_logits = self.discriminator(
            self._complex_encode(fake_features.detach(), neighbor_idx)
        )
        discriminator_loss = 0.5 * (
            criterion(real_logits, torch.ones_like(real_logits))
            + criterion(fake_logits, torch.zeros_like(fake_logits))
        )
        generator_logits = self.discriminator(
            self._complex_encode(fake_features, neighbor_idx)
        )
        generator_loss = criterion(generator_logits, torch.ones_like(generator_logits))
        return generator_loss, discriminator_loss










@dataclass




@dataclass
class SparNicheOutput:
    """Outputs required by the RNA-only SparNiche training and benchmark flow."""

    embedding: torch.Tensor
    reconstruction: torch.Tensor
    graph_logits: torch.Tensor
    q: torch.Tensor
    aux: dict[str, torch.Tensor]


class SparNiche(nn.Module):
    """Complete SparNiche RNA encoder without a synthetic second view."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        sparniche_config: dict | None = None,
        feature_graph_fusion_mode: str = "gated",
        adaptive_graph: bool = False,
        local_graph_mode: str = "normalized",
        local_graph_hops: int = 1,
        local_graph_normalize: bool = True,
        residual_gcn_layers: int = 1,
        mlp_layers: int = 2,
    ) -> None:
        super().__init__()
        config = dict(sparniche_config or {})
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = SparNicheEncoder(
            self.input_dim,
            self.latent_dim,
            hidden_dims=config.get("hidden_dims", (64, 16)),
            num_heads=int(config.get("num_heads", 1)),
            dropout=float(config.get("dropout", 0.2)),
            dec_cluster_n=int(config.get("dec_cluster_n", 10)),
            fusion_mode=str(feature_graph_fusion_mode),
            adaptive_graph=bool(adaptive_graph),
            local_graph_mode=str(local_graph_mode),
            local_graph_hops=int(local_graph_hops),
            local_graph_normalize=bool(local_graph_normalize),
            residual_gcn_layers=int(residual_gcn_layers),
            mlp_layers=int(mlp_layers),
            attention_mode=str(config.get("attention_mode", "global")),
            attention_neighbors=config.get("attention_neighbors"),
            attention_chunk_size=config.get("attention_chunk_size"),
        )

    def configure_graph(
        self, graph: dict, graph_mask: torch.Tensor | None = None
    ) -> None:
        self.encoder.configure_graph(graph, graph_mask=graph_mask)

    def forward(
        self, view1: torch.Tensor, neighbor_idx: torch.Tensor
    ) -> SparNicheOutput:
        aux = self.encoder(view1, neighbor_idx, return_aux=True)
        if not isinstance(aux, dict):
            raise TypeError("SparNiche encoder did not return its training outputs")
        return SparNicheOutput(
            embedding=aux["embedding"],
            reconstruction=aux["reconstruction"],
            graph_logits=aux["graph_logits"],
            q=aux["q"],
            aux=aux,
        )
