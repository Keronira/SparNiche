from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from .config import normalize_sparniche_config
from .data import build_sparniche_negative_mask
from .models import SparNicheEncoder, SparNiche, SparNicheOutput


@dataclass
class TrainingResult:
    model: SparNiche
    output: SparNicheOutput
    history: list[dict[str, Any]]
    last_epoch: int



STAGE_ORDER = {
    "gan": 0,
    "non_dec": 1,
    "dec": 2,
    "complete": 3,
}


def _sce_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha: int = 3,
) -> torch.Tensor:
    if prediction.numel() == 0:
        return prediction.sum() * 0.0
    prediction = F.normalize(prediction, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    return (1.0 - (prediction * target).sum(dim=-1)).pow(int(alpha)).mean()


def _sparniche_target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = batch.pow(2) / batch.sum(dim=0).clamp_min(1e-8)
    return (weight.t() / weight.sum(dim=1).clamp_min(1e-8)).t()


def _sparniche_contrastive_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    temperature: float = 0.2,
    weight: float = 1.0,
) -> torch.Tensor:
    """InfoNCE consistency for same-spot expression/graph representations."""
    if weight == 0.0:
        return left.sum() * 0.0
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("contrastive inputs must have matching rank-2 shapes")
    if temperature <= 0.0:
        raise ValueError("contrastive temperature must be positive")
    left = F.normalize(left, dim=1)
    right = F.normalize(right, dim=1)
    logits = left @ right.t() / float(temperature)
    labels = torch.arange(left.shape[0], device=left.device)
    return float(weight) * 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )


def _sparniche_losses(
    aux: dict[str, torch.Tensor],
    clean_rna: torch.Tensor,
    target_distribution: torch.Tensor | None,
    contrastive_w: float = 0.0,
    contrastive_temperature: float = 0.2,
    clean_adt: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    reconstruction = F.mse_loss(aux["reconstruction"], clean_rna)
    graph_mask = aux["graph_mask"].coalesce()
    graph = aux["graph_norm"] * F.binary_cross_entropy_with_logits(
        aux["graph_logits"], graph_mask.values()
    )
    mu = aux["mu"]
    logvar = aux["logvar"]
    graph = graph - 0.5 / max(clean_rna.shape[0], 1) * torch.mean(
        torch.sum(
            1.0 + 2.0 * logvar - mu.pow(2) - logvar.exp().pow(2),
            dim=1,
        )
    )
    self_construction = _sce_loss(
        aux["masked_reconstruction"], aux["masked_target"], alpha=3
    )
    contrastive = _sparniche_contrastive_loss(
        aux["feature_embedding"],
        aux["graph_embedding"],
        temperature=contrastive_temperature,
        weight=contrastive_w,
    )
    if target_distribution is None:
        dec = aux["embedding"].sum() * 0.0
    else:
        dec = F.kl_div(
            aux["q"].clamp_min(1e-8).log(),
            target_distribution.to(aux["q"].device),
            reduction="mean",
        )
    losses = {
        "reconstruction": reconstruction,
        "graph": graph,
        "self": self_construction,
        "contrastive": contrastive,
        "dec": dec,
    }
    if clean_adt is not None:
        losses["adt_reconstruction"] = F.mse_loss(aux["adt_reconstruction"], clean_adt)
    return losses


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@torch.no_grad()
















def _checkpoint_config_identity(config: dict[str, Any]) -> str:
    identity = copy.deepcopy(config)
    training = identity.get("training", {})
    for key in ("epochs", "checkpoint_every", "device"):
        training.pop(key, None)
    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_sparniche_attention_config(
    sparniche_view1_config: dict[str, Any],
    *,
    data_neighbors: int,
    available_neighbors: int,
) -> dict[str, Any]:
    """Resolve local-attention defaults without treating YAML null as a value."""
    resolved = dict(sparniche_view1_config)
    if str(resolved.get("attention_mode", "global")) != "spatial_local":
        return resolved
    if resolved.get("attention_neighbors") is None:
        resolved["attention_neighbors"] = int(data_neighbors)
    if resolved.get("attention_chunk_size") is None:
        resolved["attention_chunk_size"] = 4096
    if int(resolved["attention_neighbors"]) > int(available_neighbors):
        raise ValueError("attention_neighbors exceeds available neighbor_idx width")
    return resolved


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_checkpoint(
    path: Path,
    model: SparNiche,
    optimizers: dict[str, torch.optim.Optimizer],
    stage: str,
    epoch: int,
    config: dict[str, Any],
    data_fingerprint: str | None,
    extras: dict[str, Any] | None = None,
) -> None:
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown checkpoint stage: {stage}")
    encoder = model.encoder
    graph_mask = (
        encoder.graph_mask_cpu() if isinstance(encoder, SparNicheEncoder) else None
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "stage": stage,
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dicts": {
                name: optimizer.state_dict() for name, optimizer in optimizers.items()
            },
            "sparniche_graph_mask": graph_mask,
            "extras": dict(extras or {}),
            "config": config,
            "config_identity_sha256": _checkpoint_config_identity(config),
            "data_fingerprint": data_fingerprint,
            "rng_state": _rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    model: SparNiche,
    config: dict[str, Any],
    data_fingerprint: str | None,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("config_identity_sha256") != _checkpoint_config_identity(config):
        raise ValueError("checkpoint configuration does not match the requested run")
    if checkpoint.get("data_fingerprint") != data_fingerprint:
        raise ValueError("checkpoint data fingerprint does not match the selected features")
    if checkpoint.get("stage") not in STAGE_ORDER:
        raise ValueError("checkpoint predates the official SparNiche staged training flow")
    model.load_state_dict(checkpoint["model_state_dict"])
    graph_mask = checkpoint.get("sparniche_graph_mask")
    if graph_mask is not None:
        encoder = model.encoder
        if not isinstance(encoder, SparNicheEncoder):
            raise TypeError("checkpoint contains a SparNiche graph for a non-SparNiche model")
        encoder.set_graph_mask(graph_mask)
    rng_state = checkpoint.get("rng_state")
    if rng_state is None:
        raise ValueError("checkpoint is missing RNG state required for exact resume")
    _restore_rng_state(rng_state)
    return checkpoint


def _stage_start(checkpoint: dict[str, Any] | None, stage: str) -> int | None:
    if checkpoint is None:
        return 1
    checkpoint_stage = str(checkpoint["stage"])
    if STAGE_ORDER[checkpoint_stage] > STAGE_ORDER[stage]:
        return None
    if checkpoint_stage == stage:
        return int(checkpoint["epoch"]) + 1
    return 1


def _checkpoint_due(epoch: int, total: int, every: int) -> bool:
    return epoch == total or epoch % max(1, every) == 0


def _record(
    history: list[dict[str, Any]],
    metrics_path: Path,
    stage: str,
    epoch: int,
    loss: torch.Tensor,
    components: dict[str, torch.Tensor],
) -> None:
    record = {
        "stage": stage,
        "epoch": int(epoch),
        "loss": float(loss.detach().cpu()),
        **{
            f"loss_{name}": float(value.detach().cpu())
            for name, value in components.items()
        },
    }
    history.append(record)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def _prepare_epoch_metrics(
    path: Path,
    checkpoint: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    if checkpoint is not None and not path.is_file():
        raise ValueError(
            "checkpoint is missing epoch_metrics.jsonl required for exact resume"
        )
    if checkpoint is not None:
        checkpoint_stage = str(checkpoint["stage"])
        checkpoint_epoch = int(checkpoint["epoch"])
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_stage = str(record["stage"])
                record_epoch = int(record["epoch"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid epoch metrics record at line {line_number}"
                ) from error
            if record_stage not in STAGE_ORDER:
                raise ValueError(
                    f"unknown epoch metrics stage at line {line_number}: "
                    f"{record_stage}"
                )
            if (
                STAGE_ORDER[record_stage] < STAGE_ORDER[checkpoint_stage]
                or (
                    record_stage == checkpoint_stage
                    and record_epoch <= checkpoint_epoch
                )
            ):
                retained.append(record)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in retained:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    return retained


def _sparniche_aux(
    model: SparNiche,
    clean_rna: torch.Tensor,
    neighbor_idx: torch.Tensor,
    clean_adt: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return model(clean_rna, neighbor_idx, clean_adt).aux


def _initialize_negative_graph_mask(
    encoder: SparNicheEncoder,
    graph: dict[str, Any],
    repeats: int,
) -> None:
    graph_mask = build_sparniche_negative_mask(
        graph["adj_label"],
        repeats=repeats,
        seed=None,
    )
    encoder.set_graph_mask(graph_mask)


def _refresh_graph_logits(
    encoder: SparNicheEncoder,
    aux: dict[str, torch.Tensor],
) -> None:
    logits, graph_mask = encoder.decode_graph(aux["embedding"])
    aux["graph_logits"] = logits
    aux["graph_mask"] = graph_mask


def train_tensors(
    view1: torch.Tensor,
    neighbor_idx: torch.Tensor,
    sparniche_graph: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    resume: bool = False,
    data_fingerprint: str | None = None,
    view2: torch.Tensor | None = None,
) -> TrainingResult:
    config = normalize_sparniche_config(config)
    training = config.get("training", {})
    model_config = config.get("model", {})
    sparniche_config = model_config.get("sparniche", {})
    sparniche_view1_config = _resolve_sparniche_attention_config(
        model_config.get("sparniche_view1", {}),
        data_neighbors=int(config.get("data", {}).get("n_neighbors", neighbor_idx.shape[1])),
        available_neighbors=int(neighbor_idx.shape[1]),
    )
    model_config["sparniche_view1"] = sparniche_view1_config
    seed = int(training.get("seed", 2023))
    set_seed(seed)
    device = _resolve_device(str(training.get("device", "auto")))
    clean_view = view1.float().to(device)
    double_view = bool(model_config.get("double_view", False))
    if double_view != (view2 is not None):
        raise ValueError("double_view and supplied ADT view2 must agree")
    clean_adt = view2.float().to(device) if view2 is not None else None
    neighbor_idx = neighbor_idx.long().to(device)

    model = SparNiche(
        input_dim=view1.shape[1],
        latent_dim=int(model_config.get("latent_dim", 32)),
        sparniche_config=sparniche_view1_config,
        feature_graph_fusion_mode=str(
            model_config.get("feature_graph_fusion_mode", "gated")
        ),
        adaptive_graph=bool(model_config.get("sparniche_adaptive_graph", False)),
        local_graph_mode=str(model_config.get("local_graph_mode", "normalized")),
        local_graph_hops=int(model_config.get("local_graph_hops", 1)),
        local_graph_normalize=bool(model_config.get("local_graph_normalize", True)),
        residual_gcn_layers=int(model_config.get("residual_gcn_layers", 1)),
        mlp_layers=int(model_config.get("mlp_layers", 2)),
        double_view=double_view,
        view2_dim=clean_adt.shape[1] if clean_adt is not None else None,
    ).to(device)
    sparniche_encoder = model.encoder
    model.configure_graph(sparniche_graph)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    checkpoint = None
    if resume and checkpoint_path.is_file():
        checkpoint = _load_checkpoint(
            checkpoint_path,
            model,
            config,
            data_fingerprint,
            device,
        )

    epoch_metrics_path = output_dir / "epoch_metrics.jsonl"
    history = _prepare_epoch_metrics(epoch_metrics_path, checkpoint)
    checkpoint_every = max(1, int(training.get("checkpoint_every", 10)))
    negative_repeats = int(sparniche_config.get("negative_repeats", 1))
    graph_mask_ready = bool(
        checkpoint is not None
        and checkpoint.get("stage") in {"non_dec", "dec", "complete"}
        and checkpoint.get("sparniche_graph_mask") is not None
    )

    gan_epochs = int(sparniche_config.get("gan_epochs", 80))
    gan_start = _stage_start(checkpoint, "gan")
    if gan_start is not None and gan_start <= gan_epochs:
        discriminator_parameters = [
            *sparniche_encoder.layers.parameters(),
            *sparniche_encoder.layer.parameters(),
            *sparniche_encoder.adjust_layer.parameters(),
            *sparniche_encoder.discriminator.parameters(),
        ]
        discriminator_optimizer = torch.optim.Adam(
            discriminator_parameters,
            lr=float(sparniche_config.get("gan_lr", 1e-4)),
        )
        generator_optimizer = torch.optim.Adam(
            sparniche_encoder.generator.parameters(),
            lr=float(sparniche_config.get("gan_lr", 1e-4)),
        )
        if checkpoint is not None and checkpoint["stage"] == "gan":
            states = checkpoint.get("optimizer_state_dicts", {})
            discriminator_optimizer.load_state_dict(states["discriminator"])
            generator_optimizer.load_state_dict(states["generator"])
        progress = tqdm(
            range(gan_start, gan_epochs + 1),
            desc="[SparNiche] GAN",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in progress:
            model.train()
            discriminator_loss = None
            for _ in range(max(1, int(sparniche_config.get("gan_discriminator_steps", 1)))):
                discriminator_optimizer.zero_grad(set_to_none=True)
                _, discriminator_loss = sparniche_encoder.gan_losses(clean_view, neighbor_idx)
                discriminator_loss.backward()
                discriminator_optimizer.step()
            generator_optimizer.zero_grad(set_to_none=True)
            generator_loss, _ = sparniche_encoder.gan_losses(clean_view, neighbor_idx)
            generator_loss.backward()
            generator_optimizer.step()
            if discriminator_loss is None:
                raise RuntimeError("SparNiche discriminator stage produced no loss")
            total = generator_loss + discriminator_loss
            _record(
                history,
                epoch_metrics_path,
                "gan",
                epoch,
                total,
                {
                    "gan_generator": generator_loss,
                    "gan_discriminator": discriminator_loss,
                },
            )
            progress.set_postfix(loss=f"{float(total.detach().cpu()):.8f}")
            if _checkpoint_due(epoch, gan_epochs, checkpoint_every):
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    {
                        "discriminator": discriminator_optimizer,
                        "generator": generator_optimizer,
                    },
                    "gan",
                    epoch,
                    config,
                    data_fingerprint,
                )

    sparniche_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(sparniche_config.get("lr", 0.01)),
        weight_decay=float(sparniche_config.get("weight_decay", 0.01)),
    )
    if checkpoint is not None and checkpoint["stage"] in {"non_dec", "dec"}:
        state = checkpoint.get("optimizer_state_dicts", {}).get("sparniche")
        if state is None:
            raise ValueError("SparNiche checkpoint is missing its optimizer state")
        sparniche_optimizer.load_state_dict(state)

    rec_w = float(sparniche_config.get("rec_w", 10.0))
    gcn_w = float(sparniche_config.get("gcn_w", 0.1))
    self_w = float(sparniche_config.get("self_w", 1.0))
    contrastive_w = float(sparniche_config.get("contrastive_w", 0.0))
    contrastive_temperature = float(sparniche_config.get("contrastive_temperature", 0.2))
    dec_w = float(sparniche_config.get("dec_kl_w", 1.0))
    adt_w = float(sparniche_config.get("adt_rec_w", 1.0))
    if adt_w < 0:
        raise ValueError("adt_rec_w must be non-negative")

    non_dec_epochs = int(sparniche_config.get("pretrain_epochs", 80))
    non_dec_start = _stage_start(checkpoint, "non_dec")
    if non_dec_start is not None and non_dec_start <= non_dec_epochs:
        progress = tqdm(
            range(non_dec_start, non_dec_epochs + 1),
            desc="[SparNiche] pretrain",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in progress:
            model.train()
            sparniche_optimizer.zero_grad(set_to_none=True)
            aux = _sparniche_aux(model, clean_view, neighbor_idx, clean_adt)
            if not graph_mask_ready:
                _initialize_negative_graph_mask(
                    sparniche_encoder,
                    sparniche_graph,
                    negative_repeats,
                )
                _refresh_graph_logits(sparniche_encoder, aux)
                graph_mask_ready = True
            components = _sparniche_losses(
                aux, clean_view, None, contrastive_w, contrastive_temperature,
                clean_adt=clean_adt,
            )
            total = (
                rec_w * components["reconstruction"]
                + gcn_w * components["graph"]
                + self_w * components["self"]
                + components["contrastive"]
                + adt_w * components.get("adt_reconstruction", 0.0)
            )
            total.backward()
            sparniche_optimizer.step()
            _record(
                history,
                epoch_metrics_path,
                "non_dec",
                epoch,
                total,
                components,
            )
            progress.set_postfix(loss=f"{float(total.detach().cpu()):.8f}")
            if _checkpoint_due(epoch, non_dec_epochs, checkpoint_every):
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    {"sparniche": sparniche_optimizer},
                    "non_dec",
                    epoch,
                    config,
                    data_fingerprint,
                )

    if not graph_mask_ready:
        _initialize_negative_graph_mask(
            sparniche_encoder,
            sparniche_graph,
            negative_repeats,
        )
        graph_mask_ready = True

    dec_epochs = int(training.get("epochs", 550))
    dec_start = _stage_start(checkpoint, "dec")
    target_distribution = None
    previous_labels = None
    if checkpoint is not None and checkpoint["stage"] == "dec":
        extras = checkpoint.get("extras", {})
        target_distribution = extras.get("dec_target_distribution")
        previous_labels = extras.get("dec_previous_labels")
        if target_distribution is not None:
            target_distribution = target_distribution.to(device)
        if previous_labels is not None:
            previous_labels = np.asarray(previous_labels)
    elif dec_start is not None:
        model.eval()
        with torch.no_grad():
            initial_aux = _sparniche_aux(model, clean_view, neighbor_idx, clean_adt)
        cluster_count = sparniche_encoder.dec_cluster_n
        if clean_view.shape[0] < cluster_count:
            raise ValueError(
                "SparNiche DEC requires at least dec_cluster_n spots for KMeans initialization"
            )
        kmeans = KMeans(
            n_clusters=cluster_count,
            n_init=cluster_count * 2,
            random_state=42,
        )
        previous_labels = kmeans.fit_predict(
            initial_aux["embedding"].detach().cpu().numpy()
        )
        sparniche_encoder.cluster_layer.data = torch.as_tensor(
            kmeans.cluster_centers_,
            dtype=clean_view.dtype,
            device=device,
        )

    if dec_start is not None and dec_start <= dec_epochs:
        dec_interval = max(1, int(sparniche_config.get("dec_interval", 20)))
        dec_tolerance = float(sparniche_config.get("dec_tol", 0.0))
        progress = tqdm(
            range(dec_start, dec_epochs + 1),
                desc="[SparNiche] DEC",
            unit="epoch",
            dynamic_ncols=True,
        )
        for epoch in progress:
            epoch_id = epoch - 1
            if epoch_id % dec_interval == 0:
                model.eval()
                with torch.no_grad():
                    update_aux = _sparniche_aux(
                        model,
                        clean_view,
                        neighbor_idx,
                        clean_adt,
                    )
                target_distribution = _sparniche_target_distribution(
                    update_aux["q"].detach()
                ).detach()
                labels = target_distribution.detach().cpu().numpy().argmax(1)
                if previous_labels is not None:
                    delta = float(np.mean(labels != previous_labels))
                    if epoch_id > 0 and delta < dec_tolerance:
                        break
                previous_labels = labels.copy()

            if target_distribution is None:
                raise RuntimeError("SparNiche DEC target distribution was not initialized")
            model.train()
            sparniche_optimizer.zero_grad(set_to_none=True)
            aux = _sparniche_aux(model, clean_view, neighbor_idx, clean_adt)
            components = _sparniche_losses(
                aux, clean_view, target_distribution,
                contrastive_w, contrastive_temperature,
                clean_adt=clean_adt,
            )
            total = (
                gcn_w * components["graph"]
                + dec_w * components["dec"]
                + rec_w * components["reconstruction"]
                + components["contrastive"]
                + adt_w * components.get("adt_reconstruction", 0.0)
            )
            total.backward()
            sparniche_optimizer.step()
            _record(
                history,
                epoch_metrics_path,
                "dec",
                epoch,
                total,
                components,
            )
            progress.set_postfix(loss=f"{float(total.detach().cpu()):.8f}")
            if _checkpoint_due(epoch, dec_epochs, checkpoint_every):
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    {"sparniche": sparniche_optimizer},
                    "dec",
                    epoch,
                    config,
                    data_fingerprint,
                    extras={
                        "dec_target_distribution": target_distribution.detach().cpu(),
                        "dec_previous_labels": previous_labels,
                    },
                )

    model.eval()
    with torch.no_grad():
        final_output = model(clean_view, neighbor_idx, clean_adt)
    last_epoch = int(training.get("epochs", 550))
    _save_checkpoint(
        checkpoint_path,
        model,
        {},
        "complete",
        last_epoch,
        config,
        data_fingerprint,
    )
    return TrainingResult(
        model=model,
        output=final_output,
        history=history,
        last_epoch=last_epoch,
    )
