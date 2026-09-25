# SparNiche

SparNiche is a single-view spatial transcriptomics model for spatial-domain discovery. It consumes RNA expression features and a spatial-neighbor graph; tissue images and image embeddings are not read by the model.

## Architecture

1. Filter, normalize, scale, and reduce RNA counts to a configurable feature matrix (PCA-200 by default).
2. Build a symmetric spatial k-nearest-neighbor graph from `adata.obsm["spatial"]`.
3. Encode RNA features with global or memory-safe spatial-local attention and graph convolutions.
4. Combine the RNA-feature and graph representations with an internal gate.
5. Train in three stages: adversarial pretraining, non-DEC reconstruction/graph training, and DEC refinement.
6. Export the latent embedding and select Leiden labels with the project resolution search (`K` to `K+2` clusters by default).

The word *gate* refers to fusion inside the single RNA/graph encoder. SparNiche has no second modality, image encoder, cross-view expert, or multimodal router.

## Input contract

Each input is one `.h5ad` file.

- RNA matrix: `adata.X` (raw counts or normalized values; detected automatically by the batch runner).
- Spatial coordinates: `adata.obsm["spatial"]`, shape `[n_spots, 2]` or wider.
- Ground-truth labels for benchmark evaluation: `adata.obs["annotation_final"]`.
- Optional reusable RNA features: `adata.obsm["feat"]`, accepted only when its preprocessing metadata matches the configured contract.
- Optional ADT matrix: `adata.obsm["adt"]`, aligned by spot with `adata.X`.

Image content in `adata.uns["spatial"]` or `adata.obsm["image_features"]` is ignored.

ADT training is disabled by default (`model.double_view: false`). To enable
RNA-primary gated fusion and ADT reconstruction, pass `--double-view`; use
`--view2-key NAME` when the ADT matrix is stored under another `obsm` key.
Every double-view sample must contain both RNA in `X` and aligned ADT in that
key. An RNA-only sample should be run without `--double-view`.

## Installation

```bash
git clone https://github.com/Keronira/SparNiche.git
cd SparNiche
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py"
```

## Run experiments

Run a directory of `.h5ad` samples with three seeds:

```bash
python scripts/run_experiments.py \
  --experiment-set single \
  --source /root/autodl-fs/data/source24 \
  --attention-mode spatial_local \
  --device cuda \
  --continue-on-error
```

For source30, run S1 with ADT and S2 as RNA-only:

```bash
python scripts/run_experiments.py --experiment-set single \
  --source /root/autodl-fs/data/source30/S1.h5ad \
  --double-view --view2-key adt --device cuda
python scripts/run_experiments.py --experiment-set single \
  --source /root/autodl-fs/data/source30/S2.h5ad --device cuda
```

The shared default configuration uses `latent_dim=32`, a SparNiche learning
rate of `0.01`, and null local-attention neighbor and chunk settings. Source24
continues to use its dedicated overlay in `configs/source24.yaml`, which sets
`latent_dim=64`, `lr=0.005`, `attention_neighbors=12`, and
`attention_chunk_size=4096`. Passing `--config` disables the automatic overlay.

The default output root is `/root/autodl-fs/bench_results`. Seeds map to stable method directories:

```text
bench_results/<source>/SparNiche_rep1/
bench_results/<source>/SparNiche_rep2/
bench_results/<source>/SparNiche_rep3/
```

Each method directory contains `results/<sample>.csv`, `results/<sample>_embedding.csv`, `results/<sample>_profile.json`, and training artifacts under `artifacts/`.

Run one worker directly with a resolved YAML configuration:

```bash
python scripts/run_sparniche.py \
  --config configs/config.yaml \
  --output-dir outputs/example
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for experiment-set semantics, parameter screening, Leiden evaluation, and output summaries.

## Repository layout

```text
configs/    Default model and experiment matrices
scripts/    Batch runners, parameter experiments, aggregation, and plotting
src/        Data preparation, graph/model/training, evaluation, and exports
tests/      Unit and small integration tests
```

Datasets, checkpoints, generated embeddings, benchmark outputs, and trained `.h5ad` files are intentionally excluded from version control.
