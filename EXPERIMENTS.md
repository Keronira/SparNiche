# SparNiche experiments

## Default model

| Setting | Default |
| --- | --- |
| RNA preprocessing | filter genes/spots, normalize to `1e6`, HVG-2000, scale, PCA-200 |
| Spatial graph | 12 nearest neighbors, symmetric normalization |
| Attention | `spatial_local`, neighbor/chunk settings left `null` by default |
| Latent dimension | 32 |
| Internal RNA/graph fusion | gated |
| Training | 80 GAN + 80 non-DEC + 550 DEC epochs |
| Optimizer | Adam, learning rate 0.01, weight decay 0.01 |
| Labels | `adata.obs["annotation_final"]` |
| Repeats | seeds 1234, 1235, 1236 |

The global default uses latent dimension 32 and SparNiche learning rate 0.01.
Source24 keeps a dedicated overlay with latent dimension 64, learning rate
0.005, 12 attention neighbors, and chunk size 4096. An explicit `--config`
path remains authoritative.

Ambiguous labels such as `Unknown` are excluded from benchmark scoring. `n_clusters` is inferred from the remaining ground-truth categories unless it is supplied explicitly.

## `run_experiments.py` experiment sets

| `--experiment-set` | Meaning |
| --- | --- |
| `single` | Standard SparNiche run. This is the normal benchmark entry point. |
| `local_graph` | Runs the normalized local-graph configuration declared in `configs/experiment_local_graph.yaml`. |
| `ablation_local_graph` | Runs the anchor and graph-loss-zero cases in `configs/experiment_ablation_local_graph.yaml`. Each case receives separate `SparNiche__<case>_rep*` directories. |

Useful controls include `--seeds`, `--repeats`, `--no-repeats`, `--epochs`, `--attention-mode`, `--leiden-min-resolution`, `--leiden-max-resolution`, `--device`, `--resume`, `--force`, and `--continue-on-error`.

Example:

```bash
python scripts/run_experiments.py \
  --experiment-set single \
  --source /root/autodl-fs/data/source24 \
  --seeds 1234 1235 1236 \
  --attention-mode spatial_local \
  --device cuda \
  --continue-on-error
```

## Leiden prediction

Clustering uses the SparNiche latent embedding. The search first probes the configured lower bound, midpoint, and upper bound, then narrows by midpoint probes until the observed cluster count reaches the target window. It traverses the relevant local interval and selects the candidate with the highest ARI. The default accepted cluster-count window is `K` through `K+2`, where `K` is the number of non-ambiguous ground-truth classes. If the window is not reached, the same ARI rule is applied to the evaluated fallback candidates.

## Embedding parameter experiments

`scripts/run_final_embedding_experiment.py` evaluates the last six source24 samples with three seeds and reports:

- ARI, NMI, FMI, accuracy, and Macro-F1;
- layer-order score;
- spatial fidelity;
- mean and standard deviation by sample, seed, and parameter set.

The named sets cover spatial neighbors, latent dimension, graph normalization, dropout, learning rate, weight decay, and DEC epochs. The `final` set keeps the selected one-factor settings and the `lr_low_latent_64` interaction. Existing anchor results are read from `SparNiche_rep*` rather than retrained.

```bash
python scripts/run_final_embedding_experiment.py \
  --experiment-set final \
  --source /root/autodl-fs/data/source24 \
  --anchor-root '/root/autodl-fs/bench_results/source24/SparNiche_rep*' \
  --output-root /root/autodl-fs/bench_results/source24/SparNiche_final \
  --device cuda
```

Primary summary files are:

```text
SparNiche_final/metrics.csv
SparNiche_final/summary_overall.csv
SparNiche_final/summary_by_sample.csv
SparNiche_final/summary_by_seed.csv
```

## Output contract

For each benchmark sample:

- `<sample>.csv`: spot IDs, ground truth, and predicted labels;
- `<sample>_embedding.csv`: spot IDs and latent dimensions;
- `<sample>_profile.json`: runtime and memory information;
- `artifacts/<run_id>/checkpoint.pt`: resumable model state;
- `artifacts/<run_id>/epoch_metrics.jsonl`: epoch-level training history;
- `artifacts/<run_id>/trained.h5ad`: AnnData with `obsm["sparniche"]`.

No image features, multimodal routing weights, reliability scores, or cross-view corruption outputs are produced.
