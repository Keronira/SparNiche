from __future__ import annotations

import pandas as pd

from scripts.run_multimodal_sensitivity import (
    SAMPLES,
    stage1_specs,
    stage2_specs,
    stage3_specs,
    select_winner,
)


def test_source29_atac_only_sample_is_not_in_rna_screen() -> None:
    assert ("source29", "E18.5_S1") not in SAMPLES
    assert len(SAMPLES) == 4


def test_three_stage_grid_sizes_and_carried_factors() -> None:
    first = stage1_specs()
    assert len(first) == 16
    assert len({(x["hvg"], x["pca"]) for x in first}) == 16
    second = stage2_specs(first[0])
    assert len(second) == 13
    assert all(x["hvg"] == first[0]["hvg"] and x["pca"] == first[0]["pca"] for x in second)
    third = stage3_specs(second[0])
    assert len(third) == 12
    assert len({(x["latent"], x["neighbors"]) for x in third}) == 12


def test_winner_requires_all_samples_and_seeds() -> None:
    rows = [
        {"config_id": spec, "sample": sample, "seed": seed, "ari": score,
         "status": "completed"}
        for spec, score in (("complete", 0.4), ("incomplete", 0.9))
        for sample in ("E13.5_S1", "E15.5_S1", "S1", "S2")
        for seed in (1234, 1235, 1236)
        if not (spec == "incomplete" and sample == "S2" and seed == 1236)
    ]
    assert select_winner(pd.DataFrame(rows)) == "complete"
