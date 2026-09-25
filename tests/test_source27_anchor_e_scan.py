"""Contract tests for the temporary source27 anchor-E search."""

import pandas as pd

from scripts.tmp_run_source27_anchor_e_scan import (
    ANCHOR,
    build_stage1_specs,
    choose_stage2_specs,
    select_stage3_specs,
)


def test_stage1_changes_only_one_factor_and_keeps_anchor():
    specs = build_stage1_specs()
    assert len(specs) == 21
    assert len({spec["config_id"] for spec in specs}) == 21
    assert specs[0]["factors"] == ANCHOR
    assert all(
        sum(spec["factors"][key] != ANCHOR[key] for key in ANCHOR) <= 1
        for spec in specs
    )


def test_stage2_uses_top_three_screen_effects_and_six_local_combinations():
    specs = build_stage1_specs()
    rows = []
    gains = {"hvg": 0.03, "pca": 0.08, "latent": 0.12,
             "neighbors": 0.10, "lr": 0.02, "dropout": 0.01,
             "weight_decay": 0.00, "epochs": -0.01}
    for spec in specs:
        changed = [key for key in ANCHOR if spec["factors"][key] != ANCHOR[key]]
        rows.append({"config_id": spec["config_id"], "status": "completed",
                     "ari": 0.20 + (gains[changed[0]] if changed else 0.0),
                     "nmi": 0.4})
    selected, factors = choose_stage2_specs(pd.DataFrame(rows), specs)
    assert factors == ["latent", "neighbors", "pca"]
    assert len(selected) == 7  # anchor, three single changes, three pairs
    assert len({spec["config_id"] for spec in selected}) == 7


def test_stage3_keeps_anchor_and_selects_top_two_nonanchors():
    rows = []
    for config_id, value in (("anchor", 0.20), ("one", 0.24),
                             ("two", 0.23), ("three", 0.21)):
        for seed in (1234, 1235, 1236):
            rows.append({"config_id": config_id, "status": "completed",
                         "seed": seed, "ari": value, "nmi": value,
                         "fmi": value, "accuracy": value, "macro_f1": value})
    selected = select_stage3_specs(pd.DataFrame(rows),
                                   [{"config_id": name, "factors": dict(ANCHOR)}
                                    for name in ("anchor", "one", "two", "three")])
    assert [spec["config_id"] for spec in selected] == ["anchor", "one", "two"]
