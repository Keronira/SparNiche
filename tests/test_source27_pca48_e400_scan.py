"""Selection contracts for the source27 scan around PCA48/400 epochs."""

import pandas as pd

from scripts.tmp_run_source27_pca48_e400_scan import (
    ANCHOR,
    build_stage1_specs,
    choose_final_spec,
    prepare_output_root,
    select_stage2_specs,
)


def test_stage1_is_unique_one_factor_scan_around_new_anchor():
    specs = build_stage1_specs()
    assert len(specs) == 15
    assert len({spec["config_id"] for spec in specs}) == 15
    assert specs[0] == {"config_id": "anchor", "factors": ANCHOR}
    assert all(
        sum(spec["factors"][key] != ANCHOR[key] for key in ANCHOR) <= 1
        for spec in specs
    )
    assert {spec["factors"]["latent"] for spec in specs} == {16, 24, 32}
    assert {spec["factors"]["neighbors"] for spec in specs} == {12, 16, 20}
    assert {spec["factors"]["epochs"] for spec in specs} == {300, 400, 500}


def test_stage2_uses_three_sample_means_and_excludes_incomplete_candidates():
    specs = build_stage1_specs()
    samples = ["s1", "s2", "s3"]
    values = {
        "anchor": [0.20, 0.20, 0.20],
        "pca_32": [0.23, 0.23, 0.23],
        "pca_64": [0.32, 0.18, 0.18],
        "latent_24": [0.22, 0.22, 0.22],
        "neighbors_20": [0.215, 0.215, 0.215],
        "epochs_300": [0.19, 0.19, 0.19],
        "lr_0p005": [0.50, 0.50],
    }
    rows = [
        {"config_id": name, "sample": sample, "seed": 1234,
         "status": "completed", "ari": ari, "nmi": ari}
        for name, scores in values.items()
        for sample, ari in zip(samples, scores)
    ]
    selected, factors = select_stage2_specs(pd.DataFrame(rows), specs, samples)
    assert factors == ["pca", "latent", "neighbors"]
    assert len(selected) == 7
    assert {spec["config_id"] for spec in selected} == {
        "anchor", "pca_32", "latent_24", "neighbors_20",
        "pca_32__latent_24", "pca_32__neighbors_20",
        "latent_24__neighbors_20",
    }


def test_final_selection_requires_complete_runs_and_two_sample_ari_gain():
    specs = [
        {"config_id": name, "factors": dict(ANCHOR)}
        for name in ("anchor", "one_sample", "balanced", "incomplete")
    ]
    rows = []
    for name, aris, nseeds in (
        ("anchor", [0.20, 0.20, 0.20], 3),
        ("one_sample", [0.40, 0.19, 0.19], 3),
        ("balanced", [0.23, 0.22, 0.19], 3),
        ("incomplete", [0.50, 0.50, 0.50], 2),
    ):
        for sample, ari in zip(("s1", "s2", "s3"), aris):
            for seed in (1234, 1235, 1236)[:nseeds]:
                rows.append({"config_id": name, "sample": sample, "seed": seed,
                             "status": "completed", "ari": ari, "nmi": ari,
                             "fmi": ari, "accuracy": ari, "macro_f1": ari})
    selected = choose_final_spec(pd.DataFrame(rows), specs, ("s1", "s2", "s3"),
                                 (1234, 1235, 1236))
    assert selected["config_id"] == "balanced"


def test_final_selection_keeps_anchor_when_candidate_loses_weighted_score():
    specs = [{"config_id": name, "factors": dict(ANCHOR)}
             for name in ("anchor", "ari_only")]
    rows = []
    for sample in ("s1", "s2", "s3"):
        for seed in (1234, 1235, 1236):
            rows.append({"config_id": "anchor", "sample": sample, "seed": seed,
                         "status": "completed", "ari": 0.20, "nmi": 0.30,
                         "fmi": 0.30, "accuracy": 0.30, "macro_f1": 0.30})
            rows.append({"config_id": "ari_only", "sample": sample, "seed": seed,
                         "status": "completed", "ari": 0.21, "nmi": 0.10,
                         "fmi": 0.10, "accuracy": 0.10, "macro_f1": 0.10})
    selected = choose_final_spec(pd.DataFrame(rows), specs, ("s1", "s2", "s3"),
                                 (1234, 1235, 1236))
    assert selected["config_id"] == "anchor"


def test_matching_manifest_allows_resuming_output_directory(tmp_path):
    root = tmp_path / "scan"
    manifest = {"samples": ("s1", "s2"), "seeds": (1234, 1235)}
    prepare_output_root(root, manifest)
    prepare_output_root(root, manifest)
    assert (root / "manifest.json").is_file()
