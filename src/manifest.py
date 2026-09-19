from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any


def _dataset_name(dataset: str | dict[str, Any]) -> str:
    if isinstance(dataset, str):
        return dataset
    if isinstance(dataset, dict) and dataset.get("name"):
        return str(dataset["name"])
    raise ValueError("each dataset must be a name or mapping with a name")


def expand_manifest(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    datasets = experiment.get("datasets")
    seeds = experiment.get("seeds")
    grid = experiment.get("grid", {})
    cases = experiment.get("cases", [{"name": "default", "overrides": {}}])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("datasets must be a non-empty list")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list")
    if not isinstance(grid, dict):
        raise ValueError("grid must be a mapping")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty list when provided")
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"grid dimension {key!r} must be a non-empty list")
    normalized_cases = []
    for case in cases:
        if not isinstance(case, dict) or not case.get("name"):
            raise ValueError("each case must be a mapping with a name")
        overrides = case.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError("case overrides must be a mapping")
        normalized_cases.append((str(case["name"]), overrides))

    keys = list(grid)
    rows = []
    for dataset in datasets:
        dataset_name = _dataset_name(dataset)
        for seed in seeds:
            for case_name, case_overrides in normalized_cases:
                combinations = itertools.product(*(grid[key] for key in keys)) if keys else [()]
                for values in combinations:
                    overrides = dict(zip(keys, values))
                    overrides.update(case_overrides)
                    identity = {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "case": case_name,
                        "overrides": overrides,
                    }
                    digest = hashlib.sha256(
                        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()[:12]
                    rows.append({"run_id": f"{dataset_name}-{int(seed)}-{digest}", **identity})
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError(
            "duplicate run identity generated; remove duplicate datasets, seeds, grid values, or cases"
        )
    return rows
