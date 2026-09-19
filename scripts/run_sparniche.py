#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import apply_overrides, load_yaml  # noqa: E402
from src.runner import run_experiment  # noqa: E402


def _parse_overrides(values: list[str]) -> dict[str, object]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"override must use KEY=VALUE: {value!r}")
        key, raw = value.split("=", 1)
        overrides[key] = yaml.safe_load(raw)
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one scheduler-independent SparNiche experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = apply_overrides(load_yaml(args.config), _parse_overrides(args.overrides))
    result = run_experiment(config, args.output_dir, resume=args.resume, force=args.force)
    print("skipped" if result["skipped"] else "completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
