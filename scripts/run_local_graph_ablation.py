#!/usr/bin/env python3
"""Run the two-arm graph-loss comparison on all samples and three seeds."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_experiments import main as run_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", required=True, nargs="+")
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--experiment-set", "ablation_local_graph", "--source", *known.source,
                "--seeds", "2024", "2025", "2026",
                "--skip-visual-artifacts", *remaining]
    raise SystemExit(run_main())
