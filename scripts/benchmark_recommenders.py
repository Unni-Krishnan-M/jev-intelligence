"""Reproducible recommender benchmark: user-temporal and global-temporal protocols, bootstrap CIs,
paired tests, cold-start strategies, confidence calibration and equally-tuned baselines.

Writes experiments/rec-benchmark-<ts>/{results.json, per_user.json, config.yaml, REPORT.md}.
Every number in docs/ML_IMPROVEMENT_REPORT.md comes from a run of this script.

  uv run python scripts/benchmark_recommenders.py            # full run (~11 min on a laptop CPU)
  uv run python scripts/benchmark_recommenders.py --quick    # CI smoke (~1-2 min, subsampled users)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from jev_ml.evaluation.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="experiment YAML (default configs/experiment.yaml)"
    )
    parser.add_argument("--quick", action="store_true", help="smoke mode: subsampled users, no retuning")
    parser.add_argument(
        "--protocol",
        action="append",
        choices=["user_temporal", "global_temporal"],
        help="restrict to one protocol (repeatable; default: both)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("jev_ml.evaluation.evaluator").setLevel(logging.WARNING)
    res, out = run_benchmark(args.config, quick=args.quick, protocols=args.protocol)
    print(
        json.dumps(
            {
                "run_id": res["run_id"],
                "out": str(out),
                "decision": res["decision"],
                "seconds": res["seconds_total"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
