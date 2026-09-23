"""Stage 3: split → tune → evaluate → train production models → register version → report."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from jev_ml.training import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=None, help="experiment YAML (default configs/experiment.yaml)"
    )
    parser.add_argument("--quick", action="store_true", help="skip hyper-parameter tuning")
    parser.add_argument("--no-activate", action="store_true", help="register without activating")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    res = run_pipeline(args.config, quick=args.quick, activate=not args.no_activate)
    print(
        json.dumps(
            {
                "run_id": res["run_id"],
                "model_version": res["model_version"],
                "seconds": round(res["seconds_total"], 1),
                "test_ndcg@10": {k: round(v["ndcg@10"], 4) for k, v in res["metrics"]["test"].items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
