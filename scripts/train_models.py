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
    # Phase 2 governance: a new version registers as a CANDIDATE (gate it with scripts/retrain.py
    # evaluate VERSION, then promote). Only the bootstrap model of an empty registry is activated.
    parser.add_argument(
        "--activate",
        action="store_true",
        help="activate the new version even if one is active (bypasses the gate)",
    )
    parser.add_argument(
        "--no-activate", action="store_true", help="never activate, not even the bootstrap model"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="afterwards fit the recommendation-confidence calibrator (validation split) for the new "
        "version; same as scripts/calibrate_recommendations.py --model VERSION",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from jev_ml.registry import active_version

    bootstrap = active_version() is None
    activate = not args.no_activate and (args.activate or bootstrap)
    res = run_pipeline(args.config, quick=args.quick, activate=activate)
    calibration = None
    if args.calibrate and res["model_version"]:
        from jev_ml.calibration import calibrate_model

        cal = calibrate_model(res["model_version"])
        calibration = {"calibration_version": cal["calibration_version"], "headline": cal["headline"]}
    print(
        json.dumps(
            {
                "run_id": res["run_id"],
                "model_version": res["model_version"],
                "activated": activate,
                "seconds": round(res["seconds_total"], 1),
                "test_ndcg@10": {k: round(v["ndcg@10"], 4) for k, v in res["metrics"]["test"].items()},
                "calibration": calibration,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
