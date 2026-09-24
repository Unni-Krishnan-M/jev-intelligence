"""Fit the recommendation-confidence calibrator for a model version and write its calibration.json.

The calibrator (isotonic regression, P(rating >= 4)) is fitted on the VALIDATION split of the
model's recorded experiment protocol and evaluated on the TEST split; see jev_ml/calibration.py.

--model VERSION   model version to calibrate (default: the active one in models/registry.json)
--k N             list length scored per user (default 50; ranks beyond it get no confidence)
--method M        isotonic | logistic (per stratum, the lower validation cross-fit log-loss of
                  isotonic and logistic is served; see jev_ml/calibration.py)
--dry-run         compute and print the metrics without writing the file
"""

from __future__ import annotations

import argparse
import json
import logging

from jev_ml.calibration import DEFAULT_K, METHODS, calibrate_model
from jev_ml.paths import MODELS_DIR
from jev_ml.registry import active_version

DEFAULT_METHOD = "isotonic"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=None, help="model version (default: active)")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--method", choices=METHODS, default=DEFAULT_METHOD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    version = args.model or active_version(MODELS_DIR)
    if not version:
        raise SystemExit("no active model; pass --model VERSION")
    out = calibrate_model(version, MODELS_DIR, k=args.k, write=not args.dry_run, method=args.method)
    print(
        json.dumps(
            {
                "model_version": out["model_version"],
                "calibration_version": out["calibration_version"],
                "target": out["target"],
                "strata": [
                    {
                        "name": st["name"],
                        "applies_to": st["applies_to"],
                        "feature": st["feature"],
                        "serving_method": st.get("serving_method", "isotonic"),
                        "selection": st["selection"]["candidates"],
                        "validation_base_rate": st["base_rate"],
                        "test": {
                            k: st.get("test_served", st["test"]).get(k)
                            for k in (
                                "n",
                                "observed_rate",
                                "mean_predicted",
                                "ece",
                                "ece_equal_mass",
                                "brier",
                                "base_rate_brier",
                                "brier_skill_vs_base_rate",
                                "auc",
                            )
                        },
                    }
                    for st in out["strata"]
                ],
                "written": None if args.dry_run else str(MODELS_DIR / version / "calibration.json"),
                "seconds": out["seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
