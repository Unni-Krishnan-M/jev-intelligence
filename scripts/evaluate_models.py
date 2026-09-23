"""Stage 4: re-run the evaluation protocol for a registered model version's parameters.

Fits the components on train+val with the version's stored (tuned) parameters, evaluates every
model on the test split (+ cold-start protocol) and writes a new experiment report. No new model
version is produced.
"""

from __future__ import annotations

import argparse
import json
import logging

from jev_ml.registry import active_version, load_manifest
from jev_ml.training import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=None, help="model version (default: active)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    version = args.version or active_version()
    if version is None:
        raise SystemExit("no model registered; run scripts/train_models.py first")
    manifest = load_manifest(version)
    res = run_pipeline(train_production=False, fixed_models=manifest["training_config"]["models"])
    print(
        json.dumps(
            {
                "run_id": res["run_id"],
                "evaluated_params_of": version,
                "test": {
                    m: {
                        k: round(v[k], 4)
                        for k in ("precision@10", "recall@10", "ndcg@10", "map@10", "hit_rate@10")
                    }
                    for m, v in res["metrics"]["test"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
