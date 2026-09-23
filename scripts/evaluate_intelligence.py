"""Offline evaluation of the intelligence layer on the real data.

Writes experiments/intel-eval-<UTC ts>/{report.json, REPORT.md} and prints the headline numbers.
--as-of YYYY-MM-DD   evaluate as of a past date (default: last MovieLens event)
--latency-runs N     pipeline runs for the latency measurement (default 5)
"""

from __future__ import annotations

import argparse
import json
import logging

from jev_ml.intel import load_default_inputs
from jev_ml.intel.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--latency-runs", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rep, out = run_evaluation(load_default_inputs(as_of=args.as_of), n_latency_runs=args.latency_runs)
    print(
        json.dumps(
            {
                "run_dir": str(out),
                "forecast": rep["forecast"]["summary"],
                "lapse": {
                    k: rep["lapse"]["metrics"].get(k)
                    for k in ("auc", "baseline_auc", "brier", "ece", "base_rate")
                },
                "shilling": [
                    {
                        k: a[k]
                        for k in ("type", "precision", "recall", "f1", "auc", "baseline_f1", "baseline_auc")
                    }
                    for a in rep["anomaly"]["injection"]["attack_types"]
                ],
                "series_anomaly": {
                    k: rep["anomaly"]["series"][k] for k in ("detection_rate", "false_alarm_rate", "n_trials")
                },
                "change_point": {
                    k: rep["change_point"][k]
                    for k in (
                        "detection_rate",
                        "false_alarm_rate",
                        "mean_abs_location_error_months",
                        "n_trials",
                    )
                },
                "latency": {k: rep["latency"][k] for k in ("pipeline_ms_mean", "pipeline_ms_p95", "n_runs")},
                "notes": rep["notes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
