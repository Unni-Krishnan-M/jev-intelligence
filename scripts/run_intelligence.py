"""Run the intelligence pipeline on the real data and print its summary and stage timings.

--as-of YYYY-MM-DD   replay the pipeline as of a past date (default: last MovieLens event)
--out FILE.json      also write the full PipelineResult.to_dict() as JSON
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from jev_ml.intel import load_default_inputs, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD (UTC midnight)")
    parser.add_argument("--out", default=None, help="write the full result JSON here")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    result = run_pipeline(load_default_inputs(as_of=args.as_of))
    d = result.to_dict()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "as_of": d["run"]["as_of"],
                "data_version": d["run"]["data_version"],
                "model_version": d["run"]["model_version"],
                "summary": d["summary"],
                "quality_score": d["data"]["quality"]["score"],
                "warnings": [
                    {"severity": w["severity"], "key": w["key"], "title": w["title"]} for w in d["warnings"]
                ],
                "decisions": [
                    {
                        "key": x["key"],
                        "entity": x["entity"],
                        "answer": x["answer"],
                        "confidence": x["confidence"],
                        "confidence_kind": x["confidence_kind"],
                        "abstained": x["abstained"],
                    }
                    for x in d["decisions"]
                    if x["key"] != "genre_programming" or x["answer"] != "hold"
                ],
                "actions": [{"priority": a["priority"], "title": a["title"]} for a in d["actions"]],
                "diagnostics": d["diagnostics"],
                "stage_ms": d["run"]["stage_ms"],
                "written": args.out,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
