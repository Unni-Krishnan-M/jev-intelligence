"""Run the JEV decision & early-warning pipeline on a domain and print its summary and timings.

--domain KEY         movie (default) or generic:<name> (configs/domains/<name>.yaml); --list shows all
--as-of YYYY-MM-DD   replay as of a past date (movie default: last MovieLens event; generic: now)
--out FILE.json      also write the full PipelineResult.to_dict() as JSON
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from jev_ml.core import run_domain
from jev_ml.domains import available, get_adapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--domain", default="movie", help="adapter key (default: movie)")
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD (UTC midnight)")
    parser.add_argument("--out", default=None, help="write the full result JSON here")
    parser.add_argument("--list", action="store_true", help="list the registered domains and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.list:
        for d in available():
            print(f"{d['key']:28s} available={d['available']}  {d['reason'] or ''}")
        return

    result = run_domain(get_adapter(args.domain), as_of=args.as_of)
    d = result.to_dict()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "domain": d["run"]["domain"],
                "as_of": d["run"]["as_of"],
                "data_version": d["run"]["data_version"],
                "model_version": d["run"]["model_version"],
                "summary": d["summary"],
                "quality_score": d["data"]["quality"]["score"],
                "warnings": [
                    {
                        "severity": w["severity"],
                        "level": w.get("early_warning_level"),
                        "key": w["key"],
                        "title": w["title"],
                    }
                    for w in d["warnings"]
                ],
                "decisions": [
                    {
                        "key": x["key"],
                        "entity": x.get("situation") or x["entity"],
                        "answer": x["answer"],
                        "confidence": x["confidence"],
                        "confidence_kind": x["confidence_kind"],
                        "abstained": x["abstained"],
                    }
                    for x in d["decisions"]
                    if not (x["key"] == "genre_programming" and x["answer"] == "hold")
                    and not (x["key"] == "early_warning_level" and x["answer"] in ("NO_ACTION", "MONITOR"))
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
