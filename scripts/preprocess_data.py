"""Stage 2: validate, clean, transform raw data into data/processed/."""

from __future__ import annotations

import argparse
import json
import logging

from jev_ml.data.preprocess import preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-enrichment", action="store_true", help="ignore Wikidata metadata")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    meta = preprocess(enrichment_path=None) if args.no_enrichment else preprocess()
    print(
        json.dumps(
            {
                k: meta[k]
                for k in ("dataset_version", "n_movies", "n_interactions", "n_users", "metadata_coverage")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
