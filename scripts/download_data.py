"""Stage 1: download MovieLens (and optionally Wikidata metadata enrichment)."""

from __future__ import annotations

import argparse
import logging

from jev_ml.data.download import download_movielens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="do not query Wikidata for director/cast/keywords (offline mode)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    path = download_movielens(force=args.force)
    print(f"MovieLens ready at {path}")
    if not args.skip_enrichment:
        from jev_ml.data.enrich import enrich_from_wikidata

        out = enrich_from_wikidata(path)
        print(f"Wikidata enrichment written to {out}")


if __name__ == "__main__":
    main()
