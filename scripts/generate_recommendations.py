"""Stage 5: batch-generate explained recommendations with the active model.

--user-id N          a MovieLens user (profile = their ratings in data/processed)
--rate ID=RATING ... an ad-hoc new user (fold-in path, e.g. --rate 109487=5 79132=4.5)
--genres Sci-Fi ...  onboarding genres for a new user
--all --out FILE     every MovieLens user → CSV (top-K each)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys

from jev_ml.data.dataset import load_interactions
from jev_ml.engine import Interaction, RecommendationEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--user-id", type=int)
    parser.add_argument("--rate", nargs="*", default=[])
    parser.add_argument("--genres", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--out", default="experiments/batch_recommendations.csv")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    engine = RecommendationEngine.load_active()

    def events_for(uid: int) -> list[Interaction]:
        rows = ratings[ratings["user_id"] == uid]
        return [
            Interaction(int(r.movie_id), "rating", float(r.rating), float(r.timestamp))
            for r in rows.itertuples()
        ]

    if args.all:
        ratings = load_interactions()
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["user_id", "rank", "movie_id", "title", "score", "reason"])
            for uid in sorted(ratings["user_id"].unique()):
                for r in engine.recommend(engine.build_profile(events_for(int(uid))), k=args.k):
                    w.writerow([uid, r.rank, r.movie_id, r.title, r.score, r.reason])
        print(f"wrote {args.out}")
        return

    events: list[Interaction] = []
    if args.user_id is not None:
        ratings = load_interactions()
        events = events_for(args.user_id)
        if not events:
            sys.exit(f"user {args.user_id} has no ratings")
    for spec in args.rate:
        mid, val = spec.split("=")
        events.append(Interaction(int(mid), "rating", float(val)))
    profile = engine.build_profile(events, genre_prefs=args.genres)
    recs = engine.recommend(profile, k=args.k)
    print(
        json.dumps(
            {
                "model_version": engine.version,
                "effective_weights": {k: round(v, 3) for k, v in engine.effective_weights(profile).items()},
                "items": [
                    {
                        "rank": r.rank,
                        "movie_id": r.movie_id,
                        "title": r.title,
                        "score": r.score,
                        "reason": r.reason,
                    }
                    for r in recs
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
