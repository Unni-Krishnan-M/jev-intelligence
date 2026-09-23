# ML pipeline

| Stage | Entry point | Output |
|---|---|---|
| download | `scripts/download_data.py` | `data/raw/ml-latest-small/` (MD5 verified against GroupLens), `data/raw/wikidata_enrichment.json` |
| validate / clean / transform | `scripts/preprocess_data.py` | `data/processed/{movies,interactions}.csv`, `dataset_meta.json` |
| feature engineering | `jev_ml.features` (inside training) | TF-IDF item vectors, genre matrix |
| split | `jev_ml.evaluation.split` | train / val / test (per-user temporal) |
| train + tune | `scripts/train_models.py` | `models/<version>/` + `models/registry.json` |
| evaluate | same run, or `scripts/evaluate_models.py` for a registered version | `experiments/<run>/` |
| generate | `scripts/generate_recommendations.py` | explained recommendations (JSON / CSV) |

```bash
uv run python scripts/download_data.py        # ~1 MB MovieLens + Wikidata (≈20 min first time, then cached; --skip-enrichment for offline)
uv run python scripts/preprocess_data.py
uv run python scripts/train_models.py         # tuned (~4 min on a laptop CPU)  |  --quick (~20 s)
uv run python scripts/evaluate_models.py      # re-evaluate the active version's parameters
uv run python scripts/generate_recommendations.py --rate 79132=5 109487=5 --genres Sci-Fi -k 10
uv run python scripts/generate_recommendations.py --user-id 42
```

## Dataset
**MovieLens ml-latest-small** (F. M. Harper and J. A. Konstan, 2015, *The MovieLens Datasets: History and Context*, ACM TiiS),
<https://grouplens.org/datasets/movielens/>. Research and non-commercial licence: it is downloaded by the script and **not
committed**. After loading: 9,742 movies, 100,836 ratings, 610 users, 3,683 tags. Ratings run 0.5–5 in half steps,
from 1996-03-29 to 2018-09-24. Density is 1.70%. Validation checks (unique ids, no nulls, rating grid, referential
integrity, duplicates) all pass. Their results are stored in `dataset_meta.json`.

**Wikidata** (CC0) enrichment is keyed by the IMDb ids in `links.csv`, with one SPARQL request per property per batch
of 250. The script is resumable, and batches are cached in `data/raw/wikidata_cache/`. 9,651/9,742 IMDb ids matched.
Coverage in the processed catalogue: directors 98.5%, cast 90.4%, Wikidata genres 98.0%, description 99.0%,
keywords 36.2%, MovieLens tags 16.1%.
Missing values stay missing, with one exception that is labelled: when Wikidata has no director statement but the
English description reads "…directed by X / film by X", X is used and marked `director_source = wikidata_description`
(94 films, e.g. several Christopher Nolan titles).

Processed `movies.csv` fields: `movie_id, title, raw_title, year, genres, tags, directors, director_source, cast, keywords,
wd_genres, countries, description, runtime_min, imdb_id, tmdb_id, n_ratings, mean_rating` (list fields `|`-joined).

**Dataset version** = `ml-latest-small-<md5[:8]>-wd-<sha256(enrichment)[:12]>`. It is recorded in every model manifest
and experiment.

## Artifacts (`models/<version>/`)
`manifest.json` (version, timestamp, dataset version, seed, full training config, test metrics, component params,
ALS loss curve, artifact list and size, git commit, Python version), `hybrid.json`, `movies.csv` (the catalogue snapshot
the indices refer to), `popularity/popularity.{npz,json}`, `content/{item_vectors,neighbors}.npz` + `featurizer.json`,
`itemknn/item_similarity.npz`, `als/als_factors.npz`. A version is about 15 MB. No pickles are used.

## Reproducibility
One seed (`configs/experiment.yaml: seed`) drives ALS initialisation, hybrid random search and evaluation sampling.
Every stochastic step uses an explicit `numpy.random.Generator`. `tests/ml/test_engine_pipeline.py::test_pipeline_is_reproducible`
asserts that two runs give identical metrics. Versions are content-addressed (timestamp + hash of the config).
