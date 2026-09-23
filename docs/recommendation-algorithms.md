# Recommendation algorithms

All models share one interface (`jev_ml.models.base.Recommender`): `fit(TrainContext)`, `score(UserProfile) → scores for
every item`, `explain(profile, item) → contributions`, and `save/load` using portable `.npz` + `.json` artifacts.

## Interaction signals (`jev_ml/signals.py`)

| Event | Implicit weight (CF / MF confidence) | Preference weight (content / genre taste) |
|---|---|---|
| rating r ≥ 4 | 1 + (r − 4) → 1.0 … 2.0 | (r − 3) / 2 → +0.5 … +1 |
| 2.5 < r < 4 | 0.5 | −0.25 … +0.25 |
| r ≤ 2.5 | 0.1 (watched, but disliked) | −0.25 … −1 |
| favourite / onboarding pick | 2.0 / 1.5 | +1 |
| watched (not rated) | 0.6 | +0.3 |
| "not interested" / "dislike" feedback | excluded from candidates | — |

Why this design: a rating is evidence that the user **watched** a film, whatever its value, and that predicts what they
will watch next. So collaborative models see every consumed item. Only content and genre profiles use the *signed*
taste, where a disliked film pushes the vector away.

## 1. Popularity baseline (`models/popularity.py`)
`score = log1p(#ratings ≥ 4) + 0.25·log1p(#ratings)`. The model also computes:
- **trending**: exponentially time-decayed counts (half-life 365 days, relative to the newest rating), optionally blended
  with the last 7 days of in-app activity;
- **Bayesian average rating** `(Σr + m·μ)/(n + m)`, with m = 10, used for the *Popular & acclaimed* shelf and the quality-floor
  business rule.

## 2. Content-based (`models/content.py`, `features.py`)
Features come from MovieLens (title, genres, user tags, year) and Wikidata (directors, cast, "main subject" keywords,
fine-grained genres, description). They are normalised (NFKD, lower-case, stop-words, genre aliasing so that
`science fiction film` and MovieLens `Sci-Fi` become the same token) and prefixed per field (`d:`, `c:`, `k:`, …).
- Each field has **its own TF-IDF** (sublinear TF, min_df = 2). The field matrices are scaled by √weight and stacked,
  then rows are L2-normalised. Field weights: genres 1.0, directors 1.0, keywords 0.8, tags 0.8, cast 0.6,
  description 0.3, decade 0.3, title 0.2.
- User vector = Σ preference_weight · item vector (signed), normalised. Score = cosine.
- The top-50 content neighbours per item are precomputed in blocks, which keeps memory O(n·k) and avoids an n² matrix.
- Explanations: the liked item with the highest cosine to the candidate ("Because you liked …"), and the shared features
  ranked by `x_candidate[f] · u[f]` ("your interest in Christopher Nolan").
- **Item cold start**: the vocabulary and IDF are persisted, so a movie added after training is vectorised at
  request time and its neighbours come from metadata alone.

## 3. Collaborative filtering: item-kNN (`models/itemknn.py`)
Cosine similarity between item columns of the users × items confidence matrix, with shrinkage,
`sim = ⟨xᵢ,xⱼ⟩ / (‖xᵢ‖‖xⱼ‖ + λ)`, keeping the top-k neighbours per item. Similarities are computed in 1,024-row blocks
over sparse matrices. Scores for any user come from one sparse vector-matrix product. Explanation: the history item
with the largest `wᵤᵢ · sim(i, j)` ("Viewers who enjoyed *X* also loved this").

## 4. Matrix factorization: implicit ALS (`models/als.py`)
Hu, Koren & Volinsky (2008). Confidence is `c = 1 + α·w` and the model minimises
`Σ c_ui (p_ui − xᵤ·yᵢ)² + λ(‖X‖² + ‖Y‖²)`. Each half-step is an exact solve using the YᵀY trick. The training objective
is logged every iteration (dashboard: "ALS training objective").
- **Known users** (MovieLens ids) use their learned factor. **New users** are folded in:
  `xᵤ = (YᵀY + Yᵤᵀ(Cᵤ − I)Yᵤ + λI)⁻¹ Yᵤᵀcᵤ`.
- **Explanations** use the paper's exact score decomposition `s_ui = Σⱼ (yᵢᵀ Wᵤ yⱼ) c_uj`, which shows which past items
  produced the score ("Because you liked *X*").

## 5. Hybrid ranking engine (`models/hybrid.py`)
Six signals: `content_similarity`, `collaborative_score` (item-kNN), `latent_factor_score` (ALS), `popularity_score`,
`user_preference_score` (genre vector from onboarding picks + genres of rated films, normalised by √#genres of the item),
`recency_score` (exp(−age/15 years)).

Pipeline, deterministic for fixed model state and input, with ties always broken by item index:
1. **candidate filtering**: mask consumed items, "not interested" items, and request filters (genres, year range,
   min/max rating count);
2. **candidate generation**: union of the top-200 from each active signal;
3. **feature computation**: all six raw signals for each candidate. Content is multiplied by a support prior
   `((log1p n + 1)/(log1p n_max + 1))^d` so that pure metadata matches on obscure titles don't dominate; the +1 keeps
   unrated movies discoverable;
4. **normalisation**: per-signal min-max (or rank) within the candidate set;
5. **hybrid scoring**: weighted sum with **interaction-adaptive weights**. The CF and ALS weights are multiplied by
   `β = min(1, n/ramp)`, content by `min(1, n_liked/3)`, and popularity and preference by `1 + boost·(1 − β)`.
   Weights are renormalised to sum to 1, so a brand-new user is served popularity plus their genre picks, and those
   signals give way to behavioural ones as interactions accumulate;
6. **business rules**: quality floor, dropping items whose Bayesian average is < 2.0 with at least 5 ratings;
7. **diversity**: Maximal Marginal Relevance over content cosine, `λ·rel − (1−λ)·max_sim_to_selected`, and an optional
   per-primary-genre cap. Users can choose *Focused* (λ = 1), *Balanced* (tuned λ) or *Adventurous* (λ = 0.7);
8. **final ranking and top-K** with offset pagination (a page is exactly the slice of a longer ranking).

Every ranked item carries `{raw, normalized, weight, contribution}` for all six signals. The API stores these with the
reason.

### Weights and tuning
Weights, ramp, boost, damping, normalisation and λ live in `configs/experiment.yaml`. `scripts/train_models.py` tunes
them on the validation split: grids for item-kNN (k, shrinkage) and ALS (factors, λ, α), then 24 seeded Dirichlet
samples of hybrid weights plus the configured defaults, then a cold-start sweep (ramp × boost × damping, scored on the
mean of warm and truncated-profile NDCG), then an MMR λ sweep. The λ chosen is the most diverse setting within 2% of
the best NDCG.

## Explanations (`explain.py`)
Reasons are templates **filled only from model contributions**:
- they are ordered by contribution to the hybrid score, times an informativeness prior (personal signals 1.0,
  popularity and recency 0.5);
- a signal without attributable evidence (no anchor item with positive attribution, no shared feature) produces **no
  sentence**, and the next signal is used instead;
- `anchor_movie_ids` lists every film cited. Tests assert that these are always films from the user's own history.

Examples actually produced: *Recommended because of your interest in Christopher Nolan* · *Because you liked Inception* ·
*Viewers who enjoyed The Dark Knight also loved this* · *Matches your preference for Sci-Fi & Thriller* ·
*Rated 4+ stars by 122 viewers*.

## Cold start
- **New user**: onboarding asks for genres (preference signal), favourite films (strong positives) and optional quick
  ratings. With zero events the ranking is popularity + genre preference. Behavioural signals ramp in as described above.
- **New movie**: metadata-only neighbours via the persisted featurizer (`/recommendations/similar/{id}` falls back
  automatically). Unrated catalogue items keep a non-zero content support prior.
