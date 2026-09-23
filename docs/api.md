# API reference

Base URL: `http://localhost:8000` (direct) or `http://localhost:3000/api` (through the web proxy). Interactive OpenAPI
docs are at `/docs` when `JEV_ENV != production`.

**Auth:** `POST /auth/login` returns `{access_token, expires_in, user}` and also sets an httpOnly `jev_session` cookie
(SameSite=Lax). Send either `Authorization: Bearer <token>` or the cookie. **Cookie-authenticated writes must include
`X-JEV-CSRF: 1`.** Errors are always `{"detail": str, "request_id": str}`, and 422 responses add `errors: [{loc, msg, type}]`.
Rate limits: 240 requests/min per client and 20/min on `/auth/login` and `/auth/register` (429 with `Retry-After`).

| Method & path | Auth | Description |
|---|---|---|
| `POST /auth/register` | – | `{email, password(≥8), display_name}` → 201 token response; 409 if the email exists |
| `POST /auth/login` | – | `{email, password}` → token response; 401 on bad credentials |
| `POST /auth/logout` | – | clears the cookie |
| `GET /users/me` | user | profile incl. `favorite_genres`, `onboarding_completed` |
| `POST /users/me/onboarding` | user | `{genres: [str], movie_ids: [int]}`: stores genre prefs + onboarding favourites |
| `PATCH /users/me/preferences` | user | `{genres?, diversity?: focused\|balanced\|adventurous}` |
| `GET /users/me/profile` | user | taste profile from real interactions: genre affinity, directors, actors, rating histogram, preference vector, effective hybrid weights, stage |
| `GET /users/me/ratings` · `/history` · `/favorites` | user | paginated (`page`, `page_size`) |
| `GET /genres` | – | genres with movie counts |
| `GET /movies` | – | `page, page_size, genre, year_min, year_max, min_ratings, sort=popular\|rating\|year\|title` |
| `GET /movies/search?q=` | – | title / director / cast search (parameterised LIKE, wildcards escaped) |
| `GET /movies/popular` | – | Bayesian-average top rated (model popularity component) |
| `GET /movies/{id}` | optional | detail, plus `user_rating`, `is_favorite`, `watched` when authenticated, and `in_model` |
| `POST /movies/{id}/rate` | user | `{rating: 0.5…5 in steps of 0.5}`; `DELETE` removes it |
| `POST /movies/{id}/favorite` | user | `{favorite: bool}` |
| `POST /movies/{id}/watch` | user | appends to watch history |
| `POST /movies` | admin | add a catalogue movie unseen by the model (item cold start) |
| `GET /recommendations` | user | `limit (≤100), offset, genres[], year_min, year_max, min_ratings, max_ratings, context` → see below |
| `GET /recommendations/similar/{movie_id}` | optional | item-to-item (content + co-watch; metadata only for new movies) |
| `GET /recommendations/trending` | – | time-decayed popularity + last-7-days JEV activity |
| `GET /recommendations/because-you-watched` | user | neighbours of the latest watched / 4★+ film (`anchor`, `anchor_kind`) |
| `GET /recommendations/similar-to-favorites` | user | neighbours of recent favourites, each attributed to one |
| `POST /recommendations/feedback` | user | `{movie_id, feedback: like\|dislike\|not_interested\|clicked, recommendation_id?}`; dislike and not_interested exclude the movie from future lists |
| `GET /recommendations/history` | user | served recommendations with reason, score and feedback |
| `GET /models` · `GET /models/{id}` | admin | model registry (metrics, config, manifest, ALS loss curve) |
| `POST /models/{id}/activate` | admin | hot-swap the serving model (loaded before swapping; recommendation cache cleared) |
| `GET /models/active/summary` | – | public headline metrics of the serving model |
| `GET /experiments` · `GET /experiments/{id}` | admin | runs and every metric (model × protocol × metric × K) |
| `GET /admin/stats` | admin | users, ratings, recommendations served per day, feedback breakdown, reason codes |
| `GET /health` | – | DB and cache status (503 when degraded) |
| `GET /health/ml` | – | model status plus a live self-check recommendation |

### `GET /recommendations` response
```json
{
  "items": [{
    "recommendation_id": 123, "movie_id": 4226, "title": "Memento", "year": 2000,
    "genres": ["Mystery", "Thriller"], "directors": ["Christopher Nolan"],
    "score": 0.7743, "rank": 1,
    "reason": "Recommended because of your interest in Christopher Nolan", "reason_code": "director",
    "secondary_reasons": ["Rated 4+ stars by 122 viewers"], "anchor_movie_ids": [],
    "signals": {"content": {"raw": 0.41, "normalized": 0.93, "weight": 0.2, "contribution": 0.186}, "...": {}}
  }],
  "model_version": "jev-20260923T100141Z-bbb2e4c9",
  "generated_at": "2026-09-23T10:05:00Z", "request_id": "…", "limit": 20, "offset": 0,
  "effective_weights": {"content": 0.2, "collaborative": 0.1, "...": 0},
  "profile": {"interactions": 5, "genres": 2, "excluded": 0},
  "cached": false
}
```
Each call (on a cache miss) is persisted to the `recommendations` table, including the signals and reason.

## Intelligence layer (`/intel/*`)

All `/intel/*` endpoints and `GET /admin/metrics` are **admin only** (401 anonymous, 403 for members); cookie-authenticated
writes need `X-JEV-CSRF: 1` like every other write. The design and the full JSON contract are in
[intelligence.md](intelligence.md) (sections 4 and 6). "Latest run" means the most recently started *successful* run,
including replays; pass `?run_id=<uuid>` to read an older one.

| Method & path | Description |
|---|---|
| `GET /intel/status` | overview: `latest_run` (any status), `summary`/`data` of the latest successful run, active `model`, `warnings_open` by severity, 5 `recent_decisions`, 5 `top_signals`, 5 `top_risks`, `confidence_histogram` (5 bins over the latest run's decisions), `health {database, cache, model, pipeline}` |
| `GET /intel/runs` | `limit, offset` → `{items: [Run], total}`, newest first |
| `POST /intel/runs` | `{as_of?: "YYYY-MM-DD" \| ISO}`. Runs synchronously (about 1 s on MovieLens-small) and returns the finished Run. A pipeline failure comes back as 200 with `status: "failed"` and `error`. 409 while another run is in progress; 422 for an unparsable date, a future date or one outside the MovieLens range |
| `GET /intel/runs/{id}` | one Run by numeric `id` or `run_id` (never includes the full result) |
| `GET /intel/signals` | `run_id?, kind, entity_type, direction, limit (1..200, default 50), offset` → `{items, total, limit, offset, run_id, as_of}` |
| `GET /intel/trends` | same envelope; filters `direction (up\|down\|flat), metric, entity` |
| `GET /intel/anomalies` | filters `kind, severity, entity_type, suppressed` |
| `GET /intel/risks` · `GET /intel/actions` | filters `kind, level` · `priority (P1\|P2\|P3)` |
| `GET /intel/predictions` | `run_id?, series_id?` → `{run_id, as_of, forecasts, lapse}` |
| `GET /intel/series/{series_id}` | series ids contain colons (`volume:genre:Drama`, URL-encoded or not) → `{run_id, as_of, series, trend, anomalies, forecast}`; 404 for an unknown series |
| `GET /intel/warnings` | `status, severity, key, limit, offset` → `{items, total, limit, offset}`, most recently seen first |
| `GET /intel/warnings/{id}` | one warning plus `history` (every status change: from, to, note, actor, at) |
| `PATCH /intel/warnings/{id}` | `{status, note?(≤1000)}`. Allowed: new→acknowledged/investigating/resolved/dismissed, acknowledged→investigating/resolved/dismissed, investigating→resolved/dismissed; resolved and dismissed are terminal. 409 on any other transition |
| `GET /intel/decisions` | decision log across runs: `key, run_id, entity, limit, offset` → DB-backed Decision (`db_id, run_id, as_of, created_at, feedback {correct, incorrect}`) |
| `GET /intel/decisions/{id}` | by `db_id`, or by contract id `dec-…` (its most recent copy) |
| `POST /intel/feedback` | `{target_type: decision\|warning\|action\|prediction, target_id: str, verdict, note?, outcome?}` → 201. `target_id` is the contract id for decisions (`dec-…`) and actions (`act-…`), the forecast id for predictions (`fc-…`) and the numeric id (as a string) for warnings. Actions and predictions must exist in the latest run. 404 for an unknown target, 422 when the verdict does not fit the target (decision/prediction: correct\|incorrect; warning: useful\|not_useful\|false_positive; action: useful\|not_useful) |
| `GET /intel/feedback` | `target_type?, limit, offset` → `{items, total, limit, offset, summary}`; `accuracy`/`precision` are null until there is a verdict |
| `POST /intel/scenarios` | scenario input (`series_id, horizon_months, as_of?, scenarios[]`) plus `save, title?` → scenario output plus `id` (null unless saved). Invalid specs → 422 with the validator's message; 404 before any run (unless `as_of` is given) |
| `GET /intel/scenarios` | saved analyses: `series_id?, limit, offset` → `{items: [{id, title, series_id, created_at, input, output}], total}` |
| `GET /intel/evaluation` | newest `experiments/intel-eval-*/report.json` → `{available, run_dir, report}` |
| `GET /admin/metrics` | in-process counters since the process started, grouped: `http_requests_by_status`, `http_requests_by_route` (route templates), `http_errors`, `http_latency_ms`, `http` (rate-limited, unhandled), `intel_pipeline` (runs per trigger, succeeded/failed, last duration/status, in progress), `intel_pipeline_ms`, `intel_stage_ms` (per-stage timings, including `forecast` and `lapse`), `intel_warnings` (lifecycle counters), `intel_warnings_open`, `intel_decisions_latest_run` (by spec and answer), `data_freshness` (source ages of the latest run), `model` (loaded, version, load errors) |
