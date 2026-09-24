# Online experimentation (A/B testing)

Phase 2, WS5. This document covers online A/B tests on the `/recommendations` surface: the exposure log,
persisted member strategy decisions, the statistics, and an offline demonstration on MovieLens that is
labelled as such throughout.

The whole system runs inside the existing API and database. There is no experimentation service or
feature-flag SaaS. Assignment uses deterministic hashing in Python, and state lives in PostgreSQL or
SQLite.

| Piece | Where |
|---|---|
| Tables | `backend/jev_api/models/experiments.py`, migration `0009_experiments.py` |
| Assignment, lifecycle, exposures, attribution, results | `backend/jev_api/services/experiments.py` |
| Statistics (pure functions) | `backend/jev_api/services/experiment_stats.py` |
| Serving integration | `backend/jev_api/services/recommend.py` (`personalized`) |
| Admin API | `backend/jev_api/routers/experiments.py` (`/experiments/online/...`) |
| Offline replay demo | `scripts/simulate_ab_replay.py` → `experiments/ab-replay-<ts>/` |
| Tests | `tests/integration/test_experiments.py`, `tests/integration/test_experiments_stats.py` |

The existing offline `Experiment` table and `GET /experiments[/{id}]` keep their meaning: offline
evaluation runs. The new tables are all prefixed `ab_`.

## 1. Schema (migration 0009)

| Table | Purpose | Key constraints |
|---|---|---|
| `ab_experiments` | key, name, surface, status, hypothesis, primary metric, guardrails (JSON), `traffic_percent`, `salt`, `attribution_window_hours`, `analysis` (alpha, power, MDE, min users, `data_source`), frozen `result`, start/stop/conclude times, `created_by` | unique `key`; CHECKs on surface, status and primary metric; **partial unique index `uq_ab_experiments_active_surface` on `surface` WHERE status IN ('running','paused')** |
| `ab_variants` | name, position, `is_control`, weight, description, `config` (JSON serving payload) | unique (experiment, name) |
| `ab_assignments` | sticky (experiment, user) → variant, plus the enrolment bucket | **unique (experiment_id, user_id)** |
| `ab_exposures` | one row per served list, **inside or outside an experiment**: fresh `request_id`, `source_request_id`, user, surface, context, experiment and variant (NULL outside), `cached`, model version, decision id, items `[{movie_id, rank, recommendation_id}]`, latency, `served_at` | unique `request_id` |
| `ab_outcomes` | an interaction attributed to an exposure: kind, value, served rank (NULL when the item was not on the list), `source` (`live` or `replay`), `occurred_at` | unique (exposure, kind, movie); CHECKs on kind and source |
| `member_decisions` | every served `recommendation_strategy` decision: decision id, user, spec, policy version, answer, served strategy, confidence (+ kind), abstained, `state_hash`, state, rationale, evidence, model version, source, `created_at` | unique (decision_id, state_hash); CHECK on confidence_kind |
| `recommendations` | adds `experiment_id` and `variant` (lineage columns, no FK) | index (experiment_id, variant) |

The new CHECK vocabularies are appended to `models/enums.py` (`AB_*`) and registered in `CHECK_ENUMS`.
The migration keeps literal copies of them. The upgrade → downgrade → upgrade round trip, CHECK parity and
`alembic check` pass on SQLite and on PostgreSQL 17. The downgrade drops experiment data and persisted
decisions, and keeps served recommendations.

### Member decisions (gap P1 #7)

`strategy_choice` persists each freshly taken decision (on a strategy-cache miss), and so does
`GET /me/intelligence`. Rows are deduplicated by (decision_id, state_hash). The decision id is stable for
the same member and history, so new feedback under the same id is a new row. To resolve an id, take the
newest row. Members read their own decisions with `GET /me/intelligence/decisions/{decision_id}`. The
id also resolves for `POST /me/intelligence/feedback` after the cache has expired. Persisting never fails
a request: on an error it logs and counts `member_decisions.errors`.

Note for the integrator: the audit (§8) planned member decisions for WS4 in a later `0011`. The table
now exists in 0009, so WS4 should build on `member_decisions` and not add a second table.

## 2. Assignment

```
bucket   = int(sha256(f"{salt}:{user_id}")[:8 bytes]) mod 10000          # enrolment
enrolled = bucket < round(traffic_percent * 100)
u        = int(sha256(f"{salt}:variant:{user_id}")[:8 bytes]) / 2**64   # in [0, 1)
variant  = the first variant (in position order) whose cumulative weight share exceeds u
```

- **Deterministic.** Assignment depends only on the salt and the user id, so it is identical in every
  worker and after every restart. Each experiment gets a fresh 128-bit random salt at creation, so
  experiments are independent of each other.
- **Two independent hashes.** Enrolment and the variant split are uncorrelated. The first 10 % of
  traffic has the same variant mix as the whole population (tested).
- **Sticky and stored.** At a member's first request while enrolled, an `ab_assignments` row is written
  and then reused for the life of the experiment. The unique (experiment, user) constraint makes "one
  variant per member per experiment" a database guarantee. A race between two concurrent first requests
  resolves to the committed row.
- **Ramp semantics.** Only `traffic_percent` can change after start (`POST .../ramp`, audited as
  `experiment.ramp` with from/to):
  - *Ramp up* adds members. Everyone enrolled at 10 % is still enrolled at 50 %, since their bucket is
    below both thresholds, and nobody changes variant.
  - *Ramp down* only stops new enrolment. Members who already have an assignment stay in their variant.
    Silently moving them back to control would contaminate both arms.
  - Variant weights and configs are immutable once started. Changing them means creating a new
    experiment.

## 3. Contamination controls

| Control | How |
|---|---|
| One active experiment per surface | Enforced in code (`transition` checks for another running or paused experiment and returns **409**) and in the database (partial unique index, portable to SQLite and PostgreSQL; a concurrent start that slips past the code check fails on commit and also returns 409). "Paused" still holds the surface. |
| One variant per member per experiment | unique (experiment_id, user_id) on `ab_assignments` |
| Admins and test accounts excluded | `JEV_EXPERIMENTS_EXCLUDE_ADMINS=true` (default) and `JEV_EXPERIMENTS_EXCLUDED_EMAIL_PATTERNS` (comma-separated fnmatch patterns, default `*@test.invalid,*+jevtest@*`). Excluded members are served the default and never assigned. Their exposures are logged untagged. |
| Sample-ratio mismatch | χ² goodness of fit of the arm sizes against the weights, on assigned members **and** on exposed members, at α = 0.001 (`JEV_EXPERIMENTS_SRM_ALPHA`). A detected SRM forces "inconclusive". |
| Cache isolation | The recommendation cache key carries `x<experiment id>.<variant>` and the serving model version. |
| Kill switch | `JEV_EXPERIMENTS_ENABLED=false` serves everyone the default. Exposures are still logged. |

## 4. Serving: what a variant can change

`VariantConfig` (validated at create time):

| Field | Effect in `personalized()` |
|---|---|
| `hybrid_overrides` | Any `HybridConfig` field. Examples: `diversity_lambda` (MMR λ, in [0, 1]), `weights`, `recency_tau_years`, `cold_start_boost`, `behavioral_ramp`, and WS3's `cold_stages` (profile-size cold-start staging, already in `HybridConfig`). Unknown fields are rejected with 422. The override replaces the base config for that arm. A member's explicit diversity preference and the strategy decision still apply on top of it, exactly as for the champion. |
| `model_version` | A **challenger** model, which must be registered. It is loaded lazily through the normal loader (`RecommendationEngine(models_dir/version)`) on first use and kept in an LRU of `JEV_EXPERIMENTS_MAX_CHALLENGER_MODELS` (default 2) engines per process. **Memory:** each challenger is one full engine resident next to the champion. The MovieLens model is about 12 MB on disk and a few tens of MB in memory. Load time is about 0.1 s for the MovieLens model (measured at startup), paid by the first request that hits the arm. |
| `strategy_decision` | `false` turns the member strategy decision off. The list is served as standard and the response has no `intelligence` block. |
| `recency_half_life_days` | Profile recency decay (the `adapt_to_recent` mechanism), applied unless the strategy already decayed the profile |

If a variant fails (for example the challenger cannot load), the request is served the default, untagged,
and `experiments.errors` is counted. The exposed-member SRM check then flags the experiment. A request
never fails because of an experiment.

### Exposure log and request ids (gap P1 #6): the choice made

**Every response carries a fresh `request_id`, and every served list writes one `ab_exposures` row,
cache hits included.** On a cache hit, the response and the exposure are **exposure-linked**:

- the `recommendation_id`s are the rows written when the list was generated (no duplicate rows per hit);
- `source_request_id`, in both the response (a new optional field on `RecommendationResponse`) and the
  exposure row, names the request that generated them.

This was chosen over re-writing recommendation rows on every hit because it is one row per request
instead of k rows. The `recommendations` table stays "what was generated", and `ab_exposures` is "what
was shown, when".

Outside an experiment the list is exactly what v1.2 served, including items, scores, ranks, reasons and
the strategy decision (tested). The only visible difference is the fresh request id on cache hits.

## 5. Metrics and statistics

The **unit of analysis is the member**, which is also the unit of randomisation. A member's exposures
are correlated, and treating each exposure as an independent trial would overstate significance.
Per-exposure rates are reported alongside the member-level ones, as descriptive figures only.

### Outcome attribution

`attribute_outcomes` runs on every results read. It is idempotent: a unique key on
(exposure, kind, movie) prevents duplicates.

- **Sources:** `recommendation_feedback` (like, dislike, not_interested, clicked→click), `ratings`
  (`updated_at`), `watch_history` and user `favorites`.
- **Window:** an event at time *t* by an enrolled member is attributed when
  `served_at ≤ t ≤ served_at + attribution_window_hours` (default 24 h, set per experiment).
- **Which exposure:** the member's latest in-window exposure that *listed* the item, with its served
  rank. Otherwise, the latest in-window exposure with rank NULL. An unlisted item is a relevant item the
  list missed: it counts toward the ideal DCG but not toward the list's rates.

### Per variant

| Metric | Definition (member level) |
|---|---|
| `interaction_rate` (CTR) | share of members with ≥ 1 click, like, rating, watch or favourite on a listed item |
| `positive_rate` | ≥ 1 like, favourite, watch or rating ≥ 4 on a listed item |
| `rating_rate` | ≥ 1 rating of a listed item |
| `feedback_rate` | ≥ 1 like, dislike or not_interested on a listed item |
| `negative_rate` (guardrail) | ≥ 1 dislike, not_interested or rating ≤ 2 on a listed item |
| `ndcg_at_10` | online binary NDCG@10: the served ranks of positive items, over an IDCG of min(10, all distinct positives in the window, listed or not); averaged per member |
| `diversity` | intra-list diversity = mean pairwise (1 − genre Jaccard) of the top 10 |
| `novelty` | mean self-information −log2((n_ratings+1)/(Σ+N)) of the listed items (bits) |
| `coverage` | distinct exposed items / catalogue size (per arm, descriptive) |
| latency | p50 and p95 of the server-side serving time, including cache hits; also p95 of generated lists only |

### Tests

| Question | Method |
|---|---|
| Rates | two-proportion z-test (pooled SE for p) with a Wald CI (unpooled SE) on the difference |
| NDCG, diversity, novelty | percentile bootstrap (default 2000 resamples, seeded from the salt so reports reproduce) on the difference of member means, with a two-sided bootstrap p-value |
| SRM | χ² goodness of fit (above) |
| Power | required members per arm for the configured relative MDE, α and power. Rates use the closed form; means use 2(z_α/2 + z_β)²σ²/δ². A warning is emitted when the smallest arm is below it, or below `min_users_per_variant`. |
| Multiple arms | Bonferroni: the primary metric is tested at α / (number of treatments) |

The statistics are checked on known inputs:

- the z-test against the textbook numbers;
- the bootstrap CI width against the normal-theory standard error;
- SRM chi-square values;
- the sample-size formula;
- NDCG and diversity edge cases.

The P2.6 A/A acceptance ("≤ 5 % false positives") is tested with 2000 simulated z-tests and 400
simulated bootstraps inside a 3σ band. The last run gave a 5.9 % rate for the z-test and 6.0 % for the
bootstrap. A literal "≤ 10 of 200 runs" check fails about 42 % of the time even for a perfectly
calibrated test, so it was not used.

### Guardrails and conclusion

Guardrails are set per experiment. The default is `negative_rate` with `max_increase` 0.02, plus
`latency_p95_ms` with `max_ratio` 1.5 once p95 latency is also more than 5 ms above control. A guardrail
is **breached** in either of two cases:

- the observed degradation exceeds its threshold;
- a rate or mean guardrail degrades significantly in the harmful direction.

The conclusion (`POST .../conclude` freezes it into `result`) is one of:

- **ship**: the treatment's primary metric is significantly better (at the adjusted α, CI excluding 0),
  and none of its guardrails is breached. If several treatments qualify, the one with the largest
  effect wins.
- **keep_control**: every treatment is significantly worse on the primary metric.
- **inconclusive**: everything else, including an SRM, an arm below `min_users_per_variant`, a primary
  effect that is not significant, or a significant effect with a breached guardrail. No winner is
  reported.

## 6. Lifecycle and API

```
draft ──start──▶ running ◀──start── paused
  │                 │  └──pause──▶ ─┘  │
delete            stop               stop
                    ▼                  ▼
                 stopped ──conclude──▶ concluded
```

- Any other transition returns **409**.
- A draft can be edited (PATCH) or deleted. After start, only the ramp changes anything.
- A paused experiment serves everyone the default, keeps its assignments and still holds the surface.
- Every action is audited: `experiment.create`, `start`, `pause`, `ramp`, `stop` and `conclude`. The
  conclude entry records the decision and winner.

All endpoints are admin only.

| Method and path | Action |
|---|---|
| `POST /experiments/online` | create a draft (201) |
| `GET /experiments/online/list?status=` | list |
| `GET /experiments/online/{key}` | detail, with `allowed_actions` and assigned users per variant |
| `PATCH /experiments/online/{key}` | edit a draft (409 after start) |
| `DELETE /experiments/online/{key}` | delete a draft (204) |
| `POST /experiments/online/{key}/start` \| `pause` \| `stop` \| `conclude` | lifecycle actions |
| `POST /experiments/online/{key}/ramp` | `{"traffic_percent": 0..100}` |
| `GET /experiments/online/{key}/results` | metrics, comparisons, SRM, sample size, guardrails, warnings, conclusion and `label` |

The listing lives at `/list` rather than at `GET /experiments/online` because the offline router's
`GET /experiments/{experiment_id}`, mounted first, would capture that path. `list` is therefore a
reserved key.

On the member side the system is transparent. Members are assigned and served inside
`GET /recommendations`, and no endpoint tells a member which variant they are in. The only new member
endpoint is `GET /me/intelligence/decisions/{id}`.

Metrics exposed at `/admin/metrics`:

- `experiments.exposures` (`generated`, `cache_hit`);
- `experiments.exposures_by_variant`;
- `experiments.assignments`;
- `experiments.outcomes`;
- `experiments.transitions`;
- `experiments.challenger_loads`;
- `experiments.errors`;
- `member_decisions`.

## 7. Demonstration: offline replay on MovieLens

> **Offline replay using held-out ratings, not live traffic.** MovieLens has no online traffic. Nobody saw
> these lists, so the replay compares rankings through the online pipeline. It cannot measure the causal
> effect of showing a list.

`uv run python scripts/simulate_ab_replay.py` does the following:

1. It splits the 100,836 ratings by `user_temporal` (per member: 70 % train, 10 % validation, the newest
   20 % test). The history is train + validation (80,672 rows). The future is the test split (20,164
   ratings, 610 members).
2. It retrains the active model's configuration (`jev-20260923T100141Z-bbb2e4c9`: component params and
   tuned hybrid config) **on the history only**, in 5.8 s. Catalogue statistics are recomputed from the
   history. The production models were trained on all rows, so replaying them would leak the future.
3. It builds a throwaway SQLite database through all migrations, with one member per MovieLens user and
   their history ratings at the original timestamps.
4. It creates and starts an experiment through the same service code as the API, and serves every
   member once (top 10) through `personalized()`. That covers hash assignment, variant config, the
   strategy decision (persisted) and exposure logging.
5. It records each member's held-out ratings as outcomes (`source = replay`; ≥ 4 counts as positive,
   ≤ 2 as negative). It then stops and concludes the experiment, which runs the same metric and
   statistics code as the results endpoint.

The run on 2026-09-24 is in `experiments/ab-replay-20260924T173730Z/` (`report.json`, `REPORT.md`).
Its primary metric is NDCG@10, tested at α = 0.05 with a Bonferroni-adjusted α of 0.025 per treatment.
The arms were:

- **control**: the champion config (MMR λ 0.8);
- **diverse**: MMR λ 0.6;
- **recency**: a 365-day profile recency half-life.

| variant | members | NDCG@10 | positive rate | rating rate | negative rate | diversity | novelty (bits) | coverage | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| control | 198 | 0.1105 | 52.53 % | 60.10 % | 4.04 % | 0.7153 | 9.48 | 3.51 % | 73.7 |
| diverse | 195 | 0.1042 | 46.67 % | 55.38 % | 3.08 % | 0.7157 | 9.54 | 3.65 % | 69.7 |
| recency | 217 | 0.1364 | 51.61 % | 59.91 % | 2.77 % | 0.7199 | 9.45 | 3.47 % | 63.0 |

| vs control | ΔNDCG@10 [95 % bootstrap CI] | p | Δpositive rate [95 % CI] | p |
|---|---|---|---|---|
| diverse | −0.0064 [−0.0400, +0.0295] | 0.708 | −0.0586 [−0.1573, +0.0401] | 0.246 |
| recency | +0.0259 [−0.0094, +0.0613] | 0.092 | −0.0091 [−0.1053, +0.0871] | 0.853 |

Checks and outcome of the run:

- **SRM:** observed 198 / 195 / 217 against an expected 203.3 each; p = 0.50, so no mismatch.
- **Guardrails:** none breached on the rate guardrails (negative rate is lower in both treatments);
  latency is not assessed in a replay (a shared laptop replay is not a serving measurement).
- **Power:** detecting a 5 % relative NDCG change would need about 13,400 members per arm in this
  unpaired design. The smallest arm has 195, and the report says so. A paired offline design needs far
  fewer members.
- **Conclusion:** **inconclusive, no winner.** Recency decay has the largest point estimate (+23 %
  relative, 95 % CI −8 % to +55 %; different members in each arm), but its CI includes 0 and p = 0.092 is above the adjusted 0.025. The
  diversity-λ arm shows no measurable change. Genre-Jaccard intra-list diversity barely moves either,
  because MMR in the ranker diversifies on content similarity, not on genres.
- **Design caveat.** An offline replay can score every member under every variant (a paired design);
  this between-arm replay demonstrates the pipeline, not the effect. Arm baselines move by about 0.01
  NDCG@10 through assignment alone (control 0.1105 against the benchmark's full-population 0.1207), so
  part of the recency difference is between-arm variation.
- **Next step:** a paired within-member replay (every member × every variant) for the offline estimate,
  and a real online test for the causal effect.

## 8. Limitations

- **Attribution is by time and item.** It does not model position bias. The first 24 h window is a
  default, not a tuned value. An interaction during two experiments' windows cannot happen, because
  there is at most one active experiment per surface. A member's organic ratings of listed items count
  as "interactions" even when they came from another page.
- **Only one surface** (`recommendations`) is experimentable. `/similar`, `/trending` and the
  because-you-watched rows are not, and they log no exposures.
- **Exposure rows grow with traffic**, at one row per served list. There is no retention job yet. The
  audit's `retention.prune` action is reserved for operations.
- **Results are computed on read**, with an O(exposures + events) scan per call, which is fine at this
  scale. A large deployment would want an incremental attribution job.
- **Challenger engines are per process.** Each worker loads its own copy lazily. The first request per
  worker per challenger pays the load time, and that latency lands in the challenger arm's p95. The
  latency guardrail's p95 of generated lists excludes cache hits but not the first load.
- **Assignment happens at the first request after enrolment**, not at randomisation time. Members who
  never request recommendations are never assigned, so the analysis is on exposed members.
- **Replay is not an online result**. See §7.
- **PostgreSQL** was verified locally against `postgres:17` (round trip, CHECK parity, `alembic check`).
  CI runs the `*_postgres` tests.
