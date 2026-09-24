# Streaming architecture: events, idempotency, bitemporal replay

_WS1 (events and ingestion), Phase 2. Covers P0 #2 of `docs/PHASE2_GAP_MATRIX.md` (a rating was overwritten
in place, a watch retry added a duplicate row, and there was no Idempotency-Key) and acceptance criteria P2.1
of `docs/PHASE2_ARCHITECTURE_AUDIT.md`._

JEV now records every member interaction, and every pushed observation of a generic domain, as a row in
one append-only `events` table. The current-state tables stay in place, but as projections of that log,
written in the same transaction. Intelligence runs read the log bitemporally, so a replay is deterministic
for a fixed code version, data files and active model: a movie replay reads the model that is active at replay
time (section 4.1).
A debounced in-process refresher turns fresh events into a live run.

There is no Kafka and no broker. The design uses PostgreSQL or SQLite, the existing FastAPI process and a
thread. Section 8 explains why that is enough.

| Piece | Where |
|---|---|
| Tables | `backend/jev_api/models/events.py`, migration `0007_events.py` |
| Appends, folds, projections, idempotency, as-of reads, refresher | `backend/jev_api/services/events.py` |
| HTTP API | `backend/jev_api/routers/events.py`; the write paths in `routers/movies.py`, `services/feedback.py` and onboarding in `routers/users.py` |
| Intelligence input gathering | `IntelService.gather_movie_inputs` / `gather_generic_adapter` in `services/intel.py` |
| Schemas | `backend/jev_api/schemas/events.py` |
| Tests | `tests/integration/test_events.py`, `tests/integration/test_events_ingest.py` |

---

## 1. Event schema

### 1.1 The `events` table (append-only)

| Column | Type | Meaning |
|---|---|---|
| `id` | integer PK | **Sequence number.** It increases monotonically in insertion order. Watermarks and "events since" counts use it. |
| `event_id` | UUID string, unique | A global identity, returned to clients. |
| `source` | string(32) | The producer channel: `app` (member endpoints and `POST /events`), `ingest` (observations), `backfill` (migration 0007). |
| `domain` | string(64) | `movie` for member interactions, `generic:<key>` for observations. |
| `event_type` | CHECK `ck_event_type` | `rating`, `rating_removed`, `watch`, `favorite`, `unfavorite`, `rec_feedback`, `observation` (`enums.EVENT_TYPES`). |
| `user_id` | integer, nullable | The member. Observations have none. There is deliberately no foreign key: the log outlives the mutable rows it describes. |
| `entity_id` | string(128) | The movie id (as text) or the generic domain's entity name. |
| `value` | float, nullable | The rating value or the observed value. |
| `payload` | JSON | Type-specific fields: `feedback` and `recommendation_id` (rec_feedback), `source` (favorite: `user` or `onboarding`), `attributes` (observation groups or calendar columns), `created_at` (backfilled rating). |
| `event_time` | timestamptz | **Valid time:** when it happened (client-supplied or server time). |
| `ingested_at` | timestamptz | **Knowledge time:** when JEV stored it. Always the server clock. |
| `idempotency_key` | string(200), nullable | Unique per `(source, idempotency_key)` (`uq_events_source_key`). Member keys are stored as `u<user_id>:<key>` and observation keys as `<domain>:<key>`, so keys are scoped. |
| `batch_id` | UUID string, nullable | The request batch that carried the event. |
| `schema_version` | integer | `1`. A payload change bumps it, and readers branch on it. |

Indexes:
- `ix_events_domain_time (domain, event_time, id)`: as-of reads;
- `ix_events_domain_ingested (domain, ingested_at)`: the knowledge-time cut and lag;
- `ix_events_user_entity (user_id, entity_id, event_type)`: projection refolds;
- `ix_events_batch`.

**Append-only is enforced by the database.** Migration 0007 installs BEFORE UPDATE and BEFORE DELETE triggers
that abort the statement, on SQLite and PostgreSQL, plus BEFORE TRUNCATE on PostgreSQL (tested on both). A
production deployment should also withhold the UPDATE and DELETE grants from the application role. That is a
deployment step, because the migration runs as the schema owner.

**Retention pruning is out of scope.** Rows are never deleted by the application. A future `retention.prune`
job (the audit action is pre-registered) would have to run as a role that may drop the trigger, or archive
by partition. Right-to-erasure is the same job: because `user_id` has no foreign key, deleting a user does
not touch the log.

### 1.2 Companion tables

- **`idempotency_keys`** `(scope, key)` unique, plus `fingerprint`, `status_code`, `response` and
  `created_at`. It holds the first response of a request that carried an `Idempotency-Key` header. `scope`
  is `user:<id>`.
- **`event_daily_counts`** `(domain, day)` PK, plus `accepted`, `duplicates` and `rejected`. These counters
  are maintained incrementally by an upsert in the ingest transaction, and they are what monitoring reads.
- **`intel_runs.event_watermark`** (JSON): the events a run read (section 4.3).

### 1.3 Lifecycle

```
request ─► validate ─► (Idempotency-Key seen? → return stored response)
        ─► append event(s)            ┐
        ─► refold projection row(s)   │ one transaction
        ─► upsert event_daily_counts  │
        ─► store idempotent response  ┘ commit
        ─► mark domain dirty ─► (debounce) ─► live intelligence run ─► run.event_watermark
```

### 1.4 Backfill (migration 0007)

Every row that existed in `ratings`, `favorites`, `watch_history` and `recommendation_feedback` becomes one
event:
- `source = 'backfill'`;
- `event_time = ingested_at =` the row's own timestamp (`updated_at` for a rating, whose `created_at` travels
  in the payload);
- key `<table>:<row id>`.

The folded backfill equals the tables exactly (tested). The knowledge time of pre-log rows is unknown, so it
is taken as their event time.

---

## 2. Projections

The current-state tables keep their shape, their API responses and their consumers. They are now derived
from the log.

| Projection | Fold (events of one key, ordered by `(event_time, id)`) |
|---|---|
| `ratings` (user, movie) | `rating` sets the value. `created_at` is the first rating after a removal. `updated_at` moves **only when the value changes** (an in-place UPDATE of an unchanged value never moved it). `rating_removed` deletes the row. |
| `favorites` (user, movie) | `favorite` adds; a repeat keeps the first `source` and time. `unfavorite` removes. |
| `watch_history` | Additive: one row per `watch` event, `watched_at = event_time`. |
| `recommendation_feedback` | The `upsert_feedback` rules as a fold. There is one verdict slot and one click slot per recommendation, or per movie when there is none. A changed verdict replaces the old one and moves `created_at`; a repeat is a no-op. |

- **Ordering is by event time, not arrival.** An event that arrives late (with an older `event_time`, which
  `POST /events` allows) lands where it belongs, so an old rating cannot overwrite a newer one. The live
  write path therefore *refolds* the key from its events on every append instead of patching the row. The
  live tables and a full rebuild use the same fold functions (`fold_rating`, `fold_favorite` and
  `fold_feedback` in `services/events.py`), so they cannot disagree.
- **Rebuild.** `fold_log()` folds the whole log, `live_projections()` reads the tables, and
  `diff_projections()` compares them row for row (missing, extra and changed per table). `POST /events/replay`
  exposes this to admins:
  - `{"apply": false}` (the default) verifies;
  - `{"apply": true}` rewrites drifted rows in place, audits `events.replay` and clears the recommendation
    cache.

  Tests check that the rebuild equals the live tables after mixed traffic (re-rates, unrates, favorite
  toggles, watches, verdict changes, repeated clicks, onboarding), and that a deliberately corrupted
  projection is detected and repaired.
- Every writer goes through the log: the four movie endpoints, `POST /events`, `upsert_feedback` (behind
  `POST /recommendations/feedback`) and onboarding favorites. A new writer of these tables must call
  `services.events.apply_member_event`. A direct INSERT shows up as `extra` in the replay diff.

---

## 3. Idempotency

### 3.1 The `Idempotency-Key` header

These endpoints accept an optional `Idempotency-Key` header (printable ASCII, 1 to 200 characters):
- `POST /movies/{id}/rate`, `DELETE /movies/{id}/rate`;
- `POST /movies/{id}/favorite`, `POST /movies/{id}/watch`;
- `POST /events`, `POST /intel/domains/{key}/observations`.

- **First request:** the work, the event append and the stored response commit together.
- **Retry with the same key and the same request** (a fingerprint of method, path and body): the stored
  response comes back byte for byte (the same `profile_version` too), and nothing is written. Tested: 10
  identical rates give 1 event, 1 row and 10 identical responses.
- **Same key, different request:** 422 `this Idempotency-Key was already used for a different request`.
  Keys are single-purpose.
- **Concurrent duplicates:** the unique constraints (`uq_idempotency_scope_key`, `uq_events_source_key`)
  make the loser's commit fail. It rolls back and returns the winner's stored response.
- `upsert_feedback(..., idempotency_key=)` takes a key too. WS5's `POST /recommendations/feedback` can pass
  the header through; feedback is already an upsert, so a keyless retry is harmless.

### 3.2 Per-item keys in batches

Every item of `POST /events` and of the observations endpoint carries a required `idempotency_key`:
- a known key with the same content is a `duplicate`, and the response points at the original `event_id`
  and `seq`;
- a known key with different content is `rejected` (`idempotency_key already used for a different event`);
- a key repeated inside one batch is a duplicate of its first occurrence.

### 3.3 Keyless retries: the rules

| Write | Without a key |
|---|---|
| rate, unrate, favorite / unfavorite, rec feedback | **State-setting.** A retry appends a second identical event, and the fold absorbs it (same value, same state), so the projection and every read are unchanged. Responses are exactly as before Phase 2 (including the `profile_version` bump). |
| watch | **Additive.** A keyless watch identical to one of the same member and movie appended within `events_watch_dedupe_seconds` (default 60 s, by `ingested_at`) is a retry. It returns `{watched: true, profile_version: <current>}`, appends nothing and counts as a duplicate. A keyed watch always counts: a new key is a new intent. |

---

## 4. Bitemporal reads and replay determinism

### 4.1 The rule

An intelligence run is a cut through the log on two axes:

```
visible(event)  ⇔  event.event_time  <= as_of            (it had happened)
               AND event.ingested_at <= knowledge_time    (JEV knew about it)
```

| Run | as_of | knowledge_time |
|---|---|---|
| live | now | now |
| replay (explicit `as_of`) | as_of | **as_of** (default); `knowledge_time=` can be passed to `gather_*` |

Because a replay's knowledge time defaults to its `as_of`, events that arrive **after** the replay date never
change it. That covers late mobile uploads, backfills and revised observations. Replaying the same `as_of`
tomorrow therefore gives the same inputs as today, however many events arrived in between. A replay can
still ask what JEV knew later about that date by passing a later `knowledge_time` (for example to measure
publication lag). Tested with a known event, a late event and a future event:
- the replay sees only the known one;
- the replay with the later knowledge time also sees the late one;
- the live run sees the newest.

**Replay determinism holds for a fixed code version, data files and active model.** The event cut above makes a
replay independent of events that arrived after its `as_of`. It does not freeze two other inputs:

- **The active model.** A movie replay reads the manifest that is active *at replay time*
  (`ml/jev_ml/domains/movie/ingest.py`, `load_default_inputs`), so a promotion or rollback between two replays of
  the same `as_of` changes the model-based stages (model governance, lapse, served-strategy evidence). Every run
  records the version it used (`run.model_version` in the result and `intel_runs.model_version`), and
  `load_default_inputs(..., model_version=...)` pins a replay to that manifest from Python
  (`tests/intel/test_replay_model_version.py`). The API does not expose pinning yet.
- **Code and data files.** The pipeline version, the config hash and the input fingerprint are in every run
  (`run.pipeline_version`, `run.config_hash`, `run.input_fingerprint`); a replay under different code or a
  refreshed MovieLens/CSV file is a different computation.

What is tested: the frame-level cut (`test_replay_never_sees_late_arriving_events`) and two back-to-back replays
with no intervening change (`test_same_as_of_gives_identical_decisions_via_api`). An end-to-end test with events
arriving *between* two replays that compares full outputs does not exist yet. On PostgreSQL the watermark can miss
in-flight commits (section 4.3).

Served recommendations are not events (WS5 owns them). A replay cuts them at `created_at <= as_of`, and a
live run reads all of them.

### 4.2 What the movie run reads

`IntelService.gather_movie_inputs` calls `events.app_frames(db, as_of, now, knowledge_time)`:
- `app_ratings`: the rating fold at the cut, with timestamp = `updated_at`;
- `app_feedback`: the feedback fold, then the newest verdict and newest click per member per movie (the
  `latest_feedback()` rule);
- `app_served`: the served recommendations, cut as above.

A live run therefore reproduces the projections exactly, and a replay no longer mixes 2026 app events into a
2017 analysis (gap-matrix row "Replay vs live: app-event clock").

### 4.3 The watermark (lineage)

Every run stores `intel_runs.event_watermark`, which is also exposed as `event_watermark` on `GET /intel/runs*`:

```json
{"domain": "generic:us-unemployment", "as_of": "…", "knowledge_time": "…",
 "max_event_id": 3, "max_ingested_at": "…", "max_event_time": "…", "n_events": 2}
```

The run reads exactly the visible events with `id <= max_event_id`, so its output is traceable to a log
prefix. `GET /events/health` reports `events_since_last_run` (events with an id above the latest live run's
watermark). For WS4 lineage, `(max_event_id, knowledge_time)` plus the data version identify the event input.

Caveat: on PostgreSQL, two transactions that commit out of id order can leave a gap. An event with a smaller
id may commit just after the watermark was read. SQLite serialises writers, so the watermark there is exact.
The effect is bounded by the in-flight transactions at run start, and those events are always visible to the
next run.

### 4.4 Generic domains: pushed observations

`POST /intel/domains/{key}/observations` appends `observation` events. `gather_generic_adapter` folds the
visible ones and merges them into the adapter's dataset through `GenericAdapter(cfg, frame=…)` (the `frame`
hook; `ml/` is not modified):
- **Revisions:** the newest ingest per (entity, period) wins, and it replaces the file's row for that period.
- **Row template:** a pushed row copies its entity's newest file row (entity type and group columns). The
  payload's `attributes` can override the group and calendar columns, and a domain with a `calendar_check`
  column requires it.
- **Availability:** the file's `availability_lag_days` models a publisher's release delay for rows whose
  knowledge time is unknown. A pushed row's knowledge time is its `ingested_at`, which the bitemporal cut has
  already applied, so pushed rows are usable immediately (lag 0). For example, a September 2026 rate pushed on
  24 September is visible to a live run the same day, and invisible to a replay as of 20 September.
  (`ObservedAdapter` overrides the adapter's private `_lag_days`, `_raw` and `_provenance`, and a test guards
  that contract.)
- **Lineage:** the data version changes, because the checksum covers the file plus the pushed rows.
  Provenance reads `…; + N pushed observations (events up to id M)`.
- Without any pushed observation, the plain adapter is used, so behaviour is byte-identical to before.

---

## 5. Ingestion API

### `POST /events` (authenticated member)

```json
{"events": [
  {"event_type": "rating", "movie_id": 356, "value": 4.5, "idempotency_key": "r-356-1",
   "event_time": "2026-09-22T18:04:00Z"},
  {"event_type": "rec_feedback", "movie_id": 1, "feedback": "like", "recommendation_id": 812,
   "idempotency_key": "fb-812"}
]}
```

- **Types:** `rating` (a value from 0.5 to 5.0 in half steps), `rating_removed`, `watch`, `favorite`,
  `unfavorite` and `rec_feedback` (a `feedback` value, plus an optional `recommendation_id` that must be the
  member's own and for that movie). `user_id` may be omitted; when given it must be the caller, and another
  member's id is a **403** for the whole request.
- **Structure → 422 for the whole batch:** an empty batch, more than `events_batch_max` items (500), an
  unknown type or field, a missing key, a value on a watch, or a rating that is not a half step.
- **Per item → `rejected` with a reason:**
  - `movie not found`;
  - `recommendation not found`;
  - `event_time is in the future` (beyond `events_max_future_skew_seconds`, 300 s);
  - `event_time is more than 30 days old` (`events_max_age_days`);
  - key reuse with different content.
- **Response:**

  ```json
  {"batch_id": "…", "domain": "movie", "accepted": 1, "duplicates": 0, "rejected": 1,
   "items": [{"index": 0, "status": "accepted", "event_id": "…", "seq": 41, "reason": null}, …]}
  ```

- `event_time` is preserved in the log and in the projections (`ratings.updated_at`, `watch_history.watched_at`).
  `ingested_at` is always the server clock. `profile_version` is bumped once per batch when anything changed.

### `POST /intel/domains/{key}/observations` (admin; pluggable)

The auth dependency is `require_observation_writer` in `routers/events.py`. Admins pass today, and service
tokens plug in there without touching the handler.

```json
{"observations": [{"entity": "US", "event_time": "2026-09-01T00:00:00Z", "value": 5.9,
                   "idempotency_key": "us-2026-09", "attributes": {}}]}
```

Validation is against the domain config:
- a known entity (present in the dataset);
- a finite value inside `value_range`;
- `event_time` at a period start at midnight UTC (the first day of the month for monthly domains), and not in
  the future;
- attributes limited to the domain's group and calendar columns.

It is 404 for an unknown domain, 409 for an unavailable one and 422 for `movie`. The cap is
`events_observations_batch_max` (5000). The response has the same shape as `POST /events`, and each call is
audited as `events.ingest`.

### `POST /events/replay` (admin)

`{"apply": false}` returns `{applied, consistent_before, consistent_after, diff}`. `diff` has
`missing`/`extra`/`changed` counts and samples for each of the four projections. `apply: true` repairs them
and audits `events.replay`.

### `GET /events/health` (admin)

For each domain:
- `events`, `max_event_id`, `max_event_time`, `max_ingested_at`;
- `event_time_lag_s` (now minus the newest event time; for monthly statistics this is naturally weeks);
- `ingest_lag_s` (now minus the newest ingest: silence on the feed);
- `today` and `last_7_days` `{accepted, duplicates, rejected}` (from `event_daily_counts`);
- `last_live_run`, `watermark` and `events_since_last_run`;
- `refresh {dirty, last_refresh}`.

A top-level `refresher` block holds `enabled`, `debounce_seconds`, `max_delay_seconds`, `dirty_domains` and
`running`.

---

## 6. Near-real-time refresh and monitoring

- **Dirty marks.** Every accepted append (member endpoints, `POST /events`, observations) marks its domain
  dirty in the process's `EventRefresher` (`services/events.refresher_for(app)`). It is created on first use,
  because `main.py` is frozen, and its worker thread starts on the first mark.
- **Debounce.** A domain runs once it has been quiet for `events_refresh_debounce_seconds` (default 30), or
  `events_refresh_max_delay_seconds` (default 300) after its first mark under steady traffic. That gives at
  most one run per domain per burst. The run is `IntelService.run("schedule", domain=…)`: a **live** run
  through the existing one-run-at-a-time lock and the per-domain logic. If the lock is busy
  (`IntelBusyError`), the domain stays dirty and is retried after another debounce period, so the refresher
  never blocks and never stacks runs.
- **Enablement.** `events_refresh_enabled` defaults to `None` (auto): on, except under pytest
  (`PYTEST_CURRENT_TEST`) or `JEV_ENV=test`. The test suite therefore never starts background runs, and the
  unit tests drive `EventRefresher.tick()` with a fake clock.
- **Incremental aggregation.** `event_daily_counts` is upserted in the ingest transaction, so health reads are
  O(days × domains) and never scan the log. The only log scans in health are per-domain `max`/`count`
  aggregates, served by the indexes.
- **Metrics** (`GET /admin/metrics`, in-process registry):
  - `events_ingested{domain:type}`, `events_duplicates{domain}`, `events_rejected{domain}`;
  - `events_refresh{marked_dirty, busy, errors, runs_succeeded, runs_failed, last_domain}`.

**What the frontend should show (an admin "Events" page, WS6):** one card per domain with:
- today's accepted, duplicate and rejected counts, and a 7-day sparkline from `last_7_days`;
- ingest lag, for example "last event 5 s ago", in amber after about an hour of silence on a feed that should
  be live;
- event-time lag, with a per-domain expectation: a monthly statistic is weeks old by nature;
- `events_since_last_run`, shown as "N events not yet in intelligence", together with the dirty and
  last-refresh state and a link to the last live run;
- the watermark (`max_event_id`, `max_ingested_at`), shown next to the run.

A global strip shows whether the refresher is enabled and running, and the debounce. The **rejected** count
should be the most prominent alert: rejections mean a producer is sending bad data.

---

## 7. Configuration (`JEV_EVENTS_*`)

| Setting | Default | Meaning |
|---|---|---|
| `events_batch_max` | 500 | Items per `POST /events` |
| `events_observations_batch_max` | 5000 | Items per observations request |
| `events_max_future_skew_seconds` | 300 | Tolerated client clock skew |
| `events_max_age_days` | 30 | Oldest accepted member `event_time` |
| `events_watch_dedupe_seconds` | 60 | Keyless watch retry window (0 disables) |
| `events_refresh_enabled` | auto | `true`/`false`; auto = on outside tests |
| `events_refresh_debounce_seconds` | 30 | Quiet time before a live run |
| `events_refresh_max_delay_seconds` | 300 | Upper bound under steady traffic |

---

## 8. Limitations, and why this is enough

- **No broker.** The log is a table in the same database as the projections, so the append and the projection
  commit atomically. There is no dual-write problem, no outbox and no consumer offsets: `events.id` is the
  offset. JEV has one writer tier (the API) and one consumer (the intelligence run). At laptop scale, SQLite
  sustains the P2.1 targets: measured at **730 events/s, and 10 000 events in 13.7 s** through `POST /events`
  (20 × 500 mixed events, single process, with projection refolds). PostgreSQL does better. A broker would add
  an operational dependency, and it could not run on "any laptop".
- **Single-process refresher.** Dirty marks live in memory, per process. With several uvicorn workers, each
  debounces its own ingests. The run lock is per process as well, so two workers could each start a live run
  of the same domain (the run itself is safe; the cost is duplicated work). Run one worker, or set
  `JEV_EVENTS_REFRESH_ENABLED=false` on all but one and trigger runs from a scheduler. A lost mark (for
  example after a restart) only delays a refresh: `events_since_last_run` makes the backlog visible, and the
  next ingest or the startup refresh catches up.
- **Refolds on write.** Each member event refolds one (user, movie) key. That is O(events for that key), which
  is tiny for real members. A key with thousands of events would call for a snapshot, and none is needed today.
- **Replays read the active model** (section 4.1): determinism holds for a fixed code version, data files and
  active model; the run records `model_version`, but the API cannot pin it.
- **The watermark under concurrent PostgreSQL writers** can miss an event committed out of id order at run
  start (section 4.3). It is exact on SQLite.
- **Retention, erasure and pruning of `idempotency_keys`** are out of scope. Both tables grow without bound
  until the `retention.prune` job exists.
- **Recommendation exposures** (`recommendations` rows) are not events yet. That is WS5's cache-hit exposure
  logging item.
- **Clock trust.** `event_time` from a member is trusted within the skew and age window. `ingested_at` is the
  server's clock and is the only time used for knowledge cuts.
