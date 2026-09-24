# Retraining and model governance

Phase 2, WS2 (docs/PHASE2_ARCHITECTURE_AUDIT.md §6 P2.2, gap matrix P0 #3). This document covers how member
feedback becomes training data, and how a retrained model becomes a **candidate** that the promotion
gate must pass before it can serve. It also covers rollback, lineage, the scheduler, the reproducibility
commands and the known limitations.

Code:

| Where | What |
|---|---|
| `ml/jev_ml/governance/snapshot.py` | Versioned, content-hashed training snapshots (no database) |
| `ml/jev_ml/governance/retrain.py` | Trains a snapshot into a registered candidate and writes `lineage.json` |
| `ml/jev_ml/governance/gates.py` | The promotion gate |
| `ml/jev_ml/governance/promotion.py` | The promotion rules, shared by the API, the CLI and the scheduler |
| `ml/jev_ml/registry.py` | Lifecycle states, `previous`, and the activation history |
| `backend/jev_api/services/governance.py` | Reads app feedback from the DB. Also: jobs, the retrain lock, promotion, rollback and the scheduler tick |
| `backend/jev_api/routers/governance.py` | `/governance/*` (admin) |
| `backend/jev_api/models/governance.py`, migration `0008` | `dataset_snapshots`, `training_jobs`, `model_governance`, `governance_locks` |
| `scripts/export_training_snapshot.py`, `scripts/retrain.py` | The CLI |
| `docker-compose.yml` | `retrain` (profile `train`) and `scheduler` (profile `scheduler`) services |

## 1. Data flow

```
ratings / favorites / watch_history / recommendation_feedback  (DB projections)
        │  read_app_events()                                   + data/processed (MovieLens)
        ▼
snapshot  data/snapshots/snap-<sha16>/   ── dataset_snapshots row, audit dataset.snapshot
        │  train_candidate()  → run_pipeline(processed_dir=snapshot, activate=False)
        ▼
candidate  models/<version>/ + lineage.json ── registry state "candidate", audit model.retrain + model.register
        │  (calibrate, if the incumbent is calibrated)
        │  evaluate_candidate(candidate, active, snapshot)
        ▼
gate ──fail──► state "rejected", audit model.reject (reasons stored)
  │
  pass ─► promotable. POST /governance/models/{v}/promote (or auto-promote if configured)
        ▼
EngineHolder.activate: load first → registry.set_active(action="promote") → swap → clear caches
        │  audit model.promote (detail.promoted, detail.gate_ok, forced, reason)
        ▼
active  ──POST /governance/models/rollback──► previous version restored, audit model.rollback
```

Registering never activates. `register_version(..., activate=False)` is now the default, and the old
implicit rule ("activate if nothing is active") is gone. Two paths still activate, and both do it
explicitly:

- `scripts/train_models.py` activates the **bootstrap** model of an empty registry, or any model
  when you pass `--activate`.
- The test fixture passes `activate=True`.

`run_pipeline`'s own default is still `activate=True`. That default belongs to `training.py` and is
listed as a hook in §8. The governance path always passes `activate=False`, and `train_candidate`
refuses a candidate that ended up active.

## 2. Snapshots: how feedback becomes training data

A snapshot is a directory shaped like `data/processed`, so the training API and calibration read it
without changes. It holds the following files:

- `interactions.csv`: the MovieLens rows plus the app rows.
- `movies.csv`: the base catalogue, byte for byte.
- `dataset_meta.json`: the base metadata, with `dataset_version = snapshot_id`.
- `app_interactions.csv`: the provenance of each app row.
- `exclusions.csv`.
- `manifest.json`: the sources, row counts, cut-off, watermark, semantics and file hashes.

Training reads one `rating` per (user, film). That rating sets both the implicit confidence
(`signals.rating_weight`) and the signed taste (`signals.preference_weight`). Each app signal is
therefore mapped to the **pseudo-rating whose training weight matches the serving weight in
`signals.py`**:

| App signal | Training row | Training weight | Serving weight (engine) |
|---|---|---|---|
| explicit rating | the rating itself (latest). Wins over every other signal on that film | `rating_weight(r)` | same |
| favourite | 5.0 | 2.0 | `FAVORITE_WEIGHT` 2.0 |
| onboarding pick | 4.5 | 1.5 | `ONBOARDING_PICK_WEIGHT` 1.5 |
| like (latest verdict) | 4.0 (counts as relevant, like a rating ≥ 4) | 1.0 | — (feedback only) |
| watch | 3.5, at the first watch time | 0.5 | `WATCH_WEIGHT` 0.6 (no rating maps to 0.6) |
| dislike (latest verdict) | 0.5 | 0.1, taste −1 | excluded from the member's lists |
| not_interested (latest verdict) | **no row**. The pair goes to `exclusions.csv` | — | excluded |
| clicked | ignored (an interaction, not a judgement) | — | — |

Precedence per (user, film): explicit rating, then a latest verdict of dislike or not_interested, then
favourite, onboarding pick, like and watch.

What these rules do and do not achieve:

- **A dislike is not a true negative.** The implicit CF and ALS models only see positive confidences,
  so a disliked film still counts as weak consumption (0.1). This matches how MovieLens ratings ≤ 2.5
  are treated. Its content and genre taste is −1.
- **not_interested can only remove evidence.** It drops the member's other signals on that film. The
  training API has no way to express "never recommend this". Serving already excludes these films.
- **App users stay distinguishable.** Their id is `app_user_id + JEV_GOVERNANCE_APP_USER_OFFSET`
  (default 10,000,000, far above MovieLens ids). A snapshot fails if the base ids would collide.
- **Rows outside the snapshot are dropped and counted.** This covers films outside the base catalogue
  (for example, admin-added films) and events after the cut-off.

**Determinism.** `snapshot_id = "snap-" + sha256(semantics version, interactions.csv, movies.csv,
exclusions.csv)[:16]`. The rows are sorted before they are written. The default cut-off is the newest
app event, not "now", so the same database always produces the same id. A second export returns the
existing snapshot with `new: false`, and the tests assert this. The manifest also records a
**watermark**: the newest row of each source table and, when WS1's `events` table exists, its
`max(id)`, `max(ingested_at)` and row count.

## 3. The gate: definitions and rationale

`evaluate_candidate(candidate, incumbent, snapshot_dir, GateConfig)` returns a JSON result. The result
is stored in `model_governance.gate` and `models/<v>/gate.json`, and summarised per gate by
`GET /governance/models`.

**What is compared.** Both production artifacts are trained on all of their data. Scoring either one on
any held-out split of the candidate's snapshot would therefore leak, because the candidate has already
seen that split. The gate instead compares the two **training recipes** (the component
hyper-parameters and the hybrid config, read from each manifest). Both are refit with the same seed on
train+val of the candidate snapshot's split (the candidate's configured protocol) and scored on its
test part. Both see the same users, the same profiles and the same relevant items. The test part is
fingerprinted (`split.test_fingerprint`). The artifact itself is then checked for the things only an
artifact can show.

| Gate | Pass rule (default) | Why |
|---|---|---|
| `split` | `evaluation.leakage.check_split` holds | Never compare on a split that breaks its own temporal contract |
| `artifact` | The candidate loads (`RecommendationEngine` validates shapes and finite factors). The self-check serves a cold and a warm list: k unique films, finite scores, nothing already in the profile. A failure skips every other gate. | Never serve a model that cannot load |
| `ndcg@10` | Lower bound of the paired bootstrap CI of the per-user difference (candidate − incumbent) ≥ −0.010 | Non-inferiority, not superiority: a retrain on fresh data should be allowed through when it is statistically no worse. Paired, per user (`evaluation.stats.paired_comparison`). The margin follows the power rule below |
| `recall@10` | The same test, margin 0.011 | A second accuracy view, less sensitive to rank position |
| `cold_start` | The same NDCG@10 test on profiles truncated to `cold_start_profile_size`, margin 0.006 | New members must not pay for gains on warm members |
| `coverage@10` | ≥ (1 − 0.2) × the incumbent's | Catches a collapse into popularity. Accuracy can look fine while the catalogue shrinks |
| `calibration` | If the incumbent is calibrated: the candidate must be too, with headline test ECE ≤ incumbent + 0.01 and AUC ≥ incumbent − 0.01. Skipped if neither is calibrated | Served confidences must not silently get worse or disappear |
| `latency` | p95 of `build_profile` + `recommend(k=10)` over sampled test users ≤ 250 ms | Laptop serving budget |

The CI is a two-sided `1 − alpha` interval. With `alpha = 0.10`, its lower bound is the one-sided 95 %
non-inferiority bound. The resampling unit is the user: B = 2000 resamples, also with `--quick` (quick mode
only reduces the permutations to 1000 and the latency probes; the bootstrap is cheap next to the refits).
Content features of both refits use only the tags written before the test period
(`evaluation.leakage.movies_with_tags_before`) when the raw MovieLens `tags.csv` of the same dataset is
available; `gate.json` records which (`split.tags`).

**Power rule for the margins (gate-1.1.0).** The gate passes when `diff − z·SE ≥ −margin`, with
z = 1.645 at alpha 0.10. A candidate that is exactly as good as the incumbent (true Δ = 0) therefore
passes with probability `Φ(margin/SE − z)`, and that probability reaches 80 % when
`margin ≥ (z + z_0.80)·SE = 2.49·SE` (`gates.power_margin`). The defaults apply this rule to the paired
SEs observed on the MovieLens split (594 users, `models/jev-20260924T174707Z-0298b516/gate.json`):

| gate | observed SE | old margin → P(pass \| Δ = 0) | new margin → P(pass \| Δ = 0) |
|---|---|---|---|
| `ndcg@10` | 0.0039 | 0.005 → 0.36 | **0.010** → 0.80 |
| `recall@10` | 0.0044 | 0.010 → 0.74 | **0.011** → 0.79 |
| `cold_start` | 0.0022 | 0.005 → 0.75 | **0.006** → 0.83 |

The three accuracy gates are conjunctive, so the joint pass rate of an equivalent candidate is lower
than each (about 0.5 if they were independent; they are positively correlated, so somewhat higher).
An NDCG margin of 0.010 is about 8 % of the incumbent's 0.122: the gate now accepts a candidate whose
NDCG@10 is at most about one point lower, with 95 % one-sided confidence. Every non-inferiority gate
records its observed `se`, the `power_if_equivalent` it had and the `margin_for_80pct` the rule would
require, so an underpowered comparison (a smaller snapshot, `max_users`) is visible in `gate.json`.
Tests: `tests/ml/test_governance.py::test_gate_margins_follow_the_power_rule`,
`test_gate_records_its_power`; the deliberately worse (popularity-only) model is still rejected and the
equivalent retrain still passes (`test_gate_rejects_a_worse_model`, `test_gate_passes_an_equivalent_model`).

**Reading a rejection.** A rejection means "non-inferiority not shown", not "worse". The real quick retrain
of 2026-09-24 (`models/jev-20260924T174707Z-0298b516/gate.json`, gate-1.0.0) was not shown non-inferior
(Δ −0.0009, 90 % CI −0.0073..+0.0055), so the gate rejected it: the gate could not distinguish it from the
incumbent. Under the new margins that NDCG result would pass (lower bound −0.0073 ≥ −0.010), and the
cold-start result (lower bound −0.0055) would pass the 0.006 margin; that gate was not re-run.
Without an incumbent (bootstrap), gates 2–6 are `skipped`, and the decision rests on `artifact` and
`latency`. Every threshold is a setting (`JEV_GOVERNANCE_GATE_*`, `config.py` governance section).

**Pass or fail.** When a gate fails, the version is marked `rejected` in both the registry and the DB,
and a `model.reject` audit row stores the reasons. When every gate passes, the version stays a
`candidate` and becomes *promotable*.

## 4. Promotion and rollback

- `POST /governance/models/{version}/promote` is **manual by default**. Without `force`, the latest
  gate must have passed **against the model that is active now**. A gate run against an incumbent that
  has since been replaced is stale, and the response says "evaluate again". With `force: true` a
  `reason` is required. Force bypasses the gate but never the load check. Every attempt is audited as
  `model.promote`, refused ones included, with `detail.promoted`, `detail.gate_ok`, `forced`, `reason`
  and `blockers`.
  - Note: the audit redactor masks any key that contains "pass" (`logging_setup._SENSITIVE_KEYS`), so
    the §9.3 name `detail.passed` cannot be used. The equivalent fields are `promoted` and `gate_ok`.
- **Automatic promotion** happens only when `JEV_GOVERNANCE_AUTO_PROMOTE=true` **and** the gate passed.
  It is off by default.
- `POST /governance/models/rollback` restores `previous`, the version that served before the active
  one. The body must carry a `reason` (3–500 non-blank characters; 422 otherwise). It is audited as
  `model.rollback` with from, to and reason. A second rollback walks further back
  and does not toggle between two versions. A rollback target has already served, so it is not gated
  again.
- **Hot swap.** `EngineHolder.activate` loads the version first, and a broken model is never swapped
  in. It then points `registry.json` at the version (recording the action in `history`), swaps the
  engine under a lock and clears the recommendation caches (`rec:`, `sim:`, `trend:`, `meintel:`,
  `strat:`; every key also carries the engine version).
- **Other processes** (other API workers, the CLI, the scheduler) follow a promotion by polling. On a
  request, at most every `JEV_GOVERNANCE_ENGINE_POLL_SECONDS` (default 5), the holder stats
  `registry.json`. If the file's mtime has changed and it names another version, that version is loaded
  in a background thread while the old one keeps serving.
- **The legacy `POST /models/{id}/activate` now enforces the gate.** Re-activating the serving version
  reloads it, as before. Any other version goes through the same `promote` rule without force, and gets
  a 409 with the blockers when the gate has not passed. Every 409 from promote/activate has the shape
  `{"detail": "<message>", "blockers": ["<reason>", …], "request_id": "…"}`: `detail` is always a string, as
  for every other error.
- **Where a listed model comes from.** `GET /governance/models` marks each entry with `source`: `"job"` when a
  `training_jobs` row of this database produced it (its `job_id` resolves via `/governance/jobs/{id}`), or
  `"artifact"` when lineage and gate were mirrored from `models/<version>/lineage.json` and `gate.json`
  (for example a fresh database); `job_id` then names a job of another database. To force, use the governance endpoint with a
  reason. The endpoint still writes `model.activate` on success.

Lifecycle: `candidate → rejected` (the gate failed), `candidate → active` (promoted), and
`active → retired` (superseded or rolled back). `registry.json` keeps its v1 keys (`active`,
`versions`) and adds `states`, `previous` and `history`. A v1 file is read with the active version as
`active` and every other version as `retired`.

## 5. Lineage

`models/<version>/lineage.json` is written next to the untouched manifest and mirrored into
`model_governance.lineage`. It records:

- the snapshot id, content hash, cut-off and row counts;
- `dataset_version` (equal to the snapshot id);
- the config path and **config hash** (SHA-256 of the parsed YAML);
- the **git commit**, `jev_ml_version` and **seed**;
- the quick flag, the experiment run and the split summary;
- the training test metrics (hybrid, warm and cold);
- the training time and the incumbent at training time;
- the **job id**, trigger and requester, and the **retrain decision id** when a decision triggered the
  job.

`GET /governance/models/{version}/lineage` combines this with the following:

- the snapshot row and its manifest;
- the job, with its steps;
- the gate result;
- who promoted the version and when, and whether it was forced and why;
- the registry history of the version.

## 6. Triggers, jobs and the scheduler

- **Jobs** (`training_jobs`) record the following:
  - kind: `retrain` or `evaluate`;
  - trigger: `manual`, `schedule`, `decision` or `cli`;
  - status: `queued`, `running`, `succeeded` or `failed`;
  - the snapshot, the model version and the incumbent;
  - `gate_passed` and `promoted`;
  - timings;
  - `steps` (the logs summary: snapshot, train, calibrate, gate and promote, each with its key
    numbers);
  - `error`, holding the traceback tail on failure.
- **One job at a time, across processes.** On PostgreSQL the lock is `pg_try_advisory_lock` on a
  dedicated AUTOCOMMIT connection, which the server releases if the process dies. On SQLite it is a
  lease row in `governance_locks` that expires after `JEV_GOVERNANCE_JOB_TIMEOUT_MINUTES`. The API
  answers 409 while the lock is held.
- **API.** `POST /governance/retrain` (202) and `POST /governance/models/{v}/evaluate` (202) run in a
  worker thread.
- **CLI.** `scripts/retrain.py run` runs the same job inline.
- **Scheduler.** `scripts/retrain.py schedule` is the compose `scheduler` service. Each tick
  (`JEV_GOVERNANCE_SCHEDULE_POLL_SECONDS`) does the following:
  1. It reads the `retrain_model` decision of the latest successful **live** movie run from
     `intel_decisions`. Replays are ignored.
  2. With `JEV_GOVERNANCE_SCHEDULE_REQUIRE_DECISION=true` (the default), it retrains only when the
     answer is `yes`, it is not abstained, and no job has used that `decision_id` yet. The decision id
     is stored on the job and in the candidate's lineage, which connects the decision to the action.
  3. With `false`, it retrains on the interval alone.
  4. Automatic jobs are always at least `JEV_GOVERNANCE_SCHEDULE_INTERVAL_MINUTES` apart.
- **Where the loop runs.** The loop is disabled unless `JEV_GOVERNANCE_SCHEDULE_ENABLED=true`, and it
  is off in dev and in tests. It runs in its own process, not inside the API, because `main.py` is
  integrator-frozen. One scheduler process can serve any number of API workers, since they follow the
  registry.

## 7. Reproducibility commands

```bash
# export a snapshot (same DB + cut-off → same id); lands in data/snapshots/ (gitignored)
uv run python scripts/export_training_snapshot.py [--cutoff 2026-09-24T00:00:00Z]

# snapshot → train (no tuning) → candidate → gate against the active model; never activates
uv run python scripts/retrain.py run --quick        # --full tunes hyper-parameters (minutes)
uv run python scripts/retrain.py evaluate VERSION   # gate an existing version on its snapshot
uv run python scripts/retrain.py status             # active, previous, states, last jobs
uv run python scripts/retrain.py promote VERSION    # 2 = blocked (prints the blockers)
uv run python scripts/retrain.py promote VERSION --force --reason "incident 42"
uv run python scripts/retrain.py rollback --reason "negative feedback spike"
uv run python scripts/retrain.py schedule --once    # one scheduler tick (needs ..._SCHEDULE_ENABLED=true)

# Docker
docker compose --profile train run --rm retrain                 # RETRAIN_ARGS=--full for tuning
docker compose --profile scheduler up -d scheduler

# re-run a candidate's training exactly: its snapshot + its config + its seed
uv run python -c "from pathlib import Path; from jev_ml.training import run_pipeline; \
  run_pipeline(Path('<config_path>'), quick=True, processed_dir=Path('data/snapshots/<snapshot_id>'), activate=False)"
```

Tests: `tests/ml/test_governance.py` covers snapshots, registry states and the gate pass, reject and
broken-artifact cases on the synthetic fixture. `tests/integration/test_governance_api.py` covers the
API, promotion and rollback, audit, the lock and the decision-driven scheduler.
`tests/integration/test_governance_migration.py` runs the 0008 round trip on SQLite, and on PostgreSQL
when `JEV_TEST_POSTGRES_URL` is set.

## 8. Limitations and hooks

- **The gate compares recipes, not the served artifacts.** This avoids leakage (§3). The downside is
  that a defect that exists only in the production fit (all data) is caught only by the `artifact`
  and `latency` checks.
- **The gate uses one split.** It is the candidate's configured split of its own snapshot, so the
  incumbent's recipe is re-evaluated on the new data. That is the question that matters for promotion,
  but the result is not the incumbent's historical metric.
- **Calibration is compared on each model's own headline test ECE.** Both are fitted by the same code
  on each model's own data, not on one shared calibration set.
- **The feedback semantics are approximations.** Dislike is weak consumption rather than a negative, a
  watch is weighted 0.5 rather than 0.6, and not_interested can only remove evidence (§2).
- **No automatic post-promotion monitor yet** (the P2.2 "3σ negative-feedback" rollback). Rollback is
  one request or one CLI call.
- **Multi-worker swap works by polling.** A worker follows a promotion within the poll interval plus
  the load time (well under the 60 s P2.2 criterion with the default of 5 s). A shared read-only
  models volume is required.
- **Dev databases stamped past the empty 0006–0010 stubs** (auto-migrated before the migrations were
  filled in) lack tables and columns although `alembic current` says head. `uv run python
  scripts/repair_dev_db.py` backs the file up, rebuilds the schema at head and copies every row back
  (it aborts without touching the file if any user row would be lost); then `alembic check` is clean.
- **Power of the gate** is bounded by the snapshot size. At 594 users the margins above give an
  equivalent candidate about an 80 % pass rate per gate. With fewer users (`max_users`, a small app
  snapshot) the SE grows and `power_if_equivalent` in `gate.json` drops; widen the margin, use B = 2000,
  or evaluate on more users.
- **`training.py` hooks** (owned by WS3, not edited here):
  1. `run_pipeline(activate=...)` should default to `False`.
  2. An optional `manifest_extra` passthrough on `run_pipeline` would let the lineage sit inside the
     manifest instead of `lineage.json`.
