# JEV decision engine

_Phase 2, WS4a. Covers `ml/jev_ml/core/{decisions,batches,early_warning,lineage}.py`, the movie
decisions (`ml/jev_ml/domains/movie/decisions.py`) and the API around them
(`backend/jev_api/services/intel.py`, `routers/intel.py`). Result contract: [intelligence.md](intelligence.md) §4 and §9.1;
platform contract: [platform.md](platform.md)._

JEV decisions are **bounded, typed and versioned rules** over evidence that ordinary code computes
first. No LLM is involved; every answer can be recomputed from the recorded state.

## 1. Decision specs

A `DecisionSpec` declares a question once:

| Field | Meaning |
|---|---|
| `key` | stable question id, e.g. `early_warning_level`, `retrain_model` |
| `policy_version` | the version of the rule that answers it (`ewl-1.1.0`, …). A change in behaviour bumps it. |
| `kind` | `boolean`, `choice` (a fixed option set) or `score` (a number, optionally with `answer_interval`) |
| `options` | the allowed answers (`choice`, `boolean`) |
| `confidence_kind` | what the confidence number means (§3). It is fixed per spec. |

`spec.decide(...)` builds the wire record: `id` (= `stable_id("dec", key, entity, as_of)`), `answer`,
`option_scores`, `confidence`, `confidence_kind`, `state` (the inputs the rule read), `rationale`,
`evidence`, `policy_version`, `batch_id`. `spec.abstain(reason, …)` records `answer: null`,
`abstained: true` and `fallback_reason`.

### Decisions in the product

| Key | Domain | Kind / options | Policy | Confidence kind |
|---|---|---|---|---|
| `early_warning_level` | every domain | choice: NO_ACTION, MONITOR, WARNING, URGENT_ACTION | `ewl-1.1.0` | `margin` |
| `retrain_model` | movie | boolean | `retrain-1.0.0` | `rule` |
| `serving_model` | movie | choice over evaluated models | `serving-1.0.0` | `probability` (paired bootstrap P(best > runner-up)) |
| `genre_programming` | movie | choice: promote / keep / demote per genre | `genre-1.0.0` | `margin` |
| `editorial_slot_share` | movie | score: slot share % with an interval | `slot-share-1.0.0` | `interval` (nominal coverage) |
| `rater_action` | movie | choice per flagged rater | `rater-1.0.0` | `margin` |
| `reengagement_campaign` | movie | boolean | `reengage-1.0.0` | `probability` (Poisson-binomial tail) |
| `recommendation_strategy` | movie (member) | choice: standard / adapt_to_recent / explore | `strategy-1.0.0` | `margin` |

Versions: `domains/movie/decisions.py` (`POLICY_VERSIONS`), `core/config.py` (`EWL_POLICY_VERSION`), `domains/movie/user_intel.py`.

## 2. Batches: one snapshot per multi-question call

Questions that must agree are answered together in a **batch** (`core/batches.run_batch`):
- the inputs are deep-copied once and hashed (`state_hash`, sha1 over a canonical encoding);
- every question reads that same snapshot;
- the snapshot is re-hashed after the last question.

If a policy raised an error or mutated the snapshot, **none** of the batch's answers is kept: each
question records an abstention and the batch reports `status: "failed"`.

Batches in a movie run: `model_governance`, `genre_programming`, `audience`, `early_warning`. Every
generic run has one batch, `early_warning`.

## 3. Confidence kinds

Every confidence the engine emits carries an explicit `confidence_kind` (tested:
`tests/core/test_decision_quality.py::test_every_emitted_confidence_has_a_kind`). None of them is a
probability that the answer is right, unless the kind says `probability`.

| Kind | Definition | Where |
|---|---|---|
| `probability` | from a probabilistic computation (bootstrap, calibrated model) | `serving_model`, `reengagement_campaign`, one movie signal |
| `margin` | normalised distance from a decision boundary, (s1 − s2)/(s1 + s2) over option scores, or a normalised detector strength | `early_warning_level`, anomalies, forecasts' skill signals, anomaly and situation warnings |
| `rule` | a deterministic threshold rule (1.0 when it fires) | `retrain_model`; `data_quality`, `data_staleness` and `model_staleness` risks |
| `interval` | the nominal coverage of `answer_interval` | `editorial_slot_share` |
| `evidence` | 1 − (adjusted) p-value, or a coverage × (1 − q) strength of support | trends, trend/change-point signals, trend and anomaly risks |

**Changes in core-1.1.0:**
- Risks had a `confidence` factor with no kind. They now carry `confidence_kind` from
  `core.risk.RISK_CONFIDENCE_KINDS`; the movie `model_staleness` risk declares `rule` itself.
- A warning raised from a risk used to be labelled `margin` whatever the risk measured. It now
  carries the risk's own kind.

The DB `ck_intel_decision_confidence_kind` allows all five kinds (migration 0006).

**Base-rate context.** Wherever precision is reported, the engine also reports the base rate, the lift
(precision / base rate) and the flag rate (`core.evaluation.confusion`). A precision of 0.36 means
nothing on its own: at a base rate of 0.30 it is only 1.2× better than flagging units at random.

## 4. Abstention

A decision abstains, rather than guessing, when:
- **the evidence cannot answer**: every item of an early-warning situation is a skipped stage (too
  little history), there is too little feedback to test, or too few users;
- **it would leak**: in a replay whose as_of predates the active model's training cutoff, the
  model-governance decisions abstain with "model was trained on data after as_of";
- **the batch failed** (§2).

Abstentions are persisted like answers (`abstained: true`, `fallback_reason`). They are counted in
`summary.counts.decisions_abstained` and in the `early_warning` counts.

## 5. The early-warning policy (`ewl-1.1.0`)

Evidence is grouped into **situations**: `series:<id>`, or `entity:<type>:<entity>` for evidence not
about a series. Each evidence item earns points:
- a risk or anomaly ("component") earns 1/3/5/7 points at the low/medium/high/critical band edges,
  interpolated inside each band;
- an adverse trend, change point or forecast is context and earns 1–2.5 points.

The aggregate is P = max + corroboration × (independent agreeing stages − 1). The answer is the level
whose threshold P reaches: MONITOR at 1, WARNING at 3, URGENT_ACTION at 5.

**1.1.0 change (recovery-aware).** A trend whose series is already *reversing* earns 0 points, and so
does a change point in the trend's direction. Reversing means the latest value has fallen back from
the window's extreme by more than 2σ√3 (σ = the MAD of period changes; `trend.recent_move`). No
`adverse_trend` risk is raised for it either. The rationale and the component's `detail` say
"reversing". Measured effect: [INTELLIGENCE_ENGINE_AUDIT.md](INTELLIGENCE_ENGINE_AUDIT.md) §4.

## 6. Evidence lineage

Every decision and every warning resolves to an **evidence bundle**:

```
decision ─evidence→ run objects (signal · trend · anomaly · forecast · risk · decision)
                        └─about→ series ─built_from→ data source ─read_by→ run
                                                                   (mode, versions, config_hash,
                                                                    input_fingerprint, event_watermark)
warning ─decided_by→ its early_warning_level decision ─→ …  (plus ─source→ the risk/anomaly it came from)
```

- **Pure core** (`jev_ml.core.lineage`): `decision_lineage(result, decision_id)` and
  `warning_lineage(result, key)`. Refs are taken from `evidence[].ref` and, for early-warning
  decisions, from `state.components[].ref`. They are followed transitively, up to depth 8.
- **Series and sources.** An object with a `series_id` links to that series. A series links to the
  run's primary source (the first `data.sources` row). An object with a `source` field links to that
  source directly.
- **Run-level decisions.** A decision with no object refs (`retrain_model`, `serving_model`,
  `reengagement_campaign`, an abstention) links to every source of the run, with
  `relation: "derived_from_run"`. Its basis is the run's inputs as a whole, and the graph says so.
- **Completeness.** `complete` is true when every non-null ref resolves inside the run and at least
  one source is reached. For a warning, its decision must also resolve. `unresolved` lists what did
  not resolve.
- **Run node.** The API adds the run: `mode`, `pipeline_version`, `core_version`, `data_version`,
  `model_version`, `config_hash` (sha1 of `run.config`), `input_fingerprint` (sha1 of the data
  version, as_of and per-source rows / first / last event), and WS1's `event_watermark`
  (`max_event_id`, `max_ingested_at`, `max_event_time`, `n_events`, `as_of`, `knowledge_time`; null
  for runs that predate the event log).

**Endpoints** (admin, domain-scoped; a decision or warning of another domain is a 404):
- `GET /intel/decisions/{db_id | dec-id}/lineage[?run_id=]`: a contract id resolves to the copy in
  `run_id`, else to the newest run's copy.
- `GET /intel/warnings/{id}/lineage[?run_id=]`: resolved in the run that last saw the warning, or in
  `run_id`. The root carries `decision_id` and `decision_db_id`.

Both return `IntelLineage`: `version`, `root`, `run`, `evidence` (the root's items with
`resolves_to`), `nodes`, `edges` (`from`, `to`, `relation`), `series`, `sources`, `node_types`,
`unresolved`, `complete`, `event_watermark`.

**Tests.**
- Every decision and warning of synthetic movie and unemployment runs, and of **real** MovieLens and
  FRED runs (live, and replays at 2017-07-01 and 2008-06-01), has a complete lineage:
  `tests/core/test_decision_quality.py`.
- The same holds through the API for every decision and open warning of a live movie and a live
  generic run: `tests/integration/test_intel_replay_lineage.py`.
- The smoke test on real data resolved 55/55 movie and 9/9 unemployment decisions, and 1/1 and 6/6
  warnings.

No migration was needed. The hashes live in the run's result JSON (`result.run`), and the watermark
is WS1's `intel_runs.event_watermark` (0007).

## 7. Determinism

The same as-of state gives the same decisions: ids, answers, confidences, policy versions, abstentions
and batch `state_hash`es. More strongly, the whole result is identical apart from timings and `now`.
- **Core.** `test_same_as_of_state_gives_identical_decisions` covers the movie full / as-of variants
  and an unemployment replay. `test_replay_decisions_do_not_depend_on_the_wall_clock` runs a replay
  with a different `now` and gets the same decisions.
- **API.** `test_same_as_of_gives_identical_decisions_via_api` compares:
  - two movie replays at the same as_of (different run ids, identical decisions and state hashes,
    equal `config_hash` and `input_fingerprint`);
  - two live generic runs on the same day.
- **Why it holds.**
  - Every random draw is seeded per object (`sub_seed(seed, object key)`).
  - Ids are `stable_id`s of (kind, entity, as_of).
  - A replay ignores live operator suppression (§8), so what operators dismissed later cannot change
    a replay.
  - A replay's app events are cut at as_of, and at `knowledge_time` for ingestion (WS1).

## 8. Replay versus live

A run with an explicit `as_of` is a **replay** (`run.mode = "replay"`, migration 0006). A replay:
- never creates, updates, reopens or resolves a live warning (its warnings are in its result only:
  `GET /intel/runs/{id}/warnings`);
- is never the default "latest" (§9 of [EARLY_WARNING_SYSTEM.md](EARLY_WARNING_SYSTEM.md));
- ignores live operator suppression;
- sees app events only up to as_of, and runs live-window stages (live feedback, app freshness) on the
  replay clock (`domains/movie/ingest.py`, `raters.py`);
- accepts `knowledge_time` (`POST /intel/runs {"as_of", "knowledge_time"}`): events ingested after it
  are unknown to the run. It defaults to as_of, is rejected without an as_of, and is rejected in the
  future.
