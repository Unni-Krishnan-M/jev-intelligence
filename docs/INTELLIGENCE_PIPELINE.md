# Intelligence pipeline

One page on how a JEV run turns data into decisions and warnings. It links to the detailed documents
instead of repeating them.

```
 adapter.load(as_of)                      ml/jev_ml/core/pipeline.py: run_domain(adapter, as_of, now)
 ─────────────────                        ────────────────────────────────────────────────────────────
 observations + series specs ─► VALIDATE ─► SERIES ─► SIGNALS ─► TRENDS / CHANGE POINTS ─► ANOMALIES
   (movie: MovieLens + app       quality,     monthly,   dedup,     Mann–Kendall + Theil–Sen,   robust z
    event log cut at as_of;      freshness    weekly or  strength   BH FDR; AR(1)-null change   (seasonal
    generic: CSV + YAML)                      daily                  points, reversal state       basis)
                                                                                               │
 ◄──────────────────────────────────────────────────────────────────────────────────────────────┘
 FORECASTS ─► RISK ─► DECISIONS ─► EARLY WARNINGS ─► ACTIONS ─► RESULT (+ lineage ids)
 rolling-     likelihood  typed, versioned   only downstream of     plan with     run.{pipeline_version,
 origin       × impact    policies with      early_warning_level    effort        config_hash,
 selection,   × confidence confidence kind    = WARNING | URGENT                  input_fingerprint,
 conformal    × quality    and abstention                                          model_version}
 80 % band
                                        │ persisted by backend/jev_api/services/intel.py
                                        ▼
        live run: warnings upserted (dedup key, lifecycle, auto-resolve after K live runs)
        replay:   warnings stay in the run's result; live state untouched
```

| Stage | What it does | Detail |
|---|---|---|
| Adapter | Turns a domain into observations and `SeriesSpec`s; movie adds raters, lapse, model governance, preference drift | [platform.md §3](platform.md), [SECOND_DOMAIN_CASE_STUDY.md](SECOND_DOMAIN_CASE_STUDY.md) |
| Validation, freshness | Weighted quality checks, publication lag, SLA | [intelligence.md §2](intelligence.md) |
| Trends, change points, anomalies | Mann–Kendall/Theil–Sen with BH FDR; AR(1) null with φ from history (≈ 1.3 % false alarms at nominal 1 %); robust z | [INTELLIGENCE_ENGINE_AUDIT.md §2, §4.4](INTELLIGENCE_ENGINE_AUDIT.md) |
| Forecasts | Naive, moving average, damped Holt; seasonal models for daily data; chosen by rolling-origin backtest | [SECOND_DOMAIN_CASE_STUDY.md §5.1](SECOND_DOMAIN_CASE_STUDY.md) |
| Risk | Score = 100 · likelihood · impact · confidence · data quality, with factors | [intelligence.md §4](intelligence.md) |
| Decisions | Bounded, typed, versioned rules; confidence kinds `probability`, `margin`, `interval`, `evidence`, `rule`; explicit abstention; atomic batches | [DECISION_ENGINE.md](DECISION_ENGINE.md) |
| Early warnings | `early_warning_level` gates every warning; lifecycle new → acknowledged → investigating → resolved/dismissed, with audit | [EARLY_WARNING_SYSTEM.md](EARLY_WARNING_SYSTEM.md) |
| Lineage | decision → evidence → run objects → series → data sources, resolved per decision and warning | [DECISION_ENGINE.md](DECISION_ENGINE.md), `ml/jev_ml/core/lineage.py` |
| Replay | `as_of` cut on event time and knowledge time; deterministic for a fixed code version, data files and active model | [STREAMING_ARCHITECTURE.md §4](STREAMING_ARCHITECTURE.md) |
| Evaluation | Leak-free monthly replays; precision, FPR, recall, base rate, lift with a month-cluster bootstrap CI | [EARLY_WARNING_SYSTEM.md §4](EARLY_WARNING_SYSTEM.md), [EVALUATION_AND_VALIDITY_REVIEW.md](EVALUATION_AND_VALIDITY_REVIEW.md) |

## Run it

```bash
uv run python scripts/run_intelligence.py                                   # movie, latest data, JSON to stdout
uv run python scripts/run_intelligence.py --domain generic:us-unemployment --as-of 2008-06-01
uv run python scripts/run_intelligence.py --domain generic:cta-ridership --as-of 2020-03-20
uv run python scripts/evaluate_domains.py                                   # all domains, ~6 min → experiments/platform-eval-*/
```

Through the API (admin): `POST /intel/runs` with an optional `as_of` (a replay) and `?domain=`; reads default to
the latest live run (`?run_id=`, `?mode=replay|any` select others). See [api.md](api.md).
