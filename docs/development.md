# Development

Requirements: **Python 3.11–3.14** (managed by [uv](https://docs.astral.sh/uv/)), **Node 20.9+** with pnpm (via
`corepack enable`). Docker is optional. Tested on Linux x86-64; everything is pure Python/JS, so macOS (Intel/Apple
silicon) and Windows (WSL2 or native) work the same way. A `--quick` training run peaked at 616 MB RSS in our measurement.

```bash
git clone … && cd JEV
cp .env.example .env                     # edit secrets
uv sync                                  # Python env (.venv)
cd frontend && pnpm install && cd ..

# data + models (one-off)
uv run python scripts/download_data.py   # add --skip-enrichment if offline
uv run python scripts/preprocess_data.py
uv run python scripts/train_models.py --calibrate   # or --quick; --calibrate also writes models/<v>/calibration.json
uv run python scripts/evaluate_intelligence.py      # optional: intelligence offline evaluation → experiments/intel-eval-*/

# run (SQLite + in-memory cache, no Docker needed)
uv run uvicorn jev_api.main:app --reload --port 8000
cd frontend && pnpm dev                  # http://localhost:3000
```
The frontend reads `JEV_API_URL` (default `http://127.0.0.1:8000`) when you run `pnpm build`/`pnpm dev`, not at
runtime, so rebuild after changing it. `JEV_HTTPS=true` (runtime) adds HSTS and `upgrade-insecure-requests`; leave it
unset on http://localhost. Every page gets a per-request nonce CSP from `src/proxy.ts`.

Startup runs Alembic migrations (`JEV_AUTO_MIGRATE=true`; 0001 initial, 0002 intelligence, 0003 normalised intel
tables + audit log, 0004 recommendation-feedback dedup), loads the active model, seeds the movie catalogue on first
run, syncs model versions and experiments, and creates the admin from `JEV_ADMIN_EMAIL`/`JEV_ADMIN_PASSWORD`.
To use local Postgres/Redis instead: `docker compose up -d db cache` and set `JEV_DATABASE_URL` / `JEV_REDIS_URL`
(the ports are not published by default; add them in a `docker-compose.override.yml`).

## Intelligence layer
The layer (`ml/jev_ml/intel`, `/intel/*`, the `/intel` console) needs no extra services. It reads three things:
- the processed MovieLens files (`JEV_PROCESSED_DIR`, default `data/processed`);
- the active model and its experiment run (`JEV_MODELS_DIR`, `JEV_EXPERIMENTS_DIR`);
- the app database.

Design and contract: [intelligence.md](intelligence.md).

| Variable | Default | Meaning |
|---|---|---|
| `JEV_PROCESSED_DIR` | `data/processed` | inputs of every run |
| `JEV_MODELS_DIR` / `JEV_EXPERIMENTS_DIR` | `models` / `experiments` | registry, model, `calibration.json`; experiment runs and `intel-eval-*` reports |
| `JEV_INTEL_RUN_ON_STARTUP` | `true` | background run on startup when the latest successful run is missing or stale |
| `JEV_INTEL_MIN_INTERVAL_HOURS` | `24` | staleness threshold for the startup run |
| `JEV_INTEL_SUPPRESS_DAYS` | `30` | quiet period of a dismissed warning key (escalation breaks it) |
| `JEV_INTEL_RUN_RATE_LIMIT_PER_MINUTE` | `10` | `POST /intel/runs` per admin per minute |

```bash
uv run python scripts/run_intelligence.py --as-of 2017-07-01   # one leak-free run, JSON to stdout (no DB)
uv run python scripts/evaluate_intelligence.py                 # forecast backtests, lapse holdout, shilling/anomaly/change-point studies, latency
uv run python scripts/calibrate_recommendations.py [--model V] [--dry-run]   # recommendation-confidence calibrator
```
**Calibration step.** `calibrate_recommendations.py` fits isotonic regression of P(rating ≥ 4 among the next 5 ratings)
on the validation split of the model's own experiment protocol. It evaluates on test and writes
`models/<version>/calibration.json` (19 s for the committed model). `train_models.py --calibrate` does the same for the version it has
just trained. The API reads the file when it loads the model, so after calibrating, restart the API or re-activate the
model.

## Quality gates
```bash
uv run pytest                              # 174 tests: unit 47, ml 9, intel 39, integration 79 (≈25 s; 1 skips without JEV_TEST_POSTGRES_URL; real-data checks skip without data/models)
uv run ruff check . && uv run ruff format --check .
uv run mypy                                # type check ML + API packages
cd frontend && pnpm test                   # Vitest + Testing Library (jsdom): 65 tests in 6 files, about 1 s
cd frontend && pnpm exec next typegen && pnpm exec tsc --noEmit && pnpm exec eslint src && pnpm build
cd frontend && pnpm audit                  # includes dev dependencies (Vitest); "No known vulnerabilities found" on 2026-09-24
uv run python scripts/acceptance_test.py --base http://localhost:3000/api   # against a running stack (58 checks)
uv run python scripts/capture_screenshots.py                                  # browser journey (needs Chrome); --only 22 23 24 saves a subset
```
Frontend tests live in `frontend/src/__tests__` (config: `frontend/vitest.config.mts`). They are not routes, and
`.dockerignore` keeps them out of the web image.

**Acceptance test.** `scripts/acceptance_test.py` needs a running API and web. It goes through the web proxy, so it
exercises Next.js → FastAPI → DB/cache → model. Part 1 is the member journey (registration, onboarding, recommendations
with reasons and `confidence`, feedback). Part 2 runs as the admin, against `--admin-email`/`--admin-password`:
- the 401/403 access rules;
- status, a run, and a replay (`--as-of`, default 2017-07-01, which must show the May-2017 Horror spike);
- every run-backed list;
- a score decision with its interval, and decision batches;
- a warning transition and its history row;
- evidence search, and history across the two runs;
- a saved scenario;
- decision and warning feedback, and the summary counts;
- evaluation and evaluation runs, recommender monitoring;
- the audit log (login, runs, transition, feedback and scenario rows, matched by request id) and `/admin/metrics`.

Each step prints `[PASS]`/`[FAIL]`, and the exit code is non-zero on any failure. `--skip-intel` runs part 1 only. It
is safe to re-run: the acknowledged warning is resolved at the end, so the next run reopens it.

A local stack for it:
```bash
JEV_DATABASE_URL=sqlite:////tmp/jev-accept.db JEV_ADMIN_EMAIL=admin@example.com JEV_ADMIN_PASSWORD=admin-pass-123 \
  JEV_JWT_SECRET=$(python -c 'import secrets;print(secrets.token_urlsafe(48))') uv run uvicorn jev_api.main:app --port 8000 &
(cd frontend && pnpm build && pnpm start -p 3000) &
uv run python scripts/acceptance_test.py --base http://localhost:3000/api
```

## Migrations
```bash
uv run alembic revision --autogenerate -m "describe change"
uv run alembic upgrade head
```
Migrations use batch mode, so they also run on SQLite.

## Conventions
- Training and inference are separate: never import `jev_ml.training` from `jev_api`.
- Any new interaction type gets its weights in `jev_ml/signals.py` and nowhere else.
- New explanation text must come from a `Contribution`; add a test in `tests/unit/test_hybrid.py`.
- The frontend talks only to `/api/*` through `src/lib/api.ts` (it adds the CSRF header).
