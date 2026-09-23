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
uv run python scripts/train_models.py    # or --quick

# run (SQLite + in-memory cache, no Docker needed)
uv run uvicorn jev_api.main:app --reload --port 8000
cd frontend && pnpm dev                  # http://localhost:3000
```
Startup runs Alembic migrations (`JEV_AUTO_MIGRATE=true`), loads the active model, seeds the movie catalogue on first
run, syncs model versions and experiments, and creates the admin from `JEV_ADMIN_EMAIL`/`JEV_ADMIN_PASSWORD`.
To use local Postgres/Redis instead: `docker compose up -d db cache` and set `JEV_DATABASE_URL` / `JEV_REDIS_URL`
(the ports are not published by default; add them in a `docker-compose.override.yml`).

## Quality gates
```bash
uv run pytest                              # unit + ml + integration (≈10 s, synthetic data; real-model checks run if models/ exists)
uv run ruff check . && uv run ruff format --check .
uv run mypy                                # type check ML + API packages
cd frontend && pnpm exec tsc --noEmit && pnpm exec eslint src && pnpm build
uv run python scripts/acceptance_test.py --base http://localhost:3000/api   # against a running stack
uv run python scripts/capture_screenshots.py                                  # browser journey (needs Chrome)
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
