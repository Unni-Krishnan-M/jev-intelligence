# Deployment

## Docker Compose (recommended)
```bash
cp .env.example .env          # set POSTGRES_PASSWORD, JEV_JWT_SECRET, JEV_ADMIN_*
docker compose --profile train run --rm trainer   # first run only: download, preprocess, train (writes ./data ./models ./experiments)
docker compose up -d --build                      # db, cache, api, web
open http://localhost:3000
```
Services: `db` (postgres:17-alpine, volume `pgdata`), `cache` (redis:7-alpine, 128 MB LRU, no persistence), `api`
(FastAPI, non-root, health-checked, migrates on start), `web` (Next.js standalone, non-root), `trainer` (profile
`train`; `TRAIN_ARGS=--quick` for a fast run). Model artifacts are bind-mounted, so training on the host and serving
in Docker (or the reverse) share one registry.

Resource use on a laptop: about 350 MB RAM for api + web + db + cache at idle; training peaks around 1.5 GB.

## Production checklist
- `JEV_ENV=production`: disables `/docs` and makes `JEV_JWT_SECRET` mandatory.
- Serve over HTTPS behind a reverse proxy; set `JEV_COOKIE_SECURE=true`, `JEV_CORS_ORIGINS=https://your.domain`,
  and keep `JEV_TRUST_PROXY=true` only when the proxy sets `X-Forwarded-For`.
- Use a managed Postgres with backups; Redis can be ephemeral (cache + rate-limit counters only).
- Run several API workers (`uvicorn --workers N`): each loads the model (about 60 MB RSS). Rate limits need Redis
  when there is more than one worker.
- Retrain by running the trainer, then activate the new version from `/admin/models` (or `models/registry.json`).
  The API swaps it in without a restart.
