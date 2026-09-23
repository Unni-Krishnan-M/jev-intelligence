# JEV Progress

_Last updated: 2026-09-23_

## Completed
- Phase 1 — Environment audit: Linux x86_64, 28 cores / 62 GB RAM (dev box; target is *any* laptop),
  Python 3.14 system + 3.13 via uv, Node 26, pnpm, Docker 29 + Compose v5.
  No local PostgreSQL/Redis → provided by Docker Compose. Empty repository.

## In progress
- Phase 2 — Architecture and project structure

## Blocked
- (none)

## Next
- Phase 3 — Dependencies
- Phase 4 — Dataset

## Design constraints decided in Phase 1
- **Runs on any laptop**: MovieLens `ml-latest-small` (100k ratings) so the full pipeline trains in
  well under a minute on CPU; no GPU/torch dependency; numpy/scipy/sklearn only.
- **Portable dev mode**: backend works with SQLite + in-memory cache when Docker is not available;
  PostgreSQL + Redis via Docker Compose for the full stack.
- **No pickles**: model artifacts are `.npz` / `.json` so they are portable and safe to load.
