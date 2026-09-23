# JEV web app

Next.js 16 (App Router), React 19, TypeScript, Tailwind v4, shadcn/ui, Recharts and SWR. Every call
goes to `/api/*`, which `next.config.ts` proxies to the FastAPI backend (`JEV_API_URL`, default
`http://127.0.0.1:8000`). `JEV_API_URL` is read when you run `pnpm build`, not at runtime. The Docker image is
built with `http://api:8000`.

Set `JEV_HTTPS=true` (at runtime) only when the site is served over HTTPS. `src/proxy.ts` then adds
`Strict-Transport-Security`, and the CSP adds `upgrade-insecure-requests`. Leave it unset for local http.

```bash
pnpm install
pnpm dev                      # http://localhost:3000
pnpm exec next typegen && pnpm exec tsc --noEmit
pnpm lint
pnpm test                     # Vitest + Testing Library (jsdom), tests in src/__tests__
pnpm build && pnpm start
```

The version shown in the footer comes from `src/lib/version.ts`; keep it equal to `package.json`
(a unit test checks).

## Intelligence console (`/intel`)

Admin-only views over the intelligence layer (contract: `docs/intelligence.md`, sections 4, 6
and 9). v1.1 adds score decisions and decision batches, run-history timelines on signals, risks
and trends, the evidence explorer (`/intel/evidence`), recommender monitoring
(`/intel/recommendations`), the audit log (`/intel/audit`) and the evaluation-run history. When
the API answering predates an endpoint, the view says "not available on this API version" rather
than failing.

## Design references

Structure only; the visual style stays JEV's (serif display, mono data, one amber accent,
hairlines).

- **Evidence explorer**: Grafana Explore's logs view
  (<https://grafana.com/docs/grafana/latest/explore/logs-integration/>). A line-filter search
  above the results, rows that carry their fields, and "filter on this value" from inside a row
  (the owner-type chip on each row sets the owner filter).
- **Audit log**: common audit-trail viewer patterns, for example
  <https://dev.to/dangtony98/guide-to-building-audit-logs-for-application-software-49fh>. Actor
  and action filters in one row above a table of time (UTC), action, actor, target and request id.
  Failed actions are set apart by icon and word, the raw detail sits one click away, and a filter
  that matches nothing gets its own empty state. Clicking an actor filters by that actor.
- **Calibration**: scikit-learn's probability calibration guide
  (<https://scikit-learn.org/stable/modules/calibration.html>) and
  <https://github.com/hollance/reliability-diagrams>. A reliability diagram against y = x with a
  per-bin table, ECE and Brier beside it (Brier next to the base-rate forecaster for scale), and
  the histogram of served confidences below, so sparse bins show. Without a calibration file the
  page says "Not calibrated" and draws nothing.
