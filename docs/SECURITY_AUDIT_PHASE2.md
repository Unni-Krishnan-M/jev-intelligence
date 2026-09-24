# JEV Phase 2 security audit

_Application security review of the Phase 2 API (events, governance, online experiments, member
intelligence) and of the session lifecycle, 2026-09-24. It builds on the two earlier passes in
`SECURITY.md`. Regression tests: `tests/integration/test_security_phase2.py` (20 tests), plus the route
walkers in `tests/integration/test_security.py`._

## 1. Threat model

**Assets**
- Admin authority: `/admin/*`, `/intel/*`, `/governance/*` (model promotion and rollback),
  `/experiments*`, `/events/replay`.
- Model state: the active model, `models/registry.json`, dataset snapshots under `data/snapshots/`.
- Member data and member sessions.
- The event log and the generic-domain observations that feed live warnings and decisions.
- The audit log.
- Availability of sign-in for every member.

**Actors**
- Anonymous internet clients. Registration is open and emails are not verified.
- Signed-in members.
- Admins. They are trusted, but their browser session or token can be stolen.
- Machines: ingestion jobs and schedulers. They need to write observations and nothing else.
- The web proxy (Next.js). It is the API's only network peer in compose.

**Trust boundaries**
- browser → web (`/api/*` rewrite) → API → PostgreSQL/Redis. The API port is not published.
- Machine clients → API, with a bearer service token.
- Operator environment (`.env`) → API startup (`JEV_ADMIN_EMAIL`/`JEV_ADMIN_PASSWORD`, secrets).

**Out of scope**
- A compromised host or filesystem.
- A malicious admin.
- TLS termination (a deployment concern, see `SECURITY.md`).

## 2. Endpoint authorization matrix

Levels:
- **Public**: no credential.
- **Optional**: personalised when signed in.
- **Member**: any valid session.
- **Admin**: a session whose user has `is_admin`, re-read from the database on every request.
- **Admin or token**: an admin, or a service token with the matching scope.

Every cookie-authenticated non-GET request also needs `X-JEV-CSRF: 1`.

| Level | Routes |
|---|---|
| Public | `GET /health`, `GET /health/ml`, `POST /auth/register`, `POST /auth/login`, `GET /genres`, `GET /movies`, `GET /movies/search`, `GET /movies/popular`, `GET /recommendations/trending`, `GET /models/active/summary` |
| Optional | `POST /auth/logout` (revokes the presented session, if any), `GET /movies/{movie_id}`, `GET /recommendations/similar/{movie_id}` |
| Member | `POST /auth/logout-all`, `POST /auth/password`, `GET /users/me`, `POST /users/me/onboarding`, `PATCH /users/me/preferences`, `GET /users/me/{profile,ratings,history,favorites}`, `GET /me/intelligence`, `GET /me/intelligence/decisions/{decision_id}` (own decisions only), `POST /me/intelligence/scenarios` (plus a per-member limit), `POST\|GET /me/intelligence/feedback`, `POST\|DELETE /movies/{id}/rate`, `POST /movies/{id}/favorite`, `POST /movies/{id}/watch`, `GET /recommendations`, `GET /recommendations/because-you-watched`, `GET /recommendations/similar-to-favorites`, `POST /recommendations/feedback` (own recommendation ids only), `GET /recommendations/history`, `POST /events` (own events only: another `user_id` is a 403) |
| Admin | `POST /movies`, `GET /models`, `GET /models/{id}`, `POST /models/{id}/activate`, `GET /experiments`, `GET /experiments/{id}`, `GET /admin/{stats,metrics,audit}`, the other 31 `/intel/*` routes, `POST /events/replay`, `GET /events/health`, the 10 `/governance/*` routes, the 11 `/experiments/online*` routes, **new:** `POST\|GET /admin/service-tokens`, `DELETE /admin/service-tokens/{id}`, `POST /admin/users/{id}/revoke-sessions` |
| Admin or token | `POST /intel/domains/{key}/observations`: an admin, or a service token with `events:ingest:<key>` or `events:ingest:*` |

A service token is **not** a session. `get_optional_user` only accepts JWTs, so a `jevst_…` token gets
401 on every route except the observation endpoint. This is tested for `/admin/stats`, `/admin/audit`,
`/intel/runs`, `/admin/service-tokens`, `/users/me`, `/events` and `/events/replay`.

`test_every_intel_and_admin_route_requires_admin` walks every registered `/intel*` and `/admin*`
route. It fails when a route has neither `require_admin` nor `require_observation_writer`, which is
the one allowed exception and delegates to `require_admin` for users.
`test_admin_routes_reject_anonymous_and_members` sends each method anonymously (expects 401) and as a
member (expects 403).

## 3. Findings

| # | Severity | Finding | Status |
|---|---|---|---|
| F1 | **High** | `ensure_admin` promoted any existing account with the admin email | fixed |
| F2 | Medium | JWTs could not be revoked: logout, a password change or incident response did not end a session | fixed |
| F3 | Medium | No machine credential: ingestion needed a 24 h admin token | fixed |
| F4 | Medium | One login bucket for all clients behind the proxy, and no per-account brute-force limit | fixed (API; web proxy signing in `frontend/src/proxy.ts`; compose passes one `JEV_PROXY_SECRET` to both, v1.3.0) |
| F5 | Medium | No supply-chain checks in CI | fixed |
| F6 | Low | `GET /health/ml` (public) returns the model-load exception text | fixed in v1.3.0 (fixed message; detail in the log and admin metrics) |

### F1: account squatting becomes admin (High)

**Evidence.** In `services/sync.py`, `ensure_admin` ran on every startup:
`elif not user.is_admin: user.is_admin = True`. It did this whenever an account with
`JEV_ADMIN_EMAIL` existed, and never checked who owned that account.

**Exploit.** `POST /auth/register` is open and emails are not verified. An attacker registers
`admin@<company>` with their own password before the operator sets `JEV_ADMIN_EMAIL`. The same works
after the admin row is deleted, or when the operator changes the configured address to one the
attacker registered. On the next restart, the attacker's account becomes admin. The attacker can
then promote or roll back models, read the audit log, run the pipeline and view every member's
aggregates.

**Fix.**
- A missing account is still created, with `is_admin=True`.
- An existing member account is promoted only when `JEV_ADMIN_PASSWORD` verifies against its password
  hash, which proves the operator owns it. Otherwise startup logs an error and leaves the account a
  member.
- A promotion bumps `token_version`, so sessions issued before it end, and writes an audit entry
  `user.role_change` (`{"is_admin": true, "by": "ensure_admin"}`).

**Tests.**
- `test_ensure_admin_does_not_promote_a_squatted_account`
- `test_ensure_admin_promotes_only_with_the_configured_password`, which also checks the audit row and
  that the old session gets 401.

### F2: JWT lifecycle and revocation (Medium)

**Evidence.** Tokens carried `sub, adm, iat, exp, typ` and lived 24 h. `POST /auth/logout` only
deleted the cookie. A stolen token (from logs, a shared machine or an XSS on another origin) stayed
valid for a day. There was no password change at all.

**Fix.** The design combines a per-token denylist with a per-account version.
- Every JWT now has a random `jti` and `ver`, the account's `users.token_version` at issue time.
- `decode_access_token` requires both claims (tokens from before migration 0010 get 401). It also
  checks that `sub` is ASCII digits.
- `get_optional_user` rejects a token when `ver != user.token_version`, or when its `jti` is in
  `revoked_tokens`. That is one primary-key lookup, and the user was already loaded every request.
- `POST /auth/logout` denylists the presented `jti` until the token's own `exp`, clears the cookie and
  audits `auth.logout`. It needs the CSRF header when the caller is authenticated by cookie; the
  frontend already sends it.
- `POST /auth/logout-all` bumps `token_version` and audits `auth.revoke_all` (`by: self`).
- `POST /auth/password` (`current_password`, `new_password` 8–128) checks the current password. Wrong
  guesses count toward the per-account throttle (F4). It then rehashes, bumps `token_version`, audits
  `auth.password_change` and returns a fresh token, so only the caller's new session survives. The
  auth rate-limit bucket covers it.
- `POST /admin/users/{id}/revoke-sessions` is the incident-response tool for admins. It audits
  `auth.revoke_all` (`by: admin`).
- Migration 0010 adds `users.token_version` (NOT NULL, default 0) and `revoked_tokens` (`jti` PK,
  `user_id` FK with CASCADE, `revoked_at`, `expires_at`, both indexed).

**Tests.**
- `test_logout_revokes_the_token`
- `test_logout_revokes_only_that_session`
- `test_cookie_logout_needs_csrf_header`
- `test_logout_all_revokes_every_session`
- `test_password_change_revokes_other_sessions`
- `test_admin_can_end_a_members_sessions`
- `test_tokens_without_jti_or_version_are_rejected`, which forges tokens with a missing `jti`, a
  missing `ver` and a string `ver`.

### F3: service tokens for machine ingestion (Medium)

**Evidence.** The only credential was a user JWT. `require_observation_writer` accepted admins only,
so an ingestion job needed an admin password, or an admin token that has full console power and
lasts 24 h.

**Fix.**
- Token format: `jevst_` plus 32 random bytes (`secrets.token_urlsafe`).
- Only the SHA-256 is stored. A fast hash is enough for a 256-bit random secret. The table has a
  unique index on the hash, so the lookup is a single indexed read.
- Stored alongside: a 12-character display prefix, scopes, `expires_at` (1–365 days,
  `JEV_SECURITY_SERVICE_TOKEN_MAX_DAYS`), `revoked_at`/`revoked_by`, `last_used_at` and the creator.
- Scopes are validated against `^events:ingest:(\*|generic:<key>)$`. `admin`, `*`, `movie` and path-like
  values are 422.
- `POST /admin/service-tokens` returns the plaintext exactly once. `GET` lists metadata only, never the
  token or its hash. `DELETE /admin/service-tokens/{id}` revokes immediately.
- Audits: `token.create` (name, scopes, expiry; never the secret) and `token.revoke`.
- `require_observation_writer` (`routers/events.py`) now returns a `Writer`: an admin, or a token that
  has the path's scope. A wrong scope is 403. An unknown, revoked or expired token is 401.
- The audit actor for a token is `service-token:<id>:<name>`.
- The idempotency scope is `token:<id>`, separate from `user:<id>`. A token can never replay a stored
  response that belongs to an admin with the same numeric id.

**Tests.**
- `test_service_token_is_shown_once_and_stored_hashed`
- `test_service_token_posts_observations_but_reads_nothing_else`
- `test_service_token_scope_is_enforced` (exact scope and wildcard)
- `test_service_token_validation` (bad scopes, lifetimes, member → 403)
- `test_revoked_and_expired_service_tokens_are_rejected`
- `test_idempotency_scopes_of_tokens_and_users_cannot_collide`

### F4: login throttling behind the proxy (Medium)

**Evidence.**
- The auth limiter keyed on `request.client.host`. In compose that is always the web container,
  because Next.js rewrites forward `X-Forwarded-For` verbatim and uvicorn rightly trusts no one
  (`docker-compose.yml`, progress.md §10).
- The 20/min login bucket was therefore global. One client sending 20 bad logins a minute locked
  every member out.
- There was no per-account limit. Password guessing against one account was bounded only by that
  global bucket.

**Fix (API).**
1. **Per-account throttle.** `routers/auth.py` counts failed logins per email, stored as a SHA-256
   so the cache holds no addresses. After `JEV_SECURITY_LOGIN_MAX_FAILURES` (10) failures in
   `JEV_SECURITY_LOGIN_WINDOW_SECONDS` (900 s), further attempts get 429 with `Retry-After`, **before**
   the password is checked. That also saves the Argon2 CPU. Details:
   - It does not depend on the client address, so a shared or spoofed address changes nothing.
   - Unknown emails are throttled the same way, so the throttle does not reveal which accounts exist.
   - `auth.login.throttled` is audited once per account per window, so a flood cannot grow the audit
     table.
   - Wrong current passwords on `POST /auth/password` count too.
2. **Signed client address.** When `JEV_SECURITY_PROXY_SECRET` is set, the API accepts
   `X-JEV-Client: v1:<ip>:<unix ts>:<hmac>`, where the HMAC is HMAC-SHA256 over `v1|<ip>|<ts>`.
   - The header is used only if the HMAC verifies (constant-time compare), the timestamp is within
     `JEV_SECURITY_PROXY_MAX_SKEW_SECONDS` (60 s) and the IP parses. Otherwise the socket peer is used,
     so a forged header never opens a fresh bucket.
   - The verified address keys the API and auth rate limits and the audit `client`.
   - A backstop, `JEV_SECURITY_PROXY_RATE_LIMIT_PER_MINUTE` (6000/min), caps everything one proxy
     peer carries. A client that can choose its signed address (see residual risks) still cannot flood
     without bound.

**Frontend change needed.** See §5; it is not made here because `frontend/**` is owned by the
frontend engineer. Until it ships, the API behaves exactly as before for per-IP buckets. The
per-account throttle works now either way.

**Tests.**
- `test_per_account_throttle_blocks_even_the_right_password`: another account on the same address is
  unaffected, unknown emails are throttled too, and the audit row is written once.
- `test_per_account_throttle_ignores_forged_client_addresses`
- `test_signed_client_addresses_get_their_own_auth_bucket`: a flood from one signed client is
  throttled while another client signs in.
- `test_forged_or_stale_client_signatures_fall_back_to_the_peer`: wrong secret, stale timestamp,
  zero MAC, garbage, bare XFF.
- `test_signed_addresses_are_audited` (IPv6).
- The earlier XFF-spoofing tests in `test_security_review2.py` still pass.

### F5: supply chain (Medium, process)

**Fix.** `.github/workflows/ci.yml` now has two new steps:
- `python` job: `uv export --frozen` then `uvx pip-audit --disable-pip --no-deps --strict`.
- `web` job: `pnpm audit --prod --audit-level high`.

`.github/dependabot.yml` adds weekly updates for uv, npm and GitHub Actions.

**Results (2026-09-24).** pip-audit found no known vulnerabilities in the 83 locked packages.
`pnpm audit --prod` found no known vulnerabilities. No upgrades were needed.

### Areas reviewed without findings

| Area | Result |
|---|---|
| SQL injection | Only ORM queries and bound parameters. The `text()` uses are constants (`SELECT 1`, the events watermark `SELECT MAX(id)…`, partial-index predicates) or bound (`pg_try_advisory_lock(:k)`). The only user-chosen ordering is `/movies?sort=`, a closed `Literal` mapped to fixed columns. |
| Path handling | `/governance/models/{version}` must match `^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$` and exist in `registry.json`. The retrain `config` is a file *name* (`^…\.ya?ml$`), resolved inside `configs/` with a parent check. Job ids are UUID-shaped. Experiment keys and domain keys are pattern-bound (`DOMAIN_PATTERN`). Snapshot ids come from content hashes, never from requests. |
| Input bounds | Event batches are capped (500 member events, 5000 observations). Idempotency keys are pattern and length bound. Time skew and age are checked. NaN is rejected. Every list endpoint caps `limit` at 200 or 500. |
| Cookies | httpOnly, SameSite=Lax, `Secure` configurable (`JEV_COOKIE_SECURE`; set it with TLS). |
| CSRF | Unchanged and still enforced on every cookie-authenticated non-GET, including the new auth and admin routes. Service tokens are bearer-only, so no CSRF applies. |
| Secrets and logging | New secrets are `SecretStr`. The service token appears only in the creation response. Audit details never include it. The log redactor already masks `token`/`secret` keys and bearer values. |
| Error leakage | Unchanged generic 500 envelope. Service-token errors do not say whether a token exists, was revoked or has expired (all are "invalid service token"). |

## 4. Validation

| Check | Result |
|---|---|
| New tests | `tests/integration/test_security_phase2.py`, 20 tests. |
| Full suite and static checks | `uv run pytest -q`, `ruff check`, `ruff format --check`, `mypy`, `alembic check`. 459 passed, 8 skipped (PostgreSQL-only); ruff, format and mypy clean; `alembic check` shows no drift. |
| Migration 0010 | Round trip on SQLite (upgrade → downgrade 0009 → upgrade), then `alembic check`: no drift. It adds no partial index or trigger, so the chain's PostgreSQL CI job covers it. |
| Smoke test | Real uvicorn on port 8770 with a temp DB and admin env: a revoked token is rejected, a service token can post observations but cannot read admin data, and a login flood from different signed clients is limited per client. |

## 5. Frontend change (`frontend/src/proxy.ts`; shipped)

The change signs the client address on `/api/*` requests. It has shipped (`frontend/src/proxy.ts`,
`frontend/src/lib/client-sign.ts`, `frontend/src/__tests__/proxy.test.ts`). `docker-compose.yml` now sets
`JEV_SECURITY_PROXY_SECRET` on `api` and `JEV_PROXY_SECRET` on `web` from one `.env` variable,
`JEV_PROXY_SECRET` (`.env.example`); leaving it empty keeps the old single shared bucket. The design
notes below are kept for reference.

```ts
// frontend/src/proxy.ts: add near the top
const CLIENT_HEADER = "x-jev-client";

function clientIp(headers: Headers): string | null {
  // an edge proxy that OVERWRITES this header (e.g. nginx `proxy_set_header X-Real-IP $remote_addr`) is
  // authoritative; otherwise the right-most X-Forwarded-For hop (Next sets it from the socket when absent)
  const edge = process.env.JEV_CLIENT_IP_HEADER;
  const raw = edge ? headers.get(edge) : headers.get("x-forwarded-for")?.split(",").pop();
  const ip = raw?.trim() ?? "";
  return ip && ip.length <= 45 ? ip : null;
}

async function signClient(ip: string, secret: string): Promise<string> {
  const ts = Math.floor(Date.now() / 1000);
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(`v1|${ip}|${ts}`));
  const hex = Array.from(new Uint8Array(mac), (b) => b.toString(16).padStart(2, "0")).join("");
  return `v1:${ip}:${ts}:${hex}`;
}

// make proxy async and, as its FIRST statement:
export async function proxy(request: NextRequest) {
  if (request.nextUrl.pathname.startsWith("/api/")) {
    const headers = new Headers(request.headers);
    headers.delete(CLIENT_HEADER); // never forward a client-supplied value
    const secret = process.env.JEV_PROXY_SECRET;
    const ip = secret ? clientIp(request.headers) : null;
    if (secret && ip) headers.set(CLIENT_HEADER, await signClient(ip, secret));
    return NextResponse.next({ request: { headers } });
  }
  // ... existing page logic unchanged ...
}

export const config = {
  // pages AND the API rewrite; not build assets or the favicon
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
```

A unit test should cover four cases:
- a client-sent `x-jev-client` is dropped;
- the signed value verifies with the Python `verify_client_address` format;
- no secret means no header;
- `/api/*` gets no CSP rewrite.

In compose:
- add `JEV_SECURITY_PROXY_SECRET: ${JEV_PROXY_SECRET:-}` to `x-api-env`;
- add `JEV_PROXY_SECRET: ${JEV_PROXY_SECRET:-}` to `web.environment`.

## 6. Residual risks

- **Client-chosen signed address without an edge proxy.** When the browser connects straight to
  Next.js, the right-most `X-Forwarded-For` hop is whatever the client sent, because Next only sets
  the header when it is absent. The web proxy would sign that value, so a client can spread its
  requests over many per-IP buckets. These controls still hold:
  - the per-account throttle (unaffected);
  - the 6000/min proxy backstop;
  - the fact that the API cannot be reached without the secret.

  For true per-client limits, put an edge proxy in front of web that overwrites `X-Real-IP`, and set
  `JEV_CLIENT_IP_HEADER=x-real-ip`.
- **Targeted lockout.** Anyone can hold a known account in the 429 state by sending 10 bad passwords
  every 15 minutes. This is the usual trade-off of per-account throttling. The owner's existing
  sessions keep working. An admin can raise the limit, and the audit shows the attempts.
- **Memory-cache counters** (no Redis) are per process and evict under pressure (5000 keys). An
  attacker who fills the cache can reset counters early. Use Redis in shared deployments, as compose
  does.
- **`revoked_tokens` is not pruned.** Rows are useless after `expires_at` (at most 24 h) and are
  indexed for a future `retention.prune` job. Growth is one row per logout.
- **Existing sessions end at upgrade.** Tokens without `jti`/`ver` get 401 after migration 0010, so
  every member signs in once more.
- **Service tokens are bearer secrets.** Whoever holds one can post observations for its domains until
  it expires or is revoked. Observations drive live warnings, so scope tokens per domain, keep
  lifetimes short and watch `last_used_at`. There is no IP allow-list per token.
- **F6 (fixed in v1.3.0).** `GET /health/ml` returned `holder.last_error`, including the exception text
  of a failed model load. It now returns a fixed message; the detail stays in the log and in the
  admin-only `/admin/metrics` (`test_public_health_ml_hides_the_model_load_error`).
- **No admin role-change endpoint.** Promotion happens only through `ensure_admin`, and demotion only
  through the database. `is_admin` is re-read on every request, so a demotion takes effect
  immediately. A future role endpoint must bump `token_version` and audit `user.role_change`.
- **Carried over from `SECURITY.md`:** `style-src 'unsafe-inline'`, no email verification or password
  reset, and a per-process run lock.
