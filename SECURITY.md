# Security

## Controls in place
| Area | Implementation |
|---|---|
| Secrets | Only from environment / `.env` (git-ignored); `.env.example` has placeholders. No secret in code. Production refuses to start without `JEV_JWT_SECRET`; development uses an ephemeral random secret and logs a warning. |
| Passwords | Argon2id (`argon2-cffi`, library defaults) with transparent rehash; 8–128 chars; login does a dummy verify for unknown emails so response time doesn't reveal which accounts exist; identical error for unknown email and bad password. |
| Sessions | HS256 JWT (`sub`, `adm`, `iat`, `exp`, `typ`), 24 h. Delivered as an httpOnly SameSite=Lax cookie (`Secure` configurable) and in the body for API clients. |
| CSRF | Cookie-authenticated non-GET requests require the `X-JEV-CSRF: 1` header. Cross-site forms cannot set it, and CORS blocks it from other origins. |
| Authorization | Admin-only routers (`/models*`, `/experiments*`, `/admin/*`, `POST /movies`); users can only reference their own recommendation ids in feedback. |
| Input validation | Pydantic schemas with bounds (rating grid, lengths, list sizes, regex for `context`); query parameter limits on pagination. |
| SQL injection | SQLAlchemy ORM / bound parameters everywhere; the search escapes LIKE wildcards. Covered by tests. |
| CORS | Explicit origin allow-list, restricted methods and headers. |
| Rate limiting | Fixed-window per client IP (Redis, in-memory fallback): 240/min API, 20/min auth. `X-Forwarded-For` honoured only with `JEV_TRUST_PROXY`. |
| Errors | Uniform `{detail, request_id}`; unhandled exceptions are logged server-side with traceback and returned as a generic 500. |
| Logging | JSON logs with request id; keys matching password/secret/token/authorization/cookie/api_key are redacted, as are bearer tokens, JWT-shaped strings and credentials in URLs. |
| Headers | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`; `Cache-Control: no-store` on API responses; `X-Powered-By` removed. |
| Artifacts | No pickle/joblib: model files are `.npz` / JSON, so loading a model cannot execute code. The dataset archive is MD5-checked, and zip members are path-validated before extraction. |
| Containers | Non-root users, slim/alpine bases, health checks, no ports for db/cache on the host. |
| Open redirect | The login `next` parameter accepts only same-site paths (`/…`, not `//…`). |

## Review notes (phase 22)
- Reviewed routes for authorization gaps: every state-changing route requires a user; admin routes require
  `is_admin`. `GET /movies/{id}` personalisation fields are only filled for the authenticated user.
- The proxy guard (`frontend/src/proxy.ts`) is an *optimistic* redirect only. Authorization is always enforced by the API.
- Known limitations: no email verification or password reset flow; JWTs are not revocable before expiry (logout clears
  the cookie); the rate limiter is per-IP and approximate (fixed window); the in-memory cache is per process.

## Reporting
Please report vulnerabilities privately to the maintainers rather than opening a public issue.
