"""End-to-end acceptance scenario against a *running* stack (local or Docker Compose).

    uv run python scripts/acceptance_test.py --base http://localhost:3000/api \
        --admin-email admin@example.com --admin-password ...

Goes through the web proxy by default, so it exercises Next.js → FastAPI → Postgres/Redis → model.
Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import requests

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{f' — {detail}' if detail else ''}")
    return ok


class Api:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")
        self.s = requests.Session()
        self.token: str | None = None

    def call(self, method: str, path: str, **kw: Any) -> requests.Response:
        headers = kw.pop("headers", {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return self.s.request(method, f"{self.base}{path}", headers=headers, timeout=30, **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:3000/api")
    ap.add_argument("--admin-email", default="admin@example.com")
    ap.add_argument("--admin-password", default="admin-pass-123")
    args = ap.parse_args()

    api = Api(args.base)
    h = api.call("GET", "/health")
    check(
        "API healthy through proxy (db + cache)",
        h.ok and h.json()["status"] == "ok",
        str(h.json().get("cache")),
    )
    ml = api.call("GET", "/health/ml").json()
    check("model loaded + self-check", ml.get("status") == "ok", ml.get("model_version", ml.get("error", "")))

    stamp = int(time.time())
    reg = api.call(
        "POST",
        "/auth/register",
        json={
            "email": f"accept{stamp}@example.com",
            "password": "acceptance-pass",
            "display_name": "Acceptance",
        },
    )
    check("register new user", reg.status_code == 201)
    api.token = reg.json()["access_token"]
    me = api.call("GET", "/users/me").json()
    check("authenticated /users/me", me["email"] == f"accept{stamp}@example.com")

    ob = api.call(
        "POST", "/users/me/onboarding", json={"genres": ["Sci-Fi", "Thriller"], "movie_ids": [2571]}
    )
    check(
        "complete onboarding with genres + favourite",
        ob.ok and ob.json()["favorite_genres"] == ["Sci-Fi", "Thriller"],
    )
    cold = api.call("GET", "/recommendations?limit=10").json()
    check(
        "cold-start user gets recommendations",
        len(cold["items"]) == 10,
        f"weights {cold['effective_weights']}",
    )

    rated = {79132: 5.0, 109487: 5.0, 58559: 4.5, 1: 2.0, 318: 4.0}
    for mid, val in rated.items():
        api.call("POST", f"/movies/{mid}/rate", json={"rating": val}).raise_for_status()
    api.call("POST", "/movies/2959/watch").raise_for_status()
    consumed = set(rated) | {2571, 2959}

    recs = api.call("GET", "/recommendations?limit=20").json()
    items = recs["items"]
    ids = [i["movie_id"] for i in items]
    check("personalized recommendations returned", len(items) == 20, recs["model_version"])
    check("previously consumed movies excluded", not set(ids) & consumed)
    check(
        "every item has a reason and six signals", all(i["reason"] and len(i["signals"]) == 6 for i in items)
    )
    check(
        "scores finite and ranked",
        all(0 <= i["score"] <= 1.0001 for i in items) and [i["rank"] for i in items] == list(range(1, 21)),
    )
    grounded = all(set(i["anchor_movie_ids"]) <= consumed for i in items)
    check("explanations cite only the user's own movies", grounded)
    check(
        "personalized list differs from cold-start list", ids[:10] != [i["movie_id"] for i in cold["items"]]
    )

    # a second user with opposite taste must get a different list (not hard-coded)
    other = Api(args.base)
    r2 = other.call(
        "POST",
        "/auth/register",
        json={"email": f"kids{stamp}@example.com", "password": "acceptance-pass", "display_name": "Kids"},
    )
    other.token = r2.json()["access_token"]
    other.call("POST", "/users/me/onboarding", json={"genres": ["Animation", "Children"], "movie_ids": [1]})
    for mid in (2355, 6377, 4886):
        other.call("POST", f"/movies/{mid}/rate", json={"rating": 5.0}).raise_for_status()
    ids2 = [i["movie_id"] for i in other.call("GET", "/recommendations?limit=20").json()["items"]]
    overlap = len(set(ids) & set(ids2))
    check(
        "recommendations are not hard-coded (different tastes → different lists)",
        overlap <= 5,
        f"overlap {overlap}/20",
    )

    movie = api.call("GET", f"/movies/{ids[0]}").json()
    check("open a movie", movie["id"] == ids[0], movie["title"])
    sim = api.call("GET", f"/recommendations/similar/{ids[0]}?limit=8").json()
    check(
        "similar movies returned",
        len(sim["items"]) >= 5 and all(s["movie_id"] != ids[0] for s in sim["items"]),
    )

    target = items[0]
    fb = api.call(
        "POST",
        "/recommendations/feedback",
        json={
            "movie_id": target["movie_id"],
            "feedback": "not_interested",
            "recommendation_id": target["recommendation_id"],
        },
    )
    check("submit feedback", fb.status_code == 201)
    hist = api.call("GET", "/recommendations/history?limit=100").json()
    check(
        "feedback persisted in recommendation history",
        any(x["id"] == target["recommendation_id"] and x["feedback"] == "not_interested" for x in hist),
    )
    after = [i["movie_id"] for i in api.call("GET", "/recommendations?limit=20").json()["items"]]
    check("'not interested' movie no longer recommended", target["movie_id"] not in after)

    adm = Api(args.base)
    lr = adm.call("POST", "/auth/login", json={"email": args.admin_email, "password": args.admin_password})
    if check("admin login", lr.ok):
        adm.token = lr.json()["access_token"]
        models = adm.call("GET", "/models").json()
        check(
            "admin model dashboard lists versions", bool(models) and sum(m["is_active"] for m in models) == 1
        )
        exps = adm.call("GET", "/experiments").json()
        detail = adm.call("GET", f"/experiments/{exps[0]['id']}").json() if exps else {"metrics": []}
        hy = {
            m["metric"]: m["value"]
            for m in detail["metrics"]
            if m["model_name"] == "hybrid" and m["protocol"] == "test" and m["k"] == 10
        }
        check(
            "experiment metrics visible",
            {"ndcg", "recall", "precision", "f1", "map", "hit_rate"} <= set(hy),
            f"hybrid NDCG@10={hy.get('ndcg', float('nan')):.4f}",
        )
    check("non-admin blocked from admin API", api.call("GET", "/models").status_code == 403)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
