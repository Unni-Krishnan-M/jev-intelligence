"""End-to-end acceptance scenario against a *running* stack (local or Docker Compose).

    uv run python scripts/acceptance_test.py --base http://localhost:3000/api \
        --admin-email admin@example.com --admin-password ...

Goes through the web proxy by default, so it exercises Next.js → FastAPI → Postgres/Redis → model.

Part 1 is the recommender journey (member). Part 2 (skip with --skip-intel) drives the intelligence
layer as the admin: status, a run and a leak-free replay (--as-of, default 2017-07-01), every
run-backed list, a score decision with its interval, decision batches, a warning transition and its
history row, evidence search, history across the two runs, a saved what-if scenario, feedback on a
decision and a warning (and the summary counts), evaluation, recommender monitoring, the audit log
and /admin/metrics, plus the 401/403 access rules. It can be re-run against the same database: the
acknowledged warning is resolved at the end, so the next run reopens it.

Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import requests

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{f' — {detail}' if detail else ''}")
    return ok


@contextmanager
def step(name: str) -> Iterator[None]:
    """An unexpected response shape (KeyError, bad JSON, HTTP error) fails the step, not the script."""
    try:
        yield
    except Exception as exc:  # reported as a FAIL line
        check(name, False, f"{type(exc).__name__}: {exc}")


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


def intelligence(args: argparse.Namespace, member: Api, adm: Api, login_rid: str | None) -> None:
    """Part 2: the intelligence layer end to end, as the admin (see the module docstring)."""
    print("\n-- intelligence layer --")
    anon = Api(args.base)
    code = anon.call("GET", "/intel/status").status_code
    check("anonymous gets 401 on /intel/status", code == 401, str(code))
    code = member.call("GET", "/intel/status").status_code
    check("member gets 403 on /intel/status", code == 403, str(code))
    code = member.call("GET", "/admin/audit").status_code
    check("member gets 403 on /admin/audit", code == 403, str(code))
    if adm.token is None:
        check("intelligence checks need an admin session", False, "admin login failed")
        return

    def get(path: str, **kw: Any) -> Any:
        r = adm.call("GET", path, **kw)
        r.raise_for_status()
        return r.json()

    rids: dict[str, str] = {}  # request ids of the writes, to find their audit rows
    with step("status"):
        st = get("/intel/status")
        health = st["health"]
        sources = {s["source"]: s for s in (st.get("data") or {}).get("sources", [])}
        check(
            "status: health (database, cache, model) ok and data freshness reported",
            all(health[k] == "ok" for k in ("database", "cache", "model")) and "movielens" in sources,
            f"pipeline {health['pipeline']}, open warnings {st['warnings_open']['total']}, "
            f"MovieLens last event {sources.get('movielens', {}).get('last_event')}",
        )

    current = replay = None
    with step("trigger run"):
        r = adm.call("POST", "/intel/runs", json={})
        current = r.json()
        rids["run"] = r.headers.get("x-request-id", "")
        check(
            "trigger an intelligence run (latest data)",
            r.ok and current["status"] == "succeeded",
            f"as_of {current.get('as_of')}, {current.get('duration_ms')} ms, "
            f"{current.get('summary', {}).get('headline')}",
        )
    with step("replay run"):
        r = adm.call("POST", "/intel/runs", json={"as_of": args.as_of})
        replay = r.json()
        rids["replay"] = r.headers.get("x-request-id", "")
        check(
            f"replay run as of {args.as_of} (leak-free)",
            r.ok and replay["status"] == "succeeded" and replay["as_of"].startswith(args.as_of),
            f"{replay.get('duration_ms')} ms, {replay.get('summary', {}).get('headline')}",
        )
        code = adm.call("POST", "/intel/runs", json={"as_of": "2999-01-01"}).status_code
        check("future as_of rejected with 422", code == 422, str(code))
        one = get(f"/intel/runs/{replay['run_id']}")
        check("run readable by uuid", one["id"] == replay["id"])
    if not (current and replay and current.get("status") == replay.get("status") == "succeeded"):
        check("intelligence checks need two successful runs", False)
        return
    cur_id, rep_id = current["run_id"], replay["run_id"]

    # the replay is now the latest successful run, so the lists below read it
    with step("signals"):
        sig = get("/intel/signals?limit=200")
        check(
            "signals: latest run is the replay, every signal carries evidence",
            sig["run_id"] == rep_id and sig["total"] > 0 and all(s["evidence"] for s in sig["items"]),
            f"{sig['total']} signals",
        )
    with step("trends"):
        tr = get("/intel/trends?limit=200")
        check(
            "trends: Theil–Sen slope CI + p/q values on every trend",
            tr["total"] > 0
            and all(len(t["slope_ci"]) == 2 and t["q_value"] is not None for t in tr["items"]),
            f"{tr['total']} trends, {sum(t['direction'] != 'flat' for t in tr['items'])} not flat",
        )
    with step("anomalies"):
        an = get("/intel/anomalies?limit=200")
        horror = [
            a for a in an["items"] if a["series_id"] == "share:genre:Horror" and a["kind"] == "series_spike"
        ]
        if args.as_of == "2017-07-01":
            ok = bool(horror) and horror[0]["severity"] == "high" and horror[0]["detected_at"] == "2017-05-01"
            detail = (
                f"Horror share spike {horror[0]['detected_at']} robust z {horror[0]['score']} "
                f"({horror[0]['severity']})"
                if horror
                else "no Horror spike"
            )
            check("anomalies: replay surfaces the May-2017 Horror spike (high)", ok, detail)
        else:
            check("anomalies listed", an["total"] > 0, f"{an['total']} anomalies")
    with step("predictions"):
        pr = get("/intel/predictions")
        fc = pr["forecasts"]
        bands = all(p["lo80"] <= p["mean"] <= p["hi80"] for f in fc for p in f["points"])
        lapse = (pr.get("lapse") or {}).get("metrics", {})
        check(
            "predictions: forecasts with 80 % bands + lapse model metrics",
            bool(fc) and bands and lapse.get("auc") is not None,
            f"{len(fc)} forecasts, lapse AUC {lapse.get('auc')}, ECE {lapse.get('ece')}",
        )
        s = get("/intel/series/share:genre:Horror")
        check("series detail (trend + anomalies + forecast)", s["series"]["id"] == "share:genre:Horror")
    with step("risks"):
        rk = get("/intel/risks")
        check(
            "risks: scored 0–100 with contributing factors",
            rk["total"] > 0 and all(0 <= x["score"] <= 100 and x["factors"] for x in rk["items"]),
            ", ".join(f"{x['kind']} {x['score']} ({x['level']})" for x in rk["items"][:3]),
        )
        acts = get("/intel/actions")
        check(
            "recommended actions listed",
            acts["total"] > 0,
            acts["items"][0]["title"] if acts["items"] else "",
        )

    score_dec: dict[str, Any] = {}
    with step("score decision"):
        decs = get(f"/intel/decisions?run_id={cur_id}&key=editorial_slot_share&limit=200")["items"]
        answered = [d for d in decs if not d["abstained"]]
        score_dec = answered[0] if answered else {}
        lo, hi = (score_dec.get("answer_interval") or [None, None])[:2]
        ok = (
            bool(score_dec)
            and score_dec["kind"] == "score"
            and score_dec["confidence_kind"] == "interval"
            and isinstance(score_dec["answer"], int | float)
            and lo is not None
            and lo <= score_dec["answer"] <= hi
            and score_dec["scale"]["max"] == 100
        )
        check(
            "score decision with an interval (editorial_slot_share)",
            ok,
            f"{score_dec.get('entity')}: {score_dec.get('answer')} "
            f"{score_dec.get('scale', {}).get('unit')}, {score_dec.get('confidence')} interval "
            f"[{lo}, {hi}]; {len(answered)}/{len(decs)} genres answered",
        )
        full = get(f"/intel/decisions/{score_dec['db_id']}")
        check(
            "decision detail carries state, rationale and evidence",
            bool(full["state"]) and bool(full["rationale"]) and bool(full["evidence"]),
        )
        gov = get(f"/intel/decisions?run_id={rep_id}&limit=200")["items"]
        abst = {d["key"] for d in gov if d["abstained"]}
        check(
            "replay: retrain + serving decisions abstain (model trained on later data)",
            {"retrain_model", "serving_model"} <= abst,
            f"{len([d for d in gov if d['abstained']])}/{len(gov)} abstained",
        )
    with step("decision batches"):
        bat = get(f"/intel/decisions/batches?run_id={cur_id}")["items"]
        names = {b["name"] for b in bat}
        gov_b = next(b for b in bat if b["name"] == "model_governance")
        members = get(f"/intel/decisions?run_id={cur_id}&batch_id={gov_b['id']}")
        check(
            "decision batches share one hashed state; batch_id filter matches",
            {"model_governance", "genre_programming", "audience"} <= names
            and all(b.get("state_hash") for b in bat)
            and members["total"] == len(gov_b["decision_ids"]),
            ", ".join(f"{b['name']} ({len(b['decision_ids'])})" for b in bat),
        )

    wid = None
    with step("warning transition"):
        new = get("/intel/warnings?status=new&limit=200")["items"]
        pick = next((w for w in new if "Horror" in w["key"]), new[0] if new else None)
        if pick is None:
            check("an open 'new' warning to acknowledge", False, "none")
        else:
            wid = pick["id"]
            r = adm.call(
                "PATCH", f"/intel/warnings/{wid}", json={"status": "acknowledged", "note": "acceptance test"}
            )
            rids["transition"] = r.headers.get("x-request-id", "")
            w = r.json()
            last = (w.get("history") or [{}])[-1]
            check(
                "acknowledge a warning; history row records it",
                r.ok
                and w["status"] == "acknowledged"
                and last.get("from_status") == "new"
                and last.get("to_status") == "acknowledged"
                and last.get("actor") == args.admin_email,
                f"#{wid} {pick['title']} ({pick['severity']}): {len(w.get('history') or [])} history rows",
            )
            code = adm.call("PATCH", f"/intel/warnings/{wid}", json={"status": "new"}).status_code
            check("illegal transition back to 'new' rejected", code in (409, 422), str(code))

    with step("evidence search"):
        ev = get("/intel/evidence?q=horror&limit=200")
        hit = all(
            "horror" in " ".join(str(e.get(k) or "") for k in ("label", "detail", "owner_title")).lower()
            for e in ev["items"]
        )
        check(
            "evidence search (q=horror) returns only matching items",
            ev["total"] > 0 and hit,
            f"{ev['total']} items",
        )
        evd = get("/intel/evidence?owner_type=decision&limit=5")
        check("evidence filter by owner_type", all(e["owner_type"] == "decision" for e in evd["items"]))
    with step("history"):
        hs = get("/intel/history/trends?key=volume:all")
        runs = {p["run_id"] for p in hs["items"]}
        check(
            "history across the two runs (trend volume:all)",
            {cur_id, rep_id} <= runs,
            f"{len(hs['items'])} points over {len(runs)} runs",
        )
        hr = get("/intel/history/risks?key=risk:audience_lapse:all")
        check("risk history across runs", {cur_id, rep_id} <= {p["run_id"] for p in hr["items"]})

    with step("scenario"):
        r = adm.call(
            "POST",
            "/intel/scenarios",
            json={
                "series_id": "volume:all",
                "horizon_months": 6,
                "scenarios": [
                    {"name": "Trend continues", "kind": "continue"},
                    {"name": "Trend slows", "kind": "slow", "trend_multiplier": 0.5},
                    {"name": "Shock -20 %", "kind": "shock", "level_shift_pct": -20, "shock_month": 1},
                ],
                "save": True,
                "title": "Acceptance scenario",
            },
        )
        sc = r.json()
        saved = get("/intel/scenarios?series_id=volume:all&limit=200")["items"]
        check(
            "what-if scenario run and saved",
            r.ok
            and len(sc["scenarios"]) == 3
            and sc["id"] is not None
            and any(s["id"] == sc["id"] for s in saved),
            ", ".join(f"{c['name']} {c['delta_pct']:+.1f} %" for c in sc.get("comparison", [])),
        )
        rids["scenario"] = r.headers.get("x-request-id", "")

    with step("feedback"):
        before = get("/intel/feedback")["summary"]
        r1 = adm.call(
            "POST",
            "/intel/feedback",
            json={
                "target_type": "decision",
                "target_id": score_dec["id"],
                "verdict": "correct",
                "note": "acceptance",
            },
        )
        rids["feedback_decision"] = r1.headers.get("x-request-id", "")
        r2 = adm.call(
            "POST",
            "/intel/feedback",
            json={"target_type": "warning", "target_id": str(wid), "verdict": "useful"},
        )
        rids["feedback_warning"] = r2.headers.get("x-request-id", "")
        bad = adm.call(
            "POST",
            "/intel/feedback",
            json={"target_type": "warning", "target_id": str(wid), "verdict": "correct"},
        ).status_code
        after = get("/intel/feedback")["summary"]
        check(
            "feedback on a decision and a warning; summary counts move",
            r1.status_code == 201
            and r2.status_code == 201
            and after["decision"]["correct"] == before["decision"]["correct"] + 1
            and after["warning"]["useful"] == before["warning"]["useful"] + 1,
            f"decision accuracy {after['decision']['accuracy']}, "
            f"warning precision {after['warning']['precision']}",
        )
        check("verdict that does not fit the target rejected (422)", bad == 422, str(bad))

    with step("evaluation"):
        ev = get("/intel/evaluation")
        runs = get("/intel/evaluation/runs")
        head = runs["items"][0]["headline"] if runs["items"] else {}
        check(
            "offline evaluation report + evaluation runs",
            ev["available"] and runs["total"] > 0 and head.get("median_mase") is not None,
            f"{ev['run_dir']}: median MASE {head.get('median_mase')}, lapse AUC {head.get('lapse_auc')}",
        )
    with step("feedback dedup"):
        # POST /recommendations/feedback is an upsert (migration 0004): the same verdict twice counts once
        rec = member.call("GET", "/recommendations?limit=5").json()["items"][0]
        before_d = get("/intel/recommendations?recent=0")["feedback_totals"]["dislike"]
        body = {
            "movie_id": rec["movie_id"],
            "feedback": "dislike",
            "recommendation_id": rec["recommendation_id"],
        }
        codes = [member.call("POST", "/recommendations/feedback", json=body).status_code for _ in range(2)]
        after_d = get("/intel/recommendations?recent=0")["feedback_totals"]["dislike"]
        check(
            "the same dislike posted twice counts once (feedback upsert)",
            codes == [201, 201] and after_d == before_d + 1,
            f"HTTP {codes}, dislikes {before_d} → {after_d}",
        )
    with step("recommender monitoring"):
        mon = get("/intel/recommendations")
        check(
            "recommender monitoring: served rows, confidence histogram, calibration",
            mon["served"]["total"] > 0
            and len(mon["confidence_histogram"]) == 10
            and mon["calibration"] is not None
            and mon["served"]["with_confidence"] > 0,
            f"served {mon['served']['total']} ({mon['served']['with_confidence']} with confidence), "
            f"feedback {mon['feedback_totals']}",
        )

    with step("audit log"):

        def audit_rids(action: str) -> dict[str, dict[str, Any]]:
            items = get(f"/admin/audit?action={action}&limit=200")["items"]
            return {e["request_id"]: e for e in items if e.get("request_id")}

        logins = audit_rids("auth.login.success")
        runs_a = audit_rids("intel.run")
        trans = audit_rids("warning.transition")
        fb = audit_rids("feedback.create")
        found = {
            "login": login_rid in logins,
            "run": runs_a.get(rids.get("run", ""), {}).get("target_id") == cur_id,
            "replay": runs_a.get(rids.get("replay", ""), {}).get("target_id") == rep_id,
            "transition": rids.get("transition", "") in trans,
            "feedback×2": rids.get("feedback_decision", "") in fb and rids.get("feedback_warning", "") in fb,
            "scenario": rids.get("scenario", "") in audit_rids("scenario.save"),
        }
        check(
            "audit log records the login, runs, transition, feedback and scenario",
            all(found.values()),
            ", ".join(f"{k} {'ok' if v else 'MISSING'}" for k, v in found.items()),
        )
        dump = str(get("/admin/audit?limit=200"))
        check("audit log never contains the admin password", args.admin_password not in dump)
    with step("metrics"):
        m = get("/admin/metrics")
        pipe = m["intel_pipeline"]
        check(
            "/admin/metrics: requests, pipeline, audit and persisted-row counters",
            bool(m["http_requests_by_status"])
            and pipe.get("succeeded", 0) >= 2
            and m["audit_entries_by_action"].get("intel.run", 0) >= 2
            and m["intel_rows_persisted"].get("intel_evidence", 0) > 0,
            f"pipeline {pipe.get('succeeded')} succeeded, last {pipe.get('last_duration_ms')} ms, "
            f"audit {m['audit_entries_by_action']}",
        )

    # leave the stack re-runnable: a resolved warning reopens when a later run fires it again
    if wid is not None:
        with step("resolve warning"):
            r = adm.call(
                "PATCH", f"/intel/warnings/{wid}", json={"status": "resolved", "note": "acceptance test"}
            )
            check("resolve the acknowledged warning (terminal)", r.ok and r.json()["status"] == "resolved")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:3000/api")
    ap.add_argument("--admin-email", default="admin@example.com")
    ap.add_argument("--admin-password", default="admin-pass-123")
    ap.add_argument("--as-of", default="2017-07-01", help="replay date for the intelligence run")
    ap.add_argument("--skip-intel", action="store_true", help="only the recommender checks")
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
    with_conf = [i for i in items if i.get("confidence") is not None]
    check(
        "every item carries confidence + confidence_kind keys (calibrated P(rating ≥ 4) or null)",
        all("confidence" in i and "confidence_kind" in i for i in items)
        and all(0 <= i["confidence"] <= 1 and i["confidence_kind"] == "probability" for i in with_conf),
        f"{len(with_conf)}/{len(items)} with a value"
        + (
            f", range {min(i['confidence'] for i in with_conf):.4f}"
            f"–{max(i['confidence'] for i in with_conf):.4f}"
            if with_conf
            else ""
        ),
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

    if not args.skip_intel:
        intelligence(args, api, adm, login_rid=lr.headers.get("x-request-id") if lr.ok else None)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
