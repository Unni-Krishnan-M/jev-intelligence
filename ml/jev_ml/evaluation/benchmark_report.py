"""REPORT.md of a recommender benchmark run (every number comes from results.json)."""

from __future__ import annotations

from typing import Any

WARM_ORDER = (
    "hybrid",
    "hybrid_cold",
    "hybrid_retuned",
    "hybrid_all_tags",
    "als",
    "als_retuned",
    "itemknn",
    "itemknn_retuned",
    "popularity",
    "popularity_tuned",
    "content",
    "random",
)
COLD_ORDER = ("hybrid", "hybrid_cold", "popularity", "popularity_tuned", "als", "itemknn", "content")


def _ci(e: dict[str, Any] | None) -> str:
    if not e or e.get("mean") is None:
        return "n/a"
    return f"{e['mean']:.4f} [{e['lo']:.4f}, {e['hi']:.4f}]"


def _d(e: dict[str, Any] | None) -> str:
    if not e or e.get("diff") is None:
        return "n/a"
    star = " *" if significant(e) else ""
    if e.get("p_holm") is not None:
        p = f"p_holm={e['p_holm']:.3g}"
    else:
        p = f"p={e['p_value']:.3g}" if e.get("p_value") is not None else "p=n/a"
    return f"{e['diff']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]{star} ({p})"


def significant(e: dict[str, Any]) -> bool:
    """The star rule: Holm-adjusted p < 0.05 where the comparison belongs to an adjusted family
    (the warm paired comparisons); otherwise (no family, e.g. a single cold bucket) the CI excludes 0."""
    if e.get("p_holm") is not None:
        return bool(e["p_holm"] < 0.05)
    return bool(e.get("significant"))


def _f(v: Any, nd: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def write_markdown(res: dict[str, Any]) -> str:
    md = [
        f"# Recommender benchmark `{res['run_id']}`",
        "",
        f"- dataset `{res['dataset_version']}`, git `{res['git_commit']}`, "
        f"config hash `{res['config_hash']}`, "
        f"seed {res['seed']}, quick={res['quick']}",
        f"- incumbent parameters: `{res['incumbent']['source']}`",
        f"- command: `{res['command']}`; runtime {res['seconds_total']:.0f}s",
        "- CIs: per-user percentile bootstrap 95 %; paired differences: bootstrap CI and sign-flip "
        "permutation p (Holm-adjusted within the warm family). `*` = Holm-adjusted p < 0.05 where "
        "a Holm p is shown, otherwise CI excludes 0.",
        "",
    ]
    for p, r in res["protocols"].items():
        md += [f"## Protocol `{p}`", ""]
        s = r["leakage"]["summary"]
        md.append(
            f"Split: {s['train_rows']} / {s['val_rows']} / {s['test_rows']} rows; "
            f"leakage check `{r['leakage']['split']}`; tags {r['leakage']['tags']}; "
            f"future items excluded {r['leakage']['future_items_excluded']}."
        )
        md += [
            "",
            "### Warm test (full training profile)",
            "",
            "| model | users | NDCG@10 | Recall@10 |",
            "|---|---|---|---|",
        ]
        for m in WARM_ORDER:
            if m in r["warm"]:
                w = r["warm"][m]
                md.append(f"| {m} | {w['n_users']} | {_ci(w['ndcg@10'])} | {_ci(w['recall@10'])} |")
        md += ["", "Paired NDCG@10 differences:", "", "| comparison | diff [95% CI] |", "|---|---|"]
        for k, v in r["warm_paired_ndcg10"].items():
            md.append(f"| {k} | {_d(v)} |")
        md += ["", "Paired Recall@10 differences:", "", "| comparison | diff [95% CI] |", "|---|---|"]
        for k, v in r["warm_paired_recall10"].items():
            md.append(f"| {k} | {_d(v)} |")
        md += ["", "### Cold start: buckets (user-level mean over the bucket's sizes)", ""]
        md += [
            "| bucket | users | hybrid | hybrid_cold | popularity | popularity_tuned "
            "| cold - hybrid | cold - popularity |"
        ]
        md.append("|---|---|---|---|---|---|---|---|")
        for key, e in r["cold_buckets"].items():
            n = e["ndcg@10"]
            md.append(
                f"| {key} | {e['n_users']} | {_ci(n['hybrid'])} | {_ci(n['hybrid_cold'])} "
                f"| {_ci(n['popularity'])} "
                f"| {_ci(n['popularity_tuned'])} | {_d(e['paired']['hybrid_cold_vs_hybrid'])} "
                f"| {_d(e['paired']['hybrid_cold_vs_popularity'])} |"
            )
        md += ["", "### Cold start: NDCG@10 per profile size", ""]
        md += ["| cell | users | " + " | ".join(COLD_ORDER) + " |", "|---|---|" + "---|" * len(COLD_ORDER)]
        for cell, models in r["cold"].items():
            users = next(iter(models.values()))["n_users"]
            md.append(
                f"| {cell} | {users} | "
                + " | ".join(_f(models[m]["ndcg@10"]["mean"]) for m in COLD_ORDER)
                + " |"
            )
        st = r["cold_stage_tuning"]
        md += [
            "",
            "### Cold stages chosen on validation",
            "",
            "| bucket | chosen | val objective | incumbent val | best per family |",
            "|---|---|---|---|---|",
        ]
        for b, e in st["buckets"].items():
            fam = ", ".join(f"{k} {v:.4f}" for k, v in e["best_per_family"].items())
            md.append(
                f"| {b} | {e['chosen_family']} `{e['chosen_stage']}` | {e['validation_objective']:.4f} "
                f"| {e['incumbent_objective']:.4f} | {fam} |"
            )
        md += ["", "### Baseline tuning (validation)", ""]
        for name, t in r["tuning"].items():
            md.append(f"- {name}: best `{t['best']}` ({t['budget']} configurations)")
        cal = r.get("calibration") or {}
        md += ["", "### Confidence calibration (test; fitted on validation)", ""]
        if "error" in cal:
            md.append(f"Not computed: {cal['error']}")
        elif not all("served" in s_ for s_ in cal.get("strata", [])):
            md.append("Stored in an earlier results.json schema; not rendered here (see results.json).")
        else:
            md += [
                f"Target: {cal['target']}. Logistic features: {cal['features']}.",
                "",
                "| stratum | rows | method | AUC [95% CI] | Brier | Brier skill | ECE | ECE (equal mass) |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for s_ in cal["strata"]:
                for meth in ("isotonic", "logistic", "served"):
                    m = s_[meth]
                    ci = m["auc_ci"]
                    auc = "n/a" if m["auc"] is None else f"{m['auc']:.4f} [{_f(ci[0])}, {_f(ci[1])}]"
                    label = f"served ({s_['serving_method']})" if meth == "served" else meth
                    md.append(
                        f"| {s_['name']} | {s_['n_test_rows']} | {label} | {auc} | {_f(m['brier'], 5)} "
                        f"| {_f(m['brier_skill_vs_base_rate'])} | {_f(m['ece'], 5)} "
                        f"| {_f(m['ece_equal_mass'], 5)} |"
                    )
                for meth in ("logistic", "served"):
                    a = s_[f"auc_diff_{meth}_minus_isotonic"]
                    bd = s_[f"brier_diff_{meth}_minus_isotonic"]
                    md.append(
                        f"| {s_['name']} | | {meth} - isotonic "
                        f"| {_f(a['value'])} [{_f(a['lo'])}, {_f(a['hi'])}] "
                        f"| {_f(bd['value'], 6)} [{_f(bd['lo'], 6)}, {_f(bd['hi'], 6)}] | | | |"
                    )
            md += ["", "Logistic coefficients (standardised features) and validation cross-fit log-loss:", ""]
            for s_ in cal["strata"]:
                md.append(
                    f"- {s_['name']}: C={s_['logistic_C']}, "
                    f"{ {k: round(v, 3) for k, v in s_['logistic_coef'].items()} }, "
                    f"log-loss {s_['crossfit_log_loss']} -> serves {s_['serving_method']}"
                )
        md += ["", f"Protocol runtime {r['seconds']:.0f}s.", ""]
    md += [
        "## Decisions (test data)",
        "",
        "```",
        *[f"{k}: {v}" for k, v in res["decision"].items()],
        "```",
        "",
    ]
    if res.get("latency_ms"):
        md += ["## Serving latency (median ms per 20-item request)", "", f"`{res['latency_ms']}`", ""]
    return "\n".join(md) + "\n"
