"""Experiment report: metrics.json, comparison tables (CSV + Markdown) and PNG charts.

Charts follow one visual system: hairline recessive axes, thin marks, one highlighted series
(the hybrid) against gray peers, categorical colors in a fixed order, and no dual axes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PEER = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_ORDER = ["hybrid", "als", "itemknn", "content", "popularity", "random"]
MODEL_LABEL = {
    "hybrid": "Hybrid",
    "als": "ALS (MF)",
    "itemknn": "Item-kNN (CF)",
    "content": "Content (TF-IDF)",
    "popularity": "Popularity",
    "random": "Random",
}
TABLE_METRICS = [
    "precision@10",
    "recall@10",
    "f1@10",
    "ndcg@10",
    "map@10",
    "hit_rate@10",
    "coverage@10",
    "diversity@10",
    "novelty@10",
]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def comparison_frame(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for m in MODEL_ORDER:
        if m in metrics:
            rows.append({"model": m, **{k: metrics[m].get(k) for k in TABLE_METRICS}})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    best = {c: df[c].max() for c in cols if c != "model"}
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "model":
                cells.append(MODEL_LABEL.get(v, v))
            else:
                s = f"{v:.4f}" if c != "novelty@10" else f"{v:.2f}"
                cells.append(f"**{s}**" if v == best[c] else s)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_metric_panels(metrics: dict[str, dict[str, float]], out: Path, title: str) -> None:
    panels = ["ndcg@10", "recall@10", "precision@10", "map@10", "hit_rate@10", "coverage@10"]
    models = [m for m in MODEL_ORDER if m in metrics][::-1]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), facecolor=SURFACE)
    for ax, metric in zip(axes.ravel(), panels, strict=True):
        _style(ax)
        vals = [metrics[m][metric] for m in models]
        colors = [CATEGORICAL[0] if m == "hybrid" else PEER for m in models]
        ax.barh(range(len(models)), vals, color=colors, height=0.62, edgecolor=SURFACE, linewidth=1.5)
        ax.set_yticks(range(len(models)), [MODEL_LABEL[m] for m in models], color=INK_2, fontsize=8)
        ax.set_title(metric.upper().replace("_", " "), loc="left", fontsize=9, color=INK, pad=6)
        vmax = max(vals) if vals else 1
        ax.set_xlim(0, vmax * 1.22 if vmax > 0 else 1)
        for i, (m, v) in enumerate(zip(models, vals, strict=True)):
            if m == "hybrid" or v == vmax:
                ax.text(v + vmax * 0.02, i, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_ndcg_vs_k(metrics: dict[str, dict[str, float]], ks: list[int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    for idx, m in enumerate(m for m in MODEL_ORDER if m in metrics):
        ys = [metrics[m][f"ndcg@{k}"] for k in ks]
        color = CATEGORICAL[idx]
        ax.plot(ks, ys, color=color, linewidth=2, marker="o", markersize=5, label=MODEL_LABEL[m])
    ax.set_xticks(ks, [f"K={k}" for k in ks])
    ax.set_xlim(ks[0] - 1, ks[-1] + 1)
    ax.set_ylabel("NDCG@K", color=MUTED, fontsize=8)
    # >4 series: the legend carries identity, placed outside the plot so it never covers data
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("NDCG by cut-off K (test split)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_loss(loss: list[float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    it = list(range(1, len(loss) + 1))
    ax.plot(it, loss, color=CATEGORICAL[0], linewidth=2, marker="o", markersize=4)
    ax.annotate(
        f"{loss[-1]:.4f}",
        (it[-1], loss[-1]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=7.5,
        color=INK,
    )
    ax.set_xlabel("iteration", color=MUTED, fontsize=8)
    ax.set_ylabel("weighted loss / nnz", color=MUTED, fontsize=8)
    ax.set_title("ALS training objective (production fit)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_report(result: dict[str, Any], exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    plots = exp_dir / "plots"
    plots.mkdir(exist_ok=True)
    test = result["metrics"]["test"]
    cold = result["metrics"].get("cold_start") or {}
    ks = result["config"]["evaluation"]["ks"]

    (exp_dir / "metrics.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "per_user_ndcg10"}, indent=2, default=str)
    )
    (exp_dir / "per_user_ndcg10.json").write_text(json.dumps(result["per_user_ndcg10"]))

    df = comparison_frame(test)
    df.to_csv(exp_dir / "comparison_test.csv", index=False)
    md = [
        f"# Experiment `{result['run_id']}`",
        "",
        f"- dataset: `{result['dataset_version']}`",
        f"- seed: {result['training_seed']}",
        f"- split: {result['split']}",
        f"- model version: `{result['model_version']}`",
        f"- total runtime: {result['seconds_total']:.0f}s",
        "",
        f"## Test split — {result['metrics']['n_eval_users']['test']} users, K=10",
        "",
        to_markdown(df),
    ]
    plot_metric_panels(test, plots / "test_metrics.png", "Test split · all models · K=10")
    plot_ndcg_vs_k(test, ks, plots / "ndcg_vs_k.png")
    if cold:
        cdf = comparison_frame(cold)
        cdf.to_csv(exp_dir / "comparison_cold_start.csv", index=False)
        n = result["config"]["evaluation"]["cold_start_profile_size"]
        md += [
            "",
            f"## Cold start — profiles truncated to first {n} interactions, K=10",
            "",
            to_markdown(cdf),
        ]
        plot_metric_panels(
            cold, plots / "cold_start_metrics.png", f"Cold start (first {n} interactions) · K=10"
        )
    if result.get("als_loss_history"):
        plot_loss(result["als_loss_history"], plots / "als_loss.png")
    tuning = result.get("tuning") or {}
    if tuning:
        md += [
            "",
            "## Tuning (validation split)",
            "",
            f"- selection metric: `{tuning['selection_metric']}`",
            f"- item-kNN best: `{tuning['itemknn']}`",
            f"- ALS best: `{tuning['als']}`",
            f"- hybrid best weights: `{tuning['hybrid']['weights']}`, "
            f"λ={tuning['hybrid']['diversity_lambda']}",
            f"- trials: kNN {len(tuning['trials']['itemknn'])}, ALS {len(tuning['trials']['als'])}, "
            f"hybrid {len(tuning['trials']['hybrid'])}",
        ]
    md += ["", "![test metrics](plots/test_metrics.png)", "![ndcg vs k](plots/ndcg_vs_k.png)"]
    (exp_dir / "REPORT.md").write_text("\n".join(md) + "\n")
