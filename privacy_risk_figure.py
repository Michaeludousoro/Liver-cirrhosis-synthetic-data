"""
Privacy Disclosure Risk Analysis (Figure and Summary Table)
===========================================================

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Why this script exists
----------------------
The nearest-neighbour privacy analysis was originally computed inside notebook
02. That notebook also regenerates synthetic data using an earlier consensus
implementation which does not match the one in master_runner.py, so executing it
overwrites output/data with a different realisation and invalidates the numbers
reported in the paper.

This script reproduces exactly the same privacy analysis, but reads the datasets
that master_runner.py has already written to output/data/. It is therefore safe
to run at any time and always describes the same synthetic data the rest of the
results are computed from.

It writes:
    output/results/privacy_risk_summary.csv
    output/figures/privacy_risk_analysis.png

No figure title is drawn: in the manuscript the LaTeX caption supplies it, and an
in-image title would duplicate the caption and bake a figure number into the
image that could later go stale.

Usage
-----
    python master_runner.py        # first, to produce output/data
    python privacy_risk_figure.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "output", "data")
FIG_DIR  = os.path.join(BASE_DIR, "output", "figures")
RES_DIR  = os.path.join(BASE_DIR, "output", "results")

TARGET_COL = "Status"

# Percentile of the real-to-real nearest-neighbour distance used as the
# disclosure-risk threshold. A synthetic record closer to a real patient than
# this is counted as a near-duplicate.
RISK_PERCENTILE = 5


def load():
    real = pd.read_csv(os.path.join(DATA_DIR, "train_real.csv"))
    sets = [
        ("GAN (filtered)",   pd.read_csv(os.path.join(DATA_DIR, "filtered_gan.csv")),   "#0066FF"),
        ("CTGAN (filtered)", pd.read_csv(os.path.join(DATA_DIR, "filtered_ctgan.csv")), "#FF4500"),
        ("TVAE (filtered)",  pd.read_csv(os.path.join(DATA_DIR, "filtered_tvae.csv")),  "#00B43C"),
        ("Consensus",        pd.read_csv(os.path.join(DATA_DIR, "consensus_equalised.csv")), "#9900CC"),
    ]
    return real, sets


def compute(real, sets):
    feat_cols = [c for c in real.columns if c != TARGET_COL]
    fill = real[feat_cols].mean()

    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real[feat_cols].fillna(fill))

    # k=2 because the first neighbour of a real patient is itself.
    nn_real = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(real_scaled)
    real_real_min = nn_real.kneighbors(real_scaled)[0][:, 1]
    threshold = np.percentile(real_real_min, RISK_PERCENTILE)

    nn_synth = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(real_scaled)

    rows, dists_by_method = [], {}
    for name, df, _ in sets:
        scaled = scaler.transform(df[feat_cols].fillna(fill))
        min_dists = nn_synth.kneighbors(scaled)[0][:, 0]
        dists_by_method[name] = min_dists
        n_at_risk = int((min_dists < threshold).sum())
        rows.append({
            "Method":             name,
            "Synthetic records":  len(min_dists),
            "Mean dist to real":  round(float(min_dists.mean()), 4),
            "Min dist to real":   round(float(min_dists.min()), 4),
            "Near-duplicate (n)": n_at_risk,
            "Near-duplicate (%)": round(n_at_risk / len(min_dists) * 100, 1),
        })

    return pd.DataFrame(rows), dists_by_method, real_real_min, threshold


def plot(summary, dists_by_method, real_real_min, threshold, sets):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    data   = [real_real_min] + [dists_by_method[n] for n, _, _ in sets]
    labels = ["Real-to-Real\n(baseline)"] + [n.replace(" ", "\n") for n, _, _ in sets]
    colors = ["#505050"] + [c for _, _, c in sets]

    parts = ax.violinplot(data, positions=range(len(data)),
                          showmedians=True, showextrema=True, widths=0.6)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.65)
        pc.set_edgecolor("white")
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    for part in ["cbars", "cmaxes", "cmins"]:
        parts[part].set_color("#444444")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Nearest-neighbour distance (standardised)")
    ax.set_title("Distance from each record to its nearest real patient")
    ax.axhline(threshold, color="#DC0000", linewidth=1.5, linestyle="--",
               label=f"Risk threshold ({RISK_PERCENTILE}th pct real-to-real = {threshold:.3f})")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    methods = summary["Method"].tolist()
    risks   = summary["Near-duplicate (%)"].tolist()
    bars = ax2.bar(methods, risks, color=[c for _, _, c in sets],
                   edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, risks):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Percentage of records flagged as near-duplicates (%)")
    ax2.set_title("Near-duplicate risk by synthetic method\n"
                  "(records closer to a real patient than the risk threshold)")
    ax2.set_ylim(0, max(risks) * 1.35 + 1)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, "privacy_risk_analysis.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    real, sets = load()
    summary, dists, real_real_min, threshold = compute(real, sets)

    print(f"  Risk threshold ({RISK_PERCENTILE}th pct real-to-real): {threshold:.4f}\n")
    print(summary.to_string(index=False))

    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    csv_out = os.path.join(RES_DIR, "privacy_risk_summary.csv")
    summary.to_csv(csv_out, index=False)
    fig_out = plot(summary, dists, real_real_min, threshold, sets)

    print(f"\n  Summary written to {csv_out}")
    print(f"  Figure written to  {fig_out}")


if __name__ == "__main__":
    main()
