"""
Corrected/New Figures
======================

Generates the figures the paper rewrite needs that either changed (feature
importance, now N_Days-excluded) or are new (six-generator FID-vs-TSTR-AUC
scatter, Kaplan-Meier survival-curve comparison, corrected power analysis).
Run after rerun_full_analysis.py and run_ctgan_proper.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from src.data_loader import CLASSIFICATION_FEATURE_COLS, TARGET_COL
from src.predictive_modeling import _build_classifiers
from src.survival_analysis import DURATION_COL
from lifelines import KaplanMeierFitter

_S = 2.0
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5*_S, "axes.titlesize": 9.5*_S,
    "axes.labelsize": 8.5*_S, "xtick.labelsize": 7.5*_S, "ytick.labelsize": 7.5*_S,
    "legend.fontsize": 7.5*_S, "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.linewidth": 0.8*_S, "lines.linewidth": 1.4*_S, "xtick.major.width": 0.8*_S,
    "ytick.major.width": 0.8*_S, "patch.linewidth": 0.6*_S, "savefig.dpi": 300,
})

DATA_DIR = "output/data"
RESULTS_DIR = "output/results"
FIG_DIR = "paper/figures"


def feature_importance_figure():
    train_real = pd.read_csv(f"{DATA_DIR}/train_real.csv")
    X = train_real[CLASSIFICATION_FEATURE_COLS].values
    y = np.round(train_real[TARGET_COL].values).astype(int)

    clfs = _build_classifiers()
    rf, gb = clfs["Random Forest"], clfs["Gradient Boosting"]
    rf.fit(X, y)
    gb.fit(X, y)

    rf_imp = pd.Series(rf.feature_importances_, index=CLASSIFICATION_FEATURE_COLS).sort_values(ascending=False)
    gb_imp = pd.Series(gb.feature_importances_, index=CLASSIFICATION_FEATURE_COLS).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, imp, model_name, bar_color in [
        (axes[0], rf_imp, "Random Forest", "#0066FF"),
        (axes[1], gb_imp, "Gradient Boosting", "#FF4500"),
    ]:
        bars = ax.barh(imp.index[::-1], imp.values[::-1],
                       color=bar_color, alpha=0.82, edgecolor="white")
        for bar, val in zip(bars, imp.values[::-1]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", ha="left", fontsize=9*_S)
        ax.set_title(f"Feature Importance (N_Days excluded)\n{model_name}",
                    fontsize=12*_S, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=11*_S)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    # Save to output/figures/ (the canonical source organise_figures.py
    # copies from) AND directly to paper/figures/, so a stray rerun of
    # organise_figures.py won't silently restore the stale leaked version.
    plt.savefig("output/figures/feature_importance.png", bbox_inches="tight")
    out = f"{FIG_DIR}/fig07_feature_importance.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    print("Top 5 RF:", rf_imp.head(5).round(4).to_dict())
    print("Top 5 GB:", gb_imp.head(5).round(4).to_dict())
    return rf_imp, gb_imp


def fid_vs_tstr_scatter():
    df = pd.read_csv(f"{RESULTS_DIR}/scenario_e_six_generators.csv")
    cls_df = df[df["Framing"].str.startswith("Classification")].dropna(subset=["AUC"])

    per_gen = cls_df.groupby("Generator").agg(FID=("FID", "first"), AUC=("AUC", "mean")).reset_index()

    pearson_r, pearson_p = scipy_stats.pearsonr(per_gen["FID"], per_gen["AUC"])
    spearman_r, spearman_p = scipy_stats.spearmanr(per_gen["FID"], per_gen["AUC"])
    print(f"FID vs mean TSTR-AUC across {len(per_gen)} generators: "
          f"Pearson r={pearson_r:.3f} (p={pearson_p:.3f}), "
          f"Spearman rho={spearman_r:.3f} (p={spearman_p:.3f})")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(per_gen)))
    for (_, row), c in zip(per_gen.iterrows(), colors):
        ax.scatter(row["FID"], row["AUC"], s=140, color=c, edgecolor="black",
                  linewidth=1.2, zorder=3)
        ax.annotate(row["Generator"], (row["FID"], row["AUC"]),
                   textcoords="offset points", xytext=(8, 6), fontsize=9*_S)

    y_lo, y_hi = per_gen["AUC"].min(), per_gen["AUC"].max()
    y_pad = max((y_hi - y_lo) * 0.35, 0.01)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
    x_lo, x_hi = per_gen["FID"].min(), per_gen["FID"].max()
    x_pad = (x_hi - x_lo) * 0.15
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad * 2.5)

    ax.set_xlabel("FID (lower = higher distributional fidelity)", fontsize=11*_S)
    ax.set_ylabel("Mean TSTR AUC (RF, GB, LR)", fontsize=11*_S)
    ax.set_title(f"FID vs. train-on-synthetic-test-on-real AUC\n"
                f"Pearson r={pearson_r:.2f} (p={pearson_p:.2f}), "
                f"Spearman $\\rho$={spearman_r:.2f} (p={spearman_p:.2f})",
                fontsize=10*_S)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{FIG_DIR}/fig11_fid_vs_tstr.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    return per_gen, pearson_r, pearson_p, spearman_r, spearman_p


def km_curves_figure():
    train_real = pd.read_csv(f"{DATA_DIR}/train_real.csv")
    pool_files = {
        "Vanilla GAN": "filtered_gan.csv", "cGAN": "filtered_ctgan.csv",
        "VAE": "filtered_tvae.csv", "Masked-loss VAE": "filtered_masked_vae.csv",
        "Consensus": "consensus_equalised.csv",
    }
    if os.path.exists(f"{DATA_DIR}/filtered_ctgan_proper.csv"):
        pool_files["CTGAN"] = "filtered_ctgan_proper.csv"

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    real_kmf = KaplanMeierFitter()
    real_kmf.fit(train_real[DURATION_COL], event_observed=np.round(train_real[TARGET_COL]).astype(int),
                label="Real")

    km_df = pd.read_csv(f"{RESULTS_DIR}/km_logrank_six_generators.csv").set_index("Generator")

    for ax, (name, fname) in zip(axes, pool_files.items()):
        synth_df = pd.read_csv(f"{DATA_DIR}/{fname}")
        synth_kmf = KaplanMeierFitter()
        synth_kmf.fit(synth_df[DURATION_COL].clip(lower=1e-6),
                      event_observed=np.round(synth_df[TARGET_COL]).astype(int), label=name)

        real_kmf.plot_survival_function(ax=ax, color="black", linewidth=2)
        synth_kmf.plot_survival_function(ax=ax, color="#dc2626", linewidth=2)
        p_val = km_df.loc[name, "p_value"] if name in km_df.index else np.nan
        ax.set_title(f"{name}\nlog-rank p={p_val:.4f}", fontsize=10*_S)
        ax.set_xlabel("N_Days", fontsize=9*_S)
        ax.set_ylabel("Survival probability", fontsize=9*_S)
        ax.get_legend().remove()
        ax.grid(True, alpha=0.3)

    for ax in axes[len(pool_files):]:
        ax.axis("off")

    handles = [plt.Line2D([0], [0], color="black", lw=2, label="Real"),
              plt.Line2D([0], [0], color="#dc2626", lw=2, label="Synthetic")]
    fig.legend(handles=handles, loc="lower right", ncol=1, fontsize=11*_S,
              bbox_to_anchor=(0.97, 0.08), frameon=True)

    plt.tight_layout()
    out = f"{FIG_DIR}/fig12_km_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    feature_importance_figure()
    print()
    fid_vs_tstr_scatter()
    print()
    km_curves_figure()
