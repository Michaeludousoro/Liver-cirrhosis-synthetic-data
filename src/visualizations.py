"""
Visualizations
==============

Project
-------
Synthetic Data Generation and Predictive Utility Evaluation
for Primary Biliary Cirrhosis (PBC) — Mayo Clinic Dataset

Purpose of this module
-----------------------
This module generates all publication-ready figures for the paper. Every figure
is saved at 300 DPI, which meets the resolution requirements of most medical
and data science journals. The figures are designed to be informative, clean,
and readable without emojis, decorative symbols, or unnecessary visual clutter.

Figures produced
    Figure 0: Training loss curves for all three generative models.
              This shows the generator and discriminator loss over training
              epochs, which helps the reader assess model convergence.

    Figure 1: FID score bar chart comparing GAN, cGAN, VAE, and Consensus.
              Lower bars indicate better fidelity to the real data distribution.

    Figure 2: IQR filtering effect on bilirubin distributions.
              Shows the distribution before and after filtering for each method,
              illustrating how the filter removes physiologically implausible values.

    Figure 3: Feature distribution comparison across all methods.
              Side-by-side histograms for six key clinical features comparing
              the real data distribution against all four synthetic datasets.

    Figure 4: Pearson correlation heatmaps for real and consensus data.
              Comparing the correlation structure helps us verify that the
              synthetic data preserves the biological relationships between features.

    Figure 5: Consensus source distribution bar chart and pie chart.
              Shows what proportion of the consensus records came from each
              generative method.

    Figure 6: Predictive performance heatmap.
              A Scenario by Classifier grid showing the F1 score (or other metric)
              with colour intensity indicating performance level.

    Figure 7: Grouped bar chart of classifier performance across scenarios.
              Compares all three classifiers side by side for the three metrics
              Accuracy, F1, and AUC.

    Figure 8: Line plot of all five metrics across the three training scenarios
              for every classifier. Provides a comprehensive view of how
              augmentation changes the full performance profile.

    Figure 9: Pipeline flowchart summarising the full methodology.
              A schematic diagram that can be directly reproduced in the paper
              to illustrate the end-to-end workflow.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
# Force the non-interactive backend. _save_figure() calls plt.show(), which
# blocks indefinitely under a GUI backend when the pipeline is run headless,
# and that silently hangs master_runner.py partway through figure generation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR  = os.path.join(BASE_DIR, "output", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Global style settings for all figures.
#
# Font sizes are chosen for PRINT, not for the notebook. Matplotlib points are
# relative to the figure canvas, so a figure built at 14 inches wide and placed
# by LaTeX at \textwidth (7.16 in) is scaled down by about 2x, halving every
# label. The sizes below are therefore roughly twice their intended printed
# size: a 17 pt label here reaches the reader at about 8.5 pt, which matches the
# caption text around it. See print_style.py for the full derivation.
#
# The earlier settings (11 pt base on canvases up to 18 inches wide) printed at
# 2 to 4 pt and were illegible on A4.
_PRINT_SCALE = 2.0

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         8.5 * _PRINT_SCALE,
    "axes.titlesize":    9.5 * _PRINT_SCALE,
    "axes.labelsize":    8.5 * _PRINT_SCALE,
    "xtick.labelsize":   7.5 * _PRINT_SCALE,
    "ytick.labelsize":   7.5 * _PRINT_SCALE,
    "legend.fontsize":   7.5 * _PRINT_SCALE,

    # High contrast for print: black text on a white panel.
    "text.color":        "black",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "axes.edgecolor":    "black",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",

    # Strokes shrink with the figure too, so scale them as well.
    "axes.linewidth":    0.8 * _PRINT_SCALE,
    "lines.linewidth":   1.4 * _PRINT_SCALE,
    "xtick.major.width": 0.8 * _PRINT_SCALE,
    "ytick.major.width": 0.8 * _PRINT_SCALE,
    "patch.linewidth":   0.6 * _PRINT_SCALE,

    "figure.dpi":        100,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# Colour palette used consistently across all figures
METHOD_COLORS = {
    "Real":      "#505050",
    "GAN":       "#0066FF",
    "cGAN":     "#FF4500",
    "VAE":      "#00B43C",
    "Consensus": "#9900CC",
}
SCENARIO_COLORS = ["#0052CC", "#E63800", "#008C38"]


def _save_figure(fig, filename):
    """Save a figure to the output figures directory, display it inline, then close it."""
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.show()
    plt.close(fig)
    print(f"  Figure saved: {path}")


def plot_training_losses(gan_model, ctgan_model, tvae_model):
    """
    Figure 0: Training loss curves for all three generative models.

    For the GAN and cGAN we plot both the generator and discriminator loss
    on the same axes. An ideal training run shows the two losses converging
    toward a stable equilibrium. For the VAE we plot the total ELBO loss,
    which should decrease smoothly over training.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(gan_model.g_losses, label="Generator",     color="#0066FF", linewidth=1.8)
    ax.plot(gan_model.d_losses, label="Discriminator", color="#CC0000", linewidth=1.8)
    ax.set_title("Vanilla GAN Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(ctgan_model.g_losses, label="Generator",     color="#FF4500", linewidth=1.8)
    ax.plot(ctgan_model.d_losses, label="Discriminator", color="#CC0000", linewidth=1.8)
    ax.set_title("cGAN Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    ax = axes[2]
    ax.plot(tvae_model.losses, label="ELBO Loss", color="#00B43C", linewidth=1.8)
    ax.set_title("VAE Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    fig.tight_layout()
    _save_figure(fig, "fig0_training_losses.png")


def plot_fid_comparison(fid_df):
    """
    Figure 1: FID scores for each synthetic method compared to real data.

    A lower bar indicates that the synthetic distribution is closer to the
    real data distribution. The Consensus bar is expected to be the lowest
    because it represents the high-confidence intersection of all three methods.

    Parameters
    ----------
    fid_df : DataFrame with columns Method, FID, and n_samples
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bar_colors = [
        METHOD_COLORS.get(name.split()[0], "#999999")
        for name in fid_df["Method"]
    ]
    bars = ax.bar(fid_df["Method"], fid_df["FID"],
                  color=bar_colors, edgecolor="white", linewidth=0.8, width=0.6)

    for bar, val in zip(bars, fid_df["FID"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + fid_df["FID"].max() * 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9*_PRINT_SCALE
        )

    ax.set_ylabel("FID score (lower is better)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    _save_figure(fig, "fig1_fid_comparison.png")


def plot_iqr_filtering(real_df, raw_synthetics, filtered_synthetics,
                       col="Bilirubin"):
    """
    Figure 2: The effect of IQR filtering on a key clinical feature.

    We show three overlaid histograms per method: the real distribution,
    the raw synthetic distribution before filtering, and the filtered
    distribution after removing out-of-range records. Bilirubin is used
    as the default example because it is one of the most diagnostically
    important and most variable features in this dataset.

    Parameters
    ----------
    real_df             : real training DataFrame
    raw_synthetics      : dictionary with keys GAN, cGAN, VAE (unfiltered)
    filtered_synthetics : dictionary with keys GAN, cGAN, VAE (after IQR filter)
    col                 : the feature column to visualise
    """
    methods = list(raw_synthetics.keys())
    fig, axes = plt.subplots(1, len(methods),
                             figsize=(5 * len(methods), 4),
                             sharey=False)
    if len(methods) == 1:
        axes = [axes]

    real_vals = real_df[col].dropna().values

    for ax, method in zip(axes, methods):
        raw_vals  = (raw_synthetics[method][col].dropna().values
                     if col in raw_synthetics[method].columns else np.array([]))
        filt_vals = (filtered_synthetics[method][col].dropna().values
                     if col in filtered_synthetics[method].columns else np.array([]))

        upper_limit = np.percentile(
            np.concatenate([real_vals, raw_vals]) if len(raw_vals) else real_vals,
            99
        )
        bins = np.linspace(0, upper_limit, 40)

        ax.hist(real_vals,  bins=bins, alpha=0.55, color=METHOD_COLORS["Real"],
                label="Real patients", density=True)
        ax.hist(raw_vals,   bins=bins, alpha=0.45, color="#CC0000",
                label="Synthetic before filter", density=True)
        ax.hist(filt_vals,  bins=bins, alpha=0.55, color=METHOD_COLORS.get(method, "#999"),
                label="Synthetic after filter", density=True)

        ax.set_title(method)
        ax.set_xlabel(col + " (mg/dL)")
        ax.set_ylabel("Density")
        # legend drawn once for the whole figure, below the panels

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=7.5*_PRINT_SCALE, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _save_figure(fig, "fig2_iqr_filtering.png")


def plot_distribution_comparison(real_df, synthetic_dict, cols=None):
    """
    Figure 3: Distribution comparison for six key clinical features.

    For each feature we overlay the real distribution with all four synthetic
    distributions (GAN, cGAN, VAE, Consensus). This gives a comprehensive
    visual impression of how well each method reproduces the clinical data.

    Parameters
    ----------
    real_df        : real training DataFrame
    synthetic_dict : dictionary mapping method names to DataFrames
    cols           : list of up to six feature columns to show
                     defaults to the six most clinically informative continuous features
    """
    if cols is None:
        cols = ["Bilirubin", "Albumin", "Prothrombin",
                "Cholesterol", "Copper", "SGOT"]
    cols = [c for c in cols if c in real_df.columns][:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.hist(real_df[col].dropna().values, bins=25, alpha=0.6,
                color=METHOD_COLORS["Real"], label="Real", density=True)

        for method, synth_df in synthetic_dict.items():
            if col not in synth_df.columns:
                continue
            color = METHOD_COLORS.get(method.split()[0], "#999999")
            ax.hist(synth_df[col].dropna().values, bins=25, alpha=0.4,
                    color=color, label=method, density=True)

        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend(fontsize=7*_PRINT_SCALE)

    for j in range(len(cols), 6):
        axes[j].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, "fig3_distribution_comparison.png")


def plot_correlation_heatmap(real_df, consensus_df, numeric_cols=None):
    """
    Figure 4: Pearson correlation heatmaps for real and consensus data.

    Preserving the correlation structure between features is important for
    generating realistic synthetic patient records. If bilirubin and prothrombin
    are highly correlated in the real data, a good synthetic dataset should
    preserve that relationship.

    Parameters
    ----------
    real_df       : real training DataFrame
    consensus_df  : consensus synthetic DataFrame
    numeric_cols  : continuous features to include in the correlation matrix
    """
    from .data_loader import CONTINUOUS_COLS
    if numeric_cols is None:
        numeric_cols = [c for c in CONTINUOUS_COLS if c in real_df.columns]

    cmap = LinearSegmentedColormap.from_list(
        "blue_white_pink", ["#0066FF", "#FFFFFF", "#CC0000"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (title, df) in zip(axes,
            [("Real Patient Data", real_df), ("Consensus Synthetic Data", consensus_df)]):
        corr   = df[numeric_cols].corr()
        image  = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=8*_PRINT_SCALE)
        ax.set_yticklabels(numeric_cols, fontsize=8*_PRINT_SCALE)
        ax.set_title(title)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    _save_figure(fig, "fig4_correlation_heatmap.png")


def plot_consensus_distribution(source_counts):
    """
    Figure 5: Proportion of consensus records contributed by each generative method.

    The bar chart shows absolute counts and the pie chart shows proportions.
    A roughly equal distribution across methods suggests the three models are
    exploring similar regions of feature space. Dominance by one method may
    indicate the other methods have not converged as well.

    Parameters
    ----------
    source_counts : dictionary from run_consensus, e.g. {'GAN': 45, 'cGAN': 60, 'VAE': 95}
    """
    labels = list(source_counts.keys())
    counts = [source_counts[k] for k in labels]
    colors = [METHOD_COLORS.get(k, "#999999") for k in labels]
    total  = sum(counts)

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(10, 5))

    bars = ax_bar.bar(labels, counts, color=colors,
                      edgecolor="white", linewidth=0.8, width=0.5)
    for bar, v in zip(bars, counts):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + total * 0.01,
                    str(v), ha="center", va="bottom", fontsize=10*_PRINT_SCALE)
    ax_bar.set_ylabel("Number of consensus records")
    ax_bar.set_title("Records contributed per method")

    wedge_style = {"linewidth": 1.5, "edgecolor": "white"}
    ax_pie.pie(counts, labels=labels, colors=colors,
               autopct="%1.1f%%", startangle=140, wedgeprops=wedge_style)
    ax_pie.set_title("Proportional contribution")

    fig.tight_layout()
    _save_figure(fig, "fig5_consensus_distribution.png")


def plot_performance_heatmap(results_df, metric="F1"):
    """
    Figure 6: Heatmap of classifier performance across training scenarios.

    The rows represent the three training scenarios and the columns represent
    the three classifiers. Each cell shows the value of the chosen metric.
    Darker blue indicates higher performance.

    Parameters
    ----------
    results_df : output of run_all_scenarios
    metric     : the performance metric to display (default F1)
    """
    pivot = results_df.pivot(
        index="Scenario", columns="Classifier", values=metric
    )

    cmap = LinearSegmentedColormap.from_list("performance", ["#CCE5FF", "#003399"])
    fig, ax = plt.subplots(figsize=(10, 4))

    image = ax.imshow(
        pivot.values, cmap=cmap,
        vmin=pivot.values[~np.isnan(pivot.values)].min() * 0.98,
        vmax=pivot.values[~np.isnan(pivot.values)].max() * 1.01,
        aspect="auto"
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticklabels([s[:40] for s in pivot.index])

    mean_val = np.nanmean(pivot.values)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > mean_val else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        color=text_color, fontsize=10*_PRINT_SCALE, fontweight="bold")

    plt.colorbar(image, ax=ax, fraction=0.02, pad=0.02, label=metric)
    fig.tight_layout()
    _save_figure(fig, "fig6_performance_heatmap.png")


def plot_model_comparison(results_df, metrics=None):
    """
    Figure 7: Grouped bar chart comparing classifiers across training scenarios.

    For each metric (Accuracy, F1, and AUC) we show a group of bars, one per
    classifier, for each training scenario. This makes it easy to compare how
    each classifier responds to the different augmentation strategies.

    Parameters
    ----------
    results_df : output of run_all_scenarios
    metrics    : list of metric names to include (default: Accuracy, F1, AUC)
    """
    if metrics is None:
        metrics = ["Accuracy", "F1", "AUC"]
    metrics     = [m for m in metrics if m in results_df.columns]
    scenarios   = results_df["Scenario"].unique().tolist()
    classifiers = results_df["Classifier"].unique().tolist()

    n_clf = len(classifiers)
    bar_w = 0.8 / n_clf
    x     = np.arange(len(scenarios))

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for i, clf in enumerate(classifiers):
            subset = results_df[results_df["Classifier"] == clf].set_index("Scenario")
            values = [
                subset.loc[s, metric] if s in subset.index else np.nan
                for s in scenarios
            ]
            offset = (i - n_clf / 2 + 0.5) * bar_w
            bars   = ax.bar(x + offset, values, width=bar_w * 0.9,
                            label=clf,
                            color=plt.cm.Set2(i / n_clf),
                            edgecolor="white")
            for bar, v in zip(bars, values):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=7*_PRINT_SCALE, rotation=90
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [t.split(":")[0].strip() for t in scenarios],
            rotation=25, ha="right", fontsize=8*_PRINT_SCALE
        )
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_ylim(0, min(1.15, results_df[metric].max() * 1.2))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 1.02), ncol=len(classifiers), fontsize=8*_PRINT_SCALE)
    fig.tight_layout()
    _save_figure(fig, "fig7_model_comparison.png")


def plot_all_metrics(results_df):
    """
    Figure 8: Line plots of all five metrics across three training scenarios.

    Each line represents one classifier. The x-axis shows the three scenarios
    labelled A (baseline), B (real plus cGAN), and C (real plus consensus).
    Upward-trending lines from A to B or C indicate that augmentation improved
    performance for that metric and classifier combination.

    Parameters
    ----------
    results_df : output of run_all_scenarios
    """
    all_metrics = ["Accuracy", "F1", "Precision", "Recall", "AUC"]
    metrics     = [m for m in all_metrics if m in results_df.columns]
    classifiers = results_df["Classifier"].unique().tolist()
    scenarios   = results_df["Scenario"].unique().tolist()

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    marker_styles = ["o", "s", "^", "D", "v"]

    for ax, metric in zip(axes, metrics):
        for j, clf in enumerate(classifiers):
            subset = results_df[results_df["Classifier"] == clf].set_index("Scenario")
            values = [
                subset.loc[s, metric] if s in subset.index else np.nan
                for s in scenarios
            ]
            ax.plot(
                range(len(scenarios)), values,
                marker=marker_styles[j % len(marker_styles)],
                label=clf.replace(" ", "\n"),
                color=plt.cm.Set1(j / max(len(classifiers), 1)),
                linewidth=1.8, markersize=7
            )
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([s.split(":")[0].strip() for s in scenarios])
        ax.set_xlabel("Training scenario")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_ylim(0, 1.1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.12), ncol=len(classifiers), fontsize=8*_PRINT_SCALE)
    fig.tight_layout()
    _save_figure(fig, "fig8_all_metrics.png")


def plot_pipeline_flowchart():
    """
    Figure 9: Schematic flowchart of the complete pipeline methodology.

    This diagram illustrates the sequence of processing steps from the raw
    dataset through synthetic generation, IQR filtering, consensus voting,
    FID evaluation, predictive modeling, and statistical analysis. It can be
    reproduced directly in the paper's methods section.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Each entry is (x_centre, y_centre, label_text, background_colour)
    steps = [
        (5.0, 9.2, "Raw Dataset\n418 patients (cirrhosis.csv)",               "#505050"),
        (5.0, 8.0, "Complete-Case Filter\n(drop rows with missing values)",  "#505050"),
        (5.0, 6.8, "Encode, scale to [0,1]\nsplit 70/30",             "#505050"),
        (2.5, 5.5, "Vanilla GAN\n500 samples",  "#505050"),
        (5.0, 5.5, "cGAN\n500 samples",         "#505050"),
        (7.5, 5.5, "VAE\n500 samples",          "#505050"),
        (2.5, 4.3, "IQR Filter",                 "#505050"),
        (5.0, 4.3, "IQR Filter",                 "#505050"),
        (7.5, 4.3, "IQR Filter",                 "#505050"),
        (5.0, 3.1, "Consensus Voting\ntolerance 5.0, min votes 2",              "#505050"),
        (5.0, 2.0, "FID Scores and\nPredictive Utility Evaluation",             "#505050"),
        (5.0, 0.9, "Statistical Tests and\nPublication Figures",                "#505050"),
    ]

    box_height = 0.55
    WIDE, NARROW = 5.2, 2.2          # single-step vs three-across rows
    TRIPLE_ROWS = {5.5, 4.3}         # y-levels holding three boxes side by side

    for (x, y, label, color) in steps:
        box_width = NARROW if y in TRIPLE_ROWS else WIDE
        box = mpatches.FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2),
            box_width, box_height,
            boxstyle="round,pad=0.06",
            facecolor=color, edgecolor="white",
            linewidth=1.2, alpha=0.92
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center",
                color="white", fontsize=6.5*_PRINT_SCALE, fontweight="bold",
                multialignment="center")

    arrow_style = dict(arrowstyle="->", color="#333333",
                       linewidth=1.4, mutation_scale=12)
    connections = [
        ((5, 8.92), (5, 8.28)),
        ((5, 7.72), (5, 7.08)),
        ((5, 6.52), (2.5, 5.78)), ((5, 6.52), (5, 5.78)), ((5, 6.52), (7.5, 5.78)),
        ((2.5, 5.22), (2.5, 4.58)), ((5, 5.22), (5, 4.58)), ((7.5, 5.22), (7.5, 4.58)),
        ((2.5, 4.02), (5, 3.38)), ((5, 4.02), (5, 3.38)), ((7.5, 4.02), (5, 3.38)),
        ((5, 2.78), (5, 2.28)),
        ((5, 1.72), (5, 1.18)),
    ]
    for (x0, y0), (x1, y1) in connections:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=arrow_style)

    fig.tight_layout()
    _save_figure(fig, "fig9_pipeline_flowchart.png")
