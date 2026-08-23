"""
Remaining Corrected Figures: ROC, PR, AUC-comparison, and the three
visualizations.py-based figures (performance heatmap, model comparison,
all-metrics), all rebuilt on the leak-fixed (N_Days-excluded) classifiers.
Saves under the exact filenames organise_figures.py expects in
output/figures/, so the normal organise_figures.py / optimise_figures.py
pipeline can copy/resize them into paper/figures and
paper/figures_supplementary unchanged.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler

from src.data_loader import CLASSIFICATION_FEATURE_COLS, TARGET_COL
from src.predictive_modeling import _build_classifiers
from src.visualizations import plot_performance_heatmap, plot_model_comparison, plot_all_metrics

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
FEAT_COLS = CLASSIFICATION_FEATURE_COLS
CLASSIFIERS = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
SCENARIOS = ["A: Baseline (real data only)", "B: Real data plus filtered cGAN",
            "C: Real data plus consensus synthetic"]
SCENARIO_SHORT = ["A: Baseline", "B: +cGAN", "C: +Consensus"]
COLORS = ["#0052CC", "#E63800", "#008C38"]


def fit_scenarios_get_probs():
    train_real = pd.read_csv(f"{DATA_DIR}/train_real.csv")
    test_real = pd.read_csv(f"{DATA_DIR}/test_real.csv")
    filtered_ctgan = pd.read_csv(f"{DATA_DIR}/filtered_ctgan.csv")
    consensus_df = pd.read_csv(f"{DATA_DIR}/consensus_equalised.csv")

    def combine(*dfs):
        return pd.concat(list(dfs), ignore_index=True).dropna(subset=FEAT_COLS + [TARGET_COL])

    scenario_dfs = {
        SCENARIOS[0]: train_real,
        SCENARIOS[1]: combine(train_real, filtered_ctgan),
        SCENARIOS[2]: combine(train_real, consensus_df),
    }

    X_test = test_real[FEAT_COLS].values
    y_test = np.round(test_real[TARGET_COL].values).astype(int)

    probs = {s: {} for s in SCENARIOS}
    for scenario, train_df in scenario_dfs.items():
        X_train = train_df[FEAT_COLS].values
        y_train = np.round(train_df[TARGET_COL].values).astype(int)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        for clf_name, clf in _build_classifiers().items():
            clf.fit(X_train_s, y_train)
            probs[scenario][clf_name] = clf.predict_proba(X_test_s)[:, 1]

    return probs, y_test


def plot_roc(probs, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, clf_name in zip(axes, CLASSIFIERS):
        for scenario, short, color in zip(SCENARIOS, SCENARIO_SHORT, COLORS):
            y_prob = probs[scenario][clf_name]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{short} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(clf_name)
        ax.legend(fontsize=7*_S, loc="lower right")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/figures/roc_curves.png", bbox_inches="tight")
    plt.close()
    print("Saved output/figures/roc_curves.png")


def plot_pr(probs, y_test):
    baseline_rate = y_test.mean()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, clf_name in zip(axes, CLASSIFIERS):
        for scenario, short, color in zip(SCENARIOS, SCENARIO_SHORT, COLORS):
            y_prob = probs[scenario][clf_name]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            ap = average_precision_score(y_test, y_prob)
            ax.plot(recall, precision, color=color, linewidth=2, label=f"{short} (AP={ap:.3f})")
        ax.axhline(baseline_rate, color="gray", linestyle="--", linewidth=1,
                  label=f"Baseline rate ({baseline_rate:.2f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(clf_name)
        ax.legend(fontsize=7*_S, loc="lower left")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/figures/pr_curves.png", bbox_inches="tight")
    plt.close()
    print("Saved output/figures/pr_curves.png")


def plot_auc_comparison(performance_df):
    fig, ax = plt.subplots(figsize=(15, 5.5))
    x = np.arange(len(CLASSIFIERS))
    width = 0.25

    for i, (scenario, short, color) in enumerate(zip(SCENARIOS, SCENARIO_SHORT, COLORS)):
        auc_vals = [
            float(performance_df[(performance_df["Scenario"] == scenario) &
                                 (performance_df["Classifier"] == clf)]["AUC"].values[0])
            for clf in CLASSIFIERS
        ]
        bars = ax.bar(x + i * width, auc_vals, width, label=short, color=color,
                      alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, auc_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                   f"{val:.3f}", ha="center", va="bottom", fontsize=6.5*_S,
                   fontweight="bold", rotation=90)

    ax.axhline(0.5, color="red", linewidth=1.2, linestyle="--", label="Random classifier (AUC = 0.5)")
    ax.axhline(0.8, color="green", linewidth=1.2, linestyle="--", label="Good discrimination threshold (AUC = 0.8)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASSIFIERS, fontsize=10*_S)
    ax.set_ylabel("AUC Score")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("AUC Comparison Across Training Scenarios and Classifiers", fontsize=13*_S)
    ax.legend(fontsize=7.5*_S, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig("output/figures/auc_detailed_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved output/figures/auc_detailed_comparison.png")


def main():
    probs, y_test = fit_scenarios_get_probs()
    plot_roc(probs, y_test)
    plot_pr(probs, y_test)

    performance_df = pd.read_csv("output/results/model_performance.csv")
    plot_auc_comparison(performance_df)

    smote_df = pd.read_csv("output/results/smote_results.csv")
    plot_df = pd.concat([performance_df, smote_df], ignore_index=True).sort_values("Scenario").reset_index(drop=True)

    plot_performance_heatmap(plot_df, metric="F1")
    plot_model_comparison(plot_df)
    plot_all_metrics(plot_df)
    print("Saved fig6_performance_heatmap.png, fig7_model_comparison.png, fig8_all_metrics.png")


if __name__ == "__main__":
    main()
