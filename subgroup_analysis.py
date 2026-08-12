"""
Subgroup Consistency Analysis
==============================

This script evaluates whether the main findings (null result on held-out
accuracy, stability benefit in cross-validation) hold consistently across
three clinical subgroups of the 83-patient test set:

    Stage subgroup:   early disease (Stage 1 or 2) vs late disease (Stage 3 or 4)
    Sex subgroup:     female vs male
    Treatment arm:    D-penicillamine vs placebo

For each subgroup, classifiers trained under Scenario A (real only) and
Scenario C (real plus equalised consensus) are evaluated on that subgroup
alone. AUC is reported. Because subgroups are small (the smallest has 11
patients), formal statistical tests are not applied; the purpose is to show
directional consistency rather than significance within subgroups.

If the null result holds across all subgroups, this provides evidence that
the finding is not driven by a specific patient profile and partially
addresses the single-dataset external validity concern.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Print-safe sizing: matplotlib points are relative to the figure canvas, so a
# wide canvas placed at \\textwidth by LaTeX shrinks every label. Sizes here are
# about twice their intended printed size so they land near 8 pt on the page.
import matplotlib.pyplot as _plt
_S = 2.0
_plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5*_S, "axes.titlesize": 9.5*_S,
    "axes.labelsize": 8.5*_S, "xtick.labelsize": 7.5*_S, "ytick.labelsize": 7.5*_S,
    "legend.fontsize": 7.5*_S, "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.linewidth": 0.8*_S, "lines.linewidth": 1.4*_S, "xtick.major.width": 0.8*_S,
    "ytick.major.width": 0.8*_S, "patch.linewidth": 0.6*_S, "savefig.dpi": 300,
})


from sklearn.ensemble       import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import TARGET_COL, ALL_FEATURE_COLS

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA     = os.path.join(BASE, "output", "data")
RESULTS  = os.path.join(BASE, "output", "results")
FIGURES  = os.path.join(BASE, "output", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

feat = ALL_FEATURE_COLS


def load_data():
    train  = pd.read_csv(os.path.join(DATA, "train_real.csv"))
    test   = pd.read_csv(os.path.join(DATA, "test_real.csv"))
    cons   = pd.read_csv(os.path.join(DATA, "consensus_equalised.csv"))
    return train, test, cons


def build_training_sets(train, consensus):
    X_tr = train[feat].values
    y_tr = train[TARGET_COL].values
    X_cn = consensus[feat].values
    y_cn = consensus[TARGET_COL].values.astype(int).clip(0, 1)

    scenarios = {
        "A: Real only":        (X_tr, y_tr),
        "C: Real + Consensus": (np.vstack([X_tr, X_cn]), np.hstack([y_tr, y_cn])),
    }
    return scenarios


def evaluate_subgroup(clf, X_train, y_train, X_sub, y_sub):
    if len(np.unique(y_sub)) < 2:
        return None
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_sub)[:, 1]
    return round(roc_auc_score(y_sub, probs), 4)


def run_analysis(train, test, consensus):
    scenarios = build_training_sets(train, consensus)

    subgroups = {
        "Early stage (1-2)": test[test["Stage"].isin([1, 2])],
        "Late stage (3-4)":  test[test["Stage"].isin([3, 4])],
        "Female":            test[test["Sex"] == 0],
        "Male":              test[test["Sex"] == 1],
        "D-penicillamine":   test[test["Drug"] == 1],
        "Placebo":           test[test["Drug"] == 0],
    }

    clfs = {
        "Random Forest":       RandomForestClassifier(random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    }

    rows = []
    for sg_name, sg_df in subgroups.items():
        X_sub = sg_df[feat].values
        y_sub = sg_df[TARGET_COL].values
        n     = len(sg_df)
        dec   = int(y_sub.sum())

        for sc_name, (X_tr, y_tr) in scenarios.items():
            for clf_name, clf in clfs.items():
                auc = evaluate_subgroup(clf, X_tr, y_tr, X_sub, y_sub)
                rows.append({
                    "Subgroup":   sg_name,
                    "n":          n,
                    "Deceased":   dec,
                    "Scenario":   sc_name,
                    "Classifier": clf_name,
                    "AUC":        auc,
                })

    return pd.DataFrame(rows)


def print_table(results_df):
    print()
    print("=" * 80)
    print("SUBGROUP CONSISTENCY ANALYSIS")
    print("AUC on held-out subgroups: Scenario A (real only) vs Scenario C (consensus)")
    print("=" * 80)
    print()
    print(f"{'Subgroup':<22} {'n':>4} {'Classifier':<22} {'Scenario A AUC':>15} {'Scenario C AUC':>15} {'Delta':>7}")
    print("-" * 80)

    for sg in ["Early stage (1-2)", "Late stage (3-4)", "Female", "Male",
               "D-penicillamine", "Placebo"]:
        sg_df = results_df[results_df["Subgroup"] == sg]
        n = sg_df["n"].iloc[0]
        first = True
        for clf in ["Random Forest", "Gradient Boosting", "Logistic Regression"]:
            row_a = sg_df[(sg_df["Scenario"].str.startswith("A")) & (sg_df["Classifier"] == clf)]
            row_c = sg_df[(sg_df["Scenario"].str.startswith("C")) & (sg_df["Classifier"] == clf)]
            if row_a.empty or row_c.empty:
                continue
            auc_a = row_a["AUC"].values[0]
            auc_c = row_c["AUC"].values[0]
            if auc_a is None or auc_c is None:
                continue
            delta = auc_c - auc_a
            sg_label = sg if first else ""
            n_label  = str(n) if first else ""
            first = False
            print(f"{sg_label:<22} {n_label:>4} {clf:<22} {auc_a:>15.4f} {auc_c:>15.4f} {delta:>+7.4f}")
        print()


def plot_subgroup(results_df):
    subgroups = ["Early stage (1-2)", "Late stage (3-4)", "Female",
                 "Male", "D-penicillamine", "Placebo"]
    clfs = ["Random Forest", "Gradient Boosting", "Logistic Regression"]
    colors = {"Random Forest": "#3b82f6", "Gradient Boosting": "#059669",
               "Logistic Regression": "#ef4444"}

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax_idx, scenario_prefix in enumerate(["A", "C"]):
        sc_label = "A: Real only" if scenario_prefix == "A" else "C: Real + Consensus"
        ax = axes[ax_idx]

        x    = np.arange(len(subgroups))
        w    = 0.25
        offsets = [-w, 0, w]

        for clf_idx, clf_name in enumerate(clfs):
            aucs = []
            for sg in subgroups:
                row = results_df[(results_df["Subgroup"] == sg) &
                                 (results_df["Scenario"].str.startswith(scenario_prefix)) &
                                 (results_df["Classifier"] == clf_name)]
                aucs.append(row["AUC"].values[0] if not row.empty and row["AUC"].values[0] is not None else 0)

            ax.bar(x + offsets[clf_idx], aucs, w,
                   label=clf_name, color=colors[clf_name], alpha=0.85)

        ax.axhline(0.8, color="black", linestyle="--", linewidth=1,
                   label="AUC = 0.8 (clinical threshold)")
        ax.set_xticks(x)
        sg_labels = [f"{s}\n(n={results_df[results_df['Subgroup']==s]['n'].values[0]})"
                     for s in subgroups]
        ax.set_xticklabels(sg_labels, fontsize=9*_S)
        ax.set_ylabel("AUC", fontsize=11*_S)
        sc_title = "Scenario A: Real data only" if scenario_prefix == "A" else "Scenario C: Real + Consensus"
        ax.set_title(sc_title, fontsize=11*_S)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9*_S)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(FIGURES, "subgroup_analysis.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Figure saved: {out}")


def main():
    train, test, consensus = load_data()
    print(f"Training set: {len(train)} | Test set: {len(test)} | Consensus: {len(consensus)}")

    results_df = run_analysis(train, test, consensus)
    results_df.to_csv(os.path.join(RESULTS, "subgroup_analysis.csv"), index=False)

    print_table(results_df)
    plot_subgroup(results_df)
    print(f"Results saved: {os.path.join(RESULTS, 'subgroup_analysis.csv')}")


if __name__ == "__main__":
    main()
